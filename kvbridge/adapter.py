"""One explainable cross-model KV adapter: resize, then affine projection."""

from __future__ import annotations

import json
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file


@dataclass(frozen=True)
class LayerKV:
    key: torch.Tensor
    value: torch.Tensor


def cache_layers(cache: Any) -> list[LayerKV]:
    """Read the DynamicCache representation supported by the pinned runtime."""
    raw_layers = getattr(cache, "layers", None)
    if raw_layers is None and hasattr(cache, "to_legacy_cache"):
        raw_layers = cache.to_legacy_cache()
    if raw_layers is None and isinstance(cache, (tuple, list)):
        raw_layers = cache
    if raw_layers is None:
        raise TypeError("Expected a Hugging Face DynamicCache or legacy cache tuple")

    result: list[LayerKV] = []
    for layer in raw_layers:
        if isinstance(layer, (tuple, list)):
            key, value = layer[:2]
        else:
            key, value = layer.keys, layer.values
        if key.ndim != 4 or key.shape != value.shape:
            raise ValueError("KV tensors must share [batch, heads, tokens, head_dim] shape")
        result.append(LayerKV(key, value))
    if not result:
        raise ValueError("Cannot adapt an empty cache")
    return result


def config_shape(config: Any) -> tuple[int, int, int]:
    """Return (layers, KV heads, head dimension) for the three target families."""
    layers = int(config.num_hidden_layers)
    heads = int(getattr(config, "num_key_value_heads", config.num_attention_heads))
    head_dim = int(getattr(config, "head_dim", config.hidden_size // config.num_attention_heads))
    return layers, heads, head_dim


def relative_layer_map(source_layers: int, target_layers: int) -> tuple[int, ...]:
    if source_layers < 1 or target_layers < 1:
        raise ValueError("layer counts must be positive")
    if target_layers == 1:
        return (0,)
    return tuple(
        round(index * (source_layers - 1) / (target_layers - 1))
        for index in range(target_layers)
    )


def resize_tokens(tensor: torch.Tensor, target_tokens: int) -> torch.Tensor:
    """Linearly interpolate the sequence axis."""
    if target_tokens < 1:
        raise ValueError("target_tokens must be positive")
    if tensor.shape[-2] == target_tokens:
        return tensor
    dtype = tensor.dtype
    batch, heads, tokens, width = tensor.shape
    rows = tensor.float().permute(0, 1, 3, 2).reshape(batch * heads, width, tokens)
    rows = F.interpolate(rows, size=target_tokens, mode="linear", align_corners=True)
    return rows.reshape(batch, heads, width, target_tokens).permute(0, 1, 3, 2).to(dtype)


def resize_heads(tensor: torch.Tensor, target_heads: int) -> torch.Tensor:
    """Average proportional groups when shrinking and repeat when expanding."""
    source_heads = int(tensor.shape[1])
    if target_heads < 1:
        raise ValueError("target_heads must be positive")
    if source_heads == target_heads:
        return tensor
    if source_heads < target_heads:
        indices = torch.floor(
            torch.arange(target_heads, device=tensor.device) * source_heads / target_heads
        ).long()
        return tensor.index_select(1, indices)
    groups = []
    for index in range(target_heads):
        start = math.floor(index * source_heads / target_heads)
        end = max(start + 1, math.floor((index + 1) * source_heads / target_heads))
        groups.append(tensor[:, start:end].mean(dim=1, keepdim=True))
    return torch.cat(groups, dim=1)


def resized_source(tensor: torch.Tensor, target_heads: int, target_tokens: int) -> torch.Tensor:
    return resize_heads(resize_tokens(tensor, target_tokens), target_heads)


def build_cache(layers: Iterable[LayerKV], config: Any) -> Any:
    from transformers import DynamicCache

    cache = DynamicCache(config=config)
    for index, layer in enumerate(layers):
        cache.update(layer.key, layer.value, index)
    return cache


class KVAdapter(ABC):
    """The required three-argument adapter interface."""

    approved: bool = False
    rejection_reason: str = "quality_gate_rejected"

    def prepare_target(self, target_prefix_ids: torch.Tensor) -> None:
        """Bind per-request target length/device without changing adapt's public signature."""
        self._target_prefix_ids = target_prefix_ids

    @abstractmethod
    def adapt(self, source_cache: Any, source_config: Any, target_config: Any) -> Any:
        """Return a target-shaped cache for the previously bound target prefix."""


class LinearKVAdapter(KVAdapter):
    """Pair-specific affine projections shared across heads within each target layer."""

    def __init__(
        self,
        source_model: str,
        target_model: str,
        source_revision: str = "unknown",
        target_revision: str = "unknown",
        ridge_lambda: float = 0.001,
    ):
        self.source_model = source_model
        self.target_model = target_model
        self.source_revision = source_revision
        self.target_revision = target_revision
        self.ridge_lambda = ridge_lambda
        self.weights: dict[str, torch.Tensor] = {}
        self.layer_mapping: tuple[int, ...] = ()
        self.approved = False
        self.quality_score: float | None = None
        self.quality_threshold = 0.15
        self.rejection_reason = "adapter_not_calibrated"
        self.fit_examples = 0
        self.validation_examples = 0
        self._target_prefix_ids: torch.Tensor | None = None

    def adapt(self, source_cache: Any, source_config: Any, target_config: Any) -> Any:
        if self._target_prefix_ids is None:
            raise RuntimeError("prepare_target must be called before adapt")
        if not self.weights:
            raise RuntimeError("adapter has no fitted weights")
        source = cache_layers(source_cache)
        target_layers, target_heads, target_width = config_shape(target_config)
        target_tokens = int(self._target_prefix_ids.shape[-1])
        mapping = self.layer_mapping or relative_layer_map(len(source), target_layers)
        if len(mapping) != target_layers:
            raise ValueError("stored layer mapping does not match target configuration")

        converted: list[LayerKV] = []
        for target_index, source_index in enumerate(mapping):
            pair: dict[str, torch.Tensor] = {}
            for kind in ("key", "value"):
                base = resized_source(
                    getattr(source[source_index], kind), target_heads, target_tokens
                ).to(self._target_prefix_ids.device)
                weight = self.weights[f"layer.{target_index}.{kind}.weight"].to(
                    device=base.device, dtype=torch.float32
                )
                bias = self.weights[f"layer.{target_index}.{kind}.bias"].to(
                    device=base.device, dtype=torch.float32
                )
                if weight.shape != (base.shape[-1], target_width):
                    raise ValueError("projection width does not match model configurations")
                pair[kind] = (base.float() @ weight + bias).to(base.dtype)
                if not torch.isfinite(pair[kind]).all():
                    raise ValueError("adapter produced non-finite values")
            converted.append(LayerKV(pair["key"], pair["value"]))
        return build_cache(converted, target_config)

    def fit(
        self,
        cache_pairs: Iterable[tuple[Any, Any]],
        source_config: Any,
        target_config: Any,
        *,
        max_positions: int = 64,
    ) -> None:
        """Fit ridge regression from streaming paired caches using sufficient statistics."""
        source_layer_count, _, source_width = config_shape(source_config)
        target_layer_count, target_heads, target_width = config_shape(target_config)
        self.layer_mapping = relative_layer_map(source_layer_count, target_layer_count)
        stats: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        for layer in range(target_layer_count):
            for kind in ("key", "value"):
                key = f"layer.{layer}.{kind}"
                stats[key] = (
                    torch.zeros((source_width + 1, source_width + 1), dtype=torch.float64),
                    torch.zeros((source_width + 1, target_width), dtype=torch.float64),
                )

        count = 0
        for source_cache, target_cache in cache_pairs:
            source_layers = cache_layers(source_cache)
            target_layers = cache_layers(target_cache)
            count += 1
            for target_index, source_index in enumerate(self.layer_mapping):
                for kind in ("key", "value"):
                    target = getattr(target_layers[target_index], kind).detach().cpu()
                    source = getattr(source_layers[source_index], kind).detach().cpu()
                    source = resized_source(source, target_heads, int(target.shape[-2]))
                    positions = torch.linspace(
                        0,
                        target.shape[-2] - 1,
                        steps=min(max_positions, target.shape[-2]),
                    ).round().long().unique()
                    x = source[:, :, positions].reshape(-1, source_width).double()
                    y = target[:, :, positions].reshape(-1, target_width).double()
                    x = torch.cat((x, torch.ones((x.shape[0], 1), dtype=x.dtype)), dim=1)
                    key = f"layer.{target_index}.{kind}"
                    xtx, xty = stats[key]
                    xtx.add_(x.T @ x)
                    xty.add_(x.T @ y)

        if count == 0:
            raise ValueError("fit requires at least one cache pair")
        self.fit_examples = count
        penalty = torch.eye(source_width + 1, dtype=torch.float64) * self.ridge_lambda
        penalty[-1, -1] = 0.0
        for key, (xtx, xty) in stats.items():
            coefficients = torch.linalg.solve(xtx + penalty, xty).float()
            self.weights[f"{key}.weight"] = coefficients[:-1]
            self.weights[f"{key}.bias"] = coefficients[-1]
        self.rejection_reason = "quality_gate_not_run"

    def set_quality(self, score: float, threshold: float, validation_examples: int) -> None:
        self.quality_score = float(score)
        self.quality_threshold = float(threshold)
        self.validation_examples = int(validation_examples)
        self.approved = math.isfinite(score) and score <= threshold
        self.rejection_reason = (
            "" if self.approved else "degradation_threshold_exceeded"
        )

    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        stem = f"{self.source_model}__to__{self.target_model}"
        save_file({key: value.contiguous() for key, value in self.weights.items()}, directory / f"{stem}.safetensors")
        manifest = {
            "source_model": self.source_model,
            "target_model": self.target_model,
            "source_revision": self.source_revision,
            "target_revision": self.target_revision,
            "ridge_lambda": self.ridge_lambda,
            "layer_mapping": list(self.layer_mapping),
            "fit_examples": self.fit_examples,
            "validation_examples": self.validation_examples,
            "quality_metric": "mean_kl_cold_to_adapted",
            "quality_score": self.quality_score,
            "quality_threshold": self.quality_threshold,
            "approved": self.approved,
        }
        (directory / f"{stem}.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, directory: str | Path, source_model: str, target_model: str) -> "LinearKVAdapter":
        directory = Path(directory)
        stem = f"{source_model}__to__{target_model}"
        manifest = json.loads((directory / f"{stem}.json").read_text(encoding="utf-8"))
        adapter = cls(
            source_model,
            target_model,
            manifest["source_revision"],
            manifest["target_revision"],
            manifest["ridge_lambda"],
        )
        adapter.weights = load_file(directory / f"{stem}.safetensors")
        adapter.layer_mapping = tuple(manifest["layer_mapping"])
        adapter.fit_examples = manifest["fit_examples"]
        adapter.validation_examples = manifest["validation_examples"]
        adapter.quality_score = manifest["quality_score"]
        adapter.quality_threshold = manifest["quality_threshold"]
        adapter.approved = manifest["approved"]
        adapter.rejection_reason = "" if adapter.approved else "degradation_threshold_exceeded"
        return adapter
