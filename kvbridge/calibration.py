"""Fit and validate the single linear adapter on realistic hand-off transcripts."""

from __future__ import annotations

import gc
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from .adapter import LinearKVAdapter
from .pipeline import CacheState, ModelBundle


def tokenize(bundle: ModelBundle, text: str, *, special: bool) -> tuple[torch.Tensor, torch.Tensor]:
    device = bundle.model.get_input_embeddings().weight.device
    encoded = bundle.tokenizer(text, return_tensors="pt", add_special_tokens=special)
    ids = encoded["input_ids"].to(device)
    mask = encoded.get("attention_mask", torch.ones_like(ids)).to(device)
    return ids, mask


def collect_cache(bundle: ModelBundle, text: str) -> tuple[Any, torch.Tensor]:
    """Cold-prefill one transcript and return its cache and exact token IDs."""
    ids, mask = tokenize(bundle, text, special=True)
    with torch.inference_mode():
        output = bundle.model(
            input_ids=ids,
            attention_mask=mask,
            cache_position=torch.arange(ids.shape[-1], device=ids.device),
            use_cache=True,
        )
    return output.past_key_values, ids


def paired_caches(
    source: ModelBundle,
    target: ModelBundle,
    transcripts: Iterable[str],
    source_states: Iterable[CacheState] | None = None,
) -> Iterable[tuple[Any, Any]]:
    """Yield caches for identical text, then release each pair after fitting consumes it."""
    texts = list(transcripts)
    states = list(source_states) if source_states is not None else [None] * len(texts)
    if len(states) != len(texts):
        raise ValueError("source states must match transcripts")
    for text, state in zip(texts, states, strict=True):
        if state is None:
            source_cache, _ = collect_cache(source, text)
        else:
            source_cache = state.past_key_values
        target_cache, _ = collect_cache(target, text)
        yield source_cache, target_cache
        del source_cache, target_cache
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _cold_probe(
    target: ModelBundle, prefix_ids: torch.Tensor, suffix_ids: torch.Tensor, max_tokens: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate probe tokens and retain the cold distribution predicting each token."""
    model = target.model
    full_ids = torch.cat((prefix_ids, suffix_ids), dim=-1)
    mask = torch.ones_like(full_ids)
    with torch.inference_mode():
        output = model(
            input_ids=full_ids,
            attention_mask=mask,
            cache_position=torch.arange(full_ids.shape[-1], device=full_ids.device),
            use_cache=True,
        )
    logits: list[torch.Tensor] = []
    tokens: list[torch.Tensor] = []
    eos = target.tokenizer.eos_token_id
    eos_ids = {int(eos)} if isinstance(eos, int) else {int(token) for token in (eos or [])}
    for index in range(max_tokens):
        distribution = output.logits[:, -1]
        token = distribution.argmax(dim=-1, keepdim=True)
        if int(token.item()) in eos_ids:
            break
        logits.append(distribution)
        tokens.append(token)
        mask = torch.cat((mask, mask.new_ones((1, 1))), dim=-1)
        with torch.inference_mode():
            output = model(
                input_ids=token,
                attention_mask=mask,
                past_key_values=output.past_key_values,
                cache_position=torch.tensor([full_ids.shape[-1] + index], device=full_ids.device),
                use_cache=True,
            )
    if not tokens:
        raise RuntimeError("cold probe generated no non-EOS tokens")
    return torch.stack(logits, dim=1), torch.cat(tokens, dim=-1)


def _adapted_probe(
    target: ModelBundle,
    adapted_cache: Any,
    prefix_length: int,
    suffix_ids: torch.Tensor,
    teacher_tokens: torch.Tensor,
) -> torch.Tensor:
    """Teacher-force the cold tokens through the adapted cache."""
    model = target.model
    total = prefix_length + suffix_ids.shape[-1]
    mask = torch.ones((1, total), dtype=torch.long, device=suffix_ids.device)
    with torch.inference_mode():
        output = model(
            input_ids=suffix_ids,
            attention_mask=mask,
            past_key_values=adapted_cache,
            cache_position=torch.arange(prefix_length, total, device=suffix_ids.device),
            use_cache=True,
        )
    logits = [output.logits[:, -1]]
    for index in range(teacher_tokens.shape[-1] - 1):
        token = teacher_tokens[:, index : index + 1]
        mask = torch.cat((mask, mask.new_ones((1, 1))), dim=-1)
        with torch.inference_mode():
            output = model(
                input_ids=token,
                attention_mask=mask,
                past_key_values=output.past_key_values,
                cache_position=torch.tensor([total + index], device=suffix_ids.device),
                use_cache=True,
            )
        logits.append(output.logits[:, -1])
    return torch.stack(logits, dim=1)


def mean_kl(cold_logits: torch.Tensor, adapted_logits: torch.Tensor) -> float:
    if cold_logits.shape != adapted_logits.shape:
        raise ValueError("cold and adapted logits must have equal shapes")
    cold_log = F.log_softmax(cold_logits.float(), dim=-1)
    adapted_log = F.log_softmax(adapted_logits.float(), dim=-1)
    return float((cold_log.exp() * (cold_log - adapted_log)).sum(dim=-1).mean().item())


def calibrate_adapter(
    adapter: LinearKVAdapter,
    source: ModelBundle,
    target: ModelBundle,
    fit_transcripts: list[str],
    validation_transcripts: list[str],
    next_instruction: str,
    artifact_dir: str | Path,
    *,
    validation_source_states: list[CacheState] | None = None,
    fit_source_states: list[CacheState] | None = None,
    max_positions: int = 64,
    probe_tokens: int = 16,
    threshold: float = 0.15,
) -> dict[str, Any]:
    """Fit, evaluate on held-out hand-offs, freeze the gate decision, and save."""
    started = time.perf_counter()
    for state in (fit_source_states or []) + (validation_source_states or []):
        if state.model_name != adapter.source_model:
            raise ValueError("calibration source state belongs to the wrong model")
    adapter.fit(
        paired_caches(source, target, fit_transcripts, fit_source_states),
        source.model.config,
        target.model.config,
        max_positions=max_positions,
    )
    if validation_source_states is not None and len(validation_source_states) != len(
        validation_transcripts
    ):
        raise ValueError("validation source states must match validation transcripts")

    scores: list[float] = []
    for index, transcript in enumerate(validation_transcripts):
        if validation_source_states is None:
            source_cache, _ = collect_cache(source, transcript)
        else:
            state = validation_source_states[index]
            if state.model_name != adapter.source_model:
                raise ValueError("validation source state belongs to the wrong model")
            source_cache = state.past_key_values
        prefix_ids, _ = tokenize(target, transcript, special=True)
        suffix_ids, _ = tokenize(target, next_instruction, special=False)
        adapter.prepare_target(prefix_ids)
        adapted_cache = adapter.adapt(source_cache, source.model.config, target.model.config)
        cold_logits, teacher_tokens = _cold_probe(target, prefix_ids, suffix_ids, probe_tokens)
        adapted_logits = _adapted_probe(
            target, adapted_cache, prefix_ids.shape[-1], suffix_ids, teacher_tokens
        )
        scores.append(mean_kl(cold_logits, adapted_logits))

    score = sum(scores) / len(scores) if scores else float("inf")
    adapter.set_quality(score, threshold, len(scores))
    adapter.save(artifact_dir)
    return {
        "metric": "mean_kl_cold_to_adapted",
        "score": score,
        "threshold": threshold,
        "approved": adapter.approved,
        "calibration_seconds": time.perf_counter() - started,
        "per_example_scores": scores,
    }
