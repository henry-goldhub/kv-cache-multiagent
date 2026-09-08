"""Append-only three-stage inference with explicit KV-cache reuse."""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Any

import torch

from .adapter import KVAdapter


STEP_INSTRUCTIONS = (
    "\n\nExtract the quantities and variables in the problem. Be concise.\nExtraction:\n",
    "\n\nUse the extraction above to write the arithmetic expression or equations. "
    "Do not compute the final answer yet.\nPlan:\n",
    "\n\nCompute the result from the plan. End with exactly `#### number`.\nComputation:\n",
)


@dataclass(frozen=True)
class ModelBundle:
    """The only wrapper used around a Hugging Face model and tokenizer."""

    model: Any
    tokenizer: Any
    revision: str = "unknown"


@dataclass
class CacheState:
    """A cache for the exact tokens represented by one transcript."""

    model_name: str
    token_ids: torch.Tensor
    transcript: str
    past_key_values: Any


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _model_device(model: Any) -> torch.device:
    return model.get_input_embeddings().weight.device


def _cache_position(start: int, length: int, device: torch.device) -> torch.Tensor:
    return torch.arange(start, start + length, device=device)


class Pipeline:
    """Run extract -> plan -> compute with disabled or reusable cross-step caches."""

    def __init__(self, models: dict[str, ModelBundle], config: dict[str, Any] | None = None):
        self.models = models
        self.config = config or {}
        self.cache_policy = self.config.get("cache_policy", "reuse")
        if self.cache_policy not in {"disabled", "reuse"}:
            raise ValueError("cache_policy must be 'disabled' or 'reuse'")
        self.seed = int(self.config.get("seed", 42))
        self.max_new_tokens = int(self.config.get("max_new_tokens", 128))
        self.context_limit = int(self.config.get("context_limit", 2048))
        self.adapters: dict[str, KVAdapter] = self.config.get("adapters", {})

    def run(self, task_input: str, step_assignment: list[str]) -> tuple[str, dict[str, Any]]:
        """Run exactly three stages, as required by the task's public API."""
        if len(step_assignment) != 3:
            raise ValueError("step_assignment must contain exactly three model names")
        result, logs, _ = self._execute(task_input, step_assignment)
        return result, logs

    def run_steps(
        self, task_input: str, step_assignment: list[str]
    ) -> tuple[str, dict[str, Any], CacheState | None]:
        """Run one to three stages; calibration uses this to create real hand-off text."""
        if not 1 <= len(step_assignment) <= 3:
            raise ValueError("step_assignment must contain one to three model names")
        return self._execute(task_input, step_assignment)

    def _execute(
        self, task_input: str, step_assignment: list[str]
    ) -> tuple[str, dict[str, Any], CacheState | None]:
        if any(name not in self.models for name in step_assignment):
            raise KeyError("step_assignment names a model that was not supplied")
        random.seed(self.seed)
        torch.manual_seed(self.seed)
        transcript = f"Question: {task_input.strip()}{STEP_INSTRUCTIONS[0]}"
        previous: CacheState | None = None
        output = ""
        logs: dict[str, Any] = {"steps": [], "failed": False}
        started = time.perf_counter()

        for stage, model_name in enumerate(step_assignment):
            output, state, step_log = self._stage(stage + 1, model_name, transcript, previous)
            logs["steps"].append(step_log)
            if state is None:
                logs["failed"] = True
                logs["failure_reason"] = step_log["failure_reason"]
                output = ""
                break
            transcript += output
            previous = state
            if stage + 1 < len(step_assignment):
                transcript += STEP_INSTRUCTIONS[stage + 1]

        if previous is not None:
            _sync(_model_device(self.models[previous.model_name].model))
        logs["total_seconds"] = time.perf_counter() - started
        logs["final_transcript"] = transcript
        return output.strip(), logs, previous

    def _tokenize(
        self, bundle: ModelBundle, text: str, *, add_special_tokens: bool
    ) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = bundle.tokenizer(
            text, return_tensors="pt", add_special_tokens=add_special_tokens
        )
        device = _model_device(bundle.model)
        ids = encoded["input_ids"].to(device)
        mask = encoded.get("attention_mask", torch.ones_like(ids)).to(device)
        return ids, mask

    def _stage(
        self, stage: int, model_name: str, transcript: str, previous: CacheState | None
    ) -> tuple[str, CacheState | None, dict[str, Any]]:
        bundle = self.models[model_name]
        model = bundle.model
        device = _model_device(model)
        same_model = previous is not None and previous.model_name == model_name

        if same_model:
            suffix = transcript[len(previous.transcript) :]
            suffix_ids, _ = self._tokenize(bundle, suffix, add_special_tokens=False)
            full_ids = torch.cat((previous.token_ids.to(device), suffix_ids), dim=-1)
            new_ids = suffix_ids
            prefix_tokens = previous.token_ids.shape[-1]
        elif previous is not None:
            suffix = transcript[len(previous.transcript) :]
            prefix_ids, _ = self._tokenize(bundle, previous.transcript, add_special_tokens=True)
            suffix_ids, _ = self._tokenize(bundle, suffix, add_special_tokens=False)
            full_ids = torch.cat((prefix_ids, suffix_ids), dim=-1)
            new_ids = full_ids
            prefix_tokens = int(prefix_ids.shape[-1])
        else:
            full_ids, _ = self._tokenize(bundle, transcript, add_special_tokens=True)
            new_ids = full_ids
            prefix_tokens = 0

        full_length = int(full_ids.shape[-1])
        log: dict[str, Any] = {
            "stage": stage,
            "model": model_name,
            "prompt_tokens": full_length,
            "processed_tokens": full_length,
            "exact_reused_tokens": 0,
            "transferred_tokens": 0,
            "prefill_seconds": 0.0,
            "decode_seconds": 0.0,
            "adapter_seconds": 0.0,
            "adapter_attempted": False,
            "adapter_accepted": False,
            "fallback_reason": None,
            "truncated": False,
            "context_limit_reached": False,
            "failure_reason": None,
        }
        if full_length >= self.context_limit:
            log["failure_reason"] = (
                f"context_limit_leaves_no_decode_room:{full_length}>={self.context_limit}"
            )
            return "", None, log

        past = None
        if previous is not None and self.cache_policy == "reuse":
            if same_model:
                past = previous.past_key_values
                log["exact_reused_tokens"] = int(previous.token_ids.shape[-1])
                log["processed_tokens"] = int(new_ids.shape[-1])
            else:
                key = f"{previous.model_name}->{model_name}"
                adapter = self.adapters.get(key)
                if adapter is None:
                    log["fallback_reason"] = "adapter_missing"
                else:
                    log["adapter_attempted"] = True
                    if not adapter.approved:
                        log["fallback_reason"] = adapter.rejection_reason
                    else:
                        adapter.prepare_target(full_ids[:, :prefix_tokens])
                        _sync(device)
                        adapter_started = time.perf_counter()
                        try:
                            adapted = adapter.adapt(
                                previous.past_key_values,
                                self.models[previous.model_name].model.config,
                                model.config,
                            )
                        except (RuntimeError, ValueError, TypeError) as error:
                            adapted = None
                            log["fallback_reason"] = f"adapter_error:{type(error).__name__}"
                        _sync(device)
                        log["adapter_seconds"] = time.perf_counter() - adapter_started
                        if adapted is not None:
                            past = adapted
                            new_ids = full_ids[:, prefix_tokens:]
                            log["adapter_accepted"] = True
                            log["transferred_tokens"] = prefix_tokens
                            log["processed_tokens"] = int(new_ids.shape[-1])

        attention_mask = torch.ones((1, full_length), dtype=torch.long, device=device)
        _sync(device)
        prefill_started = time.perf_counter()
        with torch.inference_mode():
            outputs = model(
                input_ids=new_ids,
                attention_mask=attention_mask,
                past_key_values=past,
                cache_position=_cache_position(full_length - new_ids.shape[-1], new_ids.shape[-1], device),
                use_cache=True,
            )
        _sync(device)
        log["prefill_seconds"] = time.perf_counter() - prefill_started

        generated, completed_cache, decode_seconds = self._greedy_decode(
            bundle,
            outputs,
            attention_mask,
            full_length,
            min(self.max_new_tokens, self.context_limit - full_length),
        )
        log["decode_seconds"] = decode_seconds
        log["context_limit_reached"] = (
            generated.shape[-1] == self.context_limit - full_length
            and generated.shape[-1] < self.max_new_tokens
        )
        complete_ids = torch.cat((full_ids, generated), dim=-1)
        raw_output = bundle.tokenizer.decode(
            generated[0], skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        state = CacheState(model_name, complete_ids, transcript + raw_output, completed_cache)
        return raw_output, state, log

    def _greedy_decode(
        self,
        bundle: ModelBundle,
        outputs: Any,
        attention_mask: torch.Tensor,
        prefix_length: int,
        max_tokens: int,
    ) -> tuple[torch.Tensor, Any, float]:
        model, tokenizer = bundle.model, bundle.tokenizer
        device = attention_mask.device
        eos = getattr(tokenizer, "eos_token_id", None)
        eos_ids = {int(eos)} if isinstance(eos, int) else set(eos or [])
        generated: list[torch.Tensor] = []
        cache = outputs.past_key_values
        _sync(device)
        started = time.perf_counter()
        for index in range(max_tokens):
            token = outputs.logits[:, -1:].argmax(dim=-1)
            if int(token.item()) in eos_ids:
                break
            generated.append(token)
            attention_mask = torch.cat((attention_mask, attention_mask.new_ones((1, 1))), dim=-1)
            with torch.inference_mode():
                outputs = model(
                    input_ids=token,
                    attention_mask=attention_mask,
                    past_key_values=cache,
                    cache_position=torch.tensor([prefix_length + index], device=device),
                    use_cache=True,
                )
            cache = outputs.past_key_values
        _sync(device)
        elapsed = time.perf_counter() - started
        if generated:
            return torch.cat(generated, dim=-1), cache, elapsed
        return torch.empty((1, 0), dtype=torch.long, device=device), cache, elapsed

    def warmup(self) -> None:
        """Run one untimed forward for each loaded model before benchmarking."""
        for bundle in self.models.values():
            ids, mask = self._tokenize(bundle, "Warmup", add_special_tokens=True)
            with torch.inference_mode():
                bundle.model(input_ids=ids, attention_mask=mask, use_cache=True)
            _sync(ids.device)

    def check_same_model_reuse(self, model_name: str, prefix: str, suffix: str) -> dict[str, Any]:
        """Compare cold and reused suffix logits for a deterministic correctness check."""
        bundle = self.models[model_name]
        model = bundle.model
        prefix_ids, _ = self._tokenize(bundle, prefix, add_special_tokens=True)
        suffix_ids, _ = self._tokenize(bundle, suffix, add_special_tokens=False)
        full_ids = torch.cat((prefix_ids, suffix_ids), dim=-1)
        full_mask = torch.ones_like(full_ids)
        with torch.inference_mode():
            cold = model(
                input_ids=full_ids,
                attention_mask=full_mask,
                cache_position=_cache_position(0, full_ids.shape[-1], full_ids.device),
                use_cache=True,
            )
            prefix_output = model(
                input_ids=prefix_ids,
                attention_mask=torch.ones_like(prefix_ids),
                cache_position=_cache_position(0, prefix_ids.shape[-1], prefix_ids.device),
                use_cache=True,
            )
            reused = model(
                input_ids=suffix_ids,
                attention_mask=full_mask,
                past_key_values=prefix_output.past_key_values,
                cache_position=_cache_position(
                    prefix_ids.shape[-1], suffix_ids.shape[-1], suffix_ids.device
                ),
                use_cache=True,
            )
        cold_logits = cold.logits[:, -suffix_ids.shape[-1] :].float()
        reused_logits = reused.logits.float()
        difference = (cold_logits - reused_logits).abs()
        return {
            "prefix_tokens": int(prefix_ids.shape[-1]),
            "suffix_tokens": int(suffix_ids.shape[-1]),
            "max_absolute_logit_difference": float(difference.max().item()),
            "mean_absolute_logit_difference": float(difference.mean().item()),
            "top1_agreement": float(
                (cold_logits.argmax(-1) == reused_logits.argmax(-1)).float().mean().item()
            ),
        }
