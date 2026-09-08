"""Paired GSM8K evaluation with auditable per-example records."""

from __future__ import annotations

import json
import math
import platform
import random
import re
import statistics
from collections.abc import Iterable
from decimal import Decimal, InvalidOperation
from importlib.metadata import version
from pathlib import Path
from typing import Any

import torch

from .pipeline import Pipeline


ANSWER_PATTERN = re.compile(r"####\s*(-?(?:\d+(?:,\d{3})*|\d+)(?:\.\d+)?)")


def numeric_answer(text: str) -> str | None:
    """Parse only the requested final marker and normalize it with Decimal."""
    matches = ANSWER_PATTERN.findall(text)
    if not matches:
        return None
    try:
        value = Decimal(matches[-1].replace(",", ""))
    except InvalidOperation:
        return None
    normalized = format(value, "f")
    if "." in normalized:
        normalized = normalized.rstrip("0").rstrip(".")
    return normalized if normalized not in {"", "-0"} else "0"


def fixed_subset(dataset: Iterable[dict[str, Any]], size: int, seed: int = 42) -> list[dict[str, Any]]:
    rows = list(dataset)
    if size < 0 or size > len(rows):
        raise ValueError("subset size must be between zero and dataset size")
    indices = list(range(len(rows)))
    random.Random(seed).shuffle(indices)
    return [rows[index] for index in indices[:size]]


def _question_and_gold(row: dict[str, Any]) -> tuple[str, str]:
    question = row.get("question", row.get("task_input"))
    answer = row.get("answer", row.get("gold_answer"))
    if not isinstance(question, str) or not question.strip():
        raise ValueError("each row needs a non-empty question")
    gold = numeric_answer(str(answer))
    if gold is None:
        raise ValueError("each gold answer must contain a #### numeric marker")
    return question, gold


def _wilson(correct: int, total: int) -> list[float]:
    if total == 0:
        return [0.0, 0.0]
    z = 1.959963984540054
    p = correct / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [max(0.0, center - radius), min(1.0, center + radius)]


def _bootstrap_speedup(
    baseline: list[float], reuse: list[float], seed: int, samples: int = 2000
) -> list[float] | None:
    if not baseline or len(baseline) != len(reuse) or any(value <= 0 for value in reuse):
        return None
    generator = random.Random(seed)
    ratios = []
    for _ in range(samples):
        indices = [generator.randrange(len(baseline)) for _ in baseline]
        base_mean = statistics.fmean(baseline[index] for index in indices)
        reuse_mean = statistics.fmean(reuse[index] for index in indices)
        ratios.append(base_mean / reuse_mean)
    ratios.sort()
    return [ratios[int(0.025 * samples)], ratios[int(0.975 * samples)]]


def _record(
    policy: str,
    question: str,
    gold: str,
    result: str,
    logs: dict[str, Any],
) -> dict[str, Any]:
    prediction = numeric_answer(result)
    return {
        "policy": policy,
        "question": question,
        "gold": gold,
        "output": result,
        "prediction": prediction,
        "correct": prediction == gold,
        "logs": logs,
    }


def _summarize(records: list[dict[str, Any]], assignment: list[str]) -> dict[str, Any]:
    correct = sum(record["correct"] for record in records)
    totals = [record["logs"]["total_seconds"] for record in records]
    steps = [step for record in records for step in record["logs"]["steps"]]
    prompt_tokens = sum(step["prompt_tokens"] for step in steps)
    exact = sum(step["exact_reused_tokens"] for step in steps)
    transferred = sum(step["transferred_tokens"] for step in steps)
    handoffs = sum(assignment[index] != assignment[index - 1] for index in range(1, 3))
    handoff_attempts = handoffs * len(records)
    fallbacks = sum(step["fallback_reason"] is not None for step in steps)

    def mean(field: str) -> float:
        return statistics.fmean(step[field] for step in steps) if steps else 0.0

    return {
        "examples": len(records),
        "accuracy": correct / len(records) if records else 0.0,
        "accuracy_95ci": _wilson(correct, len(records)),
        "failed_examples": sum(record["logs"]["failed"] for record in records),
        "latency_seconds": {
            "total_mean": statistics.fmean(totals) if totals else 0.0,
            "total_median": statistics.median(totals) if totals else 0.0,
            "prefill_mean_per_stage": mean("prefill_seconds"),
            "decode_mean_per_stage": mean("decode_seconds"),
            "adapter_mean_per_stage": mean("adapter_seconds"),
        },
        "prompt_tokens": prompt_tokens,
        "exact_reused_tokens": exact,
        "transferred_tokens": transferred,
        "exact_reuse_rate": exact / prompt_tokens if prompt_tokens else 0.0,
        "transferred_token_rate": transferred / prompt_tokens if prompt_tokens else 0.0,
        "handoff_fallback_rate": fallbacks / handoff_attempts if handoff_attempts else 0.0,
    }


def evaluate(
    pipeline: Pipeline,
    dataset: Iterable[dict[str, Any]],
    step_assignments: list[list[str]],
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run a paired cold/reuse evaluation, alternating execution order."""
    settings = {**pipeline.config, **(config or {})}
    rows = list(dataset)
    size = int(settings.get("eval_size", len(rows)))
    rows = fixed_subset(rows, size, int(settings.get("seed", 42)))
    baseline = Pipeline(pipeline.models, {**settings, "cache_policy": "disabled"})
    reuse = Pipeline(pipeline.models, {**settings, "cache_policy": "reuse"})
    if settings.get("warmup", True):
        pipeline.warmup()

    record_path = settings.get("records_path")
    if record_path:
        destination = Path(record_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text("", encoding="utf-8")

    report: dict[str, Any] = {
        "seed": int(settings.get("seed", 42)),
        "eval_examples": len(rows),
        "dataset_fingerprint": settings.get("dataset_fingerprint"),
        "experiment": {
            "max_new_tokens": pipeline.max_new_tokens,
            "context_limit": pipeline.context_limit,
            "quality_threshold": 0.15,
            "models": {
                name: {
                    "model_id": getattr(bundle.model.config, "_name_or_path", "unknown"),
                    "revision": bundle.revision,
                }
                for name, bundle in pipeline.models.items()
            },
        },
        "runtime": {
            "python": platform.python_version(),
            "torch": version("torch"),
            "transformers": version("transformers"),
            "safetensors": version("safetensors"),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "gpu_total_bytes": (
                torch.cuda.get_device_properties(0).total_memory
                if torch.cuda.is_available()
                else None
            ),
        },
        "settings": {},
    }
    for assignment_index, assignment in enumerate(step_assignments):
        if len(assignment) != 3:
            raise ValueError("every assignment must contain three model names")
        records = {"disabled": [], "reuse": []}
        for example_index, row in enumerate(rows):
            question, gold = _question_and_gold(row)
            order = (
                (("disabled", baseline), ("reuse", reuse))
                if (assignment_index + example_index) % 2 == 0
                else (("reuse", reuse), ("disabled", baseline))
            )
            for policy, candidate in order:
                result, logs = candidate.run(question, assignment)
                item = _record(policy, question, gold, result, logs)
                records[policy].append(item)
                if record_path:
                    saved = {"assignment": assignment, **item}
                    with Path(record_path).open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(saved) + "\n")

        cold_summary = _summarize(records["disabled"], assignment)
        reuse_summary = _summarize(records["reuse"], assignment)
        cold_times = [item["logs"]["total_seconds"] for item in records["disabled"]]
        reuse_times = [item["logs"]["total_seconds"] for item in records["reuse"]]
        cold_mean = cold_summary["latency_seconds"]["total_mean"]
        reuse_mean = reuse_summary["latency_seconds"]["total_mean"]
        cold_prefill = cold_summary["latency_seconds"]["prefill_mean_per_stage"]
        reuse_prefill = reuse_summary["latency_seconds"]["prefill_mean_per_stage"]
        disagreements = sum(
            left["prediction"] != right["prediction"]
            for left, right in zip(records["disabled"], records["reuse"], strict=True)
        )
        report["settings"][" -> ".join(assignment)] = {
            "disabled": cold_summary,
            "reuse": reuse_summary,
            "paired_answer_disagreements": disagreements,
            "total_speedup": cold_mean / reuse_mean if reuse_mean else None,
            "prefill_speedup": cold_prefill / reuse_prefill if reuse_prefill else None,
            "total_speedup_95ci": _bootstrap_speedup(
                cold_times, reuse_times, int(settings.get("seed", 42)) + assignment_index
            ),
        }

    report_path = settings.get("report_path")
    if report_path:
        destination = Path(report_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def report_markdown(report: dict[str, Any]) -> str:
    headers = (
        "Setting",
        "Cold acc.",
        "Reuse acc.",
        "Disagreements",
        "Cold total (s)",
        "Reuse total (s)",
        "Reuse prefill (s)",
        "Reuse decode (s)",
        "Adapter (s)",
        "Exact reuse",
        "Transferred",
        "Fallback",
        "Total speedup",
    )
    lines = [f"| {' | '.join(headers)} |", f"| {' | '.join(['---'] * len(headers))} |"]
    for name, values in report["settings"].items():
        cold, reuse = values["disabled"], values["reuse"]
        speedup = values["total_speedup"]
        lines.append(
            "| "
            + " | ".join(
                (
                    name,
                    f"{cold['accuracy']:.3f}",
                    f"{reuse['accuracy']:.3f}",
                    str(values["paired_answer_disagreements"]),
                    f"{cold['latency_seconds']['total_mean']:.3f}",
                    f"{reuse['latency_seconds']['total_mean']:.3f}",
                    f"{reuse['latency_seconds']['prefill_mean_per_stage']:.3f}",
                    f"{reuse['latency_seconds']['decode_mean_per_stage']:.3f}",
                    f"{reuse['latency_seconds']['adapter_mean_per_stage']:.3f}",
                    f"{reuse['exact_reuse_rate']:.3f}",
                    f"{reuse['transferred_token_rate']:.3f}",
                    f"{reuse['handoff_fallback_rate']:.3f}",
                    "n/a" if speedup is None else f"{speedup:.2f}x",
                )
            )
            + " |"
        )
    return "\n".join(lines) + "\n"
