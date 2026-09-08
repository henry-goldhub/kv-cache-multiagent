"""Fit and quality-gate the two adapters used by the mixed pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from kvbridge import LinearKVAdapter, Pipeline
from kvbridge.adapter import cache_layers
from kvbridge.calibration import calibrate_adapter
from kvbridge.evaluation import fixed_subset
from kvbridge.pipeline import CacheState, STEP_INSTRUCTIONS

from common import load_models


def cpu_state(state: CacheState) -> CacheState:
    layers = cache_layers(state.past_key_values)
    cache = tuple(
        (layer.key.detach().cpu().clone(), layer.value.detach().cpu().clone())
        for layer in layers
    )
    return CacheState(state.model_name, state.token_ids.cpu(), state.transcript, cache)


def contexts(
    pipeline: Pipeline,
    rows: list[dict[str, str]],
    assignment: list[str],
    *,
    keep_states_from: int,
) -> tuple[list[str], list[CacheState]]:
    transcripts, states = [], []
    for number, row in enumerate(rows, start=1):
        _, logs, state = pipeline.run_steps(row["question"], assignment)
        if logs["failed"] or state is None:
            raise RuntimeError(f"context generation failed at example {number}")
        transcripts.append(state.transcript)
        if number > keep_states_from:
            states.append(cpu_state(state))
        del state
        torch.cuda.empty_cache()
        print(f"context {number}/{len(rows)} for {' -> '.join(assignment)}")
    return transcripts, states


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", default="artifacts/adapters")
    parser.add_argument("--summary", default="results/calibration.json")
    args = parser.parse_args()

    from datasets import load_dataset

    models = load_models()
    dataset = load_dataset("openai/gsm8k", "main", split="train")
    rows = fixed_subset(list(dataset), 64, 42)
    base_config = {
        "cache_policy": "disabled",
        "seed": 42,
        "max_new_tokens": 128,
        "context_limit": 2048,
    }
    cold = Pipeline(models, base_config)
    cold.warmup()
    print(f"memory smoke-test peak bytes: {torch.cuda.max_memory_allocated()}")
    extraction_texts, extraction_states = contexts(
        cold, rows, ["model_1"], keep_states_from=0
    )

    first = LinearKVAdapter(
        "model_1",
        "model_2",
        models["model_1"].revision,
        models["model_2"].revision,
    )
    summary = {
        "experiment": {
            "dataset": "openai/gsm8k:main:train",
            "dataset_fingerprint": getattr(dataset, "_fingerprint", None),
            "seed": 42,
            "fit_examples": 32,
            "validation_examples": 32,
            "max_positions": 64,
            "probe_tokens": 16,
            "quality_threshold": 0.15,
            "model_revisions": {name: bundle.revision for name, bundle in models.items()},
        },
        "model_1->model_2": calibrate_adapter(
            first,
            models["model_1"],
            models["model_2"],
            extraction_texts[:32],
            extraction_texts[32:],
            STEP_INSTRUCTIONS[1],
            args.artifact_dir,
            fit_source_states=extraction_states[:32],
            validation_source_states=extraction_states[32:],
        )
    }
    del extraction_states

    runtime = Pipeline(
        models,
        {
            **base_config,
            "cache_policy": "reuse",
            "adapters": {"model_1->model_2": first},
        },
    )
    planning_texts, planning_states = contexts(
        runtime, rows, ["model_1", "model_2"], keep_states_from=0
    )
    second = LinearKVAdapter(
        "model_2",
        "model_3",
        models["model_2"].revision,
        models["model_3"].revision,
    )
    summary["model_2->model_3"] = calibrate_adapter(
        second,
        models["model_2"],
        models["model_3"],
        planning_texts[:32],
        planning_texts[32:],
        STEP_INSTRUCTIONS[2],
        args.artifact_dir,
        fit_source_states=planning_states[:32],
        validation_source_states=planning_states[32:],
    )
    destination = Path(args.summary)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
