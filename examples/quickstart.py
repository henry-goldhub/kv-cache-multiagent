"""Run the four required settings and save an auditable report."""

from __future__ import annotations

import argparse
from pathlib import Path

from kvbridge import Pipeline, evaluate
from kvbridge.evaluation import report_markdown

from common import ASSIGNMENTS, load_adapters, load_models


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-examples", type=int, default=200)
    parser.add_argument("--split", choices=("train", "test"), default="test")
    parser.add_argument("--adapter-dir", default="artifacts/adapters")
    parser.add_argument("--output-dir", default="results/final")
    args = parser.parse_args()

    from datasets import load_dataset

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    models = load_models()
    adapters = load_adapters(args.adapter_dir)
    dataset = load_dataset("openai/gsm8k", "main", split=args.split)
    config = {
        "cache_policy": "reuse",
        "adapters": adapters,
        "seed": 42,
        "max_new_tokens": 128,
        "context_limit": 2048,
        "eval_size": args.num_examples,
        "dataset_fingerprint": getattr(dataset, "_fingerprint", None),
        "warmup": True,
        "records_path": str(output / "records.jsonl"),
        "report_path": str(output / "report.json"),
    }
    pipeline = Pipeline(models, config)

    checks = {
        name: pipeline.check_same_model_reuse(
            name,
            "Question: Jo has 2 apples.",
            "\n\nHow many apples?\n",
        )
        for name in models
    }
    (output / "same_model_checks.json").write_text(
        __import__("json").dumps(checks, indent=2), encoding="utf-8"
    )
    failed_checks = {
        name: values
        for name, values in checks.items()
        if values["top1_agreement"] < 1.0
        or values["max_absolute_logit_difference"] > 0.1
    }
    if failed_checks:
        raise RuntimeError(
            "same-model cache correctness check failed; inspect same_model_checks.json"
        )
    report = evaluate(pipeline, dataset, ASSIGNMENTS, config)
    markdown = report_markdown(report)
    (output / "report.md").write_text(markdown, encoding="utf-8")
    print(markdown)


if __name__ == "__main__":
    main()
