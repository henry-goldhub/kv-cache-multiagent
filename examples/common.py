"""Cloud-only model and dataset setup shared by the two small entry points."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from kvbridge import LinearKVAdapter, ModelBundle


MODELS = {
    "model_1": (
        "Qwen/Qwen2.5-7B-Instruct",
        "a09a35458c702b33eeacc393d103063234e8bc28",
    ),
    "model_2": (
        "mistralai/Mistral-7B-Instruct-v0.3",
        "c170c708c41dac9275d15a8fff4eca08d52bab71",
    ),
    "model_3": (
        "microsoft/Phi-3.5-mini-instruct",
        "2fe192450127e6a83f7441aef6e3ca586c338b77",
    ),
}

ASSIGNMENTS = [
    ["model_1", "model_1", "model_1"],
    ["model_2", "model_2", "model_2"],
    ["model_3", "model_3", "model_3"],
    ["model_1", "model_2", "model_3"],
]


def load_models() -> dict[str, ModelBundle]:
    if not torch.cuda.is_available():
        raise RuntimeError("The named-model experiment requires a CUDA GPU")
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    bundles = {}
    try:
        for alias, (model_id, revision) in MODELS.items():
            tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                revision=revision,
                device_map={"": 0},
                quantization_config=quantization,
                dtype=torch.float16,
                attn_implementation="eager",
            )
            model.eval()
            bundles[alias] = ModelBundle(model, tokenizer, revision)
    except torch.OutOfMemoryError as error:
        allocated = torch.cuda.max_memory_allocated()
        raise RuntimeError(
            f"Named models do not fit this GPU; peak allocated bytes: {allocated}"
        ) from error
    print(
        json.dumps(
            {
                "gpu": torch.cuda.get_device_name(0),
                "loaded_models": list(bundles),
                "allocated_bytes": torch.cuda.memory_allocated(),
                "reserved_bytes": torch.cuda.memory_reserved(),
            },
            indent=2,
        )
    )
    return bundles


def load_adapters(directory: str | Path) -> dict[str, LinearKVAdapter]:
    directory = Path(directory)
    result = {}
    for source, target in (("model_1", "model_2"), ("model_2", "model_3")):
        manifest = directory / f"{source}__to__{target}.json"
        weights = directory / f"{source}__to__{target}.safetensors"
        if manifest.exists() and weights.exists():
            result[f"{source}->{target}"] = LinearKVAdapter.load(directory, source, target)
    return result
