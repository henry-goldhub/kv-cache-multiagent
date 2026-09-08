# KVBridge

KVBridge is a deliberately small experiment in reusing transformer KV caches
across an append-only, three-step GSM8K pipeline:

```text
question -> extract -> plan -> compute -> #### answer
```

The claim is intentionally narrow:

> I reuse previously computed keys and values exactly when the model stays the
> same. When the model changes, I try a small learned conversion. I validate
> that conversion on separate examples and fall back when it changes predictions
> too much.

Cross-model transfer is an experiment, not a guaranteed optimization. A result
where both adapters are rejected is useful evidence about this method.

## What is cached

At each transformer layer, attention projects every processed token into a key
and value. Hugging Face stores one tensor of each kind with shape:

```text
[batch, KV heads, tokens, head dimension]
```

With batch size one, the expected shapes for the pinned model revisions are:

| Model | Layers | KV heads | Head dimension | One key/value layer |
| --- | ---: | ---: | ---: | --- |
| Qwen2.5-7B | 28 | 4 | 128 | `[1, 4, tokens, 128]` |
| Mistral-7B v0.3 | 32 | 8 | 128 | `[1, 8, tokens, 128]` |
| Phi-3.5-mini | 32 | 32 | 96 | `[1, 32, tokens, 96]` |

Keys and values exist at every layer, so a complete Qwen cache is 28 pairs of
the first shape. Future tokens can attend to those saved tensors instead of
recomputing the prefix.

For background, see Hugging Face's [cache explanation](https://huggingface.co/docs/transformers/main/cache_explanation)
and the [Prompt Cache paper](https://arxiv.org/abs/2311.04934).

Suppose step 1 has 100 prompt tokens and generates 20 tokens. Its completed
cache represents exactly 120 tokens. If the next instruction is 12 tokens and
the same model handles step 2, KVBridge passes the 120-token cache plus only the
12 new tokens. The cold baseline processes all 132 tokens. Both paths construct
the same logical token sequence.

The generated stop token is not retained in the transcript or cache. Every
retained output token is explicitly forwarded once, so `len(token_ids)` equals
the cache length.

## The cross-model hypothesis

A Qwen cache cannot be passed directly to Mistral because its shape and learned
coordinate system differ. `LinearKVAdapter` applies four understandable steps:

1. Map target layers to the nearest source layer at the same relative depth.
2. Linearly interpolate the token axis to the target tokenizer's prefix length.
3. Average proportional KV-head groups when shrinking, or repeat them when growing.
4. Apply a learned affine map to every head vector: `target = source @ W + b`.

There are separate `W` and `b` values for every target layer and for keys and
values. A layer shares its projection across heads.

For a tiny example, if a source key is `x = [2, 3]`, then

```text
W = [[1, 0],    b = [1, -1]
     [0, 2]]

x @ W + b = [3, 5]
```

Calibration obtains source and cold-target caches for the same transcript and
fits `W` and `b` with ridge regression. It uses 32 GSM8K training questions and
at most 64 positions from each cache. Pretrained model weights never change.
The source side is the exact cache retained from the real extraction or planning
run, so calibration does not recreate generated tokens by re-tokenizing them.

The assumptions are weak. Relative-depth layers need not perform equivalent
work. Interpolation aligns relative token positions, not pieces of text. Keys
also include model-specific positional transformations. These are expected
failure modes rather than implementation details to hide.

## Quality gate and fallback

Another 32 training questions are held out from fitting. For each realistic
plan or compute hand-off, the cold target generates up to 16 tokens. KVBridge
teacher-forces those same tokens through the adapted cache and compares the two
next-token distributions:

```text
mean KL(cold target || adapted target)
```

An adapter is approved only when the mean is finite and at most `0.15`. That is
a frozen experimental tolerance, not a claim that KL guarantees GSM8K accuracy.
The decision is stored with the weights. Timed inference reads that decision;
it never runs a hidden cold quality check. Missing, rejected, or invalid
adapters trigger a logged cold prefill.

Example rejected hand-off:

```text
Qwen cache -> resize/project -> KL 1.20 > 0.15 -> reject -> Mistral cold prefill
```

## Install and run on a Colab T4

The experiment pins the versions and model revisions used by the prior hardware
feasibility run, but produces its own results. Python 3.11–3.13 is supported.

```bash
pip install -r requirements-colab.lock
pip install -e .
python examples/calibrate.py
python examples/quickstart.py --num-examples 5 --split train --output-dir results/smoke
python examples/quickstart.py --num-examples 200 --split test --output-dir results/final
```

All three models use 4-bit NF4 weights, float16 computation, eager attention,
batch size one, and one CUDA device. Model loading prints allocated memory and
turns CUDA out-of-memory into an explicit hardware failure. It does not silently
substitute models or offload layers.

The notebook at `examples/colab.ipynb` contains only setup and calls to these
scripts. Download `results/` before the Colab runtime ends.

## Public API

```python
from kvbridge import Pipeline, evaluate

pipeline = Pipeline(models, config)
result, logs = pipeline.run(task_input, step_assignment)
report = evaluate(pipeline, dataset, step_assignments, config)
```

The two cache policies are `disabled` and `reuse`. Disabled means cold prefill
at each pipeline stage; ordinary autoregressive decode caching remains enabled.
Configuration defaults are seed 42, 128 generated tokens per stage, and a 2,048
token context limit. Over-limit examples fail explicitly and are never silently
truncated.

## Evaluation and result interpretation

The final script evaluates a seed-42 subset of 200 GSM8K test questions under:

```text
Qwen -> Qwen -> Qwen
Mistral -> Mistral -> Mistral
Phi -> Phi -> Phi
Qwen -> Mistral -> Phi
```

Each setting runs both policies. Their order alternates per question to reduce
timing bias. CUDA is synchronized around prefill, decode, and adaptation. The
report includes exact-match accuracy with Wilson intervals, paired answer
disagreements, mean/median total latency, component timings, cache-token rates,
fallback rate, and a paired bootstrap interval for total speedup. Every raw
record is appended to `records.jsonl` before the next run.

Before evaluation, `check_same_model_reuse` compares cold and reused logits on
the same tokens. Evaluation stops if any top-1 token differs or the maximum
absolute logit difference exceeds `0.1`. Unexplained answer disagreements also
block a correctness claim.

No final T4 experiment has been run from this fresh repository yet. The scripts
write measured results to `results/final/report.json` and `report.md`; this README
must only be updated with those numbers after that run. Results from MLTask are
not evidence for this implementation.

Reduced prefill time may produce only a small total speedup because generation
is often dominated by one-token-at-a-time decoding. Report both numbers.

## Tests

```bash
pytest
ruff check kvbridge tests examples
```

The unit tests use deterministic CPU fakes. They cover exact cache accounting,
final-token bookkeeping, teacher-forced logit equivalence, adapter shapes and
finite values, recovery of a known affine map, forced and missing-adapter
fallbacks, numeric parsing, and report aggregation. The five-example cloud run
is the integration smoke test for the pinned real models.
