# GPT-2 Capture Data Specification

This document describes the JSON artefacts emitted by
`gpt_extraction/extract_gpt2_data.py`. The format is optimised for the
three.js visualisation pipeline – everything is stored as compact,
quantised payloads that can be decoded in the browser without running any
Python code.

## File overview

A capture file is a single JSON object with the following top-level
fields:

| key | description |
| --- | ----------- |
| `prompt` | Original prompt text and token ids. |
| `completion` | Selected continuation text and token ids. |
| `generation_trace` | Optional step-by-step sampling trace. |
| `sequence_length` | Number of tokens processed in the capture (prompt + completion). |
| `quantization` | Quantisation mode used for all stored activations (`float32`, `float16`, or `int8`). |
| `stride` | Sampling stride for vector states. |
| `embeddings` | Sampled residual stream states before the first transformer block. |
| `layers` | Instrumentation data for each transformer block. |
| `final_layernorm` | Breakdown of the final layer normalisation. |
| `filtered_logits` | Per-position logit summaries matching the sampling strategy. |
| `flops` | Running FLOP counts at every capture checkpoint. |
| `parameter_checkpoints` | Static parameter usage table (identical for every capture). |

All numeric tensors are stored as quantised payloads to minimise size.

## Quantised arrays

Vector states, logits and attention matrices are stored using the
`QuantizedArray` schema:

```json
{
  "dtype": "float16",
  "shape": [T, N],
  "data": "<base64 encoded little-endian bytes>",
  "layout": "tokens x sample_dim"
}
```

* `dtype` – `float32`, `float16` or `int8`.
* `shape` – original tensor shape after sampling.
* `data` – base64 encoded raw bytes.
* `layout` – optional human-readable hint describing how to interpret the
  flattened payload.
* `scale` and `zero_point` – present only for `int8` tensors using symmetric
  quantisation (`value ≈ scale * (int8 - zero_point)`).

All tensors are sampled along their last dimension using the configured
stride (default: 32). For example, a 768-d residual vector produces 24
sampled values per token.

## Embedding block

```json
"embeddings": {
  "token": [QuantizedArray per token],
  "position": [QuantizedArray per token],
  "summed": [QuantizedArray per token]
}
```

Each list has `sequence_length` entries capturing the token embedding,
position embedding, and their sum (the residual stream input to layer 0).

## Transformer layers

`"layers"` is an array with one entry per GPT-2 block. Each entry contains:

* `residual_in` – residual stream state entering the block.
* `ln1` / `ln2` – dictionaries with `norm`, `scaled` and `shifted` vector
  states (before γβ, after γ and after β respectively).
* `attention` – payload with:
  * `q`, `k`, `v` – sampled head vectors (shape: heads × tokens).
  * `pre_softmax`, `post_softmax` – lower-triangular attention matrices for
    each head. The `values` field contains a `QuantizedArray` of the packed
    lower-triangular entries; `size` gives the sequence dimension so the
    matrix can be reconstructed.
  * `projected` – sampled 768-d vectors after the output projection.
* `residual_after_attn` – residual stream state after the first skip
  connection.
* `mlp` – dictionaries for the feed-forward network (`up`, `activated`,
  `down`).
* `residual_out` – output residual stream (input to the next block).

## Final layer norm

Structure mirrors the per-layer layer-norm breakdown:

```json
"final_layernorm": {
  "norm": [...],
  "scaled": [...],
  "shifted": [...]
}
```

## Logit summaries

`filtered_logits` reports, for each token position:

```json
{
  "position": 12,
  "token_id": 50257,
  "top_k": {
    "token_ids": [...],
    "logits": QuantizedArray,
    "probs": QuantizedArray
  },
  "top_p": {
    "token_ids": [...],
    "logits": QuantizedArray,
    "probs": QuantizedArray
  }
}
```

Only the filters enabled for the capture (top-k, top-p) are present. The
token ids correspond to rows in the shared GPT-2 vocabulary.

## Sampling trace

`generation_trace` is an ordered list of dictionaries (one per generated
token) mirroring the structure above and including the sampled token at
each step. This lets the visualiser replay the sampling decisions without
re-running the model.

## FLOP checkpoints

`flops` is a list of objects:

```json
{
  "name": "layer_00/attn/out_proj",
  "increment": 1.18e8,
  "cumulative": 2.34e8
}
```

The checkpoints follow the order requested in the project brief (embeddings
→ layer norm steps → attention → MLP → residual additions). FLOP counts are
analytical estimates based on dense matmul costs and element-wise
operations; they scale with sequence length and can be aggregated by name
in the front-end.

## Parameter checkpoints

`parameter_checkpoints` mirrors the FLOP checkpoints but records the number
of distinct parameters touched by each stage. Values are constant for every
capture because they depend only on the GPT-2 small architecture.

## Quantisation and strides

The capture metadata exposes two knobs for future extensions:

* `quantization` – default `float16`. Switch to `int8` for more aggressive
  compression or `float32` for lossless exports.
* `stride` – default `32`. Adjust to collect denser or sparser samples from
  the underlying tensors.

Any consumer should respect both fields when decoding the payload. The
`QuantizedArray` objects always encode their actual shape; stride only
describes how samples relate to the original tensors.

