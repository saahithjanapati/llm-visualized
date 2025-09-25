# GPT-2 Trace Payload Format

This document captures the layout of the JSON artefacts produced by
`scripts/extract_gpt2_data.py`.  The format is designed so Three.js based
clients (and other tools) can stream the activations required by the
`llm-visualized` project without decoding large tensors at runtime.

## Overview

Running `extract_gpt2_data.py` results in a single JSON object with the
following top-level keys:

| Key | Description |
| --- | ----------- |
| `meta` | Model metadata, quantisation options, and sampling stride. |
| `prompt` | Original prompt string and its token identifiers. |
| `tokens` | Sequence (prompt + completion) token metadata. |
| `embeddings` | Token, position, and summed embedding vector states. |
| `layers` | Per-layer activation snapshots for residual, LN, attention, and MLP blocks. |
| `final_layernorm` | Final layer norm intermediary vector states. |
| `logits` | Sampling logits for the generated tokens (top-k or top-p depending on settings). |
| `parameters_used` | Cumulative parameter counts at each instrumentation checkpoint. |
| `flops_used` | FLOP estimates grouped by checkpoint for the generated tokens. |

All vectors are sampled at evenly spaced strides (default 32) along the last
dimension.  The same schema is used for 768-dim residual vectors, 64-dim head
vectors, and 3072-dim MLP activations.  Sampling stride and quantisation mode
are recorded in `meta` so the original signals can be reconstructed via
interpolation or simple scaling.

## Vector Encoding

Every sampled vector is stored as:

```json
{
  "data": [/* values */],
  "scale": 0.015625  // present only for int8 quantisation
}
```

The representation depends on the chosen quantisation mode:

* `none`: `data` is a float32 list.
* `float16`: `data` is a float16 list (serialised as JSON numbers).
* `int8`: `data` is an int8 list and `scale` holds the de-quantisation factor
  for that vector (value = `int8 * scale`).

## Embedding Section

```
"embeddings": {
  "token":   [vector_state_t0, vector_state_t1, ...],
  "position": [...],
  "sum":      [...]
}
```

Each entry is aligned with the token order in `tokens`.  The `token` and
`position` arrays describe the raw embedding lookup outputs, while `sum` is the
post-addition residual stream prior to entering the first layer.

## Layer Objects

Each layer entry contains the following keys:

```
{
  "index": 0,
  "residual_in": [ ... ],
  "ln1": {
    "norm":  [...],
    "scale": [...],
    "shift": [...]
  },
  "attention": {
    "q": [[... per head ...]],
    "k": [[... per head ...]],
    "v": [[... per head ...]],
    "pre_softmax": [[[row-wise lower triangle]]],
    "post_softmax": [[[row-wise lower triangle]]],
    "projection": [...],
    "residual":   [...]
  },
  "ln2": { ... },
  "mlp": {
    "up": [...],
    "act": [...],
    "down": [...],
    "residual": [...]
  }
}
```

### Attention Matrices

`pre_softmax` and `post_softmax` use a nested list representation.  For each
head, `pre_softmax[h][t]` contains the logits for row `t` with only causal
(`s ≤ t`) columns preserved, making it straightforward to reconstruct the
triangular band client-side.  Probabilities follow the same structure.

### Query/Key/Value Samples

`q`, `k`, and `v` are arrays of shape `[n_head][sequence_length]` containing
sampled vector states (64-dimensional head space sampled with the global
stride).

### Residual Streams

`residual_in`, `attention.residual`, and `mlp.residual` provide the state of the
768-d residual stream at the start of the layer, after the attention residual
addition, and after the MLP residual addition respectively.

## Final Layer Norm

```
"final_layernorm": {
  "norm": [...],
  "scale": [...],
  "shift": [...]
}
```

Identical structure to the per-layer layer norm snapshots, capturing the
post-transformer activations prior to the language modelling head.

## Parameter Usage

`parameters_used` is a dictionary keyed by checkpoint paths (e.g.
`"layer_03/attn/qkv"`).  Values are cumulative counts – e.g. the value for
`layer_03/attn/qkv` includes all parameters consumed up to and including that
projection.  These counts are deterministic for the GPT-2 124M weights and can
be reused across traces.

## FLOP Tracking

`flops_used` mirrors the checkpoint naming scheme, but each entry is a list of
integers, one per generated token (prompt tokens are excluded).  The values are
coarse estimates that assume autoregressive decoding with cached keys and
values; they provide a consistent relative scale for animation timing.

## Logit Snapshots

`logits` holds an array of `{ "position", "candidates" }` objects.  Each
`candidates` list contains the logits for either the configured top-k tokens or
the tokens retained under top-p sampling.  These values can be normalised on
the client for colour mapping or display.

## Token Metadata

`tokens` is aligned with the combined prompt + completion sequence.  Each entry
stores the GPT-2 token identifier and the raw string decoded without cleanup so
clients can match token surfaces exactly.

## Extensibility

The script records the sampling stride and quantisation strategy inside `meta`
(`"vector_stride"`, `"quantization"`).  Future extensions (for example storing
additional strides for specific tensors or alternative quantisers) can extend
`meta` and add new keys under the relevant sections without breaking backwards
compatibility – clients should ignore keys they do not recognise.

