# GPT-2 Trace Export Format

This document describes the structure of the JSON blobs produced by
`scripts/collect_gpt2_trace.py`.  Each blob contains the minimal activation
signals required by the Three.js visualisation along with bookkeeping metadata
that allows the renderer to reconstruct context vectors and colour maps.

## High level structure

```
{
  "meta": { ... },
  "prompt": { ... },
  "completion": { ... },
  "sequence": { ... },
  "activations": { ... },
  "flops": { ... },
  "parameters": { ... }
}
```

All tensors inside the payload are quantised and base64 encoded.  The quantised
representation is described in the "Quantised tensors" section below.

### `meta`

Metadata describing how the trace was produced:

| Field | Description |
| --- | --- |
| `model` | Hugging Face identifier or path that was used to load GPT-2 |
| `quantization` | Quantisation mode (`float32`, `float16`, `int8`) |
| `stride`, `head_stride`, `mlp_stride` | Sampling stride for residual, attention and MLP vectors |
| `generation` | Sampling settings used while producing the completion (max tokens, top-k/p, temperature, seed) |

### `prompt`

Stores the seed text and corresponding token identifiers:

```
{
  "text": "Once upon a time",
  "token_ids": [123, ...],
  "tokens": ["Once", "Ġupon", "Ġa", "Ġtime"]
}
```

### `completion`

The sampled (or manually supplied) continuation.  `steps` is the generation
trace captured during sampling.  Each step exposes the candidate logits used for
top-k / top-p sampling.

```
{
  "text": " there was...",
  "token_ids": [...],
  "tokens": [...],
  "steps": [
    {
      "step": 0,
      "token_id": 318,
      "token": "Ġthere",
      "candidates": [
        {"token_id": 318, "logit": -2.14},
        {"token_id": 257, "logit": -2.42},
        ...
      ]
    },
    ...
  ]
}
```

### `sequence`

Combined prompt + completion text/tokens (`token_ids` mirrors the tensor fed to
the collector).

### `activations`

The heart of the export.  Activations are organised into the following sections:

```
{
  "embeddings": {
    "token": QuantisedTensor,
    "position": QuantisedTensor,
    "summed": QuantisedTensor
  },
  "layers": [LayerActivations, ...],
  "final_layernorm": {
    "norm": QuantisedTensor,
    "scale": QuantisedTensor,
    "shift": QuantisedTensor
  }
}
```

Each `LayerActivations` entry contains:

```
{
  "residual_in": QuantisedTensor,
  "ln1": {
    "norm": QuantisedTensor,
    "scale": QuantisedTensor,
    "shift": QuantisedTensor
  },
  "attention": {
    "q": QuantisedTensor,
    "k": QuantisedTensor,
    "v": QuantisedTensor,
    "pre_softmax": QuantisedTensor,
    "post_softmax": QuantisedTensor,
    "post_projection": QuantisedTensor,
    "residual": QuantisedTensor
  },
  "ln2": { ... analogous ... },
  "mlp": {
    "up": QuantisedTensor,
    "activation": QuantisedTensor,
    "down": QuantisedTensor,
    "residual": QuantisedTensor
  }
}
```

All residual stream tensors are sampled every `stride` dimensions.  Q/K/V are
sampled every `head_stride` dimensions (defaults to `stride`).  The MLP
intermediate activations (dimension 3072) are sampled every `mlp_stride`
dimensions.

`pre_softmax` and `post_softmax` are exported with shape `(sequence_length,
num_heads, sequence_length)` and rely on a lower-triangular causal mask (the
metadata of the quantised tensor includes `"mask": "lower_triangular"`).

The per-head context vectors are deliberately omitted.  They can be recomputed at
render time from the stored attention weights and sampled value vectors using
``ctx[h, t, d] = Σ_s post_softmax[t, h, s] * V[t, h, d]`` for any sampled value
dimension `d`.

### `flops`

The FLOP book-keeping mirrors the structure of the activations and reports the
estimated floating point operations incurred by each checkpoint during the
forward pass.  Counts are approximate and help drive progress indicators in the
visualisation.

### `parameters`

Static parameter counts grouped by checkpoint.  This payload can be shared
across traces because the numbers depend only on the model architecture.

## Quantised tensors

Every tensor payload follows the schema:

```
{
  "data": "...",          // base64 encoded bytes
  "dtype": "float16",      // numpy dtype of the payload
  "shape": [T, ...],        // tensor shape
  "scale": 0.03125,         // optional, only for integer formats
  "zero_point": 0.0,        // optional, only for integer formats
  "meta": { ... }           // optional auxiliary information
}
```

For integer formats the decoder should reconstruct real values via
``value = scale * (int_value - zero_point)``.

## Workflow summary

1. Run `scripts/collect_gpt2_trace.py` to interactively sample completions and
   emit trace files.
2. Serve the JSON blob(s) to the front-end.  Each blob contains everything
   required to colour residual vectors, attention patterns and MLP stages.
3. Optional: call the script with `--parameters-output` once to generate a
   reusable parameter summary file.

