# GPT-2 Activation Capture Format

This document describes the JSON payload emitted by `scripts/gpt2_capture.py`. The
format is designed for compact transport over a CDN and efficient parsing inside
the Three.js visualiser.

## Top-level layout

Each capture serialises to UTF-8 JSON with the following keys:

| Key | Type | Description |
| --- | ---- | ----------- |
| `meta` | object | Static metadata (model hyperparameters and sampling strides). |
| `tokens` | int[] | GPT-2 BPE token ids for the prompt + chosen continuation. |
| `prompt_token_count` | int | Number of prompt tokens (continuation starts here). |
| `prompt` / `completion` | string | Human-readable strings for convenience. |
| `embeddings` | object | Sampled token/position/sum vector states. |
| `layers` | object[] | Per-layer activation snapshots (one per transformer block). |
| `final_layernorm` | object | Final layer norm states (norm/scale/shift). |
| `logits` | packed array | Quantised LM head logits for the last token. |
| `vocab_size` | int | Model vocabulary size. |
| `seq_len` | int | Total sequence length captured. |
| `sampling_trace` | object[] | Per-step sampler diagnostics (see below). |
| `sampling_config` | object | Temperature / top-k / top-p / max-new-tokens used. |
| `generation_flops` | object[] | FLOP checkpoints for each generated token. |

All numeric tensors are stored via the *packed array* helper described next.

## Packed arrays

To keep files small, multi-dimensional arrays are quantised and stored as
base64-encoded binary blobs. Each packed array has the shape:

```json
{
  "encoding": "base64" | "raw",
  "dtype": "float16" | "int8" | "float32",
  "shape": [dim0, dim1, ...],
  "data": "...",
  "scale": <float, optional when dtype == int8>
}
```

* `encoding = "raw"` means the payload is an array of JSON numbers (used when
  `--quant none` is selected). Otherwise `data` is base64 and should be decoded
  then reinterpreted as the specified dtype in row-major order.
* For `int8` quantisation the payload captures symmetric values with
  `real_value ≈ int8_value * scale`.
* Arrays always have batch dimension first. Examples: residual snapshots are
  shaped `[seq_len, samples]`, per-head Q/K/V states are `[n_heads, seq_len, samples]`,
  and flattened causal attention is `[n_heads, tri_count]` where
  `tri_count = seq_len * (seq_len + 1) / 2`.

The capture metadata echoes the quantisation mode selected via `--quant` so the
viewer can pick the correct decode path.

## Embedding states

`embeddings` contains three packed arrays sampled with stride `meta.stride_residual`:

* `token`: token embedding output (`wte(input_ids)`)
* `position`: position embedding output (`wpe(position_ids)`)
* `sum`: elementwise sum of token + position embeddings (residual stream input)

Sampling stride defaults to 32 (capturing 24 values from a 768-dim vector).

## Per-layer structure

Each entry of `layers` is an object with:

* `residual_in`: residual stream just before layer `i` (shared with previous layer output)
* `ln1_norm`, `ln1_scale`, `ln1_shift`: layer-norm states before MHSA
* `q_heads`, `k_heads`, `v_heads`: sampled 64-dim head vectors for every token (stride `meta.stride_qkv`)
* `attention_pre_softmax`, `attention_post_softmax`: packed `[n_heads, tri_count]`
  causal triangles (`tri_count = seq_len*(seq_len+1)/2`). `attention_tri_seq_len`
  records the sequence length used for reconstruction.
* `attn_out_proj`: 768-dim vector after the output projection matrix
* `residual_after_attn`: residual stream after the first skip connection
* `ln2_norm`, `ln2_scale`, `ln2_shift`: layer-norm states before the MLP
* `mlp_up_proj`: sampled 3072-dim activations after the up-projection
* `mlp_activation`: sampled 3072-dim activations after GELU
* `mlp_down_proj`: sampled 768-dim activations after the down-projection
* `residual_after_mlp`: residual stream handed to the next layer

All vectors are sampled along their feature dimension using the configured stride.

## Final layernorm and logits

`final_layernorm` mirrors the per-layer layout (norm/scale/shift). The raw
vocabulary logits for the final token are stored in `logits` using the same
quantisation wrapper.

## Sampling trace payload

`scripts/gpt2_capture.py` always recomputes a sampler trace for the final token
sequence, regardless of whether it was auto-generated or entered manually. Each
entry of `sampling_trace` corresponds to one generated token and contains:

```json
{
  "candidate_ids": [int, ...],
  "logits": <packed array shape [1, K]>,
  "probs": <packed array shape [1, K]>,
  "chosen_token": int
}
```

`candidate_ids` lists every token considered by the configured top-k/top-p filter.
When the user-provided continuation contains ids outside that set, the tool
appends the chosen id and its logit so the viewer can still highlight it.

## FLOP checkpoints

`generation_flops` is an array with one entry per generated token (prompt tokens
are excluded because we assume they are pre-fed). Each entry is a dictionary
mapping checkpoint names to cumulative FLOPs. Names match the stage list used in
the visualisation, e.g.:

```
after_embeddings
layer_00/ln1_norm
layer_00/ln1_scale
layer_00/ln1_shift
layer_00/qkv_proj
layer_00/self_attention_scores
layer_00/self_attention_context
layer_00/self_attention_concat
layer_00/attn_output_proj
layer_00/residual_add_1
layer_00/ln2_norm
layer_00/ln2_scale
layer_00/ln2_shift
layer_00/mlp_up_proj
layer_00/mlp_activation
layer_00/mlp_down_proj
layer_00/residual_add_2
...
final_ln_norm
final_ln_scale
final_ln_shift
lm_head
```

The estimator relies on analytic formulas (matrix multiply ≈ `2MNK` FLOPs,
layernorm ≈ `6 * hidden_size`, etc.). While approximate, it is consistent across
captures so relative comparisons remain meaningful.

## Parameter checkpoints

`gpt2_parameter_report.json` uses the same checkpoint names as the FLOP trace
and stores cumulative parameter counts after each stage. LayerNorm weights are
attributed to the scale stage, biases to the shift stage, and projection matrices
are counted when their stage first appears.

## Customisation knobs

* `meta.stride_residual`, `meta.stride_qkv`, and `meta.stride_mlp` mirror the
  CLI `--stride`, `--qkv-stride`, and `--mlp-stride` arguments. Adjust these to
  trade granularity for file size.
* `meta.quantisation` reflects the `--quant` flag (`float16` by default). The
  CLI also supports `--quant int8` and `--quant none`.
* Future compression strategies can extend the packed-array wrapper while
  keeping the rest of the schema untouched.

## Reconstructing attention matrices

The triangular attention stores rows concatenated in order `t = 0..T-1`. Given
`tri_len = seq_len*(seq_len+1)/2`, you can recover row `t` by slicing the range
`offset = t*(t+1)/2` to `offset + t + 1` and reshaping back to `[t+1]`.

## Versioning

`meta.model_name` captures the Hugging Face identifier that produced the
activations (currently `"gpt2"`). Any future schema revisions should bump a
separate version field inside `meta` so the frontend can branch accordingly.
