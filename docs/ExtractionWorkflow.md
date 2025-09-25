# GPT-2 Extraction Workflow Summary

The `extract_gpt2_data.py` script provides an end-to-end workflow for sampling
GPT-2 completions and exporting the activation bundle consumed by the
`llm-visualized` Three.js scenes.

## High-Level Flow

1. **Prompt input** – supply a starting string via `--prompt` or interactively.
2. **Candidate generation** – draw `--num-completions` sequences using
   configurable sampling controls (temperature, top-k, top-p, seed).
3. **Selection** – pick a completion, truncate it, or type a custom
   continuation directly in the CLI.
4. **Tracing run** – the script replays the combined prompt + completion
   through GPT-2 Small (124M) while capturing residual, attention, layer norm,
   and MLP vector states together with attention matrices.
5. **Metric capture** – cumulative parameter counts and per-token FLOP
   estimates are recorded at checkpoints matching the visualisation pipeline.
6. **Serialisation** – activations are sampled at a configurable stride,
   quantised (`none`, `float16`, or `int8`), and written to a JSON artefact that
   adheres to the spec documented in `docs/GPT2TraceFormat.md`.

## Customisation Hooks

* `--vector-stride` adjusts the sampling stride (default 32).
* `--quantization` chooses the storage format for sampled vectors.
* Generation knobs (`--temperature`, `--top-k`, `--top-p`, `--seed`) allow the
  same prompt to be reproduced or explored interactively.
* Output path is controlled via `--output`.

The payload is intentionally compact and omits expensive tensors (such as full
context vectors) in favour of reconstructible samples.  All sections include
enough metadata for client-side interpolation or recolouring.

