# GPT-2 Activation Capture Toolkit Overview

This repository now ships with a data-extraction pipeline that turns live GPT-2
(124M) forward passes into compact artefacts for the llm-visualized scene. The
goal is to replace placeholder colours with faithful values sampled from the
model's residual stream, attention machinery, and MLP activations.

## What the new tooling does

* **Interactive prompt + sampler UI** – `scripts/gpt2_capture.py` lets you enter a
  seed prompt, preview multiple auto-completions (temperature / top-k / top-p
  configurable), trim the generated tokens, or provide your own continuation.
* **Activation harvesting** – once a continuation is chosen the script replays a
  full forward pass and records sampled snapshots of every major tensor in GPT-2:
  residual vectors, LayerNorm phases, per-head Q/K/V vectors, attention weights,
  MLP intermediates, and the final layer norm.
* **Sampler diagnostics** – regardless of whether a continuation was sampled or
  typed manually, the tool records the logits/probabilities for the configured
  top-k/top-p filter at each generation step (plus the actual chosen token).
* **Performance accounting** – FLOP estimates are tracked at each visualisation
  checkpoint for every generated token. A separate JSON report enumerates how
  many parameters have been touched after the same checkpoints.
* **Size-aware output** – activations are quantised (float16 by default, int8 or
  raw floats optional) and stored as base64-encoded binary blobs to keep CDN
  payloads light.

## Files added in this change

| Path | Purpose |
| ---- | ------- |
| `scripts/gpt2_capture.py` | Main CLI tool that orchestrates sampling, activation capture, quantisation, and report generation. |
| `docs/capture_data_format.md` | Formal specification of the JSON schema emitted by the capture script (packed arrays, checkpoints, trace format, etc.). |
| `docs/capture_tool_overview.md` | This friendly overview that distils the original brief and highlights the key features. |

## Running the capture tool

```bash
python scripts/gpt2_capture.py \
  --prompt "The secret ingredient" \
  --num-completions 5 \
  --max-new-tokens 64 \
  --temperature 0.8 \
  --top-k 40 \
  --top-p 0.95 \
  --quant float16 \
--output ./captures/secret.json
```

> **Dependencies:** install `torch`, `transformers`, and `numpy` in your Python
> environment before running the script (e.g. `pip install torch transformers numpy`).

You will see playful console prompts to pick a completion, optionally trim it,
and confirm the capture. Two artefacts are produced by default:

* `capture.json` (or the file given via `--output`) – the activation payload
  described in the data format spec.
* `gpt2_parameter_report.json` – cumulative parameter counts per checkpoint.

## Customisation knobs

* **Sampling behaviour** – adjust `--temperature`, `--top-k`, `--top-p`,
  `--max-new-tokens`, and `--num-completions` to explore the completion space.
* **Quantisation** – toggle `--quant` between `float16`, `int8`, or `none`.
* **Sampling density** – `--stride`, `--qkv-stride`, and `--mlp-stride` control
  how many dimensions are retained from residual, attention-head, and MLP
  vectors respectively. Lower strides = richer colour detail; higher strides =
  smaller files.

## A note on FLOPs and parameters

FLOP tallies are analytic approximations tuned for consistency rather than exact
hardware cost. Parameter counts, on the other hand, are exact and stored once
per checkpoint because they never change between runs. Use these numbers to sync
UI counters with the simulation timeline.

Happy capturing! 🌈
