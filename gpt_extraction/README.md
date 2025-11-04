# GPT-2 Activation Capture Toolkit

This folder hosts utilities for extracting real GPT-2 (124M) activations for
the llm-visualized project. The key entry point is
`extract_gpt2_data.py`, an interactive CLI that lets you:

1. Enter a seed prompt.
2. Sample multiple GPT-2 completions with configurable temperature, top-k and
   top-p settings.
3. Inspect the candidates, optionally truncate them, or type your own
   completion.
4. Capture a richly annotated forward pass over the combined prompt and
   completion, including sampled residual stream vectors, attention scores,
   MLP activations, FLOP estimates and logit traces.

The output is a compact JSON artefact designed for direct consumption in the
browser visualisation. All tensors are quantised (defaults to `float16`) and
sampled with a configurable stride (defaults to 32) to keep payload sizes
small enough for CDN distribution.

## Quick start

```bash
python -m gpt_extraction.extract_gpt2_data \
  --prompt "Once upon a time" \
  --num-completions 6 \
  --max-new-tokens 24 \
  --top-k 40 \
  --top-p 0.95 \
  --quantization float16 \
  --stride 32 \
  --output captures/once_upon.json
```

During the run you will be shown the sampled completions. Choose one by
index, trim it to the desired length, or type your own follow-up text. When
the script finishes the requested capture appears at the path passed via
`--output` (directories are created automatically).

## Customisation knobs

* **Quantisation** – `--quantization` accepts `float32`, `float16` or `int8`.
  Integer mode performs symmetric quantisation with a per-array scale. Use it
  for ultra-small artefacts.
* **Sampling stride** – `--stride` controls how densely we sample each
  residual vector. The default stride of 32 matches the design brief
  (768/32 = 24 samples). Lower strides collect more points.
* **Sampling strategy** – mix-and-match temperature, top-k and top-p to mirror
  the settings used in the visual experience. The script records the
  resulting candidate logits at each generation step as well as in the final
  forward pass (see `DATA_SPEC.md`).
* **Device and seed** – use `--device cuda` for GPU acceleration (if
  available) and `--seed` for deterministic sampling.

## Output format

Full schema and decoding notes live in [`DATA_SPEC.md`](DATA_SPEC.md).

## Dependencies

The script relies on the Hugging Face `transformers` stack. Install the Python
requirements in your virtual environment before running the capture:

```bash
pip install torch transformers
```

The GPT-2 model weights are downloaded automatically on first use and cached
in the standard Hugging Face location (`~/.cache/huggingface`).

