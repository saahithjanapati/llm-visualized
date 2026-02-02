# llm-visualized
3d visualization of gpt2

## Main GPT-2 tower demo
- `index.html`: production entrypoint for the 12-layer GPT-2 tower.
- `src/app/gpt-tower/index.js`: main orchestration for loading data, building the tower, and wiring UI.
- `src/app/gpt-tower/`: demo-specific modules (activation loading, token chips, top logit bars, controls).

## Core engine and visuals
- `src/engine/LayerPipeline.js`: orchestrates N layers and lane handoff.
- `src/engine/CoreEngine.js`: Three.js setup, render loop, controls, raycasting.
- `src/engine/layers/Gpt2Layer.js`: single transformer layer visuals and sequencing.
- `src/animations/`: MHSA + LayerNorm animation controllers.
- `src/components/`: visual primitives (vectors, matrices, layer norms).
- `src/utils/`: constants, colors, trail helpers, precomputed geometry loader.
- `src/ui/`: overlays and settings UI.

## Assets
- `public/twelve-layer-stack.css`: main demo styling.
- `precomputed_components_slice.glb` / `precomputed_components_qkv.glb`: cached geometry.
- `metal_grate_rusty_1k.gltf` / `rogland_clear_night_64.exr`: environment assets.
- `cool-fun.gif`: loading overlay asset.

## GPT-2 capture utility
The repository includes an interactive CLI (`scripts/extract_gpt2_data.py`) for sampling real GPT-2 activations.

```
pip install transformers torch  # if not already installed
python scripts/extract_gpt2_data.py --max-new-tokens 40 --num-completions 6 --output capture.json
```

Key flags:
- `--quantisation`: choose between `float16` (default), `int8`, or `float32` encodings for stored activations.
- `--residual-stride`, `--attention-stride`, `--mlp-stride`: control the sampling stride for 768-d residual vectors, 64-d head vectors, and 3072-d MLP activations respectively (default stride = 32).
- `--top-k` / `--top-p`: match the sampler used during completion generation so the stored logits line up with the visualiser's needs.
