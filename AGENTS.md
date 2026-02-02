# AGENTS.md

## Repo summary
This repo is a Three.js-based 3D visualization of GPT-2 internals. The primary demo is a 12-layer GPT-2 stack with layered animations (LN, MHSA, MLP), backed by a custom engine, animation system, and precomputed geometry assets. Vite is used for local dev/build.

## Key entry points
- `index.html` - Main HTML entry for the GPT-2 tower.
- `src/app/gpt-tower/index.js` - Bootstraps the 12-layer pipeline and connects UI, timing, and scene setup.
- `src/app/gpt-tower/` - Demo-specific modules (activation loading, token chips, top logit bars, controls).

## Core engine and visuals
- `src/engine/CoreEngine.js` - Three.js scene setup, renderer, camera, render loop, and controls.
- `src/engine/LayerPipeline.js` - Orchestrates N layers and lane handoff.
- `src/engine/layers/Gpt2Layer.js` - Single transformer layer visuals and sequencing.
- `src/animations/` - MHSA and other animation controllers (routing, pass-through, attention).
- `src/components/` - Visual primitives (vectors, weight matrices, layer norms).
- `src/utils/` - Constants, colors, trail helpers, precomputed geometry loader.
- `src/ui/` - UI overlays (intro, pause, settings, status).
- `src/state/appState.js` - Shared runtime state.

## Assets and precomputed geometry
- `precomputed_components_slice.glb` / `precomputed_components_qkv.glb` - Cached geometry to speed startup (slice + full-depth QKV).
- `metal_grate_rusty_1k.gltf` / `rogland_clear_night_64.exr` - Environment assets.
- `cool-fun.gif` - Loading overlay asset.
- `public/` - Static assets and CSS (including `public/twelve-layer-stack.css`).

## GPT-2 data extraction utilities
- `scripts/extract_gpt2_data.py` - CLI for sampling GPT-2 activations for visualization.
- `scripts/generate_precomputed_components.mjs` - Generates cached geometry.

## Build and dev
- `package.json` - Vite scripts.
- `vite.config.js` - Vite config for the main entry.

If you need to change behavior, start with `src/app/gpt-tower/index.js` (demo wiring) or `src/engine/` + `src/animations/` (core visuals).
