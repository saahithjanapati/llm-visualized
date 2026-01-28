# AGENTS.md

## Repo summary
This repo is a Three.js-based 3D visualization of GPT-2 internals. The primary demo is a 12-layer GPT-2 stack with layered animations (LN, MHSA, MLP), backed by a custom engine, animation system, and precomputed geometry assets. Vite is used for local dev/build and a set of HTML test pages live under `tests/`.

## Key entry points
- `index.html` - Main HTML entry that loads the 12-layer demo via `tests/twelve-layer-stack.js`.
- `tests/twelve-layer-stack.html` - Demo page wiring overlays, settings, and canvas setup.
- `tests/twelve-layer-stack.js` - Bootstraps the 12-layer pipeline and connects UI, timing, and scene setup.
- `main.js` - Minimal single-layer entry (useful for quick engine checks).

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
- `package.json` - Vite scripts; `predev` and `prebuild` run `generate_index.js`.
- `vite.config.js` - Vite inputs for test pages.
- `generate_index.js` / `index.template.html` - Generates `tests/index.html` menu of demo pages.

## Tests and demos
- `tests/` - Many HTML demo pages for specific components and animations.

If you need to change behavior, start with `tests/twelve-layer-stack.js` (demo wiring) or `src/engine/` + `src/animations/` (core visuals).
