# AGENTS.md

_Last reviewed: 2026-03-11._

## Repo Summary
This repo is a Three.js visualization of GPT-2 internals, centered on a 12-layer transformer stack demo.

## Primary Runtime Path
- `index.html` loads the tower demo.
- `src/app/gpt-tower/index.js` wires engine, pipeline, UI, and activation data.
- `src/engine/LayerPipeline.js` creates and sequences layers.
- `src/engine/layers/Gpt2Layer.js` drives LN -> MHSA -> MLP per-layer flow.
- `src/animations/MHSAAnimation.js` orchestrates attention routing/pass-through.
- `src/animations/mhsa/VisualSetup.js` builds static MHSA matrices and output projection visuals.

## Key Areas
- `src/engine/`: renderer, camera, layer orchestration, raycast helpers.
- `src/engine/layers/`: lane construction, phase logic, data-access helpers, watchdog/debug signature/logging helpers.
- `src/animations/` and `src/animations/mhsa/`: animation controllers and MHSA internals.
- `src/components/`: visual primitives (instanced vectors, matrices, layer norm geometry).
- `src/ui/`: DOM controls, overlays, and split selection-panel utility modules (copy, reveal, formatting, vector clone, constants).
- `src/utils/`: constants, trails, materials, activation metadata/precompute helpers, shared matrix-visual tuning helpers.
- `src/data/`: random/captured activation sources + parameter datasets.
- `scripts/`: dataset extraction and geometry-generation scripts.
- `public/`: static assets and CSS.

## Current Complexity Hotspots
- `src/ui/selectionPanel.js` (large UI + preview composition module).
- `src/engine/layers/Gpt2Layer.js` (large phase/state machine).
- `src/animations/MHSAAnimation.js` (broad lifecycle + fallback paths).

## Dev Commands
- `npm run dev`
- `npm run test`
- `npm run build`

## Build Notes
- `vite.config.js` uses `manualChunks` to keep app-owned runtime code split (`index`, `scene-runtime`, `ui-selection-panel`) and isolate large vendor domains (`vendor-three-core`, `vendor-three-examples`).

## Activation Capture Semantics
- `scripts/extract_gpt2_data.py` captures compact activations with stride sampling, not pooled 64-d buckets. The helper `sample_tensor()` selects coordinates `0, stride, 2 * stride, ...` along the last dimension.
- The current shipped captures (`public/capture.json`, `public/capture_2.json`) use `residual_stride=64`, `attention_stride=64`, and `mlp_stride=64`, so a 768-d residual is stored as 12 sampled GPT coordinates at dims `0, 64, 128, ..., 704`.
- For the residual `X` summaries in 2D and the related sidebar/vector previews, the 12 stored values are the color-driving values. The visible strip/card gradient interpolates between the 12 resulting colors; it is not a 64-d average per bucket.
- If a future capture is regenerated with a different stride, the sampled-coordinate interpretation changes with that config. Check `meta.config` in the capture file before assuming the meaning of a stored vector length.

## 2D View Roadmap
- Treat the 2D inspector as a parallel view of the same runtime/model state shown in 3D, not as a separate static explainer.
- Any 3D component that can open the selection panel should eventually be able to route into a matching 2D target via a user action such as `Move to 2D`.
- Keep 2D navigation keyed by stable semantic identifiers instead of DOM-only selectors. Expected routing payloads will likely include fields such as `componentKind`, `layerIndex`, `headIndex`, `tokenIndex` / `tokenIndices`, `stage`, and matrix/vector role.
- The long-term 2D canvas should have a canonical world/canvas coordinate system so every major component can report deterministic bounds/anchor points.
- Prefer a dedicated layout/bounds registry for the 2D view. Given a semantic target, the app should be able to resolve `{x, y, width, height}` (or equivalent focus anchors) without querying ad hoc rendered DOM state.
- When a 3D selection opens a 2D target, the 2D view should center/zoom to the corresponding region and reuse the same live token window / pass context that the 3D scene is showing.
- The expected interaction model is a viewport "fly to" or animated focus move into the exact 2D region corresponding to the selected 3D object, followed by a locked/selected state for that region.
- Prefer scalable vector-style rendering and semantic zoom for the long-term 2D canvas so the user can zoom deeply without the view turning into a blurred scaled bitmap.
- Keep the mapping layer modular. Prefer dedicated helpers/registries for `3D selection -> 2D target -> viewport focus` instead of hard-coding more routing logic directly into `src/ui/selectionPanel.js`.

## Change Guidance
Prefer adding helpers in focused modules (`src/engine/layers/*`, `src/animations/mhsa/*`, `src/ui/selectionPanel*Utils.js`) instead of expanding already-large controllers.
