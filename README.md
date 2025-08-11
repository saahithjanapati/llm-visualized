# llm-visualized
3d visualization of gpt2

### 12‑Layer GPT‑2 Stack demo (tests/twelve-layer-stack.html)
Brief notes on the modules and assets used by `tests/twelve-layer-stack.html`.

- `tests/twelve-layer-stack.html`: Entry page for the tower demo. Sets playback speed via `setPlaybackSpeed`, enables self‑attention for MHSA, shows an intro type animation, creates a `LayerPipeline` with 12 layers on `gptCanvas`, loads an HDRI (`rogland_clear_night_1k.exr`), shows a status overlay, and provides a settings modal for speed.

- `src/engine/LayerPipeline.js`: Orchestrates a single bundle of residual vectors (lanes) through N stacked GPT‑2 layers. Holds a `CoreEngine` instance, pre‑creates `Gpt2Layer`s, transfers lanes upward when each layer finishes, and exposes `engine` used by the page.
- `src/engine/CoreEngine.js`: Three.js engine setup (scene, camera, renderer, post‑processing bloom), `OrbitControls`, hover label raycasting, a single RAF loop that updates active layers, global pause/resume, and optional `Stats` overlay.
- `src/engine/layers/Gpt2Layer.js`: One transformer layer. Builds LN1 → MHSA → LN2 → MLP visuals; manages lanes, trails, and synchronised phase gates; delegates MHSA details to `MHSAAnimation`; triggers residual addition and hands off to the next layer via `onFinished`.

- `src/animations/MHSAAnimation.js`: High‑level MHSA controller. Builds head visuals, routes vectors via `VectorRouter`, runs parallel pass‑through via `PassThroughAnimator`, optional above‑matrix effects via `SelfAttentionAnimator`, handles merge/output‑projection phases, and kicks the residual addition animation.
- `src/animations/mhsa/index.js`: Barrel exports for the MHSA sub‑modules used above.
- `src/animations/mhsa/VectorRouter.js`: Positions travelling vectors at each head, spawns upward K copies and side Q/V copies, and signals readiness for pass‑through.
- `src/animations/mhsa/PassThroughAnimator.js`: Launches all K/Q/V pass‑through tweens for every head and lane; on completion triggers self‑attention or final colour transitions.
- `src/animations/mhsa/SelfAttentionAnimator.js`: Above‑matrix choreography. Adds extra V rise, aligns K under V, and runs a conveyor‑belt traversal for Q vectors; durations scale with `SELF_ATTENTION_TIME_MULT`.
- `src/animations/mhsa/VectorMatrixPassThrough.js`: Per‑matrix pass‑through animation; swaps heavy vectors for lightweight 64‑dim variants inside matrices; pulses matrix emissive colour.
- `src/animations/mhsa/VisualSetup.js`: Builds Q/K/V matrices for all heads and the Output‑Projection matrix; returns coordinates and references used by the MHSA controller.
 - `src/animations/PrismLayerNormAnimation.js`: Per‑unit LayerNorm visual animation for prism vectors (rise/flash sequence), used in LN1 and LN2.
 - `src/animations/LayerAnimationConstants.js`: Shared colour/layout presets for MHSA/LN/MLP used by `Gpt2Layer` and MHSA visuals (e.g., final Q/K/V colours, output‑projection params).

- `src/utils/constants.js`: Centralised constants and runtime knobs. Includes `setPlaybackSpeed`/`GLOBAL_ANIM_SPEED_MULT`, geometry grouping (e.g., `PRISM_DIMENSIONS_PER_UNIT`, `VECTOR_LENGTH_PRISM`), lane counts, timing scalars, CAPTION placement (`CAPTION_TEXT_Y_POS`), quality flags, and MHSA/MLP timings.
- `src/utils/precomputedGeometryLoader.js`: Optionally loads `precomputed_components.glb` to prime cached `BufferGeometry` for WeightMatrix/LayerNorm shapes (skips heavy CSG at runtime).
- `src/utils/trailUtils.js` and `src/utils/trailConstants.js`: Trail rendering utilities (`StraightLineTrail`) and shared trail styling; used widely for residual and branch motion traces.
 - `src/utils/additionUtils.js`: Prism‑by‑prism residual addition helper used to merge processed vectors back into the residual stream; manages lane flags and trail opacity.
 - `src/utils/colors.js`: Colour mapping helpers (HSL gradients, bright/monochromatic variants) used by vector visuals and animations.

- `src/components/VectorVisualizationInstancedPrism.js`: Instanced‑mesh vector (grouped prisms) used for all vectors in the demo; supports `rawData`, gradient colouring, and `applyProcessedVisuals`.
- `src/components/LayerNormalizationVisualization.js`: LayerNorm “ring” mesh with configurable slits; supports colour/emissive/opacity updates.
- `src/components/WeightMatrixVisualization.js`: Tapered/rectangular weight matrix block; exposes colour/emissive/material property setters; used for Q/K/V and MLP matrices.
 - `src/engine/BaseLayer.js`: Minimal base class for layers providing a `root` group and lifecycle (`init`, `update`, `dispose`).

- `src/data/RandomActivationSource.js`: Factory used by `LayerPipeline` to seed per‑layer random vector data.

- Assets referenced by the page:
  - `precomputed_components.glb` (and optional `precomputed_components_slice.glb`): pre‑baked component geometries for faster startup.
  - `rogland_clear_night_1k.exr`: HDRI environment map applied to both scenes.
  - `cool-fun.gif`: loading overlay GIF shown during startup.

External CDN/runtime deps (not in this repo):
- `@tweenjs/tween.js` UMD (global `TWEEN`), used throughout animations.
- `stats.js` (optional FPS overlay).
- Fonts via `FontLoader` from three.js examples.
