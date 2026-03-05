# AGENT.md

_Last reviewed: 2026-03-05._

## Scope
Internal modules used by `MHSAAnimation`.

## Key Files
- `VisualSetup.js`: builds Q/K/V matrix sets + output projection matrix and returns layout metadata.
- `VectorRouter.js`: splits/routs lane vectors to head parking positions.
- `PassThroughAnimator.js`: manages pass-through stage sequencing.
- `VectorMatrixPassThrough.js`: per-vector matrix traversal helper.
- `SelfAttentionAnimator.js`: optional self-attention conveyor/weighted-sum stage.
- `laneIndex.js`: lane/head indexing helpers.
- `mhsaTimingUtils.js`: pause-aware delay/timing helpers.
- `index.js`: exports MHSA submodules.

## Notes
- `buildMHAVisuals(...)` in `VisualSetup.js` is the active matrix construction path.
- Matrix label/surface tweaks are shared through `src/utils/matrixVisualUtils.js`.
- Keep lane identity/index conventions aligned with `LayerPipeline` and `Gpt2Layer`.
