# AGENT.md

## Scope
Multi-head self-attention routing internals.

## Key files
- `SelfAttentionAnimator.js`: per-head attention flow.
- `VectorRouter.js`: lane routing and split/merge logic.
- `PassThroughAnimator.js`: skip/pass-through paths.
- `VisualSetup.js`: builds MHSA visuals for the animation.
- `index.js`: public exports for MHSA helpers.

## Notes
- Lane ordering must stay consistent with `LayerPipeline` and `MHSAAnimation`.
- Update `laneIndex.js` if you change lane conventions.
