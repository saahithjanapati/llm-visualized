# AGENT.md

_Last reviewed: 2026-03-05._

## Scope
Animation controllers and sequencing logic.

## Key Files
- `MHSAAnimation.js`: top-level MHSA controller (routing, pass-through, merge/projection phases, fallbacks).
- `PrismLayerNormAnimation.js`: layer-norm vector animation behavior.
- `LayerAnimationConstants.js`: shared MHSA/LN colors and tuning constants.
- `mhsa/`: split-out MHSA internals (`VisualSetup`, router, pass-through, timing, lane index helpers).

## Notes
- Static MHSA matrix/output-projection creation lives in `mhsa/VisualSetup.js`.
- Matrix label/material tuning shared with other systems should go through `src/utils/matrixVisualUtils.js`.
- Keep per-frame update paths allocation-light; place setup/config transforms outside hot loops.
