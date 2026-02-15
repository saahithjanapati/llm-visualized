# AGENT.md

## Scope
Animation controllers and sequencing logic.

## Key Files
- `MHSAAnimation.js`: top-level MHSA controller (routing, pass-through, merge/projection phases, fallbacks).
- `PrismLayerNormAnimation.js`: layer-norm vector animation behavior.
- `LayerAnimationConstants.js`: shared MHSA/LN colors and tuning constants.
- `mhsa/`: split-out MHSA internals (`VisualSetup`, router, pass-through, timing, lane index helpers).

## Notes
- Static MHSA matrix/output-projection creation lives in `mhsa/VisualSetup.js`.
- Keep per-frame update paths allocation-light; place setup/config transforms outside hot loops.
