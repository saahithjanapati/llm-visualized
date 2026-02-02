# AGENT.md

## Scope
Animation controllers and sequencing helpers.

## Key files
- `MHSAAnimation.js`: attention routing block used by the main tower.
- `PrismLayerNormAnimation.js`: LN animation and prism flows.
- `LayerAnimationConstants.js`: shared colors and lane settings.
- `mhsa/`: detailed attention routing helpers.

## Notes
- Most animations are constructed by `Gpt2Layer.js`.
- Prefer updating or adding a controller rather than embedding timing logic in components.
