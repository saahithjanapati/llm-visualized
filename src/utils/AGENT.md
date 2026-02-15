# AGENT.md

## Scope
Cross-cutting constants and utility helpers.

## Key Files
- `constants.js`: shared layout, timings, and feature toggles.
- `visualTuningProfiles.js`: rendering/animation profile tuning constants.
- `trailUtils.js` and `trailConstants.js`: trail implementations and limits.
- `sciFiMaterial.js` and `materialUtils.js`: material shader/uniform helpers.
- `activationMetadata.js` and `activationPrecompute.js`: activation coloring/metadata prep.
- `precomputedGeometryLoader.js`: cached geometry loading.
- `additionUtils.js`, `prismLayout.js`, `colors.js`: animation/layout/color helpers.

## Notes
- Utility changes can affect multiple subsystems; audit call sites before modifying shared constants.
- Keep helpers side-effect-light unless a function is explicitly mutating visuals/materials.
