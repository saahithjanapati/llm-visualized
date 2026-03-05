# AGENT.md

_Last reviewed: 2026-03-05._

## Scope
Cross-cutting constants and utility helpers.

## Key Files
- `constants.js`: shared layout, timings, and feature toggles.
- `visualTuningProfiles.js`: rendering/animation profile tuning constants.
- `trailUtils.js` and `trailConstants.js`: trail implementations and limits.
- `sciFiMaterial.js` and `materialUtils.js`: material shader/uniform helpers.
- `matrixVisualUtils.js`: shared matrix labeling/userData/material tweak helpers.
- `activationMetadata.js` and `activationPrecompute.js`: activation coloring/metadata prep.
- `precomputedGeometryLoader.js`: cached geometry loading.
- `additionUtils.js`, `prismLayout.js`, `colors.js`: animation/layout/color helpers.

## Notes
- Utility changes can affect multiple subsystems; audit call sites before modifying shared constants.
- Keep helpers side-effect-light unless a function is explicitly mutating visuals/materials.
