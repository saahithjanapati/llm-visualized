# AGENT.md

## Scope
Shared constants, materials, and helpers.

## Key files
- `constants.js`: layout, timings, colors, global toggles.
- `precomputedGeometryLoader.js`: GLB loader for cached geometry.
- `trailUtils.js` / `trailConstants.js`: residual stream trails.
- `sciFiMaterial.js`: shared materials.
- `activationPrecompute.js`: cache prep for capture data.

## Notes
- Many values are global; changing constants affects all demos.
- Keep utility functions side-effect free where possible.
