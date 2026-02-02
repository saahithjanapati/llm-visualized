# AGENT.md

## Scope
Single transformer-layer visuals and lane construction.

## Key files
- `Gpt2Layer.js`: builds LN/MHSA/MLP geometry and sequencing.
- `gpt2LaneBuilder.js`: creates lane groups and placeholders.
- `gpt2LayerUtils.js`: helpers for materials, data application, math.

## Notes
- If changing per-layer layout, update constants in `src/utils/constants.js`.
- For performance, reuse cached materials/colors and avoid per-frame allocations.
