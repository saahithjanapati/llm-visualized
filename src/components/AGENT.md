# AGENT.md

_Last reviewed: 2026-03-05._

## Scope
Reusable visual primitives.

## Key Files
- `VectorVisualizationInstancedPrism.js`: instanced prism vector implementation.
- `BatchedPrismVectorSet.js`: batched vector copies used to reduce draw calls.
- `WeightMatrixVisualization.js`: weight-matrix geometry/material wrapper.
- `LayerNormalizationVisualization.js`: layer norm ring geometry.

## Notes
- Components should stay focused on visuals/state containers.
- Higher-level animation sequencing belongs in `src/animations/` and `src/engine/layers/`.
