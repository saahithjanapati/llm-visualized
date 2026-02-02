# AGENT.md

## Scope
Reusable visual primitives (vectors, matrices, layer norm rings).

## Key files
- `VectorVisualizationInstancedPrism.js`: instanced prism vectors (main tower).
- `WeightMatrixVisualization.js`: weight matrix blocks.
- `LayerNormalizationVisualization.js`: LN ring geometry.

## Notes
- Instanced versions are preferred for performance.
- Components should be pure visuals; animation logic lives in `src/animations/`.
