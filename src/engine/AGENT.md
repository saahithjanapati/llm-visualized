# AGENT.md

## Scope
Three.js runtime and stack orchestration.

## Key files
- `CoreEngine.js`: renderer, scene, camera, controls, raycasting, main loop.
- `LayerPipeline.js`: builds N layers, manages lane handoff, camera follow/skip logic.
- `BaseLayer.js`: interface for per-layer init/update/dispose.

## Notes
- Camera automation and skip-to-end live in `LayerPipeline.js`.
- Most scene objects are created in layer classes, not here.
