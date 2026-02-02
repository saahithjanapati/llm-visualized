# AGENT.md

## Scope
Three.js runtime and stack orchestration.

## Key files
- `CoreEngine.js`: renderer, scene, camera, controls, raycasting, main loop.
- `LayerPipeline.js`: builds N layers, manages lane handoff and skip logic.
- `AutoCameraController.js`: follow-mode camera offsets and auto-focus logic.
- `BaseLayer.js`: interface for per-layer init/update/dispose.

## Notes
- Most scene objects are created in layer classes, not here.
