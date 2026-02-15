# AGENT.md

## Scope
Three.js runtime, pipeline orchestration, and camera/raycast systems.

## Key Files
- `CoreEngine.js`: renderer, scene, camera, controls, render/update loop.
- `LayerPipeline.js`: layer lifecycle, lane handoff, skip/progress orchestration.
- `AutoCameraController.js`: auto-follow camera behavior.
- `autoCameraViewLogic.js`: camera target logic helpers.
- `coreRaycastResolver.js` and `coreRaycastLabels.js`: selection/raycast decoding helpers.
- `layerPipelineTopEmbedding.js` and `layerPipelineMath.js`: pipeline-specific math/layout helpers.
- `BaseLayer.js`: layer interface contract.

## Notes
- Layer geometry mostly lives in `engine/layers/`.
- Keep engine-level modules orchestration-focused; avoid embedding layer-specific animation details here.
