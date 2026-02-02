# AGENT.md

## Scope
Source code for the visualization. This folder hosts the rendering engine, animations, UI overlays, and data adapters used by the main demo.

## Map
- engine/: Three.js core (renderer, camera, layer stack).
- engine/layers/: single transformer layer visuals.
- animations/: animation controllers (MHSA, LN, vector ops).
- components/: reusable geometry/visual primitives.
- utils/: constants, materials, trails, loaders.
- ui/: DOM overlays and controls.
- data/: activation sources and parameter data.
- state/: shared runtime state.
- app/gpt-tower/: main demo entry and helper modules.

## Where to start
- For the main tower: `src/app/gpt-tower/index.js` wires the pipeline.
- For per-layer visuals: `engine/layers/Gpt2Layer.js`.
