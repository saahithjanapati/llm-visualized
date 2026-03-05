# AGENTS.md

_Last reviewed: 2026-03-05._

## Repo Summary
This repo is a Three.js visualization of GPT-2 internals, centered on a 12-layer transformer stack demo.

## Primary Runtime Path
- `index.html` loads the tower demo.
- `src/app/gpt-tower/index.js` wires engine, pipeline, UI, and activation data.
- `src/engine/LayerPipeline.js` creates and sequences layers.
- `src/engine/layers/Gpt2Layer.js` drives LN -> MHSA -> MLP per-layer flow.
- `src/animations/MHSAAnimation.js` orchestrates attention routing/pass-through.
- `src/animations/mhsa/VisualSetup.js` builds static MHSA matrices and output projection visuals.

## Key Areas
- `src/engine/`: renderer, camera, layer orchestration, raycast helpers.
- `src/engine/layers/`: lane construction, phase logic, data-access helpers, watchdog/debug signature helpers.
- `src/animations/` and `src/animations/mhsa/`: animation controllers and MHSA internals.
- `src/components/`: visual primitives (instanced vectors, matrices, layer norm geometry).
- `src/ui/`: DOM controls, overlays, and selection-panel helper utilities.
- `src/utils/`: constants, trails, materials, activation metadata/precompute helpers, shared matrix-visual tuning helpers.
- `src/data/`: random/captured activation sources + parameter datasets.
- `scripts/`: dataset extraction and geometry-generation scripts.
- `public/`: static assets and CSS.

## Current Complexity Hotspots
- `src/ui/selectionPanel.js` (large UI + preview composition module).
- `src/engine/layers/Gpt2Layer.js` (large phase/state machine).
- `src/animations/MHSAAnimation.js` (broad lifecycle + fallback paths).

## Dev Commands
- `npm run dev`
- `npm run test`
- `npm run build`

## Change Guidance
Prefer adding helpers in focused modules (`src/engine/layers/*`, `src/animations/mhsa/*`, `src/ui/selectionPanel*Utils.js`) instead of expanding already-large controllers.
