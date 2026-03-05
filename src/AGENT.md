# AGENT.md

_Last reviewed: 2026-03-05._

## Scope
Main source tree for rendering, animation, UI, and data plumbing.

## Layout
- `app/gpt-tower/`: demo bootstrap, controls, token UI, activation wiring.
- `engine/`: scene runtime, camera/raycast systems, layer pipeline.
- `engine/layers/`: per-layer state machine + lane/data/watchdog helpers.
- `animations/`: top-level animation controllers.
- `animations/mhsa/`: MHSA routing, pass-through, visual setup, timing helpers.
- `components/`: reusable visual primitives.
- `ui/`: overlays, interactive controls, and selection-panel helper modules.
- `utils/`: constants, material/trail helpers, activation utilities, shared matrix tuning helpers.
- `data/`: activation sources and model parameter datasets.
- `state/`: shared runtime state container.

## Start Points
- Demo wiring: `src/app/gpt-tower/index.js`
- Layer behavior: `src/engine/layers/Gpt2Layer.js`
- MHSA orchestration: `src/animations/MHSAAnimation.js`
