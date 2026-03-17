# AGENT.md

_Last reviewed: 2026-03-05._

## Scope
Static assets served directly by Vite.

## Key Files
- `twelve-layer-stack.css`: core tower styling.
- `capture.json`: captured activation dataset.
- `flops_per_step.csv` and `flops_per_step.svg`: stats/visualization assets.

## Notes
- Keep executable app code in `src/`.
- `index.html` loads the app entry from `src/app/gpt-tower/index.js`.
