# AGENT.md

_Last reviewed: 2026-03-05._

## Scope
Per-transformer-layer visuals, state machine phases, and lane construction helpers.

## Key Files
- `Gpt2Layer.js`: LN/MHSA/MLP sequencing and lane lifecycle.
- `gpt2LaneBuilder.js`: lane object construction and placeholder setup.
- `gpt2LanePhases.js`: lane phase enums + transition guards.
- `gpt2LaneWatchdogUtils.js`: lane-progress signatures and LN debug vector helpers.
- `gpt2LayerDataAccess.js`: activation/parameter data lookup helpers.
- `gpt2LayerNormVisuals.js`: layer norm visual state updates.
- `gpt2LayerUtils.js`: vector/trail/material utility operations.

## Notes
- `Gpt2Layer.js` is intentionally central but large; prefer extracting helpers into sibling modules.
- Preserve lane phase transition validity when changing sequencing behavior.
