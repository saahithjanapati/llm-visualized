# AGENT.md

## Scope
Shared runtime state for UI and engine.

## Key files
- `appState.js`: global flags (playback, dev mode, selection, etc).

## Notes
- Keep state updates centralized; avoid ad-hoc globals.
