# AGENT.md

_Last reviewed: 2026-03-05._

## Scope
Shared runtime state.

## Key Files
- `appState.js`: global flags for playback, selection, debug toggles, and UI integration.

## Notes
- Treat `appState` as shared contract between engine, animation, and UI layers.
- Prefer explicit field additions over ad-hoc globals.
