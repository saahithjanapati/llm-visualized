# AGENT.md

## Scope
DOM overlays, controls, and inspection panels.

## Key Files
- `selectionPanel.js`: primary selection details panel and preview composition.
- `selectionPanelSelectionUtils.js`: selection filtering/normalization helpers.
- `selectionPanelNarrativeUtils.js`: narrative text generation helpers.
- `statusOverlay.js`, `parameterCounter.js`, `perfOverlay.js`: HUD widgets.
- `settingsModal.js`, `pauseButton.js`, `skipMenu.js`, `skip*Button.js`, `conveyorSkipButton.js`: runtime controls.
- `introAnimation.js`, `liveVisualControls.js`, `touchClickFallback.js`: UX glue.

## Notes
- `selectionPanel.js` is a current complexity hotspot; prefer extracting narrow helpers rather than growing it.
- Keep UI modules aligned with DOM ids/classes defined by `index.html` and `public/twelve-layer-stack.css`.
