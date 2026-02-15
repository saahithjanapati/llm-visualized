# AGENT.md

## Scope
Activation sources and parameter datasets.

## Key Files
- `RandomActivationSource.js`: synthetic fallback activations.
- `CaptureActivationSource.js`: reads captured activations for deterministic playback.
- `layerNormParams.js` and `gpt2_layernorm_params.json`: layer norm parameter data.
- `parameterCheckpoints.js`: parameter-count milestones for overlays.

## Notes
- Capture artifacts are served from `public/`.
- Keep source interfaces stable so `gpt-tower` wiring can switch sources cleanly.
