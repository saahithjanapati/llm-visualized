# AGENT.md

_Last reviewed: 2026-03-05._

## Scope
Activation sources and parameter datasets.

## Key Files
- `RandomActivationSource.js`: synthetic fallback activations.
- `CaptureActivationSource.js`: reads captured activations for deterministic playback.
- `layerNormParams.js` and `gpt2_layernorm_params.json`: layer norm parameter data.
- `biasParams.js` and `gpt2_bias_params.json`: sampled GPT-2 bias data.
- `parameterCheckpoints.js`: parameter-count milestones for overlays.

## Notes
- Capture artifacts are served from `public/`.
- Keep source interfaces stable so `gpt-tower` wiring can switch sources cleanly.
