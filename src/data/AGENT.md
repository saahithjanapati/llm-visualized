# AGENT.md

## Scope
Activation sources and parameter data.

## Key files
- `RandomActivationSource.js`: synthetic activations.
- `CaptureActivationSource.js`: loads captured GPT-2 activations.
- `layerNormParams.js` / `gpt2_layernorm_params.json`: LN parameter data.
- `parameterCheckpoints.js`: parameter counts for overlays.

## Notes
- Capture files typically live under `public/` (for example capture.json).
