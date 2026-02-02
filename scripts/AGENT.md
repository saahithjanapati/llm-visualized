# AGENT.md

## Scope
Data extraction and precomputed geometry generation.

## Key files
- `extract_gpt2_data.py`: capture activations for visualization.
- `extract_gpt2_layernorm_params.py`: LN params.
- `generate_precomputed_components.mjs`: build cached GLB geometry.
- `generate_qkv_components.mjs`: QKV geometry variants.

## Notes
- Generated assets are large; check output paths before running.
