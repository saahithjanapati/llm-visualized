# AGENT.md

_Last reviewed: 2026-03-05._

## Scope
Offline data extraction and precompute tooling.

## Key Files
- `extract_gpt2_data.py`: activation capture extraction.
- `extract_gpt2_layernorm_params.py`: layer norm parameter extraction.
- `generate_precomputed_components.mjs`: main precompute generator.
- `generate_qkv_components.mjs`: QKV-focused geometry generation.
- `generate_single_lane_components.mjs`: single-lane geometry generation.

## Notes
- Generated outputs are large and intended for `public/assets/`.
- Validate output paths before running generators.
