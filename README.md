# llm-visualized

`llm-visualized` is an interactive Three.js project that turns a GPT-2 Small forward pass into a navigable visual scene. The main experience is a 12-layer transformer tower where tokens, residual vectors, layer norms, attention heads, and MLP blocks are shown as animated geometry instead of abstract diagrams.

The project is meant to be both explanatory and inspectable. You can watch the model pipeline play out in 3D, open richer detail panels, and use the 2D inspector work to look at the same model state from a more semantic diagram view.

## What the app shows

The main runtime walks through the same major stages you would expect in a GPT-2 Small forward pass:

- token and position embeddings
- residual-stream addition
- per-layer `LayerNorm -> MHSA -> residual add -> LayerNorm -> MLP -> residual add`
- final layer norm and output logits
- generation-time token selection and next-step continuation

Instead of showing raw tensors directly, the app uses repeated visual metaphors such as:

- tapered matrix prisms for learned projections
- sampled vector strips for activations
- lane-based token flow through the model stack
- animated attention routing for query/key/value and weighted-sum behavior
- detail panels and 2D views for inspecting the currently selected semantic target

## Main routes

- `index.html`: primary GPT-2 tower demo
- `info/index.html`: project information page
- `essay/index.html`: essay / long-form companion page

## Getting started

### Install

```bash
npm install
```

### Run the app locally

```bash
npm run dev
```

Vite serves the project over local HTTPS. The app entry point is the tower demo in `index.html`.

### Run tests

```bash
npm run test
```

### Build for production

```bash
npm run build
```

## Useful scripts

- `npm run dev`: start the local development server
- `npm run test`: run the Vitest suite
- `npm run build`: produce a production build in `dist/`
- `npm run preview`: preview the production build locally
- `npm run precompute`: regenerate precomputed GLB geometry used by the runtime
- `npm run test:activation-integrity`: run the activation-integrity checks

Note: the `predev`, `prebuild`, and `pretest` hooks automatically run `scripts/prepareRuntimeAssets.mjs` so capture manifests stay in sync with the files in `public/`.

## Project structure

### App entry and orchestration

- `src/app/gpt-tower/index.js`: bootstraps the tower experience
- `src/app/gpt-tower/`: app-specific runtime helpers such as capture loading, token chips, and top-logit UI
- `src/app/weight-matrix-noncsg-test/`: auxiliary app entry for weight-matrix experimentation

### Scene engine

- `src/engine/CoreEngine.js`: Three.js renderer, camera, controls, picking, and render loop
- `src/engine/LayerPipeline.js`: creates and sequences the transformer layers
- `src/engine/layers/Gpt2Layer.js`: drives the per-layer phase logic and handoff behavior

### Animation and visual logic

- `src/animations/`: higher-level animation controllers
- `src/animations/mhsa/`: attention-specific setup, timing, routing, and output-projection helpers
- `src/components/`: reusable visual primitives such as vectors, matrices, and layer-norm geometry

### UI and inspector work

- `src/ui/`: overlays, controls, selection panel logic, and 2D transformer view helpers
- `src/view2d/`: semantic 2D scene helpers and detail-view rendering utilities
- `src/content/`: longer-form markdown content used by UI surfaces

### Data and assets

- `public/`: static assets and shipped activation captures such as `capture.json`
- `src/assets/runtime/precomputed/`: cached GLB geometry used at runtime
- `src/assets/runtime/environments/`: EXR environment maps for lighting/reflections
- `src/data/`: activation-source adapters, model parameters, and packaged datasets
- `scripts/`: geometry-generation and GPT-2 activation extraction utilities

## Activation capture workflow

The repo includes a capture utility for exporting sampled GPT-2 activations into the format the visualizer expects:

```bash
pip install transformers torch
python scripts/extract_gpt2_data.py --max-new-tokens 40 --num-completions 6 --output capture.json
```

Useful flags:

- `--quantisation`: `float16` (default), `int8`, or `float32`
- `--residual-stride`: sampling stride for 768-d residual vectors
- `--attention-stride`: sampling stride for 64-d head vectors
- `--mlp-stride`: sampling stride for 3072-d MLP activations
- `--top-k` / `--top-p`: generation controls so captures line up with the runtime behavior you want to visualize

These captures are compact by design. The visualizer uses sampled coordinates rather than full tensors so the demo stays lightweight enough to run interactively in the browser.

## Notes for contributors

- Prefer adding focused helpers in smaller modules rather than expanding already-large controllers.
- The biggest complexity hotspots are currently the selection panel, `Gpt2Layer`, and `MHSAAnimation`.
- If you regenerate runtime geometry, keep the generated assets under `src/assets/runtime/` rather than the repo root.
