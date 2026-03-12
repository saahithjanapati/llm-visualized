# 2D Transformer Blueprint

Last updated: 2026-03-12

## Goal

Build a scalable 2D view of the same live transformer state shown in 3D.

The 2D view should support:

- full-model pan and zoom
- deterministic `3D selection -> 2D target -> fly to`
- the same activation-driven colors already used in 3D and the selection preview
- progressive detail instead of rendering every head at maximum fidelity at once

The target visual language stays close to the current MHSA CSS view:

- rounded matrix and vector blocks
- row and column subdivisions
- attention grids
- connectors
- labels, operators, and parentheses

The part that must change is the rendering architecture, not the aesthetic.

## Core Rule

Do not duplicate the current full-detail CSS MHSA block across the whole model.

Use one canonical 2D world with semantic zoom:

- far zoom: layer and module summaries
- medium zoom: module internals and head stacks
- close zoom: one focused module or head expands into the full equation view

This keeps the same visual vocabulary while making the full transformer navigable.

## Canonical World

The full 2D world should be laid out in model execution order from left to right:

1. token and positional embeddings
2. residual input rail
3. layer 1
4. layer 2
5. ...
6. layer 12
7. final layer norm
8. unembedding or logits projection
9. logits or token output area

Each transformer layer is its own world-space group. Inside each layer, also lay out left to right:

1. residual in
2. LN1
3. MHSA super-block
4. output projection
5. residual add
6. LN2
7. MLP
8. residual add
9. residual out

Inside the MHSA super-block, stack the 12 heads vertically.

Inside a focused head, render the current equation-style composition:

- `X_ln`
- `W_q`, `b_q`, `Q`
- `W_k`, `b_k`, `K`
- `W_v`, `b_v`, `V`
- `Q K^T / sqrt(d_h)`
- pre-score, mask, post-score
- post-score times `V`
- head output

## Primitive Set

The renderer should only need a small set of reusable primitives.

### Structural primitives

- `group`
- `stage`
- `connector`
- `rail`
- `focusFrame`

### Matrix and vector primitives

- `matrixCard`
  - rounded card for parameter matrices without dense sampled cells
- `matrixRows`
  - rounded outer rect with row bars
- `matrixColumns`
  - rounded outer rect with column bars
- `matrixGrid`
  - rounded outer rect with visible cells
- `vectorBar`
  - single row or column strip
- `stackSummary`
  - repeated tiny bars or cards for compressed overview states

### Text primitives

- `label`
- `caption`
- `operator`
- `paren`
- `badge`

These map cleanly onto canvas rendering and can later map to WebGL if needed.

## Primitive Schema

Suggested scene node shape:

```js
{
  id: "componentKind:mhsa/layerIndex:6/headIndex:6/stage:attention/role:head-detail",
  kind: "matrix" | "text" | "connector" | "group" | "rail",
  primitive: "matrixRows" | "matrixGrid" | "matrixCard" | "label" | "operator",
  semantic: {
    componentKind: "mhsa",
    layerIndex: 6,
    headIndex: 6,
    stage: "attention",
    role: "head-detail",
    tokenIndex: 3,
    rowIndex: 2,
    colIndex: 1
  },
  bounds: { x, y, width, height },
  anchors: { left, right, top, bottom, center },
  lod: {
    minScreenPx: 0,
    maxScreenPx: null,
    detailTier: "overview" | "module" | "detail" | "focused"
  },
  visual: {
    styleKey: "mhsa.q",
    background: "...",
    stroke: "...",
    cornerRadius: 12
  },
  data: {
    rows: [...],
    cols: [...],
    cells: [...],
    label: "Q",
    dims: "(5, 64)"
  },
  children: [...]
}
```

## Semantic Target Contract

The 2D view should resolve targets by semantics, not by DOM lookup.

Suggested target payload:

```js
{
  componentKind: "mhsa" | "output-projection" | "mlp" | "layer-norm" | "embedding" | "residual" | "logits",
  layerIndex: 6,
  headIndex: 6,
  stage: "attention" | "projection-q" | "projection-k" | "projection-v" | "softmax" | "post" | "ln1" | "ln2" | "mlp-up" | "mlp-down",
  role: "module" | "head" | "matrix" | "cell" | "residual-add",
  tokenIndex: 3,
  tokenIndices: [0, 1, 2, 3, 4],
  rowIndex: 2,
  colIndex: 1,
  focusKey: "softmax"
}
```

Examples:

```js
{ componentKind: "mhsa", layerIndex: 6, role: "module" }
{ componentKind: "mhsa", layerIndex: 6, headIndex: 6, role: "head" }
{ componentKind: "layer-norm", layerIndex: 6, stage: "ln1", role: "module" }
{ componentKind: "mlp", layerIndex: 6, stage: "mlp-up", role: "matrix" }
{ componentKind: "residual", layerIndex: 6, stage: "post-attn-add", role: "residual-add" }
```

## Zoom Levels

Use detail tiers based on projected screen size of the target bounds, not raw camera zoom alone.

Suggested tiers:

- `overview`
  - component is smaller than about `140px`
  - show only summary silhouettes, rails, tiny head stacks, and lightweight text
- `module`
  - component is roughly `140px` to `320px`
  - show major internal stages and compressed matrices
- `detail`
  - component is roughly `320px` to `720px`
  - show readable row and column subdivisions and small attention grids
- `focused`
  - component is larger than about `720px` or explicitly selected
  - show full equation composition with operators and richer captions

Important rule:

- only a focused MHSA head gets the full equation-style layout
- the whole layer never renders 12 expanded heads at once

## Component-Level Detail Mapping

### Embeddings

`overview`

- token strip
- embedding block
- positional embedding block
- residual merge rail

`module`

- show sampled row bars for selected token window
- show add operator and residual output rail

`detail`

- show token lookup rows, positional rows, and merged residual rows

`focused`

- if needed, show one token lookup path in the same rounded-rect style as the MHSA detail view

### Residual stream

`overview`

- one horizontal rail through the entire model

`module`

- highlight entry and exit points for each layer
- show add nodes as compact circular or rounded badges

`detail`

- show sampled residual row bars around a selected stage

`focused`

- show exact add participants and resulting residual block

### Layer norm

`overview`

- compact `LN1` or `LN2` chip attached to the residual rail

`module`

- input block, normalized block, `gamma`, `beta`, output block

`detail`

- mean and variance annotations if useful
- sampled rows before and after normalization

`focused`

- optional equation labels and parameter cards

### MHSA super-block

`overview`

- one MHSA module frame
- inside it, 12 tiny head cards stacked vertically
- output projection block on the right

`module`

- 12 head cards become readable
- each head card can show tiny `Q`, `K`, `V`, and a small attention heatmap summary
- output projection and residual add stay visible

`detail`

- selected head expands relative to neighbors
- neighboring heads stay compressed
- Q, K, V and attention blocks become readable

`focused`

- render the current equation-style composition for the selected head
- all other heads collapse to compact summaries

### Output projection

`overview`

- one `W_O` card after the head stack

`module`

- show concatenated head output summary on the left
- `W_O` and output residual update on the right

`detail`

- show sampled post-concatenation rows and projected output rows

`focused`

- show `concat(head_i) x W_O + b_O = attn_out`

### MLP

`overview`

- one compact MLP block

`module`

- `W_up`, activation block, `W_down`

`detail`

- sampled rows for up projection, activation, and down projection
- optional activation histogram or compact neuron banding

`focused`

- render `X_ln x W_up + b_up -> GELU -> x W_down + b_down`

### Final norm and logits

`overview`

- final LN block and logits block

`module`

- final normalized residual and unembedding card

`detail`

- top logits and projected token rows

`focused`

- selected-token path if needed

## Layout Strategy

Use authored world-space layout, not measured DOM layout.

At the top level:

- reserve a fixed horizontal band for each layer
- keep a stable vertical baseline for the residual rail
- place LN, MHSA, output projection, MLP, and add nodes at deterministic x offsets within each layer

Inside MHSA:

- use a stable vertical grid for the 12 heads
- reserve expansion space for one focused head
- when a head is focused, redistribute available height instead of rebuilding the world origin

This is important for smooth `flyTo` behavior. Bounds should not jump when detail changes.

## Connectors

Connectors should be semantic primitives too, not DOM overlays.

Use only a few route families:

- straight horizontal
- straight vertical
- elbow

Connector styling:

- neutral connector for math flow
- Q, K, V colored connectors where that improves readability
- dim connectors with non-selected nodes during focus mode

## Text Strategy

Do not use KaTeX across the entire full-model view.

Use:

- lightweight canvas text for overview and module tiers
- KaTeX only for selected or focused detail nodes

Recommended rule:

- if a node is not selected and its projected size is below a readability threshold, do not render equation-grade text

This prevents text layout from becoming the next bottleneck.

## Interaction Model

The 2D scene should support:

- hover
- select
- pan
- wheel zoom
- pinch zoom
- keyboard nudging
- semantic `flyTo`

Selection behavior:

1. resolve semantic target
2. resolve world bounds from layout registry
3. animate viewport to padded bounds
4. mark target as selected
5. switch target subtree to the required detail tier

If a 3D selection arrives:

- map the 3D object userData to a 2D semantic target
- call `flyToSemanticTarget(target)`
- keep the selected token window and pass context in sync with 3D

## Bounds Registry

The layout registry should expose semantic resolution helpers:

```js
resolveNodeIdForTarget(target)
resolveBoundsForTarget(target)
resolveAnchorsForTarget(target)
resolveFocusPathForTarget(target)
```

Matching priority should go from most specific to least specific:

1. exact cell
2. exact matrix
3. exact head
4. exact module stage
5. layer module
6. layer

This makes `Move to 2D` robust even when some detailed child is not rendered at the current tier.

## Rendering Rules

### Rule 1

The renderer is authoritative for layout, interaction, and focus.

The old DOM tree may remain as a short-term visual reference, but not as the source of viewport math or hit testing.

### Rule 2

LOD is a rendering choice, not a semantic identity change.

The selected head in layer 7 is still the same semantic target whether shown as:

- a tiny head badge
- a medium card
- a full equation view

### Rule 3

Do not recreate the whole scene on every hover.

Use:

- stable scene nodes
- stable bounds
- a small render-state overlay for hover, selection, and dimming

### Rule 4

Favor sampled visual summaries for large tensors.

Examples:

- residual and MLP vectors use sampled row gradients
- parameter matrices use cards or sampled bars
- attention score matrices render actual cells only when focused enough

## Suggested File Shape

Suggested additions under `src/view2d/`:

- `model/buildTransformerSceneModel.js`
- `model/buildLayerSceneModel.js`
- `model/buildEmbeddingSceneModel.js`
- `model/buildMhsaModuleSceneModel.js`
- `model/buildOutputProjectionSceneModel.js`
- `model/buildLayerNormSceneModel.js`
- `model/buildMlpSceneModel.js`
- `layout/resolveSemanticTargetBounds.js`
- `runtime/View2dViewportController.js`
- `render/canvas/CanvasHitTester.js`
- `render/canvas/lodResolvers.js`

Suggested extensions:

- extend `src/view2d/schema/sceneTypes.js` with explicit primitive kinds and LOD metadata
- extend `src/view2d/layout/LayoutRegistry.js` with semantic target lookup helpers
- keep `src/view2d/model/buildMhsaSceneModel.js` as the focused-head builder, not the full-world builder

## Implementation Order

### Phase 1

Build the full transformer world in overview tier only.

Deliverables:

- deterministic world bounds
- pan and zoom
- `flyToSemanticTarget`
- layer, module, and head summary nodes

### Phase 2

Add MHSA module tier and focused head tier.

Deliverables:

- 12-head vertical stack per layer
- expand one head into the current equation-style detail view
- collapse neighbors into summaries

### Phase 3

Add LN, output projection, and MLP detail tiers.

Deliverables:

- coherent per-layer left-to-right story
- residual add points
- sampled activation-driven visuals

### Phase 4

Add 3D to 2D routing and polish.

Deliverables:

- map 3D selections to semantic targets
- viewport fly-to animations
- selected and dimmed states
- lightweight labels at overview scale

## Immediate Next Step

The next implementation step should be:

1. create `buildTransformerSceneModel.js`
2. create a stable world layout for `embedding -> 12 layers -> final ln -> logits`
3. make `LayoutRegistry` resolve semantic target bounds
4. add a viewport controller with `flyTo(bounds)`

Do not start by trying to make the full-model view beautiful at maximum detail.

Start by making the world stable, navigable, and semantically addressable.

Once that works, the current MHSA CSS aesthetic can be reintroduced as the focused-detail tier where it belongs.
