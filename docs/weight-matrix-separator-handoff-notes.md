# Weight Matrix Separator Artifact Handoff Notes

Date: 2026-02-06

Owner context: `WeightMatrixVisualization` slit artifacts are mostly suppressed by a distance/screen-size LOD swap, but the visual transition feels janky. Goal is to preserve visible internal lane separators while eliminating far-distance separator bleed/shimmer without hard popping.

Primary file:
- `/Users/saahithjanapati/Desktop/llm-visualized/src/components/WeightMatrixVisualization.js`

Related scene/depth file:
- `/Users/saahithjanapati/Desktop/llm-visualized/src/engine/CoreEngine.js`

Prior investigation log:
- `/Users/saahithjanapati/Desktop/llm-visualized/docs/weight-matrix-slit-artifacts-investigation.md`

## 1) Problem Statement

Symptom:
- When camera is far from a slit-enabled weight matrix, inner separator faces "fight" with the outer shell and appear as flickering/dashed artifacts.

User constraints:
- Keep inner lane separators visible at close/normal viewing distances.
- Avoid transparent/inside-view regressions.
- Keep matrix depth aligned with LayerNorm depth (no "matrices are longer than layernorm" regression).
- Avoid heavy jank/pop behavior.

Current state:
- A near/far LOD swap exists: near uses full geometry (with separators), far uses stripped-shell geometry (internal separator faces removed). This suppresses artifacts but introduces noticeable behavior changes.

## 2) Current Implementation Snapshot (Important)

Current branch behavior in `WeightMatrixVisualization`:
- Geometry cleanup and dedupe:
  - `sanitizeCsgGeometry(...)` removes degenerate + duplicate triangles.
  - Optional `stripInternalZFaces` removes internal separator-aligned z-faces for far shell.
- Slit material hardening:
  - `enforceStableSlitMaterial(...)` forces opaque/depth-stable behavior and disables transmission/thickness.
- Depth alignment correction:
  - `_resolveGeometryDepth(...)` normalizes slit matrix depth from `(lanes + 1) * spacing` to `lanes * spacing` when matching canonical lane depth.
- Slit LOD state:
  - `_separatorShellMesh` is built from stripped geometry.
  - `_updateSlitLodVisibility(...)` switches between full mesh and stripped-shell mesh.
  - Current trigger uses world distance + projected screen size hysteresis.
- Slit instancing:
  - Instanced-slice path is disabled for slit-enabled matrices.

Current camera/depth context in `CoreEngine`:
- `WebGLRenderer` uses `logarithmicDepthBuffer: true`.
- Adaptive near/far is active:
  - base near defaults to `5`
  - adaptive near ratio defaults to `0.003`
  - near cap defaults to `80`
  - far/near ratio cap defaults to `6000`

## 3) Investigation Summary (What Was Already Tried)

Across this branch and prior investigation cycles, the following classes of fixes were attempted:

1. Geometry topology and CSG cleanup
- Degenerate triangle removal.
- Duplicate triangle dedupe.
- Corner radius clamps for slit-enabled geometry to avoid sliver triangles.
- Normal cleanup and snap passes.
- Multiple slit topology rewrites in prior logs.

2. Material and shading stabilization
- Slit materials forced opaque.
- Transmission/thickness disabled for slit meshes.
- Depth writes/tests forced on.
- Shader noise/stripe-like effects reduced or disabled in some variants.

3. Render ordering and cap adjustments
- Explicit renderOrder for shell and caps.
- Cap polygon offset tuning.
- Cap offset epsilon tuning.

4. Instancing and depth/scene tuning
- Slit-enabled instanced path disabled (for seam stability).
- Core camera near/far adaptive behavior adjusted.
- Log depth kept enabled.

5. Visibility strategy
- Added near/far slit separator LOD shell swap (current workaround).
- Then changed LOD trigger to include projected screen size to engage at realistic distances.

Observed outcomes:
- No single geometry-only or material-only tweak fully solved the far artifact while keeping separators visible everywhere.
- LOD swap suppresses artifact, but visual continuity is not ideal.

Conclusion:
- This is primarily a raster/depth precision + coplanar-near-coplanar interior-vs-shell visibility problem under minification.
- There is no one-line "never show inner object on outer shell" toggle if the inner object is rendered normally in the same depth pipeline.

## 4) Non-Negotiable Acceptance Criteria

Any candidate fix must satisfy all:

1. No visible separator bleed/flicker on closed shell surfaces at far zoom.
2. Lane separators still visible when looking through slit openings.
3. No "everything transparent from inside" regression.
4. No matrix-vs-layernorm depth mismatch regression.
5. No severe pop/jank during camera motion.
6. `npm run build` passes.

## 5) Handoff Plan: Two Parallel Agent Tracks

Both agents should work in separate branches and behind explicit runtime flags.

Suggested branches:
- Agent A: `codex/wm-separator-depth-rules`
- Agent B: `codex/wm-separator-stencil-mask`

Suggested common flag style:
- URL flags or globals, e.g. `?wmSeparatorMode=depth` / `?wmSeparatorMode=stencil`.
- Keep current LOD behavior as fallback mode until one candidate is accepted.

---

## 6) Agent A Plan: Depth-Rule Separators (No Hard LOD Swap)

### Objective
Replace near/far visibility swapping with a stable dual-mesh depth strategy:
- Shell mesh renders normally.
- Separator mesh renders only when truly visible through openings.

### Core idea
Use depth behavior to naturally reject separator fragments behind shell:
- Separator mesh should test against shell depth but avoid writing its own unstable depth.
- Add a slight depth bias so separators remain "behind" shell in ambiguous pixels.

### Implementation steps

1. Split separator geometry from shell geometry
- Extend `sanitizeCsgGeometry(...)` to support:
  - `keepOnlyInternalZFaces` (new option), inverse of `stripInternalZFaces`.
- Build three geometries when slits are enabled:
  - `fullGeometry` (existing full mesh)
  - `shellGeometry` (strip internal z faces)
  - `separatorGeometry` (keep only internal separator-relevant faces)

2. Replace LOD swap with persistent shell + separator meshes
- Use shell mesh as primary visible matrix body.
- Add a dedicated `_separatorMesh` (not `_separatorShellMesh` LOD fallback).
- Remove/disable `_updateSlitLodVisibility` path for this mode.

3. Separator material depth policy
- Keep opaque.
- Set:
  - `depthTest = true`
  - `depthWrite = false`
  - `depthFunc = THREE.LessDepth`
  - `polygonOffset = true`
  - start with `polygonOffsetFactor = 2`, `polygonOffsetUnits = 2`
- Side recommendation:
  - start with `THREE.DoubleSide` (to preserve inside/view-through behavior)
  - if leakage persists, test `THREE.FrontSide` with consistent winding.

4. Shell material policy
- Keep current stable slit policy:
  - opaque, depthWrite true, depthTest true.
- Keep existing render order shell first, then separators, then caps.

5. Preserve current depth alignment behavior
- Do not modify `_resolveGeometryDepth(...)` logic unless a regression appears.

6. Add runtime flag and preserve fallback
- Gate this path behind a mode flag.
- Keep current LOD mode available for A/B.

### Validation checklist

1. Near view:
- separators are visible through slit openings.
- no obvious dimming/lighting mismatch.

2. Mid/far view:
- closed shell has no separator shimmer/bleed.
- no hard pop transitions.

3. Inside view:
- matrix should not become globally transparent.

4. System checks:
- `npm run build` pass.
- verify no material clone issues break sci-fi uniforms.

### Risks
- If `depthWrite=false` separators appear too weak or disappear at some angles, tune polygon offset and side mode.
- If separator geometry extraction is incomplete, some expected lane dividers may vanish.

### Exit criteria
- Accepted if artifacts are suppressed without visible popping and separators remain legible in slit openings.

---

## 7) Agent B Plan: Stencil Aperture Mask + Optional Depth Prepass

### Objective
Enforce a hard visibility rule in rasterization:
- Separators never render on pixels where shell already rendered.
- Separators can render through true openings.

### Core idea
Use stencil as a per-matrix aperture mask instead of distance-based hiding.

### Implementation steps

1. Renderer stencil support
- In `CoreEngine`, create renderer with stencil enabled:
  - add `stencil: true` in `WebGLRenderer` options.

2. Mesh split required
- Same as Agent A:
  - shell geometry
  - separator geometry
- Avoid using full geometry for both in stencil mode.

3. Stencil write on shell pass
- Shell mesh material state:
  - `stencilWrite = true`
  - `stencilRef = 1`
  - `stencilFunc = THREE.AlwaysStencilFunc`
  - `stencilZPass = THREE.ReplaceStencilOp`
  - `stencilFail = THREE.KeepStencilOp`
  - `stencilZFail = THREE.KeepStencilOp`
- Keep shell color/depth writes enabled.

4. Stencil test on separator pass
- Separator material state:
  - `stencilWrite = false`
  - `stencilRef = 1`
  - `stencilFunc = THREE.NotEqualStencilFunc`
  - depth settings:
    - `depthTest = true`
    - `depthWrite = false`
    - start with `depthFunc = THREE.LessDepth`

5. Prevent cross-matrix stencil contamination
- This is the hard part. Choose one and document:
  - Strategy A: clear stencil per matrix before shell draw and force shell+separator adjacency in render order.
  - Strategy B: assign scoped stencil refs (limited by stencil bits; likely not scalable for many matrices).
- Preferred initial attempt:
  - clear stencil in shell `onBeforeRender` (`renderer.clear(false, false, true)`),
  - render shell then separator consecutively.

6. Optional depth prepass
- If needed, add a colorWrite=false depth prepass mesh for shell before stencil/color pass.
- Keep this optional and behind an additional debug flag.

7. Mode flagging
- Gate with runtime mode flag.
- Preserve current LOD and Agent A mode for quick A/B.

### Validation checklist

1. Visual correctness:
- No separator bleed on closed shell at far zoom.
- Separators visible through openings.

2. Inter-object correctness:
- No obvious separator loss caused by other matrices writing stencil.

3. Performance:
- Watch for frame drops with per-mesh stencil clears.

4. Build/system:
- `npm run build` pass.

### Risks
- Stencil is global per frame; poor scoping can cause cross-object artifacts.
- Per-matrix stencil clearing can be expensive and order-sensitive.

### Exit criteria
- Accepted only if visual correctness holds across many matrices simultaneously, not just a single isolated matrix.

---

## 8) Shared Test Protocol for Both Agents

Use same manual repro to compare apples-to-apples:

1. Start app:
- `npm run dev`

2. Test locations:
- Q/K/V blocks in MHSA (many instances close together).
- MLP up/down matrices.
- Output projection matrix.

3. Camera sweeps:
- close to shell
- inside one matrix
- medium distance
- extreme zoom out
- grazing angles

4. Pass/fail observations to capture:
- separator bleed on closed shell
- separator visibility inside slit openings
- pop/jank during motion
- transparency regressions
- depth/length alignment with LayerNorm

5. Build check:
- `npm run build`

## 9) Recommendation on Priority

Run Agent A first:
- Lower implementation risk.
- Smaller renderer-wide blast radius.
- More likely to preserve current behavior with less infrastructure.

Run Agent B in parallel as a higher-robustness but higher-complexity candidate.

## 10) Deliverable Format Requested from Each Agent

Each agent should return:

1. Exact files changed.
2. Mode flag used.
3. Before/after behavior summary for near, mid, far, inside views.
4. Known tradeoffs and any residual artifact cases.
5. Build result (`npm run build`).

