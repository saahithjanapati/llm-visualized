# Weight Matrix Slit Artifact Investigation Log

## Scope
This document records all major fixes attempted so far for the dashed-line artifacts around `WeightMatrixVisualization` slit openings, specifically when slits are enabled.

Status: unresolved (user still reports the artifact).

Primary file touched:
- `/Users/saahithjanapati/Desktop/llm-visualized/src/components/WeightMatrixVisualization.js`

Related file touched (imported helper):
- `/Users/saahithjanapati/Desktop/llm-visualized/src/utils/sciFiMaterial.js` (read only; no direct edits in this investigation)

---

## Baseline Symptom
- Dashed/striped artifact appears on weight matrix surfaces when slit channels are active.
- Artifact is visible especially when zoomed out.
- User repeatedly confirmed issue persists after multiple geometry and material changes.
- Issue appears slit-specific, not a general matrix rendering issue.

---

## Chronological Attempts That Did Not Resolve the Issue

## 1. Removed CSG slit subtraction; replaced with first-principles slit mesh generation
### Change
- Removed `three-csg-ts` usage from `WeightMatrixVisualization`.
- Added deterministic geometry builders:
  - `createTrapezoidShape(...)`
  - `buildMergedSlitRanges(...)`
  - `buildSlitPlan(...)`
  - `buildCsgFreeWeightMatrixGeometry(...)`
  - helper quad/triangle emitters for slit walls/floors/mouth faces

### Rationale
- CSG booleans were the original suspected source of artifacts.
- Goal: eliminate boolean precision/coplanar artifacts completely.

### Outcome
- Did not resolve dashed slit artifacts.
- Introduced regressions in early iterations (open/continuous-looking top/bottom cuts, visible interior exposure) that were subsequently patched.

---

## 2. Rebuilt top/bottom slit-facing surfaces from strips instead of preserving original cap triangles
### Change
- Removed top/bottom triangles from generated body and rebuilt planar face strips with slit-aware openings (`addPlanarFaceWithOpenings` path).

### Rationale
- Original triangle-level filtering was deleting too much surface (large triangles crossing slit regions).

### Outcome
- Helped with early “large continuous opening” regression.
- Did not eliminate dashed artifacts.

---

## 3. Mouth-width clamping and corner handling changes
### Change
- Added safe radius handling and slit-mouth width constraints:
  - `computeSafeCornerRadius(...)`
  - effective outer-half limits at top/bottom mouth (`effectiveOuterHalfAtY`)

### Rationale
- Suspected detached slit-side geometry and slit walls extending too close to/through shoulder regions.

### Outcome
- Improved panel flushness in some views.
- Did not eliminate dashed artifacts.

---

## 4. Tightened cap-face filtering logic
### Change
- Restricted top/bottom face removal to triangles whose three vertices are on the cap plane.
- Increased normal gate strictness for cap classification.

### Rationale
- Prevent accidental deletion of shoulder/adjacent faces that can create shell tearing and visual noise.

### Outcome
- Reduced accidental stripping.
- Did not eliminate dashed artifacts.

---

## 5. Disabled corner fillets for slit-enabled geometry
### Change
- Added `_getGeometryCornerRadius()`:
  - when slits are active, corner radius forced to `0` for geometry build.

### Rationale
- Rounded corners conflicted with assumptions in slit mouth/sidewall generation and could produce detached or near-coplanar surfaces.

### Outcome
- Helped with flushness/stability.
- Did not eliminate dashed artifacts.

---

## 6. Disabled legacy post-cleanup mutation for slit-enabled meshes
### Change
- Skipped slit-enabled:
  - vertex “snap to top/bottom plane” pass
  - top/bottom normal-forcing cleanup pass

### Rationale
- Those passes were originally tuned around CSG residue and might distort slit-edge geometry.

### Outcome
- No complete fix; dashed artifacts remained.

---

## 7. Creased-normal handling changes for slit-enabled meshes
### Change
- Bypassed `toCreasedNormals(...)` when slits are active; kept recomputed vertex normals.

### Rationale
- Hard creases can over-emphasize micro-seams and show dashed edge transitions at distance.

### Outcome
- No complete fix.

---

## 8. Instanced-slice path experiments
### Change A
- Forced non-instanced path for slit-enabled matrices.

### Rationale
- Instanced lane seams were suspected as dashed artifacts.

### Outcome
- Did not solve artifact per user report.
- Also caused undesirable perceived “longer component” behavior, so this was reverted.

### Change B
- Restored original instanced eligibility logic.

### Outcome
- Preserved expected component lengths.
- Dashed artifact still present.

---

## 9. Side material transparency/transmission suppression for slit-enabled meshes
### Change
- For slit-enabled side shell:
  - `transparent = false`
  - `opacity = 1`
  - `depthWrite = true`
  - `depthTest = true`
  - transmission/thickness forced to 0
  - `side` toggled during iterations (`DoubleSide` then `FrontSide`)

### Rationale
- Transparent/transmissive blending between front/back slit interior faces can manifest as dashed-looking lines at distance.

### Outcome
- No complete fix.

---

## 10. Slit-only shader effect suppression
### Change
- Using `updateSciFiMaterialUniforms(...)` for slit-enabled side shell:
  - `stripeStrength = 0`
  - `scanlineStrength = 0`
  - `depthAccentStrength = 0`
  - `noiseStrength = 0`
  - `glintStrength = 0`

### Rationale
- Procedural shader overlays can read as edge dashing over narrow slit geometry at distance.

### Outcome
- Did not eliminate artifact.

---

## 11. Increased minimum slit wall thickness
### Change
- Introduced stronger side-wall reserve near slit mouths:
  - `requiredWall = max(1.25, outerHalf * 0.18)`

### Rationale
- Very thin residual walls near slit boundaries can alias into dashed artifacts under minification.

### Outcome
- Did not eliminate artifact.

---

## 12. Cache key/version invalidation across attempts
### Change
- Incremented geometry cache version repeatedly (`wm-fp-nocsg-v1 ... v7`).

### Rationale
- Ensure old geometry/material cache entries were not masking new changes.

### Outcome
- Prevented stale cache confusion.
- Artifact still reported.

---

## 13. Re-checked historical CSG implementation on `main`
### Change
- Reviewed `main:src/components/WeightMatrixVisualization.js` to confirm original CSG slit subtraction path and compare behavior assumptions with current branch.
- User confirmed the same dashed artifact existed with that original CSG method as well.

### Rationale
- Validate whether this was introduced by the no-CSG migration or existed beforehand.

### Outcome
- Confirms the issue is not a regression introduced by removing CSG.
- Artifact predates the first-principles migration.

---

## 14. Replaced hybrid slit build with full deterministic shell builder (`v8`)
### Change
- Removed slit-enabled dependency on `ExtrudeGeometry` surface reuse and triangle filtering.
- Added a full first-principles slit-enabled geometry path:
  - `buildDepthBreakpoints(...)`
  - `addTrapezoidSideBands(...)`
  - `addTrapezoidEndCap(...)`
  - `buildFirstPrinciplesSlitWeightMatrixGeometry(...)`
- New path explicitly emits:
  - outer side bands,
  - top/bottom planes with openings,
  - front/back end caps,
  - slit interior walls/floors/mouth faces.
- `_createMesh()` now uses the deterministic builder whenever slit cuts are active; non-slit geometry still uses `ExtrudeGeometry`.
- Bumped geometry cache version to `wm-fp-nocsg-v8`.

### Rationale
- Eliminate inherited triangulation/minification artifacts from extrude-then-remove/rebuild flows.
- Ensure slit-enabled geometry is generated from one controlled topology pipeline.

### Outcome
- Build/runtime sanity checks pass.
- User still reports the dashed artifact.

---

## 15. Rendering pipeline sanity checks
### Change
- Verified scene settings around common artifact sources:
  - shadows disabled for demo path,
  - no SSAO/outline pass in use,
  - bloom path exists but is disabled by default.

### Rationale
- Rule out common post-processing/shadow artifacts before deeper material/raster debugging.

### Outcome
- No direct smoking gun from scene-level toggles.
- Artifact remains unresolved.

---

## 16. Stabilized slit rendering in instanced path
### Change
- Updated instanced slit side material in `WeightMatrixVisualization` to match the non-instanced slit stability constraints:
  - `transparent = false`
  - `opacity = 1`
  - `depthWrite = true`
  - `depthTest = true`
  - `side = FrontSide`
  - `transmission = 0`
  - `thickness = 0`
  - slit shader overlays clamped via `updateSciFiMaterialUniforms(...)`
- Applied the same opaque/no-transmission treatment to instanced front/back cap materials.

### Rationale
- Instanced slit path still used transparent/transmissive sci-fi materials even after non-instanced slit path was hardened.
- This could preserve distance artifacts despite geometry changes.

### Outcome
- Build passes.
- Visual outcome pending user verification.

---

## 17. Removed auxiliary cap overlays for non-instanced slit-enabled matrices
### Change
- Non-instanced slit-enabled matrices now skip extra front/back cap meshes.
- Slit-enabled first-principles body mesh is treated as the authoritative watertight shell.
- Added guards around cap material dimension updates when cap meshes are absent.
- Instanced cap geometry generation was decoupled from temporary `WeightMatrixVisualization` cap meshes and now built directly from `createTrapezoidShape(...) + ShapeGeometry`.

### Rationale
- Slit-enabled geometry already includes end caps; adding separate transparent cap overlays can introduce additional depth/blend interactions visible through slit channels.

### Outcome
- Build passes.
- Visual outcome pending user verification.

---

## 18. Slit-only material path switched to plain MeshPhysicalMaterial
### Change
- Introduced `createSlitStablePhysicalMaterial(...)` and routed slit-enabled matrices to this material path (no custom sci-fi shader injection / `onBeforeCompile`).
- Applied to:
  - non-instanced slit side shell,
  - instanced slit side shell,
  - instanced slit front/back caps.
- Maintained opaque/front-side/depth-write settings and zero transmission/thickness.

### Rationale
- Prior slit material tweaks still used the sci-fi shader base, leaving gradient/fresnel/rim logic active.
- This change performs a true shader A/B by fully removing custom shader math from slit rendering while preserving overall PBR look.

### Outcome
- Build passes.
- Visual outcome pending user verification.

---

## 19. Restored default slit lighting; kept plain-material path as opt-in debug mode
### Change
- Added a debug flag `USE_PLAIN_SLIT_DEBUG_MATERIAL` sourced from:
  - URL param `?wmSlitPlainMat=1`, or
  - runtime global `window.__WM_SLIT_PLAIN_MATERIAL = true`.
- Default behavior now uses the normal slit sci-fi material path again (with prior slit stability clamps) to preserve scene lighting.
- Plain `MeshPhysicalMaterial` slit path from attempt 18 remains available only when debug flag is enabled.

### Rationale
- User reported that attempt 18 made the whole scene look darker even though slit artifacts persisted.
- Need to preserve baseline visual quality while retaining A/B tooling for diagnosis.

### Outcome
- Build passes.
- Scene lighting should match pre-attempt-18 baseline by default.

---

## 20. Added logarithmic-depth toggle for runtime A/B testing
### Change
- In `CoreEngine`, renderer creation now supports disabling log-depth without code edits:
  - URL param: `?wmNoLogDepth=1`
  - Runtime global: `window.__WM_NO_LOG_DEPTH = true`
- Default remains unchanged (`logarithmicDepthBuffer: true`) unless explicitly disabled.

### Rationale
- Next diagnostic target is depth precision interaction. This toggle enables immediate A/B comparison in the same build.

### Outcome
- Build passes.
- Visual outcome pending user verification.

---

## 21. Welded slit geometry vertices to remove segment-boundary shading seams
### Change
- Updated slit first-principles geometry finalization to call `mergeVertices(...)` before normal generation.
- Added `mergeVertices` import from `BufferGeometryUtils`.
- Slit geometry now computes normals on a welded indexed mesh rather than fully non-indexed duplicated vertices.

### Rationale
- Non-indexed slit construction duplicates vertices across coplanar strip boundaries.
- That can create per-segment normal discontinuities that look like dashed/stippled lines at distance, especially along long top/bottom runs.

### Outcome
- Build passes.
- Visual outcome pending user verification.

---

## 22. Reverted vertex weld; removed side-surface Z segmentation
### Change
- Reverted slit geometry vertex welding (`mergeVertices`) due visible gradient regression.
- Bumped geometry cache version to `wm-fp-nocsg-v9`.
- Reworked first-principles side shell generation:
  - removed side-surface banding at slit `z` breakpoints,
  - now emits one continuous quad per trapezoid side (`addTrapezoidSideSurface(...)`).

### Rationale
- Vertex welding introduced broad shading gradients across matrix surfaces.
- Side-surface band segmentation itself can create subtle seam bands along long coplanar faces; removing those boundaries targets dashed/diagonal seam visibility without changing slit topology.

### Outcome
- Build passes.
- Visual outcome pending user verification.

---

## Instrumentation/Checks Performed
- Multiple successful production builds (`npm run build`) after each major change.
- Quick local geometry sanity checks (finite vertices, geometry creation for representative matrix presets).
- Post-`v8` deterministic rewrite:
  - successful `npm run build`,
  - Node-side geometry sanity checks for slit and non-slit matrices (finite vertex buffers and valid bounding boxes).
- Post-instanced/cap stabilization changes:
  - successful `npm run build`.
- Post plain-material slit path change:
  - successful `npm run build`.
- Post debug-toggle restoration change:
  - successful `npm run build`.
- Post log-depth toggle addition:
  - successful `npm run build`.
- Post vertex-weld normal pass:
  - successful `npm run build`.
- Post side-surface continuity rewrite (`v9`):
  - successful `npm run build`.
- Temporary edge-topology scripts were used during debugging and removed.

Note: these checks validated build/runtime integrity, not final visual acceptance.

---

## Current Working Hypothesis (Still Unproven)
- Artifact is likely not a single-source bug from one toggle.
- Most likely candidates now:
  - interaction between slit interior geometry and the custom sci-fi shader under distance/minification,
  - raster/depth precision behavior for narrow slit faces at distance,
  - or subtle shell overlap/ordering behavior not fully controlled by current material flags.

Given `main` CSG and both no-CSG builders show the same symptom, root cause is likely downstream of CSG itself.

---

## What Was Explicitly Reverted
- Forcing non-instanced geometry for all slit-enabled matrices was reverted because it changed perceived component lengths.

---

## Current Code State Snapshot
At the time of this log:
- CSG remains removed from `WeightMatrixVisualization`.
- Slit generation uses a fully deterministic slit-enabled path (`buildFirstPrinciplesSlitWeightMatrixGeometry`) that does not reuse slit-region `ExtrudeGeometry` triangles.
- Slit-only material/shader clamps are active.
- Slit-only geometry wall-thickness clamp is active.
- Geometry cache version is currently `wm-fp-nocsg-v8`.
- Instanced slit materials are now forced opaque/front-side with transmission disabled.
- Non-instanced slit-enabled matrices skip auxiliary cap overlay meshes.
- Slit-enabled matrices use the standard slit sci-fi path by default.
- Plain slit material path is available only via `?wmSlitPlainMat=1` (or `window.__WM_SLIT_PLAIN_MATERIAL = true`) for debugging.
- Log-depth buffer can be disabled via `?wmNoLogDepth=1` (or `window.__WM_NO_LOG_DEPTH = true`) for depth-precision A/B.
- Slit first-principles side surfaces are now continuous quads (no slit-boundary side band splits).
- Artifact still reported by user.

---

## Suggested Next Diagnostic Step (Not Yet Applied)
To isolate root cause cleanly, run a strict A/B at runtime for slit-enabled side meshes only:
- Mode A: plain `THREE.MeshStandardMaterial` (no custom sci-fi shader, opaque, front-side).
- Mode B: current sci-fi shader material.

If A removes artifacts and B restores them, the issue is shader/minification/depth interaction.
If both retain artifacts, issue is likely raster/depth precision or topology at slit scales.

Follow-up if both retain artifacts:
- run one pass with `WebGLRenderer({ logarithmicDepthBuffer: false })` to test depth precision interaction with slit bands.

---

## 23. Full CSG-free rewrite of `WeightMatrixVisualization` (`wm-fp-v10`)
### Change
- Replaced `src/components/WeightMatrixVisualization.js` with a new deterministic implementation that does not import or use `three-csg-ts`.
- Added first-principles slit shell generation using explicit quad/triangle emission (`TriangleBuilder`) instead of boolean subtraction.
- Slit-enabled matrices now build one watertight body mesh directly (front/back caps + side shell + top/bottom slit openings + slit interior walls).
- Non-slit matrices continue to use an extruded trapezoid path.
- Kept instanced-slice support, but switched it to the same deterministic geometry path for slice generation.
- Introduced geometry cache version key suffix: `wm-fp-v10`.

### Rationale
- Prior local file state had drifted back to a CSG-based implementation.
- A full rewrite ensures slit topology is controlled end-to-end with no boolean/coplanar residue from subtraction workflows.

### Outcome
- `npm run build` passes.
- Node-side geometry sanity checks pass for representative slit/non-slit presets (finite bounds, mesh creation succeeds).
- Visual outcome pending user verification.

---

## 24. Legacy precomputed cache-key compatibility mapping
### Change
- Updated `WeightMatrixVisualization.registerPrecomputedGeometry(...)` to register both:
  - legacy keys (without version suffix), and
  - normalized keys with `|wm-fp-v10`.

### Rationale
- Existing precomputed assets and scripts currently emit legacy weight-matrix cache keys.
- Without mapping, the new versioned key format would bypass precomputed cache hits.

### Outcome
- Build passes after compatibility patch.
- Runtime can consume legacy precomputed geometry entries while still using the new versioned key path.

---

## Addendum: Current Code State (After Attempt 24)
- `WeightMatrixVisualization` is now fully CSG-free in this branch.
- Slit geometry is first-principles and deterministic (`TriangleBuilder` path).
- Geometry cache key includes version suffix `wm-fp-v10`.
- Legacy precomputed key mapping is active in `registerPrecomputedGeometry(...)`.
- Latest checks run:
  - `npm run build` (pass),
  - Node geometry sanity script for multiple matrix presets (pass).

---

## 25. Slit-only sci-fi uniform suppression on v10 geometry
### Change
- Imported `updateSciFiMaterialUniforms` into `WeightMatrixVisualization`.
- Added `SLIT_SHADER_STABILITY_UNIFORMS` and apply it only for slit-enabled materials (side + caps):
  - `stripeStrength = 0`
  - `depthAccentStrength = 0`
  - `scanlineStrength = 0`
  - `glintStrength = 0`
  - `noiseStrength = 0`
- Kept the same base sci-fi material path (no switch to plain `MeshPhysicalMaterial`).

### Rationale
- Even with deterministic slit geometry, procedural slit shader overlays can still read as dashed/stippled artifacts under minification.
- This keeps scene lighting/palette style while removing high-frequency slit overlays.

### Outcome
- `npm run build` passes after this change.
- Visual outcome pending user verification.

---

## 26. Removed instanced-slice rendering for slit-enabled matrices (with lane-span depth matching)
### Change
- Updated `_createMesh()` instancing gate so slit-enabled matrices no longer use `_createInstancedSlices()`.
- Added `_resolveGeometryDepth()`:
  - For slit-enabled lane-dependent configs near `depth ~= (N+1)*VECTOR_DEPTH_SPACING`, geometry depth is normalized to `N*VECTOR_DEPTH_SPACING` (matching historical instanced visual span).
  - Otherwise, keep caller-provided `depth`.
- Routed cache keys, mesh build, cap placement, and sci-fi dimension uniforms through resolved geometry depth.
- Kept instanced path active for non-slit matrices.

### Rationale
- User screenshots continue to show seam-like artifacts consistent with per-slice lane boundaries.
- Deterministic full-depth slit meshes are now cheap enough to render directly; this removes the instanced seam boundary class while preserving historical depth span in canonical lane configs.

### Outcome
- `npm run build` passes.
- Quick runtime check confirms:
  - slit-enabled matrices are non-instanced,
  - non-slit matrices still use instanced path when eligible.
- Visual outcome pending user verification.

---

## 27. Strengthened slit geometry anti-alias guards (minimum slit gaps + thicker side reserves)
### Change
- Updated `buildMergedSlitRanges(...)`:
  - slit width now clamps to preserve a minimum inter-slit gap (`minGap = max(1.0, spacing * 0.08)`).
- Updated `resolveHoleHalfWidth(...)`:
  - increased minimum side reserve from `max(1.25, outerHalf * 0.03)` to `max(1.5, outerHalf * 0.08)`.

### Rationale
- Extremely thin residual walls and near-touching slit windows can minify into dashed/stippled edge artifacts.
- These guards keep slit topology visually robust under distance while retaining the same overall form.

### Outcome
- `npm run build` passes.
- Node-side geometry sanity checks pass for representative slit presets (finite bounds, geometry creation succeeds).
- Visual outcome pending user verification.
