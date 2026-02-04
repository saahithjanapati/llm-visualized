import * as THREE from 'three';
import BaseLayer from '../BaseLayer.js';
import { LayerNormalizationVisualization } from '../../components/LayerNormalizationVisualization.js';
import { WeightMatrixVisualization } from '../../components/WeightMatrixVisualization.js';
import { VectorVisualizationInstancedPrism } from '../../components/VectorVisualizationInstancedPrism.js';
import { BatchedPrismVectorSet } from '../../components/BatchedPrismVectorSet.js';
import { StraightLineTrail, buildMergedLineSegmentsFromSegments, collectTrailsUnder, mergeTrailsIntoLineSegments } from '../../utils/trailUtils.js';
import { TRAIL_MIN_SEGMENT_DISTANCE } from '../../utils/trailConstants.js';
import {
    LN_PARAMS,
    LN_NORM_START_FRACTION_FROM_BOTTOM,
    LN_TO_MHA_GAP,
    BRANCH_X,
    LN2_TO_MLP_GAP,
    LAYER_NORM_1_Y_POS,
    LAYER_NORM_2_Y_POS,
    MLP_MATRIX_PARAMS_UP,
    MLP_MATRIX_PARAMS_DOWN,
    MLP_INTER_MATRIX_GAP,
    NUM_VECTOR_LANES,
    ANIM_RISE_SPEED_ORIGINAL,
    VECTOR_LENGTH_PRISM,
    GLOBAL_ANIM_SPEED_MULT,
    ANIM_RISE_SPEED_INSIDE_LN,
    ANIM_RISE_SPEED_POST_SPLIT_LN1,
    ANIM_RISE_SPEED_POST_SPLIT_LN2,
    ORIGINAL_TO_PROCESSED_GAP,
    ANIM_HORIZ_SPEED,
    INACTIVE_COMPONENT_COLOR,
    PRISM_ADD_ANIM_BASE_DURATION,
    PRISM_ADD_ANIM_BASE_FLASH_DURATION,
    PRISM_ADD_ANIM_BASE_DELAY_BETWEEN_PRISMS,
    PRISM_ADD_ANIM_SPEED_MULT,
    LAYER_STACK_SPACING_Y,
    SKIP_COMPONENT_COLOR_LERP_ALPHA,
    SKIP_MLP_COLOR_MIN_MS,
    SKIP_TRAIL_FADE_IN_MS
} from '../../utils/constants.js';
import { PrismLayerNormAnimation } from '../../animations/PrismLayerNormAnimation.js';
import { MHSAAnimation } from '../../animations/MHSAAnimation.js';
import { startPrismAdditionAnimation } from '../../utils/additionUtils.js';
import { getLayerNormParamData } from '../../data/layerNormParams.js';
import {
    applyLayerNormMaterial,
    applyVectorData,
    copyVectorAppearance,
    freezeStaticTransforms,
    formatTokenLabel,
    geluApprox,
    LN_INTERNAL_TRAIL_MIN_SEGMENT
} from './gpt2LayerUtils.js';
import {
    buildSingleLane,
    createAdditionPlaceholders,
    createFreshLanes,
    createLanesFromExternal,
    LN_PARAM_MONOCHROME
} from './gpt2LaneBuilder.js';


// Slightly reduced spacing between stacked layers for a tighter layout.
// Keep this just above the per-layer vertical extent so MLP tops don't collide.
const DEFAULT_LAYER_STACK_SPACING = LAYER_STACK_SPACING_Y;
// Reusable scratch vector to avoid per-frame allocations when working with
// world-space trail coordinates.
const TMP_WORLD_POS = new THREE.Vector3();

// Shared colour constants reused across the layer to avoid per-frame
// allocations inside the animation loop.
const COLOR_DARK_GRAY = new THREE.Color(0x333333);
const COLOR_LIGHT_YELLOW = new THREE.Color(0xffffff);
const COLOR_BRIGHT_YELLOW = new THREE.Color(0xffffff);
const COLOR_INACTIVE_COMPONENT = new THREE.Color(INACTIVE_COMPONENT_COLOR);

const TMP_LN_TRAIL_POS = new THREE.Vector3();

const MLP_REFLECTIVITY_TWEAKS = {
    roughnessMin: 0.4,
    metalnessMax: 0.65,
    clearcoatMax: 0.6,
    clearcoatRoughnessMin: 0.45,
    iridescenceMax: 0.25,
    envMapIntensityMax: 1.1
};

const applyMatrixReflectivityTweak = (matrix, tweaks) => {
    if (!matrix || !tweaks) return;
    const applyToMaterial = (mat) => {
        if (!mat) return;
        const mats = Array.isArray(mat) ? mat : [mat];
        mats.forEach(m => {
            if (!m) return;
            if (typeof tweaks.roughnessMin === 'number' && typeof m.roughness === 'number') {
                m.roughness = Math.max(m.roughness, tweaks.roughnessMin);
            }
            if (typeof tweaks.metalnessMax === 'number' && typeof m.metalness === 'number') {
                m.metalness = Math.min(m.metalness, tweaks.metalnessMax);
            }
            if (typeof tweaks.clearcoatMax === 'number' && typeof m.clearcoat === 'number') {
                m.clearcoat = Math.min(m.clearcoat, tweaks.clearcoatMax);
            }
            if (typeof tweaks.clearcoatRoughnessMin === 'number' && typeof m.clearcoatRoughness === 'number') {
                m.clearcoatRoughness = Math.max(m.clearcoatRoughness, tweaks.clearcoatRoughnessMin);
            }
            if (typeof tweaks.iridescenceMax === 'number' && typeof m.iridescence === 'number') {
                m.iridescence = Math.min(m.iridescence, tweaks.iridescenceMax);
            }
            if (typeof tweaks.envMapIntensityMax === 'number' && typeof m.envMapIntensity === 'number') {
                m.envMapIntensity = Math.min(m.envMapIntensity, tweaks.envMapIntensityMax);
            }
        });
    };
    applyToMaterial(matrix.mesh?.material);
    applyToMaterial(matrix.frontCapMesh?.material);
    applyToMaterial(matrix.backCapMesh?.material);
};


export default class Gpt2Layer extends BaseLayer {
    /**
     * @param {number} index – 0-based index of this layer in the transformer.
     * @param {object} random – object returned by createRandomSource().
     * @param {number} yOffset – Optional y-offset for this layer.
     * @param {Layer} waitForLayer – Optional layer to wait for before starting.
     * @param {Array} externalLanes – Optional array of lanes to re-use.
     * @param {Function} onFinished – Optional callback to invoke when all lanes finish.
     * @param {object} activationSource – Optional capture data source for real activations.
     */
    constructor(index, random, yOffset = 0, externalLanes = null, onFinished = null, isActive = true, activationSource = null, laneCount = NUM_VECTOR_LANES, layerSpacing = DEFAULT_LAYER_STACK_SPACING) {
        super(index);
        this.random = random;
        this.yOffset = yOffset;
        this.externalLanes = externalLanes;
        this.onFinished = typeof onFinished === 'function' ? onFinished : null;
        this.isActive = isActive;
        this.activationSource = activationSource || null;
        this._laneCount = Math.max(1, Math.floor(laneCount || NUM_VECTOR_LANES));
        this._layerStackSpacing = Number.isFinite(layerSpacing) ? layerSpacing : DEFAULT_LAYER_STACK_SPACING;
        this._baseVectorLength = (this.activationSource && typeof this.activationSource.getBaseVectorLength === 'function')
            ? this.activationSource.getBaseVectorLength()
            : VECTOR_LENGTH_PRISM;
        this._completed = false;
        this._pendingAdditions = 0; // Track ongoing addition animations
        // Synchronisation flag – ensures all lanes begin the LN-2 branch at the exact same time.
        this._ln2Ready = false;
        // Flag to indicate that all lanes have reached MLP readiness.
        this._mlpStart = false;
        // NEW: Barrier flags
        this._ln1Start = false;        // start LN-1 branch
        this._mhsaStart = false;       // start horizontal travel to heads
        this._progressEmitter = null;  // external emitter for progress events
        this._skipToEndActive = false;
        this._skipConcatTriggered = false;
        this._skipHiddenMaterials = new WeakMap();
        this.raycastRoot = null;

        // Placeholder vectors shown inside inactive LayerNorms so the
        // scale/shift prisms are visible throughout the full stack even
        // before a layer becomes active. When a layer is activated the
        // placeholders are re-used as the live targets and removed
        // from these arrays (see _buildSingleLane).
        this._ln1AddPlaceholders = [];
        this._ln2AddPlaceholders = [];
        this._ln1ScalePlaceholders = [];
        this._ln2ScalePlaceholders = [];

        // Cached colours reused during per-frame updates to avoid heap churn.
        this._ln1TargetColor = new THREE.Color();
        this._ln2TargetColor = new THREE.Color();
        this._ln1LockedColor = new THREE.Color();
        this._ln1ColorLocked = false;
        this._ln2LockedColor = new THREE.Color();
        this._ln2ColorLocked = false;
        this._ln1MaterialState = { color: new THREE.Color(), opacity: 1.0, transparent: false, initialized: false };
        this._ln2MaterialState = { color: new THREE.Color(), opacity: 1.0, transparent: false, initialized: false };
        this._trailUpdateFrameId = 0;
        this._vecsToCheckScratch = new Array(9);
    }

    setProgressEmitter(emitter) { this._progressEmitter = emitter; }
    _emitProgress() {
        if (this._progressEmitter && typeof this._progressEmitter.dispatchEvent === 'function') {
            this._progressEmitter.dispatchEvent(new Event('progress'));
        }
    }

    init(scene) {
        super.init(scene);

        // Keep a reference to the global scene so certain visuals (like the
        // continuous residual-stream trails) can render in world-space and
        // survive lane hand-offs between stacked layers.
        this._globalScene = scene;

        // Offset root vertically for stack layout
        this.root.position.y = this.index * this._layerStackSpacing + this.yOffset;
        freezeStaticTransforms(this.root);

        this.raycastRoot = new THREE.Group();
        this.raycastRoot.name = `Layer${this.index}_RaycastRoot`;
        this.raycastRoot.userData.layerIndex = this.index;
        this.root.add(this.raycastRoot);

        const offsetX = BRANCH_X; // all branched components share this X

        // ────────────────────────────────────────────────────────────────
        // 1) LayerNorm 1
        // ────────────────────────────────────────────────────────────────
        const ln1CenterY = LAYER_NORM_1_Y_POS;
        const ln1 = new LayerNormalizationVisualization(
            new THREE.Vector3(offsetX, ln1CenterY, 0),
            LN_PARAMS.width,
            LN_PARAMS.height,
            LN_PARAMS.depth,
            LN_PARAMS.wallThickness,
            LN_PARAMS.numberOfHoles,
            LN_PARAMS.holeWidth,
            LN_PARAMS.holeWidthFactor
        );
        // Start LN1 in the same "inactive" palette used for LN2 so it only lights up once
        // vectors begin to enter the ring.
        const inactiveDark = COLOR_INACTIVE_COMPONENT;
        ln1.setColor(inactiveDark);
        // Start fully opaque to avoid early depth-sorting costs.
        ln1.setMaterialProperties({ opacity: 1.0, transparent: false, emissiveIntensity: 0.05 });
        this.raycastRoot.add(ln1.group);
        freezeStaticTransforms(ln1.group, true);

        const ln1TopY = ln1CenterY + LN_PARAMS.height / 2;
        this.ln1TopY = ln1TopY;

        // ────────────────────────────────────────────────────────────────
        // 2) Multi-Head Self-Attention matrices
        // ────────────────────────────────────────────────────────────────
        // Skipped: Legacy per-layer Q/K/V matrix group construction.
        // MHSA visuals are now created exclusively by MHSAAnimation to avoid
        // duplicate geometry and unnecessary CPU/GPU overhead.

        // ────────────────────────────────────────────────────────────────
        //  MHSAAnimation controller – handles vector routing through the
        //  12-head attention block and subsequent merging logic.
        // ────────────────────────────────────────────────────────────────

        const mhaBaseY_local = ln1TopY + LN_TO_MHA_GAP;
        const mhaBaseY = mhaBaseY_local;
        // Create an internal clock for sub-animations handled by the MHSA helper
        this._mhsaClock = new THREE.Clock();
        // Pass empty opts – twelve-layer stack keeps self-attention disabled via global static flag.
        this.mhsaAnimation = new MHSAAnimation(this.raycastRoot, BRANCH_X, mhaBaseY, this._mhsaClock, 'temp', {
            activationSource: this.activationSource,
            layerIndex: this.index,
            vectorPrismCount: this._getBaseVectorLength(),
            laneCount: this._laneCount,
        });

        // ────────────────────────────────────────────────────────────────
        // 2.5) Output-projection matrix
        //      Now created solely inside MHSAAnimation.  The former static
        //      placeholder has been removed to stop overlapping transparent
        //      meshes that caused colour flickering.

        // ────────────────────────────────────────────────────────────────
        // 3) LayerNorm 2
        // ────────────────────────────────────────────────────────────────
        const ln2CenterY = LAYER_NORM_2_Y_POS + (this.mhsaAnimation?.outputProjMatrixYOffset || 0);
        const ln2 = new LayerNormalizationVisualization(
            new THREE.Vector3(offsetX, ln2CenterY, 0),
            LN_PARAMS.width,
            LN_PARAMS.height,
            LN_PARAMS.depth,
            LN_PARAMS.wallThickness,
            LN_PARAMS.numberOfHoles,
            LN_PARAMS.holeWidth,
            LN_PARAMS.holeWidthFactor
        );
        ln2.setColor(inactiveDark);
        ln2.setMaterialProperties({ opacity: 1.0, transparent: false, emissiveIntensity: 0.05 });
        this.raycastRoot.add(ln2.group);
        freezeStaticTransforms(ln2.group, true);

        const ln2TopY = ln2CenterY + LN_PARAMS.height / 2;

        // Create LayerNorm parameter banks (shared across lanes) so both active
        // and inactive layers show the same LN scale/shift visuals.
        this._createAdditionPlaceholders(offsetX, ln1CenterY, ln2CenterY);

        // ────────────────────────────────────────────────────────────────
        // 4) MLP Up-projection matrix (orange)
        // ────────────────────────────────────────────────────────────────
        const mlpUpCenterY = ln2TopY + LN2_TO_MLP_GAP + MLP_MATRIX_PARAMS_UP.height / 2;
        const mlpUp = new WeightMatrixVisualization(
            null,
            new THREE.Vector3(offsetX, mlpUpCenterY, 0),
            MLP_MATRIX_PARAMS_UP.width,
            MLP_MATRIX_PARAMS_UP.height,
            MLP_MATRIX_PARAMS_UP.depth,
            MLP_MATRIX_PARAMS_UP.topWidthFactor,
            MLP_MATRIX_PARAMS_UP.cornerRadius,
            MLP_MATRIX_PARAMS_UP.numberOfSlits,
            MLP_MATRIX_PARAMS_UP.slitWidth,
            MLP_MATRIX_PARAMS_UP.slitDepthFactor,
            MLP_MATRIX_PARAMS_UP.slitBottomWidthFactor,
            MLP_MATRIX_PARAMS_UP.slitTopWidthFactor
        );
        mlpUp.setColor(inactiveDark.clone());
        mlpUp.setMaterialProperties({ opacity: 1.0, transparent: false, emissiveIntensity: 0.08 });
        applyMatrixReflectivityTweak(mlpUp, MLP_REFLECTIVITY_TWEAKS);
        {
            const lbl = 'MLP Up Weight Matrix';
            mlpUp.group.userData.label = lbl;
            if (mlpUp.mesh) mlpUp.mesh.userData.label = lbl;
            if (mlpUp.frontCapMesh) mlpUp.frontCapMesh.userData.label = lbl;
            if (mlpUp.backCapMesh)  mlpUp.backCapMesh.userData.label  = lbl;
        }
        this.raycastRoot.add(mlpUp.group);
        freezeStaticTransforms(mlpUp.group, true);

        // 5) MLP Down-projection matrix (same orange)
        const mlpDownCenterY = mlpUpCenterY + MLP_MATRIX_PARAMS_UP.height / 2 + MLP_INTER_MATRIX_GAP + MLP_MATRIX_PARAMS_DOWN.height / 2;
        const mlpDown = new WeightMatrixVisualization(
            null,
            new THREE.Vector3(offsetX, mlpDownCenterY, 0),
            MLP_MATRIX_PARAMS_DOWN.width,
            MLP_MATRIX_PARAMS_DOWN.height,
            MLP_MATRIX_PARAMS_DOWN.depth,
            MLP_MATRIX_PARAMS_DOWN.topWidthFactor,
            MLP_MATRIX_PARAMS_DOWN.cornerRadius,
            MLP_MATRIX_PARAMS_DOWN.numberOfSlits,
            MLP_MATRIX_PARAMS_DOWN.slitWidth,
            MLP_MATRIX_PARAMS_DOWN.slitDepthFactor,
            MLP_MATRIX_PARAMS_DOWN.slitBottomWidthFactor,
            MLP_MATRIX_PARAMS_DOWN.slitTopWidthFactor
        );
        mlpDown.setColor(inactiveDark.clone());
        mlpDown.setMaterialProperties({ opacity: 1.0, transparent: false, emissiveIntensity: 0.1 });
        applyMatrixReflectivityTweak(mlpDown, MLP_REFLECTIVITY_TWEAKS);
        {
            const lbl = 'MLP Down Weight Matrix';
            mlpDown.group.userData.label = lbl;
            if (mlpDown.mesh) mlpDown.mesh.userData.label = lbl;
            if (mlpDown.frontCapMesh) mlpDown.frontCapMesh.userData.label = lbl;
            if (mlpDown.backCapMesh)  mlpDown.backCapMesh.userData.label  = lbl;
        }
        this.raycastRoot.add(mlpDown.group);
        freezeStaticTransforms(mlpDown.group, true);

        // ---------- Residual vectors (original stream) ----------
        this.lanes = [];
        if (this.isActive) {
            if (this.externalLanes && this.externalLanes.length) {
                this._createLanesFromExternal(this.externalLanes, offsetX, ln1CenterY, ln2CenterY, ln1TopY);
            } else {
                this._createFreshLanes(offsetX, ln1CenterY, ln2CenterY, ln1TopY);
            }
        }

        // Keep references for per-frame updates
        this.ln1 = ln1;
        this.ln2 = ln2;
        this.mlpUp = mlpUp;
        this.mlpDown = mlpDown;
    }

    update(dt) {
        const skipActive = this._skipToEndActive;
        const bottomY_ln1_abs = LAYER_NORM_1_Y_POS - LN_PARAMS.height / 2;
        const midY_ln1_abs    = LAYER_NORM_1_Y_POS;
        const topY_ln1_abs    = LAYER_NORM_1_Y_POS + LN_PARAMS.height / 2;
        const ln2CenterY = this.ln2 && this.ln2.group && Number.isFinite(this.ln2.group.position.y)
            ? this.ln2.group.position.y
            : LAYER_NORM_2_Y_POS;
        const ln2CenterX = this.ln2 && this.ln2.group && Number.isFinite(this.ln2.group.position.x)
            ? this.ln2.group.position.x
            : BRANCH_X;
        const ln2EntryXEpsilon = 0.5;
        const bottomY_ln2_abs = ln2CenterY - LN_PARAMS.height / 2;
        const midY_ln2_abs    = ln2CenterY;
        const topY_ln2_abs    = ln2CenterY + LN_PARAMS.height / 2;
        const lanes = Array.isArray(this.lanes) ? this.lanes : [];
        const laneCount = lanes.length;
        const exitTransitionRange = 5; // world–unit distance for final fade
        const needsPositioningCheck = this._transitionPhase === 'positioning';
        let allVectorsInPosition = needsPositioningCheck && laneCount > 0;
        let posAddDone = true;
        let allLn1Ready = !this._ln1Start && laneCount > 0;
        let allMhsaReady = !this._mhsaStart && laneCount > 0;
        let allLn2Ready = !this._ln2Ready && laneCount > 0;
        let allMlpReady = !this._mlpStart && laneCount > 0;
        let highestLN1VecY = -Infinity;
        let anyVectorInLN1 = false;
        let highestLN2VecY = -Infinity;
        let anyVectorInLN2 = false;
        const ln2SyncY = bottomY_ln2_abs + 5;
        this._trailUpdateFrameId = (this._trailUpdateFrameId + 1) | 0;
        // ──────────────── Update straight-line trails ────────────────
        if (laneCount) {
            lanes.forEach(lane => {
                const vecsToCheck = this._vecsToCheckScratch;
                vecsToCheck[0] = lane.originalVec;
                vecsToCheck[1] = lane.dupVec;
                vecsToCheck[2] = lane.travellingVec;
                vecsToCheck[3] = lane.resultVec;
                vecsToCheck[4] = lane.postAdditionVec;
                vecsToCheck[5] = lane.movingVecLN2;
                vecsToCheck[6] = lane.resultVecLN2;
                vecsToCheck[7] = lane.finalVecAfterMlp;
                vecsToCheck[8] = lane.posVec;

                const trailFrameId = this._trailUpdateFrameId;

                for (let i = 0; i < vecsToCheck.length; i++) {
                    const v = vecsToCheck[i];
                    if (!v || !v.userData || !v.userData.trail) continue;
                    // Let MHSA routing own the travel trail updates to preserve sharp corners.
                    if (v === lane.travellingVec && (lane.horizPhase === 'readyMHSA' || lane.horizPhase === 'travelMHSA')) {
                        continue;
                    }
                    const trailRef = v.userData.trail;
                    if (trailRef.__lastUpdateFrameId === trailFrameId) continue;
                    trailRef.__lastUpdateFrameId = trailFrameId;

                    // During residual addition, let MHSAAnimation drive world-space
                    // residual trail updates (it follows the centre prism). Skip here
                    // to avoid double-writing the same world trail in the same frame.
                    if (lane.stopRise && v.userData.trailWorld) continue;

                    if (!v.userData.trailWorld) {
                        const zPos = Number.isFinite(lane.zPos) ? lane.zPos : v.group.position.z;
                        if (
                            lane.horizPhase === 'insideLN'
                            && (v === lane.dupVec || v === lane.resultVec)
                        ) {
                            const clampedY = Math.min(topY_ln1_abs, Math.max(bottomY_ln1_abs, v.group.position.y));
                            TMP_LN_TRAIL_POS.set(BRANCH_X, clampedY, zPos);
                            trailRef.update(TMP_LN_TRAIL_POS);
                            continue;
                        }
                        if (
                            lane.ln2Phase === 'insideLN'
                            && (v === lane.movingVecLN2 || v === lane.resultVecLN2)
                        ) {
                            const clampedY = Math.min(topY_ln2_abs, Math.max(bottomY_ln2_abs, v.group.position.y));
                            TMP_LN_TRAIL_POS.set(BRANCH_X, clampedY, zPos);
                            trailRef.update(TMP_LN_TRAIL_POS);
                            continue;
                        }
                    }

                    if (v.userData.trailWorld) {
                        v.group.getWorldPosition(TMP_WORLD_POS);
                        trailRef.update(TMP_WORLD_POS);
                    } else {
                        trailRef.update(v.group.position);
                    }
                }

                // Aggregate LN1/LN2 state and readiness flags in a single pass.
                const dupVec = lane.dupVec;
                if (dupVec && dupVec.group && dupVec.group.visible) {
                    const y = dupVec.group.position.y;
                    highestLN1VecY = Math.max(highestLN1VecY, y);
                    if (y >= bottomY_ln1_abs - exitTransitionRange) anyVectorInLN1 = true;
                }
                const resultVec = lane.resultVec;
                if (resultVec && resultVec.group && resultVec.group.visible) {
                    const y = resultVec.group.position.y;
                    highestLN1VecY = Math.max(highestLN1VecY, y);
                    if (y >= bottomY_ln1_abs - exitTransitionRange) anyVectorInLN1 = true;
                }

                const movingLn2 = lane.movingVecLN2;
                if (movingLn2 && movingLn2.group && movingLn2.group.visible) {
                    const x = movingLn2.group.position.x;
                    if (Math.abs(x - ln2CenterX) <= ln2EntryXEpsilon) {
                        const y = movingLn2.group.position.y;
                        highestLN2VecY = Math.max(highestLN2VecY, y);
                        if (y >= bottomY_ln2_abs - exitTransitionRange) anyVectorInLN2 = true;
                    }
                } else {
                    const resultLn2 = lane.resultVecLN2;
                    if (resultLn2 && resultLn2.group && resultLn2.group.visible) {
                        const x = resultLn2.group.position.x;
                        if (Math.abs(x - ln2CenterX) <= ln2EntryXEpsilon) {
                            const y = resultLn2.group.position.y;
                            highestLN2VecY = Math.max(highestLN2VecY, y);
                            if (y >= bottomY_ln2_abs - exitTransitionRange) anyVectorInLN2 = true;
                        }
                    }
                }

                if (needsPositioningCheck && allVectorsInPosition) {
                    const ov = lane.originalVec;
                    const targetY = lane.branchStartY;
                    if (!ov || !ov.group || !Number.isFinite(targetY) || ov.group.position.y < targetY - 5) {
                        allVectorsInPosition = false;
                    }
                }
                if (this.index === 0 && posAddDone) {
                    if (lane.posVec && !lane.posAddComplete) {
                        posAddDone = false;
                    }
                }
                if (allLn1Ready) {
                    const ov = lane.originalVec;
                    const targetY = lane.branchStartY;
                    if (!ov || !ov.group || !Number.isFinite(targetY) || ov.group.position.y < targetY) {
                        allLn1Ready = false;
                    }
                }
                if (allMhsaReady && lane.horizPhase !== 'readyMHSA') {
                    allMhsaReady = false;
                }
                if (allLn2Ready) {
                    const ready = lane.ln2Phase === 'preRise'
                        && lane.postAdditionVec
                        && lane.postAdditionVec.group
                        && lane.postAdditionVec.group.position.y >= ln2SyncY - 0.01;
                    if (!ready) allLn2Ready = false;
                }
                if (allMlpReady && lane.ln2Phase !== 'mlpReady') {
                    allMlpReady = false;
                }

                // Expanded 4× vector group trail
                if (lane.expandedVecGroup && lane.expandedVecTrail) {
                    lane.expandedVecTrail.update(lane.expandedVecGroup.position);
                }

                // Trails for K/Q/V copies are updated inside VectorRouter.
                // Avoid double-updating here to reduce CPU work.
            });
        }
        // Handle transition phase - wait for vectors to reach position
        if (this._transitionPhase === 'positioning') {
            if (allVectorsInPosition) {
                this._transitionPhase = 'complete';
                this.isActive = true; // now start the actual animation
                console.log(`Layer ${this.index}: All vectors in position, starting animation`);
            } else {
                // Keep vectors rising toward the target
                lanes.forEach(lane => {
                    const targetY = lane.branchStartY;
                    if (lane.originalVec.group.position.y < targetY) {
                        lane.originalVec.group.position.y = Math.min(targetY, 
                            lane.originalVec.group.position.y + ANIM_RISE_SPEED_ORIGINAL * GLOBAL_ANIM_SPEED_MULT * dt);
                    }
                });
                if (skipActive) this._applySkipVectorVisibility();
                return; // keep waiting
            }
        }

        if (!this.isActive) {
            if (skipActive) this._applySkipVectorVisibility();
            return; // Skip processing when inactive / placeholder
        }

        if (skipActive && this.mhsaAnimation && !this._skipConcatTriggered) {
            if (this.mhsaAnimation.mhaPassThroughPhase === 'mha_pass_through_complete'
                && typeof this.mhsaAnimation.skipSelfAttentionAndStartConcat === 'function') {
                this.mhsaAnimation.skipSelfAttentionAndStartConcat();
                this._skipConcatTriggered = true;
            }
        }
        // ────────────────────────────────────────────────────────────
        // Dynamic colour / opacity transition for the FIRST LayerNorm
        // ────────────────────────────────────────────────────────────
        const opaqueOpacity = 1.0;
        const semiTransparentOpacity = 0.6;

        const ln1TargetColor = this._ln1TargetColor;
        ln1TargetColor.copy(COLOR_DARK_GRAY);
        let targetOpacity = opaqueOpacity;

        if (anyVectorInLN1 && highestLN1VecY > -Infinity) {
            if (highestLN1VecY >= bottomY_ln1_abs && highestLN1VecY < midY_ln1_abs) {
                // Entering LN-1
                const t = (highestLN1VecY - bottomY_ln1_abs) / (midY_ln1_abs - bottomY_ln1_abs);
                ln1TargetColor.lerpColors(COLOR_DARK_GRAY, COLOR_LIGHT_YELLOW, t);
                targetOpacity = THREE.MathUtils.lerp(opaqueOpacity, semiTransparentOpacity, t);
            } else if (highestLN1VecY >= midY_ln1_abs && highestLN1VecY < topY_ln1_abs) {
                // Inside LN-1
                ln1TargetColor.copy(COLOR_LIGHT_YELLOW);
                targetOpacity = semiTransparentOpacity;
            } else if (highestLN1VecY >= topY_ln1_abs) {
                // Exiting LN-1
                const tRaw = (highestLN1VecY - topY_ln1_abs) / exitTransitionRange;
                const t = Math.min(1, Math.max(0, tRaw));
                ln1TargetColor.lerpColors(COLOR_LIGHT_YELLOW, COLOR_BRIGHT_YELLOW, t);
                targetOpacity = THREE.MathUtils.lerp(semiTransparentOpacity, opaqueOpacity, t);
            }
        }

        // -------------------------------------------------------------
        // Once a vector has risen sufficiently above LN-1 we want to
        // "bake" the bright colour so the ring doesn't revert to the
        // inactive palette when no vectors are nearby (e.g. while the
        // MHSA animation runs).  We do this by latching a flag the first
        // frame the exit transition completes.
        if (!this._ln1ColorLocked && highestLN1VecY >= topY_ln1_abs + exitTransitionRange) {
            this._ln1ColorLocked = true;
            this._ln1LockedColor.copy(COLOR_BRIGHT_YELLOW);
        }

        if (this._ln1ColorLocked) {
            ln1TargetColor.copy(this._ln1LockedColor);
            targetOpacity = opaqueOpacity;
        }

        if (skipActive && this._ln1MaterialState.initialized) {
            const smoothAlpha = SKIP_COMPONENT_COLOR_LERP_ALPHA;
            if (smoothAlpha > 0 && smoothAlpha < 1) {
                ln1TargetColor.lerpColors(this._ln1MaterialState.color, ln1TargetColor, smoothAlpha);
                targetOpacity = THREE.MathUtils.lerp(this._ln1MaterialState.opacity, targetOpacity, smoothAlpha);
            }
        }

        // Apply to mesh material(s)
        applyLayerNormMaterial(this.ln1 && this.ln1.group, ln1TargetColor, targetOpacity, this._ln1MaterialState);

        // ────────────────────────────────────────────────────────────
        // Dynamic colour / opacity transition for the SECOND LayerNorm
        // ────────────────────────────────────────────────────────────
        // Find the highest Y position of any vector moving through LN2

        const ln2TargetColor = this._ln2TargetColor;
        ln2TargetColor.copy(COLOR_DARK_GRAY);
        let ln2TargetOpacity = opaqueOpacity;

        if (anyVectorInLN2 && highestLN2VecY > -Infinity) {
            if (highestLN2VecY >= bottomY_ln2_abs && highestLN2VecY < midY_ln2_abs) {
                // Entering LN2
                const t = (highestLN2VecY - bottomY_ln2_abs) / (midY_ln2_abs - bottomY_ln2_abs);
                ln2TargetColor.lerpColors(COLOR_DARK_GRAY, COLOR_LIGHT_YELLOW, t);
                ln2TargetOpacity = THREE.MathUtils.lerp(opaqueOpacity, semiTransparentOpacity, t);
            } else if (highestLN2VecY >= midY_ln2_abs && highestLN2VecY < topY_ln2_abs) {
                // Inside LN2
                ln2TargetColor.copy(COLOR_LIGHT_YELLOW);
                ln2TargetOpacity = semiTransparentOpacity;
            } else if (highestLN2VecY >= topY_ln2_abs) {
                // Exiting LN2
                const tRaw = (highestLN2VecY - topY_ln2_abs) / exitTransitionRange;
                const t = Math.min(1, Math.max(0, tRaw));
                ln2TargetColor.lerpColors(COLOR_LIGHT_YELLOW, COLOR_BRIGHT_YELLOW, t);
                ln2TargetOpacity = THREE.MathUtils.lerp(semiTransparentOpacity, opaqueOpacity, t);
            }
        }

        if (!this._ln2ColorLocked && highestLN2VecY >= topY_ln2_abs + exitTransitionRange) {
            this._ln2ColorLocked = true;
            this._ln2LockedColor.copy(COLOR_BRIGHT_YELLOW);
        }

        if (this._ln2ColorLocked) {
            ln2TargetColor.copy(this._ln2LockedColor);
            ln2TargetOpacity = opaqueOpacity;
        }

        if ((anyVectorInLN2 && highestLN2VecY > -Infinity) || this._ln2ColorLocked) {
            if (skipActive && this._ln2MaterialState.initialized) {
                const smoothAlpha = SKIP_COMPONENT_COLOR_LERP_ALPHA;
                if (smoothAlpha > 0 && smoothAlpha < 1) {
                    ln2TargetColor.lerpColors(this._ln2MaterialState.color, ln2TargetColor, smoothAlpha);
                    ln2TargetOpacity = THREE.MathUtils.lerp(this._ln2MaterialState.opacity, ln2TargetOpacity, smoothAlpha);
                }
            }

            // Apply to LN2
            applyLayerNormMaterial(this.ln2 && this.ln2.group, ln2TargetColor, ln2TargetOpacity, this._ln2MaterialState);
        }

        // ────────────────────────────────────────────────────────────
        // LN-2 synchronisation check – only once all lanes have reached
        // the staging height (ln2Phase === 'preRise' and held position)
        // do we allow any of them to continue into the branch.
        // ────────────────────────────────────────────────────────────
        if (!this._ln2Ready && allLn2Ready) {
            this._ln2Ready = true;
            this._emitProgress();
            console.log(`Layer ${this.index}: All lanes ready – starting LN2 simultaneously`);
        }

        // ----------------------------------------------------------------
        // MLP synchronisation: wait until every lane has completed LN-2 and
        // is marked as 'mlpReady' before triggering the up-projection.
        // ----------------------------------------------------------------
        if (!this._mlpStart && allMlpReady) {
            this._mlpStart = true;
            this._emitProgress();
            console.log(`Layer ${this.index}: All lanes ready – starting MLP up-projection simultaneously`);
        }

        const speedMult = GLOBAL_ANIM_SPEED_MULT;

        // ────────────────────────────────────────────────────────────────
        //  NEW: LayerNorm-1 synchronisation barrier – wait until EVERY
        //  lane's original residual-stream vector has reached the branching
        //  height before triggering the horizontal duplicate move.  This
        //  guarantees that all lanes enter LN-1 in perfect lock-step.
        // ────────────────────────────────────────────────────────────────
        if (!this._ln1Start) {
            const readyToStartLn1 = allLn1Ready && posAddDone;
            if (readyToStartLn1) {
                this._ln1Start = true;
                this._emitProgress();
                console.log(`Layer ${this.index}: All lanes ready – starting LN-1 branch simultaneously`);

                // Kick every lane out of the waiting state together.
                this.lanes.forEach(l => {
                    if (l.horizPhase === 'waiting') {
                        l.horizPhase = 'right';
                        this._emitProgress();
                        // Ensure the branch duplicate matches the latest residual data
                        // (e.g., after positional embedding addition).
                        copyVectorAppearance(l.dupVec, l.originalVec);
                        l.dupVec.group.visible = true;
                        // Snap duplicate to the LN-1 branch staging height to avoid
                        // any vertical drift while moving horizontally into the ring.
                        if (typeof l.branchStartY === 'number') {
                            l.dupVec.group.position.y = l.branchStartY;
                        } else {
                            l.dupVec.group.position.y = l.originalVec.group.position.y;
                        }
                    }
                });
            }
        }

        //  NEW: MHSA travel synchronisation barrier – wait until every lane
        //  has its duplicate result vector staged above LN-1 before letting
        //  them begin horizontal travel to the attention heads.
        // ────────────────────────────────────────────────────────────────
        if (!this._mhsaStart && allMhsaReady) {
            this._mhsaStart = true;
            this._emitProgress();
            console.log(`Layer ${this.index}: All lanes ready – starting travel to MHSA heads simultaneously`);
            this.lanes.forEach(l => {
                if (l.horizPhase === 'readyMHSA') {
                    l.horizPhase = 'travelMHSA';
                    l.__mhsaTrailCornerPending = true;
                    this._emitProgress();
                }
            });
        }

        this.lanes.forEach(lane => {
            const { originalVec, dupVec } = lane;
            
            // The Gpt2Layer's direct update logic is now ONLY responsible for
            // handling the initial branching toward the first LayerNorm.
            // ALL subsequent movement, including the continuous rise of the
            // original residual-stream vector, is now managed by the
            // dedicated MHSAAnimation controller. This prevents conflicting
            // updates.

            // branch logic
            switch (lane.horizPhase) {
                case 'waiting':
                    // Hold at the branching height until the global LN-1
                    // barrier is released.
                    if (originalVec.group.position.y >= lane.branchStartY) {
                        // Clamp position so early-arriving lanes don’t drift.
                        originalVec.group.position.y = lane.branchStartY;
                    } else {
                        // Continue rising towards the branching height.
                        originalVec.group.position.y = Math.min(lane.branchStartY, originalVec.group.position.y + ANIM_RISE_SPEED_ORIGINAL * speedMult * dt);
                    }
                    break;
                case 'right':
                    // Mirror LN-2: lock Y at staging height and move X only.
                    if (typeof lane.branchStartY === 'number') {
                        dupVec.group.position.y = lane.branchStartY;
                    }
                    dupVec.group.position.x = Math.min(BRANCH_X, dupVec.group.position.x + ANIM_HORIZ_SPEED * speedMult * dt);
                    if (dupVec.group.position.x >= BRANCH_X - 0.01) {
                        // Ensure alignment with LN-1 centre
                        dupVec.group.position.x = BRANCH_X;
                        // Show the multiplication target inside LN-1 (parity with LN-2 behaviour)
                        if (lane.multTarget && lane.multTarget.group) {
                            lane.multTarget.group.visible = true;
                        }
                        if (lane.addTarget && lane.addTarget.group) {
                            lane.addTarget.group.visible = true;
                        }
                        lane.horizPhase = 'insideLN';
                        this._emitProgress();
                    }
                    break;
                case 'insideLN':
                    // Start the normalisation animation once the vector has
                    // climbed the configured fraction of the LayerNorm's
                    // height from its bottom edge.
                    const normStartY =
                        bottomY_ln1_abs + LN_PARAMS.height * LN_NORM_START_FRACTION_FROM_BOTTOM;
                    if (!lane.ln1ParamColored && dupVec.group.position.y >= normStartY) {
                        // Switch LN1 param vectors from placeholder gray to active (blue-ish) colors.
                        if (lane.multTarget) this._applyLayerNormParamVector(lane.multTarget, 'ln1', 'scale', null);
                        if (lane.addTarget) this._applyLayerNormParamVector(lane.addTarget, 'ln1', 'shift', null);
                        lane.ln1ParamColored = true;
                    }
                    const riseStep = ANIM_RISE_SPEED_INSIDE_LN * speedMult * dt;
                    const ln1RiseTargetY = (() => {
                        const multTargetGroup = lane.multTarget && lane.multTarget.group;
                        if (multTargetGroup && Number.isFinite(multTargetGroup.position.y)) {
                            return Math.max(lane.ln1MidY, multTargetGroup.position.y);
                        }
                        return lane.ln1MidY;
                    })();
                    const startLn1Norm = () => {
                        const ln1NormData = this._getLn1Data(lane, 'norm');
                        const normInput = ln1NormData ? ln1NormData.slice() : dupVec.rawData.slice();
                        lane.pendingNormData = ln1NormData || null;
                        lane.pendingNormLabel = lane.tokenLabel ? `LN1 Normed - ${lane.tokenLabel}` : 'LN1 Normed';
                        lane.pendingNormMeta = this._getLaneMeta(lane, 'ln1.norm');
                        lane.normApplied = false;
                        if (!skipActive && lane.normAnim) {
                            lane.normAnim.start(normInput, { deferDataUpdate: true });
                        }
                        lane.normStarted = true;
                        if (skipActive) {
                            if (lane.pendingNormData) {
                                applyVectorData(
                                    dupVec,
                                    lane.pendingNormData,
                                    lane.pendingNormLabel,
                                    lane.pendingNormMeta
                                );
                            }
                            lane.normApplied = true;
                        }
                    };
                    if (!lane.normStarted) {
                        if (dupVec.group.position.y >= normStartY) {
                            dupVec.group.position.y = normStartY;
                            startLn1Norm();
                        } else {
                            dupVec.group.position.y = Math.min(normStartY, dupVec.group.position.y + riseStep);
                            if (dupVec.group.position.y >= normStartY) {
                                startLn1Norm();
                            }
                        }
                    }
                    if (lane.normStarted && lane.normAnim) {
                        if (!skipActive) {
                            lane.normAnim.update(dt);
                        } else if (lane.normAnim.isAnimating) {
                            lane.normAnim.isAnimating = false;
                        }
                    }
                    const normAnimating = !skipActive && lane.normStarted && lane.normAnim && lane.normAnim.isAnimating;
                    if (lane.normStarted && !normAnimating && !lane.normApplied) {
                        if (lane.pendingNormData) {
                            applyVectorData(
                                dupVec,
                                lane.pendingNormData,
                                lane.pendingNormLabel,
                                lane.pendingNormMeta
                            );
                        }
                        lane.normApplied = true;
                    }
                    if (lane.normStarted && !normAnimating) {
                        dupVec.group.position.y = Math.min(
                            ln1RiseTargetY,
                            dupVec.group.position.y + riseStep
                        );
                    }
                    if (
                        !lane.multStarted &&
                        lane.normStarted &&
                        !lane.normAnim.isAnimating &&
                        dupVec.group.position.y >= ln1RiseTargetY - 0.01
                    ) {
                        lane.multStarted = true;
                        const scaleParam = lane.multTarget;
                        const shiftParam = lane.addTarget;
                        if (shiftParam && shiftParam.group) {
                            shiftParam.group.visible = true;
                        }
                        if (scaleParam && scaleParam.group) {
                            scaleParam.group.position.y = ln1RiseTargetY;
                            scaleParam.group.visible = true;
                        }

                        const scaleParamData = this._getLayerNormParamData('ln1', 'scale');
                        const sourceRaw = Array.isArray(dupVec.rawData) ? dupVec.rawData : [];
                        const multData = sourceRaw.slice();
                        if (scaleParamData && Array.isArray(scaleParamData)) {
                            const len = Math.min(multData.length, scaleParamData.length);
                            for (let i = 0; i < len; i++) {
                                multData[i] = (sourceRaw[i] || 0) * (scaleParamData[i] || 0);
                            }
                        }

                        const multSeed = multData.length
                            ? multData
                            : this.random.nextVector(this._getBaseVectorLength());
                        const multResult = this._createPrismVector(
                            multSeed,
                            (scaleParam && scaleParam.group)
                                ? scaleParam.group.position.clone()
                                : dupVec.group.position.clone(),
                            30,
                            dupVec.instanceCount
                        );
                        this.raycastRoot.add(multResult.group);
                        dupVec.group.visible = false;
                        const ln1ScaledData = this._getLn1Data(lane, 'scale');
                        if (ln1ScaledData) {
                            applyVectorData(
                                multResult,
                                ln1ScaledData,
                                lane.tokenLabel ? `LN1 Scaled - ${lane.tokenLabel}` : 'LN1 Scaled',
                                this._getLaneMeta(lane, 'ln1.scale')
                            );
                        }
                        if (scaleParam && scaleParam.group) {
                            scaleParam.group.visible = false;
                        }

                        const reusedTrail = dupVec && dupVec.userData && dupVec.userData.trail;
                        const reusedTrailIsWorld = Boolean(dupVec && dupVec.userData && dupVec.userData.trailWorld);
                        if (this._skipToEndActive && reusedTrail && !reusedTrailIsWorld && dupVec && dupVec.group) {
                            const zPos = Number.isFinite(lane.zPos) ? lane.zPos : dupVec.group.position.z;
                            const clampedY = Math.min(topY_ln1_abs, Math.max(bottomY_ln1_abs, dupVec.group.position.y));
                            TMP_LN_TRAIL_POS.set(BRANCH_X, clampedY, zPos);
                            if (typeof reusedTrail.update === 'function') {
                                reusedTrail.update(TMP_LN_TRAIL_POS);
                            }
                        }
                        let trailForResult;
                        if (reusedTrail) {
                            trailForResult = reusedTrail;
                            if (reusedTrailIsWorld) {
                                multResult.group.getWorldPosition(TMP_WORLD_POS);
                                if (typeof trailForResult.snapLastPointTo === 'function') {
                                    trailForResult.snapLastPointTo(TMP_WORLD_POS);
                                } else if (typeof trailForResult.update === 'function') {
                                    trailForResult.update(TMP_WORLD_POS);
                                }
                            } else if (typeof trailForResult.snapLastPointTo === 'function') {
                                trailForResult.snapLastPointTo(multResult.group.position);
                            } else if (typeof trailForResult.update === 'function') {
                                trailForResult.update(multResult.group.position);
                            }
                            if (dupVec.userData) {
                                delete dupVec.userData.trail;
                                delete dupVec.userData.trailWorld;
                            }
                        } else {
                            trailForResult = new StraightLineTrail(this.root, 0xffffff, 1, undefined, undefined, TRAIL_MIN_SEGMENT_DISTANCE);
                            trailForResult.start(multResult.group.position);
                        }
                        multResult.userData = multResult.userData || {};
                        multResult.userData.trail = trailForResult;
                        if (reusedTrailIsWorld) {
                            multResult.userData.trailWorld = true;
                        } else if (multResult.userData.trailWorld) {
                            delete multResult.userData.trailWorld;
                        }

                        const shiftParamData = this._getLayerNormParamData('ln1', 'shift');
                        const shiftSeed = (shiftParamData && shiftParamData.length)
                            ? shiftParamData
                            : (sourceRaw.length ? sourceRaw.slice() : this.random.nextVector(this._getBaseVectorLength()));
                        const addResult = this._createPrismVector(
                            shiftSeed,
                            (shiftParam && shiftParam.group)
                                ? shiftParam.group.position.clone()
                                : multResult.group.position.clone(),
                            30,
                            multResult.instanceCount
                        );
                        this.raycastRoot.add(addResult.group);
                        // Keep shift/addition parameter in active (blue-ish) colors during the add animation.
                        this._applyLayerNormParamVector(addResult, 'ln1', 'shift', null);
                        if (shiftParam && shiftParam.group) {
                            shiftParam.group.visible = false;
                        }

                        const ln1ShiftedData = this._getLn1Data(lane, 'shift');
                        lane.resultVec = addResult;
                        lane.ln1AddStarted = true;
                        startPrismAdditionAnimation(multResult, addResult, null, () => {
                            lane.ln1AddComplete = true;
                            if (ln1ShiftedData) {
                                applyVectorData(
                                    addResult,
                                    ln1ShiftedData,
                                    lane.tokenLabel ? `LN1 Shifted - ${lane.tokenLabel}` : 'LN1 Shifted',
                                    this._getLaneMeta(lane, 'ln1.shift')
                                );
                            }
                            const additionTrail = multResult.userData && multResult.userData.trail;
                            if (additionTrail) {
                                addResult.userData = addResult.userData || {};
                                const additionTrailIsWorld = Boolean(multResult.userData && multResult.userData.trailWorld);
                                const prevTrail = addResult.userData.trail;
                                if (prevTrail && prevTrail !== additionTrail) {
                                    try {
                                        if (typeof prevTrail.dispose === 'function') {
                                            prevTrail.dispose();
                                        } else if (prevTrail._line && prevTrail._line.parent) {
                                            prevTrail._line.parent.remove(prevTrail._line);
                                        }
                                    } catch (_) { /* non-fatal cleanup */ }
                                }
                                addResult.userData.trail = additionTrail;
                                addResult.userData.trailWorld = additionTrailIsWorld;
                                if (additionTrailIsWorld) {
                                    addResult.group.getWorldPosition(TMP_WORLD_POS);
                                    if (typeof additionTrail.snapLastPointTo === 'function') {
                                        additionTrail.snapLastPointTo(TMP_WORLD_POS);
                                    } else {
                                        additionTrail.update(TMP_WORLD_POS);
                                    }
                                } else {
                                    if (typeof additionTrail.snapLastPointTo === 'function') {
                                        additionTrail.snapLastPointTo(addResult.group.position);
                                    } else {
                                        additionTrail.update(addResult.group.position);
                                    }
                                }
                                delete multResult.userData.trail;
                                delete multResult.userData.trailWorld;
                            }
                            if (multResult.group && multResult.group.parent) {
                                multResult.group.parent.remove(multResult.group);
                            }
                            addResult.userData = addResult.userData || {};
                            if (!addResult.userData.trail) {
                                const fallbackTrail = new StraightLineTrail(this.root, 0xffffff, 1, undefined, undefined, TRAIL_MIN_SEGMENT_DISTANCE);
                                fallbackTrail.start(addResult.group.position);
                                addResult.userData.trail = fallbackTrail;
                                addResult.userData.trailWorld = false;
                            }
                            lane.horizPhase = 'riseAboveLN';
                            this._emitProgress();
                        }, { finalData: ln1ShiftedData, progressTarget: lane, progressKey: 'ln1ShiftProgress' });
                    }
                    break;
                case 'riseAboveLN':
                    // Rise to just above LN1 before starting horizontal travel
                    const rv = lane.resultVec;
                    if (rv) {
                        const targetY = this.ln1TopY + 5; // Same as meetY in original
                        if (rv.group.position.y < targetY) {
                            rv.group.position.y = Math.min(targetY, rv.group.position.y + ANIM_RISE_SPEED_INSIDE_LN * speedMult * dt);
                        } else {
                            // Now that we're above LN1, mark lane ready for MHSA travel.
                            lane.travellingVec = rv;
                            lane.headIndex = 0;
                            lane.horizPhase = 'readyMHSA'; // wait for global barrier
                            lane.__mhsaTrailCornerPending = true;
                            this._emitProgress();
                        }
                    }
                    break;
                case 'readyMHSA':
                    // Hold at staging height until global _mhsaStart flag triggers.
                    // Ensure vector stays exactly at meetY.
                    if (lane.travellingVec) {
                        lane.travellingVec.group.position.y = this.ln1TopY + 5;
                    }
                    break;
                case 'travelMHSA':
                    // MHSAAnimation will handle the horizontal movement
                    break;
                case 'postMHSAAddition':
                    // After MHSA addition completes, start LN2 phase
                    // This state is set by MHSAAnimation._startAdditionAnimation
                    if (lane.ln2Phase === 'preRise' && lane.postAdditionVec) {
                        lane.horizPhase = 'waitingForLN2'; // Move to next state
                        this._emitProgress();
                    }
                    break;
                case 'waitingForLN2':
                    // Just a placeholder state while LN2 animation runs
                    break;
                default:
                    break;
            }

            // ─────────────────────────────────────────────────────────────
            // LayerNorm2 / MLP Pipeline
            // ─────────────────────────────────────────────────────────────
            switch (lane.ln2Phase) {
                case 'preRise': {
                    // Rise the post-addition vector before branching to LN2
                    const v = lane.postAdditionVec;
                    if (!v) break;
                    
                    // Stage the vector at the same relative position used for LayerNorm-1
                    // (5 units above the bottom of the norm ring) so that the ensuing
                    // normalisation begins at a consistent height across both LayerNorms.
                    const targetY = bottomY_ln2_abs + 5; // align with LN1 offset
                    if (v.group.position.y < targetY) {
                        v.group.position.y = Math.min(targetY, v.group.position.y + ANIM_RISE_SPEED_POST_SPLIT_LN2 * speedMult * dt);
                    } else {
                        if (!this._ln2Ready) {
                            // Wait here until every lane reaches the staging height.
                            break;
                        }

                        // ────────────────────────────────────────────────
                        //  Reached staging height – begin LN-2 branch
                        // ────────────────────────────────────────────────

                        // Allow residual stream to keep rising while the
                        // duplicate goes through LN-2/MLP.
                        if (this.mhsaAnimation && typeof this.mhsaAnimation.finalOriginalY === 'number') {
                            const newTarget = this.mlpUp.group.position.y + MLP_MATRIX_PARAMS_UP.height / 2 - ORIGINAL_TO_PROCESSED_GAP;
                            if (newTarget > this.mhsaAnimation.finalOriginalY) {
                                this.mhsaAnimation.finalOriginalY = newTarget;
                            }
                            this.mhsaAnimation.postSplitRiseSpeed = ANIM_RISE_SPEED_POST_SPLIT_LN2;
                        }

                        // Spawn duplicate vector that will travel into LN-2
                        const mv = this._createPrismVector(
                            v.rawData.slice(),
                            v.group.position.clone(),
                            30,
                            v.instanceCount
                        );
                        copyVectorAppearance(mv, v);
                        this.raycastRoot.add(mv.group);
                        // ---- Trail for LN2 moving vector ----
                        const mvTrail = new StraightLineTrail(this.root, 0xffffff, 1, undefined, undefined, LN_INTERNAL_TRAIL_MIN_SEGMENT);
                        mvTrail.start(mv.group.position);
                        mv.userData = mv.userData || {};
                        mv.userData.trail = mvTrail;
                        lane.movingVecLN2 = mv;
                        lane.normAnimationLN2 = new PrismLayerNormAnimation(mv);

                        lane.ln2Phase = 'right';
                        this._emitProgress();
                    }
                    break;
                }
                
                case 'right': {
                    // Move horizontally to LN2
                    const mv = lane.movingVecLN2;
                    if (!mv) break;
                    
                    mv.group.visible = true;
                    const dx = ANIM_HORIZ_SPEED * speedMult * dt;
                    mv.group.position.x = Math.min(BRANCH_X, mv.group.position.x + dx);
                    
                    if (mv.group.position.x >= BRANCH_X - 0.01) {
                        mv.group.position.x = BRANCH_X;
                        if (lane.multTargetLN2 && lane.multTargetLN2.group) {
                            lane.multTargetLN2.group.visible = true;
                        }
                        if (lane.addTargetLN2 && lane.addTargetLN2.group) {
                            lane.addTargetLN2.group.visible = true;
                        }
                        lane.ln2Phase = 'insideLN';
                        this._emitProgress();
                    }

                    break;
                }
                
                case 'insideLN': {
                    if (!lane.ln2ParamColored) {
                        // Switch LN2 param vectors from placeholder gray to active (blue-ish) colors.
                        if (lane.multTargetLN2) this._applyLayerNormParamVector(lane.multTargetLN2, 'ln2', 'scale', null);
                        if (lane.addTargetLN2) this._applyLayerNormParamVector(lane.addTargetLN2, 'ln2', 'shift', null);
                        lane.ln2ParamColored = true;
                    }
                    // Inside LayerNorm2 - normalize and multiply
                    const mv = lane.movingVecLN2;
                    if (!mv) break;

                    const ln2RiseTargetY = (() => {
                        const multTargetGroup = lane.multTargetLN2 && lane.multTargetLN2.group;
                        if (multTargetGroup && Number.isFinite(multTargetGroup.position.y)) {
                            return Math.max(midY_ln2_abs, multTargetGroup.position.y);
                        }
                        return midY_ln2_abs;
                    })();

                    const startLn2Rise = (vec) => {
                        if (!vec) return;
                        const destY = this.mlpUp.group.position.y - MLP_MATRIX_PARAMS_UP.height / 2 - 10;
                        const dist = destY - vec.group.position.y;
                        const durationMs = (dist / (ANIM_RISE_SPEED_INSIDE_LN * GLOBAL_ANIM_SPEED_MULT)) * 1000;

                        if (typeof TWEEN !== 'undefined') {
                            new TWEEN.Tween(vec.group.position)
                                .to({ y: destY }, durationMs)
                                .easing(TWEEN.Easing.Linear.None)
                                .onUpdate(() => {})
                                .onComplete(() => {
                                    lane.ln2Phase = 'mlpReady';
                                    this._emitProgress();
                                })
                                .start();
                        } else {
                            vec.group.position.y = destY;
                            lane.ln2Phase = 'mlpReady';
                            this._emitProgress();
                        }
                    };

                    // Use the same fraction as LayerNorm-1 so the
                    // normalisation animation begins at an identical
                    // relative height inside the ring.
                    const normStartY2 =
                        bottomY_ln2_abs + LN_PARAMS.height * LN_NORM_START_FRACTION_FROM_BOTTOM;
                    const ln2HoleHeight = Math.min(
                        LN_PARAMS.wallThickness * 1.5 * LN_PARAMS.holeWidthFactor,
                        LN_PARAMS.height / 2
                    );
                    const ln2SolidEntryY = bottomY_ln2_abs + ln2HoleHeight;
                    const minInsideY = Math.max(bottomY_ln2_abs + 6, ln2SolidEntryY);
                    const effectiveNormStartY2 = Math.max(normStartY2, minInsideY);
                    if (!lane.ln2ParamColored && mv.group.position.y >= minInsideY) {
                        // Switch LN2 param vectors from placeholder gray to active (blue-ish) colors.
                        if (lane.multTargetLN2) this._applyLayerNormParamVector(lane.multTargetLN2, 'ln2', 'scale', null);
                        if (lane.addTargetLN2) this._applyLayerNormParamVector(lane.addTargetLN2, 'ln2', 'shift', null);
                        lane.ln2ParamColored = true;
                    }
                    const riseStep = ANIM_RISE_SPEED_INSIDE_LN * speedMult * dt;
                    const startLn2Norm = () => {
                        const ln2NormData = this._getLn2Data(lane, 'norm');
                        const normInput = ln2NormData ? ln2NormData.slice() : mv.rawData.slice();
                        lane.pendingNormDataLN2 = ln2NormData || null;
                        lane.pendingNormLabelLN2 = lane.tokenLabel ? `LN2 Normed - ${lane.tokenLabel}` : 'LN2 Normed';
                        lane.pendingNormMetaLN2 = this._getLaneMeta(lane, 'ln2.norm');
                        lane.normAppliedLN2 = false;
                        if (!skipActive && lane.normAnimationLN2) {
                            lane.normAnimationLN2.start(normInput, { deferDataUpdate: true });
                        }
                        lane.normStartedLN2 = true;
                        if (skipActive) {
                            if (lane.pendingNormDataLN2) {
                                applyVectorData(
                                    mv,
                                    lane.pendingNormDataLN2,
                                    lane.pendingNormLabelLN2,
                                    lane.pendingNormMetaLN2
                                );
                            }
                            lane.normAppliedLN2 = true;
                        }
                    };
                    if (!lane.normStartedLN2) {
                        if (mv.group.position.y < minInsideY) {
                            mv.group.position.y = Math.min(minInsideY, mv.group.position.y + riseStep);
                        } else if (mv.group.position.y >= effectiveNormStartY2) {
                            mv.group.position.y = effectiveNormStartY2;
                            startLn2Norm();
                        } else {
                            mv.group.position.y = Math.min(effectiveNormStartY2, mv.group.position.y + riseStep);
                            if (mv.group.position.y >= effectiveNormStartY2) {
                                startLn2Norm();
                            }
                        }
                    }
                    if (lane.normStartedLN2 && lane.normAnimationLN2) {
                        if (!skipActive) {
                            lane.normAnimationLN2.update(dt);
                        } else if (lane.normAnimationLN2.isAnimating) {
                            lane.normAnimationLN2.isAnimating = false;
                        }
                    }
                    const normAnimating2 = !skipActive && lane.normStartedLN2 && lane.normAnimationLN2 && lane.normAnimationLN2.isAnimating;
                    if (lane.normStartedLN2 && !normAnimating2 && !lane.normAppliedLN2) {
                        if (lane.pendingNormDataLN2) {
                            applyVectorData(
                                mv,
                                lane.pendingNormDataLN2,
                                lane.pendingNormLabelLN2,
                                lane.pendingNormMetaLN2
                            );
                        }
                        lane.normAppliedLN2 = true;
                    }
                    if (!lane.multDoneLN2 && lane.normStartedLN2 && !normAnimating2) {
                        mv.group.position.y = Math.min(
                            ln2RiseTargetY,
                            mv.group.position.y + riseStep
                        );
                    }
                    
                    // Trigger multiplication at center of LN2
                    if (
                        !lane.multDoneLN2 &&
                        lane.normStartedLN2 &&
                        !lane.normAnimationLN2.isAnimating &&
                        mv.group.position.y >= ln2RiseTargetY - 0.01
                    ) {
                        lane.multDoneLN2 = true;
                        const scaleParam = lane.multTargetLN2;
                        const shiftParam = lane.addTargetLN2;
                        if (shiftParam && shiftParam.group) {
                            shiftParam.group.visible = true;
                        }
                        if (scaleParam && scaleParam.group) {
                            scaleParam.group.position.y = ln2RiseTargetY;
                            scaleParam.group.visible = true;
                        }

                        const scaleParamData = this._getLayerNormParamData('ln2', 'scale');
                        const sourceRaw = Array.isArray(mv.rawData) ? mv.rawData : [];
                        const multData = sourceRaw.slice();
                        if (scaleParamData && Array.isArray(scaleParamData)) {
                            const len = Math.min(multData.length, scaleParamData.length);
                            for (let i = 0; i < len; i++) {
                                multData[i] = (sourceRaw[i] || 0) * (scaleParamData[i] || 0);
                            }
                        }

                        // Reuse the moving LN2 vector as the scaled output to
                        // avoid a one-frame overlap/glitch from swapping meshes.
                        const resVec = mv;
                        if (scaleParam && scaleParam.group) {
                            scaleParam.group.visible = false;
                        }

                        const ln2ScaledData = this._getLn2Data(lane, 'scale');
                        const scaledFallback = (ln2ScaledData && ln2ScaledData.length)
                            ? ln2ScaledData
                            : multData;
                        if (scaledFallback && scaledFallback.length) {
                            applyVectorData(
                                resVec,
                                scaledFallback,
                                lane.tokenLabel ? `LN2 Scaled - ${lane.tokenLabel}` : 'LN2 Scaled',
                                this._getLaneMeta(lane, 'ln2.scale')
                            );
                        }

                        const reusedTrailLn2 = resVec && resVec.userData && resVec.userData.trail;
                        const reusedTrailLn2IsWorld = Boolean(resVec && resVec.userData && resVec.userData.trailWorld);
                        if (this._skipToEndActive && reusedTrailLn2 && !reusedTrailLn2IsWorld && resVec && resVec.group) {
                            const zPos = Number.isFinite(lane.zPos) ? lane.zPos : resVec.group.position.z;
                            const clampedY = Math.min(topY_ln2_abs, Math.max(bottomY_ln2_abs, resVec.group.position.y));
                            TMP_LN_TRAIL_POS.set(BRANCH_X, clampedY, zPos);
                            if (typeof reusedTrailLn2.update === 'function') {
                                reusedTrailLn2.update(TMP_LN_TRAIL_POS);
                            }
                        }
                        if (!reusedTrailLn2 && resVec && resVec.group) {
                            const fallbackTrailLn2 = new StraightLineTrail(this.root, 0xffffff, 1, undefined, undefined, TRAIL_MIN_SEGMENT_DISTANCE);
                            fallbackTrailLn2.start(resVec.group.position);
                            resVec.userData = resVec.userData || {};
                            resVec.userData.trail = fallbackTrailLn2;
                            resVec.userData.trailWorld = false;
                        }

                        const shiftParamData = this._getLayerNormParamData('ln2', 'shift');
                        const shiftSeed = (shiftParamData && shiftParamData.length)
                            ? shiftParamData
                            : (sourceRaw.length ? sourceRaw.slice() : this.random.nextVector(this._getBaseVectorLength()));
                        const addResult = this._createPrismVector(
                            shiftSeed,
                            (shiftParam && shiftParam.group)
                                ? shiftParam.group.position.clone()
                                : resVec.group.position.clone(),
                            30,
                            resVec.instanceCount
                        );
                        this.raycastRoot.add(addResult.group);
                        // Keep shift/addition parameter in active (blue-ish) colors during the add animation.
                        this._applyLayerNormParamVector(addResult, 'ln2', 'shift', null);
                        if (shiftParam && shiftParam.group) {
                            shiftParam.group.visible = false;
                        }

                        const ln2ShiftedData = this._getLn2Data(lane, 'shift');
                        lane.resultVecLN2 = addResult;
                        lane.ln2AddStarted = true;
                        startPrismAdditionAnimation(resVec, addResult, null, () => {
                            lane.ln2AddComplete = true;
                            if (ln2ShiftedData) {
                                applyVectorData(
                                    addResult,
                                    ln2ShiftedData,
                                    lane.tokenLabel ? `LN2 Shifted - ${lane.tokenLabel}` : 'LN2 Shifted',
                                    this._getLaneMeta(lane, 'ln2.shift')
                                );
                            }
                            const ln2Trail = resVec.userData && resVec.userData.trail;
                            if (ln2Trail) {
                                addResult.userData = addResult.userData || {};
                                const ln2TrailIsWorld = Boolean(resVec.userData && resVec.userData.trailWorld);
                                const prevTrailLN2 = addResult.userData.trail;
                                if (prevTrailLN2 && prevTrailLN2 !== ln2Trail) {
                                    try {
                                        if (typeof prevTrailLN2.dispose === 'function') {
                                            prevTrailLN2.dispose();
                                        } else if (prevTrailLN2._line && prevTrailLN2._line.parent) {
                                            prevTrailLN2._line.parent.remove(prevTrailLN2._line);
                                        }
                                    } catch (_) { /* cleanup best-effort */ }
                                }
                                addResult.userData.trail = ln2Trail;
                                addResult.userData.trailWorld = ln2TrailIsWorld;
                                if (ln2TrailIsWorld) {
                                    addResult.group.getWorldPosition(TMP_WORLD_POS);
                                    if (typeof ln2Trail.snapLastPointTo === 'function') {
                                        ln2Trail.snapLastPointTo(TMP_WORLD_POS);
                                    } else {
                                        ln2Trail.update(TMP_WORLD_POS);
                                    }
                                } else {
                                    if (typeof ln2Trail.snapLastPointTo === 'function') {
                                        ln2Trail.snapLastPointTo(addResult.group.position);
                                    } else {
                                        ln2Trail.update(addResult.group.position);
                                    }
                                }
                                delete resVec.userData.trail;
                                delete resVec.userData.trailWorld;
                            }
                            if (resVec.group && resVec.group.parent) {
                                resVec.group.parent.remove(resVec.group);
                            }
                            lane.movingVecLN2 = null;
                            lane.normAnimationLN2 = null;
                            addResult.userData = addResult.userData || {};
                            if (!addResult.userData.trail) {
                                const fallbackTrailLn2 = new StraightLineTrail(this.root, 0xffffff, 1, undefined, undefined, TRAIL_MIN_SEGMENT_DISTANCE);
                                fallbackTrailLn2.start(addResult.group.position);
                                addResult.userData.trail = fallbackTrailLn2;
                                addResult.userData.trailWorld = false;
                            }
                            startLn2Rise(addResult);
                        }, { finalData: ln2ShiftedData, progressTarget: lane, progressKey: 'ln2ShiftProgress', suppressResidualTrailUpdates: true });
                    }


                    break;
                }
                
                case 'mlpReady':
                    // Ready for MLP animation – begin only once every lane
                    // has reached this state to maintain synchronisation.
                    if (!this._mlpStart) {
                        break; // hold until global start flag set
                    }

                    if (!lane.mlpUpStarted && lane.resultVecLN2) {
                        lane.mlpUpStarted = true;
                        this._emitProgress();
                        this._animateMlpUpProjection(lane);
                    }
                    break;
                    
                case 'done':
                default:
                    break;
            }
        });

        // Ensure LayerNorm parameter banks reflect any visibility/position toggles.
        if (this._lnParamBanks) {
            const banks = this._lnParamBanks;
            if (banks.ln1Scale && typeof banks.ln1Scale.syncAll === 'function') banks.ln1Scale.syncAll();
            if (banks.ln1Shift && typeof banks.ln1Shift.syncAll === 'function') banks.ln1Shift.syncAll();
            if (banks.ln2Scale && typeof banks.ln2Scale.syncAll === 'function') banks.ln2Scale.syncAll();
            if (banks.ln2Shift && typeof banks.ln2Shift.syncAll === 'function') banks.ln2Shift.syncAll();
        }

        // Update the MHSA controller so vectors travel to attention heads
        if (this.mhsaAnimation) {
            this.mhsaAnimation.update(dt, performance.now(), this.lanes);
        }

        // ----------------------------------------------------------
        // Notify LayerPipeline once **all** lanes have finished AND all additions complete
        // ----------------------------------------------------------
        if (this.isActive && !this._completed && this.lanes.length && 
            this.lanes.every(l => l.ln2Phase === 'done') && this._pendingAdditions === 0) {
            this._completed = true;
            this._makeLayerOpaque(); // disable expensive transparency; dynamic vectors culled later by pipeline
            if (this.onFinished) this.onFinished();
        }

        // ------------------------------------------------------------------
        // Final hard clamp: if the pipeline has provided a top-embedding stop
        // height (in this layer's local space), ensure residual vectors cannot
        // rise above it under any circumstance (e.g. stray tweens).
        // ------------------------------------------------------------------
        // Do not force residual vectors downward here; MHSAAnimation already clamps upward motion.
        if (skipActive) this._applySkipVectorVisibility();
    }

    /**
     * Animate vector through MLP up-projection (768 → 3072 dimensions)
     */
    _animateMlpUpProjection(lane) {
        const vec = lane.resultVecLN2;
        if (!vec || typeof TWEEN === 'undefined') return;

        const bottomY = this.mlpUp.group.position.y - MLP_MATRIX_PARAMS_UP.height / 2;
        const topY = this.mlpUp.group.position.y + MLP_MATRIX_PARAMS_UP.height / 2;
        const distance = topY - vec.group.position.y;
        const duration = (distance / (ANIM_RISE_SPEED_INSIDE_LN * GLOBAL_ANIM_SPEED_MULT)) * 1000;
        const colorDuration = this._skipToEndActive
            ? Math.max(duration, SKIP_MLP_COLOR_MIN_MS)
            : duration;
        
        const matrixStartColor = new THREE.Color(INACTIVE_COMPONENT_COLOR);
        const matrixEndColor = new THREE.Color(0xc07a12); // orange
        const startIntensity = 0.1;
        const peakIntensity = 0.24;
        const finalIntensity = 0.30;

        // Animate matrix colour and emissive intensity for a glow effect
        const state = { t: 0, emissive: startIntensity };
        new TWEEN.Tween(state)
            .to({ t: 1, emissive: peakIntensity }, colorDuration * 0.6)
            .easing(TWEEN.Easing.Quadratic.InOut)
            .onUpdate(() => {
                const col = matrixStartColor.clone().lerp(matrixEndColor, state.t);
                this.mlpUp.setColor(col);
                this.mlpUp.setEmissive(col, state.emissive);
            })
            .onComplete(() => {
                new TWEEN.Tween(state)
                    .to({ emissive: finalIntensity }, colorDuration * 0.4)
                    .easing(TWEEN.Easing.Quadratic.InOut)
                    .onUpdate(() => {
                        this.mlpUp.setEmissive(matrixEndColor, state.emissive);
                    })
                    .start();
            })
            .start();
            
        // Move vector through matrix
        new TWEEN.Tween(vec.group.position)
            .to({ y: topY }, duration)
            .easing(TWEEN.Easing.Linear.None)
            .onUpdate(() => {
            })
            .onStart(() => {
                // Shrink to fit in narrowing matrix
                vec.group.scale.setScalar(0.6);
            })
            .onComplete(() => {
                // Restore scale
                vec.group.scale.setScalar(0.6);
                this.mlpUp.setColor(matrixEndColor);
                this.mlpUp.setEmissive(matrixEndColor, finalIntensity);

                const mlpUpData = this._getMlpUpData(lane);
                // Expand to 4x width (3072 dimensions)
                this._expandTo4x(lane, vec, mlpUpData);
            })
            .start();
    }

    /**
     * Expand vector to 4x width representing 3072 dimensions
     */
    _expandTo4x(lane, vec, mlpUpData = null) {
        const segments = 4;
        const segmentLength = Number.isFinite(vec?.instanceCount) ? vec.instanceCount : this._getBaseVectorLength();
        const segWidth = vec.getBaseWidthConstant() * vec.getWidthScale() * segmentLength;
        const expandedGroup = new THREE.Group();
        const segmentVecs = [];
        const segmentBatch = new BatchedPrismVectorSet({
            vectorCount: segments,
            prismCount: segmentLength,
            parentGroup: expandedGroup,
            label: 'MLP Expanded Segments',
        });
        const paddedData = Array.isArray(mlpUpData) ? mlpUpData.slice() : null;
        if (paddedData && paddedData.length < segments * segmentLength) {
            while (paddedData.length < segments * segmentLength) paddedData.push(0);
        }

        // Create 4 segments side by side
        for (let s = 0; s < segments; s++) {
            const segVec = segmentBatch.getVectorRef(s);
            segVec.group.visible = true;
            segVec.rawData = Array.isArray(vec.rawData) ? vec.rawData.slice() : [];

            if (paddedData) {
                const start = s * segmentLength;
                const slice = paddedData.slice(start, start + segmentLength);
                applyVectorData(
                    segVec,
                    slice,
                    lane && lane.tokenLabel
                        ? `MLP Up Projection - ${lane.tokenLabel}`
                        : 'MLP Up Projection',
                    this._getLaneMeta(lane, 'mlp.up', { segmentIndex: s })
                );
            } else if (typeof segVec.copyColorsFrom === 'function') {
                segVec.copyColorsFrom(vec);
            }
            
            // Position segments horizontally
            const localX = (s - (segments - 1) / 2) * segWidth;
            segVec.group.position.set(localX, 0, 0);
            segmentVecs.push(segVec);
        }
        segmentBatch.syncAll();

        expandedGroup.userData.fullWidth = segWidth * segments;
        expandedGroup.position.copy(vec.group.position);
        this.raycastRoot.add(expandedGroup);
        vec.group.visible = false;

        // ---- Trail for expanded 4x vector group ----
        const expTrail = new StraightLineTrail(this.root, 0xffffff, 1, undefined, undefined, TRAIL_MIN_SEGMENT_DISTANCE);
        expTrail.start(expandedGroup.position);
        lane.expandedVecTrail = expTrail;
        
        lane.expandedVecGroup = expandedGroup;
        lane.expandedVecSegments = segmentVecs;
        
        // Rise before down-projection
        const extraRise = 30;
        const pauseMs = 0;

        new TWEEN.Tween(expandedGroup.position)
            .to({ y: expandedGroup.position.y + extraRise }, 500)
            .easing(TWEEN.Easing.Quadratic.InOut)
            .onUpdate(() => {
            })
            .onComplete(() => {
                const startDown = () => this._animateMlpDownProjection(lane);
                const activationData = this._getMlpActivationData(lane);
                const applyActivation = () => {
                    if (activationData && lane.expandedVecSegments) {
                        const activationSlices = activationData.slice();
                        while (activationSlices.length < segments * segmentLength) activationSlices.push(0);
                        lane.expandedVecSegments.forEach((segVec, idx) => {
                            const start = idx * segmentLength;
                            const slice = activationSlices.slice(start, start + segmentLength);
                            applyVectorData(
                                segVec,
                                slice,
                                lane && lane.tokenLabel
                                    ? `MLP Activation - ${lane.tokenLabel}`
                                    : 'MLP Activation',
                                this._getLaneMeta(lane, 'mlp.activation', { segmentIndex: idx })
                            );
                        });
                    }
                };
                const runGeluTransition = () => {
                    this._animateMlpActivationGelu(lane, applyActivation, startDown);
                };

                if (typeof TWEEN !== 'undefined' && pauseMs > 0) {
                    new TWEEN.Tween({ t: 0 })
                        .to({ t: 1 }, pauseMs)
                        .onComplete(runGeluTransition)
                        .start();
                } else {
                    runGeluTransition();
                }
            })
            .start();
    }

    _animateMlpActivationGelu(lane, applyActivation, onComplete, options = {}) {
        const expandedGroup = lane.expandedVecGroup;
        const segmentVecs = lane.expandedVecSegments;
        const durationMs = Number.isFinite(options.durationMs) ? options.durationMs : 500;
        const riseExtra = Number.isFinite(options.riseExtra) ? options.riseExtra : 24;
        const curveHeight = Number.isFinite(options.curveHeight) ? options.curveHeight : 36;
        const curveDomain = Number.isFinite(options.curveDomain) ? options.curveDomain : 2.5;
        const negativeScale = Number.isFinite(options.negativeScale) ? options.negativeScale : 2.6;
        const activationSwitchT = THREE.MathUtils.clamp(
            Number.isFinite(options.activationSwitchT) ? options.activationSwitchT : 0.35,
            0,
            0.9
        );
        const safeCurveDomain = curveDomain > 0 ? curveDomain : 1;

        if (lane) {
            lane.mlpGeluActive = true;
            lane.mlpGeluComplete = false;
        }
        const finishGelu = () => {
            if (!lane) return;
            lane.mlpGeluActive = false;
            lane.mlpGeluComplete = true;
        };

        if (!expandedGroup || !segmentVecs || segmentVecs.length === 0) {
            finishGelu();
            if (typeof applyActivation === 'function') applyActivation();
            if (typeof onComplete === 'function') onComplete();
            return;
        }
        if (typeof TWEEN === 'undefined' || durationMs <= 0) {
            finishGelu();
            if (typeof applyActivation === 'function') applyActivation();
            if (typeof onComplete === 'function') onComplete();
            return;
        }

        const segmentCounts = segmentVecs.map(seg => Math.max(1, Math.floor(seg.instanceCount || 1)));
        const totalCount = segmentCounts.reduce((sum, count) => sum + count, 0);
        const offsets = segmentVecs.map((_, idx) => new Float32Array(segmentCounts[idx]));
        let globalIndex = 0;
        for (let s = 0; s < segmentVecs.length; s++) {
            const count = segmentCounts[s];
            for (let i = 0; i < count; i++) {
                const t = totalCount > 1 ? globalIndex / (totalCount - 1) : 0.5;
                const x = (t * 2 - 1) * safeCurveDomain;
                const y = geluApprox(x);
                const scaledY = y < 0 ? y * negativeScale : y;
                offsets[s][i] = (scaledY / safeCurveDomain) * curveHeight;
                globalIndex++;
            }
        }

        const baseY = expandedGroup.position.y;
        let activationApplied = false;
        const state = { t: 0 };

        const applyActivationOnce = () => {
            if (activationApplied) return;
            activationApplied = true;
            if (typeof applyActivation === 'function') applyActivation();
        };

        // Bend into a GELU curve, switch colors mid-flight, then return to the flat vector.
        new TWEEN.Tween(state)
            .to({ t: 1 }, durationMs)
            .easing(TWEEN.Easing.Quadratic.InOut)
            .onUpdate(() => {
                const curveT = Math.sin(Math.PI * state.t);
                if (state.t >= activationSwitchT) {
                    applyActivationOnce();
                }
                const liftProgress = state.t >= activationSwitchT
                    ? (state.t - activationSwitchT) / Math.max(1e-6, 1 - activationSwitchT)
                    : 0;
                const liftT = TWEEN.Easing.Quadratic.Out(THREE.MathUtils.clamp(liftProgress, 0, 1));
                expandedGroup.position.y = baseY + riseExtra * liftT;

                for (let s = 0; s < segmentVecs.length; s++) {
                    const segVec = segmentVecs[s];
                    const offsetRow = offsets[s];
                    const count = segmentCounts[s];
                    for (let i = 0; i < count; i++) {
                        segVec.setInstanceAppearance(i, offsetRow[i] * curveT, null, null, false);
                    }
                    if (typeof segVec.markInstanceMatrixDirty === 'function') {
                        segVec.markInstanceMatrixDirty();
                    } else if (segVec.mesh) {
                        segVec.mesh.instanceMatrix.needsUpdate = true;
                    }
                }
            })
            .onComplete(() => {
                applyActivationOnce();
                expandedGroup.position.y = baseY + riseExtra;
                for (let s = 0; s < segmentVecs.length; s++) {
                    const segVec = segmentVecs[s];
                    const count = segmentCounts[s];
                    for (let i = 0; i < count; i++) {
                        segVec.resetInstanceAppearance(i);
                    }
                    if (typeof segVec.markInstanceMatrixDirty === 'function') {
                        segVec.markInstanceMatrixDirty();
                    } else if (segVec.mesh) {
                        segVec.mesh.instanceMatrix.needsUpdate = true;
                    }
                }
                finishGelu();
                if (typeof onComplete === 'function') onComplete();
            })
            .start();
    }

    /**
     * Animate through MLP down-projection (3072 → 768 dimensions)
     */
    _animateMlpDownProjection(lane) {
        const expandedGroup = lane.expandedVecGroup;
        if (!expandedGroup || typeof TWEEN === 'undefined') {
            if (lane) lane.mlpDownComplete = true;
            return;
        }
        lane.mlpDownStarted = true;
        lane.mlpDownComplete = false;
        
        const orangeColor = new THREE.Color(0xc07a12);
        const downBottomY = this.mlpDown.group.position.y - MLP_MATRIX_PARAMS_DOWN.height / 2;
        const downTopY = this.mlpDown.group.position.y + MLP_MATRIX_PARAMS_DOWN.height / 2;
        
        const startY = expandedGroup.position.y;
        const totalDist = downTopY - startY;
        const durationDown = (Math.abs(totalDist) / (ANIM_RISE_SPEED_INSIDE_LN * GLOBAL_ANIM_SPEED_MULT)) * 1000;
        const colorDurationDown = this._skipToEndActive
            ? Math.max(durationDown, SKIP_MLP_COLOR_MIN_MS)
            : durationDown;

        const matrixBottomWidth = MLP_MATRIX_PARAMS_DOWN.width;
        const matrixTopWidth = MLP_MATRIX_PARAMS_DOWN.width * MLP_MATRIX_PARAMS_DOWN.topWidthFactor;
        const widthPadding = 0.95;
        let expandedFullWidth = Number.isFinite(expandedGroup.userData.fullWidth)
            ? expandedGroup.userData.fullWidth
            : null;
        if (!Number.isFinite(expandedFullWidth) && lane.expandedVecSegments && lane.expandedVecSegments[0]) {
            const baseSegment = lane.expandedVecSegments[0];
            const segLength = Number.isFinite(baseSegment.instanceCount)
                ? baseSegment.instanceCount
                : this._getBaseVectorLength();
            const segWidth = baseSegment.getBaseWidthConstant() * baseSegment.getWidthScale() * segLength;
            expandedFullWidth = segWidth * 4;
        }
        const clampScaleForY = (yPos) => {
            if (!Number.isFinite(expandedFullWidth) || expandedFullWidth <= 0) return null;
            const denom = downTopY - downBottomY;
            if (!Number.isFinite(denom) || denom === 0) return null;
            const t = THREE.MathUtils.clamp((yPos - downBottomY) / denom, 0, 1);
            const matrixWidth = THREE.MathUtils.lerp(matrixBottomWidth, matrixTopWidth, t);
            const maxScale = (matrixWidth * widthPadding) / expandedFullWidth;
            return Math.min(1, Math.max(0.01, maxScale));
        };
        
        // Matrix colour + emissive animation for glow
        const startIntensity = 0.1;
        const peakIntensity = 0.24;
        const finalIntensity = 0.30;
        const downState = { t: 0, emissive: startIntensity };

        new TWEEN.Tween(downState)
            .to({ t: 1, emissive: peakIntensity }, colorDurationDown * 0.6)
            .easing(TWEEN.Easing.Quadratic.InOut)
            .onUpdate(() => {
                const col = new THREE.Color(INACTIVE_COMPONENT_COLOR).lerp(orangeColor, downState.t);
                this.mlpDown.setColor(col);
                this.mlpDown.setEmissive(col, downState.emissive);
            })
            .onComplete(() => {
                new TWEEN.Tween(downState)
                    .to({ emissive: finalIntensity }, colorDurationDown * 0.4)
                    .easing(TWEEN.Easing.Quadratic.InOut)
                    .onUpdate(() => {
                        this.mlpDown.setEmissive(orangeColor, downState.emissive);
                    })
                    .start();
            })
            .start();
            
        // Move expanded vector through matrix
        new TWEEN.Tween(expandedGroup.position)
            .to({ y: downTopY }, durationDown)
            .easing(TWEEN.Easing.Linear.None)
            .onUpdate(() => {
                if (expandedGroup.position.y >= downBottomY) {
                    const maxScale = clampScaleForY(expandedGroup.position.y);
                    if (Number.isFinite(maxScale)) {
                        const nextScale = Math.min(expandedGroup.scale.x, maxScale);
                        if (nextScale !== expandedGroup.scale.x) {
                            expandedGroup.scale.setScalar(nextScale);
                        }
                    }
                }

                // Check if we've reached the middle of the down-projection matrix
                const midY = this.mlpDown.group.position.y;
                if (!lane.collapsedInMatrix && expandedGroup.position.y >= midY) {
                    lane.collapsedInMatrix = true;
                    
                    // Create collapsed vector at current position
                    const collapseVec = this._createPrismVector(
                        lane.expandedVecSegments[0].rawData.slice(),
                        expandedGroup.position.clone(),
                        30,
                        lane.expandedVecSegments[0].instanceCount
                    );
                    // Do not start a local trail yet; we'll create a clean path trail
                    // when rising above the MLP to avoid zig-zag artifacts.
                    
                    // Copy gradient colors
                    if (Array.isArray(lane.expandedVecSegments[0].currentKeyColors) && lane.expandedVecSegments[0].currentKeyColors.length) {
                        collapseVec.currentKeyColors = lane.expandedVecSegments[0].currentKeyColors.map(c => c.clone());
                        collapseVec.updateInstanceGeometryAndColors();
                    }
                    
                    this.raycastRoot.add(collapseVec.group);
                    expandedGroup.visible = false;
                    
                    lane.finalVecAfterMlp = collapseVec;
                    const mlpDownData = this._getMlpDownData(lane);
                    if (mlpDownData) {
                        applyVectorData(
                            collapseVec,
                            mlpDownData,
                            lane.tokenLabel ? `MLP Down Projection - ${lane.tokenLabel}` : 'MLP Down Projection',
                            this._getLaneMeta(lane, 'mlp.down')
                        );
                    }
                    
                    // Continue animating the collapsed vector for the rest of the journey
                    new TWEEN.Tween(collapseVec.group.position)
                        .to({ y: downTopY }, durationDown / 2) // Half duration since we're halfway
                        .easing(TWEEN.Easing.Linear.None)
                        .onUpdate(() => {
                        })
                        .start();
                }
            })
            .onComplete(() => {
                lane.mlpDownComplete = true;
                this.mlpDown.setColor(orangeColor);
                this.mlpDown.setEmissive(orangeColor, finalIntensity);
                
                // Ensure both MLP matrices are fully opaque at the end
                this.mlpUp.setMaterialProperties({ opacity: 1.0, transparent: false });
                this.mlpDown.setMaterialProperties({ opacity: 1.0, transparent: false });
                
                // If we haven't collapsed yet (shouldn't happen), do it now
                if (!lane.collapsedInMatrix) {
                    this._collapseToSingle(lane);
                } else {
                    const mlpDownData = this._getMlpDownData(lane);
                    if (mlpDownData && lane.finalVecAfterMlp) {
                        applyVectorData(
                            lane.finalVecAfterMlp,
                            mlpDownData,
                            lane.tokenLabel ? `MLP Down Projection - ${lane.tokenLabel}` : 'MLP Down Projection',
                            this._getLaneMeta(lane, 'mlp.down')
                        );
                    }
                    // Continue with the rise animation
                    this._riseAfterMlp(lane);
                }
            })
            .start();
    }

    /**
     * Collapse expanded vector back to single 768-dim vector
     */
    _collapseToSingle(lane) {
        const expandedGroup = lane.expandedVecGroup;
        const segmentVecs = lane.expandedVecSegments;
        if (!expandedGroup || !segmentVecs || segmentVecs.length === 0) return;
        
        // Create collapsed vector
        const collapseVec = this._createPrismVector(
            segmentVecs[0].rawData.slice(),
            expandedGroup.position.clone(),
            30,
            segmentVecs[0].instanceCount
        );
        // Defer trail creation until after the rise above the MLP.
        
        // Copy gradient colors
        if (Array.isArray(segmentVecs[0].currentKeyColors) && segmentVecs[0].currentKeyColors.length) {
            collapseVec.currentKeyColors = segmentVecs[0].currentKeyColors.map(c => c.clone());
            collapseVec.updateInstanceGeometryAndColors();
        }
        
        this.raycastRoot.add(collapseVec.group);
        expandedGroup.visible = false;
        
        lane.finalVecAfterMlp = collapseVec;
        const mlpDownData = this._getMlpDownData(lane);
        if (mlpDownData) {
            applyVectorData(
                collapseVec,
                mlpDownData,
                lane.tokenLabel ? `MLP Down Projection - ${lane.tokenLabel}` : 'MLP Down Projection',
                this._getLaneMeta(lane, 'mlp.down')
            );
        }
        
        this._riseAfterMlp(lane);
    }
    
    /**
     * Rise above matrix after MLP processing
     */
    _riseAfterMlp(lane) {
        const vec = lane.finalVecAfterMlp;
        if (!vec) return;
        
        // Rise above matrix
        const riseAbove = 40;
        const riseDur = (riseAbove / (ANIM_RISE_SPEED_INSIDE_LN * GLOBAL_ANIM_SPEED_MULT)) * 1000;
        
        new TWEEN.Tween(vec.group.position)
            .to({ y: vec.group.position.y + riseAbove }, riseDur)
            .easing(TWEEN.Easing.Quadratic.InOut)
            .onStart(() => {
                // Start a dedicated post-MLP path trail for a clean rise-then-right
                try {
                    vec.userData = vec.userData || {};
                    if (!vec.userData.mlpTrail) {
                        const pathTrail = new StraightLineTrail(this.root, 0xffffff, 1, undefined, undefined, TRAIL_MIN_SEGMENT_DISTANCE);
                        pathTrail.start(vec.group.position.clone());
                        vec.userData.mlpTrail = pathTrail;
                    }
                } catch (_) { /* optional visual */ }
            })
            .onUpdate(() => {
                const t = vec && vec.userData && vec.userData.mlpTrail;
                if (t && typeof t.update === 'function') {
                    t.update(vec.group.position);
                }
            })
            .onComplete(() => {
                // Update residual stream target height
                if (this.mhsaAnimation) {
                    this.mhsaAnimation.finalOriginalY = vec.group.position.y - ORIGINAL_TO_PROCESSED_GAP;
                }
                
                // Move back to residual stream
                this._returnToResidualStream(lane, vec);
            })
            .start();
    }

    /**
     * Move processed vector back to residual stream for final addition
     */
    _returnToResidualStream(lane, vec) {
        const horizDist = Math.abs(vec.group.position.x);
        const horizDur = (horizDist / (ANIM_HORIZ_SPEED * GLOBAL_ANIM_SPEED_MULT)) * 1000;
        // Lock Y to eliminate tiny easing jitter at the corner so the trail
        // forms a clean right-angle when switching from vertical to horizontal.
        const lockedY = vec.group.position.y;

        new TWEEN.Tween(vec.group.position)
            .to({ x: 0, y: lockedY }, horizDur)
            .easing(TWEEN.Easing.Quadratic.InOut)
            .onStart(() => {
                vec.group.position.y = lockedY;
            })
            .onUpdate(() => {
                // Ensure Y stays perfectly constant during horizontal travel
                // to avoid any up-then-down wobble in the trail.
                vec.group.position.y = lockedY;
                // Continue updating the post-MLP path trail during horizontal move
                try {
                    const t = vec && vec.userData && vec.userData.mlpTrail;
                    if (t && typeof t.update === 'function') t.update(vec.group.position);
                } catch (_) { /* optional */ }
            })
            .onComplete(() => {
                // Freeze the temporary post-MLP path trail into a static line so
                // the horizontal segment persists visually after the move back.
                try {
                    const t = vec && vec.userData && vec.userData.mlpTrail;
                    if (t) {
                        if (typeof t.snapLastPointTo === 'function') {
                            t.snapLastPointTo(vec.group.position);
                        }
                        const colorHex = (t._material && t._material.color)
                            ? t._material.color.getHex() : undefined;
                        const frozenOpacity = (typeof t._opacity === 'number') ? t._opacity : undefined;
                        mergeTrailsIntoLineSegments(
                            [t],
                            this.root,
                            colorHex,
                            undefined,
                            frozenOpacity,
                            null
                        );
                        if (typeof t.dispose === 'function') t.dispose();
                        if (vec && vec.userData) delete vec.userData.mlpTrail;
                    }
                } catch (_) { /* optional */ }
                // Prevent double-drawing along the residual stream: retire any temporary
                // local trail for the collapsed vector. Keep the existing world-space
                // residual trail owned by the original vector/lane to avoid concurrent
                // updates from both vectors at x=0.
                try {
                    const localTrail = vec && vec.userData && vec.userData.trail;
                    if (localTrail && localTrail._scene === this.root) {
                        if (typeof localTrail.snapLastPointTo === 'function') {
                            localTrail.snapLastPointTo(vec.group.position);
                        }
                        const colorHex = (localTrail._material && localTrail._material.color)
                            ? localTrail._material.color.getHex() : undefined;
                        const frozenOpacity = (typeof localTrail._opacity === 'number') ? localTrail._opacity : undefined;
                        mergeTrailsIntoLineSegments(
                            [localTrail],
                            this.root,
                            colorHex,
                            undefined,
                            frozenOpacity,
                            null
                        );
                        if (vec.userData) delete vec.userData.trail;
                    }
                } catch (_) { /* no-op */ }

                // Perform final addition with original vector
                if (this.mhsaAnimation && lane.originalVec) {
                    // Track this addition animation
                    this._pendingAdditions++;
                    
                    // Trigger the final addition animation (originalVec ➔ vec)
                    // Prisms should rise from the lower original vector up into the processed one.
                    const postMlpData = this._getPostMlpResidualData(lane);
                    if (postMlpData) {
                        lane.additionTargetData = postMlpData;
                    }
                    this.mhsaAnimation._startAdditionAnimation(lane.originalVec, vec, lane, () => {
                        if (postMlpData) {
                            applyVectorData(
                                lane.originalVec,
                                postMlpData,
                                lane.tokenLabel ? `Post-MLP Residual - ${lane.tokenLabel}` : 'Post-MLP Residual',
                                this._getLaneMeta(lane, 'residual.post_mlp')
                            );
                        }
                    });
                    
                    // Set up completion callback for when addition finishes
                    this._scheduleAdditionCompletion(lane);
                }
                lane.ln2Phase = 'done';
                this._emitProgress();
            })
            .start();
    }

    // ------------------------------------------------------------
    // Public helpers
    // ------------------------------------------------------------

    /** Inject external lanes from the previous layer and activate animation */
    activateWithLanes(externalLanes) {
        if (this.isActive) return; // already active
        
        // Set up lanes but don't start animation yet - wait for positioning
        this.isActive = false; // keep inactive during transition
        this._transitionPhase = 'positioning';
        const ln1CenterY = (this.ln1 && this.ln1.group && Number.isFinite(this.ln1.group.position.y))
            ? this.ln1.group.position.y
            : LAYER_NORM_1_Y_POS;
        const ln2CenterY = (this.ln2 && this.ln2.group && Number.isFinite(this.ln2.group.position.y))
            ? this.ln2.group.position.y
            : (LAYER_NORM_2_Y_POS + (this.mhsaAnimation?.outputProjMatrixYOffset || 0));
        const ln1TopY = ln1CenterY + LN_PARAMS.height / 2;
        this._createLanesFromExternal(externalLanes, BRANCH_X, ln1CenterY, ln2CenterY, ln1TopY);

        // MHSAAnimation's constructor already sets the correct initial rise speed.
        // No override is needed here.
    }

    /** Replace/assign onFinished callback after construction */
    setOnFinished(cb) { this.onFinished = typeof cb === 'function' ? cb : null; }

    setSkipToEndMode(enabled = true) {
        this._skipToEndActive = !!enabled;
        if (this._skipToEndActive) {
            this._applySkipVectorVisibility();
        }
        if (this.mhsaAnimation && typeof this.mhsaAnimation.setSkipToEndMode === 'function') {
            this.mhsaAnimation.setSkipToEndMode(this._skipToEndActive);
        }
    }

    postUpdate() {
        if (this._skipToEndActive) {
            this._applySkipVectorVisibility();
        }
    }

    refreshSkipVisibility() {
        if (this._skipToEndActive) {
            this._applySkipVectorVisibility();
        }
    }

    restoreResidualVectorVisibility(lanes) {
        if (!Array.isArray(lanes) || !lanes.length) return;
        lanes.forEach(lane => {
            const vec = lane && lane.originalVec;
            if (!vec || !vec.group) return;
            this._restoreVectorVisibility(vec);
        });
    }

    _restoreVectorVisibility(vec) {
        if (!vec || !vec.group) return;
        const hidden = this._skipHiddenMaterials;
        vec.group.traverse(obj => {
            if (!obj || !obj.isMesh || !obj.material) return;
            obj.visible = true;
            const mats = Array.isArray(obj.material) ? obj.material : [obj.material];
            mats.forEach(mat => {
                if (!mat) return;
                const prev = hidden.get(mat);
                if (prev) {
                    mat.opacity = prev.opacity;
                    mat.transparent = prev.transparent;
                    mat.depthWrite = prev.depthWrite;
                    mat.needsUpdate = true;
                    hidden.delete(mat);
                }
            });
        });
    }

    _isVectorVisual(obj) {
        if (!obj) return false;
        const data = obj.userData;
        if (data && data.isVector) return true;
        if (data && data.mergedKVMeta) return true;
        if (data && typeof data.label === 'string' && data.label.includes('Vector')) return true;
        const parent = obj.parent;
        if (parent && parent.userData && parent.userData.isVector) return true;
        if (parent && parent.userData && typeof parent.userData.label === 'string' && parent.userData.label.includes('Vector')) return true;
        return false;
    }

    _applySkipVectorVisibility() {
        if (!this._skipToEndActive || !this.root) return;
        const hidden = this._skipHiddenMaterials;
        this.root.traverse(obj => {
            if (!obj || !obj.isMesh || !obj.material) return;
            if (!this._isVectorVisual(obj)) return;
            const allowVisible = (obj.userData && obj.userData.skipVisible)
                || (obj.parent && obj.parent.userData && obj.parent.userData.skipVisible);
            const mats = Array.isArray(obj.material) ? obj.material : [obj.material];
            if (allowVisible) {
                obj.visible = true;
                mats.forEach(mat => {
                    if (!mat) return;
                    const prev = hidden.get(mat);
                    if (prev) {
                        mat.opacity = prev.opacity;
                        mat.transparent = prev.transparent;
                        mat.depthWrite = prev.depthWrite;
                        mat.needsUpdate = true;
                        hidden.delete(mat);
                    }
                });
                return;
            }
            mats.forEach(mat => {
                if (!mat) return;
                if (!hidden.has(mat)) {
                    hidden.set(mat, {
                        opacity: mat.opacity,
                        transparent: mat.transparent,
                        depthWrite: mat.depthWrite
                    });
                }
                const needsUpdate = mat.opacity !== 0 || mat.transparent !== true || mat.depthWrite !== false;
                mat.transparent = true;
                mat.opacity = 0;
                mat.depthWrite = false;
                if (needsUpdate) mat.needsUpdate = true;
            });
            obj.visible = false;
        });
    }

    _getLaneMeta(lane, stage, extra = {}) {
        return {
            stage,
            layerIndex: this.index,
            tokenIndex: lane && Number.isFinite(lane.tokenIndex) ? lane.tokenIndex : undefined,
            tokenLabel: lane && lane.tokenLabel ? lane.tokenLabel : undefined,
            ...extra
        };
    }

    _getBaseVectorLength() {
        return Number.isFinite(this._baseVectorLength) ? this._baseVectorLength : VECTOR_LENGTH_PRISM;
    }

    _getInstanceCountFromData(values, fallback = null) {
        if (Array.isArray(values) || ArrayBuffer.isView(values)) {
            return Math.max(1, values.length || 1);
        }
        const base = Number.isFinite(fallback) ? fallback : this._getBaseVectorLength();
        return Math.max(1, base);
    }

    _createPrismVector(values, position, numSubsections = 30, instanceCount = null) {
        const count = Number.isFinite(instanceCount)
            ? Math.max(1, Math.floor(instanceCount))
            : this._getInstanceCountFromData(values);
        const data = Array.isArray(values)
            ? values
            : ArrayBuffer.isView(values)
                ? Array.from(values)
                : null;
        return new VectorVisualizationInstancedPrism(data, position, numSubsections, count);
    }

    _getLayerNormParamData(kind, param) {
        const baseLength = this._getBaseVectorLength();
        return getLayerNormParamData(this.index, kind, param, baseLength);
    }

    _applyLayerNormParamVector(targetVec, kind, param, colorOptions = null) {
        if (!targetVec) return false;
        const data = this._getLayerNormParamData(kind, param);
        if (!data) return false;
        const lnLabel = kind === 'ln1' ? 'LN1' : kind === 'ln2' ? 'LN2' : String(kind).toUpperCase();
        const paramLabel = param === 'scale' ? 'Scale (gamma)' : 'Shift (beta)';
        const label = `${lnLabel} ${paramLabel}`;
        const meta = {
            stage: `${kind}.param.${param}`,
            layerIndex: this.index,
            notes: param === 'scale'
                ? 'LayerNorm scale (gamma) parameter'
                : 'LayerNorm shift (beta) parameter'
        };
        return applyVectorData(targetVec, data, label, meta, colorOptions);
    }

    _getTokenIndexForLane(laneIdx) {
        if (!this.activationSource) return laneIdx;
        return this.activationSource.getLaneTokenIndex(laneIdx, this._laneCount);
    }

    _getTokenLabel(tokenIndex) {
        if (!this.activationSource) return null;
        return formatTokenLabel(this.activationSource.getTokenString(tokenIndex));
    }

    _getEmbeddingData(lane, kind) {
        if (!this.activationSource || !lane) return null;
        return this.activationSource.getEmbedding(kind, lane.tokenIndex, this._getBaseVectorLength());
    }

    _getLayerIncomingData(lane) {
        if (!this.activationSource || !lane) return null;
        return this.activationSource.getLayerIncoming(this.index, lane.tokenIndex, this._getBaseVectorLength());
    }

    _getLn1Data(lane, stage) {
        if (!this.activationSource || !lane) return null;
        return this.activationSource.getLayerLn1(this.index, stage, lane.tokenIndex, this._getBaseVectorLength());
    }

    _getLn2Data(lane, stage) {
        if (!this.activationSource || !lane) return null;
        return this.activationSource.getLayerLn2(this.index, stage, lane.tokenIndex, this._getBaseVectorLength());
    }

    _getAttentionOutputProjectionData(lane) {
        if (!this.activationSource || !lane) return null;
        return this.activationSource.getAttentionOutputProjection(this.index, lane.tokenIndex, this._getBaseVectorLength());
    }

    _getPostAttentionResidualData(lane) {
        if (!this.activationSource || !lane) return null;
        return this.activationSource.getPostAttentionResidual(this.index, lane.tokenIndex, this._getBaseVectorLength());
    }

    _getMlpUpData(lane) {
        if (!this.activationSource || !lane) return null;
        const targetLength = this._getBaseVectorLength() * 4;
        return this.activationSource.getMlpUp(this.index, lane.tokenIndex, targetLength);
    }

    _getMlpActivationData(lane) {
        if (!this.activationSource || !lane) return null;
        const targetLength = this._getBaseVectorLength() * 4;
        return this.activationSource.getMlpActivation(this.index, lane.tokenIndex, targetLength);
    }

    _getMlpDownData(lane) {
        if (!this.activationSource || !lane) return null;
        return this.activationSource.getMlpDown(this.index, lane.tokenIndex, this._getBaseVectorLength());
    }

    _getPostMlpResidualData(lane) {
        if (!this.activationSource || !lane) return null;
        return this.activationSource.getPostMlpResidual(this.index, lane.tokenIndex, this._getBaseVectorLength());
    }

    // ------------------------------------------------------------
    // Internal lane creation helpers (delegated to gpt2LaneBuilder.js)
    // ------------------------------------------------------------

    _createFreshLanes(offsetX, ln1CenterY, ln2CenterY, ln1TopY) {
        createFreshLanes(this, offsetX, ln1CenterY, ln2CenterY, ln1TopY);
    }

    _createAdditionPlaceholders(offsetX, ln1CenterY, ln2CenterY) {
        createAdditionPlaceholders(this, offsetX, ln1CenterY, ln2CenterY);
    }

    _createLanesFromExternal(externalLanes, offsetX, ln1CenterY, ln2CenterY, ln1TopY) {
        createLanesFromExternal(this, externalLanes, offsetX, ln1CenterY, ln2CenterY, ln1TopY);
    }

    _buildSingleLane(oldLane, offsetX, ln1CenterY, ln2CenterY, startY_override, meetY, laneIdx, slitSpacing) {
        buildSingleLane(this, oldLane, offsetX, ln1CenterY, ln2CenterY, startY_override, meetY, laneIdx, slitSpacing);
    }

    /**
     * Schedule callback for when addition animation completes
     */
    _scheduleAdditionCompletion(lane) {
        // Mirror timings from additionUtils to ensure consistent completion detection
        const duration      = PRISM_ADD_ANIM_BASE_DURATION             / PRISM_ADD_ANIM_SPEED_MULT;
        const flashDuration = PRISM_ADD_ANIM_BASE_FLASH_DURATION       / PRISM_ADD_ANIM_SPEED_MULT;
        const delayBetween  = PRISM_ADD_ANIM_BASE_DELAY_BETWEEN_PRISMS / PRISM_ADD_ANIM_SPEED_MULT;
        const vectorLength = lane?.addTarget?.instanceCount
            || lane?.originalVec?.instanceCount
            || this._getBaseVectorLength();
        const totalAnimTime = duration + flashDuration + vectorLength * delayBetween;

        lane.additionComplete = false;

        const complete = () => {
            if (this._pendingAdditions > 0) {
                this._pendingAdditions--;
            }
            lane.additionComplete = true;
            if (lane._additionCompletionTween) {
                try { lane._additionCompletionTween.stop(); } catch (_) { /* no-op */ }
                lane._additionCompletionTween = null;
            }
        };

        if (typeof TWEEN !== 'undefined') {
            const tween = new TWEEN.Tween({ progress: 0 })
                .to({ progress: 1 }, totalAnimTime + 100)
                .onComplete(complete)
                .start();
            lane._additionCompletionTween = tween;
        } else {
            setTimeout(complete, totalAnimTime + 100);
        }
    }

    /**
     * Walk every mesh in this layer and turn off Three.js transparency for
     * materials that are already fully opaque.  This removes them from the
     * per-frame depth-sorting list, improving performance once the layer is
     * finished and static.
     */
    _makeLayerOpaque() {
        this.root.traverse(obj => {
            if (obj.isMesh && obj.material) {
                const mats = Array.isArray(obj.material) ? obj.material : [obj.material];
                mats.forEach(mat => {
                    // Only change materials that use blending but are visually opaque
                    const fullyOpaque = (mat.opacity === undefined) || (mat.opacity >= 0.99);
                    if (mat.transparent && fullyOpaque) {
                        mat.transparent = false;
                        mat.depthWrite = true;
                        mat.needsUpdate = true;
                    }
                });
            }
        });

        // After making meshes opaque, fuse the numerous per-lane residual
        // trails into a single static LineSegments and trim the live trails so
        // they can continue seamlessly in higher layers with minimal cost.
        try {
            // 1) Freeze residual trails completed so far into a static merged line
            if (Array.isArray(this.lanes) && this.lanes.length) {
                const origTrailSet = new Set();
                const residualGroups = new Map();
                const getTrailStyle = (trail) => {
                    let color = null;
                    if (trail && trail._material && trail._material.color) {
                        color = trail._material.color.clone();
                    } else if (trail && typeof trail._color !== 'undefined') {
                        color = new THREE.Color(trail._color);
                    }
                    const opacity = (trail && typeof trail._opacity === 'number') ? trail._opacity : null;
                    const colorKey = color
                        ? `${color.r.toFixed(4)},${color.g.toFixed(4)},${color.b.toFixed(4)}`
                        : 'default';
                    const opacityKey = opacity != null ? opacity.toFixed(4) : 'default';
                    return { color, opacity, key: `${colorKey}|${opacityKey}` };
                };
                this.lanes.forEach(l => {
                    // Prefer the dedicated world-space residual trail reference if available
                    const t = (l && l.originalTrail)
                        || (l && l.originalVec && l.originalVec.userData && l.originalVec.userData.trail);
                    if (t) {
                        origTrailSet.add(t);
                    }
                    if (t && typeof t.extractSegmentsAndTrim === 'function') {
                        const seg = t.extractSegmentsAndTrim({ preserveSegments: 1 });
                        if (seg && seg.length) {
                            const style = getTrailStyle(t);
                            let group = residualGroups.get(style.key);
                            if (!group) {
                                group = { segments: [], color: style.color, opacity: style.opacity };
                                residualGroups.set(style.key, group);
                            }
                            group.segments.push(seg);
                        }
                    }
                });
                residualGroups.forEach((group) => {
                    if (!group.segments.length) return;
                    buildMergedLineSegmentsFromSegments(
                        group.segments,
                        this._globalScene || this.root,
                        group.color || undefined,
                        undefined,
                        group.opacity != null ? group.opacity : undefined,
                        null
                    );
                });

                // 2) Merge all other per-layer (non-residual) trails under this.root
                const allLayerTrails = collectTrailsUnder(this.root);
                const otherTrails = allLayerTrails.filter(t => !origTrailSet.has(t));
                if (otherTrails.length) {
                    const groups = new Map();
                    otherTrails.forEach((trail) => {
                        const style = getTrailStyle(trail);
                        let group = groups.get(style.key);
                        if (!group) {
                            group = { trails: [], color: style.color, opacity: style.opacity };
                            groups.set(style.key, group);
                        }
                        group.trails.push(trail);
                    });
                    groups.forEach((group) => {
                        if (!group.trails.length) return;
                        mergeTrailsIntoLineSegments(
                            group.trails,
                            this.root,
                            group.color || undefined,
                            undefined,
                            group.opacity != null ? group.opacity : undefined,
                            null
                        );
                    });
                }
            }
        } catch (e) {
            // Best-effort optimisation; ignore failures to avoid breaking the demo.
        }
    }

    /**
     * Remove heavy dynamic geometry (vectors) after the layer has
     * handed its residual stream to the next layer.  This is more aggressive
     * than the previous implementation: we now detach the objects from the
     * scene graph and dispose of their GPU resources so they no longer add
     * draw-calls, memory pressure or ray-casting traversal cost.
     */
    hideDynamicGeometry() {
        const disposeObj = (obj) => {
            if (!obj) return;

            // Dispose materials
            if (obj.material) {
                const materials = Array.isArray(obj.material) ? obj.material : [obj.material];
                materials.forEach(mat => mat && mat.dispose && mat.dispose());
            }

            // Dispose geometry
            if (obj.geometry) {
                obj.geometry.dispose();
            }

            // Recursively dispose children
            if (obj.children && obj.children.length) {
                [...obj.children].forEach(child => disposeObj(child));
            }

            // Finally, detach from parent to remove from scene traversal
            if (obj.parent) {
                obj.parent.remove(obj);
            }
        };

        const vectorGroups = [];
        this.root.traverse(obj => {
            if (!obj || !obj.userData) return;
            if (!obj.isGroup) return;
            if (obj.userData.isVector || obj.userData.label === 'Vector' || obj.userData.label === 'Vector24') {
                vectorGroups.push(obj);
            }
        });

        vectorGroups.forEach(obj => disposeObj(obj));
    }
}
