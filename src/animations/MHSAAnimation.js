import * as THREE from 'three';
import { WeightMatrixVisualization } from '../components/WeightMatrixVisualization.js';
import { VectorVisualizationInstancedPrism } from '../components/VectorVisualizationInstancedPrism.js';
import { BatchedPrismVectorSet } from '../components/BatchedPrismVectorSet.js';
import { SegmentTrailBatch, StraightLineTrail, mergeTrailsIntoLineSegments } from '../utils/trailUtils.js';
import { TRAIL_COLOR, TRAIL_MIN_SEGMENT_DISTANCE, TRAIL_OPACITY } from '../utils/trailConstants.js';
import { logRandomColorDebug } from '../utils/randomColorDebug.js';



import { buildActivationData, applyActivationDataToVector } from '../utils/activationMetadata.js';
import { buildHueRangeOptions, mapValueToHueRange } from '../utils/colors.js';
import { MHSA_MATRIX_INITIAL_RESTING_COLOR, MHSA_BRIGHT_GREEN, MHSA_DARK_TINTED_GREEN, MHSA_BRIGHT_BLUE, MHSA_DARK_TINTED_BLUE, MHSA_BRIGHT_RED, MHSA_DARK_TINTED_RED, MHA_FINAL_Q_COLOR, MHA_FINAL_K_COLOR, MHA_FINAL_V_COLOR, MHA_OUTPUT_PROJECTION_MATRIX_Y_OFFSET_ABOVE_ROW, MHA_OUTPUT_PROJECTION_MATRIX_PARAMS, MHA_OUTPUT_PROJECTION_MATRIX_COLOR, MHA_VALUE_SPECTRUM_COLOR, MHA_VALUE_HUE_SPREAD, MHA_VALUE_LIGHTNESS_MIN, MHA_VALUE_LIGHTNESS_MAX, MHA_VALUE_RANGE_MIN, MHA_VALUE_RANGE_MAX, MHA_VALUE_CLAMP_MAX, MHA_VALUE_KEY_COLOR_COUNT, MHA_WEIGHTED_SUM_DOCK_OFFSET } from './LayerAnimationConstants.js';
import { INACTIVE_COMPONENT_COLOR, MHSA_DUPLICATE_VECTOR_RISE_SPEED, MHSA_PASS_THROUGH_TOTAL_DURATION_MS, MHSA_RESULT_RISE_OFFSET_Y, MHSA_HEAD_VECTOR_STOP_BELOW, MHA_RESULT_RISE_DURATION_BASE_MS, DECORATIVE_FADE_MS, DECORATIVE_FADE_DELAY_MS, MERGE_TO_ROW_DELAY_AFTER_FADE_MS, HEAD_COLOR_TRANSITION_MS, MERGE_POST_COLOR_TRANSITION_DELAY_MS, MERGE_EXTRA_BUFFER_MS, OUTPUT_PROJ_STAGE1_MS, OUTPUT_PROJ_STAGE2_MS, OUTPUT_PROJ_STAGE3_MS, GLOBAL_ANIM_SPEED_MULT, MHSA_MATRIX_MAX_EMISSIVE_INTENSITY, SKIP_TRAIL_FADE_IN_MS, SA_RED_EXTRA_RISE } from '../utils/constants.js';
import {
    // Constants needed for setup & animation
    MHA_MATRIX_PARAMS,
    NUM_VECTOR_LANES,
    NUM_HEAD_SETS_LAYER,
    HEAD_SET_GAP_LAYER,
    MHA_INTERNAL_MATRIX_SPACING,
    HEAD_VECTOR_STOP_BELOW,
    ANIM_HORIZ_SPEED,
    SIDE_COPY_DELAY_MS,
    SIDE_COPY_HORIZ_SPEED,
    ROW_MERGE_HORIZ_SPEED,
    ROW_SEGMENT_SPACING,
    VECTOR_LENGTH_PRISM,
    PRISM_BASE_WIDTH,
    PRISM_BASE_DEPTH,
    PRISM_MAX_HEIGHT,
    PRISM_HEIGHT_SCALE_FACTOR,
    HIDE_INSTANCE_Y_OFFSET,
    ANIM_RISE_SPEED_ORIGINAL,
    ANIM_RISE_SPEED_POST_SPLIT_LN1,
    ANIM_RISE_SPEED_POST_SPLIT_LN2,
    ORIGINAL_TO_PROCESSED_GAP,
    MHSA_RESIDUAL_ADDITION_EXTRA_GAP,
    PRISM_DIMENSIONS_PER_UNIT,
    BRANCH_X
} from '../utils/constants.js';
import { startPrismAdditionAnimation } from '../utils/additionUtils.js';
import { computeCenteredPrismX, getPrismSpacing, PRISM_INSTANCE_WIDTH_SCALE } from '../utils/prismLayout.js';
import { buildMHAVisuals, VectorRouter, PassThroughAnimator, SelfAttentionAnimator } from './mhsa/index.js';
import { updateSciFiMaterialUniforms } from '../utils/sciFiMaterial.js';
import { scaleGlobalEmissiveIntensity } from '../utils/materialUtils.js';
import { getSideCopyEntry } from './mhsa/laneIndex.js';
import { animateVectorMatrixPassThrough as animateVectorMatrixPassThroughExternal } from './mhsa/VectorMatrixPassThrough.js';
import { clearScheduledDelays, resolveSkipDelay, resolveSkipDuration, scheduleAfterDelay } from './mhsa/mhsaTimingUtils.js';
import { appState } from '../state/appState.js';
import {
    HORIZ_PHASE,
    LN2_PHASE,
    isLn2PrimedPhase,
    primeLaneForLn2Fallback
} from '../engine/layers/gpt2LanePhases.js';
import { GPT2_LAYER_VISUAL_TUNING } from '../utils/visualTuningProfiles.js';

const _tmpWorld = new THREE.Vector3();
const _tmpWorld2 = new THREE.Vector3();
const _tmpMatrix = new THREE.Matrix4();
const QKV_TRAIL_OPACITY = TRAIL_OPACITY;
const OUTPUT_PROJ_RETURN_TRAIL_OPACITY = 0.1;
const QKV_FINAL_MATRIX_EMISSIVE_INTENSITY = GPT2_LAYER_VISUAL_TUNING.mhsa.qkvFinalMatrixEmissiveIntensity;
const OUTPUT_PROJ_RETURN_WATCHDOG_MIN_MS = 5000;
const OUTPUT_PROJ_RETURN_WATCHDOG_GRACE_MS = 2000;

function isMhsaDebugEnabled() {
    return typeof window !== 'undefined' && window.__MHSA_DEBUG === true;
}

function logMhsaDebug(...args) {
    if (!isMhsaDebugEnabled()) return;
    console.log(...args);
}

const SOFTENED_MATRIX_UNIFORMS = {
    stripeStrength: 0.0,
    scanlineStrength: 0.0,
    glintStrength: 0.0,
    noiseStrength: 0.0,
    rimIntensity: 0.18,
    depthAccentStrength: 0.06,
    fresnelBoost: 0.12,
    accentMix: 0.7
};

const BASE_MATRIX_SURFACE_TWEAKS = {
    roughnessMin: 0.55,
    metalnessMax: 0.55,
    clearcoatMax: 0.25,
    clearcoatRoughnessMin: 0.55,
    iridescenceMax: 0.18,
    envMapIntensityMax: 0.7
};

const QKV_SURFACE_TWEAKS = {
    roughnessMin: 0.6,
    metalnessMax: 0.5,
    clearcoatMax: 0.2,
    clearcoatRoughnessMin: 0.6,
    iridescenceMax: 0.15,
    envMapIntensityMax: 0.6
};

const softenMatrixSurface = (matrix, tweaks = BASE_MATRIX_SURFACE_TWEAKS) => {
    if (!matrix) return;
    const {
        roughnessMin,
        metalnessMax,
        clearcoatMax,
        clearcoatRoughnessMin,
        iridescenceMax,
        envMapIntensityMax
    } = tweaks || BASE_MATRIX_SURFACE_TWEAKS;
    const mats = [matrix.mesh?.material, matrix.frontCapMesh?.material, matrix.backCapMesh?.material];
    mats.forEach((mat) => {
        if (!mat) return;
        if (typeof mat.roughness === 'number') mat.roughness = Math.max(mat.roughness, roughnessMin);
        if (typeof mat.metalness === 'number') mat.metalness = Math.min(mat.metalness, metalnessMax);
        if (typeof mat.clearcoat === 'number') mat.clearcoat = Math.min(mat.clearcoat, clearcoatMax);
        if (typeof mat.clearcoatRoughness === 'number') {
            mat.clearcoatRoughness = Math.max(mat.clearcoatRoughness, clearcoatRoughnessMin);
        }
        if (typeof mat.iridescence === 'number') mat.iridescence = Math.min(mat.iridescence, iridescenceMax);
        if (typeof mat.envMapIntensity === 'number') {
            mat.envMapIntensity = Math.min(mat.envMapIntensity, envMapIntensityMax);
        }
    });
    updateSciFiMaterialUniforms(matrix.mesh?.material, SOFTENED_MATRIX_UNIFORMS);
    updateSciFiMaterialUniforms(matrix.frontCapMesh?.material, SOFTENED_MATRIX_UNIFORMS);
    updateSciFiMaterialUniforms(matrix.backCapMesh?.material, SOFTENED_MATRIX_UNIFORMS);
};

// Use live binding of GLOBAL_ANIM_SPEED_MULT at each use; do not cache

export class MHSAAnimation {
    /**
     * Global toggle.  Set `MHSAAnimation.ENABLE_SELF_ATTENTION = true` **before**
     * constructing an instance to activate the self-attention sub-animation.
     * The main tower demo enables this by default; other entrypoints may opt out.
     */
    static ENABLE_SELF_ATTENTION = false;

    constructor(parentGroup, branchX, mhsaBaseY, clock, mode = 'temp', opts = {}) {
        // ------------------------------------------------------------------
        // MHSAAnimation flow map (high level):
        // 1) Build head matrices + output projection visuals.
        // 2) Route duplicated vectors to head parking positions (VectorRouter).
        // 3) Pass-through animation inside Q/K/V matrices (VectorMatrixPassThrough).
        // 4) Optional self-attention conveyor that produces weighted sums.
        // 5) Merge weighted sums into a row, project, then add back to residual.
        // ------------------------------------------------------------------
        this.parentGroup = parentGroup;
        this.branchX = branchX;
        this.mhsaBaseY = mhsaBaseY;
        this.clock = clock;
        this.activationSource = opts.activationSource || null;
        this.layerIndex = Number.isFinite(opts.layerIndex) ? opts.layerIndex : null;
        this.vectorPrismCount = Number.isFinite(opts.vectorPrismCount)
            ? Math.max(1, Math.floor(opts.vectorPrismCount))
            : VECTOR_LENGTH_PRISM;
        this._laneCount = Number.isFinite(opts.laneCount)
            ? Math.max(1, Math.floor(opts.laneCount))
            : NUM_VECTOR_LANES;
        this._kvCacheDecodeActive = !!opts.kvCacheDecodeActive;
        this._useBatchedVectorCopies = opts.useBatchedVectorCopies !== false;
        // Batched trails collapse to single segments; keep legacy multi-segment
        // trails by default so paths show right-angle turns.
        this._useBatchedTrails = opts.useBatchedTrails !== undefined
            ? !!opts.useBatchedTrails
            : false;
        this.useBatchedPassThrough = opts.useBatchedPassThrough !== undefined
            ? !!opts.useBatchedPassThrough
            : true;
        this._shareVectorData = opts.shareVectorData !== undefined
            ? !!opts.shareVectorData
            : true;
        this._vectorPool = new Map();
        this._vectorPoolSize = 0;
        this._vectorPoolLimit = Number.isFinite(opts.vectorPoolLimit) ? Math.max(0, Math.floor(opts.vectorPoolLimit)) : 512;
        this._batchedVectorSets = null;
        if (this._useBatchedVectorCopies) {
            const totalCopies = this._laneCount * NUM_HEAD_SETS_LAYER;
            this._batchedVectorSets = {
                K: new BatchedPrismVectorSet({
                    vectorCount: totalCopies,
                    prismCount: this.vectorPrismCount,
                    parentGroup: this.parentGroup,
                    label: 'MHSA K Copies',
                }),
                Q: new BatchedPrismVectorSet({
                    vectorCount: totalCopies,
                    prismCount: this.vectorPrismCount,
                    parentGroup: this.parentGroup,
                    label: 'MHSA Q Copies',
                }),
                V: new BatchedPrismVectorSet({
                    vectorCount: totalCopies,
                    prismCount: this.vectorPrismCount,
                    parentGroup: this.parentGroup,
                    label: 'MHSA V Copies',
                }),
            };
        }
        this._qkvTrailBatch = null;
        this._trailFactory = null;
        if (this._useBatchedTrails) {
            const capacity = this._laneCount * NUM_HEAD_SETS_LAYER * 3;
            this._qkvTrailBatch = new SegmentTrailBatch(
                this.parentGroup,
                capacity,
                TRAIL_COLOR,
                1,
                QKV_TRAIL_OPACITY
            );
            this._trailFactory = () => (this._qkvTrailBatch ? this._qkvTrailBatch.acquireTrail() : null);
        }

        // Speed at which residual-stream vectors rise while branched
        // during MHSA/MLP processing. Starts with the LN1 value.
        this.postSplitRiseSpeed = ANIM_RISE_SPEED_POST_SPLIT_LN1;
        // Allow the pipeline to pause residual rise when a higher-level
        // animation sequence (e.g. final LayerNorm) takes over.
        this.suppressResidualRise = false;

        // Core positional helpers & state flags
        this.mhaPassThroughPhase = 'positioning_mha_vectors';
        // Track the merge-to-row phase so UI can switch equations as soon as
        // vectors start travelling back toward the output projection matrix.
        // Values: 'not_started' | 'merging' | 'merged'
        this.rowMergePhase = 'not_started';

        this.mhsa_matrix_center_y = this.mhsaBaseY + MHA_MATRIX_PARAMS.height / 2;
        this.headStopY            = this.mhsa_matrix_center_y - MHSA_HEAD_VECTOR_STOP_BELOW;
        this.mhaPassThroughTargetY = this.mhsa_matrix_center_y + MHA_MATRIX_PARAMS.height / 2 + 20;

        // Durations & dimensional constants
        this.outputVectorLength      = 64;
        this.mhaResultRiseOffsetY    = MHSA_RESULT_RISE_OFFSET_Y;
        // Note: durations are exposed as getters to reflect runtime speed changes

        // Colours & material defaults
        this.matrixInitialRestingColor     = new THREE.Color(MHSA_MATRIX_INITIAL_RESTING_COLOR);
        this.matrixRestingEmissiveIntensity = 0.0;
        this.matrixRestingOpacity           = 1.0;

        this.brightGreen      = new THREE.Color(MHSA_BRIGHT_GREEN);
        this.darkTintedGreen  = new THREE.Color(MHSA_DARK_TINTED_GREEN);
        this.brightBlue       = new THREE.Color(MHSA_BRIGHT_BLUE);
        this.darkTintedBlue   = new THREE.Color(MHSA_DARK_TINTED_BLUE);
        this.brightRed        = new THREE.Color(MHSA_BRIGHT_RED);
        this.darkTintedRed    = new THREE.Color(MHSA_DARK_TINTED_RED);
        this.finalHeadColors = {
            Q: new THREE.Color(MHA_FINAL_Q_COLOR),
            K: new THREE.Color(MHA_FINAL_K_COLOR),
            V: new THREE.Color(MHA_FINAL_V_COLOR),
        };

        // --------------------------------------------------------------
        //  Build static visuals (matrices, head layout, output projection)
        // --------------------------------------------------------------
        const visuals = buildMHAVisuals(this.parentGroup, {
            branchX: this.branchX,
            mhsaBaseY: this.mhsaBaseY,
            matrixRestingOpacity: 1.0, // retains original behaviour
            layerIndex: this.layerIndex,
        });

        this.mhaVisualizations           = visuals.mhaVisualizations;
        this.headsCentersX               = visuals.headsCentersX;
        this.headCoords                  = visuals.headCoords;
        this.outputProjectionMatrix      = visuals.outputProjectionMatrix;
        this.outputProjMatrixCenterY     = visuals.outputProjMatrixCenterY;
        this.outputProjMatrixHeight      = visuals.outputProjMatrixHeight;
        this.outputProjMatrixDefaultColor = visuals.outputProjMatrixDefaultColor;
        this.outputProjMatrixActiveColor  = visuals.outputProjMatrixActiveColor;
        this.finalCombinedY              = visuals.finalCombinedY;
        this.finalOriginalY              = visuals.finalOriginalY;
        this.outputProjMatrixYOffset = 0;
        if (!this.outputProjMatrixBasePosition && this.outputProjectionMatrix && this.outputProjectionMatrix.group) {
            this.outputProjMatrixBasePosition = new THREE.Vector3(
                this.outputProjectionMatrix.group.position.x,
                this.outputProjMatrixCenterY,
                this.outputProjectionMatrix.group.position.z
            );
        }

        // Additional arrays required by later stages
        this.outputProjMatrixAnimationPhase = 'waiting';
        this.outputProjMatrixVectors        = [];
        this.outputProjMatrixReturnComplete = false;
        this._outputProjReturnCount = 0;
        this._outputProjReturnTargetCount = 0;

        // Mode control (e.g., 'temp', 'perm', etc.)
        this.mode = mode;

        // Flag to indicate a global matrix pulse is active (throttles per-vector updates)
        this._mhaPulseActive = false;
        this._headColorsFinalized = false;
        this._headFinalColorCallbackRegistered = false;
        this._skipMatrixColorsLocked = false;
        this._skipMatrixColorsPending = false;

        // --------------------------------------------------------------
        //  Self-attention toggle (defaults to global static value)
        // --------------------------------------------------------------
        this.enableSelfAttentionAnimation =
            opts.enableSelfAttention ?? MHSAAnimation.ENABLE_SELF_ATTENTION;

        // Self-attention helper (placeholder)
        this.selfAttentionAnimator = new SelfAttentionAnimator(this);
        // Dock offset used when weighted sums park above V vectors.
        this.weightedSumDockOffset = MHA_WEIGHTED_SUM_DOCK_OFFSET;

        // Raise the output projection matrix to the weighted-sum row height immediately
        // so it does not jump later during concatenation.
        try {
            if (!this.enableSelfAttentionAnimation) {
                // Keep legacy placement when the conveyor belt is disabled.
            } else {
            const BASE_RISE_ADJUST = -30;
            const weightedRowY = this.mhaPassThroughTargetY
                + this.mhaResultRiseOffsetY
                + BASE_RISE_ADJUST
                + (this.selfAttentionAnimator ? this.selfAttentionAnimator.RED_EXTRA_RISE : 0)
                + this.weightedSumDockOffset;
            const desiredBottomY = weightedRowY + MHA_OUTPUT_PROJECTION_MATRIX_Y_OFFSET_ABOVE_ROW;
            const desiredCenterY = desiredBottomY + this.outputProjMatrixHeight / 2;
            if (Number.isFinite(desiredCenterY) && Number.isFinite(this.outputProjMatrixCenterY)) {
                this.outputProjMatrixYOffset = desiredCenterY - this.outputProjMatrixCenterY;
                if (Math.abs(this.outputProjMatrixYOffset) > 0.01) {
                    this.outputProjMatrixCenterY = desiredCenterY;
                    if (this.outputProjMatrixBasePosition) {
                        this.outputProjMatrixBasePosition.set(
                            this.outputProjMatrixBasePosition.x,
                            desiredCenterY,
                            this.outputProjMatrixBasePosition.z
                        );
                    }
                    if (this.outputProjectionMatrix && this.outputProjectionMatrix.group) {
                        this.outputProjectionMatrix.group.position.y = desiredCenterY;
                    }
                    const matrixTopY = desiredCenterY + this.outputProjMatrixHeight / 2;
                    this.finalCombinedY = matrixTopY + 60;
                    this.finalOriginalY = this.finalCombinedY - ORIGINAL_TO_PROCESSED_GAP - MHSA_RESIDUAL_ADDITION_EXTRA_GAP;
                }
            }
            }
        } catch (_) { /* optional */ }

        // Cap residual-stream rise to the baseline target computed during MHSA setup.
        // Later stage code may raise `finalOriginalY` while branches run; this ceiling
        // preserves the historical max-rise behavior for the original stream vectors.
        this.maxResidualRiseY = Number.isFinite(this.finalOriginalY)
            ? this.finalOriginalY
            : null;

        // Temp-mode bookkeeping
        this._tempModeCompleted = false;
        this._tempAllOutputVectors = []; // K,Q,V combined
        this._tempKOutputVectors   = []; // Only central K vectors

        // Track per-head merged fixed vectors so we only merge once.
        this._mergedHeads = new Set();
        this._mergedGroupsByHead = new Map(); // headIdx -> { K: Group, V: Group }
        this._allFixedMerged = false; // single merged K and V meshes across all heads created
        this._attentionWeightedSums = new Map(); // key -> { vec, laneZ, headIdx, laneIndex }
        this._laneByZ = new Map();
        this._laneZsSorted = [];
        this._attentionConveyorLanes = [];
        this._laneZKeyPrecision = 10; // 0.1 world-unit resolution
        this._cachedKvEntries = [];
        this._cachedKvBatchesToSync = new Set();

        // Pause-aware scheduling helpers ensure delayed callbacks respect manual pauses.
        this._scheduledDelayTweens = new Set();
        this._scheduledTimeoutIds = new Set();
        this._cancelOutputProjectionWatchdog = null;
        this._skipToEndActive = false;
        this._passThroughJobs = [];
        this._lastUpdateTimeNow = null;
        this._kvCacheVectorsPreserved = false;
        this._batchSyncScratch = new Set();
        this._laneComposeCachedInput = null;
        this._laneComposeCachedLen = -1;
        this._laneComposeCachedKvVersion = -1;
        this._cachedKvEntriesVersion = 0;
        this._outputProjColorScratch = new THREE.Color();
        this.setCachedKvEntries(opts.cachedKvEntries);

        // --------------------------------------------------------------
        //  Vector router: handles horizontal travel + K/Q/V parking.
        //  Once all copies are parked it triggers the parallel pass-through.
        // --------------------------------------------------------------
        this.vectorRouter = new VectorRouter(this.parentGroup, this.headsCentersX, this.headCoords, this.headStopY, this.mhaVisualizations, {
            acquireVector: this._acquireVectorCopy.bind(this),
            shareVectorData: this._shareVectorData,
            trailFactory: this._trailFactory,
            copyTrailOpacity: this._isKvCacheModeEnabled() ? 0.16 : undefined
        });
        this.vectorRouter.onReady(() => {
            this.mhaPassThroughPhase = 'ready_for_parallel_pass_through';
            logMhsaDebug('MHSAAnimation: All MHSA vectors are in position. Ready for PARALLEL pass-through.');
            this.passThroughAnimator = new PassThroughAnimator(this);
            this.passThroughAnimator.start(this.currentLanes);
        });

        // ----------------------------------------------
        //  Stage pipeline scaffolding (legacy)
        // ----------------------------------------------
        // These two helper calls are replaced by the new builder.
        // Keeping methods defined below for now so update() logic referring to
        // them compiles, but we skip execution to avoid duplicate visuals.
        // this._setupMHSAVisualizations();
        // this._setupOutputProjectionMatrix();
    }

    // Dynamic durations that adapt to GLOBAL_ANIM_SPEED_MULT at access time
    get mhaResultRiseDuration() {
        return MHA_RESULT_RISE_DURATION_BASE_MS / GLOBAL_ANIM_SPEED_MULT;
    }
    get mhaPassThroughDuration() {
        return MHSA_PASS_THROUGH_TOTAL_DURATION_MS / GLOBAL_ANIM_SPEED_MULT;
    }

    // ------------------------------------------------------------------
    //  Placeholder self-attention phase – waits 3 s before continuing
    // ------------------------------------------------------------------
    _runSelfAttentionPhase(onDone) {
        if (this.selfAttentionAnimator) {
            this.selfAttentionAnimator.start(onDone);
        } else if (onDone) {
            onDone();
        }
    }

    setCachedKvEntries(entries = null) {
        if (!Array.isArray(entries) || !entries.length) {
            this._cachedKvEntries = [];
            this._cachedKvBatchesToSync.clear();
            this._cachedKvEntriesVersion += 1;
            return;
        }
        this._cachedKvEntries = entries
            .filter((entry) => entry && Number.isFinite(entry.zPos))
            .slice()
            .sort((a, b) => {
                const aLayout = Number.isFinite(a?.laneLayoutIndex) ? a.laneLayoutIndex : Infinity;
                const bLayout = Number.isFinite(b?.laneLayoutIndex) ? b.laneLayoutIndex : Infinity;
                if (aLayout !== bLayout) return aLayout - bLayout;
                return a.zPos - b.zPos;
            });
        this._refreshCachedKvBatchesToSync(this._cachedKvEntries);
        this._snapCachedKeyVectorsUnderValues(this._cachedKvEntries, { cachedOnly: true });
        this._cachedKvEntriesVersion += 1;
    }

    _refreshCachedKvBatchesToSync(entries = null) {
        if (!this._cachedKvBatchesToSync) {
            this._cachedKvBatchesToSync = new Set();
        } else {
            this._cachedKvBatchesToSync.clear();
        }
        if (!Array.isArray(entries) || !entries.length) return;

        const addBatchFromVec = (vec) => {
            const batch = vec && vec.isBatchedVectorRef ? vec._batch : null;
            if (!batch || typeof batch.syncAll !== 'function') return;
            this._cachedKvBatchesToSync.add(batch);
        };

        entries.forEach((entry) => {
            if (!entry) return;
            if (Array.isArray(entry.upwardCopies)) {
                entry.upwardCopies.forEach((vec) => addBatchFromVec(vec));
            }
            if (Array.isArray(entry.sideCopies)) {
                entry.sideCopies.forEach((side) => addBatchFromVec(side?.vec));
            }
        });
    }

    getAttentionConveyorLanes() {
        if (Array.isArray(this._attentionConveyorLanes) && this._attentionConveyorLanes.length) {
            return this._attentionConveyorLanes;
        }
        return Array.isArray(this.currentLanes) ? this.currentLanes : [];
    }

    _composeAttentionConveyorLanes(lanes) {
        const liveLanes = Array.isArray(lanes) ? lanes : [];
        const cached = Array.isArray(this._cachedKvEntries) ? this._cachedKvEntries : [];
        if (!cached.length) return liveLanes;

        const seen = new Set();
        const composed = [];
        const append = (lane) => {
            if (!lane || !Number.isFinite(lane.zPos)) return;
            const tokenKey = Number.isFinite(lane.tokenIndex) ? `t:${lane.tokenIndex}` : null;
            const layoutKey = Number.isFinite(lane.laneLayoutIndex) ? `l:${lane.laneLayoutIndex}` : null;
            const zKey = `z:${lane.zPos.toFixed(4)}`;
            const key = tokenKey || layoutKey || zKey;
            if (seen.has(key)) return;
            seen.add(key);
            composed.push(lane);
        };

        cached.forEach(append);
        liveLanes.forEach(append);
        composed.sort((a, b) => {
            const aLayout = Number.isFinite(a?.laneLayoutIndex) ? a.laneLayoutIndex : Infinity;
            const bLayout = Number.isFinite(b?.laneLayoutIndex) ? b.laneLayoutIndex : Infinity;
            if (aLayout !== bLayout) return aLayout - bLayout;
            return a.zPos - b.zPos;
        });
        return composed;
    }

    _laneKey(zPos) {
        const precision = Number.isFinite(this._laneZKeyPrecision) ? this._laneZKeyPrecision : 10;
        if (!Number.isFinite(zPos)) return null;
        return Math.round(zPos * precision) / precision;
    }

    _rebuildLaneIndex(lanes) {
        if (!this._laneByZ) this._laneByZ = new Map();
        this._laneByZ.clear();
        const zList = [];
        (lanes || []).forEach((lane) => {
            if (!lane || !Number.isFinite(lane.zPos)) return;
            const key = this._laneKey(lane.zPos);
            if (key !== null && !this._laneByZ.has(key)) {
                this._laneByZ.set(key, lane);
            }
            zList.push(lane.zPos);
        });
        zList.sort((a, b) => a - b);
        this._laneZsSorted = zList;
        this.sortedLaneZs = zList;
    }

    _getVectorPoolBucket(instanceCount) {
        const key = Number.isFinite(instanceCount) ? Math.max(1, Math.floor(instanceCount)) : this.vectorPrismCount;
        let bucket = this._vectorPool.get(key);
        if (!bucket) {
            bucket = [];
            this._vectorPool.set(key, bucket);
        }
        return { key, bucket };
    }

    _acquireVectorCopy({ rawData, position, instanceCount, numSubsections = 30, shareData = false, kind = null, headIndex = null, lane = null } = {}) {
        if (this._batchedVectorSets && kind && this._batchedVectorSets[kind]) {
            const laneIndex = (lane && Number.isFinite(lane.laneIndex))
                ? lane.laneIndex
                : (this.currentLanes ? this.currentLanes.indexOf(lane) : -1);
            const headIdx = Number.isFinite(headIndex) ? headIndex : 0;
            const safeLaneIndex = laneIndex >= 0 ? laneIndex : 0;
            const vectorIndex = safeLaneIndex * NUM_HEAD_SETS_LAYER + headIdx;
            const batch = this._batchedVectorSets[kind];
            const vec = batch.getVectorRef(vectorIndex);
            vec.group.visible = true;
            if (position) {
                vec.group.position.copy(position);
            }
            vec.updateDataInternal(rawData, { copyData: true });
            vec.userData = vec.userData || {};
            vec.userData.qkvProcessed = false;
            vec.userData.qkvOutputLength = null;
            vec.userData.qkvProcessedCategory = null;
            vec.userData.vectorCategory = kind;
            delete vec.userData.activationData;
            if (vec.group && vec.group.userData) {
                delete vec.group.userData.activationData;
            }
            return vec;
        }
        const { key, bucket } = this._getVectorPoolBucket(instanceCount);
        let vec = null;
        if (bucket.length) {
            vec = bucket.pop();
            this._vectorPoolSize = Math.max(0, this._vectorPoolSize - 1);
        }
        if (!vec) {
            vec = new VectorVisualizationInstancedPrism(
                rawData,
                position ? position.clone() : new THREE.Vector3(),
                numSubsections,
                key,
                { copyData: !shareData }
            );
        } else {
            if (vec.group) {
                vec.group.visible = true;
                vec.group.scale.set(1, 1, 1);
                if (position) {
                    vec.group.position.copy(position);
                }
            }
            if (typeof vec.updateDataInternal === 'function') {
                vec.updateDataInternal(rawData, { copyData: !shareData });
            }
            if (vec.mesh) {
                vec.mesh.visible = true;
                if (vec.mesh.material) {
                    const mats = Array.isArray(vec.mesh.material) ? vec.mesh.material : [vec.mesh.material];
                    mats.forEach((mat) => this._normalizeVectorMaterialVisible(mat, 1));
                }
            }
        }
        if (vec.mesh && vec.mesh.isMesh) {
            // Instanced prism copies can be pooled and reused across passes.
            // Keep culling disabled so stale instance bounds never cause
            // angle-dependent popping in KV-cache decode runs.
            vec.mesh.frustumCulled = false;
            const mats = Array.isArray(vec.mesh.material) ? vec.mesh.material : [vec.mesh.material];
            mats.forEach((mat) => {
                this._normalizeVectorMaterialVisible(mat, 1);
            });
        }
        vec.userData = vec.userData || {};
        vec.userData.qkvProcessed = false;
        vec.userData.qkvOutputLength = null;
        vec.userData.qkvProcessedCategory = null;
        if (typeof kind === 'string') {
            vec.userData.vectorCategory = kind;
        } else {
            delete vec.userData.vectorCategory;
        }
        delete vec.userData.activationData;
        if (vec.group && vec.group.userData) {
            delete vec.group.userData.activationData;
        }
        if (vec.mesh && vec.mesh.userData) {
            delete vec.mesh.userData.activationData;
        }
        if (this.parentGroup && vec.group && vec.group.parent !== this.parentGroup) {
            this.parentGroup.add(vec.group);
        }
        return vec;
    }

    _releaseVectorCopy(vec) {
        if (!vec) return;
        if (vec.isBatchedVectorRef) {
            if (vec.group) {
                vec.group.visible = false;
            }
            if (vec.userData) {
                delete vec.userData.trail;
                delete vec.userData.trailWorld;
                delete vec.userData.parentLane;
            }
            try {
                vec.rawData = [];
                vec.normalizedData = [];
            } catch (_) { /* ignore */ }
            return;
        }
        try {
            if (vec.group && vec.group.parent) vec.group.parent.remove(vec.group);
        } catch (_) { /* ignore */ }
        if (vec.group) {
            vec.group.visible = false;
        }
        if (vec.mesh) {
            vec.mesh.visible = false;
        }
        if (vec.userData) {
            delete vec.userData.trail;
            delete vec.userData.trailWorld;
            delete vec.userData.parentLane;
        }
        try {
            vec.rawData = [];
            vec.normalizedData = [];
        } catch (_) { /* ignore */ }
        const { key, bucket } = this._getVectorPoolBucket(vec.instanceCount);
        if (this._vectorPoolSize < this._vectorPoolLimit) {
            bucket.push(vec);
            this._vectorPoolSize += 1;
        } else {
            try { if (typeof vec.dispose === 'function') vec.dispose(); } catch (_) { /* ignore */ }
        }
    }

    getLaneForZ(zPos) {
        if (!Number.isFinite(zPos)) return null;
        if (this._laneByZ && this._laneByZ.size) {
            const key = this._laneKey(zPos);
            if (key !== null && this._laneByZ.has(key)) {
                return this._laneByZ.get(key);
            }
        }
        // Fallback for any small drift beyond the map tolerance.
        const lanes = Array.isArray(this._attentionConveyorLanes) && this._attentionConveyorLanes.length
            ? this._attentionConveyorLanes
            : (Array.isArray(this.currentLanes) ? this.currentLanes : null);
        if (!lanes) return null;
        let bestLane = null;
        let bestDist = Infinity;
        for (let i = 0; i < lanes.length; i++) {
            const lane = lanes[i];
            if (!lane || !Number.isFinite(lane.zPos)) continue;
            const dist = Math.abs(lane.zPos - zPos);
            if (dist < bestDist) {
                bestDist = dist;
                bestLane = lane;
            }
        }
        return bestDist <= 0.25 ? bestLane : null;
    }

    _setupMHSAVisualizations() {
        const darkGrayColor = new THREE.Color(0x404040);
        const matrixOpacity = this.matrixRestingOpacity;
        const matrixCenterY = this.mhsaBaseY + MHA_MATRIX_PARAMS.height / 2;

        for (let i = 0; i < NUM_HEAD_SETS_LAYER; i++) {
            const headSetWidth = MHA_INTERNAL_MATRIX_SPACING * 2 + MHA_MATRIX_PARAMS.width;
            const currentHeadSetBaseX = this.branchX - MHA_INTERNAL_MATRIX_SPACING + i * (headSetWidth + HEAD_SET_GAP_LAYER);

            const x_q = currentHeadSetBaseX;
            const x_k = currentHeadSetBaseX + MHA_INTERNAL_MATRIX_SPACING;
            const x_v = currentHeadSetBaseX + MHA_INTERNAL_MATRIX_SPACING * 2;

            const queryMatrix = new WeightMatrixVisualization(
                null, new THREE.Vector3(x_q, matrixCenterY, 0),
                MHA_MATRIX_PARAMS.width, MHA_MATRIX_PARAMS.height, MHA_MATRIX_PARAMS.depth,
                MHA_MATRIX_PARAMS.topWidthFactor, MHA_MATRIX_PARAMS.cornerRadius, MHA_MATRIX_PARAMS.numberOfSlits,
                MHA_MATRIX_PARAMS.slitWidth, MHA_MATRIX_PARAMS.slitDepthFactor,
                MHA_MATRIX_PARAMS.slitBottomWidthFactor, MHA_MATRIX_PARAMS.slitTopWidthFactor
            );
            queryMatrix.setColor(darkGrayColor);
            {
                const lbl = 'Query Weight Matrix';
                queryMatrix.group.userData.label = lbl;
                if (queryMatrix.mesh) queryMatrix.mesh.userData.label = lbl;
                if (queryMatrix.frontCapMesh) queryMatrix.frontCapMesh.userData.label = lbl;
                if (queryMatrix.backCapMesh)  queryMatrix.backCapMesh.userData.label  = lbl;
            }
            queryMatrix.group.children.forEach(child => {
                if (child.material) {
                    child.material.transparent = matrixOpacity < 1.0;
                    child.material.opacity = matrixOpacity;
                }
            });
            this.parentGroup.add(queryMatrix.group);
            this.mhaVisualizations.push(queryMatrix);
            softenMatrixSurface(queryMatrix, QKV_SURFACE_TWEAKS);

            const keyMatrix = new WeightMatrixVisualization(
                null, new THREE.Vector3(x_k, matrixCenterY, 0),
                MHA_MATRIX_PARAMS.width, MHA_MATRIX_PARAMS.height, MHA_MATRIX_PARAMS.depth,
                MHA_MATRIX_PARAMS.topWidthFactor, MHA_MATRIX_PARAMS.cornerRadius, MHA_MATRIX_PARAMS.numberOfSlits,
                MHA_MATRIX_PARAMS.slitWidth, MHA_MATRIX_PARAMS.slitDepthFactor,
                MHA_MATRIX_PARAMS.slitBottomWidthFactor, MHA_MATRIX_PARAMS.slitTopWidthFactor
            );
            keyMatrix.setColor(darkGrayColor);
            {
                const lbl = 'Key Weight Matrix';
                keyMatrix.group.userData.label = lbl;
                if (keyMatrix.mesh) keyMatrix.mesh.userData.label = lbl;
                if (keyMatrix.frontCapMesh) keyMatrix.frontCapMesh.userData.label = lbl;
                if (keyMatrix.backCapMesh)  keyMatrix.backCapMesh.userData.label  = lbl;
            }
            keyMatrix.group.children.forEach(child => {
                if (child.material) {
                    child.material.transparent = matrixOpacity < 1.0;
                    child.material.opacity = matrixOpacity;
                }
            });
            this.parentGroup.add(keyMatrix.group);
            this.mhaVisualizations.push(keyMatrix);
            softenMatrixSurface(keyMatrix, QKV_SURFACE_TWEAKS);

            const valueMatrix = new WeightMatrixVisualization(
                null, new THREE.Vector3(x_v, matrixCenterY, 0),
                MHA_MATRIX_PARAMS.width, MHA_MATRIX_PARAMS.height, MHA_MATRIX_PARAMS.depth,
                MHA_MATRIX_PARAMS.topWidthFactor, MHA_MATRIX_PARAMS.cornerRadius, MHA_MATRIX_PARAMS.numberOfSlits,
                MHA_MATRIX_PARAMS.slitWidth, MHA_MATRIX_PARAMS.slitDepthFactor,
                MHA_MATRIX_PARAMS.slitBottomWidthFactor, MHA_MATRIX_PARAMS.slitTopWidthFactor
            );
            valueMatrix.setColor(darkGrayColor);
            {
                const lbl = 'Value Weight Matrix';
                valueMatrix.group.userData.label = lbl;
                if (valueMatrix.mesh) valueMatrix.mesh.userData.label = lbl;
                if (valueMatrix.frontCapMesh) valueMatrix.frontCapMesh.userData.label = lbl;
                if (valueMatrix.backCapMesh)  valueMatrix.backCapMesh.userData.label  = lbl;
            }
            valueMatrix.group.children.forEach(child => {
                if (child.material) {
                    child.material.transparent = matrixOpacity < 1.0;
                    child.material.opacity = matrixOpacity;
                }
            });
            this.parentGroup.add(valueMatrix.group);
            this.mhaVisualizations.push(valueMatrix);
            softenMatrixSurface(valueMatrix, QKV_SURFACE_TWEAKS);

            this.headsCentersX.push(x_k);
            this.headCoords.push({ q: x_q, k: x_k, v: x_v });
        }
    }

    _setupOutputProjectionMatrix() {
        // Positioned above the merged row, aligned with the first head's K matrix X-coordinate
        const firstHeadKMatrixX = this.headCoords.length > 0 ? this.headCoords[0].k : this.branchX; // Fallback to branchX if no heads

        // Y position calculation:
        // Base Y of K vectors after passing through heads and initial rise:
        const postPassThroughBaseY = this.mhaPassThroughTargetY + this.mhaResultRiseOffsetY;
        // Y of the decorative vectors that form the merged row:
        const decorativeVectorsY = postPassThroughBaseY + 60; // 60 is the verticalOffset from _applyTempModeBehaviour
        // Center Y for the new output projection matrix:
        const matrixHeight = MHA_MATRIX_PARAMS.height * MHA_OUTPUT_PROJECTION_MATRIX_PARAMS.heightFactor;
        const outputProjMatrixCenterY = decorativeVectorsY + MHA_OUTPUT_PROJECTION_MATRIX_Y_OFFSET_ABOVE_ROW + matrixHeight / 2;

        // Use same depth as other matrices for consistency
        const inputDepth = MHA_MATRIX_PARAMS.depth;

        this.outputProjectionMatrix = new WeightMatrixVisualization(
            null, // No specific data array needed for this visualization
            new THREE.Vector3(firstHeadKMatrixX, outputProjMatrixCenterY, 0),
            MHA_OUTPUT_PROJECTION_MATRIX_PARAMS.width,
            matrixHeight,
            inputDepth,
            MHA_OUTPUT_PROJECTION_MATRIX_PARAMS.topWidthFactor,
            MHA_OUTPUT_PROJECTION_MATRIX_PARAMS.cornerRadius,
            NUM_VECTOR_LANES,
            MHA_OUTPUT_PROJECTION_MATRIX_PARAMS.slitWidth,
            MHA_OUTPUT_PROJECTION_MATRIX_PARAMS.slitDepthFactor,
            MHA_OUTPUT_PROJECTION_MATRIX_PARAMS.slitBottomWidthFactor,
            MHA_OUTPUT_PROJECTION_MATRIX_PARAMS.slitTopWidthFactor
        );

        // Initialise pitch-black; will brighten once vectors pass through
        const initDarkColor = new THREE.Color(MHSA_MATRIX_INITIAL_RESTING_COLOR);
        this.outputProjectionMatrix.setColor(initDarkColor);
        {
            const lbl = 'Output Projection Matrix';
            this.outputProjectionMatrix.group.userData.label = lbl;
            if (this.outputProjectionMatrix.mesh) this.outputProjectionMatrix.mesh.userData.label = lbl;
            if (this.outputProjectionMatrix.frontCapMesh) this.outputProjectionMatrix.frontCapMesh.userData.label = lbl;
            if (this.outputProjectionMatrix.backCapMesh)  this.outputProjectionMatrix.backCapMesh.userData.label  = lbl;
        }
        this.outputProjectionMatrix.group.children.forEach(child => {
            if (child.material) {
                // Ensure the output projection matrix starts fully opaque
                child.material.transparent = false;
                child.material.opacity = 1.0;
                child.material.emissive = initDarkColor;
                child.material.emissiveIntensity = scaleGlobalEmissiveIntensity(0.1); // Low initial emissive intensity
            }
        });
        this.parentGroup.add(this.outputProjectionMatrix.group);
        // Keep output projection reflectivity aligned with QKV matrices.
        softenMatrixSurface(this.outputProjectionMatrix, QKV_SURFACE_TWEAKS);
        
        // Store the matrix's Y position for later animations
        this.outputProjMatrixCenterY = outputProjMatrixCenterY;
        this.outputProjMatrixBasePosition = new THREE.Vector3(firstHeadKMatrixX, outputProjMatrixCenterY, 0);
        this.outputProjMatrixHeight = matrixHeight;
        
        // Store default and target colors for animation
        this.outputProjMatrixDefaultColor = initDarkColor;
        this.outputProjMatrixActiveColor = new THREE.Color(MHA_OUTPUT_PROJECTION_MATRIX_COLOR);
        
        // Animation state
        this.outputProjMatrixAnimationPhase = 'waiting'; // 'waiting', 'vectors_entering', 'vectors_inside', 'completed'
        this.outputProjMatrixVectors = [];
        
        // Log the matrix dimensions to confirm they match desired specifications
        logMhsaDebug(`MHSAAnimation: Output Projection Matrix added - Width: ${MHA_OUTPUT_PROJECTION_MATRIX_PARAMS.width}, Height: ${matrixHeight}, Depth: ${inputDepth}`);

        // ------------------------------------------------------------------
        //  Pre-compute the Y coordinate where the processed vectors will finish
        //  ("finalCombinedY"), and where the original residual-stream vectors
        //  should end up ("finalOriginalY").  These values let us start moving
        //  the original vectors immediately rather than waiting until the
        //  Output-Projection phase kicks in.
        // ------------------------------------------------------------------

        // The processed vectors rise to:  matrixTopY + 60.
        const matrixTopY = outputProjMatrixCenterY + matrixHeight / 2;
        this.finalCombinedY = matrixTopY + 60; // 30 (rise to matrix) + 30 (extraRise)
        this.finalOriginalY = this.finalCombinedY - ORIGINAL_TO_PROCESSED_GAP - MHSA_RESIDUAL_ADDITION_EXTRA_GAP;
    }

    areAllMHAVectorsInPosition(lanes) {
        if (!lanes || !lanes.length) return false;

        for (const lane of lanes) {
            if (!lane.upwardCopies || lane.upwardCopies.length !== NUM_HEAD_SETS_LAYER) {
                return false; 
            }

            for (let headIdx = 0; headIdx < NUM_HEAD_SETS_LAYER; headIdx++) {
                const kVec = lane.upwardCopies[headIdx];
                if (!kVec || Math.abs(kVec.group.position.y - this.headStopY) > 0.1) {
                    return false; 
                }
                if (!kVec.userData.sideSpawned) {
                    return false; 
                }
            }

            if (!lane.sideCopies || lane.sideCopies.length !== NUM_HEAD_SETS_LAYER * 2) {
                return false;
            }

            for (const sideCopyObj of lane.sideCopies) {
                if (!sideCopyObj || !sideCopyObj.vec) {
                    return false;
                }
                if (Math.abs(sideCopyObj.vec.group.position.y - this.headStopY) > 0.1) {
                    return false; 
                }
                if (Math.abs(sideCopyObj.vec.group.position.x - sideCopyObj.targetX) > 0.1) {
                    return false; 
                }
            }
        }
        return true;
    }

    animateVectorMatrixPassThrough(vector, matrix, brightMatrixColor, darkTintedMatrixColor, finalVectorColor, passThroughY, duration, riseOffset, riseDurationVal, outLength, animationCompletionCallback, vectorCategory = 'K') {
        // Thin wrapper delegating to extracted helper for maintainability.
        return animateVectorMatrixPassThroughExternal(
            this,
            vector,
            matrix,
            brightMatrixColor,
            darkTintedMatrixColor,
            finalVectorColor,
            passThroughY,
            duration,
            riseOffset,
            riseDurationVal,
            outLength,
            animationCompletionCallback,
            vectorCategory
        );
    }

    // Launch pass-through tweens for every head's K/Q/V copy in parallel.
    initiateParallelHeadPassThroughAnimations(allLanes) {
        if (this.mhaPassThroughPhase !== 'ready_for_parallel_pass_through') return;
        logMhsaDebug('MHSAAnimation: Initiating Parallel MHSA Head Pass-Through Animations...');
        this.mhaPassThroughPhase = 'parallel_pass_through_active';

        if (this.enableSelfAttentionAnimation && this.selfAttentionAnimator && !this._headFinalColorCallbackRegistered) {
            this._headFinalColorCallbackRegistered = true;
            this.selfAttentionAnimator.start(() => {
                this._transitionHeadColorsToFinal(HEAD_COLOR_TRANSITION_MS);
            });
        }

        // Start one shared pulse per matrix instead of per-vector material updates
        try { this._startMatrixPulseDuringPassThrough(this.mhaPassThroughDuration); } catch (_) { /* optional */ }

        // Before launching pass-through tweens, merge parked K/Q/V copy trails
        // into a single static LineSegments to reduce draw calls and skip their
        // per-frame updates while vectors move through matrices.
        try { this._mergeCopyTrailsBeforePassThrough(); } catch (_) { /* optional */ }

        let totalAnimationsToComplete = allLanes.length * NUM_HEAD_SETS_LAYER * 3;
        let animationsCompleted = 0;

        const singleAnimationDone = () => {
            animationsCompleted++;
            if (animationsCompleted >= totalAnimationsToComplete) {
                logMhsaDebug('MHSAAnimation: All MHSA parallel pass-through animations complete.');
                this.mhaPassThroughPhase = 'mha_pass_through_complete';

                if (this._skipMatrixColorsLocked && this._skipMatrixColorsPending) {
                    this._applyFinalMatrixColorsImmediate();
                    this._skipMatrixColorsPending = false;
                }

                // In KV-cache decode passes (non-first token), skip the full
                // self-attention conveyor and jump directly into concat/output.
                if (this._kvCacheDecodeActive) {
                    this.skipSelfAttentionAndStartConcat();
                    return;
                }

                // ---------------------------------------------------------------
                //  TEMP MODE: post pass-through behaviour
                // ---------------------------------------------------------------
                if (this.mode === 'temp' && !this._tempModeCompleted) {
                    this._applyTempModeBehaviour();
                    this._tempModeCompleted = true;
                } else if (this.mode !== 'temp' && !this.enableSelfAttentionAnimation) {
                    // For perm mode, trigger final color transition here
                    this._transitionHeadColorsToFinal(HEAD_COLOR_TRANSITION_MS); // 1 second duration
                }

                // After pass-through completes for all vectors, try merging all fixed K/V across heads
                try { this._mergeAllFixedKVIfReady(); } catch (_) { /* optional */ }

                // Schedule disposal of merged K/V groups only AFTER the very last
                // blue (Q) vector completes its conveyor-belt traversal.
                try { this._waitAndDisposeMergedKVWhenBlueConveyorComplete(); } catch (_) { /* optional */ }
            }
        };

        allLanes.forEach((lane) => {
            for (let headIdx = 0; headIdx < NUM_HEAD_SETS_LAYER; headIdx++) {
                const kVec = lane.upwardCopies[headIdx];
                const kMatrix = this.mhaVisualizations[headIdx * 3 + 1];
                this.animateVectorMatrixPassThrough(kVec, kMatrix, this.brightGreen, this.darkTintedGreen, this.finalHeadColors.K, this.mhaPassThroughTargetY, this.mhaPassThroughDuration, this.mhaResultRiseOffsetY, this.mhaResultRiseDuration, this.outputVectorLength, singleAnimationDone, 'K');

                const qSideCopy = getSideCopyEntry(lane, headIdx, 'Q');
                if (qSideCopy && qSideCopy.vec) {
                    this.animateVectorMatrixPassThrough(qSideCopy.vec, qSideCopy.matrixRef, this.brightBlue, this.darkTintedBlue, this.finalHeadColors.Q, this.mhaPassThroughTargetY, this.mhaPassThroughDuration, this.mhaResultRiseOffsetY, this.mhaResultRiseDuration, this.outputVectorLength, singleAnimationDone, 'Q');
                } else { totalAnimationsToComplete--; }

                const vSideCopy = getSideCopyEntry(lane, headIdx, 'V');
                if (vSideCopy && vSideCopy.vec) {
                    this.animateVectorMatrixPassThrough(vSideCopy.vec, vSideCopy.matrixRef, this.brightRed, this.darkTintedRed, this.finalHeadColors.V, this.mhaPassThroughTargetY, this.mhaPassThroughDuration, this.mhaResultRiseOffsetY, this.mhaResultRiseDuration, this.outputVectorLength, singleAnimationDone, 'V');
                } else { totalAnimationsToComplete--; }
            }
        });
        if (totalAnimationsToComplete === 0 && allLanes.length > 0) {
             logMhsaDebug('MHSAAnimation: No valid K,Q,V vectors found to animate for parallel pass-through.');
             this.mhaPassThroughPhase = 'mha_pass_through_complete';
             if (this._skipMatrixColorsLocked && this._skipMatrixColorsPending) {
                 this._applyFinalMatrixColorsImmediate();
                 this._skipMatrixColorsPending = false;
             }
             if (this._kvCacheDecodeActive) {
                 this.skipSelfAttentionAndStartConcat();
             }
        }
    }

    // ------------------------------------------------------------------
    //  One shared pulse per Q/K/V matrix during pass-through (perf)
    // ------------------------------------------------------------------
    _startMatrixPulseDuringPassThrough(totalDurationMs) {
        if (typeof TWEEN === 'undefined') return;
        if (this._skipMatrixColorsLocked) {
            return;
        }
        this._mhaPulseActive = true;
        this._headColorsFinalized = !this.enableSelfAttentionAnimation;

        const restingColor = this.matrixInitialRestingColor;
        const restIntensity = this.matrixRestingEmissiveIntensity;

        const makePulse = (matrix, brightCol, finalCol) => {
            if (!matrix || !matrix.mesh || !matrix.mesh.material) return null;
            const state = { p: 0 };
            const currentColor = new THREE.Color();
            return new TWEEN.Tween(state)
                .to({ p: 1 }, totalDurationMs)
                .easing(TWEEN.Easing.Quadratic.InOut)
                .onUpdate(() => {
                    const t = THREE.MathUtils.smoothstep(state.p, 0, 1);
                    const pulse = Math.sin(Math.PI * t);
                    if (t < 0.5) {
                        currentColor.copy(restingColor).lerp(brightCol, t / 0.5);
                    } else {
                        currentColor.copy(brightCol).lerp(finalCol, (t - 0.5) / 0.5);
                    }
                    const currentEmissive = THREE.MathUtils.lerp(restIntensity, MHSA_MATRIX_MAX_EMISSIVE_INTENSITY, pulse);
                    matrix.setColor(currentColor);
                    matrix.setEmissive(currentColor, currentEmissive);
                })
                .onComplete(() => {
                    matrix.setColor(finalCol);
                    matrix.setEmissive(finalCol, restIntensity);
                })
                .start();
        };

        let pulsesStarted = 0;
        const totalMatrices = NUM_HEAD_SETS_LAYER * 3;
        const useDarkFinal = this.enableSelfAttentionAnimation;
        const qFinal = useDarkFinal ? this.darkTintedBlue : this.finalHeadColors.Q;
        const kFinal = useDarkFinal ? this.darkTintedGreen : this.finalHeadColors.K;
        const vFinal = useDarkFinal ? this.darkTintedRed : this.finalHeadColors.V;
        for (let i = 0; i < NUM_HEAD_SETS_LAYER; i++) {
            const qMatrix = this.mhaVisualizations[i * 3];
            const kMatrix = this.mhaVisualizations[i * 3 + 1];
            const vMatrix = this.mhaVisualizations[i * 3 + 2];
            if (makePulse(qMatrix, this.brightBlue, qFinal)) pulsesStarted++;
            if (makePulse(kMatrix, this.brightGreen, kFinal)) pulsesStarted++;
            if (makePulse(vMatrix, this.brightRed, vFinal)) pulsesStarted++;
        }

        // Safely clear the pulse flag after the pulses end
        this._scheduleAfterDelay(() => { this._mhaPulseActive = false; }, totalDurationMs + 50);
        if (pulsesStarted !== totalMatrices) {
            // If some matrices missing, still clear after duration to avoid stuck flag
            this._scheduleAfterDelay(() => { this._mhaPulseActive = false; }, totalDurationMs + 60);
        }
    }
    
    update(deltaTime, timeNow, lanes) {
        // Per-frame responsibilities: route vectors, update trails, and keep
        // residual vectors rising while the MHSA branch is active.
        // Increment a lightweight frame counter to de-duplicate residual trail writes
        if (typeof this._frameCounter !== 'number') this._frameCounter = 0;
        this._frameCounter++;
        this._lastUpdateTimeNow = timeNow;
        // Keep a reference to the latest lanes array so that other internal
        // methods (triggered asynchronously) can access the original vectors.
        this.currentLanes = lanes;
        const laneCount = Array.isArray(lanes) ? lanes.length : 0;
        const composeNeedsRefresh = lanes !== this._laneComposeCachedInput
            || laneCount !== this._laneComposeCachedLen
            || this._laneComposeCachedKvVersion !== this._cachedKvEntriesVersion;
        if (composeNeedsRefresh) {
            this._attentionConveyorLanes = this._composeAttentionConveyorLanes(lanes);
            this._rebuildLaneIndex(this._attentionConveyorLanes);
            this._laneComposeCachedInput = lanes;
            this._laneComposeCachedLen = laneCount;
            this._laneComposeCachedKvVersion = this._cachedKvEntriesVersion;
        }
        if ((this._skipToEndActive || this._kvCacheDecodeActive) && this._cachedKvEntries.length) {
            this._snapCachedKeyVectorsUnderValues(this._cachedKvEntries, { cachedOnly: true });
        }

        if (this._skipMatrixColorsLocked && this._skipMatrixColorsPending
            && this.mhaPassThroughPhase === 'mha_pass_through_complete') {
            this._applyFinalMatrixColorsImmediate();
            this._skipMatrixColorsPending = false;
        }

        // ---------------- Vector routing (refactored) ----------------
        if (this.vectorRouter) {
            this.vectorRouter.update(deltaTime, timeNow, lanes);
        }

        if (this._passThroughJobs && this._passThroughJobs.length) {
            this._updatePassThroughJobs(timeNow);
        }

        // Output-projection trails are updated by their tween callbacks; avoid
        // duplicate per-frame writes here to reduce shimmer/brightness artifacts.

        // ---------------- End VectorRouter section -------------------
        /* Legacy inline routing logic has been moved to VectorRouter.js
           and will be removed once the migration is complete. */

        // ------------------------------------------------------------------
        //  CONTINUOUSLY MOVE ORIGINAL RESIDUAL-STREAM VECTORS UPWARDS
        // ------------------------------------------------------------------
        if (this.finalOriginalY !== undefined && !this.suppressResidualRise) {
            const riseStep = this.postSplitRiseSpeed * GLOBAL_ANIM_SPEED_MULT * deltaTime;
            for (let laneIndex = 0; laneIndex < lanes.length; laneIndex++) {
                const lane = lanes[laneIndex];
                if (!lane || !lane.originalVec || !lane.originalVec.group) continue;
                if (lane.horizPhase === HORIZ_PHASE.WAITING) continue;

                const curY = lane.originalVec.group.position.y;
                let targetY = this.finalOriginalY;
                // Hard clamp: once the pipeline signals the top embedding stop height,
                // never allow the residual vectors to rise past that entrance.
                if (typeof this.topEmbeddingStopY === 'number') {
                    targetY = Math.min(targetY, this.topEmbeddingStopY);
                }
                if (Number.isFinite(this.maxResidualRiseY)) {
                    targetY = Math.min(targetY, this.maxResidualRiseY);
                }
                let shouldMove = true;

                // Enforce suspension during addition animation
                if (lane.stopRise) {
                    if (lane.stopRiseTarget) {
                        // Allow drifting up to just below the target vector
                        targetY = Math.min(targetY, lane.stopRiseTarget.position.y - ORIGINAL_TO_PROCESSED_GAP);
                    } else {
                        // If no target, freeze completely
                        shouldMove = false;
                    }
                }
                
                // Only allow upward movement toward the target/clamp; never force downward snaps
                if (shouldMove && curY < targetY) {
                    lane.originalVec.group.position.y = Math.min(curY + riseStep, targetY);
                }

                // Update trail after movement – respect world-space trails.
                // During residual addition we update the trail using the centre prism's
                // world position instead (handled just below), so skip the group-centre
                // update while lane.stopRise is active to avoid double-writing.
                if (!lane.stopRise) {
                    try {
                        // Prefer attached trail; fall back to lane.originalTrail if not attached yet
                        const attached = lane.originalVec && lane.originalVec.userData && lane.originalVec.userData.trail;
                        const residualTrail = attached || lane.originalTrail;
                        if (residualTrail && typeof residualTrail.update === 'function') {
                            const wp = _tmpWorld;
                            lane.originalVec.group.getWorldPosition(wp);
                            // Monotonic Y clamp: only extend the trail upwards
                            if (typeof lane.__residualMaxY !== 'number') lane.__residualMaxY = wp.y;
                            if (wp.y >= lane.__residualMaxY) {
                                if (lane.__lastResidualTrailFrame !== this._frameCounter) {
                                    residualTrail.update(wp);
                                    lane.__lastResidualTrailFrame = this._frameCounter;
                                }
                                lane.__residualMaxY = wp.y;
                            }
                        }
                    } catch (_) { /* defensive */ }
                }
            }
        }


        // Residual trail tracking during addition: follow the centre prism in world space.
        // Skip when a higher-level animation (e.g. top LayerNorm) has taken over
        // residual control to avoid double-updating the same trail.
        if (!this.suppressResidualRise) {
            for (let laneIndex = 0; laneIndex < lanes.length; laneIndex++) {
                const lane = lanes[laneIndex];
                if (!lane || !lane.originalVec) continue;

                // Only follow while the addition animation is active (stopRise flag present)
                if (!lane.stopRise) continue;
                // Some additions drive residual trail samples directly from
                // prism motion; skip this fallback sampler to avoid double writes.
                if (lane.__additionOwnsResidualTrail) continue;

                try {
                    // Compute world position of the centre prism for the ORIGINAL (source) vector
                    // so the residual trail extends continuously as prisms rise during addition.
                    const centreIdx = Math.floor(
                        ((lane.originalVec && lane.originalVec.instanceCount) ? lane.originalVec.instanceCount : this.vectorPrismCount) / 2
                    );
                    const instMat = _tmpMatrix;
                    lane.originalVec.mesh.getMatrixAt(centreIdx, instMat);
                    const wPos = _tmpWorld2;
                    wPos.setFromMatrixPosition(instMat);
                    wPos.applyMatrix4(lane.originalVec.group.matrixWorld);

                    // Skip bogus updates when the centre prism is hidden far below the scene
                    // during/addition (it is moved to HIDE_INSTANCE_Y_OFFSET to disappear).
                    const hideThreshold = HIDE_INSTANCE_Y_OFFSET / 10; // e.g. -5000 for -50000 offset
                    if (wPos.y < hideThreshold) continue;

                    // Update trail all the way to the top vector so there is no visible gap.
                    // Prefer the dedicated world-space residual trail reference carried by the lane.
                    const residualTrail = (lane.originalTrail)
                        || (lane.originalVec && lane.originalVec.userData && lane.originalVec.userData.trail)
                        || (lane.postAdditionVec && lane.postAdditionVec.userData && lane.postAdditionVec.userData.trail);
                    if (residualTrail && typeof residualTrail.update === 'function') {
                        // Guard against accidental double-write in the same frame and enforce monotonic Y extension
                        // Allow immediate extension from the centre prism upward when addition begins.
                        if (typeof lane.__residualMaxY !== 'number') lane.__residualMaxY = wPos.y - 0.001;
                        if (wPos.y >= lane.__residualMaxY) {
                            const anchor = lane.__residualTrailAnchor;
                            if (anchor) {
                                if (Number.isFinite(anchor.x)) wPos.x = anchor.x;
                                if (Number.isFinite(anchor.z)) wPos.z = anchor.z;
                            }
                            if (lane.__lastResidualTrailFrame !== this._frameCounter) {
                                residualTrail.update(wPos);
                                lane.__lastResidualTrailFrame = this._frameCounter;
                            }
                            lane.__residualMaxY = wPos.y;
                        }
                    }
                } catch (_) { /* defensive */ }
            }
        }

        const batchesToSync = this._batchSyncScratch;
        batchesToSync.clear();
        if (this._batchedVectorSets) {
            if (this._batchedVectorSets.K && typeof this._batchedVectorSets.K.syncAll === 'function') {
                batchesToSync.add(this._batchedVectorSets.K);
            }
            if (this._batchedVectorSets.Q && typeof this._batchedVectorSets.Q.syncAll === 'function') {
                batchesToSync.add(this._batchedVectorSets.Q);
            }
            if (this._batchedVectorSets.V && typeof this._batchedVectorSets.V.syncAll === 'function') {
                batchesToSync.add(this._batchedVectorSets.V);
            }
        }
        if (this._cachedKvBatchesToSync && this._cachedKvBatchesToSync.size) {
            for (const batch of this._cachedKvBatchesToSync) {
                if (batch && typeof batch.syncAll === 'function') {
                    batchesToSync.add(batch);
                }
            }
        }
        for (const batch of batchesToSync) {
            try {
                batch.syncAll();
            } catch (_) { /* optional batch sync */ }
        }
        if (batchesToSync.size === 0 && this._cachedKvEntries.length) {
            // Cached entries can be re-bound after construction (during decode);
            // refresh lazily so newly attached batched refs also stay in sync.
            this._refreshCachedKvBatchesToSync(this._cachedKvEntries);
        }
    }

    _registerPassThroughJob(job) {
        if (!job) return;
        this._passThroughJobs.push(job);
    }

    _updatePassThroughJobs(timeNow) {
        const jobs = this._passThroughJobs;
        if (!jobs || jobs.length === 0) return;
        let write = 0;
        for (let i = 0; i < jobs.length; i++) {
            const job = jobs[i];
            if (!job || typeof job.update !== 'function') continue;
            const done = job.update(timeNow);
            if (!done) {
                jobs[write++] = job;
            }
        }
        jobs.length = write;
    }

    setSkipToEndMode(enabled = false) {
        this._skipToEndActive = !!enabled;
        if (this.vectorRouter && typeof this.vectorRouter.setSkipToEndMode === 'function') {
            this.vectorRouter.setSkipToEndMode(this._skipToEndActive);
        }
        if (this._skipToEndActive) {
            this._skipMatrixColorsLocked = true;
            this._skipMatrixColorsPending = true;
            this._headColorsFinalized = true;
            if (this.mhaPassThroughPhase === 'mha_pass_through_complete') {
                this._applyFinalMatrixColorsImmediate();
                this._skipMatrixColorsPending = false;
            }
        } else {
            this._skipMatrixColorsLocked = false;
            this._skipMatrixColorsPending = false;
        }
    }

    _applyFinalMatrixColorsImmediate() {
        const qColor = this.finalHeadColors.Q;
        const kColor = this.finalHeadColors.K;
        const vColor = this.finalHeadColors.V;
        for (let i = 0; i < NUM_HEAD_SETS_LAYER; i++) {
            const qMatrix = this.mhaVisualizations[i * 3];
            const kMatrix = this.mhaVisualizations[i * 3 + 1];
            const vMatrix = this.mhaVisualizations[i * 3 + 2];
            const apply = (matrix, color) => {
                if (!matrix) return;
                matrix.setColor(color);
                matrix.setEmissive(color, QKV_FINAL_MATRIX_EMISSIVE_INTENSITY);
                matrix.setMaterialProperties({ opacity: 1.0, transparent: false });
            };
            apply(qMatrix, qColor);
            apply(kMatrix, kColor);
            apply(vMatrix, vColor);
        }
    }

    // ------------------------------------------------------------------
    //  Merge faint trails drawn by the parked K/Q/V copies (pre pass-through)
    // ------------------------------------------------------------------
    _mergeCopyTrailsBeforePassThrough() {
        if (this._isKvCacheModeEnabled()) return;
        const lanes = this.currentLanes || [];
        if (!lanes.length) return;

        /** @type {import('../utils/trailUtils.js').StraightLineTrail[]} */
        const trailsToMerge = [];
        const vectorsWithTrails = [];

        lanes.forEach(lane => {
            // Upward K copies
            if (Array.isArray(lane.upwardCopies)) {
                lane.upwardCopies.forEach(v => {
                    const t = v && v.userData && v.userData.trail;
                    if (t && !t.isBatchedTrail && typeof t.toSegmentsFloat32 === 'function') {
                        trailsToMerge.push(t);
                        vectorsWithTrails.push(v);
                    }
                });
            }
            // Side Q/V copies
            if (Array.isArray(lane.sideCopies)) {
                lane.sideCopies.forEach(obj => {
                    const v = obj && obj.vec;
                    const t = v && v.userData && v.userData.trail;
                    if (t && !t.isBatchedTrail && typeof t.toSegmentsFloat32 === 'function') {
                        trailsToMerge.push(t);
                        vectorsWithTrails.push(v);
                    }
                });
            }
            // Travelling vector trail inside MHSA branch (not the residual stream)
            const tv = lane && lane.travellingVec;
            const tvTrail = tv && tv.userData && tv.userData.trail;
            if (tvTrail && !tvTrail.isBatchedTrail && typeof tvTrail.toSegmentsFloat32 === 'function') {
                trailsToMerge.push(tvTrail);
                vectorsWithTrails.push(tv);
            }
        });

        if (!trailsToMerge.length) return;

        // Preserve appearance based on first trail
        let colorHex = undefined;
        let baseLineWidth = undefined;
        let baseOpacity = undefined;
        const first = trailsToMerge[0];
        if (first) {
            if (first._material && first._material.color) colorHex = first._material.color.getHex();
            else if (typeof first._color !== 'undefined') colorHex = first._color;
            if (typeof first._lineWidth === 'number') baseLineWidth = first._lineWidth;
            if (typeof first._opacity === 'number') baseOpacity = first._opacity;
        }

        // Build merged static segments and dispose originals
        mergeTrailsIntoLineSegments(
            trailsToMerge,
            this.parentGroup,
            colorHex,
            baseLineWidth,
            baseOpacity,
            null
        );

        // Remove trail refs from vectors to avoid updating disposed trails
        vectorsWithTrails.forEach(v => { if (v && v.userData) delete v.userData.trail; });
    }

    dispose() {
        // Standard THREE.js objects added to scene are usually handled by scene traversal on global cleanup.
        this._cancelOutputProjectionReturnWatchdog();
        this._clearScheduledDelays();
    }

    // ------------------------------------------------------------------
    //  Merge fixed K (green) and V (red) vectors for a head into instanced
    //  batches to reduce draw calls once alignment is complete. Original
    //  VectorVisualizationInstancedPrism meshes are disposed, but their
    //  empty groups remain for positional queries used by the conveyor.
    // ------------------------------------------------------------------
    _mergeFixedVectorsForHead(headIdx) {
        if (this._isKvCacheModeEnabled()) return;
        if (this._mergedHeads.has(headIdx)) return; // already merged
        if (!Array.isArray(this.currentLanes) || this.currentLanes.length === 0) return;

        const greens = [];
        const reds = [];

        // Collect K and V vectors for this head across all lanes
        this.currentLanes.forEach((lane) => {
            if (!lane) return;
            const kVec = lane.upwardCopies && lane.upwardCopies[headIdx];
            if (kVec && kVec.mesh) greens.push(kVec);
            const vObj = getSideCopyEntry(lane, headIdx, 'V');
            if (vObj && vObj.vec && vObj.vec.mesh) reds.push(vObj.vec);
        });

        const keepRedMeshesVisible = !!this.enableSelfAttentionAnimation;
        // Build merged instanced meshes (one for K, one for V) and add to scene
        const mergedGroups = { K: null, V: null };
        try {
            if (greens.length) mergedGroups.K = this._buildMergedPrismsFromVectors(greens, `MergedK_head${headIdx}`);
        } catch (_) { /* non-fatal */ }
        try {
            if (reds.length && !keepRedMeshesVisible) mergedGroups.V = this._buildMergedPrismsFromVectors(reds, `MergedV_head${headIdx}`);
        } catch (_) { /* non-fatal */ }

        if (mergedGroups.K) this.parentGroup.add(mergedGroups.K);
        if (mergedGroups.V) this.parentGroup.add(mergedGroups.V);

        // For temp mode, simply hide original meshes to avoid double-drawing,
        // but keep them around for any logic that still references them.
        // For other modes, dispose meshes to reclaim resources.
        const retireOriginalMesh = (vec, { keepVisible = false } = {}) => {
            try {
                if (!vec || !vec.mesh) return;
                if (keepVisible) return;
                if (this.mode === 'temp') {
                    vec.mesh.visible = false;
                } else {
                    if (vec.mesh.material) vec.mesh.material.dispose();
                    if (vec.mesh.geometry) vec.mesh.geometry.dispose();
                    if (vec.group) vec.group.remove(vec.mesh);
                    vec.mesh = null;
                }
            } catch (_) { /* ignore */ }
        };
        greens.forEach((vec) => retireOriginalMesh(vec));
        reds.forEach((vec) => retireOriginalMesh(vec, { keepVisible: keepRedMeshesVisible }));

        this._mergedHeads.add(headIdx);
        this._mergedGroupsByHead.set(headIdx, mergedGroups);
    }

    // Helper to create a single InstancedMesh representing all prisms from a list of vectors
    _buildMergedPrismsFromVectors(vecList, debugName = 'MergedKV') {
        if (!Array.isArray(vecList) || vecList.length === 0) return null;

        // Mirror VectorVisualizationInstancedPrism internal constants
        const PRISM_WIDTH_SCALE = PRISM_INSTANCE_WIDTH_SCALE;
        const PRISM_DEPTH_SCALE = 1.5;
        const uniformCalculatedHeight = Math.max(0.01, PRISM_MAX_HEIGHT * PRISM_HEIGHT_SCALE_FACTOR * 2.0);
        const baseWidth = PRISM_BASE_WIDTH;
        const baseDepth = PRISM_BASE_DEPTH;
        const hideY = HIDE_INSTANCE_Y_OFFSET;

        // Determine if this merged group represents V (red) vectors.
        // If so, we anchor them at the canonical above-matrix height rather than
        // whatever transient Y the fixed copies happen to have when we merge.
        const isRedCategory = /V(\b|_|head|$)/.test(debugName);
        const extraRise = (this.selfAttentionAnimator && this.selfAttentionAnimator.RED_EXTRA_RISE) || SA_RED_EXTRA_RISE;
        // BASE_RISE_ADJUST from VectorMatrixPassThrough is -30; replicate here to match final resting height
        const canonicalRaisedBaseY = this.mhaPassThroughTargetY + this.mhaResultRiseOffsetY - 30 + extraRise;

        const vectorLength = vecList[0]?.instanceCount || this.vectorPrismCount || VECTOR_LENGTH_PRISM;
        const totalInstances = vecList.length * vectorLength;
        const baseGeo = new THREE.BoxGeometry(baseWidth, 1, baseDepth);
        const material = new THREE.MeshBasicMaterial({
            color: 0xffffff,
            side: THREE.DoubleSide
        });
        // Shader patch to support per-instance left→right gradient identical to VectorVisualizationInstancedPrism
        material.customProgramCacheKey = () => 'InstancedPrismGradientV1';
        material.onBeforeCompile = (shader) => {
            shader.uniforms.prismHalfWidth = { value: getPrismSpacing(PRISM_WIDTH_SCALE) / 2 };
            shader.vertexShader = shader.vertexShader.replace(
                '#include <common>',
                `#include <common>\nattribute vec3 colorStart;\nattribute vec3 colorEnd;\nvarying vec3 vColorStart;\nvarying vec3 vColorEnd;\nvarying float vGradientT;\nuniform float prismHalfWidth;`
            );
            shader.vertexShader = shader.vertexShader.replace(
                '#include <begin_vertex>',
                `#include <begin_vertex>\n    vColorStart = colorStart;\n    vColorEnd   = colorEnd;\n    vGradientT  = clamp( (position.x + prismHalfWidth) / (2.0 * prismHalfWidth), 0.0, 1.0 );`
            );
            shader.fragmentShader = shader.fragmentShader.replace(
                '#include <common>',
                `#include <common>\nvarying vec3 vColorStart;\nvarying vec3 vColorEnd;\nvarying float vGradientT;`
            );
            shader.fragmentShader = shader.fragmentShader.replace(
                'vec4 diffuseColor = vec4( diffuse, opacity );',
                `vec3 grad = mix( vColorStart, vColorEnd, vGradientT );\n    vec4 diffuseColor = vec4( grad, opacity );`
            );
        };

        const instanced = new THREE.InstancedMesh(baseGeo, material, totalInstances);
        instanced.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
        instanced.frustumCulled = false;

        // Gradient buffers (per instance)
        const colorStartArr = new Float32Array(totalInstances * 3);
        const colorEndArr   = new Float32Array(totalInstances * 3);
        const colorStartAttr = new THREE.InstancedBufferAttribute(colorStartArr, 3);
        const colorEndAttr   = new THREE.InstancedBufferAttribute(colorEndArr, 3);
        instanced.geometry.setAttribute('colorStart', colorStartAttr);
        instanced.geometry.setAttribute('colorEnd',   colorEndAttr);

        const tmpMatrix = new THREE.Matrix4();
        const tmpPos = new THREE.Vector3();
        const tmpQuat = new THREE.Quaternion();
        const tmpScale = new THREE.Vector3();
        const tmpColor = new THREE.Color();
        const dummy = new THREE.Object3D();
        const hiddenScale = 0.001;

        // Populate instances by copying per-prism transforms and colours from the original vectors
        let instIndex = 0;
        // Build mapping so raycasts (instanceId) can be decoded back to vector/prism/head/lane
        const vectorRefs = vecList.slice();
        const headIdxs = vecList.map(v => (v && v.userData && typeof v.userData.headIndex === 'number') ? v.userData.headIndex : null);
        const laneIdxs = vecList.map(v => {
            const z = v && v.group ? v.group.position.z : NaN;
            const lanes = this.currentLanes || [];
            let best = -1, bestDist = Infinity;
            for (let i = 0; i < lanes.length; i++) {
                const d = Math.abs(lanes[i].zPos - z);
                if (d < bestDist) { best = i; bestDist = d; }
            }
            return best;
        });
        for (let vIdx = 0; vIdx < vecList.length; vIdx++) {
            const vec = vecList[vIdx];
            if (!vec || !vec.group) { instIndex += vectorLength; continue; }

            // Access original gradient attrs if available
            const srcCS = vec.mesh && vec.mesh.geometry && vec.mesh.geometry.getAttribute ? vec.mesh.geometry.getAttribute('colorStart') : null;
            const srcCE = vec.mesh && vec.mesh.geometry && vec.mesh.geometry.getAttribute ? vec.mesh.geometry.getAttribute('colorEnd')   : null;

            for (let i = 0; i < vectorLength; i++, instIndex++) {
                // Determine if this prism is visible or hidden in the source vector by reading its matrix
                let hidden = false;
                if (vec.mesh && typeof vec.mesh.getMatrixAt === 'function') {
                    vec.mesh.getMatrixAt(i, tmpMatrix);
                    tmpMatrix.decompose(tmpPos, tmpQuat, tmpScale);
                    // Hidden prisms are scaled down and translated to a large negative Y
                    if (tmpPos.y < hideY * 0.5 || tmpScale.y < 0.01) hidden = true;
                }

                // Compute world position for this prism
                const baseX = computeCenteredPrismX(i, vectorLength);
                const worldX = vec.group.position.x + baseX;
                const baseYForCategory = isRedCategory ? canonicalRaisedBaseY : (vec.group && vec.group.position ? vec.group.position.y : 0);
                const worldY = baseYForCategory + uniformCalculatedHeight / 2;
                const worldZ = vec.group.position.z;

                dummy.position.set(worldX, worldY, worldZ);
                // Apply fixed uniform dimensions (mirrors original visual size)
                // Fully collapse hidden prisms so they never show up below the tower.
                if (hidden) {
                    dummy.scale.set(hiddenScale, hiddenScale, hiddenScale);
                } else {
                    dummy.scale.set(PRISM_WIDTH_SCALE, uniformCalculatedHeight, PRISM_DEPTH_SCALE);
                }
                dummy.updateMatrix();
                instanced.setMatrixAt(instIndex, dummy.matrix);

                // Copy gradient colours from source if present; otherwise derive from vector helper
                if (srcCS && srcCE) {
                    const i3 = i * 3;
                    colorStartArr[instIndex * 3 + 0] = srcCS.array[i3 + 0] || 0.5;
                    colorStartArr[instIndex * 3 + 1] = srcCS.array[i3 + 1] || 0.5;
                    colorStartArr[instIndex * 3 + 2] = srcCS.array[i3 + 2] || 0.5;
                    colorEndArr[instIndex * 3 + 0] = srcCE.array[i3 + 0] || 0.5;
                    colorEndArr[instIndex * 3 + 1] = srcCE.array[i3 + 1] || 0.5;
                    colorEndArr[instIndex * 3 + 2] = srcCE.array[i3 + 2] || 0.5;
                    tmpColor.setRGB(
                        (colorStartArr[instIndex * 3 + 0] + colorEndArr[instIndex * 3 + 0]) * 0.5,
                        (colorStartArr[instIndex * 3 + 1] + colorEndArr[instIndex * 3 + 1]) * 0.5,
                        (colorStartArr[instIndex * 3 + 2] + colorEndArr[instIndex * 3 + 2]) * 0.5,
                    );
                } else if (typeof vec.getDefaultColorForIndex === 'function') {
                    const leftColor  = vec.getDefaultColorForIndex(Math.max(0, i - 1));
                    const rightColor = vec.getDefaultColorForIndex(Math.min(vectorLength - 1, i + 1));
                    colorStartArr[instIndex * 3 + 0] = leftColor.r;  colorStartArr[instIndex * 3 + 1] = leftColor.g;  colorStartArr[instIndex * 3 + 2] = leftColor.b;
                    colorEndArr[instIndex * 3 + 0]   = rightColor.r; colorEndArr[instIndex * 3 + 1]   = rightColor.g; colorEndArr[instIndex * 3 + 2]   = rightColor.b;
                    tmpColor.copy(leftColor).lerp(rightColor, 0.5);
                } else {
                    // Neutral grey fallback
                    colorStartArr[instIndex * 3 + 0] = 0.5; colorStartArr[instIndex * 3 + 1] = 0.5; colorStartArr[instIndex * 3 + 2] = 0.5;
                    colorEndArr[instIndex * 3 + 0]   = 0.5; colorEndArr[instIndex * 3 + 1]   = 0.5; colorEndArr[instIndex * 3 + 2]   = 0.5;
                    tmpColor.setRGB(0.5, 0.5, 0.5);
                }

                // Preserve per-instance colour used by effects that rely on instanceColor buffers.
                if (vec.mesh && typeof vec.mesh.getColorAt === 'function' && vec.mesh.instanceColor) {
                    try {
                        vec.mesh.getColorAt(i, tmpColor);
                    } catch (_) {
                        // Fall back to the average of gradient colours computed above.
                    }
                }
                if (typeof instanced.setColorAt === 'function') {
                    instanced.setColorAt(instIndex, tmpColor);
                }
            }
        }

        instanced.instanceMatrix.needsUpdate = true;
        colorStartAttr.needsUpdate = true;
        colorEndAttr.needsUpdate = true;
        if (instanced.instanceColor) {
            instanced.instanceColor.needsUpdate = true;
        }

        const group = new THREE.Group();
        group.name = debugName;
        group.add(instanced);
        // Attach metadata for decoding instanceId on raycast
        const category = /V(\b|_|head|$)/.test(debugName) ? 'V' : 'K';
        // Assign descriptive label so hover shows full text for merged groups
        group.userData.label = (category === 'V') ? 'Merged Value Vectors (Orange)' : 'Merged Key Vectors (Green)';
        group.userData.isVector = true;
        if (Number.isFinite(this.layerIndex)) {
            group.userData.layerIndex = this.layerIndex;
        }
        const mergedKVMeta = {
            category,
            vectorPrismCount: vectorLength,
            vectorRefs,
            headIdxs,
            laneIdxs
        };
        group.userData.mergedKVMeta = mergedKVMeta;
        instanced.userData.mergedKVMeta = mergedKVMeta;
        // Also set label on mesh for direct hits
        instanced.userData.label = (category === 'V') ? 'Merged Value Vectors (Orange)' : 'Merged Key Vectors (Green)';
        instanced.userData.isVector = true;
        if (Number.isFinite(this.layerIndex)) {
            instanced.userData.layerIndex = this.layerIndex;
        }
        return group;
    }

    /** Decode a THREE.Raycaster intersection against a merged K/V InstancedMesh. */
    decodeMergedKVIntersection(intersection) {
        if (!intersection || typeof intersection.instanceId !== 'number') return null;
        const mesh = intersection.object;
        if (!mesh || !mesh.isInstancedMesh) return null;
        const meta = (mesh.userData && mesh.userData.mergedKVMeta)
            || (mesh.parent && mesh.parent.userData && mesh.parent.userData.mergedKVMeta)
            || null;
        if (!meta) return null;
        const instanceId = intersection.instanceId;
        const vectorIndex = Math.floor(instanceId / meta.vectorPrismCount);
        const prismIndex = instanceId % meta.vectorPrismCount;
        const headIndex = Array.isArray(meta.headIdxs) ? meta.headIdxs[vectorIndex] : null;
        const laneIndex = Array.isArray(meta.laneIdxs) ? meta.laneIdxs[vectorIndex] : null;
        const vectorRef = Array.isArray(meta.vectorRefs) ? meta.vectorRefs[vectorIndex] : null;
        const activationData = vectorRef && vectorRef.userData ? vectorRef.userData.activationData : null;
        return {
            category: meta.category, // 'K' or 'V'
            vectorIndex,
            prismIndex,
            headIndex,
            laneIndex,
            vectorRef,
            activationData
        };
    }

    // Merge ALL fixed K and V vectors across heads into two meshes
    _mergeAllFixedKVIfReady() {
        if (this._isKvCacheModeEnabled()) return;
        if (this._allFixedMerged) return;
        if (!Array.isArray(this.currentLanes) || this.currentLanes.length === 0) return;
        if (this.mhaPassThroughPhase !== 'mha_pass_through_complete') return;
        // Ensure all heads are aligned before collapsing geometry
        for (let h = 0; h < NUM_HEAD_SETS_LAYER; h++) {
            if (!this.selfAttentionAnimator || this.selfAttentionAnimator.greensAligned[h] !== true) {
                return;
            }
        }

        const allGreens = [];
        const allReds = [];
        this.currentLanes.forEach(lane => {
            if (!lane) return;
            if (Array.isArray(lane.upwardCopies)) {
                for (let h = 0; h < NUM_HEAD_SETS_LAYER; h++) {
                    const kVec = lane.upwardCopies[h];
                    if (kVec && kVec.mesh) allGreens.push(kVec);
                }
            }
            if (Array.isArray(lane.sideCopies)) {
                lane.sideCopies.forEach(sc => {
                    if (sc && sc.type === 'V' && sc.vec && sc.vec.mesh) allReds.push(sc.vec);
                });
            }
        });
        if (!allGreens.length && !allReds.length) return;

        const keepRedMeshesVisible = !!this.enableSelfAttentionAnimation;
        let mergedKGroup = null;
        let mergedVGroup = null;
        try { if (allGreens.length) mergedKGroup = this._buildMergedPrismsFromVectors(allGreens, 'MergedAllK'); } catch (_) {}
        try { if (allReds.length && !keepRedMeshesVisible) mergedVGroup = this._buildMergedPrismsFromVectors(allReds,   'MergedAllV'); } catch (_) {}
        if (mergedKGroup) this.parentGroup.add(mergedKGroup);
        if (mergedVGroup) this.parentGroup.add(mergedVGroup);

        const stripMeshKeepGroup = (vec) => {
            try {
                if (vec && vec.mesh) {
                    if (vec.mesh.material) vec.mesh.material.dispose();
                    if (vec.mesh.geometry) vec.mesh.geometry.dispose();
                    if (vec.group) vec.group.remove(vec.mesh);
                    vec.mesh = null;
                }
            } catch (_) {}
        };
        allGreens.forEach(stripMeshKeepGroup);
        if (!keepRedMeshesVisible) {
            allReds.forEach(stripMeshKeepGroup);
        }
        this._allFixedMerged = true;
        if (keepRedMeshesVisible) {
            logMhsaDebug('MHSAAnimation: Merged all fixed K vectors into one instanced mesh (kept V vectors visible).');
        } else {
            logMhsaDebug('MHSAAnimation: Merged all fixed K and V vectors into two instanced meshes.');
        }
    }

    // Fade any existing merged K/V instanced meshes to a target opacity
    _fadeMergedKVOpacity(targetOpacity, durationMs, delayMs = 0) {
        const doFade = (group) => {
            if (!group) return;
            const inst = group.children && group.children[0];
            if (!inst || !inst.isMesh) return;
            const mat = inst.material;
            if (!mat) return;
            mat.transparent = true;
            const state = { op: typeof mat.opacity === 'number' ? mat.opacity : 1.0 };
            if (typeof TWEEN !== 'undefined') {
                new TWEEN.Tween(state)
                    .to({ op: targetOpacity }, durationMs)
                    .delay(delayMs)
                    .easing(TWEEN.Easing.Quadratic.InOut)
                    .onUpdate(() => { mat.opacity = state.op; mat.needsUpdate = true; })
                    .start();
            } else {
                mat.opacity = targetOpacity;
                mat.needsUpdate = true;
            }
        };
        this._mergedGroupsByHead.forEach((grp) => {
            if (grp && grp.K) doFade(grp.K);
            if (grp && grp.V) doFade(grp.V);
        });
    }

    // Temp-mode pipeline: dim outputs, spawn decoratives, then schedule merge/projection.
    _applyTempModeBehaviour() {
        const grayColor = new THREE.Color(0x606060);
        const fadeDuration = this._resolveSkipDuration(DECORATIVE_FADE_MS);
        const fadeDelay = this._resolveSkipDelay(DECORATIVE_FADE_DELAY_MS);
        const preserveKvCache = this._shouldPreserveKVCacheVectors();
        const shouldPreserveVecForKvCache = (vec) => {
            if (!preserveKvCache || !vec) return false;
            const category = vec?.userData?.vectorCategory || vec?.userData?.qkvProcessedCategory || null;
            return category === 'K' || category === 'V';
        };
        // Visible prism window for gray-out and gradient calculations
        const visiblePrismCountTemp = Math.min(this.vectorPrismCount, Math.ceil(this.outputVectorLength / PRISM_DIMENSIONS_PER_UNIT));
        const startVisibleIdx = Math.max(0, Math.floor((this.vectorPrismCount - visiblePrismCountTemp) / 2));
        const endVisibleIdx = startVisibleIdx + visiblePrismCountTemp - 1;

        this._tempAllOutputVectors.forEach(vec => {
            if (!vec || !vec.mesh) return;
            if (shouldPreserveVecForKvCache(vec)) {
                this._setVectorVisible(vec, { opacity: 1 });
                return;
            }

            // Gray-out only the visible 64-dim region so outer hidden prisms stay hidden
            for (let i = startVisibleIdx; i <= endVisibleIdx; i++) {
                vec.setInstanceAppearance(i, 0, grayColor);
            }

            // Lower emissive intensity on the shared material (if present)
            if (vec.mesh.material && typeof vec.mesh.material.emissiveIntensity === 'number') {
                vec.mesh.material.emissiveIntensity = scaleGlobalEmissiveIntensity(0.05);
            }

            // Ensure material supports transparency
            if (vec.mesh.material) {
                vec.mesh.material.transparent = true;
                vec.mesh.material.opacity = 1.0; // start fully opaque gray
                vec.mesh.material.needsUpdate = true;
            }

            // Fade out the gray vectors to make them less visible
            if (vec.mesh.material && typeof TWEEN !== 'undefined') {
                new TWEEN.Tween({ op: 1.0 })
                    .to({ op: 0.2 }, fadeDuration)
                    .easing(TWEEN.Easing.Quadratic.Out)
                    .onUpdate(function(obj){
                        vec.mesh.material.opacity = obj.op;
                        vec.mesh.material.needsUpdate = true;
                    })
                    .start();
            } else if (vec.mesh.material) {
                vec.mesh.material.opacity = 0.2;
            }
        });

        // 2) Spawn new decorative vectors (64-dim) above each central K vector and fade them in
        const weightedDecoratives = this._getWeightedSumDecoratives();
        if (weightedDecoratives.length) {
            this._tempDecorativeVecs = weightedDecoratives;
            this._restoreWeightedSumColors(fadeDuration, 0);
        } else {
            this._spawnTempDecorativesFromK({ fadeDuration, fadeDelay });
        }

        // After decorative vectors begin fading in, further dim gray vectors for subtlety
        if (typeof TWEEN !== 'undefined') {
            this._tempAllOutputVectors.forEach(vec => {
                if (!vec || !vec.mesh || !vec.mesh.material) return;
                if (shouldPreserveVecForKvCache(vec)) {
                    this._setVectorVisible(vec, { opacity: 1 });
                    return;
                }
                const mat = vec.mesh.material;
                new TWEEN.Tween({ op: mat.opacity })
                    .to({ op: 0.05 }, fadeDuration)
                    .delay(fadeDelay)
                    .onUpdate(function(o){
                        mat.opacity = o.op;
                        mat.needsUpdate = true;
                    })
                    .onComplete(() => {
                        // Once the gray vector has fully faded, remove it from the scene
                        if (vec && vec.group) {
                            // Hide, detach, and dispose to reclaim resources
                            vec.group.visible = false;
                            this.parentGroup.remove(vec.group);
                            if (typeof vec.dispose === 'function') {
                                vec.dispose();
                            }
                        }
                    })
                    .start();
            });
        }

        // ------------------------------------------------------
        //   Begin merge-to-row-vector phase after fade-in delay
        // ------------------------------------------------------
        if (typeof TWEEN !== 'undefined') {
            const mergeDelayMs = MERGE_TO_ROW_DELAY_AFTER_FADE_MS / GLOBAL_ANIM_SPEED_MULT;
            if (this.enableSelfAttentionAnimation && this.selfAttentionAnimator) {
                // Wait for the conveyor to finish so all weighted sums exist
                this.selfAttentionAnimator.start(() => {
                    this._scheduleAfterDelay(() => {
                        this._startMergeToRowVectors();
                    }, mergeDelayMs);
                });
            } else {
                // Start after decorative fade-in completes
                this._scheduleAfterDelay(() => {
                    this._startMergeToRowVectors();
                }, mergeDelayMs);
            }
        } else if (this._tempDecorativeVecs && this._tempDecorativeVecs.length) {
            this._startMergeToRowVectors();
        }
    }

    _clearTempDecoratives(options = {}) {
        const dispose = options && typeof options.dispose === 'boolean' ? options.dispose : true;
        if (!Array.isArray(this._tempDecorativeVecs)) {
            this._tempDecorativeVecs = [];
            return;
        }
        this._tempDecorativeVecs.forEach((entry) => {
            const vec = entry && entry.vec ? entry.vec : null;
            if (!vec) return;
            try {
                if (vec.group && vec.group.parent) vec.group.parent.remove(vec.group);
            } catch (_) { /* optional cleanup */ }
            if (dispose && typeof vec.dispose === 'function') {
                try { vec.dispose(); } catch (_) { /* optional cleanup */ }
            }
        });
        this._tempDecorativeVecs = [];
    }

    _clearWeightedSumVectors(options = {}) {
        const dispose = options && typeof options.dispose === 'boolean' ? options.dispose : true;
        if (!this._attentionWeightedSums || this._attentionWeightedSums.size === 0) return;
        this._attentionWeightedSums.forEach((entry) => {
            const vec = entry && entry.vec ? entry.vec : null;
            if (!vec) return;
            try {
                if (vec.group && vec.group.parent) vec.group.parent.remove(vec.group);
            } catch (_) { /* optional cleanup */ }
            if (dispose && typeof vec.dispose === 'function') {
                try { vec.dispose(); } catch (_) { /* optional cleanup */ }
            }
        });
        this._attentionWeightedSums.clear();
    }

    _spawnTempDecorativesFromK(options = {}) {
        const fadeDuration = options && Number.isFinite(options.fadeDuration) ? options.fadeDuration : 0;
        const fadeDelay = options && Number.isFinite(options.fadeDelay) ? options.fadeDelay : 0;
        const clearExisting = options && typeof options.clearExisting === 'boolean' ? options.clearExisting : false;
        if (clearExisting) this._clearTempDecoratives();

        const visiblePrismCountTemp = Math.min(this.vectorPrismCount, Math.ceil(this.outputVectorLength / PRISM_DIMENSIONS_PER_UNIT));
        const startVisibleIdx = Math.max(0, Math.floor((this.vectorPrismCount - visiblePrismCountTemp) / 2));
        const verticalOffset = 60; // world units above existing vector

        const nextDecoratives = [];
        (this._tempKOutputVectors || []).forEach((kVec) => {
            if (!kVec || !kVec.group) return;

            // Build raw 768-dim data with 30 random switch points for varied gradient
            const rawData = [];
            const desiredSwitches = Math.min(30, this.vectorPrismCount);
            logRandomColorDebug('MHSAAnimation.decorativeVector.randomRawData', {
                layerIndex: this.layerIndex,
                vectorPrismCount: this.vectorPrismCount,
                desiredSwitches
            });
            const switchPoints = new Set();
            while (switchPoints.size < desiredSwitches) {
                const idx = Math.floor(Math.random() * this.vectorPrismCount);
                switchPoints.add(idx);
            }
            const sortedSwitch = Array.from(switchPoints).sort((a, b) => a - b);
            let curVal = Math.random() * 2 - 1;
            let nextSwitch = sortedSwitch.shift();
            for (let i = 0; i < this.vectorPrismCount; i++) {
                if (i === nextSwitch) {
                    curVal = Math.random() * 2 - 1;
                    nextSwitch = sortedSwitch.shift();
                }
                rawData.push(curVal);
            }

            const spawnPos = kVec.group.position.clone().add(new THREE.Vector3(0, verticalOffset, 0));
            const decoVec = new VectorVisualizationInstancedPrism(
                rawData,
                spawnPos,
                3,
                this.vectorPrismCount
            );

            // Hide outer prisms and snap decorative vec to 64-dim geometry
            decoVec.applyProcessedVisuals(
                rawData.slice(startVisibleIdx, startVisibleIdx + visiblePrismCountTemp),
                this.outputVectorLength,
                { numKeyColors: this.outputVectorLength },
                { setHiddenToBlack: true }
            );

            // Colour the visible prism(s) with a smooth two-colour gradient so
            // each 64-D decorative vector has its own distinctive colour.
            const startHue = Math.random();
            const endHue = Math.random();
            logRandomColorDebug('MHSAAnimation.decorativeVector.randomGradientHues', {
                layerIndex: this.layerIndex,
                startHue,
                endHue
            });
            const startColor = new THREE.Color().setHSL(startHue, 0.9, 0.6);
            const endColor   = new THREE.Color().setHSL(endHue, 0.9, 0.6);
            const visibleCount = visiblePrismCountTemp;
            for (let vi = 0; vi < visibleCount; vi++) {
                const idx = startVisibleIdx + vi;
                const t   = visibleCount > 1 ? vi / (visibleCount - 1) : 0;
                const col = startColor.clone().lerp(endColor, t);
                decoVec.setInstanceAppearance(idx, 0, col);
            }

            this.parentGroup.add(decoVec.group);

            // Keep reference for merge phase. Carry lane identity so later
            // output-projection return can map back to lanes robustly.
            const laneZ = kVec.group.position.z;
            nextDecoratives.push({
                vec: decoVec,
                laneZ,
                laneIndex: this._resolveLaneIndexFromZ(laneZ)
            });

            // Start invisible: set material opacity to 0
            if (decoVec.mesh && decoVec.mesh.material) {
                decoVec.mesh.material.transparent = true;
                decoVec.mesh.material.opacity = 0.0;
            }

            if (typeof TWEEN !== 'undefined') {
                const mat = decoVec.mesh.material;
                new TWEEN.Tween({ op: 0.0 })
                    .to({ op: 1.0 }, fadeDuration)
                    .easing(TWEEN.Easing.Quadratic.InOut)
                    .onUpdate(function(o){
                        mat.opacity = o.op;
                        mat.needsUpdate = true;
                    })
                    .delay(fadeDelay)
                    .start();
            } else if (decoVec.mesh && decoVec.mesh.material) {
                decoVec.mesh.material.opacity = 1.0;
                decoVec.mesh.material.needsUpdate = true;
            }
        });

        this._tempDecorativeVecs = nextDecoratives;
        return nextDecoratives;
    }

    _resolveLaneIndexFromZ(laneZ) {
        if (!Array.isArray(this.currentLanes) || !Number.isFinite(laneZ)) return null;
        let bestIdx = null;
        let bestDist = Infinity;
        for (let i = 0; i < this.currentLanes.length; i++) {
            const lane = this.currentLanes[i];
            const z = lane && Number.isFinite(lane.zPos) ? lane.zPos : null;
            if (!Number.isFinite(z)) continue;
            const dist = Math.abs(z - laneZ);
            if (dist < bestDist) {
                bestDist = dist;
                bestIdx = i;
            }
        }
        return bestDist <= 0.25 ? bestIdx : null;
    }

    registerWeightedSumVector(vec, laneZ, headIdx, lane = null) {
        if (!vec || !vec.group) return;
        const resolvedLaneZ = Number.isFinite(lane?.zPos) ? lane.zPos
            : Number.isFinite(laneZ) ? laneZ : vec.group.position.z;
        const laneIndex = Number.isFinite(lane?.laneIndex)
            ? lane.laneIndex
            : this._resolveLaneIndexFromZ(resolvedLaneZ);
        const keyLane = Number.isFinite(laneIndex)
            ? laneIndex
            : Number.isFinite(resolvedLaneZ) ? resolvedLaneZ.toFixed(3) : 'unknown';
        const key = `${headIdx}|${keyLane}`;
        this._attentionWeightedSums.set(key, {
            vec,
            laneZ: resolvedLaneZ,
            headIdx,
            laneIndex
        });
    }

    _getWeightedSumDecoratives() {
        if (!this._attentionWeightedSums || this._attentionWeightedSums.size === 0) return [];
        const entries = Array.from(this._attentionWeightedSums.values())
            .filter(entry => entry && entry.vec && entry.vec.group);
        entries.sort((a, b) => {
            const laneA = Number.isFinite(a.laneIndex) ? a.laneIndex : 0;
            const laneB = Number.isFinite(b.laneIndex) ? b.laneIndex : 0;
            if (laneA !== laneB) return laneA - laneB;
            const headA = Number.isFinite(a.headIdx) ? a.headIdx : 0;
            const headB = Number.isFinite(b.headIdx) ? b.headIdx : 0;
            return headA - headB;
        });
        return entries.map(entry => ({
            vec: entry.vec,
            laneZ: Number.isFinite(entry.laneZ) ? entry.laneZ : entry.vec.group.position.z,
            laneIndex: Number.isFinite(entry.laneIndex) ? entry.laneIndex : this._resolveLaneIndexFromZ(entry.laneZ)
        }));
    }

    _resolveLaneByIdentity(laneIndex, laneZ) {
        if (!Array.isArray(this.currentLanes) || this.currentLanes.length === 0) return null;
        if (Number.isFinite(laneIndex)) {
            const idx = Math.floor(laneIndex);
            if (idx >= 0 && idx < this.currentLanes.length) {
                return this.currentLanes[idx] || null;
            }
        }
        const resolvedIndex = this._resolveLaneIndexFromZ(laneZ);
        if (Number.isFinite(resolvedIndex) && resolvedIndex >= 0 && resolvedIndex < this.currentLanes.length) {
            return this.currentLanes[resolvedIndex] || null;
        }
        return null;
    }

    _ensureLaneReadyForLn2Fallback(lane) {
        return primeLaneForLn2Fallback(lane);
    }

    _forcePostAttentionFallback(reason = 'unknown reason', options = {}) {
        const lanes = Array.isArray(this.currentLanes) ? this.currentLanes : [];
        if (!lanes.length) return 0;
        const onlyIncomplete = !!options.onlyIncomplete;
        let forcedCount = 0;
        for (let i = 0; i < lanes.length; i++) {
            const lane = lanes[i];
            if (!lane || lane.ln2Phase === LN2_PHASE.DONE) continue;
            if (onlyIncomplete) {
                const hasPostVec = !!(lane.postAdditionVec && lane.postAdditionVec.group);
                const ln2Primed = isLn2PrimedPhase(lane.ln2Phase);
                if (hasPostVec && ln2Primed) continue;
            }
            if (this._ensureLaneReadyForLn2Fallback(lane)) {
                forcedCount += 1;
            }
        }
        if (forcedCount > 0) {
            const layer = lanes[0] && lanes[0].layer;
            if (layer && typeof layer._emitProgress === 'function') {
                layer._emitProgress();
            }
            console.warn(
                `MHSAAnimation: forced post-attention lane fallback for ${forcedCount} lane(s)`
                + ` (layer=${this.layerIndex ?? '?'}, reason=${reason}).`
            );
        }
        return forcedCount;
    }

    _cancelOutputProjectionReturnWatchdog() {
        if (typeof this._cancelOutputProjectionWatchdog === 'function') {
            try {
                this._cancelOutputProjectionWatchdog();
            } catch (_) { /* best effort */ }
        }
        this._cancelOutputProjectionWatchdog = null;
    }

    _scheduleOutputProjectionReturnWatchdog(delayMs, reason = 'output projection return watchdog') {
        this._cancelOutputProjectionReturnWatchdog();
        const safeDelayMs = Number.isFinite(delayMs) ? Math.max(0, delayMs) : 0;
        if (safeDelayMs <= 0) return;
        this._cancelOutputProjectionWatchdog = this._scheduleAfterDelay(() => {
            this._cancelOutputProjectionWatchdog = null;
            if (this.outputProjMatrixReturnComplete) return;
            const targetCount = Number.isFinite(this._outputProjReturnTargetCount)
                ? Math.max(0, Math.floor(this._outputProjReturnTargetCount))
                : 0;
            const returnCount = Number.isFinite(this._outputProjReturnCount)
                ? Math.max(0, Math.floor(this._outputProjReturnCount))
                : 0;
            if (targetCount > 0 && returnCount >= targetCount) {
                this.outputProjMatrixReturnComplete = true;
                this.outputProjMatrixAnimationPhase = 'completed';
                return;
            }
            console.warn(
                `MHSAAnimation: output projection watchdog fired for layer ${this.layerIndex ?? '?'} `
                + `(${returnCount}/${targetCount}).`
            );
            this._completeOutputProjectionFallback(reason, {
                fallbackLanes: true,
                onlyIncompleteLanes: false,
                logReason: false
            });
        }, safeDelayMs);
    }

    _completeOutputProjectionFallback(reason, options = {}) {
        const fallbackLanes = options.fallbackLanes !== false;
        const onlyIncompleteLanes = !!options.onlyIncompleteLanes;
        const logReason = options.logReason !== false;
        this._cancelOutputProjectionReturnWatchdog();
        const targetCount = Number.isFinite(this._outputProjReturnTargetCount)
            ? Math.max(0, Math.floor(this._outputProjReturnTargetCount))
            : 0;
        this._outputProjReturnTargetCount = targetCount;
        this._outputProjReturnCount = targetCount;
        this.outputProjMatrixAnimationPhase = 'completed';
        this.outputProjMatrixReturnComplete = true;
        if (this.rowMergePhase === 'not_started') {
            this.rowMergePhase = 'merged';
        }
        if (fallbackLanes) {
            this._forcePostAttentionFallback(reason || 'output projection fallback', {
                onlyIncomplete: onlyIncompleteLanes
            });
        }
        if (logReason && reason) {
            console.warn(
                `MHSAAnimation: forcing output projection completion `
                + `(layer=${this.layerIndex ?? '?'}, reason=${reason}).`
            );
        }
    }

    _adjustOutputProjectionMatrixForRow(rowY) {
        if (!Number.isFinite(rowY)) return;
        if (!this.outputProjectionMatrix || !this.outputProjectionMatrix.group) return;
        const desiredBottomY = rowY + MHA_OUTPUT_PROJECTION_MATRIX_Y_OFFSET_ABOVE_ROW;
        const newCenterY = desiredBottomY + this.outputProjMatrixHeight / 2;
        if (!Number.isFinite(newCenterY)) return;
        if (Math.abs(newCenterY - this.outputProjMatrixCenterY) < 0.01) return;

        this.outputProjMatrixCenterY = newCenterY;
        if (this.outputProjMatrixBasePosition) {
            this.outputProjMatrixBasePosition.set(
                this.outputProjMatrixBasePosition.x,
                newCenterY,
                this.outputProjMatrixBasePosition.z
            );
        }
        this.outputProjectionMatrix.group.position.y = newCenterY;

        const matrixTopY = newCenterY + this.outputProjMatrixHeight / 2;
        this.finalCombinedY = matrixTopY + 60;
        this.finalOriginalY = this.finalCombinedY - ORIGINAL_TO_PROCESSED_GAP - MHSA_RESIDUAL_ADDITION_EXTRA_GAP;
    }

    _restoreWeightedSumColors(fadeDuration = 0, fadeDelay = 0) {
        const entries = this._getWeightedSumDecoratives();
        if (!entries.length) return;
        const outputLength = this.outputVectorLength || 64;
        const rangeOptions = buildHueRangeOptions(MHA_VALUE_SPECTRUM_COLOR, {
            hueSpread: MHA_VALUE_HUE_SPREAD,
            minLightness: MHA_VALUE_LIGHTNESS_MIN,
            maxLightness: MHA_VALUE_LIGHTNESS_MAX,
            valueMin: MHA_VALUE_RANGE_MIN,
            valueMax: MHA_VALUE_RANGE_MAX,
            valueClampMax: MHA_VALUE_CLAMP_MAX,
        });
        entries.forEach(({ vec }) => {
            if (!vec || !vec.mesh) return;
            const raw = Array.isArray(vec.rawData)
                ? vec.rawData.slice(0, outputLength)
                : [];
            const numKeyColors = raw.length <= 1 ? 1 : Math.min(MHA_VALUE_KEY_COLOR_COUNT, raw.length);
            vec.applyProcessedVisuals(
                raw,
                outputLength,
                { numKeyColors, generationOptions: rangeOptions },
                { setHiddenToBlack: true },
                raw
            );
            if (raw.length === 1 && typeof vec.setUniformColor === 'function') {
                vec.setUniformColor(mapValueToHueRange(raw[0], rangeOptions));
            }
            if (vec.userData) {
                vec.userData.weightedSumReadyForConcat = true;
            }
            vec.group.visible = true;
            if (vec.mesh) vec.mesh.visible = true;
            if (vec.mesh.material) {
                vec.mesh.material.transparent = true;
                vec.mesh.material.opacity = 0.25;
                vec.mesh.material.needsUpdate = true;
            }
            if (typeof TWEEN !== 'undefined' && vec.mesh && vec.mesh.material) {
                const mat = vec.mesh.material;
                new TWEEN.Tween({ op: mat.opacity })
                    .to({ op: 1.0 }, fadeDuration)
                    .easing(TWEEN.Easing.Quadratic.InOut)
                    .delay(fadeDelay)
                    .onUpdate((o) => {
                        mat.opacity = o.op;
                        mat.needsUpdate = true;
                    })
                    .start();
            } else if (vec.mesh && vec.mesh.material) {
                vec.mesh.material.opacity = 1.0;
                vec.mesh.material.needsUpdate = true;
            }
        });
    }

    // Merge per-head decorative vectors into a single horizontal row per lane.
    _startMergeToRowVectors() {
        if (this.rowMergePhase && this.rowMergePhase !== 'not_started') return;
        if (this.enableSelfAttentionAnimation) {
            const weighted = this._getWeightedSumDecoratives();
            if (weighted.length) this._tempDecorativeVecs = weighted;
        }
        if (!this._tempDecorativeVecs || this._tempDecorativeVecs.length === 0) return;

        // Mark that decorative vectors are now travelling horizontally back
        // toward the output projection alignment row.
        this.rowMergePhase = 'merging';

        // Build map lane -> decorative vectors. Prefer lane index identity
        // (stable across animation) and fall back to lane Z.
        const laneVectors = new Map();
        this._tempDecorativeVecs.forEach(obj => {
            if (!obj || !obj.vec || !obj.vec.group) return;
            const laneZ = Number.isFinite(obj.laneZ) ? obj.laneZ : obj.vec.group.position.z;
            const laneIndex = Number.isFinite(obj.laneIndex)
                ? Math.floor(obj.laneIndex)
                : this._resolveLaneIndexFromZ(laneZ);
            const key = Number.isFinite(laneIndex)
                ? `idx:${laneIndex}`
                : `z:${Number.isFinite(laneZ) ? laneZ.toFixed(3) : 'unknown'}`;
            if (!laneVectors.has(key)) {
                laneVectors.set(key, { laneZ, laneIndex, vectors: [] });
            }
            const laneEntry = laneVectors.get(key);
            if (!Number.isFinite(laneEntry.laneZ) && Number.isFinite(laneZ)) laneEntry.laneZ = laneZ;
            if (!Number.isFinite(laneEntry.laneIndex) && Number.isFinite(laneIndex)) laneEntry.laneIndex = laneIndex;
            laneEntry.vectors.push(obj.vec);
        });

        const firstHeadCenterX = this.headsCentersX.length ? this.headsCentersX[0] : 0; 
        const targetX = firstHeadCenterX; // Existing merge target for centralised row
        let maxDurationMs = 0;

        laneVectors.forEach((laneEntry) => {
            const vecList = laneEntry && Array.isArray(laneEntry.vectors) ? laneEntry.vectors : [];
            // Ensure vecList ordered so we can grab a representative Y coordinate
            vecList.sort((a, b) => a.group.position.x - b.group.position.x);
            const yPos = vecList.length ? vecList[0].group.position.y : 0;



            // --------------------------------------------------------------
            //  Launch horizontal merge tweens for the decorative vectors
            // --------------------------------------------------------------
            vecList.forEach((vec, idx) => {
                const destX = targetX + (idx - (NUM_HEAD_SETS_LAYER - 1) / 2) * ROW_SEGMENT_SPACING;
                const distance = Math.abs(vec.group.position.x - destX);
                const durationMs = (distance / (ROW_MERGE_HORIZ_SPEED * GLOBAL_ANIM_SPEED_MULT)) * 1000;
                if (durationMs > maxDurationMs) maxDurationMs = durationMs;

                if (typeof TWEEN !== 'undefined') {
                    new TWEEN.Tween(vec.group.position)
                        .to({ x: destX }, durationMs)
                        .easing(TWEEN.Easing.Quadratic.InOut)
                        .start();
                } else {
                    vec.group.position.x = destX;
                }
            });
        });

        // After all merge tweens are initiated, schedule the next phases.
        if (typeof TWEEN !== 'undefined') {
            this._scheduleAfterDelay(() => {
                this._transitionHeadColorsToFinal(HEAD_COLOR_TRANSITION_MS);
                this._scheduleAfterDelay(() => {
                    this._startVectorsThroughOutputProjection(laneVectors);
                }, MERGE_POST_COLOR_TRANSITION_DELAY_MS);
            }, maxDurationMs + MERGE_EXTRA_BUFFER_MS);
        } else {
            this._transitionHeadColorsToFinal(0);
            this._startVectorsThroughOutputProjection(laneVectors);
        }
    }
    
    // Run output-projection matrix animation and return vectors to residual stream.
    _startVectorsThroughOutputProjection(laneVectors) {
        // Combine decorative vectors in each lane into a single vector, then animate those combined vectors
        // Mark merge as complete prior to entering the output projection stage
        this._cancelOutputProjectionReturnWatchdog();
        this.rowMergePhase = 'merged';
        this.outputProjMatrixAnimationPhase = 'vectors_entering';
        this.outputProjMatrixReturnComplete = false;
        this._outputProjReturnCount = 0;
        const markOutputProjectionReturn = () => {
            this._outputProjReturnCount += 1;
            if (this._outputProjReturnCount >= this._outputProjReturnTargetCount) {
                this._cancelOutputProjectionReturnWatchdog();
                this.outputProjMatrixReturnComplete = true;
                this.outputProjMatrixAnimationPhase = 'completed';
            }
        };

        // Keep K/V visuals present during concatenation; they will be disposed
        // once the very last blue (Q) vector finishes its conveyor belt.

        const combinedVectors = [];

        // Central X coordinate for combined vector (align with first head center)
        const centerX = this.headsCentersX.length ? this.headsCentersX[0] : 0;

        // Determine central prism range for visible region.  Each prism now
        // represents 64 real dimensions, so calculate required prism count
        // and clamp to avoid negative indices.
        const visiblePrismCount = Math.min(this.vectorPrismCount, Math.ceil(this.outputVectorLength / PRISM_DIMENSIONS_PER_UNIT));
        const startVisibleIdx = Math.max(0, Math.floor((this.vectorPrismCount - visiblePrismCount) / 2));
        const endVisibleIdx = startVisibleIdx + visiblePrismCount - 1;

        laneVectors.forEach((laneEntry, laneKey) => {
            const vecList = Array.isArray(laneEntry)
                ? laneEntry
                : (laneEntry && Array.isArray(laneEntry.vectors) ? laneEntry.vectors : null);
            if (!vecList || vecList.length === 0) return;
            const laneZ = Array.isArray(laneEntry)
                ? (Number.isFinite(laneKey) ? laneKey : vecList[0].group.position.z)
                : (Number.isFinite(laneEntry.laneZ) ? laneEntry.laneZ : vecList[0].group.position.z);
            const laneIndex = Array.isArray(laneEntry)
                ? this._resolveLaneIndexFromZ(laneZ)
                : (Number.isFinite(laneEntry.laneIndex) ? laneEntry.laneIndex : this._resolveLaneIndexFromZ(laneZ));

            // Ensure vecList is sorted by X position
            vecList.sort((a, b) => a.group.position.x - b.group.position.x);

            // Build combined raw data by concatenating the 64-dim slices of each decorative vector (preserves lane-specific data)
            const combinedRaw = [];
            vecList.forEach(v => {
                const slice = v.rawData.slice(startVisibleIdx, startVisibleIdx + visiblePrismCount);
                combinedRaw.push(...slice);
            });

            // Ensure the final length is exactly the configured prism count
            if (combinedRaw.length < this.vectorPrismCount) {
                while (combinedRaw.length < this.vectorPrismCount) combinedRaw.push(0);
            } else if (combinedRaw.length > this.vectorPrismCount) {
                combinedRaw.length = this.vectorPrismCount;
            }

            const spawnPos = new THREE.Vector3(centerX, vecList[0].group.position.y, laneZ);
            const combinedVec = new VectorVisualizationInstancedPrism(
                combinedRaw,
                spawnPos,
                3,
                this.vectorPrismCount
            );

            // ------------------------------------------------------------------
            //  Re-colour combined vector with smooth gradient across its 12 prisms
            // ------------------------------------------------------------------
            // Copy colours from each decorative vector's visible prism so the
            // combined vector visually matches its inputs (avoids a colour
            // shift before the Output-Projection matrix).
            const tmpColor = new THREE.Color();
            const destCS = combinedVec.mesh?.geometry?.getAttribute?.('colorStart');
            const destCE = combinedVec.mesh?.geometry?.getAttribute?.('colorEnd');
            const destIC = combinedVec.mesh?.instanceColor;
            vecList.forEach((srcVec, i) => {
                const srcPrismIdx  = Math.min(startVisibleIdx, Math.max(0, (srcVec.instanceCount || this.vectorPrismCount) - 1));
                const destPrismIdx = Math.min(i, Math.max(0, (combinedVec.instanceCount || this.vectorPrismCount) - 1));
                const srcCS = srcVec.mesh?.geometry?.getAttribute?.('colorStart');
                const srcCE = srcVec.mesh?.geometry?.getAttribute?.('colorEnd');
                if (srcCS && srcCE && destCS && destCE) {
                    const srcIdx3 = srcPrismIdx * 3;
                    const destIdx3 = destPrismIdx * 3;
                    destCS.array[destIdx3] = srcCS.array[srcIdx3];
                    destCS.array[destIdx3 + 1] = srcCS.array[srcIdx3 + 1];
                    destCS.array[destIdx3 + 2] = srcCS.array[srcIdx3 + 2];
                    destCE.array[destIdx3] = srcCE.array[srcIdx3];
                    destCE.array[destIdx3 + 1] = srcCE.array[srcIdx3 + 1];
                    destCE.array[destIdx3 + 2] = srcCE.array[srcIdx3 + 2];
                    if (destIC && destIC.array) {
                        destIC.array[destIdx3] = srcCS.array[srcIdx3];
                        destIC.array[destIdx3 + 1] = srcCS.array[srcIdx3 + 1];
                        destIC.array[destIdx3 + 2] = srcCS.array[srcIdx3 + 2];
                    }
                } else if (srcVec.mesh && srcVec.mesh.getColorAt && combinedVec.mesh) {
                    srcVec.mesh.getColorAt(srcPrismIdx, tmpColor);
                    combinedVec.mesh.setColorAt(destPrismIdx, tmpColor);
                }
            });
            if (destCS) destCS.needsUpdate = true;
            if (destCE) destCE.needsUpdate = true;
            if (destIC) destIC.needsUpdate = true;

            this.parentGroup.add(combinedVec.group);
            // Do NOT attach a trail yet. We only want a horizontal trail
            // when returning to the residual stream (no vertical segments).

            combinedVectors.push({ vec: combinedVec, laneZ, laneIndex });

            // Hide original decorative vectors
            vecList.forEach(v => { v.group.visible = false; });


        });

        if (combinedVectors.length === 0) {
            console.warn("No combined vectors created for output projection animation");
            this._outputProjReturnTargetCount = 0;
            this._completeOutputProjectionFallback('no combined vectors for output projection', {
                fallbackLanes: true,
                onlyIncompleteLanes: false,
                logReason: false
            });
            return;
        }

        this._outputProjReturnTargetCount = combinedVectors.length;

        // Store for later reference
        this.outputProjMatrixVectors = combinedVectors.map(obj => obj.vec);

        // Matrix positions
        const matrixBottomY = this.outputProjMatrixCenterY - this.outputProjMatrixHeight / 2;
        const matrixTopY = this.outputProjMatrixCenterY + this.outputProjMatrixHeight / 2;
        const targetYAboveMatrix = matrixTopY + 30;

        // Durations
        const outputProjectionTuning = GPT2_LAYER_VISUAL_TUNING.mhsa.outputProjection;
        const minStageEnterDurationMs = (!this._skipToEndActive && Number.isFinite(outputProjectionTuning.minStageEnterDurationMs))
            ? outputProjectionTuning.minStageEnterDurationMs
            : 0;
        const minStageThroughDurationMs = (!this._skipToEndActive && Number.isFinite(outputProjectionTuning.minStageThroughDurationMs))
            ? outputProjectionTuning.minStageThroughDurationMs
            : 0;
        const minStageExitDurationMs = (!this._skipToEndActive && Number.isFinite(outputProjectionTuning.minStageExitDurationMs))
            ? outputProjectionTuning.minStageExitDurationMs
            : 0;
        const duration1 = Math.max(
            this._resolveSkipDuration(OUTPUT_PROJ_STAGE1_MS / GLOBAL_ANIM_SPEED_MULT),
            minStageEnterDurationMs
        );
        const duration2 = Math.max(
            this._resolveSkipDuration(OUTPUT_PROJ_STAGE2_MS / GLOBAL_ANIM_SPEED_MULT),
            minStageThroughDurationMs
        );
        const duration3 = Math.max(
            this._resolveSkipDuration(OUTPUT_PROJ_STAGE3_MS / GLOBAL_ANIM_SPEED_MULT),
            minStageExitDurationMs
        );

        if (typeof TWEEN === 'undefined') {
            console.warn("TWEEN not available for output projection matrix animation");
            this._completeOutputProjectionFallback('TWEEN unavailable for output projection', {
                fallbackLanes: true,
                onlyIncompleteLanes: false,
                logReason: false
            });
            return;
        }

        const horizDurRaw = (Math.abs(centerX) / (ANIM_HORIZ_SPEED * GLOBAL_ANIM_SPEED_MULT)) * 1000;
        const horizDurEstimate = Number.isFinite(horizDurRaw)
            ? this._resolveSkipDuration(horizDurRaw)
            : 0;
        const expectedReturnMs = Math.max(0, duration1 + duration2 + duration3 + horizDurEstimate);
        const watchdogDelayMs = Math.max(
            OUTPUT_PROJ_RETURN_WATCHDOG_MIN_MS,
            expectedReturnMs * 2 + OUTPUT_PROJ_RETURN_WATCHDOG_GRACE_MS
        );
        this._scheduleOutputProjectionReturnWatchdog(watchdogDelayMs);

        if (this._skipToEndActive && this._outputProjReturnTargetCount > 0) {
            const fallbackMs = this._resolveSkipDelay(400);
            this._scheduleAfterDelay(() => {
                if (this.outputProjMatrixReturnComplete) return;
                if (this._outputProjReturnCount >= this._outputProjReturnTargetCount) return;
                console.warn(
                    `MHSAAnimation: forcing output projection completion for layer ${this.layerIndex ?? '?'} `
                    + `(${this._outputProjReturnCount}/${this._outputProjReturnTargetCount}).`
                );
                this._completeOutputProjectionFallback('skip output projection return watchdog', {
                    fallbackLanes: true,
                    onlyIncompleteLanes: false,
                    logReason: false
                });
            }, fallbackMs);
        }

        combinedVectors.forEach((vecObj, idx) => {
            const vec = vecObj.vec;
            let returnMarkedForVector = false;
            const markReturnForVector = () => {
                if (returnMarkedForVector) return;
                returnMarkedForVector = true;
                markOutputProjectionReturn();
            };
            const laneZ = vecObj.laneZ;
            const laneIndex = Number.isFinite(vecObj.laneIndex)
                ? Math.floor(vecObj.laneIndex)
                : this._resolveLaneIndexFromZ(laneZ);
            const resolveLaneForVector = () => {
                const mappedLane = this._resolveLaneByIdentity(laneIndex, laneZ);
                if (mappedLane) return mappedLane;
                if (Array.isArray(this.currentLanes) && idx >= 0 && idx < this.currentLanes.length) {
                    return this.currentLanes[idx] || null;
                }
                if (Array.isArray(this.currentLanes) && Number.isFinite(laneZ)) {
                    let nearestLane = null;
                    let nearestDist = Infinity;
                    for (let laneIdx = 0; laneIdx < this.currentLanes.length; laneIdx++) {
                        const lane = this.currentLanes[laneIdx];
                        if (!lane) continue;
                        const zPos = Number.isFinite(lane.zPos)
                            ? lane.zPos
                            : (lane.originalVec && lane.originalVec.group ? lane.originalVec.group.position.z : NaN);
                        if (!Number.isFinite(zPos)) continue;
                        const dist = Math.abs(zPos - laneZ);
                        if (dist < nearestDist) {
                            nearestDist = dist;
                            nearestLane = lane;
                        }
                    }
                    if (nearestLane) return nearestLane;
                }
                if (Array.isArray(this.currentLanes)) {
                    for (let laneIdx = 0; laneIdx < this.currentLanes.length; laneIdx++) {
                        const lane = this.currentLanes[laneIdx];
                        if (!lane || lane.ln2Phase === LN2_PHASE.DONE) continue;
                        if (lane.originalVec && lane.originalVec.group) return lane;
                    }
                }
                return null;
            };

            new TWEEN.Tween(vec.group.position)
                .to({ y: matrixBottomY }, duration1)
                .easing(TWEEN.Easing.Quadratic.InOut)
                .onUpdate(() => {
                    // Keep trail brightness consistent during stage 1 rise
                    if (vec && vec.userData && vec.userData.trail) {
                        vec.userData.trail.update(vec.group.position);
                    }
                })
                .onComplete(() => {
                    if (idx === 0) {
                        this._animateOutputMatrixBrightening(duration2);
                    }
                    new TWEEN.Tween(vec.group.position)
                        .to({ y: matrixTopY }, duration2)
                        .onStart(() => {
                            // Apply transformation INSIDE the projection matrix
                            const lane = resolveLaneForVector();
                            const outputData = (this.activationSource && lane && Number.isFinite(this.layerIndex))
                                ? this.activationSource.getAttentionOutputProjection(this.layerIndex, lane.tokenIndex, this.vectorPrismCount)
                                : null;
                            const newRaw = outputData || this._generateRawDataWithSwitchPoints(30);
                            const cacheKeyData = outputData || null;
                            const numKeyColors = Math.min(30, Math.max(1, newRaw.length || 1));
                            vec.applyProcessedVisuals(
                                newRaw,
                                NUM_HEAD_SETS_LAYER * this.outputVectorLength, // 12 * 64  = 768 visible output units
                                { numKeyColors, generationOptions: null },
                                { setHiddenToBlack: false },
                                cacheKeyData
                            );
                            const label = lane && lane.tokenLabel
                                ? `Attention Output Projection - ${lane.tokenLabel}`
                                : 'Attention Output Projection';
                            const activationData = buildActivationData({
                                label,
                                stage: 'attention.output_projection',
                                layerIndex: this.layerIndex,
                                tokenIndex: lane ? lane.tokenIndex : undefined,
                                tokenLabel: lane ? lane.tokenLabel : undefined,
                                values: newRaw,
                            });
                            applyActivationDataToVector(vec, activationData, label);
                        })

                        .onUpdate(() => {
                            // Continue updating trail through matrix travel
                            if (vec && vec.userData && vec.userData.trail) {
                                vec.userData.trail.update(vec.group.position);
                            }
                        })
                        .onComplete(() => {
                            const extraRise = 30; // additional upward distance
                            const finalCombinedY = targetYAboveMatrix + extraRise;

                            // Final rise after transformation done
                            new TWEEN.Tween(vec.group.position)
                                .to({ y: finalCombinedY }, duration3)
                                .easing(TWEEN.Easing.Quadratic.InOut)
                                .onStart(() => {
                                    // Start a trail as the vector exits the output-projection matrix
                                    try {
                                        vec.userData = vec.userData || {};
                                        if (!vec.userData.trail) {
                                            const tr = new StraightLineTrail(
                                                this.parentGroup,
                                                undefined,
                                                undefined,
                                                undefined,
                                                OUTPUT_PROJ_RETURN_TRAIL_OPACITY,
                                                TRAIL_MIN_SEGMENT_DISTANCE
                                            );
                                            tr.start(vec.group.position.clone());
                                            vec.userData.trail = tr;
                                        }
                                        const outputTrail = vec.userData && vec.userData.trail;
                                        if (outputTrail && typeof outputTrail.setBaseOpacity === 'function') {
                                            outputTrail.setBaseOpacity(OUTPUT_PROJ_RETURN_TRAIL_OPACITY);
                                        }
                                    } catch (_) { /* optional visual */ }
                                })
                                .onUpdate(() => {
                                    if (vec && vec.userData && vec.userData.trail) {
                                        vec.userData.trail.update(vec.group.position);
                                    }
                                })
                                .onComplete(() => {
                                    const riseTrail = vec && vec.userData && vec.userData.trail;
                                    if (riseTrail) {
                                        if (typeof riseTrail.snapLastPointTo === 'function') {
                                            riseTrail.snapLastPointTo(vec.group.position);
                                        } else if (typeof riseTrail.update === 'function') {
                                            riseTrail.update(vec.group.position);
                                        }
                                    }
                                    // Horizontal move back to residual stream centre (x = 0),
                                    // then perform the addition with the lane's original vector
                                    const horizDistance = Math.abs(vec.group.position.x);
                                    const horizDurRaw = (horizDistance / (ANIM_HORIZ_SPEED * GLOBAL_ANIM_SPEED_MULT)) * 1000;
                                    const horizDur = Number.isFinite(horizDurRaw) ? Math.max(0, horizDurRaw) : 0;

                                    new TWEEN.Tween(vec.group.position)
                                        .to({ x: 0 }, horizDur)
                                        .easing(TWEEN.Easing.Quadratic.InOut)
                                        .onStart(() => {
                                            vec.group.position.y = finalCombinedY;
                                            const tr = vec && vec.userData && vec.userData.trail;
                                            if (tr) {
                                                if (typeof tr.snapLastPointTo === 'function') {
                                                    tr.snapLastPointTo(vec.group.position);
                                                } else if (typeof tr.update === 'function') {
                                                    tr.update(vec.group.position);
                                                }
                                            }
                                        })
                                        .onUpdate(() => {
                                            vec.group.position.y = finalCombinedY;
                                            // Continue updating the same trail during horizontal travel
                                            const tr = vec && vec.userData && vec.userData.trail;
                                            if (tr) tr.update(vec.group.position);
                                        })

                                        .onComplete(() => {
                                            try {
                                                vec.group.position.x = 0;
                                                vec.group.position.y = finalCombinedY;
                                                // Freeze the trail into static segments so the horizontal
                                                // return path remains visible, then remove live trail ref
                                                try {
                                                    const tr = vec && vec.userData && vec.userData.trail;
                                                    if (tr) {
                                                        if (typeof tr.snapLastPointTo === 'function') {
                                                            tr.snapLastPointTo(vec.group.position);
                                                        } else if (typeof tr.update === 'function') {
                                                            tr.update(vec.group.position);
                                                        }
                                                        mergeTrailsIntoLineSegments(
                                                            [tr],
                                                            this.parentGroup,
                                                            undefined,
                                                            (typeof tr._lineWidth === 'number') ? tr._lineWidth : undefined,
                                                            (typeof tr._opacity === 'number') ? tr._opacity : undefined,
                                                            null
                                                        );
                                                        if (vec && vec.userData) delete vec.userData.trail;
                                                    }
                                                } catch (_) { /* optional visual */ }
                                                if (this.currentLanes) {
                                                    const matchingLane = resolveLaneForVector();
                                                    if (matchingLane && matchingLane.originalVec) {
                                                        const postData = (this.activationSource && Number.isFinite(this.layerIndex))
                                                            ? this.activationSource.getPostAttentionResidual(this.layerIndex, matchingLane.tokenIndex, this.vectorPrismCount)
                                                            : null;
                                                        if (postData) {
                                                            matchingLane.additionTargetData = postData;
                                                        }
                                                        this._startAdditionAnimation(matchingLane.originalVec, vec, matchingLane, () => {
                                                            if (postData) {
                                                                const label = matchingLane.tokenLabel
                                                                    ? `Post-Attention Residual - ${matchingLane.tokenLabel}`
                                                                    : 'Post-Attention Residual';
                                                                matchingLane.originalVec.rawData = postData.slice();
                                                                const numKeyColors = Math.min(30, Math.max(1, postData.length || 1));
                                                                matchingLane.originalVec.updateKeyColorsFromData(
                                                                    matchingLane.originalVec.rawData,
                                                                    numKeyColors,
                                                                    null,
                                                                    postData
                                                                );
                                                                const activationData = buildActivationData({
                                                                    label,
                                                                    stage: 'residual.post_attention',
                                                                    layerIndex: this.layerIndex,
                                                                    tokenIndex: matchingLane.tokenIndex,
                                                                    tokenLabel: matchingLane.tokenLabel,
                                                                    values: postData,
                                                                });
                                                                applyActivationDataToVector(matchingLane.originalVec, activationData, label);
                                                            }
                                                        });
                                                    } else {
                                                        console.warn(`[MHSAAnimation] Failed to map output vector back to lane (layer=${this.layerIndex}, laneIndex=${laneIndex}, laneZ=${laneZ}).`);
                                                    }
                                                }
                                            } catch (err) {
                                                console.error(
                                                    `[MHSAAnimation] Output projection return failed `
                                                    + `(layer=${this.layerIndex ?? '?'}, laneIndex=${laneIndex}, laneZ=${laneZ})`,
                                                    err
                                                );
                                            } finally {
                                                markReturnForVector();
                                            }
                                        })
                                        .start();
                                })
                                .start();
                        })
                        .start();
                })
                .start();
        });

        logMhsaDebug('Starting animation of combined lane vectors through output projection matrix');
    }

    _shouldPreserveKVCacheVectors() {
        return !!appState.kvCacheModeEnabled;
    }

    _isKvCacheModeEnabled() {
        return !!appState.kvCacheModeEnabled;
    }

    _markSkipVisible(target) {
        if (!target) return;
        target.userData = target.userData || {};
        target.userData.skipVisible = true;
    }

    _normalizeVectorMaterialVisible(mat, opacity = 1) {
        if (!mat) return;
        const clampedOpacity = Number.isFinite(opacity)
            ? THREE.MathUtils.clamp(opacity, 0, 1)
            : 1;
        const fullyOpaque = clampedOpacity >= 0.999;
        mat.transparent = !fullyOpaque;
        mat.opacity = clampedOpacity;
        if ('depthWrite' in mat) mat.depthWrite = fullyOpaque;
        if ('depthTest' in mat) mat.depthTest = true;
        if ('alphaTest' in mat) mat.alphaTest = 0;
        mat.side = THREE.DoubleSide;
        mat.needsUpdate = true;
    }

    _setInstancedMeshVisible(mesh, { opacity = 1 } = {}) {
        if (!mesh || !mesh.isMesh) return;
        mesh.frustumCulled = false;
        if (mesh.material) {
            const mats = Array.isArray(mesh.material) ? mesh.material : [mesh.material];
            mats.forEach((mat) => {
                this._normalizeVectorMaterialVisible(mat, opacity);
            });
        }
        this._markSkipVisible(mesh);
        if (mesh.parent) this._markSkipVisible(mesh.parent);
        mesh.visible = true;
        if (mesh.parent) mesh.parent.visible = true;
    }

    _setVectorVisible(vec, { opacity = 1 } = {}) {
        if (!vec) return;
        try {
            if (vec.group) {
                this._markSkipVisible(vec.group);
                vec.group.visible = true;
            }
            if (vec.mesh) {
                this._markSkipVisible(vec.mesh);
                vec.mesh.visible = true;
                vec.mesh.frustumCulled = false;
                const mats = Array.isArray(vec.mesh.material) ? vec.mesh.material : [vec.mesh.material];
                mats.forEach((mat) => {
                    this._normalizeVectorMaterialVisible(mat, opacity);
                });
            }
            if (vec.isBatchedVectorRef && vec._batch?.mesh) {
                this._setInstancedMeshVisible(vec._batch.mesh, { opacity });
            }
            if (vec.isBatchedVectorRef && vec._batch?.syncAll) {
                vec._batch.syncAll();
            }
        } catch (_) { /* best effort */ }
    }

    _markVectorAsCachedKv(vec, category = null) {
        if (!vec) return;
        const resolvedCategory = String(
            category
            || vec?.userData?.vectorCategory
            || vec?.userData?.qkvProcessedCategory
            || 'K'
        ).toUpperCase() === 'V' ? 'V' : 'K';
        const label = resolvedCategory === 'V' ? 'Cached Value Vector' : 'Cached Key Vector';

        vec.userData = vec.userData || {};
        vec.userData.cachedKv = true;
        vec.userData.kvCachePersistent = true;
        vec.userData.vectorCategory = resolvedCategory;

        if (vec.group) {
            vec.group.userData = vec.group.userData || {};
            vec.group.userData.cachedKv = true;
            vec.group.userData.kvCachePersistent = true;
            vec.group.userData.vectorCategory = resolvedCategory;
            vec.group.userData.label = label;
        }
        if (vec.mesh) {
            vec.mesh.userData = vec.mesh.userData || {};
            vec.mesh.userData.cachedKv = true;
            vec.mesh.userData.kvCachePersistent = true;
            vec.mesh.userData.vectorCategory = resolvedCategory;
            vec.mesh.userData.label = label;
        }
    }

    _snapCachedKeyVectorsUnderValues(lanes = null, { cachedOnly = false } = {}) {
        const laneList = Array.isArray(lanes) ? lanes : [];
        if (!laneList.length) return;

        const targetXByHead = new Map();
        laneList.forEach((lane) => {
            if (!Array.isArray(lane?.sideCopies)) return;
            lane.sideCopies.forEach((sc) => {
                if (!sc || sc.type !== 'V') return;
                const headIndex = Number.isFinite(sc.headIndex)
                    ? Math.floor(sc.headIndex)
                    : (Number.isFinite(sc?.vec?.userData?.headIndex) ? Math.floor(sc.vec.userData.headIndex) : null);
                if (!Number.isFinite(headIndex) || targetXByHead.has(headIndex)) return;
                const targetX = Number.isFinite(sc.targetX)
                    ? sc.targetX
                    : (Number.isFinite(sc?.vec?.group?.position?.x) ? sc.vec.group.position.x : null);
                if (Number.isFinite(targetX)) {
                    targetXByHead.set(headIndex, targetX);
                }
            });
        });
        if (!targetXByHead.size) return;

        const touchedBatches = new Set();
        laneList.forEach((lane) => {
            if (cachedOnly && lane?.bootstrapFromActivation === true) return;
            if (!Array.isArray(lane?.upwardCopies)) return;
            lane.upwardCopies.forEach((kVec, headIndex) => {
                if (!kVec || !kVec.group || !kVec.group.position) return;
                if (cachedOnly) {
                    const isCached = !!(
                        kVec.userData?.cachedKv
                        || kVec.userData?.kvCachePersistent
                        || kVec.group?.userData?.cachedKv
                        || kVec.group?.userData?.kvCachePersistent
                    );
                    if (!isCached) return;
                }
                const targetX = targetXByHead.get(headIndex);
                if (!Number.isFinite(targetX)) return;
                kVec.group.position.x = targetX;
                if (kVec.isBatchedVectorRef && kVec._batch && typeof kVec._batch.syncAll === 'function') {
                    touchedBatches.add(kVec._batch);
                }
            });
        });

        touchedBatches.forEach((batch) => {
            try {
                batch.syncAll();
            } catch (_) { /* optional cache snap sync */ }
        });
    }

    _preserveKVVectorsForCache() {
        const lanes = Array.isArray(this.currentLanes) ? this.currentLanes : [];
        const cachedLanes = Array.isArray(this._cachedKvEntries) ? this._cachedKvEntries : [];
        if (this._kvCacheVectorsPreserved) {
            this._snapCachedKeyVectorsUnderValues(lanes);
            this._snapCachedKeyVectorsUnderValues(cachedLanes, { cachedOnly: true });
            return;
        }
        this._kvCacheVectorsPreserved = true;

        if (this._mergedGroupsByHead && typeof this._mergedGroupsByHead.forEach === 'function') {
            this._mergedGroupsByHead.forEach((grp) => {
                if (grp?.K?.children?.[0]) this._setInstancedMeshVisible(grp.K.children[0], { opacity: 1 });
                if (grp?.V?.children?.[0]) this._setInstancedMeshVisible(grp.V.children[0], { opacity: 1 });
            });
        }
        if (this.parentGroup && typeof this.parentGroup.getObjectByName === 'function') {
            const allK = this.parentGroup.getObjectByName('MergedAllK');
            const allV = this.parentGroup.getObjectByName('MergedAllV');
            if (allK?.children?.[0]) this._setInstancedMeshVisible(allK.children[0], { opacity: 1 });
            if (allV?.children?.[0]) this._setInstancedMeshVisible(allV.children[0], { opacity: 1 });
        }

        lanes.forEach((lane) => {
            if (Array.isArray(lane?.upwardCopies)) {
                lane.upwardCopies.forEach((kVec) => {
                    this._setVectorVisible(kVec, { opacity: 1 });
                    this._markVectorAsCachedKv(kVec, 'K');
                });
            }
            if (Array.isArray(lane?.sideCopies)) {
                lane.sideCopies.forEach((sc) => {
                    if (!sc || sc.type !== 'V') return;
                    this._setVectorVisible(sc.vec, { opacity: 1 });
                    this._markVectorAsCachedKv(sc.vec, 'V');
                });
            }
        });
        this._snapCachedKeyVectorsUnderValues(lanes);
        this._snapCachedKeyVectorsUnderValues(cachedLanes, { cachedOnly: true });
    }

    _finalizeKVVisualStateAfterConveyor() {
        if (this._shouldPreserveKVCacheVectors()) {
            this._preserveKVVectorsForCache();
            return;
        }
        this._disposeMergedKVGroups();
        this._disposeAllIndividualKandVVectorsImmediately();
    }

    // Hide all merged K/V instanced groups (both per-head and global-all)
    _hideMergedKVGroups() {
        // Per-head merged groups
        if (this._mergedGroupsByHead && typeof this._mergedGroupsByHead.forEach === 'function') {
            this._mergedGroupsByHead.forEach((grp) => {
                if (grp && grp.K) grp.K.visible = false;
                if (grp && grp.V) grp.V.visible = false;
            });
        }
        // Global merged groups (if created)
        if (this.parentGroup && typeof this.parentGroup.getObjectByName === 'function') {
            const allK = this.parentGroup.getObjectByName('MergedAllK');
            const allV = this.parentGroup.getObjectByName('MergedAllV');
            if (allK) allK.visible = false;
            if (allV) allV.visible = false;
        }
    }

    // Dispose and remove any merged K/V instanced groups (both per-head and global)
    _disposeMergedKVGroups() {
        const safeDisposeGroup = (group) => {
            if (!group) return;
            try {
                const inst = group.children && group.children[0];
                if (inst && inst.isMesh) {
                    if (inst.material) {
                        if (Array.isArray(inst.material)) {
                            inst.material.forEach((m) => { if (m && m.dispose) m.dispose(); });
                        } else if (inst.material.dispose) {
                            inst.material.dispose();
                        }
                    }
                    if (inst.geometry && inst.geometry.dispose) inst.geometry.dispose();
                }
                if (group.parent) {
                    group.parent.remove(group);
                } else if (this.parentGroup) {
                    this.parentGroup.remove(group);
                }
            } catch (_) { /* ignore */ }
        };

        // Per-head merged groups
        if (this._mergedGroupsByHead && typeof this._mergedGroupsByHead.forEach === 'function') {
            this._mergedGroupsByHead.forEach((grp) => {
                if (grp && grp.K) safeDisposeGroup(grp.K);
                if (grp && grp.V) safeDisposeGroup(grp.V);
            });
            try { this._mergedGroupsByHead.clear(); } catch (_) { /* ignore */ }
        }

        // Global merged groups (if created)
        if (this.parentGroup && typeof this.parentGroup.getObjectByName === 'function') {
            const allK = this.parentGroup.getObjectByName('MergedAllK');
            const allV = this.parentGroup.getObjectByName('MergedAllV');
            if (allK) safeDisposeGroup(allK);
            if (allV) safeDisposeGroup(allV);
        }

        // Reset tracking flags/collections
        try { if (this._mergedHeads && this._mergedHeads.clear) this._mergedHeads.clear(); } catch (_) { /* ignore */ }
        this._allFixedMerged = false;
    }

    // Poll self-attention animator until all blue (Q) conveyors across heads
    // have completed, then finalize K/V visuals exactly after the last blue
    // finishes its path.
    _waitAndDisposeMergedKVWhenBlueConveyorComplete() {
        // Prefer a direct completion callback from SelfAttentionAnimator
        try {
            if (this.selfAttentionAnimator && typeof this.selfAttentionAnimator.start === 'function') {
                this.selfAttentionAnimator.start(() => {
                    try { this._finalizeKVVisualStateAfterConveyor(); } catch (_) {}
                });
                return;
            }
        } catch (_) { /* ignore and fall back to polling */ }

        // Fallback: poll phase with a safety timeout
        const checkIntervalMs = 100;
        const maxWaitMs = 120000;
        let waited = 0;
        const tick = () => {
            try {
                if (!this.selfAttentionAnimator || this.selfAttentionAnimator.phase === 'complete') {
                    this._finalizeKVVisualStateAfterConveyor();
                    return;
                }
            } catch (_) { /* ignore */ }
            waited += checkIntervalMs;
            if (waited >= maxWaitMs) {
                try { this._finalizeKVVisualStateAfterConveyor(); } catch (_) {}
                return;
            }
            this._scheduleAfterDelay(tick, checkIntervalMs);
        };
        this._scheduleAfterDelay(tick, checkIntervalMs);
    }

    // Immediately hide all K (green) and V (red) vectors, including any merged
    // instanced groups and any leftover individual K/V copies in lanes.
    _hideAllKandVVectorsImmediately() {
        try { this._hideMergedKVGroups(); } catch (_) {}
        const lanes = this.currentLanes || [];
        lanes.forEach((lane) => {
            // Hide K upward copies for every head
            if (Array.isArray(lane.upwardCopies)) {
                lane.upwardCopies.forEach((kVec) => {
                    if (!kVec) return;
                    try {
                        if (kVec.mesh) kVec.mesh.visible = false;
                        if (kVec.group) kVec.group.visible = false;
                    } catch (_) {}
                });
            }
            // Hide V side copies
            if (Array.isArray(lane.sideCopies)) {
                lane.sideCopies.forEach((sc) => {
                    if (!sc || sc.type !== 'V') return;
                    const v = sc.vec;
                    if (!v) return;
                    try {
                        if (v.mesh) v.mesh.visible = false;
                        if (v.group) v.group.visible = false;
                    } catch (_) {}
                });
            }
        });
    }

    // Hide all Q vectors (blue side copies) so a manual skip can immediately clear conveyors.
    _hideAllQVectorsImmediately() {
        const lanes = this.currentLanes || [];
        lanes.forEach((lane) => {
            if (Array.isArray(lane.sideCopies)) {
                lane.sideCopies.forEach((sc) => {
                    if (!sc || sc.type !== 'Q') return;
                    const q = sc.vec;
                    if (!q) return;
                    try {
                        if (q.mesh) q.mesh.visible = false;
                        if (q.group) q.group.visible = false;
                    } catch (_) {}
                });
            }
        });
    }

    // Permanently dispose any remaining individual K (green) and V (red) vectors
    // across all lanes. Keeps the empty groups (for positional queries) but
    // removes meshes and their GPU resources so nothing remains visible.
    _disposeAllIndividualKandVVectorsImmediately() {
        const stripMeshKeepGroup = (vec) => {
            try {
                if (!vec) return;
                if (vec.mesh) {
                    try { if (vec.mesh.material) vec.mesh.material.dispose(); } catch (_) {}
                    try { if (vec.mesh.geometry) vec.mesh.geometry.dispose(); } catch (_) {}
                    if (vec.group) vec.group.remove(vec.mesh);
                    vec.mesh = null;
                }
                if (vec.group) vec.group.visible = false;
            } catch (_) { /* ignore */ }
        };
        const lanes = this.currentLanes || [];
        lanes.forEach((lane) => {
            // Upward K copies for each head
            if (Array.isArray(lane.upwardCopies)) {
                lane.upwardCopies.forEach((kVec, idx) => {
                    stripMeshKeepGroup(kVec);
                });
            }
            // Side V copies for each head
            if (Array.isArray(lane.sideCopies)) {
                lane.sideCopies.forEach((sc) => {
                    if (!sc || sc.type !== 'V') return;
                    stripMeshKeepGroup(sc.vec);
                });
            }
        });
    }

    /**
     * Skip the attention conveyor belt animation and jump straight into the concatenation phase.
     * Ensures weighted sums exist so concat/output-projection can proceed deterministically.
     */
    skipSelfAttentionAndStartConcat() {
        if (this.mhaPassThroughPhase !== 'mha_pass_through_complete') return;
        const rowMergePhase = this.rowMergePhase || 'not_started';
        const outputProjPhase = this.outputProjMatrixAnimationPhase || 'waiting';
        if (rowMergePhase !== 'not_started' || outputProjPhase !== 'waiting') {
            return;
        }
        if (this._skipMatrixColorsLocked && this._skipMatrixColorsPending) {
            this._applyFinalMatrixColorsImmediate();
            this._skipMatrixColorsPending = false;
        }
        const preserveTrails = !!this._skipToEndActive;
        const preserveKVCache = this._shouldPreserveKVCacheVectors();
        try {
            if (this.selfAttentionAnimator?.forceComplete) {
                this.selfAttentionAnimator.forceComplete({
                    preserveTrails,
                    createWeightedSums: true,
                    replaceWeightedSums: true,
                });
            }
        } catch (_) {}
        try { this._hideAllQVectorsImmediately(); } catch (_) {}
        if (preserveKVCache) {
            try { this._preserveKVVectorsForCache(); } catch (_) {}
        } else {
            try { this._hideAllKandVVectorsImmediately(); } catch (_) {}
        }
        const weightedDecoratives = this._getWeightedSumDecoratives();
        if (weightedDecoratives.length) {
            this._tempDecorativeVecs = weightedDecoratives;
            const fadeDuration = this._resolveSkipDuration(DECORATIVE_FADE_MS);
            this._restoreWeightedSumColors(fadeDuration, 0);
        } else {
            const fadeDuration = this._resolveSkipDuration(DECORATIVE_FADE_MS);
            this._spawnTempDecorativesFromK({ fadeDuration, fadeDelay: 0, clearExisting: true });
        }
        this.outputProjMatrixReturnComplete = false;
        this._outputProjReturnCount = 0;
        this._outputProjReturnTargetCount = 0;
        // Ensure decorative vectors exist before starting the merge/concat sequence
        if (!this._tempModeCompleted) {
            try { this._applyTempModeBehaviour(); this._tempModeCompleted = true; } catch (_) {}
        }
        if (this.rowMergePhase === 'not_started') {
            if (this._tempDecorativeVecs && this._tempDecorativeVecs.length) {
                this._startMergeToRowVectors();
            } else {
                // Skip fallback: if no decorative vectors exist, synthesize
                // post-attention residual handoff lanes so LN2 can continue.
                this._outputProjReturnTargetCount = 0;
                this._completeOutputProjectionFallback('skip concat had no decorative vectors', {
                    fallbackLanes: true,
                    onlyIncompleteLanes: false,
                    logReason: false
                });
            }
        }
    }
    
    _animateOutputMatrixBrightening(duration) {
        if (typeof TWEEN === 'undefined') return;
        const outputProjectionTuning = GPT2_LAYER_VISUAL_TUNING.mhsa.outputProjection;
        const minFlashDurationMs = (!this._skipToEndActive && Number.isFinite(outputProjectionTuning.minFlashDurationMs))
            ? outputProjectionTuning.minFlashDurationMs
            : 0;
        const effectiveDuration = Math.max(this._resolveSkipDuration(duration), minFlashDurationMs);
        
        this.outputProjMatrixAnimationPhase = 'vectors_inside';
        
        // Animation parameters
        const startColor = this.outputProjMatrixDefaultColor.clone();
        const brightColor = this.outputProjMatrixActiveColor.clone();
        const startEmissiveIntensity = Number.isFinite(outputProjectionTuning.startEmissiveIntensity)
            ? outputProjectionTuning.startEmissiveIntensity
            : 0.05;
        const peakEmissiveIntensity = Number.isFinite(outputProjectionTuning.peakEmissiveIntensity)
            ? outputProjectionTuning.peakEmissiveIntensity
            : 0.38;
        const endEmissiveIntensity = Number.isFinite(outputProjectionTuning.endEmissiveIntensity)
            ? outputProjectionTuning.endEmissiveIntensity
            : 0.14;

        // Ensure matrix begins in its dark resting state
        this.outputProjectionMatrix.setColor(startColor);
        this.outputProjectionMatrix.setEmissive(startColor, startEmissiveIntensity);

        // First brighten the matrix
        const state = {
            r: startColor.r,
            g: startColor.g,
            b: startColor.b,
            emissiveIntensity: startEmissiveIntensity
        };
        const currentColor = this._outputProjColorScratch;
        
        new TWEEN.Tween(state)
            .to({ 
                r: brightColor.r, 
                g: brightColor.g, 
                b: brightColor.b,
                emissiveIntensity: peakEmissiveIntensity
            }, effectiveDuration * 0.6) // 60% of the total duration
            .easing(TWEEN.Easing.Quadratic.InOut)
            .onUpdate(() => {
                currentColor.setRGB(state.r, state.g, state.b);
                this.outputProjectionMatrix.setColor(currentColor);
                this.outputProjectionMatrix.setEmissive(currentColor, state.emissiveIntensity);
            })
            .onComplete(() => {
                // Then dim slightly to the final state
                new TWEEN.Tween(state)
                    .to({ emissiveIntensity: endEmissiveIntensity }, effectiveDuration * 0.4) // 40% of the total duration
                    .easing(TWEEN.Easing.Quadratic.InOut)
                    .onUpdate(() => {
                        this.outputProjectionMatrix.setEmissive(brightColor, state.emissiveIntensity);
                    })
                    .onComplete(() => {
                        this.outputProjMatrixAnimationPhase = 'completed';
                        // Ensure final opacity is fully opaque
                        this.outputProjectionMatrix.setMaterialProperties({ opacity: 1.0, transparent: false });
                    })
                    .start();
            })
            .start();
    }

    _transitionHeadColorsToFinal(duration) {
        if (this._headColorsFinalized) return;
        const effectiveDuration = this._resolveSkipDuration(duration);
        if (typeof TWEEN === 'undefined') {
            console.warn("TWEEN not available for final head color transition.");
            // Set colors directly if TWEEN is not available
            for (let i = 0; i < NUM_HEAD_SETS_LAYER; i++) {
                const qMatrix = this.mhaVisualizations[i * 3];
                const kMatrix = this.mhaVisualizations[i * 3 + 1];
                const vMatrix = this.mhaVisualizations[i * 3 + 2];

                if (qMatrix) qMatrix.setColor(this.finalHeadColors.Q);
                if (kMatrix) kMatrix.setColor(this.finalHeadColors.K);
                if (vMatrix) vMatrix.setColor(this.finalHeadColors.V);
            }
            this._headColorsFinalized = true;
            return;
        }

        this._headColorsFinalized = true;
        const finalQColor = this.finalHeadColors.Q;
        const finalKColor = this.finalHeadColors.K;
        const finalVColor = this.finalHeadColors.V;
        for (let i = 0; i < NUM_HEAD_SETS_LAYER; i++) {
            const qMatrix = this.mhaVisualizations[i * 3];
            const kMatrix = this.mhaVisualizations[i * 3 + 1];
            const vMatrix = this.mhaVisualizations[i * 3 + 2];

            // Ensure matrices are fully opaque before colour tween begins.
            [qMatrix, kMatrix, vMatrix].forEach(m => {
                if (m) m.setMaterialProperties({ opacity: 1.0, transparent: false });
            });

            if (qMatrix && qMatrix.mesh && qMatrix.mesh.material) {
                const initialQColor = qMatrix.mesh.material.color.clone();
                new TWEEN.Tween(initialQColor)
                    .to(finalQColor, effectiveDuration)
                    .easing(TWEEN.Easing.Quadratic.InOut)
                    .onUpdate(() => {
                        qMatrix.setColor(initialQColor);
                        qMatrix.setEmissive(initialQColor, QKV_FINAL_MATRIX_EMISSIVE_INTENSITY); // Subtle emissiveness
                    })
                    .start();
            }

            if (kMatrix && kMatrix.mesh && kMatrix.mesh.material) {
                const initialKColor = kMatrix.mesh.material.color.clone();
                new TWEEN.Tween(initialKColor)
                    .to(finalKColor, effectiveDuration)
                    .easing(TWEEN.Easing.Quadratic.InOut)
                    .onUpdate(() => {
                        kMatrix.setColor(initialKColor);
                        kMatrix.setEmissive(initialKColor, QKV_FINAL_MATRIX_EMISSIVE_INTENSITY); // Subtle emissiveness
                    })
                    .start();
            }

            if (vMatrix && vMatrix.mesh && vMatrix.mesh.material) {
                const initialVColor = vMatrix.mesh.material.color.clone();
                new TWEEN.Tween(initialVColor)
                    .to(finalVColor, effectiveDuration)
                    .easing(TWEEN.Easing.Quadratic.InOut)
                    .onUpdate(() => {
                        vMatrix.setColor(initialVColor);
                        vMatrix.setEmissive(initialVColor, QKV_FINAL_MATRIX_EMISSIVE_INTENSITY); // Subtle emissiveness
                    })
                    .start();
            }
        }
        logMhsaDebug('MHSAAnimation: Initiated final head color transitions.');
    }

    // ----------------------------------------------------------------------
    // Helper: Generate raw data with switch points (similar to earlier)
    // ----------------------------------------------------------------------
    _generateRawDataWithSwitchPoints(numSwitchPoints = 30) {
        const raw = [];
        // Clamp switch point count to available prisms to avoid infinite loop.
        numSwitchPoints = Math.min(numSwitchPoints, this.vectorPrismCount);
        logRandomColorDebug('MHSAAnimation.generateRawDataWithSwitchPoints', {
            layerIndex: this.layerIndex,
            vectorPrismCount: this.vectorPrismCount,
            numSwitchPoints
        });
        const switchPoints = new Set();
        while (switchPoints.size < numSwitchPoints) {
            const idx = Math.floor(Math.random() * this.vectorPrismCount);
            switchPoints.add(idx);
        }
        const sortedSwitches = Array.from(switchPoints).sort((a, b) => a - b);
        let curVal = Math.random() * 2 - 1;
        let nextSwitch = sortedSwitches.shift();
        for (let i = 0; i < this.vectorPrismCount; i++) {
            if (i === nextSwitch) {
                curVal = Math.random() * 2 - 1;
                nextSwitch = sortedSwitches.shift();
            }
            raw.push(curVal);
        }
        return raw;
    }

    _resolveSkipDelay(delayMs) {
        return resolveSkipDelay(this._skipToEndActive, delayMs);
    }

    _resolveSkipDuration(durationMs) {
        return resolveSkipDuration(this._skipToEndActive, durationMs);
    }

    _scheduleAfterDelay(callback, delayMs) {
        return scheduleAfterDelay({
            callback,
            delayMs,
            skipToEndActive: this._skipToEndActive,
            scheduledDelayTweens: this._scheduledDelayTweens,
            scheduledTimeoutIds: this._scheduledTimeoutIds,
            tweenLib: typeof TWEEN !== 'undefined' ? TWEEN : null,
            onError: (err) => console.error(err)
        });
    }

    _clearScheduledDelays() {
        clearScheduledDelays(this._scheduledDelayTweens, this._scheduledTimeoutIds);
    }

    // ----------------------------------------------------------------------
    // Helper: Addition animation between two InstancedPrism vectors
    // ----------------------------------------------------------------------
    _startAdditionAnimation(sourceVec, targetVec, lane, onComplete = null, options = null) {
        // Initiate prism-by-prism addition animation where prisms from sourceVec
        // move into their corresponding positions in targetVec.
        // The lane object is forwarded so the helper can update lane state
        // (stopRise flags, phase transitions, etc.).
        const finalData = lane && lane.additionTargetData ? lane.additionTargetData : null;
        const cameraHoldAfterAdditionMs = Number.isFinite(options?.cameraHoldAfterAdditionMs)
            ? Math.max(0, options.cameraHoldAfterAdditionMs)
            : 140;
        if (lane && lane.additionTargetData) {
            delete lane.additionTargetData;
        }
        startPrismAdditionAnimation(sourceVec, targetVec, lane, () => {
            if (typeof onComplete === 'function') {
                try {
                    onComplete();
                } catch (_) { /* no-op */ }
            }
        }, {
            finalData,
            cameraHoldAfterAdditionMs,
            progressTarget: lane,
            progressKey: 'mhsaResidualAddProgress'
        });
        // Don't force absolute positions here – vectors should keep their
        // natural flow handled by the tween callbacks inside the helper.
    }
}
