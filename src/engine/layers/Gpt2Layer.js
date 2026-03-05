import * as THREE from 'three';
import BaseLayer from '../BaseLayer.js';
import { LayerNormalizationVisualization } from '../../components/LayerNormalizationVisualization.js';
import { WeightMatrixVisualization } from '../../components/WeightMatrixVisualization.js';
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
import { applyMatrixLabel, applyMatrixMaterialTweaks } from '../../utils/matrixVisualUtils.js';
import {
    applyVectorData,
    copyVectorAppearance,
    freezeStaticTransforms,
    geluApprox,
    LN_INTERNAL_TRAIL_MIN_SEGMENT
} from './gpt2LayerUtils.js';
import { updateLayerNormVisualState } from './gpt2LayerNormVisuals.js';
import {
    buildSingleLane,
    createAdditionPlaceholders,
    createFreshLanes,
    createLanesFromExternal,
    LN_PARAM_MONOCHROME
} from './gpt2LaneBuilder.js';
import {
    HORIZ_PHASE,
    LN2_PHASE,
    isAllowedLanePhaseTransition
} from './gpt2LanePhases.js';
import {
    buildDebugVectorSum,
    getLaneProgressSignature
} from './gpt2LaneWatchdogUtils.js';
import { logLayerNormVectorDump } from './gpt2LayerDebugUtils.js';
import { GPT2_LAYER_VISUAL_TUNING } from '../../utils/visualTuningProfiles.js';
import {
    applyLayerNormParamVectorForLayer,
    createPrismVectorForLayer,
    getAttentionOutputProjectionDataForLane,
    getEmbeddingDataForLane,
    getLayerIncomingDataForLane,
    getLn1DataForLane,
    getLn2DataForLane,
    getMlpActivationDataForLane,
    getMlpDownDataForLane,
    getMlpUpDataForLane,
    getPostAttentionResidualDataForLane,
    getPostMlpResidualDataForLane,
    resolveActiveLaneLayoutIndices,
    resolveBaseVectorLength,
    resolveInstanceCountFromData,
    resolveLaneLayoutCount,
    resolveLayerNormParamDataForLayer,
    resolveTokenIndexForLane,
    resolveTokenLabelForLayer
} from './gpt2LayerDataAccess.js';


// Slightly reduced spacing between stacked layers for a tighter layout.
// Keep this just above the per-layer vertical extent so MLP tops don't collide.
const DEFAULT_LAYER_STACK_SPACING = LAYER_STACK_SPACING_Y;
// Reusable scratch vector to avoid per-frame allocations when working with
// world-space trail coordinates.
const TMP_WORLD_POS = new THREE.Vector3();

// Shared colour constants reused across the layer to avoid per-frame
// allocations inside the animation loop.
const COLOR_DARK_GRAY = new THREE.Color(GPT2_LAYER_VISUAL_TUNING.layerNorm.inactiveColor);
const COLOR_LIGHT_YELLOW = new THREE.Color(GPT2_LAYER_VISUAL_TUNING.layerNorm.activeColor);
const COLOR_BRIGHT_YELLOW = new THREE.Color(GPT2_LAYER_VISUAL_TUNING.layerNorm.finalColor);
const COLOR_WHITE = new THREE.Color(0xffffff);
const COLOR_INACTIVE_COMPONENT = new THREE.Color(INACTIVE_COMPONENT_COLOR);

const TMP_LN_TRAIL_POS = new THREE.Vector3();
const SKIP_VISIBILITY_REFRESH_MS = 56;
const SKIP_VISIBILITY_MESH_INDEX_REFRESH_MS = 220;

const MLP_REFLECTIVITY_TWEAKS = {
    // Match the same reflectivity profile used for QKV matrices.
    roughnessMin: 0.45,
    metalnessMax: 0.62,
    clearcoatMax: 0.35,
    clearcoatRoughnessMin: 0.45,
    iridescenceMax: 0.2,
    envMapIntensityMax: 0.9
};

const POS_ADD_STALL_TIMEOUT_MS = 12000;
const POS_PASS_START_PAUSE_MS = 140;
const FIRST_LAYER_LN1_START_PAUSE_MS = 180;
const FIRST_LAYER_LN1_RISE_SPEED_MULT = 0.48;
const LANE_PHASE_STALL_TIMEOUT_MS_SKIP = 3000;
const LANE_PHASE_STALL_TIMEOUT_MS_NORMAL = 4500;
// Allow extra headroom during debug-heavy sessions so watchdog fallback does not
// fire purely because console instrumentation slowed animation frames.
const LN2_HANDOFF_STALL_TIMEOUT_MS_NORMAL = 7000;
// Long hidden-tab / focus gaps advance performance.now() while updates are paused.
// Reset watchdog baselines after such gaps to avoid false "stalled" recoveries
// that can spawn fallback vectors/trails on top of active ones.
const WATCHDOG_FRAME_GAP_RESET_MS = 900;
const POST_ATTENTION_LN2_PRE_RISE_SPEED_MULT = 1.25;
const MULTIPLY_TRANSITION_DURATION_MS = 160;
const MULTIPLY_SOURCE_SHRINK = 0.94;
const MLP_POST_PASS_THROUGH_FINAL_EMISSIVE = GPT2_LAYER_VISUAL_TUNING.mlp.postPassFinalEmissiveIntensity;
const MLP_MATRIX_FLASH_START_EMISSIVE = Number.isFinite(GPT2_LAYER_VISUAL_TUNING.mlp.flashStartEmissiveIntensity)
    ? GPT2_LAYER_VISUAL_TUNING.mlp.flashStartEmissiveIntensity
    : 0.04;
const MLP_MATRIX_FLASH_PEAK_EMISSIVE_BASE = Number.isFinite(GPT2_LAYER_VISUAL_TUNING.mlp.flashPeakEmissiveIntensity)
    ? GPT2_LAYER_VISUAL_TUNING.mlp.flashPeakEmissiveIntensity
    : 0.4;
const MLP_MATRIX_FLASH_PEAK_EMISSIVE_UP = Number.isFinite(GPT2_LAYER_VISUAL_TUNING.mlp.flashPeakEmissiveIntensityUp)
    ? GPT2_LAYER_VISUAL_TUNING.mlp.flashPeakEmissiveIntensityUp
    : MLP_MATRIX_FLASH_PEAK_EMISSIVE_BASE;
const MLP_MATRIX_FLASH_PEAK_EMISSIVE_DOWN = Number.isFinite(GPT2_LAYER_VISUAL_TUNING.mlp.flashPeakEmissiveIntensityDown)
    ? GPT2_LAYER_VISUAL_TUNING.mlp.flashPeakEmissiveIntensityDown
    : MLP_MATRIX_FLASH_PEAK_EMISSIVE_BASE;
const MLP_MATRIX_FLASH_MIN_DURATION_MS = Number.isFinite(GPT2_LAYER_VISUAL_TUNING.mlp.flashMinDurationMs)
    ? GPT2_LAYER_VISUAL_TUNING.mlp.flashMinDurationMs
    : 340;
const POST_MLP_RETURN_TRAIL_OPACITY = 0.1;
const MLP_TRANSITION_PROFILE_DEFAULT = Object.freeze({
    expandRiseUnits: 30,
    expandRiseMs: 500,
    geluDurationMs: 500,
    geluRiseExtra: 24,
    geluCurveHeight: 36,
    geluActivationSwitchT: 0.35,
    geluLiteMode: false,
    maxUpDurationMs: null,
    maxDownDurationMs: null,
    colorMinDurationMs: MLP_MATRIX_FLASH_MIN_DURATION_MS
});
const MLP_TRANSITION_PROFILE_TOUCH = Object.freeze({
    // Keep touch/mobile behavior aligned with desktop so GELU curve bending
    // remains visible instead of falling back to the lite vertical-lift path.
    expandRiseUnits: 30,
    expandRiseMs: 500,
    geluDurationMs: 500,
    geluRiseExtra: 24,
    geluCurveHeight: 36,
    geluActivationSwitchT: 0.35,
    geluLiteMode: false,
    maxUpDurationMs: null,
    maxDownDurationMs: null,
    colorMinDurationMs: MLP_MATRIX_FLASH_MIN_DURATION_MS
});
const MLP_TRANSITION_PROFILE_SKIP = Object.freeze({
    expandRiseUnits: 16,
    expandRiseMs: 180,
    geluDurationMs: 150,
    geluRiseExtra: 12,
    geluCurveHeight: 16,
    geluActivationSwitchT: 0.22,
    geluLiteMode: true,
    maxUpDurationMs: 260,
    maxDownDurationMs: 320,
    colorMinDurationMs: 220
});
const MLP_TRANSITION_PROFILE_SKIP_TOUCH = Object.freeze({
    expandRiseUnits: 12,
    expandRiseMs: 140,
    geluDurationMs: 110,
    geluRiseExtra: 8,
    geluCurveHeight: 10,
    geluActivationSwitchT: 0.2,
    geluLiteMode: true,
    maxUpDurationMs: 220,
    maxDownDurationMs: 240,
    colorMinDurationMs: 180
});

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
    constructor(index, random, yOffset = 0, externalLanes = null, onFinished = null, isActive = true, activationSource = null, laneCount = NUM_VECTOR_LANES, layerSpacing = DEFAULT_LAYER_STACK_SPACING, passConfig = null) {
        super(index);
        this.random = random;
        this.yOffset = yOffset;
        this.externalLanes = externalLanes;
        this.onFinished = typeof onFinished === 'function' ? onFinished : null;
        this.isActive = isActive;
        this.activationSource = activationSource || null;
        this._laneCount = Math.max(1, Math.floor(laneCount || NUM_VECTOR_LANES));
        const cfg = passConfig || {};
        this._laneLayoutCount = Math.max(this._laneCount, Math.floor(cfg.laneLayoutCount || this._laneCount));
        const configuredActive = Array.isArray(cfg.activeLaneLayoutIndices)
            ? cfg.activeLaneLayoutIndices.slice(0, this._laneCount)
            : null;
        this._activeLaneLayoutIndices = (configuredActive && configuredActive.length)
            ? configuredActive.map((laneIdx) => {
                const maxIdx = Math.max(0, this._laneLayoutCount - 1);
                const idx = Number.isFinite(laneIdx) ? Math.floor(laneIdx) : 0;
                return Math.max(0, Math.min(maxIdx, idx));
            })
            : Array.from({ length: this._laneCount }, (_, idx) => idx);
        while (this._activeLaneLayoutIndices.length < this._laneCount) {
            this._activeLaneLayoutIndices.push(this._activeLaneLayoutIndices.length);
        }
        this._passLaneTokenIndices = Array.isArray(cfg.laneTokenIndices)
            ? cfg.laneTokenIndices.slice(0, this._laneCount)
            : null;
        this._kvCacheModeEnabled = !!cfg.kvCacheModeEnabled;
        this._kvCacheDecodeActive = !!cfg.kvCacheDecodeActive;
        this._cachedKvEntries = Array.isArray(cfg.cachedKvEntries) ? cfg.cachedKvEntries : [];
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
        // Barrier flag for synchronized post-MLP return motion.
        this._mlpReturnStart = false;
        // NEW: Barrier flags
        this._ln1Start = false;        // start LN-1 branch
        this._ln1StartBarrierArmed = false;
        this._ln1StartAtMs = NaN;
        this._mhsaStart = false;       // start horizontal travel to heads
        this._progressEmitter = null;  // external emitter for progress events
        this._skipToEndActive = false;
        this._skipConcatTriggered = false;
        this._skipHiddenMaterials = new WeakMap();
        this._skipVectorMeshes = new Set();
        this._skipVectorMeshIndexDirty = true;
        this._skipVectorMeshIndexLastBuildMs = 0;
        this._skipVisibilityDirty = false;
        this._skipVisibilityLastApplyMs = 0;
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
        this._mlpMatrixInactiveColor = new THREE.Color(INACTIVE_COMPONENT_COLOR);
        this._mlpMatrixActiveColor = new THREE.Color(0xc07a12);
        this._mlpUpTweenColor = new THREE.Color();
        this._mlpDownTweenColor = new THREE.Color();
        this._posPassBarrierArmed = false;
        this._posPassStartAtMs = NaN;
        this._trailUpdateFrameId = 0;
        this._vecsToCheckScratch = new Array(9);
        this._postAttentionWatchLanes = [];
        this._lanePhaseDebugOverride = undefined;
        this._ln2HandoffStallSinceMs = NaN;
        this._lastUpdateNowMs = NaN;
    }

    setProgressEmitter(emitter) { this._progressEmitter = emitter; }
    _emitProgress() {
        if (this._progressEmitter && typeof this._progressEmitter.dispatchEvent === 'function') {
            this._progressEmitter.dispatchEvent(new Event('progress'));
        }
    }

    setLanePhaseDebugLogging(enabled = true) {
        this._lanePhaseDebugOverride = !!enabled;
    }

    _isLanePhaseDebugEnabled() {
        if (typeof this._lanePhaseDebugOverride === 'boolean') return this._lanePhaseDebugOverride;
        if (typeof window === 'undefined') return false;
        // Runtime toggle for ad-hoc diagnostics in DevTools:
        // window.__GPT2_LAYER_PHASE_DEBUG = true
        return !!window.__GPT2_LAYER_PHASE_DEBUG;
    }

    _isLayerLifecycleDebugEnabled() {
        if (typeof window === 'undefined') return false;
        return window.__GPT2_LAYER_DEBUG === true || this._isLanePhaseDebugEnabled();
    }

    _debugLayerLifecycleLog(message) {
        if (!this._isLayerLifecycleDebugEnabled()) return;
        console.log(message);
    }

    _debugLanePhaseTransition(lane, key, from, to, reason = '') {
        if (!this._isLanePhaseDebugEnabled()) return;
        const laneId = lane && Number.isFinite(lane.laneIndex) ? lane.laneIndex : '?';
        const why = reason ? ` (${reason})` : '';
        console.info(`[LanePhase][L${this.index} lane ${laneId}] ${key}: ${from || 'na'} -> ${to || 'na'}${why}`);
    }

    _setLanePhase(lane, key, nextValue, reason = '') {
        if (!lane || !key) return false;
        const prevValue = lane[key];
        if (prevValue === nextValue) return false;
        const allowed = isAllowedLanePhaseTransition(key, prevValue, nextValue);
        if (!allowed && this._isLanePhaseDebugEnabled()) {
            const laneId = Number.isFinite(lane?.laneIndex) ? lane.laneIndex : '?';
            console.warn(
                `[LanePhase][L${this.index} lane ${laneId}] unexpected ${key} transition: `
                + `${prevValue || 'na'} -> ${nextValue || 'na'}`
                + (reason ? ` (${reason})` : '')
            );
        }
        lane[key] = nextValue;
        this._debugLanePhaseTransition(lane, key, prevValue, nextValue, reason);
        return true;
    }

    _setLaneHorizPhase(lane, nextValue, reason = '') {
        return this._setLanePhase(lane, 'horizPhase', nextValue, reason);
    }

    _setLaneLn2Phase(lane, nextValue, reason = '') {
        return this._setLanePhase(lane, 'ln2Phase', nextValue, reason);
    }

    _isTouchPrimaryDevice() {
        if (typeof window === 'undefined') return false;
        try {
            if (typeof window.matchMedia === 'function') {
                const coarse = window.matchMedia('(pointer: coarse)').matches
                    || window.matchMedia('(hover: none) and (pointer: coarse)').matches;
                if (coarse) return true;
            }
        } catch (_) { /* non-browser/test env */ }
        const touchPoints = Number.isFinite(window?.navigator?.maxTouchPoints)
            ? window.navigator.maxTouchPoints
            : 0;
        return touchPoints > 0;
    }

    _resolveMlpTransitionProfile() {
        const skipFastPath = !!(this._skipToEndActive || this._skipLayerActive);
        const touchPrimary = this._isTouchPrimaryDevice();
        if (skipFastPath && touchPrimary) return MLP_TRANSITION_PROFILE_SKIP_TOUCH;
        if (skipFastPath) return MLP_TRANSITION_PROFILE_SKIP;
        if (touchPrimary) return MLP_TRANSITION_PROFILE_TOUCH;
        return MLP_TRANSITION_PROFILE_DEFAULT;
    }

    _resolveMlpColorDuration(baseDurationMs, profile = null) {
        const safeBase = Number.isFinite(baseDurationMs) ? Math.max(0, baseDurationMs) : 0;
        const resolved = profile || this._resolveMlpTransitionProfile();
        if (resolved && Number.isFinite(resolved.colorMinDurationMs)) {
            return Math.max(safeBase, resolved.colorMinDurationMs);
        }
        if (this._skipToEndActive) {
            return Math.max(safeBase, SKIP_MLP_COLOR_MIN_MS);
        }
        return safeBase;
    }

    _getBasePrismAdditionDurationMs() {
        const vectorLength = Math.max(1, this._getBaseVectorLength());
        const durationMs = PRISM_ADD_ANIM_BASE_DURATION / PRISM_ADD_ANIM_SPEED_MULT;
        const flashDurationMs = PRISM_ADD_ANIM_BASE_FLASH_DURATION / PRISM_ADD_ANIM_SPEED_MULT;
        const delayBetweenMs = PRISM_ADD_ANIM_BASE_DELAY_BETWEEN_PRISMS / PRISM_ADD_ANIM_SPEED_MULT;
        return durationMs + flashDurationMs + Math.max(0, (vectorLength - 1) * delayBetweenMs);
    }

    _computeGateBarrierRemainders(lanes) {
        const list = Array.isArray(lanes) ? lanes : [];
        const laneCount = list.length;
        const speedMult = GLOBAL_ANIM_SPEED_MULT;
        const basePrismAddDurationMs = this._getBasePrismAdditionDurationMs();
        const ln1RiseBaseSpeed = ANIM_RISE_SPEED_INSIDE_LN * speedMult;
        let ln1BarrierMaxRemaining = 0;
        let mhsaBarrierMaxRemaining = 0;
        let ln2BarrierMaxRemaining = 0;

        if (laneCount) {
            const ln1TravelTargetY = this.ln1TopY + 5;
            for (let laneIdx = 0; laneIdx < laneCount; laneIdx++) {
                const lane = list[laneIdx];
                if (!lane) continue;

                if (!this._ln1Start
                    && lane.horizPhase === HORIZ_PHASE.WAITING
                    && lane.originalVec
                    && lane.originalVec.group
                    && Number.isFinite(lane.branchStartY)) {
                    const remaining = Math.max(0, lane.branchStartY - lane.originalVec.group.position.y);
                    if (remaining > ln1BarrierMaxRemaining) ln1BarrierMaxRemaining = remaining;
                }

                if (!this._mhsaStart
                    && lane.horizPhase === HORIZ_PHASE.RISE_ABOVE_LN
                    && lane.resultVec
                    && lane.resultVec.group) {
                    const remaining = Math.max(0, ln1TravelTargetY - lane.resultVec.group.position.y);
                    if (remaining > mhsaBarrierMaxRemaining) mhsaBarrierMaxRemaining = remaining;
                }

                if (!this._mhsaStart && lane.ln1AddStarted && !lane.ln1AddComplete) {
                    const addProgress = THREE.MathUtils.clamp(
                        Number.isFinite(lane.ln1ShiftProgress) ? lane.ln1ShiftProgress : 0,
                        0,
                        1
                    );
                    const remainingSeconds = (basePrismAddDurationMs * (1 - addProgress)) / 1000;
                    const virtualRemaining = ln1RiseBaseSpeed * remainingSeconds;
                    if (virtualRemaining > mhsaBarrierMaxRemaining) mhsaBarrierMaxRemaining = virtualRemaining;
                }

                // Keep LN2 handoff lane-local: do not slow already-ready lanes
                // to synchronize with lanes still finishing post-attention add.
            }
        }

        return {
            speedMult,
            ln1BarrierMaxRemaining,
            mhsaBarrierMaxRemaining,
            ln2BarrierMaxRemaining,
        };
    }

    _maybeStartSkipConcat(skipActive) {
        if (!skipActive || !this.mhsaAnimation || this._skipConcatTriggered) return;
        const mhsa = this.mhsaAnimation;
        const rowMergePhase = mhsa.rowMergePhase || 'not_started';
        const outputProjPhase = mhsa.outputProjMatrixAnimationPhase || 'waiting';
        const concatNotStarted = rowMergePhase === 'not_started' && outputProjPhase === 'waiting';
        if (concatNotStarted
            && mhsa.mhaPassThroughPhase === 'mha_pass_through_complete'
            && typeof mhsa.skipSelfAttentionAndStartConcat === 'function') {
            mhsa.skipSelfAttentionAndStartConcat();
            this._skipConcatTriggered = true;
        }
    }

    _applyLn2HandoffWatchdog(lanes, nowMs, mhsaOutputComplete) {
        if (!Array.isArray(lanes) || !lanes.length || !Number.isFinite(nowMs)) {
            this._ln2HandoffStallSinceMs = NaN;
            return false;
        }
        const mhsaOutputStageActive = this._isMhsaOutputStageActive();
        const handoffWindowOpen = mhsaOutputComplete || mhsaOutputStageActive;
        if (!handoffWindowOpen || this._ln2Ready || this._mlpStart) {
            this._ln2HandoffStallSinceMs = NaN;
            return false;
        }

        const pendingLanes = lanes.filter((lane) => {
            if (!lane || lane.ln2Phase === LN2_PHASE.DONE) return false;
            if (lane.ln2Phase === LN2_PHASE.NOT_STARTED) return true;
            if (lane.ln2Phase === LN2_PHASE.PRE_RISE) {
                return !(lane.postAdditionVec && lane.postAdditionVec.group);
            }
            return false;
        });
        const blockedByOutputCompletion = !mhsaOutputComplete
            && lanes.some((lane) => (
                lane
                && lane.ln2Phase === LN2_PHASE.PRE_RISE
                && lane.postAdditionVec
                && lane.postAdditionVec.group
            ));

        if (!pendingLanes.length && !blockedByOutputCompletion) {
            this._ln2HandoffStallSinceMs = NaN;
            return false;
        }

        if (!Number.isFinite(this._ln2HandoffStallSinceMs)) {
            this._ln2HandoffStallSinceMs = nowMs;
            return false;
        }

        if ((nowMs - this._ln2HandoffStallSinceMs) < LN2_HANDOFF_STALL_TIMEOUT_MS_NORMAL) {
            return false;
        }

        let forcedCount = 0;
        let forcedOutputCompletion = false;
        if (this.mhsaAnimation && !mhsaOutputComplete
            && typeof this.mhsaAnimation._completeOutputProjectionFallback === 'function') {
            try {
                this.mhsaAnimation._completeOutputProjectionFallback(
                    'LN2 handoff watchdog: output projection did not complete',
                    {
                        fallbackLanes: true,
                        onlyIncompleteLanes: false,
                        logReason: false
                    }
                );
                forcedOutputCompletion = true;
            } catch (_) { /* fallback below */ }
        }
        if (
            this.mhsaAnimation
            && !forcedOutputCompletion
            && typeof this.mhsaAnimation._forcePostAttentionFallback === 'function'
        ) {
            try {
                forcedCount += this.mhsaAnimation._forcePostAttentionFallback(
                    'LN2 handoff watchdog after output projection',
                    { onlyIncomplete: false }
                ) || 0;
            } catch (_) { /* fallback below */ }
        }

        for (let i = 0; i < pendingLanes.length; i++) {
            const lane = pendingLanes[i];
            if (!lane || lane.ln2Phase === LN2_PHASE.DONE) continue;
            if (lane.ln2Phase !== LN2_PHASE.NOT_STARTED && lane.ln2Phase !== LN2_PHASE.PRE_RISE) continue;
            const postVec = this._ensureLanePostAdditionVector(lane, { allowOriginalFallback: true });
            if (!postVec || !postVec.group) continue;
            this._setLaneHorizPhase(lane, HORIZ_PHASE.WAITING_FOR_LN2, 'ln2 handoff watchdog local fallback');
            this._setLaneLn2Phase(lane, LN2_PHASE.PRE_RISE, 'ln2 handoff watchdog local fallback');
            forcedCount += 1;
        }

        if (forcedCount > 0 || forcedOutputCompletion) {
            console.warn(
                `Layer ${this.index}: LN2 handoff watchdog forced ${forcedCount} lane(s)`
                + ` (completedOutputStage=${forcedOutputCompletion}).`
            );
            this._emitProgress();
        }
        this._ln2HandoffStallSinceMs = nowMs;
        return forcedCount > 0 || forcedOutputCompletion;
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
            useBatchedVectorCopies: !this._kvCacheModeEnabled,
            kvCacheDecodeActive: this._kvCacheDecodeActive,
        });
        if (this.mhsaAnimation && typeof this.mhsaAnimation.setCachedKvEntries === 'function') {
            this.mhsaAnimation.setCachedKvEntries(this._cachedKvEntries);
        }

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
        const mlpUp = this._createMlpMatrix({
            offsetX,
            centerY: mlpUpCenterY,
            params: MLP_MATRIX_PARAMS_UP,
            label: 'MLP Up Weight Matrix',
            color: inactiveDark,
            emissiveIntensity: 0.08
        });

        // 5) MLP Down-projection matrix (same orange)
        const mlpDownCenterY = mlpUpCenterY + MLP_MATRIX_PARAMS_UP.height / 2 + MLP_INTER_MATRIX_GAP + MLP_MATRIX_PARAMS_DOWN.height / 2;
        const mlpDown = this._createMlpMatrix({
            offsetX,
            centerY: mlpDownCenterY,
            params: MLP_MATRIX_PARAMS_DOWN,
            label: 'MLP Down Weight Matrix',
            color: inactiveDark,
            emissiveIntensity: 0.1
        });

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

    _createMlpMatrix({ offsetX, centerY, params, label, color, emissiveIntensity }) {
        const matrix = new WeightMatrixVisualization(
            null,
            new THREE.Vector3(offsetX, centerY, 0),
            params.width,
            params.height,
            params.depth,
            params.topWidthFactor,
            params.cornerRadius,
            params.numberOfSlits,
            params.slitWidth,
            params.slitDepthFactor,
            params.slitBottomWidthFactor,
            params.slitTopWidthFactor
        );
        const matrixColor = color && typeof color.clone === 'function' ? color.clone() : color;
        matrix.setColor(matrixColor);
        matrix.setMaterialProperties({ opacity: 1.0, transparent: false, emissiveIntensity });
        applyMatrixMaterialTweaks(matrix, MLP_REFLECTIVITY_TWEAKS);
        applyMatrixLabel(matrix, label);
        this.raycastRoot.add(matrix.group);
        freezeStaticTransforms(matrix.group, true);
        return matrix;
    }

    update(dt) {
        const skipActive = this._skipToEndActive;
        const nowMs = this._getNowMs();
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
        const lastUpdateNowMs = this._lastUpdateNowMs;
        this._lastUpdateNowMs = nowMs;
        if (Number.isFinite(lastUpdateNowMs) && Number.isFinite(nowMs) && (nowMs - lastUpdateNowMs) > WATCHDOG_FRAME_GAP_RESET_MS) {
            this._ln2HandoffStallSinceMs = NaN;
            this._resetLaneStallWatchdog(lanes);
        }
        const exitTransitionRange = GPT2_LAYER_VISUAL_TUNING.layerNorm.exitTransitionRange; // world–unit distance for final fade
        const needsPositioningCheck = this._transitionPhase === 'positioning';
        let allVectorsInPosition = needsPositioningCheck && laneCount > 0;
        let posAddDone = true;
        let allPosPassReady = this.index === 0 && laneCount > 0;
        let hasPendingPosPass = false;
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
                    // Positional pass-through trail is manually sampled in its tween
                    // so skip the generic updater to avoid duplicate corner writes.
                    if (v === lane.posVec && lane.__manualPosTrail) continue;
                    // Let MHSA routing own the travel trail updates to preserve sharp corners.
                    if (v === lane.travellingVec && (lane.horizPhase === HORIZ_PHASE.READY_MHSA || lane.horizPhase === HORIZ_PHASE.TRAVEL_MHSA)) {
                        continue;
                    }

                    // MLP return vectors are always trail-driven by dedicated
                    // tweens (down-projection/rise/return); skipping here avoids
                    // double-writing the same trail in one frame.
                    if (v === lane.finalVecAfterMlp && lane.mlpDownStarted) continue;
                    // Residual-stream world trails are owned by MHSAAnimation after
                    // the initial WAITING phase. Skipping here avoids duplicate
                    // writes that can make LN2 approach segments look brighter.
                    if (
                        this.mhsaAnimation
                        && v.userData.trailWorld
                        && lane.horizPhase !== HORIZ_PHASE.WAITING
                        && (v === lane.originalVec || v === lane.postAdditionVec)
                    ) {
                        continue;
                    }
                    const trailRef = v.userData.trail;
                    if (trailRef.__lastUpdateFrameId === trailFrameId) continue;
                    trailRef.__lastUpdateFrameId = trailFrameId;

                    // During residual addition, let MHSAAnimation drive world-space
                    // residual trail updates (it follows the centre prism). Skip here
                    // to avoid double-writing the same world trail in the same frame.
                    const topLnTrailControlled = lane.stopRise && lane.__topLnStopRise;
                    if (lane.stopRise && (v.userData.trailWorld || topLnTrailControlled)) continue;

                    if (!v.userData.trailWorld) {
                        const zPos = Number.isFinite(lane.zPos) ? lane.zPos : v.group.position.z;
                        if (
                            lane.horizPhase === HORIZ_PHASE.RIGHT
                            && v === lane.dupVec
                        ) {
                            const rightY = Number.isFinite(lane.branchStartY)
                                ? lane.branchStartY
                                : v.group.position.y;
                            TMP_LN_TRAIL_POS.set(v.group.position.x, rightY, zPos);
                            trailRef.update(TMP_LN_TRAIL_POS);
                            continue;
                        }
                        if (
                            lane.ln2Phase === LN2_PHASE.RIGHT
                            && v === lane.movingVecLN2
                        ) {
                            const rightY = Number.isFinite(lane.__ln2RightY)
                                ? lane.__ln2RightY
                                : v.group.position.y;
                            TMP_LN_TRAIL_POS.set(v.group.position.x, rightY, zPos);
                            trailRef.update(TMP_LN_TRAIL_POS);
                            continue;
                        }
                        if (
                            lane.horizPhase === HORIZ_PHASE.INSIDE_LN
                            && (v === lane.dupVec || v === lane.resultVec)
                        ) {
                            const clampedY = Math.min(topY_ln1_abs, Math.max(bottomY_ln1_abs, v.group.position.y));
                            TMP_LN_TRAIL_POS.set(BRANCH_X, clampedY, zPos);
                            trailRef.update(TMP_LN_TRAIL_POS);
                            continue;
                        }
                        if (
                            lane.ln2Phase === LN2_PHASE.INSIDE_LN
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
                if (this.index === 0) {
                    const hasPositionalPass = !!(lane.posVec && typeof lane.startPositionalPassThrough === 'function');
                    if (hasPositionalPass && !lane.posAddComplete) {
                        if (!lane.__posPassStarted) {
                            hasPendingPosPass = true;
                            if (!skipActive) {
                                const ov = lane.originalVec;
                                const exitY = Number.isFinite(lane.vocabEmbeddingExitY)
                                    ? lane.vocabEmbeddingExitY
                                    : NaN;
                                const vocabExited = !!(
                                    ov
                                    && ov.group
                                    && Number.isFinite(exitY)
                                    && ov.group.position.y >= exitY - 0.01
                                );
                                if (!vocabExited) {
                                    allPosPassReady = false;
                                }
                            }
                        }
                        if (lane.__posPassStarted) {
                            // Count only active update time so hidden-tab gaps
                            // don't falsely trip the positional-add watchdog.
                            const prevElapsedMs = Number.isFinite(lane.__posAddWatchElapsedMs)
                                ? lane.__posAddWatchElapsedMs
                                : 0;
                            const stepMs = Number.isFinite(dt)
                                ? Math.max(0, dt * 1000)
                                : 0;
                            const elapsedMs = prevElapsedMs + stepMs;
                            lane.__posAddWatchElapsedMs = elapsedMs;
                            if (elapsedMs >= POS_ADD_STALL_TIMEOUT_MS) {
                                const sumData = this._getEmbeddingData(lane, 'sum');
                                if (sumData && lane.originalVec) {
                                    applyVectorData(
                                        lane.originalVec,
                                        sumData,
                                        lane.tokenLabel ? `Embedding Sum - ${lane.tokenLabel}` : 'Embedding Sum',
                                        this._getLaneMeta(lane, 'embedding.sum')
                                    );
                                }
                                lane.posAddComplete = true;
                                console.warn(`Layer ${this.index}: positional addition watchdog forced completion for lane ${lane.laneIndex ?? '?'}`);
                            } else {
                                posAddDone = false;
                            }
                        } else {
                            if (Number.isFinite(lane.__posAddWatchElapsedMs)) {
                                delete lane.__posAddWatchElapsedMs;
                            }
                            posAddDone = false;
                        }
                    } else if (lane.posAddComplete && Number.isFinite(lane.__posAddWatchElapsedMs)) {
                        delete lane.__posAddWatchElapsedMs;
                    }
                }
                if (allLn1Ready) {
                    const ov = lane.originalVec;
                    const targetY = lane.branchStartY;
                    if (!ov || !ov.group || !Number.isFinite(targetY) || ov.group.position.y < targetY) {
                        allLn1Ready = false;
                    }
                }
                if (allMhsaReady && lane.horizPhase !== HORIZ_PHASE.READY_MHSA) {
                    allMhsaReady = false;
                }
                if (allLn2Ready) {
                    const ready = lane.ln2Phase === LN2_PHASE.PRE_RISE
                        && lane.postAdditionVec
                        && lane.postAdditionVec.group
                        && lane.postAdditionVec.group.position.y >= ln2SyncY - 0.01;
                    if (!ready) allLn2Ready = false;
                }
                if (allMlpReady && lane.ln2Phase !== LN2_PHASE.MLP_READY) {
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
        this._updatePositionalPassBarrier({
            lanes,
            nowMs,
            skipActive,
            hasPendingPosPass,
            allPosPassReady
        });
        // Handle transition phase - wait for vectors to reach position
        if (this._transitionPhase === 'positioning') {
            if (allVectorsInPosition) {
                this._transitionPhase = 'complete';
                this.isActive = true; // now start the actual animation
                this._debugLayerLifecycleLog(`Layer ${this.index}: All vectors in position, starting animation`);
            } else {
                // Keep vectors rising toward the target
                lanes.forEach(lane => {
                    const targetY = lane.branchStartY;
                    if (lane.originalVec.group.position.y < targetY) {
                        lane.originalVec.group.position.y = Math.min(targetY, 
                            lane.originalVec.group.position.y + ANIM_RISE_SPEED_ORIGINAL * GLOBAL_ANIM_SPEED_MULT * dt);
                    }
                });
                return; // keep waiting
            }
        }

        if (!this.isActive) {
            return; // Skip processing when inactive / placeholder
        }

        this._maybeStartSkipConcat(skipActive);
        const layerNormTuning = GPT2_LAYER_VISUAL_TUNING.layerNorm;
        this._ln1ColorLocked = updateLayerNormVisualState({
            layerNorm: this.ln1,
            targetColor: this._ln1TargetColor,
            lockedColor: this._ln1LockedColor,
            isColorLocked: this._ln1ColorLocked,
            materialState: this._ln1MaterialState,
            highestVecY: highestLN1VecY,
            anyVectorInNorm: anyVectorInLN1,
            bottomY: bottomY_ln1_abs,
            midY: midY_ln1_abs,
            topY: topY_ln1_abs,
            exitTransitionRange,
            inactiveColor: COLOR_DARK_GRAY,
            activeColor: COLOR_LIGHT_YELLOW,
            finalColor: COLOR_BRIGHT_YELLOW,
            opaqueOpacity: layerNormTuning.opaqueOpacity,
            activeOpacity: layerNormTuning.activeOpacity,
            skipActive,
            skipColorLerpAlpha: SKIP_COMPONENT_COLOR_LERP_ALPHA,
            applyWhenInactive: true,
        }).colorLocked;

        this._ln2ColorLocked = updateLayerNormVisualState({
            layerNorm: this.ln2,
            targetColor: this._ln2TargetColor,
            lockedColor: this._ln2LockedColor,
            isColorLocked: this._ln2ColorLocked,
            materialState: this._ln2MaterialState,
            highestVecY: highestLN2VecY,
            anyVectorInNorm: anyVectorInLN2,
            bottomY: bottomY_ln2_abs,
            midY: midY_ln2_abs,
            topY: topY_ln2_abs,
            exitTransitionRange,
            inactiveColor: COLOR_DARK_GRAY,
            activeColor: COLOR_LIGHT_YELLOW,
            finalColor: COLOR_BRIGHT_YELLOW,
            opaqueOpacity: layerNormTuning.opaqueOpacity,
            activeOpacity: layerNormTuning.activeOpacity,
            skipActive,
            skipColorLerpAlpha: SKIP_COMPONENT_COLOR_LERP_ALPHA,
            applyWhenInactive: false,
        }).colorLocked;

        let mhsaOutputComplete = this._isMhsaOutputStageComplete();
        if (!skipActive && !this._skipLayerActive) {
            const watchdogMutated = this._applyLn2HandoffWatchdog(lanes, nowMs, mhsaOutputComplete);
            if (watchdogMutated) {
                mhsaOutputComplete = this._isMhsaOutputStageComplete();
            }
        } else {
            this._ln2HandoffStallSinceMs = NaN;
        }
        this._updateLn2AndMlpStartGates({
            allLn2Ready,
            allMlpReady
        });

        const {
            speedMult,
            ln1BarrierMaxRemaining,
            mhsaBarrierMaxRemaining,
            ln2BarrierMaxRemaining
        } = this._computeGateBarrierRemainders(lanes);

        this._updateLn1AndMhsaStartGates({
            allLn1Ready,
            posAddDone,
            allMhsaReady,
            nowMs,
            skipActive
        });

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
                case HORIZ_PHASE.WAITING:
                    // Hold at the branching height until the global LN-1
                    // barrier is released.
                    if (this._isWaitingForInputVocabChipGate(lane, nowMs, skipActive)) {
                        if (originalVec && originalVec.group) {
                            originalVec.group.visible = false;
                            // Keep first-layer residual vectors fully inside the bottom
                            // embedding until the matching token chip has entered.
                            if (!lane.__inputVocabGateAdjustedStartY) {
                                const halfPrismHeight = Number.isFinite(originalVec._basePrismCenterY)
                                    ? originalVec._basePrismCenterY
                                    : 0;
                                const prismHeight = halfPrismHeight > 0 ? halfPrismHeight * 2 : 10.5;
                                originalVec.group.position.y -= prismHeight;
                                lane.__inputVocabGateAdjustedStartY = true;
                                const trail = originalVec.userData && originalVec.userData.trail;
                                if (trail) {
                                    if (originalVec.userData && originalVec.userData.trailWorld) {
                                        originalVec.group.getWorldPosition(TMP_WORLD_POS);
                                        if (typeof trail.snapLastPointTo === 'function') {
                                            trail.snapLastPointTo(TMP_WORLD_POS);
                                        } else if (typeof trail.update === 'function') {
                                            trail.update(TMP_WORLD_POS);
                                        }
                                    } else if (typeof trail.snapLastPointTo === 'function') {
                                        trail.snapLastPointTo(originalVec.group.position);
                                    } else if (typeof trail.update === 'function') {
                                        trail.update(originalVec.group.position);
                                    }
                                }
                            }
                        }
                        break;
                    }
                    if (originalVec && originalVec.group && !originalVec.group.visible) {
                        originalVec.group.visible = true;
                    }
                    if (originalVec.group.position.y >= lane.branchStartY) {
                        // Clamp position so early-arriving lanes don’t drift.
                        originalVec.group.position.y = lane.branchStartY;
                    } else {
                        const baseRiseSpeed = ANIM_RISE_SPEED_ORIGINAL * speedMult;
                        const syncedRiseSpeed = this._getSynchronizedRiseSpeed(
                            originalVec.group.position.y,
                            lane.branchStartY,
                            baseRiseSpeed,
                            ln1BarrierMaxRemaining
                        );
                        // Continue rising towards the branching height.
                        originalVec.group.position.y = Math.min(
                            lane.branchStartY,
                            originalVec.group.position.y + syncedRiseSpeed * dt
                        );
                    }
                    break;
                case HORIZ_PHASE.RIGHT:
                    // Mirror LN-2: lock Y at staging height and move X only.
                    if (typeof lane.branchStartY === 'number') {
                        dupVec.group.position.y = lane.branchStartY;
                    }
                    dupVec.group.position.x = Math.min(
                        BRANCH_X,
                        dupVec.group.position.x + ANIM_HORIZ_SPEED * speedMult * dt
                    );
                    if (dupVec.group.position.x >= BRANCH_X - 0.01) {
                        // Ensure alignment with LN-1 centre
                        dupVec.group.position.x = BRANCH_X;
                        if (typeof lane.branchStartY === 'number') {
                            dupVec.group.position.y = lane.branchStartY;
                        }
                        try {
                            const ln1Trail = dupVec.userData && dupVec.userData.trail;
                            if (ln1Trail) {
                                TMP_LN_TRAIL_POS.set(dupVec.group.position.x, dupVec.group.position.y, dupVec.group.position.z);
                                if (typeof ln1Trail.snapLastPointTo === 'function') {
                                    ln1Trail.snapLastPointTo(TMP_LN_TRAIL_POS);
                                } else if (typeof ln1Trail.update === 'function') {
                                    ln1Trail.update(TMP_LN_TRAIL_POS);
                                }
                            }
                        } catch (_) { /* non-fatal trail alignment */ }
                        // Show the multiplication target inside LN-1 (parity with LN-2 behaviour)
                        if (lane.multTarget && lane.multTarget.group) {
                            lane.multTarget.group.visible = true;
                        }
                        if (lane.addTarget && lane.addTarget.group) {
                            lane.addTarget.group.visible = true;
                        }
                        this._setLaneHorizPhase(lane, HORIZ_PHASE.INSIDE_LN, 'ln1 branch reached ring');
                        this._emitProgress();
                    }
                    break;
                case HORIZ_PHASE.INSIDE_LN:
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
                    const ln1RiseSpeedMult = (!skipActive && this.index === 0)
                        ? FIRST_LAYER_LN1_RISE_SPEED_MULT
                        : 1;
                    const riseStep = ANIM_RISE_SPEED_INSIDE_LN * speedMult * ln1RiseSpeedMult * dt;
                    const ln1RiseTargetY = (() => {
                        const multTargetGroup = lane.multTarget && lane.multTarget.group;
                        if (multTargetGroup && Number.isFinite(multTargetGroup.position.y)) {
                            return Math.max(lane.ln1MidY, multTargetGroup.position.y);
                        }
                        return lane.ln1MidY;
                    })();
                    const ln1VectorHeight = (dupVec && typeof dupVec.getUniformHeight === 'function')
                        ? dupVec.getUniformHeight()
                        : NaN;
                    const ln1TouchTriggerY = (Number.isFinite(ln1VectorHeight) && ln1VectorHeight > 0)
                        ? (ln1RiseTargetY - ln1VectorHeight)
                        : ln1RiseTargetY;
                    const startLn1Norm = () => {
                        const ln1NormData = this._getLn1Data(lane, 'norm');
                        const normInput = ln1NormData ? ln1NormData.slice() : dupVec.rawData.slice();
                        lane.pendingNormData = ln1NormData || null;
                        lane.pendingNormLabel = lane.tokenLabel ? `LN1 Normed - ${lane.tokenLabel}` : 'LN1 Normed';
                        lane.pendingNormMeta = this._getLaneMeta(lane, 'ln1.norm');
                        lane.normApplied = false;
                        if (!skipActive && lane.normAnim) {
                            lane.normAnim.start(normInput, {
                                deferDataUpdate: true,
                                sourceAlreadyNormalized: !!(ln1NormData && ln1NormData.length)
                            });
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
                        const ln1PreMultiplyTargetY = Math.max(dupVec.group.position.y, ln1TouchTriggerY);
                        dupVec.group.position.y = Math.min(
                            ln1PreMultiplyTargetY,
                            dupVec.group.position.y + riseStep
                        );
                    }
                    const ln1NormAnimationActive = !!(lane.normAnim && lane.normAnim.isAnimating);
                    if (
                        !lane.multStarted &&
                        lane.normStarted &&
                        !ln1NormAnimationActive &&
                        dupVec.group.position.y >= ln1TouchTriggerY - 0.01
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
                        const sourceRaw = (Array.isArray(dupVec.rawData) || ArrayBuffer.isView(dupVec.rawData))
                            ? dupVec.rawData
                            : [];
                        const multData = sourceRaw.slice();
                        if (scaleParamData && (Array.isArray(scaleParamData) || ArrayBuffer.isView(scaleParamData))) {
                            const len = Math.min(multData.length, scaleParamData.length);
                            for (let i = 0; i < len; i++) {
                                multData[i] = (sourceRaw[i] || 0) * (scaleParamData[i] || 0);
                            }
                        }

                        const multSeed = multData.length
                            ? multData
                            : (sourceRaw.length
                                ? sourceRaw.slice()
                                : new Array(Math.max(1, Math.floor(dupVec.instanceCount || this._getBaseVectorLength()))).fill(0));
                        const multResult = this._createPrismVector(
                            multSeed,
                            (scaleParam && scaleParam.group)
                                ? scaleParam.group.position.clone()
                                : dupVec.group.position.clone(),
                            30,
                            dupVec.instanceCount
                        );
                        this.raycastRoot.add(multResult.group);
                        // Keep hidden until colors are fully applied and the multiply
                        // handoff executes, to avoid one-frame constructor defaults.
                        multResult.group.visible = false;
                        const ln1ScaledData = this._getLn1Data(lane, 'scale');
                        const scaledFallback = (ln1ScaledData && ln1ScaledData.length)
                            ? ln1ScaledData
                            : multData;
                        if (scaledFallback && scaledFallback.length) {
                            applyVectorData(
                                multResult,
                                scaledFallback,
                                lane.tokenLabel ? `LN1 Scaled - ${lane.tokenLabel}` : 'LN1 Scaled',
                                this._getLaneMeta(lane, 'ln1.scale')
                            );
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

                        const startLn1Addition = () => {
                            const shiftParamData = this._getLayerNormParamData('ln1', 'shift');
                            const shiftSeed = (shiftParamData && shiftParamData.length)
                                ? shiftParamData
                                : (sourceRaw.length
                                    ? sourceRaw.slice()
                                    : new Array(Math.max(1, Math.floor(multResult.instanceCount || this._getBaseVectorLength()))).fill(0));
                            const usingShiftTarget = !!(shiftParam && shiftParam.group);
                            const addResult = usingShiftTarget
                                ? shiftParam
                                : this._createPrismVector(
                                    shiftSeed,
                                    multResult.group.position.clone(),
                                    30,
                                    multResult.instanceCount
                                );
                            if (!usingShiftTarget) {
                                this.raycastRoot.add(addResult.group);
                                addResult.group.visible = false;
                                // Fallback path when param bank target is unavailable.
                                this._applyLayerNormParamVector(addResult, 'ln1', 'shift', null);
                            } else {
                                // Ensure the true shift parameter vector is visible as the
                                // top addend while addition begins.
                                addResult.group.visible = true;
                            }
                            addResult.group.visible = true;

                            const ln1ShiftedData = this._getLn1Data(lane, 'shift');
                            const finalLn1ShiftData = (ln1ShiftedData && ln1ShiftedData.length)
                                ? ln1ShiftedData
                                : (() => {
                                    const count = Math.max(1, Math.floor(multResult.instanceCount || addResult.instanceCount || 1));
                                    const scaledRaw = (Array.isArray(multResult.rawData) || ArrayBuffer.isView(multResult.rawData))
                                        ? multResult.rawData
                                        : [];
                                    const shiftRaw = (Array.isArray(addResult.rawData) || ArrayBuffer.isView(addResult.rawData))
                                        ? addResult.rawData
                                        : [];
                                    const sum = new Array(count);
                                    for (let i = 0; i < count; i++) {
                                        sum[i] = (scaledRaw[i] || 0) + (shiftRaw[i] || 0);
                                    }
                                    return sum;
                                })();
                            logLayerNormVectorDump({
                                layerIndex: this.index,
                                kind: 'ln1',
                                lane,
                                normalizedSaved: this._getLn1Data(lane, 'norm'),
                                normalizedRuntime: sourceRaw,
                                scaleParamSaved: scaleParamData,
                                productComputed: multData,
                                productSaved: ln1ScaledData,
                                productUsedForColor: scaledFallback,
                                shiftParamSaved: shiftParamData,
                                shiftRuntime: addResult.rawData,
                                productPlusShiftComputed: buildDebugVectorSum(multData, shiftParamData, multResult.instanceCount),
                                productPlusShiftSaved: ln1ShiftedData,
                                productPlusShiftUsedForColor: finalLn1ShiftData
                            });
                            lane.resultVec = addResult;
                            lane.ln1AddStarted = true;
                            startPrismAdditionAnimation(multResult, addResult, null, () => {
                                lane.ln1AddComplete = true;
                                if (finalLn1ShiftData && finalLn1ShiftData.length) {
                                    applyVectorData(
                                        addResult,
                                        finalLn1ShiftData,
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
                                this._setLaneHorizPhase(lane, HORIZ_PHASE.RISE_ABOVE_LN, 'ln1 add complete');
                                this._emitProgress();
                            }, {
                                finalData: finalLn1ShiftData,
                                flashOnTargetImpact: false,
                                progressTarget: lane,
                                progressKey: 'ln1ShiftProgress'
                            });
                        };

                        this._animateMultiplyTransition({
                            sourceVec: dupVec,
                            multResult,
                            scaleParam,
                            instant: true,
                            onComplete: startLn1Addition
                        });
                    }
                    break;
                case HORIZ_PHASE.RISE_ABOVE_LN:
                    // Rise to just above LN1 before starting horizontal travel
                    const rv = lane.resultVec;
                    if (rv) {
                        const targetY = this.ln1TopY + 5; // Same as meetY in original
                        if (rv.group.position.y < targetY) {
                            const ln1RiseSpeedMult = (!skipActive && this.index === 0)
                                ? FIRST_LAYER_LN1_RISE_SPEED_MULT
                                : 1;
                            const baseRiseSpeed = ANIM_RISE_SPEED_INSIDE_LN * speedMult * ln1RiseSpeedMult;
                            const syncedRiseSpeed = this._getSynchronizedRiseSpeed(
                                rv.group.position.y,
                                targetY,
                                baseRiseSpeed,
                                mhsaBarrierMaxRemaining
                            );
                            rv.group.position.y = Math.min(targetY, rv.group.position.y + syncedRiseSpeed * dt);
                        } else {
                            // Now that we're above LN1, mark lane ready for MHSA travel.
                            lane.travellingVec = rv;
                            lane.headIndex = 0;
                            this._setLaneHorizPhase(lane, HORIZ_PHASE.READY_MHSA, 'ln1 rise complete');
                            lane.__mhsaTrailCornerPending = true;
                            this._emitProgress();
                        }
                    }
                    break;
                case HORIZ_PHASE.READY_MHSA:
                    // Hold at staging height until global _mhsaStart flag triggers.
                    // Ensure vector stays exactly at meetY.
                    if (lane.travellingVec) {
                        lane.travellingVec.group.position.y = this.ln1TopY + 5;
                    }
                    break;
                case HORIZ_PHASE.TRAVEL_MHSA:
                    // MHSAAnimation will handle the horizontal movement
                    break;
                case HORIZ_PHASE.POST_MHSA_ADDITION:
                    // After MHSA addition completes, start LN2 phase
                    // This state is set by MHSAAnimation._startAdditionAnimation
                    if (lane.ln2Phase === LN2_PHASE.PRE_RISE && lane.postAdditionVec) {
                        this._setLaneHorizPhase(lane, HORIZ_PHASE.WAITING_FOR_LN2, 'mhsa addition complete');
                        this._emitProgress();
                    }
                    break;
                case HORIZ_PHASE.WAITING_FOR_LN2:
                    // Just a placeholder state while LN2 animation runs
                    break;
                default:
                    break;
            }

            // ─────────────────────────────────────────────────────────────
            // LayerNorm2 / MLP Pipeline
            // ─────────────────────────────────────────────────────────────
            switch (lane.ln2Phase) {
                case LN2_PHASE.PRE_RISE: {
                    // Rise the post-addition vector before branching to LN2
                    const v = lane.postAdditionVec;
                    if (!v) break;
                    
                    // Stage the vector at the same relative position used for LayerNorm-1
                    // (5 units above the bottom of the norm ring) so that the ensuing
                    // normalisation begins at a consistent height across both LayerNorms.
                    const targetY = bottomY_ln2_abs + 5; // align with LN1 offset
                    if (v.group.position.y < targetY) {
                        const baseRiseSpeed =
                            ANIM_RISE_SPEED_POST_SPLIT_LN2
                            * POST_ATTENTION_LN2_PRE_RISE_SPEED_MULT
                            * speedMult;
                        const syncedRiseSpeed = this._getSynchronizedRiseSpeed(
                            v.group.position.y,
                            targetY,
                            baseRiseSpeed,
                            ln2BarrierMaxRemaining
                        );
                        v.group.position.y = Math.min(targetY, v.group.position.y + syncedRiseSpeed * dt);
                    } else {
                        // Always stage LN2 entry at the ring-bottom offset (same as LN1).
                        // Under skip acceleration, post-MHSA vectors may already be above
                        // LN2; without this clamp, the horizontal entry trail can appear
                        // from near LN2 top back to residual.
                        v.group.position.y = targetY;

                        // ────────────────────────────────────────────────
                        //  Reached staging height – begin LN-2 branch
                        // ────────────────────────────────────────────────

                        // Allow residual stream to keep rising while the
                        // duplicate goes through LN-2/MLP.
                        this._updateResidualRiseForMlp();

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
                        // Preserve exact right-angle LN entry geometry during skip: internal LN trails
                        // should not be throttled by global trail-step clamping.
                        if (typeof mvTrail.setMaxStepDistance === 'function') {
                            mvTrail.setMaxStepDistance(1e9);
                        }
                        mvTrail.start(mv.group.position);
                        mv.userData = mv.userData || {};
                        mv.userData.trail = mvTrail;
                        lane.__ln2RightY = mv.group.position.y;
                        lane.movingVecLN2 = mv;
                        lane.normAnimationLN2 = new PrismLayerNormAnimation(mv);

                        this._setLaneLn2Phase(lane, LN2_PHASE.RIGHT, 'ln2 branch spawned');
                        this._emitProgress();
                    }
                    break;
                }
                
                case LN2_PHASE.RIGHT: {
                    // Move horizontally to LN2
                    const mv = lane.movingVecLN2;
                    if (!mv) break;
                    
                    mv.group.visible = true;
                    const rightY = Number.isFinite(lane.__ln2RightY)
                        ? lane.__ln2RightY
                        : mv.group.position.y;
                    mv.group.position.y = rightY;
                    const dx = ANIM_HORIZ_SPEED * speedMult * dt;
                    mv.group.position.x = Math.min(BRANCH_X, mv.group.position.x + dx);
                    
                    if (mv.group.position.x >= BRANCH_X - 0.01) {
                        mv.group.position.x = BRANCH_X;
                        mv.group.position.y = rightY;
                        try {
                            const ln2Trail = mv.userData && mv.userData.trail;
                            if (ln2Trail) {
                                TMP_LN_TRAIL_POS.set(mv.group.position.x, mv.group.position.y, mv.group.position.z);
                                if (typeof ln2Trail.snapLastPointTo === 'function') {
                                    ln2Trail.snapLastPointTo(TMP_LN_TRAIL_POS);
                                } else if (typeof ln2Trail.update === 'function') {
                                    ln2Trail.update(TMP_LN_TRAIL_POS);
                                }
                            }
                        } catch (_) { /* non-fatal trail alignment */ }
                        if (lane.multTargetLN2 && lane.multTargetLN2.group) {
                            lane.multTargetLN2.group.visible = true;
                        }
                        if (lane.addTargetLN2 && lane.addTargetLN2.group) {
                            lane.addTargetLN2.group.visible = true;
                        }
                        this._setLaneLn2Phase(lane, LN2_PHASE.INSIDE_LN, 'ln2 entry reached');
                        this._emitProgress();
                    }

                    break;
                }
                
                case LN2_PHASE.INSIDE_LN: {
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
                    const ln2VectorHeight = (mv && typeof mv.getUniformHeight === 'function')
                        ? mv.getUniformHeight()
                        : NaN;
                    const ln2TouchTriggerY = (Number.isFinite(ln2VectorHeight) && ln2VectorHeight > 0)
                        ? (ln2RiseTargetY - ln2VectorHeight)
                        : ln2RiseTargetY;

                    const startLn2Rise = (vec) => {
                        if (!vec || !vec.group) return;
                        const destY = this.mlpUp.group.position.y - MLP_MATRIX_PARAMS_UP.height / 2 - 10;
                        const dist = destY - vec.group.position.y;
                        const riseSpeed = ANIM_RISE_SPEED_INSIDE_LN * GLOBAL_ANIM_SPEED_MULT;
                        const rawDurationMs = (Number.isFinite(dist) && riseSpeed > 0)
                            ? (dist / riseSpeed) * 1000
                            : 0;
                        const durationMs = Number.isFinite(rawDurationMs) ? rawDurationMs : 0;

                        if (typeof TWEEN === 'undefined' || durationMs <= 0) {
                            vec.group.position.y = Math.max(vec.group.position.y, destY);
                            this._setLaneLn2Phase(lane, LN2_PHASE.MLP_READY, 'ln2 rise complete without tween');
                            this._emitProgress();
                            return;
                        }

                        new TWEEN.Tween(vec.group.position)
                            .to({ y: destY }, durationMs)
                            .easing(TWEEN.Easing.Linear.None)
                            .onUpdate(() => {})
                            .onComplete(() => {
                                this._setLaneLn2Phase(lane, LN2_PHASE.MLP_READY, 'ln2 rise tween complete');
                                this._emitProgress();
                            })
                            .start();
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
                            lane.normAnimationLN2.start(normInput, {
                                deferDataUpdate: true,
                                sourceAlreadyNormalized: !!(ln2NormData && ln2NormData.length)
                            });
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
                        const ln2PreMultiplyTargetY = Math.max(mv.group.position.y, ln2TouchTriggerY);
                        mv.group.position.y = Math.min(
                            ln2PreMultiplyTargetY,
                            mv.group.position.y + riseStep
                        );
                    }
                    
                    // Trigger multiplication at center of LN2
                    const ln2NormAnimationActive = !!(lane.normAnimationLN2 && lane.normAnimationLN2.isAnimating);
                    if (
                        !lane.multDoneLN2 &&
                        lane.normStartedLN2 &&
                        !ln2NormAnimationActive &&
                        mv.group.position.y >= ln2TouchTriggerY - 0.01
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
                        const sourceRaw = (Array.isArray(mv.rawData) || ArrayBuffer.isView(mv.rawData))
                            ? mv.rawData
                            : [];
                        const multData = sourceRaw.slice();
                        if (scaleParamData && (Array.isArray(scaleParamData) || ArrayBuffer.isView(scaleParamData))) {
                            const len = Math.min(multData.length, scaleParamData.length);
                            for (let i = 0; i < len; i++) {
                                multData[i] = (sourceRaw[i] || 0) * (scaleParamData[i] || 0);
                            }
                        }

                        const multSeed = multData.length
                            ? multData
                            : (sourceRaw.length
                                ? sourceRaw.slice()
                                : new Array(Math.max(1, Math.floor(mv.instanceCount || this._getBaseVectorLength()))).fill(0));
                        const multResult = this._createPrismVector(
                            multSeed,
                            (scaleParam && scaleParam.group)
                                ? scaleParam.group.position.clone()
                                : mv.group.position.clone(),
                            30,
                            mv.instanceCount
                        );
                        this.raycastRoot.add(multResult.group);
                        // Keep hidden until colors are fully applied and the multiply
                        // handoff executes, to avoid one-frame constructor defaults.
                        multResult.group.visible = false;

                        const ln2ScaledData = this._getLn2Data(lane, 'scale');
                        const scaledFallback = (ln2ScaledData && ln2ScaledData.length)
                            ? ln2ScaledData
                            : multData;
                        if (scaledFallback && scaledFallback.length) {
                            applyVectorData(
                                multResult,
                                scaledFallback,
                                lane.tokenLabel ? `LN2 Scaled - ${lane.tokenLabel}` : 'LN2 Scaled',
                                this._getLaneMeta(lane, 'ln2.scale')
                            );
                        }

                        const reusedTrailLn2 = mv && mv.userData && mv.userData.trail;
                        const reusedTrailLn2IsWorld = Boolean(mv && mv.userData && mv.userData.trailWorld);
                        if (this._skipToEndActive && reusedTrailLn2 && !reusedTrailLn2IsWorld && mv && mv.group) {
                            const zPos = Number.isFinite(lane.zPos) ? lane.zPos : mv.group.position.z;
                            const clampedY = Math.min(topY_ln2_abs, Math.max(bottomY_ln2_abs, mv.group.position.y));
                            TMP_LN_TRAIL_POS.set(BRANCH_X, clampedY, zPos);
                            if (typeof reusedTrailLn2.update === 'function') {
                                reusedTrailLn2.update(TMP_LN_TRAIL_POS);
                            }
                        }
                        let trailForLn2Result;
                        if (reusedTrailLn2) {
                            trailForLn2Result = reusedTrailLn2;
                            if (reusedTrailLn2IsWorld) {
                                multResult.group.getWorldPosition(TMP_WORLD_POS);
                                if (typeof trailForLn2Result.snapLastPointTo === 'function') {
                                    trailForLn2Result.snapLastPointTo(TMP_WORLD_POS);
                                } else if (typeof trailForLn2Result.update === 'function') {
                                    trailForLn2Result.update(TMP_WORLD_POS);
                                }
                            } else if (typeof trailForLn2Result.snapLastPointTo === 'function') {
                                trailForLn2Result.snapLastPointTo(multResult.group.position);
                            } else if (typeof trailForLn2Result.update === 'function') {
                                trailForLn2Result.update(multResult.group.position);
                            }
                            if (mv.userData) {
                                delete mv.userData.trail;
                                delete mv.userData.trailWorld;
                            }
                        } else {
                            trailForLn2Result = new StraightLineTrail(this.root, 0xffffff, 1, undefined, undefined, TRAIL_MIN_SEGMENT_DISTANCE);
                            trailForLn2Result.start(multResult.group.position);
                        }
                        multResult.userData = multResult.userData || {};
                        multResult.userData.trail = trailForLn2Result;
                        if (reusedTrailLn2IsWorld) {
                            multResult.userData.trailWorld = true;
                        } else if (multResult.userData.trailWorld) {
                            delete multResult.userData.trailWorld;
                        }

                        const startLn2Addition = () => {
                            lane.movingVecLN2 = multResult;
                            lane.normAnimationLN2 = null;
                            const resVec = multResult;

                            const shiftParamData = this._getLayerNormParamData('ln2', 'shift');
                            const shiftSeed = (shiftParamData && shiftParamData.length)
                                ? shiftParamData
                                : (sourceRaw.length
                                    ? sourceRaw.slice()
                                    : new Array(Math.max(1, Math.floor(resVec.instanceCount || this._getBaseVectorLength()))).fill(0));
                            const usingShiftTarget = !!(shiftParam && shiftParam.group);
                            const addResult = usingShiftTarget
                                ? shiftParam
                                : this._createPrismVector(
                                    shiftSeed,
                                    resVec.group.position.clone(),
                                    30,
                                    resVec.instanceCount
                                );
                            if (!usingShiftTarget) {
                                this.raycastRoot.add(addResult.group);
                                addResult.group.visible = false;
                                // Fallback path when param bank target is unavailable.
                                this._applyLayerNormParamVector(addResult, 'ln2', 'shift', null);
                            } else {
                                // Ensure the true shift parameter vector is visible as the
                                // top addend while addition begins.
                                addResult.group.visible = true;
                            }
                            addResult.group.visible = true;

                            const ln2ShiftedData = this._getLn2Data(lane, 'shift');
                            const finalLn2ShiftData = (ln2ShiftedData && ln2ShiftedData.length)
                                ? ln2ShiftedData
                                : (() => {
                                    const count = Math.max(1, Math.floor(resVec.instanceCount || addResult.instanceCount || 1));
                                    const scaledRaw = (Array.isArray(resVec.rawData) || ArrayBuffer.isView(resVec.rawData))
                                        ? resVec.rawData
                                        : [];
                                    const shiftRaw = (Array.isArray(addResult.rawData) || ArrayBuffer.isView(addResult.rawData))
                                        ? addResult.rawData
                                        : [];
                                    const sum = new Array(count);
                                    for (let i = 0; i < count; i++) {
                                        sum[i] = (scaledRaw[i] || 0) + (shiftRaw[i] || 0);
                                    }
                                    return sum;
                                })();
                            logLayerNormVectorDump({
                                layerIndex: this.index,
                                kind: 'ln2',
                                lane,
                                normalizedSaved: this._getLn2Data(lane, 'norm'),
                                normalizedRuntime: sourceRaw,
                                scaleParamSaved: scaleParamData,
                                productComputed: multData,
                                productSaved: ln2ScaledData,
                                productUsedForColor: scaledFallback,
                                shiftParamSaved: shiftParamData,
                                shiftRuntime: addResult.rawData,
                                productPlusShiftComputed: buildDebugVectorSum(multData, shiftParamData, resVec.instanceCount),
                                productPlusShiftSaved: ln2ShiftedData,
                                productPlusShiftUsedForColor: finalLn2ShiftData
                            });
                            lane.resultVecLN2 = addResult;
                            lane.ln2AddStarted = true;
                            // Let the addition animation own trail updates (match LN1 behavior).
                            lane.movingVecLN2 = null;
                            lane.normAnimationLN2 = null;
                            startPrismAdditionAnimation(resVec, addResult, null, () => {
                                lane.ln2AddComplete = true;
                                if (finalLn2ShiftData && finalLn2ShiftData.length) {
                                    applyVectorData(
                                        addResult,
                                        finalLn2ShiftData,
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
                            }, {
                                finalData: finalLn2ShiftData,
                                flashOnTargetImpact: false,
                                progressTarget: lane,
                                progressKey: 'ln2ShiftProgress'
                            });
                        };

                        this._animateMultiplyTransition({
                            sourceVec: mv,
                            multResult,
                            scaleParam,
                            instant: true,
                            onComplete: startLn2Addition
                        });
                    }


                    break;
                }
                
                case LN2_PHASE.MLP_READY:
                    // Ready for MLP animation.
                    this._updateResidualRiseForMlp();
                    if (!lane.mlpUpStarted && lane.resultVecLN2) {
                        lane.mlpUpStarted = true;
                        this._emitProgress();
                        this._animateMlpUpProjection(lane);
                    }
                    break;
                    
                case LN2_PHASE.DONE:
                default:
                    break;
            }
        });

        // Keep post-MLP return motion synchronized across lanes.
        this._tryStartSynchronizedMlpReturn();

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

        // Global safety net for skip flows: if any lane's phase signature is
        // unchanged for too long, force a conservative forward step to avoid
        // whole-layer deadlocks. Keep normal playback strictly stage-gated.
        const postAttentionWatchLanes = this._postAttentionWatchLanes;
        postAttentionWatchLanes.length = 0;
        for (let i = 0; i < lanes.length; i++) {
            const lane = lanes[i];
            if (
                lane
                && (
                    lane.horizPhase === HORIZ_PHASE.POST_MHSA_ADDITION
                    || lane.horizPhase === HORIZ_PHASE.WAITING_FOR_LN2
                    || lane.ln2Phase !== LN2_PHASE.NOT_STARTED
                )
            ) {
                postAttentionWatchLanes.push(lane);
            }
        }

        if (skipActive || this._skipLayerActive) {
            this._applyLaneStallWatchdog(lanes, nowMs, {
                ln2SyncY,
                timeoutMs: LANE_PHASE_STALL_TIMEOUT_MS_SKIP,
                allowOriginalFallback: true,
                postAttentionOnly: false
            });
        } else if (postAttentionWatchLanes.length > 0) {
            // Keep normal playback strict for LN1/MHSA, but recover from post-attention
            // deadlocks that can surface after focus/visibility interruptions.
            this._applyLaneStallWatchdog(postAttentionWatchLanes, nowMs, {
                ln2SyncY,
                timeoutMs: LANE_PHASE_STALL_TIMEOUT_MS_NORMAL,
                allowOriginalFallback: false,
                postAttentionOnly: true
            });
        } else {
            this._resetLaneStallWatchdog(lanes);
        }

        // ----------------------------------------------------------
        // Notify LayerPipeline once **all** lanes have finished AND all additions complete
        // ----------------------------------------------------------
        if (this.isActive && !this._completed && this.lanes.length && 
            this.lanes.every(l => l.ln2Phase === LN2_PHASE.DONE) && this._pendingAdditions === 0) {
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
    }

    /**
     * Animate vector through MLP up-projection (768 → 3072 dimensions)
     */
    _animateMlpUpProjection(lane) {
        const vec = lane && lane.resultVecLN2;
        if (!lane || !vec || !vec.group) {
            if (lane) {
                lane.mlpUpStarted = true;
                lane.mlpDownStarted = true;
                lane.mlpDownComplete = true;
                this._setLaneLn2Phase(lane, LN2_PHASE.DONE, 'mlp up skipped (missing vec)');
                this._emitProgress();
            }
            return;
        }
        if (typeof TWEEN === 'undefined') {
            lane.mlpDownStarted = true;
            lane.mlpDownComplete = true;
            this._setLaneLn2Phase(lane, LN2_PHASE.DONE, 'mlp up skipped (no tween)');
            this._emitProgress();
            return;
        }

        const bottomY = this.mlpUp.group.position.y - MLP_MATRIX_PARAMS_UP.height / 2;
        const topY = this.mlpUp.group.position.y + MLP_MATRIX_PARAMS_UP.height / 2;
        const matrixStartColor = this._mlpMatrixInactiveColor;
        const matrixEndColor = this._mlpMatrixActiveColor;
        const tweenColor = this._mlpUpTweenColor;
        const startIntensity = MLP_MATRIX_FLASH_START_EMISSIVE;
        const peakIntensity = MLP_MATRIX_FLASH_PEAK_EMISSIVE_UP;
        const finalIntensity = MLP_POST_PASS_THROUGH_FINAL_EMISSIVE;
        const distance = topY - vec.group.position.y;
        const rawDuration = (distance / (ANIM_RISE_SPEED_INSIDE_LN * GLOBAL_ANIM_SPEED_MULT)) * 1000;
        const mlpProfile = this._resolveMlpTransitionProfile();
        let duration = Number.isFinite(rawDuration) ? Math.max(0, rawDuration) : 0;
        if (Number.isFinite(mlpProfile.maxUpDurationMs)) {
            duration = Math.min(duration, mlpProfile.maxUpDurationMs);
        }
        if (duration <= 0) {
            vec.group.position.y = Math.max(vec.group.position.y, topY);
            vec.group.scale.setScalar(0.6);
            this.mlpUp.setColor(matrixEndColor);
            this.mlpUp.setEmissive(matrixEndColor, finalIntensity);
            const mlpUpData = this._getMlpUpData(lane);
            this._expandTo4x(lane, vec, mlpUpData);
            return;
        }
        const colorDuration = this._resolveMlpColorDuration(duration, mlpProfile);

        // Animate matrix colour and emissive intensity for a glow effect
        const state = { t: 0, emissive: startIntensity };
        new TWEEN.Tween(state)
            .to({ t: 1, emissive: peakIntensity }, colorDuration * 0.6)
            .easing(TWEEN.Easing.Quadratic.InOut)
            .onUpdate(() => {
                const col = tweenColor.copy(matrixStartColor).lerp(matrixEndColor, state.t);
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
        const mlpProfile = this._resolveMlpTransitionProfile();
        const expandedGroup = new THREE.Group();
        const segmentVecs = [];
        const segmentBatch = new BatchedPrismVectorSet({
            vectorCount: segments,
            prismCount: segmentLength,
            parentGroup: expandedGroup,
            label: 'MLP Expanded Segments',
        });
        const paddedData = (Array.isArray(mlpUpData) || ArrayBuffer.isView(mlpUpData))
            ? Array.from(mlpUpData)
            : null;
        if (paddedData && paddedData.length < segments * segmentLength) {
            while (paddedData.length < segments * segmentLength) paddedData.push(0);
        }

        // Create 4 segments side by side
        for (let s = 0; s < segments; s++) {
            const segVec = segmentBatch.getVectorRef(s);
            segVec.group.visible = true;
            segVec.rawData = (Array.isArray(vec.rawData) || ArrayBuffer.isView(vec.rawData))
                ? Array.from(vec.rawData)
                : [];

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
        // During skip, carry forward the LN2 trail so we preserve a single
        // continuous LN2 -> MLP -> residual path with no shortcut segment.
        let expTrail = null;
        let expTrailWorld = false;
        if (
            this._skipToEndActive
            && vec
            && vec.userData
            && vec.userData.trail
        ) {
            expTrail = vec.userData.trail;
            expTrailWorld = !!vec.userData.trailWorld;
            delete vec.userData.trail;
            delete vec.userData.trailWorld;
            if (expTrailWorld) {
                expandedGroup.getWorldPosition(TMP_WORLD_POS);
                if (typeof expTrail.snapLastPointTo === 'function') {
                    expTrail.snapLastPointTo(TMP_WORLD_POS);
                } else if (typeof expTrail.update === 'function') {
                    expTrail.update(TMP_WORLD_POS);
                }
            } else if (typeof expTrail.snapLastPointTo === 'function') {
                expTrail.snapLastPointTo(expandedGroup.position);
            } else if (typeof expTrail.update === 'function') {
                expTrail.update(expandedGroup.position);
            }
        } else {
            expTrail = new StraightLineTrail(this.root, 0xffffff, 1, undefined, undefined, TRAIL_MIN_SEGMENT_DISTANCE);
            expTrail.start(expandedGroup.position);
        }
        lane.expandedVecTrail = expTrail;
        lane.expandedVecTrailWorld = expTrailWorld;
        
        lane.expandedVecGroup = expandedGroup;
        lane.expandedVecSegments = segmentVecs;
        
        // Rise before down-projection
        const extraRise = mlpProfile.expandRiseUnits;
        const pauseMs = 0;

        new TWEEN.Tween(expandedGroup.position)
            .to({ y: expandedGroup.position.y + extraRise }, mlpProfile.expandRiseMs)
            .easing(TWEEN.Easing.Quadratic.InOut)
            .onUpdate(() => {
                const trail = lane && lane.expandedVecTrail;
                if (trail && typeof trail.update === 'function') {
                    if (lane.expandedVecTrailWorld) {
                        expandedGroup.getWorldPosition(TMP_WORLD_POS);
                        trail.update(TMP_WORLD_POS);
                    } else {
                        trail.update(expandedGroup.position);
                    }
                }
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
                    this._animateMlpActivationGelu(lane, applyActivation, startDown, {
                        durationMs: mlpProfile.geluDurationMs,
                        riseExtra: mlpProfile.geluRiseExtra,
                        curveHeight: mlpProfile.geluCurveHeight,
                        activationSwitchT: mlpProfile.geluActivationSwitchT,
                        liteMode: !!mlpProfile.geluLiteMode
                    });
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
        const liteMode = options.liteMode === true;
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

        const baseY = expandedGroup.position.y;
        let activationApplied = false;
        const applyActivationOnce = () => {
            if (activationApplied) return;
            activationApplied = true;
            if (typeof applyActivation === 'function') applyActivation();
        };
        const updateExpandedTrail = () => {
            const expandedTrail = lane && lane.expandedVecTrail;
            if (!expandedTrail || typeof expandedTrail.update !== 'function') return;
            if (lane.expandedVecTrailWorld) {
                expandedGroup.getWorldPosition(TMP_WORLD_POS);
                expandedTrail.update(TMP_WORLD_POS);
            } else {
                expandedTrail.update(expandedGroup.position);
            }
        };

        if (liteMode) {
            const state = { t: 0 };
            new TWEEN.Tween(state)
                .to({ t: 1 }, durationMs)
                .easing(TWEEN.Easing.Quadratic.InOut)
                .onUpdate(() => {
                    if (state.t >= activationSwitchT) {
                        applyActivationOnce();
                    }
                    const liftProgress = state.t >= activationSwitchT
                        ? (state.t - activationSwitchT) / Math.max(1e-6, 1 - activationSwitchT)
                        : 0;
                    const liftT = TWEEN.Easing.Quadratic.Out(THREE.MathUtils.clamp(liftProgress, 0, 1));
                    expandedGroup.position.y = baseY + riseExtra * liftT;
                    updateExpandedTrail();
                })
                .onComplete(() => {
                    applyActivationOnce();
                    expandedGroup.position.y = baseY + riseExtra;
                    finishGelu();
                    if (typeof onComplete === 'function') onComplete();
                })
                .start();
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

        const state = { t: 0 };

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
                updateExpandedTrail();

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
            if (lane) {
                lane.mlpDownStarted = true;
                lane.mlpDownComplete = true;
                if (!lane.finalVecAfterMlp && lane.resultVecLN2 && lane.resultVecLN2.group) {
                    lane.finalVecAfterMlp = lane.resultVecLN2;
                }
                this._tryStartSynchronizedMlpReturn();
            }
            return;
        }
        lane.mlpDownStarted = true;
        lane.mlpDownComplete = false;
        lane.collapsedInMatrix = false;
        lane.finalVecAfterMlp = null;
        
        const orangeColor = this._mlpMatrixActiveColor;
        const downTweenColor = this._mlpDownTweenColor;
        const downBottomY = this.mlpDown.group.position.y - MLP_MATRIX_PARAMS_DOWN.height / 2;
        const downTopY = this.mlpDown.group.position.y + MLP_MATRIX_PARAMS_DOWN.height / 2;
        // Collapse before matrix entry so the 3072-dim visual never protrudes through the shell.
        const preEntryCollapseLead = THREE.MathUtils.clamp(MLP_MATRIX_PARAMS_DOWN.height * 0.22, 8, 30);
        const collapseTriggerY = downBottomY - preEntryCollapseLead;
        
        const startY = expandedGroup.position.y;
        const totalDist = downTopY - startY;
        const mlpProfile = this._resolveMlpTransitionProfile();
        let durationDown = (Math.abs(totalDist) / (ANIM_RISE_SPEED_INSIDE_LN * GLOBAL_ANIM_SPEED_MULT)) * 1000;
        if (Number.isFinite(mlpProfile.maxDownDurationMs)) {
            durationDown = Math.min(durationDown, mlpProfile.maxDownDurationMs);
        }
        const colorDurationDown = this._resolveMlpColorDuration(durationDown, mlpProfile);

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
        const updateExpandedTrail = () => {
            const trail = lane && lane.expandedVecTrail;
            if (!trail || typeof trail.update !== 'function') return;
            if (lane.expandedVecTrailWorld) {
                expandedGroup.getWorldPosition(TMP_WORLD_POS);
                trail.update(TMP_WORLD_POS);
            } else {
                trail.update(expandedGroup.position);
            }
        };
        const attachExpandedTrailToVec = (targetVec) => {
            if (!targetVec || !targetVec.group || !lane || !lane.expandedVecTrail) return;
            targetVec.userData = targetVec.userData || {};
            targetVec.userData.trail = lane.expandedVecTrail;
            if (lane.expandedVecTrailWorld) {
                targetVec.userData.trailWorld = true;
                targetVec.group.getWorldPosition(TMP_WORLD_POS);
                if (typeof lane.expandedVecTrail.snapLastPointTo === 'function') {
                    lane.expandedVecTrail.snapLastPointTo(TMP_WORLD_POS);
                } else {
                    lane.expandedVecTrail.update(TMP_WORLD_POS);
                }
            } else {
                delete targetVec.userData.trailWorld;
                if (typeof lane.expandedVecTrail.snapLastPointTo === 'function') {
                    lane.expandedVecTrail.snapLastPointTo(targetVec.group.position);
                } else {
                    lane.expandedVecTrail.update(targetVec.group.position);
                }
            }
            lane.expandedVecTrail = null;
            lane.expandedVecTrailWorld = false;
        };
        
        // Matrix colour + emissive animation for glow
        const startIntensity = MLP_MATRIX_FLASH_START_EMISSIVE;
        const peakIntensity = MLP_MATRIX_FLASH_PEAK_EMISSIVE_DOWN;
        const finalIntensity = MLP_POST_PASS_THROUGH_FINAL_EMISSIVE;
        const downState = { t: 0, emissive: startIntensity };

        new TWEEN.Tween(downState)
            .to({ t: 1, emissive: peakIntensity }, colorDurationDown * 0.6)
            .easing(TWEEN.Easing.Quadratic.InOut)
            .onUpdate(() => {
                const col = downTweenColor.copy(this._mlpMatrixInactiveColor).lerp(orangeColor, downState.t);
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
                updateExpandedTrail();
                if (expandedGroup.position.y >= downBottomY) {
                    const maxScale = clampScaleForY(expandedGroup.position.y);
                    if (Number.isFinite(maxScale)) {
                        const nextScale = Math.min(expandedGroup.scale.x, maxScale);
                        if (nextScale !== expandedGroup.scale.x) {
                            expandedGroup.scale.setScalar(nextScale);
                        }
                    }
                }

                // Transition to the 768-dim collapsed vector early in the matrix so
                // the wider source visual is gone before the top taper gets narrow.
                if (!lane.collapsedInMatrix && expandedGroup.position.y >= collapseTriggerY) {
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
                    attachExpandedTrailToVec(collapseVec);
                    
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
                    const remainingDist = Math.max(0, downTopY - expandedGroup.position.y);
                    const remainingDuration = (remainingDist / (ANIM_RISE_SPEED_INSIDE_LN * GLOBAL_ANIM_SPEED_MULT)) * 1000;
                    new TWEEN.Tween(collapseVec.group.position)
                        .to({ y: downTopY }, remainingDuration)
                        .easing(TWEEN.Easing.Linear.None)
                        .onUpdate(() => {
                            const collapseTrail = collapseVec.userData && collapseVec.userData.trail;
                            if (collapseTrail && typeof collapseTrail.update === 'function') {
                                if (collapseVec.userData.trailWorld) {
                                    collapseVec.group.getWorldPosition(TMP_WORLD_POS);
                                    collapseTrail.update(TMP_WORLD_POS);
                                } else {
                                    collapseTrail.update(collapseVec.group.position);
                                }
                            }
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
                }
                this._tryStartSynchronizedMlpReturn();
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
        if (lane && lane.expandedVecTrail) {
            collapseVec.userData = collapseVec.userData || {};
            collapseVec.userData.trail = lane.expandedVecTrail;
            if (lane.expandedVecTrailWorld) {
                collapseVec.userData.trailWorld = true;
                collapseVec.group.getWorldPosition(TMP_WORLD_POS);
                if (typeof lane.expandedVecTrail.snapLastPointTo === 'function') {
                    lane.expandedVecTrail.snapLastPointTo(TMP_WORLD_POS);
                } else {
                    lane.expandedVecTrail.update(TMP_WORLD_POS);
                }
            } else {
                delete collapseVec.userData.trailWorld;
                if (typeof lane.expandedVecTrail.snapLastPointTo === 'function') {
                    lane.expandedVecTrail.snapLastPointTo(collapseVec.group.position);
                } else {
                    lane.expandedVecTrail.update(collapseVec.group.position);
                }
            }
            lane.expandedVecTrail = null;
            lane.expandedVecTrailWorld = false;
        }
        
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
    }

    _updateResidualRiseForMlp(targetY = null) {
        const mhsa = this.mhsaAnimation;
        if (!mhsa) return;

        let resolvedTarget = targetY;
        if (!Number.isFinite(resolvedTarget) && this.mlpUp && this.mlpUp.group) {
            resolvedTarget = this.mlpUp.group.position.y + MLP_MATRIX_PARAMS_UP.height / 2 - ORIGINAL_TO_PROCESSED_GAP;
        }
        if (!Number.isFinite(resolvedTarget)) return;

        if (!Number.isFinite(mhsa.finalOriginalY) || resolvedTarget > mhsa.finalOriginalY) {
            mhsa.finalOriginalY = resolvedTarget;
        }
        if (!Number.isFinite(mhsa.maxResidualRiseY) || resolvedTarget > mhsa.maxResidualRiseY) {
            mhsa.maxResidualRiseY = resolvedTarget;
        }
        mhsa.postSplitRiseSpeed = ANIM_RISE_SPEED_POST_SPLIT_LN2;
    }
    
    /**
     * Rise above matrix after MLP processing
     */
    _riseAfterMlp(lane) {
        if (!lane || lane.mlpReturnStarted) return;
        const vec = lane.finalVecAfterMlp;
        if (!vec) return;
        lane.mlpReturnStarted = true;
        
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
                    if (this._skipToEndActive && vec.userData.trail) {
                        vec.userData.mlpTrail = vec.userData.trail;
                        if (vec.userData.trailWorld) {
                            vec.userData.mlpTrailWorld = true;
                            vec.group.getWorldPosition(TMP_WORLD_POS);
                            if (typeof vec.userData.mlpTrail.snapLastPointTo === 'function') {
                                vec.userData.mlpTrail.snapLastPointTo(TMP_WORLD_POS);
                            } else if (typeof vec.userData.mlpTrail.update === 'function') {
                                vec.userData.mlpTrail.update(TMP_WORLD_POS);
                            }
                        } else {
                            delete vec.userData.mlpTrailWorld;
                            if (typeof vec.userData.mlpTrail.snapLastPointTo === 'function') {
                                vec.userData.mlpTrail.snapLastPointTo(vec.group.position);
                            } else if (typeof vec.userData.mlpTrail.update === 'function') {
                                vec.userData.mlpTrail.update(vec.group.position);
                            }
                        }
                        delete vec.userData.trail;
                        delete vec.userData.trailWorld;
                    } else if (!vec.userData.mlpTrail) {
                        // Freeze the incoming MLP traversal trail at handoff so only the
                        // dedicated post-MLP trail is updated during rise/return.
                        if (vec.userData.trail) {
                            if (vec.userData.trailWorld) {
                                vec.group.getWorldPosition(TMP_WORLD_POS);
                                if (typeof vec.userData.trail.snapLastPointTo === 'function') {
                                    vec.userData.trail.snapLastPointTo(TMP_WORLD_POS);
                                } else if (typeof vec.userData.trail.update === 'function') {
                                    vec.userData.trail.update(TMP_WORLD_POS);
                                }
                            } else if (typeof vec.userData.trail.snapLastPointTo === 'function') {
                                vec.userData.trail.snapLastPointTo(vec.group.position);
                            } else if (typeof vec.userData.trail.update === 'function') {
                                vec.userData.trail.update(vec.group.position);
                            }
                            delete vec.userData.trail;
                            delete vec.userData.trailWorld;
                        }
                        const pathTrail = new StraightLineTrail(
                            this.root,
                            0xffffff,
                            1,
                            undefined,
                            POST_MLP_RETURN_TRAIL_OPACITY,
                            TRAIL_MIN_SEGMENT_DISTANCE
                        );
                        pathTrail.start(vec.group.position.clone());
                        vec.userData.mlpTrail = pathTrail;
                        delete vec.userData.mlpTrailWorld;
                    }
                    const mlpTrail = vec.userData && vec.userData.mlpTrail;
                    if (mlpTrail && typeof mlpTrail.setBaseOpacity === 'function') {
                        mlpTrail.setBaseOpacity(POST_MLP_RETURN_TRAIL_OPACITY);
                    }
                } catch (_) { /* optional visual */ }
            })
            .onUpdate(() => {
                const t = vec && vec.userData && vec.userData.mlpTrail;
                if (t && typeof t.update === 'function') {
                    if (vec.userData && vec.userData.mlpTrailWorld) {
                        vec.group.getWorldPosition(TMP_WORLD_POS);
                        t.update(TMP_WORLD_POS);
                    } else {
                        t.update(vec.group.position);
                    }
                }
            })
            .onComplete(() => {
                // Update residual stream target height
                this._updateResidualRiseForMlp(vec.group.position.y - ORIGINAL_TO_PROCESSED_GAP);
                
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
                    if (t && typeof t.update === 'function') {
                        if (vec.userData && vec.userData.mlpTrailWorld) {
                            vec.group.getWorldPosition(TMP_WORLD_POS);
                            t.update(TMP_WORLD_POS);
                        } else {
                            t.update(vec.group.position);
                        }
                    }
                } catch (_) { /* optional */ }
            })
            .onComplete(() => {
                // Freeze the temporary post-MLP path trail into a static line so
                // the horizontal segment persists visually after the move back.
                try {
                    const t = vec && vec.userData && vec.userData.mlpTrail;
                    if (t) {
                        if (typeof t.snapLastPointTo === 'function') {
                            if (vec.userData && vec.userData.mlpTrailWorld) {
                                vec.group.getWorldPosition(TMP_WORLD_POS);
                                t.snapLastPointTo(TMP_WORLD_POS);
                            } else {
                                t.snapLastPointTo(vec.group.position);
                            }
                        }
                        const colorHex = (t._material && t._material.color)
                            ? t._material.color.getHex() : undefined;
                        const frozenLineWidth = (typeof t._lineWidth === 'number') ? t._lineWidth : undefined;
                        const frozenOpacity = (typeof t._opacity === 'number') ? t._opacity : undefined;
                        mergeTrailsIntoLineSegments(
                            [t],
                            this.root,
                            colorHex,
                            frozenLineWidth,
                            frozenOpacity,
                            null
                        );
                        if (typeof t.dispose === 'function') t.dispose();
                        if (vec && vec.userData) {
                            delete vec.userData.mlpTrail;
                            delete vec.userData.mlpTrailWorld;
                        }
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
                        const frozenLineWidth = (typeof localTrail._lineWidth === 'number') ? localTrail._lineWidth : undefined;
                        const frozenOpacity = (typeof localTrail._opacity === 'number') ? localTrail._opacity : undefined;
                        mergeTrailsIntoLineSegments(
                            [localTrail],
                            this.root,
                            colorHex,
                            frozenLineWidth,
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
                    this.mhsaAnimation._startAdditionAnimation(
                        lane.originalVec,
                        vec,
                        lane,
                        () => {
                            if (postMlpData) {
                                applyVectorData(
                                    lane.originalVec,
                                    postMlpData,
                                    lane.tokenLabel ? `Post-MLP Residual - ${lane.tokenLabel}` : 'Post-MLP Residual',
                                    this._getLaneMeta(lane, 'residual.post_mlp')
                                );
                            }
                        },
                        { cameraHoldAfterAdditionMs: 220 }
                    );
                    
                    // Set up completion callback for when addition finishes
                    this._scheduleAdditionCompletion(lane);
                }
                this._setLaneLn2Phase(lane, LN2_PHASE.DONE, 'mlp return to residual complete');
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
        const wasActive = this._skipToEndActive;
        this._skipToEndActive = !!enabled;
        if (this._skipToEndActive) {
            this._skipVectorMeshIndexDirty = true;
            this._skipVisibilityDirty = true;
            this._applySkipVectorVisibility({ force: true });
        } else if (wasActive) {
            this._restoreSkipHiddenVectorMaterials();
            this._skipVisibilityDirty = false;
            this._skipVisibilityLastApplyMs = 0;
            this._skipVectorMeshIndexDirty = true;
        }
        if (this.mhsaAnimation && typeof this.mhsaAnimation.setSkipToEndMode === 'function') {
            this.mhsaAnimation.setSkipToEndMode(this._skipToEndActive);
        }
    }

    postUpdate() {
        if (this._skipToEndActive) {
            // During fast-skip, vector visibility can be re-enabled by async
            // callbacks between frames; force a sweep every frame to prevent
            // transient "straggler" vectors from flashing onscreen.
            this._applySkipVectorVisibility({ force: true });
        }
    }

    dispose() {
        try {
            this.mhsaAnimation?.dispose?.();
        } catch (_) { /* best-effort */ }
        this.mhsaAnimation = null;
        super.dispose();
    }

    refreshSkipVisibility() {
        if (this._skipToEndActive) {
            this._skipVectorMeshIndexDirty = true;
            this._skipVisibilityDirty = true;
            this._applySkipVectorVisibility({ force: true });
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
                    if (typeof prev.depthTest === 'boolean') mat.depthTest = prev.depthTest;
                    mat.needsUpdate = true;
                    hidden.delete(mat);
                }
            });
        });
    }

    _restoreSkipHiddenVectorMaterials() {
        if (!this.root) return;
        const hidden = this._skipHiddenMaterials;
        const vectorMeshes = this._getSkipVectorMeshes(this._getNowMs(), { force: true });
        for (const obj of vectorMeshes) {
            if (!obj || !obj.isMesh || !obj.material || !obj.parent) {
                vectorMeshes.delete(obj);
                continue;
            }
            const mats = Array.isArray(obj.material) ? obj.material : [obj.material];
            let restoredAny = false;
            mats.forEach(mat => {
                if (!mat) return;
                const prev = hidden.get(mat);
                if (!prev) return;
                mat.opacity = prev.opacity;
                mat.transparent = prev.transparent;
                mat.depthWrite = prev.depthWrite;
                if (typeof prev.depthTest === 'boolean') mat.depthTest = prev.depthTest;
                mat.needsUpdate = true;
                hidden.delete(mat);
                restoredAny = true;
            });
            if (restoredAny) {
                obj.visible = true;
            }
        }
        this._skipVectorMeshIndexDirty = true;
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
        // Fallback for prism vectors that may have had their labels replaced:
        // they all share this stable shader cache key.
        if (obj.material) {
            const mats = Array.isArray(obj.material) ? obj.material : [obj.material];
            for (let i = 0; i < mats.length; i++) {
                const mat = mats[i];
                if (!mat || typeof mat.customProgramCacheKey !== 'function') continue;
                try {
                    if (mat.customProgramCacheKey() === 'InstancedPrismGradientV1') {
                        return true;
                    }
                } catch (_) { /* ignore cache-key probe failures */ }
            }
        }
        return false;
    }

    _rebuildSkipVectorMeshIndex(nowMs = this._getNowMs()) {
        const index = this._skipVectorMeshes;
        index.clear();
        if (!this.root) {
            this._skipVectorMeshIndexDirty = false;
            this._skipVectorMeshIndexLastBuildMs = nowMs;
            return;
        }
        this.root.traverse((obj) => {
            if (!obj || !obj.isMesh || !obj.material) return;
            if (!this._isVectorVisual(obj)) return;
            index.add(obj);
        });
        this._skipVectorMeshIndexDirty = false;
        this._skipVectorMeshIndexLastBuildMs = nowMs;
    }

    _getSkipVectorMeshes(nowMs = this._getNowMs(), { force = false } = {}) {
        const index = this._skipVectorMeshes;
        const shouldRebuild = this._skipVectorMeshIndexDirty
            || index.size === 0
            || force
            || (nowMs - this._skipVectorMeshIndexLastBuildMs) >= SKIP_VISIBILITY_MESH_INDEX_REFRESH_MS;
        if (shouldRebuild) {
            this._rebuildSkipVectorMeshIndex(nowMs);
        }
        return index;
    }

    _applySkipVectorVisibility({ force = false } = {}) {
        if (!this._skipToEndActive || !this.root) return;
        const now = this._getNowMs();
        if (!force) {
            const elapsed = now - this._skipVisibilityLastApplyMs;
            if (!this._skipVisibilityDirty && elapsed < SKIP_VISIBILITY_REFRESH_MS) {
                return;
            }
        }
        this._skipVisibilityDirty = false;
        this._skipVisibilityLastApplyMs = now;
        const hidden = this._skipHiddenMaterials;
        const vectorMeshes = this._getSkipVectorMeshes(now, { force });
        for (const obj of vectorMeshes) {
            if (!obj || !obj.isMesh || !obj.material || !obj.parent) {
                vectorMeshes.delete(obj);
                continue;
            }
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
                        if (typeof prev.depthTest === 'boolean') mat.depthTest = prev.depthTest;
                        mat.needsUpdate = true;
                        hidden.delete(mat);
                    }
                });
                continue;
            }
            mats.forEach(mat => {
                if (!mat) return;
                if (!hidden.has(mat)) {
                    hidden.set(mat, {
                        opacity: mat.opacity,
                        transparent: mat.transparent,
                        depthWrite: mat.depthWrite,
                        depthTest: mat.depthTest
                    });
                }
                const needsUpdate = mat.opacity !== 0 || mat.transparent !== true || mat.depthWrite !== false;
                mat.transparent = true;
                mat.opacity = 0;
                mat.depthWrite = false;
                if (needsUpdate) mat.needsUpdate = true;
            });
            obj.visible = false;
        }
    }

    _getNowMs() {
        return (typeof performance !== 'undefined' && typeof performance.now === 'function')
            ? performance.now()
            : Date.now();
    }

    _updatePositionalPassBarrier({
        lanes,
        nowMs,
        skipActive = false,
        hasPendingPosPass = false,
        allPosPassReady = false
    } = {}) {
        if (this.index !== 0) return;
        const laneList = Array.isArray(lanes) ? lanes : [];

        if (hasPendingPosPass && allPosPassReady && !this._posPassBarrierArmed) {
            const pauseMs = skipActive ? 0 : POS_PASS_START_PAUSE_MS;
            this._posPassBarrierArmed = true;
            this._posPassStartAtMs = nowMs + pauseMs;
        }
        if (
            this._posPassBarrierArmed
            && Number.isFinite(this._posPassStartAtMs)
            && nowMs >= this._posPassStartAtMs
        ) {
            const pendingPosPassLanes = laneList.filter((lane) => (
                lane
                && !lane.posAddComplete
                && !lane.__posPassStarted
                && lane.posVec
                && typeof lane.startPositionalPassThrough === 'function'
            ));
            // Keep positional pass-through synchronized: launch only when every
            // pending lane has its position-chip gate released.
            const allPositionChipGatesReleased = pendingPosPassLanes.every((lane) => (
                !this._isWaitingForInputPositionChipGate(lane, nowMs, skipActive)
            ));
            if (allPositionChipGatesReleased) {
                pendingPosPassLanes.forEach((lane) => {
                    try {
                        lane.startPositionalPassThrough({ immediate: skipActive });
                    } catch (_) {
                        lane.posAddComplete = true;
                    }
                });
            }
            const stillPending = laneList.some((lane) => {
                if (!lane || lane.posAddComplete) return false;
                return !!(lane.posVec && !lane.__posPassStarted);
            });
            if (stillPending) {
                // Continue polling until all pending lanes can launch together.
                this._posPassBarrierArmed = true;
                this._posPassStartAtMs = nowMs;
            } else {
                this._posPassBarrierArmed = false;
                this._posPassStartAtMs = NaN;
            }
        }
        if (!hasPendingPosPass) {
            this._posPassBarrierArmed = false;
            this._posPassStartAtMs = NaN;
        }
    }

    _updateLn2AndMlpStartGates({
        allLn2Ready = false,
        allMlpReady = false
    } = {}) {
        // LN-2 synchronisation check – advance as soon as every lane has
        // completed post-attention addition and reached LN-2 staging height.
        if (!this._ln2Ready && allLn2Ready) {
            this._ln2Ready = true;
            this._emitProgress();
            this._debugLayerLifecycleLog(`Layer ${this.index}: All lanes ready – starting LN2 simultaneously`);
        }

        // MLP synchronisation: wait until every lane has completed LN-2 and
        // is marked as 'mlpReady' before triggering the up-projection.
        if (!this._mlpStart && allMlpReady) {
            this._mlpStart = true;
            this._emitProgress();
            this._debugLayerLifecycleLog(`Layer ${this.index}: All lanes ready – starting MLP up-projection simultaneously`);
        }

    }

    _tryStartSynchronizedMlpReturn() {
        if (this._mlpReturnStart) return;
        const lanes = Array.isArray(this.lanes) ? this.lanes : [];
        if (!lanes.length) return;

        const pendingReturnLanes = lanes.filter((lane) => (
            lane
            && lane.ln2Phase === LN2_PHASE.MLP_READY
            && !lane.mlpReturnStarted
        ));
        if (!pendingReturnLanes.length) return;

        const allReadyForReturn = lanes.every((lane) => {
            if (!lane) return true;
            if (lane.ln2Phase === LN2_PHASE.DONE) return true;
            if (lane.ln2Phase !== LN2_PHASE.MLP_READY) return false;
            return !!(
                lane.mlpDownComplete
                && lane.finalVecAfterMlp
                && lane.finalVecAfterMlp.group
            );
        });
        if (!allReadyForReturn) return;

        this._mlpReturnStart = true;
        this._debugLayerLifecycleLog(
            `Layer ${this.index}: All lanes cleared MLP down-projection – starting post-MLP return simultaneously`
        );
        pendingReturnLanes.forEach((lane) => {
            this._riseAfterMlp(lane);
        });
        this._emitProgress();
    }

    _updateLn1AndMhsaStartGates({
        allLn1Ready = false,
        posAddDone = false,
        allMhsaReady = false,
        nowMs = NaN,
        skipActive = false
    } = {}) {
        // LayerNorm-1 synchronisation barrier – wait until every lane's
        // residual vector has reached branch height before releasing LN-1.
        if (!this._ln1Start) {
            const readyToStartLn1 = allLn1Ready && posAddDone;
            if (readyToStartLn1 && !this._ln1StartBarrierArmed) {
                const pauseMs = (!skipActive && this.index === 0)
                    ? FIRST_LAYER_LN1_START_PAUSE_MS
                    : 0;
                this._ln1StartBarrierArmed = true;
                this._ln1StartAtMs = nowMs + pauseMs;
            }

            const canStartLn1 = readyToStartLn1
                && (
                    !this._ln1StartBarrierArmed
                    || !Number.isFinite(this._ln1StartAtMs)
                    || nowMs >= this._ln1StartAtMs
                );
            if (canStartLn1) {
                this._ln1Start = true;
                this._ln1StartBarrierArmed = false;
                this._ln1StartAtMs = NaN;
                this._emitProgress();
                this._debugLayerLifecycleLog(`Layer ${this.index}: All lanes ready – starting LN-1 branch simultaneously`);

                const laneList = Array.isArray(this.lanes) ? this.lanes : [];
                laneList.forEach((lane) => {
                    if (lane.horizPhase !== HORIZ_PHASE.WAITING) return;
                    this._setLaneHorizPhase(lane, HORIZ_PHASE.RIGHT, 'ln1 barrier released');
                    this._emitProgress();
                    // Ensure the branch duplicate matches the latest residual data
                    // (e.g., after positional embedding addition).
                    copyVectorAppearance(lane.dupVec, lane.originalVec);
                    lane.dupVec.group.visible = true;
                    // Snap duplicate to the LN-1 branch staging height to avoid
                    // vertical drift while moving horizontally into the ring.
                    if (typeof lane.branchStartY === 'number') {
                        lane.dupVec.group.position.y = lane.branchStartY;
                    } else {
                        lane.dupVec.group.position.y = lane.originalVec.group.position.y;
                    }
                });
            } else if (!readyToStartLn1) {
                this._ln1StartBarrierArmed = false;
                this._ln1StartAtMs = NaN;
            }
        }

        // MHSA travel synchronisation barrier – wait until every lane has its
        // duplicate result vector staged above LN-1 before head travel starts.
        if (!this._mhsaStart && allMhsaReady) {
            this._mhsaStart = true;
            this._emitProgress();
            this._debugLayerLifecycleLog(`Layer ${this.index}: All lanes ready – starting travel to MHSA heads simultaneously`);
            const laneList = Array.isArray(this.lanes) ? this.lanes : [];
            laneList.forEach((lane) => {
                if (lane.horizPhase !== HORIZ_PHASE.READY_MHSA) return;
                this._setLaneHorizPhase(lane, HORIZ_PHASE.TRAVEL_MHSA, 'mhsa barrier released');
                lane.__mhsaTrailCornerPending = true;
                this._emitProgress();
            });
        }
    }

    _isWaitingForInputChipGate(gate, lane, nowMs, skipActive = false) {
        if (skipActive || this.index !== 0 || !lane) return false;
        if (!gate || gate.enabled === false) return false;

        if (gate.pending) {
            return true;
        }

        const tokenIndex = Number.isFinite(lane.tokenIndex) ? Math.max(0, Math.floor(lane.tokenIndex)) : null;
        const tokenKey = tokenIndex !== null ? String(tokenIndex) : null;
        // Release only when the matching chip has actually finished entering
        // the embedding matrix.
        const insideByToken = gate.insideByToken;
        if (tokenKey !== null && insideByToken && Object.prototype.hasOwnProperty.call(insideByToken, tokenKey)) {
            if (insideByToken[tokenKey] === true) return false;
            return true;
        }
        return false;
    }

    _isWaitingForInputVocabChipGate(lane, nowMs, skipActive = false) {
        const gate = this._progressEmitter && this._progressEmitter.__inputVocabChipGate;
        return this._isWaitingForInputChipGate(gate, lane, nowMs, skipActive);
    }

    _isWaitingForInputPositionChipGate(lane, nowMs, skipActive = false) {
        const gate = this._progressEmitter && this._progressEmitter.__inputPositionChipGate;
        return this._isWaitingForInputChipGate(gate, lane, nowMs, skipActive);
    }

    _ensureLanePostAdditionVector(lane, options = {}) {
        if (!lane) return null;
        if (lane.postAdditionVec && lane.postAdditionVec.group) return lane.postAdditionVec;
        if (options.allowOriginalFallback && lane.originalVec && lane.originalVec.group) {
            lane.postAdditionVec = lane.originalVec;
            return lane.postAdditionVec;
        }
        return null;
    }

    _isMhsaOutputStageComplete() {
        const mhsa = this.mhsaAnimation;
        if (!mhsa) return true;
        if (mhsa.outputProjMatrixReturnComplete === true) return true;
        if (mhsa.outputProjMatrixAnimationPhase === 'completed') return true;

        const returnTarget = Number.isFinite(mhsa._outputProjReturnTargetCount)
            ? Math.max(0, Math.floor(mhsa._outputProjReturnTargetCount))
            : 0;
        const returnCount = Number.isFinite(mhsa._outputProjReturnCount)
            ? Math.max(0, Math.floor(mhsa._outputProjReturnCount))
            : 0;
        return returnTarget > 0 && returnCount >= returnTarget;
    }

    _isMhsaOutputStageActive() {
        const mhsa = this.mhsaAnimation;
        if (!mhsa) return false;
        if (mhsa.outputProjMatrixReturnComplete === true) return true;
        const outputPhase = mhsa.outputProjMatrixAnimationPhase || 'waiting';
        if (outputPhase !== 'waiting') return true;
        const rowMergePhase = mhsa.rowMergePhase || 'not_started';
        if (rowMergePhase !== 'not_started') return true;
        const returnTarget = Number.isFinite(mhsa._outputProjReturnTargetCount)
            ? Math.max(0, Math.floor(mhsa._outputProjReturnTargetCount))
            : 0;
        const returnCount = Number.isFinite(mhsa._outputProjReturnCount)
            ? Math.max(0, Math.floor(mhsa._outputProjReturnCount))
            : 0;
        return returnTarget > 0 || returnCount > 0;
    }

    _forceAdvanceStalledLane(lane, { ln2SyncY, allowOriginalFallback = false, postAttentionOnly = false } = {}) {
        if (!lane || lane.ln2Phase === LN2_PHASE.DONE) return false;
        const laneId = lane.laneIndex ?? '?';

        if (
            !postAttentionOnly
            && lane.horizPhase === HORIZ_PHASE.WAITING
            && !this._ln1Start
            && lane.originalVec
            && lane.originalVec.group
            && Number.isFinite(lane.branchStartY)
        ) {
            lane.originalVec.group.position.y = Math.max(lane.originalVec.group.position.y, lane.branchStartY);
            if (lane.dupVec && lane.dupVec.group) {
                copyVectorAppearance(lane.dupVec, lane.originalVec);
                lane.dupVec.group.visible = true;
                lane.dupVec.group.position.y = lane.branchStartY;
            }
            this._setLaneHorizPhase(lane, HORIZ_PHASE.RIGHT, 'stall watchdog: ln1 barrier');
            this._ln1Start = true;
            this._ln1StartBarrierArmed = false;
            this._ln1StartAtMs = NaN;
            console.warn(`Layer ${this.index}: lane ${laneId} stalled in LN1 barrier; forcing branch start.`);
            return true;
        }

        if (!postAttentionOnly && lane.horizPhase === HORIZ_PHASE.READY_MHSA && !this._mhsaStart) {
            this._setLaneHorizPhase(lane, HORIZ_PHASE.TRAVEL_MHSA, 'stall watchdog: mhsa barrier');
            lane.__mhsaTrailCornerPending = true;
            this._mhsaStart = true;
            console.warn(`Layer ${this.index}: lane ${laneId} stalled before MHSA travel; forcing barrier release.`);
            return true;
        }

        if (
            lane.horizPhase === HORIZ_PHASE.POST_MHSA_ADDITION
            || lane.horizPhase === HORIZ_PHASE.WAITING_FOR_LN2
            || (this._isMhsaOutputStageComplete() && (lane.ln2Phase === LN2_PHASE.NOT_STARTED || lane.ln2Phase === LN2_PHASE.PRE_RISE))
        ) {
            const v = this._ensureLanePostAdditionVector(lane, { allowOriginalFallback });
            if (!v || !v.group) return false;
            if (Number.isFinite(ln2SyncY)) {
                // Snap to LN2 staging height so forced recovery cannot start
                // the LN2 horizontal branch from above the ring entry.
                v.group.position.y = ln2SyncY;
            }
            this._setLaneHorizPhase(lane, HORIZ_PHASE.WAITING_FOR_LN2, 'stall watchdog: ln2 pre-rise');
            this._setLaneLn2Phase(lane, LN2_PHASE.PRE_RISE, 'stall watchdog: ln2 pre-rise');
            if (lane.stopRise) {
                delete lane.stopRise;
                delete lane.stopRiseTarget;
            }
            console.warn(`Layer ${this.index}: lane ${laneId} stalled entering LN2; forcing preRise sync.`);
            return true;
        }

        if (lane.ln2Phase === LN2_PHASE.INSIDE_LN) {
            const fallback = lane.resultVecLN2 || lane.movingVecLN2 || lane.postAdditionVec || lane.originalVec;
            if (!fallback) return false;
            lane.resultVecLN2 = lane.resultVecLN2 || fallback;
            lane.movingVecLN2 = null;
            lane.normAnimationLN2 = null;
            lane.normStartedLN2 = true;
            lane.normAppliedLN2 = true;
            lane.multDoneLN2 = true;
            lane.ln2AddStarted = true;
            lane.ln2AddComplete = true;
            this._setLaneLn2Phase(lane, LN2_PHASE.MLP_READY, 'stall watchdog: ln2 inside -> mlp ready');
            if (lane.stopRise) {
                delete lane.stopRise;
                delete lane.stopRiseTarget;
            }
            console.warn(`Layer ${this.index}: lane ${laneId} stalled inside LN2; forcing MLP readiness.`);
            return true;
        }

        if (lane.ln2Phase === LN2_PHASE.MLP_READY && !this._mlpStart) {
            this._mlpStart = true;
            console.warn(`Layer ${this.index}: lane ${laneId} stalled at MLP barrier; forcing MLP start.`);
            return true;
        }

        if (lane.ln2Phase === LN2_PHASE.MLP_READY && this._mlpStart) {
            const fallback = lane.finalVecAfterMlp || lane.resultVecLN2 || lane.postAdditionVec || lane.originalVec;
            if (!fallback) return false;

            // Preserve visual correctness under skip/skip-layer: retry the
            // MLP traversal instead of forcing an immediate LN2 completion.
            lane.resultVecLN2 = lane.resultVecLN2 || fallback;

            if (!lane.mlpUpStarted && lane.resultVecLN2 && lane.resultVecLN2.group) {
                lane.mlpUpStarted = true;
                try {
                    this._animateMlpUpProjection(lane);
                    console.warn(`Layer ${this.index}: lane ${laneId} stalled at MLP gate; watchdog re-triggered MLP traversal.`);
                    return true;
                } catch (_) {
                    lane.mlpUpStarted = false;
                }
            }

            // Last-resort fallback only when re-triggering is not possible.
            lane.finalVecAfterMlp = lane.finalVecAfterMlp || lane.resultVecLN2 || fallback;
            lane.mlpUpStarted = true;
            lane.mlpDownStarted = true;
            lane.mlpDownComplete = true;
            this._setLaneLn2Phase(lane, LN2_PHASE.DONE, 'stall watchdog: mlp forced completion');
            if (lane.stopRise) {
                delete lane.stopRise;
                delete lane.stopRiseTarget;
            }
            console.warn(`Layer ${this.index}: lane ${laneId} stalled during MLP; forcing lane completion.`);
            return true;
        }

        return false;
    }

    _resetLaneStallWatchdog(lanes) {
        if (!Array.isArray(lanes) || !lanes.length) return;
        for (let i = 0; i < lanes.length; i++) {
            const lane = lanes[i];
            if (!lane) continue;
            if (lane.__phaseWatchSignature !== undefined) delete lane.__phaseWatchSignature;
            if (lane.__phaseWatchStartMs !== undefined) delete lane.__phaseWatchStartMs;
        }
    }

    _applyLaneStallWatchdog(lanes, nowMs, context = {}) {
        if (!Array.isArray(lanes) || !lanes.length || !Number.isFinite(nowMs)) return;
        const timeoutMs = Number.isFinite(context.timeoutMs)
            ? Math.max(1000, context.timeoutMs)
            : LANE_PHASE_STALL_TIMEOUT_MS_SKIP;
        let mutated = false;
        for (let i = 0; i < lanes.length; i++) {
            const lane = lanes[i];
            if (!lane || lane.ln2Phase === LN2_PHASE.DONE) continue;
            const signature = getLaneProgressSignature(lane);
            if (lane.__phaseWatchSignature !== signature) {
                lane.__phaseWatchSignature = signature;
                lane.__phaseWatchStartMs = nowMs;
                continue;
            }
            if (!Number.isFinite(lane.__phaseWatchStartMs)) {
                lane.__phaseWatchStartMs = nowMs;
                continue;
            }
            const stalledForMs = nowMs - lane.__phaseWatchStartMs;
            if (stalledForMs < timeoutMs) continue;
            if (this._forceAdvanceStalledLane(lane, context)) {
                mutated = true;
            }
            lane.__phaseWatchStartMs = nowMs;
            lane.__phaseWatchSignature = getLaneProgressSignature(lane);
        }

        if (!mutated) return;

        const ln2SyncY = Number.isFinite(context.ln2SyncY) ? context.ln2SyncY : null;
        if (!this._ln2Ready && ln2SyncY !== null) {
            const allLn2Synced = lanes.every((lane) => (
                lane
                && lane.ln2Phase === LN2_PHASE.PRE_RISE
                && lane.postAdditionVec
                && lane.postAdditionVec.group
                && lane.postAdditionVec.group.position.y >= ln2SyncY - 0.01
            ));
            if (allLn2Synced) {
                this._ln2Ready = true;
                console.warn(`Layer ${this.index}: stall watchdog forced LN2 barrier release.`);
            }
        }

        if (!this._mlpStart) {
            const allMlpReady = lanes.every((lane) => (
                lane && (lane.ln2Phase === LN2_PHASE.MLP_READY || lane.ln2Phase === LN2_PHASE.DONE)
            ));
            if (allMlpReady) {
                this._mlpStart = true;
                console.warn(`Layer ${this.index}: stall watchdog forced MLP barrier release.`);
            }
        }

        this._emitProgress();
    }

    _getSynchronizedRiseSpeed(currentY, targetY, baseSpeed, phaseMaxRemainingDistance) {
        const speed = Number.isFinite(baseSpeed) ? Math.max(0, baseSpeed) : 0;
        if (!Number.isFinite(currentY) || !Number.isFinite(targetY) || speed <= 0) return speed;
        const remaining = Math.max(0, targetY - currentY);
        if (remaining <= 1e-5) return 0;
        const maxRemaining = Number.isFinite(phaseMaxRemainingDistance)
            ? Math.max(0, phaseMaxRemainingDistance)
            : remaining;
        if (maxRemaining <= 1e-5 || maxRemaining <= remaining + 1e-5) {
            return speed;
        }
        const etaSeconds = maxRemaining / speed;
        if (etaSeconds <= 1e-5) return speed;
        return remaining / etaSeconds;
    }

    _setVectorOpacity(vec, opacity) {
        if (!vec || !vec.mesh || !vec.mesh.material) return;
        const clampedOpacity = THREE.MathUtils.clamp(opacity, 0, 1);
        const mats = Array.isArray(vec.mesh.material) ? vec.mesh.material : [vec.mesh.material];
        mats.forEach(mat => {
            if (!mat) return;
            const shouldBeTransparent = clampedOpacity < 0.999;
            if (mat.transparent !== shouldBeTransparent) {
                mat.transparent = shouldBeTransparent;
                mat.needsUpdate = true;
            }
            if (mat.opacity !== clampedOpacity) {
                mat.opacity = clampedOpacity;
            }
            if (mat.depthWrite === shouldBeTransparent) {
                mat.depthWrite = !shouldBeTransparent;
                mat.needsUpdate = true;
            }
            if (!shouldBeTransparent && mat.depthWrite !== true) {
                mat.depthWrite = true;
                mat.needsUpdate = true;
            }
        });
    }

    _animateMultiplyTransition({ sourceVec, multResult, scaleParam = null, instant = false, onComplete = null }) {
        const finish = () => {
            if (scaleParam && scaleParam.group) {
                scaleParam.group.visible = false;
            }
            if (sourceVec && sourceVec.group) {
                sourceVec.group.visible = false;
                if (sourceVec.group.parent) {
                    sourceVec.group.parent.remove(sourceVec.group);
                }
            }
            if (multResult && multResult.group) {
                multResult.group.visible = true;
                multResult.group.scale.set(1, 1, 1);
                this._setVectorOpacity(multResult, 1);
            }
            if (typeof onComplete === 'function') onComplete();
        };

        if (instant) {
            finish();
            return;
        }

        if (!multResult || !multResult.group) {
            finish();
            return;
        }

        if (!sourceVec || !sourceVec.group) {
            finish();
            return;
        }

        if (this._skipToEndActive || typeof TWEEN === 'undefined') {
            finish();
            return;
        }

        const sourceStartScale = sourceVec.group.scale.clone();
        const sourceEndScale = sourceStartScale.clone().multiplyScalar(MULTIPLY_SOURCE_SHRINK);

        multResult.group.visible = false;
        multResult.group.scale.set(1, 1, 1);
        this._setVectorOpacity(sourceVec, 1);
        this._setVectorOpacity(multResult, 1);
        if (scaleParam && scaleParam.group) {
            // Hide parameter vector during the handoff tween to avoid blended
            // overlap colors that can read as random/glitchy.
            scaleParam.group.visible = false;
        }

        const tweenState = { t: 0 };
        new TWEEN.Tween(tweenState)
            .to({ t: 1 }, MULTIPLY_TRANSITION_DURATION_MS)
            .easing(TWEEN.Easing.Quadratic.Out)
            .onUpdate(() => {
                const t = THREE.MathUtils.clamp(tweenState.t, 0, 1);
                sourceVec.group.scale.lerpVectors(sourceStartScale, sourceEndScale, t);
                this._emitProgress();
            })
            .onComplete(finish)
            .start();
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
        return resolveBaseVectorLength(this, VECTOR_LENGTH_PRISM);
    }

    _getLaneLayoutCount() {
        return resolveLaneLayoutCount(this);
    }

    _getActiveLaneLayoutIndices() {
        return resolveActiveLaneLayoutIndices(this);
    }

    _getInstanceCountFromData(values, fallback = null) {
        return resolveInstanceCountFromData(this, values, fallback, this._getBaseVectorLength());
    }

    _createPrismVector(values, position, numSubsections = 30, instanceCount = null) {
        return createPrismVectorForLayer(
            this,
            values,
            position,
            numSubsections,
            instanceCount,
            this._getBaseVectorLength()
        );
    }

    _getLayerNormParamData(kind, param) {
        return resolveLayerNormParamDataForLayer(this, kind, param, this._getBaseVectorLength());
    }

    _applyLayerNormParamVector(targetVec, kind, param, colorOptions = null) {
        return applyLayerNormParamVectorForLayer(
            this,
            targetVec,
            kind,
            param,
            colorOptions,
            this._getBaseVectorLength()
        );
    }

    _getTokenIndexForLane(laneIdx, laneLayoutIdx = null) {
        return resolveTokenIndexForLane(this, laneIdx, laneLayoutIdx);
    }

    _getTokenLabel(tokenIndex) {
        return resolveTokenLabelForLayer(this, tokenIndex);
    }

    _getEmbeddingData(lane, kind) {
        return getEmbeddingDataForLane(this, lane, kind, this._getBaseVectorLength());
    }

    _getLayerIncomingData(lane) {
        return getLayerIncomingDataForLane(this, lane, this._getBaseVectorLength());
    }

    _getLn1Data(lane, stage) {
        return getLn1DataForLane(this, lane, stage, this._getBaseVectorLength());
    }

    _getLn2Data(lane, stage) {
        return getLn2DataForLane(this, lane, stage, this._getBaseVectorLength());
    }

    _getAttentionOutputProjectionData(lane) {
        return getAttentionOutputProjectionDataForLane(this, lane, this._getBaseVectorLength());
    }

    _getPostAttentionResidualData(lane) {
        return getPostAttentionResidualDataForLane(this, lane, this._getBaseVectorLength());
    }

    _getMlpUpData(lane) {
        return getMlpUpDataForLane(this, lane, this._getBaseVectorLength());
    }

    _getMlpActivationData(lane) {
        return getMlpActivationDataForLane(this, lane, this._getBaseVectorLength());
    }

    _getMlpDownData(lane) {
        return getMlpDownDataForLane(this, lane, this._getBaseVectorLength());
    }

    _getPostMlpResidualData(lane) {
        return getPostMlpResidualDataForLane(this, lane, this._getBaseVectorLength());
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

    _buildSingleLane(oldLane, offsetX, ln1CenterY, ln2CenterY, startY_override, meetY, laneLayoutIdx, slitSpacing, laneLocalIdx = 0) {
        buildSingleLane(
            this,
            oldLane,
            offsetX,
            ln1CenterY,
            ln2CenterY,
            startY_override,
            meetY,
            laneLayoutIdx,
            slitSpacing,
            laneLocalIdx
        );
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
                    const lineWidth = (trail && typeof trail._lineWidth === 'number') ? trail._lineWidth : null;
                    const opacity = (trail && typeof trail._opacity === 'number') ? trail._opacity : null;
                    const colorKey = color
                        ? `${color.r.toFixed(4)},${color.g.toFixed(4)},${color.b.toFixed(4)}`
                        : 'default';
                    const lineWidthKey = lineWidth != null ? lineWidth.toFixed(4) : 'default';
                    const opacityKey = opacity != null ? opacity.toFixed(4) : 'default';
                    return { color, lineWidth, opacity, key: `${colorKey}|${lineWidthKey}|${opacityKey}` };
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
                                group = {
                                    segments: [],
                                    color: style.color,
                                    lineWidth: style.lineWidth,
                                    opacity: style.opacity
                                };
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
                        group.lineWidth != null ? group.lineWidth : undefined,
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
                            group = {
                                trails: [],
                                color: style.color,
                                lineWidth: style.lineWidth,
                                opacity: style.opacity
                            };
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
                            group.lineWidth != null ? group.lineWidth : undefined,
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
