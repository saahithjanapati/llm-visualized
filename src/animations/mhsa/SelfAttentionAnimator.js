import * as THREE from 'three';
import { VECTOR_LENGTH_PRISM, SA_RED_EXTRA_RISE, SA_V_RISE_DURATION_MS, SA_K_ALIGN_DURATION_MS, SA_BLUE_HORIZ_DURATION_MS, SA_BLUE_VERT_DURATION_MS, SA_BLUE_PAUSE_MS, SA_BLUE_QUEUE_SHIFT_DURATION_MS, SA_DUPLICATE_POP_IN_MS, SA_DUPLICATE_TRAVEL_MERGE_MS, SA_DUPLICATE_POP_OUT_MS, GLOBAL_ANIM_SPEED_MULT, SELF_ATTENTION_TIME_MULT } from '../../utils/constants.js';
import { VectorVisualizationInstancedPrism } from '../../components/VectorVisualizationInstancedPrism.js';
import { mapValueToColor, mapValueToGrayscale, buildMonochromeOptions, mapValueToMonochrome } from '../../utils/colors.js';
import { buildActivationData, applyActivationDataToObject } from '../../utils/activationMetadata.js';
import { MHA_VALUE_SPECTRUM_COLOR } from '../LayerAnimationConstants.js';

// Shared lightweight geometry for self-attention highlight spheres
const SHARED_SPHERE_GEOMETRY = new THREE.SphereGeometry(10, 12, 12);

/**
 * Self-attention specific, above-matrix animations.
 *
 * Responsibilities:
 * 1. Handle the legacy placeholder flow (for backwards compatibility)
 * 2. Extra rise for Value (red) vectors + horizontal alignment of
 *    corresponding Key (green) vectors underneath them.
 * 3. NEW – "conveyor-belt" style animation for Query (blue) vectors:
 *    • After the V/K alignment for a given head finishes we begin processing
 *      that head's queue of blue vectors.
 *    • The blue vectors are processed one-by-one in their ORIGINAL order
 *      (top → bottom along the slit index / lane order).
 *    • For the i-th blue vector we:
 *        a) Slide horizontally so it sits over the green matrix (x = K)
 *        b) Travel i positions along the green vectors (positive z direction
 *           – lanes are sorted by z). We pause briefly at each K vector.
 *        c) Slide horizontally so it sits over the red matrix (x = V)
 *        d) Travel i positions visiting the red vectors, pausing at each.
 *        e) Fade the vector out (dispose + remove from scene).
 *    • When a blue vector leaves the queue we shift the remaining blue vectors
 *      up by one position so they visually fill the gap (like a conveyor belt).
 *    • The same logic runs in parallel for every attention head.
 */
export class SelfAttentionAnimator {
    /**
     * @param {MHSAAnimation} ctx – parent MHSAAnimation instance.
     */
    constructor(ctx) {
        this.ctx = ctx;
        this.phase = 'waiting'; // 'waiting' | 'running' | 'complete'
        this._callbacks = [];

        // ------------------------------------------------------------------
        // V/K constants (distances remain static; durations scale via getters)
        // ------------------------------------------------------------------
        this.RED_EXTRA_RISE   = SA_RED_EXTRA_RISE;   // additional rise for red (V) vectors

        // Internal bookkeeping (per-head)
        this.blueQueues        = {};  // { headIdx: [vec, ...] }
        this.blueProcessing    = {};  // { headIdx: boolean }
        this.blueProcessedCount = {}; // { headIdx: number }
        this.greensAligned     = {};  // { headIdx: boolean } – flagged once K vectors are in place
        this.skipRequested     = false;
        this._activeBlueVectors = {}; // { headIdx: Vector }
        this._pendingTimeouts   = new Set();
        this._spawnedSpheres    = new Set();
        this._spawnedTempVectors = new Set();
        this._pendingDockCount  = 0;
        this.attentionProgress = {};
        this.attentionCompletedRows = {};
        this.attentionPostCompletedRows = {};
    }

    // Durations decoupled from GLOBAL_ANIM_SPEED_MULT to make presets clearly visible
    get V_RISE_DURATION() { return SA_V_RISE_DURATION_MS * SELF_ATTENTION_TIME_MULT; }
    get K_ALIGN_DURATION() { return SA_K_ALIGN_DURATION_MS * SELF_ATTENTION_TIME_MULT; }
    get BLUE_HORIZ_DURATION() { return SA_BLUE_HORIZ_DURATION_MS * SELF_ATTENTION_TIME_MULT; }
    get BLUE_VERT_DURATION() { return SA_BLUE_VERT_DURATION_MS * SELF_ATTENTION_TIME_MULT; }
    get BLUE_PAUSE_MS() { return SA_BLUE_PAUSE_MS * SELF_ATTENTION_TIME_MULT; }
    get BLUE_QUEUE_SHIFT_DURATION() { return SA_BLUE_QUEUE_SHIFT_DURATION_MS * SELF_ATTENTION_TIME_MULT; }
    get DUPLICATE_POP_IN_MS() { return SA_DUPLICATE_POP_IN_MS * SELF_ATTENTION_TIME_MULT; }
    get DUPLICATE_TRAVEL_MERGE_MS() { return SA_DUPLICATE_TRAVEL_MERGE_MS * SELF_ATTENTION_TIME_MULT; }
    get DUPLICATE_POP_OUT_MS() { return SA_DUPLICATE_POP_OUT_MS * SELF_ATTENTION_TIME_MULT; }

    // ------------------------------------------------------------------
    // Skip / cleanup helpers
    // ------------------------------------------------------------------
    _registerTimeout(id) {
        if (!id) return;
        this._pendingTimeouts.add(id);
    }

    _clearPendingTimeouts() {
        this._pendingTimeouts.forEach((id) => clearTimeout(id));
        this._pendingTimeouts.clear();
    }

    _cleanupAttentionScoreMeshes() {
        const root = this.ctx && this.ctx.parentGroup;
        if (!root || typeof root.traverse !== 'function') return;
        const toRemove = [];
        root.traverse((obj) => {
            if (!obj || !obj.isMesh) return;
            const ud = obj.userData || {};
            const label = typeof ud.label === 'string' ? ud.label : '';
            const stage = ud.activationData && ud.activationData.stage;
            if (label.includes('Attention Score') || stage === 'attention.pre' || stage === 'attention.post') {
                toRemove.push(obj);
            }
        });
        toRemove.forEach((obj) => {
            try { if (obj.parent) obj.parent.remove(obj); } catch (_) { /* optional cleanup */ }
            try { if (obj.material && typeof obj.material.dispose === 'function') obj.material.dispose(); } catch (_) { /* optional cleanup */ }
            try {
                if (obj.geometry && obj.geometry !== SHARED_SPHERE_GEOMETRY && typeof obj.geometry.dispose === 'function') {
                    obj.geometry.dispose();
                }
            } catch (_) { /* optional cleanup */ }
        });
    }

    _isHeadActive(headIdx) {
        if (!Number.isFinite(headIdx)) return false;
        const queued = this.blueQueues && this.blueQueues[headIdx];
        return !!(this.blueProcessing && this.blueProcessing[headIdx])
            || !!(this._activeBlueVectors && this._activeBlueVectors[headIdx])
            || (Array.isArray(queued) && queued.length > 0);
    }

    _setAttentionProgress(headIdx, rowIndex, colIndex) {
        if (!Number.isFinite(headIdx) || !Number.isFinite(rowIndex)) return;
        const prev = this.attentionProgress[headIdx] || {};
        const next = {
            activeRow: rowIndex,
            activeCol: Number.isFinite(colIndex) ? colIndex : prev.activeCol
        };
        this.attentionProgress[headIdx] = next;
    }

    _markAttentionRowComplete(headIdx, rowIndex) {
        if (!Number.isFinite(headIdx) || !Number.isFinite(rowIndex)) return;
        const completed = this.attentionCompletedRows[headIdx] || 0;
        this.attentionCompletedRows[headIdx] = Math.max(completed, rowIndex + 1);
        const progress = this.attentionProgress[headIdx];
        if (progress && progress.activeRow === rowIndex) {
            progress.activeRow = null;
            progress.activeCol = null;
        }
    }

    _markAttentionPostRowComplete(headIdx, rowIndex) {
        if (!Number.isFinite(headIdx) || !Number.isFinite(rowIndex)) return;
        const completed = this.attentionPostCompletedRows[headIdx] || 0;
        this.attentionPostCompletedRows[headIdx] = Math.max(completed, rowIndex + 1);
    }

    _markAttentionHeadComplete(headIdx) {
        if (!Number.isFinite(headIdx)) return;
        const laneCount = Array.isArray(this.ctx?.currentLanes) ? this.ctx.currentLanes.length : 0;
        if (laneCount > 0) {
            const completed = this.attentionCompletedRows[headIdx] || 0;
            this.attentionCompletedRows[headIdx] = Math.max(completed, laneCount);
            const postCompleted = this.attentionPostCompletedRows[headIdx] || 0;
            this.attentionPostCompletedRows[headIdx] = Math.max(postCompleted, laneCount);
        }
        const progress = this.attentionProgress[headIdx];
        if (progress) {
            progress.activeRow = null;
            progress.activeCol = null;
        }
    }

    getAttentionProgress(headIdx) {
        if (!this._isHeadActive(headIdx)) return null;
        const completedRows = this.attentionCompletedRows[headIdx] || 0;
        const postCompletedRows = this.attentionPostCompletedRows[headIdx] || 0;
        const progress = this.attentionProgress[headIdx] || {};
        const activeRow = Number.isFinite(progress.activeRow) ? progress.activeRow : null;
        const activeCol = Number.isFinite(progress.activeCol) ? progress.activeCol : null;
        return { completedRows, postCompletedRows, activeRow, activeCol };
    }

    _retireVector(vec, { preserveTrail = false } = {}) {
        if (!vec) return;
        try {
            if (!preserveTrail && vec.userData && vec.userData.trail && typeof vec.userData.trail.dispose === 'function') {
                vec.userData.trail.dispose();
                delete vec.userData.trail;
            }
        } catch (_) { /* optional cleanup */ }
        try {
            if (vec.group) {
                if (vec.group.parent) vec.group.parent.remove(vec.group);
                vec.group.visible = false;
                vec.group.scale.set(0.001, 0.001, 0.001);
            }
        } catch (_) { /* optional cleanup */ }
        try {
            if (typeof vec.dispose === 'function') vec.dispose();
        } catch (_) { /* optional cleanup */ }
        if (this._spawnedTempVectors) this._spawnedTempVectors.delete(vec);
    }

    _finishBlueImmediately(vector, headIdx, doneCb, options = null) {
        this._retireVector(vector, options || undefined);
        if (typeof headIdx === 'number' && this._activeBlueVectors) {
            delete this._activeBlueVectors[headIdx];
        }
        if (typeof doneCb === 'function') doneCb();
    }

    /**
     * Immediately complete the conveyor belt, clear queues, and fire callbacks.
     */
    forceComplete(options = {}) {
        if (this.phase === 'complete') {
            this._flushCallbacks();
            return;
        }
        const preserveTrails = !!options.preserveTrails;
        const createWeightedSums = !!options.createWeightedSums;
        const replaceWeightedSums = options.replaceWeightedSums !== false;
        this.skipRequested = true;
        this._clearPendingTimeouts();
        if (createWeightedSums) {
            try { this._createWeightedSumsImmediate({ replaceExisting: replaceWeightedSums }); } catch (_) {}
        }
        // Retire any active vectors and queued blues
        Object.values(this._activeBlueVectors || {}).forEach((vec) => this._retireVector(vec, { preserveTrail: preserveTrails }));
        Object.values(this.blueQueues || {}).forEach((queue) => {
            if (Array.isArray(queue)) queue.forEach((vec) => this._retireVector(vec, { preserveTrail: preserveTrails }));
        });
        this.blueQueues = {};
        this.blueProcessing = {};
        this.blueProcessedCount = {};
        this._pendingDockCount = 0;
        this.phase = 'complete';
        // Clean up any lingering highlight spheres
        this._spawnedSpheres.forEach((sp) => {
            try {
                if (sp.parent) sp.parent.remove(sp);
                if (sp.material && typeof sp.material.dispose === 'function') sp.material.dispose();
            } catch (_) { /* optional cleanup */ }
        });
        this._spawnedSpheres.clear();
        this._cleanupAttentionScoreMeshes();
        this.attentionProgress = {};
        this.attentionCompletedRows = {};
        this.attentionPostCompletedRows = {};
        // Dispose any transient attention vectors (duplicates / travellers)
        this._spawnedTempVectors.forEach((vec) => this._retireVector(vec, { preserveTrail: preserveTrails }));
        this._spawnedTempVectors.clear();
        // Dispose K/V visuals so they disappear alongside the skipped conveyor
        try { this.ctx && this.ctx._disposeMergedKVGroups && this.ctx._disposeMergedKVGroups(); } catch (_) {}
        try { this.ctx && this.ctx._disposeAllIndividualKandVVectorsImmediately && this.ctx._disposeAllIndividualKandVVectorsImmediately(); } catch (_) {}
        this._flushCallbacks();
    }

    /**
     * Public helper so UI can decide whether a skip button should be visible.
     */
    isConveyorActive() {
        return !this._isConveyorComplete();
    }

    /**
     * Public entry – dual signature for backwards compatibility.
     * 1. start(onDone)                              – legacy placeholder
     * 2. start(vector, vectorCategory, onDone)      – per-vector handling
     */
    start(vectorOrCallback = null, vectorCategory = null, onDone = null) {
        // Global completion callback registration (legacy signature)
        if (typeof vectorOrCallback === 'function') {
            this._callbacks.push(vectorOrCallback);
            // If the conveyor is already finished, flush immediately.
            if (this._isConveyorComplete()) {
                this._flushCallbacks();
                this.phase = 'complete';
            }
            return;
        }

        // Vector-aware mode
        const vector = vectorOrCallback;
        this._handleAboveMatrixAnimations(vector, vectorCategory, onDone);
    }

    // ------------------------------------------------------------------
    // Placeholder mode (kept for compatibility with older tests)
    // ------------------------------------------------------------------
    _runPlaceholderMode() {
        if (this.phase === 'complete') {
            this._flushCallbacks();
            return;
        }
        if (this.phase === 'running') {
            return;
        }

        this.phase = 'running';
        console.log('SelfAttentionAnimator: placeholder phase started');

        setTimeout(() => {
            this.phase = 'complete';
            console.log('SelfAttentionAnimator: placeholder phase complete');
            this._flushCallbacks();
        }, 3000);
    }

    // ------------------------------------------------------------------
    // Per-vector dispatch
    // ------------------------------------------------------------------
    _handleAboveMatrixAnimations(vector, vectorCategory, onDone) {
        if (this.skipRequested) {
            onDone && onDone();
            return;
        }
        if (this.phase === 'waiting') this.phase = 'running';
        if (typeof TWEEN === 'undefined') {
            console.error('Global TWEEN object not loaded for SelfAttentionAnimator!');
            onDone && onDone();
            return;
        }

        if (vectorCategory === 'V') {
            // Value (red) vector: extra rise then trigger K alignment.
            this._animateVVectorRise(vector, onDone);
            return;
        }

        if (vectorCategory === 'Q') {
            // Query (blue) vector: enqueue for conveyor-belt processing.
            this._enqueueBlueVector(vector);
            onDone && onDone(); // Immediately release the caller – we manage blue vectors asynchronously.
            return;
        }

        // Other vector categories – nothing special for now.
        onDone && onDone();
    }

    // ------------------------------------------------------------------
    // V / K alignment helpers
    // ------------------------------------------------------------------
    /**
     * Fixed Value (red) vectors rise in TWO separate stages:
     * 1. VectorMatrixPassThrough.js – while passing vertically through the V-matrix
     *    each red copy is tweened from its parking height (`headStopY`) up to
     *    `ctx.mhaPassThroughTargetY + ctx.mhaResultRiseOffsetY - 30`.
     *    This is the same base rise distance Q and K experience.
     * 2. SelfAttentionAnimator.js – this method then adds an additional
     *    `RED_EXTRA_RISE` (75 world-units) so the fixed V copies end up sitting
     *    above the green K vectors.  At the end of this tween each fixed V copy
     *    is located at the canonical "raised-V" height:
     *        ctx.mhaPassThroughTargetY + ctx.mhaResultRiseOffsetY - 30 + RED_EXTRA_RISE
     *    All subsequent duplicates and travelling red vectors snap to / match
     *    this same Y so they remain perfectly level.
     */
    _animateVVectorRise(vector, onDone) {
        if (this.skipRequested) {
            this._retireVector(vector);
            onDone && onDone();
            return;
        }
        new TWEEN.Tween({ y: vector.group.position.y })
            .to({ y: vector.group.position.y + this.RED_EXTRA_RISE }, this.V_RISE_DURATION)
            .easing(TWEEN.Easing.Quadratic.Out)
            .onUpdate(obj => { vector.group.position.y = obj.y; })
            .onComplete(() => {
                this._alignKVectorsUnderV(vector, onDone);
            })
            .start();
    }

    _alignKVectorsUnderV(redVector, onDone) {
        if (this.skipRequested) {
            onDone && onDone();
            return;
        }
        const redX = redVector.group.position.x;
        const redZ = redVector.group.position.z;
        const headIdx = (redVector.userData && typeof redVector.userData.headIndex === 'number')
            ? redVector.userData.headIndex : null;

        if (headIdx === null || !this.ctx.currentLanes) {
            onDone && onDone();
            return;
        }

        let alignmentsInProgress = 0;
        let alignmentsCompleted  = 0;

        this.ctx.currentLanes.forEach(lane => {
            if (Math.abs(lane.zPos - redZ) < 0.1 && lane.upwardCopies && lane.upwardCopies[headIdx]) {
                const green = lane.upwardCopies[headIdx];
                alignmentsInProgress++;

                new TWEEN.Tween(green.group.position)
                    .to({ x: redX }, this.K_ALIGN_DURATION)
                    .easing(TWEEN.Easing.Quadratic.Out)
                    .onComplete(() => {
                        alignmentsCompleted++;
                        if (alignmentsCompleted >= alignmentsInProgress) {
                            // Mark this head as ready for blue-vector conveyor processing.
                            this.greensAligned[headIdx] = true;
                            // Merge fixed K/V vectors for this head to reduce draw calls
                            try { this.ctx && this.ctx._mergeFixedVectorsForHead && this.ctx._mergeFixedVectorsForHead(headIdx); } catch (_) { /* optional */ }
                            this._kickoffBlueConveyor(headIdx);
                            onDone && onDone();
                        }
                    })
                    .start();
            }
        });

        // No greens to align – still flag readiness so blue vectors can run.
        if (alignmentsInProgress === 0) {
            this.greensAligned[headIdx] = true;
                // Merge fixed K/V vectors for this head immediately if nothing to align
                try { this.ctx && this.ctx._mergeFixedVectorsForHead && this.ctx._mergeFixedVectorsForHead(headIdx); } catch (_) { /* optional */ }
            this._kickoffBlueConveyor(headIdx);
            onDone && onDone();
        }
    }

    // ------------------------------------------------------------------
    // Blue (Query) conveyor-belt logic
    // ------------------------------------------------------------------
    _enqueueBlueVector(vector) {
        if (this.skipRequested) return;
        if (this.phase === 'waiting') this.phase = 'running';
        const headIdx = (vector.userData && typeof vector.userData.headIndex === 'number')
            ? vector.userData.headIndex : null;
        if (headIdx === null) return;

        if (!this.blueQueues[headIdx]) this.blueQueues[headIdx] = [];
        const queue = this.blueQueues[headIdx];

        queue.push(vector);
        // Keep queue ordered by lane z (ascending) so index == lane order (top → bottom).
        // Using the parent lane's zPos avoids jitter while vectors are mid-shift.
        queue.sort((a, b) => {
            const az = (a.userData && a.userData.parentLane && typeof a.userData.parentLane.zPos === 'number')
                ? a.userData.parentLane.zPos
                : a.group.position.z;
            const bz = (b.userData && b.userData.parentLane && typeof b.userData.parentLane.zPos === 'number')
                ? b.userData.parentLane.zPos
                : b.group.position.z;
            return az - bz;
        });

        // If the conveyor belt is already running for this head, newly
        // enqueued vectors may arrive slightly later than the initial batch.
        // Without an immediate shift they would remain at their original
        // positions until the next dequeued vector triggers a queue update,
        // causing a visible "lagging" query at the tail.  To keep all blue
        // vectors synchronized from the very start, realign the queue right
        // away whenever processing is active.
        if (this.blueProcessing[headIdx]) {
            this._shiftRemainingBlueVectors(queue, headIdx);
        }

        // If greens are already in position we can start processing immediately.
        if (this.greensAligned[headIdx]) {
            this._kickoffBlueConveyor(headIdx);
        }
    }

    _kickoffBlueConveyor(headIdx) {
        if (this.skipRequested) return;
        if (this.blueProcessing[headIdx]) return; // already running
        if (!this.blueQueues[headIdx] || this.blueQueues[headIdx].length === 0) {
            // Nothing queued – might be done already
            this._checkGlobalCompletion();
            return;
        }

        this.blueProcessing[headIdx]     = true;
        if (typeof this.blueProcessedCount[headIdx] !== 'number') this.blueProcessedCount[headIdx] = 0;

        this._processNextBlueVector(headIdx);
    }

    _processNextBlueVector(headIdx) {
        const queue = this.blueQueues[headIdx];
        if (this.skipRequested) {
            if (Array.isArray(queue)) queue.length = 0;
            this.blueProcessing[headIdx] = false;
            this._markAttentionHeadComplete(headIdx);
            this._checkGlobalCompletion();
            return;
        }
        if (!queue || queue.length === 0) {
            // Finished all blues for this head
            this.blueProcessing[headIdx] = false;
            this._markAttentionHeadComplete(headIdx);
            this._checkGlobalCompletion();
            return;
        }

        // Pop the first (top-most) blue vector
        const vector = queue.shift();

        // Shift remaining blues up to fill the gap (simple visual feedback)
        this._shiftRemainingBlueVectors(queue, headIdx);

        // i == 1-based index of this vector in processing order (1 for first vector, 2 for second, ...)
        const i = this.blueProcessedCount[headIdx] + 1; // shift to 1-based
        this.blueProcessedCount[headIdx] += 1;
        const rowIndex = i - 1;
        if (vector) {
            vector.userData = vector.userData || {};
            vector.userData.attnRowIndex = rowIndex;
        }
        this._setAttentionProgress(headIdx, rowIndex, -1);

        // Pre-compute sorted lane z positions (top → bottom)
        const laneZs = (this.ctx.currentLanes || []).map(l => l.zPos).sort((a, b) => a - b);

        this._activeBlueVectors[headIdx] = vector;
        this._animateBlueVector(vector, headIdx, i, laneZs, () => {
            delete this._activeBlueVectors[headIdx];
            // Recursive continuation
            this._processNextBlueVector(headIdx);
        });
    }

    _shiftRemainingBlueVectors(queue, headIdx) {
        if (this.skipRequested) return;
        // Existing logic (unchanged)

        const laneZs = (this.ctx.currentLanes || []).map(l => l.zPos).sort((a, b) => a - b);
        if (!laneZs.length) return;
        queue.forEach((vec, idx) => {
            const targetZ = laneZs[idx < laneZs.length ? idx : laneZs.length - 1];
            new TWEEN.Tween(vec.group.position)
                .to({ z: targetZ }, this.BLUE_QUEUE_SHIFT_DURATION)
                .easing(TWEEN.Easing.Quadratic.InOut)
                .start();
        });
    }

    _riseSpheres(spheresArr, rowIndex = null, headIdx = null) {

        if (this.skipRequested) return;
        if (!Array.isArray(spheresArr) || spheresArr.length === 0) return;
        const resolvedHeadIdx = Number.isFinite(headIdx)
            ? headIdx
            : (() => {
                const sp = spheresArr[0];
                const activationData = sp && sp.userData ? sp.userData.activationData : null;
                return activationData && Number.isFinite(activationData.headIndex) ? activationData.headIndex : null;
            })();
        if (Number.isFinite(resolvedHeadIdx) && Number.isFinite(rowIndex)) {
            this._markAttentionPostRowComplete(resolvedHeadIdx, rowIndex);
        }
        spheresArr.forEach((sp) => {
            // Position rise
            new TWEEN.Tween(sp.position)
                .to({ y: sp.position.y + this.RED_EXTRA_RISE }, this.V_RISE_DURATION)
                .easing(TWEEN.Easing.Quadratic.Out)
                .start();

            // Colour desaturation → bright mono (white-ish)
            if (sp.material) {
                const c = sp.material.color.clone();
                const hasEmissive = ('emissive' in sp.material) && ('emissiveIntensity' in sp.material);
                const state = { r: c.r, g: c.g, b: c.b, ei: hasEmissive ? sp.material.emissiveIntensity : 1.0 };
                const activationData = sp.userData ? sp.userData.activationData : null;
                const layerIndex = activationData && Number.isFinite(activationData.layerIndex) ? activationData.layerIndex : null;
                const headIdx = activationData && Number.isFinite(activationData.headIndex) ? activationData.headIndex : null;
                const queryTokenIndex = activationData && Number.isFinite(activationData.tokenIndex) ? activationData.tokenIndex : null;
                const keyTokenIndex = activationData && Number.isFinite(activationData.keyTokenIndex) ? activationData.keyTokenIndex : null;
                const postScore = (this.ctx && this.ctx.activationSource
                    && Number.isFinite(layerIndex) && Number.isFinite(headIdx)
                    && Number.isFinite(queryTokenIndex) && Number.isFinite(keyTokenIndex))
                    ? this.ctx.activationSource.getAttentionScore(layerIndex, 'post', headIdx, queryTokenIndex, keyTokenIndex)
                    : null;
                const targetColor = Number.isFinite(postScore)
                    ? mapValueToGrayscale(postScore)
                    : new THREE.Color().setHSL(0, 0, THREE.MathUtils.lerp(0.2, 0.9, Math.random()));
                const targetEI = Number.isFinite(postScore) ? 0.6 : 0.8;
                new TWEEN.Tween(state)
                    .to({ r: targetColor.r, g: targetColor.g, b: targetColor.b, ei: targetEI }, this.V_RISE_DURATION)
                    .easing(TWEEN.Easing.Quadratic.Out)
                    .onStart(() => {
                        if (activationData) {
                            activationData.stage = 'attention.post';
                            if (Number.isFinite(postScore)) activationData.postScore = postScore;
                            applyActivationDataToObject(sp, activationData, 'Post-Softmax Attention Score');
                        } else {
                            applyActivationDataToObject(sp, null, 'Post-Softmax Attention Score');
                        }
                    })
                    .onUpdate(() => {
                        sp.material.color.setRGB(state.r, state.g, state.b);
                        if (hasEmissive) {
                            sp.material.emissive.setRGB(state.r, state.g, state.b);
                            sp.material.emissiveIntensity = state.ei;
                        }
                    })
                    .start();
            }
        });
    }

    /**
     * Helper to uniformly recolor a vector (all prisms) to the given THREE.Color.
     */
    _setVectorColor(vector, color) {
        if (!vector || !vector.mesh) return;
        const col = color instanceof THREE.Color ? color : new THREE.Color(color);
        const count = vector.instanceCount || VECTOR_LENGTH_PRISM;
        for (let idx = 0; idx < count; idx++) {
            if (vector.mesh.setColorAt) {
                vector.mesh.setColorAt(idx, col);
            }
        }
        if (vector.mesh.instanceColor) {
            vector.mesh.instanceColor.needsUpdate = true;
        }
        const csAttr = vector.mesh.geometry.getAttribute('colorStart');
        const ceAttr = vector.mesh.geometry.getAttribute('colorEnd');
        if (csAttr && ceAttr) {
            for (let idx = 0; idx < count; idx++) {
                csAttr.setXYZ(idx, col.r, col.g, col.b);
                ceAttr.setXYZ(idx, col.r, col.g, col.b);
            }
            csAttr.needsUpdate = true;
            ceAttr.needsUpdate = true;
        }
    }

    _applyValueVectorScheme(vector, sourceData = null, options = {}) {
        if (!vector) return;
        const outputLength = (this.ctx && this.ctx.outputVectorLength) ? this.ctx.outputVectorLength : 64;
        const setHiddenToBlack = options && typeof options.setHiddenToBlack === 'boolean'
            ? options.setHiddenToBlack
            : true;
        const raw = Array.isArray(sourceData)
            ? sourceData.slice(0, outputLength)
            : (vector.rawData ? vector.rawData.slice(0, outputLength) : []);
        const monoOptions = buildMonochromeOptions(MHA_VALUE_SPECTRUM_COLOR);
        const numKeyColors = raw.length <= 1 ? 1 : 3;
        vector.applyProcessedVisuals(
            raw,
            outputLength,
            { numKeyColors, generationOptions: monoOptions },
            { setHiddenToBlack },
            raw
        );
        if (raw.length === 1 && typeof vector.setUniformColor === 'function') {
            const monoColor = mapValueToMonochrome(raw[0], monoOptions);
            vector.setUniformColor(monoColor);
        }
        vector.userData = vector.userData || {};
        vector.userData.weightedSumVisuals = {
            numKeyColors,
            generationOptions: monoOptions,
            outputLength
        };
    }

    _applyWeightedSumScheme(vector, sourceData = null, options = {}) {
        if (!vector) return;
        const outputLength = (this.ctx && this.ctx.outputVectorLength) ? this.ctx.outputVectorLength : 64;
        const setHiddenToBlack = options && typeof options.setHiddenToBlack === 'boolean'
            ? options.setHiddenToBlack
            : true;
        const raw = Array.isArray(sourceData)
            ? sourceData.slice(0, outputLength)
            : (vector.rawData ? vector.rawData.slice(0, outputLength) : []);
        const numKeyColors = raw.length <= 1 ? 1 : Math.min(30, raw.length);
        vector.applyProcessedVisuals(
            raw,
            outputLength,
            { numKeyColors, generationOptions: null },
            { setHiddenToBlack },
            raw
        );
        if (raw.length === 1 && typeof vector.setUniformColor === 'function') {
            vector.setUniformColor(mapValueToColor(raw[0]));
        }
        vector.userData = vector.userData || {};
        vector.userData.weightedSumVisuals = {
            numKeyColors,
            generationOptions: null,
            outputLength
        };
    }

    _buildWeightedSumData(headIdx, queryLane, lanes, outputLength) {
        const activationSource = this.ctx && this.ctx.activationSource ? this.ctx.activationSource : null;
        const layerIndex = Number.isFinite(this.ctx?.layerIndex) ? this.ctx.layerIndex : null;
        const queryTokenIndex = queryLane && Number.isFinite(queryLane.tokenIndex) ? queryLane.tokenIndex : null;
        const data = new Array(outputLength).fill(0);
        let usedWeight = false;

        if (activationSource && Number.isFinite(layerIndex) && Number.isFinite(queryTokenIndex)) {
            lanes.forEach((keyLane) => {
                const keyTokenIndex = keyLane && Number.isFinite(keyLane.tokenIndex) ? keyLane.tokenIndex : null;
                if (!Number.isFinite(keyTokenIndex)) return;
                const weight = activationSource.getAttentionScore
                    ? activationSource.getAttentionScore(layerIndex, 'post', headIdx, queryTokenIndex, keyTokenIndex)
                    : null;
                if (!Number.isFinite(weight)) return;
                const vObj = keyLane && Array.isArray(keyLane.sideCopies)
                    ? keyLane.sideCopies.find(sc => sc && sc.type === 'V' && sc.headIndex === headIdx)
                    : null;
                const vVec = vObj && vObj.vec ? vObj.vec : null;
                const vData = vVec && Array.isArray(vVec.rawData) ? vVec.rawData : null;
                if (!vData || !vData.length) return;
                usedWeight = true;
                for (let i = 0; i < outputLength; i++) {
                    data[i] += weight * (vData[i] ?? 0);
                }
            });
        }

        if (!usedWeight) {
            const fallbackObj = queryLane && Array.isArray(queryLane.sideCopies)
                ? queryLane.sideCopies.find(sc => sc && sc.type === 'V' && sc.headIndex === headIdx)
                : null;
            const fallbackVec = fallbackObj && fallbackObj.vec ? fallbackObj.vec : null;
            const fallbackData = fallbackVec && Array.isArray(fallbackVec.rawData) ? fallbackVec.rawData : null;
            if (fallbackData && fallbackData.length) {
                return fallbackData.slice(0, outputLength);
            }
            return data;
        }

        return data;
    }

    _createWeightedSumsImmediate(options = {}) {
        const ctx = this.ctx;
        const lanes = Array.isArray(ctx?.currentLanes) ? ctx.currentLanes : [];
        if (!lanes.length) return 0;
        const headCount = Array.isArray(ctx?.headCoords) ? ctx.headCoords.length : 0;
        if (!headCount) return 0;
        const outputLength = Number.isFinite(ctx?.outputVectorLength) ? ctx.outputVectorLength : 64;
        const replaceExisting = options && options.replaceExisting !== false;

        if (replaceExisting) {
            try { ctx && ctx._clearTempDecoratives && ctx._clearTempDecoratives(); } catch (_) {}
            try { ctx && ctx._clearWeightedSumVectors && ctx._clearWeightedSumVectors(); } catch (_) {}
        }

        let created = 0;
        const dockOffset = Number.isFinite(ctx?.weightedSumDockOffset) ? ctx.weightedSumDockOffset : 30;
        const fallbackBaseY = Number.isFinite(ctx?.mhaPassThroughTargetY) && Number.isFinite(ctx?.mhaResultRiseOffsetY)
            ? ctx.mhaPassThroughTargetY + ctx.mhaResultRiseOffsetY - 30 + this.RED_EXTRA_RISE
            : (this.RED_EXTRA_RISE || 0);

        lanes.forEach((lane) => {
            const laneZ = Number.isFinite(lane?.zPos) ? lane.zPos : null;
            for (let headIdx = 0; headIdx < headCount; headIdx++) {
                const vObj = lane && Array.isArray(lane.sideCopies)
                    ? lane.sideCopies.find(sc => sc && sc.type === 'V' && sc.headIndex === headIdx)
                    : null;
                const vVec = vObj && vObj.vec ? vObj.vec : null;
                const instanceCount = Number.isFinite(vVec?.instanceCount)
                    ? vVec.instanceCount
                    : (Number.isFinite(ctx?.vectorPrismCount) ? ctx.vectorPrismCount : undefined);
                const targetX = Number.isFinite(vVec?.group?.position?.x)
                    ? vVec.group.position.x
                    : (ctx?.headCoords && ctx.headCoords[headIdx] ? ctx.headCoords[headIdx].v : 0);
                const baseY = Number.isFinite(vVec?.group?.position?.y)
                    ? vVec.group.position.y
                    : fallbackBaseY;
                const targetY = baseY + dockOffset;
                const zPos = Number.isFinite(laneZ)
                    ? laneZ
                    : (Number.isFinite(vVec?.group?.position?.z) ? vVec.group.position.z : 0);

                const data = this._buildWeightedSumData(headIdx, lane, lanes, outputLength);
                const spawnPos = new THREE.Vector3(targetX, targetY, zPos);
                const wsVec = new VectorVisualizationInstancedPrism(
                    data.slice(),
                    spawnPos,
                    3,
                    instanceCount
                );
                wsVec.userData = wsVec.userData || {};
                wsVec.userData.isWeightedSum = true;
                wsVec.userData.headIndex = headIdx;
                wsVec.userData.parentLane = lane || null;
                wsVec.userData.weightedSumLaneZ = zPos;
                wsVec.userData.weightedSumReadyForConcat = true;
                wsVec.userData.weightedSumDocked = true;
                wsVec.group.userData = wsVec.group.userData || {};
                wsVec.group.userData.label = (lane && lane.tokenLabel)
                    ? `Attention Weighted Sum - ${lane.tokenLabel}`
                    : 'Attention Weighted Sum';
                if (ctx && ctx.parentGroup) {
                    ctx.parentGroup.add(wsVec.group);
                }
                this._applyWeightedSumScheme(wsVec, data);
                try {
                    if (ctx && typeof ctx.registerWeightedSumVector === 'function') {
                        ctx.registerWeightedSumVector(wsVec, zPos, headIdx, lane);
                    }
                } catch (_) { /* optional */ }
                created += 1;
            }
        });

        return created;
    }

    _copyVectorAppearance(target, source) {
        if (!target || !target.mesh || !source || !source.mesh) return;
        const srcMesh = source.mesh;
        const dstMesh = target.mesh;
        const tmpMat = new THREE.Matrix4();
        const instCount = Math.min(target.instanceCount || 0, source.instanceCount || 0);
        for (let i = 0; i < instCount; i++) {
            srcMesh.getMatrixAt(i, tmpMat);
            dstMesh.setMatrixAt(i, tmpMat);
        }
        if (dstMesh.instanceMatrix) dstMesh.instanceMatrix.needsUpdate = true;
        const srcCS = srcMesh.geometry?.getAttribute?.('colorStart');
        const srcCE = srcMesh.geometry?.getAttribute?.('colorEnd');
        const dstCS = dstMesh.geometry?.getAttribute?.('colorStart');
        const dstCE = dstMesh.geometry?.getAttribute?.('colorEnd');
        if (srcCS && dstCS && srcCS.array && dstCS.array) {
            dstCS.array.set(srcCS.array);
            dstCS.needsUpdate = true;
        }
        if (srcCE && dstCE && srcCE.array && dstCE.array) {
            dstCE.array.set(srcCE.array);
            dstCE.needsUpdate = true;
        }
        if (srcMesh.instanceColor) {
            if (!dstMesh.instanceColor) {
                dstMesh.instanceColor = new THREE.InstancedBufferAttribute(
                    new Float32Array(target.instanceCount * 3),
                    3
                );
            }
            dstMesh.instanceColor.array.set(srcMesh.instanceColor.array);
            dstMesh.instanceColor.needsUpdate = true;
        }
        if (srcMesh.material && dstMesh.material) {
            dstMesh.material.transparent = srcMesh.material.transparent;
            dstMesh.material.opacity = srcMesh.material.opacity;
            if ('emissiveIntensity' in dstMesh.material && 'emissiveIntensity' in srcMesh.material) {
                dstMesh.material.emissiveIntensity = srcMesh.material.emissiveIntensity;
            }
            if (dstMesh.material.emissive && srcMesh.material.emissive) {
                dstMesh.material.emissive.copy(srcMesh.material.emissive);
            }
            dstMesh.material.needsUpdate = true;
        }
    }

    _animateBlueVector(vector, headIdx, i, laneZs, allDoneCb) {
        if (this.skipRequested) {
            this._finishBlueImmediately(vector, headIdx, allDoneCb);
            return;
        }
        // Note: indices now start at 1, so the former i==0 no-movement case will not occur.
        const horizontalToK = this.ctx.headCoords && this.ctx.headCoords[headIdx]
            ? this.ctx.headCoords[headIdx].k
            : vector.group.position.x;


        // Convenience alias for durations / easing
        const QEasing = TWEEN.Easing.Quadratic.InOut;

        // ------------------------------------------------------------------
        // 1. Slide from Q column → K column
        // ------------------------------------------------------------------
        new TWEEN.Tween(vector.group.position)
            .to({ x: horizontalToK }, this.BLUE_HORIZ_DURATION)
            .easing(QEasing)
            .onComplete(() => {
                if (this.skipRequested) {
                    this._finishBlueImmediately(vector, headIdx, allDoneCb);
                    return;
                }
                // 2. Traverse along K vectors i times
                const spheres = [];
                this._traverseLanes(vector, laneZs, i, spheres, true, () => {
                    this._markAttentionRowComplete(headIdx, i - 1);
                    // Lift spheres upward to align with red vectors
                    this._riseSpheres(spheres, i - 1, headIdx);
                    // 3. FADE OUT at the LAST green (K) vector, then TELEPORT & FADE IN as red over V column
                    new TWEEN.Tween(vector.group.scale)
                        .to({ x: 0.001, y: 0.001, z: 0.001 }, this.BLUE_HORIZ_DURATION / 2)
                        .easing(QEasing)
                        .onComplete(() => {
                            // BEGIN NEW LOGIC – start red traversal using the first V copy and skip spawning a pre-visible red vector
                                if (vector.group.parent) vector.group.parent.remove(vector.group);
                                if (typeof vector.dispose === 'function') vector.dispose();
                                this._startRedTraversalFromFirstCopy(headIdx, i, laneZs, spheres, allDoneCb);
                                return;
                                // (legacy logic kept for reference below)
                                const targetX = horizontalToK; // stay aligned with K for pop-up effect
                            // Move (instantly) to the red-vector height on the top lane
                            vector.group.position.set(
                                targetX,
                                vector.group.position.y + this.RED_EXTRA_RISE,
                                laneZs[0]
                            );
                            // Re-colour the vector to RED before popping back in
                            this._setVectorColor(vector, this.ctx.brightRed || new THREE.Color(0xff0000));

                            // 4. FADE BACK IN (now red)
                            new TWEEN.Tween(vector.group.scale)
                                .to({ x: 1, y: 1, z: 1 }, this.BLUE_HORIZ_DURATION / 2)
                                .easing(QEasing)
                                .onComplete(() => {
                                    // 5. Traverse along lanes again i times (over red vectors)
                                    this._traverseLanes(vector, laneZs, i, spheres, false, () => {
                                        // 6. Fade / dispose after finishing red traversal
            new TWEEN.Tween(vector.group.scale)
                .to({ x: 0.001, y: 0.001, z: 0.001 }, this.DUPLICATE_POP_OUT_MS)
                                            .onComplete(() => {
                                                if (vector.group.parent) vector.group.parent.remove(vector.group);
                                                if (typeof vector.dispose === 'function') vector.dispose();
                                                allDoneCb && allDoneCb();
                                            })
                                            .start();
                                    });
                                })
                                .start();
                        })
                        .start();
                });
            })
            .start();
    }

    _startRedTraversalFromFirstCopy(headIdx, hopCount, laneZs, spheresArr, doneCb) {
        if (this.skipRequested) {
            doneCb && doneCb();
            return;
        }
        if (!this.ctx.currentLanes || laneZs.length === 0) {
            doneCb && doneCb();
            return;
        }
        const queryLaneZ = laneZs[Math.min(Math.max(0, hopCount - 1), laneZs.length - 1)];
        const queryLane = this.ctx.currentLanes.find(l => Math.abs(l.zPos - queryLaneZ) < 0.1);
        const topLaneZ = laneZs[0];
        const topLane = this.ctx.currentLanes.find(l => Math.abs(l.zPos - topLaneZ) < 0.1);
        if (!topLane || !Array.isArray(topLane.sideCopies)) {
            doneCb && doneCb();
            return;
        }
        const fixedObj = topLane.sideCopies.find(sc => sc.headIndex === headIdx && sc.type === 'V');
        if (!fixedObj || !fixedObj.vec) {
            doneCb && doneCb();
            return;
        }
        const fixedVec = fixedObj.vec;
        // Spawn travelling red vector OVER K column (horizontally offset) and ABOVE green vectors.
        const kX = (this.ctx.headCoords && this.ctx.headCoords[headIdx]) ? this.ctx.headCoords[headIdx].k : fixedVec.group.position.x;
        // Set vertical position to the **canonical** raised-V height so it always matches
        // fixed red vectors and highlight spheres, even if the fixed copy is still
        // mid-animation.
        const spawnY = this.ctx.mhaPassThroughTargetY + this.ctx.mhaResultRiseOffsetY - 30 + this.RED_EXTRA_RISE;
        const spawnPos = new THREE.Vector3(kX, spawnY, fixedVec.group.position.z);
        const travellingVec = new VectorVisualizationInstancedPrism(
            fixedVec.rawData.slice(),
            spawnPos,
            3,
            fixedVec.instanceCount
        );
        travellingVec.userData = { headIndex: headIdx, parentLane: queryLane, attnRowIndex: hopCount - 1 };
        this._activeBlueVectors[headIdx] = travellingVec;
        this.ctx.parentGroup.add(travellingVec.group);
        this._spawnedTempVectors.add(travellingVec);
        this._applyWeightedSumScheme(travellingVec, fixedVec.rawData);
        // Start invisible – will be revealed on first merge
        travellingVec.group.scale.set(0.001, 0.001, 0.001);

        // Begin traversal over red vectors
        this._traverseLanes(travellingVec, laneZs, hopCount, spheresArr, false, () => {
            this._parkWeightedSumVector(travellingVec, headIdx, queryLane, queryLaneZ);
            doneCb && doneCb();
        });
    }

    _parkWeightedSumVector(vector, headIdx, lane, laneZ) {
        if (!vector || !vector.group) return;
        const resolvedLaneZ = Number.isFinite(lane?.zPos) ? lane.zPos : (Number.isFinite(laneZ) ? laneZ : vector.group.position.z);
        const vObj = lane && Array.isArray(lane.sideCopies)
            ? lane.sideCopies.find(sc => sc && sc.headIndex === headIdx && sc.type === 'V')
            : null;
        const vVec = vObj && vObj.vec ? vObj.vec : null;
        const targetX = vVec && vVec.group ? vVec.group.position.x
            : (this.ctx.headCoords && this.ctx.headCoords[headIdx] ? this.ctx.headCoords[headIdx].v : vector.group.position.x);
        const baseY = vVec && vVec.group ? vVec.group.position.y : vector.group.position.y;
        const dockOffset = Number.isFinite(this.ctx?.weightedSumDockOffset) ? this.ctx.weightedSumDockOffset : 30;
        const targetY = baseY + dockOffset;
        vector.group.scale.set(1, 1, 1);

        vector.userData = vector.userData || {};
        vector.userData.isWeightedSum = true;
        vector.userData.headIndex = headIdx;
        vector.userData.parentLane = lane || vector.userData.parentLane || null;
        vector.userData.weightedSumLaneZ = resolvedLaneZ;
        vector.userData.weightedSumReadyForConcat = false;
        vector.userData.weightedSumDocked = false;
        vector.group.userData.label = (lane && lane.tokenLabel)
            ? `Attention Weighted Sum - ${lane.tokenLabel}`
            : 'Attention Weighted Sum';

        this._applyWeightedSumScheme(vector);

        // Ensure weighted sums survive cleanup of temp conveyor vectors
        this._spawnedTempVectors.delete(vector);
        if (this._activeBlueVectors && this._activeBlueVectors[headIdx] === vector) {
            delete this._activeBlueVectors[headIdx];
        }

        // Register with parent so it can drive concatenation later
        try {
            if (this.ctx && typeof this.ctx.registerWeightedSumVector === 'function') {
                this.ctx.registerWeightedSumVector(vector, resolvedLaneZ, headIdx, lane);
            }
        } catch (_) { /* optional */ }

        const applyGray = () => {
            if (vector.userData && vector.userData.weightedSumReadyForConcat) return;
            this._setVectorColor(vector, new THREE.Color(0x606060));
        };
        const grayDelayMs = 140;

        this._pendingDockCount += 1;

        const finalizeDock = () => {
            if (vector.userData) {
                vector.userData.weightedSumDocked = true;
            }
            this._pendingDockCount = Math.max(0, this._pendingDockCount - 1);
            this._checkGlobalCompletion();
        };

        if (typeof TWEEN !== 'undefined') {
            new TWEEN.Tween(vector.group.position)
                .to({ x: targetX, y: targetY, z: resolvedLaneZ }, this.DUPLICATE_TRAVEL_MERGE_MS)
                .easing(TWEEN.Easing.Quadratic.Out)
                .onComplete(() => {
                    const delayId = setTimeout(() => {
                        this._pendingTimeouts.delete(delayId);
                        applyGray();
                    }, grayDelayMs);
                    this._registerTimeout(delayId);
                    finalizeDock();
                })
                .start();
        } else {
            vector.group.position.set(targetX, targetY, resolvedLaneZ);
            const delayId = setTimeout(() => {
                this._pendingTimeouts.delete(delayId);
                applyGray();
            }, grayDelayMs);
            this._registerTimeout(delayId);
            finalizeDock();
        }
    }

    _traverseLanes(vector, laneZs, count, spheresArr, createSpheres, doneCb, stepIdx = 0) {
        if (this.skipRequested) {
            return;
        }
        if (count === 0 || stepIdx >= count || laneZs.length === 0) {
            // Nothing to traverse
            doneCb && doneCb();
            return;
        }

        const targetZ = laneZs[stepIdx < laneZs.length ? stepIdx : laneZs.length - 1];
        new TWEEN.Tween(vector.group.position)
            .to({ z: targetZ }, this.BLUE_VERT_DURATION)
            .easing(TWEEN.Easing.Quadratic.InOut)
            .onComplete(() => {
                if (this.skipRequested) return;
                const headIdx = (vector.userData && typeof vector.userData.headIndex === 'number') ? vector.userData.headIndex : null;
                if (createSpheres && Number.isFinite(headIdx)) {
                    const rowIndex = (vector.userData && Number.isFinite(vector.userData.attnRowIndex))
                        ? vector.userData.attnRowIndex
                        : null;
                    if (Number.isFinite(rowIndex)) {
                        this._setAttentionProgress(headIdx, rowIndex, stepIdx);
                    }
                }
                // Create a sphere between blue (vector) and corresponding green vector if enabled
                if (createSpheres) {
                    const headIdx = (vector.userData && typeof vector.userData.headIndex === 'number') ? vector.userData.headIndex : null;
                    if (headIdx !== null && this.ctx.currentLanes) {
                        const lane = this.ctx.currentLanes.find(l => Math.abs(l.zPos - targetZ) < 0.1);
                        if (lane && lane.upwardCopies && lane.upwardCopies[headIdx]) {
                            const greenVec = lane.upwardCopies[headIdx];
                            if (greenVec && greenVec.group) {
                                const midPoint = new THREE.Vector3().addVectors(vector.group.position, greenVec.group.position).multiplyScalar(0.5);
                                const sphereGeom = SHARED_SPHERE_GEOMETRY;
                                const queryLane = vector.userData ? vector.userData.parentLane : null;
                                const queryTokenIndex = queryLane && Number.isFinite(queryLane.tokenIndex) ? queryLane.tokenIndex : null;
                                const keyTokenIndex = lane && Number.isFinite(lane.tokenIndex) ? lane.tokenIndex : null;
                                const layerIndex = Number.isFinite(this.ctx?.layerIndex) ? this.ctx.layerIndex : null;
                                const preScore = (this.ctx && this.ctx.activationSource && Number.isFinite(layerIndex) && Number.isFinite(headIdx))
                                    ? this.ctx.activationSource.getAttentionScore(layerIndex, 'pre', headIdx, queryTokenIndex, keyTokenIndex)
                                    : null;
                                const baseColor = Number.isFinite(preScore)
                                    ? mapValueToColor(preScore)
                                    : new THREE.Color().setHSL(Math.random(), THREE.MathUtils.lerp(0.85, 1.0, Math.random()), THREE.MathUtils.lerp(0.45, 0.6, Math.random()));
                                const sphereMat = new THREE.MeshBasicMaterial({ color: baseColor });
                                const sphereMesh = new THREE.Mesh(sphereGeom, sphereMat);
                                sphereMesh.position.copy(midPoint);
                                sphereMesh.scale.set(0.001, 0.001, 0.001);
                                this.ctx.parentGroup.add(sphereMesh);
                                this._spawnedSpheres.add(sphereMesh);
                                const activationData = buildActivationData({
                                    label: 'Pre-Softmax Attention Score',
                                    stage: 'attention.pre',
                                    layerIndex,
                                    tokenIndex: queryTokenIndex,
                                    tokenLabel: queryLane ? queryLane.tokenLabel : null,
                                    headIndex: headIdx,
                                    preScore,
                                });
                                if (activationData && Number.isFinite(keyTokenIndex)) {
                                    activationData.keyTokenIndex = keyTokenIndex;
                                    activationData.keyTokenLabel = lane ? lane.tokenLabel : null;
                                }
                                applyActivationDataToObject(sphereMesh, activationData, 'Pre-Softmax Attention Score');
                                // Inflate animation
                                new TWEEN.Tween(sphereMesh.scale)
                                    .to({ x: 0.8, y: 0.8, z: 0.8 }, 350)
                                    .easing(TWEEN.Easing.Quadratic.Out)
                                    .start();
                                if (Array.isArray(spheresArr)) spheresArr.push(sphereMesh);
                            }
                        }
                    }
                }
                // Handle sphere removal during red-vector traversal
                if (!createSpheres && Array.isArray(spheresArr)) {
                    const idx = spheresArr.findIndex(s => Math.abs(s.position.z - targetZ) < 0.1);
                    if (idx >= 0) {
                        const sp = spheresArr[idx];
                        const ContinueTraversal = () => {
                            this._traverseLanes(vector, laneZs, count, spheresArr, createSpheres, doneCb, stepIdx + 1);
                        };
                        // ------------------------------------------------------------------
                        //  Spawn duplicate red vector from the fixed V vector and animate it
                        //  through the sphere and into the moving red vector.
                        // ------------------------------------------------------------------
                        const headIdx = (vector.userData && typeof vector.userData.headIndex === 'number') ? vector.userData.headIndex : null;
                        const lane = this.ctx.currentLanes ? this.ctx.currentLanes.find(l => Math.abs(l.zPos - targetZ) < 0.1) : null;
                        if (lane && headIdx !== null && Array.isArray(lane.sideCopies)) {
                            const fixedObj = lane.sideCopies.find(sc => sc.headIndex === headIdx && sc.type === 'V');
                            if (fixedObj && fixedObj.vec) {
                                const fixedVec = fixedObj.vec;
                                // Ensure duplicates spawn at the **raised** red-vector height (match the highlight spheres).
                                const raisedY = sp ? sp.position.y : vector.group.position.y;
                                const activationData = sp && sp.userData ? sp.userData.activationData : null;
                                const postScore = activationData && Number.isFinite(activationData.postScore)
                                    ? activationData.postScore
                                    : null;
                                const weight = Number.isFinite(postScore)
                                    ? THREE.MathUtils.clamp(postScore, 0, 1)
                                    : null;
                                const weightScale = 1.0;
                                const weightOpacity = Number.isFinite(weight)
                                    ? THREE.MathUtils.lerp(0.65, 1.0, weight)
                                    : 1.0;
                                const startPos = fixedVec.group.position.clone();
                                const dupVec = new VectorVisualizationInstancedPrism(
                                    fixedVec.rawData.slice(),
                                    startPos,
                                    3,
                                    fixedVec.instanceCount
                                );
                                // Start with the ORIGINAL value-vector look.
                                this._copyVectorAppearance(dupVec, fixedVec);
                                this.ctx.parentGroup.add(dupVec.group);
                                this._spawnedTempVectors.add(dupVec);
                                dupVec.group.scale.set(0.001, 0.001, 0.001);
                                // Optional quick pop-in (scale encodes weight)
                                new TWEEN.Tween(dupVec.group.scale)
                                    .to({ x: 1, y: 1, z: 1 }, this.DUPLICATE_POP_IN_MS)
                                    .easing(TWEEN.Easing.Quadratic.Out)
                                    .start();
                                // Pulse the post-softmax score to visualize weighting
                                if (sp) {
                                    const baseScale = sp.scale ? sp.scale.x : 1;
                                    const pulse = Number.isFinite(weight)
                                        ? THREE.MathUtils.lerp(1.15, 1.9, weight)
                                        : 1.4;
                                    new TWEEN.Tween(sp.scale)
                                        .to({ x: baseScale * pulse, y: baseScale * pulse, z: baseScale * pulse }, 180)
                                        .easing(TWEEN.Easing.Quadratic.Out)
                                        .yoyo(true)
                                        .repeat(1)
                                        .start();
                                }
                                const sumTarget = { x: vector.group.position.x, y: raisedY, z: vector.group.position.z };
                                const sumDuration = sp ? this.DUPLICATE_TRAVEL_MERGE_MS * 0.55 : this.DUPLICATE_TRAVEL_MERGE_MS;
                                const applyWeightedLook = () => {
                                    this._applyWeightedSumScheme(dupVec, fixedVec.rawData, { setHiddenToBlack: true });
                                    if (dupVec.mesh && dupVec.mesh.material) {
                                        dupVec.mesh.material.transparent = true;
                                        dupVec.mesh.material.opacity = weightOpacity;
                                        dupVec.mesh.material.needsUpdate = true;
                                    }
                                    // Keep size identical; only opacity changes for weighting.
                                    dupVec.group.scale.set(weightScale, weightScale, weightScale);
                                };
                                const flyToSum = () => {
                                    new TWEEN.Tween(dupVec.group.position)
                                        .to(sumTarget, sumDuration)
                                        .easing(TWEEN.Easing.Quadratic.InOut)
                                        .onComplete(() => {
                                        const pulseSum = () => {
                                            const baseScale = Math.max(1, vector.group.scale.x);
                                            const pulseFactor = Number.isFinite(weight)
                                                ? THREE.MathUtils.lerp(1.04, 1.14, weight)
                                                : 1.08;
                                            const state = { s: baseScale };
                                            new TWEEN.Tween(state)
                                                .to({ s: baseScale * pulseFactor }, 140)
                                                .easing(TWEEN.Easing.Quadratic.Out)
                                                .yoyo(true)
                                                .repeat(1)
                                                .onUpdate(() => {
                                                    vector.group.scale.set(state.s, state.s, state.s);
                                                })
                                                .start();
                                        };
                                        // Reveal travelling red vector on its first merge
                                        if (vector.group.scale.x < 0.5) {
                                            new TWEEN.Tween(vector.group.scale)
                                                .to({ x: 1, y: 1, z: 1 }, this.DUPLICATE_POP_IN_MS)
                                                .easing(TWEEN.Easing.Quadratic.Out)
                                                .onComplete(pulseSum)
                                                .start();
                                        } else {
                                            pulseSum();
                                        }
                                        // Fade / dispose duplicate
                                        new TWEEN.Tween(dupVec.group.scale)
                                            .to({ x: 0.001, y: 0.001, z: 0.001 }, this.DUPLICATE_POP_OUT_MS)
                                            .onComplete(() => {
                                                this._spawnedTempVectors.delete(dupVec);
                                                if (dupVec.group.parent) dupVec.group.parent.remove(dupVec.group);
                                                if (typeof dupVec.dispose === 'function') dupVec.dispose();
                                                // Continue traversal AFTER merge completes
                                                ContinueTraversal();
                                            })
                                            .start();
                                    }).start();
                                };

                                if (sp) {
                                    const scoreTarget = { x: sp.position.x, y: sp.position.y, z: sp.position.z };
                                    new TWEEN.Tween(dupVec.group.position)
                                        .to(scoreTarget, this.DUPLICATE_TRAVEL_MERGE_MS * 0.45)
                                        .easing(TWEEN.Easing.Quadratic.Out)
                                        .onComplete(() => {
                                            // Brief linger at the post-softmax score before merging into the running sum
                                            const delayId = setTimeout(() => {
                                                this._pendingTimeouts.delete(delayId);
                                                if (this.skipRequested) return;
                                                applyWeightedLook();
                                                flyToSum();
                                            }, 80);
                                            this._registerTimeout(delayId);
                                        })
                                        .start();
                                } else {
                                    applyWeightedLook();
                                    flyToSum();
                                }
                            }
                        }

                        // Shrink & remove sphere after the weighting merge completes
                        const shrinkDelay = Math.max(250, this.DUPLICATE_TRAVEL_MERGE_MS * 0.8);
                        new TWEEN.Tween(sp.scale)
                            .to({ x: 0.001, y: 0.001, z: 0.001 }, 250)
                            .delay(shrinkDelay)
                            .easing(TWEEN.Easing.Quadratic.In)
                            .onComplete(() => {
                                this._spawnedSpheres.delete(sp);
                                if (sp.parent) sp.parent.remove(sp);
                            })
                            .start();
                        spheresArr.splice(idx, 1);
                        // IMPORTANT: Return here so we do NOT recurse twice.
                        return;
                    }
                }
                // Pause briefly, then recurse
                const timeoutId = setTimeout(() => {
                    this._pendingTimeouts.delete(timeoutId);
                    if (this.skipRequested) {
                        doneCb && doneCb();
                        return;
                    }
                    this._traverseLanes(vector, laneZs, count, spheresArr, createSpheres, doneCb, stepIdx + 1);
                }, this.BLUE_PAUSE_MS);
                this._registerTimeout(timeoutId);
            })
            .start();
    }

    // ------------------------------------------------------------------
    // Utility helpers for global completion detection
    // ------------------------------------------------------------------
    _isConveyorComplete() {
        const anyProcessing = Object.values(this.blueProcessing).some(v => v);
        const anyQueued = Object.values(this.blueQueues).some(q => q && q.length > 0);
        return !anyProcessing && !anyQueued && this._pendingDockCount === 0;
    }

    _checkGlobalCompletion() {
        if (this.phase === 'complete') return;
        if (this._isConveyorComplete()) {
            this.phase = 'complete';
            // Notify parent to dispose merged K/V visuals immediately after
            // the last blue vector finishes its conveyor belt.
            try {
                // Dispose merged instanced groups first, then strip any remaining
                // individual K/V meshes to ensure nothing remains visible.
                this.ctx && this.ctx._disposeMergedKVGroups && this.ctx._disposeMergedKVGroups();
                this.ctx && this.ctx._disposeAllIndividualKandVVectorsImmediately && this.ctx._disposeAllIndividualKandVVectorsImmediately();
            } catch (_) { /* optional */ }
            this._flushCallbacks();
        }
    }

    // ------------------------------------------------------------------
    // Callback handling utilities
    // ------------------------------------------------------------------
    _flushCallbacks() {
        const list = this._callbacks.splice(0, this._callbacks.length);
        list.forEach(cb => {
            try { cb && cb(); } catch (err) { console.error(err); }
        });
    }
}
