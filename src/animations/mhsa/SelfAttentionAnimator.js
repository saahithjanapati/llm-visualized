import * as THREE from 'three';
import { VECTOR_LENGTH_PRISM, SA_RED_EXTRA_RISE, SA_V_RISE_DURATION_MS, SA_K_ALIGN_DURATION_MS, SA_BLUE_HORIZ_DURATION_MS, SA_BLUE_VERT_DURATION_MS, SA_BLUE_PAUSE_MS, SA_BLUE_QUEUE_SHIFT_DURATION_MS, SA_DUPLICATE_POP_IN_MS, SA_DUPLICATE_TRAVEL_MERGE_MS, SA_DUPLICATE_POP_OUT_MS, GLOBAL_ANIM_SPEED_MULT, SELF_ATTENTION_TIME_MULT } from '../../utils/constants.js';
import { VectorVisualizationInstancedPrism } from '../../components/VectorVisualizationInstancedPrism.js';
import { mapValueToColor, mapValueToGrayscale } from '../../utils/colors.js';
import { buildActivationData, applyActivationDataToObject } from '../../utils/activationMetadata.js';

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
                vec.group.visible = false;
                vec.group.scale.set(0.001, 0.001, 0.001);
            }
        } catch (_) { /* optional cleanup */ }
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
        this.skipRequested = true;
        this._clearPendingTimeouts();
        // Retire any active vectors and queued blues
        Object.values(this._activeBlueVectors || {}).forEach((vec) => this._retireVector(vec, { preserveTrail: preserveTrails }));
        Object.values(this.blueQueues || {}).forEach((queue) => {
            if (Array.isArray(queue)) queue.forEach((vec) => this._retireVector(vec, { preserveTrail: preserveTrails }));
        });
        this.blueQueues = {};
        this.blueProcessing = {};
        this.blueProcessedCount = {};
        this.phase = 'complete';
        // Clean up any lingering highlight spheres
        this._spawnedSpheres.forEach((sp) => {
            try {
                if (sp.parent) sp.parent.remove(sp);
            } catch (_) { /* optional cleanup */ }
        });
        this._spawnedSpheres.clear();
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
            this._checkGlobalCompletion();
            return;
        }
        if (!queue || queue.length === 0) {
            // Finished all blues for this head
            this.blueProcessing[headIdx] = false;
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

    _riseSpheres(spheresArr) {

        if (this.skipRequested) return;
        if (!Array.isArray(spheresArr) || spheresArr.length === 0) return;
        spheresArr.forEach(sp => {
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
                const postScore = activationData && Number.isFinite(activationData.postScore) ? activationData.postScore : null;
                const targetColor = Number.isFinite(postScore)
                    ? mapValueToGrayscale(postScore)
                    : new THREE.Color().setHSL(0, 0, THREE.MathUtils.lerp(0.2, 0.9, Math.random()));
                const targetEI = Number.isFinite(postScore) ? 0.6 : 0.8;
                new TWEEN.Tween(state)
                    .to({ r: targetColor.r, g: targetColor.g, b: targetColor.b, ei: targetEI }, this.V_RISE_DURATION)
                    .easing(TWEEN.Easing.Quadratic.Out)
                    .onUpdate(() => {
                        sp.material.color.setRGB(state.r, state.g, state.b);
                        if (hasEmissive) {
                            sp.material.emissive.setRGB(state.r, state.g, state.b);
                            sp.material.emissiveIntensity = state.ei;
                        }
                    })
                    .start();
                if (activationData) {
                    activationData.stage = 'attention.post';
                    applyActivationDataToObject(sp, activationData, 'Attention Score (Post-softmax)');
                } else {
                    applyActivationDataToObject(sp, null, 'Attention Score (Post-softmax)');
                }
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
                    // Lift spheres upward to align with red vectors
                    this._riseSpheres(spheres);
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
        travellingVec.userData = { headIndex: headIdx };
        this._activeBlueVectors[headIdx] = travellingVec;
        this.ctx.parentGroup.add(travellingVec.group);
        travellingVec.applyProcessedVisuals(
            fixedVec.rawData.slice(),
            this.ctx.outputVectorLength || 64,
            { numKeyColors: this.ctx.outputVectorLength || 64 },
            { setHiddenToBlack: true }
        );
        // Start invisible – will be revealed on first merge
        travellingVec.group.scale.set(0.001, 0.001, 0.001);

        // Begin traversal over red vectors
        this._traverseLanes(travellingVec, laneZs, hopCount, spheresArr, false, () => {
            new TWEEN.Tween(travellingVec.group.scale)
                .to({ x: 0.001, y: 0.001, z: 0.001 }, this.DUPLICATE_POP_OUT_MS)
                .onComplete(() => {
                    delete this._activeBlueVectors[headIdx];
                    if (travellingVec.group.parent) travellingVec.group.parent.remove(travellingVec.group);
                    if (typeof travellingVec.dispose === 'function') travellingVec.dispose();
                    doneCb && doneCb();
                })
                .start();
        });
    }

    _traverseLanes(vector, laneZs, count, spheresArr, createSpheres, doneCb, stepIdx = 0) {
        if (this.skipRequested) {
            doneCb && doneCb();
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
                                const postScore = (this.ctx && this.ctx.activationSource && Number.isFinite(layerIndex) && Number.isFinite(headIdx))
                                    ? this.ctx.activationSource.getAttentionScore(layerIndex, 'post', headIdx, queryTokenIndex, keyTokenIndex)
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
                                    label: 'Attention Score (Pre-softmax)',
                                    stage: 'attention.pre',
                                    layerIndex,
                                    tokenIndex: queryTokenIndex,
                                    tokenLabel: queryLane ? queryLane.tokenLabel : null,
                                    headIndex: headIdx,
                                    preScore,
                                    postScore,
                                });
                                if (activationData && Number.isFinite(keyTokenIndex)) {
                                    activationData.keyTokenIndex = keyTokenIndex;
                                    activationData.keyTokenLabel = lane ? lane.tokenLabel : null;
                                }
                                applyActivationDataToObject(sphereMesh, activationData, 'Attention Score (Pre-softmax)');
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
                                const startPos = new THREE.Vector3(
                                    fixedVec.group.position.x,
                                    raisedY,
                                    fixedVec.group.position.z
                                );
                                const dupVec = new VectorVisualizationInstancedPrism(
                                    fixedVec.rawData.slice(),
                                    startPos,
                                    3,
                                    fixedVec.instanceCount
                                );
                                // Make duplicate visually match the travelling 64-dim vector
                                dupVec.applyProcessedVisuals(
                                    fixedVec.rawData.slice(),
                                    this.ctx.outputVectorLength || 64,
                                    { numKeyColors: this.ctx.outputVectorLength || 64 },
                                    { setHiddenToBlack: true }
                                );
                                this.ctx.parentGroup.add(dupVec.group);
                                dupVec.group.scale.set(0.001, 0.001, 0.001);
                                // Optional quick pop-in
                new TWEEN.Tween(dupVec.group.scale).to({ x: 1, y: 1, z: 1 }, this.DUPLICATE_POP_IN_MS).start();
                                // Direct: fixed V -> travelling red vector
                                new TWEEN.Tween(dupVec.group.position)
                                    .to({ x: vector.group.position.x, y: raisedY, z: vector.group.position.z }, this.DUPLICATE_TRAVEL_MERGE_MS)
                                    .easing(TWEEN.Easing.Quadratic.InOut)
                                    .onComplete(() => {
                                                // Reveal travelling red vector on its first merge
                                                if (vector.group.scale.x < 0.5) {
                                                new TWEEN.Tween(vector.group.scale)
                                                    .to({ x: 1, y: 1, z: 1 }, this.DUPLICATE_POP_IN_MS)
                                                        .easing(TWEEN.Easing.Quadratic.Out)
                                                        .start();
                                                }
                                            // Fade / dispose duplicate
                                                new TWEEN.Tween(dupVec.group.scale)
                                                    .to({ x: 0.001, y: 0.001, z: 0.001 }, this.DUPLICATE_POP_OUT_MS)
                                                    .onComplete(() => {
                                                        if (dupVec.group.parent) dupVec.group.parent.remove(dupVec.group);
                                                        if (typeof dupVec.dispose === 'function') dupVec.dispose();
                                                        // Continue traversal AFTER merge completes
                                                        ContinueTraversal();
                                                    })
                                                    .start();
                                            }).start();
                            }
                        }

                        // Shrink & remove sphere immediately after duplicate leaves
                        new TWEEN.Tween(sp.scale)
                            .to({ x: 0.001, y: 0.001, z: 0.001 }, 250)
                            .delay(250)
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
        return !anyProcessing && !anyQueued;
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
