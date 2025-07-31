import * as THREE from 'three';

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
        // V/K specific constants
        // ------------------------------------------------------------------
        this.RED_EXTRA_RISE   = 75;   // additional rise for red (V) vectors
        this.V_RISE_DURATION  = 600;  // ms
        this.K_ALIGN_DURATION = 1000; // ms

        // ------------------------------------------------------------------
        // NEW – Query (blue) conveyor-belt constants
        // ------------------------------------------------------------------
        this.BLUE_HORIZ_DURATION      = 400;  // ms for horizontal slides
        this.BLUE_VERT_DURATION       = 400;  // ms per lane hop (z-travel)
        this.BLUE_PAUSE_MS            = 100;  // pause at each lane
        this.BLUE_QUEUE_SHIFT_DURATION = 400; // remaining blues shift up

        // Internal bookkeeping (per-head)
        this.blueQueues        = {};  // { headIdx: [vec, ...] }
        this.blueProcessing    = {};  // { headIdx: boolean }
        this.blueProcessedCount = {}; // { headIdx: number }
        this.greensAligned     = {};  // { headIdx: boolean } – flagged once K vectors are in place
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
    _animateVVectorRise(vector, onDone) {
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
            this._kickoffBlueConveyor(headIdx);
            onDone && onDone();
        }
    }

    // ------------------------------------------------------------------
    // Blue (Query) conveyor-belt logic
    // ------------------------------------------------------------------
    _enqueueBlueVector(vector) {
        const headIdx = (vector.userData && typeof vector.userData.headIndex === 'number')
            ? vector.userData.headIndex : null;
        if (headIdx === null) return;

        if (!this.blueQueues[headIdx]) this.blueQueues[headIdx] = [];
        const queue = this.blueQueues[headIdx];

        queue.push(vector);
        // Keep queue ordered by z (ascending) so index == lane order (top → bottom)
        queue.sort((a, b) => a.group.position.z - b.group.position.z);

        // If greens are already in position we can start processing immediately.
        if (this.greensAligned[headIdx]) {
            this._kickoffBlueConveyor(headIdx);
        }
    }

    _kickoffBlueConveyor(headIdx) {
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

        this._animateBlueVector(vector, headIdx, i, laneZs, () => {
            // Recursive continuation
            this._processNextBlueVector(headIdx);
        });
    }

    _shiftRemainingBlueVectors(queue, headIdx) {
        const laneZs = (this.ctx.currentLanes || []).map(l => l.zPos).sort((a, b) => a - b);
        queue.forEach((vec, idx) => {
            if (!laneZs[idx]) return;
            new TWEEN.Tween(vec.group.position)
                .to({ z: laneZs[idx] }, this.BLUE_QUEUE_SHIFT_DURATION)
                .easing(TWEEN.Easing.Quadratic.InOut)
                .start();
        });
    }

    _animateBlueVector(vector, headIdx, i, laneZs, allDoneCb) {
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
                // 2. Traverse along K vectors i times
                this._traverseLanes(vector, laneZs, i, () => {
                    // 3. Rise vertically to align with red (V) vector height while staying over K column
                    new TWEEN.Tween(vector.group.position)
                        .to({ y: vector.group.position.y + this.RED_EXTRA_RISE }, this.BLUE_HORIZ_DURATION)
                        .easing(QEasing)
                        .onComplete(() => {
                            // 4. Traverse along lanes again i times at elevated height
                            this._traverseLanes(vector, laneZs, i, () => {
                                // 5. Fade / dispose
                                new TWEEN.Tween(vector.group.scale)
                                    .to({ x: 0.001, y: 0.001, z: 0.001 }, 500)
                                    .onComplete(() => {
                                        // Remove from scene & free GPU memory if possible
                                        if (vector.group.parent) vector.group.parent.remove(vector.group);
                                        if (typeof vector.dispose === 'function') vector.dispose();
                                        allDoneCb && allDoneCb();
                                    })
                                    .start();
                            });
                        })
                        .start();
                });
            })
            .start();
    }

    _traverseLanes(vector, laneZs, count, doneCb, stepIdx = 0) {
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
                // Pause briefly, then recurse
                setTimeout(() => {
                    this._traverseLanes(vector, laneZs, count, doneCb, stepIdx + 1);
                }, this.BLUE_PAUSE_MS);
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
