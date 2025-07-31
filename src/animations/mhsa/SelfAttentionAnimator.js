import * as THREE from 'three';

export class SelfAttentionAnimator {
    /**
     * @param {MHSAAnimation} ctx – parent animation instance.
     */
    constructor(ctx) {
        this.ctx = ctx;
        this.phase = 'waiting'; // 'waiting' | 'running' | 'complete'
        this._callbacks = [];
        // Constants for above-matrix animations
        this.RED_EXTRA_RISE = 75;  // additional rise for red (V) vectors
        this.V_RISE_DURATION = 600; // duration for V vector additional rise
        this.K_ALIGN_DURATION = 1000; // duration for K vector horizontal alignment
    }

    /**
     * Start the self-attention phase. Can be called in two ways:
     * 1. Legacy mode: start(onDone) - just runs placeholder animation
     * 2. Vector mode: start(vector, vectorCategory, onDone) - handles above-matrix animations
     * @param {Object|Function} vectorOrCallback - Vector object or callback function for legacy mode
     * @param {string} vectorCategory - Either 'V', 'K', or other (only for vector mode)
     * @param {Function} onDone - Callback when animations complete (only for vector mode)
     */
    start(vectorOrCallback = null, vectorCategory = null, onDone = null) {
        // Detect calling pattern: if first argument is a function, it's legacy mode
        if (typeof vectorOrCallback === 'function') {
            // Legacy mode: start(onDone)
            this._callbacks.push(vectorOrCallback);
            this._runPlaceholderMode();
            return;
        }

        // Vector mode: start(vector, vectorCategory, onDone) - handle each vector independently
        const vector = vectorOrCallback;
        
        // Handle each vector independently without global phase management

        // Handle above-matrix animations based on vector category (independent of global phase)
        this._handleAboveMatrixAnimations(vector, vectorCategory, onDone);
    }

    /**
     * Run the placeholder animation mode (legacy behavior)
     */
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

    /**
     * Handle animations that occur above the matrices after pass-through
     * @param {Object} vector - The vector to animate
     * @param {string} vectorCategory - Either 'V', 'K', or other
     * @param {Function} onDone - Callback when this specific vector's animation completes
     */
    _handleAboveMatrixAnimations(vector, vectorCategory, onDone) {
        if (typeof TWEEN === 'undefined') {
            console.error('Global TWEEN object not loaded for SelfAttentionAnimator!');
            if (onDone) onDone();
            return;
        }

        // For V vectors: additional rise and K vector alignment
        if (vectorCategory === 'V') {
            this._animateVVectorRise(vector, onDone);
        } else {
            // For non-V vectors, complete immediately
            if (onDone) onDone();
        }
    }

    /**
     * Animate V vector additional rise and trigger K vector alignment
     * @param {Object} vector - The V vector to animate
     * @param {Function} onDone - Callback when animation completes
     */
    _animateVVectorRise(vector, onDone) {
        new TWEEN.Tween({ y: vector.group.position.y })
            .to({ y: vector.group.position.y + this.RED_EXTRA_RISE }, this.V_RISE_DURATION)
            .easing(TWEEN.Easing.Quadratic.Out)
            .onUpdate(obj => { 
                vector.group.position.y = obj.y; 
            })
            .onComplete(() => {
                this._alignKVectorsUnderV(vector, onDone);
            })
            .start();
    }

    /**
     * Move corresponding green (K) vectors horizontally under the red (V) vector
     * @param {Object} redVector - The V vector that other vectors should align under
     * @param {Function} onDone - Callback when alignment completes
     */
    _alignKVectorsUnderV(redVector, onDone) {
        const redX = redVector.group.position.x;
        const redZ = redVector.group.position.z;
        const headIdx = redVector.userData && typeof redVector.userData.headIndex === 'number' 
            ? redVector.userData.headIndex : null;
        
        if (headIdx !== null && this.ctx.currentLanes) {
            let alignmentsInProgress = 0;
            let alignmentsCompleted = 0;
            
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
                                if (onDone) onDone();
                            }
                        })
                        .start();
                }
            });
            
            // If no alignments were started, complete immediately
            if (alignmentsInProgress === 0) {
                if (onDone) onDone();
            }
        } else {
            if (onDone) onDone();
        }
    }

    /**
     * Complete the self-attention phase and flush callbacks
     */
    _completePhase() {
        this.phase = 'complete';
        console.log('SelfAttentionAnimator: above-matrix animations complete');
        this._flushCallbacks();
    }

    _flushCallbacks() {
        const list = this._callbacks.splice(0, this._callbacks.length);
        list.forEach(cb => {
            try { cb && cb(); } catch(err) { console.error(err); }
        });
    }
} 