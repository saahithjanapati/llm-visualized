import { NUM_HEAD_SETS_LAYER } from '../../utils/constants.js';

export class PassThroughAnimator {
    constructor(ctx) {
        this.ctx = ctx; // parent MHSAAnimation instance
    }

    start(allLanes) {
        const ctx = this.ctx;
        if (ctx.mhaPassThroughPhase !== 'ready_for_parallel_pass_through') return;
        console.log('PassThroughAnimator: starting parallel pass-through');
        ctx.mhaPassThroughPhase = 'parallel_pass_through_active';

        // Kick off a single shared pulse per matrix to avoid per-vector
        // material updates every frame during pass-through.
        if (typeof ctx._startMatrixPulseDuringPassThrough === 'function') {
            try { ctx._startMatrixPulseDuringPassThrough(ctx.mhaPassThroughDuration); } catch (_) {}
        }

        let totalAnimationsToComplete = allLanes.length * NUM_HEAD_SETS_LAYER * 3;
        let animationsCompleted = 0;

        const singleAnimationDone = () => {
            animationsCompleted++;
            if (animationsCompleted >= totalAnimationsToComplete) {
                console.log('PassThroughAnimator: all pass-through tweens complete');
                ctx.mhaPassThroughPhase = 'mha_pass_through_complete';

                const continueAfterSelfAttn = () => {
                    // Temp / perm mode branching identical to legacy implementation
                    if (ctx.mode === 'temp' && !ctx._tempModeCompleted) {
                        ctx._applyTempModeBehaviour();
                        ctx._tempModeCompleted = true;
                    } else if (ctx.mode !== 'temp') {
                        ctx._transitionHeadColorsToFinal(1000);
                    }
                };

                if (ctx.enableSelfAttentionAnimation && ctx.selfAttentionAnimator) {
                    ctx.selfAttentionAnimator.start(continueAfterSelfAttn);
                } else {
                    continueAfterSelfAttn();
                }
            }
        };

        allLanes.forEach(lane => {
            for (let headIdx = 0; headIdx < NUM_HEAD_SETS_LAYER; headIdx++) {
                // K vector (upward copy in the centre)
                const kVec    = lane.upwardCopies[headIdx];
                const kMatrix = ctx.mhaVisualizations[headIdx * 3 + 1];
                ctx.animateVectorMatrixPassThrough(
                    kVec,
                    kMatrix,
                    ctx.brightKey,
                    ctx.darkTintedKey,
                    0.333,
                    ctx.mhaPassThroughTargetY,
                    ctx.mhaPassThroughDuration,
                    ctx.mhaResultRiseOffsetY,
                    ctx.mhaResultRiseDuration,
                    ctx.outputVectorLength,
                    singleAnimationDone,
                    'K'
                );

                // Q side copy
                const qSideCopy = lane.sideCopies.find(sc => sc.headIndex === headIdx && sc.type === 'Q');
                if (qSideCopy && qSideCopy.vec) {
                    ctx.animateVectorMatrixPassThrough(
                        qSideCopy.vec,
                        qSideCopy.matrixRef,
                        ctx.brightBlue,
                        ctx.darkTintedBlue,
                        0.666,
                        ctx.mhaPassThroughTargetY,
                        ctx.mhaPassThroughDuration,
                        ctx.mhaResultRiseOffsetY,
                        ctx.mhaResultRiseDuration,
                        ctx.outputVectorLength,
                        singleAnimationDone,
                        'Q'
                    );
                } else {
                    totalAnimationsToComplete--;
                }

                // V side copy
                const vSideCopy = lane.sideCopies.find(sc => sc.headIndex === headIdx && sc.type === 'V');
                if (vSideCopy && vSideCopy.vec) {
                    ctx.animateVectorMatrixPassThrough(
                        vSideCopy.vec,
                        vSideCopy.matrixRef,
                        ctx.brightValue,
                        ctx.darkTintedValue,
                        0.0,
                        ctx.mhaPassThroughTargetY,
                        ctx.mhaPassThroughDuration,
                        ctx.mhaResultRiseOffsetY,
                        ctx.mhaResultRiseDuration,
                        ctx.outputVectorLength,
                        singleAnimationDone,
                        'V'
                    );
                } else {
                    totalAnimationsToComplete--;
                }
            }
        });

        if (totalAnimationsToComplete === 0 && allLanes.length > 0) {
            console.log('PassThroughAnimator: no valid vectors found to animate');
            ctx.mhaPassThroughPhase = 'mha_pass_through_complete';
        }
    }
}