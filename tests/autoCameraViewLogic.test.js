import { describe, expect, it } from 'vitest';
import {
    getAutoCameraViewSwitchHoldMs,
    resolveAutoCameraViewState,
    resolveStableAutoCameraViewKey
} from '../src/engine/autoCameraViewLogic.js';
import { HORIZ_PHASE, LN2_PHASE } from '../src/engine/layers/gpt2LanePhases.js';

describe('autoCameraViewLogic.resolveAutoCameraViewState', () => {
    it('returns default when no layers exist', () => {
        const result = resolveAutoCameraViewState({ layers: [] });
        expect(result.rawKey).toBe('default');
        expect(result.viewContext).toBeNull();
    });

    it('returns final when forward pass is complete', () => {
        const pipeline = { isForwardPassComplete: () => true };
        const result = resolveAutoCameraViewState({
            pipeline,
            layers: [{ lanes: [{ horizPhase: HORIZ_PHASE.WAITING, ln2Phase: LN2_PHASE.NOT_STARTED }] }],
            currentLayerIdx: 0
        });
        expect(result.rawKey).toBe('final');
    });

    it('returns ln when active lane is inside layer norm', () => {
        const pipeline = { isForwardPassComplete: () => false };
        const result = resolveAutoCameraViewState({
            pipeline,
            layers: [{ lanes: [{ horizPhase: HORIZ_PHASE.INSIDE_LN, ln2Phase: LN2_PHASE.NOT_STARTED }] }],
            currentLayerIdx: 0
        });
        expect(result.rawKey).toBe('ln');
    });

    it('returns top-ln for the final LayerNorm endgame phase', () => {
        const pipeline = { isForwardPassComplete: () => false };
        const result = resolveAutoCameraViewState({
            pipeline,
            layers: [{
                index: 11,
                lanes: [{ horizPhase: HORIZ_PHASE.WAITING, ln2Phase: LN2_PHASE.NOT_STARTED }]
            }],
            currentLayerIdx: 11,
            isTopLayerNormCameraPhase: () => true
        });
        expect(result.rawKey).toBe('top-ln');
        expect(result.viewContext?.inTopLn).toBe(true);
    });

    it('returns ln when active lane is moving right toward first layer norm', () => {
        const pipeline = { isForwardPassComplete: () => false };
        const result = resolveAutoCameraViewState({
            pipeline,
            layers: [{ lanes: [{ horizPhase: HORIZ_PHASE.RIGHT, ln2Phase: LN2_PHASE.NOT_STARTED }] }],
            currentLayerIdx: 0
        });
        expect(result.rawKey).toBe('ln');
    });

    it('returns ln when active lane is moving right toward second layer norm', () => {
        const pipeline = { isForwardPassComplete: () => false };
        const result = resolveAutoCameraViewState({
            pipeline,
            layers: [{
                mhsaAnimation: {
                    mhaPassThroughPhase: 'mha_pass_through_complete',
                    rowMergePhase: 'merged',
                    outputProjMatrixAnimationPhase: 'vectors_inside',
                    outputProjMatrixReturnComplete: false
                },
                lanes: [{
                    horizPhase: HORIZ_PHASE.WAITING_FOR_LN2,
                    ln2Phase: LN2_PHASE.RIGHT
                }]
            }],
            currentLayerIdx: 0
        });
        expect(result.rawKey).toBe('ln');
    });

    it('keeps ln view while vectors rise out of layer norm', () => {
        const pipeline = { isForwardPassComplete: () => false };
        const result = resolveAutoCameraViewState({
            pipeline,
            layers: [{ lanes: [{ horizPhase: HORIZ_PHASE.RISE_ABOVE_LN, ln2Phase: LN2_PHASE.NOT_STARTED }] }],
            currentLayerIdx: 0
        });
        expect(result.rawKey).toBe('ln');
    });

    it('switches directly to travel when MHSA staging starts', () => {
        const pipeline = { isForwardPassComplete: () => false };
        const result = resolveAutoCameraViewState({
            pipeline,
            layers: [{
                mhsaAnimation: {
                    mhaPassThroughPhase: 'positioning_mha_vectors',
                    rowMergePhase: 'not_started',
                    outputProjMatrixAnimationPhase: 'waiting',
                    outputProjMatrixReturnComplete: false
                },
                lanes: [{ horizPhase: HORIZ_PHASE.READY_MHSA, ln2Phase: LN2_PHASE.NOT_STARTED }]
            }],
            currentLayerIdx: 0
        });
        expect(result.rawKey).toBe('travel');
    });

    it('holds prior view during handoff when transition hold is active', () => {
        const pipeline = { isForwardPassComplete: () => false };
        const result = resolveAutoCameraViewState({
            pipeline,
            layers: [
                { lanes: [{ horizPhase: HORIZ_PHASE.WAITING, ln2Phase: LN2_PHASE.NOT_STARTED }] },
                { lanes: [{ horizPhase: HORIZ_PHASE.WAITING, ln2Phase: LN2_PHASE.NOT_STARTED }] }
            ],
            currentLayerIdx: 1,
            isLargeDesktopViewport: true
        });
        expect(result.rawKey).toBe('default');
    });

    it('returns layer-end-desktop during handoff when prior key already settled', () => {
        const pipeline = { isForwardPassComplete: () => false };
        const result = resolveAutoCameraViewState({
            pipeline,
            layers: [
                { lanes: [{ horizPhase: HORIZ_PHASE.WAITING, ln2Phase: LN2_PHASE.NOT_STARTED }] },
                { lanes: [{ horizPhase: HORIZ_PHASE.WAITING, ln2Phase: LN2_PHASE.NOT_STARTED }] }
            ],
            currentLayerIdx: 1,
            priorViewKey: 'layer-end-desktop',
            isLargeDesktopViewport: true
        });
        expect(result.rawKey).toBe('layer-end-desktop');
    });

    it('returns concat during row merge before output projection return completes', () => {
        const pipeline = { isForwardPassComplete: () => false };
        const result = resolveAutoCameraViewState({
            pipeline,
            layers: [{
                mhsaAnimation: {
                    mhaPassThroughPhase: 'mha_pass_through_complete',
                    rowMergePhase: 'merging',
                    outputProjMatrixAnimationPhase: 'waiting',
                    outputProjMatrixReturnComplete: false
                },
                lanes: [{ horizPhase: HORIZ_PHASE.WAITING, ln2Phase: LN2_PHASE.DONE }]
            }],
            currentLayerIdx: 0
        });
        expect(result.rawKey).toBe('concat');
    });

    it('returns final as soon as the last-layer vectors enter the top unembedding path', () => {
        const pipeline = { isForwardPassComplete: () => false };
        const result = resolveAutoCameraViewState({
            pipeline,
            layers: [{
                index: 11,
                __topEmbedEntryYLocal: 20000,
                lanes: [{
                    horizPhase: HORIZ_PHASE.WAITING,
                    ln2Phase: LN2_PHASE.NOT_STARTED,
                    originalVec: {
                        group: {
                            position: { y: 20000.25 }
                        }
                    }
                }]
            }],
            currentLayerIdx: 11
        });
        expect(result.rawKey).toBe('final');
        expect(result.viewContext?.inTopEmbedding).toBe(true);
    });

    it('holds concat framing during output projection/residual add after concat', () => {
        const pipeline = { isForwardPassComplete: () => false };
        const result = resolveAutoCameraViewState({
            pipeline,
            priorViewKey: 'concat',
            layers: [{
                mhsaAnimation: {
                    mhaPassThroughPhase: 'mha_pass_through_complete',
                    rowMergePhase: 'merged',
                    outputProjMatrixAnimationPhase: 'vectors_inside',
                    outputProjMatrixReturnComplete: false
                },
                lanes: [{
                    horizPhase: HORIZ_PHASE.POST_MHSA_ADDITION,
                    ln2Phase: LN2_PHASE.PRE_RISE,
                    stopRise: true,
                    stopRiseTarget: {}
                }]
            }],
            currentLayerIdx: 0
        });
        expect(result.rawKey).toBe('concat');
        expect(result.viewContext?.holdViewDuringResidualAdd).toBe(true);
        expect(result.viewContext?.holdViewUntilLn2Inside).toBe(false);
    });
});

describe('autoCameraViewLogic hold and stable key', () => {
    it('increases hold window for residual-add and layer handoff contexts', () => {
        const hold = getAutoCameraViewSwitchHoldMs({
            fromKey: 'default',
            toKey: 'travel',
            baseHoldMs: 90,
            viewContext: {
                holdViewDuringResidualAdd: true,
                inLayerHandoff: true
            }
        });
        expect(hold).toBeGreaterThanOrEqual(220);
    });

    it('keeps current key until hold threshold elapses', () => {
        const first = resolveStableAutoCameraViewKey({
            rawKey: 'mhsa',
            currentKey: 'default',
            pendingKey: 'default',
            pendingSinceMs: 0,
            nowMs: 100,
            baseHoldMs: 90
        });
        expect(first.key).toBe('default');
        expect(first.pendingKey).toBe('mhsa');
        expect(first.pendingSinceMs).toBe(100);

        const second = resolveStableAutoCameraViewKey({
            rawKey: 'mhsa',
            currentKey: 'default',
            pendingKey: first.pendingKey,
            pendingSinceMs: first.pendingSinceMs,
            nowMs: 220,
            baseHoldMs: 90
        });
        expect(second.key).toBe('mhsa');
        expect(second.pendingSinceMs).toBe(0);
    });

    it('extends hold when switching from concat to default', () => {
        const hold = getAutoCameraViewSwitchHoldMs({
            fromKey: 'concat',
            toKey: 'default',
            baseHoldMs: 90
        });
        expect(hold).toBe(110);
    });

    it('shortens hold when switching from ln to travel', () => {
        const hold = getAutoCameraViewSwitchHoldMs({
            fromKey: 'ln',
            toKey: 'travel',
            baseHoldMs: 90
        });
        expect(hold).toBe(28);
    });

    it('uses the layer-norm hold floor when switching into top-ln', () => {
        const hold = getAutoCameraViewSwitchHoldMs({
            fromKey: 'default',
            toKey: 'top-ln',
            baseHoldMs: 40
        });
        expect(hold).toBe(72);
    });
});
