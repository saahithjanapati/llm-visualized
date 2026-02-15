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
});
