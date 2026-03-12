import { afterEach, describe, expect, it, vi } from 'vitest';

vi.mock('../src/state/appState.js', () => ({
    AppState: class AppState {},
    appState: {},
    default: {}
}));

import Gpt2Layer from '../src/engine/layers/Gpt2Layer.js';
import { LN2_PHASE } from '../src/engine/layers/gpt2LanePhases.js';

describe('Gpt2Layer pending-addition watchdog', () => {
    afterEach(() => {
        vi.restoreAllMocks();
    });

    it('forces overdue final additions once every lane is otherwise done', () => {
        const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {});
        const lane = {
            laneIndex: 2,
            ln2Phase: LN2_PHASE.DONE,
            additionComplete: false,
            __additionPending: true,
            __additionExpectedCompleteAtMs: 1000,
            __additionWatchdogGraceMs: 200
        };
        const completePendingAddition = vi.fn((targetLane) => {
            targetLane.additionComplete = true;
            return true;
        });
        const layer = {
            index: 5,
            _pendingAdditions: 1,
            _completePendingAddition: completePendingAddition,
            _emitProgress: vi.fn()
        };

        const mutated = Gpt2Layer.prototype._applyPendingAdditionWatchdog.call(layer, [lane], 1301);

        expect(mutated).toBe(true);
        expect(completePendingAddition).toHaveBeenCalledWith(lane);
        expect(warnSpy).toHaveBeenCalled();
    });

    it('does not force completion before the expected deadline plus grace period', () => {
        const lane = {
            laneIndex: 0,
            ln2Phase: LN2_PHASE.DONE,
            additionComplete: false,
            __additionPending: true,
            __additionExpectedCompleteAtMs: 1000,
            __additionWatchdogGraceMs: 400
        };
        const layer = {
            index: 1,
            _pendingAdditions: 1,
            _completePendingAddition: vi.fn(() => true),
            _emitProgress: vi.fn()
        };

        const mutated = Gpt2Layer.prototype._applyPendingAdditionWatchdog.call(layer, [lane], 1399);

        expect(mutated).toBe(false);
        expect(layer._completePendingAddition).not.toHaveBeenCalled();
    });

    it('clears leaked pending-addition counters once no lane still tracks a pending addition', () => {
        const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {});
        const layer = {
            index: 7,
            _pendingAdditions: 2,
            _completePendingAddition: vi.fn(() => false),
            _emitProgress: vi.fn()
        };
        const lanes = [
            { laneIndex: 0, ln2Phase: LN2_PHASE.DONE, additionComplete: true },
            { laneIndex: 1, ln2Phase: LN2_PHASE.DONE, additionComplete: true }
        ];

        const mutated = Gpt2Layer.prototype._applyPendingAdditionWatchdog.call(layer, lanes, 5000);

        expect(mutated).toBe(true);
        expect(layer._pendingAdditions).toBe(0);
        expect(layer._emitProgress).toHaveBeenCalled();
        expect(warnSpy).toHaveBeenCalled();
    });
});
