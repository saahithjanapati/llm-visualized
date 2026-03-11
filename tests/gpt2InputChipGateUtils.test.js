import { describe, expect, it } from 'vitest';
import { shouldWaitForInputChipGate } from '../src/engine/layers/gpt2InputChipGateUtils.js';

describe('gpt2InputChipGateUtils.shouldWaitForInputChipGate', () => {
    it('waits while a gate is still pending before its fallback time', () => {
        const gate = {
            enabled: true,
            pending: true,
            pendingFallbackAt: 5000
        };

        expect(shouldWaitForInputChipGate(gate, 0, 4999)).toBe(true);
    });

    it('releases a pending gate once its fallback time has elapsed', () => {
        const gate = {
            enabled: true,
            pending: true,
            pendingFallbackAt: 5000
        };

        expect(shouldWaitForInputChipGate(gate, 0, 5000)).toBe(false);
    });

    it('waits for per-token completion before the scheduled release time', () => {
        const gate = {
            enabled: true,
            pending: false,
            releaseByToken: { '2': 1800 },
            defaultReleaseAt: 2500,
            insideByToken: { '2': false }
        };

        expect(shouldWaitForInputChipGate(gate, 2, 1799)).toBe(true);
    });

    it('releases a stalled token gate once its per-token release time passes', () => {
        const gate = {
            enabled: true,
            pending: false,
            releaseByToken: { '2': 1800 },
            defaultReleaseAt: 2500,
            insideByToken: { '2': false }
        };

        expect(shouldWaitForInputChipGate(gate, 2, 1800)).toBe(false);
    });

    it('uses the default release time when no token-specific entry exists', () => {
        const gate = {
            enabled: true,
            pending: false,
            defaultReleaseAt: 2500,
            insideByToken: Object.create(null)
        };

        expect(shouldWaitForInputChipGate(gate, 7, 2000)).toBe(true);
        expect(shouldWaitForInputChipGate(gate, 7, 2500)).toBe(false);
    });

    it('releases immediately once the token is marked inside the matrix', () => {
        const gate = {
            enabled: true,
            pending: false,
            releaseByToken: { '1': 4000 },
            defaultReleaseAt: 4000,
            insideByToken: { '1': true }
        };

        expect(shouldWaitForInputChipGate(gate, 1, 1000)).toBe(false);
    });

    it('can keep waiting after the token is inside until its release time', () => {
        const gate = {
            enabled: true,
            pending: false,
            waitForReleaseAfterInside: true,
            releaseByToken: { '1': 4000 },
            defaultReleaseAt: 4000,
            insideByToken: { '1': true }
        };

        expect(shouldWaitForInputChipGate(gate, 1, 3999)).toBe(true);
        expect(shouldWaitForInputChipGate(gate, 1, 4000)).toBe(false);
    });
});
