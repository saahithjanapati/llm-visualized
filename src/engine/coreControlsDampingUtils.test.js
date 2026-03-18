import { describe, expect, it } from 'vitest';

import { resolveFrameRateIndependentDamping } from './coreControlsDampingUtils.js';

describe('coreControlsDampingUtils', () => {
    it('preserves the configured damping factor at the reference frame step', () => {
        expect(resolveFrameRateIndependentDamping(0.08, 1 / 60)).toBeCloseTo(0.08, 8);
    });

    it('increases the effective damping on longer frames', () => {
        const shortFrameDamping = resolveFrameRateIndependentDamping(0.08, 1 / 60);
        const longFrameDamping = resolveFrameRateIndependentDamping(0.08, 1 / 30);

        expect(longFrameDamping).toBeGreaterThan(shortFrameDamping);
        expect(longFrameDamping).toBeCloseTo(0.1536, 4);
    });

    it('returns no effective damping when no time has elapsed', () => {
        expect(resolveFrameRateIndependentDamping(0.08, 0)).toBe(0);
    });
});
