import { describe, expect, it } from 'vitest';
import { computeMlpMatrixPassEmissive } from '../src/engine/layers/gpt2MlpMatrixVisualUtils.js';

describe('gpt2MlpMatrixVisualUtils', () => {
    it('starts at the flash start intensity and settles to the final intensity', () => {
        expect(computeMlpMatrixPassEmissive(0, 0.04, 0.78, 0.22)).toBeCloseTo(0.04);
        expect(computeMlpMatrixPassEmissive(1, 0.04, 0.78, 0.22)).toBeCloseTo(0.22);
    });

    it('keeps growing across the body of the pass instead of plateauing early', () => {
        const samples = [
            computeMlpMatrixPassEmissive(0.1, 0.04, 0.78, 0.22),
            computeMlpMatrixPassEmissive(0.35, 0.04, 0.78, 0.22),
            computeMlpMatrixPassEmissive(0.65, 0.04, 0.78, 0.22),
            computeMlpMatrixPassEmissive(0.9, 0.04, 0.78, 0.22)
        ];

        expect(samples[1]).toBeGreaterThan(samples[0]);
        expect(samples[2]).toBeGreaterThan(samples[1]);
        expect(samples[3]).toBeGreaterThan(samples[2]);
    });

    it('stays above the final resting emissive right until the settle tail', () => {
        const latePass = computeMlpMatrixPassEmissive(0.95, 0.04, 0.78, 0.22);

        expect(latePass).toBeGreaterThan(0.22);
        expect(latePass).toBeLessThanOrEqual(0.78);
    });
});
