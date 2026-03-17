import { describe, expect, it } from 'vitest';

import { computeMlpMatrixPassEmissive } from './gpt2MlpMatrixVisualUtils.js';

describe('computeMlpMatrixPassEmissive', () => {
    const startIntensity = 0.04;
    const peakIntensity = 0.78;
    const finalIntensity = 0.22;

    function run(progress) {
        return computeMlpMatrixPassEmissive(
            progress,
            startIntensity,
            peakIntensity,
            finalIntensity
        );
    }

    it('starts at the resting intensity and settles at the post-pass intensity', () => {
        expect(run(0)).toBeCloseTo(startIntensity, 6);
        expect(run(1)).toBeCloseTo(finalIntensity, 6);
    });

    it('reaches a clear peak before transitioning into a longer fade-out', () => {
        const early = run(0.2);
        const peak = run(0.68);
        const late = run(0.85);

        expect(early).toBeGreaterThan(startIntensity);
        expect(peak).toBeCloseTo(peakIntensity, 6);
        expect(late).toBeLessThan(peak);
        expect(late).toBeGreaterThan(finalIntensity);
    });

    it('continues dimming after the peak instead of holding near maximum until the end', () => {
        const midFade = run(0.78);
        const lateFade = run(0.92);

        expect(lateFade).toBeLessThan(midFade);
        expect(midFade).toBeLessThan(peakIntensity);
    });
});
