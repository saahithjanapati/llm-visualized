import { describe, expect, it } from 'vitest';
import {
    buildGeluWaveField,
    computeGeluWhipOffset
} from '../src/engine/layers/gpt2GeluMotionUtils.js';

describe('gpt2GeluMotionUtils', () => {
    it('builds phase offsets and normalized indices across the full expanded vector', () => {
        const { phaseOffsets, normalizedIndices, totalCount } = buildGeluWaveField([3, 2], 1.25);

        expect(totalCount).toBe(5);
        expect(phaseOffsets[0][0]).toBeCloseTo(0);
        expect(normalizedIndices[0][0]).toBeCloseTo(0);
        expect(phaseOffsets[1][1]).toBeCloseTo(Math.PI * 2 * 1.25);
        expect(normalizedIndices[1][1]).toBeCloseTo(1);
    });

    it('stays quiescent at the endpoints and hits harder near the tip', () => {
        expect(computeGeluWhipOffset({
            phaseOffset: Math.PI * 0.5,
            normalizedIndex: 1,
            progress: 0,
            waveHeight: 1,
            waveTravelCycles: 0
        })).toBeCloseTo(0);

        expect(computeGeluWhipOffset({
            phaseOffset: Math.PI * 0.5,
            normalizedIndex: 1,
            progress: 1,
            waveHeight: 1,
            waveTravelCycles: 0
        })).toBeCloseTo(0);

        const rootOffset = computeGeluWhipOffset({
            phaseOffset: Math.PI * 0.5,
            normalizedIndex: 0,
            progress: 0.5,
            waveHeight: 1,
            waveTravelCycles: 0
        });
        const tipOffset = computeGeluWhipOffset({
            phaseOffset: Math.PI * 0.5,
            normalizedIndex: 1,
            progress: 0.5,
            waveHeight: 1,
            waveTravelCycles: 0
        });

        expect(Math.abs(tipOffset)).toBeGreaterThan(Math.abs(rootOffset));
    });

    it('produces both positive and negative offsets across a fuller sinusoid', () => {
        const { phaseOffsets, normalizedIndices } = buildGeluWaveField([6], 1.15);
        const samples = Array.from(phaseOffsets[0], (phaseOffset, index) => computeGeluWhipOffset({
            phaseOffset,
            normalizedIndex: normalizedIndices[0][index],
            progress: 0.5,
            waveHeight: 1,
            waveTravelCycles: 0.2
        }));

        expect(samples.some((value) => value > 0)).toBe(true);
        expect(samples.some((value) => value < 0)).toBe(true);
    });
});
