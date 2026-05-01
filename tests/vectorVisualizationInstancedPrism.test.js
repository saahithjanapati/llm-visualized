import { afterAll, beforeAll, describe, expect, it, vi } from 'vitest';

let VectorVisualizationInstancedPrism;

function createVectorHelper() {
    return Object.create(VectorVisualizationInstancedPrism.prototype);
}

describe('VectorVisualizationInstancedPrism normalization helpers', () => {
    beforeAll(async () => {
        vi.stubGlobal('localStorage', {
            getItem: vi.fn(() => null),
            setItem: vi.fn(),
            removeItem: vi.fn()
        });
        ({ VectorVisualizationInstancedPrism } = await import('../src/components/VectorVisualizationInstancedPrism.js'));
    });

    afterAll(() => {
        vi.unstubAllGlobals();
    });

    it('normalizes finite values in one pass while preserving neutral fallbacks', () => {
        const vector = createVectorHelper();

        expect(vector.minMaxNormalize([2, 4, Number.NaN, 6, Infinity])).toEqual([
            0,
            0.5,
            0.5,
            1,
            0.5
        ]);
    });

    it('returns ordinary arrays for typed activation data', () => {
        const vector = createVectorHelper();
        const normalized = vector.layerNormalize(new Float32Array([5, 10, 15]));

        expect(Array.isArray(normalized)).toBe(true);
        expect(normalized).toEqual([0, 0.5, 1]);
    });
});
