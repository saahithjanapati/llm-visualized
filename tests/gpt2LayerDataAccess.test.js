import { describe, expect, it, vi } from 'vitest';
import {
    getLn1DataForLane,
    getMlpUpDataForLane,
    resolveActiveLaneLayoutIndices,
    resolveBaseVectorLength,
    resolveInstanceCountFromData,
    resolveLaneLayoutCount,
    resolveTokenIndexForLane
} from '../src/engine/layers/gpt2LayerDataAccess.js';

describe('gpt2LayerDataAccess helpers', () => {
    it('resolves base vector length with fallback', () => {
        expect(resolveBaseVectorLength({ _baseVectorLength: 256 }, 64)).toBe(256);
        expect(resolveBaseVectorLength({}, 64)).toBe(64);
        expect(resolveBaseVectorLength(null, 0)).toBe(1);
    });

    it('resolves lane layout count and active indices', () => {
        const layer = { _laneCount: 4, _laneLayoutCount: 6, _activeLaneLayoutIndices: [2, 3, 4, 5, 6] };
        expect(resolveLaneLayoutCount(layer)).toBe(6);
        expect(resolveActiveLaneLayoutIndices(layer)).toEqual([2, 3, 4, 5]);

        const fallbackLayer = { _laneCount: 3 };
        expect(resolveActiveLaneLayoutIndices(fallbackLayer)).toEqual([0, 1, 2]);
    });

    it('resolves instance count from data length when available', () => {
        const layer = { _baseVectorLength: 64 };
        expect(resolveInstanceCountFromData(layer, [1, 2, 3], null, 64)).toBe(3);
        expect(resolveInstanceCountFromData(layer, new Float32Array([1, 2]), null, 64)).toBe(2);
        expect(resolveInstanceCountFromData(layer, null, null, 64)).toBe(64);
        expect(resolveInstanceCountFromData(layer, null, 7, 64)).toBe(7);
    });

    it('prefers configured lane token indices and falls back to activation source mapping', () => {
        const directLayer = {
            _passLaneTokenIndices: [10, 11, 12],
            _laneCount: 3,
            _laneLayoutCount: 4
        };
        expect(resolveTokenIndexForLane(directLayer, 1, 2)).toBe(11);

        const getLaneTokenIndex = vi.fn(() => 42);
        const mappedLayer = {
            _laneCount: 3,
            _laneLayoutCount: 5,
            activationSource: { getLaneTokenIndex }
        };
        expect(resolveTokenIndexForLane(mappedLayer, 1, 4)).toBe(42);
        expect(getLaneTokenIndex).toHaveBeenCalledWith(4, 5);
    });

    it('passes expanded target length for MLP up data', () => {
        const getMlpUp = vi.fn(() => [0.1, 0.2]);
        const layer = {
            index: 2,
            _baseVectorLength: 32,
            activationSource: { getMlpUp }
        };
        const lane = { tokenIndex: 7 };

        const result = getMlpUpDataForLane(layer, lane, 32);
        expect(result).toEqual([0.1, 0.2]);
        expect(getMlpUp).toHaveBeenCalledWith(2, 7, 128);
    });

    it('requests ln1 data with expected index, stage, token, and length', () => {
        const getLayerLn1 = vi.fn(() => [1, 2, 3]);
        const layer = {
            index: 5,
            _baseVectorLength: 48,
            activationSource: { getLayerLn1 }
        };
        const lane = { tokenIndex: 9 };

        const result = getLn1DataForLane(layer, lane, 'normed', 48);
        expect(result).toEqual([1, 2, 3]);
        expect(getLayerLn1).toHaveBeenCalledWith(5, 'normed', 9, 48);
    });
});
