import { describe, expect, it } from 'vitest';

import {
    resolveKvCachePassMode,
    resolveKvPrefillBaseLaneCount
} from './kvCachePassMode.js';

describe('kvCachePassMode', () => {
    it('uses the prompt/base lane count as the KV prefill baseline', () => {
        expect(resolveKvPrefillBaseLaneCount({
            initialLaneCount: 8,
            baseLaneCount: 6
        })).toBe(6);
    });

    it('treats the full prompt/base window as prefill', () => {
        const passMode = resolveKvCachePassMode({
            laneCount: 6,
            kvModeEnabled: true,
            prefillBaseLaneCount: 6
        });

        expect(passMode).toMatchObject({
            totalLaneCount: 6,
            passIndex: 0,
            kvCachePrefillActive: true,
            kvCacheDecodeActive: false,
            activeLaneCount: 6
        });
    });

    it('treats later token windows as decode relative to the prompt/base window', () => {
        const passMode = resolveKvCachePassMode({
            laneCount: 7,
            kvModeEnabled: true,
            prefillBaseLaneCount: 6
        });

        expect(passMode).toMatchObject({
            totalLaneCount: 7,
            passIndex: 1,
            kvCachePrefillActive: false,
            kvCacheDecodeActive: true,
            activeLaneCount: 1
        });
    });
});
