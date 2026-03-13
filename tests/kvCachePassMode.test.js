import { describe, expect, it } from 'vitest';

import {
    resolveKvCachePassMode,
    resolveKvPrefillBaseLaneCount
} from '../src/app/gpt-tower/kvCachePassMode.js';

describe('kvCachePassMode', () => {
    it('treats only the very first token as a KV prefill pass', () => {
        const prefillBaseLaneCount = resolveKvPrefillBaseLaneCount({ initialLaneCount: 5 });
        const passMode = resolveKvCachePassMode({
            laneCount: 5,
            kvModeEnabled: true,
            prefillBaseLaneCount
        });

        expect(prefillBaseLaneCount).toBe(1);
        expect(passMode).toMatchObject({
            totalLaneCount: 5,
            passIndex: 4,
            kvCachePrefillActive: false,
            kvCacheDecodeActive: true,
            activeLaneCount: 1
        });
    });

    it('keeps token 1 as the only KV prefill case', () => {
        const prefillBaseLaneCount = resolveKvPrefillBaseLaneCount({ initialLaneCount: 1 });
        const passMode = resolveKvCachePassMode({
            laneCount: 1,
            kvModeEnabled: true,
            prefillBaseLaneCount
        });

        expect(passMode).toMatchObject({
            totalLaneCount: 1,
            passIndex: 0,
            kvCachePrefillActive: true,
            kvCacheDecodeActive: false,
            activeLaneCount: 1
        });
    });
});
