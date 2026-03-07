import { describe, expect, it } from 'vitest';
import {
    getLogitTokenChipColorCss,
    resolveAdjacentLogitTokenChipColorKeys,
    resolveLogitTokenChipColorKey
} from '../src/app/gpt-tower/logitColor.js';

function findCollidingSeeds() {
    const seenByColorKey = new Map();
    for (let seed = 0; seed < 4096; seed += 1) {
        const colorKey = resolveLogitTokenChipColorKey({ seed }, seed);
        const previousSeed = seenByColorKey.get(colorKey);
        if (previousSeed !== undefined) {
            return [previousSeed, seed];
        }
        seenByColorKey.set(colorKey, seed);
    }
    throw new Error('Expected to find at least one prompt-chip color collision.');
}

describe('logitColor prompt chip palette', () => {
    it('separates adjacent distinct tokens when they collide onto the same canonical color', () => {
        const [leftSeed, rightSeed] = findCollidingSeeds();
        const leftCanonicalColorKey = resolveLogitTokenChipColorKey({ seed: leftSeed }, 0);
        const rightCanonicalColorKey = resolveLogitTokenChipColorKey({ seed: rightSeed }, 1);

        expect(leftCanonicalColorKey).toBe(rightCanonicalColorKey);

        const entries = [
            { seed: leftSeed, tokenId: 11, tokenLabel: 'alpha' },
            { seed: rightSeed, tokenId: 17, tokenLabel: 'beta' }
        ];
        const resolvedOnce = resolveAdjacentLogitTokenChipColorKeys(entries);
        const resolvedTwice = resolveAdjacentLogitTokenChipColorKeys(entries);

        expect(resolvedOnce[0]).toBe(leftCanonicalColorKey);
        expect(resolvedOnce[1]).not.toBe(resolvedOnce[0]);
        expect(resolvedTwice).toEqual(resolvedOnce);
    });

    it('keeps adjacent identical token ids on the same color', () => {
        const entries = [
            { tokenId: 42, tokenLabel: 'same' },
            { tokenId: 42, tokenLabel: 'same' }
        ];
        const canonicalColorKey = resolveLogitTokenChipColorKey(entries[0], 0);
        const resolved = resolveAdjacentLogitTokenChipColorKeys(entries);

        expect(resolved).toEqual([canonicalColorKey, canonicalColorKey]);
    });

    it('formats prompt chip colors from the fixed neon palette', () => {
        expect(getLogitTokenChipColorCss(0, 0.92)).toBe('rgb(255 88 171 / 0.920)');
        expect(getLogitTokenChipColorCss(12, 0.2)).toBe('rgb(255 88 171 / 0.200)');
    });
});
