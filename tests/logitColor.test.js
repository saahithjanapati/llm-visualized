import { describe, expect, it } from 'vitest';
import {
    getLogitTokenChipColorCss,
    resolveAdjacentLogitTokenChipColorKeys,
    resolveLogitTokenChipColorKey
} from '../src/app/gpt-tower/logitColor.js';

function findCollidingTokenIds() {
    const seenByColorKey = new Map();
    for (let tokenId = 0; tokenId < 4096; tokenId += 1) {
        const colorKey = resolveLogitTokenChipColorKey({ tokenId, tokenLabel: `token-${tokenId}` }, tokenId);
        const previousTokenId = seenByColorKey.get(colorKey);
        if (previousTokenId !== undefined) {
            return [previousTokenId, tokenId];
        }
        seenByColorKey.set(colorKey, tokenId);
    }
    throw new Error('Expected to find at least one prompt-chip token-id collision.');
}

describe('logitColor prompt chip palette', () => {
    it('separates adjacent distinct tokens when they collide onto the same canonical color', () => {
        const [leftTokenId, rightTokenId] = findCollidingTokenIds();
        const leftCanonicalColorKey = resolveLogitTokenChipColorKey({
            tokenId: leftTokenId,
            tokenLabel: `token-${leftTokenId}`
        }, 0);
        const rightCanonicalColorKey = resolveLogitTokenChipColorKey({
            tokenId: rightTokenId,
            tokenLabel: `token-${rightTokenId}`
        }, 1);

        expect(leftCanonicalColorKey).toBe(rightCanonicalColorKey);

        const entries = [
            { tokenId: leftTokenId, tokenLabel: `token-${leftTokenId}` },
            { tokenId: rightTokenId, tokenLabel: `token-${rightTokenId}` }
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

    it('prefers token identity over a temporary seed for prompt chip colors', () => {
        const identityColorKey = resolveLogitTokenChipColorKey({
            tokenId: 0,
            tokenLabel: 'beta'
        }, 1);
        const seededColorKey = resolveLogitTokenChipColorKey({
            tokenId: 0,
            tokenLabel: 'beta',
            seed: 1
        }, 1);

        expect(seededColorKey).toBe(identityColorKey);
    });

    it('formats prompt chip colors from the fixed neon palette', () => {
        expect(getLogitTokenChipColorCss(0, 0.92)).toBe('rgb(255 88 171 / 0.920)');
        expect(getLogitTokenChipColorCss(12, 0.2)).toBe('rgb(255 88 171 / 0.200)');
    });
});
