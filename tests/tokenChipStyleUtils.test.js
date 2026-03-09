// @vitest-environment jsdom
import { afterEach, describe, expect, it } from 'vitest';
import { resolveLogitTokenChipColorKey } from '../src/app/gpt-tower/logitColor.js';
import {
    buildPromptTokenChipEntries,
    setActivePromptTokenChipEntries,
    resolveTokenChipColors
} from '../src/ui/tokenChipColorUtils.js';
import { applyPromptTokenChipColors } from '../src/utils/tokenChipStyleUtils.js';

describe('tokenChipStyleUtils', () => {
    afterEach(() => {
        setActivePromptTokenChipEntries([]);
        document.body.innerHTML = '';
    });

    it('uses the active prompt-strip color mapping for hover-style token chips', () => {
        setActivePromptTokenChipEntries(buildPromptTokenChipEntries({
            tokenLabels: ['Can', ' machines', ' think', '?', ' '],
            tokenIndices: [0, 1, 2, 3, 4],
            tokenIds: [6090, 8217, 892, 30, 220]
        }));
        setActivePromptTokenChipEntries(buildPromptTokenChipEntries({
            tokenLabels: ['Can', ' machines', ' think', '?', ' '],
            tokenIndices: [0, 1, 2, 3, 4],
            tokenIds: [6090, 8217, 892, 30, 220],
            generatedToken: {
                tokenLabel: ' ',
                tokenIndex: 5,
                tokenId: 1849,
                seed: 1849
            }
        }));
        setActivePromptTokenChipEntries(buildPromptTokenChipEntries({
            tokenLabels: ['Can', ' machines', ' think', '?', ' ', ' '],
            tokenIndices: [0, 1, 2, 3, 4, 5],
            tokenIds: [6090, 8217, 892, 30, 220, 1849]
        }));

        const element = document.createElement('span');
        document.body.appendChild(element);

        const colorKey = applyPromptTokenChipColors(element, {
            tokenText: ' ',
            tokenIndex: 5,
            tokenId: 1849
        });
        const expected = resolveTokenChipColors({
            tokenLabel: ' ',
            tokenIndex: 5,
            tokenId: 1849
        }, 5);
        const canonical = resolveLogitTokenChipColorKey({
            tokenId: 1849,
            tokenLabel: ' '
        }, 5);

        expect(colorKey).toBe(expected.colorKey);
        expect(element.style.getPropertyValue('--token-color-border')).toBe(expected.border);
        expect(colorKey).not.toBe(canonical);
    });
});
