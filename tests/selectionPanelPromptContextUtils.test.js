import { describe, expect, it } from 'vitest';
import { resolveAdjacentLogitTokenSeeds, resolveLogitTokenSeed } from '../src/app/gpt-tower/logitColor.js';
import { buildSelectionPromptContext } from '../src/ui/selectionPanelPromptContextUtils.js';

describe('selectionPanelPromptContextUtils', () => {
    it('keeps selected prompt chips on their canonical token color', () => {
        const activationSource = {
            getTokenId(tokenIndex) {
                return [0, 8][tokenIndex] ?? null;
            }
        };

        const result = buildSelectionPromptContext({
            activationSource,
            laneTokenIndices: [0, 1],
            tokenLabels: ['alpha', 'beta'],
            selectedTokenIndex: 1,
            selectedTokenId: 8,
            selectedTokenText: 'beta'
        });

        const canonicalSeed = resolveLogitTokenSeed({ token_id: 8, token: 'beta' }, 1);
        const legacyAdjacentSeed = resolveAdjacentLogitTokenSeeds([
            { tokenId: 0, tokenLabel: 'alpha' },
            { tokenId: 8, tokenLabel: 'beta' }
        ])[1];

        expect(result.activeIndex).toBe(1);
        expect(result.entries[1]?.seed).toBe(canonicalSeed);
        expect(legacyAdjacentSeed).not.toBe(canonicalSeed);
    });

    it('does not mark a trailing space token as selected when no token is selected', () => {
        const activationSource = {
            getTokenId(tokenIndex) {
                return 1000 + tokenIndex;
            }
        };

        const result = buildSelectionPromptContext({
            activationSource,
            laneTokenIndices: [0, 1, 2, 3, 4],
            tokenLabels: ['Can', ' machines', ' think', '?', ' '],
            selectedTokenIndex: null,
            selectedTokenId: null,
            selectedTokenText: ''
        });

        expect(result.entries.map((entry) => entry.tokenLabel)).toEqual([
            'Can',
            ' machines',
            ' think',
            '?',
            '" "'
        ]);
        expect(result.activeIndex).toBe(-1);
    });
});
