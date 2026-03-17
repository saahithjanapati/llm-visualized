import { beforeEach, describe, expect, it } from 'vitest';
import { resolveLogitTokenChipColorKey } from './logitColor.js';
import { resolveChosenLogitDisplayColorKey } from './topLogitBars.js';
import {
    buildPromptTokenChipEntries,
    resolvePromptTokenChipColorState,
    resolveTokenChipColorKey,
    setActivePromptTokenChipEntries
} from '../../ui/tokenChipColorUtils.js';

describe('resolveChosenLogitDisplayColorKey', () => {
    beforeEach(() => {
        setActivePromptTokenChipEntries([]);
    });

    it('matches the prompt token strip color context for the generated chosen token', () => {
        const laneTokenIndices = [4, 5];
        const visibleTokenLabels = ['Alpha', 'Beta'];
        const visibleTokenIds = [104, 105];
        const activationSource = {
            getTokenString(tokenIndex) {
                const index = laneTokenIndices.indexOf(tokenIndex);
                return index >= 0 ? visibleTokenLabels[index] : '';
            },
            getTokenId(tokenIndex) {
                const index = laneTokenIndices.indexOf(tokenIndex);
                return index >= 0 ? visibleTokenIds[index] : null;
            }
        };

        const visibleEntries = buildPromptTokenChipEntries({
            tokenLabels: visibleTokenLabels,
            tokenIndices: laneTokenIndices,
            tokenIds: visibleTokenIds
        });
        const visibleState = setActivePromptTokenChipEntries(visibleEntries);
        const previousVisibleEntry = {
            tokenIndex: laneTokenIndices[1],
            tokenId: visibleTokenIds[1],
            tokenLabel: visibleTokenLabels[1]
        };
        const previousColorKey = resolveTokenChipColorKey(previousVisibleEntry, 1, {
            lookup: visibleState.lookup
        });

        let generatedTokenId = null;
        let generatedTokenLabel = '';
        for (let candidate = 200; candidate < 4000; candidate += 1) {
            const candidateLabel = `Gamma ${candidate}`;
            if (resolveLogitTokenChipColorKey({
                tokenId: candidate,
                tokenLabel: candidateLabel
            }, 2) === previousColorKey) {
                generatedTokenId = candidate;
                generatedTokenLabel = candidateLabel;
                break;
            }
        }

        expect(generatedTokenId).not.toBeNull();

        const generatedToken = {
            tokenIndex: 6,
            tokenId: generatedTokenId,
            tokenLabel: generatedTokenLabel
        };
        const expectedEntries = buildPromptTokenChipEntries({
            tokenLabels: visibleTokenLabels,
            tokenIndices: laneTokenIndices,
            tokenIds: visibleTokenIds,
            generatedToken
        });
        const expectedState = resolvePromptTokenChipColorState(expectedEntries, {
            previousLookup: visibleState.lookup
        });
        const expectedColorKey = resolveTokenChipColorKey(generatedToken, 2, {
            lookup: expectedState.lookup
        });

        const actualColorKey = resolveChosenLogitDisplayColorKey({
            activationSource,
            laneTokenIndices,
            chosenEntry: {
                token_id: generatedTokenId,
                token: generatedTokenLabel
            },
            chosenTokenIndex: generatedToken.tokenIndex,
            fallbackIndex: 2
        });

        expect(actualColorKey).toBe(expectedColorKey);
        expect(actualColorKey).not.toBe(previousColorKey);
    });
});
