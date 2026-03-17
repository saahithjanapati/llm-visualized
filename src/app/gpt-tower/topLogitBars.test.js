import * as THREE from 'three';
import { beforeEach, describe, expect, it } from 'vitest';
import { getLogitTokenChipColorHex, resolveLogitTokenChipColorKey } from './logitColor.js';
import { addTopLogitBars, resolveChosenLogitDisplayColorKey } from './topLogitBars.js';
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

    it('colors the chosen-token prism with the same chip color as the chosen token chip', () => {
        const laneTokenIndices = [0];
        const visibleTokenLabels = ['Alpha', 'Beta'];
        const visibleTokenIds = [101, 202];
        const activationSource = {
            getTokenCount() {
                return visibleTokenLabels.length;
            },
            getTokenString(tokenIndex) {
                return visibleTokenLabels[tokenIndex] ?? '';
            },
            getTokenId(tokenIndex) {
                return visibleTokenIds[tokenIndex] ?? null;
            },
            getLogitTopK() {
                return 2;
            },
            getLogitsForToken(tokenIndex, limit = null) {
                if (tokenIndex !== 0) return [];
                const row = [
                    { token_id: 303, token: 'Omega', prob: 0.7 },
                    { token_id: 202, token: 'Beta', prob: 0.2 }
                ];
                return limit == null ? row : row.slice(0, limit);
            }
        };

        const visibleEntries = buildPromptTokenChipEntries({
            tokenLabels: [visibleTokenLabels[0]],
            tokenIndices: laneTokenIndices,
            tokenIds: [visibleTokenIds[0]]
        });
        setActivePromptTokenChipEntries(visibleEntries);

        const barGroup = addTopLogitBars({
            activationSource,
            laneTokenIndices,
            laneZs: [0],
            vocabCenter: new THREE.Vector3(0, 0, 0),
            scene: new THREE.Group(),
            engine: null
        });

        expect(barGroup).toBeTruthy();
        const chosenEntry = barGroup.userData.chosenEntries[0];
        expect(chosenEntry?.instanceIndex).toBe(1);
        expect(chosenEntry?.useChipColor).toBe(true);

        const actualColor = new THREE.Color();
        barGroup.userData.instancedMesh.getColorAt(chosenEntry.instanceIndex, actualColor);

        const expectedColorKey = resolveChosenLogitDisplayColorKey({
            activationSource,
            laneTokenIndices,
            chosenEntry: chosenEntry.entry,
            chosenTokenIndex: chosenEntry.tokenIndex,
            fallbackIndex: chosenEntry.fallbackIndex
        });
        const expectedColor = new THREE.Color(getLogitTokenChipColorHex(expectedColorKey));

        expect(actualColor.r).toBeCloseTo(expectedColor.r, 5);
        expect(actualColor.g).toBeCloseTo(expectedColor.g, 5);
        expect(actualColor.b).toBeCloseTo(expectedColor.b, 5);
    });
});
