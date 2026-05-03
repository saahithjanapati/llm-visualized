import * as THREE from 'three';
import { beforeEach, describe, expect, it } from 'vitest';

import {
    getLogitTokenChipColorHex,
    getLogitTokenColorUnit,
    resolveLogitTokenChipColorKey,
    resolveLogitTokenSeed
} from './logitColor.js';
import {
    addTopLogitBars,
    buildTopLogitHoverLabel,
    resolveChosenLogitDisplayColorKey
} from './topLogitBars.js';
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
        const nonChosenColor = new THREE.Color();
        barGroup.userData.instancedMesh.getColorAt(0, nonChosenColor);

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

        const nonChosenSeed = resolveLogitTokenSeed(activationSource.getLogitsForToken(0)[0], 0);
        const brightNonChosenUnit = getLogitTokenColorUnit(nonChosenSeed);
        const brightNonChosenColor = new THREE.Color().setHSL(
            brightNonChosenUnit.h,
            brightNonChosenUnit.s,
            brightNonChosenUnit.l
        );
        const brightHsl = { h: 0, s: 0, l: 0 };
        const dimmedHsl = { h: 0, s: 0, l: 0 };
        brightNonChosenColor.getHSL(brightHsl);
        nonChosenColor.getHSL(dimmedHsl);

        expect(dimmedHsl.l).toBeLessThan(brightHsl.l * 0.5);
        expect(dimmedHsl.s).toBeGreaterThan(0.5);
    });
});

describe('buildTopLogitHoverLabel', () => {
    it('formats regular logit bars into separate tooltip rows', () => {
        expect(buildTopLogitHoverLabel({
            token: 'Beta',
            token_id: 9001,
            prob: 0.08
        })).toBe('Logit token: "Beta"\nID: 9001\nProbability: 8%');
    });

    it('keeps id and probability rows when token text is unavailable', () => {
        expect(buildTopLogitHoverLabel({
            token_id: 42,
            prob: 0.125
        })).toBe('Logit token\nID: 42\nProbability: 12.5%');
    });
});
