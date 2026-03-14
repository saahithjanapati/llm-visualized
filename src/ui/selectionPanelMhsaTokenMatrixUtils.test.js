import { describe, expect, it } from 'vitest';

import { buildMhsaTokenMatrixPreviewData } from './selectionPanelMhsaTokenMatrixUtils.js';

function createActivationSource() {
    const tokenLabels = ['Alpha', 'Beta', 'Gamma'];
    return {
        meta: {
            prompt_tokens: [0, 1, 2],
            token_display_strings: tokenLabels
        },
        getTokenCount() {
            return tokenLabels.length;
        },
        getTokenString(tokenIndex) {
            return tokenLabels[tokenIndex] || `Token ${tokenIndex + 1}`;
        },
        getLayerLn1() {
            return new Array(12).fill(0.1);
        },
        getLayerQKVScalar(layerIndex, kind, headIndex, tokenIndex) {
            expect(layerIndex).toBe(0);
            expect(headIndex).toBe(0);
            return tokenIndex + (kind === 'q' ? 0.1 : (kind === 'k' ? 0.2 : 0.3));
        },
        getLayerQKVVector(layerIndex, kind, headIndex, tokenIndex, targetLength) {
            expect(layerIndex).toBe(0);
            expect(headIndex).toBe(0);
            const base = tokenIndex + (kind === 'q' ? 0.1 : (kind === 'k' ? 0.2 : 0.3));
            return new Array(targetLength).fill(base);
        },
        getAttentionScoresRow(layerIndex, mode, headIndex, tokenIndex) {
            expect(layerIndex).toBe(0);
            expect(headIndex).toBe(0);
            if (mode === 'pre') {
                return [
                    [0.4, -0.2, 0.7],
                    [0.1, 0.6, -0.3],
                    [-0.5, 0.2, 0.9]
                ][tokenIndex] || null;
            }
            return [
                [1],
                [0.3, 0.7],
                [0.2, 0.3, 0.5]
            ][tokenIndex] || null;
        },
        getAttentionWeightedSum(layerIndex, headIndex, tokenIndex, targetLength) {
            expect(layerIndex).toBe(0);
            expect(headIndex).toBe(0);
            return new Array(targetLength).fill(tokenIndex + 0.4);
        }
    };
}

describe('buildMhsaTokenMatrixPreviewData', () => {
    it('keeps saved pre-softmax upper cells muted by default and renders post-softmax upper cells as zero', () => {
        const previewData = buildMhsaTokenMatrixPreviewData({
            activationSource: createActivationSource(),
            layerIndex: 0,
            headIndex: 0,
            tokenIndices: [0, 1, 2]
        });

        const scoreStage = previewData?.attentionScoreStage;
        expect(scoreStage).toBeTruthy();

        const upperPreCell = scoreStage.outputRows[0].cells[2];
        expect(upperPreCell.preScore).toBe(0.7);
        expect(upperPreCell.isMasked).toBe(true);
        expect(upperPreCell.defaultMuted).toBe(true);

        const upperPostCell = scoreStage.postRows[0].cells[2];
        expect(upperPostCell.postScore).toBe(0);
        expect(upperPostCell.rawValue).toBe(0);
        expect(upperPostCell.isEmpty).toBe(false);
        expect(upperPostCell.useMaskedStyle).toBe(false);
        expect(upperPostCell.fillCss).not.toBe('rgba(0, 0, 0, 0.94)');
    });
});
