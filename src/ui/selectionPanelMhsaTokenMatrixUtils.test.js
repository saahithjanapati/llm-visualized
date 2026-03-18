import * as THREE from 'three';
import { describe, expect, it } from 'vitest';

import { buildMhsaTokenMatrixPreviewData } from './selectionPanelMhsaTokenMatrixUtils.js';
import {
    MHA_FINAL_K_COLOR,
    MHA_FINAL_Q_COLOR,
    MHA_FINAL_V_COLOR
} from '../animations/LayerAnimationConstants.js';

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

function extractGradientHexColors(gradientCss = '') {
    const matches = String(gradientCss || '').match(/#([0-9a-f]{6})/gi) || [];
    return matches.map((token) => Number.parseInt(token.slice(1), 16));
}

function getHueDistance(colorHex, baseHex) {
    const sampleHsl = { h: 0, s: 0, l: 0 };
    const baseHsl = { h: 0, s: 0, l: 0 };
    new THREE.Color(colorHex).getHSL(sampleHsl);
    new THREE.Color(baseHex).getHSL(baseHsl);
    const rawDelta = Math.abs(sampleHsl.h - baseHsl.h);
    return Math.min(rawDelta, 1 - rawDelta);
}

function createExtremeFamilyActivationSource() {
    const tokenLabels = ['Alpha'];
    const qValues = [-2, -1.5, -1, -0.5, -0.25, 0, 0.25, 0.5, 1, 1.5, 2, -2];
    const kValues = [2, 1.5, 1, 0.5, 0.25, 0, -0.25, -0.5, -1, -1.5, -2, 2];
    const vValues = [-2, -1, -0.5, 0, 0.5, 1, 2, 1, 0.5, 0, -0.5, -1];
    const weightedValues = [2, 1.5, 1, 0.5, 0.25, 0, -0.25, -0.5, -1, -1.5, -2, 2];
    const projectionVectors = {
        q: qValues,
        k: kValues,
        v: vValues
    };

    return {
        meta: {
            prompt_tokens: [0],
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
        getLayerQKVScalar(layerIndex, kind, headIndex) {
            expect(layerIndex).toBe(0);
            expect(headIndex).toBe(0);
            const values = projectionVectors[kind] || [];
            return values[0] ?? 0;
        },
        getLayerQKVVector(layerIndex, kind, headIndex, tokenIndex, targetLength) {
            expect(layerIndex).toBe(0);
            expect(headIndex).toBe(0);
            expect(tokenIndex).toBe(0);
            const source = projectionVectors[kind] || [];
            return Array.from({ length: targetLength }, (_, index) => source[index % source.length] ?? 0);
        },
        getAttentionScoresRow(layerIndex, mode, headIndex, tokenIndex) {
            expect(layerIndex).toBe(0);
            expect(headIndex).toBe(0);
            expect(tokenIndex).toBe(0);
            return mode === 'pre' ? [0.4] : [1];
        },
        getAttentionWeightedSum(layerIndex, headIndex, tokenIndex, targetLength) {
            expect(layerIndex).toBe(0);
            expect(headIndex).toBe(0);
            expect(tokenIndex).toBe(0);
            return Array.from({ length: targetLength }, (_, index) => weightedValues[index % weightedValues.length] ?? 0);
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

    it('keeps Q, K, and V detail gradients locked to their family hues', () => {
        const previewData = buildMhsaTokenMatrixPreviewData({
            activationSource: createExtremeFamilyActivationSource(),
            layerIndex: 0,
            headIndex: 0,
            tokenIndices: [0]
        });

        const queryProjection = previewData?.projections?.find((entry) => entry?.kind === 'Q') || null;
        const keyProjection = previewData?.projections?.find((entry) => entry?.kind === 'K') || null;
        const valueProjection = previewData?.projections?.find((entry) => entry?.kind === 'V') || null;
        const valueOperandRow = previewData?.attentionScoreStage?.valueRows?.[0] || null;
        const headOutputRow = previewData?.attentionScoreStage?.headOutputRows?.[0] || null;

        const expectations = [
            { gradientCss: queryProjection?.outputRows?.[0]?.gradientCss, baseHex: MHA_FINAL_Q_COLOR },
            { gradientCss: keyProjection?.outputRows?.[0]?.gradientCss, baseHex: MHA_FINAL_K_COLOR },
            { gradientCss: valueProjection?.outputRows?.[0]?.gradientCss, baseHex: MHA_FINAL_V_COLOR },
            { gradientCss: valueOperandRow?.gradientCss, baseHex: MHA_FINAL_V_COLOR },
            { gradientCss: headOutputRow?.gradientCss, baseHex: MHA_FINAL_V_COLOR }
        ];

        expectations.forEach(({ gradientCss, baseHex }) => {
            const colors = extractGradientHexColors(gradientCss);
            expect(colors.length).toBeGreaterThan(0);
            colors.forEach((colorHex) => {
                expect(getHueDistance(colorHex, baseHex)).toBeLessThan(0.02);
            });
        });

        const weightExpectations = [
            { gradientCss: queryProjection?.weightGradientCss, baseHex: MHA_FINAL_Q_COLOR },
            { gradientCss: keyProjection?.weightGradientCss, baseHex: MHA_FINAL_K_COLOR },
            { gradientCss: valueProjection?.weightGradientCss, baseHex: MHA_FINAL_V_COLOR }
        ];

        weightExpectations.forEach(({ gradientCss, baseHex }) => {
            const colors = extractGradientHexColors(gradientCss);
            expect(colors.length).toBeGreaterThan(0);
            expect(String(gradientCss || '')).not.toMatch(/rgba?\(/i);
            colors.forEach((colorHex) => {
                expect(getHueDistance(colorHex, baseHex)).toBeLessThan(0.02);
            });
        });
    });
});
