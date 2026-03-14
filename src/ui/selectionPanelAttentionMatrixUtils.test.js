import { describe, expect, it } from 'vitest';

import {
    buildAttentionMatrixValues,
    resolveAttentionMatrixCellValue,
    shouldMuteCausalUpperPreAttentionCell
} from './selectionPanelAttentionMatrixUtils.js';

describe('selectionPanelAttentionMatrixUtils', () => {
    it('fills post-softmax causal upper-triangle cells with zero while preserving saved pre-softmax values', () => {
        const activationSource = {
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
            }
        };

        const preValues = buildAttentionMatrixValues({
            activationSource,
            layerIndex: 0,
            headIndex: 0,
            tokenIndices: [0, 1, 2],
            mode: 'pre'
        });
        const postValues = buildAttentionMatrixValues({
            activationSource,
            layerIndex: 0,
            headIndex: 0,
            tokenIndices: [0, 1, 2],
            mode: 'post'
        });

        expect(preValues).toEqual([
            [0.4, -0.2, 0.7],
            [0.1, 0.6, -0.3],
            [-0.5, 0.2, 0.9]
        ]);
        expect(postValues).toEqual([
            [1, 0, 0],
            [0.3, 0.7, 0],
            [0.2, 0.3, 0.5]
        ]);
    });

    it('marks only pre-softmax causal upper-triangle cells for default muting', () => {
        expect(shouldMuteCausalUpperPreAttentionCell({
            mode: 'pre',
            queryTokenIndex: 0,
            keyTokenIndex: 2
        })).toBe(true);
        expect(shouldMuteCausalUpperPreAttentionCell({
            mode: 'pre',
            queryTokenIndex: 2,
            keyTokenIndex: 0
        })).toBe(false);
        expect(shouldMuteCausalUpperPreAttentionCell({
            mode: 'post',
            queryTokenIndex: 0,
            keyTokenIndex: 2
        })).toBe(false);
        expect(resolveAttentionMatrixCellValue({
            mode: 'post',
            value: null,
            queryTokenIndex: 0,
            keyTokenIndex: 2
        })).toBe(0);
    });
});
