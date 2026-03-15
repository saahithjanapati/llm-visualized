import { describe, expect, it } from 'vitest';

import {
    CAUSAL_MASK_PREVIEW_BLOCKED_VALUE,
    buildAttentionMatrixValues,
    resolveAttentionMatrixCellValue,
    shouldClearPinnedAttentionOnDocumentPointerDown,
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

    it('builds a deterministic causal-mask preview grid without activation rows', () => {
        const maskValues = buildAttentionMatrixValues({
            tokenIndices: [0, 1, 2],
            mode: 'mask'
        });

        expect(maskValues).toEqual([
            [0, CAUSAL_MASK_PREVIEW_BLOCKED_VALUE, CAUSAL_MASK_PREVIEW_BLOCKED_VALUE],
            [0, 0, CAUSAL_MASK_PREVIEW_BLOCKED_VALUE],
            [0, 0, 0]
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

    it('keeps a pinned attention cell active for taps elsewhere in the detail panel', () => {
        expect(shouldClearPinnedAttentionOnDocumentPointerDown({
            isPinned: true,
            panelHit: true
        })).toBe(false);
    });

    it('clears a pinned attention cell when tapping blank space inside the attention matrix', () => {
        expect(shouldClearPinnedAttentionOnDocumentPointerDown({
            isPinned: true,
            panelHit: true,
            insideAttentionBody: true,
            insideAttentionMatrix: true,
            validMatrixCell: false
        })).toBe(true);
    });

    it('clears a pinned attention cell when tapping elsewhere in the attention body', () => {
        expect(shouldClearPinnedAttentionOnDocumentPointerDown({
            isPinned: true,
            panelHit: true,
            insideAttentionBody: true
        })).toBe(true);
    });

    it('clears a pinned attention cell when tapping outside the detail panel', () => {
        expect(shouldClearPinnedAttentionOnDocumentPointerDown({
            isPinned: true
        })).toBe(true);
    });
});
