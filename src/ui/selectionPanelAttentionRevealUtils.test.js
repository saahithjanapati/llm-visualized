import { describe, expect, it } from 'vitest';

import {
    resolveMutedAttentionCellRevealDelay,
    shouldAnimateAttentionCellReveal,
    shouldRevealAttentionCell
} from './selectionPanelAttentionRevealUtils.js';

describe('selectionPanelAttentionRevealUtils', () => {
    it('keeps previously completed rows visible during decode traversal', () => {
        expect(shouldRevealAttentionCell(
            { completedRows: 0, activeRow: 3, activeCol: 1 },
            2,
            4,
            'pre',
            { enabled: true, animating: true, highlightRow: 3 }
        )).toBe(true);
    });

    it('does not animate newly revealed muted pre-softmax cells', () => {
        expect(shouldAnimateAttentionCellReveal({
            mode: 'pre',
            wasEmpty: true,
            isMuted: true
        })).toBe(false);
    });

    it('still animates newly revealed visible pre-softmax cells', () => {
        expect(shouldAnimateAttentionCellReveal({
            mode: 'pre',
            wasEmpty: true,
            isMuted: false
        })).toBe(true);
    });

    it('still animates newly revealed post-softmax cells', () => {
        expect(shouldAnimateAttentionCellReveal({
            mode: 'post',
            wasEmpty: true,
            isMuted: true
        })).toBe(true);
    });

    it('delays muted pre-softmax cells until after the scored row settles', () => {
        expect(resolveMutedAttentionCellRevealDelay(2, 3, 6, 12, 72)).toBe(108);
        expect(resolveMutedAttentionCellRevealDelay(2, 4, 6, 12, 72)).toBe(120);
    });
});
