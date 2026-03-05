import { describe, it, expect } from 'vitest';
import {
    computeAttentionCellSize,
    countVisibleAttentionCellsInRow,
    getAttentionRevealOrder,
    shouldRevealAttentionCell
} from '../src/ui/selectionPanelAttentionRevealUtils.js';

describe('selectionPanelAttentionRevealUtils', () => {
    it('computes clamped attention cell size', () => {
        const options = { targetPx: 320, minCell: 4, maxCell: 24 };
        expect(computeAttentionCellSize(1, options)).toBe(24);
        expect(computeAttentionCellSize(16, options)).toBe(20);
        expect(computeAttentionCellSize(300, options)).toBe(4);
    });

    it('respects pre and post reveal progress', () => {
        const preProgress = { completedRows: 1, activeRow: 1, activeCol: 2 };
        expect(shouldRevealAttentionCell(preProgress, 0, 5, 'pre')).toBe(true);
        expect(shouldRevealAttentionCell(preProgress, 1, 2, 'pre')).toBe(true);
        expect(shouldRevealAttentionCell(preProgress, 1, 3, 'pre')).toBe(false);

        const postProgress = { postCompletedRows: 2 };
        expect(shouldRevealAttentionCell(postProgress, 1, 0, 'post')).toBe(true);
        expect(shouldRevealAttentionCell(postProgress, 2, 0, 'post')).toBe(false);
    });

    it('honors decode-profile row gating while animating', () => {
        const decodeProfile = { enabled: true, animating: true, highlightRow: 3 };
        expect(shouldRevealAttentionCell(null, 2, 0, 'pre', decodeProfile)).toBe(true);
        expect(shouldRevealAttentionCell(null, 3, 0, 'pre', decodeProfile)).toBe(false);
    });

    it('computes visible cell counts for lower and upper triangles', () => {
        expect(countVisibleAttentionCellsInRow(2, 6, 'lower')).toBe(3);
        expect(countVisibleAttentionCellsInRow(2, 6, 'upper')).toBe(4);
    });

    it('computes reveal order for lower and upper triangles', () => {
        expect(getAttentionRevealOrder(3, 2, 8, 'lower')).toBe(2);
        expect(getAttentionRevealOrder(3, 2, 8, 'upper')).toBe(0);
        expect(getAttentionRevealOrder(3, 6, 8, 'upper')).toBe(3);
    });
});
