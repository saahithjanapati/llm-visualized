import { describe, expect, it } from 'vitest';
import {
    formatActivationData,
    formatAttentionSubtitleTokenPart,
    formatTokenLabelForPreview,
    formatTokenWithIndex,
    formatValues,
    normalizeAttentionValuePart
} from '../src/ui/selectionPanelFormatUtils.js';

describe('selectionPanelFormatUtils', () => {
    it('formats numeric vectors in wrapped lines', () => {
        expect(formatValues(null)).toBe('(empty)');
        expect(formatValues([1, 2, 3], 2)).toBe('1.0000, 2.0000\n3.0000');
    });

    it('normalizes token labels and attention value placeholders', () => {
        expect(formatTokenLabelForPreview('  hello   world ')).toBe('hello world');
        expect(normalizeAttentionValuePart('   ')).toBe('--');
        expect(normalizeAttentionValuePart(' token ')).toBe('token');
    });

    it('formats token+position helper strings', () => {
        expect(formatTokenWithIndex(2, 'hello')).toBe('3 (hello)');
        expect(formatAttentionSubtitleTokenPart('hello', 1, 'Source')).toBe('Source hello (Position 2)');
    });

    it('formats activation payload summaries with stage and scores', () => {
        const output = formatActivationData({
            stage: 'attention.post',
            layerIndex: 1,
            tokenIndex: 2,
            tokenLabel: 'source',
            keyTokenIndex: 4,
            keyTokenLabel: 'target',
            headIndex: 0,
            preScore: 0.12345,
            postScore: 0.67891,
            values: [0.1, 0.2]
        });

        expect(output).toContain('Stage: attention.post');
        expect(output).toContain('Layer: 2');
        expect(output).toContain('Attention score (post-softmax): 0.6789');
        expect(output).toContain('Values (2):');
    });
});
