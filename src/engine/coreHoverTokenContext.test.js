import { describe, expect, it } from 'vitest';

import { resolveHoverTokenContext } from './coreHoverTokenContext.js';

function buildAttentionInfo({
    stage,
    preScore = 0.25,
    postScore = 0.15,
    maskValue = Number.NEGATIVE_INFINITY,
    showMaskValue = false
} = {}) {
    return {
        activationData: {
            stage,
            queryTokenIndex: 1,
            queryTokenLabel: 'Token B',
            keyTokenIndex: 0,
            keyTokenLabel: 'Token A',
            preScore,
            postScore,
            maskValue,
            showMaskValue
        }
    };
}

describe('resolveHoverTokenContext attention metrics', () => {
    it('shows only the pre-softmax metric for pre-score hovers', () => {
        const context = resolveHoverTokenContext({
            label: 'Pre-Softmax Attention Score',
            info: buildAttentionInfo({
                stage: 'attention.pre'
            })
        });

        expect(context?.detailKind).toBe('attention-token-pair');
        expect(context?.attentionMetrics).toEqual([{
            roleLabel: 'Score',
            valueText: '0.2500'
        }]);
    });

    it('shows only the post-softmax metric for post-score hovers', () => {
        const context = resolveHoverTokenContext({
            label: 'Post-Softmax Attention Score',
            info: buildAttentionInfo({
                stage: 'attention.post'
            })
        });

        expect(context?.detailKind).toBe('attention-token-pair');
        expect(context?.attentionMetrics).toEqual([{
            roleLabel: 'Score',
            valueText: '0.1500'
        }]);
    });

    it('shows only the causal-mask metric for mask hovers', () => {
        const context = resolveHoverTokenContext({
            label: 'Causal Mask',
            info: buildAttentionInfo({
                stage: 'attention.mask',
                showMaskValue: true
            })
        });

        expect(context?.detailKind).toBe('attention-token-pair');
        expect(context?.attentionMetrics).toEqual([{
            roleLabel: 'Causal mask',
            valueText: '-∞'
        }]);
    });
});
