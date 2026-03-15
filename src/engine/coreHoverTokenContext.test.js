import { describe, expect, it } from 'vitest';

import {
    resolveHoverLabelSubtitle,
    resolveHoverTokenChipSyncEntries,
    resolveHoverTokenChipSyncEntry,
    resolveHoverTokenContext
} from './coreHoverTokenContext.js';

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
            roleLabel: 'Score:',
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
            roleLabel: 'Score:',
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

    it('keeps post-layernorm residual subtitles for layernorm-specific labels', () => {
        const subtitle = resolveHoverLabelSubtitle({
            label: 'Post LayerNorm 2 Residual Vector',
            info: {
                activationData: {
                    stage: 'ln2.shift',
                    tokenIndex: 1,
                    layerIndex: 3
                }
            }
        });

        expect(subtitle).toBe('Position 2 • Layer 4');
    });

    it('resolves weighted-sum vector hovers to a token chip sync entry', () => {
        const entry = resolveHoverTokenChipSyncEntry({
            label: 'Attention Weighted Sum',
            info: {
                activationData: {
                    stage: 'attention.weighted_sum',
                    isWeightedSum: true,
                    tokenIndex: 2,
                    tokenId: 102,
                    tokenLabel: 'Token C'
                }
            }
        });

        expect(entry).toEqual({
            tokenIndex: 2,
            tokenId: 102,
            tokenLabel: 'Token C'
        });
    });

    it('ignores attention-score hovers for token chip sync', () => {
        const entry = resolveHoverTokenChipSyncEntry({
            label: 'Pre-Softmax Attention Score',
            info: buildAttentionInfo({
                stage: 'attention.pre'
            })
        });

        expect(entry).toBeNull();
    });

    it('resolves attention-score hovers to source and target token chip sync entries', () => {
        const entries = resolveHoverTokenChipSyncEntries({
            label: 'Pre-Softmax Attention Score',
            info: buildAttentionInfo({
                stage: 'attention.pre'
            })
        });

        expect(entries).toEqual([
            {
                tokenIndex: 1,
                tokenId: null,
                tokenLabel: 'Token B'
            },
            {
                tokenIndex: 0,
                tokenId: null,
                tokenLabel: 'Token A'
            }
        ]);
    });

    it('ignores layer norm parameter hovers for token chip sync even with lane token metadata', () => {
        const entry = resolveHoverTokenChipSyncEntry({
            label: 'LayerNorm 1 Scale',
            info: {
                tokenIndex: 0,
                tokenLabel: 'Token A',
                activationData: {
                    stage: 'ln1.param.scale',
                    layerIndex: 3
                }
            }
        });

        expect(entry).toBeNull();
    });

    it('honors explicit MLP down token-chip suppression for shared hover labels', () => {
        const context = resolveHoverTokenContext({
            label: 'MLP Down Projection',
            info: {
                activationData: {
                    stage: 'mlp.down',
                    suppressTokenChip: true,
                    tokenIndex: 1,
                    tokenLabel: 'Token B'
                }
            }
        });
        const entry = resolveHoverTokenChipSyncEntry({
            label: 'MLP Down Projection',
            info: {
                activationData: {
                    stage: 'mlp.down',
                    suppressTokenChip: true,
                    tokenIndex: 1,
                    tokenLabel: 'Token B'
                }
            }
        });

        expect(context).toBeNull();
        expect(entry).toBeNull();
    });

    it('resolves position-chip hovers to a token chip sync entry', () => {
        const entry = resolveHoverTokenChipSyncEntry({
            label: 'Position: 3',
            object: {
                userData: {
                    tokenIndex: 2,
                    tokenId: 102,
                    tokenLabel: 'Token C'
                }
            }
        });

        expect(entry).toEqual({
            tokenIndex: 2,
            tokenId: 102,
            tokenLabel: 'Token C'
        });
    });

    it('resolves position-embedding vector hovers to a token chip sync entry', () => {
        const entry = resolveHoverTokenChipSyncEntry({
            label: 'Position Embedding - Token B',
            info: {
                activationData: {
                    stage: 'embedding.position',
                    tokenIndex: 1,
                    tokenId: 101,
                    tokenLabel: 'Token B'
                }
            }
        });

        expect(entry).toEqual({
            tokenIndex: 1,
            tokenId: 101,
            tokenLabel: 'Token B'
        });
    });
});
