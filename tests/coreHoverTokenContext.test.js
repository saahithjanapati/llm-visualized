import { describe, expect, it } from 'vitest';
import {
    findHoverTokenString,
    isVectorLikeHoverSelection,
    resolveHoverLabelSubtitle,
    resolveHoverTokenContext
} from '../src/engine/coreHoverTokenContext.js';

describe('coreHoverTokenContext', () => {
    it('reads token metadata from vectorRef parent-lane fallbacks', () => {
        const info = {
            vectorRef: {
                userData: {
                    parentLane: {
                        tokenIndex: 3,
                        tokenLabel: 'world'
                    }
                }
            }
        };

        expect(findHoverTokenString(info, null, 'tokenLabel')).toBe('world');
        expect(resolveHoverTokenContext({ label: 'LN1 Normed', info })).toEqual({
            suppressHoverLabel: false,
            showPrimaryLabel: true,
            primaryLabelText: '',
            detailKind: 'token-chip',
            detailText: 'world',
            tokenIndex: 3,
            tokenId: null,
            tokenLabel: 'world'
        });
    });

    it('uses activation-source fallbacks for vector hovers with only a token index', () => {
        const info = {
            kind: 'mergedKV',
            tokenIndex: 1
        };
        const activationSource = {
            getTokenString: (tokenIndex) => (tokenIndex === 1 ? 'beta' : ''),
            getTokenId: (tokenIndex) => (tokenIndex === 1 ? 502 : null)
        };

        expect(resolveHoverTokenContext({
            label: 'Cached Key Vector',
            info,
            activationSource
        })).toEqual({
            suppressHoverLabel: false,
            showPrimaryLabel: true,
            primaryLabelText: '',
            detailKind: 'token-chip',
            detailText: 'beta',
            tokenIndex: 1,
            tokenId: 502,
            tokenLabel: 'beta'
        });
    });

    it('keeps weighted-sum hover labels free of top-row detail text', () => {
        const info = {
            isWeightedSum: true,
            tokenIndex: 4,
            tokenLabel: 'delta'
        };

        expect(resolveHoverTokenContext({
            label: 'Attention Weighted Sum',
            info
        })).toBeNull();
    });

    it('returns source and target token rows for attention-score hovers', () => {
        const info = {
            activationData: {
                stage: 'attention.pre',
                tokenIndex: 1,
                tokenLabel: 'beta',
                keyTokenIndex: 0,
                keyTokenLabel: 'alpha'
            }
        };
        const activationSource = {
            getTokenId: (tokenIndex) => (tokenIndex === 1 ? 22 : (tokenIndex === 0 ? 11 : null))
        };

        expect(resolveHoverTokenContext({
            label: 'Pre-Softmax Attention Score',
            info,
            activationSource
        })).toEqual({
            suppressHoverLabel: false,
            showPrimaryLabel: true,
            detailKind: 'attention-token-pair',
            attentionRows: [
                {
                    roleLabel: 'Source',
                    tokenLabel: 'beta',
                    tokenIndex: 1,
                    tokenId: 22,
                    positionText: 'Position 2'
                },
                {
                    roleLabel: 'Target',
                    tokenLabel: 'alpha',
                    tokenIndex: 0,
                    tokenId: 11,
                    positionText: 'Position 1'
                }
            ]
        });
    });

    it('shows qkv hover subtitles as position before head and layer', () => {
        const info = {
            headIndex: 1,
            layerIndex: 3,
            tokenIndex: 6,
            stage: 'qkv.k'
        };

        expect(resolveHoverLabelSubtitle({
            label: 'Key Vector',
            info
        })).toBe('Position 7 • Head 2 • Layer 4');
    });

    it('shows weighted-sum hover subtitles as position before head and layer', () => {
        const info = {
            isWeightedSum: true,
            headIndex: 1,
            layerIndex: 3,
            tokenIndex: 6,
            stage: 'attention.weighted_sum'
        };

        expect(resolveHoverLabelSubtitle({
            label: 'Attention Weighted Sum',
            info
        })).toBe('Position 7 • Head 2 • Layer 4');
    });

    it('appends token position to incoming residual hover subtitles', () => {
        const info = {
            layerIndex: 2,
            tokenIndex: 4,
            stage: 'layer.incoming'
        };

        expect(resolveHoverLabelSubtitle({
            label: 'Residual Stream Vector',
            info
        })).toBe('Position 5 • Layer 3');
    });

    it('shows position before layer for post-layernorm residual hover subtitles', () => {
        const info = {
            layerIndex: 1,
            tokenIndex: 3,
            stage: 'ln1.shift'
        };

        expect(resolveHoverLabelSubtitle({
            label: 'Post LayerNorm Residual Vector',
            info
        })).toBe('Position 4 • Layer 2');
    });

    it('shows position for embedding-sum residual hover subtitles even without a layer', () => {
        const info = {
            tokenIndex: 1,
            stage: 'embedding.sum'
        };

        expect(resolveHoverLabelSubtitle({
            label: 'Residual Stream Vector',
            info
        })).toBe('Position 2');
    });

    it('shows token chips for mlp down-projection vector hovers', () => {
        const object = {
            userData: {
                activationData: {
                    stage: 'mlp.down',
                    tokenIndex: 5,
                    tokenLabel: 'theta'
                }
            },
            parent: null
        };

        expect(resolveHoverTokenContext({
            label: 'MLP Down Projection',
            object
        })).toEqual({
            suppressHoverLabel: false,
            showPrimaryLabel: true,
            primaryLabelText: '',
            detailKind: 'token-chip',
            detailText: 'theta',
            tokenIndex: 5,
            tokenId: null,
            tokenLabel: 'theta'
        });
    });

    it('keeps token chips off layernorm parameter hovers even with token metadata present', () => {
        const info = {
            activationData: {
                stage: 'ln1.param.scale',
                tokenIndex: 5,
                tokenLabel: 'theta'
            },
            layerNormKind: 'ln1'
        };

        expect(resolveHoverTokenContext({
            label: 'LayerNorm 1 Scale',
            info
        })).toBeNull();
    });

    it('keeps token chips off suppressed semantic module hovers even if token metadata leaks in', () => {
        const info = {
            suppressTokenChip: true,
            activationData: {
                stage: 'attention.output_projection',
                suppressTokenChip: true,
                tokenIndex: 5,
                tokenLabel: 'theta'
            }
        };

        expect(resolveHoverTokenContext({
            label: 'Output Projection Matrix',
            info
        })).toBeNull();
    });

    it('shows bottom token-chip hovers as chip-only', () => {
        const info = {
            tokenIndex: 2,
            tokenLabel: 'omega',
            positionIndex: 3
        };

        expect(resolveHoverTokenContext({
            label: 'Token: omega',
            info
        })).toEqual({
            suppressHoverLabel: false,
            showPrimaryLabel: true,
            primaryLabelText: 'Token',
            detailKind: 'token-chip',
            detailText: 'omega',
            tokenIndex: 2,
            tokenId: null,
            tokenLabel: 'omega'
        });
        expect(resolveHoverLabelSubtitle({
            label: 'Token: omega',
            info
        })).toBe('Position 3');
    });

    it('keeps bottom position-chip hovers text-only', () => {
        const info = {
            tokenIndex: 2,
            tokenLabel: 'omega'
        };

        expect(resolveHoverTokenContext({
            label: 'Position: 3',
            info
        })).toBeNull();
    });

    it('keeps embedding-matrix hovers text-only even if token metadata is present', () => {
        const info = {
            tokenIndex: 2,
            tokenLabel: 'omega'
        };

        expect(resolveHoverTokenContext({
            label: 'Vocabulary Embedding',
            info
        })).toBeNull();
        expect(resolveHoverTokenContext({
            label: 'Positional Embedding',
            info
        })).toBeNull();
    });

    it('keeps token chips off weight matrices even when token metadata is present', () => {
        const info = {
            vectorRef: {
                userData: {
                    parentLane: {
                        tokenIndex: 0,
                        tokenLabel: 'alpha'
                    }
                }
            }
        };

        expect(isVectorLikeHoverSelection('Query Weight Matrix', info)).toBe(false);
        expect(resolveHoverTokenContext({
            label: 'Query Weight Matrix',
            info
        })).toBeNull();
    });

    it('keeps token chips off bias vectors even when token metadata is present', () => {
        const info = {
            tokenIndex: 2,
            tokenLabel: 'gamma',
            headIndex: 4,
            layerIndex: 1
        };

        expect(isVectorLikeHoverSelection('Value Bias Vector', info)).toBe(false);
        expect(resolveHoverTokenContext({
            label: 'Value Bias Vector',
            info
        })).toBeNull();
        expect(resolveHoverLabelSubtitle({
            label: 'Value Bias Vector',
            info
        })).toBe('Head 5 • Layer 2');
    });

    it('does not attach token chips to non-vector token-bearing hovers', () => {
        const info = {
            tokenIndex: 2,
            tokenLabel: 'gamma'
        };

        expect(resolveHoverTokenContext({
            label: 'Attention Score',
            info
        })).toBeNull();
    });
});
