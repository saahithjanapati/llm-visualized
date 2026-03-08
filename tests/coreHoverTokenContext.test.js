import { describe, expect, it } from 'vitest';
import {
    findHoverTokenString,
    isVectorLikeHoverSelection,
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

    it('shows plain position text instead of a token chip for weighted sums', () => {
        const info = {
            isWeightedSum: true,
            tokenIndex: 4,
            tokenLabel: 'delta'
        };

        expect(resolveHoverTokenContext({
            label: 'Attention Weighted Sum',
            info
        })).toEqual({
            suppressHoverLabel: false,
            showPrimaryLabel: true,
            detailKind: 'position-text',
            detailText: 'Position 5',
            tokenIndex: 4,
            tokenId: null,
            tokenLabel: ''
        });
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

    it('shows bottom token-chip hovers as chip-only', () => {
        const info = {
            tokenIndex: 2,
            tokenLabel: 'omega'
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
