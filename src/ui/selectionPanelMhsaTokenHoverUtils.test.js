import { describe, expect, it } from 'vitest';

import {
    resolveMhsaTokenMatrixHoverTokenEntries
} from './selectionPanelMhsaTokenHoverUtils.js';

describe('selectionPanelMhsaTokenHoverUtils', () => {
    const activationSource = {
        getTokenId(tokenIndex) {
            return tokenIndex + 100;
        },
        getTokenString(tokenIndex) {
            return `Token ${tokenIndex + 1}`;
        }
    };

    const previewData = {
        rows: [
            { rowIndex: 0, tokenIndex: 7, tokenLabel: 'Alpha' },
            { rowIndex: 1, tokenIndex: 8, tokenLabel: 'Beta' },
            { rowIndex: 2, tokenIndex: 9, tokenLabel: 'Gamma' }
        ]
    };

    it('resolves a single token chip entry for X_ln, Q, K, V, and head-output row hovers', () => {
        [
            { kind: 'row', rowIndex: 1, sourceType: 'x' },
            { kind: 'row', rowIndex: 1, sourceType: 'query' },
            { kind: 'key-link', rowIndex: 1, sourceType: 'query' },
            { kind: 'key-link', rowIndex: 1, sourceType: 'transpose' },
            { kind: 'row', rowIndex: 1, sourceType: 'value-post' },
            { kind: 'row', rowIndex: 1, sourceType: 'head-output' }
        ].forEach((targetInfo) => {
            expect(resolveMhsaTokenMatrixHoverTokenEntries(targetInfo, {
                previewData,
                activationSource
            })).toEqual([
                {
                    tokenIndex: 8,
                    tokenId: 108,
                    tokenLabel: 'Beta'
                }
            ]);
        });
    });

    it('returns both query and key token entries for score hovers', () => {
        expect(resolveMhsaTokenMatrixHoverTokenEntries({
            kind: 'score',
            rowIndex: 2,
            colIndex: 0,
            sourceType: 'pre'
        }, {
            previewData,
            activationSource
        })).toEqual([
            {
                tokenIndex: 9,
                tokenId: 109,
                tokenLabel: 'Gamma'
            },
            {
                tokenIndex: 7,
                tokenId: 107,
                tokenLabel: 'Alpha'
            }
        ]);
    });

    it('deduplicates score hovers on the diagonal', () => {
        expect(resolveMhsaTokenMatrixHoverTokenEntries({
            kind: 'score',
            rowIndex: 1,
            colIndex: 1,
            sourceType: 'pre'
        }, {
            previewData,
            activationSource
        })).toEqual([
            {
                tokenIndex: 8,
                tokenId: 108,
                tokenLabel: 'Beta'
            }
        ]);
    });

    it('falls back to token index and label arrays when preview rows are unavailable', () => {
        expect(resolveMhsaTokenMatrixHoverTokenEntries({
            kind: 'row',
            rowIndex: 0,
            sourceType: 'x'
        }, {
            activationSource,
            tokenIndices: [4],
            tokenLabels: ['Fallback']
        })).toEqual([
            {
                tokenIndex: 4,
                tokenId: 104,
                tokenLabel: 'Fallback'
            }
        ]);
    });

    it('returns no entries for non-token targets', () => {
        expect(resolveMhsaTokenMatrixHoverTokenEntries({
            kind: 'attention-block',
            focusKey: 'query'
        }, {
            previewData,
            activationSource
        })).toEqual([]);
    });
});
