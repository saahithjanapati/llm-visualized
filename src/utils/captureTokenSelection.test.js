import { describe, expect, it } from 'vitest';

import {
    findMatchingLogitEntryIndex,
    resolveChosenTokenCandidateForToken,
    resolveHiddenTerminalToken
} from './captureTokenSelection.js';

describe('captureTokenSelection', () => {
    it('resolves hidden terminal token metadata from the capture source', () => {
        const activationSource = {
            meta: {
                hidden_terminal_token: {
                    token_id: 50256,
                    token: '<|endoftext|>',
                    token_display: '<|endoftext|>',
                    token_hf: '<|endoftext|>'
                }
            }
        };

        expect(resolveHiddenTerminalToken(activationSource)).toEqual({
            tokenId: 50256,
            tokenRaw: '<|endoftext|>',
            tokenDisplay: '<|endoftext|>',
            tokenHf: '<|endoftext|>'
        });
    });

    it('matches a hidden terminal token from the final visible logit row', () => {
        const activationSource = {
            meta: {
                hidden_terminal_token: {
                    token_id: 50256,
                    token: '<|endoftext|>',
                    token_display: '<|endoftext|>'
                }
            },
            getTokenCount() {
                return 3;
            },
            getLogitsForToken(tokenIndex) {
                return tokenIndex === 2
                    ? [
                        { token_id: 198, token: '\n', prob: 0.1737 },
                        { token_id: 50256, token: '<|endoftext|>', prob: 0.0111 },
                        { token_id: 464, token: 'The', prob: 0.0658 }
                    ]
                    : [];
            }
        };

        const chosenToken = resolveChosenTokenCandidateForToken(activationSource, 2);
        expect(chosenToken).toMatchObject({
            resolution: 'hidden-terminal-token',
            tokenIndex: null,
            tokenId: 50256,
            tokenRaw: '<|endoftext|>',
            tokenDisplay: '<|endoftext|>',
            logitEntryIndex: 1
        });
        expect(chosenToken?.logitEntry).toEqual({
            token_id: 50256,
            token: '<|endoftext|>',
            prob: 0.0111
        });
    });

    it('matches visible next-token rows before considering hidden terminal metadata', () => {
        const activationSource = {
            meta: {
                hidden_terminal_token: {
                    token_id: 50256,
                    token: '<|endoftext|>',
                    token_display: '<|endoftext|>'
                }
            },
            getTokenCount() {
                return 3;
            },
            getTokenString(tokenIndex) {
                return ['Alpha', 'Beta', 'Gamma'][tokenIndex] || '';
            },
            getTokenId(tokenIndex) {
                return [101, 102, 103][tokenIndex] ?? null;
            },
            getLogitsForToken(tokenIndex) {
                return tokenIndex === 1
                    ? [
                        { token_id: 103, token: 'Gamma', prob: 0.4 },
                        { token_id: 50256, token: '<|endoftext|>', prob: 0.3 }
                    ]
                    : [];
            }
        };

        const chosenToken = resolveChosenTokenCandidateForToken(activationSource, 1);
        expect(chosenToken).toMatchObject({
            resolution: 'next-visible-token',
            tokenIndex: 2,
            tokenId: 103,
            tokenRaw: 'Gamma',
            tokenDisplay: 'Gamma',
            logitEntryIndex: 0
        });
    });

    it('matches logit rows by token id before raw token text', () => {
        const logitRow = [
            { token_id: 1, token: 'A' },
            { token_id: 50256, token: 'Different text' },
            { token_id: 7, token: '<|endoftext|>' }
        ];

        expect(findMatchingLogitEntryIndex(logitRow, {
            tokenId: 50256,
            tokenRaw: '<|endoftext|>'
        })).toBe(1);
    });
});
