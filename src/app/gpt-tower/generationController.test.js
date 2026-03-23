// @vitest-environment jsdom

import { afterEach, beforeEach, describe, expect, it } from 'vitest';

let resolvePromptStripGeneratedTokenRef = null;

describe('resolvePromptStripGeneratedToken', () => {
    beforeEach(async () => {
        globalThis.localStorage = {
            getItem: () => null,
            setItem: () => {},
            removeItem: () => {}
        };

        ({ resolvePromptStripGeneratedToken: resolvePromptStripGeneratedTokenRef } = await import('./generationController.js'));
    });

    afterEach(() => {
        delete globalThis.localStorage;
    });

    it('keeps regular next-token chips hidden until the current pass completes', () => {
        const activationSource = {
            getTokenCount() {
                return 2;
            },
            getTokenString(tokenIndex) {
                return ['Hello', 'world'][tokenIndex] || '';
            },
            getTokenId(tokenIndex) {
                return [11, 12][tokenIndex] ?? null;
            },
            getLogitTopK() {
                return 2;
            },
            getLogitsForToken(tokenIndex) {
                return tokenIndex === 0
                    ? [
                        { token_id: 12, token: 'world', prob: 0.7 },
                        { token_id: 99, token: 'other', prob: 0.3 }
                    ]
                    : [];
            }
        };

        expect(resolvePromptStripGeneratedTokenRef(activationSource, [0], {
            currentLaneCount: 1,
            maxLaneCount: 2,
            passComplete: false
        })).toBeNull();

        expect(resolvePromptStripGeneratedTokenRef(activationSource, [0], {
            currentLaneCount: 1,
            maxLaneCount: 2,
            passComplete: true
        })).toMatchObject({
            tokenIndex: 1,
            tokenId: 12,
            tokenLabel: 'world',
            generatedState: 'pending'
        });
    });

    it('shows the hidden terminal token only after the final visible pass completes and keeps it pending', () => {
        const activationSource = {
            getTokenCount() {
                return 3;
            },
            getTokenString(tokenIndex) {
                return ['Alpha', 'Beta', 'Gamma'][tokenIndex] || '';
            },
            getTokenId(tokenIndex) {
                return [101, 102, 103][tokenIndex] ?? null;
            },
            getHiddenTerminalToken() {
                return {
                    token_id: 50256,
                    token: '<|endoftext|>',
                    token_display: '<|endoftext|>'
                };
            },
            getLogitTopK() {
                return 2;
            },
            getLogitsForToken(tokenIndex) {
                return tokenIndex === 2
                    ? [
                        { token_id: 103, token: 'Gamma', prob: 0.3 },
                        { token_id: 50256, token: '<|endoftext|>', prob: 0.2 }
                    ]
                    : [];
            }
        };

        expect(resolvePromptStripGeneratedTokenRef(activationSource, [0, 1, 2], {
            currentLaneCount: 3,
            maxLaneCount: 3,
            passComplete: false
        })).toBeNull();

        expect(resolvePromptStripGeneratedTokenRef(activationSource, [0, 1, 2], {
            currentLaneCount: 3,
            maxLaneCount: 3,
            passComplete: true
        })).toMatchObject({
            tokenIndex: null,
            tokenId: 50256,
            tokenLabel: '<|endoftext|>',
            generatedState: 'pending'
        });
    });
});
