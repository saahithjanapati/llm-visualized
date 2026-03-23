// @vitest-environment jsdom

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

let appStateRef = null;
let initPromptTokenStripRef = null;
let tokenChipHoverSyncEventRef = '';

describe('promptTokenStrip mirrored hover state', () => {
    beforeEach(async () => {
        globalThis.localStorage = {
            getItem: () => null,
            setItem: () => {},
            removeItem: () => {}
        };

        ({ appState: appStateRef } = await import('../state/appState.js'));
        ({ initPromptTokenStrip: initPromptTokenStripRef } = await import('./promptTokenStrip.js'));
        ({ TOKEN_CHIP_HOVER_SYNC_EVENT: tokenChipHoverSyncEventRef } = await import('./tokenChipHoverSync.js'));
        appStateRef.showPromptTokenStrip = true;
    });

    afterEach(() => {
        document.body.innerHTML = '';
        delete document.body.dataset.promptTokenStripVisible;
        document.body.style.removeProperty('--prompt-token-strip-height');
        if (appStateRef) appStateRef.showPromptTokenStrip = true;
        delete globalThis.localStorage;
    });

    it('lifts the matching chip when another surface mirrors a token hover', () => {
        const strip = initPromptTokenStripRef();
        strip.update({
            tokenLabels: ['Hello', 'world'],
            tokenIndices: [0, 1],
            tokenIds: [11, 12]
        });

        window.dispatchEvent(new CustomEvent(tokenChipHoverSyncEventRef, {
            detail: {
                active: true,
                source: 'scene-raycast',
                tokenIndex: 1,
                tokenId: 12,
                tokenLabel: 'world'
            }
        }));

        const firstChip = strip.getTokenElement(0);
        const secondChip = strip.getTokenElement(1);
        const tokensRoot = document.querySelector('#promptTokenStrip [data-role="tokens"]');
        expect(firstChip?.classList.contains('is-token-chip-active')).toBe(false);
        expect(firstChip?.classList.contains('is-token-chip-hover-synced')).toBe(false);
        expect(secondChip?.classList.contains('is-token-chip-active')).toBe(true);
        expect(secondChip?.classList.contains('is-token-chip-hover-synced')).toBe(true);
        expect(tokensRoot?.dataset.tokenFocusActive).toBe('true');

        window.dispatchEvent(new CustomEvent(tokenChipHoverSyncEventRef, {
            detail: {
                active: false,
                source: 'scene-raycast',
                tokenIndex: null,
                tokenId: null,
                tokenLabel: ''
            }
        }));

        expect(secondChip?.classList.contains('is-token-chip-active')).toBe(false);
        expect(secondChip?.classList.contains('is-token-chip-hover-synced')).toBe(false);
        expect(tokensRoot?.dataset.tokenFocusActive).toBe('false');

        strip.dispose();
    });

    it('lifts both matching chips when another surface mirrors an attention-score hover pair', () => {
        const strip = initPromptTokenStripRef();
        strip.update({
            tokenLabels: ['Hello', 'world', '!'],
            tokenIndices: [0, 1, 2],
            tokenIds: [11, 12, 13]
        });

        window.dispatchEvent(new CustomEvent(tokenChipHoverSyncEventRef, {
            detail: {
                active: true,
                source: 'scene-raycast',
                entries: [
                    {
                        tokenIndex: 0,
                        tokenId: 11,
                        tokenLabel: 'Hello'
                    },
                    {
                        tokenIndex: 2,
                        tokenId: 13,
                        tokenLabel: '!'
                    }
                ],
                tokenIndex: 0,
                tokenId: 11,
                tokenLabel: 'Hello'
            }
        }));

        expect(strip.getTokenElement(0)?.classList.contains('is-token-chip-active')).toBe(true);
        expect(strip.getTokenElement(1)?.classList.contains('is-token-chip-active')).toBe(false);
        expect(strip.getTokenElement(2)?.classList.contains('is-token-chip-active')).toBe(true);

        strip.dispose();
    });

    it('does not keep a chip active from non-focus-visible focus alone', () => {
        const strip = initPromptTokenStripRef();
        strip.update({
            tokenLabels: ['Hello', 'world'],
            tokenIndices: [0, 1],
            tokenIds: [11, 12]
        });

        const chip = strip.getTokenElement(0);
        chip.matches = vi.fn((selector) => {
            if (selector === ':focus-visible') return false;
            return Element.prototype.matches.call(chip, selector);
        });

        chip.dispatchEvent(new FocusEvent('focusin', {
            bubbles: true
        }));

        expect(chip.classList.contains('is-token-chip-active')).toBe(false);
        expect(chip.classList.contains('is-token-chip-hover-synced')).toBe(false);

        strip.dispose();
    });

    it('keeps generated chips dashed in the token strip', () => {
        const strip = initPromptTokenStripRef();
        strip.update({
            tokenLabels: ['Hello'],
            tokenIndices: [0],
            tokenIds: [11],
            generatedToken: {
                tokenLabel: '<|endoftext|>',
                tokenId: 50256,
                generatedState: 'pending'
            }
        });

        expect(strip.getTokenElement(1)?.classList.contains('prompt-token-strip__token--generated')).toBe(true);

        strip.dispose();
    });
});
