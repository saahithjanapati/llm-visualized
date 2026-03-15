// @vitest-environment jsdom

import { afterEach, describe, expect, it, vi } from 'vitest';

import { TOKEN_CHIP_HOVER_SYNC_EVENT } from './tokenChipHoverSync.js';
import {
    createTransformerView2dTokenHoverSync,
    TRANSFORMER_VIEW2D_TOKEN_HOVER_SOURCE,
    resolveTransformerView2dTokenEntriesFromHoverPayload,
    resolveTransformerView2dTokenEntryFromHoverPayload,
    resolveTransformerView2dTokenEntryFromResidualHoverPayload
} from './selectionPanelTransformerView2dTokenHoverUtils.js';

function buildTokenChip({
    tokenText = '',
    tokenIndex = null,
    tokenId = null
} = {}) {
    const chip = document.createElement('button');
    chip.type = 'button';
    chip.className = 'detail-transformer-view2d-token-strip__token prompt-token-strip__token';
    chip.dataset.tokenNav = tokenText.length ? 'true' : 'false';
    chip.dataset.tokenText = tokenText;
    if (Number.isFinite(tokenIndex)) {
        chip.dataset.tokenIndex = String(tokenIndex);
    }
    if (Number.isFinite(tokenId)) {
        chip.dataset.tokenId = String(tokenId);
    }
    chip.textContent = tokenText;
    return chip;
}

describe('selectionPanelTransformerView2dTokenHoverUtils', () => {
    afterEach(() => {
        document.body.innerHTML = '';
    });

    it('normalizes token metadata from residual hover payloads', () => {
        expect(resolveTransformerView2dTokenEntryFromResidualHoverPayload({
            info: {
                tokenIndex: 4,
                tokenId: 99,
                tokenLabel: 'world'
            }
        })).toEqual({
            tokenIndex: 4,
            tokenId: 99,
            tokenLabel: 'world'
        });
    });

    it('normalizes token metadata from detailed hover payloads', () => {
        expect(resolveTransformerView2dTokenEntryFromHoverPayload({
            info: {
                activationData: {
                    queryTokenIndex: 2,
                    queryTokenLabel: 'Token C'
                }
            }
        })).toEqual({
            tokenIndex: 2,
            tokenId: null,
            tokenLabel: 'Token C'
        });
    });

    it('suppresses token metadata for MLP down weight-matrix hovers to match 3D behavior', () => {
        expect(resolveTransformerView2dTokenEntryFromHoverPayload({
            label: 'MLP Down Weight Matrix',
            info: {
                activationData: {
                    stage: 'mlp.down',
                    tokenIndex: 1,
                    tokenLabel: 'Token B'
                }
            }
        })).toBeNull();
    });

    it('keeps token metadata for MLP down projection row hovers', () => {
        expect(resolveTransformerView2dTokenEntryFromHoverPayload({
            label: 'MLP Down Projection',
            info: {
                activationData: {
                    stage: 'mlp.down',
                    tokenIndex: 1,
                    tokenLabel: 'Token B'
                }
            }
        })).toEqual({
            tokenIndex: 1,
            tokenId: null,
            tokenLabel: 'Token B'
        });
    });

    it('normalizes source and target token metadata from attention hover payloads', () => {
        expect(resolveTransformerView2dTokenEntriesFromHoverPayload({
            info: {
                activationData: {
                    queryTokenIndex: 2,
                    queryTokenLabel: 'Token C',
                    keyTokenIndex: 0,
                    keyTokenLabel: 'Token A'
                }
            }
        })).toEqual([
            {
                tokenIndex: 2,
                tokenId: null,
                tokenLabel: 'Token C'
            },
            {
                tokenIndex: 0,
                tokenId: null,
                tokenLabel: 'Token A'
            }
        ]);
    });

    it('keeps both attention tokens active when the canvas sync payload includes a token pair', () => {
        const container = document.createElement('div');
        const firstChip = buildTokenChip({
            tokenText: 'Token A',
            tokenIndex: 0,
            tokenId: 11
        });
        const secondChip = buildTokenChip({
            tokenText: 'Token B',
            tokenIndex: 1,
            tokenId: 12
        });
        const thirdChip = buildTokenChip({
            tokenText: 'Token C',
            tokenIndex: 2,
            tokenId: 13
        });
        container.append(firstChip, secondChip, thirdChip);
        document.body.appendChild(container);

        const hoverSync = createTransformerView2dTokenHoverSync({ container });
        hoverSync.setCanvasEntry([
            {
                tokenIndex: 2,
                tokenId: 13,
                tokenLabel: 'Token C'
            },
            {
                tokenIndex: 0,
                tokenId: 11,
                tokenLabel: 'Token A'
            }
        ]);

        expect(firstChip.classList.contains('is-token-chip-active')).toBe(true);
        expect(secondChip.classList.contains('is-token-chip-active')).toBe(false);
        expect(thirdChip.classList.contains('is-token-chip-active')).toBe(true);
        expect(container.dataset.tokenFocusActive).toBe('true');

        hoverSync.dispose({ emit: false });
    });

    it('highlights the matching 2D token chip and emits sync when the canvas hovers a residual row', () => {
        const container = document.createElement('div');
        const firstChip = buildTokenChip({
            tokenText: 'Hello',
            tokenIndex: 0,
            tokenId: 11
        });
        const secondChip = buildTokenChip({
            tokenText: 'world',
            tokenIndex: 1,
            tokenId: 12
        });
        container.append(firstChip, secondChip);
        document.body.appendChild(container);

        const syncDetails = [];
        const onSync = vi.fn((event) => {
            syncDetails.push(event.detail);
        });
        window.addEventListener(TOKEN_CHIP_HOVER_SYNC_EVENT, onSync);

        const hoverSync = createTransformerView2dTokenHoverSync({ container });
        hoverSync.setCanvasEntryFromResidualHoverPayload({
            info: {
                tokenIndex: 1,
                tokenId: 12,
                tokenLabel: 'world'
            }
        });

        expect(firstChip.classList.contains('is-token-chip-active')).toBe(false);
        expect(secondChip.classList.contains('is-token-chip-active')).toBe(true);
        expect(secondChip.dataset.tokenActive).toBe('true');
        expect(secondChip.classList.contains('is-token-chip-hover-synced')).toBe(false);
        expect(container.dataset.tokenFocusActive).toBe('true');
        expect(syncDetails.at(-1)).toEqual({
            active: true,
            source: TRANSFORMER_VIEW2D_TOKEN_HOVER_SOURCE,
            entries: [{
                tokenIndex: 1,
                tokenId: 12,
                tokenLabel: 'world'
            }],
            tokenIndex: 1,
            tokenId: 12,
            tokenLabel: 'world'
        });

        hoverSync.clearCanvasEntry();

        expect(secondChip.classList.contains('is-token-chip-active')).toBe(false);
        expect(secondChip.dataset.tokenActive).toBe('false');
        expect(container.dataset.tokenFocusActive).toBe('false');
        expect(syncDetails.at(-1)).toEqual({
            active: false,
            source: TRANSFORMER_VIEW2D_TOKEN_HOVER_SOURCE,
            entries: [],
            tokenIndex: null,
            tokenId: null,
            tokenLabel: ''
        });

        hoverSync.dispose({ emit: false });
        window.removeEventListener(TOKEN_CHIP_HOVER_SYNC_EVENT, onSync);
    });

    it('highlights the matching 2D token chip when the canvas hovers a detailed-view row', () => {
        const container = document.createElement('div');
        const firstChip = buildTokenChip({
            tokenText: 'Token A',
            tokenIndex: 0,
            tokenId: 11
        });
        const secondChip = buildTokenChip({
            tokenText: 'Token B',
            tokenIndex: 1,
            tokenId: 12
        });
        container.append(firstChip, secondChip);
        document.body.appendChild(container);

        const hoverSync = createTransformerView2dTokenHoverSync({ container });
        hoverSync.setCanvasEntryFromHoverPayload({
            info: {
                activationData: {
                    tokenIndex: 1,
                    tokenLabel: 'Token B'
                }
            }
        });

        expect(firstChip.classList.contains('is-token-chip-active')).toBe(false);
        expect(secondChip.classList.contains('is-token-chip-active')).toBe(true);

        hoverSync.dispose({ emit: false });
    });

    it('does not highlight a token chip when the canvas hovers the MLP down weight matrix', () => {
        const container = document.createElement('div');
        const firstChip = buildTokenChip({
            tokenText: 'Token A',
            tokenIndex: 0,
            tokenId: 11
        });
        const secondChip = buildTokenChip({
            tokenText: 'Token B',
            tokenIndex: 1,
            tokenId: 12
        });
        container.append(firstChip, secondChip);
        document.body.appendChild(container);

        const hoverSync = createTransformerView2dTokenHoverSync({ container });
        hoverSync.setCanvasEntryFromHoverPayload({
            label: 'MLP Down Weight Matrix',
            info: {
                activationData: {
                    stage: 'mlp.down',
                    tokenIndex: 1,
                    tokenLabel: 'Token B'
                }
            }
        });

        expect(firstChip.classList.contains('is-token-chip-active')).toBe(false);
        expect(secondChip.classList.contains('is-token-chip-active')).toBe(false);
        expect(container.dataset.tokenFocusActive).toBe('false');

        hoverSync.dispose({ emit: false });
    });

    it('highlights the matching token chip when the canvas hovers the MLP down projection row', () => {
        const container = document.createElement('div');
        const firstChip = buildTokenChip({
            tokenText: 'Token A',
            tokenIndex: 0,
            tokenId: 11
        });
        const secondChip = buildTokenChip({
            tokenText: 'Token B',
            tokenIndex: 1,
            tokenId: 12
        });
        container.append(firstChip, secondChip);
        document.body.appendChild(container);

        const hoverSync = createTransformerView2dTokenHoverSync({ container });
        hoverSync.setCanvasEntryFromHoverPayload({
            label: 'MLP Down Projection',
            info: {
                activationData: {
                    stage: 'mlp.down',
                    tokenIndex: 1,
                    tokenLabel: 'Token B'
                }
            }
        });

        expect(firstChip.classList.contains('is-token-chip-active')).toBe(false);
        expect(secondChip.classList.contains('is-token-chip-active')).toBe(true);
        expect(container.dataset.tokenFocusActive).toBe('true');

        hoverSync.dispose({ emit: false });
    });

    it('does not clear an active strip hover when the canvas hover path clears', () => {
        const container = document.createElement('div');
        const chip = buildTokenChip({
            tokenText: 'Hello',
            tokenIndex: 0,
            tokenId: 11
        });
        container.appendChild(chip);
        document.body.appendChild(container);

        const syncDetails = [];
        const onSync = vi.fn((event) => {
            syncDetails.push(event.detail);
        });
        window.addEventListener(TOKEN_CHIP_HOVER_SYNC_EVENT, onSync);

        const hoverSync = createTransformerView2dTokenHoverSync({ container });
        chip.dispatchEvent(new Event('pointerover', {
            bubbles: true
        }));

        expect(chip.classList.contains('is-token-chip-active')).toBe(true);

        hoverSync.clearCanvasEntry();

        expect(chip.classList.contains('is-token-chip-active')).toBe(true);
        expect(syncDetails).toHaveLength(1);
        expect(syncDetails[0]).toEqual({
            active: true,
            source: TRANSFORMER_VIEW2D_TOKEN_HOVER_SOURCE,
            entries: [{
                tokenIndex: 0,
                tokenId: 11,
                tokenLabel: 'Hello'
            }],
            tokenIndex: 0,
            tokenId: 11,
            tokenLabel: 'Hello'
        });

        hoverSync.dispose({ emit: false });
        window.removeEventListener(TOKEN_CHIP_HOVER_SYNC_EVENT, onSync);
    });

    it('mirrors token hover sync events from other surfaces onto the 2D token strip', () => {
        const container = document.createElement('div');
        const firstChip = buildTokenChip({
            tokenText: 'Hello',
            tokenIndex: 0,
            tokenId: 11
        });
        const secondChip = buildTokenChip({
            tokenText: 'world',
            tokenIndex: 1,
            tokenId: 12
        });
        container.append(firstChip, secondChip);
        document.body.appendChild(container);

        const hoverSync = createTransformerView2dTokenHoverSync({ container });

        window.dispatchEvent(new CustomEvent(TOKEN_CHIP_HOVER_SYNC_EVENT, {
            detail: {
                active: true,
                source: 'prompt-token-strip',
                tokenIndex: 0,
                tokenId: 11,
                tokenLabel: 'Hello'
            }
        }));

        expect(firstChip.classList.contains('is-token-chip-active')).toBe(true);
        expect(secondChip.classList.contains('is-token-chip-active')).toBe(false);
        expect(firstChip.classList.contains('is-token-chip-hover-synced')).toBe(true);
        expect(container.dataset.tokenFocusActive).toBe('true');

        window.dispatchEvent(new CustomEvent(TOKEN_CHIP_HOVER_SYNC_EVENT, {
            detail: {
                active: false,
                source: 'prompt-token-strip',
                tokenIndex: null,
                tokenId: null,
                tokenLabel: ''
            }
        }));

        expect(firstChip.classList.contains('is-token-chip-active')).toBe(false);
        expect(firstChip.classList.contains('is-token-chip-hover-synced')).toBe(false);
        expect(container.dataset.tokenFocusActive).toBe('false');

        hoverSync.dispose({ emit: false });
    });

    it('clears stale mirrored token hover state when the 2D view resets', () => {
        const container = document.createElement('div');
        const firstChip = buildTokenChip({
            tokenText: 'Hello',
            tokenIndex: 0,
            tokenId: 11
        });
        const secondChip = buildTokenChip({
            tokenText: 'world',
            tokenIndex: 1,
            tokenId: 12
        });
        container.append(firstChip, secondChip);
        document.body.appendChild(container);

        const hoverSync = createTransformerView2dTokenHoverSync({ container });

        window.dispatchEvent(new CustomEvent(TOKEN_CHIP_HOVER_SYNC_EVENT, {
            detail: {
                active: true,
                source: 'prompt-token-strip',
                entries: [{
                    tokenIndex: 1,
                    tokenId: 12,
                    tokenLabel: 'world'
                }],
                tokenIndex: 1,
                tokenId: 12,
                tokenLabel: 'world'
            }
        }));

        expect(firstChip.classList.contains('is-token-chip-active')).toBe(false);
        expect(secondChip.classList.contains('is-token-chip-active')).toBe(true);
        expect(container.dataset.tokenFocusActive).toBe('true');

        hoverSync.clear({ emit: false });

        expect(firstChip.classList.contains('is-token-chip-active')).toBe(false);
        expect(secondChip.classList.contains('is-token-chip-active')).toBe(false);
        expect(container.dataset.tokenFocusActive).toBe('false');

        hoverSync.dispose({ emit: false });
    });

    it('mirrors multi-token hover sync events from other surfaces onto the 2D token strip', () => {
        const container = document.createElement('div');
        const firstChip = buildTokenChip({
            tokenText: 'Hello',
            tokenIndex: 0,
            tokenId: 11
        });
        const secondChip = buildTokenChip({
            tokenText: 'world',
            tokenIndex: 1,
            tokenId: 12
        });
        const thirdChip = buildTokenChip({
            tokenText: '!',
            tokenIndex: 2,
            tokenId: 13
        });
        container.append(firstChip, secondChip, thirdChip);
        document.body.appendChild(container);

        const hoverSync = createTransformerView2dTokenHoverSync({ container });

        window.dispatchEvent(new CustomEvent(TOKEN_CHIP_HOVER_SYNC_EVENT, {
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

        expect(firstChip.classList.contains('is-token-chip-active')).toBe(true);
        expect(secondChip.classList.contains('is-token-chip-active')).toBe(false);
        expect(thirdChip.classList.contains('is-token-chip-active')).toBe(true);

        hoverSync.dispose({ emit: false });
    });

    it('does not keep a 2D token chip active from non-focus-visible focus alone', () => {
        const container = document.createElement('div');
        const chip = buildTokenChip({
            tokenText: 'Hello',
            tokenIndex: 0,
            tokenId: 11
        });
        chip.matches = vi.fn((selector) => {
            if (selector === ':focus-visible') return false;
            return Element.prototype.matches.call(chip, selector);
        });
        container.appendChild(chip);
        document.body.appendChild(container);

        const hoverSync = createTransformerView2dTokenHoverSync({ container });

        chip.dispatchEvent(new FocusEvent('focusin', {
            bubbles: true
        }));

        expect(chip.classList.contains('is-token-chip-active')).toBe(false);
        expect(chip.classList.contains('is-token-chip-hover-synced')).toBe(false);
        expect(container.dataset.tokenFocusActive === 'true').toBe(false);

        hoverSync.dispose({ emit: false });
    });
});
