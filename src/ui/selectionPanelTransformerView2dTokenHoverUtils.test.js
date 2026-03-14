// @vitest-environment jsdom

import { afterEach, describe, expect, it, vi } from 'vitest';

import { TOKEN_CHIP_HOVER_SYNC_EVENT } from './tokenChipHoverSync.js';
import {
    createTransformerView2dTokenHoverSync,
    TRANSFORMER_VIEW2D_TOKEN_HOVER_SOURCE,
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
        expect(syncDetails.at(-1)).toEqual({
            active: true,
            source: TRANSFORMER_VIEW2D_TOKEN_HOVER_SOURCE,
            tokenIndex: 1,
            tokenId: 12,
            tokenLabel: 'world'
        });

        hoverSync.clearCanvasEntry();

        expect(secondChip.classList.contains('is-token-chip-active')).toBe(false);
        expect(secondChip.dataset.tokenActive).toBe('false');
        expect(syncDetails.at(-1)).toEqual({
            active: false,
            source: TRANSFORMER_VIEW2D_TOKEN_HOVER_SOURCE,
            tokenIndex: null,
            tokenId: null,
            tokenLabel: ''
        });

        hoverSync.dispose({ emit: false });
        window.removeEventListener(TOKEN_CHIP_HOVER_SYNC_EVENT, onSync);
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

        hoverSync.dispose({ emit: false });
    });
});
