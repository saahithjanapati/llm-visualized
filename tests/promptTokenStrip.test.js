// @vitest-environment jsdom
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { resolveLogitTokenChipColorKey } from '../src/app/gpt-tower/logitColor.js';

function findCollidingTokenIds() {
    const seenByColorKey = new Map();
    for (let tokenId = 0; tokenId < 4096; tokenId += 1) {
        const colorKey = resolveLogitTokenChipColorKey({ tokenId, tokenLabel: `token-${tokenId}` }, tokenId);
        const previousTokenId = seenByColorKey.get(colorKey);
        if (previousTokenId !== undefined) {
            return [previousTokenId, tokenId];
        }
        seenByColorKey.set(colorKey, tokenId);
    }
    throw new Error('Expected to find at least one prompt-strip token-id color collision.');
}

describe('promptTokenStrip', () => {
    let promptTokenStrip = null;
    let initPromptTokenStrip = null;
    let appState = null;
    let originalLocalStorage = null;
    let originalMatchMedia = null;
    let originalResizeObserver = null;

    beforeEach(async () => {
        document.body.innerHTML = '';
        originalLocalStorage = globalThis.localStorage;
        originalMatchMedia = window.matchMedia;
        originalResizeObserver = globalThis.ResizeObserver;
        const storage = new Map();
        Object.defineProperty(globalThis, 'localStorage', {
            configurable: true,
            writable: true,
            value: {
                getItem(key) {
                    return storage.has(key) ? storage.get(key) : null;
                },
                setItem(key, value) {
                    storage.set(key, String(value));
                },
                removeItem(key) {
                    storage.delete(key);
                }
            }
        });
        Object.defineProperty(window, 'matchMedia', {
            configurable: true,
            writable: true,
            value: () => ({
                matches: false,
                addEventListener() {},
                removeEventListener() {},
                addListener() {},
                removeListener() {}
            })
        });
        Object.defineProperty(globalThis, 'ResizeObserver', {
            configurable: true,
            writable: true,
            value: class ResizeObserver {
                observe() {}
                disconnect() {}
            }
        });
        ({ appState } = await import('../src/state/appState.js'));
        ({ initPromptTokenStrip } = await import('../src/ui/promptTokenStrip.js'));
        appState.showPromptTokenStrip = true;
    });

    afterEach(() => {
        promptTokenStrip?.dispose?.();
        promptTokenStrip = null;
        initPromptTokenStrip = null;
        appState = null;
        document.body.innerHTML = '';
        if (originalLocalStorage) {
            Object.defineProperty(globalThis, 'localStorage', {
                configurable: true,
                writable: true,
                value: originalLocalStorage
            });
        } else {
            delete globalThis.localStorage;
        }
        if (originalMatchMedia) {
            Object.defineProperty(window, 'matchMedia', {
                configurable: true,
                writable: true,
                value: originalMatchMedia
            });
        } else {
            delete window.matchMedia;
        }
        if (originalResizeObserver) {
            Object.defineProperty(globalThis, 'ResizeObserver', {
                configurable: true,
                writable: true,
                value: originalResizeObserver
            });
        } else {
            delete globalThis.ResizeObserver;
        }
    });

    it('renders different chip colors for adjacent prompt tokens with different token ids', () => {
        const [leftTokenId, rightTokenId] = findCollidingTokenIds();
        promptTokenStrip = initPromptTokenStrip();

        promptTokenStrip.update({
            tokenLabels: ['alpha', 'beta'],
            tokenIndices: [0, 1],
            tokenIds: [leftTokenId, rightTokenId]
        });

        const chips = [...document.querySelectorAll('#promptTokenStrip .prompt-token-strip__token')];
        expect(chips).toHaveLength(2);
        expect(chips[0].style.getPropertyValue('--token-color-border')).not.toBe(
            chips[1].style.getPropertyValue('--token-color-border')
        );
    });

    it('keeps repeated adjacent token ids on the same chip color', () => {
        promptTokenStrip = initPromptTokenStrip();

        promptTokenStrip.update({
            tokenLabels: ['same', 'same'],
            tokenIndices: [0, 1],
            tokenIds: [42, 42]
        });

        const chips = [...document.querySelectorAll('#promptTokenStrip .prompt-token-strip__token')];
        expect(chips).toHaveLength(2);
        expect(chips[0].style.getPropertyValue('--token-color-border')).toBe(
            chips[1].style.getPropertyValue('--token-color-border')
        );
    });
});
