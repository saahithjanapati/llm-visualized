// @vitest-environment jsdom
import { afterEach, beforeEach, describe, expect, it } from 'vitest';

describe('generationController generated token preview', () => {
    let originalLocalStorage = null;
    let resolveGeneratedLogitToken = null;

    beforeEach(async () => {
        originalLocalStorage = globalThis.localStorage;
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

        ({ resolveGeneratedLogitToken } = await import('../src/app/gpt-tower/generationController.js'));
    });

    afterEach(() => {
        resolveGeneratedLogitToken = null;
        if (originalLocalStorage) {
            Object.defineProperty(globalThis, 'localStorage', {
                configurable: true,
                writable: true,
                value: originalLocalStorage
            });
        } else {
            delete globalThis.localStorage;
        }
    });

    it('uses the actual next captured token when it is missing from the top-k logit row', () => {
        const activationSource = {
            getLogitTopK() {
                return 2;
            },
            getLogitsForToken() {
                return [
                    { token_id: 13, token: '.', prob: 0.9 },
                    { token_id: 0, token: '!', prob: 0.1 }
                ];
            },
            getTokenCount() {
                return 2;
            },
            getTokenString(tokenIndex) {
                return ['V', 'ect'][tokenIndex] ?? null;
            },
            getTokenId(tokenIndex) {
                return [53, 478][tokenIndex] ?? null;
            }
        };

        const result = resolveGeneratedLogitToken(activationSource, [0]);

        expect(result).toMatchObject({
            tokenLabel: 'ect',
            tokenId: 478,
            tokenIndex: 1,
            selectionLabel: 'Chosen token: ect',
            logitEntry: null
        });
    });

    it('keeps logit metadata when the actual next captured token is present in top-k', () => {
        const matchingEntry = { token_id: 478, token: 'ect', prob: 0.3 };
        const activationSource = {
            getLogitTopK() {
                return 3;
            },
            getLogitsForToken() {
                return [
                    { token_id: 13, token: '.', prob: 0.4 },
                    matchingEntry,
                    { token_id: 0, token: '!', prob: 0.3 }
                ];
            },
            getTokenCount() {
                return 2;
            },
            getTokenString(tokenIndex) {
                return ['V', 'ect'][tokenIndex] ?? null;
            },
            getTokenId(tokenIndex) {
                return [53, 478][tokenIndex] ?? null;
            }
        };

        const result = resolveGeneratedLogitToken(activationSource, [0]);

        expect(result?.tokenLabel).toBe('ect');
        expect(result?.tokenId).toBe(478);
        expect(result?.logitEntry).toBe(matchingEntry);
    });
});
