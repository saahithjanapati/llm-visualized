import { beforeAll, describe, expect, it, vi } from 'vitest';

let resolveLogitPreviewTokenText;

beforeAll(async () => {
    vi.stubGlobal('localStorage', {
        getItem: () => null,
        setItem: () => {},
        removeItem: () => {}
    });
    ({ resolveLogitPreviewTokenText } = await import('./selectionPanel.js'));
});

describe('resolveLogitPreviewTokenText', () => {
    it('prefers an explicit token label over a numeric token id fallback', () => {
        const tokenText = resolveLogitPreviewTokenText('Chosen token: hello', {
            info: {
                tokenLabel: 'hello',
                logitEntry: {
                    token_id: 31337,
                    prob: 0.42
                }
            }
        });

        expect(tokenText).toBe('hello');
    });

    it('falls back to the chosen-token label text before rendering a numeric id', () => {
        const tokenText = resolveLogitPreviewTokenText('Chosen token: world', {
            info: {
                logitEntry: {
                    token_id: 4242,
                    prob: 0.15
                }
            }
        });

        expect(tokenText).toBe('world');
    });

    it('falls back to the logit hover token text before rendering a numeric id', () => {
        const tokenText = resolveLogitPreviewTokenText('Logit\nToken "Beta"\nID 9001', {
            info: {
                logitEntry: {
                    token_id: 9001,
                    prob: 0.08
                }
            }
        });

        expect(tokenText).toBe('Beta');
    });
});
