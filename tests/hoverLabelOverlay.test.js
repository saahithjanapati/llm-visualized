import { afterEach, describe, expect, it } from 'vitest';
import { JSDOM } from 'jsdom';
import { createHoverLabelOverlay } from '../src/ui/hoverLabelOverlay.js';

describe('hoverLabelOverlay', () => {
    afterEach(() => {
        globalThis.window?.close?.();
        delete globalThis.window;
        delete globalThis.document;
    });

    it('renders source and target token rows for attention-score hovers', () => {
        const dom = new JSDOM('<!doctype html><html><body></body></html>');
        globalThis.window = dom.window;
        globalThis.document = dom.window.document;

        const overlay = createHoverLabelOverlay({
            documentRef: dom.window.document,
            parent: dom.window.document.body
        });

        const shown = overlay.show({
            clientX: 24,
            clientY: 32,
            label: 'Pre-Softmax Attention Score',
            info: {
                activationData: {
                    stage: 'attention.pre',
                    tokenIndex: 1,
                    tokenLabel: 'beta',
                    keyTokenIndex: 0,
                    keyTokenLabel: 'alpha',
                    headIndex: 2,
                    layerIndex: 0
                }
            },
            activationSource: {
                getTokenId: (tokenIndex) => (tokenIndex === 1 ? 22 : (tokenIndex === 0 ? 11 : null))
            }
        });

        expect(shown).toBe(true);

        const tooltip = dom.window.document.body.querySelector('.scene-hover-label');
        const rows = Array.from(tooltip?.querySelectorAll('.scene-hover-label__attention-row') || []);

        expect(tooltip?.style.display).toBe('block');
        expect(tooltip?.querySelector('.scene-hover-label__text')?.textContent).toBe('Pre-Softmax Attention Score');
        expect(tooltip?.querySelector('.scene-hover-label__subtitle')?.textContent).toBe('Head 3 • Layer 1');
        expect(rows).toHaveLength(2);
        expect(rows[0]?.querySelector('.scene-hover-label__attention-role')?.textContent).toBe('Source');
        expect(rows[0]?.querySelector('.scene-hover-label__attention-chip')?.textContent).toContain('beta');
        expect(rows[0]?.querySelector('.scene-hover-label__attention-position')?.textContent).toBe('(Position 2)');
        expect(rows[1]?.querySelector('.scene-hover-label__attention-role')?.textContent).toBe('Target');
        expect(rows[1]?.querySelector('.scene-hover-label__attention-chip')?.textContent).toContain('alpha');
        expect(rows[1]?.querySelector('.scene-hover-label__attention-position')?.textContent).toBe('(Position 1)');
    });
});
