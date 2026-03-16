// @vitest-environment jsdom

import { afterEach, describe, expect, it, vi } from 'vitest';
import { initTransformerView2dTouchActionFallback } from './selectionPanelTransformerView2dTouchFallback.js';

function createTouchPointerEvent(type, {
    pointerId = 1,
    clientX = 16,
    clientY = 24
} = {}) {
    const event = new Event(type, {
        bubbles: true,
        cancelable: true
    });
    Object.defineProperties(event, {
        pointerId: { configurable: true, value: pointerId },
        pointerType: { configurable: true, value: 'touch' },
        clientX: { configurable: true, value: clientX },
        clientY: { configurable: true, value: clientY }
    });
    return event;
}

afterEach(() => {
    document.body.innerHTML = '';
});

describe('selectionPanelTransformerView2dTouchFallback', () => {
    it('activates 2D toolbar buttons on touch pointerdown', () => {
        document.body.innerHTML = `
            <section class="detail-transformer-view2d">
                <div class="detail-transformer-view2d-hud">
                    <button type="button" class="detail-transformer-view2d-action">Fit scene</button>
                </div>
            </section>
        `;
        const root = document.querySelector('.detail-transformer-view2d');
        const button = document.querySelector('.detail-transformer-view2d-action');
        const onClick = vi.fn();
        button.addEventListener('click', onClick);

        const cleanup = initTransformerView2dTouchActionFallback(root);
        button.dispatchEvent(createTouchPointerEvent('pointerdown'));

        expect(onClick).toHaveBeenCalledTimes(1);

        cleanup();
    });

    it('activates the selection sidebar close button on touch pointerdown', () => {
        document.body.innerHTML = `
            <section class="detail-transformer-view2d">
                <aside class="detail-transformer-view2d-selection-sidebar">
                    <button type="button" class="detail-transformer-view2d-selection-sidebar-close">×</button>
                </aside>
            </section>
        `;
        const root = document.querySelector('.detail-transformer-view2d');
        const button = document.querySelector('.detail-transformer-view2d-selection-sidebar-close');
        const onClick = vi.fn();
        button.addEventListener('click', onClick);

        const cleanup = initTransformerView2dTouchActionFallback(root);
        button.dispatchEvent(createTouchPointerEvent('pointerdown'));

        expect(onClick).toHaveBeenCalledTimes(1);

        cleanup();
    });

    it('activates shared history buttons in the selection sidebar on touch pointerdown', () => {
        document.body.innerHTML = `
            <section class="detail-transformer-view2d">
                <aside class="detail-transformer-view2d-selection-sidebar">
                    <div class="detail-history-nav">
                        <button type="button" class="detail-history-btn detail-history-btn--back">‹</button>
                    </div>
                </aside>
            </section>
        `;
        const root = document.querySelector('.detail-transformer-view2d');
        const button = document.querySelector('.detail-history-btn--back');
        const onClick = vi.fn();
        button.addEventListener('click', onClick);

        const cleanup = initTransformerView2dTouchActionFallback(root);
        button.dispatchEvent(createTouchPointerEvent('pointerdown'));

        expect(onClick).toHaveBeenCalledTimes(1);

        cleanup();
    });

    it('ignores non-2D-action buttons inside the view', () => {
        document.body.innerHTML = `
            <section class="detail-transformer-view2d">
                <div class="detail-transformer-view2d-hud">
                    <button type="button" class="other-action">Other</button>
                </div>
            </section>
        `;
        const root = document.querySelector('.detail-transformer-view2d');
        const button = document.querySelector('.other-action');
        const onClick = vi.fn();
        button.addEventListener('click', onClick);

        const cleanup = initTransformerView2dTouchActionFallback(root);
        button.dispatchEvent(createTouchPointerEvent('pointerdown'));

        expect(onClick).not.toHaveBeenCalled();

        cleanup();
    });
});
