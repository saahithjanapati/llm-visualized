// @vitest-environment jsdom

import { afterEach, describe, expect, it, vi } from 'vitest';
import {
    initTransformerView2dTouchActionFallback,
    TRANSFORMER_VIEW2D_TOUCH_ACTION_SELECTOR
} from './selectionPanelTransformerView2dTouchFallback.js';

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
            <div class="detail-transformer-view2d-hud">
                <button type="button" class="detail-transformer-view2d-action">Fit scene</button>
            </div>
        `;
        const hud = document.querySelector('.detail-transformer-view2d-hud');
        const button = document.querySelector(TRANSFORMER_VIEW2D_TOUCH_ACTION_SELECTOR);
        const onClick = vi.fn();
        button.addEventListener('click', onClick);

        const cleanup = initTransformerView2dTouchActionFallback(hud);
        button.dispatchEvent(createTouchPointerEvent('pointerdown'));

        expect(onClick).toHaveBeenCalledTimes(1);

        cleanup();
    });

    it('ignores non-toolbar targets', () => {
        document.body.innerHTML = `
            <div class="detail-transformer-view2d-hud">
                <button type="button" class="other-action">Other</button>
            </div>
        `;
        const hud = document.querySelector('.detail-transformer-view2d-hud');
        const button = document.querySelector('.other-action');
        const onClick = vi.fn();
        button.addEventListener('click', onClick);

        const cleanup = initTransformerView2dTouchActionFallback(hud);
        button.dispatchEvent(createTouchPointerEvent('pointerdown'));

        expect(onClick).not.toHaveBeenCalled();

        cleanup();
    });
});
