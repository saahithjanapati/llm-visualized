// @vitest-environment jsdom

import { afterEach, describe, expect, it, vi } from 'vitest';
import { initTouchClickFallback } from './touchClickFallback.js';

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

describe('initTouchClickFallback', () => {
    it('suppresses a retargeted native click after a touch pointerdown activation', () => {
        document.body.innerHTML = `
            <div class="touch-root">
                <button type="button" class="close-btn">Close</button>
                <button type="button" class="exit-btn">Go to 3D</button>
            </div>
        `;
        const root = document.querySelector('.touch-root');
        const closeBtn = document.querySelector('.close-btn');
        const exitBtn = document.querySelector('.exit-btn');
        const closeSpy = vi.fn(() => {
            closeBtn.remove();
        });
        const exitSpy = vi.fn();
        const originalAddEventListener = root.addEventListener.bind(root);
        let clickHandler = null;
        root.addEventListener = (type, listener, options) => {
            if (type === 'click') {
                clickHandler = listener;
            }
            return originalAddEventListener(type, listener, options);
        };
        closeBtn.addEventListener('click', closeSpy);
        exitBtn.addEventListener('click', exitSpy);

        const cleanup = initTouchClickFallback(root, {
            selector: 'button',
            activateOnPointerDownSelector: 'button'
        });

        closeBtn.dispatchEvent(createTouchPointerEvent('pointerdown', {
            clientX: 22,
            clientY: 18
        }));

        expect(closeSpy).toHaveBeenCalledTimes(1);

        clickHandler?.({
            isTrusted: true,
            target: exitBtn,
            clientX: 22,
            clientY: 18,
            preventDefault: vi.fn(),
            stopPropagation: vi.fn()
        });

        expect(exitSpy).not.toHaveBeenCalled();

        cleanup();
    });

    it('does not suppress an unrelated trusted click away from the original touch point', () => {
        document.body.innerHTML = `
            <div class="touch-root">
                <button type="button" class="close-btn">Close</button>
                <button type="button" class="exit-btn">Go to 3D</button>
            </div>
        `;
        const root = document.querySelector('.touch-root');
        const closeBtn = document.querySelector('.close-btn');
        const exitBtn = document.querySelector('.exit-btn');
        const closeSpy = vi.fn();
        const exitSpy = vi.fn();
        const originalAddEventListener = root.addEventListener.bind(root);
        let clickHandler = null;
        root.addEventListener = (type, listener, options) => {
            if (type === 'click') {
                clickHandler = listener;
            }
            return originalAddEventListener(type, listener, options);
        };
        closeBtn.addEventListener('click', closeSpy);
        exitBtn.addEventListener('click', exitSpy);

        const cleanup = initTouchClickFallback(root, {
            selector: 'button',
            activateOnPointerDownSelector: 'button'
        });

        closeBtn.dispatchEvent(createTouchPointerEvent('pointerdown', {
            clientX: 22,
            clientY: 18
        }));

        expect(closeSpy).toHaveBeenCalledTimes(1);

        clickHandler?.({
            isTrusted: true,
            target: exitBtn,
            clientX: 200,
            clientY: 180,
            preventDefault: vi.fn(),
            stopPropagation: vi.fn()
        });

        exitBtn.click();

        expect(exitSpy).toHaveBeenCalledTimes(1);

        cleanup();
    });
});
