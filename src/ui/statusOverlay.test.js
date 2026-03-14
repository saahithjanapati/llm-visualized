// @vitest-environment jsdom

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

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

describe('statusOverlay KV cache link touch fallback', () => {
    let appState;
    let initStatusOverlay;
    let KV_CACHE_INFO_REQUEST_EVENT;

    beforeEach(async () => {
        vi.resetModules();
        vi.stubGlobal('localStorage', {
            getItem: vi.fn(() => null),
            setItem: vi.fn(),
            removeItem: vi.fn(),
            clear: vi.fn()
        });
        vi.stubGlobal('requestAnimationFrame', vi.fn(() => 1));
        ({ appState } = await import('../state/appState.js'));
        ({ initStatusOverlay } = await import('./statusOverlay.js'));
        ({ KV_CACHE_INFO_REQUEST_EVENT } = await import('./kvCacheInfoUtils.js'));
        document.body.innerHTML = `
            <div id="statusOverlay"></div>
            <section id="equationsPanel">
                <div id="equationsTitle"></div>
                <div id="equationsBody"></div>
            </section>
        `;
        appState.kvCacheModeEnabled = true;
        appState.kvCachePrefillActive = true;
        appState.showEquations = true;
        appState.equationsSuppressed = false;
    });

    afterEach(() => {
        document.body.innerHTML = '';
        vi.unstubAllGlobals();
    });

    it('opens KV cache details on touch pointerdown', () => {
        const pipeline = {
            _currentLayerIdx: 0,
            _layers: [],
            engine: {
                scene: {
                    traverse() {}
                }
            },
            addEventListener: vi.fn()
        };
        const openListener = vi.fn();
        window.addEventListener(KV_CACHE_INFO_REQUEST_EVENT, openListener);

        initStatusOverlay(pipeline, 12);

        const link = document.querySelector('.status-overlay__kv-link');
        expect(link).toBeTruthy();
        expect(link.hidden).toBe(false);

        link.dispatchEvent(createTouchPointerEvent('pointerdown'));

        expect(openListener).toHaveBeenCalledTimes(1);
        expect(openListener.mock.calls[0][0]?.detail).toEqual({ phase: 'prefill' });

        window.removeEventListener(KV_CACHE_INFO_REQUEST_EVENT, openListener);
    });
});

describe('statusOverlay layer norm equations', () => {
    let appState;
    let initStatusOverlay;

    beforeEach(async () => {
        vi.resetModules();
        vi.stubGlobal('localStorage', {
            getItem: vi.fn(() => null),
            setItem: vi.fn(),
            removeItem: vi.fn(),
            clear: vi.fn()
        });
        vi.stubGlobal('requestAnimationFrame', vi.fn(() => 1));
        vi.stubGlobal('ResizeObserver', class {
            observe() {}
            disconnect() {}
        });
        ({ appState } = await import('../state/appState.js'));
        ({ initStatusOverlay } = await import('./statusOverlay.js'));
        document.body.innerHTML = `
            <div id="statusOverlay"></div>
            <section id="equationsPanel">
                <div id="equationsTitle"></div>
                <div id="equationsBody"></div>
            </section>
        `;
        appState.showEquations = true;
        appState.equationsSuppressed = false;
    });

    afterEach(() => {
        document.body.innerHTML = '';
        vi.unstubAllGlobals();
    });

    it('renders LayerNorm 2 with x minus mu in the normalization term', () => {
        const render = vi.fn();
        window.katex = { render };
        const pipeline = {
            _currentLayerIdx: 0,
            _layers: [{
                index: 0,
                isActive: false,
                lanes: [{
                    ln2Phase: 'active',
                    mlpUpStarted: false,
                    stopRise: false,
                    horizPhase: 'waitingForLN2'
                }]
            }],
            engine: {
                scene: {
                    traverse() {}
                }
            },
            addEventListener: vi.fn()
        };

        initStatusOverlay(pipeline, 1);

        expect(render).toHaveBeenCalled();
        const [tex] = render.mock.calls.at(-1);
        expect(tex).toContain('\\frac{x - \\mu}{\\sqrt{\\sigma^2 + \\epsilon}}');
        expect(tex).not.toContain('\\frac{u - \\mu}{\\sqrt{\\sigma^2 + \\epsilon}}');
    });
});
