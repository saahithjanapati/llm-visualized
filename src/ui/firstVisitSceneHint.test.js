// @vitest-environment jsdom

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

function createMockLocalStorage() {
    const store = new Map();
    return {
        getItem: vi.fn((key) => (store.has(key) ? store.get(key) : null)),
        setItem: vi.fn((key, value) => {
            store.set(String(key), String(value));
        }),
        removeItem: vi.fn((key) => {
            store.delete(String(key));
        }),
        clear: vi.fn(() => {
            store.clear();
        })
    };
}

describe('firstVisitSceneHint', () => {
    let initFirstVisitSceneHint;

    beforeEach(async () => {
        vi.resetModules();
        vi.useFakeTimers();
        vi.stubGlobal('localStorage', createMockLocalStorage());
        ({ initFirstVisitSceneHint } = await import('./firstVisitSceneHint.js'));
    });

    afterEach(() => {
        document.body.innerHTML = '';
        vi.useRealTimers();
        vi.unstubAllGlobals();
    });

    it('shows once, auto-hides after five seconds, and persists that it was shown', () => {
        const hint = initFirstVisitSceneHint();

        expect(document.getElementById('firstVisitSceneHint')?.dataset.visible).toBe('false');
        expect(hint.showIfEligible()).toBe(true);

        const root = document.getElementById('firstVisitSceneHint');
        expect(root?.textContent).toBe('Be sure to click around on different components to learn more about them.');
        expect(root?.dataset.visible).toBe('true');
        expect(root?.getAttribute('aria-hidden')).toBe('false');

        vi.advanceTimersByTime(5000);

        expect(root?.dataset.visible).toBe('false');
        expect(root?.getAttribute('aria-hidden')).toBe('true');
        expect(JSON.parse(localStorage.getItem('firstVisitSceneHintShown'))).toBe(true);
    });

    it('does not show again after the first visit has been recorded', () => {
        localStorage.setItem('firstVisitSceneHintShown', JSON.stringify(true));
        const hint = initFirstVisitSceneHint();

        expect(hint.showIfEligible()).toBe(false);
        expect(document.getElementById('firstVisitSceneHint')?.dataset.visible).toBe('false');
    });
});
