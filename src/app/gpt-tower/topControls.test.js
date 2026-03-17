// @vitest-environment jsdom

import { afterEach, describe, expect, it, vi } from 'vitest';

import { initFollowModeControls } from './topControls.js';

describe('initFollowModeControls', () => {
    const originalMatchMedia = window.matchMedia;

    afterEach(() => {
        window.matchMedia = originalMatchMedia;
    });

    it('uses a short mobile follow-mode label and updates it on resize', () => {
        let smallScreen = false;
        window.matchMedia = vi.fn((query) => ({
            matches: query === '(max-aspect-ratio: 1/1), (max-width: 880px)' ? smallScreen : false,
            media: query,
            addListener: vi.fn(),
            removeListener: vi.fn(),
            addEventListener: vi.fn(),
            removeEventListener: vi.fn(),
            dispatchEvent: vi.fn()
        }));

        const followModeBtn = document.createElement('button');
        const pipeline = {
            isAutoCameraFollowEnabled: vi.fn(() => false),
            setAutoCameraFollow: vi.fn(),
            engine: {
                controls: {
                    addEventListener: vi.fn(),
                    domElement: {
                        addEventListener: vi.fn()
                    }
                },
                camera: {}
            }
        };
        const appState = {
            autoCameraFollow: false
        };

        initFollowModeControls({
            pipeline,
            appState,
            followModeBtn
        });

        expect(followModeBtn.textContent).toBe('Enable Follow Mode');
        expect(followModeBtn.getAttribute('aria-label')).toBe('Enable follow mode');

        smallScreen = true;
        window.dispatchEvent(new Event('resize'));

        expect(followModeBtn.textContent).toBe('Follow Mode');
        expect(followModeBtn.getAttribute('aria-label')).toBe('Enable follow mode');
        expect(followModeBtn.getAttribute('title')).toBe('Enable follow mode');
    });

    it('keeps the short mobile label even when follow mode is already enabled', () => {
        window.matchMedia = vi.fn((query) => ({
            matches: query === '(max-aspect-ratio: 1/1), (max-width: 880px)',
            media: query,
            addListener: vi.fn(),
            removeListener: vi.fn(),
            addEventListener: vi.fn(),
            removeEventListener: vi.fn(),
            dispatchEvent: vi.fn()
        }));

        const followModeBtn = document.createElement('button');
        const pipeline = {
            isAutoCameraFollowEnabled: vi.fn(() => true),
            setAutoCameraFollow: vi.fn(),
            engine: {
                controls: {
                    addEventListener: vi.fn(),
                    domElement: {
                        addEventListener: vi.fn()
                    }
                },
                camera: {}
            }
        };
        const appState = {
            autoCameraFollow: true
        };

        initFollowModeControls({
            pipeline,
            appState,
            followModeBtn
        });

        expect(followModeBtn.textContent).toBe('Follow Mode');
        expect(followModeBtn.disabled).toBe(true);
        expect(followModeBtn.getAttribute('aria-label')).toBe('Follow mode enabled');
    });
});
