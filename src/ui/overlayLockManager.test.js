// @vitest-environment jsdom

import { beforeEach, describe, expect, it, vi } from 'vitest';

describe('overlayLockManager', () => {
    beforeEach(() => {
        vi.resetModules();
        vi.stubGlobal('localStorage', {
            getItem: vi.fn(() => null),
            setItem: vi.fn(),
            removeItem: vi.fn()
        });
        document.body.innerHTML = '';
        document.body.style.overflow = '';
    });

    it('keeps modal UI state locked until the final nested release', async () => {
        const { appState } = await import('../state/appState.js');
        const { acquireModalUiLock } = await import('./overlayLockManager.js');

        appState.modalPaused = false;
        document.body.style.overflow = '';

        const releaseOuter = acquireModalUiLock();
        const releaseInner = acquireModalUiLock();

        expect(appState.modalPaused).toBe(true);
        expect(document.body.style.overflow).toBe('hidden');

        releaseOuter();

        expect(appState.modalPaused).toBe(true);
        expect(document.body.style.overflow).toBe('hidden');

        releaseInner();

        expect(appState.modalPaused).toBe(false);
        expect(document.body.style.overflow).toBe('');
    });

    it('restores the pre-existing modal UI state after nested locks release', async () => {
        const { appState } = await import('../state/appState.js');
        const { acquireModalUiLock } = await import('./overlayLockManager.js');

        appState.modalPaused = true;
        document.body.style.overflow = 'clip';

        const release = acquireModalUiLock();

        expect(appState.modalPaused).toBe(true);
        expect(document.body.style.overflow).toBe('hidden');

        release();

        expect(appState.modalPaused).toBe(true);
        expect(document.body.style.overflow).toBe('clip');
    });

    it('keeps scene interaction disabled until the final nested release', async () => {
        const { acquireSceneBackgroundInteractionLock } = await import('./overlayLockManager.js');

        const resetInteractionState = vi.fn();
        const engine = {
            resetInteractionState,
            controls: {
                enabled: true,
                enableRotate: true,
                enablePan: true,
                enableZoom: true
            },
            renderer: {
                domElement: {
                    style: {
                        pointerEvents: ''
                    }
                }
            }
        };

        const releaseOuter = acquireSceneBackgroundInteractionLock(engine);
        const releaseInner = acquireSceneBackgroundInteractionLock(engine);

        expect(engine.controls.enabled).toBe(false);
        expect(engine.controls.enableRotate).toBe(false);
        expect(engine.controls.enablePan).toBe(false);
        expect(engine.controls.enableZoom).toBe(false);
        expect(engine.renderer.domElement.style.pointerEvents).toBe('none');

        releaseOuter();

        expect(engine.controls.enabled).toBe(false);
        expect(engine.controls.enableRotate).toBe(false);
        expect(engine.controls.enablePan).toBe(false);
        expect(engine.controls.enableZoom).toBe(false);
        expect(engine.renderer.domElement.style.pointerEvents).toBe('none');

        releaseInner();

        expect(engine.controls.enabled).toBe(true);
        expect(engine.controls.enableRotate).toBe(true);
        expect(engine.controls.enablePan).toBe(true);
        expect(engine.controls.enableZoom).toBe(true);
        expect(engine.renderer.domElement.style.pointerEvents).toBe('');
        expect(resetInteractionState).toHaveBeenCalledTimes(4);
    });
});
