// @vitest-environment jsdom

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

let appState;
let initProjectInfoOverlay;
let PROJECT_INFO_OVERLAY_PAUSE_REASON;

function renderOverlayDom() {
    document.body.innerHTML = `
        <button id="trigger" type="button">Open info</button>
        <div id="projectInfoOverlay" aria-hidden="true" style="display:none;">
            <div class="project-info-modal" role="dialog" aria-modal="true" aria-labelledby="projectInfoOverlayTitle">
                <div class="project-info-modal-header">
                    <div class="project-info-modal-title-group">
                        <div class="project-info-modal-eyebrow">About the project</div>
                        <div id="projectInfoOverlayTitle" class="project-info-modal-title">LLM-Visualized</div>
                    </div>
                    <div class="project-info-modal-header-actions">
                        <button id="projectInfoClose" type="button">Back to visualization</button>
                    </div>
                </div>
                <div id="projectInfoOverlayContent" class="project-info-modal-content"></div>
            </div>
        </div>
    `;
}

describe('projectInfoOverlay', () => {
    beforeEach(async () => {
        vi.resetModules();
        vi.stubGlobal('localStorage', {
            getItem: vi.fn(() => null),
            setItem: vi.fn(),
            removeItem: vi.fn()
        });
        ({ appState } = await import('../state/appState.js'));
        ({
            initProjectInfoOverlay,
            PROJECT_INFO_OVERLAY_PAUSE_REASON
        } = await import('./projectInfoOverlay.js'));
        renderOverlayDom();
        window.history.replaceState({}, '', '/?view=2d&component=mhsa&layer=3');
        appState.modalPaused = false;
    });

    afterEach(() => {
        vi.restoreAllMocks();
    });

    it('opens as an in-scene overlay, pushes the info URL, and returns to the prior route on close', () => {
        const pause = vi.fn();
        const resume = vi.fn();
        const resetInteractionState = vi.fn();
        const pipeline = {
            engine: {
                pause,
                resume,
                resetInteractionState,
                controls: {
                    enabled: true,
                    enableRotate: true,
                    enablePan: true,
                    enableZoom: true
                },
                renderer: {
                    domElement: {
                        style: {}
                    }
                }
            }
        };
        const historyBackSpy = vi.spyOn(window.history, 'back').mockImplementation(() => {
            window.history.replaceState(window.history.state, '', '/?view=2d&component=mhsa&layer=3');
            window.dispatchEvent(new PopStateEvent('popstate'));
        });

        const controller = initProjectInfoOverlay({ pipeline });
        const trigger = document.getElementById('trigger');
        trigger?.focus();

        const opened = controller.open();

        expect(opened).toBe(true);
        expect(pause).toHaveBeenCalledWith(PROJECT_INFO_OVERLAY_PAUSE_REASON);
        expect(appState.modalPaused).toBe(true);
        expect(document.getElementById('projectInfoOverlay')?.style.display).toBe('flex');
        expect(document.getElementById('projectInfoOverlay')?.getAttribute('aria-hidden')).toBe('false');
        expect(document.getElementById('projectInfoOverlayContent')?.innerHTML).toContain('<h2>Overview</h2>');
        expect(window.location.pathname).toBe('/info/');
        expect(window.location.search).toBe('?returnTo=%2F%3Fview%3D2d%26component%3Dmhsa%26layer%3D3');
        expect(resetInteractionState).toHaveBeenCalledTimes(1);
        expect(pipeline.engine.controls.enabled).toBe(false);
        expect(pipeline.engine.renderer.domElement.style.pointerEvents).toBe('none');

        document.getElementById('projectInfoClose')?.dispatchEvent(new MouseEvent('click', { bubbles: true }));

        expect(historyBackSpy).toHaveBeenCalledTimes(1);
        expect(resume).toHaveBeenCalledWith(PROJECT_INFO_OVERLAY_PAUSE_REASON);
        expect(appState.modalPaused).toBe(false);
        expect(document.getElementById('projectInfoOverlay')?.style.display).toBe('none');
        expect(document.getElementById('projectInfoOverlay')?.getAttribute('aria-hidden')).toBe('true');
        expect(window.location.pathname).toBe('/');
        expect(window.location.search).toBe('?view=2d&component=mhsa&layer=3');
        expect(pipeline.engine.controls.enabled).toBe(true);
        expect(pipeline.engine.renderer.domElement.style.pointerEvents).toBe('');
        expect(resetInteractionState).toHaveBeenCalledTimes(2);
    });

    it('closes on Escape without clearing a prior modal pause state', () => {
        appState.modalPaused = true;
        const pause = vi.fn();
        const resume = vi.fn();
        const pipeline = {
            engine: {
                pause,
                resume,
                resetInteractionState: vi.fn(),
                controls: null,
                renderer: {
                    domElement: {
                        style: {}
                    }
                }
            }
        };
        vi.spyOn(window.history, 'back').mockImplementation(() => {
            window.history.replaceState(window.history.state, '', '/?view=2d&component=mhsa&layer=3');
            window.dispatchEvent(new PopStateEvent('popstate'));
        });

        const controller = initProjectInfoOverlay({ pipeline });
        controller.open();

        document.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true }));

        expect(controller.isOpen()).toBe(false);
        expect(resume).toHaveBeenCalledWith(PROJECT_INFO_OVERLAY_PAUSE_REASON);
        expect(appState.modalPaused).toBe(true);
    });

    it('reopens from browser history when the info route is revisited', () => {
        const pause = vi.fn();
        const pipeline = {
            engine: {
                pause,
                resume: vi.fn(),
                resetInteractionState: vi.fn(),
                controls: null,
                renderer: {
                    domElement: {
                        style: {}
                    }
                }
            }
        };

        const controller = initProjectInfoOverlay({ pipeline });

        window.history.pushState({}, '', '/info/?returnTo=%2F%3Fview%3D2d');
        window.dispatchEvent(new PopStateEvent('popstate'));

        expect(controller.isOpen()).toBe(true);
        expect(pause).toHaveBeenCalledWith(PROJECT_INFO_OVERLAY_PAUSE_REASON);
        expect(document.getElementById('projectInfoOverlay')?.getAttribute('aria-hidden')).toBe('false');
    });
});
