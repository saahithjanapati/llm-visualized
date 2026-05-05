// @vitest-environment jsdom

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

function createMockLocalStorage() {
    const store = new Map();
    return {
        getItem: vi.fn((key) => (store.has(String(key)) ? store.get(String(key)) : null)),
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

function createPipelineMock() {
    return {
        setAutoCameraFollow: vi.fn(),
        setDevMode: vi.fn(),
        engine: {
            pause: vi.fn(),
            resume: vi.fn(),
            setCameraDebugEnabled: vi.fn(),
            setDevMode: vi.fn(),
            renderer: {
                domElement: document.createElement('canvas')
            }
        }
    };
}

function installSettingsDom() {
    document.body.innerHTML = `
        <button id="settingsBtn" type="button">Settings</button>
        <div id="settingsOverlay" aria-hidden="true">
            <div class="settings-modal">
                <button id="settingsClose" type="button">Close</button>
                <label class="toggle-row">
                    <span class="toggle-copy">
                        <span class="toggle-text">Enable KV cache mode</span>
                        <span class="toggle-description">Caches attention keys and values.</span>
                    </span>
                    <input id="toggleKvCacheMode" type="checkbox">
                    <span class="toggle-track" aria-hidden="true"></span>
                </label>
                <div id="kvCacheStatusHint" class="settings-hint" hidden>KV cache enabled for prefill and decode.</div>
            </div>
        </div>
        <div id="equationsPanel"></div>
    `;
}

describe('settingsModal KV cache mode toggle', () => {
    let appState;
    let initSettingsModal;
    let kvCacheModeChangedEvent;

    beforeEach(async () => {
        vi.resetModules();
        vi.stubGlobal('localStorage', createMockLocalStorage());
        vi.useFakeTimers();
        const requestAnimationFrameMock = vi.fn((callback) => (
            setTimeout(() => callback(16), 16)
        ));
        const cancelAnimationFrameMock = vi.fn((id) => clearTimeout(id));
        vi.stubGlobal('requestAnimationFrame', requestAnimationFrameMock);
        vi.stubGlobal('cancelAnimationFrame', cancelAnimationFrameMock);
        window.setTimeout = setTimeout;
        window.clearTimeout = clearTimeout;
        window.requestAnimationFrame = requestAnimationFrameMock;
        window.cancelAnimationFrame = cancelAnimationFrameMock;
        ({ appState } = await import('../state/appState.js'));
        ({ KV_CACHE_MODE_CHANGED_EVENT: kvCacheModeChangedEvent } = await import('../state/kvCacheModeEvents.js'));
        ({ initSettingsModal } = await import('./settingsModal.js'));
        localStorage.clear();
        appState.kvCacheModeEnabled = false;
        appState.kvCachePrefillActive = false;
        appState.kvCachePassIndex = 0;
        installSettingsDom();
    });

    afterEach(() => {
        document.body.innerHTML = '';
        localStorage.clear();
        if (appState) {
            appState.kvCacheModeEnabled = false;
            appState.kvCachePrefillActive = false;
            appState.kvCachePassIndex = 0;
        }
        vi.useRealTimers();
        vi.unstubAllGlobals();
    });

    it('paints a switching state before dispatching the heavy KV cache mode change', () => {
        initSettingsModal(createPipelineMock());
        const changes = [];
        const onChange = (event) => {
            changes.push(event.detail);
        };
        window.addEventListener(kvCacheModeChangedEvent, onChange);

        const toggle = document.getElementById('toggleKvCacheMode');
        const row = toggle.closest('.toggle-row');
        const hint = document.getElementById('kvCacheStatusHint');

        toggle.checked = true;
        toggle.dispatchEvent(new Event('change', { bubbles: true }));

        expect(toggle.checked).toBe(true);
        expect(row.dataset.switching).toBe('true');
        expect(row.getAttribute('aria-busy')).toBe('true');
        expect(hint.hidden).toBe(false);
        expect(hint.dataset.state).toBe('switching');
        expect(hint.textContent).toBe('Switching to KV cache mode...');
        expect(changes).toHaveLength(0);

        const transitionOverlay = document.getElementById('kvCacheModeTransitionOverlay');
        expect(transitionOverlay?.dataset.visible).toBe('true');
        expect(transitionOverlay?.textContent).toContain('Switching to KV cache mode');

        vi.runOnlyPendingTimers();
        expect(changes).toHaveLength(0);
        expect(transitionOverlay?.dataset.visible).toBe('true');

        vi.runOnlyPendingTimers();

        expect(changes).toEqual([{
            enabled: true,
            previousEnabled: false
        }]);
        expect(row.dataset.switching).toBe('true');
        expect(transitionOverlay?.dataset.visible).toBe('true');

        vi.runOnlyPendingTimers();
        vi.runOnlyPendingTimers();

        expect(row.dataset.switching).toBeUndefined();
        expect(row.hasAttribute('aria-busy')).toBe(false);
        expect(hint.dataset.state).toBeUndefined();
        expect(hint.textContent).toBe('KV cache enabled for prefill and decode.');
        expect(hint.hidden).toBe(false);
        expect(transitionOverlay?.dataset.visible).toBe('false');

        window.removeEventListener(kvCacheModeChangedEvent, onChange);
    });
});
