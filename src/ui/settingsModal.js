import {
    clampPlaybackSpeedPercent,
    DEFAULT_PLAYBACK_SPEED_PERCENT,
    PLAYBACK_SPEED_PERCENT_MAX,
    PLAYBACK_SPEED_PERCENT_MIN,
    setPlaybackSpeed
} from '../utils/constants.js';
import { getPreference, setPreference } from '../utils/preferences.js';
import { appState } from '../state/appState.js';
import {
    KV_CACHE_MODE_STATE_SYNC_EVENT,
    dispatchKvCacheModeChanged
} from '../state/kvCacheModeEvents.js';
import { ENVIRONMENT_MAP_OPTIONS } from '../utils/environmentMaps.js';
import { initPerfOverlay } from './perfOverlay.js';
import { createModalReopenGuard } from './modalReopenGuard.js';
import { initTouchClickFallback } from './touchClickFallback.js';

const BRIGHTNESS_PREF_KEY = 'displayBrightnessScale';
const PLAYBACK_SPEED_PREF_KEY = 'playbackSpeedPercent';
const PROMPT_TOKEN_STRIP_PREF_KEY = 'showPromptTokenStrip';
const BRIGHTNESS_MIN = 0.5;
const BRIGHTNESS_MAX = 1.8;
const BRIGHTNESS_DEFAULT = 1.2;
const BRIGHTNESS_UI_BASELINE = 1.2;
const BRIGHTNESS_UI_SCALE = 100;
const BRIGHTNESS_UI_MIN = Math.round((BRIGHTNESS_MIN / BRIGHTNESS_UI_BASELINE) * BRIGHTNESS_UI_SCALE);
const BRIGHTNESS_UI_MAX = Math.round((BRIGHTNESS_MAX / BRIGHTNESS_UI_BASELINE) * BRIGHTNESS_UI_SCALE);
const BRIGHTNESS_UI_DEFAULT = Math.round((BRIGHTNESS_DEFAULT / BRIGHTNESS_UI_BASELINE) * BRIGHTNESS_UI_SCALE);

function clampBrightness(value) {
    const next = Number(value);
    if (!Number.isFinite(next)) return BRIGHTNESS_DEFAULT;
    return Math.max(BRIGHTNESS_MIN, Math.min(BRIGHTNESS_MAX, next));
}

function clampBrightnessUi(value) {
    const next = Number(value);
    if (!Number.isFinite(next)) return BRIGHTNESS_UI_DEFAULT;
    return Math.max(BRIGHTNESS_UI_MIN, Math.min(BRIGHTNESS_UI_MAX, next));
}

function brightnessInternalToUi(value) {
    return Math.round((clampBrightness(value) / BRIGHTNESS_UI_BASELINE) * BRIGHTNESS_UI_SCALE);
}

function brightnessUiToInternal(value) {
    return clampBrightness((clampBrightnessUi(value) / BRIGHTNESS_UI_SCALE) * BRIGHTNESS_UI_BASELINE);
}

function formatBrightness(value) {
    return `${brightnessInternalToUi(value)}%`;
}

function formatPlaybackSpeed(value) {
    return `${clampPlaybackSpeedPercent(value)}%`;
}

// Wires up the settings modal controls.
export function initSettingsModal(pipeline, {
    initialKvCacheModeEnabled = false
} = {}) {
    const settingsBtn = document.getElementById('settingsBtn');
    const settingsOverlay = document.getElementById('settingsOverlay');
    const settingsClose = document.getElementById('settingsClose');
    const equationsPanel = document.getElementById('equationsPanel');
    const settingsModal = settingsOverlay?.querySelector('.settings-modal') || null;
    const brightnessSlider = document.getElementById('brightnessSlider');
    const brightnessValue = document.getElementById('brightnessValue');
    const brightnessInput = document.getElementById('brightnessInput');
    const playbackSpeedSlider = document.getElementById('playbackSpeedSlider');
    const playbackSpeedValue = document.getElementById('playbackSpeedValue');
    const playbackSpeedInput = document.getElementById('playbackSpeedInput');
    const environmentMapSelect = document.getElementById('environmentMapSelect');
    if (playbackSpeedSlider) {
        playbackSpeedSlider.min = String(PLAYBACK_SPEED_PERCENT_MIN);
        playbackSpeedSlider.max = String(PLAYBACK_SPEED_PERCENT_MAX);
        playbackSpeedSlider.step = '1';
    }
    if (playbackSpeedInput) {
        playbackSpeedInput.min = String(PLAYBACK_SPEED_PERCENT_MIN);
        playbackSpeedInput.max = String(PLAYBACK_SPEED_PERCENT_MAX);
        playbackSpeedInput.step = '1';
    }
    if (brightnessSlider) {
        brightnessSlider.min = String(BRIGHTNESS_UI_MIN);
        brightnessSlider.max = String(BRIGHTNESS_UI_MAX);
        brightnessSlider.step = '1';
    }
    if (brightnessInput) {
        brightnessInput.min = String(BRIGHTNESS_UI_MIN);
        brightnessInput.max = String(BRIGHTNESS_UI_MAX);
        brightnessInput.step = '1';
    }

    if (environmentMapSelect && environmentMapSelect.options.length === 0) {
        ENVIRONMENT_MAP_OPTIONS.forEach((option) => {
            const el = document.createElement('option');
            el.value = option.key;
            el.textContent = option.label;
            environmentMapSelect.appendChild(el);
        });
    }

    const reopenGuard = createModalReopenGuard();

    initTouchClickFallback(settingsModal, {
        selector: 'button, .toggle-row',
        activateOnPointerDownSelector: '#settingsClose'
    });

    appState.autoCameraFollow = getPreference('autoCameraFollow', true);
    appState.showCameraDebug = getPreference('showCameraDebug', false);
    appState.showFollowViewInspector = false;
    appState.devMode = getPreference('devMode', false);
    appState.showPerfOverlay = getPreference('showPerfOverlay', false);
    appState.showPromptTokenStrip = getPreference(PROMPT_TOKEN_STRIP_PREF_KEY, true);
    // Do not restore sticky persisted KV mode. Only an explicit route should
    // be able to start the app with KV cache enabled.
    appState.kvCacheModeEnabled = !!initialKvCacheModeEnabled;
    setPreference('kvCacheModeEnabled', false);
    pipeline?.setAutoCameraFollow?.(appState.autoCameraFollow, { immediate: true });
    pipeline?.engine?.setCameraDebugEnabled?.(appState.showCameraDebug);
    pipeline?.engine?.setDevMode?.(appState.devMode);
    pipeline?.setDevMode?.(appState.devMode);

    let followInspectorEl = null;
    let followInspectorRaf = null;
    let perfOverlayController = null;

    const setPerfOverlayEnabled = (enabled) => {
        const nextValue = !!enabled;
        if (nextValue === appState.showPerfOverlay && ((nextValue && perfOverlayController) || (!nextValue && !perfOverlayController))) {
            return;
        }
        appState.showPerfOverlay = nextValue;
        setPreference('showPerfOverlay', appState.showPerfOverlay);
        if (appState.showPerfOverlay) {
            if (!perfOverlayController) {
                perfOverlayController = initPerfOverlay();
            }
        } else if (perfOverlayController) {
            perfOverlayController.dispose();
            perfOverlayController = null;
        }
    };

    const ensureFollowInspector = () => {
        if (followInspectorEl) return followInspectorEl;
        const el = document.createElement('div');
        el.id = 'followViewInspector';
        Object.assign(el.style, {
            position: 'fixed',
            left: '10px',
            bottom: '70px',
            padding: '10px 12px',
            fontFamily: 'monospace',
            fontSize: '12px',
            lineHeight: '1.35',
            color: '#fff',
            background: 'rgba(12,12,12,0.55)',
            border: '1px solid rgba(255,255,255,0.18)',
            borderRadius: '10px',
            backdropFilter: 'blur(6px)',
            WebkitBackdropFilter: 'blur(6px)',
            whiteSpace: 'pre-wrap',
            wordBreak: 'break-word',
            maxWidth: 'min(360px, 90vw)',
            pointerEvents: 'none',
            zIndex: 6,
            display: 'none'
        });
        document.body.appendChild(el);
        followInspectorEl = el;
        return el;
    };

    const formatNum = (value) => (Number.isFinite(value) ? value.toFixed(2) : '—');
    const formatVec = (v) => (v ? `(${formatNum(v.x)}, ${formatNum(v.y)}, ${formatNum(v.z)})` : '(—, —, —)');

    const updateFollowInspector = () => {
        if (!appState.showFollowViewInspector) {
            if (followInspectorEl) followInspectorEl.style.display = 'none';
            followInspectorRaf = null;
            return;
        }

        const el = ensureFollowInspector();
        const cam = pipeline?.engine?.camera;
        const target = pipeline?.engine?.controls?.target || null;
        const refInfo = pipeline?.getAutoCameraReference?.();
        const ref = refInfo?.position || null;
        const laneLabel = Number.isFinite(refInfo?.laneIndex) ? refInfo.laneIndex + 1 : '—';

        let camOffset = null;
        let targetOffset = null;
        if (cam && ref) {
            camOffset = {
                x: cam.position.x - ref.x,
                y: cam.position.y - ref.y,
                z: cam.position.z - ref.z
            };
        }
        if (target && ref) {
            targetOffset = {
                x: target.x - ref.x,
                y: target.y - ref.y,
                z: target.z - ref.z
            };
        }

        const enabled = pipeline?.isAutoCameraFollowEnabled?.() ? 'on' : 'off';
        el.textContent = [
            'Follow View Inspector',
            `Follow mode: ${enabled}`,
            `Lane ref: ${laneLabel}`,
            `Ref: ${formatVec(ref)}`,
            `Cam pos: ${cam ? formatVec(cam.position) : '(—, —, —)'}`,
            `Cam offset: ${formatVec(camOffset)}`,
            `Target: ${formatVec(target)}`,
            `Target offset: ${formatVec(targetOffset)}`
        ].join('\n');
        el.style.display = 'block';
        followInspectorRaf = requestAnimationFrame(updateFollowInspector);
    };

    const setFollowInspectorEnabled = (enabled) => {
        appState.showFollowViewInspector = !!enabled;
        if (appState.showFollowViewInspector) {
            if (followInspectorRaf === null) {
                followInspectorRaf = requestAnimationFrame(updateFollowInspector);
            }
        } else {
            if (followInspectorRaf !== null && typeof cancelAnimationFrame === 'function') {
                cancelAnimationFrame(followInspectorRaf);
                followInspectorRaf = null;
            }
            if (followInspectorEl) followInspectorEl.style.display = 'none';
        }
    };

    const updateKvCacheStatusHint = (enabled) => {
        const hint = document.getElementById('kvCacheStatusHint');
        if (!hint) return;
        hint.hidden = !enabled;
    };

    const syncKvCacheModeUi = (enabled) => {
        const kvCacheModeToggle = document.getElementById('toggleKvCacheMode');
        if (kvCacheModeToggle) kvCacheModeToggle.checked = !!enabled;
        updateKvCacheStatusHint(enabled);
    };

    const initInlinePercentEditor = ({
        buttonEl,
        inputEl,
        readValue,
        commitValue
    }) => {
        const shell = inputEl?.closest('.slider-value-input-shell') || null;
        if (!buttonEl || !inputEl || !shell) return;

        let editing = false;

        const closeEditor = ({ commit = false, restoreFocus = false } = {}) => {
            if (!editing) return;
            editing = false;

            if (commit) {
                const raw = inputEl.value;
                if (raw !== '') {
                    commitValue(raw);
                }
            } else {
                inputEl.value = String(readValue());
            }

            shell.hidden = true;
            buttonEl.hidden = false;
            if (restoreFocus) {
                requestAnimationFrame(() => {
                    if (document.body.contains(buttonEl)) {
                        buttonEl.focus();
                    }
                });
            }
        };

        buttonEl.addEventListener('click', () => {
            if (editing) return;
            editing = true;
            inputEl.value = String(readValue());
            buttonEl.hidden = true;
            shell.hidden = false;
            requestAnimationFrame(() => {
                inputEl.focus();
                inputEl.select();
            });
        });

        inputEl.addEventListener('keydown', (event) => {
            if (event.key === 'Enter') {
                event.preventDefault();
                closeEditor({ commit: true, restoreFocus: true });
            } else if (event.key === 'Escape') {
                event.preventDefault();
                closeEditor({ commit: false, restoreFocus: true });
            }
        });

        inputEl.addEventListener('blur', () => {
            closeEditor({ commit: true });
        });
    };

    const applyBrightness = (value, { persist = true, valueIsUi = false } = {}) => {
        const next = valueIsUi ? brightnessUiToInternal(value) : clampBrightness(value);
        const brightnessCss = `brightness(${next.toFixed(2)})`;
        const uiValue = brightnessInternalToUi(next);

        if (brightnessSlider) brightnessSlider.value = String(uiValue);
        if (brightnessValue) brightnessValue.textContent = formatBrightness(next);
        if (brightnessInput && document.activeElement !== brightnessInput) {
            brightnessInput.value = String(uiValue);
        }

        const gptCanvas = document.getElementById('gptCanvas');
        const introCanvas = document.getElementById('introCanvas');
        const engineCanvas = pipeline?.engine?.renderer?.domElement || null;
        const targets = [gptCanvas, introCanvas, engineCanvas];
        const seen = new Set();
        targets.forEach((canvas) => {
            if (!canvas || seen.has(canvas)) return;
            seen.add(canvas);
            canvas.style.filter = brightnessCss;
        });

        if (persist) {
            setPreference(BRIGHTNESS_PREF_KEY, next);
        }
    };

    const applyPlaybackSpeed = (value, { persist = true } = {}) => {
        const next = clampPlaybackSpeedPercent(value);
        const profile = setPlaybackSpeed(next);

        if (playbackSpeedSlider) playbackSpeedSlider.value = String(next);
        if (playbackSpeedValue) playbackSpeedValue.textContent = formatPlaybackSpeed(next);
        if (playbackSpeedInput && document.activeElement !== playbackSpeedInput) {
            playbackSpeedInput.value = String(next);
        }
        if (profile && typeof profile.engineSpeed === 'number') {
            pipeline?.engine?.setSpeed?.(profile.engineSpeed);
        }

        if (persist) {
            setPreference(PLAYBACK_SPEED_PREF_KEY, next);
        }
    };

    const isSettingsOpen = () => settingsOverlay?.getAttribute('aria-hidden') === 'false';

    function openSettings() {
        if (!settingsOverlay || isSettingsOpen()) return;
        if (!reopenGuard.shouldAllowOpen()) return;
        settingsOverlay.style.display = 'flex';
        settingsOverlay.setAttribute('aria-hidden', 'false');
        document.body.style.overflow = 'hidden';
        pipeline?.engine?.pause?.('modal');
        appState.modalPaused = true;
        const eq = document.getElementById('toggleEquations');
        if (eq) eq.checked = !!appState.showEquations;
        const promptTokenStrip = document.getElementById('togglePromptTokenStrip');
        if (promptTokenStrip) promptTokenStrip.checked = !!appState.showPromptTokenStrip;
        const bg = document.getElementById('toggleHdrBackground');
        if (bg) bg.checked = !!appState.showHdrBackground;
        if (environmentMapSelect) {
            environmentMapSelect.value = appState.selectedEnvironmentKey;
        }
        const devMode = document.getElementById('toggleDevMode');
        if (devMode) devMode.checked = !!appState.devMode;
        const camDebug = document.getElementById('toggleCameraDebug');
        if (camDebug) camDebug.checked = !!appState.showCameraDebug;
        const followInspector = document.getElementById('toggleFollowViewInspector');
        if (followInspector) followInspector.checked = !!appState.showFollowViewInspector;
        const perfOverlay = document.getElementById('togglePerfOverlay');
        if (perfOverlay) perfOverlay.checked = !!appState.showPerfOverlay;
        const kvCacheModeToggle = document.getElementById('toggleKvCacheMode');
        if (kvCacheModeToggle) kvCacheModeToggle.checked = !!appState.kvCacheModeEnabled;
        updateKvCacheStatusHint(appState.kvCacheModeEnabled);
    }

    function closeSettings({ guardReopen = false } = {}) {
        if (!settingsOverlay || !isSettingsOpen()) return;
        if (guardReopen) {
            // On mobile, the close tap can fall through to the opener after the
            // overlay hides. Block that one immediate reopen attempt.
            reopenGuard.markClosed();
        }
        settingsOverlay.style.display = 'none';
        settingsOverlay.setAttribute('aria-hidden', 'true');
        document.body.style.overflow = '';
        pipeline?.engine?.resume?.('modal');
        appState.modalPaused = false;
    }

    settingsBtn?.addEventListener('click', openSettings);
    settingsClose?.addEventListener('click', () => {
        closeSettings({ guardReopen: true });
    });
    settingsOverlay?.addEventListener('click', (e) => {
        if (e.target === settingsOverlay) closeSettings({ guardReopen: true });
    });
    window.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && isSettingsOpen()) {
            closeSettings();
        }
    });

    const eqToggle = document.getElementById('toggleEquations');
    eqToggle?.addEventListener('change', () => {
        appState.showEquations = !!eqToggle.checked;
        setPreference('showEquations', appState.showEquations);
        if (equationsPanel) {
            equationsPanel.style.display = (appState.showEquations && !appState.equationsSuppressed)
                ? 'block'
                : 'none';
        }
        appState.lastEqKey = '';
    });

    const promptTokenStripToggle = document.getElementById('togglePromptTokenStrip');
    promptTokenStripToggle?.addEventListener('change', () => {
        appState.showPromptTokenStrip = !!promptTokenStripToggle.checked;
        setPreference(PROMPT_TOKEN_STRIP_PREF_KEY, appState.showPromptTokenStrip);
        if (typeof window !== 'undefined' && typeof window.dispatchEvent === 'function') {
            window.dispatchEvent(new CustomEvent('promptTokenStripVisibilityChanged', {
                detail: { enabled: appState.showPromptTokenStrip }
            }));
        }
    });

    const bgToggle = document.getElementById('toggleHdrBackground');
    bgToggle?.addEventListener('change', () => {
        appState.showHdrBackground = !!bgToggle.checked;
        setPreference('showHdrBackground', appState.showHdrBackground);
        appState.applyEnvironmentBackground(pipeline);
    });

    environmentMapSelect?.addEventListener('change', async () => {
        const previousValue = appState.selectedEnvironmentKey;
        const nextValue = environmentMapSelect.value;
        environmentMapSelect.disabled = true;
        try {
            await appState.setEnvironmentKey(nextValue, pipeline);
        } catch (err) {
            console.warn('Failed to switch environment map:', err);
            environmentMapSelect.value = previousValue;
        } finally {
            environmentMapSelect.disabled = false;
        }
    });

    const devToggle = document.getElementById('toggleDevMode');
    devToggle?.addEventListener('change', () => {
        appState.devMode = !!devToggle.checked;
        setPreference('devMode', appState.devMode);
        pipeline?.engine?.setDevMode?.(appState.devMode);
        pipeline?.setDevMode?.(appState.devMode);
        if (typeof window !== 'undefined' && typeof window.dispatchEvent === 'function') {
            window.dispatchEvent(new CustomEvent('selectionPanelDevModeChanged', {
                detail: { enabled: appState.devMode }
            }));
        }
    });

    const perfToggle = document.getElementById('togglePerfOverlay');
    perfToggle?.addEventListener('change', () => {
        setPerfOverlayEnabled(!!perfToggle.checked);
    });

    const camDebugToggle = document.getElementById('toggleCameraDebug');
    camDebugToggle?.addEventListener('change', () => {
        appState.showCameraDebug = !!camDebugToggle.checked;
        setPreference('showCameraDebug', appState.showCameraDebug);
        pipeline?.engine?.setCameraDebugEnabled?.(appState.showCameraDebug);
    });

    const followInspectorToggle = document.getElementById('toggleFollowViewInspector');
    followInspectorToggle?.addEventListener('change', () => {
        setFollowInspectorEnabled(!!followInspectorToggle.checked);
    });

    const kvCacheModeToggle = document.getElementById('toggleKvCacheMode');
    kvCacheModeToggle?.addEventListener('change', () => {
        const prevEnabled = !!appState.kvCacheModeEnabled;
        const nextEnabled = !!kvCacheModeToggle.checked;
        appState.kvCacheModeEnabled = nextEnabled;
        setPreference('kvCacheModeEnabled', nextEnabled);
        syncKvCacheModeUi(nextEnabled);
        dispatchKvCacheModeChanged({
            enabled: nextEnabled,
            previousEnabled: prevEnabled
        });
    });

    if (typeof window !== 'undefined' && typeof window.addEventListener === 'function') {
        window.addEventListener(KV_CACHE_MODE_STATE_SYNC_EVENT, (event) => {
            const nextEnabled = !!event?.detail?.enabled;
            appState.kvCacheModeEnabled = nextEnabled;
            syncKvCacheModeUi(nextEnabled);
        });
    }

    const initialBrightness = getPreference(BRIGHTNESS_PREF_KEY, BRIGHTNESS_DEFAULT);
    applyBrightness(initialBrightness, { persist: false });

    const initialPlaybackSpeed = getPreference(PLAYBACK_SPEED_PREF_KEY, DEFAULT_PLAYBACK_SPEED_PERCENT);
    applyPlaybackSpeed(initialPlaybackSpeed, { persist: false });

    initInlinePercentEditor({
        buttonEl: brightnessValue,
        inputEl: brightnessInput,
        readValue: () => clampBrightnessUi(brightnessSlider?.value),
        commitValue: (value) => applyBrightness(value, { persist: true, valueIsUi: true })
    });

    initInlinePercentEditor({
        buttonEl: playbackSpeedValue,
        inputEl: playbackSpeedInput,
        readValue: () => clampPlaybackSpeedPercent(playbackSpeedSlider?.value),
        commitValue: (value) => applyPlaybackSpeed(value, { persist: true })
    });

    brightnessSlider?.addEventListener('input', () => {
        applyBrightness(brightnessSlider.value, { persist: false, valueIsUi: true });
    });

    brightnessSlider?.addEventListener('change', () => {
        applyBrightness(brightnessSlider.value, { persist: true, valueIsUi: true });
    });

    playbackSpeedSlider?.addEventListener('input', () => {
        applyPlaybackSpeed(playbackSpeedSlider.value, { persist: false });
    });

    playbackSpeedSlider?.addEventListener('change', () => {
        applyPlaybackSpeed(playbackSpeedSlider.value, { persist: true });
    });

    if (appState.showPerfOverlay) {
        setPerfOverlayEnabled(true);
    }

    if (appState.showFollowViewInspector) {
        setFollowInspectorEnabled(true);
    }

    syncKvCacheModeUi(appState.kvCacheModeEnabled);
}
