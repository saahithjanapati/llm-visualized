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
import { acquireModalUiLock } from './overlayLockManager.js';

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
const KV_CACHE_READY_HINT = 'KV cache enabled for prefill and decode.';
const KV_CACHE_SWITCHING_ON_HINT = 'Switching to KV cache mode...';
const KV_CACHE_SWITCHING_OFF_HINT = 'Returning to standard attention mode...';
const KV_CACHE_SWITCH_DISPATCH_DELAY_MS = 260;

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

function scheduleAfterNextPaint(callback, { delayMs = 0 } = {}) {
    if (typeof callback !== 'function') return () => {};
    if (typeof window === 'undefined') {
        callback();
        return () => {};
    }

    const safeDelayMs = Number.isFinite(delayMs) ? Math.max(0, delayMs) : 0;
    let rafId = null;
    let timeoutId = null;
    let cancelled = false;

    const run = () => {
        if (cancelled) return;
        const setTimer = typeof window.setTimeout === 'function' ? window.setTimeout.bind(window) : setTimeout;
        timeoutId = setTimer(() => {
            timeoutId = null;
            if (!cancelled) callback();
        }, safeDelayMs);
    };

    if (typeof window.requestAnimationFrame === 'function') {
        rafId = window.requestAnimationFrame(run);
    } else {
        run();
    }

    return () => {
        cancelled = true;
        if (rafId !== null && typeof window.cancelAnimationFrame === 'function') {
            window.cancelAnimationFrame(rafId);
        }
        if (timeoutId !== null) {
            const clearTimer = typeof window.clearTimeout === 'function' ? window.clearTimeout.bind(window) : clearTimeout;
            clearTimer(timeoutId);
        }
    };
}

function ensureKvCacheTransitionOverlay() {
    if (typeof document === 'undefined') return null;
    let root = document.getElementById('kvCacheModeTransitionOverlay');
    if (!root) {
        root = document.createElement('div');
        root.id = 'kvCacheModeTransitionOverlay';
        root.dataset.visible = 'false';
        root.setAttribute('role', 'status');
        root.setAttribute('aria-live', 'polite');
        root.innerHTML = `
            <div class="kv-cache-mode-transition-card">
                <div class="kv-cache-mode-transition-spinner" aria-hidden="true"></div>
                <div class="kv-cache-mode-transition-copy" data-role="copy"></div>
            </div>
        `;
        document.body.appendChild(root);
    }
    return root;
}

function setKvCacheTransitionOverlayVisible(visible, {
    enabled = false
} = {}) {
    const root = ensureKvCacheTransitionOverlay();
    if (!root) return;
    const copy = root.querySelector('[data-role="copy"]');
    if (copy) {
        copy.textContent = enabled
            ? 'Switching to KV cache mode'
            : 'Returning to standard attention mode';
    }
    root.dataset.visible = visible ? 'true' : 'false';
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
    let releaseModalUiLock = null;
    let kvCacheModeChangeNonce = 0;
    let kvCacheModeSwitchPending = false;
    let kvCacheModeSwitchTarget = !!appState.kvCacheModeEnabled;
    let cancelPendingKvCacheModeDispatch = null;
    let cancelPendingKvCacheModeSettle = null;

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
        const laneLabel = Number.isFinite(refInfo?.laneLabel)
            ? refInfo.laneLabel
            : (Number.isFinite(refInfo?.laneIndex) ? refInfo.laneIndex + 1 : '—');

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

    const updateKvCacheStatusHint = (enabled, {
        switching = false,
        switchingTarget = enabled
    } = {}) => {
        const hint = document.getElementById('kvCacheStatusHint');
        if (!hint) return;
        if (switching) {
            hint.hidden = false;
            hint.textContent = switchingTarget
                ? KV_CACHE_SWITCHING_ON_HINT
                : KV_CACHE_SWITCHING_OFF_HINT;
            hint.dataset.state = 'switching';
            return;
        }
        hint.textContent = KV_CACHE_READY_HINT;
        hint.hidden = !enabled;
        delete hint.dataset.state;
    };

    const syncKvCacheModeUi = (enabled, {
        switching = kvCacheModeSwitchPending,
        switchingTarget = kvCacheModeSwitchTarget
    } = {}) => {
        const kvCacheModeToggle = document.getElementById('toggleKvCacheMode');
        if (kvCacheModeToggle) kvCacheModeToggle.checked = !!enabled;
        const toggleRow = kvCacheModeToggle?.closest?.('.toggle-row') || null;
        if (toggleRow) {
            if (switching) {
                toggleRow.dataset.switching = 'true';
                toggleRow.setAttribute('aria-busy', 'true');
            } else {
                delete toggleRow.dataset.switching;
                toggleRow.removeAttribute('aria-busy');
            }
        }
        updateKvCacheStatusHint(enabled, { switching, switchingTarget });
    };

    const cancelScheduledKvCacheModeWork = () => {
        if (cancelPendingKvCacheModeDispatch) {
            cancelPendingKvCacheModeDispatch();
            cancelPendingKvCacheModeDispatch = null;
        }
        if (cancelPendingKvCacheModeSettle) {
            cancelPendingKvCacheModeSettle();
            cancelPendingKvCacheModeSettle = null;
        }
    };

    const settleKvCacheModeSwitch = (nonce) => {
        cancelPendingKvCacheModeSettle = scheduleAfterNextPaint(() => {
            cancelPendingKvCacheModeSettle = null;
            if (nonce !== kvCacheModeChangeNonce) return;
            kvCacheModeSwitchPending = false;
            syncKvCacheModeUi(appState.kvCacheModeEnabled, { switching: false });
            setKvCacheTransitionOverlayVisible(false, {
                enabled: appState.kvCacheModeEnabled
            });
        });
    };

    const scheduleKvCacheModeDispatch = ({
        nextEnabled,
        previousEnabled
    }) => {
        const nonce = ++kvCacheModeChangeNonce;
        kvCacheModeSwitchPending = true;
        kvCacheModeSwitchTarget = !!nextEnabled;
        cancelScheduledKvCacheModeWork();
        setKvCacheTransitionOverlayVisible(true, { enabled: nextEnabled });
        syncKvCacheModeUi(nextEnabled, {
            switching: true,
            switchingTarget: nextEnabled
        });
        cancelPendingKvCacheModeDispatch = scheduleAfterNextPaint(() => {
            cancelPendingKvCacheModeDispatch = null;
            if (nonce !== kvCacheModeChangeNonce) return;
            try {
                dispatchKvCacheModeChanged({
                    enabled: nextEnabled,
                    previousEnabled
                });
            } finally {
                settleKvCacheModeSwitch(nonce);
            }
        }, {
            delayMs: KV_CACHE_SWITCH_DISPATCH_DELAY_MS
        });
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
        releaseModalUiLock = acquireModalUiLock();
        pipeline?.engine?.pause?.('modal');
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
        syncKvCacheModeUi(appState.kvCacheModeEnabled);
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
        pipeline?.engine?.resume?.('modal');
        if (releaseModalUiLock) {
            releaseModalUiLock();
            releaseModalUiLock = null;
        }
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
        scheduleKvCacheModeDispatch({
            nextEnabled,
            previousEnabled: prevEnabled
        });
    });

    if (typeof window !== 'undefined' && typeof window.addEventListener === 'function') {
        window.addEventListener(KV_CACHE_MODE_STATE_SYNC_EVENT, (event) => {
            const nextEnabled = !!event?.detail?.enabled;
            kvCacheModeChangeNonce += 1;
            kvCacheModeSwitchPending = false;
            kvCacheModeSwitchTarget = nextEnabled;
            cancelScheduledKvCacheModeWork();
            setKvCacheTransitionOverlayVisible(false, { enabled: nextEnabled });
            appState.kvCacheModeEnabled = nextEnabled;
            syncKvCacheModeUi(nextEnabled, { switching: false });
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
