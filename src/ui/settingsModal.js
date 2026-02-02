import { setPlaybackSpeed } from '../utils/constants.js';
import { getPreference, setPreference } from '../utils/preferences.js';
import { appState } from '../state/appState.js';
import { initPerfOverlay } from './perfOverlay.js';

// Wires up the settings modal controls.
export function initSettingsModal(pipeline) {
    const settingsBtn = document.getElementById('settingsBtn');
    const settingsOverlay = document.getElementById('settingsOverlay');
    const settingsClose = document.getElementById('settingsClose');
    const equationsPanel = document.getElementById('equationsPanel');

    appState.autoCameraFollow = getPreference('autoCameraFollow', true);
    appState.showCameraDebug = getPreference('showCameraDebug', false);
    appState.showFollowViewInspector = false;
    appState.devMode = getPreference('devMode', false);
    appState.showPerfOverlay = getPreference('showPerfOverlay', false);
    pipeline?.setAutoCameraFollow?.(appState.autoCameraFollow, { immediate: true });
    pipeline?.engine?.setCameraDebugEnabled?.(appState.showCameraDebug);
    pipeline?.setDevMode?.(appState.devMode);
    pipeline?.engine?.setDevMode?.(appState.devMode);

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

    function applySpeed(value) {
        setPlaybackSpeed(value);
    }

    function openSettings() {
        settingsOverlay.style.display = 'flex';
        settingsOverlay.setAttribute('aria-hidden', 'false');
        document.body.style.overflow = 'hidden';
        pipeline?.engine?.pause?.('modal');
        appState.modalPaused = true;
        const checked = settingsOverlay.querySelector('input[name="playbackSpeed"]:checked');
        if (checked) {
            const selectedLabel = checked.closest('.speed-option');
            updateSpeedChecked(selectedLabel?.dataset.value || 'normal');
        }
        const rc = document.getElementById('toggleRaycast');
        if (rc && pipeline?.engine?.isRaycastingEnabled) {
            rc.checked = !!pipeline.engine.isRaycastingEnabled();
        }
        const eq = document.getElementById('toggleEquations');
        if (eq) eq.checked = !!appState.showEquations;
        const bg = document.getElementById('toggleHdrBackground');
        if (bg) bg.checked = !!appState.showHdrBackground;
        const autoCam = document.getElementById('toggleAutoCamera');
        if (autoCam) {
            const enabled = typeof pipeline?.isAutoCameraFollowEnabled === 'function'
                ? pipeline.isAutoCameraFollowEnabled()
                : appState.autoCameraFollow;
            autoCam.checked = !!enabled;
        }
        const devMode = document.getElementById('toggleDevMode');
        if (devMode) devMode.checked = !!appState.devMode;
        const camDebug = document.getElementById('toggleCameraDebug');
        if (camDebug) camDebug.checked = !!appState.showCameraDebug;
        const followInspector = document.getElementById('toggleFollowViewInspector');
        if (followInspector) followInspector.checked = !!appState.showFollowViewInspector;
        const perfOverlay = document.getElementById('togglePerfOverlay');
        if (perfOverlay) perfOverlay.checked = !!appState.showPerfOverlay;
    }

    function closeSettings() {
        settingsOverlay.style.display = 'none';
        settingsOverlay.setAttribute('aria-hidden', 'true');
        document.body.style.overflow = '';
        pipeline?.engine?.resume?.('modal');
        appState.modalPaused = false;
    }

    function updateSpeedChecked(value) {
        const labels = settingsOverlay.querySelectorAll('.speed-option');
        labels.forEach((label) => {
            const v = label.getAttribute('data-value');
            label.setAttribute('data-checked', String(v === value));
            const input = label.querySelector('input');
            if (input) input.checked = v === value;
        });
    }

    settingsBtn?.addEventListener('click', openSettings);
    settingsClose?.addEventListener('click', closeSettings);
    settingsOverlay?.addEventListener('click', (e) => {
        if (e.target === settingsOverlay) closeSettings();
    });
    window.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && settingsOverlay?.getAttribute('aria-hidden') === 'false') {
            closeSettings();
        }
    });

    settingsOverlay?.querySelectorAll('.speed-option').forEach((label) => {
        label.addEventListener('click', (e) => {
            e.preventDefault();
            const value = label.getAttribute('data-value');
            if (!value) return;
            updateSpeedChecked(value);
            applySpeed(value);
        });
    });

    const rayToggle = document.getElementById('toggleRaycast');
    rayToggle?.addEventListener('change', () => {
        pipeline?.engine?.setRaycastingEnabled?.(!!rayToggle.checked);
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

    const bgToggle = document.getElementById('toggleHdrBackground');
    bgToggle?.addEventListener('change', () => {
        appState.showHdrBackground = !!bgToggle.checked;
        setPreference('showHdrBackground', appState.showHdrBackground);
        appState.applyEnvironmentBackground(pipeline);
    });

    const autoCamToggle = document.getElementById('toggleAutoCamera');
    autoCamToggle?.addEventListener('change', () => {
        appState.autoCameraFollow = !!autoCamToggle.checked;
        setPreference('autoCameraFollow', appState.autoCameraFollow);
        pipeline?.setAutoCameraFollow?.(appState.autoCameraFollow, {
            immediate: appState.autoCameraFollow,
            resetView: appState.autoCameraFollow,
            smoothReset: appState.autoCameraFollow
        });
    });

    const devToggle = document.getElementById('toggleDevMode');
    devToggle?.addEventListener('change', () => {
        appState.devMode = !!devToggle.checked;
        setPreference('devMode', appState.devMode);
        pipeline?.setDevMode?.(appState.devMode);
        pipeline?.engine?.setDevMode?.(appState.devMode);
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

    if (appState.showPerfOverlay) {
        setPerfOverlayEnabled(true);
    }

    if (appState.showFollowViewInspector) {
        setFollowInspectorEnabled(true);
    }
}
