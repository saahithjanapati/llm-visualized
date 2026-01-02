import { setPlaybackSpeed } from '../utils/constants.js';
import { getPreference, setPreference } from '../utils/preferences.js';
import { appState } from '../state/appState.js';

// Wires up the settings modal controls.
export function initSettingsModal(pipeline) {
    const settingsBtn = document.getElementById('settingsBtn');
    const settingsOverlay = document.getElementById('settingsOverlay');
    const settingsClose = document.getElementById('settingsClose');
    const equationsPanel = document.getElementById('equationsPanel');

    appState.autoCameraFollow = getPreference('autoCameraFollow', true);
    pipeline?.setAutoCameraFollow?.(appState.autoCameraFollow, { immediate: true });

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
        pipeline?.setAutoCameraFollow?.(appState.autoCameraFollow, { immediate: appState.autoCameraFollow });
    });
}
