import { setPlaybackSpeed } from '../utils/constants.js';
import { setPreference } from '../utils/preferences.js';
import { appState } from '../state/appState.js';

// Wires up the settings modal controls.
export function initSettingsModal(pipeline) {
    const settingsBtn = document.getElementById('settingsBtn');
    const settingsOverlay = document.getElementById('settingsOverlay');
    const settingsClose = document.getElementById('settingsClose');
    const equationsPanel = document.getElementById('equationsPanel');

    function applySpeed(value) {
        try { setPlaybackSpeed(value); } catch (_) {}
    }

    function openSettings() {
        settingsOverlay.style.display = 'flex';
        settingsOverlay.setAttribute('aria-hidden', 'false');
        document.body.style.overflow = 'hidden';
        try { pipeline?.engine?.pause?.(); } catch (_) {}
        appState.modalPaused = true;
        try {
            const checked = settingsOverlay.querySelector('input[name="playbackSpeed"]:checked');
            if (checked) {
                const selectedLabel = checked.closest('.speed-option');
                updateSpeedChecked(selectedLabel?.dataset.value || 'medium');
            }
            const rc = document.getElementById('toggleRaycast');
            if (rc && pipeline?.engine?.isRaycastingEnabled) {
                rc.checked = !!pipeline.engine.isRaycastingEnabled();
            }
            const eq = document.getElementById('toggleEquations');
            if (eq) eq.checked = !!appState.showEquations;
        } catch (_) {}
    }

    function closeSettings() {
        settingsOverlay.style.display = 'none';
        settingsOverlay.setAttribute('aria-hidden', 'true');
        document.body.style.overflow = '';
        try { pipeline?.engine?.resume?.(); } catch (_) {}
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
        try { pipeline?.engine?.setRaycastingEnabled?.(!!rayToggle.checked); } catch (_) {}
    });

    const eqToggle = document.getElementById('toggleEquations');
    eqToggle?.addEventListener('change', () => {
        appState.showEquations = !!eqToggle.checked;
        setPreference('showEquations', appState.showEquations);
        if (equationsPanel) equationsPanel.style.display = appState.showEquations ? 'block' : 'none';
        appState.lastEqKey = '';
    });
}
