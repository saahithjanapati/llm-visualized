import { setPlaybackSpeed } from '../src/utils/constants.js';

export function setupSettingsPanel(pipeline, eqControls, setModalPaused) {
    const settingsBtn = document.getElementById('settingsBtn');
    const settingsOverlay = document.getElementById('settingsOverlay');
    const settingsClose = document.getElementById('settingsClose');

    function applySpeed(value) {
        try { setPlaybackSpeed(value); } catch (_) { /* no-op */ }
    }

    function openSettings() {
        settingsOverlay.style.display = 'flex';
        settingsOverlay.setAttribute('aria-hidden', 'false');
        document.body.style.overflow = 'hidden';
        try { pipeline?.engine?.pause?.(); } catch (_) { /* no-op */ }
        setModalPaused(true);
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
            if (eq) eq.checked = !!eqControls.getShowEquations();
        } catch (_) { /* no-op */ }
    }

    function closeSettings() {
        settingsOverlay.style.display = 'none';
        settingsOverlay.setAttribute('aria-hidden', 'true');
        document.body.style.overflow = '';
        try { pipeline?.engine?.resume?.(); } catch (_) { /* no-op */ }
        setModalPaused(false);
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
        try { pipeline?.engine?.setRaycastingEnabled?.(!!rayToggle.checked); } catch (_) { /* no-op */ }
    });

    const eqToggle = document.getElementById('toggleEquations');
    eqToggle?.addEventListener('change', () => {
        const val = !!eqToggle.checked;
        eqControls.setShowEquations(val);
    });
}
