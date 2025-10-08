import { setPlaybackSpeed } from '../utils/constants.js';
import { setPreference } from '../utils/preferences.js';
import { appState } from '../state/appState.js';
import { getAvailableThemes, getCurrentThemeId, onThemeChange, setTheme } from '../state/themeState.js';

// Wires up the settings modal controls.
export function initSettingsModal(pipeline) {
    const settingsBtn = document.getElementById('settingsBtn');
    const settingsOverlay = document.getElementById('settingsOverlay');
    const settingsClose = document.getElementById('settingsClose');
    const equationsPanel = document.getElementById('equationsPanel');

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
            updateSpeedChecked(selectedLabel?.dataset.value || 'medium');
        }
        updateThemeChecked(getCurrentThemeId());
        const rc = document.getElementById('toggleRaycast');
        if (rc && pipeline?.engine?.isRaycastingEnabled) {
            rc.checked = !!pipeline.engine.isRaycastingEnabled();
        }
        const eq = document.getElementById('toggleEquations');
        if (eq) eq.checked = !!appState.showEquations;
        const bg = document.getElementById('toggleHdrBackground');
        if (bg) bg.checked = !!appState.showHdrBackground;
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

    const themeGroup = document.getElementById('themeOptions');

    function updateThemeChecked(themeId) {
        if (!themeGroup) return;
        themeGroup.querySelectorAll('.theme-option').forEach((label) => {
            const value = label.getAttribute('data-value');
            label.setAttribute('data-checked', String(value === themeId));
            const input = label.querySelector('input');
            if (input) input.checked = value === themeId;
        });
    }

    if (themeGroup) {
        const themes = getAvailableThemes();
        themeGroup.innerHTML = '';
        const currentTheme = getCurrentThemeId();
        themes.forEach((theme) => {
            const label = document.createElement('label');
            label.className = 'theme-option';
            label.setAttribute('data-value', theme.id);

            const input = document.createElement('input');
            input.type = 'radio';
            input.name = 'colorTheme';
            input.value = theme.id;
            input.checked = theme.id === currentTheme;
            label.appendChild(input);

            const name = document.createElement('div');
            name.className = 'theme-name';
            name.textContent = theme.label;
            label.appendChild(name);

            const swatch = document.createElement('div');
            swatch.className = 'theme-swatch';
            if (Array.isArray(theme.swatch) && theme.swatch.length >= 2) {
                swatch.style.background = `linear-gradient(90deg, ${theme.swatch[0]}, ${theme.swatch[1]})`;
            }
            label.appendChild(swatch);

            label.addEventListener('click', (e) => {
                e.preventDefault();
                const value = label.getAttribute('data-value');
                if (!value || value === getCurrentThemeId()) return;
                setTheme(value);
            });

            themeGroup.appendChild(label);
        });
        updateThemeChecked(currentTheme);
    }

    const detachThemeListener = onThemeChange((theme) => {
        updateThemeChecked(theme?.id ?? getCurrentThemeId());
    });

    const rayToggle = document.getElementById('toggleRaycast');
    rayToggle?.addEventListener('change', () => {
        pipeline?.engine?.setRaycastingEnabled?.(!!rayToggle.checked);
    });

    const eqToggle = document.getElementById('toggleEquations');
    eqToggle?.addEventListener('change', () => {
        appState.showEquations = !!eqToggle.checked;
        setPreference('showEquations', appState.showEquations);
        if (equationsPanel) equationsPanel.style.display = appState.showEquations ? 'block' : 'none';
        appState.lastEqKey = '';
    });

    const bgToggle = document.getElementById('toggleHdrBackground');
    bgToggle?.addEventListener('change', () => {
        appState.showHdrBackground = !!bgToggle.checked;
        setPreference('showHdrBackground', appState.showHdrBackground);
        appState.applyEnvironmentBackground(pipeline);
    });

    if (typeof window !== 'undefined') {
        window.addEventListener('unload', detachThemeListener, { once: true });
    }
}
