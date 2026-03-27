import { getPreference, setPreference } from '../utils/preferences.js';

const FIRST_VISIT_SCENE_HINT_ID = 'firstVisitSceneHint';
const FIRST_VISIT_SCENE_HINT_TEXT = 'Be sure to click around on different components to learn more about them.';
const FIRST_VISIT_SCENE_HINT_PREFERENCE_KEY = 'firstVisitSceneHintShown';
const FIRST_VISIT_SCENE_HINT_AUTO_HIDE_MS = 5000;

function buildDom() {
    let root = document.getElementById(FIRST_VISIT_SCENE_HINT_ID);
    if (!root) {
        root = document.createElement('div');
        root.id = FIRST_VISIT_SCENE_HINT_ID;
        document.body.appendChild(root);
    }
    root.textContent = FIRST_VISIT_SCENE_HINT_TEXT;
    root.dataset.visible = 'false';
    root.setAttribute('role', 'status');
    root.setAttribute('aria-live', 'polite');
    root.setAttribute('aria-atomic', 'true');
    root.setAttribute('aria-hidden', 'true');
    return root;
}

export function initFirstVisitSceneHint({
    autoHideMs = FIRST_VISIT_SCENE_HINT_AUTO_HIDE_MS,
    preferenceKey = FIRST_VISIT_SCENE_HINT_PREFERENCE_KEY
} = {}) {
    if (typeof document === 'undefined' || !document.body) {
        return {
            showIfEligible: () => false,
            hide: () => false,
            dispose: () => false
        };
    }

    const root = buildDom();
    let hideTimerId = null;

    const clearHideTimer = () => {
        if (hideTimerId === null) return;
        clearTimeout(hideTimerId);
        hideTimerId = null;
    };

    const hide = () => {
        clearHideTimer();
        root.dataset.visible = 'false';
        root.setAttribute('aria-hidden', 'true');
        return true;
    };

    const showIfEligible = () => {
        if (getPreference(preferenceKey, false)) return false;
        setPreference(preferenceKey, true);
        root.dataset.visible = 'true';
        root.setAttribute('aria-hidden', 'false');
        clearHideTimer();
        hideTimerId = setTimeout(() => {
            hide();
        }, Math.max(0, Number.isFinite(autoHideMs) ? autoHideMs : FIRST_VISIT_SCENE_HINT_AUTO_HIDE_MS));
        return true;
    };

    const dispose = () => {
        clearHideTimer();
        if (root.parentElement) {
            root.parentElement.removeChild(root);
        }
        return true;
    };

    return {
        showIfEligible,
        hide,
        dispose
    };
}
