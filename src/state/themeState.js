import { getPreference, setPreference } from '../utils/preferences.js';

const THEME_PREF_KEY = 'colorTheme';

const THEMES = Object.freeze({
    original: Object.freeze({
        id: 'original',
        label: 'Original',
        cssVars: Object.freeze({
            '--app-body-bg': '#000000',
            '--app-text-primary': '#ffffff',
            '--app-control-border': 'rgba(255,255,255,0.18)',
            '--app-control-bg': 'rgba(30,30,30,0.55)',
            '--app-control-hover-bg': 'rgba(40,40,40,0.65)',
            '--app-control-focus-outline': 'rgba(122,202,255,0.9)',
            '--app-control-active-bg': 'rgba(56,132,255,0.18)',
            '--app-control-active-border': 'rgba(122,202,255,0.8)',
            '--app-overlay-bg': 'rgba(0,0,0,0.45)',
            '--app-status-bg': 'rgba(20,20,20,0.35)',
            '--app-status-text': '#ffffff',
            '--app-panel-bg': 'rgba(255,255,255,0.04)',
            '--app-panel-border': 'rgba(255,255,255,0.12)',
            '--app-panel-hover-bg': 'rgba(255,255,255,0.08)',
            '--app-panel-checked-bg': 'rgba(56,132,255,0.18)',
            '--app-panel-checked-border': 'rgba(122,202,255,0.8)',
            '--app-muted-text': '#cfd3dc',
            '--app-equation-bg': 'rgba(20,20,20,0.35)',
            '--app-equation-text': '#ffffff',
            '--app-katex-color': '#ffffff',
            '--app-hover-label-bg': 'rgba(20,20,20,0.35)',
            '--app-hover-label-text': '#ffffff'
        }),
        three: Object.freeze({
            sceneBackground: 0x000000,
            ambientLight: Object.freeze({ color: 0xffffff, intensity: 0.7 }),
            directionalLight: Object.freeze({ color: 0xffffff, intensity: 0.9 }),
            hoverLabel: Object.freeze({ background: 'rgba(20,20,20,0.35)', color: '#ffffff' }),
            inactiveComponentColor: 0x202020,
            mhsaMatrixRestingColor: 0x404040,
            mhsaBrightGreen: 0x33ff33,
            mhsaDarkGreen: 0x002200,
            mhsaBrightBlue: 0x6666ff,
            mhsaDarkBlue: 0x000022,
            mhsaBrightRed: 0xff3333,
            mhsaDarkRed: 0x220000,
            mhaFinalQColor: 0x1e5299,
            mhaFinalKColor: 0x1d9752,
            mhaFinalVColor: 0x811b2d,
            mhaOutputProjectionActiveColor: 0xaf5faf,
            embeddingVocabColor: 0x1e5299,
            embeddingPosColor: 0x1d9752,
            embeddingTopColor: 0x000000,
            trailColor: 0xffffff
        })
    }),
    aurora: Object.freeze({
        id: 'aurora',
        label: 'Aurora drift',
        cssVars: Object.freeze({
            '--app-body-bg': '#07111d',
            '--app-text-primary': '#e8f9ff',
            '--app-control-border': 'rgba(125,223,255,0.35)',
            '--app-control-bg': 'rgba(12,32,48,0.65)',
            '--app-control-hover-bg': 'rgba(18,48,68,0.75)',
            '--app-control-focus-outline': 'rgba(140,230,255,0.9)',
            '--app-control-active-bg': 'rgba(68,198,255,0.25)',
            '--app-control-active-border': 'rgba(140,230,255,0.9)',
            '--app-overlay-bg': 'rgba(4,18,36,0.7)',
            '--app-status-bg': 'rgba(12,32,48,0.55)',
            '--app-status-text': '#e8f9ff',
            '--app-panel-bg': 'rgba(18,48,68,0.6)',
            '--app-panel-border': 'rgba(125,223,255,0.25)',
            '--app-panel-hover-bg': 'rgba(68,198,255,0.15)',
            '--app-panel-checked-bg': 'rgba(68,198,255,0.25)',
            '--app-panel-checked-border': 'rgba(140,230,255,0.85)',
            '--app-muted-text': '#9ad4f7',
            '--app-equation-bg': 'rgba(12,32,48,0.55)',
            '--app-equation-text': '#e8f9ff',
            '--app-katex-color': '#e8f9ff',
            '--app-hover-label-bg': 'rgba(12,32,48,0.65)',
            '--app-hover-label-text': '#e8f9ff'
        }),
        three: Object.freeze({
            sceneBackground: 0x07111d,
            ambientLight: Object.freeze({ color: 0x88d6ff, intensity: 0.6 }),
            directionalLight: Object.freeze({ color: 0x6ed6ff, intensity: 0.95 }),
            hoverLabel: Object.freeze({ background: 'rgba(12,32,48,0.65)', color: '#e8f9ff' }),
            inactiveComponentColor: 0x153043,
            mhsaMatrixRestingColor: 0x1c3f57,
            mhsaBrightGreen: 0x6cf0a2,
            mhsaDarkGreen: 0x0c2f29,
            mhsaBrightBlue: 0x6fc4ff,
            mhsaDarkBlue: 0x102a3f,
            mhsaBrightRed: 0xff7fa6,
            mhsaDarkRed: 0x3a0f22,
            mhaFinalQColor: 0x63c6ff,
            mhaFinalKColor: 0x4de0a0,
            mhaFinalVColor: 0xc77dff,
            mhaOutputProjectionActiveColor: 0x7f9cff,
            embeddingVocabColor: 0x63c6ff,
            embeddingPosColor: 0x4de0a0,
            embeddingTopColor: 0x0a2233,
            trailColor: 0x7fe8ff
        })
    }),
    midnight: Object.freeze({
        id: 'midnight',
        label: 'Midnight brass',
        cssVars: Object.freeze({
            '--app-body-bg': '#05070f',
            '--app-text-primary': '#f5f1e8',
            '--app-control-border': 'rgba(255,215,141,0.28)',
            '--app-control-bg': 'rgba(18,20,34,0.72)',
            '--app-control-hover-bg': 'rgba(24,28,44,0.82)',
            '--app-control-focus-outline': 'rgba(255,215,141,0.8)',
            '--app-control-active-bg': 'rgba(214,169,96,0.2)',
            '--app-control-active-border': 'rgba(255,215,141,0.75)',
            '--app-overlay-bg': 'rgba(6,8,18,0.78)',
            '--app-status-bg': 'rgba(18,20,34,0.62)',
            '--app-status-text': '#f5f1e8',
            '--app-panel-bg': 'rgba(24,28,44,0.7)',
            '--app-panel-border': 'rgba(214,169,96,0.25)',
            '--app-panel-hover-bg': 'rgba(214,169,96,0.18)',
            '--app-panel-checked-bg': 'rgba(214,169,96,0.24)',
            '--app-panel-checked-border': 'rgba(255,215,141,0.75)',
            '--app-muted-text': '#d6ba7c',
            '--app-equation-bg': 'rgba(18,20,34,0.62)',
            '--app-equation-text': '#f5f1e8',
            '--app-katex-color': '#f5f1e8',
            '--app-hover-label-bg': 'rgba(18,20,34,0.7)',
            '--app-hover-label-text': '#f5f1e8'
        }),
        three: Object.freeze({
            sceneBackground: 0x05070f,
            ambientLight: Object.freeze({ color: 0xffd78d, intensity: 0.55 }),
            directionalLight: Object.freeze({ color: 0xffc169, intensity: 0.85 }),
            hoverLabel: Object.freeze({ background: 'rgba(18,20,34,0.7)', color: '#f5f1e8' }),
            inactiveComponentColor: 0x2a2e41,
            mhsaMatrixRestingColor: 0x35394d,
            mhsaBrightGreen: 0xf0c766,
            mhsaDarkGreen: 0x3d2f18,
            mhsaBrightBlue: 0x6a9cfa,
            mhsaDarkBlue: 0x0f1b3c,
            mhsaBrightRed: 0xf28a6a,
            mhsaDarkRed: 0x2f1410,
            mhaFinalQColor: 0x6a9cfa,
            mhaFinalKColor: 0xf0c766,
            mhaFinalVColor: 0xc46a5c,
            mhaOutputProjectionActiveColor: 0xb58a5a,
            embeddingVocabColor: 0x6a9cfa,
            embeddingPosColor: 0xf0c766,
            embeddingTopColor: 0x1a1f30,
            trailColor: 0xffd78d
        })
    }),
    sunset: Object.freeze({
        id: 'sunset',
        label: 'Neon sunset',
        cssVars: Object.freeze({
            '--app-body-bg': '#1a0b1a',
            '--app-text-primary': '#ffe5f0',
            '--app-control-border': 'rgba(255,136,192,0.38)',
            '--app-control-bg': 'rgba(42,18,42,0.7)',
            '--app-control-hover-bg': 'rgba(60,24,60,0.78)',
            '--app-control-focus-outline': 'rgba(255,136,192,0.9)',
            '--app-control-active-bg': 'rgba(255,112,152,0.28)',
            '--app-control-active-border': 'rgba(255,136,192,0.85)',
            '--app-overlay-bg': 'rgba(30,12,30,0.78)',
            '--app-status-bg': 'rgba(42,18,42,0.62)',
            '--app-status-text': '#ffe5f0',
            '--app-panel-bg': 'rgba(60,24,60,0.72)',
            '--app-panel-border': 'rgba(255,136,192,0.28)',
            '--app-panel-hover-bg': 'rgba(255,112,152,0.2)',
            '--app-panel-checked-bg': 'rgba(255,112,152,0.28)',
            '--app-panel-checked-border': 'rgba(255,136,192,0.85)',
            '--app-muted-text': '#ff9bc9',
            '--app-equation-bg': 'rgba(42,18,42,0.62)',
            '--app-equation-text': '#ffe5f0',
            '--app-katex-color': '#ffe5f0',
            '--app-hover-label-bg': 'rgba(42,18,42,0.68)',
            '--app-hover-label-text': '#ffe5f0'
        }),
        three: Object.freeze({
            sceneBackground: 0x1a0b1a,
            ambientLight: Object.freeze({ color: 0xff88c0, intensity: 0.6 }),
            directionalLight: Object.freeze({ color: 0xff7098, intensity: 0.95 }),
            hoverLabel: Object.freeze({ background: 'rgba(42,18,42,0.68)', color: '#ffe5f0' }),
            inactiveComponentColor: 0x331f33,
            mhsaMatrixRestingColor: 0x472c47,
            mhsaBrightGreen: 0xffc877,
            mhsaDarkGreen: 0x3d2312,
            mhsaBrightBlue: 0xff88f0,
            mhsaDarkBlue: 0x321336,
            mhsaBrightRed: 0xff7098,
            mhsaDarkRed: 0x3d121c,
            mhaFinalQColor: 0xff88f0,
            mhaFinalKColor: 0xffc877,
            mhaFinalVColor: 0xff7098,
            mhaOutputProjectionActiveColor: 0xff9bd1,
            embeddingVocabColor: 0xff88f0,
            embeddingPosColor: 0xffc877,
            embeddingTopColor: 0x2b152d,
            trailColor: 0xffa8de
        })
    }),
    forest: Object.freeze({
        id: 'forest',
        label: 'Emerald forest',
        cssVars: Object.freeze({
            '--app-body-bg': '#04130c',
            '--app-text-primary': '#e5ffe8',
            '--app-control-border': 'rgba(140,214,161,0.3)',
            '--app-control-bg': 'rgba(16,40,28,0.68)',
            '--app-control-hover-bg': 'rgba(24,64,40,0.78)',
            '--app-control-focus-outline': 'rgba(140,214,161,0.85)',
            '--app-control-active-bg': 'rgba(122,214,164,0.26)',
            '--app-control-active-border': 'rgba(140,214,161,0.85)',
            '--app-overlay-bg': 'rgba(6,24,16,0.78)',
            '--app-status-bg': 'rgba(16,40,28,0.6)',
            '--app-status-text': '#e5ffe8',
            '--app-panel-bg': 'rgba(24,64,40,0.72)',
            '--app-panel-border': 'rgba(140,214,161,0.28)',
            '--app-panel-hover-bg': 'rgba(122,214,164,0.2)',
            '--app-panel-checked-bg': 'rgba(122,214,164,0.26)',
            '--app-panel-checked-border': 'rgba(140,214,161,0.85)',
            '--app-muted-text': '#a9e4ba',
            '--app-equation-bg': 'rgba(16,40,28,0.6)',
            '--app-equation-text': '#e5ffe8',
            '--app-katex-color': '#e5ffe8',
            '--app-hover-label-bg': 'rgba(16,40,28,0.68)',
            '--app-hover-label-text': '#e5ffe8'
        }),
        three: Object.freeze({
            sceneBackground: 0x04130c,
            ambientLight: Object.freeze({ color: 0x8cd6a1, intensity: 0.6 }),
            directionalLight: Object.freeze({ color: 0xa9f0c1, intensity: 0.9 }),
            hoverLabel: Object.freeze({ background: 'rgba(16,40,28,0.68)', color: '#e5ffe8' }),
            inactiveComponentColor: 0x1f3a2a,
            mhsaMatrixRestingColor: 0x2a4d36,
            mhsaBrightGreen: 0x9bf0a5,
            mhsaDarkGreen: 0x153220,
            mhsaBrightBlue: 0x6ed7c6,
            mhsaDarkBlue: 0x12322d,
            mhsaBrightRed: 0xf2b574,
            mhsaDarkRed: 0x3a2516,
            mhaFinalQColor: 0x6ed7c6,
            mhaFinalKColor: 0x9bf0a5,
            mhaFinalVColor: 0xf2b574,
            mhaOutputProjectionActiveColor: 0x7fdab0,
            embeddingVocabColor: 0x6ed7c6,
            embeddingPosColor: 0x9bf0a5,
            embeddingTopColor: 0x183025,
            trailColor: 0x9bf0a5
        })
    }),
    cyberpunk: Object.freeze({
        id: 'cyberpunk',
        label: 'Cyber bloom',
        cssVars: Object.freeze({
            '--app-body-bg': '#04000f',
            '--app-text-primary': '#f2f6ff',
            '--app-control-border': 'rgba(255,0,204,0.42)',
            '--app-control-bg': 'rgba(20,0,38,0.78)',
            '--app-control-hover-bg': 'rgba(32,0,58,0.85)',
            '--app-control-focus-outline': 'rgba(0,204,255,0.9)',
            '--app-control-active-bg': 'rgba(0,204,255,0.28)',
            '--app-control-active-border': 'rgba(0,204,255,0.85)',
            '--app-overlay-bg': 'rgba(10,0,28,0.82)',
            '--app-status-bg': 'rgba(20,0,38,0.7)',
            '--app-status-text': '#f2f6ff',
            '--app-panel-bg': 'rgba(32,0,58,0.78)',
            '--app-panel-border': 'rgba(255,0,204,0.35)',
            '--app-panel-hover-bg': 'rgba(0,204,255,0.2)',
            '--app-panel-checked-bg': 'rgba(0,204,255,0.28)',
            '--app-panel-checked-border': 'rgba(0,204,255,0.85)',
            '--app-muted-text': '#a2b4ff',
            '--app-equation-bg': 'rgba(20,0,38,0.7)',
            '--app-equation-text': '#f2f6ff',
            '--app-katex-color': '#f2f6ff',
            '--app-hover-label-bg': 'rgba(20,0,38,0.75)',
            '--app-hover-label-text': '#f2f6ff'
        }),
        three: Object.freeze({
            sceneBackground: 0x04000f,
            ambientLight: Object.freeze({ color: 0x00ccff, intensity: 0.65 }),
            directionalLight: Object.freeze({ color: 0xff00cc, intensity: 1.05 }),
            hoverLabel: Object.freeze({ background: 'rgba(20,0,38,0.75)', color: '#f2f6ff' }),
            inactiveComponentColor: 0x260037,
            mhsaMatrixRestingColor: 0x35004d,
            mhsaBrightGreen: 0x00ffcc,
            mhsaDarkGreen: 0x00332a,
            mhsaBrightBlue: 0x2ea8ff,
            mhsaDarkBlue: 0x001d33,
            mhsaBrightRed: 0xff4fd8,
            mhsaDarkRed: 0x3f0029,
            mhaFinalQColor: 0x2ea8ff,
            mhaFinalKColor: 0x00ffcc,
            mhaFinalVColor: 0xff4fd8,
            mhaOutputProjectionActiveColor: 0x7e4bff,
            embeddingVocabColor: 0x2ea8ff,
            embeddingPosColor: 0x00ffcc,
            embeddingTopColor: 0x12002b,
            trailColor: 0x7edbff
        })
    })
});

let currentThemeId = 'original';
let initialized = false;
const registeredPipelines = new Set();
const listeners = new Set();

function resolveTheme(themeId) {
    if (themeId && Object.prototype.hasOwnProperty.call(THEMES, themeId)) {
        return THEMES[themeId];
    }
    return THEMES.original;
}

function applyCssVariables(theme) {
    if (typeof document === 'undefined') return;
    const root = document.documentElement;
    if (!root || !theme?.cssVars) return;
    for (const [key, value] of Object.entries(theme.cssVars)) {
        if (typeof value === 'string') {
            root.style.setProperty(key, value);
        }
    }
}

function notifyThemeChange(theme) {
    listeners.forEach((listener) => {
        try {
            listener(theme);
        } catch (err) {
            console.warn('Theme listener failed:', err);
        }
    });
}

function applyThemeInternal(themeId, { skipSave = false, pipeline = null } = {}) {
    const theme = resolveTheme(themeId);
    currentThemeId = theme.id;
    applyCssVariables(theme);
    if (!skipSave) {
        try {
            setPreference(THEME_PREF_KEY, currentThemeId);
        } catch (_) {
            /* ignore storage errors */
        }
    }
    const targets = new Set(registeredPipelines);
    if (pipeline) targets.add(pipeline);
    targets.forEach((pipe) => {
        try {
            pipe?.applyTheme?.(theme);
        } catch (err) {
            console.warn('Pipeline theme application failed:', err);
        }
    });
    notifyThemeChange(theme);
    return theme;
}

export function initializeTheme(pipeline = null) {
    if (!initialized) {
        const saved = getPreference(THEME_PREF_KEY, currentThemeId);
        applyThemeInternal(saved, { skipSave: true });
        initialized = true;
    }
    if (pipeline) {
        registerPipeline(pipeline);
    }
    return getCurrentTheme();
}

export function applyTheme(themeId, pipeline = null) {
    return applyThemeInternal(themeId, { pipeline });
}

export function registerPipeline(pipeline) {
    if (!pipeline) return;
    registeredPipelines.add(pipeline);
    const theme = getCurrentTheme();
    try {
        pipeline.applyTheme?.(theme);
    } catch (err) {
        console.warn('Pipeline theme application failed:', err);
    }
}

export function unregisterPipeline(pipeline) {
    if (!pipeline) return;
    registeredPipelines.delete(pipeline);
}

export function getThemeOptions() {
    return Object.values(THEMES).map(({ id, label }) => ({ id, label }));
}

export function getCurrentThemeId() {
    return currentThemeId;
}

export function getCurrentTheme() {
    return resolveTheme(currentThemeId);
}

export function subscribeThemeChange(listener) {
    if (typeof listener !== 'function') return () => {};
    listeners.add(listener);
    return () => listeners.delete(listener);
}

export function getThemeColor(key) {
    const theme = getCurrentTheme();
    if (!theme?.three || !(key in theme.three)) return undefined;
    return theme.three[key];
}
