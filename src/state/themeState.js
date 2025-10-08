import { getPreference, setPreference } from '../utils/preferences.js';

const THEME_DEFINITIONS = {
    original: {
        id: 'original',
        label: 'Original',
        scene: {
            background: 0x081129,
            introBackground: 0x000000
        },
        layerAccents: [0xff6f61, 0xffd166, 0x06d6a0, 0x118ab2, 0x9d4edd],
        vectorGradient: null,
        mhsa: {
            rest: 0x404040,
            brightGreen: 0x33ff33,
            darkGreen: 0x002200,
            brightBlue: 0x6666ff,
            darkBlue: 0x000022,
            brightRed: 0xff3333,
            darkRed: 0x220000,
            finalQ: 0x1e5299,
            finalK: 0x1d9752,
            finalV: 0x811b2d,
            output: 0xaf5faf,
            baseQ: 0x3388ff,
            baseK: 0x33ff88,
            baseV: 0xff3355
        },
        trailColor: 0xffffff,
        embeddings: {
            vocab: 0x3388ff,
            positional: 0x33ff88,
            top: 0x000000
        },
        ui: {
            bodyBg: '#000000',
            controlBg: 'rgba(30,30,30,0.55)',
            controlText: '#ffffff',
            controlBorder: 'rgba(255,255,255,0.18)',
            controlHoverBg: 'rgba(40,40,40,0.65)',
            controlActiveBg: 'rgba(56,132,255,0.18)',
            controlActiveBorder: 'rgba(122,202,255,0.8)',
            controlFocus: 'rgba(122,202,255,0.9)',
            modalBg: 'rgba(20,20,24,0.85)',
            modalBorder: 'rgba(255,255,255,0.1)',
            modalText: '#ffffff',
            sectionLabel: '#cfd3dc',
            optionBg: 'rgba(255,255,255,0.04)',
            optionBorder: 'rgba(255,255,255,0.12)',
            optionHoverBg: 'rgba(255,255,255,0.08)',
            optionCheckedBg: 'rgba(56,132,255,0.18)',
            optionCheckedBorder: 'rgba(122,202,255,0.8)',
            toggleBg: 'rgba(255,255,255,0.04)',
            toggleBorder: 'rgba(255,255,255,0.12)',
            equationsBg: 'rgba(20,20,20,0.35)',
            equationsText: '#ffffff',
            equationsTitle: '#cfd3dc',
            statusBg: 'rgba(20,20,20,0.35)',
            statusText: '#ffffff',
            katexColor: '#ffffff',
            swatch: ['#3388ff', '#ff3355'],
            themeBorder: 'rgba(255,255,255,0.12)'
        }
    },
    aurora: {
        id: 'aurora',
        label: 'Aurora',
        scene: {
            background: 0x03111f,
            introBackground: 0x01070f
        },
        layerAccents: [0x48f6c2, 0x7a5cff, 0x49a6ff, 0xff7ab4, 0xf9ff8f],
        vectorGradient: [
            { stop: -1, color: '#1f3b73' },
            { stop: -0.05, color: '#38f9d7' },
            { stop: 0.35, color: '#6d50ff' },
            { stop: 1, color: '#ff7ac7' }
        ],
        mhsa: {
            rest: 0x14263a,
            brightGreen: 0x7cffe3,
            darkGreen: 0x0a362b,
            brightBlue: 0x8ba8ff,
            darkBlue: 0x111c3d,
            brightRed: 0xff7ebd,
            darkRed: 0x3b0d24,
            finalQ: 0x6fc8ff,
            finalK: 0x6ffac9,
            finalV: 0xff9fd6,
            output: 0xc59bff,
            baseQ: 0x4aa5ff,
            baseK: 0x6df7d8,
            baseV: 0xff7dad
        },
        trailColor: 0x9fffe0,
        embeddings: {
            vocab: 0x4aa5ff,
            positional: 0x6df7d8,
            top: 0x10172a
        },
        ui: {
            bodyBg: '#020b16',
            controlBg: 'rgba(20,36,55,0.65)',
            controlText: '#e9fffb',
            controlBorder: 'rgba(119,255,220,0.35)',
            controlHoverBg: 'rgba(27,52,79,0.75)',
            controlActiveBg: 'rgba(111,240,220,0.25)',
            controlActiveBorder: 'rgba(168,255,242,0.8)',
            controlFocus: 'rgba(143,255,236,0.95)',
            modalBg: 'rgba(10,25,40,0.92)',
            modalBorder: 'rgba(120,255,235,0.22)',
            modalText: '#e7fff9',
            sectionLabel: '#9fe0e2',
            optionBg: 'rgba(19,46,62,0.55)',
            optionBorder: 'rgba(111,240,220,0.3)',
            optionHoverBg: 'rgba(54,103,134,0.55)',
            optionCheckedBg: 'rgba(111,240,220,0.35)',
            optionCheckedBorder: 'rgba(164,255,242,0.85)',
            toggleBg: 'rgba(19,46,62,0.55)',
            toggleBorder: 'rgba(111,240,220,0.3)',
            equationsBg: 'rgba(10,25,40,0.75)',
            equationsText: '#e7fff9',
            equationsTitle: '#9fe0e2',
            statusBg: 'rgba(10,25,40,0.55)',
            statusText: '#e7fff9',
            katexColor: '#e7fff9',
            swatch: ['#6fc8ff', '#ff9fd6'],
            themeBorder: 'rgba(111,240,220,0.45)'
        }
    },
    sunset: {
        id: 'sunset',
        label: 'Sunset',
        scene: {
            background: 0x1c0a29,
            introBackground: 0x13051b
        },
        layerAccents: [0xff9478, 0xffc15e, 0xf96d9b, 0x8d5bfd, 0xfff38b],
        vectorGradient: [
            { stop: -1, color: '#1f164f' },
            { stop: -0.3, color: '#f06c9b' },
            { stop: 0.25, color: '#ffb37a' },
            { stop: 1, color: '#ffd56f' }
        ],
        mhsa: {
            rest: 0x2b1436,
            brightGreen: 0xfff494,
            darkGreen: 0x402b19,
            brightBlue: 0xff90e8,
            darkBlue: 0x2a1136,
            brightRed: 0xff7363,
            darkRed: 0x3c1310,
            finalQ: 0xff9b82,
            finalK: 0xffd66f,
            finalV: 0xff6f91,
            output: 0xffb5f2,
            baseQ: 0xff957e,
            baseK: 0xffd86f,
            baseV: 0xff7aa8
        },
        trailColor: 0xffd8c2,
        embeddings: {
            vocab: 0xff957e,
            positional: 0xffd86f,
            top: 0x1a0a24
        },
        ui: {
            bodyBg: '#14061f',
            controlBg: 'rgba(64,28,74,0.65)',
            controlText: '#fff0eb',
            controlBorder: 'rgba(255,180,140,0.35)',
            controlHoverBg: 'rgba(82,36,98,0.75)',
            controlActiveBg: 'rgba(255,155,130,0.35)',
            controlActiveBorder: 'rgba(255,210,140,0.85)',
            controlFocus: 'rgba(255,205,158,0.95)',
            modalBg: 'rgba(46,19,57,0.92)',
            modalBorder: 'rgba(255,166,143,0.28)',
            modalText: '#fff2ec',
            sectionLabel: '#ffcfb6',
            optionBg: 'rgba(60,23,67,0.6)',
            optionBorder: 'rgba(255,166,143,0.35)',
            optionHoverBg: 'rgba(92,37,104,0.6)',
            optionCheckedBg: 'rgba(255,166,143,0.4)',
            optionCheckedBorder: 'rgba(255,214,160,0.85)',
            toggleBg: 'rgba(60,23,67,0.6)',
            toggleBorder: 'rgba(255,166,143,0.35)',
            equationsBg: 'rgba(46,19,57,0.72)',
            equationsText: '#fff2ec',
            equationsTitle: '#ffcfb6',
            statusBg: 'rgba(46,19,57,0.55)',
            statusText: '#fff2ec',
            katexColor: '#fff2ec',
            swatch: ['#ff9b82', '#ff6f91'],
            themeBorder: 'rgba(255,166,143,0.45)'
        }
    },
    ocean: {
        id: 'ocean',
        label: 'Ocean',
        scene: {
            background: 0x021f30,
            introBackground: 0x01121f
        },
        layerAccents: [0x4bd9ff, 0x2ec7b6, 0x1e88e5, 0x9b8cff, 0x2ef0ff],
        vectorGradient: [
            { stop: -1, color: '#082f5f' },
            { stop: -0.2, color: '#0fb9d8' },
            { stop: 0.25, color: '#0fe6b9' },
            { stop: 1, color: '#9fffe3' }
        ],
        mhsa: {
            rest: 0x103047,
            brightGreen: 0x7bffe0,
            darkGreen: 0x09352b,
            brightBlue: 0x63b4ff,
            darkBlue: 0x0c223c,
            brightRed: 0xff9fb1,
            darkRed: 0x30111d,
            finalQ: 0x63b4ff,
            finalK: 0x5cf2d2,
            finalV: 0xffa6c9,
            output: 0x8bc3ff,
            baseQ: 0x3fa6ff,
            baseK: 0x46e6d0,
            baseV: 0xff8fbf
        },
        trailColor: 0x7bf7ff,
        embeddings: {
            vocab: 0x3fa6ff,
            positional: 0x46e6d0,
            top: 0x012435
        },
        ui: {
            bodyBg: '#011622',
            controlBg: 'rgba(9,44,66,0.72)',
            controlText: '#e8fbff',
            controlBorder: 'rgba(75,217,255,0.35)',
            controlHoverBg: 'rgba(12,58,85,0.82)',
            controlActiveBg: 'rgba(70,214,255,0.32)',
            controlActiveBorder: 'rgba(111,245,255,0.85)',
            controlFocus: 'rgba(125,244,255,0.95)',
            modalBg: 'rgba(5,30,47,0.94)',
            modalBorder: 'rgba(75,217,255,0.3)',
            modalText: '#e8fbff',
            sectionLabel: '#9be7ff',
            optionBg: 'rgba(6,36,54,0.65)',
            optionBorder: 'rgba(75,217,255,0.35)',
            optionHoverBg: 'rgba(17,70,96,0.65)',
            optionCheckedBg: 'rgba(70,214,255,0.4)',
            optionCheckedBorder: 'rgba(111,245,255,0.9)',
            toggleBg: 'rgba(6,36,54,0.65)',
            toggleBorder: 'rgba(75,217,255,0.35)',
            equationsBg: 'rgba(5,30,47,0.8)',
            equationsText: '#e8fbff',
            equationsTitle: '#9be7ff',
            statusBg: 'rgba(5,30,47,0.6)',
            statusText: '#e8fbff',
            katexColor: '#e8fbff',
            swatch: ['#63b4ff', '#5cf2d2'],
            themeBorder: 'rgba(75,217,255,0.45)'
        }
    },
    neon: {
        id: 'neon',
        label: 'Neon',
        scene: {
            background: 0x050014,
            introBackground: 0x020009
        },
        layerAccents: [0xff37a6, 0x37fff7, 0x8d43ff, 0xfff75f, 0x46ff9a],
        vectorGradient: [
            { stop: -1, color: '#32004f' },
            { stop: -0.15, color: '#b300ff' },
            { stop: 0.25, color: '#00e5ff' },
            { stop: 1, color: '#7dff62' }
        ],
        mhsa: {
            rest: 0x14002b,
            brightGreen: 0x7bffcd,
            darkGreen: 0x003320,
            brightBlue: 0x7ad7ff,
            darkBlue: 0x03163b,
            brightRed: 0xff6ec7,
            darkRed: 0x3a0024,
            finalQ: 0x7ad7ff,
            finalK: 0x72ffce,
            finalV: 0xff6ec7,
            output: 0xd28bff,
            baseQ: 0x4bbfff,
            baseK: 0x51ffcb,
            baseV: 0xff58ba
        },
        trailColor: 0x94fff2,
        embeddings: {
            vocab: 0x4bbfff,
            positional: 0x51ffcb,
            top: 0x070018
        },
        ui: {
            bodyBg: '#03000d',
            controlBg: 'rgba(35,0,70,0.75)',
            controlText: '#f9e9ff',
            controlBorder: 'rgba(148,0,255,0.4)',
            controlHoverBg: 'rgba(55,0,100,0.85)',
            controlActiveBg: 'rgba(80,0,180,0.5)',
            controlActiveBorder: 'rgba(132,255,246,0.9)',
            controlFocus: 'rgba(148,255,246,0.95)',
            modalBg: 'rgba(20,0,45,0.92)',
            modalBorder: 'rgba(132,255,246,0.35)',
            modalText: '#f9e9ff',
            sectionLabel: '#a77cff',
            optionBg: 'rgba(30,0,60,0.65)',
            optionBorder: 'rgba(132,255,246,0.35)',
            optionHoverBg: 'rgba(55,0,110,0.7)',
            optionCheckedBg: 'rgba(80,0,180,0.45)',
            optionCheckedBorder: 'rgba(170,255,250,0.9)',
            toggleBg: 'rgba(30,0,60,0.65)',
            toggleBorder: 'rgba(132,255,246,0.35)',
            equationsBg: 'rgba(20,0,45,0.75)',
            equationsText: '#f9e9ff',
            equationsTitle: '#a77cff',
            statusBg: 'rgba(20,0,45,0.65)',
            statusText: '#f9e9ff',
            katexColor: '#f9e9ff',
            swatch: ['#7ad7ff', '#ff6ec7'],
            themeBorder: 'rgba(148,0,255,0.45)'
        }
    },
    noir: {
        id: 'noir',
        label: 'Noir',
        scene: {
            background: 0x070707,
            introBackground: 0x000000
        },
        layerAccents: [0xffffff, 0xd3d3d3, 0xa0a0a0, 0x7a7a7a, 0x4c4c4c],
        vectorGradient: [
            { stop: -1, color: '#0a0a0a' },
            { stop: 0, color: '#707070' },
            { stop: 1, color: '#f5f5f5' }
        ],
        mhsa: {
            rest: 0x1a1a1a,
            brightGreen: 0xe0e0e0,
            darkGreen: 0x0f0f0f,
            brightBlue: 0xcacaca,
            darkBlue: 0x080808,
            brightRed: 0xf0f0f0,
            darkRed: 0x111111,
            finalQ: 0xb0b0b0,
            finalK: 0xe0e0e0,
            finalV: 0x8e8e8e,
            output: 0xc0c0c0,
            baseQ: 0xb0b0b0,
            baseK: 0xe0e0e0,
            baseV: 0x8e8e8e
        },
        trailColor: 0xffffff,
        embeddings: {
            vocab: 0xb0b0b0,
            positional: 0xe0e0e0,
            top: 0x050505
        },
        ui: {
            bodyBg: '#050505',
            controlBg: 'rgba(28,28,28,0.82)',
            controlText: '#f1f1f1',
            controlBorder: 'rgba(220,220,220,0.3)',
            controlHoverBg: 'rgba(45,45,45,0.88)',
            controlActiveBg: 'rgba(210,210,210,0.35)',
            controlActiveBorder: 'rgba(255,255,255,0.7)',
            controlFocus: 'rgba(255,255,255,0.9)',
            modalBg: 'rgba(15,15,15,0.92)',
            modalBorder: 'rgba(220,220,220,0.25)',
            modalText: '#f1f1f1',
            sectionLabel: '#cccccc',
            optionBg: 'rgba(20,20,20,0.7)',
            optionBorder: 'rgba(220,220,220,0.25)',
            optionHoverBg: 'rgba(36,36,36,0.75)',
            optionCheckedBg: 'rgba(210,210,210,0.35)',
            optionCheckedBorder: 'rgba(255,255,255,0.7)',
            toggleBg: 'rgba(20,20,20,0.7)',
            toggleBorder: 'rgba(220,220,220,0.25)',
            equationsBg: 'rgba(15,15,15,0.82)',
            equationsText: '#f1f1f1',
            equationsTitle: '#cccccc',
            statusBg: 'rgba(15,15,15,0.7)',
            statusText: '#f1f1f1',
            katexColor: '#f1f1f1',
            swatch: ['#b0b0b0', '#8e8e8e'],
            themeBorder: 'rgba(220,220,220,0.4)'
        }
    }
};

let currentThemeId = getPreference('colorTheme', 'original');
if (!THEME_DEFINITIONS[currentThemeId]) {
    currentThemeId = 'original';
}

const listeners = new Set();

function applyCssVariables(theme) {
    if (typeof document === 'undefined') return;
    const root = document.documentElement;
    const vars = theme.ui || {};
    Object.entries(vars).forEach(([key, value]) => {
        if (Array.isArray(value) || value === null || typeof value === 'object') return;
        root.style.setProperty(`--${key.replace(/[A-Z]/g, m => `-${m.toLowerCase()}`)}`, value);
    });
}

function notify(theme, previous) {
    listeners.forEach((cb) => {
        try {
            cb(theme, previous);
        } catch (err) {
            console.error('Theme listener failed:', err);
        }
    });
}

applyCssVariables(THEME_DEFINITIONS[currentThemeId]);

export function getAvailableThemes() {
    return Object.values(THEME_DEFINITIONS).map(({ id, label, ui }) => ({
        id,
        label,
        swatch: Array.isArray(ui?.swatch) ? ui.swatch.slice(0, 2) : null
    }));
}

export function getCurrentThemeId() {
    return currentThemeId;
}

export function getCurrentTheme() {
    return THEME_DEFINITIONS[currentThemeId];
}

export function setTheme(themeId) {
    const next = THEME_DEFINITIONS[themeId];
    if (!next || next.id === currentThemeId) return;
    const previous = THEME_DEFINITIONS[currentThemeId];
    currentThemeId = next.id;
    setPreference('colorTheme', currentThemeId);
    applyCssVariables(next);
    notify(next, previous);
}

export function onThemeChange(listener) {
    if (typeof listener !== 'function') return () => {};
    listeners.add(listener);
    return () => listeners.delete(listener);
}

export function getThemeColorStops() {
    const theme = getCurrentTheme();
    return Array.isArray(theme.vectorGradient) ? theme.vectorGradient : null;
}

export function getThemeLayerAccents() {
    const theme = getCurrentTheme();
    return theme.layerAccents || [];
}

export function getThemeSceneBackground() {
    const theme = getCurrentTheme();
    return theme.scene?.background ?? 0x000000;
}

export function getThemeIntroBackground() {
    const theme = getCurrentTheme();
    return theme.scene?.introBackground ?? 0x000000;
}

export function getThemeMhsaColors() {
    const theme = getCurrentTheme();
    return theme.mhsa;
}

export function getThemeTrailColor() {
    const theme = getCurrentTheme();
    return theme.trailColor ?? 0xffffff;
}

export function getThemeEmbeddingColors() {
    const theme = getCurrentTheme();
    return theme.embeddings || {};
}
