import { getPreference, setPreference } from '../utils/preferences.js';

const THEMES = {
    original: {
        id: 'original',
        label: 'Original',
        description: 'Classic contrast with warm amber matrices.',
        scene: {
            background: 0x000000,
            ambientLight: { color: 0xffffff, intensity: 0.7 },
            directionalLight: { color: 0xffffff, intensity: 0.9 },
            inactiveComponent: 0x202020,
            mlpActive: 0xb07c13,
            trailColor: 0xffffff,
            layerAccents: [0xff6f61, 0xffd166, 0x06d6a0, 0x118ab2, 0x9d4edd],
            mhsa: {
                matrixRest: 0x404040,
                brightGreen: 0x33ff33,
                darkGreen: 0x002200,
                brightBlue: 0x6666ff,
                darkBlue: 0x000022,
                brightRed: 0xff3333,
                darkRed: 0x220000,
                finalQ: 0x1e5299,
                finalK: 0x1d9752,
                finalV: 0x811b2d,
                outputProjection: 0xaf5faf,
            },
        },
    },
    midnight: {
        id: 'midnight',
        label: 'Midnight Aurora',
        description: 'Deep navy with electric blues and violets.',
        scene: {
            background: 0x020622,
            ambientLight: { color: 0x2c3d73, intensity: 0.55 },
            directionalLight: { color: 0xaad4ff, intensity: 0.8 },
            inactiveComponent: 0x1b2546,
            mlpActive: 0x76a0ff,
            trailColor: 0x9fb5ff,
            layerAccents: [0x4f6cff, 0x8dacef, 0x34d1ff, 0x7f77ff, 0x4ef3ff],
            mhsa: {
                matrixRest: 0x1d2749,
                brightGreen: 0x5af5c5,
                darkGreen: 0x053a2b,
                brightBlue: 0x6a8dff,
                darkBlue: 0x09163f,
                brightRed: 0xff6e8f,
                darkRed: 0x2d0b1d,
                finalQ: 0x6aa7ff,
                finalK: 0x5fffd0,
                finalV: 0xff7fb3,
                outputProjection: 0x7e8aff,
            },
        },
    },
    aurora: {
        id: 'aurora',
        label: 'Aurora Bloom',
        description: 'Glacial teal with magenta highlights.',
        scene: {
            background: 0x04110f,
            ambientLight: { color: 0x4bd6c6, intensity: 0.5 },
            directionalLight: { color: 0xff7bc2, intensity: 0.7 },
            inactiveComponent: 0x10332d,
            mlpActive: 0x5fffe4,
            trailColor: 0xff9be7,
            layerAccents: [0x4de0d2, 0xff93d1, 0x79ffa2, 0x7ab8ff, 0xffc169],
            mhsa: {
                matrixRest: 0x123c37,
                brightGreen: 0x64ffd3,
                darkGreen: 0x062924,
                brightBlue: 0x73cfff,
                darkBlue: 0x07243c,
                brightRed: 0xff93d8,
                darkRed: 0x381224,
                finalQ: 0x5fe0ff,
                finalK: 0x79ffb0,
                finalV: 0xff9be7,
                outputProjection: 0xff8bd1,
            },
        },
    },
    sunset: {
        id: 'sunset',
        label: 'Desert Sunset',
        description: 'Amber dusk with coral accents.',
        scene: {
            background: 0x120304,
            ambientLight: { color: 0xffb48b, intensity: 0.6 },
            directionalLight: { color: 0xfff2d0, intensity: 0.8 },
            inactiveComponent: 0x30140d,
            mlpActive: 0xff7a45,
            trailColor: 0xffd3a8,
            layerAccents: [0xff9a62, 0xffc56f, 0xf76d7c, 0xff8da1, 0xffd080],
            mhsa: {
                matrixRest: 0x2f1410,
                brightGreen: 0xffd86f,
                darkGreen: 0x3a1a07,
                brightBlue: 0xff8fa3,
                darkBlue: 0x2d0a0f,
                brightRed: 0xff6b5a,
                darkRed: 0x3f0b09,
                finalQ: 0xffa45e,
                finalK: 0xffd27a,
                finalV: 0xff7fa7,
                outputProjection: 0xff9154,
            },
        },
    },
    forest: {
        id: 'forest',
        label: 'Forest Canopy',
        description: 'Verdant greens with soft sky highlights.',
        scene: {
            background: 0x07140a,
            ambientLight: { color: 0x88ffb2, intensity: 0.5 },
            directionalLight: { color: 0xf2ffe0, intensity: 0.75 },
            inactiveComponent: 0x153120,
            mlpActive: 0x8aff7a,
            trailColor: 0xd2ffd6,
            layerAccents: [0x8aff7a, 0x4dd088, 0xb3ff9a, 0x6ee6c4, 0xf1ffb4],
            mhsa: {
                matrixRest: 0x1a3725,
                brightGreen: 0x9bff90,
                darkGreen: 0x0b2a15,
                brightBlue: 0x6fd6ff,
                darkBlue: 0x0d2130,
                brightRed: 0xffa877,
                darkRed: 0x35160b,
                finalQ: 0x6fe0ff,
                finalK: 0x9bff90,
                finalV: 0xffa877,
                outputProjection: 0x8cff82,
            },
        },
    },
    mono: {
        id: 'mono',
        label: 'Monochrome Glow',
        description: 'Minimal grayscale with soft highlights.',
        scene: {
            background: 0x101010,
            ambientLight: { color: 0xffffff, intensity: 0.55 },
            directionalLight: { color: 0xffffff, intensity: 0.8 },
            inactiveComponent: 0x2a2a2a,
            mlpActive: 0xb5b5b5,
            trailColor: 0xe0e0e0,
            layerAccents: [0xffffff, 0xcccccc, 0x999999, 0x777777, 0x555555],
            mhsa: {
                matrixRest: 0x2a2a2a,
                brightGreen: 0xd0ffd0,
                darkGreen: 0x202020,
                brightBlue: 0xbfd8ff,
                darkBlue: 0x1b1b1b,
                brightRed: 0xffbfbf,
                darkRed: 0x1f1f1f,
                finalQ: 0xbfd8ff,
                finalK: 0xd0ffd0,
                finalV: 0xffbfbf,
                outputProjection: 0xc8c8c8,
            },
        },
    },
};

let currentThemeId = getPreference('colorTheme', 'original');
if (!Object.prototype.hasOwnProperty.call(THEMES, currentThemeId)) {
    currentThemeId = 'original';
}

let currentTheme = THEMES[currentThemeId];
const listeners = new Set();

function applyThemeAttribute(themeId) {
    if (typeof document === 'undefined') return;
    const apply = () => {
        if (document.body) {
            document.body.setAttribute('data-theme', themeId);
        }
    };
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', apply, { once: true });
    } else {
        apply();
    }
}

function notify(theme) {
    listeners.forEach((listener) => {
        try {
            listener(theme);
        } catch (err) {
            console.error('Theme listener failed:', err);
        }
    });
}

export function getAvailableThemes() {
    return Object.values(THEMES).map(({ id, label, description }) => ({ id, label, description }));
}

export function getCurrentThemeId() {
    return currentThemeId;
}

export function getCurrentTheme() {
    return currentTheme;
}

export function setTheme(themeId) {
    if (!Object.prototype.hasOwnProperty.call(THEMES, themeId)) return;
    if (themeId === currentThemeId) return;
    currentThemeId = themeId;
    currentTheme = THEMES[themeId];
    try {
        setPreference('colorTheme', themeId);
    } catch (err) {
        console.warn('Failed to persist theme preference:', err);
    }
    applyThemeAttribute(themeId);
    notify(currentTheme);
}

export function subscribeToThemeChanges(listener) {
    if (typeof listener !== 'function') return () => {};
    listeners.add(listener);
    return () => {
        listeners.delete(listener);
    };
}

applyThemeAttribute(currentThemeId);

export { THEMES };

