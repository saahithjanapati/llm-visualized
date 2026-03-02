import { getPreference, setPreference } from '../utils/preferences.js';
import { setGlobalEmissiveIntensityScale } from '../utils/materialUtils.js';

const BRIGHTNESS_PREF_KEY = 'displayBrightnessScale';
const FIXED_EMISSIVE_SCALE = 0.75;
const BRIGHTNESS_DEFAULT = 1.0;

const BRIGHTNESS_MIN = 0.5;
const BRIGHTNESS_MAX = 1.8;

const clamp = (value, min, max, fallback) => {
    const next = Number(value);
    if (!Number.isFinite(next)) return fallback;
    return Math.max(min, Math.min(max, next));
};

const fmt = (value) => `${Number(value).toFixed(2)}x`;

export function initLiveVisualControls(pipeline) {
    const root = document.getElementById('liveVisualControls');
    const brightnessInput = document.getElementById('liveBrightness');
    const brightnessValue = document.getElementById('liveBrightnessValue');
    if (!root || !brightnessInput || !brightnessValue) return;

    const applyBrightness = (value, { persist = true } = {}) => {
        const next = clamp(value, BRIGHTNESS_MIN, BRIGHTNESS_MAX, BRIGHTNESS_DEFAULT);
        brightnessInput.value = next.toFixed(2);
        brightnessValue.textContent = fmt(next);
        const css = `brightness(${next.toFixed(2)})`;
        const gptCanvas = document.getElementById('gptCanvas');
        const introCanvas = document.getElementById('introCanvas');
        if (gptCanvas) gptCanvas.style.filter = css;
        if (introCanvas) introCanvas.style.filter = css;
        const engineCanvas = pipeline?.engine?.renderer?.domElement;
        if (engineCanvas && engineCanvas !== gptCanvas) engineCanvas.style.filter = css;
        if (persist) setPreference(BRIGHTNESS_PREF_KEY, next);
    };

    setGlobalEmissiveIntensityScale(FIXED_EMISSIVE_SCALE, pipeline?.engine?.scene || null);

    const initialBrightness = getPreference(BRIGHTNESS_PREF_KEY, BRIGHTNESS_DEFAULT);
    applyBrightness(initialBrightness, { persist: false });

    brightnessInput.addEventListener('input', () => applyBrightness(brightnessInput.value, { persist: false }));
    brightnessInput.addEventListener('change', () => applyBrightness(brightnessInput.value, { persist: true }));
}

export default initLiveVisualControls;
