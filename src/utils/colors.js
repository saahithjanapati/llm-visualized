import * as THREE from 'three';
import { VECTOR_LENGTH } from './constants.js'; // Added import for logging

const MONO_MIN_LIGHTNESS = 0.45;
const MONO_MAX_LIGHTNESS = 0.35;
const SPECTRUM_CLAMP_MAX = 2;
const SPECTRUM_HUE_MIN = 0;
const SPECTRUM_HUE_MID = 0.5; // cyan/light blue at zero
const SPECTRUM_HUE_MAX = 2 / 3; // blue at max
const SPECTRUM_CENTER_RANGE = 0.35; // Portion of value range treated as "center" (0-1, normalized)
const SPECTRUM_CENTER_SPAN = 0.9; // Portion of hue span reserved for center values (0-1)

function applySpectrumStretch(normalized) {
    const safeRange = Number.isFinite(SPECTRUM_CENTER_RANGE)
        ? Math.max(0, Math.min(1, SPECTRUM_CENTER_RANGE))
        : 0;
    const safeSpan = Number.isFinite(SPECTRUM_CENTER_SPAN)
        ? Math.max(0, Math.min(1, SPECTRUM_CENTER_SPAN))
        : 0;

    if (safeRange <= 0 || safeRange >= 1 || safeSpan <= 0 || safeSpan >= 1) {
        return normalized;
    }

    const absNorm = Math.abs(normalized);
    if (absNorm <= safeRange) {
        const t = safeRange > 0 ? absNorm / safeRange : 0;
        return Math.sign(normalized) * t * safeSpan;
    }

    const t = (absNorm - safeRange) / (1 - safeRange);
    return Math.sign(normalized) * (safeSpan + t * (1 - safeSpan));
}

function mapCurvedToHue(curved) {
    if (curved <= 0) {
        const t = curved + 1; // [-1, 0] -> [0, 1]
        return THREE.MathUtils.lerp(SPECTRUM_HUE_MIN, SPECTRUM_HUE_MID, t);
    }
    const t = curved; // [0, 1] -> [0, 1]
    return THREE.MathUtils.lerp(SPECTRUM_HUE_MID, SPECTRUM_HUE_MAX, t);
}

// Map a value (normalized, potentially outside -1 to 1) to a rainbow color (HSL)
export function mapValueToColor(value) {
    // console.log(`mapValueToColor input value: ${value}`); // Log input value

    // Clamp raw values to the visualised range before mapping to hue.
    const safeValue = Number.isFinite(value) ? value : 0;
    const clampedValue = Math.max(-SPECTRUM_CLAMP_MAX, Math.min(SPECTRUM_CLAMP_MAX, safeValue));
    const normalized = clampedValue / SPECTRUM_CLAMP_MAX;
    const curved = applySpectrumStretch(normalized);

    // Map [-1, 0, 1] -> [red, cyan, blue] with a shifted midpoint.
    const hue = mapCurvedToHue(curved);
    const saturation = 1.0;
    // Keep lightness higher so colors remain readable on black backgrounds.
    const lightness = 0.6;
    const finalColor = new THREE.Color().setHSL(hue, saturation, lightness);
    // console.log(`mapValueToColor output: value=${value}, clamped=${clampedValue}, hue=${hue}, color R=${finalColor.r} G=${finalColor.g} B=${finalColor.b}`);
    return finalColor;
}

export function buildMonochromeOptions(color, { valueMin = -2, valueMax = 2 } = {}) {
    const hsl = { h: 0, s: 0, l: 0 };
    let sourceColor = color;
    if ((typeof color === 'number' || typeof color === 'string') && THREE.Color) {
        try {
            sourceColor = new THREE.Color(color);
        } catch (_) {
            sourceColor = null;
        }
    }
    if (sourceColor && typeof sourceColor.getHSL === 'function') {
        sourceColor.getHSL(hsl);
    }
    const baseSat = Number.isFinite(hsl.s) ? hsl.s : 0.9;
    return {
        type: 'monochromatic',
        baseHue: hsl.h,
        saturation: Math.min(1, Math.max(0.85, baseSat * 1.2)),
        minLightness: MONO_MIN_LIGHTNESS,
        maxLightness: MONO_MAX_LIGHTNESS,
        valueMin,
        valueMax,
    };
}

export function mapValueToMonochrome(value, options = {}) {
    const minLightness = Number.isFinite(options.minLightness) ? options.minLightness : MONO_MIN_LIGHTNESS;
    const maxLightness = Number.isFinite(options.maxLightness) ? options.maxLightness : MONO_MAX_LIGHTNESS;
    const baseHue = Number.isFinite(options.baseHue) ? options.baseHue : 0;
    const saturation = Number.isFinite(options.saturation) ? options.saturation : 0.9;
    const valueMin = Number.isFinite(options.valueMin) ? options.valueMin : -2;
    const valueMax = Number.isFinite(options.valueMax) ? options.valueMax : 2;
    const denom = valueMax - valueMin;
    const clamped = Number.isFinite(value) ? THREE.MathUtils.clamp(value, valueMin, valueMax) : valueMin + denom * 0.5;
    const t = denom > 0 ? (clamped - valueMin) / denom : 0.5;
    const lightness = THREE.MathUtils.lerp(minLightness, maxLightness, t);
    return new THREE.Color().setHSL(baseHue, saturation, lightness);
}

export function mapValueToGrayscale(value) {
    const t = Math.max(0, Math.min(1, Number.isFinite(value) ? value : 0));
    return new THREE.Color(t, t, t);
}

// Add a counter to limit logging if needed, e.g., for mapValueToColor
// let mapValueToColorCallCount = 0;
// export function mapValueToColor(value) {
//     mapValueToColorCallCount++;
//     if (mapValueToColorCallCount <= 10 || mapValueToColorCallCount > VECTOR_LENGTH - 10) { // Log first 10 and last 10 for a typical vector
//         console.log(`mapValueToColor (call #${mapValueToColorCallCount}) input value: ${value}`);
//     }
// ... (rest of the function) ...
//     if (mapValueToColorCallCount <= 10 || mapValueToColorCallCount > VECTOR_LENGTH - 10) {
//        console.log(`mapValueToColor output: value=${value}, clamped=${clampedValue}, hue=${hue}, color R=${finalColor.r} G=${finalColor.g} B=${finalColor.b}`);
//     }
//     return finalColor;
// }
// For more targeted logging from VectorVisualization:
// Modify mapValueToColor to accept an optional index for logging context
let mapValueToColorCallCount = 0;
export function mapValueToColor_LOG(value, index) {
    mapValueToColorCallCount++;
    // Log for first 5, last 5, and a few in the middle to avoid excessive logging for VECTOR_LENGTH=100
    const shouldLog = index < 5 || 
                      index >= VECTOR_LENGTH - 5 || 
                      (index >= Math.floor(VECTOR_LENGTH/2) - 2 && index <= Math.floor(VECTOR_LENGTH/2) + 2);

    if (shouldLog) {
        console.log(`mapValueToColor_LOG (idx ${index}, call #${mapValueToColorCallCount}): input=${value !== undefined && value !== null ? value.toFixed(3) : value}`);
    }

    // Original logic from mapValueToColor
    const safeValue = Number.isFinite(value) ? value : 0;
    const clampedValue = Math.max(-SPECTRUM_CLAMP_MAX, Math.min(SPECTRUM_CLAMP_MAX, safeValue));
    const normalized = clampedValue / SPECTRUM_CLAMP_MAX;
    const curved = applySpectrumStretch(normalized);
    const hue = mapCurvedToHue(curved);
    const saturation = 1.0;
    const lightness = 0.6;
    const finalColor = new THREE.Color().setHSL(hue, saturation, lightness);

    if (shouldLog) {
        console.log(` -> clamped=${clampedValue.toFixed(3)}, curved=${curved.toFixed(3)}, hue=${hue.toFixed(3)}, L=0.6, RGB=(${finalColor.r.toFixed(3)}, ${finalColor.g.toFixed(3)}, ${finalColor.b.toFixed(3)})`);
    }
    return finalColor;
}

// Original function if the _LOG version is not used or for easy revert
// export function mapValueToColor(value) {
//     const clampedValue = Math.max(-1, Math.min(1, value / 2));
//     const hue = (clampedValue + 1) / 2;
//     const saturation = 1.0;
//     const lightness = 0.4;
//     return new THREE.Color().setHSL(hue, saturation, lightness);
// }

// To use the logging version, you'd call mapValueToColor_LOG from VectorVisualization.js
// For now, let's make the original function log directly but less frequently to avoid flooding.

// Re-defining the original function with some conditional logging:
// A global variable to import VECTOR_LENGTH if not already available for logging conditions
// import { VECTOR_LENGTH } from './constants.js'; // Assuming this path is correct from colors.js
// ^ This import might cause circular dependency issues if constants.js imports from colors.js implicitly or explicitly.
// It's safer to pass VECTOR_LENGTH or index if complex conditional logging is needed.

// Simpler logging for now in the main function, will be very verbose for VECTOR_LENGTH=100
// export function mapValueToColor(value) {
//     console.log(`mapValueToColor input value: ${value}`); // Log input value
//     const clampedValue = Math.max(-1, Math.min(1, value / 2));
//     const hue = (clampedValue + 1) / 2;
//     const saturation = 1.0;
//     const lightness = 0.4;
//     const finalColor = new THREE.Color().setHSL(hue, saturation, lightness);
//     console.log(`mapValueToColor output: value=${value}, clamped=${clampedValue}, hue=${hue}, color R=${finalColor.r} G=${finalColor.g} B=${finalColor.b}`);
//     return finalColor;
// }

export function mapNormalizedValueToBrightColor(value, targetColorInstance) {
    let h = 0.33; // Default hue (green)
    let s = 0.9;  // Default saturation
    let l = 0.6;  // Default lightness

    if (typeof value === 'number' && isFinite(value)) {
        const clampedValue = Math.max(0, Math.min(1, value));
        h = clampedValue * 0.8; // Map the 0-1 value to a hue range (e.g., 0 to 0.8 for red to blue)
        s = 0.9; // High saturation
        l = 0.7;  // High lightness
    } else {
        // console.warn(`mapNormalizedValueToBrightColor received non-finite value: ${value}. Defaulting color.`);
        // Using a less intrusive log, or remove if too noisy during normal operation with fixed layerNormalize
        if (typeof value !== 'number') console.log(`Bad color value: ${value}`); 
    }

    if (targetColorInstance) {
        targetColorInstance.setHSL(h, s, l);
        return targetColorInstance;
    }
    return new THREE.Color().setHSL(h, s, l);
}
