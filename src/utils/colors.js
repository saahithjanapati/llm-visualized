import * as THREE from 'three';
import { VECTOR_LENGTH } from './constants.js';
import { getThemeColorStops } from '../state/themeState.js';

function mapValueUsingStops(value, stops) {
    if (!Array.isArray(stops) || stops.length === 0) return null;
    const clamped = Math.max(-1, Math.min(1, Number(value)));
    const sorted = stops.slice().sort((a, b) => (a.stop ?? 0) - (b.stop ?? 0));
    const first = sorted[0];
    const last = sorted[sorted.length - 1];

    if (clamped <= (first.stop ?? -1)) {
        return new THREE.Color(first.color ?? 0xffffff);
    }
    if (clamped >= (last.stop ?? 1)) {
        return new THREE.Color(last.color ?? 0xffffff);
    }

    for (let i = 1; i < sorted.length; i += 1) {
        const prev = sorted[i - 1];
        const next = sorted[i];
        const start = prev.stop ?? -1;
        const end = next.stop ?? 1;
        if (clamped <= end) {
            const span = Math.max(1e-6, end - start);
            const t = THREE.MathUtils.clamp((clamped - start) / span, 0, 1);
            const startColor = new THREE.Color(prev.color ?? 0xffffff);
            const endColor = new THREE.Color(next.color ?? 0xffffff);
            return startColor.lerp(endColor, t);
        }
    }

    return new THREE.Color(last.color ?? 0xffffff);
}

export function mapValueToColor(value) {
    const stops = getThemeColorStops();
    if (Array.isArray(stops) && stops.length) {
        const gradient = mapValueUsingStops(value, stops);
        if (gradient) return gradient;
    }

    const clampedValue = Math.max(-1, Math.min(1, value / 2));
    const hue = (clampedValue + 1) / 2;
    const saturation = 1.0;
    const lightness = 0.4;
    return new THREE.Color().setHSL(hue, saturation, lightness);
}

let mapValueToColorCallCount = 0;

export function mapValueToColor_LOG(value, index) {
    mapValueToColorCallCount += 1;
    const shouldLog = index < 5
        || index >= VECTOR_LENGTH - 5
        || (index >= Math.floor(VECTOR_LENGTH / 2) - 2 && index <= Math.floor(VECTOR_LENGTH / 2) + 2);

    if (shouldLog) {
        const formatted = (typeof value === 'number' && Number.isFinite(value)) ? value.toFixed(3) : value;
        console.log(`mapValueToColor_LOG (idx ${index}, call #${mapValueToColorCallCount}): input=${formatted}`);
    }

    const stops = getThemeColorStops();
    if (Array.isArray(stops) && stops.length) {
        const gradient = mapValueUsingStops(value, stops);
        if (gradient) {
            if (shouldLog) {
                console.log(` -> gradient RGB=(${gradient.r.toFixed(3)}, ${gradient.g.toFixed(3)}, ${gradient.b.toFixed(3)})`);
            }
            return gradient;
        }
    }

    const clampedValue = Math.max(-1, Math.min(1, value / 2));
    const hue = (clampedValue + 1) / 2;
    const saturation = 1.0;
    const lightness = 0.6;
    const finalColor = new THREE.Color().setHSL(hue, saturation, lightness);

    if (shouldLog) {
        console.log(` -> clamped=${clampedValue.toFixed(3)}, hue=${hue.toFixed(3)}, L=0.6, RGB=(${finalColor.r.toFixed(3)}, ${finalColor.g.toFixed(3)}, ${finalColor.b.toFixed(3)})`);
    }
    return finalColor;
}

export function mapNormalizedValueToBrightColor(value, targetColorInstance) {
    let h = 0.33;
    let s = 0.9;
    let l = 0.6;

    if (typeof value === 'number' && isFinite(value)) {
        const clampedValue = Math.max(0, Math.min(1, value));
        h = clampedValue * 0.8;
        s = 0.9;
        l = 0.7;
    } else if (typeof value !== 'number') {
        console.log(`Bad color value: ${value}`);
    }

    if (targetColorInstance) {
        targetColorInstance.setHSL(h, s, l);
        return targetColorInstance;
    }
    return new THREE.Color().setHSL(h, s, l);
}
