import * as THREE from 'three';

const OUTPUT_PROJ_COLOR_RAMP_END_T = 0.4;
const OUTPUT_PROJ_EMISSIVE_PEAK_T = 0.58;

export function applyOutputProjectionPassVisual({
    progress,
    startColor,
    activeColor,
    targetColor,
    startEmissiveIntensity,
    peakEmissiveIntensity,
    endEmissiveIntensity,
}) {
    if (!startColor || !activeColor || !targetColor) {
        return {
            color: targetColor || null,
            emissiveIntensity: Number.isFinite(endEmissiveIntensity) ? endEmissiveIntensity : 0,
        };
    }

    const clampedProgress = THREE.MathUtils.clamp(
        Number.isFinite(progress) ? progress : 0,
        0,
        1
    );
    const colorT = THREE.MathUtils.smootherstep(
        clampedProgress,
        0,
        OUTPUT_PROJ_COLOR_RAMP_END_T
    );
    const start = Number.isFinite(startEmissiveIntensity) ? startEmissiveIntensity : 0;
    const peak = Number.isFinite(peakEmissiveIntensity) ? peakEmissiveIntensity : start;
    const end = Number.isFinite(endEmissiveIntensity) ? endEmissiveIntensity : peak;

    targetColor.copy(startColor).lerp(activeColor, colorT);

    let emissiveIntensity = peak;
    if (clampedProgress <= OUTPUT_PROJ_EMISSIVE_PEAK_T) {
        const riseT = THREE.MathUtils.smootherstep(
            clampedProgress,
            0,
            OUTPUT_PROJ_EMISSIVE_PEAK_T
        );
        emissiveIntensity = THREE.MathUtils.lerp(start, peak, riseT);
    } else {
        const fallT = THREE.MathUtils.smootherstep(
            clampedProgress,
            OUTPUT_PROJ_EMISSIVE_PEAK_T,
            1
        );
        emissiveIntensity = THREE.MathUtils.lerp(peak, end, fallT);
    }

    return {
        color: targetColor,
        emissiveIntensity,
    };
}
