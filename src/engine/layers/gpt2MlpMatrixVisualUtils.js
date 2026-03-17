import * as THREE from 'three';

const MLP_MATRIX_EMISSIVE_PEAK_T = 0.68;

export function computeMlpMatrixPassEmissive(progress, startIntensity, peakIntensity, finalIntensity) {
    const clampedProgress = THREE.MathUtils.clamp(
        Number.isFinite(progress) ? progress : 0,
        0,
        1
    );
    const start = Number.isFinite(startIntensity) ? startIntensity : 0;
    const peak = Number.isFinite(peakIntensity) ? peakIntensity : start;
    const finalValue = Number.isFinite(finalIntensity) ? finalIntensity : peak;

    // Let the matrix keep charging through most of the pass, then spend the
    // last third cooling down so the glow reads as a fade instead of a snap.
    if (clampedProgress <= MLP_MATRIX_EMISSIVE_PEAK_T) {
        const riseT = THREE.MathUtils.smootherstep(
            clampedProgress,
            0,
            MLP_MATRIX_EMISSIVE_PEAK_T
        );
        return THREE.MathUtils.lerp(start, peak, riseT);
    }

    const fallT = THREE.MathUtils.smootherstep(
        clampedProgress,
        MLP_MATRIX_EMISSIVE_PEAK_T,
        1
    );
    return THREE.MathUtils.lerp(peak, finalValue, fallT);
}
