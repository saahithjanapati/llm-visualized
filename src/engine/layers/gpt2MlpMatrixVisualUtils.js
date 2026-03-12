import * as THREE from 'three';

const MLP_MATRIX_EMISSIVE_GROWTH_END_T = 0.92;
const MLP_MATRIX_EMISSIVE_EXIT_START_T = 0.92;

export function computeMlpMatrixPassEmissive(progress, startIntensity, peakIntensity, finalIntensity) {
    const clampedProgress = THREE.MathUtils.clamp(
        Number.isFinite(progress) ? progress : 0,
        0,
        1
    );
    const start = Number.isFinite(startIntensity) ? startIntensity : 0;
    const peak = Number.isFinite(peakIntensity) ? peakIntensity : start;
    const finalValue = Number.isFinite(finalIntensity) ? finalIntensity : peak;

    // Let the glow keep growing through almost the whole pass so longer
    // matrix traversals read as accumulating energy instead of an early hold.
    const growthT = THREE.MathUtils.smootherstep(
        clampedProgress,
        0,
        MLP_MATRIX_EMISSIVE_GROWTH_END_T
    );
    const activeEmissive = THREE.MathUtils.lerp(start, peak, growthT);
    const exitT = THREE.MathUtils.smootherstep(
        clampedProgress,
        MLP_MATRIX_EMISSIVE_EXIT_START_T,
        1
    );

    return THREE.MathUtils.lerp(activeEmissive, finalValue, exitT);
}
