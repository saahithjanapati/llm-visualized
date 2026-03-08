import * as THREE from 'three';

export function smootherstep01(value) {
    const t = THREE.MathUtils.clamp(value, 0, 1);
    return t * t * t * (t * (t * 6 - 15) + 10);
}

export function smoothSegmentProgress(progress, start, end) {
    if (!Number.isFinite(start) || !Number.isFinite(end)) {
        return smootherstep01(progress);
    }
    if (end <= start) {
        return progress >= end ? 1 : 0;
    }
    return smootherstep01((progress - start) / (end - start));
}

export function computeTravelArcHeight(distance, {
    minArc = 0,
    maxArc = Number.POSITIVE_INFINITY,
    distanceScale = 0,
} = {}) {
    if (!Number.isFinite(distance) || distance <= 0) return 0;
    const scaled = distance * distanceScale;
    return THREE.MathUtils.clamp(scaled, minArc, maxArc);
}

export function getArcOffset(progress, arcHeight) {
    if (!Number.isFinite(arcHeight) || arcHeight <= 0) return 0;
    const t = smootherstep01(progress);
    return Math.sin(t * Math.PI) * arcHeight;
}
