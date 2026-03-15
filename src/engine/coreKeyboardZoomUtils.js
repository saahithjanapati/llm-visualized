const KEYBOARD_ZOOM_BOOST_START_DISTANCE_RATIO = 0.32;
const KEYBOARD_ZOOM_BOOST_END_DISTANCE_RATIO = 0.94;
const KEYBOARD_ZOOM_BOOST_MAX_MULTIPLIER = 5.0;

function clamp(value, min, max) {
    return Math.min(max, Math.max(min, value));
}

function smoothstep(edge0, edge1, value) {
    if (!(Number.isFinite(edge0) && Number.isFinite(edge1)) || edge0 === edge1) {
        return value >= edge1 ? 1 : 0;
    }
    const t = clamp((value - edge0) / (edge1 - edge0), 0, 1);
    return t * t * (3 - (2 * t));
}

function resolveKeyboardZoomBoostMultiplier({
    direction,
    currentDistance,
    minDistance,
    maxDistance,
    boostStartDistanceRatio = KEYBOARD_ZOOM_BOOST_START_DISTANCE_RATIO,
    boostEndDistanceRatio = KEYBOARD_ZOOM_BOOST_END_DISTANCE_RATIO,
    boostMaxMultiplier = KEYBOARD_ZOOM_BOOST_MAX_MULTIPLIER
} = {}) {
    if (direction <= 0) return 1;
    if (!(Number.isFinite(currentDistance) && currentDistance > 0)) return 1;
    if (!(Number.isFinite(minDistance) && minDistance >= 0)) return 1;
    if (!(Number.isFinite(maxDistance) && maxDistance > minDistance)) return 1;
    if (!(Number.isFinite(boostMaxMultiplier) && boostMaxMultiplier > 1)) return 1;

    const distanceRatio = clamp(
        (currentDistance - minDistance) / (maxDistance - minDistance),
        0,
        1
    );
    const boostT = smoothstep(boostStartDistanceRatio, boostEndDistanceRatio, distanceRatio);
    return 1 + ((boostMaxMultiplier - 1) * boostT);
}

export function resolveKeyboardZoomStep({
    direction,
    currentDistance,
    deltaSeconds,
    zoomSpeed,
    minUnitsPerSecond,
    maxUnitsPerSecond,
    minDistance = 0,
    maxDistance = Infinity,
    boostStartDistanceRatio = KEYBOARD_ZOOM_BOOST_START_DISTANCE_RATIO,
    boostEndDistanceRatio = KEYBOARD_ZOOM_BOOST_END_DISTANCE_RATIO,
    boostMaxMultiplier = KEYBOARD_ZOOM_BOOST_MAX_MULTIPLIER
} = {}) {
    if (!(Number.isFinite(currentDistance) && currentDistance > 0)) return null;
    if (!(Number.isFinite(deltaSeconds) && deltaSeconds > 0)) return null;
    if (!(Number.isFinite(zoomSpeed) && zoomSpeed > 0)) return null;

    const zoomScale = Math.exp(-direction * zoomSpeed * deltaSeconds);
    const scaledDistance = currentDistance * zoomScale;
    const scaledStep = Math.abs(scaledDistance - currentDistance);
    const minimumStep = Math.max(0, (Number.isFinite(minUnitsPerSecond) ? minUnitsPerSecond : 0) * deltaSeconds);
    const baseMaximumStep = Math.max(
        minimumStep,
        Math.max(0, Number.isFinite(maxUnitsPerSecond) ? maxUnitsPerSecond : 0) * deltaSeconds
    );
    const boostMultiplier = resolveKeyboardZoomBoostMultiplier({
        direction,
        currentDistance,
        minDistance,
        maxDistance,
        boostStartDistanceRatio,
        boostEndDistanceRatio,
        boostMaxMultiplier
    });
    const maximumStep = Math.max(minimumStep, baseMaximumStep * boostMultiplier);
    return clamp(scaledStep, minimumStep, maximumStep);
}

export function resolveKeyboardZoomTargetDistance({
    direction,
    currentDistance,
    deltaSeconds,
    zoomSpeed,
    minUnitsPerSecond,
    maxUnitsPerSecond,
    minDistance = 0,
    maxDistance = Infinity,
    boostStartDistanceRatio = KEYBOARD_ZOOM_BOOST_START_DISTANCE_RATIO,
    boostEndDistanceRatio = KEYBOARD_ZOOM_BOOST_END_DISTANCE_RATIO,
    boostMaxMultiplier = KEYBOARD_ZOOM_BOOST_MAX_MULTIPLIER
} = {}) {
    const safeCurrentDistance = Number.isFinite(currentDistance) ? currentDistance : null;
    if (!(safeCurrentDistance > 0)) return safeCurrentDistance;

    const zoomStep = resolveKeyboardZoomStep({
        direction,
        currentDistance: safeCurrentDistance,
        deltaSeconds,
        zoomSpeed,
        minUnitsPerSecond,
        maxUnitsPerSecond,
        minDistance,
        maxDistance,
        boostStartDistanceRatio,
        boostEndDistanceRatio,
        boostMaxMultiplier
    });
    if (!(Number.isFinite(zoomStep) && zoomStep > 0)) return safeCurrentDistance;

    const desiredDistance = safeCurrentDistance + (direction > 0 ? -zoomStep : zoomStep);
    const safeMinDistance = Number.isFinite(minDistance) ? minDistance : 0;
    const safeMaxDistance = Number.isFinite(maxDistance) ? maxDistance : Infinity;
    return clamp(desiredDistance, safeMinDistance, safeMaxDistance);
}
