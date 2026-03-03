// Centralised visual parameters for StraightLineTrail lines
// Modify these values to tweak trail appearance globally

import { DEFAULT_NUM_VECTOR_LANES, NUM_VECTOR_LANES, resolveRenderPixelRatio } from './constants.js';

export const TRAIL_COLOR = 0xffffff;          // Default hex colour (match other trails)
export const TRAIL_LINE_WIDTH = 1;            // Pixel width (hardware-dependent)
export const TRAIL_OPACITY = 0.14;             // 0 (fully transparent) → 1 (fully opaque)
export const TRAIL_MAX_SEGMENTS = 5000;       // Preallocated straight-line segments
// Dim trails as lane count grows to reduce clutter.
export const TRAIL_LANE_OPACITY_EXPONENT = 0.25;
export const TRAIL_LANE_OPACITY_MIN_SCALE = 0.6;
// Minimum distance (world units) between recorded trail points to reduce churn.
export const TRAIL_MIN_SEGMENT_DISTANCE = 0.4;
let TRAIL_OPACITY_RUNTIME_MULTIPLIER = 1.0;
let TRAIL_LINE_WIDTH_RUNTIME_MULTIPLIER = 1.0;
const TRAIL_LINE_WIDTH_DPR_EXPONENT = 0.5;
const TRAIL_MIN_SCREEN_WIDTH_PX = 1.25;
const TRAIL_OPACITY_DPR_DARKEN_START = 1.4;
const TRAIL_OPACITY_DPR_DARKEN_EXPONENT = 0.6;
const TRAIL_OPACITY_DPR_DARKEN_MIN = 0.68;
let TRAIL_PIXEL_RATIO_CACHE = {
    width: -1,
    height: -1,
    dpr: -1,
    ratio: 1
};

// Reserved for future extensions – THREE.LineBasicMaterial has no emissive term but
// we expose a placeholder in case the implementation switches materials later.
export const TRAIL_EMISSIVE_INTENSITY = 0.0;

// ---------------------------------------------------------------------------
// DPI-aware helpers to normalize trail appearance across displays
// ---------------------------------------------------------------------------

/**
 * Returns the effective renderer pixel ratio used for rendering trails.
 * This tracks the same supersampling logic as CoreEngine so trail widths stay
 * consistent across built-in and external displays.
 */
export function getEffectiveDevicePixelRatio() {
    if (typeof window === 'undefined') return 1;

    const width = Number.isFinite(window.innerWidth) ? Math.round(window.innerWidth) : 0;
    const height = Number.isFinite(window.innerHeight) ? Math.round(window.innerHeight) : 0;
    const dpr = (typeof window.devicePixelRatio === 'number' && window.devicePixelRatio > 0)
        ? window.devicePixelRatio
        : 1;

    const cacheValid = TRAIL_PIXEL_RATIO_CACHE
        && TRAIL_PIXEL_RATIO_CACHE.width === width
        && TRAIL_PIXEL_RATIO_CACHE.height === height
        && Math.abs(TRAIL_PIXEL_RATIO_CACHE.dpr - dpr) < 0.001
        && Number.isFinite(TRAIL_PIXEL_RATIO_CACHE.ratio)
        && TRAIL_PIXEL_RATIO_CACHE.ratio > 0;
    if (cacheValid) {
        return TRAIL_PIXEL_RATIO_CACHE.ratio;
    }

    const resolved = resolveRenderPixelRatio({ viewportWidth: width, viewportHeight: height });
    const ratio = (Number.isFinite(resolved) && resolved > 0) ? resolved : dpr;
    TRAIL_PIXEL_RATIO_CACHE = { width, height, dpr, ratio };
    return ratio;
}

/**
 * Scale a base trail opacity for the current display. DPR scaling is intentionally
 * disabled so trails keep a consistent brightness across devices; only the
 * lane-count boost remains.
 */
export function scaleOpacityForDisplay(baseOpacity) {
    const laneScale = getLaneOpacityScale();
    const dpr = getEffectiveDevicePixelRatio();
    // Retina/HiDPI displays can make thin translucent trails read brighter;
    // apply a gentle high-DPR darkening so brightness better matches large
    // low/medium-DPR external monitors.
    const dprOpacityCompensation = (dpr > TRAIL_OPACITY_DPR_DARKEN_START)
        ? Math.max(
            TRAIL_OPACITY_DPR_DARKEN_MIN,
            Math.pow(TRAIL_OPACITY_DPR_DARKEN_START / dpr, TRAIL_OPACITY_DPR_DARKEN_EXPONENT)
        )
        : 1;
    // Slightly boost trail readability on low-DPR displays (external monitors),
    // while leaving Retina/high-DPR displays unchanged.
    const scaled = baseOpacity * laneScale * TRAIL_OPACITY_RUNTIME_MULTIPLIER * dprOpacityCompensation;
    return Math.min(1, Math.max(0, scaled));
}

/**
 * Runtime multiplier for trail opacity. Useful for mode-specific visual tuning
 * without changing base constants.
 */
export function setTrailOpacityRuntimeMultiplier(multiplier = 1) {
    const next = Number(multiplier);
    if (!Number.isFinite(next) || next <= 0) {
        TRAIL_OPACITY_RUNTIME_MULTIPLIER = 1.0;
        return TRAIL_OPACITY_RUNTIME_MULTIPLIER;
    }
    TRAIL_OPACITY_RUNTIME_MULTIPLIER = Math.min(2.5, Math.max(0.25, next));
    return TRAIL_OPACITY_RUNTIME_MULTIPLIER;
}

/**
 * Runtime multiplier for trail line width. This mainly helps on platforms
 * where LineBasicMaterial.linewidth is honored.
 */
export function setTrailLineWidthRuntimeMultiplier(multiplier = 1) {
    const next = Number(multiplier);
    if (!Number.isFinite(next) || next <= 0) {
        TRAIL_LINE_WIDTH_RUNTIME_MULTIPLIER = 1.0;
        return TRAIL_LINE_WIDTH_RUNTIME_MULTIPLIER;
    }
    TRAIL_LINE_WIDTH_RUNTIME_MULTIPLIER = Math.min(6.0, Math.max(0.5, next));
    return TRAIL_LINE_WIDTH_RUNTIME_MULTIPLIER;
}

/**
 * Scale a base trail linewidth for the current display. Many platforms ignore
 * LineBasicMaterial.linewidth, but on those that respect it this keeps the
 * perceived thickness similar across DPRs.
 */
export function scaleLineWidthForDisplay(baseWidth) {
    const dpr = getEffectiveDevicePixelRatio();
    // Sublinear scaling keeps high-DPR displays from looking excessively thick/
    // bright while still widening enough on lower-DPR displays to reduce shimmer.
    const dprWidthFactor = Math.pow(dpr, TRAIL_LINE_WIDTH_DPR_EXPONENT);
    const scaled = baseWidth * dprWidthFactor * TRAIL_LINE_WIDTH_RUNTIME_MULTIPLIER;
    // Keep trails above 1px for more stable line rasterization while orbiting.
    return Math.max(TRAIL_MIN_SCREEN_WIDTH_PX, scaled);
}

/** Dims trails when there are more lanes than the baseline configuration. */
export function getLaneOpacityScale(laneCount = NUM_VECTOR_LANES) {
    const lanes = Math.max(1, Math.floor(laneCount || 1));
    if (lanes <= DEFAULT_NUM_VECTOR_LANES) return 1;
    const ratio = lanes / DEFAULT_NUM_VECTOR_LANES;
    const scaled = Math.pow(ratio, -TRAIL_LANE_OPACITY_EXPONENT);
    return Math.min(1, Math.max(TRAIL_LANE_OPACITY_MIN_SCALE, scaled));
}
