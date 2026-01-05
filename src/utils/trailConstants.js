// Centralised visual parameters for StraightLineTrail lines
// Modify these values to tweak trail appearance globally

import { DEFAULT_NUM_VECTOR_LANES, NUM_VECTOR_LANES } from './constants.js';

export const TRAIL_COLOR = 0xffffff;          // Default hex colour (match other trails)
export const TRAIL_LINE_WIDTH = 1;            // Pixel width (hardware-dependent)
export const TRAIL_OPACITY = 0.14;             // 0 (fully transparent) → 1 (fully opaque)
export const TRAIL_MAX_SEGMENTS = 5000;       // Preallocated straight-line segments
// Subtle brightening as lane count grows to keep dense trails readable.
export const TRAIL_LANE_OPACITY_EXPONENT = 0.1;
export const TRAIL_LANE_OPACITY_MAX_SCALE = 1.1;
// Minimum distance (world units) between recorded trail points to reduce churn.
export const TRAIL_MIN_SEGMENT_DISTANCE = 0.4;

// Reserved for future extensions – THREE.LineBasicMaterial has no emissive term but
// we expose a placeholder in case the implementation switches materials later.
export const TRAIL_EMISSIVE_INTENSITY = 0.0;

// ---------------------------------------------------------------------------
// DPI-aware helpers to normalize trail appearance across displays
// ---------------------------------------------------------------------------

/**
 * Returns the effective device pixel ratio used for rendering, capped to 2 to
 * align with renderer pixel ratio capping elsewhere in the app.
 */
export function getEffectiveDevicePixelRatio() {
    if (typeof window === 'undefined' || typeof window.devicePixelRatio !== 'number') return 1;
    // Cap at 2 because some renderers clamp to 2 for performance/quality balance
    return Math.min(window.devicePixelRatio, 2);
}

/**
 * Scale a base trail opacity for the current display so perceived brightness of
 * thin 1px lines remains similar on Retina/high-DPI screens, plus a lane-count
 * brightness factor to keep dense trails readable.
 * Uses a gentle sqrt growth with DPR and a soft lane scaling curve.
 */
export function scaleOpacityForDisplay(baseOpacity) {
    const dpr = getEffectiveDevicePixelRatio();
    const laneScale = getLaneOpacityScale();
    // On standard-density panels (≈1 DPR) thin 1px lines lose brightness due to
    // limited pixel coverage and anti-alias falloff.  Give them a gentle boost
    // so they match the perceived brightness of the same trail on Retina/hi-DPI
    // displays where multiple physical pixels contribute to each fragment.
    const lowDprBoost = 1 + 0.6 * Math.max(0, (1 / Math.max(dpr, 1)) - 0.5);
    const scaled = baseOpacity * laneScale * Math.sqrt(dpr) * lowDprBoost;
    return Math.min(1, scaled);
}

/**
 * Scale a base trail linewidth for the current display. Many platforms ignore
 * LineBasicMaterial.linewidth, but on those that respect it this keeps the
 * perceived thickness similar across DPRs.
 */
export function scaleLineWidthForDisplay(baseWidth) {
    const dpr = getEffectiveDevicePixelRatio();
    const scaled = baseWidth * dpr;
    // Ensure at least 1 to avoid sub-pixel rounding artifacts
    return Math.max(1, scaled);
}

/** Brightens trails when there are more lanes than the baseline configuration. */
export function getLaneOpacityScale(laneCount = NUM_VECTOR_LANES) {
    const lanes = Math.max(1, Math.floor(laneCount || 1));
    if (lanes <= DEFAULT_NUM_VECTOR_LANES) return 1;
    const ratio = lanes / DEFAULT_NUM_VECTOR_LANES;
    const scaled = Math.pow(ratio, TRAIL_LANE_OPACITY_EXPONENT);
    return Math.max(1, Math.min(TRAIL_LANE_OPACITY_MAX_SCALE, scaled));
}
