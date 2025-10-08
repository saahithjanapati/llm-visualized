// Centralised visual parameters for StraightLineTrail lines
// Modify these values to tweak trail appearance globally

import { getThemeTrailColor, onThemeChange } from '../state/themeState.js';

export let TRAIL_COLOR = getThemeTrailColor();          // Default hex colour (match other trails)
export const TRAIL_LINE_WIDTH = 1;            // Pixel width (hardware-dependent)
export const TRAIL_OPACITY = 0.13;             // 0 (fully transparent) → 1 (fully opaque)
export const TRAIL_MAX_SEGMENTS = 5000;       // Preallocated straight-line segments

// Reserved for future extensions – THREE.LineBasicMaterial has no emissive term but
// we expose a placeholder in case the implementation switches materials later.
export const TRAIL_EMISSIVE_INTENSITY = 0.0;

onThemeChange(() => {
    TRAIL_COLOR = getThemeTrailColor();
});

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
 * thin 1px lines remains similar on Retina/high-DPI screens.
 * Uses a gentle sqrt growth with DPR.
 */
export function scaleOpacityForDisplay(baseOpacity) {
    const dpr = getEffectiveDevicePixelRatio();
    const scaled = baseOpacity * Math.sqrt(dpr);
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
