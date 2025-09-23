/**
 * Lightweight easing helpers shared by multiple animation modules.
 * Implemented manually so we can keep bespoke motion curves without
 * relying on the global TWEEN easing catalogue everywhere.
 */

/**
 * Smoothly accelerates and decelerates using a sine wave.
 * @param {number} t - Normalised progress (0 → 1)
 * @returns {number}
 */
export function easeInOutSine(t) {
    const clamped = Math.min(Math.max(t, 0), 1);
    return -(Math.cos(Math.PI * clamped) - 1) / 2;
}

/**
 * Overshooting ease-out used to create playful anticipation / follow-through.
 * @param {number} t - Normalised progress (0 → 1)
 * @param {number} [overshoot=1.70158] - Strength of the overshoot.
 * @returns {number}
 */
export function easeOutBack(t, overshoot = 1.70158) {
    const clamped = Math.min(Math.max(t, 0), 1);
    const inv = clamped - 1;
    return 1 + inv * inv * ((overshoot + 1) * inv + overshoot);
}

/**
 * Fast-out curve that gently glides into the target.
 * @param {number} t - Normalised progress (0 → 1)
 * @returns {number}
 */
export function easeOutCubic(t) {
    const clamped = Math.min(Math.max(t, 0), 1);
    const inv = 1 - clamped;
    return 1 - inv * inv * inv;
}

/**
 * Convenience clamp used by several motion helpers.
 * @param {number} value
 * @param {number} min
 * @param {number} max
 * @returns {number}
 */
export function clamp(value, min, max) {
    return Math.min(Math.max(value, min), max);
}
