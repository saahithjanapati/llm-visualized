// Centralised visual parameters for StraightLineTrail lines
// Modify these values to tweak trail appearance globally

export const TRAIL_COLOR = 0x444544;          // Default hex colour
export const TRAIL_LINE_WIDTH = 1;            // Pixel width (hardware-dependent)
export const TRAIL_OPACITY = 0.1;             // 0 (fully transparent) → 1 (fully opaque)
export const TRAIL_MAX_SEGMENTS = 5000;       // Preallocated straight-line segments

// Reserved for future extensions – THREE.LineBasicMaterial has no emissive term but
// we expose a placeholder in case the implementation switches materials later.
export const TRAIL_EMISSIVE_INTENSITY = 0.0;
