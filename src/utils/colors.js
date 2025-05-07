import * as THREE from 'three';

// Map a value (normalized, potentially outside -1 to 1) to a rainbow color (HSL)
export function mapValueToColor(value) {
    // Clamp or scale the normalized value to the -1 to 1 range for hue mapping
    // Simple clamping for now:
    const clampedValue = Math.max(-1, Math.min(1, value / 2)); // Divide by 2 assuming norm keeps most data in [-2, 2]

    const hue = (clampedValue + 1) / 2; // Full hue range based on clamped value
    const saturation = 1.0;
    // Increase lightness for brighter instances
    const lightness = 0.4;
    return new THREE.Color().setHSL(hue, saturation, lightness);
}
