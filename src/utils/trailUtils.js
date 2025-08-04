// Stubbed-out trail utilities – trail-line functionality has been fully
// removed from the project.  These helpers now return no-op placeholders so
// that the rest of the animation codebase can remain unchanged.

// Global toggle (kept for API compatibility)
export const TRAILS_ENABLED = false;

/**
 * Returns a dummy "trail" object so existing animation code that formerly
 * expected a geometry + material can still access nested properties without
 * throwing runtime errors.  No real Three.js objects are created.
 */
export function createTrailLine() {
  return {
    line: {
      material: {
        opacity: 0,
        needsUpdate: false,
      },
    },
    geometry: {
      // minimal stub attributes
      attributes: {
        position: {
          setXYZ: () => {},
          needsUpdate: false,
        },
      },
      setDrawRange: () => {},
      computeBoundingSphere: () => {},
    },
    positions: [],
    points: [],
  };
}

// No-op – trails are gone.
export function updateTrail() {}
