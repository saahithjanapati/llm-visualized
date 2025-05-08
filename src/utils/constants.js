export const VECTOR_LENGTH = 100;
export const SPHERE_RADIUS = 0.15;
export const SPHERE_DIAMETER = SPHERE_RADIUS * 2;
export const VECTOR_VISUAL_WIDTH = VECTOR_LENGTH * SPHERE_DIAMETER;
export const SPAWN_Y = -50;
export const DESPAWN_Y = 90;
export const SPAWN_X_RANGE = 70;
export const SPAWN_Z_RANGE = 10;
export const VECTOR_SPEED = 0.1;
export const EPSILON = 1e-5; // Small value for Layer Norm stability
export const SPAWN_INTERVAL = 120; // Faster spawning
export const HIDE_INSTANCE_Y_OFFSET = 10000; // Used to "hide" instanced mesh instances by moving them far away


// ------------------------------------------------------------
// Prism constants
// ------------------------------------------------------------

export const VECTOR_LENGTH_PRISM = 768;
export const PRISM_BASE_WIDTH = 0.1; // Width of the prism along the main vector axis
export const PRISM_BASE_DEPTH = 5; // Depth of the prism
export const PRISM_MAX_HEIGHT = 1.5; // Max height for a prism when data is at its peak
export const PRISM_HEIGHT_SCALE_FACTOR = 0.75; // Scales normalized data to prism height


// Constants for PrismAdditionAnimation
export const PRISM_ADD_ANIM_BASE_DURATION = 400 // ms, for one prism to move
export const PRISM_ADD_ANIM_BASE_FLASH_DURATION = 80; // ms, for the target flash (if re-enabled)
export const PRISM_ADD_ANIM_BASE_DELAY_BETWEEN_PRISMS = 15; // ms, delay between starting each prism's animation
export const PRISM_ADD_ANIM_BASE_Y_OFFSET_FACTOR = 1.1; // How much higher source moves relative to target base
