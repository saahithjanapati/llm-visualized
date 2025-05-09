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
export const PRISM_BASE_DEPTH = 3; // Depth of the prism
export const PRISM_MAX_HEIGHT = 3; // Max height for a prism when data is at its peak
export const PRISM_HEIGHT_SCALE_FACTOR = 0.75; // Scales normalized data to prism height

// ------------------------------------------------------------
// Common depth spacing (Z-axis) — edit this to change spacing for
// vectors/components across the entire scene.
// ------------------------------------------------------------

/** Distance (in world units) between adjacent vector lanes along Z. */
export const VECTOR_DEPTH_SPACING = 100;

// Constants for PrismAdditionAnimation
export const PRISM_ADD_ANIM_BASE_DURATION = 400 // ms, for one prism to move
export const PRISM_ADD_ANIM_BASE_FLASH_DURATION = 80; // ms, for the target flash (if re-enabled)
export const PRISM_ADD_ANIM_BASE_DELAY_BETWEEN_PRISMS = 15; // ms, delay between starting each prism's animation
export const PRISM_ADD_ANIM_BASE_Y_OFFSET_FACTOR = 1.1; // How much higher source moves relative to target base

// ------------------------------------------------------------
// Constants from LayerAnimationConstants.js
// ------------------------------------------------------------

// General Layout Constants
/** Vertical gap between the top of one component and the bottom of the next. */
export const LN_TO_MHA_GAP = 150;
export const VERTICAL_GAP_COMPONENTS = LN_TO_MHA_GAP; // Backwards-compat alias

/** Horizontal X-offset from the main residual stream (x=0) for branched components like LayerNorms, MHSA, MLP. */
export const BRANCH_X = 350;

// LayerNorm1 (LN1) Parameters
/** Y-position for the center of the first LayerNormalization block. */
export const LAYER_NORM_1_Y_POS = -10;

/**
 * Parameters defining the geometry and appearance of the first LayerNormalization block.
 * @property {number} width - Width of the LayerNorm block.
 * @property {number} height - Height of the LayerNorm block.
 * @property {number} depth - Depth of the LayerNorm block.
 * @property {number} wallThickness - Thickness of the walls of the LayerNorm block.
 * @property {number} numberOfHoles - Number of holes/slits along the depth of the block.
 * @property {number} holeWidth - Width of each hole/slit.
 * @property {number} holeWidthFactor - Factor to adjust the visual width of the holes.
 */
export const LN_PARAMS = {
    width: 150,
    height: 100,
    depth: 6 * VECTOR_DEPTH_SPACING,
    wallThickness: 1.0,
    numberOfHoles: 5, // This is also used as numVectors for lanes
    holeWidth: 5,
    holeWidthFactor: 20
};

// Multi-Head Self-Attention (MHSA) Parameters
/** Number of attention head sets (each set has Q, K, V matrices). */
export const NUM_HEAD_SETS_LAYER = 12;

/** Horizontal gap between adjacent attention head sets. */
export const HEAD_SET_GAP_LAYER = 40;

/** Horizontal center-to-center spacing for Q, K, V matrices within a single attention head. */
export const MHA_INTERNAL_MATRIX_SPACING = 130;

/**
 * Parameters defining the geometry of individual Q, K, V matrices in the MHSA block.
 * Properties are consistent with WeightMatrixVisualization constructor.
 */
export const MHA_MATRIX_PARAMS = {
    width: 120,
    height: 40,
    depth: 6 * VECTOR_DEPTH_SPACING,
    topWidthFactor: 0.1,
    cornerRadius: 5,
    numberOfSlits: 5, // Visually, might want to link to VECTOR_LENGTH or a fraction
    slitWidth: 5,
    slitDepthFactor: 1.0,
    slitBottomWidthFactor: 1,
    slitTopWidthFactor: 0.08
};

// MLP (Multi-Layer Perceptron) Layer Parameters
/** Factor by which the vector dimension is multiplied in the MLP's up-projection. */
export const MLP_VECTOR_MULTIPLIER = 4;

/**
 * Base styling parameters for the WeightMatrixVisualizations in the MLP layer.
 * Width and depth are scaled based on MLP_VECTOR_MULTIPLIER.
 * numberOfSlits is also scaled.
 */
export const MLP_MATRIX_STYLE_PARAMS = {
    height: 15,
    topWidthFactor: 0.6,
    cornerRadius: 1.0,
    // numberOfSlits will be VECTOR_LENGTH for the d_model side, and VECTOR_LENGTH * MLP_VECTOR_MULTIPLIER for the 4*d_model side
    slitWidth: 5,
    slitDepthFactor: 1.0,
    slitBottomWidthFactor: 0.9,
    slitTopWidthFactor: 0.5
};




/** Visual representation of depth for d_model in MLP matrices. Used for scaling. */
export const MLP_D_MODEL_VISUAL_DEPTH = 20;



// Animation Path & Behavior Constants
/** Vertical offset below LayerNorm1 for spawning original vectors. */
export const ANIM_OFFSET_Y_ORIGINAL_SPAWN = 10;

/** Vertical offset above LayerNorm1 top where original and processed vectors meet (or would have met). */
export const ANIM_MEET_Y_OFFSET_ABOVE_LN1 = 5;

/** Speed at which original vectors rise along the main path (x=0). */
export const ANIM_RISE_SPEED_ORIGINAL = 3;

/** Horizontal speed for vectors moving towards/away from branched components. */
export const ANIM_HORIZ_SPEED = 15;

/** Horizontal speed for side copies moving to Q/V matrices. */
export const SIDE_COPY_HORIZ_SPEED = 5;

/** Vertical speed for vectors moving upwards inside the LayerNorm block. */
export const ANIM_RISE_SPEED_INSIDE_LN = 6;

// Trail Line Constants
/** Maximum number of points to store for each trail line, affecting trail length. */
export const MAX_TRAIL_POINTS = 100000;

// Vector behaviour within MHSA heads ------------------------------------------------
/** Vertical speed for vectors rising into heads. */
export const ANIM_RISE_SPEED_HEAD = 0.5;
/** Distance below the centre of a head where vectors stop. */
export const HEAD_VECTOR_STOP_BELOW = 35;

/** Delay (ms) between centre copy arriving and side copies spawning. */
export const SIDE_COPY_DELAY_MS = 500;

// ------------------------------------------------------------
// Global animation speed multiplier (1 = normal speed). Increase to speed up everything.
// ------------------------------------------------------------
export const GLOBAL_ANIM_SPEED_MULT = 1000;
