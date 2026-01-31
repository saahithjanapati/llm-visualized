import { NUM_VECTOR_LANES } from '../utils/constants.js';
// export const VECTOR_LENGTH = 5; // REMOVED: This was causing the length mismatch.
// The main VECTOR_LENGTH (100) should be imported from utils/constants.js directly where needed,
// or LayerAnimationConstants.js should correctly re-export it if preferred as a central point for this animation.
// For now, removing it here means LayerAnimation.js will need to adjust its import.

// -----------------------------------------------------------------------------
// General Layout Constants
// -----------------------------------------------------------------------------

/** Vertical gap between the top of one component and the bottom of the next. */
export const VERTICAL_GAP_COMPONENTS = 20;

/** Horizontal X-offset from the main residual stream (x=0) for branched components like LayerNorms, MHSA, MLP. */
export const BRANCH_X = 250; // widened to maintain proportion with revised utils constant

// -----------------------------------------------------------------------------
// LayerNorm1 (LN1) Parameters
// -----------------------------------------------------------------------------

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
    width: 70,
    height: 35,
    depth: 120,
    wallThickness: 1.0,
    numberOfHoles: NUM_VECTOR_LANES, // This is also used as numVectors for lanes
    holeWidth: 5,
    holeWidthFactor: 3.75
};

// -----------------------------------------------------------------------------
// Multi-Head Self-Attention (MHSA) Parameters
// -----------------------------------------------------------------------------

/** Number of attention head sets (each set has Q, K, V matrices). */
export const NUM_HEAD_SETS_LAYER = 12;

/** Horizontal gap between adjacent attention head sets. */
export const HEAD_SET_GAP_LAYER = 10;

/** Horizontal center-to-center spacing for Q, K, V matrices within a single attention head. */
export const MHA_INTERNAL_MATRIX_SPACING = 37.5;

/**
 * Parameters defining the geometry of individual Q, K, V matrices in the MHSA block.
 * Properties are consistent with WeightMatrixVisualization constructor.
 */
export const MHA_MATRIX_PARAMS = {
    width: 37.5,
    height: 12,
    depth: 100,
    topWidthFactor: 0.47,
    cornerRadius: 1.2,
    // Dynamically match the current lane count
    numberOfSlits: NUM_VECTOR_LANES, // matches vector lanes
    slitWidth: 8,
    // Keep slits shallow to avoid visible side dashes around the openings.
    slitDepthFactor: 0.3,
    slitBottomWidthFactor: 0.95,
    slitTopWidthFactor: 0.90
};

// -----------------------------------------------------------------------------
// LayerNorm2 (LN2) Parameters
// -----------------------------------------------------------------------------
// LN2 uses LN_PARAMS for its geometry by default in LayerAnimation.js.
// If specific params are needed for LN2, they can be defined here.
// e.g., export const LN2_PARAMS = { ... };

// -----------------------------------------------------------------------------
// MLP (Multi-Layer Perceptron) Layer Parameters
// -----------------------------------------------------------------------------

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
    slitWidth: 3,
    // Match the shallow slit cut used in the main constants.
    slitDepthFactor: 0.3,
    slitBottomWidthFactor: 0.9,
    slitTopWidthFactor: 0.5
};

/** Visual representation of depth for d_model in MLP matrices. Used for scaling. */
export const MLP_D_MODEL_VISUAL_DEPTH = 20;

// -----------------------------------------------------------------------------
// Color Constants – keep **all** colour values in this one place
// -----------------------------------------------------------------------------

/** Base (resting) colour for MHSA matrices. */
export const MHSA_MATRIX_INITIAL_RESTING_COLOR = 0x404040;

/** Bright / activated colours used inside the MHSA animation. */
export const MHSA_BRIGHT_GREEN        = 0x33FF33;
export const MHSA_DARK_TINTED_GREEN   = 0x002200;
export const MHSA_BRIGHT_BLUE         = 0x6666FF;
export const MHSA_DARK_TINTED_BLUE    = 0x000022;
export const MHSA_BRIGHT_RED          = 0xFF7A33;
export const MHSA_DARK_TINTED_RED     = 0x2B1100;

/** Colour for trail lines behind moving prisms. */


/** Final Q, K, V head colours used once attention heads have merged. */
export const MHA_FINAL_Q_COLOR = 0x276ebb;
export const MHA_FINAL_K_COLOR = 0x1e9f57;
// Base tint for V vectors (head visuals). Value-spectrum tint for the lightweight V outputs is separate.
export const MHA_FINAL_V_COLOR = 0xaa3420;
// Base tint for the value-spectrum applied to lightweight V vectors after head projection.
export const MHA_VALUE_SPECTRUM_COLOR = 0xf28b30;

/** Active colour of the output-projection matrix that follows MHSA. */
export const MHA_OUTPUT_PROJECTION_MATRIX_COLOR = 0x9C27B0;

/** Active colours for the MLP up-/down-projection matrices. */
export const MLP_UP_MATRIX_COLOR = 0xe35400;
export const MLP_DOWN_MATRIX_COLOR = 0xff6a00;

// -----------------------------------------------------------------------------
// MHSA (Multi-Head Self-Attention) Specific Animation Constants
// -----------------------------------------------------------------------------

/** Speed at which duplicated vectors rise towards their attention head. */
export const MHSA_DUPLICATE_VECTOR_RISE_SPEED = 6;

// Moved timing constants to utils/constants.js (kept colours and geometry here)

// Vertical offset applied to vectors after pass-through through matrices
export const MHSA_RESULT_RISE_OFFSET_Y = 60;

// Distance below the matrix center where vectors should stop before entering
export const MHSA_HEAD_VECTOR_STOP_BELOW = 70;

// Trail line visual properties


// -----------------------------------------------------------------------------
// Animation Path & Behavior Constants
// -----------------------------------------------------------------------------

/** Vertical offset below LayerNorm1 for spawning original vectors. */
export const ANIM_OFFSET_Y_ORIGINAL_SPAWN = 10;

/** Vertical offset above LayerNorm1 top where original and processed vectors meet (or would have met). */
export const ANIM_MEET_Y_OFFSET_ABOVE_LN1 = 5;

/** Speed at which original vectors rise along the main path (x=0). */
export const ANIM_RISE_SPEED_ORIGINAL = 3;

/** Horizontal speed for vectors moving towards/away from branched components. */
export const ANIM_HORIZ_SPEED = 15;

/** Vertical speed for vectors moving upwards inside the LayerNorm block. */
export const ANIM_RISE_SPEED_INSIDE_LN = 6;

// mergeGap was used for the original merge logic, may not be needed if merge is redesigned
// export const ANIM_MERGE_GAP = 7;

// -----------------------------------------------------------------------------
// Trail Line Constants
// -----------------------------------------------------------------------------

/** Maximum number of points to store for each trail line, affecting trail length. */


// -----------------------------------------------------------------------------
// MHA Output Projection Matrix (Post-Concatenation)
// -----------------------------------------------------------------------------

/** Vertical offset for the output projection matrix above the merged row of vectors. */
export const MHA_OUTPUT_PROJECTION_MATRIX_Y_OFFSET_ABOVE_ROW = 20;

/**
 * Parameters defining the geometry and appearance of the MHA output projection matrix.
 * This matrix has a rectangular shape (topWidthFactor=1.0) to indicate it preserves 
 * the full dimensionality on both input and output sides.
 */
export const MHA_OUTPUT_PROJECTION_MATRIX_PARAMS = {
    // Size parameters
    width: 150,                   // Same as MHA_MATRIX_PARAMS.width
    heightFactor: 2.5,             // Height relative to standard MHA matrices
    
    // Shape parameters
    topWidthFactor: 1.0,           // 1.0 = rectangular (no tapering)
    cornerRadius: 20,             // Same as MHA_MATRIX_PARAMS.cornerRadius
    
    // Slit parameters
    // Keep in sync with lane count
    numberOfSlits: NUM_VECTOR_LANES,              // Same as MHA_MATRIX_PARAMS.numberOfSlits
    slitWidth: 20,
    slitDepthFactor: 1.0,          // Same as MHA_MATRIX_PARAMS.slitDepthFactor
    slitBottomWidthFactor: 0.92,   // Same as MHA_MATRIX_PARAMS.slitBottomWidthFactor
    slitTopWidthFactor: 0.92      // Same as MHA_MATRIX_PARAMS.slitTopWidthFactor
}; 

export function setAnimationLaneCount(laneCount = NUM_VECTOR_LANES) {
    const clamped = Math.max(1, Math.floor(laneCount || 1));
    LN_PARAMS.numberOfHoles = clamped;
    MHA_MATRIX_PARAMS.numberOfSlits = clamped;
    MHA_OUTPUT_PROJECTION_MATRIX_PARAMS.numberOfSlits = clamped;
    return clamped;
}
