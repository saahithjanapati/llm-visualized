// ------------------------------------------------------------
// Global animation speed multiplier (1 = normal speed). Increase to speed up everything.
// ------------------------------------------------------------
export const GLOBAL_ANIM_SPEED_MULT = 50;

// Always render in high-quality mode. Remove lower-quality mobile detection.
export const IS_MOBILE = false;
export const QUALITY_PRESET = 'high';

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
export const HIDE_INSTANCE_Y_OFFSET = -50000; // Used to "hide" instanced mesh instances by moving them far away (placed well below the scene)


// ------------------------------------------------------------
// Prism constants
// ------------------------------------------------------------

export const VECTOR_LENGTH_PRISM = 768;
export const PRISM_BASE_WIDTH = 0.1; // Width of the prism along the main vector axis
export const PRISM_BASE_DEPTH = 5; // Depth of the prism
export const PRISM_MAX_HEIGHT = 5; // Max height for a prism when data is at its peak
export const PRISM_HEIGHT_SCALE_FACTOR = 0.75; // Scales normalized data to prism height

// ─────────────────────────────────────────────────────────────
// Feature toggles
// ─────────────────────────────────────────────────────────────

/**
 * When true the app skips expensive CSG work and loads pre-baked
 * BufferGeometries from `precomputed_components.glb`.
 * You can disable it via a URL query string: `?fresh=1` or by setting
 * `window.__USE_PRECOMPUTED_GEOMETRIES = false` in DevTools before reload.
 */
export const USE_PRECOMPUTED_GEOMETRIES = true;

// ------------------------------------------------------------
// Common depth spacing (Z-axis) — edit this to change spacing for
// vectors/components across the entire scene.
// ------------------------------------------------------------

/** Distance (in world units) between adjacent vector lanes along Z. */
export const VECTOR_DEPTH_SPACING = 150;

// ------------------------------------------------------------
// Lane / vector configuration – controls how many lanes/vectors
// are present in the scene and component depths that depend on it.
// ------------------------------------------------------------

export const NUM_VECTOR_LANES = 5; // Master switch: number of vector "lanes" active in the scene

// Depth large enough to fit all lanes plus one spacing margin at each end
export const LANE_DEPENDENT_DEPTH = (NUM_VECTOR_LANES + 1) * VECTOR_DEPTH_SPACING;

// Constants for PrismAdditionAnimation
export const PRISM_ADD_ANIM_BASE_DURATION = 400 // ms, for one prism to move
export const PRISM_ADD_ANIM_BASE_FLASH_DURATION = 80; // ms, for the target flash (if re-enabled)
export const PRISM_ADD_ANIM_BASE_DELAY_BETWEEN_PRISMS = 15; // ms, delay between starting each prism's animation
export const PRISM_ADD_ANIM_BASE_Y_OFFSET_FACTOR = 1.1; // How much higher source moves relative to target base
/** Multiplier for the prism addition animation speed.  
 * 1 = base speed; 2 = twice as fast (half the duration); 0.5 = half as fast (double duration). */
export const PRISM_ADD_ANIM_SPEED_MULT = 4;

// ------------------------------------------------------------
// Constants from LayerAnimationConstants.js
// ------------------------------------------------------------

// General Layout Constants
/** Vertical gap between the top of one component and the bottom of the next. */
export const LN_TO_MHA_GAP = 150;
export const VERTICAL_GAP_COMPONENTS = LN_TO_MHA_GAP; // Backwards-compat alias

// Increased horizontal separation between the residual stream (x=0) and
// branched components (LayerNorms, MHSA, MLP).  A wider gap improves visual
// clarity when many layers are stacked, and reduces the chance of artefacts
// from overlapping transparent geometry.
export const BRANCH_X = 600;

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
    depth: LANE_DEPENDENT_DEPTH,
    wallThickness: 1.0,
    numberOfHoles: NUM_VECTOR_LANES,
    holeWidth: 10,
    holeWidthFactor: 20
};

// Multi-Head Self-Attention (MHSA) Parameters
/** Number of attention head sets (each set has Q, K, V matrices). */
export const NUM_HEAD_SETS_LAYER = 12;

/** Horizontal gap between adjacent attention head sets. */
export const HEAD_SET_GAP_LAYER = 240;

// Centre-to-centre spacing between the Q, K and V matrices within a single
// attention head.  Setting this equal to the matrix width means the matrices
// now sit flush against one another (touching edges) without any gap.
export const MHA_INTERNAL_MATRIX_SPACING = 116;

/**
 * Parameters defining the geometry of individual Q, K, V matrices in the MHSA block.
 * Properties are consistent with WeightMatrixVisualization constructor.
 */
export const MHA_MATRIX_PARAMS = {
    width: 120,
    height: 40,
    depth: LANE_DEPENDENT_DEPTH,
    topWidthFactor: 0.1,
    cornerRadius: 5,
    numberOfSlits: 5, // Visually, might want to link to VECTOR_LENGTH or a fraction
    slitWidth: 20, // significantly wider slits for clearer view
    slitDepthFactor: 1.0,
    slitBottomWidthFactor: 1,
    slitTopWidthFactor: 0.90
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
export const ANIM_RISE_SPEED_ORIGINAL = 1.5; // Slower rise speed for residual stream vectors

/**
 * Speed at which residual-stream vectors rise after branching at the first
 * LayerNorm (during the MHSA computation).
 */
export const ANIM_RISE_SPEED_POST_SPLIT_LN1 = 0.5;

/**
 * Speed at which residual-stream vectors rise after branching at the second
 * LayerNorm (during the MLP computation).  Increased to quicken the upward
 * motion during the MLP stage.
 */
export const ANIM_RISE_SPEED_POST_SPLIT_LN2 = 0.5;

/** Horizontal speed for vectors moving towards/away from branched components. */
export const ANIM_HORIZ_SPEED = 25;

/** Horizontal speed for side copies moving to Q/V matrices. */
export const SIDE_COPY_HORIZ_SPEED = 15;

/** Vertical speed for vectors moving upwards inside the LayerNorm block. */
export const ANIM_RISE_SPEED_INSIDE_LN = 6;

// Trail Line Constants
/** Maximum number of points to store for each trail line, affecting trail length.  Increase for longer-duration animations. */
export const MAX_TRAIL_POINTS = QUALITY_PRESET === 'low' ? 4000 : 100000;

// Vector behaviour within MHSA heads ------------------------------------------------
/** Vertical speed for vectors rising into heads. */
export const ANIM_RISE_SPEED_HEAD = 0.44;
/** Distance below the centre of a head where vectors stop. */
export const HEAD_VECTOR_STOP_BELOW = 35;

/** Delay (ms) between centre copy arriving and side copies spawning. */
export const SIDE_COPY_DELAY_MS = 0;


/** Horizontal speed for decorative vectors merging into row output. */
export const ROW_MERGE_HORIZ_SPEED = 20;

/** Centre-to-centre spacing between 64-dim segments when assembling the 768-dim row. */
export const ROW_SEGMENT_SPACING = PRISM_BASE_WIDTH * 1.5 * 64; // matches InstancedPrism width scaling

// -----------------------------------------------------------------------------
// Residual Stream Alignment Constants
// -----------------------------------------------------------------------------

/**
 * Vertical gap (in world units) between the *processed* vector (after it exits
 * the Output-Projection matrix) and the *original* residual-stream vector that
 * continues to flow underneath.  Increase this value to leave a larger space
 * or set it to a negative number to let the original vector overlap.
 */
export const ORIGINAL_TO_PROCESSED_GAP = 10;

// Vertical gap from LayerNorm2 top to first MLP matrix
export const LN2_TO_MLP_GAP = 150;

// Y-position for the centre of the second LayerNormalization block
// (approximate – may be tweaked from the animation file for best fit)
export const LAYER_NORM_2_Y_POS = 750; // Increased for clearer separation from MHSA block

// Gap between the two stacked MLP weight matrices
export const MLP_INTER_MATRIX_GAP = 140; // increased gap for clearer visual separation between the two MLP weight matrices

// -----------------------------------------------------------------------------
// MLP Up-/Down-Projection Weight Matrix Geometry Parameters
// -----------------------------------------------------------------------------
// Base width is aligned with d_model size (uses same width as MHSA matrices).
export const MLP_MATRIX_BASE_WIDTH = MHA_MATRIX_PARAMS.width; // Represents 768-dim bottom width

// Up-projection matrix: bottom width = d_model, top width = 4·d_model
export const MLP_MATRIX_PARAMS_UP = {
    width: MLP_MATRIX_BASE_WIDTH,                   // bottom (input: 768)
    height: 120,                                     // visually taller than MHSA matrices
    depth: LN_PARAMS.depth,                         // match component depth
    topWidthFactor: MLP_VECTOR_MULTIPLIER,          // widens to 4× at the top (3072)
    cornerRadius: 30.0,
    numberOfSlits: LN_PARAMS.numberOfHoles, // indicate 4× channels
    slitWidth: 10,
    slitDepthFactor: 1.0,
    slitBottomWidthFactor: 0.95,
    slitTopWidthFactor: 0.95
};

// Down-projection matrix: bottom width = 4·d_model, top width = d_model
export const MLP_MATRIX_PARAMS_DOWN = {
    width: MLP_MATRIX_BASE_WIDTH * MLP_VECTOR_MULTIPLIER, // bottom (input: 3072)
    height: 120,
    depth: LN_PARAMS.depth,
    topWidthFactor: 1 / MLP_VECTOR_MULTIPLIER,            // narrows back to d_model
    cornerRadius: 30.0,
    numberOfSlits: LN_PARAMS.numberOfHoles,               // back to original channels
    slitWidth: 10,
    slitDepthFactor: 1.0,
    slitBottomWidthFactor: 0.95,
    slitTopWidthFactor: 0.95
};

// -----------------------------------------------------------------------------
// Shared colours
// -----------------------------------------------------------------------------

/** Base colour for inactive components (visible but dark). */
export const INACTIVE_COMPONENT_COLOR = 0x202020;

// ------------------------------------------------------------
// UI / Caption Constants
// ------------------------------------------------------------

/** Vertical Y-coordinate for the "Can machines think?" caption that sits under
 *  the GPT-2 tower.  Negative values move the text downward (further away from
 *  the camera).  Adjust here to reposition the caption globally. */
export const CAPTION_TEXT_Y_POS = -2500;
