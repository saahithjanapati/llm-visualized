// ------------------------------------------------------------
// Global animation speed multiplier (1 = normal speed). Increase to speed up everything.
// Use setter to update at runtime; modules should always read the live binding.
// ------------------------------------------------------------
export let GLOBAL_ANIM_SPEED_MULT = 50;
export function setGlobalAnimSpeedMult(mult) {
    const m = Number(mult);
    if (!Number.isFinite(m) || m <= 0) return;
    GLOBAL_ANIM_SPEED_MULT = m;
}

// High-level presets for playback speed used by the settings UI.
// This adjusts the global speed and certain animation multipliers together.
export let PRISM_ADD_ANIM_SPEED_MULT = 4;
export let SELF_ATTENTION_TIME_MULT = 1;
export function setPlaybackSpeed(preset) {
    // Accept string keys or fallback to medium
    const cfg = {
        slow:   { mult: 25,  addMult: 2, selfAttenTimeMult: 2   },  // slowest
        medium: { mult: 50,  addMult: 4, selfAttenTimeMult: 1   },  // default
        fast:   { mult: 100, addMult: 8, selfAttenTimeMult: 0.35 }  // fastest
    };
    const p = typeof preset === 'string' ? preset.toLowerCase() : 'medium';
    const sel = cfg[p] || cfg.medium;
    setGlobalAnimSpeedMult(sel.mult);
    PRISM_ADD_ANIM_SPEED_MULT = sel.addMult;
    SELF_ATTENTION_TIME_MULT = sel.selfAttenTimeMult;
}

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

// ------------------------------------------------------------
// Prism visualisation parameters (grouped representation)
// ------------------------------------------------------------

/**
 * How many real GPT-2 dimensions are grouped into a single prism for the
 * visualisation.  The entire renderer now shows far fewer (but wider) prisms
 * while occupying the exact same overall length as before.
 */
export const PRISM_DIMENSIONS_PER_UNIT = 64;

/**
 * Number of physical prisms used to represent a full 768-dimension vector.
 * Using the grouping above this is simply 768 / 64 = 12.
 */
export const VECTOR_LENGTH_PRISM = Math.ceil(768 / PRISM_DIMENSIONS_PER_UNIT); // 12

// ------------------------------------------------------------------
// Geometry sizing – scale the base width so the final aggregated
// prisms still cover the same overall length as the previous 768-prism
// representation (≈115 world-units).
//
// Previous spacing between prism centres:  PRISM_BASE_WIDTH * 1.5 = 0.1 * 1.5 = 0.15
// Overall length  = (768 – 1) * 0.15 ≈ 115.05
// We now have 12 prisms, and the code that positions prisms uses (N * spacing)
// when measuring overall width.  Keeping the maths simple we scale the base
// width by exactly 64 so that:
//   newSpacing = 64 × 0.15 = 9.6
//   newTotal   = 12 × 9.6 = 115.2  (matches previous length).
// Hence with width-scale 1.5 we set PRISM_BASE_WIDTH = 9.6 / 1.5 = **6.4**.
//-------------------------------------------------------------------
export const PRISM_BASE_WIDTH = 6.4; // Width of a single (grouped) prism along the vector axis
export const PRISM_BASE_DEPTH = 7; // Depth of the prism
export const PRISM_MAX_HEIGHT = 7; // Max height for a prism when data is at its peak
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
export const VECTOR_DEPTH_SPACING = 200;

// ------------------------------------------------------------
// Lane / vector configuration – controls how many lanes/vectors
// are present in the scene and component depths that depend on it.
// ------------------------------------------------------------

export const NUM_VECTOR_LANES = 10; // Master switch: number of vector "lanes" active in the scene

// Depth large enough to fit all lanes plus one spacing margin at each end
export const LANE_DEPENDENT_DEPTH = (NUM_VECTOR_LANES + 1) * VECTOR_DEPTH_SPACING;

// ------------------------------------------------------------
// Animation scaling when vectors are grouped into fewer prisms
// ------------------------------------------------------------

/**
 * Ratio of original component count (768) to current prism count.  Used to
 * slow down per-prism stagger timings so the overall animation duration
 * remains visually similar to the original fine-grained version.
 */
export const GROUPED_PRISM_SLOWDOWN = PRISM_DIMENSIONS_PER_UNIT; // 64 when grouping 64 dims → 1 prism

// Constants for PrismAdditionAnimation
// Scale per-prism timings by the slowdown factor so total animation time is
// roughly preserved (768 * 15 ms  ≈  12 * 960 ms, etc.)
export const PRISM_ADD_ANIM_BASE_DURATION = 400 * Math.sqrt(GROUPED_PRISM_SLOWDOWN); // ms, for one prism to move
export const PRISM_ADD_ANIM_BASE_FLASH_DURATION = 80 * Math.sqrt(GROUPED_PRISM_SLOWDOWN);
export const PRISM_ADD_ANIM_BASE_DELAY_BETWEEN_PRISMS = 15 * GROUPED_PRISM_SLOWDOWN; // ms delay between prism starts
export const PRISM_ADD_ANIM_BASE_Y_OFFSET_FACTOR = 1.1; // How much higher source moves relative to target base
/** Multiplier for the prism addition animation speed.  
 * 1 = base speed; 2 = twice as fast (half the duration); 0.5 = half as fast (double duration).
 * Runtime adjustable via setPlaybackSpeed(). */
// NOTE: declaration moved earlier and converted to `let` so presets can update it

// ------------------------------------------------------------
// MHSA / Pass-through speed & timing (centralised)
// ------------------------------------------------------------
export const MHSA_DUPLICATE_VECTOR_RISE_SPEED = 6;            // Upward K copy rise speed (world units / sec)
export const MHSA_PASS_THROUGH_TOTAL_DURATION_MS = 90000;     // Total duration of pass-through tween (ms)
export const MHSA_PASS_THROUGH_BRIGHTEN_RATIO = 0.4;          // Fraction of total spent brightening
export const MHSA_PASS_THROUGH_DIM_RATIO = 0.4;               // Fraction of total spent dimming
export const MHSA_MATRIX_MAX_EMISSIVE_INTENSITY = 0.80;       // Max emissive intensity during brightening
export const MHSA_RESULT_RISE_OFFSET_Y = 60;                  // Rise offset after pass-through
export const MHSA_HEAD_VECTOR_STOP_BELOW = 70;                // Stop distance below matrix centre for head parking

// Result rise duration base (scaled in code by SPEED_MULT)
export const MHA_RESULT_RISE_DURATION_BASE_MS = 500;

// Decorative/merge phase timings
export const DECORATIVE_FADE_MS = 800;
export const DECORATIVE_FADE_DELAY_MS = 800;
export const MERGE_TO_ROW_DELAY_AFTER_FADE_MS = 900;          // Begin merge after decorative fade completes
export const HEAD_COLOR_TRANSITION_MS = 1000;                 // Duration for head colour transition
export const MERGE_POST_COLOR_TRANSITION_DELAY_MS = 1000;     // Delay before starting output projection after colours
export const MERGE_EXTRA_BUFFER_MS = 200;                     // Extra buffer after merge duration

// Output projection staged timings
export const OUTPUT_PROJ_STAGE1_MS = 1000; // to matrix bottom
export const OUTPUT_PROJ_STAGE2_MS = 1000; // through matrix
export const OUTPUT_PROJ_STAGE3_MS = 500;  // final rise after matrix

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
export const HEAD_SET_GAP_LAYER = 350;

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
    // Match slit count to current lane count so each lane has its own channel.
    numberOfSlits: NUM_VECTOR_LANES,
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


// Vector behaviour within MHSA heads ------------------------------------------------
/** Vertical speed for vectors rising into heads. */
export const ANIM_RISE_SPEED_HEAD = 0.44;
/** Distance below the centre of a head where vectors stop. */
export const HEAD_VECTOR_STOP_BELOW = 35;

/** Delay (ms) between centre copy arriving and side copies spawning. */
export const SIDE_COPY_DELAY_MS = 0;


/** Horizontal speed for decorative vectors merging into row output. */
export const ROW_MERGE_HORIZ_SPEED = 20;

/**
 * Centre-to-centre spacing between **grouped** 64-dimensional segments when
 * assembling the 768-dim row vector (e.g. when merging MHSA head outputs).
 * Each segment is one physical prism, so the spacing is simply the scaled
 * prism width (no additional ×64 multiplier).
 */
export const ROW_SEGMENT_SPACING = PRISM_BASE_WIDTH * 1.5; // matches InstancedPrism width scaling

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
    // Keep slit count in sync with lane count (same as MHA_MATRIX_PARAMS)
    numberOfSlits: NUM_VECTOR_LANES, // indicate 4× channels
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
    // back to original channels
    numberOfSlits: NUM_VECTOR_LANES,               // back to original channels
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

// ------------------------------------------------------------
// Self-Attention above-matrix timings (centralised)
// ------------------------------------------------------------
export const SA_RED_EXTRA_RISE = 75;                 // Additional V-vector rise (world units)
export const SA_V_RISE_DURATION_MS = 600;            // Duration for V extra rise
export const SA_K_ALIGN_DURATION_MS = 1000;          // Duration to align K to V
export const SA_BLUE_HORIZ_DURATION_MS = 400;        // Horizontal slide duration for Q
export const SA_BLUE_VERT_DURATION_MS = 400;         // Per-lane hop duration for Q/V traversal
export const SA_BLUE_PAUSE_MS = 100;                 // Pause at each lane during traversal
export const SA_BLUE_QUEUE_SHIFT_DURATION_MS = 400;  // Duration to shift remaining blue vectors

// Self-attention duplicate vector micro-timings
export const SA_DUPLICATE_POP_IN_MS = 120;
export const SA_DUPLICATE_TRAVEL_MERGE_MS = 400;
export const SA_DUPLICATE_POP_OUT_MS = 150;

// ------------------------------------------------------------
// Prism LayerNorm animation timings (centralised)
// ------------------------------------------------------------
export const PLN_UNIT_DELAY_MS = 3 * GROUPED_PRISM_SLOWDOWN;              // ms between activations
export const PLN_UNIT_CYCLE_DURATION_MS = 200 * Math.sqrt(GROUPED_PRISM_SLOWDOWN); // ms per unit cycle

// ------------------------------------------------------------
// MLP specific micro-timings
// ------------------------------------------------------------
export const MLP_EXPAND_RISE_MS = 500;            // rise before down-projection
export const MLP_SHRINK_INSIDE_DOWN_MS = 300;     // shrink duration inside down-projection
