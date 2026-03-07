import { VECTOR_LENGTH_PRISM, NUM_HEAD_SETS_LAYER } from '../utils/constants.js';
import { MHA_FINAL_Q_COLOR } from '../animations/LayerAnimationConstants.js';

export const PREVIEW_LANES = 3;
export const PREVIEW_SOLID_LANES = 5;
export const PREVIEW_TOKEN_LANES = 1;
export const PREVIEW_MATRIX_DEPTH = 320;
export const PREVIEW_LANE_SPACING = 80;
export const PREVIEW_TARGET_SIZE = 140;
export const PREVIEW_FRAME_PADDING = 1.25;
export const PREVIEW_BASE_DISTANCE_MULT = 1.15;
export const PREVIEW_ROTATION_ENVELOPE_MARGIN = 1.06;
export const PREVIEW_VECTOR_PADDING_MULT = 2.4;
export const PREVIEW_VECTOR_DISTANCE_MULT = 1.9;
export const PREVIEW_MOBILE_MATRIX_PADDING_MULT = 1.2;
export const PREVIEW_MOBILE_MATRIX_DISTANCE_MULT = 1.22;
export const PREVIEW_ROTATION_SPEED = 0.0035;
export const PREVIEW_BASE_TILT_X = -0.12;
export const PREVIEW_BASE_ROTATION_Y = 0.38;
export const PREVIEW_TILT_AMPLITUDE = 0.02;
export const PREVIEW_TILT_OSC_SPEED = 0.32;
export const PREVIEW_FIT_LOCK_MS = 500;
export const PREVIEW_FIT_LOCK_PX = 120;
export const PREVIEW_FIT_LOCK_RATIO = 0.18;
export const FINAL_MLP_COLOR = 0xc07a12;
export const FINAL_VOCAB_TOP_COLOR = MHA_FINAL_Q_COLOR;
export const PREVIEW_QKV_LANES = 3;
export const PREVIEW_QKV_LANE_SPACING = 360;
export const PREVIEW_VECTOR_LARGE_SCALE = 1.0;
export const PREVIEW_VECTOR_SMALL_SCALE = 0.38;
export const PREVIEW_TRAIL_COLOR = 0x6ea0ff;
export const PREVIEW_QKV_X_SPREAD = 72;
export const PREVIEW_QKV_START_Y = -150;
export const PREVIEW_QKV_MATRIX_Y = -15;
export const PREVIEW_QKV_OUTPUT_Y = 95;
export const PREVIEW_QKV_EXIT_Y = PREVIEW_QKV_OUTPUT_Y + 60;
export const PREVIEW_QKV_RISE_DURATION = 900;
export const PREVIEW_QKV_CONVERT_DURATION = 420;
export const PREVIEW_QKV_HOLD_DURATION = 520;
export const PREVIEW_QKV_EXIT_DURATION = 420;
export const PREVIEW_QKV_IDLE_DURATION = 260;
export const PREVIEW_QKV_LANE_STAGGER = 0;
export const PREVIEW_VECTOR_HEAD_INSTANCES = 1;
export const PREVIEW_VECTOR_BODY_INSTANCES = VECTOR_LENGTH_PRISM;
export const ATTENTION_PREVIEW_MAX_TOKENS = 16;
export const ATTENTION_PREVIEW_TARGET_PX = 320;
export const ATTENTION_PREVIEW_MIN_CELL = 4;
export const ATTENTION_PREVIEW_MAX_CELL = 24;
export const ATTENTION_PREVIEW_GAP = 4;
export const ATTENTION_PREVIEW_TRIANGLE = 'lower';
// Matches `.detail-attention-grid` column gap in CSS.
export const ATTENTION_PREVIEW_GRID_GAP = 8;
export const ATTENTION_PRE_COLOR_CLAMP = 5;
export const ATTENTION_PREVIEW_COLOR_DARKEN_FACTOR = 0.84;
export const ATTENTION_POP_OUT_MS = 120;
export const ATTENTION_POST_REVEAL_DURATION_MS = 260;
export const ATTENTION_POST_REVEAL_SWEEP_MS = 380;
export const ATTENTION_POST_REVEAL_STAGGER_MIN_MS = 18;
export const ATTENTION_POST_REVEAL_STAGGER_MAX_MS = 52;
export const ATTENTION_PRE_REVEAL_DURATION_MS = 210;
export const ATTENTION_PRE_REVEAL_SWEEP_MS = 120;
export const ATTENTION_PRE_REVEAL_STAGGER_MIN_MS = 4;
export const ATTENTION_PRE_REVEAL_STAGGER_MAX_MS = 26;
export const ATTENTION_SCORE_DECIMALS = 4;
export const ATTENTION_VALUE_PLACEHOLDER = '--';
export const ATTENTION_DECODE_ROW_OFFSET_MULT = 0.68;
export const ATTENTION_DECODE_ROW_OFFSET_MIN_PX = 8;
export const ATTENTION_DECODE_ROW_OFFSET_MAX_PX = 20;
export const RESIDUAL_COLOR_CLAMP = 2;
export const SPACE_TOKEN_DISPLAY = '" "';
export const PANEL_SHIFT_DURATION_MS = 520;
export const DETAIL_EQUATION_FONT_MIN_PX = 9;
export const DETAIL_EQUATION_FONT_MAX_PX = 20;
export const DETAIL_EQUATION_FONT_MAX_SCALE = 1.5;
export const DETAIL_EQUATION_FIT_BUFFER_PX = 1.25;
export const COPY_CONTEXT_BUTTON_DEFAULT_LABEL = 'Have a question? Copy context to ask 🤖';
export const COPY_CONTEXT_SUCCESS_LABEL = 'Context copied to clipboard.';
export const COPY_CONTEXT_ERROR_LABEL = 'Unable to copy context.';
export const COPY_CONTEXT_EMPTY_LABEL = 'Nothing visible to copy yet.';
export const COPY_CONTEXT_FEEDBACK_MS = 1000;
export const COPY_CONTEXT_FADE_MS = 220;
export const SELECTION_PANEL_TOKEN_HOVER_SOURCE = 'selection-panel';
export const PANEL_ACTION_HISTORY_BACK = 'history-back';
export const PANEL_ACTION_HISTORY_FORWARD = 'history-forward';
export const ATTENTION_PREVIEW_SIZE_OPTIONS = Object.freeze({
    targetPx: ATTENTION_PREVIEW_TARGET_PX,
    minCell: ATTENTION_PREVIEW_MIN_CELL,
    maxCell: ATTENTION_PREVIEW_MAX_CELL
});
export const TOKEN_CHIP_STYLE = Object.freeze({
    padding: 80,
    minWidth: 220,
    minHeight: 100,
    height: 120,
    cornerRadius: 18,
    depth: 12,
    textDepth: 16,
    textSize: 52,
    textOffset: 1.2
});
export const LOGIT_PREVIEW_TEXT_STYLE = Object.freeze({
    size: 360,
    depth: 44,
    fitWidth: 960,
    fitHeight: 300,
    minScale: 0.28,
    maxScale: 1.6,
    faceOffset: 0.03
});
export const TOKEN_CHIP_FONT_URL = 'https://threejs.org/examples/fonts/helvetiker_regular.typeface.json';
export const D_MODEL = 768;
export const D_HEAD = Math.floor(D_MODEL / NUM_HEAD_SETS_LAYER);
export const VOCAB_SIZE = 50257;
export const CONTEXT_LEN = 1024;
