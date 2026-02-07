export const NUM_LAYERS = 12;
export const PROMPT_TOKENS = ['Can', '\u0120machines', '\u0120think', '?', '\u0120'];
export const POSITION_TOKENS = ['1', '2', '3', '4', '5'];
export const DEFAULT_PROMPT_LANES = Math.max(1, PROMPT_TOKENS.length);

// Visual config for the floating "token chip" labels at the base of the stack.
export const TOKEN_CHIP_STYLE = {
    padding: 140,
    minWidth: 440,
    minHeight: 150,
    height: 170,
    cornerRadius: 24,
    depth: 12,
    textSize: 90,
    textDepth: 16,
    textOffset: 0.6,
    riseDistance: 220,
    riseDelay: 120,
    riseDuration: 1200,
    vocabSlowdown: 1.6,
    positionSlowdown: 1.0,
    inset: 0,
    zOffset: 0,
    scale: 2.6,
    staticGap: 200,
    staticZOffset: 0,
    cameraHoldMs: 800,
    cameraReturnMs: 0
};

export const POSITION_CHIP_STYLE = {
    ...TOKEN_CHIP_STYLE,
    padding: 80,
    minWidth: 260,
    minHeight: 120,
    height: 130,
    textSize: 70,
    scale: 2.0
};

// GPT-2 BPE uses a leading U+0120 to indicate a space; render pure-space tokens visibly.
export const SPACE_TOKEN_DISPLAY = '" "';

// Camera + follow mode tuning for the 12-layer tower.
export const CAMERA_CONFIG = {
    initialPosition: [0, 11000, 16000],
    initialTarget: [0, 9000, 0],
    skipToEndPositionOffset: [0, 600, 36000],
    skipToEndTargetOffset: [0, 5200, 0],
    targetClampRadiusBase: 8000,
    targetClampRadiusPerLayer: 900,
    autoCameraHeadBias: 0.0,
    followDefaultCameraOffset: [-1215.87, 465.86, 3350.33],
    followDefaultTargetOffset: [1675.46, 227.33, -469.85],
    followMhsaCameraOffset: [1366.76, 1062.82, 1936.74],
    followMhsaTargetOffset: [3699.19, 110.55, 268.33],
    followConcatCameraOffset: [403.43, -14.39, 7.47],
    followConcatTargetOffset: [3383.52, 26.22, 364.05],
    followLnCameraOffset: [605.51, -78.03, 2433.13],
    followLnTargetOffset: [1026.71, 144.37, -607.81],
    followTravelCameraOffset: [1106.53, -860.48, 1389.16],
    followTravelTargetOffset: [4038.68, -398.41, 601.18],
    followTravelMobileCameraOffset: [1106.53, -860.48, 1389.16],
    followTravelMobileTargetOffset: [4038.68, -398.41, 601.18],
    autoCameraMobileScale: 1.0,
    autoCameraMobileShiftX: 0,
    autoCameraMhsaMobileShiftX: 0,
    autoCameraTravelMobileShiftX: 0,
    autoCameraScaleMinWidth: 360,
    autoCameraScaleMaxWidth: 980,
    autoCameraSmoothAlpha: 0.06,
    autoCameraOffsetLerpAlpha: 0.06,
    autoCameraViewBlendAlpha: 0.05
};
