const MHSA_DETAIL_VISUAL_CONTRACT = 'selection-panel-mhsa-v1';
const OUTPUT_PROJECTION_DETAIL_VISUAL_CONTRACT = 'selection-panel-output-projection-v1';
const MLP_DETAIL_VISUAL_CONTRACT = 'selection-panel-mlp-v1';
const LAYER_NORM_DETAIL_VISUAL_CONTRACT = 'selection-panel-layer-norm-v1';

export const MHSA_DETAIL_FIXED_SHARED_SCREEN_FONT_PX = 14;
export const MHSA_DETAIL_FIXED_TEXT_SCREEN_FONT_PX = MHSA_DETAIL_FIXED_SHARED_SCREEN_FONT_PX;
export const MHSA_DETAIL_FIXED_CAPTION_LABEL_SCREEN_FONT_PX = MHSA_DETAIL_FIXED_SHARED_SCREEN_FONT_PX;
export const MHSA_DETAIL_FIXED_CAPTION_DIMENSIONS_SCREEN_FONT_PX = MHSA_DETAIL_FIXED_SHARED_SCREEN_FONT_PX;
export const MHSA_DETAIL_FIXED_ROW_LABEL_SCREEN_FONT_PX = MHSA_DETAIL_FIXED_SHARED_SCREEN_FONT_PX;
export const MHSA_DETAIL_OPERATOR_MIN_SCREEN_FONT_PX = 12;

export const VIEW2D_TEXT_ZOOM_BEHAVIORS = Object.freeze({
    SCREEN_ADAPTIVE: 'screen-adaptive',
    SCREEN_FIXED: 'screen-fixed',
    SCENE_RELATIVE: 'scene-relative'
});

const DEFAULT_VIEW2D_SCENE_TEXT_ZOOM_POLICY = Object.freeze({
    captionBehavior: VIEW2D_TEXT_ZOOM_BEHAVIORS.SCREEN_ADAPTIVE,
    domTextBehavior: VIEW2D_TEXT_ZOOM_BEHAVIORS.SCREEN_ADAPTIVE,
    operatorBehavior: VIEW2D_TEXT_ZOOM_BEHAVIORS.SCREEN_ADAPTIVE,
    useUniformMatrixCaptions: false,
    textScreenFontPx: null,
    captionLabelScreenFontPx: null,
    captionDimensionsScreenFontPx: null,
    rowLabelScreenFontPx: null,
    operatorMinScreenFontPx: null,
    operatorMinScreenHeightPx: null
});

const MHSA_DETAIL_VIEW2D_SCENE_TEXT_ZOOM_POLICY = Object.freeze({
    captionBehavior: VIEW2D_TEXT_ZOOM_BEHAVIORS.SCENE_RELATIVE,
    domTextBehavior: VIEW2D_TEXT_ZOOM_BEHAVIORS.SCENE_RELATIVE,
    operatorBehavior: VIEW2D_TEXT_ZOOM_BEHAVIORS.SCENE_RELATIVE,
    useUniformMatrixCaptions: true,
    textScreenFontPx: null,
    captionLabelScreenFontPx: null,
    captionDimensionsScreenFontPx: null,
    rowLabelScreenFontPx: MHSA_DETAIL_FIXED_ROW_LABEL_SCREEN_FONT_PX,
    operatorMinScreenFontPx: null,
    operatorMinScreenHeightPx: 0
});

const LAYER_NORM_DETAIL_VIEW2D_SCENE_TEXT_ZOOM_POLICY = Object.freeze({
    captionBehavior: VIEW2D_TEXT_ZOOM_BEHAVIORS.SCREEN_FIXED,
    domTextBehavior: VIEW2D_TEXT_ZOOM_BEHAVIORS.SCREEN_FIXED,
    operatorBehavior: VIEW2D_TEXT_ZOOM_BEHAVIORS.SCREEN_FIXED,
    useUniformMatrixCaptions: true,
    textScreenFontPx: null,
    captionLabelScreenFontPx: null,
    captionDimensionsScreenFontPx: null,
    rowLabelScreenFontPx: null,
    operatorMinScreenFontPx: null,
    operatorMinScreenHeightPx: 0
});

function isLayerNormDetailScene(scene = null) {
    const visualContract = String(scene?.metadata?.visualContract || '').trim().toLowerCase();
    return visualContract === LAYER_NORM_DETAIL_VISUAL_CONTRACT;
}

function resolveLayerNormDetailFixedScreenFontSizing(viewportWidth = 0) {
    const safeViewportWidth = Number.isFinite(viewportWidth) ? Math.max(0, Number(viewportWidth)) : 0;
    if (safeViewportWidth > 0 && safeViewportWidth < 720) {
        return {
            textScreenFontPx: 17,
            captionLabelScreenFontPx: 15.5,
            captionDimensionsScreenFontPx: 12.5,
            rowLabelScreenFontPx: 12.5,
            operatorMinScreenFontPx: 15
        };
    }
    if (safeViewportWidth > 0 && safeViewportWidth < 1120) {
        return {
            textScreenFontPx: 18,
            captionLabelScreenFontPx: 16.5,
            captionDimensionsScreenFontPx: 13.5,
            rowLabelScreenFontPx: 13,
            operatorMinScreenFontPx: 16
        };
    }
    return {
        textScreenFontPx: 19,
        captionLabelScreenFontPx: 17.5,
        captionDimensionsScreenFontPx: 14,
        rowLabelScreenFontPx: 13.5,
        operatorMinScreenFontPx: 17
    };
}

export function isMhsaDetailScene(scene = null) {
    const visualContract = String(scene?.metadata?.visualContract || '').trim().toLowerCase();
    return visualContract === MHSA_DETAIL_VISUAL_CONTRACT
        || visualContract === OUTPUT_PROJECTION_DETAIL_VISUAL_CONTRACT
        || visualContract === MLP_DETAIL_VISUAL_CONTRACT
        || visualContract === LAYER_NORM_DETAIL_VISUAL_CONTRACT;
}

export function resolveView2dSceneTextZoomPolicy(scene = null) {
    if (isLayerNormDetailScene(scene)) {
        return LAYER_NORM_DETAIL_VIEW2D_SCENE_TEXT_ZOOM_POLICY;
    }
    return isMhsaDetailScene(scene)
        ? MHSA_DETAIL_VIEW2D_SCENE_TEXT_ZOOM_POLICY
        : DEFAULT_VIEW2D_SCENE_TEXT_ZOOM_POLICY;
}

export function resolveMhsaDetailFixedTextSizing(scene = null, viewportWidth = 0) {
    if (!isMhsaDetailScene(scene)) return null;
    const textZoomPolicy = resolveView2dSceneTextZoomPolicy(scene);
    const layerNormFixedSizing = isLayerNormDetailScene(scene)
        ? resolveLayerNormDetailFixedScreenFontSizing(viewportWidth)
        : null;
    return {
        textScreenFontPx: layerNormFixedSizing?.textScreenFontPx ?? textZoomPolicy.textScreenFontPx,
        captionLabelScreenFontPx: layerNormFixedSizing?.captionLabelScreenFontPx ?? textZoomPolicy.captionLabelScreenFontPx,
        captionDimensionsScreenFontPx: layerNormFixedSizing?.captionDimensionsScreenFontPx ?? textZoomPolicy.captionDimensionsScreenFontPx,
        rowLabelScreenFontPx: layerNormFixedSizing?.rowLabelScreenFontPx ?? textZoomPolicy.rowLabelScreenFontPx,
        operatorBehavior: textZoomPolicy.operatorBehavior,
        operatorMinScreenFontPx: layerNormFixedSizing?.operatorMinScreenFontPx ?? textZoomPolicy.operatorMinScreenFontPx,
        operatorMinScreenHeightPx: textZoomPolicy.operatorMinScreenHeightPx
    };
}
