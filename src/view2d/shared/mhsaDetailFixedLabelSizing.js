const MHSA_DETAIL_VISUAL_CONTRACT = 'selection-panel-mhsa-v1';

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

export function isMhsaDetailScene(scene = null) {
    return String(scene?.metadata?.visualContract || '').trim().toLowerCase() === MHSA_DETAIL_VISUAL_CONTRACT;
}

export function resolveView2dSceneTextZoomPolicy(scene = null) {
    return isMhsaDetailScene(scene)
        ? MHSA_DETAIL_VIEW2D_SCENE_TEXT_ZOOM_POLICY
        : DEFAULT_VIEW2D_SCENE_TEXT_ZOOM_POLICY;
}

export function resolveMhsaDetailFixedTextSizing(scene = null, viewportWidth = 0) {
    if (!isMhsaDetailScene(scene)) return null;
    const textZoomPolicy = resolveView2dSceneTextZoomPolicy(scene);
    return {
        textScreenFontPx: textZoomPolicy.textScreenFontPx,
        captionLabelScreenFontPx: textZoomPolicy.captionLabelScreenFontPx,
        captionDimensionsScreenFontPx: textZoomPolicy.captionDimensionsScreenFontPx,
        rowLabelScreenFontPx: textZoomPolicy.rowLabelScreenFontPx,
        operatorBehavior: textZoomPolicy.operatorBehavior,
        operatorMinScreenFontPx: textZoomPolicy.operatorMinScreenFontPx,
        operatorMinScreenHeightPx: textZoomPolicy.operatorMinScreenHeightPx
    };
}
