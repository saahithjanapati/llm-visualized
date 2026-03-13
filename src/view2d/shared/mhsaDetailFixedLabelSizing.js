const MHSA_DETAIL_VISUAL_CONTRACT = 'selection-panel-mhsa-v1';

export const MHSA_DETAIL_FIXED_SHARED_SCREEN_FONT_PX = 14;
export const MHSA_DETAIL_FIXED_TEXT_SCREEN_FONT_PX = MHSA_DETAIL_FIXED_SHARED_SCREEN_FONT_PX;
export const MHSA_DETAIL_FIXED_CAPTION_LABEL_SCREEN_FONT_PX = MHSA_DETAIL_FIXED_SHARED_SCREEN_FONT_PX;
export const MHSA_DETAIL_FIXED_CAPTION_DIMENSIONS_SCREEN_FONT_PX = MHSA_DETAIL_FIXED_SHARED_SCREEN_FONT_PX;
export const MHSA_DETAIL_FIXED_ROW_LABEL_SCREEN_FONT_PX = MHSA_DETAIL_FIXED_SHARED_SCREEN_FONT_PX;

export function isMhsaDetailScene(scene = null) {
    return String(scene?.metadata?.visualContract || '').trim().toLowerCase() === MHSA_DETAIL_VISUAL_CONTRACT;
}

export function resolveMhsaDetailFixedTextSizing(scene = null, viewportWidth = 0) {
    if (!isMhsaDetailScene(scene)) return null;
    return {
        textScreenFontPx: MHSA_DETAIL_FIXED_TEXT_SCREEN_FONT_PX,
        rowLabelScreenFontPx: MHSA_DETAIL_FIXED_ROW_LABEL_SCREEN_FONT_PX
    };
}
