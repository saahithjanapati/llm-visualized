export const TRANSFORMER_VIEW2D_OVERVIEW_MIN_SCALE_DEFAULT = 0.035;

const TRANSFORMER_VIEW2D_OVERVIEW_MIN_SCALE_SMALL_SCREEN_FLOOR = 0.015;
const TRANSFORMER_VIEW2D_OVERVIEW_MIN_SCALE_REFERENCE_WIDTH = 640;

export function resolveTransformerView2dOverviewMinScale({
    isSmallScreen = false,
    viewportWidth = 0
} = {}) {
    if (!isSmallScreen) {
        return TRANSFORMER_VIEW2D_OVERVIEW_MIN_SCALE_DEFAULT;
    }

    const safeViewportWidth = Math.max(1, Math.floor(Number(viewportWidth) || 0));
    const widthRatio = Math.min(1, safeViewportWidth / TRANSFORMER_VIEW2D_OVERVIEW_MIN_SCALE_REFERENCE_WIDTH);

    return Math.max(
        TRANSFORMER_VIEW2D_OVERVIEW_MIN_SCALE_SMALL_SCREEN_FLOOR,
        TRANSFORMER_VIEW2D_OVERVIEW_MIN_SCALE_DEFAULT * widthRatio
    );
}
