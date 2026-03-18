export const TRANSFORMER_VIEW2D_STAGED_FOCUS_OVERVIEW_TO_TARGET_DURATION_MS = 920;
export const TRANSFORMER_VIEW2D_STAGED_HEAD_DETAIL_OVERVIEW_TO_HEAD_DURATION_MS = 1320;
export const TRANSFORMER_VIEW2D_STAGED_DETAIL_FOCUS_SETTLE_MS = 160;

export function shouldKeepTransformerView2dHeadDetailFitView(detailSemanticTargets = null) {
    if (Array.isArray(detailSemanticTargets)) {
        return detailSemanticTargets.length === 0;
    }
    return !detailSemanticTargets;
}
