export const TRANSFORMER_VIEW2D_STAGED_HEAD_DETAIL_OVERVIEW_TO_HEAD_DURATION_MS = 1520;

export function shouldKeepTransformerView2dHeadDetailFitView(detailSemanticTargets = null) {
    if (Array.isArray(detailSemanticTargets)) {
        return detailSemanticTargets.length > 0;
    }
    return !!detailSemanticTargets;
}
