import { describeTransformerView2dTarget } from '../view2d/transformerView2dTargets.js';

export const TRANSFORMER_VIEW2D_OVERVIEW_FOCUS_LABEL = describeTransformerView2dTarget(null);

export function buildTransformerView2dOverviewState() {
    return {
        baseSemanticTarget: null,
        baseFocusLabel: TRANSFORMER_VIEW2D_OVERVIEW_FOCUS_LABEL,
        detailTargets: {
            headDetailTarget: null,
            concatDetailTarget: null,
            outputProjectionDetailTarget: null,
            mlpDetailTarget: null
        },
        detailSemanticTargets: [],
        detailFocusLabel: '',
        pendingDetailInteractionTargets: [],
        semanticTarget: null,
        focusLabel: TRANSFORMER_VIEW2D_OVERVIEW_FOCUS_LABEL
    };
}
