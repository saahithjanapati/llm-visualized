import { describe, expect, it } from 'vitest';

import {
    buildTransformerView2dOverviewState,
    TRANSFORMER_VIEW2D_OVERVIEW_FOCUS_LABEL
} from './selectionPanelTransformerView2dStateUtils.js';

describe('selectionPanelTransformerView2dStateUtils', () => {
    it('builds a cleared full-graph overview state', () => {
        expect(buildTransformerView2dOverviewState()).toEqual({
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
        });
    });
});
