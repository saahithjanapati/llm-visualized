import { describe, expect, it } from 'vitest';

import {
    TRANSFORMER_VIEW2D_STAGED_HEAD_DETAIL_OVERVIEW_TO_HEAD_DURATION_MS,
    shouldKeepTransformerView2dHeadDetailFitView
} from './selectionPanelTransformerView2dTransitionUtils.js';

describe('selectionPanelTransformerView2dTransitionUtils', () => {
    it('slows the staged MHSA overview-to-head entry motion slightly', () => {
        expect(TRANSFORMER_VIEW2D_STAGED_HEAD_DETAIL_OVERVIEW_TO_HEAD_DURATION_MS).toBe(1520);
    });

    it('keeps the head-detail scene fit when a specific MHSA component is targeted', () => {
        expect(shouldKeepTransformerView2dHeadDetailFitView([{
            componentKind: 'mhsa',
            layerIndex: 1,
            headIndex: 3,
            stage: 'attention',
            role: 'attention-post'
        }])).toBe(true);
    });

    it('allows normal head-detail focus behavior when no component target is provided', () => {
        expect(shouldKeepTransformerView2dHeadDetailFitView([])).toBe(false);
        expect(shouldKeepTransformerView2dHeadDetailFitView(null)).toBe(false);
    });
});
