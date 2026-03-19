import { describe, expect, it } from 'vitest';

import {
    TRANSFORMER_VIEW2D_STAGED_FOCUS_OVERVIEW_TO_TARGET_DURATION_MS,
    TRANSFORMER_VIEW2D_STAGED_DETAIL_FOCUS_SETTLE_MS,
    TRANSFORMER_VIEW2D_STAGED_HEAD_DETAIL_OVERVIEW_TO_HEAD_DURATION_MS,
    shouldKeepTransformerView2dHeadDetailFitView
} from './selectionPanelTransformerView2dTransitionUtils.js';

describe('selectionPanelTransformerView2dTransitionUtils', () => {
    it('keeps the staged MHSA overview-to-head entry motion brisk', () => {
        expect(TRANSFORMER_VIEW2D_STAGED_HEAD_DETAIL_OVERVIEW_TO_HEAD_DURATION_MS).toBe(1320);
    });

    it('uses a shared overview focus duration for direct in-canvas detail entry', () => {
        expect(TRANSFORMER_VIEW2D_STAGED_FOCUS_OVERVIEW_TO_TARGET_DURATION_MS).toBe(920);
    });

    it('holds briefly on the focused overview component before entering deep detail', () => {
        expect(TRANSFORMER_VIEW2D_STAGED_DETAIL_FOCUS_SETTLE_MS).toBe(160);
    });

    it('allows component focus when a specific MHSA component is targeted', () => {
        expect(shouldKeepTransformerView2dHeadDetailFitView([{
            componentKind: 'mhsa',
            layerIndex: 1,
            headIndex: 3,
            stage: 'attention',
            role: 'attention-post'
        }])).toBe(false);
    });

    it('keeps weight-matrix detail targets at fit scene', () => {
        expect(shouldKeepTransformerView2dHeadDetailFitView([{
            componentKind: 'mhsa',
            layerIndex: 1,
            headIndex: 3,
            stage: 'projection-q',
            role: 'projection-weight'
        }])).toBe(true);
        expect(shouldKeepTransformerView2dHeadDetailFitView([{
            componentKind: 'mlp',
            layerIndex: 1,
            stage: 'mlp-up',
            role: 'mlp-up-weight'
        }])).toBe(true);
        expect(shouldKeepTransformerView2dHeadDetailFitView([{
            componentKind: 'mlp',
            layerIndex: 1,
            stage: 'mlp-down',
            role: 'mlp-down-weight'
        }])).toBe(true);
    });

    it('keeps the scene fit when no component target is provided', () => {
        expect(shouldKeepTransformerView2dHeadDetailFitView([])).toBe(true);
        expect(shouldKeepTransformerView2dHeadDetailFitView(null)).toBe(true);
    });
});
