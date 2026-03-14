import { describe, expect, it } from 'vitest';

import {
    resolveTransformerView2dOverviewMinScale,
    TRANSFORMER_VIEW2D_OVERVIEW_MIN_SCALE_DEFAULT
} from './selectionPanelTransformerView2dViewportUtils.js';

describe('selectionPanelTransformerView2dViewportUtils', () => {
    it('keeps the desktop overview zoom floor unchanged', () => {
        expect(resolveTransformerView2dOverviewMinScale({
            isSmallScreen: false,
            viewportWidth: 320
        })).toBe(TRANSFORMER_VIEW2D_OVERVIEW_MIN_SCALE_DEFAULT);
    });

    it('lowers the overview zoom floor on narrow small screens', () => {
        expect(resolveTransformerView2dOverviewMinScale({
            isSmallScreen: true,
            viewportWidth: 320
        })).toBeCloseTo(0.0175, 6);
    });

    it('clamps the mobile overview zoom floor at a small but usable minimum', () => {
        expect(resolveTransformerView2dOverviewMinScale({
            isSmallScreen: true,
            viewportWidth: 240
        })).toBe(0.015);
    });
});
