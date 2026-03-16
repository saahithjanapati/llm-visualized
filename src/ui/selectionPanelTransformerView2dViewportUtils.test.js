import { describe, expect, it } from 'vitest';

import { resolveViewportFitTransform } from '../view2d/runtime/View2dViewportController.js';
import {
    isTransformerView2dViewportAtFitScene,
    resolveTransformerView2dOverviewMinScale,
    shouldShowTransformerView2dFitSceneAction,
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

    it('treats the viewport as already fit when it matches the scene fit transform', () => {
        const fitBounds = {
            x: 120,
            y: 80,
            width: 640,
            height: 280
        };
        const controllerState = {
            viewport: {
                width: 1200,
                height: 720
            }
        };
        const fitTransform = resolveViewportFitTransform(fitBounds, controllerState.viewport, {
            padding: 28,
            minScale: TRANSFORMER_VIEW2D_OVERVIEW_MIN_SCALE_DEFAULT,
            maxScale: 10
        });

        const fitSceneState = {
            ...controllerState,
            scale: fitTransform.scale,
            panX: fitTransform.panX,
            panY: fitTransform.panY
        };

        expect(isTransformerView2dViewportAtFitScene({
            controllerState: fitSceneState,
            fitBounds,
            padding: 28,
            minScale: TRANSFORMER_VIEW2D_OVERVIEW_MIN_SCALE_DEFAULT,
            maxScale: 10
        })).toBe(true);
        expect(shouldShowTransformerView2dFitSceneAction({
            controllerState: fitSceneState,
            fitBounds,
            padding: 28,
            minScale: TRANSFORMER_VIEW2D_OVERVIEW_MIN_SCALE_DEFAULT,
            maxScale: 10
        })).toBe(false);
    });

    it('keeps the fit action hidden for tiny fit-preserving numeric drift', () => {
        const fitBounds = {
            x: 48,
            y: 64,
            width: 360,
            height: 180
        };
        const controllerState = {
            viewport: {
                width: 960,
                height: 540
            }
        };
        const fitTransform = resolveViewportFitTransform(fitBounds, controllerState.viewport, {
            padding: 28,
            minScale: TRANSFORMER_VIEW2D_OVERVIEW_MIN_SCALE_DEFAULT,
            maxScale: 10
        });

        expect(shouldShowTransformerView2dFitSceneAction({
            controllerState: {
                ...controllerState,
                scale: fitTransform.scale * 1.0008,
                panX: fitTransform.panX + 1.1,
                panY: fitTransform.panY - 1.1
            },
            fitBounds,
            padding: 28,
            minScale: TRANSFORMER_VIEW2D_OVERVIEW_MIN_SCALE_DEFAULT,
            maxScale: 10
        })).toBe(false);
    });

    it('shows the fit action once the viewport has meaningfully drifted off the fit', () => {
        const fitBounds = {
            x: 12,
            y: 18,
            width: 420,
            height: 210
        };
        const controllerState = {
            viewport: {
                width: 1000,
                height: 600
            }
        };
        const fitTransform = resolveViewportFitTransform(fitBounds, controllerState.viewport, {
            padding: 28,
            minScale: TRANSFORMER_VIEW2D_OVERVIEW_MIN_SCALE_DEFAULT,
            maxScale: 10
        });

        expect(shouldShowTransformerView2dFitSceneAction({
            controllerState: {
                ...controllerState,
                scale: fitTransform.scale * 0.94,
                panX: fitTransform.panX + 8,
                panY: fitTransform.panY - 5
            },
            fitBounds,
            padding: 28,
            minScale: TRANSFORMER_VIEW2D_OVERVIEW_MIN_SCALE_DEFAULT,
            maxScale: 10
        })).toBe(true);
    });

    it('keeps the fit action hidden when no fit target is available yet', () => {
        expect(shouldShowTransformerView2dFitSceneAction({
            controllerState: {
                scale: 1,
                panX: 0,
                panY: 0,
                viewport: {
                    width: 1000,
                    height: 600
                }
            },
            fitBounds: null,
            padding: 28,
            minScale: TRANSFORMER_VIEW2D_OVERVIEW_MIN_SCALE_DEFAULT,
            maxScale: 10
        })).toBe(false);
    });
});
