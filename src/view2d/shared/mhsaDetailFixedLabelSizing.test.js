import { describe, expect, it } from 'vitest';

import {
    VIEW2D_TEXT_ZOOM_BEHAVIORS,
    resolveMhsaDetailFixedTextSizing,
    resolveView2dSceneTextZoomPolicy
} from './mhsaDetailFixedLabelSizing.js';

describe('mhsaDetailFixedLabelSizing', () => {
    it('uses scene-relative sizing for MHSA detail operators', () => {
        const sizing = resolveMhsaDetailFixedTextSizing({
            metadata: {
                visualContract: 'selection-panel-mhsa-v1'
            }
        });

        expect(sizing?.captionLabelScreenFontPx).toBeNull();
        expect(sizing?.captionDimensionsScreenFontPx).toBeNull();
        expect(sizing?.operatorBehavior).toBe(VIEW2D_TEXT_ZOOM_BEHAVIORS.SCENE_RELATIVE);
        expect(sizing?.operatorMinScreenFontPx).toBeNull();
        expect(sizing?.operatorMinScreenHeightPx).toBe(0);
    });

    it('applies the same always-visible matrix caption policy to output-projection detail scenes', () => {
        const scene = {
            metadata: {
                visualContract: 'selection-panel-output-projection-v1'
            }
        };

        const sizing = resolveMhsaDetailFixedTextSizing(scene);
        const zoomPolicy = resolveView2dSceneTextZoomPolicy(scene);

        expect(sizing?.operatorBehavior).toBe(VIEW2D_TEXT_ZOOM_BEHAVIORS.SCENE_RELATIVE);
        expect(zoomPolicy.useUniformMatrixCaptions).toBe(true);
        expect(zoomPolicy.captionBehavior).toBe(VIEW2D_TEXT_ZOOM_BEHAVIORS.SCENE_RELATIVE);
    });

    it('uses the same scene-relative caption policy for MLP detail scenes', () => {
        const scene = {
            metadata: {
                visualContract: 'selection-panel-mlp-v1'
            }
        };

        const sizing = resolveMhsaDetailFixedTextSizing(scene);
        const zoomPolicy = resolveView2dSceneTextZoomPolicy(scene);

        expect(zoomPolicy.useUniformMatrixCaptions).toBe(true);
        expect(zoomPolicy.captionBehavior).toBe(VIEW2D_TEXT_ZOOM_BEHAVIORS.SCENE_RELATIVE);
        expect(sizing?.captionLabelScreenFontPx).toBeNull();
        expect(sizing?.captionDimensionsScreenFontPx).toBeNull();
    });

    it('uses fixed larger caption and text sizing for layer norm detail scenes', () => {
        const scene = {
            metadata: {
                visualContract: 'selection-panel-layer-norm-v1'
            }
        };

        const sizing = resolveMhsaDetailFixedTextSizing(scene, 1280);
        const zoomPolicy = resolveView2dSceneTextZoomPolicy(scene);

        expect(zoomPolicy.useUniformMatrixCaptions).toBe(true);
        expect(zoomPolicy.captionBehavior).toBe(VIEW2D_TEXT_ZOOM_BEHAVIORS.SCREEN_FIXED);
        expect(zoomPolicy.domTextBehavior).toBe(VIEW2D_TEXT_ZOOM_BEHAVIORS.SCREEN_FIXED);
        expect(sizing?.textScreenFontPx).toBe(19);
        expect(sizing?.captionLabelScreenFontPx).toBe(17.5);
        expect(sizing?.captionDimensionsScreenFontPx).toBe(14);
        expect(sizing?.rowLabelScreenFontPx).toBe(13.5);
        expect(sizing?.operatorMinScreenFontPx).toBe(17);
    });

    it('ignores non-MHSA scenes', () => {
        expect(resolveMhsaDetailFixedTextSizing({
            metadata: {
                visualContract: 'transformer-overview-v1'
            }
        })).toBeNull();
    });
});
