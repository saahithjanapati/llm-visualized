import { describe, expect, it } from 'vitest';

import { createSceneModel } from '../src/view2d/schema/sceneTypes.js';
import {
    isMhsaDetailScene,
    resolveMhsaDetailFixedTextSizing,
    resolveView2dSceneTextZoomPolicy,
    VIEW2D_TEXT_ZOOM_BEHAVIORS
} from '../src/view2d/shared/mhsaDetailFixedLabelSizing.js';

function createSceneWithMetadata(metadata = {}) {
    return createSceneModel({
        semantic: {
            componentKind: 'mhsa',
            role: 'scene'
        },
        metadata
    });
}

describe('resolveView2dSceneTextZoomPolicy', () => {
    it('keeps non-detail scenes on the adaptive screen-sized path', () => {
        const scene = createSceneWithMetadata({
            source: 'test-scene'
        });

        expect(isMhsaDetailScene(scene)).toBe(false);
        expect(resolveView2dSceneTextZoomPolicy(scene)).toEqual({
            captionBehavior: VIEW2D_TEXT_ZOOM_BEHAVIORS.SCREEN_ADAPTIVE,
            domTextBehavior: VIEW2D_TEXT_ZOOM_BEHAVIORS.SCREEN_ADAPTIVE,
            useUniformMatrixCaptions: false,
            textScreenFontPx: null,
            captionLabelScreenFontPx: null,
            captionDimensionsScreenFontPx: null,
            rowLabelScreenFontPx: null,
            operatorMinScreenFontPx: null,
            operatorMinScreenHeightPx: null
        });
    });

    it('marks MHSA detail scenes as scene-relative for captions and DOM text', () => {
        const scene = createSceneWithMetadata({
            visualContract: 'selection-panel-mhsa-v1',
            source: 'selectionPanelMhsaTokenMatrixUtils'
        });

        expect(isMhsaDetailScene(scene)).toBe(true);
        expect(resolveView2dSceneTextZoomPolicy(scene)).toEqual({
            captionBehavior: VIEW2D_TEXT_ZOOM_BEHAVIORS.SCENE_RELATIVE,
            domTextBehavior: VIEW2D_TEXT_ZOOM_BEHAVIORS.SCENE_RELATIVE,
            useUniformMatrixCaptions: true,
            textScreenFontPx: null,
            captionLabelScreenFontPx: null,
            captionDimensionsScreenFontPx: null,
            rowLabelScreenFontPx: 14,
            operatorMinScreenFontPx: 12,
            operatorMinScreenHeightPx: 0
        });
    });
});

describe('resolveMhsaDetailFixedTextSizing', () => {
    it('only keeps row labels on the fixed-screen path for MHSA detail scenes', () => {
        const detailScene = createSceneWithMetadata({
            visualContract: 'selection-panel-mhsa-v1'
        });
        const nonDetailScene = createSceneWithMetadata();

        expect(resolveMhsaDetailFixedTextSizing(nonDetailScene)).toBeNull();
        expect(resolveMhsaDetailFixedTextSizing(detailScene)).toEqual({
            textScreenFontPx: null,
            captionLabelScreenFontPx: null,
            captionDimensionsScreenFontPx: null,
            rowLabelScreenFontPx: 14,
            operatorMinScreenFontPx: 12,
            operatorMinScreenHeightPx: 0
        });
    });
});
