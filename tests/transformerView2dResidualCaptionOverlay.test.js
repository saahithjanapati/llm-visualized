// @vitest-environment jsdom

import { afterEach, describe, expect, it, vi } from 'vitest';

import {
    createTransformerView2dResidualCaptionOverlay,
    resolveCaptionFontPx,
    resolveCaptionScreenExtent
} from '../src/ui/transformerView2dResidualCaptionOverlay.js';
import { buildSceneLayout } from '../src/view2d/layout/buildSceneLayout.js';
import {
    createSceneModel,
    createTextNode
} from '../src/view2d/schema/sceneTypes.js';

afterEach(() => {
    delete window.katex;
});

describe('resolveCaptionFontPx', () => {
    it('keeps bottom captions screen-anchored and capped at a constant max size', () => {
        const midRamp = resolveCaptionFontPx({
            useMatrixRelativeSizing: false,
            projectedContentHeight: 40,
            sizeProgress: 0.5,
            minFontPx: 12,
            maxFontPx: 14
        });
        const afterLock = resolveCaptionFontPx({
            useMatrixRelativeSizing: false,
            projectedContentHeight: 124,
            sizeProgress: 1,
            minFontPx: 12,
            maxFontPx: 14
        });

        expect(midRamp).toBeCloseTo(13, 6);
        expect(afterLock).toBeCloseTo(14, 6);
    });

    it('keeps matrix-relative captions proportional to the node height', () => {
        const fontPx = resolveCaptionFontPx({
            useMatrixRelativeSizing: true,
            projectedContentHeight: 100,
            minFontPx: 12,
            heightRatio: 0.34
        });

        expect(fontPx).toBeCloseTo(34, 6);
    });
});

describe('resolveCaptionScreenExtent', () => {
    it('uses the smaller projected extent for bottom captions so transpose matrices do not show labels early', () => {
        const extent = resolveCaptionScreenExtent({
            captionPosition: 'bottom',
            projectedContentWidth: 35,
            projectedContentHeight: 72
        });

        expect(extent).toBe(35);
    });

    it('keeps height-driven sizing for non-bottom or node-scaled captions', () => {
        expect(resolveCaptionScreenExtent({
            captionPosition: 'float-top',
            projectedContentWidth: 35,
            projectedContentHeight: 72
        })).toBe(72);

        expect(resolveCaptionScreenExtent({
            captionPosition: 'bottom',
            projectedContentWidth: 35,
            projectedContentHeight: 72,
            scaleWithNode: true
        })).toBe(72);
    });
});

describe('createTransformerView2dResidualCaptionOverlay', () => {
    it('renders dom-katex text nodes through the overlay instead of the canvas tex fallback', () => {
        window.katex = {
            renderToString: vi.fn(() => '<span class="katex">sqrt(d_head)</span>')
        };
        const parent = document.createElement('div');
        const canvas = document.createElement('canvas');
        Object.defineProperty(canvas, 'clientWidth', { value: 640, configurable: true });
        Object.defineProperty(canvas, 'clientHeight', { value: 360, configurable: true });
        parent.appendChild(canvas);

        const scene = createSceneModel({
            semantic: { componentKind: 'test-scene' },
            nodes: [
                createTextNode({
                    role: 'attention-scale',
                    semantic: { componentKind: 'mhsa', role: 'attention-scale' },
                    tex: '\\sqrt{d_{\\mathrm{head}}}',
                    text: 'sqrt(d_head)',
                    metadata: {
                        renderMode: 'dom-katex',
                        minScreenHeightPx: 0,
                        persistentMinScreenFontPx: 11
                    }
                })
            ]
        });
        const layout = buildSceneLayout(scene);
        const overlay = createTransformerView2dResidualCaptionOverlay({
            parent
        });

        expect(overlay.sync({
            scene,
            layout,
            canvas,
            projectBounds: (bounds) => ({ ...bounds }),
            visible: true,
            enabled: true
        })).toBe(true);

        const labelEl = overlay.element.querySelector('.detail-transformer-view2d-dom-text-line');

        expect(labelEl?.innerHTML).toContain('katex');
        expect(window.katex.renderToString).toHaveBeenCalledWith('\\sqrt{d_{\\mathrm{head}}}', {
            throwOnError: false,
            displayMode: false
        });
        overlay.destroy();
    });
});
