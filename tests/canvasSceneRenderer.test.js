// @vitest-environment jsdom

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { buildMhsaSceneModel } from '../src/view2d/model/buildMhsaSceneModel.js';
import { buildSceneLayout } from '../src/view2d/layout/buildSceneLayout.js';
import { CanvasSceneRenderer, syncCanvasResolution } from '../src/view2d/render/canvas/CanvasSceneRenderer.js';

function createActivationSource(tokenCount = 4) {
    return {
        meta: {
            prompt_tokens: Array.from({ length: tokenCount }, (_, index) => index),
            token_display_strings: Array.from({ length: tokenCount }, (_, index) => `tok_${index}`)
        },
        getLayerLn1(layerIndex, mode, tokenIndex) {
            if (mode !== 'shift') return null;
            return Array.from({ length: 768 }, (_, index) => ((index % 11) - 5) * (tokenIndex + 1) * 0.03);
        },
        getLayerQKVVector(layerIndex, kind, headIndex, tokenIndex, width) {
            const scale = kind === 'q' ? 0.09 : (kind === 'k' ? 0.12 : 0.18);
            return Array.from({ length: width }, (_, index) => ((index % 7) - 3) * scale * (tokenIndex + 1));
        },
        getLayerQKVScalar(layerIndex, kind, headIndex, tokenIndex) {
            return (tokenIndex + 1) * 0.12;
        },
        getAttentionScoresRow(layerIndex, stage, headIndex, tokenIndex) {
            return Array.from({ length: tokenCount }, (_, columnIndex) => {
                if (columnIndex > tokenIndex) return stage === 'pre' ? -1000 : 0;
                return stage === 'pre' ? ((tokenIndex + 1) * (columnIndex + 1)) / 4 : 1 / (tokenIndex + 1);
            });
        },
        getAttentionWeightedSum(layerIndex, headIndex, tokenIndex, width) {
            return Array.from({ length: width }, (_, index) => ((index % 5) - 2) * 0.07 * (tokenIndex + 1));
        }
    };
}

function createMockContext() {
    return {
        beginPath() {},
        moveTo() {},
        lineTo() {},
        arcTo() {},
        closePath() {},
        fill() {},
        stroke() {},
        fillRect() {},
        clearRect() {},
        save() {},
        restore() {},
        translate() {},
        scale() {},
        setTransform() {},
        fillText() {},
        createLinearGradient() {
            return {
                addColorStop() {}
            };
        },
        font: '',
        textAlign: 'left',
        textBaseline: 'alphabetic',
        fillStyle: '',
        strokeStyle: '',
        lineWidth: 1
    };
}

describe('CanvasSceneRenderer', () => {
    let canvas;
    let ctx;

    beforeEach(() => {
        canvas = document.createElement('canvas');
        ctx = createMockContext();
        canvas.getContext = vi.fn(() => ctx);
        canvas.getBoundingClientRect = () => ({
            width: 640,
            height: 360
        });
        Object.defineProperty(window, 'devicePixelRatio', {
            configurable: true,
            value: 2
        });
    });

    afterEach(() => {
        delete window.__RENDER_PIXEL_RATIO;
        delete window.__RENDER_DPR_CAP;
    });

    it('sizes the backing canvas using the repo DPR policy', () => {
        const resolution = syncCanvasResolution(canvas, { width: 320, height: 180 });

        expect(resolution.dpr).toBe(2);
        expect(canvas.width).toBe(640);
        expect(canvas.height).toBe(360);
    });

    it('renders a laid-out MHSA scene without throwing', () => {
        const scene = buildMhsaSceneModel({
            activationSource: createActivationSource(4),
            layerIndex: 2,
            headIndex: 6
        });
        const layout = buildSceneLayout(scene);
        const renderer = new CanvasSceneRenderer({ canvas });

        renderer.setScene(scene, layout);

        expect(renderer.render({ width: 640, height: 360, dpr: 1.5 })).toBe(true);
        expect(canvas.width).toBe(960);
        expect(canvas.height).toBe(540);
        expect(canvas.getContext).toHaveBeenCalledWith('2d');
    });

    it('honors an external viewport transform when provided', () => {
        const scene = buildMhsaSceneModel({
            activationSource: createActivationSource(4),
            layerIndex: 2,
            headIndex: 6
        });
        const layout = buildSceneLayout(scene);
        const renderer = new CanvasSceneRenderer({ canvas });

        renderer.setScene(scene, layout);

        expect(renderer.render({
            width: 640,
            height: 360,
            dpr: 1,
            viewportTransform: {
                source: 'test-fixture',
                scale: 0.42,
                offsetX: -120,
                offsetY: 36
            }
        })).toBe(true);
        expect(renderer.getLastRenderState()?.viewportSource).toBe('test-fixture');
        expect(renderer.getLastRenderState()?.worldScale).toBeCloseTo(0.42, 6);
        expect(renderer.getLastRenderState()?.offsetX).toBe(-120);
        expect(renderer.getLastRenderState()?.offsetY).toBe(36);
    });
});
