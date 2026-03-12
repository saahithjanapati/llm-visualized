// @vitest-environment jsdom

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { buildMhsaSceneModel } from '../src/view2d/model/buildMhsaSceneModel.js';
import { buildTransformerSceneModel } from '../src/view2d/model/buildTransformerSceneModel.js';
import { buildSceneLayout } from '../src/view2d/layout/buildSceneLayout.js';
import { CanvasSceneRenderer, syncCanvasResolution } from '../src/view2d/render/canvas/CanvasSceneRenderer.js';
import {
    createAnchorRef,
    createConnectorNode,
    createGroupNode,
    createSceneModel,
    createTextNode,
    VIEW2D_ANCHOR_SIDES,
    VIEW2D_CONNECTOR_ROUTES
} from '../src/view2d/schema/sceneTypes.js';
import { flattenSceneNodes } from '../src/view2d/schema/sceneTypes.js';
import { resolveView2dStyle, VIEW2D_STYLE_KEYS } from '../src/view2d/theme/visualTokens.js';

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
    const operations = [];
    return {
        operations,
        beginPath() {
            operations.push({ type: 'beginPath' });
        },
        moveTo(x, y) {
            operations.push({ type: 'moveTo', x, y });
        },
        lineTo(x, y) {
            operations.push({ type: 'lineTo', x, y });
        },
        arcTo() {},
        closePath() {},
        fill() {
            operations.push({ type: 'fill', fillStyle: this.fillStyle, globalAlpha: this.globalAlpha });
        },
        stroke() {
            operations.push({
                type: 'stroke',
                strokeStyle: this.strokeStyle,
                globalAlpha: this.globalAlpha,
                lineWidth: this.lineWidth
            });
        },
        clip() {},
        fillRect() {},
        clearRect() {},
        save() {},
        restore() {},
        translate() {},
        scale() {},
        setTransform() {},
        fillText(text, x, y) {
            operations.push({ type: 'fillText', text, x, y });
        },
        createLinearGradient() {
            return {
                addColorStop() {}
            };
        },
        createRadialGradient() {
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

    it('uses canvas-friendly vivid fills for MHSA projection cards', () => {
        const scene = buildMhsaSceneModel({
            activationSource: createActivationSource(4),
            layerIndex: 2,
            headIndex: 6
        });
        const nodes = flattenSceneNodes(scene);
        const weightNodes = nodes.filter((node) => node.role === 'projection-weight');

        expect(weightNodes).toHaveLength(3);
        weightNodes.forEach((node) => {
            expect(node.visual?.background).toMatch(/^linear-gradient\(/);
        });
        expect(resolveView2dStyle(VIEW2D_STYLE_KEYS.MHSA_Q)?.fill).toMatch(/^linear-gradient\(/);
        expect(resolveView2dStyle(VIEW2D_STYLE_KEYS.MHSA_K)?.fill).toMatch(/^linear-gradient\(/);
        expect(resolveView2dStyle(VIEW2D_STYLE_KEYS.MHSA_V)?.fill).toMatch(/^linear-gradient\(/);
        expect(resolveView2dStyle(VIEW2D_STYLE_KEYS.MHSA_HEAD_OUTPUT)?.fill).toMatch(/^linear-gradient\(/);
    });

    it('snaps connector path strokes to a stable half-pixel grid', () => {
        const sourceNode = createTextNode({
            role: 'source',
            semantic: { componentKind: 'test', role: 'source' },
            text: 'A',
            visual: { styleKey: VIEW2D_STYLE_KEYS.LABEL }
        });
        const targetNode = createTextNode({
            role: 'target',
            semantic: { componentKind: 'test', role: 'target' },
            text: 'B',
            visual: { styleKey: VIEW2D_STYLE_KEYS.LABEL }
        });
        const connectorNode = createConnectorNode({
            role: 'connector',
            semantic: { componentKind: 'test', role: 'connector' },
            source: createAnchorRef(sourceNode.id, VIEW2D_ANCHOR_SIDES.RIGHT),
            target: createAnchorRef(targetNode.id, VIEW2D_ANCHOR_SIDES.LEFT),
            route: VIEW2D_CONNECTOR_ROUTES.HORIZONTAL,
            gap: 7,
            visual: { styleKey: VIEW2D_STYLE_KEYS.CONNECTOR_NEUTRAL }
        });
        const scene = createSceneModel({
            semantic: { componentKind: 'test-scene' },
            nodes: [
                sourceNode,
                targetNode,
                createGroupNode({
                    role: 'overlay',
                    semantic: { componentKind: 'test', role: 'overlay' },
                    direction: 'overlay',
                    children: [connectorNode]
                })
            ]
        });

        const layout = buildSceneLayout(scene);
        const renderer = new CanvasSceneRenderer({ canvas });
        renderer.setScene(scene, layout);

        expect(renderer.render({ width: 640, height: 360, dpr: 1 })).toBe(true);

        const connectorStrokeIndex = ctx.operations.findIndex((entry) => entry.type === 'stroke');
        const connectorPathOps = ctx.operations.slice(Math.max(0, connectorStrokeIndex - 3), connectorStrokeIndex)
            .filter((entry) => entry.type === 'moveTo' || entry.type === 'lineTo');
        const connectorStroke = ctx.operations[connectorStrokeIndex];

        expect(connectorPathOps.length).toBeGreaterThanOrEqual(2);
        connectorPathOps.forEach((entry) => {
            expect(entry.x * 2).toBeCloseTo(Math.round(entry.x * 2), 6);
            expect(entry.y * 2).toBeCloseTo(Math.round(entry.y * 2), 6);
        });
        expect(connectorStroke.globalAlpha).toBe(1);
        expect(String(connectorStroke.strokeStyle || '')).toMatch(/^rgb\(/);
    });

    it('keeps the residual add plus sign visible at very low zoom', () => {
        const scene = buildTransformerSceneModel({
            activationSource: createActivationSource(4),
            layerCount: 1
        });
        const layout = buildSceneLayout(scene);
        const renderer = new CanvasSceneRenderer({ canvas });
        renderer.setScene(scene, layout);

        const scale = 0.05;
        const offsetX = (640 - (layout.sceneBounds.width * scale)) / 2;
        const offsetY = (360 - (layout.sceneBounds.height * scale)) / 2;

        expect(renderer.render({
            width: 640,
            height: 360,
            dpr: 1,
            viewportTransform: {
                source: 'test-low-zoom',
                scale,
                offsetX,
                offsetY
            }
        })).toBe(true);

        expect(ctx.operations.some((entry) => entry.type === 'fillText' && entry.text === '+')).toBe(true);
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
