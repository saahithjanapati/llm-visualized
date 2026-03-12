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
    createMatrixNode,
    createSceneModel,
    createTextNode,
    VIEW2D_ANCHOR_SIDES,
    VIEW2D_CONNECTOR_ROUTES
} from '../src/view2d/schema/sceneTypes.js';
import { flattenSceneNodes } from '../src/view2d/schema/sceneTypes.js';
import { VIEW2D_MATRIX_PRESENTATIONS } from '../src/view2d/schema/sceneTypes.js';
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
        quadraticCurveTo(cpx, cpy, x, y) {
            operations.push({ type: 'quadraticCurveTo', cpx, cpy, x, y });
        },
        rect(x, y, width, height) {
            operations.push({ type: 'rect', x, y, width, height });
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
        fillRect(x, y, width, height) {
            operations.push({
                type: 'fillRect',
                x,
                y,
                width,
                height,
                fillStyle: this.fillStyle,
                globalAlpha: this.globalAlpha
            });
        },
        clearRect() {},
        save() {},
        restore() {},
        translate() {},
        scale() {},
        setTransform() {},
        fillText(text, x, y) {
            operations.push({ type: 'fillText', text, x, y });
        },
        measureText(text = '') {
            return {
                width: String(text).length * 7
            };
        },
        createLinearGradient() {
            operations.push({ type: 'createLinearGradient' });
            return {
                addColorStop() {}
            };
        },
        createRadialGradient() {
            operations.push({ type: 'createRadialGradient' });
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

    it('allows a lower per-render DPR cap for interaction frames', () => {
        const resolution = syncCanvasResolution(canvas, {
            width: 320,
            height: 180,
            dprCap: 1
        });

        expect(resolution.dpr).toBe(1);
        expect(canvas.width).toBe(320);
        expect(canvas.height).toBe(180);
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

    it('renders a selected MHSA head with a black interior only in deepest head mode', () => {
        const scene = buildTransformerSceneModel({
            activationSource: createActivationSource(4),
            layerCount: 1,
            headDetailTarget: {
                layerIndex: 0,
                headIndex: 3
            }
        });
        const layout = buildSceneLayout(scene);
        const renderer = new CanvasSceneRenderer({ canvas });
        const detailEntry = layout.registry.getNodeEntries().find((entry) => (
            entry.role === 'head-card'
            && entry.semantic?.componentKind === 'mhsa'
            && entry.semantic?.layerIndex === 0
            && entry.semantic?.headIndex === 3
        ));
        const viewportPadding = 28;
        const scale = Math.min(
            (640 - (viewportPadding * 2)) / Math.max(1, detailEntry.bounds.width),
            (360 - (viewportPadding * 2)) / Math.max(1, detailEntry.bounds.height)
        );
        const offsetX = ((640 - (detailEntry.bounds.width * scale)) / 2) - (detailEntry.bounds.x * scale);
        const offsetY = ((360 - (detailEntry.bounds.height * scale)) / 2) - (detailEntry.bounds.y * scale);
        renderer.setScene(scene, layout);

        expect(renderer.render({
            width: 640,
            height: 360,
            dpr: 1,
            headDetailDepthActive: true,
            viewportTransform: {
                source: 'test-selected-head-card',
                scale,
                offsetX,
                offsetY
            }
        })).toBe(true);

        expect(ctx.operations.some((entry) => (
            entry.type === 'fillRect'
            && entry.fillStyle === '#000'
        ))).toBe(true);
        expect(ctx.operations.some((entry) => (
            entry.type === 'stroke'
            && entry.strokeStyle === 'rgb(255, 255, 255)'
        ))).toBe(true);
        expect(ctx.operations.filter((entry) => entry.type === 'createLinearGradient').length).toBeGreaterThan(0);
        expect(ctx.operations.filter((entry) => entry.type === 'fillRect').length).toBeGreaterThan(10);
        expect(ctx.operations.some((entry) => entry.type === 'stroke')).toBe(true);
        expect(renderer.getLastRenderState()?.viewportSource).toBe('test-selected-head-card');
        expect(renderer.getLastRenderState()?.headDetailBounds).toBeTruthy();
        expect(renderer.getLastRenderState()?.headDetailDepthActive).toBe(true);
    });

    it('renders the deepest concatenate detail stage with a framed concat target and inbound arrows', () => {
        const scene = buildTransformerSceneModel({
            activationSource: createActivationSource(4),
            layerCount: 1,
            concatDetailTarget: {
                layerIndex: 0
            }
        });
        const layout = buildSceneLayout(scene);
        const renderer = new CanvasSceneRenderer({ canvas });
        const detailEntry = layout.registry.getNodeEntries().find((entry) => (
            entry.role === 'concat-card'
            && entry.semantic?.componentKind === 'mhsa'
            && entry.semantic?.layerIndex === 0
            && entry.semantic?.stage === 'concatenate'
        ));
        const viewportPadding = 28;
        const scale = Math.min(
            (640 - (viewportPadding * 2)) / Math.max(1, detailEntry.bounds.width),
            (360 - (viewportPadding * 2)) / Math.max(1, detailEntry.bounds.height)
        );
        const offsetX = ((640 - (detailEntry.bounds.width * scale)) / 2) - (detailEntry.bounds.x * scale);
        const offsetY = ((360 - (detailEntry.bounds.height * scale)) / 2) - (detailEntry.bounds.y * scale);
        renderer.setScene(scene, layout);

        expect(renderer.render({
            width: 640,
            height: 360,
            dpr: 1,
            headDetailDepthActive: true,
            viewportTransform: {
                source: 'test-selected-concat-card',
                scale,
                offsetX,
                offsetY
            }
        })).toBe(true);

        expect(ctx.operations.some((entry) => (
            entry.type === 'fillRect'
            && entry.fillStyle === '#000'
        ))).toBe(true);
        expect(ctx.operations.filter((entry) => entry.type === 'lineTo').length).toBeGreaterThanOrEqual(24);
        expect(ctx.operations.filter((entry) => entry.type === 'stroke').length).toBeGreaterThanOrEqual(13);
        expect(renderer.getLastRenderState()?.detailTargetKind).toBe('concatenate');
        expect(renderer.getLastRenderState()?.concatDetailBounds).toBeTruthy();
        expect(renderer.getLastRenderState()?.headDetailDepthActive).toBe(true);
    });

    it('renders the deepest output projection detail stage with a magenta frame and one inbound arrow', () => {
        const scene = buildTransformerSceneModel({
            activationSource: createActivationSource(4),
            layerCount: 1,
            outputProjectionDetailTarget: {
                layerIndex: 0
            }
        });
        const layout = buildSceneLayout(scene);
        const renderer = new CanvasSceneRenderer({ canvas });
        const detailEntry = layout.registry.getNodeEntries().find((entry) => (
            entry.role === 'projection-weight'
            && entry.semantic?.componentKind === 'output-projection'
            && entry.semantic?.layerIndex === 0
            && entry.semantic?.stage === 'attn-out'
        ));
        const viewportPadding = 28;
        const scale = Math.min(
            (640 - (viewportPadding * 2)) / Math.max(1, detailEntry.bounds.width),
            (360 - (viewportPadding * 2)) / Math.max(1, detailEntry.bounds.height)
        );
        const offsetX = ((640 - (detailEntry.bounds.width * scale)) / 2) - (detailEntry.bounds.x * scale);
        const offsetY = ((360 - (detailEntry.bounds.height * scale)) / 2) - (detailEntry.bounds.y * scale);
        renderer.setScene(scene, layout);

        expect(renderer.render({
            width: 640,
            height: 360,
            dpr: 1,
            headDetailDepthActive: true,
            viewportTransform: {
                source: 'test-selected-output-projection',
                scale,
                offsetX,
                offsetY
            }
        })).toBe(true);

        expect(ctx.operations.some((entry) => (
            entry.type === 'fillRect'
            && entry.fillStyle === '#000'
        ))).toBe(true);
        expect(ctx.operations.filter((entry) => entry.type === 'lineTo').length).toBeGreaterThanOrEqual(1);
        expect(ctx.operations.some((entry) => (
            entry.type === 'stroke'
            && entry.strokeStyle === resolveView2dStyle(VIEW2D_STYLE_KEYS.OUTPUT_PROJECTION)?.stroke
        ))).toBe(true);
        expect(renderer.getLastRenderState()?.detailTargetKind).toBe('output-projection');
        expect(renderer.getLastRenderState()?.outputProjectionDetailBounds).toBeTruthy();
        expect(renderer.getLastRenderState()?.headDetailDepthActive).toBe(true);
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

    it('uses the interaction fast path while zooming and restores full matrix detail after settle', () => {
        const matrixNode = createMatrixNode({
            role: 'matrix',
            semantic: { componentKind: 'test-matrix', role: 'matrix' },
            presentation: VIEW2D_MATRIX_PRESENTATIONS.BANDED_ROWS,
            dimensions: { rows: 2, cols: 64 },
            label: { text: 'Test matrix', tex: 'Test matrix' },
            rowItems: [
                { rowIndex: 0, label: 'Row A', gradientCss: 'rgba(255,0,0,0.7)' },
                { rowIndex: 1, label: 'Row B', gradientCss: 'rgba(0,255,0,0.7)' }
            ]
        });
        const scene = createSceneModel({
            semantic: { componentKind: 'test-scene' },
            nodes: [matrixNode]
        });
        const layout = buildSceneLayout(scene);
        const renderer = new CanvasSceneRenderer({ canvas });
        const lowScale = 0.14;
        const highScale = 1.1;
        const highOffsetX = (640 - (layout.sceneBounds.width * highScale)) / 2;
        const highOffsetY = (360 - (layout.sceneBounds.height * highScale)) / 2;

        renderer.setScene(scene, layout);

        expect(renderer.render({
            width: 640,
            height: 360,
            dpr: 1,
            interacting: false,
            viewportTransform: {
                source: 'settled-low-scale',
                scale: lowScale,
                offsetX: (640 - (layout.sceneBounds.width * lowScale)) / 2,
                offsetY: (360 - (layout.sceneBounds.height * lowScale)) / 2
            }
        })).toBe(true);
        expect(renderer.getLastRenderState()?.detailScale).toBeCloseTo(lowScale, 6);

        ctx.operations.length = 0;
        expect(renderer.render({
            width: 640,
            height: 360,
            dpr: 1,
            interacting: true,
            viewportTransform: {
                source: 'active-high-scale',
                scale: highScale,
                offsetX: highOffsetX,
                offsetY: highOffsetY
            }
        })).toBe(true);

        const activeRowLabels = ctx.operations
            .filter((entry) => entry.type === 'fillText')
            .map((entry) => entry.text)
            .filter((text) => text === 'Row A' || text === 'Row B');
        expect(renderer.getLastRenderState()?.worldScale).toBeCloseTo(highScale, 6);
        expect(renderer.getLastRenderState()?.detailScale).toBeCloseTo(highScale, 6);
        expect(renderer.getLastRenderState()?.interactionFastPath).toBe(true);
        expect(activeRowLabels).toHaveLength(0);

        ctx.operations.length = 0;
        expect(renderer.render({
            width: 640,
            height: 360,
            dpr: 1,
            interacting: false,
            viewportTransform: {
                source: 'settled-high-scale',
                scale: highScale,
                offsetX: highOffsetX,
                offsetY: highOffsetY
            }
        })).toBe(true);

        const settledRowLabels = ctx.operations
            .filter((entry) => entry.type === 'fillText')
            .map((entry) => entry.text)
            .filter((text) => text === 'Row A' || text === 'Row B');
        expect(renderer.getLastRenderState()?.detailScale).toBeCloseTo(highScale, 6);
        expect(settledRowLabels).toEqual(['Row A', 'Row B']);
    });

    it('tracks live interaction scale while the fast path is active', () => {
        const matrixNode = createMatrixNode({
            role: 'matrix',
            semantic: { componentKind: 'test-matrix', role: 'matrix' },
            presentation: VIEW2D_MATRIX_PRESENTATIONS.BANDED_ROWS,
            dimensions: { rows: 2, cols: 64 },
            label: { text: 'Test matrix', tex: 'Test matrix' },
            rowItems: [
                { rowIndex: 0, label: 'Row A', gradientCss: 'rgba(255,0,0,0.7)' },
                { rowIndex: 1, label: 'Row B', gradientCss: 'rgba(0,255,0,0.7)' }
            ]
        });
        const scene = createSceneModel({
            semantic: { componentKind: 'test-scene' },
            nodes: [matrixNode]
        });
        const layout = buildSceneLayout(scene);
        const renderer = new CanvasSceneRenderer({ canvas });
        const settledScale = 1.1;
        const slightZoomOutScale = 1.02;
        const releaseScale = 0.94;

        renderer.setScene(scene, layout);

        expect(renderer.render({
            width: 640,
            height: 360,
            dpr: 1,
            interacting: false,
            viewportTransform: {
                source: 'settled-high-scale',
                scale: settledScale,
                offsetX: (640 - (layout.sceneBounds.width * settledScale)) / 2,
                offsetY: (360 - (layout.sceneBounds.height * settledScale)) / 2
            }
        })).toBe(true);
        expect(renderer.getLastRenderState()?.detailScale).toBeCloseTo(settledScale, 6);

        expect(renderer.render({
            width: 640,
            height: 360,
            dpr: 1,
            interacting: true,
            viewportTransform: {
                source: 'active-slight-zoom-out',
                scale: slightZoomOutScale,
                offsetX: (640 - (layout.sceneBounds.width * slightZoomOutScale)) / 2,
                offsetY: (360 - (layout.sceneBounds.height * slightZoomOutScale)) / 2
            }
        })).toBe(true);
        expect(renderer.getLastRenderState()?.interactionFastPath).toBe(true);
        expect(renderer.getLastRenderState()?.detailScale).toBeCloseTo(slightZoomOutScale, 6);

        expect(renderer.render({
            width: 640,
            height: 360,
            dpr: 1,
            interacting: true,
            viewportTransform: {
                source: 'active-release-zoom-out',
                scale: releaseScale,
                offsetX: (640 - (layout.sceneBounds.width * releaseScale)) / 2,
                offsetY: (360 - (layout.sceneBounds.height * releaseScale)) / 2
            }
        })).toBe(true);
        expect(renderer.getLastRenderState()?.detailScale).toBeCloseTo(releaseScale, 6);
    });

    it('skips expensive card surface effects while the viewport is interacting', () => {
        const cardNode = createMatrixNode({
            role: 'projection-weight',
            semantic: { componentKind: 'test-card', role: 'projection-weight' },
            presentation: VIEW2D_MATRIX_PRESENTATIONS.CARD,
            dimensions: { rows: 64, cols: 64 },
            label: { text: 'Card', tex: 'Card' },
            visual: { styleKey: VIEW2D_STYLE_KEYS.MHSA_Q }
        });
        const scene = createSceneModel({
            semantic: { componentKind: 'test-scene' },
            nodes: [cardNode]
        });
        const layout = buildSceneLayout(scene);
        const renderer = new CanvasSceneRenderer({ canvas });
        const scale = 1.15;
        const offsetX = (640 - (layout.sceneBounds.width * scale)) / 2;
        const offsetY = (360 - (layout.sceneBounds.height * scale)) / 2;

        renderer.setScene(scene, layout);

        ctx.operations.length = 0;
        expect(renderer.render({
            width: 640,
            height: 360,
            dpr: 1,
            interacting: false,
            viewportTransform: {
                source: 'settled-card',
                scale,
                offsetX,
                offsetY
            }
        })).toBe(true);
        const settledRadialGradientCount = ctx.operations
            .filter((entry) => entry.type === 'createRadialGradient')
            .length;
        expect(settledRadialGradientCount).toBeGreaterThan(0);

        ctx.operations.length = 0;
        expect(renderer.render({
            width: 640,
            height: 360,
            dpr: 1,
            interacting: true,
            viewportTransform: {
                source: 'interacting-card',
                scale,
                offsetX,
                offsetY
            }
        })).toBe(true);
        const interactingRadialGradientCount = ctx.operations
            .filter((entry) => entry.type === 'createRadialGradient')
            .length;
        expect(interactingRadialGradientCount).toBe(0);
    });

    it('resolves residual row hits and fades non-hovered rows in compact residual strips', () => {
        const scene = buildTransformerSceneModel({
            activationSource: createActivationSource(4),
            layerCount: 1
        });
        const layout = buildSceneLayout(scene);
        const renderer = new CanvasSceneRenderer({ canvas });
        const residualEntry = layout.registry.getNodeEntries().find((entry) => (
            entry.role === 'module-card'
            && entry.semantic?.componentKind === 'residual'
            && entry.semantic?.stage === 'incoming'
        ));
        const rowPoint = {
            x: residualEntry.contentBounds.x + residualEntry.layoutData.innerPaddingX + 8,
            y: residualEntry.contentBounds.y + residualEntry.layoutData.innerPaddingY + (residualEntry.layoutData.rowHeight * 0.5)
        };

        renderer.setScene(scene, layout);
        expect(renderer.render({ width: 640, height: 360, dpr: 1 })).toBe(true);

        const hit = renderer.resolveInteractiveHitAtPoint(rowPoint.x, rowPoint.y);
        expect(hit?.node?.semantic?.componentKind).toBe('residual');
        expect(hit?.rowHit?.rowIndex).toBe(0);
        expect(hit?.rowHit?.rowItem?.semantic?.tokenIndex).toBe(0);

        ctx.operations.length = 0;
        expect(renderer.render({
            width: 640,
            height: 360,
            dpr: 1,
            interactionState: {
                hoveredRow: {
                    nodeId: residualEntry.nodeId,
                    rowIndex: 0
                }
            }
        })).toBe(true);

        expect(ctx.operations.some((entry) => (
            entry.type === 'fillRect'
            && entry.globalAlpha === 0.18
        ))).toBe(true);
    });

    it('only shows residual stream captions after zooming in past the residual caption threshold', () => {
        const scene = buildTransformerSceneModel({
            activationSource: createActivationSource(4),
            layerCount: 1
        });
        const layout = buildSceneLayout(scene);
        const renderer = new CanvasSceneRenderer({ canvas });
        const residualEntry = layout.registry.getNodeEntries().find((entry) => (
            entry.role === 'module-card'
            && entry.semantic?.componentKind === 'residual'
            && entry.semantic?.stage === 'incoming'
        ));

        renderer.setScene(scene, layout);

        expect(renderer.render({
            width: 640,
            height: 360,
            dpr: 1,
            viewportTransform: {
                source: 'residual-caption-medium-zoom',
                scale: 1.4,
                offsetX: -residualEntry.bounds.x * 1.4,
                offsetY: -residualEntry.bounds.y * 1.4
            }
        })).toBe(true);

        expect(ctx.operations.some((entry) => (
            entry.type === 'fillText'
            && (entry.text === 'X' || entry.text === '4 × 768')
        ))).toBe(false);

        ctx.operations.length = 0;
        expect(renderer.render({
            width: 640,
            height: 360,
            dpr: 1,
            viewportTransform: {
                source: 'residual-caption-high-zoom',
                scale: 3.2,
                offsetX: -residualEntry.bounds.x * 3.2,
                offsetY: -residualEntry.bounds.y * 3.2
            }
        })).toBe(true);

        expect(ctx.operations.some((entry) => (
            entry.type === 'fillText'
            && entry.text === 'X'
        ))).toBe(true);
        expect(ctx.operations.some((entry) => (
            entry.type === 'fillText'
            && entry.text === '4 × 768'
        ))).toBe(true);
    });
});
