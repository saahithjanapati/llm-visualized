// @vitest-environment jsdom

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import {
    createTransformerView2dDetailView,
    describeTransformerView2dTarget,
    resolveTransformerView2dActionContext,
    resolveTransformerView2dRoute,
    syncTransformerView2dRoute
} from '../src/ui/selectionPanelTransformerView2d.js';
import { buildSceneLayout } from '../src/view2d/layout/buildSceneLayout.js';
import { buildTransformerSceneModel } from '../src/view2d/model/buildTransformerSceneModel.js';
import { CanvasSceneRenderer } from '../src/view2d/render/canvas/CanvasSceneRenderer.js';
import { resolveViewportFitTransform } from '../src/view2d/runtime/View2dViewportController.js';
import { flattenSceneNodes } from '../src/view2d/schema/sceneTypes.js';
import { D_HEAD, D_MODEL } from '../src/ui/selectionPanelConstants.js';

function createActivationSource(tokenCount = 4) {
    const promptTokens = Array.from({ length: tokenCount }, (_, index) => index);
    const tokenDisplayStrings = promptTokens.map((index) => `tok_${index}`);

    const buildVector = (length, seed = 1, scale = 0.1) => (
        Array.from({ length }, (_, index) => (((index % 17) - 8) * scale * seed))
    );

    return {
        meta: {
            prompt_tokens: promptTokens,
            token_display_strings: tokenDisplayStrings
        },
        getTokenCount() {
            return tokenCount;
        },
        getTokenString(tokenIndex) {
            return tokenDisplayStrings[tokenIndex] || `tok_${tokenIndex}`;
        },
        getEmbedding(kind, tokenIndex, targetLength = D_MODEL) {
            const scale = kind === 'position' ? 0.08 : (kind === 'sum' ? 0.12 : 0.1);
            return buildVector(targetLength, tokenIndex + 1, scale);
        },
        getLayerIncoming(layerIndex, tokenIndex, targetLength = D_MODEL) {
            return buildVector(targetLength, (layerIndex + 1) * (tokenIndex + 1), 0.015);
        },
        getLayerLn1(layerIndex, stage, tokenIndex, targetLength = D_MODEL) {
            return buildVector(targetLength, (layerIndex + 1) + tokenIndex + 1, 0.018);
        },
        getLayerLn2(layerIndex, stage, tokenIndex, targetLength = D_MODEL) {
            return buildVector(targetLength, (layerIndex + 2) + tokenIndex + 1, 0.02);
        },
        getLayerQKVVector(layerIndex, kind, headIndex, tokenIndex, targetLength = D_HEAD) {
            const scale = kind === 'q' ? 0.08 : 0.1;
            return buildVector(targetLength, (layerIndex + 1) * (headIndex + 1) * (tokenIndex + 1), scale);
        },
        getAttentionWeightedSum(layerIndex, headIndex, tokenIndex, targetLength = D_HEAD) {
            return buildVector(targetLength, (layerIndex + 1) * (headIndex + 1) * (tokenIndex + 1), 0.09);
        },
        getAttentionOutputProjection(layerIndex, tokenIndex, targetLength = D_MODEL) {
            return buildVector(targetLength, (layerIndex + 1) * (tokenIndex + 2), 0.022);
        },
        getPostAttentionResidual(layerIndex, tokenIndex, targetLength = D_MODEL) {
            return buildVector(targetLength, (layerIndex + 2) * (tokenIndex + 1), 0.024);
        },
        getMlpUp(layerIndex, tokenIndex, targetLength = D_MODEL * 4) {
            return buildVector(targetLength, (layerIndex + 3) * (tokenIndex + 1), 0.014);
        },
        getMlpActivation(layerIndex, tokenIndex, targetLength = D_MODEL * 4) {
            return buildVector(targetLength, (layerIndex + 4) * (tokenIndex + 1), 0.016);
        },
        getMlpDown(layerIndex, tokenIndex, targetLength = D_MODEL) {
            return buildVector(targetLength, (layerIndex + 5) * (tokenIndex + 1), 0.021);
        },
        getPostMlpResidual(layerIndex, tokenIndex, targetLength = D_MODEL) {
            return buildVector(targetLength, (layerIndex + 6) * (tokenIndex + 1), 0.026);
        },
        getFinalLayerNorm(stage, tokenIndex, targetLength = D_MODEL) {
            return buildVector(targetLength, tokenIndex + 1, 0.03);
        },
        getLogitsForToken(tokenIndex, limit = 8) {
            return Array.from({ length: limit }, (_, index) => ({
                token: `cand_${index}`,
                prob: Math.max(0, 0.9 - (index * 0.08))
            }));
        }
    };
}

function createMockContext() {
    return {
        beginPath() {},
        moveTo() {},
        lineTo() {},
        quadraticCurveTo() {},
        arcTo() {},
        closePath() {},
        fill() {},
        stroke() {},
        clip() {},
        fillRect() {},
        clearRect() {},
        save() {},
        restore() {},
        translate() {},
        scale() {},
        setTransform() {},
        fillText() {},
        measureText(text = '') {
            return {
                width: String(text).length * 7
            };
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

function dispatchPointerEvent(target, type, {
    clientX = 0,
    clientY = 0,
    pointerId = 1,
    pointerType = 'mouse',
    button = 0
} = {}) {
    const event = new Event(type, { bubbles: true, cancelable: true });
    Object.defineProperties(event, {
        clientX: { value: clientX },
        clientY: { value: clientY },
        pointerId: { value: pointerId },
        pointerType: { value: pointerType },
        button: { value: button }
    });
    target.dispatchEvent(event);
    return event;
}

function resolveEntryClientPoint(entry, viewport, {
    xInset = 0.5,
    yInset = 0.5
} = {}) {
    const bounds = entry?.contentBounds || entry?.bounds || null;
    return {
        clientX: viewport.panX + (((bounds?.x || 0) + ((bounds?.width || 0) * xInset)) * viewport.scale),
        clientY: viewport.panY + (((bounds?.y || 0) + ((bounds?.height || 0) * yInset)) * viewport.scale)
    };
}

function resolveRowClientPoint(entry, viewport, rowIndex = 0, {
    xInset = 0.5
} = {}) {
    const contentBounds = entry?.contentBounds || entry?.bounds || null;
    const innerPaddingX = Number(entry?.layoutData?.innerPaddingX) || 0;
    const innerPaddingY = Number(entry?.layoutData?.innerPaddingY) || 0;
    const rowGap = Number(entry?.layoutData?.rowGap) || 0;
    const contentWidth = Math.max(1, (contentBounds?.width || 0) - (innerPaddingX * 2));
    const rowWidth = Math.max(
        1,
        Math.min(Number(entry?.layoutData?.compactWidth) || contentWidth, contentWidth)
    );
    const rowHeight = Math.max(1, Number(entry?.layoutData?.rowHeight) || (contentBounds?.height || 0));
    const worldX = (contentBounds?.x || 0) + innerPaddingX + (rowWidth * xInset);
    const worldY = (contentBounds?.y || 0)
        + innerPaddingY
        + (Math.max(0, rowIndex) * (rowHeight + rowGap))
        + (rowHeight * 0.5);
    return {
        clientX: viewport.panX + (worldX * viewport.scale),
        clientY: viewport.panY + (worldY * viewport.scale)
    };
}

function resolveInteractiveRowClientPoint(scene, layout, viewport, entry, rowIndex = 0) {
    const bounds = entry?.contentBounds || entry?.bounds || null;
    const innerPaddingY = Number(entry?.layoutData?.innerPaddingY) || 0;
    const rowHeight = Math.max(1, Number(entry?.layoutData?.rowHeight) || (bounds?.height || 0));
    const rowGap = Number(entry?.layoutData?.rowGap) || 0;
    const rowTop = (bounds?.y || 0) + innerPaddingY + (Math.max(0, rowIndex) * (rowHeight + rowGap));
    const renderer = new CanvasSceneRenderer({
        canvas: {
            width: 640,
            height: 360,
            getContext: () => createMockContext()
        }
    });
    renderer.setScene(scene, layout);

    for (let yStep = 1; yStep <= 6; yStep += 1) {
        const worldY = rowTop + ((rowHeight * yStep) / 7);
        for (let xStep = 1; xStep <= 10; xStep += 1) {
            const worldX = (bounds?.x || 0) + (((bounds?.width || 0) * xStep) / 11);
            const hit = renderer.resolveInteractiveHitAtPoint(worldX, worldY);
            if (hit?.node?.id === entry?.id && hit?.rowHit?.rowIndex === rowIndex) {
                return {
                    clientX: viewport.panX + (worldX * viewport.scale),
                    clientY: viewport.panY + (worldY * viewport.scale)
                };
            }
        }
    }

    return resolveRowClientPoint(entry, viewport, rowIndex);
}

function drainRafQueue(queue, now = performance.now()) {
    while (queue.length) {
        const callbacks = queue.splice(0);
        callbacks.forEach((callback) => callback?.(now));
    }
}

function resolveSceneRenderableBounds(scene, layout) {
    const registry = layout?.registry || null;
    return flattenSceneNodes(scene)
        .flatMap((node) => {
            if (node.kind === 'group' || node?.metadata?.hidden) {
                return [];
            }
            if (node.kind === 'connector') {
                return [registry?.getConnectorEntry(node.id)?.bounds || null];
            }
            return [registry?.getNodeEntry(node.id)?.bounds || null];
        })
        .filter(Boolean)
        .reduce((acc, bounds) => {
            if (!acc) {
                return { ...bounds };
            }
            const minX = Math.min(acc.x, bounds.x);
            const minY = Math.min(acc.y, bounds.y);
            const maxX = Math.max(acc.x + acc.width, bounds.x + bounds.width);
            const maxY = Math.max(acc.y + acc.height, bounds.y + bounds.height);
            return {
                x: minX,
                y: minY,
                width: maxX - minX,
                height: maxY - minY
            };
        }, null);
}

describe('selectionPanelTransformerView2d', () => {
    let rafQueue;
    let canvasGetContextSpy;

    beforeEach(() => {
        rafQueue = [];
        vi.stubGlobal('requestAnimationFrame', vi.fn((callback) => {
            rafQueue.push(callback);
            return rafQueue.length;
        }));
        vi.stubGlobal('cancelAnimationFrame', vi.fn((id) => {
            if (id > 0 && id <= rafQueue.length) {
                rafQueue[id - 1] = null;
            }
        }));
        canvasGetContextSpy = vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockImplementation(() => createMockContext());
    });

    afterEach(() => {
        canvasGetContextSpy?.mockRestore();
        vi.unstubAllGlobals();
        vi.restoreAllMocks();
        document.body.innerHTML = '';
    });

    it('maps embedding selections into embedding-focused 2D targets', () => {
        const context = resolveTransformerView2dActionContext({
            label: 'Token embedding',
            info: {
                activationData: {
                    stage: 'embedding.token'
                }
            }
        });

        expect(context?.semanticTarget).toEqual({
            componentKind: 'embedding',
            stage: 'token',
            role: 'token-embedding'
        });
        expect(context?.focusLabel).toBe('Token embeddings');
    });

    it('maps numbered layer norm selections by activation stage', () => {
        const context = resolveTransformerView2dActionContext({
            label: 'LayerNorm shift',
            info: {
                layerIndex: 5,
                activationData: {
                    layerIndex: 5,
                    stage: 'ln2.shift'
                }
            }
        });

        expect(context?.semanticTarget).toEqual({
            componentKind: 'layer-norm',
            layerIndex: 5,
            stage: 'ln2',
            role: 'module'
        });
        expect(context?.focusLabel).toBe('Layer 6 LayerNorm 2');
    });

    it('focuses MHSA heads when attention selections carry a head index', () => {
        const context = resolveTransformerView2dActionContext({
            label: 'Query vector',
            info: {
                layerIndex: 3,
                headIndex: 8,
                activationData: {
                    layerIndex: 3,
                    headIndex: 8,
                    stage: 'qkv.q'
                }
            }
        });

        expect(context?.semanticTarget).toEqual({
            componentKind: 'mhsa',
            layerIndex: 3,
            headIndex: 8,
            stage: 'attention',
            role: 'head'
        });
        expect(context?.focusLabel).toBe('Layer 4 Attention Head 9');
    });

    it('maps concatenate selections into concat-focused 2D targets', () => {
        const context = resolveTransformerView2dActionContext({
            label: 'Concatenate',
            info: {
                layerIndex: 2,
                activationData: {
                    layerIndex: 2,
                    stage: 'attention.concatenate'
                }
            }
        });

        expect(context?.semanticTarget).toEqual({
            componentKind: 'mhsa',
            layerIndex: 2,
            stage: 'concatenate',
            role: 'concat'
        });
        expect(context?.focusLabel).toBe('Layer 3 Concatenate Heads');
    });

    it('maps output projection matrix selections to the projection weight target', () => {
        const context = resolveTransformerView2dActionContext({
            label: 'Output Projection Matrix',
            info: {
                layerIndex: 2,
                activationData: {
                    layerIndex: 2
                }
            }
        });

        expect(context?.semanticTarget).toEqual({
            componentKind: 'output-projection',
            layerIndex: 2,
            stage: 'attn-out',
            role: 'projection-weight'
        });
        expect(context?.focusLabel).toBe('Layer 3 Output Projection');
    });

    it('maps alpha projection matrix selections to the output projection target', () => {
        const context = resolveTransformerView2dActionContext({
            label: 'Alpha Projection Matrix',
            info: {
                layerIndex: 1,
                activationData: {
                    layerIndex: 1
                }
            }
        });

        expect(context?.semanticTarget).toEqual({
            componentKind: 'output-projection',
            layerIndex: 1,
            stage: 'attn-out',
            role: 'projection-weight'
        });
        expect(context?.focusLabel).toBe('Layer 2 Output Projection');
    });

    it('maps MLP projection selections to the correct stage', () => {
        const context = resolveTransformerView2dActionContext({
            label: 'MLP Up Weight Matrix',
            info: {
                layerIndex: 7,
                activationData: {
                    layerIndex: 7
                }
            }
        });

        expect(context?.semanticTarget).toEqual({
            componentKind: 'mlp',
            layerIndex: 7,
            stage: 'mlp-up',
            role: 'mlp-up'
        });
        expect(context?.focusLabel).toBe('Layer 8 Multilayer Perceptron Up Projection');
    });

    it('maps logits selections and top unembedding labels into output-space targets', () => {
        const logitContext = resolveTransformerView2dActionContext({
            label: 'Logit',
            kind: 'logitBar'
        });
        const unembeddingContext = resolveTransformerView2dActionContext({
            label: 'Vocabulary Embedding (Top)'
        });

        expect(logitContext?.semanticTarget).toEqual({
            componentKind: 'logits',
            stage: 'output',
            role: 'logits-topk'
        });
        expect(unembeddingContext?.semanticTarget).toEqual({
            componentKind: 'logits',
            stage: 'output',
            role: 'unembedding'
        });
    });

    it('parses direct 2D canvas routes from URL parameters', () => {
        const route = resolveTransformerView2dRoute(
            'https://example.test/?view=2d&component=mhsa&layer=3&head=2&stage=attention&role=head'
        );

        expect(route).toEqual({
            semanticTarget: {
                componentKind: 'mhsa',
                layerIndex: 3,
                headIndex: 2,
                stage: 'attention',
                role: 'head'
            }
        });
    });

    it('syncs direct 2D canvas routes onto the current URL', () => {
        window.history.replaceState({}, '', '/?foo=bar#2d');

        syncTransformerView2dRoute({
            active: true,
            semanticTarget: {
                componentKind: 'mlp',
                layerIndex: 7,
                stage: 'mlp-up',
                role: 'mlp-up'
            }
        });

        expect(window.location.search).toBe('?foo=bar&view=2d&component=mlp&layer=7&stage=mlp-up&role=mlp-up');
        expect(window.location.hash).toBe('');

        syncTransformerView2dRoute({ active: false });

        expect(window.location.search).toBe('?foo=bar');
        expect(window.location.hash).toBe('');
    });

    it('describes residual and final layer norm targets with human-readable labels', () => {
        expect(describeTransformerView2dTarget({
            componentKind: 'residual',
            layerIndex: 4,
            stage: 'post-attn-add'
        })).toBe('Layer 5 post-attention residual');

        expect(describeTransformerView2dTarget({
            componentKind: 'layer-norm',
            stage: 'final-ln',
            role: 'module'
        })).toBe('Final LayerNorm');
    });

    it('re-fits the overview scene when the opening canvas grows to a larger screen size', () => {
        const panel = document.createElement('section');
        panel.innerHTML = `
            <div class="detail-header"></div>
            <div class="detail-body"></div>
        `;
        document.body.appendChild(panel);

        const view = createTransformerView2dDetailView(panel);
        const canvas = panel.querySelector('.detail-transformer-view2d-canvas');
        const ctx = createMockContext();
        const activationSource = createActivationSource(4);
        let canvasRect = {
            width: 640,
            height: 360
        };

        canvas.getContext = vi.fn(() => ctx);
        canvas.getBoundingClientRect = () => ({
            left: 0,
            top: 0,
            right: canvasRect.width,
            bottom: canvasRect.height,
            width: canvasRect.width,
            height: canvasRect.height
        });

        view.setVisible(true);
        view.open({ activationSource });
        const initialViewport = view.getViewportState();

        canvasRect = {
            width: 960,
            height: 540
        };
        view.resizeAndRender();

        const resizedViewport = view.getViewportState();
        const scene = buildTransformerSceneModel({ activationSource });
        const layout = buildSceneLayout(scene);
        const expectedFit = resolveViewportFitTransform(layout.sceneBounds, canvasRect, {
            padding: 28,
            minScale: 0.035,
            maxScale: 10
        });

        expect(resizedViewport.scale).toBeCloseTo(expectedFit.scale, 6);
        expect(resizedViewport.panX).toBeCloseTo(expectedFit.panX, 6);
        expect(resizedViewport.panY).toBeCloseTo(expectedFit.panY, 6);
        expect(resizedViewport.scale).toBeGreaterThan(initialViewport.scale);
    });

    it('renders prompt token chips in the bottom-left 2D overlay from the live token window', () => {
        const panel = document.createElement('section');
        panel.innerHTML = `
            <div class="detail-header"></div>
            <div class="detail-body"></div>
        `;
        document.body.appendChild(panel);

        const view = createTransformerView2dDetailView(panel);
        const canvas = panel.querySelector('.detail-transformer-view2d-canvas');
        const ctx = createMockContext();
        const activationSource = {
            ...createActivationSource(3),
            getTokenId(tokenIndex) {
                return 1000 + tokenIndex;
            }
        };

        canvas.getContext = vi.fn(() => ctx);
        canvas.getBoundingClientRect = () => ({
            left: 0,
            top: 0,
            right: 640,
            bottom: 360,
            width: 640,
            height: 360
        });

        view.setVisible(true);
        view.open({
            activationSource,
            tokenIndices: [0, 1, 2],
            tokenLabels: ['tok_0', 'tok_1', 'tok_2'],
            semanticTarget: {
                componentKind: 'residual',
                layerIndex: 0,
                stage: 'incoming',
                role: 'module',
                tokenIndex: 1
            },
            focusLabel: 'Layer 1 incoming residual'
        });

        const tokenStrip = panel.querySelector('.detail-transformer-view2d-token-strip');
        const tokenChips = Array.from(panel.querySelectorAll('.detail-transformer-view2d-token-strip__token'));

        expect(tokenStrip?.dataset.visible).toBe('true');
        expect(tokenChips).toHaveLength(3);
        expect(tokenChips.map((chip) => chip.textContent)).toEqual(['tok_0', 'tok_1', 'tok_2']);
        expect(tokenChips[1]?.dataset.tokenIndex).toBe('1');
        expect(tokenChips[1]?.dataset.tokenId).toBe('1001');
        expect(tokenChips[1]?.dataset.tokenNav).toBe('true');
    });

    it('supports keyboard pan and zoom on the transformer 2D canvas surface', () => {
        const panel = document.createElement('section');
        panel.innerHTML = `
            <div class="detail-header"></div>
            <div class="detail-body"></div>
        `;
        document.body.appendChild(panel);

        const view = createTransformerView2dDetailView(panel);
        const canvas = panel.querySelector('.detail-transformer-view2d-canvas');
        const canvasCard = panel.querySelector('.detail-transformer-view2d-canvas-card');
        const hud = panel.querySelector('.detail-transformer-view2d-hud');
        const ctx = createMockContext();

        canvas.getContext = vi.fn(() => ctx);
        canvas.getBoundingClientRect = () => ({
            left: 0,
            top: 0,
            right: 640,
            bottom: 360,
            width: 640,
            height: 360
        });

        view.setVisible(true);
        view.open({
            activationSource: createActivationSource(4),
            semanticTarget: {
                componentKind: 'mhsa',
                layerIndex: 0,
                stage: 'attention',
                role: 'module'
            },
            focusLabel: 'Layer 1 MHSA'
        });

        expect(document.activeElement).toBe(canvasCard);

        const before = view.getViewportState();
        canvasCard.dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowRight', bubbles: true }));
        const afterPan = view.getViewportState();
        canvasCard.dispatchEvent(new KeyboardEvent('keyup', { key: 'ArrowRight', bubbles: true }));

        expect(afterPan.panX).toBeLessThan(before.panX);

        canvasCard.dispatchEvent(new KeyboardEvent('keydown', { key: '=', bubbles: true }));
        const afterZoom = view.getViewportState();
        canvasCard.dispatchEvent(new KeyboardEvent('keyup', { key: '=', bubbles: true }));

        expect(afterZoom.scale).toBeGreaterThan(afterPan.scale);
    });

    it('only enters the head detail state for explicit head targets and keeps zooming inside that deep scene', () => {
        const panel = document.createElement('section');
        panel.innerHTML = `
            <div class="detail-header"></div>
            <div class="detail-body"></div>
        `;
        document.body.appendChild(panel);

        const view = createTransformerView2dDetailView(panel);
        const canvas = panel.querySelector('.detail-transformer-view2d-canvas');
        const canvasCard = panel.querySelector('.detail-transformer-view2d-canvas-card');
        const hud = panel.querySelector('.detail-transformer-view2d-hud');
        const ctx = createMockContext();

        canvas.getContext = vi.fn(() => ctx);
        canvas.getBoundingClientRect = () => ({
            left: 0,
            top: 0,
            right: 640,
            bottom: 360,
            width: 640,
            height: 360
        });

        view.setVisible(true);
        view.open({
            activationSource: createActivationSource(4),
            semanticTarget: {
                componentKind: 'mhsa',
                layerIndex: 0,
                stage: 'attention',
                role: 'module'
            },
            focusLabel: 'Layer 1 MHSA'
        });

        const moduleScale = view.getViewportState().scale;
        expect(panel.querySelector('[data-transformer-view2d-action="close-head-detail"]')).toBeNull();
        expect(canvasCard.classList.contains('is-head-detail-active')).toBe(false);
        expect(hud.hidden).toBe(false);

        view.open({
            activationSource: createActivationSource(4),
            semanticTarget: {
                componentKind: 'mhsa',
                layerIndex: 0,
                headIndex: 4,
                stage: 'attention',
                role: 'head'
            },
            focusLabel: 'Layer 1 MHSA Head 5'
        });

        const headScale = view.getViewportState().scale;
        expect(canvasCard.classList.contains('is-head-detail-active')).toBe(true);
        expect(hud.hidden).toBe(true);
        expect(headScale).toBeGreaterThan(moduleScale);

        canvas.dispatchEvent(new WheelEvent('wheel', {
            deltaY: 160,
            clientX: 320,
            clientY: 180,
            bubbles: true,
            cancelable: true
        }));
        rafQueue.splice(0).forEach((callback) => callback?.(performance.now()));

        const zoomedOutScale = view.getViewportState().scale;
        expect(zoomedOutScale).toBeLessThan(headScale);
        expect(canvasCard.classList.contains('is-head-detail-active')).toBe(true);
        expect(hud.hidden).toBe(true);
    });

    it('caps the deepest head-detail viewport at a separate zoom-out floor', () => {
        const panel = document.createElement('section');
        panel.innerHTML = `
            <div class="detail-header"></div>
            <div class="detail-body"></div>
        `;
        document.body.appendChild(panel);

        const view = createTransformerView2dDetailView(panel);
        const canvas = panel.querySelector('.detail-transformer-view2d-canvas');
        const ctx = createMockContext();
        const activationSource = createActivationSource(4);
        const canvasRect = {
            width: 72,
            height: 72
        };

        canvas.getContext = vi.fn(() => ctx);
        canvas.getBoundingClientRect = () => ({
            left: 0,
            top: 0,
            right: canvasRect.width,
            bottom: canvasRect.height,
            width: canvasRect.width,
            height: canvasRect.height
        });

        view.setVisible(true);
        view.open({
            activationSource,
            semanticTarget: {
                componentKind: 'mhsa',
                layerIndex: 0,
                headIndex: 4,
                stage: 'attention',
                role: 'head'
            },
            focusLabel: 'Layer 1 MHSA Head 5'
        });

        const scene = buildTransformerSceneModel({
            activationSource,
            layerCount: 1,
            headDetailTarget: {
                layerIndex: 0,
                headIndex: 4
            }
        });
        const headDetailScene = scene.metadata.mhsaHeadDetailScene || scene.metadata.headDetailScene;
        const detailLayout = buildSceneLayout(headDetailScene);
        const detailBounds = resolveSceneRenderableBounds(headDetailScene, detailLayout);
        const unclampedFit = resolveViewportFitTransform(detailBounds, canvasRect, {
            padding: 28,
            minScale: 0.035,
            maxScale: 10
        });
        const expectedFit = resolveViewportFitTransform(detailBounds, canvasRect, {
            padding: 28,
            minScale: 0.06,
            maxScale: 10
        });

        expect(unclampedFit.scale).toBeLessThan(0.06);
        expect(expectedFit.scale).toBeCloseTo(0.06, 6);
        expect(view.getViewportState().scale).toBeCloseTo(expectedFit.scale, 6);
    });

    it('enters the concat detail state for concatenate targets and hides the overview HUD', () => {
        const panel = document.createElement('section');
        panel.innerHTML = `
            <div class="detail-header"></div>
            <div class="detail-body"></div>
        `;
        document.body.appendChild(panel);

        const view = createTransformerView2dDetailView(panel);
        const canvas = panel.querySelector('.detail-transformer-view2d-canvas');
        const canvasCard = panel.querySelector('.detail-transformer-view2d-canvas-card');
        const hud = panel.querySelector('.detail-transformer-view2d-hud');
        const ctx = createMockContext();

        canvas.getContext = vi.fn(() => ctx);
        canvas.getBoundingClientRect = () => ({
            left: 0,
            top: 0,
            right: 640,
            bottom: 360,
            width: 640,
            height: 360
        });

        view.setVisible(true);
        view.open({
            activationSource: createActivationSource(4),
            semanticTarget: {
                componentKind: 'mhsa',
                layerIndex: 0,
                stage: 'concatenate',
                role: 'concat'
            },
            focusLabel: 'Layer 1 Concatenate Heads'
        });

        expect(canvasCard.classList.contains('is-head-detail-active')).toBe(true);
        expect(hud.hidden).toBe(true);

        canvas.dispatchEvent(new WheelEvent('wheel', {
            deltaY: 160,
            clientX: 320,
            clientY: 180,
            bubbles: true,
            cancelable: true
        }));
        rafQueue.splice(0).forEach((callback) => callback?.(performance.now()));

        expect(canvasCard.classList.contains('is-head-detail-active')).toBe(false);
        expect(hud.hidden).toBe(false);
    });

    it('enters the output projection detail state for output projection targets and hides the overview HUD', () => {
        const panel = document.createElement('section');
        panel.innerHTML = `
            <div class="detail-header"></div>
            <div class="detail-body"></div>
        `;
        document.body.appendChild(panel);

        const view = createTransformerView2dDetailView(panel);
        const canvas = panel.querySelector('.detail-transformer-view2d-canvas');
        const canvasCard = panel.querySelector('.detail-transformer-view2d-canvas-card');
        const hud = panel.querySelector('.detail-transformer-view2d-hud');
        const ctx = createMockContext();

        canvas.getContext = vi.fn(() => ctx);
        canvas.getBoundingClientRect = () => ({
            left: 0,
            top: 0,
            right: 640,
            bottom: 360,
            width: 640,
            height: 360
        });

        view.setVisible(true);
        view.open({
            activationSource: createActivationSource(4),
            semanticTarget: {
                componentKind: 'output-projection',
                layerIndex: 0,
                stage: 'attn-out',
                role: 'projection-weight'
            },
            focusLabel: 'Layer 1 Output Projection'
        });

        expect(canvasCard.classList.contains('is-head-detail-active')).toBe(true);
        expect(hud.hidden).toBe(true);

        canvas.dispatchEvent(new WheelEvent('wheel', {
            deltaY: 160,
            clientX: 320,
            clientY: 180,
            bubbles: true,
            cancelable: true
        }));
        rafQueue.splice(0).forEach((callback) => callback?.(performance.now()));

        expect(canvasCard.classList.contains('is-head-detail-active')).toBe(false);
        expect(hud.hidden).toBe(false);
    });

    it('shows the shared residual hover tooltip for hovered 2D residual rows and hides it on leave', () => {
        const panel = document.createElement('section');
        panel.innerHTML = `
            <div class="detail-header"></div>
            <div class="detail-body"></div>
        `;
        document.body.appendChild(panel);

        const view = createTransformerView2dDetailView(panel);
        const canvas = panel.querySelector('.detail-transformer-view2d-canvas');
        const ctx = createMockContext();
        const activationSource = createActivationSource(4);

        canvas.getContext = vi.fn(() => ctx);
        canvas.getBoundingClientRect = () => ({
            left: 0,
            top: 0,
            right: 640,
            bottom: 360,
            width: 640,
            height: 360
        });

        view.setVisible(true);
        view.open({
            activationSource,
            semanticTarget: {
                componentKind: 'residual',
                layerIndex: 0,
                stage: 'incoming',
                role: 'module'
            },
            focusLabel: 'Layer 1 incoming residual'
        });

        const scene = buildTransformerSceneModel({
            activationSource,
            layerCount: 1
        });
        const layout = buildSceneLayout(scene);
        const residualEntry = layout.registry.getNodeEntries().find((entry) => (
            entry.role === 'module-card'
            && entry.semantic?.componentKind === 'residual'
            && entry.semantic?.stage === 'incoming'
        ));
        const viewport = view.getViewportState();
        const worldX = residualEntry.contentBounds.x + residualEntry.layoutData.innerPaddingX + 8;
        const worldY = residualEntry.contentBounds.y + residualEntry.layoutData.innerPaddingY + (residualEntry.layoutData.rowHeight * 0.5);
        const clientX = viewport.panX + (worldX * viewport.scale);
        const clientY = viewport.panY + (worldY * viewport.scale);

        dispatchPointerEvent(canvas, 'pointermove', {
            clientX,
            clientY,
            pointerType: 'mouse'
        });

        const tooltip = document.body.querySelector('.scene-hover-label');
        expect(tooltip?.style.display).toBe('block');
        expect(tooltip?.querySelector('.scene-hover-label__text')?.textContent).toBe('Residual Stream Vector');
        expect(tooltip?.querySelector('.scene-hover-label__token-chip')?.textContent).toContain('tok_0');
        expect(tooltip?.querySelector('.scene-hover-label__subtitle')?.textContent).toBe('Position 1 • Layer 1');

        dispatchPointerEvent(canvas, 'pointerleave', {
            clientX,
            clientY,
            pointerType: 'mouse'
        });

        expect(tooltip?.style.display).toBe('none');
    });

    it('shows the shared layer norm hover tooltip for overview layer norm pills', () => {
        const panel = document.createElement('section');
        panel.innerHTML = `
            <div class="detail-header"></div>
            <div class="detail-body"></div>
        `;
        document.body.appendChild(panel);

        const view = createTransformerView2dDetailView(panel);
        const canvas = panel.querySelector('.detail-transformer-view2d-canvas');
        const ctx = createMockContext();
        const activationSource = createActivationSource(4);

        canvas.getContext = vi.fn(() => ctx);
        canvas.getBoundingClientRect = () => ({
            left: 0,
            top: 0,
            right: 640,
            bottom: 360,
            width: 640,
            height: 360
        });

        view.setVisible(true);
        view.open({ activationSource });

        const scene = buildTransformerSceneModel({ activationSource });
        const layout = buildSceneLayout(scene);
        const layerNormEntry = layout.registry.getNodeEntries().find((entry) => (
            entry.role === 'module-card'
            && entry.semantic?.componentKind === 'layer-norm'
            && entry.semantic?.stage === 'ln1'
            && entry.semantic?.layerIndex === 0
        ));
        const viewport = view.getViewportState();
        const { clientX, clientY } = resolveEntryClientPoint(layerNormEntry, viewport);

        dispatchPointerEvent(canvas, 'pointermove', {
            clientX,
            clientY,
            pointerType: 'mouse'
        });

        const tooltip = document.body.querySelector('.scene-hover-label');
        expect(tooltip?.style.display).toBe('block');
        expect(tooltip?.querySelector('.scene-hover-label__text')?.textContent).toBe('LayerNorm 1');
        expect(tooltip?.querySelector('.scene-hover-label__token-chip')?.hidden).toBe(true);
        expect(tooltip?.querySelector('.scene-hover-label__subtitle')?.textContent).toBe('Layer 1');
    });

    it('shows the shared output projection hover tooltip for overview output projection cards', () => {
        const panel = document.createElement('section');
        panel.innerHTML = `
            <div class="detail-header"></div>
            <div class="detail-body"></div>
        `;
        document.body.appendChild(panel);

        const view = createTransformerView2dDetailView(panel);
        const canvas = panel.querySelector('.detail-transformer-view2d-canvas');
        const ctx = createMockContext();
        const activationSource = createActivationSource(4);

        canvas.getContext = vi.fn(() => ctx);
        canvas.getBoundingClientRect = () => ({
            left: 0,
            top: 0,
            right: 640,
            bottom: 360,
            width: 640,
            height: 360
        });

        view.setVisible(true);
        view.open({ activationSource });

        const scene = buildTransformerSceneModel({ activationSource });
        const layout = buildSceneLayout(scene);
        const outputProjectionEntry = layout.registry.getNodeEntries().find((entry) => (
            entry.role === 'projection-weight'
            && entry.semantic?.componentKind === 'output-projection'
            && entry.semantic?.layerIndex === 0
        ));
        const viewport = view.getViewportState();
        const { clientX, clientY } = resolveEntryClientPoint(outputProjectionEntry, viewport);

        dispatchPointerEvent(canvas, 'pointermove', {
            clientX,
            clientY,
            pointerType: 'mouse'
        });

        const tooltip = document.body.querySelector('.scene-hover-label');
        expect(tooltip?.style.display).toBe('block');
        expect(tooltip?.querySelector('.scene-hover-label__text')?.textContent).toBe('Output Projection Matrix');
        expect(tooltip?.querySelector('.scene-hover-label__token-chip')?.hidden).toBe(true);
        expect(tooltip?.querySelector('.scene-hover-label__subtitle')?.textContent).toBe('Layer 1');
    });

    it('shows the shared post-layernorm hover tooltip for deep attention-head branch copies', () => {
        const panel = document.createElement('section');
        panel.innerHTML = `
            <div class="detail-header"></div>
            <div class="detail-body"></div>
        `;
        document.body.appendChild(panel);

        const view = createTransformerView2dDetailView(panel);
        const canvas = panel.querySelector('.detail-transformer-view2d-canvas');
        const ctx = createMockContext();
        const activationSource = createActivationSource(4);

        canvas.getContext = vi.fn(() => ctx);
        canvas.getBoundingClientRect = () => ({
            left: 0,
            top: 0,
            right: 640,
            bottom: 360,
            width: 640,
            height: 360
        });

        view.setVisible(true);
        view.open({
            activationSource,
            semanticTarget: {
                componentKind: 'mhsa',
                layerIndex: 0,
                headIndex: 4,
                stage: 'attention',
                role: 'head'
            },
            focusLabel: 'Layer 1 MHSA Head 5'
        });
        rafQueue.splice(0).forEach((callback) => callback?.(performance.now()));

        const scene = buildTransformerSceneModel({
            activationSource,
            layerCount: 1,
            headDetailTarget: {
                layerIndex: 0,
                headIndex: 4
            }
        });
        const headDetailScene = scene.metadata.mhsaHeadDetailScene || scene.metadata.headDetailScene;
        const detailLayout = buildSceneLayout(headDetailScene);
        const copyEntry = detailLayout.registry.getNodeEntries().find((entry) => (
            entry.role === 'x-ln-copy'
            && entry.semantic?.branchKey === 'q'
        ));
        const detailViewport = view.getViewportState();
        const rowWidth = Math.max(
            1,
            Math.min(
                copyEntry.layoutData.compactWidth,
                copyEntry.contentBounds.width - (copyEntry.layoutData.innerPaddingX * 2)
            )
        );
        const worldX = copyEntry.contentBounds.x + copyEntry.layoutData.innerPaddingX + (rowWidth * 0.5);
        const worldY = copyEntry.contentBounds.y + copyEntry.layoutData.innerPaddingY + (copyEntry.layoutData.rowHeight * 0.5);
        const clientX = detailViewport.panX + (worldX * detailViewport.scale);
        const clientY = detailViewport.panY + (worldY * detailViewport.scale);

        dispatchPointerEvent(canvas, 'pointermove', {
            clientX,
            clientY,
            pointerType: 'mouse'
        });

        const tooltip = document.body.querySelector('.scene-hover-label');
        expect(tooltip?.style.display).toBe('block');
        expect(tooltip?.querySelector('.scene-hover-label__text')?.textContent).toBe('Post LayerNorm Residual Vector');
        expect(tooltip?.querySelector('.scene-hover-label__token-chip')?.textContent).toContain('tok_0');
        expect(tooltip?.querySelector('.scene-hover-label__subtitle')?.textContent).toBe('Position 1 • Head 5 • Layer 1');
    });

    it('keeps deep-detail focus active for value rows and copied X_ln rows', () => {
        const panel = document.createElement('section');
        panel.innerHTML = `
            <div class="detail-header"></div>
            <div class="detail-body"></div>
        `;
        document.body.appendChild(panel);

        const renderSpy = vi.spyOn(CanvasSceneRenderer.prototype, 'render');
        const view = createTransformerView2dDetailView(panel);
        const canvas = panel.querySelector('.detail-transformer-view2d-canvas');
        const ctx = createMockContext();
        const activationSource = createActivationSource(4);

        canvas.getContext = vi.fn(() => ctx);
        canvas.getBoundingClientRect = () => ({
            left: 0,
            top: 0,
            right: 640,
            bottom: 360,
            width: 640,
            height: 360
        });

        view.setVisible(true);
        view.open({
            activationSource,
            semanticTarget: {
                componentKind: 'mhsa',
                layerIndex: 0,
                headIndex: 4,
                stage: 'attention',
                role: 'head'
            },
            focusLabel: 'Layer 1 MHSA Head 5'
        });
        drainRafQueue(rafQueue);

        const scene = buildTransformerSceneModel({
            activationSource,
            layerCount: 1,
            headDetailTarget: {
                layerIndex: 0,
                headIndex: 4
            }
        });
        const headDetailScene = scene.metadata.mhsaHeadDetailScene || scene.metadata.headDetailScene;
        const detailLayout = buildSceneLayout(headDetailScene);
        const copyEntry = detailLayout.registry.getNodeEntries().find((entry) => (
            entry.role === 'x-ln-copy'
            && entry.semantic?.branchKey === 'v'
        ));
        const valueEntry = detailLayout.registry.getNodeEntries().find((entry) => (
            entry.role === 'projection-output'
            && entry.metadata?.kind === 'v'
        ));
        const detailViewport = view.getViewportState();

        const valueRenderCountBefore = renderSpy.mock.calls.length;
        dispatchPointerEvent(canvas, 'pointermove', {
            ...resolveInteractiveRowClientPoint(headDetailScene, detailLayout, detailViewport, valueEntry, 1),
            pointerType: 'mouse'
        });
        drainRafQueue(rafQueue);

        const valueTooltip = document.body.querySelector('.scene-hover-label');
        expect(valueTooltip?.querySelector('.scene-hover-label__text')?.textContent).toBe('Value Vector');

        const valueFocusCall = renderSpy.mock.calls
            .slice(valueRenderCountBefore)
            .map(([args]) => args)
            .reverse()
            .find((args) => args?.interactionState?.detailSceneFocus);
        expect(valueFocusCall?.interactionState?.detailSceneFocus?.activeNodeIds?.length).toBeGreaterThan(0);

        const copyRenderCountBefore = renderSpy.mock.calls.length;
        dispatchPointerEvent(canvas, 'pointermove', {
            ...resolveInteractiveRowClientPoint(headDetailScene, detailLayout, detailViewport, copyEntry, 0),
            pointerType: 'mouse'
        });
        drainRafQueue(rafQueue);

        const copyTooltip = document.body.querySelector('.scene-hover-label');
        expect(copyTooltip?.querySelector('.scene-hover-label__text')?.textContent).toBe('Post LayerNorm Residual Vector');

        const copyFocusCall = renderSpy.mock.calls
            .slice(copyRenderCountBefore)
            .map(([args]) => args)
            .reverse()
            .find((args) => args?.interactionState?.detailSceneFocus);
        expect(copyFocusCall?.interactionState?.detailSceneFocus?.activeNodeIds?.length).toBeGreaterThan(0);
    });

    it('supports touch pan and pinch zoom on the transformer 2D canvas surface', () => {
        const panel = document.createElement('section');
        panel.innerHTML = `
            <div class="detail-header"></div>
            <div class="detail-body"></div>
        `;
        document.body.appendChild(panel);

        const view = createTransformerView2dDetailView(panel);
        const canvas = panel.querySelector('.detail-transformer-view2d-canvas');
        const ctx = createMockContext();

        canvas.getContext = vi.fn(() => ctx);
        canvas.getBoundingClientRect = () => ({
            left: 0,
            top: 0,
            right: 640,
            bottom: 360,
            width: 640,
            height: 360
        });
        canvas.setPointerCapture = vi.fn();
        canvas.releasePointerCapture = vi.fn();

        view.setVisible(true);
        view.open({
            activationSource: createActivationSource(4),
            semanticTarget: {
                componentKind: 'mhsa',
                layerIndex: 0,
                stage: 'attention',
                role: 'module'
            },
            focusLabel: 'Layer 1 MHSA'
        });

        const beforePan = view.getViewportState();
        dispatchPointerEvent(canvas, 'pointerdown', {
            clientX: 120,
            clientY: 120,
            pointerId: 1,
            pointerType: 'touch'
        });
        dispatchPointerEvent(canvas, 'pointermove', {
            clientX: 160,
            clientY: 150,
            pointerId: 1,
            pointerType: 'touch'
        });
        const afterPan = view.getViewportState();

        expect(afterPan.panX).toBeGreaterThan(beforePan.panX);
        expect(afterPan.panY).toBeGreaterThan(beforePan.panY);

        dispatchPointerEvent(canvas, 'pointerdown', {
            clientX: 240,
            clientY: 110,
            pointerId: 2,
            pointerType: 'touch'
        });
        dispatchPointerEvent(canvas, 'pointermove', {
            clientX: 300,
            clientY: 80,
            pointerId: 2,
            pointerType: 'touch'
        });
        const afterPinch = view.getViewportState();

        expect(afterPinch.scale).toBeGreaterThan(afterPan.scale);

        dispatchPointerEvent(canvas, 'pointerup', {
            clientX: 300,
            clientY: 80,
            pointerId: 2,
            pointerType: 'touch'
        });
        dispatchPointerEvent(canvas, 'pointermove', {
            clientX: 182,
            clientY: 166,
            pointerId: 1,
            pointerType: 'touch'
        });
        const afterPinchPan = view.getViewportState();

        expect(afterPinchPan.panX).toBeGreaterThan(afterPinch.panX);
        expect(afterPinchPan.panY).toBeGreaterThan(afterPinch.panY);
    });

});
