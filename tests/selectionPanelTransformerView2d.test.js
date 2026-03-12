// @vitest-environment jsdom

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import {
    createTransformerView2dDetailView,
    describeTransformerView2dTarget,
    resolveTransformerView2dActionContext
} from '../src/ui/selectionPanelTransformerView2d.js';
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
        expect(context?.focusLabel).toBe('Layer 4 MHSA Head 9');
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
        expect(context?.focusLabel).toBe('Layer 3 output projection');
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
        expect(context?.focusLabel).toBe('Layer 8 MLP up projection');
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
