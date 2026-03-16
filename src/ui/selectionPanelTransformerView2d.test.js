// @vitest-environment jsdom

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { D_HEAD, D_MODEL } from './selectionPanelConstants.js';
import { TRANSFORMER_VIEW2D_STAGED_HEAD_DETAIL_OVERVIEW_TO_HEAD_DURATION_MS } from './selectionPanelTransformerView2dTransitionUtils.js';

function createRect(width = 960, height = 600) {
    return {
        width,
        height,
        top: 0,
        right: width,
        bottom: height,
        left: 0,
        x: 0,
        y: 0,
        toJSON() {
            return this;
        }
    };
}

function createMockCanvasContext() {
    const stateStack = [];
    return {
        canvas: null,
        globalAlpha: 1,
        fillStyle: '#000',
        strokeStyle: '#000',
        lineWidth: 1,
        font: '12px sans-serif',
        textAlign: 'left',
        textBaseline: 'alphabetic',
        globalCompositeOperation: 'source-over',
        filter: 'none',
        shadowBlur: 0,
        shadowColor: 'transparent',
        save() {
            stateStack.push({
                globalAlpha: this.globalAlpha,
                fillStyle: this.fillStyle,
                strokeStyle: this.strokeStyle,
                lineWidth: this.lineWidth,
                font: this.font,
                textAlign: this.textAlign,
                textBaseline: this.textBaseline,
                globalCompositeOperation: this.globalCompositeOperation,
                filter: this.filter,
                shadowBlur: this.shadowBlur,
                shadowColor: this.shadowColor
            });
        },
        restore() {
            const state = stateStack.pop();
            if (!state) return;
            Object.assign(this, state);
        },
        setTransform() {},
        resetTransform() {},
        clearRect() {},
        fillRect() {},
        strokeRect() {},
        beginPath() {},
        closePath() {},
        moveTo() {},
        lineTo() {},
        rect() {},
        roundRect() {},
        arc() {},
        arcTo() {},
        bezierCurveTo() {},
        quadraticCurveTo() {},
        translate() {},
        scale() {},
        stroke() {},
        fill() {},
        clip() {},
        fillText() {},
        strokeText() {},
        setLineDash() {},
        measureText(text = '') {
            return {
                width: String(text).length * 7,
                actualBoundingBoxAscent: 8,
                actualBoundingBoxDescent: 2
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
        drawImage() {}
    };
}

function setElementRect(element, width = 960, height = 600) {
    const rect = createRect(width, height);
    element.getBoundingClientRect = () => rect;
    Object.defineProperty(element, 'clientWidth', {
        configurable: true,
        value: width
    });
    Object.defineProperty(element, 'clientHeight', {
        configurable: true,
        value: height
    });
}

function createActivationSource() {
    return {
        getTokenCount() {
            return 3;
        },
        getTokenId(tokenIndex = 0) {
            return tokenIndex;
        },
        getTokenString(tokenIndex = 0) {
            return ['A', 'B', 'C'][tokenIndex] || `Token ${tokenIndex + 1}`;
        },
        getTokenRawString(tokenIndex = 0) {
            return ['A', 'B', 'C'][tokenIndex] || `Token ${tokenIndex + 1}`;
        },
        getBaseVectorLength() {
            return D_MODEL;
        },
        getLayerLn1(_layerIndex = 0, _mode = 'shift', tokenIndex = 0, targetLength = D_MODEL) {
            return Array.from({ length: targetLength || D_MODEL }, (_, index) => (
                Number(((tokenIndex * 0.05) + (index * 0.001)).toFixed(4))
            ));
        },
        getLayerLn2(_layerIndex = 0, _mode = 'shift', tokenIndex = 0, targetLength = D_MODEL) {
            return Array.from({ length: targetLength || D_MODEL }, (_, index) => (
                Number(((tokenIndex * 0.06) + (index * 0.0012)).toFixed(4))
            ));
        },
        getLayerQKVVector(_layerIndex = 0, kind = 'q', headIndex = 0, tokenIndex = 0, targetLength = D_HEAD) {
            const kindOffset = kind === 'k' ? 0.2 : (kind === 'v' ? 0.4 : 0);
            return Array.from({ length: targetLength || D_HEAD }, (_, index) => (
                Number(((headIndex * 0.1) + (tokenIndex * 0.03) + kindOffset + (index * 0.002)).toFixed(4))
            ));
        },
        getLayerQKVScalar(_layerIndex = 0, kind = 'q', headIndex = 0, tokenIndex = 0) {
            const kindOffset = kind === 'k' ? 0.15 : (kind === 'v' ? 0.3 : 0);
            return Number(((headIndex * 0.1) + (tokenIndex * 0.05) + kindOffset).toFixed(4));
        },
        getAttentionScoresRow(_layerIndex = 0, mode = 'pre', headIndex = 0, tokenIndex = 0) {
            const rowLength = 3;
            return Array.from({ length: rowLength }, (_, index) => {
                if (mode === 'post' && index > tokenIndex) return 0;
                return Number(((headIndex * 0.08) + (tokenIndex * 0.04) + (index * 0.02)).toFixed(4));
            });
        },
        getAttentionWeightedSum(_layerIndex = 0, headIndex = 0, tokenIndex = 0, targetLength = D_HEAD) {
            return Array.from({ length: targetLength || D_HEAD }, (_, index) => (
                Number(((headIndex * 0.09) + (tokenIndex * 0.04) + (index * 0.0025)).toFixed(4))
            ));
        },
        getMlpUp(_layerIndex = 0, tokenIndex = 0, targetLength = D_MODEL * 4) {
            return Array.from({ length: targetLength || (D_MODEL * 4) }, (_, index) => (
                Number(((tokenIndex * 0.07) + (index * 0.0007)).toFixed(4))
            ));
        },
        getMlpActivation(_layerIndex = 0, tokenIndex = 0, targetLength = D_MODEL * 4) {
            return Array.from({ length: targetLength || (D_MODEL * 4) }, (_, index) => (
                Number(((tokenIndex * 0.08) + (index * 0.0006)).toFixed(4))
            ));
        },
        getMlpDown(_layerIndex = 0, tokenIndex = 0, targetLength = D_MODEL) {
            return Array.from({ length: targetLength || D_MODEL }, (_, index) => (
                Number(((tokenIndex * 0.09) + (index * 0.0011)).toFixed(4))
            ));
        }
    };
}

describe('createTransformerView2dDetailView', () => {
    let rafTime = 0;
    let createTransformerView2dDetailView = null;

    beforeEach(async () => {
        vi.useFakeTimers();
        vi.resetModules();
        rafTime = 0;
        vi.spyOn(performance, 'now').mockImplementation(() => rafTime);
        vi.stubGlobal('requestAnimationFrame', vi.fn((callback) => setTimeout(() => {
            rafTime += 16;
            callback(rafTime);
        }, 16)));
        vi.stubGlobal('cancelAnimationFrame', vi.fn((id) => clearTimeout(id)));
        vi.stubGlobal('localStorage', {
            getItem: vi.fn(() => null),
            setItem: vi.fn(),
            removeItem: vi.fn()
        });

        document.body.innerHTML = `
            <div id="detailPanel">
                <div class="detail-header"></div>
            </div>
        `;

        HTMLCanvasElement.prototype.getContext = vi.fn(function getContext() {
            const context = createMockCanvasContext();
            context.canvas = this;
            return context;
        });

        ({ createTransformerView2dDetailView } = await import('./selectionPanelTransformerView2d.js'));
    });

    afterEach(() => {
        document.body.innerHTML = '';
        vi.useRealTimers();
        vi.restoreAllMocks();
    });

    it('stages MHSA entries from the tower overview into the head-detail scene', async () => {
        const panelEl = document.getElementById('detailPanel');
        const view = createTransformerView2dDetailView(panelEl);

        const canvas = panelEl.querySelector('.detail-transformer-view2d-canvas');
        const canvasCard = panelEl.querySelector('.detail-transformer-view2d-canvas-card');
        setElementRect(canvas, 960, 600);
        setElementRect(canvasCard, 960, 600);

        view.setVisible(true);
        view.open({
            activationSource: createActivationSource(),
            tokenIndices: [0, 1, 2],
            tokenLabels: ['A', 'B', 'C'],
            semanticTarget: {
                componentKind: 'mhsa',
                layerIndex: 1,
                headIndex: 2,
                stage: 'attention',
                role: 'head'
            },
            focusLabel: 'Layer 2 Attention Head 3',
            detailSemanticTargets: [{
                componentKind: 'mhsa',
                layerIndex: 1,
                headIndex: 2,
                stage: 'attention',
                role: 'attention-post'
            }],
            detailFocusLabel: 'Post-Softmax Attention Score',
            transitionMode: 'staged-head-detail'
        });

        const initialViewportScale = view.getViewportState().scale;
        expect(canvas.classList.contains('is-head-detail-scene-active')).toBe(false);

        await vi.advanceTimersByTimeAsync(700);

        const overviewFocusScale = view.getViewportState().scale;
        expect(overviewFocusScale).toBeGreaterThan(initialViewportScale);
        expect(canvas.classList.contains('is-head-detail-scene-active')).toBe(false);

        await vi.advanceTimersByTimeAsync(
            TRANSFORMER_VIEW2D_STAGED_HEAD_DETAIL_OVERVIEW_TO_HEAD_DURATION_MS + 400
        );

        expect(canvas.classList.contains('is-head-detail-scene-active')).toBe(true);
    });

    it('stages scene-backed MLP targets from overview focus into the detail scene', async () => {
        const panelEl = document.getElementById('detailPanel');
        const view = createTransformerView2dDetailView(panelEl);

        const canvas = panelEl.querySelector('.detail-transformer-view2d-canvas');
        const canvasCard = panelEl.querySelector('.detail-transformer-view2d-canvas-card');
        setElementRect(canvas, 960, 600);
        setElementRect(canvasCard, 960, 600);

        view.setVisible(true);
        view.open({
            activationSource: createActivationSource(),
            tokenIndices: [0, 1, 2],
            tokenLabels: ['A', 'B', 'C'],
            semanticTarget: {
                componentKind: 'mlp',
                layerIndex: 1,
                stage: 'mlp',
                role: 'module'
            },
            focusLabel: 'Layer 2 Multilayer Perceptron',
            detailSemanticTargets: [{
                componentKind: 'mlp',
                layerIndex: 1,
                stage: 'mlp-up',
                role: 'mlp-up-weight'
            }],
            detailFocusLabel: 'MLP Up Weight Matrix',
            transitionMode: 'staged-detail'
        });

        expect(canvas.classList.contains('is-head-detail-scene-active')).toBe(false);

        await vi.advanceTimersByTimeAsync(800);
        expect(canvas.classList.contains('is-head-detail-scene-active')).toBe(false);

        await vi.advanceTimersByTimeAsync(1200);
        expect(canvas.classList.contains('is-head-detail-scene-active')).toBe(true);
    });
});
