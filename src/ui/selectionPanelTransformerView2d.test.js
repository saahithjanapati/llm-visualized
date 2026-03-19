// @vitest-environment jsdom

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { D_HEAD, D_MODEL } from './selectionPanelConstants.js';
import {
    TRANSFORMER_VIEW2D_STAGED_FOCUS_OVERVIEW_TO_TARGET_DURATION_MS,
    TRANSFORMER_VIEW2D_STAGED_HEAD_DETAIL_OVERVIEW_TO_HEAD_DURATION_MS
} from './selectionPanelTransformerView2dTransitionUtils.js';
import { TRANSFORMER_VIEW2D_OVERVIEW_MIN_SCALE_DEFAULT } from './selectionPanelTransformerView2dViewportUtils.js';
import { TRANSFORMER_VIEW2D_OVERVIEW_LABEL } from '../view2d/transformerView2dTargets.js';
import { buildSceneLayout } from '../view2d/layout/buildSceneLayout.js';
import { resolveSemanticTargetBounds } from '../view2d/layout/resolveSemanticTargetBounds.js';
import { buildTransformerSceneModel } from '../view2d/model/buildTransformerSceneModel.js';
import { resolveViewportFitTransform } from '../view2d/runtime/View2dViewportController.js';
import { flattenSceneNodes, VIEW2D_NODE_KINDS } from '../view2d/schema/sceneTypes.js';

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

function setElementRectAt(element, {
    width = 960,
    height = 600,
    left = 0,
    top = 0
} = {}) {
    const rect = {
        width,
        height,
        top,
        right: left + width,
        bottom: top + height,
        left,
        x: left,
        y: top,
        toJSON() {
            return this;
        }
    };
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

function createPointerEvent(type, {
    pointerId = 1,
    pointerType = 'mouse',
    clientX = 24,
    clientY = 24,
    button = 0
} = {}) {
    const event = new Event(type, {
        bubbles: true,
        cancelable: true
    });
    Object.defineProperties(event, {
        pointerId: { configurable: true, value: pointerId },
        pointerType: { configurable: true, value: pointerType },
        clientX: { configurable: true, value: clientX },
        clientY: { configurable: true, value: clientY },
        button: { configurable: true, value: button }
    });
    return event;
}

function createWheelEvent({
    clientX = 24,
    clientY = 24,
    deltaY = -120
} = {}) {
    const event = new Event('wheel', {
        bubbles: true,
        cancelable: true
    });
    Object.defineProperties(event, {
        clientX: { configurable: true, value: clientX },
        clientY: { configurable: true, value: clientY },
        deltaY: { configurable: true, value: deltaY }
    });
    return event;
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

function setViewportSize(width = 1024, height = 768) {
    Object.defineProperty(window, 'innerWidth', {
        configurable: true,
        value: width
    });
    Object.defineProperty(window, 'innerHeight', {
        configurable: true,
        value: height
    });
}

function getStageReadouts(panelEl) {
    return {
        layer: panelEl.querySelector('[data-transformer-view2d-readout="layer"]'),
        stage: panelEl.querySelector('[data-transformer-view2d-readout="stage"]')
    };
}

function resolveCompactRowScreenPoint(nodeEntry = null, rowIndex = 0, viewportState = null) {
    const contentBounds = nodeEntry?.contentBounds || nodeEntry?.bounds || null;
    const layoutData = nodeEntry?.layoutData || null;
    if (!contentBounds || !layoutData) return null;

    const innerPaddingX = Math.max(0, Number(layoutData.innerPaddingX) || 0);
    const innerPaddingY = Math.max(0, Number(layoutData.innerPaddingY) || 0);
    const rowHeight = Math.max(1, Number(layoutData.rowHeight) || 0);
    const rowGap = Math.max(0, Number(layoutData.rowGap) || 0);
    const compactWidth = Math.max(
        1,
        Math.min(
            Number(layoutData.compactWidth) || 0,
            contentBounds.width - (innerPaddingX * 2)
        )
    );
    const worldX = contentBounds.x + innerPaddingX + (compactWidth * 0.5);
    const worldY = (
        contentBounds.y
        + innerPaddingY
        + (Math.max(0, Math.floor(rowIndex)) * (rowHeight + rowGap))
        + (rowHeight * 0.5)
    );
    const scale = Number(viewportState?.scale) || 1;
    const panX = Number(viewportState?.panX) || 0;
    const panY = Number(viewportState?.panY) || 0;
    return {
        clientX: (worldX * scale) + panX,
        clientY: (worldY * scale) + panY
    };
}

describe('createTransformerView2dDetailView', () => {
    let rafTime = 0;
    let createTransformerView2dDetailView = null;
    let detailHoverStateOverride = null;

    beforeEach(async () => {
        vi.useFakeTimers();
        vi.resetModules();
        rafTime = 0;
        detailHoverStateOverride = null;
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

        vi.doMock('../view2d/mhsaDetailInteraction.js', async () => {
            const actual = await vi.importActual('../view2d/mhsaDetailInteraction.js');
            return {
                ...actual,
                resolveMhsaDetailHoverState(...args) {
                    if (detailHoverStateOverride) {
                        return detailHoverStateOverride;
                    }
                    return actual.resolveMhsaDetailHoverState(...args);
                }
            };
        });

        ({ createTransformerView2dDetailView } = await import('./selectionPanelTransformerView2d.js'));
    });

    afterEach(() => {
        document.body.innerHTML = '';
        vi.useRealTimers();
        vi.restoreAllMocks();
    });

    it('keeps 2D sidebar history controls in a dedicated top row above the title block', () => {
        const panelEl = document.getElementById('detailPanel');
        createTransformerView2dDetailView(panelEl);

        const selectionSidebarHeader = panelEl.querySelector('.detail-transformer-view2d-selection-sidebar-header');
        const selectionSidebarHeaderTop = panelEl.querySelector('[data-transformer-view2d-role="selection-sidebar-header-top"]');
        const selectionSidebarCopy = panelEl.querySelector('.detail-transformer-view2d-selection-sidebar-copy');
        const historyNav = selectionSidebarHeaderTop?.querySelector('.detail-history-nav');
        const closeSelectionBtn = selectionSidebarHeaderTop?.querySelector('[data-transformer-view2d-action="close-selection"]');

        expect(selectionSidebarHeader).toBeTruthy();
        expect(selectionSidebarHeader?.firstElementChild).toBe(selectionSidebarHeaderTop);
        expect(selectionSidebarHeaderTop?.nextElementSibling).toBe(selectionSidebarCopy);
        expect(historyNav).toBeTruthy();
        expect(closeSelectionBtn).toBeTruthy();
    });

    it('places the settings action at the far right of the 2D toolbar and routes the icon actions through their callbacks', () => {
        const panelEl = document.getElementById('detailPanel');
        const onOpenInfo = vi.fn();
        const onOpenSettings = vi.fn();
        createTransformerView2dDetailView(panelEl, { onOpenInfo, onOpenSettings });

        const toolbarActions = panelEl.querySelector('.detail-transformer-view2d-toolbar-actions');
        const infoBtn = panelEl.querySelector('[data-transformer-view2d-action="open-info"]');
        const settingsBtn = panelEl.querySelector('[data-transformer-view2d-action="open-settings"]');

        expect(infoBtn).toBeTruthy();
        expect(settingsBtn).toBeTruthy();
        expect(toolbarActions?.lastElementChild).toBe(settingsBtn);
        expect(settingsBtn?.previousElementSibling).toBe(infoBtn);
        expect(infoBtn?.getAttribute('aria-label')).toBe('Open project info page');
        expect(settingsBtn?.getAttribute('aria-label')).toBe('Open settings');

        infoBtn?.dispatchEvent(new MouseEvent('click', { bubbles: true }));
        settingsBtn?.dispatchEvent(new MouseEvent('click', { bubbles: true }));

        expect(onOpenInfo).toHaveBeenCalledTimes(1);
        expect(onOpenSettings).toHaveBeenCalledTimes(1);
    });

    it('shows the 2D toolbar info/settings actions on desktop and landscape-small screens, but hides them on portrait-small screens', () => {
        const panelEl = document.getElementById('detailPanel');
        const view = createTransformerView2dDetailView(panelEl);

        const canvas = panelEl.querySelector('.detail-transformer-view2d-canvas');
        const canvasCard = panelEl.querySelector('.detail-transformer-view2d-canvas-card');
        const infoBtn = panelEl.querySelector('[data-transformer-view2d-action="open-info"]');
        const settingsBtn = panelEl.querySelector('[data-transformer-view2d-action="open-settings"]');
        setElementRect(canvas, 960, 600);
        setElementRect(canvasCard, 960, 600);

        setViewportSize(1280, 800);
        view.setVisible(true);
        view.open({
            activationSource: createActivationSource(),
            tokenIndices: [0, 1, 2],
            tokenLabels: ['A', 'B', 'C'],
            semanticTarget: null,
            focusLabel: TRANSFORMER_VIEW2D_OVERVIEW_LABEL,
            isSmallScreen: false
        });

        expect(infoBtn?.hidden).toBe(false);
        expect(infoBtn?.getAttribute('aria-hidden')).toBe('false');
        expect(settingsBtn?.hidden).toBe(false);
        expect(settingsBtn?.getAttribute('aria-hidden')).toBe('false');

        view.open({
            activationSource: createActivationSource(),
            tokenIndices: [0, 1, 2],
            tokenLabels: ['A', 'B', 'C'],
            semanticTarget: {
                componentKind: 'embedding',
                stage: 'embedding.token',
                role: 'module'
            },
            focusLabel: 'Token Embedding',
            isSmallScreen: false
        });

        expect(infoBtn?.hidden).toBe(false);
        expect(infoBtn?.getAttribute('aria-hidden')).toBe('false');
        expect(settingsBtn?.hidden).toBe(false);
        expect(settingsBtn?.getAttribute('aria-hidden')).toBe('false');

        setViewportSize(844, 390);
        view.open({
            activationSource: createActivationSource(),
            tokenIndices: [0, 1, 2],
            tokenLabels: ['A', 'B', 'C'],
            semanticTarget: {
                componentKind: 'embedding',
                stage: 'embedding.token',
                role: 'module'
            },
            focusLabel: 'Token Embedding',
            isSmallScreen: true
        });

        expect(infoBtn?.hidden).toBe(false);
        expect(infoBtn?.getAttribute('aria-hidden')).toBe('false');
        expect(settingsBtn?.hidden).toBe(false);
        expect(settingsBtn?.getAttribute('aria-hidden')).toBe('false');

        setViewportSize(390, 844);
        view.open({
            activationSource: createActivationSource(),
            tokenIndices: [0, 1, 2],
            tokenLabels: ['A', 'B', 'C'],
            semanticTarget: {
                componentKind: 'embedding',
                stage: 'embedding.token',
                role: 'module'
            },
            focusLabel: 'Token Embedding',
            isSmallScreen: true
        });

        expect(infoBtn?.hidden).toBe(true);
        expect(infoBtn?.getAttribute('aria-hidden')).toBe('true');
        expect(settingsBtn?.hidden).toBe(true);
        expect(settingsBtn?.getAttribute('aria-hidden')).toBe('true');
    });

    it('stages MHSA entries from the tower overview into the head-detail scene', async () => {
        const panelEl = document.getElementById('detailPanel');
        const view = createTransformerView2dDetailView(panelEl);

        const canvas = panelEl.querySelector('.detail-transformer-view2d-canvas');
        const canvasCard = panelEl.querySelector('.detail-transformer-view2d-canvas-card');
        const fitBtn = panelEl.querySelector('[data-transformer-view2d-action="fit-scene"]');
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
        const focusedDetailScale = view.getViewportState().scale;

        fitBtn?.click();
        await vi.advanceTimersByTimeAsync(500);

        expect(view.getViewportState().scale).toBeLessThan(focusedDetailScale);
    });

    it('opens external MHSA detail entries directly inside the head-detail scene', async () => {
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
            transitionMode: 'direct'
        });

        const { layer, stage } = getStageReadouts(panelEl);
        expect(canvas.classList.contains('is-head-detail-scene-active')).toBe(true);
        expect(stage?.textContent).toBe('Attention Head 3');
        expect(layer?.textContent).toBe('Layer 2');
        expect(layer?.hidden).toBe(false);

        await vi.advanceTimersByTimeAsync(64);

        expect(canvas.classList.contains('is-head-detail-scene-active')).toBe(true);
        expect(stage?.textContent).toBe('Attention Head 3');
    });

    it('shows the matching overview module immediately when exiting a direct deep-detail view', async () => {
        const panelEl = document.getElementById('detailPanel');
        const view = createTransformerView2dDetailView(panelEl);

        const canvas = panelEl.querySelector('.detail-transformer-view2d-canvas');
        const canvasCard = panelEl.querySelector('.detail-transformer-view2d-canvas-card');
        const backToGraphBtn = panelEl.querySelector(
            '[data-transformer-view2d-action="exit-deep-detail"]'
        );
        setElementRect(canvas, 960, 600);
        setElementRect(canvasCard, 960, 600);

        const activationSource = createActivationSource();
        const tokenIndices = [0, 1, 2];
        const tokenLabels = ['A', 'B', 'C'];

        view.setVisible(true);
        view.open({
            activationSource,
            tokenIndices,
            tokenLabels,
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
            transitionMode: 'direct'
        });

        const overviewScene = buildTransformerSceneModel({
            activationSource,
            tokenIndices,
            tokenLabels
        });
        const overviewLayout = buildSceneLayout(overviewScene, {
            isSmallScreen: false
        });
        const overviewFocusBounds = resolveSemanticTargetBounds(overviewLayout.registry, {
            componentKind: 'mlp',
            layerIndex: 1,
            stage: 'mlp',
            role: 'module-card'
        });
        const seededOverviewTransform = resolveViewportFitTransform(overviewFocusBounds, {
            width: 960,
            height: 600
        }, {
            padding: 36,
            minScale: TRANSFORMER_VIEW2D_OVERVIEW_MIN_SCALE_DEFAULT,
            maxScale: 10
        });

        expect(canvas.classList.contains('is-head-detail-scene-active')).toBe(true);

        backToGraphBtn?.click();

        const { layer, stage } = getStageReadouts(panelEl);
        const seededViewportState = view.getViewportState();
        expect(canvas.classList.contains('is-head-detail-scene-active')).toBe(false);
        expect(stage?.textContent).toBe(TRANSFORMER_VIEW2D_OVERVIEW_LABEL);
        expect(layer?.hidden).toBe(true);
        expect(seededViewportState.scale).toBeCloseTo(seededOverviewTransform.scale, 6);
        expect(seededViewportState.panX).toBeCloseTo(seededOverviewTransform.panX, 6);
        expect(seededViewportState.panY).toBeCloseTo(seededOverviewTransform.panY, 6);

        await vi.advanceTimersByTimeAsync(500);

        expect(view.getViewportState().scale).toBeLessThan(seededViewportState.scale);
    });

    it('keeps the overview title visible during staged focus entry until the canvas leaves the outer overview', async () => {
        const panelEl = document.getElementById('detailPanel');
        const view = createTransformerView2dDetailView(panelEl);

        const canvas = panelEl.querySelector('.detail-transformer-view2d-canvas');
        const canvasCard = panelEl.querySelector('.detail-transformer-view2d-canvas-card');
        const fitBtn = panelEl.querySelector('[data-transformer-view2d-action="fit-scene"]');
        setElementRect(canvas, 960, 600);
        setElementRect(canvasCard, 960, 600);

        view.setVisible(true);
        view.open({
            activationSource: createActivationSource(),
            tokenIndices: [0, 1, 2],
            tokenLabels: ['A', 'B', 'C'],
            semanticTarget: {
                componentKind: 'embedding',
                stage: 'embedding.token',
                role: 'module'
            },
            focusLabel: 'Token embeddings',
            transitionMode: 'staged-focus'
        });

        const { layer, stage } = getStageReadouts(panelEl);
        expect(stage?.textContent).toBe(TRANSFORMER_VIEW2D_OVERVIEW_LABEL);
        expect(layer?.hidden).toBe(true);

        await vi.advanceTimersByTimeAsync(420);

        expect(stage?.textContent).toBe(TRANSFORMER_VIEW2D_OVERVIEW_LABEL);
        expect(layer?.hidden).toBe(true);
    });

    it('holds on the full GPT overview for at least 150ms before staged focus zoom begins', async () => {
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
                componentKind: 'embedding',
                stage: 'embedding.token',
                role: 'module'
            },
            focusLabel: 'Token embeddings',
            transitionMode: 'staged-focus'
        });

        const initialScale = view.getViewportState().scale;

        await vi.advanceTimersByTimeAsync(160);
        expect(view.getViewportState().scale).toBeCloseTo(initialScale, 5);

        await vi.advanceTimersByTimeAsync(48);
        expect(view.getViewportState().scale).toBeGreaterThan(initialScale);
    });

    it('uses a looser initial frame when opening a focused external target', () => {
        const panelEl = document.getElementById('detailPanel');
        const view = createTransformerView2dDetailView(panelEl);

        const canvas = panelEl.querySelector('.detail-transformer-view2d-canvas');
        const canvasCard = panelEl.querySelector('.detail-transformer-view2d-canvas-card');
        setElementRect(canvas, 960, 600);
        setElementRect(canvasCard, 960, 600);

        const activationSource = createActivationSource();
        const tokenIndices = [0, 1, 2];
        const tokenLabels = ['A', 'B', 'C'];
        const semanticTarget = {
            componentKind: 'embedding',
            stage: 'embedding.token',
            role: 'module'
        };
        const scene = buildTransformerSceneModel({
            activationSource,
            tokenIndices,
            tokenLabels
        });
        const layout = buildSceneLayout(scene, {
            isSmallScreen: false
        });
        const focusBounds = resolveSemanticTargetBounds(layout.registry, semanticTarget);
        const legacyTightFocusTransform = resolveViewportFitTransform(focusBounds, {
            width: 960,
            height: 600
        }, {
            padding: 36,
            minScale: TRANSFORMER_VIEW2D_OVERVIEW_MIN_SCALE_DEFAULT,
            maxScale: 10
        });

        view.setVisible(true);
        view.open({
            activationSource,
            tokenIndices,
            tokenLabels,
            semanticTarget,
            focusLabel: 'Token embeddings',
            transitionMode: 'direct'
        });

        expect(view.getViewportState().scale).toBeLessThan(legacyTightFocusTransform.scale);
    });

    it('uses a looser overview zoom before opening a staged deep-detail target', async () => {
        const panelEl = document.getElementById('detailPanel');
        const view = createTransformerView2dDetailView(panelEl);

        const canvas = panelEl.querySelector('.detail-transformer-view2d-canvas');
        const canvasCard = panelEl.querySelector('.detail-transformer-view2d-canvas-card');
        setElementRect(canvas, 960, 600);
        setElementRect(canvasCard, 960, 600);

        const activationSource = createActivationSource();
        const tokenIndices = [0, 1, 2];
        const tokenLabels = ['A', 'B', 'C'];
        const semanticTarget = {
            componentKind: 'mlp',
            layerIndex: 1,
            stage: 'mlp',
            role: 'module'
        };
        const scene = buildTransformerSceneModel({
            activationSource,
            tokenIndices,
            tokenLabels
        });
        const layout = buildSceneLayout(scene, {
            isSmallScreen: false
        });
        const focusBounds = resolveSemanticTargetBounds(layout.registry, {
            ...semanticTarget,
            role: 'module-card'
        });
        const legacyTightFocusTransform = resolveViewportFitTransform(focusBounds, {
            width: 960,
            height: 600
        }, {
            padding: 36,
            minScale: TRANSFORMER_VIEW2D_OVERVIEW_MIN_SCALE_DEFAULT,
            maxScale: 10
        });

        view.setVisible(true);
        view.open({
            activationSource,
            tokenIndices,
            tokenLabels,
            semanticTarget,
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

        await vi.advanceTimersByTimeAsync(700);

        expect(canvas.classList.contains('is-head-detail-scene-active')).toBe(false);
        expect(view.getViewportState().scale).toBeLessThan(legacyTightFocusTransform.scale);
    });

    it('stages scene-backed MLP targets from overview focus into the detail scene', async () => {
        const panelEl = document.getElementById('detailPanel');
        const view = createTransformerView2dDetailView(panelEl);

        const canvas = panelEl.querySelector('.detail-transformer-view2d-canvas');
        const canvasCard = panelEl.querySelector('.detail-transformer-view2d-canvas-card');
        const fitBtn = panelEl.querySelector('[data-transformer-view2d-action="fit-scene"]');
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

        const { layer, stage } = getStageReadouts(panelEl);
        expect(stage?.textContent).toBe(TRANSFORMER_VIEW2D_OVERVIEW_LABEL);
        expect(layer?.hidden).toBe(true);

        await vi.advanceTimersByTimeAsync(420);

        expect(stage?.textContent).toBe(TRANSFORMER_VIEW2D_OVERVIEW_LABEL);
        expect(layer?.hidden).toBe(true);
        expect(canvas.classList.contains('is-head-detail-scene-active')).toBe(false);

        await vi.advanceTimersByTimeAsync(1200);
        expect(stage?.textContent).toBe('Multilayer Perceptron');
        expect(layer?.textContent).toBe('Layer 2');
        expect(layer?.hidden).toBe(false);
        expect(canvas.classList.contains('is-head-detail-scene-active')).toBe(true);
        const focusedDetailScale = view.getViewportState().scale;

        fitBtn?.click();
        await vi.advanceTimersByTimeAsync(500);

        expect(view.getViewportState().scale).toBeLessThan(focusedDetailScale);
    });

    it('smoothly refits within the unobscured width when the docked sidebar opens from a fit state', async () => {
        const panelEl = document.getElementById('detailPanel');
        const view = createTransformerView2dDetailView(panelEl);

        const canvas = panelEl.querySelector('.detail-transformer-view2d-canvas');
        const canvasCard = panelEl.querySelector('.detail-transformer-view2d-canvas-card');
        const selectionSidebar = panelEl.querySelector('.detail-transformer-view2d-selection-sidebar');
        const fitBtn = panelEl.querySelector('[data-transformer-view2d-action="fit-scene"]');
        setElementRect(canvas, 960, 600);
        setElementRect(canvasCard, 960, 600);
        setElementRectAt(selectionSidebar, {
            left: 560,
            top: 56,
            width: 384,
            height: 544
        });

        view.setVisible(true);
        view.open({
            activationSource: createActivationSource(),
            tokenIndices: [0, 1, 2],
            tokenLabels: ['A', 'B', 'C']
        });

        const initialViewportState = view.getViewportState();
        view.setSelectionSidebarVisible(true);

        const animatingViewportState = view.getViewportState();
        expect(animatingViewportState.viewportInsets.right).toBe(384);
        expect(fitBtn?.dataset.fitVisible).toBe('true');

        await vi.advanceTimersByTimeAsync(320);

        const refitViewportState = view.getViewportState();
        expect(refitViewportState.viewportInsets.right).toBe(384);
        expect(refitViewportState.scale).toBeLessThan(initialViewportState.scale);
        expect(refitViewportState.panX).toBeLessThanOrEqual(initialViewportState.panX);
        expect(fitBtn?.dataset.fitVisible).toBe('false');
    });

    it('keeps the current deep-detail zoom when the docked sidebar opens after a focused selection', async () => {
        const panelEl = document.getElementById('detailPanel');
        const view = createTransformerView2dDetailView(panelEl);

        const canvas = panelEl.querySelector('.detail-transformer-view2d-canvas');
        const canvasCard = panelEl.querySelector('.detail-transformer-view2d-canvas-card');
        const selectionSidebar = panelEl.querySelector('.detail-transformer-view2d-selection-sidebar');
        setElementRect(canvas, 960, 600);
        setElementRect(canvasCard, 960, 600);
        setElementRectAt(selectionSidebar, {
            left: 560,
            top: 56,
            width: 384,
            height: 544
        });

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

        await vi.advanceTimersByTimeAsync(1200);

        const focusedViewportState = view.getViewportState();
        view.setSelectionSidebarVisible(true);
        view.resizeAndRender();

        const sidebarViewportState = view.getViewportState();
        expect(sidebarViewportState.viewportInsets.right).toBe(384);
        expect(sidebarViewportState.scale).toBeCloseTo(focusedViewportState.scale, 5);
    });

    it('starts focused opens with the docked sidebar viewport already applied', () => {
        const panelEl = document.getElementById('detailPanel');
        const view = createTransformerView2dDetailView(panelEl);

        const canvas = panelEl.querySelector('.detail-transformer-view2d-canvas');
        const canvasCard = panelEl.querySelector('.detail-transformer-view2d-canvas-card');
        const selectionSidebar = panelEl.querySelector('.detail-transformer-view2d-selection-sidebar');
        setElementRect(canvas, 960, 600);
        setElementRect(canvasCard, 960, 600);
        setElementRectAt(selectionSidebar, {
            left: 560,
            top: 56,
            width: 384,
            height: 544
        });

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
            transitionMode: 'staged-detail',
            initialSelectionSidebarVisible: true
        });

        const initialViewportState = view.getViewportState();
        expect(view.isSelectionSidebarVisible()).toBe(true);
        expect(initialViewportState.viewportInsets.right).toBe(384);

        view.setSelectionSidebarVisible(true);
        view.resizeAndRender();

        const stableViewportState = view.getViewportState();
        expect(stableViewportState.viewportInsets.right).toBe(384);
        expect(stableViewportState.scale).toBeCloseTo(initialViewportState.scale, 5);
        expect(stableViewportState.panX).toBeCloseTo(initialViewportState.panX, 5);
        expect(stableViewportState.panY).toBeCloseTo(initialViewportState.panY, 5);
    });

    it('re-fits the overview after the canvas settles later on open', async () => {
        const panelEl = document.getElementById('detailPanel');
        const view = createTransformerView2dDetailView(panelEl);

        const canvas = panelEl.querySelector('.detail-transformer-view2d-canvas');
        const canvasCard = panelEl.querySelector('.detail-transformer-view2d-canvas-card');
        setElementRect(canvas, 720, 420);
        setElementRect(canvasCard, 720, 420);

        view.setVisible(true);
        view.open({
            activationSource: createActivationSource(),
            tokenIndices: [0, 1, 2],
            tokenLabels: ['A', 'B', 'C']
        });

        const initialViewportState = view.getViewportState();
        expect(initialViewportState.viewport.width).toBe(720);
        expect(initialViewportState.viewport.height).toBe(420);

        setElementRect(canvas, 960, 600);
        setElementRect(canvasCard, 960, 600);

        await vi.advanceTimersByTimeAsync(320);

        const stabilizedViewportState = view.getViewportState();
        expect(stabilizedViewportState.viewport.width).toBe(960);
        expect(stabilizedViewportState.viewport.height).toBe(600);
        expect(stabilizedViewportState.scale).toBeGreaterThan(initialViewportState.scale);
    });

    it('keeps the startup refit alive when the already-visible docked sidebar is shown again', async () => {
        const panelEl = document.getElementById('detailPanel');
        const view = createTransformerView2dDetailView(panelEl);

        const canvas = panelEl.querySelector('.detail-transformer-view2d-canvas');
        const canvasCard = panelEl.querySelector('.detail-transformer-view2d-canvas-card');
        const selectionSidebar = panelEl.querySelector('.detail-transformer-view2d-selection-sidebar');
        setElementRect(canvas, 720, 420);
        setElementRect(canvasCard, 720, 420);
        setElementRectAt(selectionSidebar, {
            left: 336,
            top: 56,
            width: 384,
            height: 364
        });

        view.setVisible(true);
        view.open({
            activationSource: createActivationSource(),
            tokenIndices: [0, 1, 2],
            tokenLabels: ['A', 'B', 'C'],
            initialSelectionSidebarVisible: true
        });

        const initialViewportState = view.getViewportState();
        expect(initialViewportState.viewport.width).toBe(720);
        expect(initialViewportState.viewport.height).toBe(420);
        expect(initialViewportState.viewportInsets.right).toBe(384);

        view.setSelectionSidebarVisible(true);

        setElementRect(canvas, 960, 600);
        setElementRect(canvasCard, 960, 600);
        setElementRectAt(selectionSidebar, {
            left: 576,
            top: 56,
            width: 384,
            height: 544
        });

        await vi.advanceTimersByTimeAsync(320);

        const stabilizedViewportState = view.getViewportState();
        expect(stabilizedViewportState.viewport.width).toBe(960);
        expect(stabilizedViewportState.viewport.height).toBe(600);
        expect(stabilizedViewportState.viewportInsets.right).toBe(384);
        expect(
            Math.abs(stabilizedViewportState.panX - initialViewportState.panX)
        ).toBeGreaterThan(1);
    });

    it('anchors keyboard zoom to the unobscured viewport center when the docked sidebar is visible', () => {
        const panelEl = document.getElementById('detailPanel');
        const view = createTransformerView2dDetailView(panelEl);

        const canvas = panelEl.querySelector('.detail-transformer-view2d-canvas');
        const canvasCard = panelEl.querySelector('.detail-transformer-view2d-canvas-card');
        const selectionSidebar = panelEl.querySelector('.detail-transformer-view2d-selection-sidebar');
        setElementRect(canvas, 960, 600);
        setElementRect(canvasCard, 960, 600);
        setElementRectAt(selectionSidebar, {
            left: 560,
            top: 56,
            width: 384,
            height: 544
        });

        view.setVisible(true);
        view.open({
            activationSource: createActivationSource(),
            tokenIndices: [0, 1, 2],
            tokenLabels: ['A', 'B', 'C']
        });

        view.setSelectionSidebarVisible(true);
        view.resizeAndRender();

        const beforeViewportState = view.getViewportState();
        const effectiveCenterX = (Number(beforeViewportState.viewportInsets?.left) || 0)
            + (
                (
                    (Number(beforeViewportState.viewport?.width) || 0)
                    - (Number(beforeViewportState.viewportInsets?.left) || 0)
                    - (Number(beforeViewportState.viewportInsets?.right) || 0)
                ) * 0.5
            );
        const effectiveCenterY = (Number(beforeViewportState.viewportInsets?.top) || 0)
            + (
                (
                    (Number(beforeViewportState.viewport?.height) || 0)
                    - (Number(beforeViewportState.viewportInsets?.top) || 0)
                    - (Number(beforeViewportState.viewportInsets?.bottom) || 0)
                ) * 0.5
            );
        const worldCenterBefore = {
            x: (effectiveCenterX - beforeViewportState.panX) / beforeViewportState.scale,
            y: (effectiveCenterY - beforeViewportState.panY) / beforeViewportState.scale
        };

        canvasCard.dispatchEvent(new KeyboardEvent('keydown', {
            key: '=',
            bubbles: true,
            cancelable: true
        }));
        canvasCard.dispatchEvent(new KeyboardEvent('keyup', {
            key: '=',
            bubbles: true,
            cancelable: true
        }));

        const afterViewportState = view.getViewportState();
        const worldCenterAfter = {
            x: (effectiveCenterX - afterViewportState.panX) / afterViewportState.scale,
            y: (effectiveCenterY - afterViewportState.panY) / afterViewportState.scale
        };

        expect(afterViewportState.scale).toBeGreaterThan(beforeViewportState.scale);
        expect(worldCenterAfter.x).toBeCloseTo(worldCenterBefore.x, 6);
        expect(worldCenterAfter.y).toBeCloseTo(worldCenterBefore.y, 6);
    });

    it('keeps a stable interaction DPR and leaves the DOM caption overlay visible during detail zoom', async () => {
        const panelEl = document.getElementById('detailPanel');
        const view = createTransformerView2dDetailView(panelEl);
        const { CanvasSceneRenderer } = await import('../view2d/render/canvas/CanvasSceneRenderer.js');
        const renderSpy = vi.spyOn(CanvasSceneRenderer.prototype, 'render');

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
            transitionMode: 'direct'
        });

        const overlay = panelEl.querySelector('.detail-transformer-view2d-caption-overlay');
        const idleRenderArgs = renderSpy.mock.calls.at(-1)?.[0] || null;
        expect(idleRenderArgs?.dprCap).toBeTruthy();
        expect(idleRenderArgs?.dpr).toBe(2);
        expect(overlay?.style.display).toBe('block');

        canvas.dispatchEvent(createWheelEvent({
            clientX: 420,
            clientY: 260,
            deltaY: -120
        }));
        await vi.advanceTimersByTimeAsync(32);

        const interactionRenderArgs = renderSpy.mock.calls.at(-1)?.[0] || null;
        expect(interactionRenderArgs?.dprCap).toBe(idleRenderArgs?.dprCap);
        expect(interactionRenderArgs?.dpr).toBe(idleRenderArgs?.dpr);
        expect(overlay?.style.display).toBe('block');

        await vi.advanceTimersByTimeAsync(220);
        expect(overlay?.style.display).toBe('block');
    });

    it('opens canvas attention-head clicks right after the overview focus zoom settles', async () => {
        const panelEl = document.getElementById('detailPanel');
        const view = createTransformerView2dDetailView(panelEl);
        const { CanvasSceneRenderer } = await import('../view2d/render/canvas/CanvasSceneRenderer.js');

        const canvas = panelEl.querySelector('.detail-transformer-view2d-canvas');
        const canvasCard = panelEl.querySelector('.detail-transformer-view2d-canvas-card');
        setElementRect(canvas, 960, 600);
        setElementRect(canvasCard, 960, 600);

        const headEntry = {
            role: 'head',
            semantic: {
                componentKind: 'mhsa',
                layerIndex: 1,
                headIndex: 2,
                stage: 'attention',
                role: 'head'
            }
        };
        vi.spyOn(CanvasSceneRenderer.prototype, 'resolveInteractiveHitAtScreenPoint').mockReturnValue({
            entry: headEntry,
            node: headEntry
        });

        view.setVisible(true);
        view.open({
            activationSource: createActivationSource(),
            tokenIndices: [0, 1, 2],
            tokenLabels: ['A', 'B', 'C']
        });

        canvas.dispatchEvent(createPointerEvent('pointerdown'));
        canvas.dispatchEvent(createPointerEvent('pointerup'));
        await vi.advanceTimersByTimeAsync(200);

        expect(canvas.classList.contains('is-head-detail-scene-active')).toBe(false);

        await vi.advanceTimersByTimeAsync(
            TRANSFORMER_VIEW2D_STAGED_FOCUS_OVERVIEW_TO_TARGET_DURATION_MS - 200 + 32
        );

        expect(canvas.classList.contains('is-head-detail-scene-active')).toBe(true);
        const openedScale = view.getViewportState().scale;

        await vi.advanceTimersByTimeAsync(240);

        expect(view.getViewportState().scale).toBeCloseTo(openedScale, 6);
    });

    it('opens canvas MLP clicks right after the overview focus zoom settles', async () => {
        const panelEl = document.getElementById('detailPanel');
        const view = createTransformerView2dDetailView(panelEl);
        const { CanvasSceneRenderer } = await import('../view2d/render/canvas/CanvasSceneRenderer.js');

        const canvas = panelEl.querySelector('.detail-transformer-view2d-canvas');
        const canvasCard = panelEl.querySelector('.detail-transformer-view2d-canvas-card');
        setElementRect(canvas, 960, 600);
        setElementRect(canvasCard, 960, 600);

        const mlpEntry = {
            role: 'module-card',
            semantic: {
                componentKind: 'mlp',
                layerIndex: 1,
                stage: 'mlp',
                role: 'module'
            }
        };
        vi.spyOn(CanvasSceneRenderer.prototype, 'resolveInteractiveHitAtScreenPoint').mockReturnValue({
            entry: mlpEntry,
            node: mlpEntry
        });

        view.setVisible(true);
        view.open({
            activationSource: createActivationSource(),
            tokenIndices: [0, 1, 2],
            tokenLabels: ['A', 'B', 'C']
        });

        canvas.dispatchEvent(createPointerEvent('pointerdown'));
        canvas.dispatchEvent(createPointerEvent('pointerup'));
        await vi.advanceTimersByTimeAsync(200);

        expect(canvas.classList.contains('is-head-detail-scene-active')).toBe(false);

        await vi.advanceTimersByTimeAsync(
            TRANSFORMER_VIEW2D_STAGED_FOCUS_OVERVIEW_TO_TARGET_DURATION_MS - 200 + 32
        );

        expect(canvas.classList.contains('is-head-detail-scene-active')).toBe(true);
        const openedScale = view.getViewportState().scale;

        await vi.advanceTimersByTimeAsync(240);

        expect(view.getViewportState().scale).toBeCloseTo(openedScale, 6);
    });

    it('opens canvas vocabulary-embedding clicks as sidebar selections', async () => {
        const panelEl = document.getElementById('detailPanel');
        const onOpenSelection = vi.fn(() => true);
        const view = createTransformerView2dDetailView(panelEl, {
            onOpenSelection
        });
        const { CanvasSceneRenderer } = await import('../view2d/render/canvas/CanvasSceneRenderer.js');

        const canvas = panelEl.querySelector('.detail-transformer-view2d-canvas');
        const canvasCard = panelEl.querySelector('.detail-transformer-view2d-canvas-card');
        setElementRect(canvas, 960, 600);
        setElementRect(canvasCard, 960, 600);

        const embeddingEntry = {
            role: 'vocabulary-embedding-card',
            semantic: {
                componentKind: 'embedding',
                stage: 'embedding.token',
                role: 'vocabulary-embedding-card'
            }
        };
        vi.spyOn(CanvasSceneRenderer.prototype, 'resolveInteractiveHitAtScreenPoint').mockReturnValue({
            entry: embeddingEntry,
            node: embeddingEntry
        });

        view.setVisible(true);
        view.open({
            activationSource: createActivationSource(),
            tokenIndices: [0, 1, 2],
            tokenLabels: ['A', 'B', 'C']
        });

        canvas.dispatchEvent(createPointerEvent('pointerdown'));
        canvas.dispatchEvent(createPointerEvent('pointerup'));

        expect(onOpenSelection).toHaveBeenCalledWith(
            expect.objectContaining({
                label: 'Vocabulary Embedding Matrix'
            })
        );
    });

    it('requires a second touch tap before opening an overview sidebar selection', async () => {
        const panelEl = document.getElementById('detailPanel');
        const onOpenSelection = vi.fn(() => true);
        const view = createTransformerView2dDetailView(panelEl, {
            onOpenSelection
        });
        const { CanvasSceneRenderer } = await import('../view2d/render/canvas/CanvasSceneRenderer.js');

        const canvas = panelEl.querySelector('.detail-transformer-view2d-canvas');
        const canvasCard = panelEl.querySelector('.detail-transformer-view2d-canvas-card');
        setElementRect(canvas, 960, 600);
        setElementRect(canvasCard, 960, 600);

        const embeddingEntry = {
            role: 'vocabulary-embedding-card',
            semantic: {
                componentKind: 'embedding',
                stage: 'embedding.token',
                role: 'vocabulary-embedding-card'
            }
        };
        vi.spyOn(CanvasSceneRenderer.prototype, 'resolveInteractiveHitAtScreenPoint').mockReturnValue({
            entry: embeddingEntry,
            node: embeddingEntry
        });

        view.setVisible(true);
        view.open({
            activationSource: createActivationSource(),
            tokenIndices: [0, 1, 2],
            tokenLabels: ['A', 'B', 'C'],
            isSmallScreen: true
        });

        canvas.dispatchEvent(createPointerEvent('pointerdown', { pointerType: 'touch' }));
        canvas.dispatchEvent(createPointerEvent('pointerup', { pointerType: 'touch' }));

        expect(onOpenSelection).not.toHaveBeenCalled();

        canvas.dispatchEvent(createPointerEvent('pointerdown', { pointerType: 'touch' }));
        canvas.dispatchEvent(createPointerEvent('pointerup', { pointerType: 'touch' }));

        expect(onOpenSelection).toHaveBeenCalledTimes(1);
        expect(onOpenSelection).toHaveBeenCalledWith(
            expect.objectContaining({
                label: 'Vocabulary Embedding Matrix'
            })
        );
    });

    it('opens an overview sidebar selection on the first touch tap on large screens', async () => {
        const panelEl = document.getElementById('detailPanel');
        const onOpenSelection = vi.fn(() => true);
        const view = createTransformerView2dDetailView(panelEl, {
            onOpenSelection
        });
        const { CanvasSceneRenderer } = await import('../view2d/render/canvas/CanvasSceneRenderer.js');

        const canvas = panelEl.querySelector('.detail-transformer-view2d-canvas');
        const canvasCard = panelEl.querySelector('.detail-transformer-view2d-canvas-card');
        setElementRect(canvas, 960, 600);
        setElementRect(canvasCard, 960, 600);

        const embeddingEntry = {
            role: 'vocabulary-embedding-card',
            semantic: {
                componentKind: 'embedding',
                stage: 'embedding.token',
                role: 'vocabulary-embedding-card'
            }
        };
        vi.spyOn(CanvasSceneRenderer.prototype, 'resolveInteractiveHitAtScreenPoint').mockReturnValue({
            entry: embeddingEntry,
            node: embeddingEntry
        });

        view.setVisible(true);
        view.open({
            activationSource: createActivationSource(),
            tokenIndices: [0, 1, 2],
            tokenLabels: ['A', 'B', 'C'],
            isSmallScreen: false
        });

        canvas.dispatchEvent(createPointerEvent('pointerdown', { pointerType: 'touch' }));
        canvas.dispatchEvent(createPointerEvent('pointerup', { pointerType: 'touch' }));

        expect(onOpenSelection).toHaveBeenCalledTimes(1);
        expect(onOpenSelection).toHaveBeenCalledWith(
            expect.objectContaining({
                label: 'Vocabulary Embedding Matrix'
            })
        );
    });

    it('still opens overview sidebar selections on touch after normal finger drift', async () => {
        const panelEl = document.getElementById('detailPanel');
        const onOpenSelection = vi.fn(() => true);
        const view = createTransformerView2dDetailView(panelEl, {
            onOpenSelection
        });
        const { CanvasSceneRenderer } = await import('../view2d/render/canvas/CanvasSceneRenderer.js');

        const canvas = panelEl.querySelector('.detail-transformer-view2d-canvas');
        const canvasCard = panelEl.querySelector('.detail-transformer-view2d-canvas-card');
        setElementRect(canvas, 960, 600);
        setElementRect(canvasCard, 960, 600);

        const embeddingEntry = {
            role: 'vocabulary-embedding-card',
            semantic: {
                componentKind: 'embedding',
                stage: 'embedding.token',
                role: 'vocabulary-embedding-card'
            }
        };
        vi.spyOn(CanvasSceneRenderer.prototype, 'resolveInteractiveHitAtScreenPoint').mockReturnValue({
            entry: embeddingEntry,
            node: embeddingEntry
        });

        view.setVisible(true);
        view.open({
            activationSource: createActivationSource(),
            tokenIndices: [0, 1, 2],
            tokenLabels: ['A', 'B', 'C'],
            isSmallScreen: true
        });

        canvas.dispatchEvent(createPointerEvent('pointerdown', {
            pointerType: 'touch',
            clientX: 120,
            clientY: 180
        }));
        canvas.dispatchEvent(createPointerEvent('pointermove', {
            pointerType: 'touch',
            clientX: 126,
            clientY: 184
        }));
        canvas.dispatchEvent(createPointerEvent('pointerup', {
            pointerType: 'touch',
            clientX: 126,
            clientY: 184
        }));

        expect(onOpenSelection).not.toHaveBeenCalled();

        canvas.dispatchEvent(createPointerEvent('pointerdown', {
            pointerType: 'touch',
            clientX: 122,
            clientY: 182
        }));
        canvas.dispatchEvent(createPointerEvent('pointermove', {
            pointerType: 'touch',
            clientX: 128,
            clientY: 186
        }));
        canvas.dispatchEvent(createPointerEvent('pointerup', {
            pointerType: 'touch',
            clientX: 128,
            clientY: 186
        }));

        expect(onOpenSelection).toHaveBeenCalledTimes(1);
        expect(onOpenSelection).toHaveBeenCalledWith(
            expect.objectContaining({
                label: 'Vocabulary Embedding Matrix'
            })
        );
    });

    it('requires a second touch tap before opening overview head detail on touch', async () => {
        const panelEl = document.getElementById('detailPanel');
        const view = createTransformerView2dDetailView(panelEl);
        const { CanvasSceneRenderer } = await import('../view2d/render/canvas/CanvasSceneRenderer.js');

        const canvas = panelEl.querySelector('.detail-transformer-view2d-canvas');
        const canvasCard = panelEl.querySelector('.detail-transformer-view2d-canvas-card');
        setElementRect(canvas, 960, 600);
        setElementRect(canvasCard, 960, 600);

        const headEntry = {
            role: 'head',
            semantic: {
                componentKind: 'mhsa',
                layerIndex: 1,
                headIndex: 2,
                stage: 'attention',
                role: 'head'
            }
        };
        vi.spyOn(CanvasSceneRenderer.prototype, 'resolveInteractiveHitAtScreenPoint').mockReturnValue({
            entry: headEntry,
            node: headEntry
        });

        view.setVisible(true);
        view.open({
            activationSource: createActivationSource(),
            tokenIndices: [0, 1, 2],
            tokenLabels: ['A', 'B', 'C'],
            isSmallScreen: true
        });

        canvas.dispatchEvent(createPointerEvent('pointerdown', { pointerType: 'touch' }));
        canvas.dispatchEvent(createPointerEvent('pointerup', { pointerType: 'touch' }));
        canvas.dispatchEvent(new Event('pointerleave', {
            bubbles: true
        }));
        await vi.advanceTimersByTimeAsync(32);

        expect(canvas.classList.contains('is-head-detail-scene-active')).toBe(false);

        canvas.dispatchEvent(createPointerEvent('pointerdown', { pointerType: 'touch' }));
        canvas.dispatchEvent(createPointerEvent('pointerup', { pointerType: 'touch' }));
        await vi.advanceTimersByTimeAsync(200);

        expect(canvas.classList.contains('is-head-detail-scene-active')).toBe(false);

        await vi.advanceTimersByTimeAsync(
            TRANSFORMER_VIEW2D_STAGED_FOCUS_OVERVIEW_TO_TARGET_DURATION_MS - 200 + 32
        );

        expect(canvas.classList.contains('is-head-detail-scene-active')).toBe(true);
    });

    it('opens overview head detail on the first touch tap on large screens', async () => {
        const panelEl = document.getElementById('detailPanel');
        const view = createTransformerView2dDetailView(panelEl);
        const { CanvasSceneRenderer } = await import('../view2d/render/canvas/CanvasSceneRenderer.js');

        const canvas = panelEl.querySelector('.detail-transformer-view2d-canvas');
        const canvasCard = panelEl.querySelector('.detail-transformer-view2d-canvas-card');
        setElementRect(canvas, 960, 600);
        setElementRect(canvasCard, 960, 600);

        const headEntry = {
            role: 'head',
            semantic: {
                componentKind: 'mhsa',
                layerIndex: 1,
                headIndex: 2,
                stage: 'attention',
                role: 'head'
            }
        };
        vi.spyOn(CanvasSceneRenderer.prototype, 'resolveInteractiveHitAtScreenPoint').mockReturnValue({
            entry: headEntry,
            node: headEntry
        });

        view.setVisible(true);
        view.open({
            activationSource: createActivationSource(),
            tokenIndices: [0, 1, 2],
            tokenLabels: ['A', 'B', 'C'],
            isSmallScreen: false
        });

        canvas.dispatchEvent(createPointerEvent('pointerdown', { pointerType: 'touch' }));
        canvas.dispatchEvent(createPointerEvent('pointerup', { pointerType: 'touch' }));
        await vi.advanceTimersByTimeAsync(200);

        expect(canvas.classList.contains('is-head-detail-scene-active')).toBe(false);

        await vi.advanceTimersByTimeAsync(
            TRANSFORMER_VIEW2D_STAGED_FOCUS_OVERVIEW_TO_TARGET_DURATION_MS - 200 + 32
        );

        expect(canvas.classList.contains('is-head-detail-scene-active')).toBe(true);
    });

    it('keeps compact-screen overview row selections armed across pointerleave so a second tap opens the sidebar', async () => {
        const panelEl = document.getElementById('detailPanel');
        const onOpenSelection = vi.fn(() => true);
        const view = createTransformerView2dDetailView(panelEl, {
            onOpenSelection
        });
        const { CanvasSceneRenderer } = await import('../view2d/render/canvas/CanvasSceneRenderer.js');

        const canvas = panelEl.querySelector('.detail-transformer-view2d-canvas');
        const canvasCard = panelEl.querySelector('.detail-transformer-view2d-canvas-card');
        setElementRect(canvas, 960, 600);
        setElementRect(canvasCard, 960, 600);

        const activationSource = createActivationSource();
        const tokenIndices = [0, 1, 2];
        const tokenLabels = ['A', 'B', 'C'];
        const scene = buildTransformerSceneModel({
            activationSource,
            tokenIndices,
            tokenLabels,
            layerCount: 1
        });
        const residualNode = flattenSceneNodes(scene).find((node) => (
            node?.kind === VIEW2D_NODE_KINDS.MATRIX
            && node?.role === 'module-card'
            && node?.semantic?.componentKind === 'residual'
            && node?.semantic?.stage === 'incoming'
        ));

        vi.spyOn(CanvasSceneRenderer.prototype, 'resolveInteractiveHitAtScreenPoint').mockReturnValue({
            entry: residualNode,
            node: residualNode,
            rowHit: {
                rowIndex: 1,
                rowItem: residualNode?.rowItems?.[1]
            }
        });

        view.setVisible(true);
        view.open({
            activationSource,
            tokenIndices,
            tokenLabels,
            isSmallScreen: true
        });

        canvas.dispatchEvent(createPointerEvent('pointerdown', {
            pointerType: 'touch',
            clientX: 220,
            clientY: 180
        }));
        canvas.dispatchEvent(createPointerEvent('pointerup', {
            pointerType: 'touch',
            clientX: 220,
            clientY: 180
        }));
        canvas.dispatchEvent(new Event('pointerleave', {
            bubbles: true
        }));

        expect(onOpenSelection).not.toHaveBeenCalled();

        canvas.dispatchEvent(createPointerEvent('pointerdown', {
            pointerType: 'touch',
            clientX: 220,
            clientY: 180
        }));
        canvas.dispatchEvent(createPointerEvent('pointerup', {
            pointerType: 'touch',
            clientX: 220,
            clientY: 180
        }));

        expect(onOpenSelection).toHaveBeenCalledTimes(1);
        expect(onOpenSelection).toHaveBeenCalledWith(
            expect.objectContaining({
                label: 'Residual Stream Vector'
            })
        );
    });

    it('resolves overview residual row clicks from the live viewport transform even when screen-hit state is stale', async () => {
        const panelEl = document.getElementById('detailPanel');
        const onOpenSelection = vi.fn(() => true);
        const view = createTransformerView2dDetailView(panelEl, {
            onOpenSelection
        });
        const { CanvasSceneRenderer } = await import('../view2d/render/canvas/CanvasSceneRenderer.js');

        const canvas = panelEl.querySelector('.detail-transformer-view2d-canvas');
        const canvasCard = panelEl.querySelector('.detail-transformer-view2d-canvas-card');
        setElementRect(canvas, 960, 600);
        setElementRect(canvasCard, 960, 600);

        const activationSource = createActivationSource();
        const tokenIndices = [0, 1, 2];
        const tokenLabels = ['A', 'B', 'C'];
        const scene = buildTransformerSceneModel({
            activationSource,
            tokenIndices,
            tokenLabels
        });
        const layout = buildSceneLayout(scene, {
            isSmallScreen: false
        });
        const residualNode = flattenSceneNodes(scene).find((node) => (
            node?.kind === VIEW2D_NODE_KINDS.MATRIX
            && node?.role === 'module-card'
            && node?.semantic?.componentKind === 'residual'
            && node?.semantic?.stage === 'incoming'
            && node?.semantic?.layerIndex === 0
        ));
        const residualEntry = residualNode ? layout.registry.getNodeEntry(residualNode.id) : null;

        view.setVisible(true);
        view.open({
            activationSource,
            tokenIndices,
            tokenLabels
        });

        const clickPoint = resolveCompactRowScreenPoint(
            residualEntry,
            1,
            view.getViewportState()
        );
        expect(clickPoint).toBeTruthy();

        const staleEmbeddingEntry = {
            role: 'vocabulary-embedding-card',
            semantic: {
                componentKind: 'embedding',
                stage: 'embedding.token',
                role: 'vocabulary-embedding-card'
            }
        };
        vi.spyOn(CanvasSceneRenderer.prototype, 'resolveInteractiveHitAtScreenPoint').mockReturnValue({
            entry: staleEmbeddingEntry,
            node: staleEmbeddingEntry
        });

        canvas.dispatchEvent(createPointerEvent('pointerdown', clickPoint));
        canvas.dispatchEvent(createPointerEvent('pointerup', clickPoint));

        expect(onOpenSelection).toHaveBeenCalledTimes(1);
        expect(onOpenSelection).toHaveBeenCalledWith(
            expect.objectContaining({
                label: 'Residual Stream Vector',
                info: expect.objectContaining({
                    tokenIndex: 1,
                    tokenLabel: 'B'
                })
            })
        );
    });

    it('prefers the more specific screen-space residual row hit when the world-space hit falls back to a broad semantic card', async () => {
        const panelEl = document.getElementById('detailPanel');
        const onOpenSelection = vi.fn(() => true);
        const view = createTransformerView2dDetailView(panelEl, {
            onOpenSelection
        });
        const { CanvasSceneRenderer } = await import('../view2d/render/canvas/CanvasSceneRenderer.js');

        const canvas = panelEl.querySelector('.detail-transformer-view2d-canvas');
        const canvasCard = panelEl.querySelector('.detail-transformer-view2d-canvas-card');
        setElementRect(canvas, 960, 600);
        setElementRect(canvasCard, 960, 600);

        const activationSource = createActivationSource();
        const tokenIndices = [0, 1, 2];
        const tokenLabels = ['A', 'B', 'C'];
        const scene = buildTransformerSceneModel({
            activationSource,
            tokenIndices,
            tokenLabels
        });
        const layout = buildSceneLayout(scene, {
            isSmallScreen: false
        });
        const residualNode = flattenSceneNodes(scene).find((node) => (
            node?.kind === VIEW2D_NODE_KINDS.MATRIX
            && node?.role === 'module-card'
            && node?.semantic?.componentKind === 'residual'
            && node?.semantic?.stage === 'incoming'
            && node?.semantic?.layerIndex === 0
        ));
        const residualEntry = residualNode ? layout.registry.getNodeEntry(residualNode.id) : null;

        view.setVisible(true);
        view.open({
            activationSource,
            tokenIndices,
            tokenLabels
        });

        const clickPoint = resolveCompactRowScreenPoint(
            residualEntry,
            1,
            view.getViewportState()
        );
        expect(clickPoint).toBeTruthy();

        const broadEmbeddingEntry = {
            role: 'vocabulary-embedding-card',
            semantic: {
                componentKind: 'embedding',
                stage: 'embedding.token',
                role: 'vocabulary-embedding-card'
            }
        };
        vi.spyOn(CanvasSceneRenderer.prototype, 'resolveInteractiveHitAtPoint').mockReturnValue({
            entry: broadEmbeddingEntry,
            node: broadEmbeddingEntry
        });
        vi.spyOn(CanvasSceneRenderer.prototype, 'resolveInteractiveHitAtScreenPoint').mockReturnValue({
            entry: residualNode,
            node: residualNode,
            rowHit: {
                rowIndex: 1,
                rowItem: residualNode?.rowItems?.[1]
            }
        });

        canvas.dispatchEvent(createPointerEvent('pointerdown', clickPoint));
        canvas.dispatchEvent(createPointerEvent('pointerup', clickPoint));

        expect(onOpenSelection).toHaveBeenCalledTimes(1);
        expect(onOpenSelection).toHaveBeenCalledWith(
            expect.objectContaining({
                label: 'Residual Stream Vector',
                info: expect.objectContaining({
                    tokenIndex: 1,
                    tokenLabel: 'B'
                })
            })
        );
    });

    it('opens canvas chosen-token chip clicks as sidebar selections', async () => {
        const panelEl = document.getElementById('detailPanel');
        const onOpenSelection = vi.fn(() => true);
        const view = createTransformerView2dDetailView(panelEl, {
            onOpenSelection
        });
        const { CanvasSceneRenderer } = await import('../view2d/render/canvas/CanvasSceneRenderer.js');

        const canvas = panelEl.querySelector('.detail-transformer-view2d-canvas');
        const canvasCard = panelEl.querySelector('.detail-transformer-view2d-canvas-card');
        setElementRect(canvas, 960, 600);
        setElementRect(canvasCard, 960, 600);

        const chosenTokenEntry = {
            role: 'chosen-token-chip',
            semantic: {
                componentKind: 'logits',
                stage: 'output',
                role: 'chosen-token-chip',
                tokenIndex: 2
            },
            metadata: {
                tokenLabel: 'Gamma',
                positionIndex: 3
            }
        };
        vi.spyOn(CanvasSceneRenderer.prototype, 'resolveInteractiveHitAtScreenPoint').mockReturnValue({
            entry: chosenTokenEntry,
            node: chosenTokenEntry
        });

        view.setVisible(true);
        view.open({
            activationSource: createActivationSource(),
            tokenIndices: [0, 1, 2],
            tokenLabels: ['A', 'B', 'Gamma']
        });

        canvas.dispatchEvent(createPointerEvent('pointerdown'));
        canvas.dispatchEvent(createPointerEvent('pointerup'));

        expect(onOpenSelection).toHaveBeenCalledWith(
            expect.objectContaining({
                label: 'Chosen Token: Gamma'
            })
        );
    });

    it('opens with the matching overview residual row already locked from a source selection target', async () => {
        const panelEl = document.getElementById('detailPanel');
        const view = createTransformerView2dDetailView(panelEl);
        const { CanvasSceneRenderer } = await import('../view2d/render/canvas/CanvasSceneRenderer.js');
        const renderSpy = vi.spyOn(CanvasSceneRenderer.prototype, 'render');

        const canvas = panelEl.querySelector('.detail-transformer-view2d-canvas');
        const canvasCard = panelEl.querySelector('.detail-transformer-view2d-canvas-card');
        setElementRect(canvas, 960, 600);
        setElementRect(canvasCard, 960, 600);

        const activationSource = createActivationSource();
        const tokenIndices = [0, 1, 2];
        const tokenLabels = ['A', 'B', 'C'];
        const scene = buildTransformerSceneModel({
            activationSource,
            tokenIndices,
            tokenLabels,
            layerCount: 1
        });
        const residualNode = flattenSceneNodes(scene).find((node) => (
            node?.kind === VIEW2D_NODE_KINDS.MATRIX
            && node?.role === 'module-card'
            && node?.semantic?.componentKind === 'residual'
            && node?.semantic?.stage === 'incoming'
        ));

        view.setVisible(true);
        view.open({
            activationSource,
            tokenIndices,
            tokenLabels,
            semanticTarget: {
                componentKind: 'residual',
                layerIndex: 0,
                stage: 'incoming',
                role: 'module'
            },
            focusLabel: 'Layer 1 residual stream',
            initialOverviewSelectionLockTarget: {
                semanticTarget: {
                    componentKind: 'residual',
                    layerIndex: 0,
                    stage: 'incoming',
                    role: 'module'
                },
                tokenIndex: 1,
                tokenLabel: 'B'
            }
        });
        await vi.advanceTimersByTimeAsync(32);

        expect(view.hasSelectionLock()).toBe(true);
        expect(
            renderSpy.mock.calls.at(-1)?.[0]?.interactionState?.overviewFocusTransition?.currentFocus?.rowSelections
        ).toContainEqual({
            nodeId: residualNode?.id,
            rowIndex: 1
        });
        expect(renderSpy.mock.calls.at(-1)?.[0]?.interactionState?.overviewFocusTransition?.dimStrength).toBe(1);
    });

    it('keeps overview residual selections dimmed until the sidebar selection closes', async () => {
        const panelEl = document.getElementById('detailPanel');
        let view = null;
        const onOpenSelection = vi.fn(() => {
            view?.setSelectionSidebarVisible(true);
            return true;
        });
        const onCloseSelection = vi.fn(() => {
            view?.setSelectionSidebarVisible(false);
            return true;
        });
        view = createTransformerView2dDetailView(panelEl, {
            onOpenSelection,
            onCloseSelection
        });
        const { CanvasSceneRenderer } = await import('../view2d/render/canvas/CanvasSceneRenderer.js');
        const renderSpy = vi.spyOn(CanvasSceneRenderer.prototype, 'render');

        const canvas = panelEl.querySelector('.detail-transformer-view2d-canvas');
        const canvasCard = panelEl.querySelector('.detail-transformer-view2d-canvas-card');
        const selectionSidebar = panelEl.querySelector('.detail-transformer-view2d-selection-sidebar');
        const closeSelectionBtn = panelEl.querySelector('[data-transformer-view2d-action="close-selection"]');
        setElementRect(canvas, 960, 600);
        setElementRect(canvasCard, 960, 600);
        setElementRectAt(selectionSidebar, {
            left: 560,
            top: 56,
            width: 384,
            height: 544
        });

        const activationSource = createActivationSource();
        const tokenIndices = [0, 1, 2];
        const tokenLabels = ['A', 'B', 'C'];
        const scene = buildTransformerSceneModel({
            activationSource,
            tokenIndices,
            tokenLabels,
            layerCount: 1
        });
        const residualNode = flattenSceneNodes(scene).find((node) => (
            node?.kind === VIEW2D_NODE_KINDS.MATRIX
            && node?.role === 'module-card'
            && node?.semantic?.componentKind === 'residual'
            && node?.semantic?.stage === 'incoming'
        ));

        vi.spyOn(CanvasSceneRenderer.prototype, 'resolveInteractiveHitAtScreenPoint').mockReturnValue({
            entry: residualNode,
            node: residualNode,
            rowHit: {
                rowIndex: 1,
                rowItem: residualNode?.rowItems?.[1]
            }
        });

        view.setVisible(true);
        view.open({
            activationSource,
            tokenIndices,
            tokenLabels
        });

        canvas.dispatchEvent(createPointerEvent('pointerdown', {
            clientX: 220,
            clientY: 180
        }));
        canvas.dispatchEvent(createPointerEvent('pointerup', {
            clientX: 220,
            clientY: 180
        }));
        await vi.advanceTimersByTimeAsync(32);

        expect(onOpenSelection).toHaveBeenCalledWith(
            expect.objectContaining({
                label: 'Residual Stream Vector'
            })
        );
        expect(view.hasSelectionLock()).toBe(true);
        expect(
            renderSpy.mock.calls.at(-1)?.[0]?.interactionState?.overviewFocusTransition?.currentFocus?.rowSelections
        ).toContainEqual({
            nodeId: residualNode?.id,
            rowIndex: 1
        });
        expect(renderSpy.mock.calls.at(-1)?.[0]?.interactionState?.overviewFocusTransition?.dimStrength).toBe(1);

        canvas.dispatchEvent(new Event('pointerleave', {
            bubbles: true
        }));
        await vi.advanceTimersByTimeAsync(32);

        expect(view.hasSelectionLock()).toBe(true);
        expect(
            renderSpy.mock.calls.at(-1)?.[0]?.interactionState?.overviewFocusTransition?.currentFocus?.rowSelections
        ).toContainEqual({
            nodeId: residualNode?.id,
            rowIndex: 1
        });

        closeSelectionBtn?.click();
        await vi.advanceTimersByTimeAsync(32);

        expect(onCloseSelection).toHaveBeenCalledTimes(1);
        expect(view.hasSelectionLock()).toBe(false);
        expect(renderSpy.mock.calls.at(-1)?.[0]?.interactionState?.overviewFocusTransition?.currentFocus).toBeNull();
    });

    it('requires a second touch tap before opening a deep-detail sidebar selection', async () => {
        const panelEl = document.getElementById('detailPanel');
        const onOpenSelection = vi.fn(() => true);
        const view = createTransformerView2dDetailView(panelEl, {
            onOpenSelection
        });

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
            transitionMode: 'staged-detail',
            isSmallScreen: true
        });

        await vi.advanceTimersByTimeAsync(1200);

        detailHoverStateOverride = {
            label: 'MLP Up Weight Matrix',
            signature: 'node-a',
            focusState: {
                activeNodeIds: ['node-a']
            }
        };

        canvas.dispatchEvent(createPointerEvent('pointerdown', { pointerType: 'touch' }));
        canvas.dispatchEvent(createPointerEvent('pointerup', { pointerType: 'touch' }));

        expect(onOpenSelection).not.toHaveBeenCalled();
        expect(view.hasSelectionLock()).toBe(false);

        canvas.dispatchEvent(createPointerEvent('pointerdown', { pointerType: 'touch' }));
        canvas.dispatchEvent(createPointerEvent('pointerup', { pointerType: 'touch' }));

        expect(onOpenSelection).toHaveBeenCalledTimes(1);
        expect(onOpenSelection).toHaveBeenCalledWith(
            expect.objectContaining({
                label: 'MLP Up Weight Matrix',
                signature: 'node-a'
            })
        );
        expect(view.hasSelectionLock()).toBe(true);
    });

    it('still opens deep-detail sidebar selections on touch after normal finger drift', async () => {
        const panelEl = document.getElementById('detailPanel');
        const onOpenSelection = vi.fn(() => true);
        const view = createTransformerView2dDetailView(panelEl, {
            onOpenSelection
        });

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
            transitionMode: 'staged-detail',
            isSmallScreen: true
        });

        await vi.advanceTimersByTimeAsync(1200);

        detailHoverStateOverride = {
            label: 'MLP Up Weight Matrix',
            signature: 'node-a',
            focusState: {
                activeNodeIds: ['node-a']
            }
        };

        canvas.dispatchEvent(createPointerEvent('pointerdown', {
            pointerType: 'touch',
            clientX: 280,
            clientY: 210
        }));
        canvas.dispatchEvent(createPointerEvent('pointermove', {
            pointerType: 'touch',
            clientX: 287,
            clientY: 214
        }));
        canvas.dispatchEvent(createPointerEvent('pointerup', {
            pointerType: 'touch',
            clientX: 287,
            clientY: 214
        }));

        expect(onOpenSelection).not.toHaveBeenCalled();
        expect(view.hasSelectionLock()).toBe(false);

        canvas.dispatchEvent(createPointerEvent('pointerdown', {
            pointerType: 'touch',
            clientX: 282,
            clientY: 212
        }));
        canvas.dispatchEvent(createPointerEvent('pointermove', {
            pointerType: 'touch',
            clientX: 289,
            clientY: 216
        }));
        canvas.dispatchEvent(createPointerEvent('pointerup', {
            pointerType: 'touch',
            clientX: 289,
            clientY: 216
        }));

        expect(onOpenSelection).toHaveBeenCalledTimes(1);
        expect(onOpenSelection).toHaveBeenCalledWith(
            expect.objectContaining({
                label: 'MLP Up Weight Matrix',
                signature: 'node-a'
            })
        );
        expect(view.hasSelectionLock()).toBe(true);
    });

    it('binds deep-detail highlight locking and sidebar selection opening to a single click', async () => {
        const panelEl = document.getElementById('detailPanel');
        let view = null;
        const onOpenSelection = vi.fn((selection) => {
            view?.setSelectionSidebarHeaderContent({
                titleHtml: selection?.label || '',
                titleClassName: 'detail-transformer-view2d-selection-sidebar-title detail-title'
            });
            view?.setSelectionSidebarVisible(true);
            return true;
        });
        const onCloseSelection = vi.fn(() => {
            view?.setSelectionSidebarVisible(false);
            return true;
        });
        view = createTransformerView2dDetailView(panelEl, {
            onOpenSelection,
            onCloseSelection
        });

        const canvas = panelEl.querySelector('.detail-transformer-view2d-canvas');
        const canvasCard = panelEl.querySelector('.detail-transformer-view2d-canvas-card');
        const selectionSidebar = panelEl.querySelector('.detail-transformer-view2d-selection-sidebar');
        const closeSelectionBtn = panelEl.querySelector('[data-transformer-view2d-action="close-selection"]');
        setElementRect(canvas, 960, 600);
        setElementRect(canvasCard, 960, 600);
        setElementRectAt(selectionSidebar, {
            left: 560,
            top: 56,
            width: 384,
            height: 544
        });

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

        await vi.advanceTimersByTimeAsync(1200);

        detailHoverStateOverride = {
            label: 'MLP Up Weight Matrix',
            signature: 'node-a',
            focusState: {
                activeNodeIds: ['node-a']
            }
        };
        canvas.dispatchEvent(createPointerEvent('pointerdown'));
        canvas.dispatchEvent(createPointerEvent('pointerup'));

        expect(view.hasSelectionLock()).toBe(true);
        expect(view.isSelectionSidebarVisible()).toBe(true);
        expect(onOpenSelection).toHaveBeenCalledTimes(1);
        expect(onOpenSelection).toHaveBeenLastCalledWith(
            expect.objectContaining({
                label: 'MLP Up Weight Matrix',
                signature: 'node-a'
            })
        );

        detailHoverStateOverride = {
            label: 'MLP Down Weight Matrix',
            signature: 'node-b',
            focusState: {
                activeNodeIds: ['node-b']
            }
        };
        canvas.dispatchEvent(createPointerEvent('pointerdown'));
        canvas.dispatchEvent(createPointerEvent('pointerup'));

        expect(view.hasSelectionLock()).toBe(true);
        expect(view.isSelectionSidebarVisible()).toBe(true);
        expect(onOpenSelection).toHaveBeenCalledTimes(2);
        expect(onOpenSelection).toHaveBeenLastCalledWith(
            expect.objectContaining({
                label: 'MLP Down Weight Matrix',
                signature: 'node-b'
            })
        );

        closeSelectionBtn?.click();

        expect(onCloseSelection).toHaveBeenCalledTimes(1);
        expect(view.hasSelectionLock()).toBe(false);
        expect(view.isSelectionSidebarVisible()).toBe(false);
    });

    it('switches a locked deep-detail highlight even when the mouse click has slight drift', async () => {
        const panelEl = document.getElementById('detailPanel');
        let view = null;
        const onOpenSelection = vi.fn((selection) => {
            view?.setSelectionSidebarHeaderContent({
                titleHtml: selection?.label || '',
                titleClassName: 'detail-transformer-view2d-selection-sidebar-title detail-title'
            });
            view?.setSelectionSidebarVisible(true);
            return true;
        });
        view = createTransformerView2dDetailView(panelEl, {
            onOpenSelection,
            onCloseSelection: vi.fn(() => true)
        });

        const canvas = panelEl.querySelector('.detail-transformer-view2d-canvas');
        const canvasCard = panelEl.querySelector('.detail-transformer-view2d-canvas-card');
        const selectionSidebar = panelEl.querySelector('.detail-transformer-view2d-selection-sidebar');
        setElementRect(canvas, 960, 600);
        setElementRect(canvasCard, 960, 600);
        setElementRectAt(selectionSidebar, {
            left: 560,
            top: 56,
            width: 384,
            height: 544
        });

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

        await vi.advanceTimersByTimeAsync(1200);

        detailHoverStateOverride = {
            label: 'MLP Up Weight Matrix',
            signature: 'node-a',
            focusState: {
                activeNodeIds: ['node-a']
            }
        };
        canvas.dispatchEvent(createPointerEvent('pointerdown', {
            clientX: 240,
            clientY: 180
        }));
        canvas.dispatchEvent(createPointerEvent('pointermove', {
            clientX: 242,
            clientY: 181
        }));
        canvas.dispatchEvent(createPointerEvent('pointerup', {
            clientX: 242,
            clientY: 181
        }));

        expect(onOpenSelection).toHaveBeenCalledTimes(1);
        expect(onOpenSelection).toHaveBeenLastCalledWith(
            expect.objectContaining({
                label: 'MLP Up Weight Matrix',
                signature: 'node-a'
            })
        );
        expect(view.hasSelectionLock()).toBe(true);

        detailHoverStateOverride = {
            label: 'MLP Down Weight Matrix',
            signature: 'node-b',
            focusState: {
                activeNodeIds: ['node-b']
            }
        };
        canvas.dispatchEvent(createPointerEvent('pointerdown', {
            clientX: 286,
            clientY: 214
        }));
        canvas.dispatchEvent(createPointerEvent('pointermove', {
            clientX: 288,
            clientY: 215
        }));
        canvas.dispatchEvent(createPointerEvent('pointerup', {
            clientX: 288,
            clientY: 215
        }));

        expect(onOpenSelection).toHaveBeenCalledTimes(2);
        expect(onOpenSelection).toHaveBeenLastCalledWith(
            expect.objectContaining({
                label: 'MLP Down Weight Matrix',
                signature: 'node-b'
            })
        );
        expect(view.hasSelectionLock()).toBe(true);
        expect(view.isSelectionSidebarVisible()).toBe(true);
    });
});
