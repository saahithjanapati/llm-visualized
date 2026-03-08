// @vitest-environment jsdom

import { beforeEach, afterEach, describe, expect, it, vi } from 'vitest';

const mockState = vi.hoisted(() => ({
    rendererInstances: [],
    resizeObservers: []
}));

vi.mock('three', async (importOriginal) => {
    const actual = await importOriginal();

    class MockWebGLRenderer {
        constructor({ canvas } = {}) {
            this.domElement = canvas || null;
            this.sizeCalls = [];
            this.renderCalls = 0;
            this.toneMapping = 0;
            this.toneMappingExposure = 1;
            this.outputColorSpace = null;
            this.outputEncoding = null;
            mockState.rendererInstances.push(this);
        }

        setPixelRatio(value) {
            this.pixelRatio = value;
        }

        setClearColor(color, alpha) {
            this.clearColor = { color, alpha };
        }

        setSize(width, height) {
            this.sizeCalls.push([width, height]);
            this.width = width;
            this.height = height;
        }

        render() {
            this.renderCalls += 1;
        }

        dispose() {}
    }

    return {
        ...actual,
        WebGLRenderer: MockWebGLRenderer
    };
});

vi.mock('three/examples/jsm/loaders/FontLoader.js', () => ({
    FontLoader: class MockFontLoader {
        load(_url, onLoad) {
            if (typeof onLoad === 'function') onLoad(null);
        }
    }
}));

import * as THREE from 'three';
import { initSelectionPanel } from '../src/ui/selectionPanel.js';

class MockResizeObserver {
    constructor(callback) {
        this.callback = callback;
        mockState.resizeObservers.push(this);
    }

    observe() {}

    disconnect() {}

    unobserve() {}

    trigger() {
        this.callback([], this);
    }
}

function make2dContext() {
    return {
        clearRect() {},
        fillRect() {},
        strokeRect() {},
        beginPath() {},
        moveTo() {},
        lineTo() {},
        stroke() {},
        fill() {},
        arc() {},
        setLineDash() {},
        fillText() {},
        measureText: () => ({ width: 96 }),
        setTransform() {},
        textAlign: 'center',
        textBaseline: 'middle',
        fillStyle: '#fff',
        strokeStyle: '#fff',
        lineWidth: 1,
        font: '12px monospace'
    };
}

function buildPanelDom() {
    document.body.innerHTML = `
        <div id="hudStack">
            <div
                id="detailPanelResizeHandle"
                class="detail-panel-resize-handle"
                role="separator"
                aria-orientation="vertical"
                aria-label="Resize selection panel"
                aria-hidden="true"
                tabindex="-1"
            ></div>
        </div>
        <div id="hudPanel"></div>
        <section id="detailPanel" aria-hidden="true">
            <div class="detail-header">
                <div class="detail-title-group">
                    <div id="detailTitle"></div>
                    <div id="detailSubtitle"></div>
                    <div id="detailSubtitleSecondary"></div>
                </div>
                <div class="detail-header-controls">
                    <button id="detailClose" type="button">Close</button>
                </div>
            </div>
            <div class="detail-preview">
                <canvas id="detailCanvas"></canvas>
            </div>
            <div id="detailVectorLegend" aria-hidden="true">
                <div id="detailVectorLegendBar"></div>
                <span id="detailVectorLegendLow"></span>
                <span id="detailVectorLegendMid"></span>
                <span id="detailVectorLegendHigh"></span>
            </div>
            <div id="detailDescription"></div>
            <div class="detail-copy-context">
                <button id="detailCopyContextBtn" class="detail-copy-context-btn" type="button">
                    <span class="detail-copy-context-copy">
                        <span id="detailCopyContextBtnLabel">Copy context</span>
                        <span id="detailCopyContextBtnAssistant" aria-hidden="true">🤖</span>
                    </span>
                </button>
            </div>
            <section id="detailEquations" aria-hidden="true">
                <div id="detailEquationsBody"></div>
            </section>
            <section id="detailAttention" aria-hidden="true">
                <label class="toggle-row">
                    <input id="detailAttentionToggle" type="checkbox" />
                    <span id="detailAttentionToggleLabel"></span>
                </label>
                <div class="attention-axis-label--left"></div>
                <div class="detail-attention-grid">
                    <div id="detailAttentionTokensTop"></div>
                    <div id="detailAttentionTokensLeft"></div>
                    <div id="detailAttentionMatrix"></div>
                </div>
                <div id="detailAttentionEmpty"></div>
                <div id="detailAttentionNote"></div>
                <div id="detailAttentionValue">
                    <span id="detailAttentionValueSource"></span>
                    <span id="detailAttentionValueTarget"></span>
                    <span id="detailAttentionValueScore"></span>
                </div>
                <div id="detailAttentionLegend">
                    <span class="attention-legend-tick" data-ratio="0"></span>
                    <span class="attention-legend-tick" data-ratio="0.5"></span>
                    <span class="attention-legend-tick" data-ratio="1"></span>
                    <span id="detailAttentionLegendLow"></span>
                    <span id="detailAttentionLegendHigh"></span>
                </div>
            </section>
            <section id="detailMeta">
                <div class="detail-row" id="detailParamsRow"><span id="detailParams"></span></div>
                <div class="detail-row">
                    <span id="detailInputDimLabel"></span>
                    <span id="detailInputDim"></span>
                    <span id="detailInputDimHalf"></span>
                </div>
                <div class="detail-row">
                    <span id="detailOutputDimLabel"></span>
                    <span id="detailOutputDim"></span>
                    <span id="detailOutputDimHalf"></span>
                </div>
                <div class="detail-row" id="detailBiasDimRow"><span id="detailBiasDim"></span></div>
                <div class="detail-token-info" id="detailTokenInfoRow">
                    <span id="detailTokenInfoHeadPrimary"></span>
                    <span id="detailTokenInfoHeadSecondary"></span>
                    <span id="detailTokenInfoHeadTertiary"></span>
                    <span id="detailTokenInfoText"></span>
                    <span id="detailTokenInfoId"></span>
                    <span id="detailTokenInfoPosition"></span>
                </div>
                <div class="detail-row" id="detailTokenEncodingRow">
                    <span id="detailTokenEncodingValue"></span>
                </div>
            </section>
            <section id="detailDataSection">
                <pre id="detailData"></pre>
            </section>
        </section>
    `;

    const canvas = document.getElementById('detailCanvas');
    const hudStack = document.getElementById('hudStack');
    let previewHidden = false;
    const resolveHudWidth = () => {
        const cssWidth = Number.parseFloat(document.documentElement.style.getPropertyValue('--hud-stack-desktop-width'));
        return Number.isFinite(cssWidth) ? cssWidth : 400;
    };
    canvas.getBoundingClientRect = () => ({
        width: previewHidden ? 0 : Math.max(1, resolveHudWidth() - 80),
        height: previewHidden ? 0 : 180,
        top: 0,
        left: 0,
        right: previewHidden ? 0 : Math.max(1, resolveHudWidth() - 80),
        bottom: previewHidden ? 0 : 180
    });
    hudStack.getBoundingClientRect = () => ({
        width: resolveHudWidth(),
        height: 720,
        top: 0,
        left: 624,
        right: 624 + resolveHudWidth(),
        bottom: 720
    });

    return {
        hudStack,
        setPreviewHidden(value) {
            previewHidden = !!value;
        }
    };
}

function makeSelection() {
    const mesh = new THREE.Mesh(
        new THREE.BoxGeometry(1, 1, 1),
        new THREE.MeshBasicMaterial({ color: 0xffffff })
    );
    mesh.userData = { label: 'MLP Up Weight Matrix' };
    return {
        label: 'MLP Up Weight Matrix',
        kind: 'mesh',
        object: mesh,
        hit: { object: mesh }
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

describe('selectionPanel history preview reuse', () => {
    beforeEach(() => {
        mockState.rendererInstances.length = 0;
        mockState.resizeObservers.length = 0;
        vi.useFakeTimers();
        vi.stubGlobal('ResizeObserver', MockResizeObserver);
        vi.stubGlobal('requestAnimationFrame', vi.fn(() => 1));
        vi.stubGlobal('cancelAnimationFrame', vi.fn());
        window.visualViewport = {
            addEventListener() {},
            removeEventListener() {}
        };
        window.matchMedia = vi.fn((query) => ({
            matches: query.includes('orientation: landscape')
                ? true
                : query.includes('max-aspect-ratio: 1/1') || query.includes('max-width: 880px')
                    ? false
                    : false,
            media: query,
            addEventListener() {},
            removeEventListener() {},
            addListener() {},
            removeListener() {},
            dispatchEvent() { return false; }
        }));
        document.fonts = { ready: Promise.resolve() };
        HTMLCanvasElement.prototype.getContext = vi.fn(() => make2dContext());
    });

    afterEach(() => {
        vi.useRealTimers();
        vi.unstubAllGlobals();
        document.body.innerHTML = '';
        document.body.className = '';
        document.documentElement.style.removeProperty('--hud-stack-desktop-width');
    });

    it('forces a resize when returning to a reused preview through history navigation', () => {
        const { setPreviewHidden } = buildPanelDom();
        const panel = initSelectionPanel();
        const selection = makeSelection();
        const backBtn = document.querySelector('.detail-history-btn--back');

        expect(backBtn).toBeTruthy();

        panel.handleSelection(selection);
        vi.advanceTimersByTime(300);

        const renderer = mockState.rendererInstances[0];
        expect(renderer).toBeTruthy();
        expect(renderer.sizeCalls.at(-1)).toEqual([320, 180]);

        const geluBtn = document.querySelector('.detail-description-action-link');
        expect(geluBtn).toBeTruthy();
        geluBtn.click();

        setPreviewHidden(true);
        mockState.resizeObservers[0]?.trigger();
        expect(renderer.sizeCalls.at(-1)).toEqual([1, 1]);

        setPreviewHidden(false);
        expect(backBtn?.disabled).toBe(false);
        backBtn.click();
        vi.advanceTimersByTime(300);

        expect(renderer.sizeCalls.at(-1)).toEqual([320, 180]);
    });

    it('allows touch pointers to resize the selection panel', () => {
        buildPanelDom();
        const panel = initSelectionPanel();
        const selection = makeSelection();
        const resizeHandle = document.getElementById('detailPanelResizeHandle');

        expect(resizeHandle).toBeTruthy();

        panel.handleSelection(selection);
        vi.advanceTimersByTime(300);

        dispatchPointerEvent(resizeHandle, 'pointerdown', {
            clientX: 500,
            pointerId: 7,
            pointerType: 'touch'
        });

        expect(document.body.classList.contains('detail-panel-resizing')).toBe(true);
        expect(document.body.classList.contains('touch-ui')).toBe(true);

        dispatchPointerEvent(window, 'pointermove', {
            clientX: 460,
            pointerId: 7,
            pointerType: 'touch'
        });

        expect(document.documentElement.style.getPropertyValue('--hud-stack-desktop-width')).toBe('440px');

        dispatchPointerEvent(window, 'pointerup', {
            pointerId: 7,
            pointerType: 'touch'
        });

        expect(document.body.classList.contains('detail-panel-resizing')).toBe(false);
    });

    it('keeps the preview drawn during sidebar resize without refitting mid-drag', () => {
        buildPanelDom();
        const panel = initSelectionPanel();
        const selection = makeSelection();
        const resizeHandle = document.getElementById('detailPanelResizeHandle');

        panel.handleSelection(selection);
        vi.advanceTimersByTime(300);

        const renderer = mockState.rendererInstances[0];
        expect(renderer).toBeTruthy();
        const initialRenderCalls = renderer.renderCalls;
        expect(renderer.sizeCalls.at(-1)).toEqual([320, 180]);

        dispatchPointerEvent(resizeHandle, 'pointerdown', {
            clientX: 500,
            pointerId: 9
        });
        dispatchPointerEvent(window, 'pointermove', {
            clientX: 460,
            pointerId: 9
        });
        vi.advanceTimersByTime(300);

        expect(renderer.sizeCalls.at(-1)).toEqual([360, 180]);
        expect(renderer.renderCalls).toBeGreaterThan(initialRenderCalls);
    });

    it('renders attention score subtitle rows with a dedicated detail wrapper next to the score pill', () => {
        buildPanelDom();
        const panel = initSelectionPanel();

        panel.updateData({
            activationSource: {
                getTokenCount: () => 8,
                getTokenString: (index) => (index === 4 ? '"' : `tok-${index}`),
                getTokenId: (index) => 1000 + index
            },
            attentionTokenIndices: [0, 1, 2, 3, 4, 5],
            attentionTokenLabels: ['tok-0', 'tok-1', 'tok-2', 'tok-3', '"', 'tok-5']
        });

        panel.handleSelection({
            label: 'Post-Softmax Attention Score',
            kind: 'attentionSphere',
            info: {
                activationData: {
                    stage: 'attention.post',
                    layerIndex: 4,
                    headIndex: 4,
                    tokenIndex: 4,
                    keyTokenIndex: 4,
                    tokenLabel: '"',
                    keyTokenLabel: '"',
                    postScore: 0.0675
                }
            }
        });

        const subtitleSecondary = document.getElementById('detailSubtitleSecondary');
        const parts = Array.from(subtitleSecondary.querySelectorAll('.detail-attention-context-part'));
        const score = subtitleSecondary.querySelector('.detail-attention-context-score-value');

        expect(subtitleSecondary.classList.contains('detail-subtitle--attention-context')).toBe(true);
        expect(parts).toHaveLength(2);
        for (const part of parts) {
            const detail = part.querySelector('.detail-attention-context-detail');
            expect(detail).toBeTruthy();
            expect(detail?.querySelector('.detail-attention-context-chip')).toBeTruthy();
            expect(detail?.querySelector('.detail-attention-context-position')?.textContent).toBe('(position 5)');
        }
        expect(score?.textContent).toBe('0.0675');
    });
});
