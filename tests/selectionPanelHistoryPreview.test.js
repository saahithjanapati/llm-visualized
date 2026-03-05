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

        render() {}

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
        <div id="hudStack"></div>
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
                    <span id="detailCopyContextBtnLabel">Copy context</span>
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
    let previewHidden = false;
    canvas.getBoundingClientRect = () => ({
        width: previewHidden ? 0 : 320,
        height: previewHidden ? 0 : 180,
        top: 0,
        left: 0,
        right: previewHidden ? 0 : 320,
        bottom: previewHidden ? 0 : 180
    });

    return {
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
        document.fonts = { ready: Promise.resolve() };
        HTMLCanvasElement.prototype.getContext = vi.fn(() => make2dContext());
    });

    afterEach(() => {
        vi.useRealTimers();
        vi.unstubAllGlobals();
        document.body.innerHTML = '';
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
});
