// @vitest-environment jsdom

import { describe, expect, it, vi } from 'vitest';

import { D_HEAD, D_MODEL, FINAL_MLP_COLOR } from './selectionPanelConstants.js';
import { createTransformerView2dResidualCaptionOverlay, resolveCaptionScreenExtent } from './transformerView2dResidualCaptionOverlay.js';
import { buildSceneLayout } from '../view2d/layout/buildSceneLayout.js';
import { createMhsaDetailSceneIndex, resolveMhsaDetailHoverState } from '../view2d/mhsaDetailInteraction.js';
import { buildMhsaSceneModel } from '../view2d/model/buildMhsaSceneModel.js';
import { buildHeadDetailSceneModel } from '../view2d/model/buildHeadDetailSceneModel.js';
import { buildLayerNormDetailSceneModel } from '../view2d/model/buildLayerNormDetailSceneModel.js';
import { buildMlpDetailSceneModel } from '../view2d/model/buildMlpDetailSceneModel.js';
import { buildOutputProjectionDetailSceneModel } from '../view2d/model/buildOutputProjectionDetailSceneModel.js';
import { flattenSceneNodes } from '../view2d/schema/sceneTypes.js';

const DEFAULT_TOKEN_LABELS = ['Token A', 'Token B'];
const MHSA_UNIFORM_MIN_SCREEN_HEIGHT_PX = 28;
const ATTENTION_MATRIX_LABEL_MIN_SCREEN_FONT_PX = 14;
const BIAS_LABEL_MIN_SCREEN_FONT_PX = 13;
const MHSA_STANDARD_LABEL_MIN_SCREEN_FONT_PX = 14.5;
const MHSA_STANDARD_LABEL_MAX_SCREEN_FONT_PX = 15.25;
const MHSA_STANDARD_DIMENSIONS_MIN_SCREEN_FONT_PX = 11.75;
const MHSA_STANDARD_DIMENSIONS_MAX_SCREEN_FONT_PX = 12.5;
function createVectorValues(seed = 0) {
    return Array.from({ length: D_HEAD }, (_, index) => Number((seed + (index * 0.01)).toFixed(4)));
}

function createResidualValues(seed = 0) {
    return Array.from({ length: D_MODEL }, (_, index) => Number((seed + (index * 0.004)).toFixed(4)));
}

function toKatexColorHex(hex = 0xFFFFFF) {
    return `#${Math.max(0, Math.min(0xFFFFFF, Math.floor(
        Number.isFinite(hex) ? hex : 0xFFFFFF
    ))).toString(16).padStart(6, '0')}`;
}

function createBaseRows(tokenLabels = DEFAULT_TOKEN_LABELS) {
    return tokenLabels.map((tokenLabel, rowIndex) => ({
        rowIndex,
        tokenIndex: rowIndex,
        tokenLabel,
        rawValues: createVectorValues(rowIndex),
        gradientCss: `rgba(${120 + (rowIndex * 16)}, 220, 255, 0.9)`
    }));
}

function createProjectionOutputRows(label = 'Q', tokenLabels = DEFAULT_TOKEN_LABELS) {
    return tokenLabels.map((tokenLabel, rowIndex) => ({
        rowIndex,
        tokenIndex: rowIndex,
        tokenLabel,
        rawValue: Number((rowIndex + 0.25).toFixed(3)),
        rawValues: createVectorValues(rowIndex + 1),
        gradientCss: `rgba(${180 - (rowIndex * 20)}, ${140 + (rowIndex * 24)}, 255, 0.88)`,
        title: `${tokenLabel}: ${label} vector`
    }));
}

function createGridRows(tokenLabels = DEFAULT_TOKEN_LABELS, fillCss = 'rgba(255, 255, 255, 0.28)') {
    return tokenLabels.map((tokenLabel, rowIndex) => ({
        rowIndex,
        tokenLabel,
        cells: tokenLabels.map((colLabel, colIndex) => ({
            rowIndex,
            colIndex,
            rowTokenLabel: tokenLabel,
            colTokenLabel: colLabel,
            rawValue: Number(((rowIndex + 1) * (colIndex + 1) * 0.125).toFixed(3)),
            fillCss,
            isMasked: false,
            isEmpty: false,
            title: `${tokenLabel} -> ${colLabel}`
        })),
        hasAnyValue: true
    }));
}

function createPreviewData({
    tokenLabels = DEFAULT_TOKEN_LABELS
} = {}) {
    const rows = createBaseRows(tokenLabels);
    const queryOutputRows = createProjectionOutputRows('Q', tokenLabels);
    const keyOutputRows = createProjectionOutputRows('K', tokenLabels);
    const valueOutputRows = createProjectionOutputRows('V', tokenLabels);

    const createProjection = (kind, outputLabelTex, outputRows) => ({
        kind,
        weightLabelTex: `W_${kind.toLowerCase()}`,
        biasLabelTex: `b_${kind.toLowerCase()}`,
        outputLabelTex,
        weightRowCount: D_MODEL,
        weightColumnCount: D_HEAD,
        biasValue: 0.15,
        biasVectorGradientCss: 'rgba(255, 255, 255, 0.2)',
        outputRowCount: outputRows.length,
        outputColumnCount: D_HEAD,
        outputRows
    });

    const attentionGridRows = createGridRows(tokenLabels);
    const postGridRows = createGridRows(tokenLabels, 'rgba(160, 220, 255, 0.34)');

    return {
        rowCount: rows.length,
        columnCount: D_MODEL,
        bandCount: 12,
        sampleStep: 64,
        rows,
        projections: [
            createProjection('Q', 'Q', queryOutputRows),
            createProjection('K', 'K', keyOutputRows),
            createProjection('V', 'V', valueOutputRows)
        ],
        attentionScoreStage: {
            queryLabelTex: 'Q',
            queryRowCount: queryOutputRows.length,
            queryColumnCount: D_HEAD,
            queryRows: queryOutputRows,
            transposeLabelTex: 'K^{\\mathsf{T}}',
            transposeRowCount: D_HEAD,
            transposeColumnCount: keyOutputRows.length,
            transposeColumns: keyOutputRows.map((rowData) => ({
                colIndex: rowData.rowIndex,
                tokenIndex: rowData.tokenIndex,
                rawValue: rowData.rawValue,
                rawValues: rowData.rawValues,
                fillCss: rowData.gradientCss,
                tokenLabel: rowData.tokenLabel
            })),
            scaleLabelTex: '\\sqrt{d_{\\mathrm{head}}}',
            outputLabelTex: 'A_{\\mathrm{pre}}',
            outputRowCount: attentionGridRows.length,
            outputColumnCount: attentionGridRows.length,
            outputRows: attentionGridRows,
            maskLabelTex: 'M_{\\mathrm{causal}}',
            maskRows: attentionGridRows,
            softmaxLabelTex: '\\mathrm{softmax}',
            postLabelTex: 'A_{\\mathrm{post}}',
            postRowCount: postGridRows.length,
            postColumnCount: postGridRows.length,
            postRows: postGridRows,
            valueLabelTex: 'V',
            valueRowCount: valueOutputRows.length,
            valueColumnCount: D_HEAD,
            valueRows: valueOutputRows,
            headOutputLabelTex: 'H_i',
            headOutputRowCount: valueOutputRows.length,
            headOutputColumnCount: D_HEAD,
            headOutputRows: valueOutputRows
        }
    };
}

function buildMhsaFixtures({
    tokenLabels = DEFAULT_TOKEN_LABELS,
    kvCacheState = null,
    canvasWidth = 2200,
    canvasHeight = 1400
} = {}) {
    const scene = buildMhsaSceneModel({
        previewData: createPreviewData({ tokenLabels }),
        layerIndex: 2,
        headIndex: 1,
        kvCacheState
    });
    const layout = buildSceneLayout(scene);
    const nodes = flattenSceneNodes(scene);
    const weightNode = nodes.find((node) => (
        node.role === 'projection-weight'
        && String(node.metadata?.kind || '').toLowerCase() === 'k'
    )) || null;
    const xLnNode = nodes.find((node) => (
        node.role === 'x-ln-copy'
        && String(node.semantic?.branchKey || '').toLowerCase() === 'k'
    )) || null;
    const qBiasNode = nodes.find((node) => (
        node.role === 'projection-bias'
        && String(node.metadata?.kind || '').toLowerCase() === 'q'
    )) || null;
    const biasNode = nodes.find((node) => (
        node.role === 'projection-bias'
        && String(node.metadata?.kind || '').toLowerCase() === 'k'
    )) || null;
    const kOutputNode = nodes.find((node) => (
        node.role === 'projection-output'
        && String(node.metadata?.kind || '').toLowerCase() === 'k'
    )) || null;
    const vOutputNode = nodes.find((node) => (
        node.role === 'projection-output'
        && String(node.metadata?.kind || '').toLowerCase() === 'v'
    )) || null;
    const kOutputCopyNode = nodes.find((node) => (
        node.role === 'projection-output-copy'
        && String(node.metadata?.kind || '').toLowerCase() === 'k'
    )) || null;
    const vOutputCopyNode = nodes.find((node) => (
        node.role === 'projection-output-copy'
        && String(node.metadata?.kind || '').toLowerCase() === 'v'
    )) || null;
    const outputNode = kOutputNode;
    const projectionStackNode = nodes.find((node) => node.role === 'projection-stack') || null;
    const qStageNode = nodes.find((node) => (
        node.role === 'projection-stage'
        && String(node.metadata?.kind || '').toLowerCase() === 'q'
    )) || null;
    const kStageNode = nodes.find((node) => (
        node.role === 'projection-stage'
        && String(node.metadata?.kind || '').toLowerCase() === 'k'
    )) || null;
    const vStageNode = nodes.find((node) => (
        node.role === 'projection-stage'
        && String(node.metadata?.kind || '').toLowerCase() === 'v'
    )) || null;
    const queryNode = nodes.find((node) => node.role === 'attention-query-source') || null;
    const transposeNode = nodes.find((node) => node.role === 'attention-key-transpose') || null;
    const preScoreNode = nodes.find((node) => node.role === 'attention-pre-score') || null;
    const maskedInputNode = nodes.find((node) => node.role === 'attention-masked-input') || null;
    const maskNode = nodes.find((node) => node.role === 'attention-mask') || null;
    const postNode = nodes.find((node) => node.role === 'attention-post') || null;
    const valuePostNode = nodes.find((node) => node.role === 'attention-value-post') || null;
    const headOutputNode = nodes.find((node) => node.role === 'attention-head-output') || null;
    const softmaxLabelNode = nodes.find((node) => node.role === 'attention-softmax-label') || null;
    const scaleNode = nodes.find((node) => node.role === 'attention-scale') || null;
    const kCacheNode = nodes.find((node) => (
        node.role === 'projection-cache'
        && String(node.semantic?.branchKey || '').toLowerCase() === 'k'
    )) || null;
    const kCacheSourceNode = nodes.find((node) => (
        node.role === 'projection-cache-source'
        && String(node.semantic?.branchKey || node.metadata?.kind || '').toLowerCase() === 'k'
    )) || null;
    const vCacheNode = nodes.find((node) => (
        node.role === 'projection-cache'
        && String(node.semantic?.branchKey || '').toLowerCase() === 'v'
    )) || null;
    const vCacheSourceNode = nodes.find((node) => (
        node.role === 'projection-cache-source'
        && String(node.semantic?.branchKey || node.metadata?.kind || '').toLowerCase() === 'v'
    )) || null;
    const kCacheSourceConnectorNode = nodes.find((node) => node.role === 'connector-k-cache-source') || null;
    const kCacheConnectorNode = nodes.find((node) => node.role === 'connector-k-cache') || null;
    const vCacheSourceConnectorNode = nodes.find((node) => node.role === 'connector-v-cache-source') || null;
    const vCacheConnectorNode = nodes.find((node) => node.role === 'connector-v-cache') || null;

    const parent = document.createElement('div');
    document.body.appendChild(parent);
    const canvas = document.createElement('canvas');
    parent.appendChild(canvas);
    Object.defineProperties(canvas, {
        clientWidth: { configurable: true, value: canvasWidth },
        clientHeight: { configurable: true, value: canvasHeight },
        offsetLeft: { configurable: true, value: 0 },
        offsetTop: { configurable: true, value: 0 }
    });
    canvas.width = canvasWidth;
    canvas.height = canvasHeight;

    const overlay = createTransformerView2dResidualCaptionOverlay({
        documentRef: document,
        parent
    });

    return {
        scene,
        layout,
        weightNode,
        xLnNode,
        qBiasNode,
        biasNode,
        kOutputNode,
        vOutputNode,
        kOutputCopyNode,
        vOutputCopyNode,
        outputNode,
        projectionStackNode,
        qStageNode,
        kStageNode,
        vStageNode,
        queryNode,
        transposeNode,
        preScoreNode,
        maskedInputNode,
        maskNode,
        postNode,
        valuePostNode,
        headOutputNode,
        softmaxLabelNode,
        scaleNode,
        kCacheNode,
        kCacheSourceNode,
        vCacheNode,
        vCacheSourceNode,
        kCacheSourceConnectorNode,
        kCacheConnectorNode,
        vCacheSourceConnectorNode,
        vCacheConnectorNode,
        canvas,
        overlay,
        cleanup() {
            overlay.destroy();
            parent.remove();
        }
    };
}

function buildHeadDetailFixtures({
    tokenLabels = DEFAULT_TOKEN_LABELS,
    canvasWidth = 2200,
    canvasHeight = 1400
} = {}) {
    const scene = buildHeadDetailSceneModel({
        headDetailPreview: {
            rowItems: createBaseRows(tokenLabels)
        },
        headDetailTarget: {
            layerIndex: 2,
            headIndex: 1
        }
    });
    const layout = buildSceneLayout(scene);
    const nodes = flattenSceneNodes(scene);
    const qCopyNode = nodes.find((node) => (
        node.role === 'x-ln-copy'
        && String(node.semantic?.branchKey || '').toLowerCase() === 'q'
    )) || null;

    const parent = document.createElement('div');
    document.body.appendChild(parent);
    const canvas = document.createElement('canvas');
    parent.appendChild(canvas);
    Object.defineProperties(canvas, {
        clientWidth: { configurable: true, value: canvasWidth },
        clientHeight: { configurable: true, value: canvasHeight },
        offsetLeft: { configurable: true, value: 0 },
        offsetTop: { configurable: true, value: 0 }
    });
    canvas.width = canvasWidth;
    canvas.height = canvasHeight;

    const overlay = createTransformerView2dResidualCaptionOverlay({
        documentRef: document,
        parent
    });

    return {
        scene,
        layout,
        qCopyNode,
        canvas,
        overlay,
        cleanup() {
            overlay.destroy();
            parent.remove();
        }
    };
}

function buildOutputProjectionFixtures({
    tokenLabels = DEFAULT_TOKEN_LABELS,
    canvasWidth = 2200,
    canvasHeight = 1400
} = {}) {
    const tokenRefs = tokenLabels.map((tokenLabel, rowIndex) => ({
        rowIndex,
        tokenIndex: rowIndex,
        tokenLabel
    }));
    const scene = buildOutputProjectionDetailSceneModel({
        activationSource: {
            getAttentionWeightedSum(_layerIndex = 0, headIndex = 0, tokenIndex = 0, targetLength = D_HEAD) {
                return Array.from({ length: targetLength }, (_, index) => Number(
                    ((headIndex * 0.15) + (tokenIndex * 0.07) + (index * 0.01)).toFixed(4)
                ));
            },
            getAttentionOutputProjection(_layerIndex = 0, tokenIndex = 0, targetLength = D_MODEL) {
                return Array.from({ length: targetLength }, (_, index) => Number(
                    ((tokenIndex * 0.05) + (index * 0.004)).toFixed(4)
                ));
            }
        },
        outputProjectionDetailTarget: {
            layerIndex: 2
        },
        tokenRefs
    });
    const layout = buildSceneLayout(scene);
    const nodes = flattenSceneNodes(scene);
    const headMatrixNode = nodes.find((node) => (
        node.role === 'head-output-matrix'
        && Number(node.semantic?.headIndex) === 0
    )) || null;
    const projectionWeightNode = nodes.find((node) => node.role === 'projection-weight') || null;
    const projectionOutputNode = nodes.find((node) => node.role === 'projection-output') || null;

    const parent = document.createElement('div');
    document.body.appendChild(parent);
    const canvas = document.createElement('canvas');
    parent.appendChild(canvas);
    Object.defineProperties(canvas, {
        clientWidth: { configurable: true, value: canvasWidth },
        clientHeight: { configurable: true, value: canvasHeight },
        offsetLeft: { configurable: true, value: 0 },
        offsetTop: { configurable: true, value: 0 }
    });
    canvas.width = canvasWidth;
    canvas.height = canvasHeight;

    const overlay = createTransformerView2dResidualCaptionOverlay({
        documentRef: document,
        parent
    });

    return {
        scene,
        layout,
        headMatrixNode,
        projectionWeightNode,
        projectionOutputNode,
        canvas,
        overlay,
        cleanup() {
            overlay.destroy();
            parent.remove();
        }
    };
}

function buildMlpDetailFixtures({
    tokenLabels = DEFAULT_TOKEN_LABELS,
    canvasWidth = 2200,
    canvasHeight = 1400
} = {}) {
    const tokenRefs = tokenLabels.map((tokenLabel, rowIndex) => ({
        rowIndex,
        tokenIndex: rowIndex,
        tokenLabel
    }));
    const scene = buildMlpDetailSceneModel({
        activationSource: {
            getLayerLn2(_layerIndex = 0, _kind = 'shift', tokenIndex = 0, targetLength = D_MODEL) {
                return createResidualValues(tokenIndex * 0.1).slice(0, targetLength);
            }
        },
        mlpDetailTarget: {
            layerIndex: 2
        },
        tokenRefs
    });
    const layout = buildSceneLayout(scene);
    const nodes = flattenSceneNodes(scene);
    const inputNode = nodes.find((node) => node.role === 'projection-source-xln') || null;
    const downOutputNode = nodes.find((node) => node.role === 'mlp-down-output') || null;

    const parent = document.createElement('div');
    document.body.appendChild(parent);
    const canvas = document.createElement('canvas');
    parent.appendChild(canvas);
    Object.defineProperties(canvas, {
        clientWidth: { configurable: true, value: canvasWidth },
        clientHeight: { configurable: true, value: canvasHeight },
        offsetLeft: { configurable: true, value: 0 },
        offsetTop: { configurable: true, value: 0 }
    });
    canvas.width = canvasWidth;
    canvas.height = canvasHeight;

    const overlay = createTransformerView2dResidualCaptionOverlay({
        documentRef: document,
        parent
    });

    return {
        scene,
        layout,
        inputNode,
        downOutputNode,
        canvas,
        overlay,
        cleanup() {
            overlay.destroy();
            parent.remove();
        }
    };
}

function buildLayerNormDetailFixtures({
    tokenLabels = DEFAULT_TOKEN_LABELS,
    canvasWidth = 2200,
    canvasHeight = 1400
} = {}) {
    const tokenRefs = tokenLabels.map((tokenLabel, rowIndex) => ({
        rowIndex,
        tokenIndex: rowIndex,
        tokenLabel
    }));
    const scene = buildLayerNormDetailSceneModel({
        activationSource: {
            getLayerIncoming(_layerIndex = 0, tokenIndex = 0, targetLength = D_MODEL) {
                return createResidualValues(tokenIndex * 0.1).slice(0, targetLength);
            },
            getLayerLn1(_layerIndex = 0, stage = 'norm', tokenIndex = 0, targetLength = D_MODEL) {
                const stageSeed = stage === 'scale' ? 0.2 : (stage === 'shift' ? 0.3 : 0.1);
                return createResidualValues(stageSeed + (tokenIndex * 0.1)).slice(0, targetLength);
            }
        },
        layerNormDetailTarget: {
            layerNormKind: 'ln1',
            layerIndex: 2
        },
        tokenRefs,
        layerCount: 12
    });
    const layout = buildSceneLayout(scene);
    const nodes = flattenSceneNodes(scene);
    const inputNode = nodes.find((node) => node.role === 'layer-norm-input') || null;
    const scaleNode = nodes.find((node) => node.role === 'layer-norm-scale') || null;
    const shiftNode = nodes.find((node) => node.role === 'layer-norm-shift') || null;

    const parent = document.createElement('div');
    document.body.appendChild(parent);
    const canvas = document.createElement('canvas');
    parent.appendChild(canvas);
    Object.defineProperties(canvas, {
        clientWidth: { configurable: true, value: canvasWidth },
        clientHeight: { configurable: true, value: canvasHeight },
        offsetLeft: { configurable: true, value: 0 },
        offsetTop: { configurable: true, value: 0 }
    });
    canvas.width = canvasWidth;
    canvas.height = canvasHeight;

    const overlay = createTransformerView2dResidualCaptionOverlay({
        documentRef: document,
        parent
    });

    return {
        scene,
        layout,
        inputNode,
        scaleNode,
        shiftNode,
        canvas,
        overlay,
        cleanup() {
            overlay.destroy();
            parent.remove();
        }
    };
}

function resolveThresholdScale(entry = null) {
    const extent = resolveCaptionScreenExtent({
        captionPosition: 'bottom',
        projectedContentWidth: entry?.contentBounds?.width || 0,
        projectedContentHeight: entry?.contentBounds?.height || 0,
        scaleWithNode: false
    });
    return {
        extent,
        scale: MHSA_UNIFORM_MIN_SCREEN_HEIGHT_PX / Math.max(0.0001, extent)
    };
}

function queryCaptionItem(nodeId = '') {
    return document.querySelector(`[data-node-id="${nodeId}"]`);
}

describe('transformerView2dResidualCaptionOverlay', () => {
    it('keeps K/V cache branches active during KV-cache prefill and decode', () => {
        const prefillFixtures = buildMhsaFixtures({
            kvCacheState: {
                kvCacheModeEnabled: true,
                kvCachePrefillActive: true
            }
        });
        const decodeFixtures = buildMhsaFixtures({
            kvCacheState: {
                kvCacheModeEnabled: true,
                kvCachePrefillActive: false,
                kvCacheDecodeActive: true,
                kvCachePassIndex: 1
            }
        });
        const disabledFixtures = buildMhsaFixtures();

        try {
            expect(prefillFixtures.kCacheNode).toBeTruthy();
            expect(prefillFixtures.vCacheNode).toBeTruthy();
            expect(prefillFixtures.kCacheConnectorNode).toBeTruthy();
            expect(prefillFixtures.vCacheConnectorNode).toBeTruthy();
            expect(prefillFixtures.kCacheNode?.visual?.opacity ?? 1).toBeCloseTo(1, 3);
            expect(prefillFixtures.vCacheNode?.visual?.opacity ?? 1).toBeCloseTo(1, 3);
            expect(prefillFixtures.kCacheNode?.label?.tex).toContain('\\mathrm{cache}');
            expect(prefillFixtures.vCacheNode?.label?.tex).toContain('\\mathrm{cache}');
            expect(prefillFixtures.kCacheConnectorNode?.target?.anchor).toBe('left');
            expect(prefillFixtures.vCacheConnectorNode?.target?.anchor).toBe('left');
            expect(prefillFixtures.kCacheConnectorNode?.metadata?.sourceAnchorMode).toBe('caption-bottom');
            expect(prefillFixtures.vCacheConnectorNode?.metadata?.sourceAnchorMode).toBe('caption-bottom');
            expect(prefillFixtures.kCacheNode?.metadata?.kvCachePhase).toBe('prefill');
            expect(prefillFixtures.vCacheNode?.metadata?.kvCachePhase).toBe('prefill');

            expect(decodeFixtures.kCacheNode).toBeTruthy();
            expect(decodeFixtures.vCacheNode).toBeTruthy();
            expect(decodeFixtures.kCacheSourceNode).toBeTruthy();
            expect(decodeFixtures.vCacheSourceNode).toBeTruthy();
            expect(decodeFixtures.kCacheSourceConnectorNode).toBeTruthy();
            expect(decodeFixtures.vCacheSourceConnectorNode).toBeTruthy();
            expect(decodeFixtures.kCacheConnectorNode).toBeNull();
            expect(decodeFixtures.vCacheConnectorNode).toBeNull();
            expect(decodeFixtures.kCacheNode?.visual?.opacity ?? 1).toBeCloseTo(1, 3);
            expect(decodeFixtures.vCacheNode?.visual?.opacity ?? 1).toBeCloseTo(1, 3);
            expect(decodeFixtures.kCacheSourceNode?.visual?.opacity ?? 1).toBeCloseTo(1, 3);
            expect(decodeFixtures.vCacheSourceNode?.visual?.opacity ?? 1).toBeCloseTo(1, 3);
            expect(decodeFixtures.kCacheNode?.metadata?.kvCachePhase).toBe('decode');
            expect(decodeFixtures.vCacheNode?.metadata?.kvCachePhase).toBe('decode');
            expect(decodeFixtures.kCacheSourceNode?.metadata?.kvCachePhase).toBe('decode');
            expect(decodeFixtures.vCacheSourceNode?.metadata?.kvCachePhase).toBe('decode');
            expect(decodeFixtures.kCacheNode?.rowItems).toHaveLength(1);
            expect(decodeFixtures.vCacheNode?.rowItems).toHaveLength(1);
            expect(decodeFixtures.kCacheSourceNode?.rowItems).toHaveLength(1);
            expect(decodeFixtures.vCacheSourceNode?.rowItems).toHaveLength(1);
            expect(decodeFixtures.kCacheSourceConnectorNode?.source?.nodeId).toBe(decodeFixtures.kCacheSourceNode?.id);
            expect(decodeFixtures.vCacheSourceConnectorNode?.source?.nodeId).toBe(decodeFixtures.vCacheSourceNode?.id);
            expect(decodeFixtures.kCacheSourceConnectorNode?.target?.nodeId).toBe(decodeFixtures.kCacheNode?.id);
            expect(decodeFixtures.vCacheSourceConnectorNode?.target?.nodeId).toBe(decodeFixtures.vCacheNode?.id);
            expect(decodeFixtures.kCacheSourceConnectorNode?.source?.anchor).toBe('right');
            expect(decodeFixtures.vCacheSourceConnectorNode?.source?.anchor).toBe('right');
            expect(decodeFixtures.kCacheSourceConnectorNode?.target?.anchor).toBe('top');
            expect(decodeFixtures.vCacheSourceConnectorNode?.target?.anchor).toBe('top');
            expect(decodeFixtures.kOutputCopyNode).toBeTruthy();
            expect(decodeFixtures.vOutputCopyNode).toBeTruthy();

            expect(disabledFixtures.kCacheNode).toBeNull();
            expect(disabledFixtures.vCacheNode).toBeNull();
        } finally {
            prefillFixtures.cleanup();
            decodeFixtures.cleanup();
            disabledFixtures.cleanup();
        }
    });

    it('keeps prefill cache captions fully visible', () => {
        const fixtures = buildMhsaFixtures({
            kvCacheState: {
                kvCacheModeEnabled: true,
                kvCachePrefillActive: true
            }
        });

        try {
            const cacheNode = fixtures.kCacheNode;
            const cacheEntry = fixtures.layout.registry.getNodeEntry(cacheNode?.id || '');
            const visibleScale = resolveThresholdScale(cacheEntry).scale * 1.1;
            const projectBounds = (bounds) => ({
                x: bounds.x,
                y: bounds.y,
                width: bounds.width * visibleScale,
                height: bounds.height * visibleScale
            });

            fixtures.overlay.sync({
                scene: fixtures.scene,
                layout: fixtures.layout,
                canvas: fixtures.canvas,
                projectBounds,
                visible: true,
                enabled: true
            });

            const cacheItem = queryCaptionItem(cacheNode?.id || '');
            expect(cacheItem?.hidden).toBe(false);
            expect(Number.parseFloat(cacheItem?.style.opacity || '0')).toBeCloseTo(1, 3);
        } finally {
            fixtures.cleanup();
        }
    });

    it('locks K/V cache branches horizontally to their source outputs', () => {
        const fixtures = buildMhsaFixtures({
            kvCacheState: {
                kvCacheModeEnabled: true,
                kvCachePrefillActive: true
            }
        });

        try {
            const kOutputEntry = fixtures.layout.registry.getNodeEntry(fixtures.kOutputNode?.id || '');
            const vOutputEntry = fixtures.layout.registry.getNodeEntry(fixtures.vOutputNode?.id || '');
            const kCacheEntry = fixtures.layout.registry.getNodeEntry(fixtures.kCacheNode?.id || '');
            const vCacheEntry = fixtures.layout.registry.getNodeEntry(fixtures.vCacheNode?.id || '');
            const resolveCenterX = (entry) => (
                (Number(entry?.contentBounds?.x) || 0)
                + ((Number(entry?.contentBounds?.width) || 0) / 2)
            );
            const resolveMaxHorizontalOffset = (outputEntry, cacheEntry) => (
                Math.max(
                    Number(outputEntry?.contentBounds?.width) || 0,
                    Number(cacheEntry?.contentBounds?.width) || 0
                ) + 36
            );

            expect(Math.abs(resolveCenterX(kOutputEntry) - resolveCenterX(kCacheEntry)))
                .toBeLessThanOrEqual(resolveMaxHorizontalOffset(kOutputEntry, kCacheEntry));
            expect(Math.abs(resolveCenterX(vOutputEntry) - resolveCenterX(vCacheEntry)))
                .toBeLessThanOrEqual(resolveMaxHorizontalOffset(vOutputEntry, vCacheEntry));
            expect(resolveCenterX(kCacheEntry)).toBeGreaterThan(resolveCenterX(kOutputEntry));
            expect(resolveCenterX(vCacheEntry)).toBeGreaterThan(resolveCenterX(vOutputEntry));
        } finally {
            fixtures.cleanup();
        }
    });

    it('keeps the projection weight, bias, and output blocks locked to their feature dimensions across token windows', () => {
        const baseFixtures = buildMhsaFixtures();
        const tallFixtures = buildMhsaFixtures({
            tokenLabels: Array.from({ length: 12 }, (_, index) => `Token ${index + 1}`)
        });

        try {
            expect(Number(baseFixtures.weightNode?.metadata?.card?.height) || 0)
                .toBe(Number(baseFixtures.xLnNode?.metadata?.compactRows?.compactWidth) || 0);
            expect(Number(tallFixtures.weightNode?.metadata?.card?.height) || 0)
                .toBe(Number(tallFixtures.xLnNode?.metadata?.compactRows?.compactWidth) || 0);
            expect(Number(baseFixtures.weightNode?.metadata?.card?.width) || 0)
                .toBe(Number(baseFixtures.outputNode?.metadata?.compactRows?.compactWidth) || 0);
            expect(Number(tallFixtures.weightNode?.metadata?.card?.width) || 0)
                .toBe(Number(tallFixtures.outputNode?.metadata?.compactRows?.compactWidth) || 0);
            expect(Number(tallFixtures.weightNode?.metadata?.card?.width) || 0)
                .toBe(Number(baseFixtures.weightNode?.metadata?.card?.width) || 0);
            expect(Number(tallFixtures.weightNode?.metadata?.card?.height) || 0)
                .toBe(Number(baseFixtures.weightNode?.metadata?.card?.height) || 0);
            expect(Number(tallFixtures.biasNode?.metadata?.compactRows?.compactWidth) || 0)
                .toBe(Number(baseFixtures.biasNode?.metadata?.compactRows?.compactWidth) || 0);
            expect(Number(tallFixtures.outputNode?.metadata?.compactRows?.compactWidth) || 0)
                .toBe(Number(baseFixtures.outputNode?.metadata?.compactRows?.compactWidth) || 0);
            expect(Number(baseFixtures.xLnNode?.metadata?.compactRows?.compactWidth) || 0)
                .toBeGreaterThan(Number(baseFixtures.outputNode?.metadata?.compactRows?.compactWidth) || 0);
            expect(Number(tallFixtures.xLnNode?.metadata?.compactRows?.compactWidth) || 0)
                .toBeGreaterThan(Number(tallFixtures.outputNode?.metadata?.compactRows?.compactWidth) || 0);
        } finally {
            tallFixtures.cleanup();
            baseFixtures.cleanup();
        }
    });

    it('keeps MHSA matrix captions visible throughout the detailed view instead of gating them by zoom', () => {
        const fixtures = buildMhsaFixtures();
        const {
            scene,
            layout,
            weightNode,
            outputNode,
            canvas,
            overlay,
            cleanup
        } = fixtures;

        try {
            const weightEntry = layout.registry.getNodeEntry(weightNode.id);
            const outputEntry = layout.registry.getNodeEntry(outputNode.id);
            const weightThreshold = resolveThresholdScale(weightEntry);
            const outputThreshold = resolveThresholdScale(outputEntry);

            expect(weightThreshold.extent).toBeGreaterThan(outputThreshold.extent);

            const betweenScale = (weightThreshold.scale + outputThreshold.scale) / 2;
            const visibleScale = outputThreshold.scale * 1.08;
            const projectBounds = (scale) => (bounds) => ({
                x: bounds.x,
                y: bounds.y,
                width: bounds.width * scale,
                height: bounds.height * scale
            });

            overlay.sync({
                scene,
                layout,
                canvas,
                projectBounds: projectBounds(visibleScale),
                visible: true,
                enabled: true
            });

            const visibleWeightItem = queryCaptionItem(weightNode.id);
            const visibleOutputItem = queryCaptionItem(outputNode.id);
            expect(visibleWeightItem).toBeTruthy();
            expect(visibleOutputItem).toBeTruthy();
            expect(visibleWeightItem.hidden).toBe(false);
            expect(visibleOutputItem.hidden).toBe(false);

            overlay.sync({
                scene,
                layout,
                canvas,
                projectBounds: projectBounds(betweenScale),
                visible: true,
                enabled: true
            });

            expect(queryCaptionItem(weightNode.id)?.hidden).toBe(false);
            expect(queryCaptionItem(outputNode.id)?.hidden).toBe(false);
        } finally {
            cleanup();
        }
    });

    it('keeps MHSA vector-strip captions in a tighter size band while still letting them scale with zoom', () => {
        const fixtures = buildMhsaFixtures();
        const {
            scene,
            layout,
            weightNode,
            outputNode,
            canvas,
            overlay,
            cleanup
        } = fixtures;

        try {
            const outputEntry = layout.registry.getNodeEntry(outputNode.id);
            const visibleScale = resolveThresholdScale(outputEntry).scale * 1.1;
            const zoomedInScale = visibleScale * 1.9;
            const projectBounds = (scale) => (bounds) => ({
                x: bounds.x,
                y: bounds.y,
                width: bounds.width * scale,
                height: bounds.height * scale
            });

            overlay.sync({
                scene,
                layout,
                canvas,
                projectBounds: projectBounds(visibleScale),
                visible: true,
                enabled: true
            });

            const weightItem = queryCaptionItem(weightNode.id);
            const outputItem = queryCaptionItem(outputNode.id);
            const zoomedOutWeightLabelSize = Number.parseFloat(
                weightItem?.style.getPropertyValue('--detail-transformer-view2d-caption-label-size') || '0'
            );
            const zoomedOutOutputLabelSize = Number.parseFloat(
                outputItem?.style.getPropertyValue('--detail-transformer-view2d-caption-label-size') || '0'
            );
            const zoomedOutWeightDimensionsSize = Number.parseFloat(
                weightItem?.style.getPropertyValue('--detail-transformer-view2d-caption-dimensions-size') || '0'
            );
            const zoomedOutOutputDimensionsSize = Number.parseFloat(
                outputItem?.style.getPropertyValue('--detail-transformer-view2d-caption-dimensions-size') || '0'
            );

            overlay.sync({
                scene,
                layout,
                canvas,
                projectBounds: projectBounds(zoomedInScale),
                visible: true,
                enabled: true
            });

            const zoomedInWeightLabelSize = Number.parseFloat(
                queryCaptionItem(weightNode.id)?.style.getPropertyValue('--detail-transformer-view2d-caption-label-size') || '0'
            );
            const zoomedInOutputLabelSize = Number.parseFloat(
                queryCaptionItem(outputNode.id)?.style.getPropertyValue('--detail-transformer-view2d-caption-label-size') || '0'
            );
            const zoomedInWeightDimensionsSize = Number.parseFloat(
                queryCaptionItem(weightNode.id)?.style.getPropertyValue('--detail-transformer-view2d-caption-dimensions-size') || '0'
            );
            const zoomedInOutputDimensionsSize = Number.parseFloat(
                queryCaptionItem(outputNode.id)?.style.getPropertyValue('--detail-transformer-view2d-caption-dimensions-size') || '0'
            );

            expect(zoomedOutOutputLabelSize).toBeGreaterThan(0);
            expect(zoomedOutOutputDimensionsSize).toBeGreaterThan(0);
            expect(zoomedOutWeightLabelSize).toBeGreaterThan(0);
            expect(zoomedOutWeightDimensionsSize).toBeGreaterThan(0);
            expect(zoomedInOutputLabelSize).toBeGreaterThan(zoomedOutOutputLabelSize);
            expect(zoomedInOutputDimensionsSize).toBeGreaterThan(zoomedOutOutputDimensionsSize);
            expect(zoomedInOutputLabelSize).toBeLessThanOrEqual(MHSA_STANDARD_LABEL_MAX_SCREEN_FONT_PX);
            expect(zoomedInOutputDimensionsSize).toBeLessThanOrEqual(MHSA_STANDARD_DIMENSIONS_MAX_SCREEN_FONT_PX);
            expect(zoomedInWeightLabelSize).toBeGreaterThan(zoomedOutWeightLabelSize);
            expect(zoomedInWeightDimensionsSize).toBeGreaterThan(zoomedOutWeightDimensionsSize);
            expect(zoomedInWeightLabelSize - zoomedOutWeightLabelSize)
                .toBeGreaterThan(zoomedInOutputLabelSize - zoomedOutOutputLabelSize);
            expect(zoomedInWeightDimensionsSize - zoomedOutWeightDimensionsSize)
                .toBeGreaterThan(zoomedInOutputDimensionsSize - zoomedOutOutputDimensionsSize);
        } finally {
            cleanup();
        }
    });

    it('keeps decode-mode cache and vector captions visible and aligned for matching peers', () => {
        const fixtures = buildMhsaFixtures({
            kvCacheState: {
                kvCacheModeEnabled: true,
                kvCachePrefillActive: false,
                kvCacheDecodeActive: true,
                kvCachePassIndex: 1
            }
        });
        const {
            scene,
            layout,
            xLnNode,
            kOutputNode,
            vOutputNode,
            kOutputCopyNode,
            vOutputCopyNode,
            queryNode,
            transposeNode,
            preScoreNode,
            valuePostNode,
            headOutputNode,
            kCacheNode,
            kCacheSourceNode,
            vCacheNode,
            vCacheSourceNode,
            canvas,
            overlay,
            cleanup
        } = fixtures;

        try {
            const contentBounds = layout.contentBounds || { x: 0, y: 0, width: 1, height: 1 };
            const fitScale = Math.min(
                (canvas.clientWidth - 160) / Math.max(1, Number(contentBounds.width) || 1),
                (canvas.clientHeight - 160) / Math.max(1, Number(contentBounds.height) || 1)
            );
            const centeredProjectBounds = (scale) => {
                const offsetX = ((canvas.clientWidth - ((Number(contentBounds.width) || 0) * scale)) / 2)
                    - ((Number(contentBounds.x) || 0) * scale);
                const offsetY = ((canvas.clientHeight - ((Number(contentBounds.height) || 0) * scale)) / 2)
                    - ((Number(contentBounds.y) || 0) * scale);
                return (bounds) => ({
                    x: (bounds.x * scale) + offsetX,
                    y: (bounds.y * scale) + offsetY,
                    width: bounds.width * scale,
                    height: bounds.height * scale
                });
            };

            overlay.sync({
                scene,
                layout,
                canvas,
                projectBounds: centeredProjectBounds(fitScale * 0.94),
                visible: true,
                enabled: true
            });

            const captionNodes = [
                xLnNode,
                kOutputNode,
                vOutputNode,
                queryNode,
                transposeNode,
                preScoreNode,
                valuePostNode,
                headOutputNode,
                kCacheSourceNode,
                vCacheSourceNode
            ].filter(Boolean);
            const captionSizes = captionNodes.map((node) => {
                const captionItem = queryCaptionItem(node.id);
                return {
                    nodeId: node.id,
                    labelSize: Number.parseFloat(
                        captionItem?.style.getPropertyValue('--detail-transformer-view2d-caption-label-size') || '0'
                    ),
                    dimensionsSize: Number.parseFloat(
                        captionItem?.style.getPropertyValue('--detail-transformer-view2d-caption-dimensions-size') || '0'
                    )
                };
            });
            const labelSizes = captionSizes.map(({ labelSize }) => labelSize);
            const dimensionsSizes = captionSizes.map(({ dimensionsSize }) => dimensionsSize);

            labelSizes.forEach((labelSize) => {
                expect(labelSize).toBeGreaterThan(0);
            });
            dimensionsSizes.forEach((dimensionsSize) => {
                expect(dimensionsSize).toBeGreaterThan(0);
            });
            const captionSizeByNodeId = new Map(captionSizes.map((entry) => [entry.nodeId, entry]));
            expect(Math.abs(
                (captionSizeByNodeId.get(kOutputNode?.id || '')?.labelSize || 0)
                - (captionSizeByNodeId.get(vOutputNode?.id || '')?.labelSize || 0)
            )).toBeLessThan(0.1);
            expect(Math.abs(
                (captionSizeByNodeId.get(kOutputNode?.id || '')?.labelSize || 0)
                - (captionSizeByNodeId.get(kCacheSourceNode?.id || '')?.labelSize || 0)
            )).toBeLessThan(0.35);
            expect(Math.abs(
                (captionSizeByNodeId.get(vOutputNode?.id || '')?.labelSize || 0)
                - (captionSizeByNodeId.get(vCacheSourceNode?.id || '')?.labelSize || 0)
            )).toBeLessThan(0.35);
            const kCacheSourceLabelSize = captionSizeByNodeId.get(kCacheSourceNode?.id || '')?.labelSize || 0;
            const vCacheSourceLabelSize = captionSizeByNodeId.get(vCacheSourceNode?.id || '')?.labelSize || 0;
            expect(vCacheSourceLabelSize).toBeGreaterThan(kCacheSourceLabelSize);
            expect(queryCaptionItem(kOutputCopyNode?.id || '')).toBeNull();
            expect(queryCaptionItem(vOutputCopyNode?.id || '')).toBeNull();
            expect(queryCaptionItem(kCacheNode?.id || '')).toBeNull();
            expect(queryCaptionItem(vCacheNode?.id || '')).toBeNull();
            const baselineKCacheLabelSize = captionSizeByNodeId.get(kCacheSourceNode?.id || '')?.labelSize || 0;
            const baselineKCacheDimensionsSize = captionSizeByNodeId.get(kCacheSourceNode?.id || '')?.dimensionsSize || 0;

            const kCacheEntry = layout.registry.getNodeEntry(kCacheSourceNode?.id || '');
            const zoomedInScale = fitScale * 2.2;
            const targetX = canvas.clientWidth / 2;
            const targetY = canvas.clientHeight / 2;
            const kCacheBounds = kCacheEntry?.contentBounds || kCacheEntry?.bounds || { x: 0, y: 0, width: 0, height: 0 };
            const cacheCenterX = (Number(kCacheBounds.x) || 0) + ((Number(kCacheBounds.width) || 0) / 2);
            const cacheCenterY = (Number(kCacheBounds.y) || 0) + ((Number(kCacheBounds.height) || 0) / 2);

            overlay.sync({
                scene,
                layout,
                canvas,
                projectBounds: (bounds) => ({
                    x: (bounds.x * zoomedInScale) + (targetX - (cacheCenterX * zoomedInScale)),
                    y: (bounds.y * zoomedInScale) + (targetY - (cacheCenterY * zoomedInScale)),
                    width: bounds.width * zoomedInScale,
                    height: bounds.height * zoomedInScale
                }),
                visible: true,
                enabled: true
            });

            const zoomedInKCacheItem = queryCaptionItem(kCacheSourceNode?.id || '');
            const zoomedInKCacheLabelSize = Number.parseFloat(
                zoomedInKCacheItem?.style.getPropertyValue('--detail-transformer-view2d-caption-label-size') || '0'
            );
            const zoomedInKCacheDimensionsSize = Number.parseFloat(
                zoomedInKCacheItem?.style.getPropertyValue('--detail-transformer-view2d-caption-dimensions-size') || '0'
            );

            expect(zoomedInKCacheLabelSize).toBeGreaterThan(baselineKCacheLabelSize || 0);
            expect(zoomedInKCacheDimensionsSize).toBeGreaterThan(baselineKCacheDimensionsSize || 0);
            expect(zoomedInKCacheLabelSize).toBeLessThanOrEqual(MHSA_STANDARD_LABEL_MAX_SCREEN_FONT_PX);
            expect(zoomedInKCacheDimensionsSize).toBeLessThanOrEqual(MHSA_STANDARD_DIMENSIONS_MAX_SCREEN_FONT_PX);
        } finally {
            cleanup();
        }
    });

    it('keeps primary MHSA single-row vector captions scene-relative after the single-row boost', () => {
        const fixtures = buildMhsaFixtures({
            tokenLabels: ['Token A']
        });
        const {
            scene,
            layout,
            vOutputNode,
            canvas,
            overlay,
            cleanup
        } = fixtures;

        try {
            const outputEntry = layout.registry.getNodeEntry(vOutputNode?.id || '');
            const zoomedOutScale = resolveThresholdScale(outputEntry).scale * 1.15;
            const zoomedInScale = resolveThresholdScale(outputEntry).scale * 3.2;
            const targetX = 60;
            const targetY = 56;
            const projectBounds = (scale) => {
                const offsetX = targetX - ((Number(outputEntry?.contentBounds?.x) || 0) * scale);
                const offsetY = targetY - ((Number(outputEntry?.contentBounds?.y) || 0) * scale);
                return (bounds) => ({
                    x: (bounds.x * scale) + offsetX,
                    y: (bounds.y * scale) + offsetY,
                    width: bounds.width * scale,
                    height: bounds.height * scale
                });
            };

            overlay.sync({
                scene,
                layout,
                canvas,
                projectBounds: projectBounds(zoomedOutScale),
                visible: true,
                enabled: true
            });

            const zoomedOutLabelSize = Number.parseFloat(
                queryCaptionItem(vOutputNode?.id || '')?.style.getPropertyValue('--detail-transformer-view2d-caption-label-size') || '0'
            );

            overlay.sync({
                scene,
                layout,
                canvas,
                projectBounds: projectBounds(zoomedInScale),
                visible: true,
                enabled: true
            });

            const zoomedInLabelSize = Number.parseFloat(
                queryCaptionItem(vOutputNode?.id || '')?.style.getPropertyValue('--detail-transformer-view2d-caption-label-size') || '0'
            );
            const labelScaleRatio = zoomedInLabelSize / Math.max(0.0001, zoomedOutLabelSize);
            const rawViewportScaleRatio = zoomedInScale / zoomedOutScale;

            expect(zoomedOutLabelSize).toBeGreaterThan(0);
            expect(zoomedInLabelSize).toBeGreaterThan(zoomedOutLabelSize);
            expect(labelScaleRatio).toBeGreaterThan(1.2);
            expect(labelScaleRatio).toBeLessThanOrEqual(rawViewportScaleRatio * 1.02);
        } finally {
            cleanup();
        }
    });

    it('keeps the Q bias caption legible across zoom levels under uniform matrix sizing', () => {
        const fixtures = buildMhsaFixtures();
        const {
            scene,
            layout,
            qBiasNode,
            canvas,
            overlay,
            cleanup
        } = fixtures;

        try {
            const biasEntry = layout.registry.getNodeEntry(qBiasNode.id);
            const zoomedOutScale = 0.2;
            const zoomedInScale = 1.1;
            const targetX = 32;
            const targetY = 40;
            const projectBounds = (scale) => {
                const offsetX = targetX - ((Number(biasEntry?.contentBounds?.x) || 0) * scale);
                const offsetY = targetY - ((Number(biasEntry?.contentBounds?.y) || 0) * scale);
                return (bounds) => ({
                    x: (bounds.x * scale) + offsetX,
                    y: (bounds.y * scale) + offsetY,
                    width: bounds.width * scale,
                    height: bounds.height * scale
                });
            };

            overlay.sync({
                scene,
                layout,
                canvas,
                projectBounds: projectBounds(zoomedOutScale),
                visible: true,
                enabled: true
            });

            const zoomedOutLabelSize = Number.parseFloat(
                queryCaptionItem(qBiasNode.id)?.style.getPropertyValue('--detail-transformer-view2d-caption-label-size') || '0'
            );
            const zoomedOutExtent = Math.min(
                (Number(biasEntry?.contentBounds?.width) || 0) * zoomedOutScale,
                (Number(biasEntry?.contentBounds?.height) || 0) * zoomedOutScale
            );

            overlay.sync({
                scene,
                layout,
                canvas,
                projectBounds: projectBounds(zoomedInScale),
                visible: true,
                enabled: true
            });

            const zoomedInLabelSize = Number.parseFloat(
                queryCaptionItem(qBiasNode.id)?.style.getPropertyValue('--detail-transformer-view2d-caption-label-size') || '0'
            );
            const zoomedInExtent = Math.min(
                (Number(biasEntry?.contentBounds?.width) || 0) * zoomedInScale,
                (Number(biasEntry?.contentBounds?.height) || 0) * zoomedInScale
            );

            expect(zoomedOutLabelSize).toBeGreaterThan(0);
            expect(zoomedInLabelSize).toBeGreaterThanOrEqual(zoomedOutLabelSize);
            expect(zoomedOutLabelSize).toBeLessThan(BIAS_LABEL_MIN_SCREEN_FONT_PX);
            expect(zoomedInExtent).toBeGreaterThan(zoomedOutExtent);
        } finally {
            cleanup();
        }
    });

    it('applies bias-caption tuning so subscripts stay smaller and bias dimensions render larger', () => {
        const fixtures = buildMhsaFixtures();
        const {
            scene,
            layout,
            qBiasNode,
            weightNode,
            canvas,
            overlay,
            cleanup
        } = fixtures;

        try {
            overlay.sync({
                scene,
                layout,
                canvas,
                projectBounds: (bounds) => ({
                    x: bounds.x,
                    y: bounds.y,
                    width: bounds.width,
                    height: bounds.height
                }),
                visible: true,
                enabled: true
            });

            const biasItem = queryCaptionItem(qBiasNode.id);
            const weightItem = queryCaptionItem(weightNode.id);
            const biasLabelRoleScale = Number.parseFloat(
                biasItem?.style.getPropertyValue('--detail-transformer-view2d-caption-label-role-scale') || '0'
            );
            const biasDimensionsRoleScale = Number.parseFloat(
                biasItem?.style.getPropertyValue('--detail-transformer-view2d-caption-dimensions-role-scale') || '0'
            );
            const biasKatexSubscriptScale = Number.parseFloat(
                biasItem?.style.getPropertyValue('--detail-transformer-view2d-caption-katex-subscript-scale') || '0'
            );
            const weightKatexSubscriptScale = Number.parseFloat(
                weightItem?.style.getPropertyValue('--detail-transformer-view2d-caption-katex-subscript-scale') || '0'
            );
            const biasInlineSubscriptScale = Number.parseFloat(
                biasItem?.style.getPropertyValue('--detail-transformer-view2d-caption-inline-subscript-scale') || '0'
            );
            const weightInlineSubscriptScale = Number.parseFloat(
                weightItem?.style.getPropertyValue('--detail-transformer-view2d-caption-inline-subscript-scale') || '0'
            );

            expect(biasLabelRoleScale).toBeGreaterThan(3);
            expect(biasLabelRoleScale).toBeLessThan(4.6);
            expect(biasDimensionsRoleScale).toBeGreaterThan(1);
            expect(biasKatexSubscriptScale).toBeLessThan(weightKatexSubscriptScale);
            expect(biasInlineSubscriptScale).toBeLessThan(weightInlineSubscriptScale);
        } finally {
            cleanup();
        }
    });

    it('applies targeted overlay label scaling to layer norm gamma and beta captions', () => {
        const fixtures = buildLayerNormDetailFixtures();
        const {
            scene,
            layout,
            inputNode,
            scaleNode,
            shiftNode,
            canvas,
            overlay,
            cleanup
        } = fixtures;

        try {
            overlay.sync({
                scene,
                layout,
                canvas,
                projectBounds: (bounds) => ({
                    x: bounds.x,
                    y: bounds.y,
                    width: bounds.width,
                    height: bounds.height
                }),
                visible: true,
                enabled: true
            });

            const inputItem = queryCaptionItem(inputNode.id);
            const scaleItem = queryCaptionItem(scaleNode.id);
            const shiftItem = queryCaptionItem(shiftNode.id);
            const inputLabelRoleScale = Number.parseFloat(
                inputItem?.style.getPropertyValue('--detail-transformer-view2d-caption-label-role-scale') || '0'
            );
            const scaleLabelRoleScale = Number.parseFloat(
                scaleItem?.style.getPropertyValue('--detail-transformer-view2d-caption-label-role-scale') || '0'
            );
            const shiftLabelRoleScale = Number.parseFloat(
                shiftItem?.style.getPropertyValue('--detail-transformer-view2d-caption-label-role-scale') || '0'
            );

            expect(inputLabelRoleScale).toBe(1);
            expect(scaleLabelRoleScale).toBeGreaterThan(1);
            expect(shiftLabelRoleScale).toBeGreaterThan(1);
            expect(scaleLabelRoleScale).toBe(shiftLabelRoleScale);
        } finally {
            cleanup();
        }
    });

    it('prefers KaTeX rendering for MHSA bias, score, and scale labels when available', () => {
        const fixtures = buildMhsaFixtures();
        const {
            scene,
            layout,
            qBiasNode,
            biasNode,
            preScoreNode,
            scaleNode,
            canvas,
            overlay,
            cleanup
        } = fixtures;
        const originalKatex = window.katex;
        window.katex = {
            renderToString: vi.fn((tex) => `<span class="katex" data-tex="${tex}"></span>`)
        };

        try {
            const qWeightNode = flattenSceneNodes(scene).find((node) => (
                node.role === 'projection-weight'
                && String(node.metadata?.kind || '').toLowerCase() === 'q'
            )) || null;
            const preScoreEntry = layout.registry.getNodeEntry(preScoreNode.id);
            const visibleScale = resolveThresholdScale(preScoreEntry).scale * 1.2;
            const projectBounds = (bounds) => ({
                x: bounds.x,
                y: bounds.y,
                width: bounds.width * visibleScale,
                height: bounds.height * visibleScale
            });

            overlay.sync({
                scene,
                layout,
                canvas,
                projectBounds,
                visible: true,
                enabled: true
            });

            expect(window.katex.renderToString).toHaveBeenCalledWith(
                'W_{\\mathrm{q}}',
                expect.objectContaining({
                    throwOnError: false,
                    displayMode: false
                })
            );
            expect(window.katex.renderToString).toHaveBeenCalledWith(
                'b_{\\mathrm{q}}',
                expect.objectContaining({
                    throwOnError: false,
                    displayMode: false
                })
            );
            expect(queryCaptionItem(qWeightNode?.id || '')?.querySelector('.katex')).toBeTruthy();
            expect(queryCaptionItem(qBiasNode.id)?.dataset.nodeRole).toBe('projection-bias');
            expect(queryCaptionItem(qBiasNode.id)?.querySelector('.katex')).toBeTruthy();
            expect(queryCaptionItem(biasNode.id)?.querySelector('.katex')).toBeTruthy();
            expect(queryCaptionItem(preScoreNode.id)?.querySelector('.katex')).toBeTruthy();
            expect(queryCaptionItem(scaleNode.id)?.querySelector('.katex')).toBeTruthy();
        } finally {
            if (originalKatex) {
                window.katex = originalKatex;
            } else {
                delete window.katex;
            }
            cleanup();
        }
    });

    it('standardizes MHSA detail matrix label sizes while keeping softmax and scale text tuned', () => {
        const fixtures = buildMhsaFixtures();
        const {
            scene,
            layout,
            preScoreNode,
            maskNode,
            postNode,
            softmaxLabelNode,
            scaleNode,
            canvas,
            overlay,
            cleanup
        } = fixtures;

        try {
            const preScoreEntry = layout.registry.getNodeEntry(preScoreNode.id);
            const targetX = 40;
            const targetY = 48;
            const scale = 0.95;
            const projectBounds = (bounds) => ({
                x: (bounds.x * scale) + (targetX - ((Number(preScoreEntry?.contentBounds?.x) || 0) * scale)),
                y: (bounds.y * scale) + (targetY - ((Number(preScoreEntry?.contentBounds?.y) || 0) * scale)),
                width: bounds.width * scale,
                height: bounds.height * scale
            });

            overlay.sync({
                scene,
                layout,
                canvas,
                projectBounds,
                visible: true,
                enabled: true
            });

            const preScoreLabelSize = Number.parseFloat(
                queryCaptionItem(preScoreNode.id)?.style.getPropertyValue('--detail-transformer-view2d-caption-label-size') || '0'
            );
            const maskLabelSize = Number.parseFloat(
                queryCaptionItem(maskNode.id)?.style.getPropertyValue('--detail-transformer-view2d-caption-label-size') || '0'
            );
            const postLabelSize = Number.parseFloat(
                queryCaptionItem(postNode.id)?.style.getPropertyValue('--detail-transformer-view2d-caption-label-size') || '0'
            );
            expect(preScoreLabelSize).toBeGreaterThan(0);
            expect(maskLabelSize).toBeGreaterThan(0);
            expect(postLabelSize).toBeGreaterThan(0);
            expect(preScoreLabelSize).toBeCloseTo(postLabelSize, 1);
            expect(maskLabelSize).toBeCloseTo(postLabelSize, 1);
            expect(softmaxLabelNode?.metadata?.fontScale).toBeGreaterThan(1);
            expect(scaleNode?.metadata?.fontScale).toBeGreaterThan(softmaxLabelNode?.metadata?.fontScale || 0);
        } finally {
            cleanup();
        }
    });

    it('lets the K weight and bias captions shrink with the scene when zooming out', () => {
        const fixtures = buildMhsaFixtures({
            tokenLabels: Array.from({ length: 12 }, (_, index) => `Token ${index + 1}`)
        });
        const {
            scene,
            layout,
            weightNode,
            biasNode,
            canvas,
            overlay,
            cleanup
        } = fixtures;

        try {
            const weightEntry = layout.registry.getNodeEntry(weightNode.id);
            const zoomedInScale = resolveThresholdScale(weightEntry).scale * 2.4;
            const zoomedOutScale = resolveThresholdScale(weightEntry).scale * 0.55;
            const projectBounds = (scale) => (bounds) => ({
                x: bounds.x,
                y: bounds.y,
                width: bounds.width * scale,
                height: bounds.height * scale
            });

            overlay.sync({
                scene,
                layout,
                canvas,
                projectBounds: projectBounds(zoomedInScale),
                visible: true,
                enabled: true
            });

            const zoomedInWeightLabelSize = Number.parseFloat(
                queryCaptionItem(weightNode.id)?.style.getPropertyValue('--detail-transformer-view2d-caption-label-size') || '0'
            );
            const zoomedInBiasLabelSize = Number.parseFloat(
                queryCaptionItem(biasNode.id)?.style.getPropertyValue('--detail-transformer-view2d-caption-label-size') || '0'
            );

            overlay.sync({
                scene,
                layout,
                canvas,
                projectBounds: projectBounds(zoomedOutScale),
                visible: true,
                enabled: true
            });

            const zoomedOutWeightLabelSize = Number.parseFloat(
                queryCaptionItem(weightNode.id)?.style.getPropertyValue('--detail-transformer-view2d-caption-label-size') || '0'
            );
            const zoomedOutBiasLabelSize = Number.parseFloat(
                queryCaptionItem(biasNode.id)?.style.getPropertyValue('--detail-transformer-view2d-caption-label-size') || '0'
            );

            expect(zoomedInWeightLabelSize).toBeGreaterThan(0);
            expect(zoomedInBiasLabelSize).toBeGreaterThan(0);
            expect(zoomedOutWeightLabelSize).toBeGreaterThan(0);
            expect(zoomedOutBiasLabelSize).toBeGreaterThan(0);
            expect(zoomedOutWeightLabelSize).toBeLessThan(zoomedInWeightLabelSize);
            expect(zoomedOutBiasLabelSize).toBeLessThan(zoomedInBiasLabelSize);
            expect(zoomedOutWeightLabelSize).toBeLessThanOrEqual(zoomedInWeightLabelSize * 0.75);
            expect(zoomedOutBiasLabelSize).toBeLessThanOrEqual(zoomedInBiasLabelSize * 0.75);
        } finally {
            cleanup();
        }
    });

    it('keeps the K weight caption readable during deep zoom', () => {
        const fixtures = buildMhsaFixtures({
            tokenLabels: Array.from({ length: 12 }, (_, index) => `Token ${index + 1}`)
        });
        const {
            scene,
            layout,
            weightNode,
            canvas,
            overlay,
            cleanup
        } = fixtures;

        try {
            const weightEntry = layout.registry.getNodeEntry(weightNode.id);
            const visibleScale = resolveThresholdScale(weightEntry).scale * 6.5;
            const projectBounds = (bounds) => ({
                x: bounds.x,
                y: bounds.y,
                width: bounds.width * visibleScale,
                height: bounds.height * visibleScale
            });

            overlay.sync({
                scene,
                layout,
                canvas,
                projectBounds,
                visible: true,
                enabled: true
            });

            const weightLabelSize = Number.parseFloat(
                queryCaptionItem(weightNode.id)?.style.getPropertyValue('--detail-transformer-view2d-caption-label-size') || '0'
            );

            expect(weightLabelSize).toBeGreaterThan(0);
            expect(weightLabelSize).toBeGreaterThan(0);
            expect(weightLabelSize).toBeGreaterThan(40);
        } finally {
            cleanup();
        }
    });

    it('scales the dom-katex sqrt(d_head) label with zoom levels', () => {
        const fixtures = buildMhsaFixtures();
        const {
            scene,
            layout,
            scaleNode,
            canvas,
            overlay,
            cleanup
        } = fixtures;

        try {
            const scaleEntry = layout.registry.getNodeEntry(scaleNode.id);
            const baseBounds = scaleEntry?.contentBounds || scaleEntry?.bounds || null;
            const zoomedOutScale = 0.72;
            const zoomedInScale = 1.4;
            const targetX = 36;
            const targetY = 42;
            const projectBounds = (scale) => {
                const offsetX = targetX - ((Number(baseBounds?.x) || 0) * scale);
                const offsetY = targetY - ((Number(baseBounds?.y) || 0) * scale);
                return (bounds) => ({
                    x: (bounds.x * scale) + offsetX,
                    y: (bounds.y * scale) + offsetY,
                    width: bounds.width * scale,
                    height: bounds.height * scale
                });
            };

            overlay.sync({
                scene,
                layout,
                canvas,
                projectBounds: projectBounds(zoomedOutScale),
                visible: true,
                enabled: true
            });

            const zoomedOutFontSize = Number.parseFloat(
                queryCaptionItem(scaleNode.id)?.style.getPropertyValue('--detail-transformer-view2d-dom-text-size') || '0'
            );
            const zoomedOutHeight = (Number(baseBounds?.height) || 0) * zoomedOutScale;

            overlay.sync({
                scene,
                layout,
                canvas,
                projectBounds: projectBounds(zoomedInScale),
                visible: true,
                enabled: true
            });

            const zoomedInFontSize = Number.parseFloat(
                queryCaptionItem(scaleNode.id)?.style.getPropertyValue('--detail-transformer-view2d-dom-text-size') || '0'
            );
            const zoomedInHeight = (Number(baseBounds?.height) || 0) * zoomedInScale;

            expect(scaleEntry).toBeTruthy();
            expect(zoomedOutFontSize).toBeGreaterThan(0);
            expect(zoomedInFontSize).toBeGreaterThan(zoomedOutFontSize);
            expect(zoomedOutFontSize / Math.max(1, zoomedOutHeight))
                .toBeCloseTo(zoomedInFontSize / Math.max(1, zoomedInHeight), 2);
        } finally {
            cleanup();
        }
    });

    it('scales the dom-katex softmax label with zoom levels on narrow MHSA detail canvases', () => {
        const fixtures = buildMhsaFixtures({
            canvasWidth: 640,
            canvasHeight: 420
        });
        const {
            scene,
            layout,
            softmaxLabelNode,
            canvas,
            overlay,
            cleanup
        } = fixtures;

        try {
            const softmaxEntry = layout.registry.getNodeEntry(softmaxLabelNode.id);
            const baseBounds = softmaxEntry?.contentBounds || softmaxEntry?.bounds || null;
            const zoomedOutScale = 0.78;
            const zoomedInScale = 1.18;
            const targetX = 28;
            const targetY = 34;
            const projectBounds = (scale) => {
                const offsetX = targetX - ((Number(baseBounds?.x) || 0) * scale);
                const offsetY = targetY - ((Number(baseBounds?.y) || 0) * scale);
                return (bounds) => ({
                    x: (bounds.x * scale) + offsetX,
                    y: (bounds.y * scale) + offsetY,
                    width: bounds.width * scale,
                    height: bounds.height * scale
                });
            };
            overlay.sync({
                scene,
                layout,
                canvas,
                projectBounds: projectBounds(zoomedOutScale),
                visible: true,
                enabled: true
            });

            const zoomedOutFontSize = Number.parseFloat(
                queryCaptionItem(softmaxLabelNode.id)?.style.getPropertyValue('--detail-transformer-view2d-dom-text-size') || '0'
            );
            const zoomedOutHeight = (Number(baseBounds?.height) || 0) * zoomedOutScale;

            overlay.sync({
                scene,
                layout,
                canvas,
                projectBounds: projectBounds(zoomedInScale),
                visible: true,
                enabled: true
            });

            const zoomedInFontSize = Number.parseFloat(
                queryCaptionItem(softmaxLabelNode.id)?.style.getPropertyValue('--detail-transformer-view2d-dom-text-size') || '0'
            );
            const zoomedInHeight = (Number(baseBounds?.height) || 0) * zoomedInScale;

            expect(zoomedOutFontSize).toBeGreaterThan(0);
            expect(zoomedInFontSize).toBeGreaterThan(zoomedOutFontSize);
            expect(zoomedOutFontSize / Math.max(1, zoomedOutHeight))
                .toBeCloseTo(zoomedInFontSize / Math.max(1, zoomedInHeight), 2);
        } finally {
            cleanup();
        }
    });

    it('scales MHSA equation operators with zoom levels', () => {
        const fixtures = buildMhsaFixtures();
        const {
            scene,
            layout,
            canvas,
            overlay,
            cleanup
        } = fixtures;

        try {
            const operatorNode = flattenSceneNodes(scene).find((node) => node.role === 'attention-equals');
            const operatorEntry = layout.registry.getNodeEntry(operatorNode?.id || '');
            const baseBounds = operatorEntry?.contentBounds || operatorEntry?.bounds || null;
            const zoomedOutScale = 0.42;
            const zoomedInScale = 1.42;
            const targetX = 52;
            const targetY = 48;
            const projectBounds = (scale) => {
                const offsetX = targetX - ((Number(baseBounds?.x) || 0) * scale);
                const offsetY = targetY - ((Number(baseBounds?.y) || 0) * scale);
                return (bounds) => ({
                    x: (bounds.x * scale) + offsetX,
                    y: (bounds.y * scale) + offsetY,
                    width: bounds.width * scale,
                    height: bounds.height * scale
                });
            };

            overlay.sync({
                scene,
                layout,
                canvas,
                projectBounds: projectBounds(zoomedOutScale),
                visible: true,
                enabled: true
            });

            const zoomedOutItem = queryCaptionItem(operatorNode?.id || '');
            const zoomedOutFontSize = Number.parseFloat(
                zoomedOutItem?.style.getPropertyValue('--detail-transformer-view2d-dom-text-size') || '0'
            );
            const zoomedOutHeight = (Number(baseBounds?.height) || 0) * zoomedOutScale;

            overlay.sync({
                scene,
                layout,
                canvas,
                projectBounds: projectBounds(zoomedInScale),
                visible: true,
                enabled: true
            });

            const zoomedInItem = queryCaptionItem(operatorNode?.id || '');
            const zoomedInFontSize = Number.parseFloat(
                zoomedInItem?.style.getPropertyValue('--detail-transformer-view2d-dom-text-size') || '0'
            );
            const zoomedInHeight = (Number(baseBounds?.height) || 0) * zoomedInScale;

            expect(operatorNode).toBeTruthy();
            expect(zoomedOutItem?.dataset.nodeKind).toBe('operator');
            expect(zoomedOutItem?.textContent).toContain('=');
            expect(zoomedOutFontSize).toBeGreaterThan(0);
            expect(zoomedInFontSize).toBeGreaterThan(zoomedOutFontSize);
            expect(zoomedOutFontSize / Math.max(1, zoomedOutHeight))
                .toBeCloseTo(zoomedInFontSize / Math.max(1, zoomedInHeight), 2);
        } finally {
            cleanup();
        }
    });

    it('scales head-detail matrix labels and dimensions with zoom', () => {
        const fixtures = buildHeadDetailFixtures();
        const {
            scene,
            layout,
            qCopyNode,
            canvas,
            overlay,
            cleanup
        } = fixtures;

        try {
            const qCopyEntry = layout.registry.getNodeEntry(qCopyNode.id);
            const zoomedOutScale = 0.38;
            const zoomedInScale = 1.46;
            const projectBounds = (scale) => (bounds) => ({
                x: bounds.x,
                y: bounds.y,
                width: bounds.width * scale,
                height: bounds.height * scale
            });

            overlay.sync({
                scene,
                layout,
                canvas,
                projectBounds: projectBounds(zoomedOutScale),
                visible: true,
                enabled: true
            });

            const zoomedOutLabelSize = Number.parseFloat(
                queryCaptionItem(qCopyNode.id)?.style.getPropertyValue('--detail-transformer-view2d-caption-label-size') || '0'
            );
            const zoomedOutDimensionsSize = Number.parseFloat(
                queryCaptionItem(qCopyNode.id)?.style.getPropertyValue('--detail-transformer-view2d-caption-dimensions-size') || '0'
            );

            overlay.sync({
                scene,
                layout,
                canvas,
                projectBounds: projectBounds(zoomedInScale),
                visible: true,
                enabled: true
            });

            const zoomedInLabelSize = Number.parseFloat(
                queryCaptionItem(qCopyNode.id)?.style.getPropertyValue('--detail-transformer-view2d-caption-label-size') || '0'
            );
            const zoomedInDimensionsSize = Number.parseFloat(
                queryCaptionItem(qCopyNode.id)?.style.getPropertyValue('--detail-transformer-view2d-caption-dimensions-size') || '0'
            );

            expect(qCopyEntry).toBeTruthy();
            expect(zoomedOutLabelSize).toBeGreaterThan(0);
            expect(zoomedOutDimensionsSize).toBeGreaterThan(0);
            expect(zoomedInLabelSize).toBeGreaterThan(zoomedOutLabelSize);
            expect(zoomedInDimensionsSize).toBeGreaterThan(zoomedOutDimensionsSize);
            expect(zoomedOutDimensionsSize).toBeLessThan(ATTENTION_MATRIX_LABEL_MIN_SCREEN_FONT_PX);
        } finally {
            cleanup();
        }
    });

    it('scales the MLP x_ln caption with zoom like the MHSA detail captions', () => {
        const fixtures = buildMlpDetailFixtures();
        const {
            scene,
            layout,
            inputNode,
            canvas,
            overlay,
            cleanup
        } = fixtures;

        try {
            const projectBounds = (scale) => (bounds) => ({
                x: bounds.x,
                y: bounds.y,
                width: bounds.width * scale,
                height: bounds.height * scale
            });

            overlay.sync({
                scene,
                layout,
                canvas,
                projectBounds: projectBounds(0.34),
                visible: true,
                enabled: true
            });

            const zoomedOutLabelSize = Number.parseFloat(
                queryCaptionItem(inputNode.id)?.style.getPropertyValue('--detail-transformer-view2d-caption-label-size') || '0'
            );
            const zoomedOutDimensionsSize = Number.parseFloat(
                queryCaptionItem(inputNode.id)?.style.getPropertyValue('--detail-transformer-view2d-caption-dimensions-size') || '0'
            );

            overlay.sync({
                scene,
                layout,
                canvas,
                projectBounds: projectBounds(1.42),
                visible: true,
                enabled: true
            });

            const zoomedInLabelSize = Number.parseFloat(
                queryCaptionItem(inputNode.id)?.style.getPropertyValue('--detail-transformer-view2d-caption-label-size') || '0'
            );
            const zoomedInDimensionsSize = Number.parseFloat(
                queryCaptionItem(inputNode.id)?.style.getPropertyValue('--detail-transformer-view2d-caption-dimensions-size') || '0'
            );

            expect(zoomedOutLabelSize).toBeGreaterThan(0);
            expect(zoomedOutDimensionsSize).toBeGreaterThan(0);
            expect(zoomedInLabelSize).toBeGreaterThan(zoomedOutLabelSize);
            expect(zoomedInDimensionsSize).toBeGreaterThan(zoomedOutDimensionsSize);
        } finally {
            cleanup();
        }
    });

    it('renders the final MLP output caption with a colored KaTeX MLP token', () => {
        const fixtures = buildMlpDetailFixtures();
        const {
            scene,
            layout,
            downOutputNode,
            canvas,
            overlay,
            cleanup
        } = fixtures;
        const originalKatex = window.katex;
        window.katex = {
            renderToString: vi.fn((tex) => `<span class="katex" data-tex="${tex}"></span>`)
        };

        try {
            overlay.sync({
                scene,
                layout,
                canvas,
                projectBounds: (bounds) => ({
                    x: bounds.x,
                    y: bounds.y,
                    width: bounds.width,
                    height: bounds.height
                }),
                visible: true,
                enabled: true
            });

            expect(window.katex.renderToString).toHaveBeenCalledWith(
                `\\textcolor{${toKatexColorHex(FINAL_MLP_COLOR)}}{\\mathrm{MLP}}(x_{\\ln})`,
                expect.objectContaining({
                    throwOnError: false,
                    displayMode: false
                })
            );
            expect(queryCaptionItem(downOutputNode?.id || '')?.querySelector('.katex')).toBeTruthy();
        } finally {
            if (originalKatex) {
                window.katex = originalKatex;
            } else {
                delete window.katex;
            }
            cleanup();
        }
    });

    it('keeps the attention query and transpose captions on the same visual scale', () => {
        const fixtures = buildMhsaFixtures();
        const {
            scene,
            layout,
            queryNode,
            transposeNode,
            canvas,
            overlay,
            cleanup
        } = fixtures;

        try {
            const queryEntry = layout.registry.getNodeEntry(queryNode.id);
            const visibleScale = resolveThresholdScale(queryEntry).scale * 1.1;
            const projectBounds = (bounds) => ({
                x: bounds.x,
                y: bounds.y,
                width: bounds.width * visibleScale,
                height: bounds.height * visibleScale
            });

            overlay.sync({
                scene,
                layout,
                canvas,
                projectBounds,
                visible: true,
                enabled: true
            });

            const queryItem = queryCaptionItem(queryNode.id);
            const transposeItem = queryCaptionItem(transposeNode.id);
            const queryLabelSize = Number.parseFloat(
                queryItem?.style.getPropertyValue('--detail-transformer-view2d-caption-label-size') || '0'
            );
            const transposeLabelSize = Number.parseFloat(
                transposeItem?.style.getPropertyValue('--detail-transformer-view2d-caption-label-size') || '0'
            );
            const queryDimensionsSize = Number.parseFloat(
                queryItem?.style.getPropertyValue('--detail-transformer-view2d-caption-dimensions-size') || '0'
            );
            const transposeDimensionsSize = Number.parseFloat(
                transposeItem?.style.getPropertyValue('--detail-transformer-view2d-caption-dimensions-size') || '0'
            );

            expect(transposeLabelSize).toBeCloseTo(queryLabelSize, 2);
            expect(transposeDimensionsSize).toBeCloseTo(queryDimensionsSize, 2);
        } finally {
            cleanup();
        }
    });

    it('keeps attention-grid captions on the same visual scale as the surrounding MHSA matrices', () => {
        const fixtures = buildMhsaFixtures();
        const {
            scene,
            layout,
            queryNode,
            transposeNode,
            preScoreNode,
            canvas,
            overlay,
            cleanup
        } = fixtures;

        try {
            const queryEntry = layout.registry.getNodeEntry(queryNode.id);
            const visibleScale = resolveThresholdScale(queryEntry).scale * 1.1;
            const projectBounds = (bounds) => ({
                x: bounds.x,
                y: bounds.y,
                width: bounds.width * visibleScale,
                height: bounds.height * visibleScale
            });

            overlay.sync({
                scene,
                layout,
                canvas,
                projectBounds,
                visible: true,
                enabled: true
            });

            const queryLabelSize = Number.parseFloat(
                queryCaptionItem(queryNode.id)?.style.getPropertyValue('--detail-transformer-view2d-caption-label-size') || '0'
            );
            const transposeLabelSize = Number.parseFloat(
                queryCaptionItem(transposeNode.id)?.style.getPropertyValue('--detail-transformer-view2d-caption-label-size') || '0'
            );
            const preScoreLabelSize = Number.parseFloat(
                queryCaptionItem(preScoreNode.id)?.style.getPropertyValue('--detail-transformer-view2d-caption-label-size') || '0'
            );
            const queryDimensionsSize = Number.parseFloat(
                queryCaptionItem(queryNode.id)?.style.getPropertyValue('--detail-transformer-view2d-caption-dimensions-size') || '0'
            );
            const transposeDimensionsSize = Number.parseFloat(
                queryCaptionItem(transposeNode.id)?.style.getPropertyValue('--detail-transformer-view2d-caption-dimensions-size') || '0'
            );
            const preScoreDimensionsSize = Number.parseFloat(
                queryCaptionItem(preScoreNode.id)?.style.getPropertyValue('--detail-transformer-view2d-caption-dimensions-size') || '0'
            );

            expect(preScoreLabelSize).toBeGreaterThan(0);
            expect(preScoreLabelSize).toBeCloseTo(queryLabelSize, 1);
            expect(preScoreLabelSize).toBeCloseTo(transposeLabelSize, 1);
            expect(preScoreDimensionsSize).toBeGreaterThan(0);
            expect(preScoreDimensionsSize).toBeCloseTo(queryDimensionsSize, 1);
            expect(preScoreDimensionsSize).toBeCloseTo(transposeDimensionsSize, 1);
        } finally {
            cleanup();
        }
    });

    it('keeps the attention score-stage labels visible across zoom levels under uniform matrix sizing', () => {
        const fixtures = buildMhsaFixtures();
        const {
            scene,
            layout,
            preScoreNode,
            maskedInputNode,
            maskNode,
            postNode,
            canvas,
            overlay,
            cleanup
        } = fixtures;

        try {
            const preScoreEntry = layout.registry.getNodeEntry(preScoreNode.id);
            const zoomedOutScale = 0.45;
            const zoomedInScale = 0.95;
            const targetX = 40;
            const targetY = 48;
            const stageNodes = [
                preScoreNode,
                maskedInputNode,
                maskNode,
                postNode
            ];
            const projectBounds = (scale) => {
                const offsetX = targetX - ((Number(preScoreEntry?.contentBounds?.x) || 0) * scale);
                const offsetY = targetY - ((Number(preScoreEntry?.contentBounds?.y) || 0) * scale);
                return (bounds) => ({
                    x: (bounds.x * scale) + offsetX,
                    y: (bounds.y * scale) + offsetY,
                    width: bounds.width * scale,
                    height: bounds.height * scale
                });
            };
            const measureLabelSizes = (scale) => {
                overlay.sync({
                    scene,
                    layout,
                    canvas,
                    projectBounds: projectBounds(scale),
                    visible: true,
                    enabled: true
                });
                return stageNodes.map((node) => Number.parseFloat(
                    queryCaptionItem(node.id)?.style.getPropertyValue('--detail-transformer-view2d-caption-label-size') || '0'
                ));
            };

            const zoomedOutLabelSizes = measureLabelSizes(zoomedOutScale);
            const zoomedInLabelSizes = measureLabelSizes(zoomedInScale);

            zoomedOutLabelSizes.forEach((labelSize) => {
                expect(labelSize).toBeGreaterThan(0);
            });
            zoomedInLabelSizes.forEach((labelSize, index) => {
                expect(labelSize).toBeGreaterThan(zoomedOutLabelSizes[index]);
                expect(labelSize).toBeLessThanOrEqual(MHSA_STANDARD_LABEL_MAX_SCREEN_FONT_PX);
            });
        } finally {
            cleanup();
        }
    });

    it('keeps output-projection H_i captions visible even when zoomed far out', () => {
        const fixtures = buildOutputProjectionFixtures({
            canvasWidth: 1280,
            canvasHeight: 840
        });
        const {
            scene,
            layout,
            headMatrixNode,
            canvas,
            overlay,
            cleanup
        } = fixtures;

        try {
            const projectedScale = 0.06;
            const headEntry = layout.registry.getNodeEntry(headMatrixNode.id);
            const targetX = 40;
            const targetY = 48;
            const offsetX = targetX - ((Number(headEntry?.contentBounds?.x) || 0) * projectedScale);
            const offsetY = targetY - ((Number(headEntry?.contentBounds?.y) || 0) * projectedScale);
            overlay.sync({
                scene,
                layout,
                canvas,
                projectBounds: (bounds) => ({
                    x: (bounds.x * projectedScale) + offsetX,
                    y: (bounds.y * projectedScale) + offsetY,
                    width: bounds.width * projectedScale,
                    height: bounds.height * projectedScale
                }),
                visible: true,
                enabled: true
            });

            const captionItem = queryCaptionItem(headMatrixNode?.id || '');
            const labelSize = Number.parseFloat(
                captionItem?.style.getPropertyValue('--detail-transformer-view2d-caption-label-size') || '0'
            );

            expect(captionItem).toBeTruthy();
            expect(captionItem?.hidden).toBe(false);
            expect(labelSize).toBeGreaterThan(0);
        } finally {
            cleanup();
        }
    });

    it('renders the output-projection W_O caption with the standard inline-subscript styling', () => {
        const fixtures = buildOutputProjectionFixtures({
            canvasWidth: 1280,
            canvasHeight: 840
        });
        const {
            scene,
            layout,
            projectionWeightNode,
            canvas,
            overlay,
            cleanup
        } = fixtures;

        try {
            const weightEntry = layout.registry.getNodeEntry(projectionWeightNode.id);
            const projectedScale = resolveThresholdScale(weightEntry).scale * 1.8;
            const targetX = 60;
            const targetY = 56;
            const offsetX = targetX - ((Number(weightEntry?.contentBounds?.x) || 0) * projectedScale);
            const offsetY = targetY - ((Number(weightEntry?.contentBounds?.y) || 0) * projectedScale);

            overlay.sync({
                scene,
                layout,
                canvas,
                projectBounds: (bounds) => ({
                    x: (bounds.x * projectedScale) + offsetX,
                    y: (bounds.y * projectedScale) + offsetY,
                    width: bounds.width * projectedScale,
                    height: bounds.height * projectedScale
                }),
                visible: true,
                enabled: true
            });

            const weightLabelItem = queryCaptionItem(projectionWeightNode.id);
            const inlineSubscript = weightLabelItem?.querySelector('.detail-transformer-view2d-inline-subscript__sub');
            const labelSize = Number.parseFloat(
                weightLabelItem?.style.getPropertyValue('--detail-transformer-view2d-caption-label-size') || '0'
            );
            const dimensionsSize = Number.parseFloat(
                weightLabelItem?.style.getPropertyValue('--detail-transformer-view2d-caption-dimensions-size') || '0'
            );

            expect(labelSize).toBeGreaterThan(0);
            expect(dimensionsSize).toBeGreaterThan(0);
            expect(inlineSubscript?.textContent).toBe('O');
            expect(
                weightLabelItem?.style.getPropertyValue('--detail-transformer-view2d-caption-inline-subscript-scale')
            ).toBe('0.84em');
            expect(
                weightLabelItem?.style.getPropertyValue('--detail-transformer-view2d-caption-inline-subscript-offset')
            ).toBe('0.28em');
        } finally {
            cleanup();
        }
    });

    it('dampens the output-projection W_O caption zoom swing', () => {
        const fixtures = buildOutputProjectionFixtures({
            canvasWidth: 1280,
            canvasHeight: 840
        });
        const {
            scene,
            layout,
            projectionWeightNode,
            canvas,
            overlay,
            cleanup
        } = fixtures;

        try {
            const weightEntry = layout.registry.getNodeEntry(projectionWeightNode.id);
            const zoomedOutScale = resolveThresholdScale(weightEntry).scale * 0.55;
            const zoomedInScale = resolveThresholdScale(weightEntry).scale * 1.8;
            const targetX = 60;
            const targetY = 56;
            const projectBounds = (scale) => {
                const offsetX = targetX - ((Number(weightEntry?.contentBounds?.x) || 0) * scale);
                const offsetY = targetY - ((Number(weightEntry?.contentBounds?.y) || 0) * scale);
                return (bounds) => ({
                    x: (bounds.x * scale) + offsetX,
                    y: (bounds.y * scale) + offsetY,
                    width: bounds.width * scale,
                    height: bounds.height * scale
                });
            };

            overlay.sync({
                scene,
                layout,
                canvas,
                projectBounds: projectBounds(zoomedOutScale),
                visible: true,
                enabled: true
            });

            const zoomedOutLabelSize = Number.parseFloat(
                queryCaptionItem(projectionWeightNode.id)?.style.getPropertyValue('--detail-transformer-view2d-caption-label-size') || '0'
            );
            const zoomedOutItem = queryCaptionItem(projectionWeightNode.id);

            overlay.sync({
                scene,
                layout,
                canvas,
                projectBounds: projectBounds(zoomedInScale),
                visible: true,
                enabled: true
            });

            const zoomedInLabelSize = Number.parseFloat(
                queryCaptionItem(projectionWeightNode.id)?.style.getPropertyValue('--detail-transformer-view2d-caption-label-size') || '0'
            );
            const zoomedInItem = queryCaptionItem(projectionWeightNode.id);
            const labelScaleRatio = zoomedInLabelSize / Math.max(0.0001, zoomedOutLabelSize);
            const rawViewportScaleRatio = zoomedInScale / zoomedOutScale;

            expect(zoomedOutItem?.hidden).toBe(false);
            expect(zoomedInItem?.hidden).toBe(false);
            expect(zoomedOutLabelSize).toBeGreaterThan(0);
            expect(zoomedInLabelSize).toBeGreaterThan(zoomedOutLabelSize);
            expect(labelScaleRatio).toBeGreaterThan(1.2);
            expect(labelScaleRatio).toBeLessThan(rawViewportScaleRatio * 0.92);
        } finally {
            cleanup();
        }
    });

    it('dims Q and V captions while keeping H_i active during K-row detail focus', () => {
        const fixtures = buildMhsaFixtures();
        const {
            scene,
            layout,
            outputNode,
            queryNode,
            valuePostNode,
            headOutputNode,
            canvas,
            overlay,
            cleanup
        } = fixtures;

        try {
            const outputEntry = layout.registry.getNodeEntry(outputNode.id);
            const visibleScale = resolveThresholdScale(outputEntry).scale * 1.1;
            const projectBounds = (bounds) => ({
                x: bounds.x,
                y: bounds.y,
                width: bounds.width * visibleScale,
                height: bounds.height * visibleScale
            });
            const index = createMhsaDetailSceneIndex(scene);
            const hoverState = resolveMhsaDetailHoverState(index, {
                node: outputNode,
                rowHit: {
                    rowIndex: 0,
                    rowItem: outputNode.rowItems[0]
                }
            });

            overlay.sync({
                scene,
                layout,
                canvas,
                projectBounds,
                visible: true,
                enabled: true,
                focusState: hoverState?.focusState || null
            });

            expect(queryCaptionItem(outputNode.id)?.hidden).toBe(false);
            expect(queryCaptionItem(queryNode.id)?.hidden).toBe(false);
            expect(queryCaptionItem(valuePostNode.id)?.hidden).toBe(false);
            expect(queryCaptionItem(headOutputNode.id)?.hidden).toBe(false);
            expect(Number.parseFloat(queryCaptionItem(outputNode.id)?.style.opacity || '0')).toBeCloseTo(1, 3);
            expect(Number.parseFloat(queryCaptionItem(queryNode.id)?.style.opacity || '0')).toBeCloseTo(0.18, 3);
            expect(Number.parseFloat(queryCaptionItem(valuePostNode.id)?.style.opacity || '0')).toBeCloseTo(0.18, 3);
            expect(Number.parseFloat(queryCaptionItem(headOutputNode.id)?.style.opacity || '0')).toBeCloseTo(1, 3);
        } finally {
            cleanup();
        }
    });

    it('increases vertical spacing between the Q, K, and V projection equations for taller token windows', () => {
        const baseFixtures = buildMhsaFixtures();
        const tallFixtures = buildMhsaFixtures({
            tokenLabels: Array.from({ length: 12 }, (_, index) => `Token ${index + 1}`)
        });

        try {
            const baseQEntry = baseFixtures.layout.registry.getNodeEntry(baseFixtures.qStageNode.id);
            const baseKEntry = baseFixtures.layout.registry.getNodeEntry(baseFixtures.kStageNode.id);
            const baseVEntry = baseFixtures.layout.registry.getNodeEntry(baseFixtures.vStageNode.id);
            const tallQEntry = tallFixtures.layout.registry.getNodeEntry(tallFixtures.qStageNode.id);
            const tallKEntry = tallFixtures.layout.registry.getNodeEntry(tallFixtures.kStageNode.id);
            const tallVEntry = tallFixtures.layout.registry.getNodeEntry(tallFixtures.vStageNode.id);

            const baseQToKGap = (baseKEntry?.bounds?.y || 0)
                - ((baseQEntry?.bounds?.y || 0) + (baseQEntry?.bounds?.height || 0));
            const baseKToVGap = (baseVEntry?.bounds?.y || 0)
                - ((baseKEntry?.bounds?.y || 0) + (baseKEntry?.bounds?.height || 0));
            const tallQToKGap = (tallKEntry?.bounds?.y || 0)
                - ((tallQEntry?.bounds?.y || 0) + (tallQEntry?.bounds?.height || 0));
            const tallKToVGap = (tallVEntry?.bounds?.y || 0)
                - ((tallKEntry?.bounds?.y || 0) + (tallKEntry?.bounds?.height || 0));

            expect(Number(baseFixtures.projectionStackNode?.metadata?.gapOverride) || 0)
                .toBeLessThan(Number(tallFixtures.projectionStackNode?.metadata?.gapOverride) || 0);
            expect(tallQToKGap).toBeGreaterThan(baseQToKGap);
            expect(tallKToVGap).toBeGreaterThan(baseKToVGap);
        } finally {
            tallFixtures.cleanup();
            baseFixtures.cleanup();
        }
    });
});
