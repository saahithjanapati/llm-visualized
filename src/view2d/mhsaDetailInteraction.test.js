import { describe, expect, it } from 'vitest';

import { D_HEAD, D_MODEL } from '../ui/selectionPanelConstants.js';
import { buildSceneLayout } from './layout/buildSceneLayout.js';
import { buildMhsaSceneModel } from './model/buildMhsaSceneModel.js';
import {
    flattenSceneNodes,
    VIEW2D_MATRIX_PRESENTATIONS
} from './schema/sceneTypes.js';
import {
    createMhsaDetailSceneIndex,
    resolveMhsaDetailHoverState
} from './mhsaDetailInteraction.js';

const TOKEN_LABELS = ['Token A', 'Token B'];
const CONNECTOR_CAPTION_EXIT_GAP = 4;
const DECODE_CONNECTOR_CAPTION_EXIT_GAP = 2;
const PRE_CONNECTOR_SOURCE_OFFSET_Y = 16;
const VALUE_CONNECTOR_SOURCE_OFFSET_Y = 0;
const VALUE_CONNECTOR_SOURCE_GAP = 8;
const VALUE_CONNECTOR_TARGET_GAP = 8;
const DECODE_KEY_CONNECTOR_SOURCE_GAP = 4;
const DECODE_VALUE_CONNECTOR_SOURCE_GAP = 4;
const DECODE_VALUE_CONNECTOR_TARGET_GAP = 4;
const DECODE_CONCAT_SOURCE_OFFSET_Y_RATIO = -0.25;
const DECODE_CACHE_NEXT_SOURCE_OFFSET_Y_RATIO = 0.25;
const DECODE_CACHE_NEXT_TARGET_GAP = 4;
const DECODE_CACHE_NEXT_OPACITY = 0.4;

function createVectorValues(seed = 0) {
    return Array.from({ length: D_HEAD }, (_, index) => Number((seed + (index * 0.01)).toFixed(4)));
}

function createTokenLabels(count = TOKEN_LABELS.length) {
    return Array.from({ length: count }, (_, index) => `Token ${String.fromCharCode(65 + index)}`);
}

function createBaseRows(tokenLabels = TOKEN_LABELS) {
    return tokenLabels.map((tokenLabel, rowIndex) => ({
        rowIndex,
        tokenIndex: rowIndex,
        tokenLabel,
        rawValues: createVectorValues(rowIndex),
        gradientCss: `rgba(${120 + (rowIndex * 16)}, 220, 255, 0.9)`
    }));
}

function createProjectionOutputRows(label = 'Q', tokenLabels = TOKEN_LABELS) {
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

function createGridRows(fillCss = 'rgba(255, 255, 255, 0.28)', tokenLabels = TOKEN_LABELS, {
    stageKey = 'pre'
} = {}) {
    return tokenLabels.map((tokenLabel, rowIndex) => ({
        rowIndex,
        tokenIndex: rowIndex,
        tokenLabel,
        cells: tokenLabels.map((colLabel, colIndex) => {
            const isMasked = colIndex > rowIndex;
            const preScore = Number(((rowIndex + 1) * (colIndex + 1) * 0.125).toFixed(3));
            const postScore = Number((((rowIndex + 1) + (colIndex + 1)) * 0.05).toFixed(3));
            const maskValue = isMasked ? Number.NEGATIVE_INFINITY : 0;
            return {
                rowIndex,
                colIndex,
                rowTokenLabel: tokenLabel,
                colTokenLabel: colLabel,
                queryTokenIndex: rowIndex,
                keyTokenIndex: colIndex,
                queryTokenLabel: tokenLabel,
                keyTokenLabel: colLabel,
                preScore,
                postScore: isMasked ? 0 : postScore,
                maskValue,
                rawValue: stageKey === 'mask'
                    ? maskValue
                    : (stageKey === 'post' ? (isMasked ? null : postScore) : preScore),
                fillCss,
                isMasked,
                isEmpty: stageKey === 'post' ? isMasked : false,
                title: `${tokenLabel} -> ${colLabel}`
            };
        }),
        hasAnyValue: true
    }));
}

function createPreviewData(tokenCount = TOKEN_LABELS.length) {
    const tokenLabels = createTokenLabels(tokenCount);
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

    const attentionGridRows = createGridRows('rgba(255, 255, 255, 0.28)', tokenLabels, {
        stageKey: 'pre'
    });
    const maskGridRows = createGridRows('rgba(0, 0, 0, 0.94)', tokenLabels, {
        stageKey: 'mask'
    });
    const postGridRows = createGridRows('rgba(160, 220, 255, 0.34)', tokenLabels, {
        stageKey: 'post'
    });

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
            maskRows: maskGridRows,
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

function buildSceneFixtures(tokenCount = TOKEN_LABELS.length, {
    isSmallScreen = false,
    kvCacheState = null
} = {}) {
    const previewData = createPreviewData(tokenCount);
    const scene = buildMhsaSceneModel({
        previewData,
        layerIndex: 2,
        headIndex: 1,
        isSmallScreen,
        kvCacheState
    });
    const nodes = flattenSceneNodes(scene);
    const projectionSourceNode = nodes.find((node) => node.role === 'projection-source-xln') || null;
    const transposeNode = nodes.find((node) => node.role === 'attention-key-transpose') || null;
    const queryNode = nodes.find((node) => node.role === 'attention-query-source') || null;
    const queryInputNode = nodes.find((node) => (
        node.role === 'x-ln-copy'
        && String(node.semantic?.branchKey || '').toLowerCase() === 'q'
    )) || null;
    const keyInputNode = nodes.find((node) => (
        node.role === 'x-ln-copy'
        && String(node.semantic?.branchKey || '').toLowerCase() === 'k'
    )) || null;
    const valueInputNode = nodes.find((node) => (
        node.role === 'x-ln-copy'
        && String(node.semantic?.branchKey || '').toLowerCase() === 'v'
    )) || null;
    const queryWeightNode = nodes.find((node) => (
        node.role === 'projection-weight'
        && String(node.metadata?.kind || '').toLowerCase() === 'q'
    )) || null;
    const queryBiasNode = nodes.find((node) => (
        node.role === 'projection-bias'
        && String(node.metadata?.kind || '').toLowerCase() === 'q'
    )) || null;
    const keyWeightNode = nodes.find((node) => (
        node.role === 'projection-weight'
        && String(node.metadata?.kind || '').toLowerCase() === 'k'
    )) || null;
    const keyBiasNode = nodes.find((node) => (
        node.role === 'projection-bias'
        && String(node.metadata?.kind || '').toLowerCase() === 'k'
    )) || null;
    const valueWeightNode = nodes.find((node) => (
        node.role === 'projection-weight'
        && String(node.metadata?.kind || '').toLowerCase() === 'v'
    )) || null;
    const valueBiasNode = nodes.find((node) => (
        node.role === 'projection-bias'
        && String(node.metadata?.kind || '').toLowerCase() === 'v'
    )) || null;
    const queryProjectionOutputNode = nodes.find((node) => (
        node.role === 'projection-output'
        && String(node.metadata?.kind || '').toLowerCase() === 'q'
    )) || null;
    const valueProjectionOutputNode = nodes.find((node) => (
        node.role === 'projection-output'
        && String(node.metadata?.kind || '').toLowerCase() === 'v'
    )) || null;
    const preScoreNode = nodes.find((node) => node.role === 'attention-pre-score') || null;
    const attentionEqualsNode = nodes.find((node) => node.role === 'attention-equals') || null;
    const attentionDivideNode = nodes.find((node) => node.role === 'attention-divide') || null;
    const attentionScaleNode = nodes.find((node) => node.role === 'attention-scale') || null;
    const divisorClusterNode = nodes.find((node) => node.role === 'attention-divisor-cluster') || null;
    const divisorCloseNode = divisorClusterNode?.children?.find((node) => (
        node.role === 'attention-close' && node.semantic?.clusterKey === 'divisor'
    )) || null;
    const attentionStageNode = nodes.find((node) => node.role === 'attention-stage') || null;
    const maskedInputNode = nodes.find((node) => node.role === 'attention-masked-input') || null;
    const postCopyNode = nodes.find((node) => node.role === 'attention-post-copy') || null;
    const softmaxLabelNode = nodes.find((node) => node.role === 'attention-softmax-label') || null;
    const softmaxOpenNode = nodes.find((node) => node.role === 'attention-softmax-open') || null;
    const softmaxCloseNode = nodes.find((node) => node.role === 'attention-softmax-close') || null;
    const findDecodeConcatNode = (role, kind) => nodes.find((node) => (
        node.role === role
        && String(node.metadata?.kind || node.semantic?.branchKey || '').toLowerCase() === kind
    )) || null;
    const keyProjectionOutputNode = nodes.find((node) => (
        node.role === 'projection-output'
        && String(node.metadata?.kind || '').toLowerCase() === 'k'
    )) || null;
    const keyProjectionOutputCopyNode = nodes.find((node) => (
        node.role === 'projection-output-copy'
        && String(node.metadata?.kind || '').toLowerCase() === 'k'
    )) || null;
    const keyCacheNode = nodes.find((node) => (
        node.role === 'projection-cache'
        && String(node.semantic?.branchKey || '').toLowerCase() === 'k'
    )) || null;
    const keyCacheSourceNode = nodes.find((node) => (
        node.role === 'projection-cache-source'
        && String(node.semantic?.branchKey || node.metadata?.kind || '').toLowerCase() === 'k'
    )) || null;
    const valueCacheNode = nodes.find((node) => (
        node.role === 'projection-cache'
        && String(node.semantic?.branchKey || '').toLowerCase() === 'v'
    )) || null;
    const valueCacheSourceNode = nodes.find((node) => (
        node.role === 'projection-cache-source'
        && String(node.semantic?.branchKey || node.metadata?.kind || '').toLowerCase() === 'v'
    )) || null;
    const valueProjectionOutputCopyNode = nodes.find((node) => (
        node.role === 'projection-output-copy'
        && String(node.metadata?.kind || '').toLowerCase() === 'v'
    )) || null;
    const keyCacheConcatLabelNode = findDecodeConcatNode('projection-cache-concat-label', 'k');
    const keyCacheConcatOpenNode = findDecodeConcatNode('projection-cache-concat-open', 'k');
    const keyCacheConcatCloseNode = findDecodeConcatNode('projection-cache-concat-close', 'k');
    const keyCacheConcatEqualsNode = findDecodeConcatNode('projection-cache-concat-equals', 'k');
    const keyCacheConcatResultNode = findDecodeConcatNode('projection-cache-concat-result', 'k');
    const keyCacheNextNode = findDecodeConcatNode('projection-cache-next', 'k');
    const valueCacheConcatLabelNode = findDecodeConcatNode('projection-cache-concat-label', 'v');
    const valueCacheConcatOpenNode = findDecodeConcatNode('projection-cache-concat-open', 'v');
    const valueCacheConcatCloseNode = findDecodeConcatNode('projection-cache-concat-close', 'v');
    const valueCacheConcatEqualsNode = findDecodeConcatNode('projection-cache-concat-equals', 'v');
    const valueCacheConcatResultNode = findDecodeConcatNode('projection-cache-concat-result', 'v');
    const valueCacheNextNode = findDecodeConcatNode('projection-cache-next', 'v');
    const qktEquationNode = nodes.find((node) => node.role === 'attention-qkt-equation') || null;
    const connectorKNode = nodes.find((node) => node.role === 'connector-k') || null;
    const connectorQNode = nodes.find((node) => node.role === 'connector-q') || null;
    const connectorPreNode = nodes.find((node) => node.role === 'connector-pre') || null;
    const connectorPostNode = nodes.find((node) => node.role === 'connector-post') || null;
    const connectorVNode = nodes.find((node) => node.role === 'connector-v') || null;
    const connectorKCacheSourceNode = nodes.find((node) => node.role === 'connector-k-cache-source') || null;
    const connectorKCacheNode = nodes.find((node) => node.role === 'connector-k-cache') || null;
    const connectorVCacheSourceNode = nodes.find((node) => node.role === 'connector-v-cache-source') || null;
    const connectorVCacheNode = nodes.find((node) => node.role === 'connector-v-cache') || null;
    const connectorKCacheCopyNode = nodes.find((node) => node.role === 'connector-k-cache-copy') || null;
    const connectorVCacheCopyNode = nodes.find((node) => node.role === 'connector-v-cache-copy') || null;
    const connectorKCacheNextNode = nodes.find((node) => node.role === 'connector-k-cache-next') || null;
    const connectorVCacheNextNode = nodes.find((node) => node.role === 'connector-v-cache-next') || null;
    const connectorXlnQNode = nodes.find((node) => node.role === 'connector-xln-q') || null;
    const connectorXlnKNode = nodes.find((node) => node.role === 'connector-xln-k') || null;
    const connectorXlnVNode = nodes.find((node) => node.role === 'connector-xln-v') || null;
    const projectionIngressConnectorNodes = nodes.filter((node) => (
        node.role === 'connector-xln-q'
        || node.role === 'connector-xln-k'
        || node.role === 'connector-xln-v'
    ));
    const valuePostNode = nodes.find((node) => node.role === 'attention-value-post') || null;
    const headOutputNode = nodes.find((node) => node.role === 'attention-head-output') || null;
    const maskNode = nodes.find((node) => node.role === 'attention-mask') || null;
    const postNode = nodes.find((node) => node.role === 'attention-post') || null;
    const index = createMhsaDetailSceneIndex(scene);
    const layout = buildSceneLayout(scene, { isSmallScreen });

    return {
        previewData,
        scene,
        layout,
        index,
        projectionSourceNode,
        transposeNode,
        queryNode,
        queryInputNode,
        keyInputNode,
        valueInputNode,
        queryWeightNode,
        queryBiasNode,
        keyWeightNode,
        keyBiasNode,
        valueWeightNode,
        valueBiasNode,
        queryProjectionOutputNode,
        valueProjectionOutputNode,
        preScoreNode,
        attentionEqualsNode,
        attentionDivideNode,
        attentionScaleNode,
        divisorClusterNode,
        divisorCloseNode,
        attentionStageNode,
        maskedInputNode,
        postCopyNode,
        softmaxLabelNode,
        softmaxOpenNode,
        softmaxCloseNode,
        keyProjectionOutputNode,
        keyProjectionOutputCopyNode,
        keyCacheNode,
        keyCacheSourceNode,
        valueCacheNode,
        valueCacheSourceNode,
        valueProjectionOutputCopyNode,
        keyCacheConcatLabelNode,
        keyCacheConcatOpenNode,
        keyCacheConcatCloseNode,
        keyCacheConcatEqualsNode,
        keyCacheConcatResultNode,
        keyCacheNextNode,
        valueCacheConcatLabelNode,
        valueCacheConcatOpenNode,
        valueCacheConcatCloseNode,
        valueCacheConcatEqualsNode,
        valueCacheConcatResultNode,
        valueCacheNextNode,
        qktEquationNode,
        connectorQNode,
        connectorKNode,
        connectorPreNode,
        connectorPostNode,
        connectorVNode,
        connectorKCacheSourceNode,
        connectorKCacheNode,
        connectorVCacheSourceNode,
        connectorVCacheNode,
        connectorKCacheCopyNode,
        connectorVCacheCopyNode,
        connectorKCacheNextNode,
        connectorVCacheNextNode,
        connectorXlnQNode,
        connectorXlnKNode,
        connectorXlnVNode,
        projectionIngressConnectorNodes,
        valuePostNode,
        headOutputNode,
        maskNode,
        postNode
    };
}

function resolveEntryCaptionBottom(entry = null) {
    if (!entry) return 0;
    const boundsBottom = (entry.bounds?.y || 0) + (entry.bounds?.height || 0);
    const labelBottom = entry.labelBounds
        ? (entry.labelBounds.y + entry.labelBounds.height)
        : 0;
    const dimensionsBottom = entry.dimensionBounds
        ? (entry.dimensionBounds.y + entry.dimensionBounds.height)
        : 0;
    return Math.max(boundsBottom, labelBottom, dimensionsBottom);
}

function resolveEntryDimensionTop(entry = null) {
    if (!entry) return 0;
    if (Number.isFinite(entry.dimensionBounds?.y)) {
        return entry.dimensionBounds.y;
    }
    const contentBottom = (entry.contentBounds?.y || 0) + (entry.contentBounds?.height || 0);
    const labelBottom = entry.labelBounds
        ? (entry.labelBounds.y + entry.labelBounds.height)
        : 0;
    return Math.max(contentBottom, labelBottom);
}

function resolveHorizontalContentGap(leftEntry = null, rightEntry = null) {
    const leftRight = (leftEntry?.contentBounds?.x || 0) + (leftEntry?.contentBounds?.width || 0);
    return Math.round((rightEntry?.contentBounds?.x || 0) - leftRight);
}

const ATTENTION_GRID_CELL_CORNER_RADIUS_SCALE = 0.9;

describe('MHSA detail transpose view', () => {
    it('renders K^T as a transpose column strip with transpose-aware dimensions', () => {
        const {
            previewData,
            transposeNode,
            queryNode
        } = buildSceneFixtures();

        expect(transposeNode).toBeTruthy();
        expect(queryNode).toBeTruthy();
        expect(transposeNode.presentation).toBe(VIEW2D_MATRIX_PRESENTATIONS.COLUMN_STRIP);
        expect(transposeNode.metadata?.columnStrip).toBeTruthy();
        expect(transposeNode.metadata?.compactRows).toBeUndefined();
        expect(transposeNode.columnItems).toHaveLength(previewData.attentionScoreStage.transposeColumnCount);
        expect(transposeNode.dimensions).toEqual({
            rows: previewData.attentionScoreStage.transposeRowCount,
            cols: previewData.attentionScoreStage.transposeColumnCount
        });
        expect(transposeNode.metadata?.measure).toEqual({
            cols: previewData.attentionScoreStage.transposeColumnCount,
            rows: previewData.attentionScoreStage.transposeRowCount
        });
        expect(transposeNode.metadata?.columnStrip?.colWidth).toBe(queryNode.metadata?.compactRows?.rowHeight);
        expect(transposeNode.metadata?.columnStrip?.colHeight).toBe(queryNode.metadata?.compactRows?.compactWidth);
    });

    it('treats direct K^T hover as a column selection on the transpose strip', () => {
        const {
            index,
            transposeNode,
            projectionSourceNode,
            queryNode,
            queryInputNode,
            queryWeightNode,
            queryBiasNode,
            queryProjectionOutputNode,
            valueInputNode,
            valueWeightNode,
            valueBiasNode,
            valueProjectionOutputNode,
            connectorKNode,
            connectorPreNode,
            connectorPostNode,
            connectorVNode,
            valuePostNode,
            headOutputNode
        } = buildSceneFixtures();

        const hoverState = resolveMhsaDetailHoverState(index, {
            node: transposeNode,
            columnHit: {
                colIndex: 1,
                columnItem: transposeNode.columnItems[1]
            }
        });

        expect(hoverState?.label).toBe('Key Vector');
        expect(hoverState?.focusState?.columnSelections).toContainEqual({
            nodeId: transposeNode.id,
            colIndex: 1
        });
        expect(hoverState?.focusState?.rowSelections).toContainEqual({
            nodeId: projectionSourceNode.id,
            rowIndex: 1
        });
        expect(hoverState?.focusState?.activeConnectorIds).toContain(connectorKNode.id);
        expect(hoverState?.focusState?.activeConnectorIds).toContain(connectorPreNode.id);
        expect(hoverState?.focusState?.activeConnectorIds).toContain(connectorPostNode.id);
        expect(hoverState?.focusState?.activeConnectorIds).not.toContain(connectorVNode.id);
        expect(hoverState?.focusState?.activeNodeIds).not.toContain(queryNode.id);
        expect(hoverState?.focusState?.activeNodeIds).not.toContain(valuePostNode.id);
        expect(hoverState?.focusState?.activeNodeIds).toContain(headOutputNode.id);
        expect(hoverState?.focusState?.activeNodeIds).not.toContain(queryInputNode.id);
        expect(hoverState?.focusState?.activeNodeIds).not.toContain(queryWeightNode.id);
        expect(hoverState?.focusState?.activeNodeIds).not.toContain(queryBiasNode.id);
        expect(hoverState?.focusState?.activeNodeIds).not.toContain(queryProjectionOutputNode.id);
        expect(hoverState?.focusState?.activeNodeIds).not.toContain(valueInputNode.id);
        expect(hoverState?.focusState?.activeNodeIds).not.toContain(valueWeightNode.id);
        expect(hoverState?.focusState?.activeNodeIds).not.toContain(valueBiasNode.id);
        expect(hoverState?.focusState?.activeNodeIds).not.toContain(valueProjectionOutputNode.id);
        expect(
            hoverState?.focusState?.rowSelections?.some((selection) => selection.nodeId === transposeNode.id)
        ).toBe(false);
    });

    it('propagates key-path focus from K rows into the transpose and post-softmax column selections', () => {
        const {
            index,
            transposeNode,
            keyProjectionOutputNode,
            queryNode,
            queryInputNode,
            queryWeightNode,
            queryBiasNode,
            queryProjectionOutputNode,
            valueInputNode,
            valueWeightNode,
            valueBiasNode,
            valueProjectionOutputNode,
            postNode,
            postCopyNode,
            connectorKNode,
            connectorPreNode,
            connectorPostNode,
            connectorVNode,
            valuePostNode,
            headOutputNode
        } = buildSceneFixtures();

        const hoverState = resolveMhsaDetailHoverState(index, {
            node: keyProjectionOutputNode,
            rowHit: {
                rowIndex: 0,
                rowItem: keyProjectionOutputNode.rowItems[0]
            }
        });

        expect(hoverState?.label).toBe('Key Vector');
        expect(hoverState?.focusState?.columnSelections).toContainEqual({
            nodeId: transposeNode.id,
            colIndex: 0
        });
        expect(hoverState?.focusState?.columnSelections).toContainEqual({
            nodeId: postNode.id,
            colIndex: 0
        });
        expect(hoverState?.focusState?.columnSelections).toContainEqual({
            nodeId: postCopyNode.id,
            colIndex: 0
        });
        expect(hoverState?.focusState?.activeNodeIds).toContain(postCopyNode.id);
        expect(hoverState?.focusState?.activeNodeIds).not.toContain(queryNode.id);
        expect(hoverState?.focusState?.activeNodeIds).not.toContain(valuePostNode.id);
        expect(hoverState?.focusState?.activeNodeIds).toContain(headOutputNode.id);
        expect(hoverState?.focusState?.activeNodeIds).not.toContain(queryInputNode.id);
        expect(hoverState?.focusState?.activeNodeIds).not.toContain(queryWeightNode.id);
        expect(hoverState?.focusState?.activeNodeIds).not.toContain(queryBiasNode.id);
        expect(hoverState?.focusState?.activeNodeIds).not.toContain(queryProjectionOutputNode.id);
        expect(hoverState?.focusState?.activeNodeIds).not.toContain(valueInputNode.id);
        expect(hoverState?.focusState?.activeNodeIds).not.toContain(valueWeightNode.id);
        expect(hoverState?.focusState?.activeNodeIds).not.toContain(valueBiasNode.id);
        expect(hoverState?.focusState?.activeNodeIds).not.toContain(valueProjectionOutputNode.id);
        expect(hoverState?.focusState?.activeConnectorIds).toContain(connectorKNode.id);
        expect(hoverState?.focusState?.activeConnectorIds).toContain(connectorPreNode.id);
        expect(hoverState?.focusState?.activeConnectorIds).toContain(connectorPostNode.id);
        expect(hoverState?.focusState?.activeConnectorIds).not.toContain(connectorVNode.id);
    });

    it('propagates query-path focus into post-softmax rows and the matching head-output row', () => {
        const {
            index,
            projectionSourceNode,
            queryProjectionOutputNode,
            keyInputNode,
            valueInputNode,
            keyWeightNode,
            keyBiasNode,
            keyProjectionOutputNode,
            transposeNode,
            valueWeightNode,
            valueBiasNode,
            valueProjectionOutputNode,
            postNode,
            postCopyNode,
            headOutputNode,
            connectorQNode,
            connectorKNode,
            connectorVNode,
            valuePostNode
        } = buildSceneFixtures();

        const hoverState = resolveMhsaDetailHoverState(index, {
            node: queryProjectionOutputNode,
            rowHit: {
                rowIndex: 1,
                rowItem: queryProjectionOutputNode.rowItems[1]
            }
        });

        expect(hoverState?.label).toBe('Query Vector');
        expect(hoverState?.focusState?.rowSelections).toContainEqual({
            nodeId: projectionSourceNode.id,
            rowIndex: 1
        });
        expect(hoverState?.focusState?.rowSelections).toContainEqual({
            nodeId: postNode.id,
            rowIndex: 1
        });
        expect(hoverState?.focusState?.rowSelections).toContainEqual({
            nodeId: postCopyNode.id,
            rowIndex: 1
        });
        expect(hoverState?.focusState?.activeNodeIds).toContain(postCopyNode.id);
        expect(hoverState?.focusState?.rowSelections).toContainEqual({
            nodeId: headOutputNode.id,
            rowIndex: 1
        });
        expect(hoverState?.focusState?.activeNodeIds).toContain(headOutputNode.id);
        expect(hoverState?.focusState?.activeConnectorIds).toContain(connectorQNode.id);
        expect(hoverState?.focusState?.activeConnectorIds).not.toContain(connectorKNode.id);
        expect(hoverState?.focusState?.activeConnectorIds).not.toContain(connectorVNode.id);
        expect(hoverState?.focusState?.activeNodeIds).not.toContain(keyInputNode.id);
        expect(hoverState?.focusState?.activeNodeIds).not.toContain(valueInputNode.id);
        expect(hoverState?.focusState?.activeNodeIds).not.toContain(keyWeightNode.id);
        expect(hoverState?.focusState?.activeNodeIds).not.toContain(keyBiasNode.id);
        expect(hoverState?.focusState?.activeNodeIds).not.toContain(keyProjectionOutputNode.id);
        expect(hoverState?.focusState?.activeNodeIds).not.toContain(transposeNode.id);
        expect(hoverState?.focusState?.activeNodeIds).not.toContain(valueWeightNode.id);
        expect(hoverState?.focusState?.activeNodeIds).not.toContain(valueBiasNode.id);
        expect(hoverState?.focusState?.activeNodeIds).not.toContain(valueProjectionOutputNode.id);
        expect(hoverState?.focusState?.activeNodeIds).not.toContain(valuePostNode.id);
    });

    it('narrows W_Q hover to the local X_ln, Q, and query-source path', () => {
        const {
            index,
            projectionSourceNode,
            queryInputNode,
            queryWeightNode,
            queryProjectionOutputNode,
            queryNode,
            keyProjectionOutputNode,
            valueProjectionOutputNode,
            connectorQNode,
            connectorXlnQNode,
            connectorKNode
        } = buildSceneFixtures();

        const hoverState = resolveMhsaDetailHoverState(index, {
            node: queryWeightNode
        });

        expect(hoverState?.label).toBe('Query Weight Matrix');
        expect(hoverState?.focusState?.activeNodeIds).toContain(projectionSourceNode.id);
        expect(hoverState?.focusState?.activeNodeIds).toContain(queryInputNode.id);
        expect(hoverState?.focusState?.activeNodeIds).toContain(queryWeightNode.id);
        expect(hoverState?.focusState?.activeNodeIds).toContain(queryProjectionOutputNode.id);
        expect(hoverState?.focusState?.activeNodeIds).toContain(queryNode.id);
        expect(hoverState?.focusState?.activeNodeIds).not.toContain(keyProjectionOutputNode.id);
        expect(hoverState?.focusState?.activeNodeIds).not.toContain(valueProjectionOutputNode.id);
        expect(hoverState?.focusState?.activeConnectorIds).toContain(connectorXlnQNode.id);
        expect(hoverState?.focusState?.activeConnectorIds).toContain(connectorQNode.id);
        expect(hoverState?.focusState?.activeConnectorIds).not.toContain(connectorKNode.id);
    });

    it('narrows W_K hover to the local X_ln, K, and K^T path', () => {
        const {
            index,
            projectionSourceNode,
            queryInputNode,
            keyInputNode,
            valueInputNode,
            queryWeightNode,
            queryBiasNode,
            keyWeightNode,
            keyProjectionOutputNode,
            queryProjectionOutputNode,
            valueWeightNode,
            valueBiasNode,
            valueProjectionOutputNode,
            transposeNode,
            queryNode,
            connectorXlnKNode,
            connectorKNode,
            connectorVNode
        } = buildSceneFixtures();

        const hoverState = resolveMhsaDetailHoverState(index, {
            node: keyWeightNode
        });

        expect(hoverState?.label).toBe('Key Weight Matrix');
        expect(hoverState?.focusState?.activeNodeIds).toContain(projectionSourceNode.id);
        expect(hoverState?.focusState?.activeNodeIds).toContain(keyInputNode.id);
        expect(hoverState?.focusState?.activeNodeIds).toContain(keyWeightNode.id);
        expect(hoverState?.focusState?.activeNodeIds).toContain(keyProjectionOutputNode.id);
        expect(hoverState?.focusState?.activeNodeIds).toContain(transposeNode.id);
        expect(hoverState?.focusState?.activeNodeIds).not.toContain(queryInputNode.id);
        expect(hoverState?.focusState?.activeNodeIds).not.toContain(valueInputNode.id);
        expect(hoverState?.focusState?.activeNodeIds).not.toContain(queryWeightNode.id);
        expect(hoverState?.focusState?.activeNodeIds).not.toContain(queryBiasNode.id);
        expect(hoverState?.focusState?.activeNodeIds).not.toContain(queryProjectionOutputNode.id);
        expect(hoverState?.focusState?.activeNodeIds).not.toContain(valueWeightNode.id);
        expect(hoverState?.focusState?.activeNodeIds).not.toContain(valueBiasNode.id);
        expect(hoverState?.focusState?.activeNodeIds).not.toContain(valueProjectionOutputNode.id);
        expect(hoverState?.focusState?.activeNodeIds).not.toContain(queryNode.id);
        expect(hoverState?.focusState?.activeConnectorIds).toContain(connectorXlnKNode.id);
        expect(hoverState?.focusState?.activeConnectorIds).toContain(connectorKNode.id);
        expect(hoverState?.focusState?.activeConnectorIds).not.toContain(connectorVNode.id);
    });

    it('narrows W_V hover to the local X_ln, V, and weighted-value path', () => {
        const {
            index,
            projectionSourceNode,
            queryInputNode,
            keyInputNode,
            queryWeightNode,
            keyWeightNode,
            queryBiasNode,
            keyBiasNode,
            queryProjectionOutputNode,
            keyProjectionOutputNode,
            queryNode,
            transposeNode,
            valueInputNode,
            valueWeightNode,
            valueProjectionOutputNode,
            valuePostNode,
            headOutputNode,
            connectorXlnVNode,
            connectorVNode,
            connectorQNode
        } = buildSceneFixtures();

        const hoverState = resolveMhsaDetailHoverState(index, {
            node: valueWeightNode
        });

        expect(hoverState?.label).toBe('Value Weight Matrix');
        expect(hoverState?.focusState?.activeNodeIds).toContain(projectionSourceNode.id);
        expect(hoverState?.focusState?.activeNodeIds).toContain(valueInputNode.id);
        expect(hoverState?.focusState?.activeNodeIds).toContain(valueWeightNode.id);
        expect(hoverState?.focusState?.activeNodeIds).toContain(valueProjectionOutputNode.id);
        expect(hoverState?.focusState?.activeNodeIds).toContain(valuePostNode.id);
        expect(hoverState?.focusState?.activeNodeIds).not.toContain(headOutputNode.id);
        expect(hoverState?.focusState?.activeConnectorIds).toContain(connectorXlnVNode.id);
        expect(hoverState?.focusState?.activeConnectorIds).toContain(connectorVNode.id);
        expect(hoverState?.focusState?.activeConnectorIds).not.toContain(connectorQNode.id);
        expect(hoverState?.focusState?.dimNodeIds).toContain(queryInputNode.id);
        expect(hoverState?.focusState?.dimNodeIds).toContain(keyInputNode.id);
        expect(hoverState?.focusState?.dimNodeIds).toContain(queryWeightNode.id);
        expect(hoverState?.focusState?.dimNodeIds).toContain(keyWeightNode.id);
        expect(hoverState?.focusState?.dimNodeIds).toContain(queryBiasNode.id);
        expect(hoverState?.focusState?.dimNodeIds).toContain(keyBiasNode.id);
        expect(hoverState?.focusState?.dimNodeIds).toContain(queryProjectionOutputNode.id);
        expect(hoverState?.focusState?.dimNodeIds).toContain(keyProjectionOutputNode.id);
        expect(hoverState?.focusState?.dimNodeIds).toContain(queryNode.id);
        expect(hoverState?.focusState?.dimNodeIds).toContain(transposeNode.id);
    });

    it('shows a 2D-only tooltip for the query bias vector with a head-bias payload', () => {
        const {
            index,
            projectionSourceNode,
            queryInputNode,
            queryWeightNode,
            queryBiasNode,
            queryProjectionOutputNode
        } = buildSceneFixtures();

        const hoverState = resolveMhsaDetailHoverState(index, {
            node: queryBiasNode,
            rowHit: {
                rowIndex: 0,
                rowItem: queryBiasNode?.rowItems?.[0]
            }
        });

        expect(hoverState?.label).toBe('Query Bias Vector');
        expect(hoverState?.info?.activationData?.stage).toBe('qkv.q.bias');
        expect(hoverState?.info?.activationData?.parameterType).toBe('bias');
        expect(hoverState?.info?.activationData?.values).toHaveLength(1);
        expect(hoverState?.focusState?.activeNodeIds).toContain(projectionSourceNode.id);
        expect(hoverState?.focusState?.activeNodeIds).toContain(queryInputNode.id);
        expect(hoverState?.focusState?.activeNodeIds).toContain(queryWeightNode.id);
        expect(hoverState?.focusState?.activeNodeIds).toContain(queryBiasNode.id);
        expect(hoverState?.focusState?.activeNodeIds).toContain(queryProjectionOutputNode.id);
    });

    it('shows a 2D-only tooltip for the key bias vector with a head-bias payload', () => {
        const {
            index,
            projectionSourceNode,
            keyInputNode,
            keyWeightNode,
            keyBiasNode,
            keyProjectionOutputNode,
            transposeNode
        } = buildSceneFixtures();

        const hoverState = resolveMhsaDetailHoverState(index, {
            node: keyBiasNode,
            rowHit: {
                rowIndex: 0,
                rowItem: keyBiasNode?.rowItems?.[0]
            }
        });

        expect(hoverState?.label).toBe('Key Bias Vector');
        expect(hoverState?.info?.activationData?.stage).toBe('qkv.k.bias');
        expect(hoverState?.info?.activationData?.parameterType).toBe('bias');
        expect(hoverState?.info?.activationData?.values).toHaveLength(1);
        expect(hoverState?.focusState?.activeNodeIds).toContain(projectionSourceNode.id);
        expect(hoverState?.focusState?.activeNodeIds).toContain(keyInputNode.id);
        expect(hoverState?.focusState?.activeNodeIds).toContain(keyWeightNode.id);
        expect(hoverState?.focusState?.activeNodeIds).toContain(keyBiasNode.id);
        expect(hoverState?.focusState?.activeNodeIds).toContain(keyProjectionOutputNode.id);
        expect(hoverState?.focusState?.activeNodeIds).toContain(transposeNode.id);
    });

    it('shows a 2D-only tooltip for the value bias vector with a head-bias payload', () => {
        const {
            index,
            projectionSourceNode,
            valueInputNode,
            valueWeightNode,
            valueBiasNode,
            valueProjectionOutputNode,
            valuePostNode
        } = buildSceneFixtures();

        const hoverState = resolveMhsaDetailHoverState(index, {
            node: valueBiasNode,
            rowHit: {
                rowIndex: 0,
                rowItem: valueBiasNode?.rowItems?.[0]
            }
        });

        expect(hoverState?.label).toBe('Value Bias Vector');
        expect(hoverState?.info?.activationData?.stage).toBe('qkv.v.bias');
        expect(hoverState?.info?.activationData?.parameterType).toBe('bias');
        expect(hoverState?.info?.activationData?.values).toHaveLength(1);
        expect(hoverState?.focusState?.activeNodeIds).toContain(projectionSourceNode.id);
        expect(hoverState?.focusState?.activeNodeIds).toContain(valueInputNode.id);
        expect(hoverState?.focusState?.activeNodeIds).toContain(valueWeightNode.id);
        expect(hoverState?.focusState?.activeNodeIds).toContain(valueBiasNode.id);
        expect(hoverState?.focusState?.activeNodeIds).toContain(valueProjectionOutputNode.id);
        expect(hoverState?.focusState?.activeNodeIds).toContain(valuePostNode.id);
    });

    it('uses a tighter local gap inside the QK^T grouping row', () => {
        const {
            qktEquationNode,
            divisorCloseNode,
            layout,
            transposeNode,
            attentionScaleNode,
            softmaxOpenNode,
            softmaxCloseNode
        } = buildSceneFixtures();

        const transposeEntry = layout?.registry?.getNodeEntry(transposeNode?.id || '');
        const divisorCloseEntry = layout?.registry?.getNodeEntry(divisorCloseNode?.id || '');
        const scaleEntry = layout?.registry?.getNodeEntry(attentionScaleNode?.id || '');
        const divisorCloseGap = resolveHorizontalContentGap(transposeEntry, divisorCloseEntry);
        const transposeScaleGap = resolveHorizontalContentGap(transposeEntry, scaleEntry);

        expect(qktEquationNode).toBeTruthy();
        expect(qktEquationNode?.metadata?.gapOverride).toBe(12);
        expect(transposeEntry).toBeTruthy();
        expect(divisorCloseEntry).toBeTruthy();
        expect(scaleEntry).toBeTruthy();
        expect(divisorCloseGap).toBeLessThanOrEqual(26);
        expect(transposeScaleGap).toBeGreaterThanOrEqual(0);
        expect(softmaxOpenNode?.metadata?.fontScale).toBeGreaterThan(1);
        expect(softmaxCloseNode?.metadata?.fontScale).toBeGreaterThan(1);
    });

    it('keeps dense 12-token softmax spacing tighter without crowding the divisor cluster', () => {
        const { scene, layout } = buildSceneFixtures(12);
        const nodes = flattenSceneNodes(scene);
        const findNode = (role) => nodes.find((node) => node.role === role) || null;
        const divisorClusterNode = findNode('attention-divisor-cluster');
        const divisorCloseNode = divisorClusterNode?.children?.find((node) => (
            node.role === 'attention-close' && node.semantic?.clusterKey === 'divisor'
        )) || null;

        const transposeEntry = layout?.registry?.getNodeEntry(findNode('attention-key-transpose')?.id || '');
        const divisorCloseEntry = layout?.registry?.getNodeEntry(divisorCloseNode?.id || '');
        const softmaxCoreFlowEntry = layout?.registry?.getNodeEntry(findNode('attention-softmax-core-flow')?.id || '');
        const maskedInputEntry = layout?.registry?.getNodeEntry(findNode('attention-masked-input')?.id || '');
        const plusEntry = layout?.registry?.getNodeEntry(findNode('attention-softmax-plus')?.id || '');
        const maskEntry = layout?.registry?.getNodeEntry(findNode('attention-mask')?.id || '');
        const softmaxCloseEntry = layout?.registry?.getNodeEntry(findNode('attention-softmax-close')?.id || '');
        const softmaxEqualsEntry = layout?.registry?.getNodeEntry(findNode('attention-softmax-equals')?.id || '');
        const postEntry = layout?.registry?.getNodeEntry(findNode('attention-post')?.id || '');

        const divisorCloseGap = resolveHorizontalContentGap(transposeEntry, divisorCloseEntry);
        const maskedPlusGap = resolveHorizontalContentGap(maskedInputEntry, plusEntry);
        const plusMaskGap = resolveHorizontalContentGap(plusEntry, maskEntry);
        const maskCloseGap = resolveHorizontalContentGap(maskEntry, softmaxCloseEntry);
        const closeEqualsGap = resolveHorizontalContentGap(softmaxCloseEntry, softmaxEqualsEntry);
        const equalsPostGap = resolveHorizontalContentGap(softmaxEqualsEntry, postEntry);

        expect(softmaxCoreFlowEntry?.layoutData?.gap).toBeLessThanOrEqual(26);
        expect(divisorCloseGap).toBeLessThanOrEqual(18);
        expect(maskedPlusGap).toBeLessThanOrEqual(30);
        expect(plusMaskGap).toBeLessThanOrEqual(30);
        expect(maskCloseGap).toBeLessThanOrEqual(30);
        expect(closeEqualsGap).toBeLessThanOrEqual(20);
        expect(equalsPostGap).toBeLessThanOrEqual(30);
    });

    it('keeps the computed A_pre row aligned with the QK^T row and moves the softmax core below it', () => {
        const {
            layout,
            queryProjectionOutputNode,
            keyProjectionOutputNode,
            preScoreNode,
            attentionEqualsNode,
            attentionDivideNode,
            attentionScaleNode,
            attentionStageNode,
            maskedInputNode,
            transposeNode,
            softmaxLabelNode,
            softmaxOpenNode
        } = buildSceneFixtures(12);

        const queryProjectionOutputEntry = layout?.registry?.getNodeEntry(queryProjectionOutputNode?.id || '');
        const keyProjectionOutputEntry = layout?.registry?.getNodeEntry(keyProjectionOutputNode?.id || '');
        const preScoreEntry = layout?.registry?.getNodeEntry(preScoreNode?.id || '');
        const attentionEqualsEntry = layout?.registry?.getNodeEntry(attentionEqualsNode?.id || '');
        const attentionDivideEntry = layout?.registry?.getNodeEntry(attentionDivideNode?.id || '');
        const attentionScaleEntry = layout?.registry?.getNodeEntry(attentionScaleNode?.id || '');
        const maskedInputEntry = layout?.registry?.getNodeEntry(maskedInputNode?.id || '');
        const transposeEntry = layout?.registry?.getNodeEntry(transposeNode?.id || '');
        const softmaxLabelEntry = layout?.registry?.getNodeEntry(softmaxLabelNode?.id || '');
        const softmaxOpenEntry = layout?.registry?.getNodeEntry(softmaxOpenNode?.id || '');

        expect(queryProjectionOutputEntry).toBeTruthy();
        expect(keyProjectionOutputEntry).toBeTruthy();
        expect(preScoreEntry).toBeTruthy();
        expect(attentionEqualsEntry).toBeTruthy();
        expect(attentionDivideEntry).toBeTruthy();
        expect(attentionScaleEntry).toBeTruthy();
        expect(attentionStageNode?.metadata?.gapOverride).toBe(-14);
        expect(maskedInputEntry).toBeTruthy();
        expect(transposeEntry).toBeTruthy();
        expect(softmaxLabelEntry).toBeTruthy();
        expect(softmaxOpenEntry).toBeTruthy();
        const queryOutputCenterY = (queryProjectionOutputEntry?.contentBounds?.y || 0)
            + ((queryProjectionOutputEntry?.contentBounds?.height || 0) * 0.5);
        const keyOutputCenterY = (keyProjectionOutputEntry?.contentBounds?.y || 0)
            + ((keyProjectionOutputEntry?.contentBounds?.height || 0) * 0.5);
        const preScoreCenterY = (preScoreEntry?.contentBounds?.y || 0)
            + ((preScoreEntry?.contentBounds?.height || 0) * 0.5);
        const attentionEqualsCenterY = (attentionEqualsEntry?.contentBounds?.y || 0)
            + ((attentionEqualsEntry?.contentBounds?.height || 0) * 0.5);
        const transposeCenterY = (transposeEntry?.contentBounds?.y || 0)
            + ((transposeEntry?.contentBounds?.height || 0) * 0.5);
        const qkMidpointY = (queryOutputCenterY + keyOutputCenterY) * 0.5;

        expect(Math.abs(preScoreCenterY - transposeCenterY)).toBeLessThanOrEqual(8);
        expect(Math.abs(attentionEqualsCenterY - transposeCenterY)).toBeLessThanOrEqual(8);
        expect(preScoreCenterY).toBeGreaterThan(queryOutputCenterY);
        expect(preScoreCenterY).toBeLessThan(keyOutputCenterY);
        expect(Math.abs(preScoreCenterY - qkMidpointY)).toBeLessThanOrEqual(28);
        expect((preScoreEntry?.contentBounds?.x || 0)).toBeGreaterThan(
            (attentionEqualsEntry?.contentBounds?.x || 0) + (attentionEqualsEntry?.contentBounds?.width || 0)
        );
        expect((preScoreEntry?.contentBounds?.x || 0) - (
            (attentionEqualsEntry?.contentBounds?.x || 0) + (attentionEqualsEntry?.contentBounds?.width || 0)
        )).toBeLessThanOrEqual(17);
        expect((attentionScaleEntry?.contentBounds?.x || 0) - (
            (attentionDivideEntry?.contentBounds?.x || 0) + (attentionDivideEntry?.contentBounds?.width || 0)
        )).toBeLessThanOrEqual(-8);
        expect((attentionEqualsEntry?.contentBounds?.x || 0) - (
            (attentionScaleEntry?.contentBounds?.x || 0) + (attentionScaleEntry?.contentBounds?.width || 0)
        )).toBeLessThanOrEqual(8);
        expect(Math.abs(
            (maskedInputEntry?.contentBounds?.x || 0)
            - (preScoreEntry?.contentBounds?.x || 0)
        )).toBeLessThanOrEqual(4);
        expect(maskedInputEntry?.contentBounds?.y).toBeGreaterThan(
            (preScoreEntry?.contentBounds?.y || 0) + (preScoreEntry?.contentBounds?.height || 0)
        );
        expect(maskedInputEntry?.contentBounds?.y).toBeGreaterThan(
            resolveEntryCaptionBottom(preScoreEntry)
        );
        expect(
            (maskedInputEntry?.contentBounds?.y || 0) - resolveEntryCaptionBottom(preScoreEntry)
        ).toBeGreaterThanOrEqual(48);
        expect(softmaxLabelEntry?.contentBounds?.x).toBeLessThan(maskedInputEntry?.contentBounds?.x || 0);
        expect((softmaxLabelEntry?.contentBounds?.x || 0)).toBeLessThan(preScoreEntry?.contentBounds?.x || 0);
        expect(
            ((softmaxOpenEntry?.contentBounds?.x || 0) + (softmaxOpenEntry?.contentBounds?.width || 0))
        ).toBeLessThanOrEqual((maskedInputEntry?.contentBounds?.x || 0) + 2);
    });

    it('lifts the QK^T row slightly above A_pre in KV-cache decode mode', () => {
        const kvCacheState = {
            kvCacheModeEnabled: true,
            kvCachePrefillActive: false,
            kvCacheDecodeActive: true,
            kvCachePassIndex: 1
        };
        const { scene, layout, qktEquationNode, preScoreNode } = buildSceneFixtures(4, {
            kvCacheState
        });
        const nodes = flattenSceneNodes(scene);
        const attentionLeftHandSideNode = nodes.find((node) => node.role === 'attention-left-hand-side') || null;
        const qktEntry = layout?.registry?.getNodeEntry(qktEquationNode?.id || '');
        const preScoreEntry = layout?.registry?.getNodeEntry(preScoreNode?.id || '');

        expect(attentionLeftHandSideNode?.layout?.anchorAlign).toMatchObject({
            axis: 'y',
            selfNodeId: qktEquationNode?.id,
            targetNodeId: preScoreNode?.id
        });
        expect(qktEntry).toBeTruthy();
        expect(preScoreEntry).toBeTruthy();

        const qktCenterY = (qktEntry?.contentBounds?.y || 0)
            + ((qktEntry?.contentBounds?.height || 0) * 0.5);
        const preScoreCenterY = (preScoreEntry?.contentBounds?.y || 0)
            + ((preScoreEntry?.contentBounds?.height || 0) * 0.5);

        expect(preScoreCenterY - qktCenterY).toBeGreaterThanOrEqual(10);
        expect(preScoreCenterY - qktCenterY).toBeLessThanOrEqual(14);
    });

    it('centers the softmax label against the masked-input / mask / post row', () => {
        const {
            layout,
            softmaxLabelNode,
            maskedInputNode,
            maskNode,
            postNode
        } = buildSceneFixtures();

        const softmaxLabelEntry = layout?.registry?.getNodeEntry(softmaxLabelNode?.id || '');
        const maskedInputEntry = layout?.registry?.getNodeEntry(maskedInputNode?.id || '');
        const maskEntry = layout?.registry?.getNodeEntry(maskNode?.id || '');
        const postEntry = layout?.registry?.getNodeEntry(postNode?.id || '');

        const softmaxCenterY = (softmaxLabelEntry?.contentBounds?.y || 0)
            + ((softmaxLabelEntry?.contentBounds?.height || 0) * 0.5);
        const rowCenterY = (
            ((maskedInputEntry?.contentBounds?.y || 0) + ((maskedInputEntry?.contentBounds?.height || 0) * 0.5))
            + ((maskEntry?.contentBounds?.y || 0) + ((maskEntry?.contentBounds?.height || 0) * 0.5))
            + ((postEntry?.contentBounds?.y || 0) + ((postEntry?.contentBounds?.height || 0) * 0.5))
        ) / 3;

        expect(softmaxLabelEntry).toBeTruthy();
        expect(Math.abs(softmaxCenterY - rowCenterY)).toBeLessThanOrEqual(8);
    });

    it('keeps the softmax parentheses vertically aligned in KV-cache decode mode', () => {
        const {
            layout,
            softmaxOpenNode,
            softmaxCloseNode
        } = buildSceneFixtures(4, {
            kvCacheState: {
                kvCacheModeEnabled: true,
                kvCachePrefillActive: false,
                kvCacheDecodeActive: true,
                kvCachePassIndex: 1
            }
        });

        const softmaxOpenEntry = layout?.registry?.getNodeEntry(softmaxOpenNode?.id || '');
        const softmaxCloseEntry = layout?.registry?.getNodeEntry(softmaxCloseNode?.id || '');
        const softmaxOpenCenterY = (softmaxOpenEntry?.contentBounds?.y || 0)
            + ((softmaxOpenEntry?.contentBounds?.height || 0) * 0.5);
        const softmaxCloseCenterY = (softmaxCloseEntry?.contentBounds?.y || 0)
            + ((softmaxCloseEntry?.contentBounds?.height || 0) * 0.5);

        expect(softmaxOpenEntry).toBeTruthy();
        expect(softmaxCloseEntry).toBeTruthy();
        expect(Math.abs(softmaxOpenCenterY - softmaxCloseCenterY)).toBeLessThanOrEqual(1);
    });

    it('routes the K to K^T connector below the transpose caption instead of through its centerline', () => {
        const {
            layout,
            transposeNode,
            connectorKNode
        } = buildSceneFixtures();

        const connectorEntry = layout?.registry?.getConnectorEntry(connectorKNode?.id || '');
        const transposeEntry = layout?.registry?.getNodeEntry(transposeNode?.id || '');

        expect(connectorEntry).toBeTruthy();
        expect(transposeEntry).toBeTruthy();
        expect(connectorEntry?.pathPoints).toHaveLength(3);
        expect(connectorEntry?.pathPoints?.[1]?.x).toBeCloseTo(connectorEntry?.pathPoints?.[2]?.x || 0, 4);
        expect(connectorEntry?.pathPoints?.[1]?.x).toBeCloseTo(transposeEntry?.anchors?.bottom?.x || 0, 4);
        expect(connectorEntry?.pathPoints?.[1]?.y).toBeCloseTo(connectorEntry?.pathPoints?.[0]?.y || 0, 4);
        expect(connectorEntry?.pathPoints?.[2]?.y).toBeCloseTo(
            resolveEntryCaptionBottom(transposeEntry) + CONNECTOR_CAPTION_EXIT_GAP,
            4
        );
    });

    it('routes the pre-softmax connector from below the A_pre caption into the softmax row', () => {
        const {
            layout,
            preScoreNode,
            maskedInputNode,
            connectorPreNode
        } = buildSceneFixtures();

        const connectorEntry = layout?.registry?.getConnectorEntry(connectorPreNode?.id || '');
        const preScoreEntry = layout?.registry?.getNodeEntry(preScoreNode?.id || '');
        const maskedInputEntry = layout?.registry?.getNodeEntry(maskedInputNode?.id || '');

        expect(connectorEntry).toBeTruthy();
        expect(preScoreEntry).toBeTruthy();
        expect(maskedInputEntry).toBeTruthy();
        expect(connectorEntry?.pathPoints?.[0]?.x).toBeCloseTo(preScoreEntry?.anchors?.bottom?.x || 0, 4);
        expect(connectorEntry?.pathPoints?.[0]?.y).toBeCloseTo(
            resolveEntryCaptionBottom(preScoreEntry) + CONNECTOR_CAPTION_EXIT_GAP + PRE_CONNECTOR_SOURCE_OFFSET_Y,
            4
        );
        expect(connectorEntry?.pathPoints?.[connectorEntry.pathPoints.length - 1]?.y).toBeLessThanOrEqual(
            maskedInputEntry?.contentBounds?.y || 0
        );
    });

    it('keeps extra pre-softmax caption clearance on the small-screen layout', () => {
        const {
            layout,
            preScoreNode,
            maskedInputNode,
            connectorPreNode
        } = buildSceneFixtures(TOKEN_LABELS.length, { isSmallScreen: true });

        const connectorEntry = layout?.registry?.getConnectorEntry(connectorPreNode?.id || '');
        const preScoreEntry = layout?.registry?.getNodeEntry(preScoreNode?.id || '');
        const maskedInputEntry = layout?.registry?.getNodeEntry(maskedInputNode?.id || '');

        expect(connectorEntry).toBeTruthy();
        expect(preScoreEntry).toBeTruthy();
        expect(maskedInputEntry).toBeTruthy();
        expect(connectorEntry?.pathPoints?.[0]?.x).toBeCloseTo(preScoreEntry?.anchors?.bottom?.x || 0, 4);
        expect(connectorEntry?.pathPoints?.[0]?.y).toBeCloseTo(
            resolveEntryCaptionBottom(preScoreEntry) + CONNECTOR_CAPTION_EXIT_GAP + PRE_CONNECTOR_SOURCE_OFFSET_Y,
            4
        );
        expect(connectorEntry?.pathPoints?.[connectorEntry.pathPoints.length - 1]?.y).toBeLessThanOrEqual(
            maskedInputEntry?.contentBounds?.y || 0
        );
    });

    it('routes the V connector directly from the projection output into the weighted value operand', () => {
        const {
            layout,
            connectorVNode,
            valueProjectionOutputNode,
            valuePostNode
        } = buildSceneFixtures();

        const connectorEntry = layout?.registry?.getConnectorEntry(connectorVNode?.id || '');
        const valueProjectionEntry = layout?.registry?.getNodeEntry(valueProjectionOutputNode?.id || '');
        const valuePostEntry = layout?.registry?.getNodeEntry(valuePostNode?.id || '');

        expect(connectorEntry).toBeTruthy();
        expect(valueProjectionEntry).toBeTruthy();
        expect(valuePostEntry).toBeTruthy();
        expect(connectorEntry?.pathPoints).toHaveLength(3);
        expect(connectorEntry?.pathPoints?.[0]?.x).toBeCloseTo(
            (valueProjectionEntry?.anchors?.right?.x || 0) + VALUE_CONNECTOR_SOURCE_GAP,
            4
        );
        expect(connectorEntry?.pathPoints?.[0]?.y).toBeCloseTo(
            (valueProjectionEntry?.anchors?.right?.y || 0) + VALUE_CONNECTOR_SOURCE_OFFSET_Y,
            4
        );
        expect(connectorEntry?.pathPoints?.[1]?.y).toBeCloseTo(
            connectorEntry?.pathPoints?.[0]?.y || 0,
            4
        );
        expect(connectorEntry?.pathPoints?.[1]?.x).toBeCloseTo(
            connectorEntry?.pathPoints?.[2]?.x || 0,
            4
        );
        expect(connectorEntry?.pathPoints?.[2]?.y).toBeLessThan(
            connectorEntry?.pathPoints?.[1]?.y || 0
        );
        expect(Math.abs(
            (connectorEntry?.pathPoints?.[2]?.x || 0) - (valuePostEntry?.anchors?.bottom?.x || 0)
        )).toBeLessThanOrEqual(4);
        expect(connectorEntry?.pathPoints?.[2]?.y).toBeCloseTo(
            resolveEntryCaptionBottom(valuePostEntry) + VALUE_CONNECTOR_TARGET_GAP,
            4
        );
    });

    it('renders value and head-output rows with the vector-strip compact row metadata', () => {
        const {
            previewData,
            valuePostNode,
            headOutputNode
        } = buildSceneFixtures();

        expect(valuePostNode).toBeTruthy();
        expect(headOutputNode).toBeTruthy();
        expect(valuePostNode.presentation).toBe(VIEW2D_MATRIX_PRESENTATIONS.COMPACT_ROWS);
        expect(headOutputNode.presentation).toBe(VIEW2D_MATRIX_PRESENTATIONS.COMPACT_ROWS);
        expect(valuePostNode.metadata?.compactRows?.variant).toBe('vector-strip');
        expect(headOutputNode.metadata?.compactRows?.variant).toBe('vector-strip');
        expect(valuePostNode.metadata?.measure?.rows).toBe(previewData.attentionScoreStage.valueRowCount);
        expect(headOutputNode.metadata?.measure?.rows).toBe(previewData.attentionScoreStage.headOutputRowCount);
        expect(valuePostNode.metadata?.compactRows?.compactWidth).toBe(72);
        expect(headOutputNode.metadata?.compactRows?.compactWidth).toBe(72);
        expect(valuePostNode.metadata?.compactRows?.rowHeight).toBe(7);
        expect(headOutputNode.metadata?.compactRows?.rowHeight).toBe(7);
    });

    it('uses larger rounded shells and insets for the attention score grids', () => {
        const {
            preScoreNode,
            maskNode,
            postNode
        } = buildSceneFixtures();

        [preScoreNode, maskNode, postNode].forEach((node) => {
            expect(node).toBeTruthy();
            expect(node?.metadata?.grid?.paddingX).toBe(4);
            expect(node?.metadata?.grid?.paddingY).toBe(4);
            expect(node?.metadata?.grid?.cellCornerRadiusScale).toBeCloseTo(ATTENTION_GRID_CELL_CORNER_RADIUS_SCALE, 5);
            expect(node?.metadata?.card?.cornerRadius).toBe(12);
        });
    });

    it('maps value-post hover rows to the Value Vector tooltip payload', () => {
        const {
            index,
            projectionSourceNode,
            queryInputNode,
            keyInputNode,
            queryWeightNode,
            keyWeightNode,
            queryBiasNode,
            keyBiasNode,
            queryProjectionOutputNode,
            keyProjectionOutputNode,
            queryNode,
            transposeNode,
            valuePostNode,
            headOutputNode,
            postCopyNode,
            connectorVNode
        } = buildSceneFixtures();

        const hoverState = resolveMhsaDetailHoverState(index, {
            node: valuePostNode,
            rowHit: {
                rowIndex: 1,
                rowItem: valuePostNode.rowItems[1]
            }
        });

        expect(hoverState?.label).toBe('Value Vector');
        expect(hoverState?.info?.activationData?.label).toBe('Value Vector');
        expect(hoverState?.info?.activationData?.stage).toBe('qkv.v');
        expect(hoverState?.info?.activationData?.tokenIndex).toBe(1);
        expect(hoverState?.info?.activationData?.tokenLabel).toBe('Token B');
        expect(hoverState?.focusState?.rowSelections).toContainEqual({
            nodeId: projectionSourceNode.id,
            rowIndex: 1
        });
        expect(hoverState?.focusState?.activeNodeIds).toContain(headOutputNode?.id);
        expect(hoverState?.focusState?.activeNodeIds).not.toContain(postCopyNode?.id);
        expect(hoverState?.focusState?.activeConnectorIds).toContain(connectorVNode?.id);
        expect(hoverState?.focusState?.dimNodeIds).toContain(queryInputNode.id);
        expect(hoverState?.focusState?.dimNodeIds).toContain(keyInputNode.id);
        expect(hoverState?.focusState?.dimNodeIds).toContain(queryWeightNode.id);
        expect(hoverState?.focusState?.dimNodeIds).toContain(keyWeightNode.id);
        expect(hoverState?.focusState?.dimNodeIds).toContain(queryBiasNode.id);
        expect(hoverState?.focusState?.dimNodeIds).toContain(keyBiasNode.id);
        expect(hoverState?.focusState?.dimNodeIds).toContain(queryProjectionOutputNode.id);
        expect(hoverState?.focusState?.dimNodeIds).toContain(keyProjectionOutputNode.id);
        expect(hoverState?.focusState?.dimNodeIds).toContain(queryNode.id);
        expect(hoverState?.focusState?.dimNodeIds).toContain(transposeNode.id);
    });

    it('keeps the full H_i matrix active when hovering a projection-side V row', () => {
        const {
            index,
            projectionSourceNode,
            valueProjectionOutputNode,
            headOutputNode,
            postCopyNode,
            connectorVNode
        } = buildSceneFixtures();

        const hoverState = resolveMhsaDetailHoverState(index, {
            node: valueProjectionOutputNode,
            rowHit: {
                rowIndex: 0,
                rowItem: valueProjectionOutputNode.rowItems[0]
            }
        });

        expect(hoverState?.label).toBe('Value Vector');
        expect(hoverState?.focusState?.rowSelections).toContainEqual({
            nodeId: projectionSourceNode.id,
            rowIndex: 0
        });
        expect(hoverState?.focusState?.rowSelections).toContainEqual({
            nodeId: valueProjectionOutputNode.id,
            rowIndex: 0
        });
        expect(hoverState?.focusState?.activeNodeIds).toContain(headOutputNode?.id);
        expect(hoverState?.focusState?.activeNodeIds).not.toContain(postCopyNode?.id);
        expect(hoverState?.focusState?.activeConnectorIds).toContain(connectorVNode?.id);
    });

    it('locks prefill V rows onto the mirrored V cache row on click', () => {
        const {
            index,
            valueProjectionOutputNode,
            valueCacheNode,
            connectorVCacheNode
        } = buildSceneFixtures(TOKEN_LABELS.length, {
            kvCacheState: {
                kvCacheModeEnabled: true,
                kvCachePrefillActive: true
            }
        });

        const hoverState = resolveMhsaDetailHoverState(index, {
            node: valueProjectionOutputNode,
            rowHit: {
                rowIndex: 1,
                rowItem: valueProjectionOutputNode?.rowItems?.[1]
            }
        });
        const clickState = resolveMhsaDetailHoverState(index, {
            node: valueProjectionOutputNode,
            rowHit: {
                rowIndex: 1,
                rowItem: valueProjectionOutputNode?.rowItems?.[1]
            }
        }, {
            interactionKind: 'click'
        });

        expect(
            hoverState?.focusState?.rowSelections?.some((selection) => (
                selection.nodeId === valueCacheNode?.id && selection.rowIndex === 1
            ))
        ).toBe(false);
        expect(clickState?.focusState?.activeNodeIds).toContain(valueCacheNode?.id);
        expect(clickState?.focusState?.activeConnectorIds).toContain(connectorVCacheNode?.id);
        expect(clickState?.focusState?.rowSelections).toContainEqual({
            nodeId: valueCacheNode?.id,
            rowIndex: 1
        });
    });

    it('can resolve deep-hover tooltip payloads without building focus state', () => {
        const {
            index,
            valuePostNode
        } = buildSceneFixtures();

        const hoverState = resolveMhsaDetailHoverState(index, {
            node: valuePostNode,
            rowHit: {
                rowIndex: 1,
                rowItem: valuePostNode.rowItems[1]
            }
        }, {
            includeFocusState: false
        });

        expect(hoverState?.label).toBe('Value Vector');
        expect(hoverState?.info?.activationData?.stage).toBe('qkv.v');
        expect(hoverState?.focusState).toBeNull();
        expect(hoverState?.signature).toBe('');
    });

    it('reuses cached deep-hover results for identical hits and focus modes', () => {
        const {
            index,
            valuePostNode
        } = buildSceneFixtures();

        const hit = {
            node: valuePostNode,
            rowHit: {
                rowIndex: 1,
                rowItem: valuePostNode.rowItems[1]
            }
        };

        const focusedHover = resolveMhsaDetailHoverState(index, hit);
        const focusedHoverRepeat = resolveMhsaDetailHoverState(index, hit);
        const tooltipHover = resolveMhsaDetailHoverState(index, hit, {
            includeFocusState: false
        });
        const tooltipHoverRepeat = resolveMhsaDetailHoverState(index, hit, {
            includeFocusState: false
        });

        expect(focusedHover).toBe(focusedHoverRepeat);
        expect(tooltipHover).toBe(tooltipHoverRepeat);
        expect(focusedHover).not.toBe(tooltipHover);
    });

    it('keeps direct value-post role hovers on the local V branch instead of the full weighted-output path', () => {
        const {
            index,
            queryInputNode,
            keyInputNode,
            queryWeightNode,
            keyWeightNode,
            queryBiasNode,
            keyBiasNode,
            queryProjectionOutputNode,
            keyProjectionOutputNode,
            valuePostNode,
            headOutputNode,
            postCopyNode,
            queryNode,
            transposeNode,
            connectorVNode,
            connectorQNode,
            connectorKNode,
            connectorPostNode
        } = buildSceneFixtures();

        const hoverState = resolveMhsaDetailHoverState(index, {
            node: valuePostNode
        });

        expect(hoverState?.label).toBe('Value Vector');
        expect(hoverState?.info?.activationData?.label).toBe('Value Vector');
        expect(hoverState?.focusState?.activeNodeIds).toContain(valuePostNode?.id);
        expect(hoverState?.focusState?.activeNodeIds).not.toContain(headOutputNode?.id);
        expect(hoverState?.focusState?.activeNodeIds).not.toContain(postCopyNode?.id);
        expect(hoverState?.focusState?.activeNodeIds).not.toContain(queryNode?.id);
        expect(hoverState?.focusState?.activeNodeIds).not.toContain(transposeNode?.id);
        expect(hoverState?.focusState?.activeConnectorIds).toContain(connectorVNode?.id);
        expect(hoverState?.focusState?.activeConnectorIds).not.toContain(connectorQNode?.id);
        expect(hoverState?.focusState?.activeConnectorIds).not.toContain(connectorKNode?.id);
        expect(hoverState?.focusState?.activeConnectorIds).not.toContain(connectorPostNode?.id);
        expect(hoverState?.focusState?.dimNodeIds).toContain(queryInputNode.id);
        expect(hoverState?.focusState?.dimNodeIds).toContain(keyInputNode.id);
        expect(hoverState?.focusState?.dimNodeIds).toContain(queryWeightNode.id);
        expect(hoverState?.focusState?.dimNodeIds).toContain(keyWeightNode.id);
        expect(hoverState?.focusState?.dimNodeIds).toContain(queryBiasNode.id);
        expect(hoverState?.focusState?.dimNodeIds).toContain(keyBiasNode.id);
        expect(hoverState?.focusState?.dimNodeIds).toContain(queryProjectionOutputNode.id);
        expect(hoverState?.focusState?.dimNodeIds).toContain(keyProjectionOutputNode.id);
        expect(hoverState?.focusState?.dimNodeIds).toContain(queryNode.id);
        expect(hoverState?.focusState?.dimNodeIds).toContain(transposeNode.id);
    });

    it('maps K/V cache row hovers to the local cache and projection-output rows', () => {
        const {
            index,
            keyInputNode,
            valueInputNode,
            transposeNode,
            valuePostNode,
            keyProjectionOutputNode,
            valueProjectionOutputNode,
            keyCacheNode,
            valueCacheNode,
            connectorKCacheNode,
            connectorVCacheNode
        } = buildSceneFixtures(TOKEN_LABELS.length, {
            kvCacheState: {
                kvCacheModeEnabled: true,
                kvCachePrefillActive: true
            }
        });

        const keyHoverState = resolveMhsaDetailHoverState(index, {
            node: keyCacheNode,
            rowHit: {
                rowIndex: 1,
                rowItem: keyCacheNode?.rowItems?.[1]
            }
        });
        const valueHoverState = resolveMhsaDetailHoverState(index, {
            node: valueCacheNode,
            rowHit: {
                rowIndex: 0,
                rowItem: valueCacheNode?.rowItems?.[0]
            }
        });

        expect(keyHoverState?.label).toBe('Cached Key Vector');
        expect(keyHoverState?.info?.activationData?.label).toBe('Cached Key Vector');
        expect(keyHoverState?.info?.cachedKv).toBe(true);
        expect(keyHoverState?.focusState?.activeNodeIds).toContain(keyProjectionOutputNode?.id);
        expect(keyHoverState?.focusState?.activeNodeIds).toContain(keyCacheNode?.id);
        expect(keyHoverState?.focusState?.activeNodeIds).not.toContain(keyInputNode?.id);
        expect(keyHoverState?.focusState?.activeNodeIds).not.toContain(transposeNode?.id);
        expect(keyHoverState?.focusState?.activeConnectorIds).toContain(connectorKCacheNode?.id);
        expect(keyHoverState?.focusState?.rowSelections).toContainEqual({
            nodeId: keyProjectionOutputNode?.id,
            rowIndex: 1
        });
        expect(keyHoverState?.focusState?.rowSelections).toContainEqual({
            nodeId: keyCacheNode?.id,
            rowIndex: 1
        });

        expect(valueHoverState?.label).toBe('Cached Value Vector');
        expect(valueHoverState?.info?.activationData?.label).toBe('Cached Value Vector');
        expect(valueHoverState?.info?.cachedKv).toBe(true);
        expect(valueHoverState?.focusState?.activeNodeIds).toContain(valueProjectionOutputNode?.id);
        expect(valueHoverState?.focusState?.activeNodeIds).toContain(valueCacheNode?.id);
        expect(valueHoverState?.focusState?.activeNodeIds).not.toContain(valueInputNode?.id);
        expect(valueHoverState?.focusState?.activeNodeIds).not.toContain(valuePostNode?.id);
        expect(valueHoverState?.focusState?.activeConnectorIds).toContain(connectorVCacheNode?.id);
        expect(valueHoverState?.focusState?.rowSelections).toContainEqual({
            nodeId: valueProjectionOutputNode?.id,
            rowIndex: 0
        });
        expect(valueHoverState?.focusState?.rowSelections).toContainEqual({
            nodeId: valueCacheNode?.id,
            rowIndex: 0
        });
    });

    it('keeps the main MHSA detail connectors on the bright preserved-color stroke path', () => {
        const {
            connectorQNode,
            connectorKNode,
            connectorPreNode,
            connectorPostNode,
            connectorVNode,
            connectorKCacheSourceNode,
            connectorVCacheSourceNode,
            connectorKCacheCopyNode,
            connectorVCacheCopyNode,
            connectorKCacheNextNode,
            connectorVCacheNextNode
        } = buildSceneFixtures(4, {
            kvCacheState: {
                kvCacheModeEnabled: true,
                kvCachePrefillActive: false,
                kvCacheDecodeActive: true,
                kvCachePassIndex: 1
            }
        });

        expect(connectorQNode?.metadata?.preserveColor).toBe(true);
        expect(connectorKNode?.metadata?.preserveColor).toBe(true);
        expect(connectorPreNode?.metadata?.preserveColor).toBe(true);
        expect(connectorPostNode?.metadata?.preserveColor).toBe(true);
        expect(connectorVNode?.metadata?.preserveColor).toBe(true);
        expect(connectorKCacheSourceNode?.metadata?.preserveColor).toBe(true);
        expect(connectorVCacheSourceNode?.metadata?.preserveColor).toBe(true);
        expect(connectorKCacheCopyNode?.metadata?.preserveColor).toBe(true);
        expect(connectorVCacheCopyNode?.metadata?.preserveColor).toBe(true);
        expect(connectorKCacheNextNode?.metadata?.preserveColor).toBe(true);
        expect(connectorVCacheNextNode?.metadata?.preserveColor).toBe(true);
        expect(connectorKNode?.visual?.stroke).toBe(connectorQNode?.visual?.stroke);
        expect(connectorPreNode?.visual?.stroke).toBe(connectorQNode?.visual?.stroke);
        expect(connectorPostNode?.visual?.stroke).toBe(connectorQNode?.visual?.stroke);
        expect(connectorVNode?.visual?.stroke).toBe(connectorQNode?.visual?.stroke);
        expect(connectorKCacheSourceNode?.visual?.stroke).toBe(connectorQNode?.visual?.stroke);
        expect(connectorVCacheSourceNode?.visual?.stroke).toBe(connectorQNode?.visual?.stroke);
        expect(connectorKCacheCopyNode?.visual?.stroke).toBe(connectorQNode?.visual?.stroke);
        expect(connectorVCacheCopyNode?.visual?.stroke).toBe(connectorQNode?.visual?.stroke);
        expect(connectorKCacheNextNode?.visual?.stroke).toBe(connectorQNode?.visual?.stroke);
        expect(connectorVCacheNextNode?.visual?.stroke).toBe(connectorQNode?.visual?.stroke);
    });

    it('maps decode cached key/value row hovers to cached matrices without re-highlighting the live copy path', () => {
        const {
            index,
            keyProjectionOutputNode,
            valueProjectionOutputNode,
            keyProjectionOutputCopyNode,
            valueProjectionOutputCopyNode,
            keyCacheNode,
            valueCacheNode,
            keyCacheSourceNode,
            valueCacheSourceNode,
            keyCacheConcatResultNode,
            valueCacheConcatResultNode,
            valueCacheNextNode,
            transposeNode,
            preScoreNode,
            maskedInputNode,
            maskNode,
            postNode,
            postCopyNode,
            valuePostNode,
            connectorKNode,
            connectorKCacheSourceNode,
            connectorKCacheNode,
            connectorKCacheCopyNode,
            connectorPreNode,
            connectorPostNode,
            connectorVNode,
            connectorVCacheSourceNode,
            connectorVCacheNode,
            connectorVCacheCopyNode,
            connectorVCacheNextNode
        } = buildSceneFixtures(TOKEN_LABELS.length, {
            kvCacheState: {
                kvCacheModeEnabled: true,
                kvCachePrefillActive: false,
                kvCacheDecodeActive: true,
                kvCachePassIndex: 1
            }
        });

        const keyHoverStates = [
            resolveMhsaDetailHoverState(index, {
                node: keyCacheNode,
                rowHit: {
                    rowIndex: 1,
                    rowItem: keyCacheNode?.rowItems?.[1]
                }
            }),
            resolveMhsaDetailHoverState(index, {
                node: keyCacheSourceNode,
                rowHit: {
                    rowIndex: 1,
                    rowItem: keyCacheSourceNode?.rowItems?.[1]
                }
            })
        ];
        const valueHoverState = resolveMhsaDetailHoverState(index, {
            node: valueCacheSourceNode,
            rowHit: {
                rowIndex: 0,
                rowItem: valueCacheSourceNode?.rowItems?.[0]
            }
        });

        keyHoverStates.forEach((keyHoverState) => {
            expect(keyHoverState?.label).toBe('Cached Key Vector');
            expect(keyHoverState?.info?.activationData?.label).toBe('Cached Key Vector');
            expect(keyHoverState?.focusState?.activeNodeIds).not.toContain(keyProjectionOutputNode?.id);
            expect(keyHoverState?.focusState?.activeNodeIds).not.toContain(keyProjectionOutputCopyNode?.id);
            expect(keyHoverState?.focusState?.activeNodeIds).not.toContain(transposeNode?.id);
            expect(keyHoverState?.focusState?.activeNodeIds).toContain(keyCacheNode?.id);
            expect(keyHoverState?.focusState?.activeNodeIds).toContain(keyCacheSourceNode?.id);
            expect(connectorKCacheNode).toBeNull();
            expect(keyHoverState?.focusState?.activeConnectorIds).toContain(connectorKCacheSourceNode?.id);
            expect(keyHoverState?.focusState?.activeConnectorIds).not.toContain(connectorKCacheCopyNode?.id);
            expect(keyHoverState?.focusState?.activeConnectorIds).not.toContain(connectorKNode?.id);
            expect(keyHoverState?.focusState?.activeConnectorIds).toContain(connectorPreNode?.id);
            expect(keyHoverState?.focusState?.activeConnectorIds).toContain(connectorPostNode?.id);
            expect(keyHoverState?.focusState?.rowSelections).toContainEqual({
                nodeId: keyCacheNode?.id,
                rowIndex: 1
            });
            expect(keyHoverState?.focusState?.rowSelections).toContainEqual({
                nodeId: keyCacheSourceNode?.id,
                rowIndex: 1
            });
            expect(keyHoverState?.focusState?.rowSelections).toContainEqual({
                nodeId: keyCacheConcatResultNode?.id,
                rowIndex: 1
            });
            expect(keyHoverState?.focusState?.columnSelections).toContainEqual({
                nodeId: preScoreNode?.id,
                colIndex: 1
            });
            expect(keyHoverState?.focusState?.columnSelections).toContainEqual({
                nodeId: maskedInputNode?.id,
                colIndex: 1
            });
            expect(keyHoverState?.focusState?.columnSelections).toContainEqual({
                nodeId: maskNode?.id,
                colIndex: 1
            });
            expect(keyHoverState?.focusState?.columnSelections).toContainEqual({
                nodeId: postNode?.id,
                colIndex: 1
            });
            expect(keyHoverState?.focusState?.columnSelections).toContainEqual({
                nodeId: postCopyNode?.id,
                colIndex: 1
            });
            expect(
                keyHoverState?.focusState?.columnSelections?.some((selection) => selection.nodeId === transposeNode?.id)
            ).toBe(false);
        });

        expect(valueHoverState?.label).toBe('Cached Value Vector');
        expect(valueHoverState?.info?.activationData?.label).toBe('Cached Value Vector');
        expect(valueHoverState?.focusState?.activeNodeIds).not.toContain(valueProjectionOutputNode?.id);
        expect(valueHoverState?.focusState?.activeNodeIds).not.toContain(valueProjectionOutputCopyNode?.id);
        expect(valueHoverState?.focusState?.activeNodeIds).toContain(valueCacheNode?.id);
        expect(valueHoverState?.focusState?.activeNodeIds).toContain(valueCacheSourceNode?.id);
        expect(connectorVCacheNode).toBeNull();
        expect(valueHoverState?.focusState?.activeConnectorIds).toContain(connectorVNode?.id);
        expect(valueHoverState?.focusState?.activeConnectorIds).toContain(connectorVCacheSourceNode?.id);
        expect(valueHoverState?.focusState?.activeConnectorIds).toContain(connectorVCacheNextNode?.id);
        expect(valueHoverState?.focusState?.activeConnectorIds).not.toContain(connectorVCacheCopyNode?.id);
        expect(valueHoverState?.focusState?.rowSelections).toContainEqual({
            nodeId: valueCacheNode?.id,
            rowIndex: 0
        });
        expect(valueHoverState?.focusState?.rowSelections).toContainEqual({
            nodeId: valueCacheSourceNode?.id,
            rowIndex: 0
        });
        expect(valueHoverState?.focusState?.rowSelections).toContainEqual({
            nodeId: valueCacheConcatResultNode?.id,
            rowIndex: 0
        });
        expect(valueHoverState?.focusState?.rowSelections).toContainEqual({
            nodeId: valueCacheNextNode?.id,
            rowIndex: 0
        });
        expect(valueHoverState?.focusState?.rowSelections).toContainEqual({
            nodeId: valuePostNode?.id,
            rowIndex: 0
        });
    });

    it('maps clicked decode K_current rows to the live concat/cache-next row and downstream score columns', () => {
        const {
            index,
            projectionSourceNode,
            keyProjectionOutputNode,
            keyProjectionOutputCopyNode,
            keyCacheNode,
            keyCacheSourceNode,
            keyCacheConcatResultNode,
            keyCacheNextNode,
            transposeNode,
            preScoreNode,
            maskedInputNode,
            maskNode,
            postNode,
            postCopyNode,
            connectorKNode,
            connectorKCacheSourceNode,
            connectorKCacheCopyNode,
            connectorKCacheNextNode,
            connectorPreNode,
            connectorPostNode
        } = buildSceneFixtures(4, {
            kvCacheState: {
                kvCacheModeEnabled: true,
                kvCachePrefillActive: false,
                kvCacheDecodeActive: true,
                kvCachePassIndex: 1
            }
        });

        const hoverState = resolveMhsaDetailHoverState(index, {
            node: keyProjectionOutputNode,
            rowHit: {
                rowIndex: 0,
                rowItem: keyProjectionOutputNode?.rowItems?.[0]
            }
        }, {
            interactionKind: 'click'
        });

        expect(hoverState?.label).toBe('Key Vector');
        expect(hoverState?.info?.cachedKv).not.toBe(true);
        expect(hoverState?.focusState?.activeNodeIds).not.toContain(keyCacheNode?.id);
        expect(hoverState?.focusState?.activeNodeIds).not.toContain(keyCacheSourceNode?.id);
        expect(hoverState?.focusState?.activeConnectorIds).toContain(connectorKNode?.id);
        expect(hoverState?.focusState?.activeConnectorIds).toContain(connectorKCacheCopyNode?.id);
        expect(hoverState?.focusState?.activeConnectorIds).toContain(connectorKCacheNextNode?.id);
        expect(hoverState?.focusState?.activeConnectorIds).toContain(connectorPreNode?.id);
        expect(hoverState?.focusState?.activeConnectorIds).toContain(connectorPostNode?.id);
        expect(hoverState?.focusState?.activeConnectorIds).not.toContain(connectorKCacheSourceNode?.id);
        expect(hoverState?.focusState?.rowSelections).toContainEqual({
            nodeId: projectionSourceNode?.id,
            rowIndex: 0
        });
        expect(hoverState?.focusState?.rowSelections).toContainEqual({
            nodeId: keyProjectionOutputNode?.id,
            rowIndex: 0
        });
        expect(hoverState?.focusState?.rowSelections).toContainEqual({
            nodeId: keyProjectionOutputCopyNode?.id,
            rowIndex: 0
        });
        expect(hoverState?.focusState?.rowSelections).toContainEqual({
            nodeId: keyCacheConcatResultNode?.id,
            rowIndex: 3
        });
        expect(hoverState?.focusState?.rowSelections).toContainEqual({
            nodeId: keyCacheNextNode?.id,
            rowIndex: 3
        });
        expect(hoverState?.focusState?.columnSelections).toContainEqual({
            nodeId: transposeNode?.id,
            colIndex: 3
        });
        expect(hoverState?.focusState?.columnSelections).toContainEqual({
            nodeId: preScoreNode?.id,
            colIndex: 3
        });
        expect(hoverState?.focusState?.columnSelections).toContainEqual({
            nodeId: maskedInputNode?.id,
            colIndex: 3
        });
        expect(hoverState?.focusState?.columnSelections).toContainEqual({
            nodeId: maskNode?.id,
            colIndex: 3
        });
        expect(hoverState?.focusState?.columnSelections).toContainEqual({
            nodeId: postNode?.id,
            colIndex: 3
        });
        expect(hoverState?.focusState?.columnSelections).toContainEqual({
            nodeId: postCopyNode?.id,
            colIndex: 3
        });
    });

    it('maps decode V_current and its copied live row to the live combined V path while mirroring the cache row', () => {
        const {
            index,
            projectionSourceNode,
            valueProjectionOutputNode,
            valueProjectionOutputCopyNode,
            valueCacheConcatResultNode,
            valueCacheNextNode,
            valueCacheNode,
            valueCacheSourceNode,
            valuePostNode,
            connectorVNode,
            connectorVCacheSourceNode,
            connectorVCacheCopyNode,
            connectorVCacheNextNode
        } = buildSceneFixtures(4, {
            kvCacheState: {
                kvCacheModeEnabled: true,
                kvCachePrefillActive: false,
                kvCacheDecodeActive: true,
                kvCachePassIndex: 1
            }
        });

        const hoverStates = [
            resolveMhsaDetailHoverState(index, {
                node: valueProjectionOutputNode,
                rowHit: {
                    rowIndex: 0,
                    rowItem: valueProjectionOutputNode?.rowItems?.[0]
                }
            }),
            resolveMhsaDetailHoverState(index, {
                node: valueProjectionOutputCopyNode,
                rowHit: {
                    rowIndex: 0,
                    rowItem: valueProjectionOutputCopyNode?.rowItems?.[0]
                }
            }),
            resolveMhsaDetailHoverState(index, {
                node: valueProjectionOutputNode,
                rowHit: {
                    rowIndex: 0,
                    rowItem: valueProjectionOutputNode?.rowItems?.[0]
                }
            }, {
                interactionKind: 'click'
            }),
            resolveMhsaDetailHoverState(index, {
                node: valueProjectionOutputCopyNode,
                rowHit: {
                    rowIndex: 0,
                    rowItem: valueProjectionOutputCopyNode?.rowItems?.[0]
                }
            }, {
                interactionKind: 'click'
            })
        ];

        hoverStates.forEach((hoverState) => {
            expect(hoverState?.label).toBe('Value Vector');
            expect(hoverState?.focusState?.activeNodeIds).toContain(valueCacheNode?.id);
            expect(hoverState?.focusState?.activeNodeIds).toContain(valueCacheSourceNode?.id);
            expect(hoverState?.focusState?.activeConnectorIds).toContain(connectorVNode?.id);
            expect(hoverState?.focusState?.activeConnectorIds).toContain(connectorVCacheCopyNode?.id);
            expect(hoverState?.focusState?.activeConnectorIds).toContain(connectorVCacheNextNode?.id);
            expect(hoverState?.focusState?.activeConnectorIds).not.toContain(connectorVCacheSourceNode?.id);
            expect(hoverState?.focusState?.rowSelections).toContainEqual({
                nodeId: projectionSourceNode?.id,
                rowIndex: 0
            });
            expect(hoverState?.focusState?.rowSelections).toContainEqual({
                nodeId: valueProjectionOutputNode?.id,
                rowIndex: 0
            });
            expect(hoverState?.focusState?.rowSelections).toContainEqual({
                nodeId: valueProjectionOutputCopyNode?.id,
                rowIndex: 0
            });
            expect(hoverState?.focusState?.rowSelections).toContainEqual({
                nodeId: valueCacheNode?.id,
                rowIndex: 0
            });
            expect(hoverState?.focusState?.rowSelections).toContainEqual({
                nodeId: valueCacheSourceNode?.id,
                rowIndex: 0
            });
            expect(hoverState?.focusState?.rowSelections).toContainEqual({
                nodeId: valueCacheConcatResultNode?.id,
                rowIndex: 3
            });
            expect(hoverState?.focusState?.rowSelections).toContainEqual({
                nodeId: valueCacheNextNode?.id,
                rowIndex: 3
            });
            expect(hoverState?.focusState?.rowSelections).toContainEqual({
                nodeId: valuePostNode?.id,
                rowIndex: 3
            });
            expect(
                hoverState?.focusState?.rowSelections?.some((selection) => (
                    selection.nodeId === valuePostNode?.id && selection.rowIndex === 0
                ))
            ).toBe(false);
        });
    });

    it('keeps the live decode V rows tied to the shared x_ln source row across V variants', () => {
        const {
            index,
            projectionSourceNode,
            valueInputNode,
            valueProjectionOutputNode,
            valueProjectionOutputCopyNode,
            valueCacheConcatResultNode,
            valueCacheNextNode
        } = buildSceneFixtures(4, {
            kvCacheState: {
                kvCacheModeEnabled: true,
                kvCachePrefillActive: false,
                kvCacheDecodeActive: true,
                kvCachePassIndex: 1
            }
        });

        const hoverStates = [
            resolveMhsaDetailHoverState(index, {
                node: valueProjectionOutputNode,
                rowHit: {
                    rowIndex: 0,
                    rowItem: valueProjectionOutputNode?.rowItems?.[0]
                }
            }),
            resolveMhsaDetailHoverState(index, {
                node: valueProjectionOutputCopyNode,
                rowHit: {
                    rowIndex: 0,
                    rowItem: valueProjectionOutputCopyNode?.rowItems?.[0]
                }
            }),
            resolveMhsaDetailHoverState(index, {
                node: valueCacheConcatResultNode,
                rowHit: {
                    rowIndex: 3,
                    rowItem: valueCacheConcatResultNode?.rowItems?.[3]
                }
            }),
            resolveMhsaDetailHoverState(index, {
                node: valueCacheNextNode,
                rowHit: {
                    rowIndex: 3,
                    rowItem: valueCacheNextNode?.rowItems?.[3]
                }
            })
        ];

        hoverStates.forEach((hoverState) => {
            expect(hoverState?.focusState?.rowSelections).toContainEqual({
                nodeId: projectionSourceNode?.id,
                rowIndex: 0
            });
            expect(hoverState?.focusState?.rowSelections).toContainEqual({
                nodeId: valueInputNode?.id,
                rowIndex: 0
            });
        });
    });

    it('maps the decode value x_ln copy to the live V row on hover and click', () => {
        const {
            index,
            projectionSourceNode,
            valueInputNode,
            valueProjectionOutputNode,
            valueProjectionOutputCopyNode,
            valueCacheNode,
            valueCacheSourceNode,
            valueCacheConcatResultNode,
            valueCacheNextNode,
            valuePostNode
        } = buildSceneFixtures(4, {
            kvCacheState: {
                kvCacheModeEnabled: true,
                kvCachePrefillActive: false,
                kvCacheDecodeActive: true,
                kvCachePassIndex: 1
            }
        });

        const focusStates = [
            resolveMhsaDetailHoverState(index, {
                node: valueInputNode,
                rowHit: {
                    rowIndex: 0,
                    rowItem: valueInputNode?.rowItems?.[0]
                }
            }),
            resolveMhsaDetailHoverState(index, {
                node: valueInputNode,
                rowHit: {
                    rowIndex: 0,
                    rowItem: valueInputNode?.rowItems?.[0]
                }
            }, {
                interactionKind: 'click'
            })
        ];

        focusStates.forEach((hoverState) => {
            expect(hoverState?.focusState?.rowSelections).toContainEqual({
                nodeId: projectionSourceNode?.id,
                rowIndex: 0
            });
            expect(hoverState?.focusState?.rowSelections).toContainEqual({
                nodeId: valueInputNode?.id,
                rowIndex: 0
            });
            expect(hoverState?.focusState?.rowSelections).toContainEqual({
                nodeId: valueProjectionOutputNode?.id,
                rowIndex: 0
            });
            expect(hoverState?.focusState?.rowSelections).toContainEqual({
                nodeId: valueProjectionOutputCopyNode?.id,
                rowIndex: 0
            });
            expect(hoverState?.focusState?.rowSelections).toContainEqual({
                nodeId: valueCacheConcatResultNode?.id,
                rowIndex: 3
            });
            expect(hoverState?.focusState?.rowSelections).toContainEqual({
                nodeId: valueCacheNextNode?.id,
                rowIndex: 3
            });
            expect(hoverState?.focusState?.rowSelections).toContainEqual({
                nodeId: valuePostNode?.id,
                rowIndex: 3
            });
            expect(
                hoverState?.focusState?.rowSelections?.some((selection) => (
                    selection.nodeId === valueCacheConcatResultNode?.id && selection.rowIndex === 0
                ))
            ).toBe(false);
            expect(
                hoverState?.focusState?.rowSelections?.some((selection) => (
                    selection.nodeId === valuePostNode?.id && selection.rowIndex === 0
                ))
            ).toBe(false);
            expect(hoverState?.focusState?.activeNodeIds).not.toContain(valueCacheNode?.id);
            expect(hoverState?.focusState?.activeNodeIds).not.toContain(valueCacheSourceNode?.id);
        });
    });

    it('maps decode concat-result rows to cached rows or the live decode row based on which row is hovered', () => {
        const {
            index,
            projectionSourceNode,
            keyInputNode,
            keyProjectionOutputNode,
            keyProjectionOutputCopyNode,
            keyCacheNode,
            keyCacheSourceNode,
            keyCacheConcatResultNode,
            transposeNode,
            preScoreNode,
            maskedInputNode,
            maskNode,
            postNode,
            postCopyNode,
            connectorKNode,
            connectorKCacheSourceNode,
            connectorKCacheCopyNode,
            connectorPreNode,
            connectorPostNode
        } = buildSceneFixtures(4, {
            kvCacheState: {
                kvCacheModeEnabled: true,
                kvCachePrefillActive: false,
                kvCacheDecodeActive: true,
                kvCachePassIndex: 1
            }
        });

        const cachedRowHoverState = resolveMhsaDetailHoverState(index, {
            node: keyCacheConcatResultNode,
            rowHit: {
                rowIndex: 1,
                rowItem: keyCacheConcatResultNode?.rowItems?.[1]
            }
        });
        const liveRowHoverState = resolveMhsaDetailHoverState(index, {
            node: keyCacheConcatResultNode,
            rowHit: {
                rowIndex: 3,
                rowItem: keyCacheConcatResultNode?.rowItems?.[3]
            }
        });

        expect(cachedRowHoverState?.label).toBe('Cached Key Vector');
        expect(cachedRowHoverState?.info?.cachedKv).toBe(true);
        expect(cachedRowHoverState?.focusState?.activeNodeIds).not.toContain(keyProjectionOutputNode?.id);
        expect(cachedRowHoverState?.focusState?.activeNodeIds).not.toContain(keyProjectionOutputCopyNode?.id);
        expect(cachedRowHoverState?.focusState?.activeNodeIds).toContain(transposeNode?.id);
        expect(cachedRowHoverState?.focusState?.activeConnectorIds).toContain(connectorKNode?.id);
        expect(cachedRowHoverState?.focusState?.activeConnectorIds).toContain(connectorKCacheSourceNode?.id);
        expect(cachedRowHoverState?.focusState?.activeConnectorIds).not.toContain(connectorKCacheCopyNode?.id);
        expect(cachedRowHoverState?.focusState?.activeConnectorIds).toContain(connectorPreNode?.id);
        expect(cachedRowHoverState?.focusState?.activeConnectorIds).toContain(connectorPostNode?.id);
        expect(cachedRowHoverState?.focusState?.rowSelections).toContainEqual({
            nodeId: keyCacheNode?.id,
            rowIndex: 1
        });
        expect(cachedRowHoverState?.focusState?.rowSelections).toContainEqual({
            nodeId: keyCacheSourceNode?.id,
            rowIndex: 1
        });
        expect(cachedRowHoverState?.focusState?.rowSelections).toContainEqual({
            nodeId: keyCacheConcatResultNode?.id,
            rowIndex: 1
        });
        expect(cachedRowHoverState?.focusState?.columnSelections).toContainEqual({
            nodeId: transposeNode?.id,
            colIndex: 1
        });
        expect(cachedRowHoverState?.focusState?.columnSelections).toContainEqual({
            nodeId: preScoreNode?.id,
            colIndex: 1
        });
        expect(cachedRowHoverState?.focusState?.columnSelections).toContainEqual({
            nodeId: maskedInputNode?.id,
            colIndex: 1
        });
        expect(cachedRowHoverState?.focusState?.columnSelections).toContainEqual({
            nodeId: maskNode?.id,
            colIndex: 1
        });
        expect(cachedRowHoverState?.focusState?.columnSelections).toContainEqual({
            nodeId: postNode?.id,
            colIndex: 1
        });
        expect(cachedRowHoverState?.focusState?.columnSelections).toContainEqual({
            nodeId: postCopyNode?.id,
            colIndex: 1
        });

        expect(liveRowHoverState?.label).toBe('Key Vector');
        expect(liveRowHoverState?.info?.cachedKv).not.toBe(true);
        expect(liveRowHoverState?.focusState?.activeNodeIds).toContain(transposeNode?.id);
        expect(liveRowHoverState?.focusState?.activeConnectorIds).toContain(connectorKNode?.id);
        expect(liveRowHoverState?.focusState?.activeConnectorIds).toContain(connectorKCacheCopyNode?.id);
        expect(liveRowHoverState?.focusState?.activeConnectorIds).not.toContain(connectorKCacheSourceNode?.id);
        expect(liveRowHoverState?.focusState?.activeNodeIds).not.toContain(keyCacheNode?.id);
        expect(liveRowHoverState?.focusState?.activeNodeIds).not.toContain(keyCacheSourceNode?.id);
        expect(liveRowHoverState?.focusState?.rowSelections).toContainEqual({
            nodeId: projectionSourceNode?.id,
            rowIndex: 0
        });
        expect(liveRowHoverState?.focusState?.rowSelections).toContainEqual({
            nodeId: keyInputNode?.id,
            rowIndex: 0
        });
        expect(liveRowHoverState?.focusState?.rowSelections).toContainEqual({
            nodeId: keyProjectionOutputNode?.id,
            rowIndex: 0
        });
        expect(liveRowHoverState?.focusState?.rowSelections).toContainEqual({
            nodeId: keyProjectionOutputCopyNode?.id,
            rowIndex: 0
        });
        expect(liveRowHoverState?.focusState?.rowSelections).toContainEqual({
            nodeId: keyCacheConcatResultNode?.id,
            rowIndex: 3
        });
        expect(liveRowHoverState?.focusState?.columnSelections).toContainEqual({
            nodeId: transposeNode?.id,
            colIndex: 3
        });
    });

    it('maps decode value-post rows through the concat result so cached rows stay cached and the live row stays live', () => {
        const {
            index,
            projectionSourceNode,
            valueInputNode,
            valueProjectionOutputNode,
            valueProjectionOutputCopyNode,
            valueCacheNode,
            valueCacheSourceNode,
            valueCacheConcatResultNode,
            valuePostNode,
            connectorVNode,
            connectorVCacheSourceNode,
            connectorVCacheCopyNode
        } = buildSceneFixtures(4, {
            kvCacheState: {
                kvCacheModeEnabled: true,
                kvCachePrefillActive: false,
                kvCacheDecodeActive: true,
                kvCachePassIndex: 1
            }
        });

        const cachedRowHoverState = resolveMhsaDetailHoverState(index, {
            node: valuePostNode,
            rowHit: {
                rowIndex: 1,
                rowItem: valuePostNode?.rowItems?.[1]
            }
        });
        const liveRowHoverState = resolveMhsaDetailHoverState(index, {
            node: valuePostNode,
            rowHit: {
                rowIndex: 3,
                rowItem: valuePostNode?.rowItems?.[3]
            }
        });

        expect(valuePostNode?.dimensions?.rows).toBe(4);
        expect(valuePostNode?.rowItems).toHaveLength(4);
        expect(valuePostNode?.rowItems?.[2]?.semantic?.concatResultPart).toBe('cache');
        expect(valuePostNode?.rowItems?.[3]?.semantic?.concatResultPart).toBe('live');

        expect(cachedRowHoverState?.label).toBe('Cached Value Vector');
        expect(cachedRowHoverState?.info?.cachedKv).toBe(true);
        expect(cachedRowHoverState?.focusState?.activeNodeIds).not.toContain(valueProjectionOutputNode?.id);
        expect(cachedRowHoverState?.focusState?.activeNodeIds).not.toContain(valueProjectionOutputCopyNode?.id);
        expect(cachedRowHoverState?.focusState?.activeConnectorIds).toContain(connectorVNode?.id);
        expect(cachedRowHoverState?.focusState?.activeConnectorIds).toContain(connectorVCacheSourceNode?.id);
        expect(cachedRowHoverState?.focusState?.activeConnectorIds).not.toContain(connectorVCacheCopyNode?.id);
        expect(cachedRowHoverState?.focusState?.rowSelections).toContainEqual({
            nodeId: valueCacheNode?.id,
            rowIndex: 1
        });
        expect(cachedRowHoverState?.focusState?.rowSelections).toContainEqual({
            nodeId: valueCacheSourceNode?.id,
            rowIndex: 1
        });
        expect(cachedRowHoverState?.focusState?.rowSelections).toContainEqual({
            nodeId: valueCacheConcatResultNode?.id,
            rowIndex: 1
        });
        expect(cachedRowHoverState?.focusState?.rowSelections).toContainEqual({
            nodeId: valuePostNode?.id,
            rowIndex: 1
        });
        expect(
            cachedRowHoverState?.focusState?.rowSelections?.some((selection) => (
                selection.nodeId === valueProjectionOutputNode?.id && selection.rowIndex === 0
            ))
        ).toBe(false);
        expect(
            cachedRowHoverState?.focusState?.rowSelections?.some((selection) => (
                selection.nodeId === valueProjectionOutputCopyNode?.id && selection.rowIndex === 0
            ))
        ).toBe(false);

        expect(liveRowHoverState?.label).toBe('Value Vector');
        expect(liveRowHoverState?.info?.cachedKv).not.toBe(true);
        expect(liveRowHoverState?.focusState?.activeConnectorIds).toContain(connectorVNode?.id);
        expect(liveRowHoverState?.focusState?.activeConnectorIds).toContain(connectorVCacheCopyNode?.id);
        expect(liveRowHoverState?.focusState?.activeConnectorIds).not.toContain(connectorVCacheSourceNode?.id);
        expect(liveRowHoverState?.focusState?.activeNodeIds).not.toContain(valueCacheNode?.id);
        expect(liveRowHoverState?.focusState?.activeNodeIds).not.toContain(valueCacheSourceNode?.id);
        expect(liveRowHoverState?.focusState?.rowSelections).toContainEqual({
            nodeId: projectionSourceNode?.id,
            rowIndex: 0
        });
        expect(liveRowHoverState?.focusState?.rowSelections).toContainEqual({
            nodeId: valueInputNode?.id,
            rowIndex: 0
        });
        expect(liveRowHoverState?.focusState?.rowSelections).toContainEqual({
            nodeId: valueProjectionOutputNode?.id,
            rowIndex: 0
        });
        expect(liveRowHoverState?.focusState?.rowSelections).toContainEqual({
            nodeId: valueProjectionOutputCopyNode?.id,
            rowIndex: 0
        });
        expect(liveRowHoverState?.focusState?.rowSelections).toContainEqual({
            nodeId: valueCacheConcatResultNode?.id,
            rowIndex: 3
        });
        expect(liveRowHoverState?.focusState?.rowSelections).toContainEqual({
            nodeId: valuePostNode?.id,
            rowIndex: 3
        });
    });

    it('treats decode cache-next live rows as cached vectors while only highlighting the live path rows', () => {
        const {
            index,
            projectionSourceNode,
            keyInputNode,
            keyProjectionOutputNode,
            keyProjectionOutputCopyNode,
            keyCacheNode,
            keyCacheSourceNode,
            keyCacheConcatResultNode,
            keyCacheNextNode,
            transposeNode,
            connectorKNode,
            connectorKCacheSourceNode,
            connectorKCacheCopyNode,
            connectorKCacheNextNode,
            valueInputNode,
            valueProjectionOutputNode,
            valueProjectionOutputCopyNode,
            valueCacheNode,
            valueCacheSourceNode,
            valueCacheConcatResultNode,
            valueCacheNextNode,
            valuePostNode,
            connectorVNode,
            connectorVCacheSourceNode,
            connectorVCacheCopyNode,
            connectorVCacheNextNode
        } = buildSceneFixtures(4, {
            kvCacheState: {
                kvCacheModeEnabled: true,
                kvCachePrefillActive: false,
                kvCacheDecodeActive: true,
                kvCachePassIndex: 1
            }
        });

        const keyHoverState = resolveMhsaDetailHoverState(index, {
            node: keyCacheNextNode,
            rowHit: {
                rowIndex: 3,
                rowItem: keyCacheNextNode?.rowItems?.[3]
            }
        });
        const valueHoverState = resolveMhsaDetailHoverState(index, {
            node: valueCacheNextNode,
            rowHit: {
                rowIndex: 3,
                rowItem: valueCacheNextNode?.rowItems?.[3]
            }
        });

        expect(keyHoverState?.label).toBe('Cached Key Vector');
        expect(keyHoverState?.info?.cachedKv).toBe(true);
        expect(keyHoverState?.focusState?.activeConnectorIds).toContain(connectorKNode?.id);
        expect(keyHoverState?.focusState?.activeConnectorIds).toContain(connectorKCacheCopyNode?.id);
        expect(keyHoverState?.focusState?.activeConnectorIds).toContain(connectorKCacheNextNode?.id);
        expect(keyHoverState?.focusState?.activeConnectorIds).not.toContain(connectorKCacheSourceNode?.id);
        expect(keyHoverState?.focusState?.rowSelections).toContainEqual({
            nodeId: projectionSourceNode?.id,
            rowIndex: 0
        });
        expect(keyHoverState?.focusState?.rowSelections).toContainEqual({
            nodeId: keyInputNode?.id,
            rowIndex: 0
        });
        expect(keyHoverState?.focusState?.rowSelections).toContainEqual({
            nodeId: keyProjectionOutputNode?.id,
            rowIndex: 0
        });
        expect(keyHoverState?.focusState?.rowSelections).toContainEqual({
            nodeId: keyProjectionOutputCopyNode?.id,
            rowIndex: 0
        });
        expect(keyHoverState?.focusState?.rowSelections).toContainEqual({
            nodeId: keyCacheConcatResultNode?.id,
            rowIndex: 3
        });
        expect(keyHoverState?.focusState?.rowSelections).toContainEqual({
            nodeId: keyCacheNextNode?.id,
            rowIndex: 3
        });
        expect(
            keyHoverState?.focusState?.rowSelections?.some((selection) => (
                selection.nodeId === keyCacheNode?.id || selection.nodeId === keyCacheSourceNode?.id
            ))
        ).toBe(false);
        expect(keyHoverState?.focusState?.columnSelections).toContainEqual({
            nodeId: transposeNode?.id,
            colIndex: 3
        });

        expect(valueHoverState?.label).toBe('Cached Value Vector');
        expect(valueHoverState?.info?.cachedKv).toBe(true);
        expect(valueHoverState?.focusState?.activeConnectorIds).toContain(connectorVNode?.id);
        expect(valueHoverState?.focusState?.activeConnectorIds).toContain(connectorVCacheCopyNode?.id);
        expect(valueHoverState?.focusState?.activeConnectorIds).toContain(connectorVCacheNextNode?.id);
        expect(valueHoverState?.focusState?.activeConnectorIds).not.toContain(connectorVCacheSourceNode?.id);
        expect(valueHoverState?.focusState?.rowSelections).toContainEqual({
            nodeId: projectionSourceNode?.id,
            rowIndex: 0
        });
        expect(valueHoverState?.focusState?.rowSelections).toContainEqual({
            nodeId: valueInputNode?.id,
            rowIndex: 0
        });
        expect(valueHoverState?.focusState?.rowSelections).toContainEqual({
            nodeId: valueProjectionOutputNode?.id,
            rowIndex: 0
        });
        expect(valueHoverState?.focusState?.rowSelections).toContainEqual({
            nodeId: valueProjectionOutputCopyNode?.id,
            rowIndex: 0
        });
        expect(valueHoverState?.focusState?.rowSelections).toContainEqual({
            nodeId: valueCacheConcatResultNode?.id,
            rowIndex: 3
        });
        expect(valueHoverState?.focusState?.rowSelections).toContainEqual({
            nodeId: valueCacheNextNode?.id,
            rowIndex: 3
        });
        expect(valueHoverState?.focusState?.rowSelections).toContainEqual({
            nodeId: valuePostNode?.id,
            rowIndex: 3
        });
        expect(
            valueHoverState?.focusState?.rowSelections?.some((selection) => (
                selection.nodeId === valueCacheNode?.id || selection.nodeId === valueCacheSourceNode?.id
            ))
        ).toBe(false);
    });

    it('stacks decode K/V caches above the copied live rows inside concat equations and feeds them from the left', () => {
        const {
            scene,
            layout,
            keyProjectionOutputNode,
            valueProjectionOutputNode,
            transposeNode,
            valuePostNode,
            keyProjectionOutputCopyNode,
            valueProjectionOutputCopyNode,
            keyCacheNode,
            keyCacheSourceNode,
            valueCacheNode,
            valueCacheSourceNode,
            keyCacheConcatLabelNode,
            keyCacheConcatOpenNode,
            keyCacheConcatCloseNode,
            keyCacheConcatEqualsNode,
            keyCacheConcatResultNode,
            keyCacheNextNode,
            valueCacheConcatLabelNode,
            valueCacheConcatOpenNode,
            valueCacheConcatCloseNode,
            valueCacheConcatEqualsNode,
            valueCacheConcatResultNode,
            valueCacheNextNode,
            connectorKNode,
            connectorKCacheSourceNode,
            connectorKCacheNode,
            connectorVNode,
            connectorVCacheSourceNode,
            connectorVCacheNode,
            connectorKCacheCopyNode,
            connectorVCacheCopyNode,
            connectorKCacheNextNode,
            connectorVCacheNextNode
        } = buildSceneFixtures(4, {
            kvCacheState: {
                kvCacheModeEnabled: true,
                kvCachePrefillActive: false,
                kvCacheDecodeActive: true,
                kvCachePassIndex: 1
            }
        });
        const nonDecodeScene = buildMhsaSceneModel({
            previewData: createPreviewData(4),
            layerIndex: 2,
            headIndex: 1
        });
        const nonDecodeLayout = buildSceneLayout(nonDecodeScene);
        const decodeProjectionStackNode = flattenSceneNodes(scene).find((node) => node.role === 'projection-stack') || null;
        const nonDecodeProjectionStackNode = flattenSceneNodes(nonDecodeScene).find((node) => node.role === 'projection-stack') || null;
        const nonDecodeTransposeNode = flattenSceneNodes(nonDecodeScene).find((node) => (
            node.role === 'attention-key-transpose'
        )) || null;

        const keyOutputEntry = layout?.registry?.getNodeEntry(keyProjectionOutputNode?.id || '');
        const valueOutputEntry = layout?.registry?.getNodeEntry(valueProjectionOutputNode?.id || '');
        const decodeTransposeEntry = layout?.registry?.getNodeEntry(transposeNode?.id || '');
        const nonDecodeTransposeEntry = nonDecodeLayout?.registry?.getNodeEntry(nonDecodeTransposeNode?.id || '');
        const keyOutputCopyEntry = layout?.registry?.getNodeEntry(keyProjectionOutputCopyNode?.id || '');
        const valueOutputCopyEntry = layout?.registry?.getNodeEntry(valueProjectionOutputCopyNode?.id || '');
        const keyCacheEntry = layout?.registry?.getNodeEntry(keyCacheNode?.id || '');
        const keyCacheSourceEntry = layout?.registry?.getNodeEntry(keyCacheSourceNode?.id || '');
        const valueCacheEntry = layout?.registry?.getNodeEntry(valueCacheNode?.id || '');
        const valueCacheSourceEntry = layout?.registry?.getNodeEntry(valueCacheSourceNode?.id || '');
        const keyConcatLabelEntry = layout?.registry?.getNodeEntry(keyCacheConcatLabelNode?.id || '');
        const keyConcatOpenEntry = layout?.registry?.getNodeEntry(keyCacheConcatOpenNode?.id || '');
        const keyConcatCloseEntry = layout?.registry?.getNodeEntry(keyCacheConcatCloseNode?.id || '');
        const keyConcatEqualsEntry = layout?.registry?.getNodeEntry(keyCacheConcatEqualsNode?.id || '');
        const keyConcatResultEntry = layout?.registry?.getNodeEntry(keyCacheConcatResultNode?.id || '');
        const keyCacheNextEntry = layout?.registry?.getNodeEntry(keyCacheNextNode?.id || '');
        const valueConcatLabelEntry = layout?.registry?.getNodeEntry(valueCacheConcatLabelNode?.id || '');
        const valueConcatOpenEntry = layout?.registry?.getNodeEntry(valueCacheConcatOpenNode?.id || '');
        const valueConcatCloseEntry = layout?.registry?.getNodeEntry(valueCacheConcatCloseNode?.id || '');
        const valueConcatEqualsEntry = layout?.registry?.getNodeEntry(valueCacheConcatEqualsNode?.id || '');
        const valueConcatResultEntry = layout?.registry?.getNodeEntry(valueCacheConcatResultNode?.id || '');
        const valueCacheNextEntry = layout?.registry?.getNodeEntry(valueCacheNextNode?.id || '');
        const valuePostEntry = layout?.registry?.getNodeEntry(valuePostNode?.id || '');
        const keyConnectorEntry = layout?.registry?.getConnectorEntry(connectorKNode?.id || '');
        const valueConnectorEntry = layout?.registry?.getConnectorEntry(connectorVNode?.id || '');
        const keySourceConnectorEntry = layout?.registry?.getConnectorEntry(connectorKCacheSourceNode?.id || '');
        const valueSourceConnectorEntry = layout?.registry?.getConnectorEntry(connectorVCacheSourceNode?.id || '');
        const keyCopyConnectorEntry = layout?.registry?.getConnectorEntry(connectorKCacheCopyNode?.id || '');
        const valueCopyConnectorEntry = layout?.registry?.getConnectorEntry(connectorVCacheCopyNode?.id || '');
        const keyNextConnectorEntry = layout?.registry?.getConnectorEntry(connectorKCacheNextNode?.id || '');
        const valueNextConnectorEntry = layout?.registry?.getConnectorEntry(connectorVCacheNextNode?.id || '');
        const keyConnectorLastPoint = keyConnectorEntry?.pathPoints?.[
            Math.max(0, (keyConnectorEntry?.pathPoints?.length || 1) - 1)
        ];
        const valueConnectorLastPoint = valueConnectorEntry?.pathPoints?.[
            Math.max(0, (valueConnectorEntry?.pathPoints?.length || 1) - 1)
        ];
        const keySourceConnectorLastPoint = keySourceConnectorEntry?.pathPoints?.[
            Math.max(0, (keySourceConnectorEntry?.pathPoints?.length || 1) - 1)
        ];
        const keySourceConnectorElbowPoint = keySourceConnectorEntry?.pathPoints?.[1] || null;
        const valueSourceConnectorLastPoint = valueSourceConnectorEntry?.pathPoints?.[
            Math.max(0, (valueSourceConnectorEntry?.pathPoints?.length || 1) - 1)
        ];
        const valueSourceConnectorElbowPoint = valueSourceConnectorEntry?.pathPoints?.[1] || null;
        const keyCopyConnectorLastPoint = keyCopyConnectorEntry?.pathPoints?.[
            Math.max(0, (keyCopyConnectorEntry?.pathPoints?.length || 1) - 1)
        ];
        const keyCopyConnectorElbowPoint = keyCopyConnectorEntry?.pathPoints?.[1] || null;
        const valueCopyConnectorLastPoint = valueCopyConnectorEntry?.pathPoints?.[
            Math.max(0, (valueCopyConnectorEntry?.pathPoints?.length || 1) - 1)
        ];
        const valueCopyConnectorElbowPoint = valueCopyConnectorEntry?.pathPoints?.[1] || null;
        const keyNextConnectorLastPoint = keyNextConnectorEntry?.pathPoints?.[
            Math.max(0, (keyNextConnectorEntry?.pathPoints?.length || 1) - 1)
        ];
        const valueNextConnectorLastPoint = valueNextConnectorEntry?.pathPoints?.[
            Math.max(0, (valueNextConnectorEntry?.pathPoints?.length || 1) - 1)
        ];

        expect(keyProjectionOutputNode?.dimensions?.rows).toBe(1);
        expect(valueProjectionOutputNode?.dimensions?.rows).toBe(1);
        expect(keyProjectionOutputNode?.rowItems).toHaveLength(1);
        expect(valueProjectionOutputNode?.rowItems).toHaveLength(1);
        expect(keyProjectionOutputCopyNode?.dimensions?.rows).toBe(1);
        expect(valueProjectionOutputCopyNode?.dimensions?.rows).toBe(1);
        expect(keyProjectionOutputCopyNode?.rowItems).toHaveLength(1);
        expect(valueProjectionOutputCopyNode?.rowItems).toHaveLength(1);
        expect(keyProjectionOutputCopyNode?.rowItems?.[0]?.semantic?.tokenIndex)
            .toBe(keyProjectionOutputNode?.rowItems?.[0]?.semantic?.tokenIndex);
        expect(valueProjectionOutputCopyNode?.rowItems?.[0]?.semantic?.tokenIndex)
            .toBe(valueProjectionOutputNode?.rowItems?.[0]?.semantic?.tokenIndex);
        expect(Number(decodeProjectionStackNode?.metadata?.gapOverride) || 0)
            .toBeGreaterThan(Number(nonDecodeProjectionStackNode?.metadata?.gapOverride) || 0);
        expect(
            (decodeTransposeEntry?.contentBounds?.y || 0) - (keyOutputEntry?.contentBounds?.y || 0)
        ).toBeLessThan(
            ((nonDecodeTransposeEntry?.contentBounds?.y || 0) - (
                nonDecodeLayout?.registry?.getNodeEntry(
                    flattenSceneNodes(nonDecodeScene).find((node) => (
                        node.role === 'projection-output'
                        && String(node.metadata?.kind || '').toLowerCase() === 'k'
                    ))?.id || ''
                )?.contentBounds?.y || 0
            ))
        );

        expect(keyCacheNode?.dimensions?.rows).toBe(3);
        expect(valueCacheNode?.dimensions?.rows).toBe(3);
        expect(keyCacheNode?.rowItems).toHaveLength(3);
        expect(valueCacheNode?.rowItems).toHaveLength(3);
        expect(keyCacheNode?.metadata?.caption).toBeNull();
        expect(valueCacheNode?.metadata?.caption).toBeNull();
        expect(keyCacheSourceNode?.dimensions?.rows).toBe(3);
        expect(valueCacheSourceNode?.dimensions?.rows).toBe(3);
        expect(keyCacheSourceNode?.rowItems).toHaveLength(3);
        expect(valueCacheSourceNode?.rowItems).toHaveLength(3);
        expect(keyProjectionOutputCopyNode?.metadata?.caption).toBeNull();
        expect(valueProjectionOutputCopyNode?.metadata?.caption).toBeNull();
        expect(keyCacheConcatLabelNode?.text).toBe('concat');
        expect(valueCacheConcatLabelNode?.text).toBe('concat');
        expect(keyCacheConcatLabelNode?.metadata?.fontScale).toBeGreaterThan(1.1);
        expect(valueCacheConcatLabelNode?.metadata?.fontScale).toBeGreaterThan(1.1);
        expect(keyCacheConcatResultNode?.dimensions?.rows).toBe(4);
        expect(valueCacheConcatResultNode?.dimensions?.rows).toBe(4);
        expect(keyCacheConcatResultNode?.rowItems).toHaveLength(4);
        expect(valueCacheConcatResultNode?.rowItems).toHaveLength(4);
        expect(keyCacheConcatResultNode?.rowItems?.[0]?.semantic?.tokenIndex).toBe(0);
        expect(keyCacheConcatResultNode?.rowItems?.[3]?.semantic?.tokenIndex).toBe(3);
        expect(valueCacheConcatResultNode?.rowItems?.[0]?.semantic?.tokenIndex).toBe(0);
        expect(valueCacheConcatResultNode?.rowItems?.[3]?.semantic?.tokenIndex).toBe(3);
        expect(keyCacheConcatResultNode?.rowItems?.[2]?.semantic?.concatResultPart).toBe('cache');
        expect(keyCacheConcatResultNode?.rowItems?.[3]?.semantic?.concatResultPart).toBe('live');
        expect(valueCacheConcatResultNode?.rowItems?.[2]?.semantic?.concatResultPart).toBe('cache');
        expect(valueCacheConcatResultNode?.rowItems?.[3]?.semantic?.concatResultPart).toBe('live');
        expect(keyCacheNextNode?.dimensions?.rows).toBe(4);
        expect(valueCacheNextNode?.dimensions?.rows).toBe(4);
        expect(keyCacheNextNode?.label?.tex).toBe('K_{\\mathrm{cache\\_next}}');
        expect(valueCacheNextNode?.label?.tex).toBe('V_{\\mathrm{cache\\_next}}');
        expect(keyCacheNextNode?.label?.text).toBe('K_cache_next');
        expect(valueCacheNextNode?.label?.text).toBe('V_cache_next');
        expect(keyCacheNextNode?.visual?.opacity).toBeCloseTo(DECODE_CACHE_NEXT_OPACITY, 3);
        expect(valueCacheNextNode?.visual?.opacity).toBeCloseTo(DECODE_CACHE_NEXT_OPACITY, 3);
        expect(keyCacheNextNode?.rowItems?.[3]?.semantic?.concatResultPart).toBe('live');
        expect(valueCacheNextNode?.rowItems?.[3]?.semantic?.concatResultPart).toBe('live');
        expect(valuePostNode?.dimensions?.rows).toBe(4);
        expect(valuePostNode?.rowItems).toHaveLength(4);
        expect(valuePostNode?.rowItems?.[2]?.semantic?.concatResultPart).toBe('cache');
        expect(valuePostNode?.rowItems?.[3]?.semantic?.concatResultPart).toBe('live');
        expect(keyCacheConcatOpenNode?.text).toBe('(');
        expect(keyCacheConcatCloseNode?.text).toBe(')');
        expect(keyCacheConcatEqualsNode?.text).toBe('=');
        expect(keyCacheConcatOpenNode?.metadata?.operatorScaleX).toBeGreaterThan(1.1);
        expect(keyCacheConcatOpenNode?.metadata?.operatorScaleX).toBeLessThan(1.3);
        expect(keyCacheConcatOpenNode?.metadata?.operatorScaleY).toBeGreaterThan(2.8);
        expect(keyCacheConcatOpenNode?.metadata?.operatorScaleY).toBeLessThan(3.2);
        expect(keyCacheConcatCloseNode?.metadata?.operatorScaleX).toBeGreaterThan(1.1);
        expect(keyCacheConcatCloseNode?.metadata?.operatorScaleX).toBeLessThan(1.3);
        expect(keyCacheConcatCloseNode?.metadata?.operatorScaleY).toBeGreaterThan(2.8);
        expect(keyCacheConcatCloseNode?.metadata?.operatorScaleY).toBeLessThan(3.2);
        expect(valueCacheConcatOpenNode?.text).toBe('(');
        expect(valueCacheConcatCloseNode?.text).toBe(')');
        expect(valueCacheConcatEqualsNode?.text).toBe('=');
        expect(valueCacheConcatOpenNode?.metadata?.operatorScaleX).toBeGreaterThan(1.1);
        expect(valueCacheConcatOpenNode?.metadata?.operatorScaleX).toBeLessThan(1.3);
        expect(valueCacheConcatOpenNode?.metadata?.operatorScaleY).toBeGreaterThan(2.8);
        expect(valueCacheConcatOpenNode?.metadata?.operatorScaleY).toBeLessThan(3.2);
        expect(valueCacheConcatCloseNode?.metadata?.operatorScaleX).toBeGreaterThan(1.1);
        expect(valueCacheConcatCloseNode?.metadata?.operatorScaleX).toBeLessThan(1.3);
        expect(valueCacheConcatCloseNode?.metadata?.operatorScaleY).toBeGreaterThan(2.8);
        expect(valueCacheConcatCloseNode?.metadata?.operatorScaleY).toBeLessThan(3.2);
        expect(connectorKNode?.source?.nodeId).toBe(keyCacheConcatResultNode?.id);
        expect(connectorKNode?.target?.nodeId).toBe(transposeNode?.id);
        expect(connectorVNode?.source?.nodeId).toBe(valueCacheConcatResultNode?.id);
        expect(connectorVNode?.target?.nodeId).toBe(valuePostNode?.id);
        expect(connectorKCacheNextNode?.source?.nodeId).toBe(keyCacheConcatResultNode?.id);
        expect(connectorKCacheNextNode?.target?.nodeId).toBe(keyCacheNextNode?.id);
        expect(connectorVCacheNextNode?.source?.nodeId).toBe(valueCacheConcatResultNode?.id);
        expect(connectorVCacheNextNode?.target?.nodeId).toBe(valueCacheNextNode?.id);
        expect(connectorKCacheNode).toBeNull();
        expect(connectorVCacheNode).toBeNull();
        expect(keyConnectorLastPoint?.y).toBeCloseTo(
            resolveEntryCaptionBottom(decodeTransposeEntry) + DECODE_CONNECTOR_CAPTION_EXIT_GAP,
            4
        );
        expect(valueConnectorLastPoint?.y).toBeCloseTo(
            resolveEntryCaptionBottom(valuePostEntry) + DECODE_VALUE_CONNECTOR_TARGET_GAP,
            4
        );
        expect(keyConnectorEntry?.pathPoints?.[0]?.y).toBeCloseTo(
            (keyConcatResultEntry?.anchors?.right?.y || 0)
            + ((Number(keyConcatResultEntry?.contentBounds?.height) || 0) * DECODE_CONCAT_SOURCE_OFFSET_Y_RATIO),
            4
        );
        expect(valueConnectorEntry?.pathPoints?.[0]?.y).toBeCloseTo(
            (valueConcatResultEntry?.anchors?.right?.y || 0)
            + ((Number(valueConcatResultEntry?.contentBounds?.height) || 0) * DECODE_CONCAT_SOURCE_OFFSET_Y_RATIO),
            4
        );
        expect(valueConnectorEntry?.pathPoints?.[0]?.x).toBeCloseTo(
            (valueConcatResultEntry?.anchors?.right?.x || 0) + DECODE_VALUE_CONNECTOR_SOURCE_GAP,
            4
        );
        expect(keyConnectorEntry?.pathPoints?.[0]?.x).toBeCloseTo(
            (keyConcatResultEntry?.anchors?.right?.x || 0) + DECODE_KEY_CONNECTOR_SOURCE_GAP,
            4
        );
        expect(keyCopyConnectorEntry?.pathPoints?.[0]?.y).toBeCloseTo(
            keyOutputEntry?.anchors?.right?.y || 0,
            4
        );
        expect(valueCopyConnectorEntry?.pathPoints?.[0]?.y).toBeCloseTo(
            valueOutputEntry?.anchors?.right?.y || 0,
            4
        );
        expect(keyNextConnectorEntry?.pathPoints?.[0]?.y).toBeCloseTo(
            (keyConcatResultEntry?.anchors?.right?.y || 0)
            + ((Number(keyConcatResultEntry?.contentBounds?.height) || 0) * DECODE_CACHE_NEXT_SOURCE_OFFSET_Y_RATIO),
            4
        );
        expect(keyNextConnectorEntry?.pathPoints?.[0]?.x).toBeCloseTo(
            (keyConcatResultEntry?.anchors?.right?.x || 0) + DECODE_KEY_CONNECTOR_SOURCE_GAP,
            4
        );
        expect(valueNextConnectorEntry?.pathPoints?.[0]?.y).toBeCloseTo(
            (valueConcatResultEntry?.anchors?.right?.y || 0)
            + ((Number(valueConcatResultEntry?.contentBounds?.height) || 0) * DECODE_CACHE_NEXT_SOURCE_OFFSET_Y_RATIO),
            4
        );
        expect(valueNextConnectorEntry?.pathPoints?.[0]?.x).toBeCloseTo(
            valueConnectorEntry?.pathPoints?.[0]?.x || 0,
            4
        );
        expect(keyNextConnectorLastPoint?.y).toBeCloseTo(
            (keyCacheNextEntry?.anchors?.top?.y || 0) - DECODE_CACHE_NEXT_TARGET_GAP,
            4
        );
        expect(valueNextConnectorLastPoint?.y).toBeCloseTo(
            (valueCacheNextEntry?.anchors?.top?.y || 0) - DECODE_CACHE_NEXT_TARGET_GAP,
            4
        );
        expect(keyCacheNextEntry?.contentBounds?.y).toBeGreaterThan(
            (keyConcatResultEntry?.contentBounds?.y || 0) + (keyConcatResultEntry?.contentBounds?.height || 0)
        );
        expect(valueCacheNextEntry?.contentBounds?.y).toBeGreaterThan(
            (valueConcatResultEntry?.contentBounds?.y || 0) + (valueConcatResultEntry?.contentBounds?.height || 0)
        );
        expect(keyCacheNextEntry?.contentBounds?.x).toBeGreaterThan(
            (keyConcatResultEntry?.contentBounds?.x || 0) + (keyConcatResultEntry?.contentBounds?.width || 0)
        );
        expect(valueCacheNextEntry?.contentBounds?.x).toBeGreaterThan(
            (valueConcatResultEntry?.contentBounds?.x || 0) + (valueConcatResultEntry?.contentBounds?.width || 0)
        );
        expect(
            (keyCacheNextEntry?.contentBounds?.x || 0)
            - ((keyConcatResultEntry?.contentBounds?.x || 0) + (keyConcatResultEntry?.contentBounds?.width || 0))
        ).toBeGreaterThan(30);
        expect(
            (valueCacheNextEntry?.contentBounds?.x || 0)
            - ((valueConcatResultEntry?.contentBounds?.x || 0) + (valueConcatResultEntry?.contentBounds?.width || 0))
        ).toBeGreaterThan(30);
        expect(
            (keyCacheNextEntry?.contentBounds?.y || 0)
            - ((keyConcatResultEntry?.contentBounds?.y || 0) + (keyConcatResultEntry?.contentBounds?.height || 0))
        ).toBeGreaterThan(24);
        expect(
            (valueCacheNextEntry?.contentBounds?.y || 0)
            - ((valueConcatResultEntry?.contentBounds?.y || 0) + (valueConcatResultEntry?.contentBounds?.height || 0))
        ).toBeGreaterThan(24);

        expect(keyOutputCopyEntry?.contentBounds?.x).toBeGreaterThan(
            (keyOutputEntry?.contentBounds?.x || 0) + (keyOutputEntry?.contentBounds?.width || 0)
        );
        expect(valueOutputCopyEntry?.contentBounds?.x).toBeGreaterThan(
            (valueOutputEntry?.contentBounds?.x || 0) + (valueOutputEntry?.contentBounds?.width || 0)
        );
        expect(Math.abs(
            (keyCacheEntry?.contentBounds?.x || 0) - (keyOutputCopyEntry?.contentBounds?.x || 0)
        )).toBeLessThanOrEqual(12);
        expect(Math.abs(
            (valueCacheEntry?.contentBounds?.x || 0) - (valueOutputCopyEntry?.contentBounds?.x || 0)
        )).toBeLessThanOrEqual(12);
        expect(
            (keyCacheSourceEntry?.contentBounds?.x || 0) + (keyCacheSourceEntry?.contentBounds?.width || 0)
        ).toBeLessThan(keyOutputEntry?.contentBounds?.x || 0);
        expect(
            (valueCacheSourceEntry?.contentBounds?.x || 0) + (valueCacheSourceEntry?.contentBounds?.width || 0)
        ).toBeLessThan(valueOutputEntry?.contentBounds?.x || 0);
        expect(
            (keyCacheSourceEntry?.contentBounds?.y || 0) + (keyCacheSourceEntry?.contentBounds?.height || 0)
        ).toBeLessThan(keyOutputEntry?.contentBounds?.y || 0);
        expect(
            (valueCacheSourceEntry?.contentBounds?.y || 0) + (valueCacheSourceEntry?.contentBounds?.height || 0)
        ).toBeLessThan(valueOutputEntry?.contentBounds?.y || 0);
        expect(
            (keyCacheSourceEntry?.contentBounds?.x || 0) + (keyCacheSourceEntry?.contentBounds?.width || 0)
        ).toBeLessThan(keyCacheEntry?.contentBounds?.x || 0);
        expect(
            (valueCacheSourceEntry?.contentBounds?.x || 0) + (valueCacheSourceEntry?.contentBounds?.width || 0)
        ).toBeLessThan(valueCacheEntry?.contentBounds?.x || 0);
        expect(
            (keyCacheSourceEntry?.contentBounds?.y || 0) + (keyCacheSourceEntry?.contentBounds?.height || 0)
        ).toBeLessThan(valueCacheSourceEntry?.contentBounds?.y || 0);
        expect(
            (keyCacheEntry?.contentBounds?.y || 0) + (keyCacheEntry?.contentBounds?.height || 0)
        ).toBeLessThan(keyOutputCopyEntry?.contentBounds?.y || 0);
        expect(
            (keyOutputCopyEntry?.contentBounds?.y || 0)
            - ((keyCacheEntry?.contentBounds?.y || 0) + (keyCacheEntry?.contentBounds?.height || 0))
        ).toBeGreaterThan(12);
        expect(
            (valueCacheEntry?.contentBounds?.y || 0) + (valueCacheEntry?.contentBounds?.height || 0)
        ).toBeLessThan(valueOutputCopyEntry?.contentBounds?.y || 0);
        expect(
            (valueOutputCopyEntry?.contentBounds?.y || 0)
            - ((valueCacheEntry?.contentBounds?.y || 0) + (valueCacheEntry?.contentBounds?.height || 0))
        ).toBeGreaterThan(12);
        expect(keyCacheEntry?.labelBounds).toBeNull();
        expect(keyCacheEntry?.dimensionBounds).toBeNull();
        expect(valueCacheEntry?.labelBounds).toBeNull();
        expect(valueCacheEntry?.dimensionBounds).toBeNull();
        expect(keyOutputCopyEntry?.labelBounds).toBeNull();
        expect(keyOutputCopyEntry?.dimensionBounds).toBeNull();
        expect(valueOutputCopyEntry?.labelBounds).toBeNull();
        expect(valueOutputCopyEntry?.dimensionBounds).toBeNull();

        expect(keyConcatOpenEntry?.contentBounds?.x).toBeGreaterThan(
            (keyConcatLabelEntry?.contentBounds?.x || 0) + (keyConcatLabelEntry?.contentBounds?.width || 0)
        );
        expect(keyConcatEqualsEntry?.contentBounds?.x).toBeGreaterThan(
            (keyConcatCloseEntry?.contentBounds?.x || 0) + (keyConcatCloseEntry?.contentBounds?.width || 0)
        );
        expect(keyConcatResultEntry?.contentBounds?.x).toBeGreaterThan(
            (keyConcatEqualsEntry?.contentBounds?.x || 0) + (keyConcatEqualsEntry?.contentBounds?.width || 0)
        );
        expect(valueConcatOpenEntry?.contentBounds?.x).toBeGreaterThan(
            (valueConcatLabelEntry?.contentBounds?.x || 0) + (valueConcatLabelEntry?.contentBounds?.width || 0)
        );
        expect(valueConcatEqualsEntry?.contentBounds?.x).toBeGreaterThan(
            (valueConcatCloseEntry?.contentBounds?.x || 0) + (valueConcatCloseEntry?.contentBounds?.width || 0)
        );
        expect(valueConcatResultEntry?.contentBounds?.x).toBeGreaterThan(
            (valueConcatEqualsEntry?.contentBounds?.x || 0) + (valueConcatEqualsEntry?.contentBounds?.width || 0)
        );
        expect(Math.abs(
            ((keyConcatLabelEntry?.contentBounds?.y || 0) + ((keyConcatLabelEntry?.contentBounds?.height || 0) * 0.5))
            - (
                (
                    ((keyCacheEntry?.contentBounds?.y || 0) + (keyCacheEntry?.contentBounds?.height || 0))
                    + (keyOutputCopyEntry?.contentBounds?.y || 0)
                ) * 0.5
            )
        )).toBeLessThan(10);
        expect(Math.abs(
            ((valueConcatLabelEntry?.contentBounds?.y || 0) + ((valueConcatLabelEntry?.contentBounds?.height || 0) * 0.5))
            - (
                (
                    ((valueCacheEntry?.contentBounds?.y || 0) + (valueCacheEntry?.contentBounds?.height || 0))
                    + (valueOutputCopyEntry?.contentBounds?.y || 0)
                ) * 0.5
            )
        )).toBeLessThan(10);
        expect(
            ((keyOutputEntry?.contentBounds?.y || 0) + ((keyOutputEntry?.contentBounds?.height || 0) * 0.5))
            - ((keyConcatLabelEntry?.contentBounds?.y || 0) + ((keyConcatLabelEntry?.contentBounds?.height || 0) * 0.5))
        ).toBeLessThan(
            ((valueOutputEntry?.contentBounds?.y || 0) + ((valueOutputEntry?.contentBounds?.height || 0) * 0.5))
            - ((valueConcatLabelEntry?.contentBounds?.y || 0) + ((valueConcatLabelEntry?.contentBounds?.height || 0) * 0.5))
        );
        expect(Math.abs(
            ((keyConcatOpenEntry?.contentBounds?.y || 0) + ((keyConcatOpenEntry?.contentBounds?.height || 0) * 0.5))
            - (
                (
                    (keyCacheEntry?.contentBounds?.y || 0)
                    + ((keyOutputCopyEntry?.contentBounds?.y || 0) + (keyOutputCopyEntry?.contentBounds?.height || 0))
                ) * 0.5
            )
        )).toBeLessThan(16);
        expect(Math.abs(
            ((valueConcatOpenEntry?.contentBounds?.y || 0) + ((valueConcatOpenEntry?.contentBounds?.height || 0) * 0.5))
            - (
                (
                    (valueCacheEntry?.contentBounds?.y || 0)
                    + ((valueOutputCopyEntry?.contentBounds?.y || 0) + (valueOutputCopyEntry?.contentBounds?.height || 0))
                ) * 0.5
            )
        )).toBeLessThan(16);

        expect(keySourceConnectorEntry?.pathPoints).toHaveLength(3);
        expect(keySourceConnectorEntry?.pathPoints?.[0]?.x).toBeCloseTo(
            (keyCacheSourceEntry?.anchors?.right?.x || 0) + 6,
            4
        );
        expect(keySourceConnectorElbowPoint?.y).toBeCloseTo(
            keySourceConnectorEntry?.pathPoints?.[0]?.y || 0,
            4
        );
        expect(keySourceConnectorElbowPoint?.x).toBeCloseTo(
            keySourceConnectorLastPoint?.x || 0,
            4
        );
        expect(keySourceConnectorLastPoint?.x).toBeCloseTo(
            keyCacheEntry?.anchors?.top?.x || 0,
            4
        );
        expect(keySourceConnectorLastPoint?.y).toBeCloseTo(
            (keyCacheEntry?.anchors?.top?.y || 0) - 6,
            4
        );
        expect(keyCopyConnectorEntry?.pathPoints).toHaveLength(3);
        expect(keyCopyConnectorEntry?.pathPoints?.[0]?.x).toBeCloseTo(
            (keyOutputEntry?.anchors?.right?.x || 0) + 6,
            4
        );
        expect(keyCopyConnectorEntry?.pathPoints?.[0]?.y).toBeCloseTo(
            keyOutputEntry?.anchors?.right?.y || 0,
            4
        );
        expect(keyCopyConnectorElbowPoint?.y).toBeCloseTo(
            keyCopyConnectorEntry?.pathPoints?.[0]?.y || 0,
            4
        );
        expect(keyCopyConnectorElbowPoint?.x).toBeCloseTo(
            keyCopyConnectorLastPoint?.x || 0,
            4
        );
        expect(keyCopyConnectorLastPoint?.x).toBeCloseTo(
            keyOutputCopyEntry?.anchors?.bottom?.x || 0,
            4
        );
        expect(keyCopyConnectorLastPoint?.y).toBeCloseTo(
            resolveEntryCaptionBottom(keyOutputCopyEntry) + 2,
            4
        );
        expect(keyCopyConnectorLastPoint?.y).toBeLessThan(
            keyCopyConnectorEntry?.pathPoints?.[0]?.y || 0
        );

        expect(valueSourceConnectorEntry?.pathPoints).toHaveLength(3);
        expect(valueSourceConnectorEntry?.pathPoints?.[0]?.x).toBeCloseTo(
            (valueCacheSourceEntry?.anchors?.right?.x || 0) + 6,
            4
        );
        expect(valueSourceConnectorElbowPoint?.y).toBeCloseTo(
            valueSourceConnectorEntry?.pathPoints?.[0]?.y || 0,
            4
        );
        expect(valueSourceConnectorElbowPoint?.x).toBeCloseTo(
            valueSourceConnectorLastPoint?.x || 0,
            4
        );
        expect(valueSourceConnectorLastPoint?.x).toBeCloseTo(
            valueCacheEntry?.anchors?.top?.x || 0,
            4
        );
        expect(valueSourceConnectorLastPoint?.y).toBeCloseTo(
            (valueCacheEntry?.anchors?.top?.y || 0) - 6,
            4
        );
        expect(keyCacheSourceEntry?.anchors?.right?.x).toBeCloseTo(
            valueCacheSourceEntry?.anchors?.right?.x || 0,
            4
        );
        expect(keySourceConnectorEntry?.pathPoints?.[0]?.x).toBeCloseTo(
            valueSourceConnectorEntry?.pathPoints?.[0]?.x || 0,
            4
        );
        expect(valueCopyConnectorEntry?.pathPoints).toHaveLength(3);
        expect(valueCopyConnectorEntry?.pathPoints?.[0]?.x).toBeCloseTo(
            (valueOutputEntry?.anchors?.right?.x || 0) + 6,
            4
        );
        expect(valueCopyConnectorEntry?.pathPoints?.[0]?.y).toBeCloseTo(
            valueOutputEntry?.anchors?.right?.y || 0,
            4
        );
        expect(valueCopyConnectorElbowPoint?.y).toBeCloseTo(
            valueCopyConnectorEntry?.pathPoints?.[0]?.y || 0,
            4
        );
        expect(valueCopyConnectorElbowPoint?.x).toBeCloseTo(
            valueCopyConnectorLastPoint?.x || 0,
            4
        );
        expect(valueCopyConnectorLastPoint?.x).toBeCloseTo(
            valueOutputCopyEntry?.anchors?.bottom?.x || 0,
            4
        );
        expect(valueCopyConnectorLastPoint?.y).toBeCloseTo(
            resolveEntryCaptionBottom(valueOutputCopyEntry) + 2,
            4
        );
        expect(valueCopyConnectorLastPoint?.y).toBeLessThan(
            valueCopyConnectorEntry?.pathPoints?.[0]?.y || 0
        );
    });

    it('maps head-output hover rows to the Attention Weighted Sum tooltip payload', () => {
        const {
            index,
            headOutputNode,
            valuePostNode,
            postNode,
            postCopyNode
        } = buildSceneFixtures();

        const hoverState = resolveMhsaDetailHoverState(index, {
            node: headOutputNode,
            rowHit: {
                rowIndex: 0,
                rowItem: headOutputNode.rowItems[0]
            }
        });

        expect(hoverState?.label).toBe('Attention Weighted Sum');
        expect(hoverState?.info?.isWeightedSum).toBe(true);
        expect(hoverState?.info?.activationData?.label).toBe('Attention Weighted Sum');
        expect(hoverState?.info?.activationData?.stage).toBe('attention.weighted_sum');
        expect(hoverState?.info?.activationData?.isWeightedSum).toBe(true);
        expect(hoverState?.info?.activationData?.tokenIndex).toBe(0);
        expect(hoverState?.info?.activationData?.tokenLabel).toBe('Token A');
        expect(hoverState?.focusState?.activeNodeIds).toContain(valuePostNode.id);
        expect(hoverState?.focusState?.rowSelections).toContainEqual({
            nodeId: postNode.id,
            rowIndex: 0
        });
        expect(hoverState?.focusState?.rowSelections).toContainEqual({
            nodeId: postCopyNode.id,
            rowIndex: 0
        });
        expect(
            hoverState?.focusState?.rowSelections?.some((selection) => (
                selection.nodeId === valuePostNode.id && selection.rowIndex === 0
            ))
        ).toBe(false);
    });

    it('ignores score-matrix background hovers when no specific attention cell is hit', () => {
        const {
            index,
            preScoreNode,
            maskedInputNode,
            maskNode,
            postNode,
            postCopyNode
        } = buildSceneFixtures();

        [
            preScoreNode,
            maskedInputNode,
            maskNode,
            postNode,
            postCopyNode
        ].forEach((node) => {
            expect(resolveMhsaDetailHoverState(index, { node })).toBeNull();
        });
    });

    it('maps pre-score hover cells to a pre-softmax attention-score tooltip payload', () => {
        const {
            index,
            projectionSourceNode,
            preScoreNode,
            maskNode,
            postNode,
            postCopyNode,
            headOutputNode,
            valuePostNode,
            connectorVNode
        } = buildSceneFixtures();

        const hoverState = resolveMhsaDetailHoverState(index, {
            node: preScoreNode,
            cellHit: {
                rowIndex: 1,
                colIndex: 0,
                cellItem: preScoreNode.rowItems[1]?.cells?.[0]
            }
        });

        expect(hoverState?.label).toBe('Pre-Softmax Attention Score');
        expect(hoverState?.info?.activationData?.label).toBe('Pre-Softmax Attention Score');
        expect(hoverState?.info?.activationData?.stage).toBe('attention.pre');
        expect(hoverState?.info?.activationData?.queryTokenIndex).toBe(1);
        expect(hoverState?.info?.activationData?.queryTokenLabel).toBe('Token B');
        expect(hoverState?.info?.activationData?.keyTokenIndex).toBe(0);
        expect(hoverState?.info?.activationData?.keyTokenLabel).toBe('Token A');
        expect(hoverState?.info?.activationData?.preScore).toBe(0.25);
        expect(hoverState?.info?.activationData?.postScore).toBe(0.15);
        expect(hoverState?.info?.activationData?.showMaskValue).toBeUndefined();
        expect(hoverState?.focusState?.rowSelections).toContainEqual({
            nodeId: projectionSourceNode.id,
            rowIndex: 1
        });
        expect(hoverState?.focusState?.rowSelections).toContainEqual({
            nodeId: projectionSourceNode.id,
            rowIndex: 0
        });
        expect(hoverState?.focusState?.rowSelections).toContainEqual({
            nodeId: headOutputNode.id,
            rowIndex: 1
        });
        expect(hoverState?.focusState?.activeNodeIds).toContain(headOutputNode.id);
        expect(hoverState?.focusState?.activeNodeIds).not.toContain(valuePostNode.id);
        expect(hoverState?.focusState?.activeConnectorIds).not.toContain(connectorVNode.id);
        expect(hoverState?.focusState?.cellSelections).toContainEqual({
            nodeId: maskNode?.id,
            rowIndex: 1,
            colIndex: 0
        });
        expect(hoverState?.focusState?.cellSelections).toContainEqual({
            nodeId: postNode?.id,
            rowIndex: 1,
            colIndex: 0
        });
        expect(hoverState?.focusState?.cellSelections).toContainEqual({
            nodeId: postCopyNode?.id,
            rowIndex: 1,
            colIndex: 0
        });
    });

    it('maps masked-input hover cells to the same pre-softmax tooltip payload used by the first A_pre matrix', () => {
        const {
            index,
            maskedInputNode,
            postCopyNode,
            headOutputNode,
            valuePostNode,
            connectorVNode
        } = buildSceneFixtures();

        const hoverState = resolveMhsaDetailHoverState(index, {
            node: maskedInputNode,
            cellHit: {
                rowIndex: 0,
                colIndex: 1,
                cellItem: maskedInputNode.rowItems[0]?.cells?.[1]
            }
        });

        expect(hoverState?.label).toBe('Pre-Softmax Attention Score');
        expect(hoverState?.info?.activationData?.label).toBe('Pre-Softmax Attention Score');
        expect(hoverState?.info?.activationData?.stage).toBe('attention.pre');
        expect(hoverState?.info?.activationData?.preScore).toBe(0.25);
        expect(hoverState?.info?.activationData?.postScore).toBe(0);
        expect(hoverState?.info?.activationData?.maskValue).toBe(Number.NEGATIVE_INFINITY);
        expect(hoverState?.info?.activationData?.showMaskValue).toBeUndefined();
        expect(hoverState?.info?.activationData?.isMasked).toBe(true);
        expect(hoverState?.focusState?.rowSelections).toContainEqual({
            nodeId: headOutputNode.id,
            rowIndex: 0
        });
        expect(hoverState?.focusState?.activeNodeIds).toContain(headOutputNode.id);
        expect(hoverState?.focusState?.activeNodeIds).not.toContain(valuePostNode.id);
        expect(hoverState?.focusState?.activeConnectorIds).not.toContain(connectorVNode.id);
        expect(hoverState?.focusState?.cellSelections).toContainEqual({
            nodeId: postCopyNode?.id,
            rowIndex: 0,
            colIndex: 1
        });
    });

    it('maps causal-mask hover cells to a causal-mask tooltip payload', () => {
        const {
            index,
            maskNode,
            postNode,
            postCopyNode,
            headOutputNode,
            valuePostNode,
            connectorVNode
        } = buildSceneFixtures();

        const hoverState = resolveMhsaDetailHoverState(index, {
            node: maskNode,
            cellHit: {
                rowIndex: 0,
                colIndex: 1,
                cellItem: maskNode.rowItems[0]?.cells?.[1]
            }
        });

        expect(hoverState?.label).toBe('Causal Mask');
        expect(hoverState?.info?.activationData?.label).toBe('Causal Mask');
        expect(hoverState?.info?.activationData?.stage).toBe('attention.mask');
        expect(hoverState?.info?.activationData?.preScore).toBe(0.25);
        expect(hoverState?.info?.activationData?.postScore).toBe(0);
        expect(hoverState?.info?.activationData?.maskValue).toBe(Number.NEGATIVE_INFINITY);
        expect(hoverState?.info?.activationData?.showMaskValue).toBe(true);
        expect(hoverState?.focusState?.rowSelections).toContainEqual({
            nodeId: headOutputNode.id,
            rowIndex: 0
        });
        expect(hoverState?.focusState?.activeNodeIds).toContain(headOutputNode.id);
        expect(hoverState?.focusState?.activeNodeIds).toContain(postNode?.id);
        expect(hoverState?.focusState?.activeNodeIds).toContain(postCopyNode?.id);
        expect(hoverState?.focusState?.activeNodeIds).not.toContain(valuePostNode.id);
        expect(hoverState?.focusState?.activeConnectorIds).not.toContain(connectorVNode.id);
        expect(
            hoverState?.focusState?.columnSelections?.some((selection) => selection.nodeId === postNode?.id)
        ).toBe(true);
        expect(hoverState?.focusState?.cellSelections).toContainEqual({
            nodeId: postNode?.id,
            rowIndex: 0,
            colIndex: 1
        });
        expect(hoverState?.focusState?.cellSelections).toContainEqual({
            nodeId: postCopyNode?.id,
            rowIndex: 0,
            colIndex: 1
        });
    });

    it('maps masked post-score hover cells to a post-softmax attention-score tooltip payload with zero post-softmax weight', () => {
        const {
            index,
            projectionSourceNode,
            postNode,
            postCopyNode,
            headOutputNode,
            valuePostNode,
            connectorVNode
        } = buildSceneFixtures();

        const hoverState = resolveMhsaDetailHoverState(index, {
            node: postNode,
            cellHit: {
                rowIndex: 0,
                colIndex: 1,
                cellItem: postNode.rowItems[0]?.cells?.[1]
            }
        });

        expect(hoverState?.label).toBe('Post-Softmax Attention Score');
        expect(hoverState?.info?.activationData?.label).toBe('Post-Softmax Attention Score');
        expect(hoverState?.info?.activationData?.stage).toBe('attention.post');
        expect(hoverState?.info?.activationData?.preScore).toBe(0.25);
        expect(hoverState?.info?.activationData?.postScore).toBe(0);
        expect(hoverState?.info?.activationData?.maskValue).toBe(Number.NEGATIVE_INFINITY);
        expect(hoverState?.info?.activationData?.showMaskValue).toBe(true);
        expect(hoverState?.focusState?.rowSelections).toContainEqual({
            nodeId: projectionSourceNode.id,
            rowIndex: 0
        });
        expect(hoverState?.focusState?.rowSelections).toContainEqual({
            nodeId: projectionSourceNode.id,
            rowIndex: 1
        });
        expect(hoverState?.focusState?.rowSelections).toContainEqual({
            nodeId: headOutputNode.id,
            rowIndex: 0
        });
        expect(hoverState?.focusState?.activeNodeIds).toContain(headOutputNode.id);
        expect(hoverState?.focusState?.activeNodeIds).not.toContain(valuePostNode.id);
        expect(hoverState?.focusState?.activeConnectorIds).not.toContain(connectorVNode.id);
        expect(hoverState?.focusState?.cellSelections).toContainEqual({
            nodeId: postCopyNode?.id,
            rowIndex: 0,
            colIndex: 1
        });
    });

    it('backpropagates clicked decode score cells on cached key columns into the cached K path', () => {
        const {
            index,
            projectionSourceNode,
            keyInputNode,
            keyWeightNode,
            keyBiasNode,
            keyProjectionOutputNode,
            keyProjectionOutputCopyNode,
            keyCacheNode,
            keyCacheSourceNode,
            keyCacheConcatResultNode,
            preScoreNode,
            connectorKCacheSourceNode,
            connectorKCacheCopyNode,
            connectorKCacheNextNode
        } = buildSceneFixtures(4, {
            kvCacheState: {
                kvCacheModeEnabled: true,
                kvCachePrefillActive: false,
                kvCacheDecodeActive: true,
                kvCachePassIndex: 1
            }
        });

        const hoverState = resolveMhsaDetailHoverState(index, {
            node: preScoreNode,
            cellHit: {
                rowIndex: 0,
                colIndex: 1,
                cellItem: preScoreNode.rowItems[0]?.cells?.[1]
            }
        }, {
            interactionKind: 'click'
        });

        expect(hoverState?.focusState?.rowSelections).toContainEqual({
            nodeId: projectionSourceNode.id,
            rowIndex: 0
        });
        expect(
            hoverState?.focusState?.rowSelections?.some((selection) => (
                selection.nodeId === projectionSourceNode.id && selection.rowIndex === 1
            ))
        ).toBe(false);
        expect(hoverState?.focusState?.rowSelections).toContainEqual({
            nodeId: keyCacheNode?.id,
            rowIndex: 1
        });
        expect(hoverState?.focusState?.rowSelections).toContainEqual({
            nodeId: keyCacheSourceNode?.id,
            rowIndex: 1
        });
        expect(hoverState?.focusState?.rowSelections).toContainEqual({
            nodeId: keyCacheConcatResultNode?.id,
            rowIndex: 1
        });
        expect(
            hoverState?.focusState?.rowSelections?.some((selection) => selection.nodeId === keyProjectionOutputNode?.id)
        ).toBe(false);
        expect(
            hoverState?.focusState?.rowSelections?.some((selection) => selection.nodeId === keyProjectionOutputCopyNode?.id)
        ).toBe(false);
        expect(
            hoverState?.focusState?.rowSelections?.some((selection) => selection.nodeId === keyInputNode?.id)
        ).toBe(false);
        expect(hoverState?.focusState?.activeNodeIds).not.toContain(keyInputNode?.id);
        expect(hoverState?.focusState?.activeNodeIds).not.toContain(keyWeightNode?.id);
        expect(hoverState?.focusState?.activeNodeIds).not.toContain(keyBiasNode?.id);
        expect(hoverState?.focusState?.activeNodeIds).not.toContain(keyProjectionOutputNode?.id);
        expect(hoverState?.focusState?.activeNodeIds).not.toContain(keyProjectionOutputCopyNode?.id);
        expect(hoverState?.focusState?.activeConnectorIds).toContain(connectorKCacheSourceNode?.id);
        expect(hoverState?.focusState?.activeConnectorIds).not.toContain(connectorKCacheCopyNode?.id);
        expect(hoverState?.focusState?.activeConnectorIds).not.toContain(connectorKCacheNextNode?.id);
    });

    it('backpropagates clicked decode score cells on the live key column into K_current and the live combined K path', () => {
        const {
            index,
            projectionSourceNode,
            keyInputNode,
            keyProjectionOutputNode,
            keyProjectionOutputCopyNode,
            keyCacheNode,
            keyCacheSourceNode,
            keyCacheConcatResultNode,
            keyCacheNextNode,
            postNode,
            connectorKCacheSourceNode,
            connectorKCacheCopyNode,
            connectorKCacheNextNode
        } = buildSceneFixtures(4, {
            kvCacheState: {
                kvCacheModeEnabled: true,
                kvCachePrefillActive: false,
                kvCacheDecodeActive: true,
                kvCachePassIndex: 1
            }
        });

        const hoverState = resolveMhsaDetailHoverState(index, {
            node: postNode,
            cellHit: {
                rowIndex: 0,
                colIndex: 3,
                cellItem: postNode.rowItems[0]?.cells?.[3]
            }
        }, {
            interactionKind: 'click'
        });

        expect(hoverState?.focusState?.rowSelections).toContainEqual({
            nodeId: projectionSourceNode.id,
            rowIndex: 0
        });
        expect(hoverState?.focusState?.rowSelections).toContainEqual({
            nodeId: keyInputNode?.id,
            rowIndex: 0
        });
        expect(hoverState?.focusState?.rowSelections).toContainEqual({
            nodeId: keyProjectionOutputNode?.id,
            rowIndex: 0
        });
        expect(hoverState?.focusState?.rowSelections).toContainEqual({
            nodeId: keyProjectionOutputCopyNode?.id,
            rowIndex: 0
        });
        expect(hoverState?.focusState?.rowSelections).toContainEqual({
            nodeId: keyCacheConcatResultNode?.id,
            rowIndex: 3
        });
        expect(hoverState?.focusState?.rowSelections).toContainEqual({
            nodeId: keyCacheNextNode?.id,
            rowIndex: 3
        });
        expect(
            hoverState?.focusState?.rowSelections?.some((selection) => (
                selection.nodeId === keyCacheNode?.id || selection.nodeId === keyCacheSourceNode?.id
            ))
        ).toBe(false);
        expect(hoverState?.focusState?.activeConnectorIds).not.toContain(connectorKCacheSourceNode?.id);
        expect(hoverState?.focusState?.activeConnectorIds).toContain(connectorKCacheCopyNode?.id);
        expect(hoverState?.focusState?.activeConnectorIds).toContain(connectorKCacheNextNode?.id);
    });

    it('maps shared X_ln source rows into the copied X_ln rows and copy connectors', () => {
        const {
            index,
            projectionSourceNode,
            projectionIngressConnectorNodes,
            queryProjectionOutputNode,
            keyProjectionOutputNode,
            valueProjectionOutputNode,
            transposeNode
        } = buildSceneFixtures();

        const hoverState = resolveMhsaDetailHoverState(index, {
            node: projectionSourceNode,
            rowHit: {
                rowIndex: 1,
                rowItem: projectionSourceNode.rowItems[1]
            }
        });

        expect(hoverState?.label).toBe('Post LayerNorm 1 Residual Vector');
        expect(hoverState?.focusState?.rowSelections).toContainEqual({
            nodeId: projectionSourceNode.id,
            rowIndex: 1
        });
        expect(hoverState?.focusState?.rowSelections).toContainEqual({
            nodeId: index?.projectionInputIdsByKind?.q,
            rowIndex: 1
        });
        expect(hoverState?.focusState?.rowSelections).toContainEqual({
            nodeId: index?.projectionInputIdsByKind?.k,
            rowIndex: 1
        });
        expect(hoverState?.focusState?.rowSelections).toContainEqual({
            nodeId: index?.projectionInputIdsByKind?.v,
            rowIndex: 1
        });
        expect(hoverState?.focusState?.rowSelections).toContainEqual({
            nodeId: queryProjectionOutputNode.id,
            rowIndex: 1
        });
        expect(hoverState?.focusState?.rowSelections).toContainEqual({
            nodeId: keyProjectionOutputNode.id,
            rowIndex: 1
        });
        expect(hoverState?.focusState?.rowSelections).toContainEqual({
            nodeId: valueProjectionOutputNode.id,
            rowIndex: 1
        });
        expect(hoverState?.focusState?.columnSelections || []).toHaveLength(0);
        expect(hoverState?.focusState?.activeConnectorIds).toHaveLength(3);
        projectionIngressConnectorNodes.forEach((connectorNode) => {
            expect(hoverState?.focusState?.activeConnectorIds).toContain(connectorNode.id);
        });
    });

    it('maps clicked decode shared X_ln source rows into the live K and V decode rows', () => {
        const {
            index,
            projectionSourceNode,
            keyProjectionOutputNode,
            keyProjectionOutputCopyNode,
            keyCacheConcatResultNode,
            keyCacheNextNode,
            valueProjectionOutputNode,
            valueProjectionOutputCopyNode,
            valueCacheConcatResultNode,
            valueCacheNextNode,
            valuePostNode,
            transposeNode
        } = buildSceneFixtures(4, {
            kvCacheState: {
                kvCacheModeEnabled: true,
                kvCachePrefillActive: false,
                kvCacheDecodeActive: true,
                kvCachePassIndex: 1
            }
        });

        const hoverState = resolveMhsaDetailHoverState(index, {
            node: projectionSourceNode,
            rowHit: {
                rowIndex: 0,
                rowItem: projectionSourceNode.rowItems[0]
            }
        }, {
            interactionKind: 'click'
        });

        expect(hoverState?.focusState?.rowSelections).toContainEqual({
            nodeId: projectionSourceNode.id,
            rowIndex: 0
        });
        expect(hoverState?.focusState?.rowSelections).toContainEqual({
            nodeId: keyProjectionOutputNode?.id,
            rowIndex: 0
        });
        expect(hoverState?.focusState?.rowSelections).toContainEqual({
            nodeId: keyProjectionOutputCopyNode?.id,
            rowIndex: 0
        });
        expect(hoverState?.focusState?.rowSelections).toContainEqual({
            nodeId: keyCacheConcatResultNode?.id,
            rowIndex: 3
        });
        expect(hoverState?.focusState?.rowSelections).toContainEqual({
            nodeId: keyCacheNextNode?.id,
            rowIndex: 3
        });
        expect(hoverState?.focusState?.rowSelections).toContainEqual({
            nodeId: valueProjectionOutputNode?.id,
            rowIndex: 0
        });
        expect(hoverState?.focusState?.rowSelections).toContainEqual({
            nodeId: valueProjectionOutputCopyNode?.id,
            rowIndex: 0
        });
        expect(hoverState?.focusState?.rowSelections).toContainEqual({
            nodeId: valueCacheConcatResultNode?.id,
            rowIndex: 3
        });
        expect(hoverState?.focusState?.rowSelections).toContainEqual({
            nodeId: valueCacheNextNode?.id,
            rowIndex: 3
        });
        expect(hoverState?.focusState?.rowSelections).toContainEqual({
            nodeId: valuePostNode?.id,
            rowIndex: 3
        });
        expect(hoverState?.focusState?.columnSelections).toContainEqual({
            nodeId: transposeNode.id,
            colIndex: 3
        });
    });

    it('treats copied X_ln rows as post-layernorm residual rows for hover labels', () => {
        const {
            index,
            projectionSourceNode,
            queryInputNode,
            preScoreNode,
            maskedInputNode,
            maskNode,
            postNode,
            postCopyNode,
            headOutputNode,
            transposeNode
        } = buildSceneFixtures();

        const hoverState = resolveMhsaDetailHoverState(index, {
            node: queryInputNode,
            rowHit: {
                rowIndex: 0,
                rowItem: queryInputNode.rowItems[0]
            }
        });

        expect(hoverState?.label).toBe('Post LayerNorm 1 Residual Vector');
        expect(hoverState?.info?.activationData?.label).toBe('Post LayerNorm 1 Residual Vector');
        expect(hoverState?.info?.activationData?.stage).toBe('ln1.output');
        expect(hoverState?.info?.activationData?.sourceStage).toBe('ln1.shift');
        expect(hoverState?.info?.activationData?.tokenIndex).toBe(0);
        expect(hoverState?.focusState?.rowSelections).toContainEqual({
            nodeId: queryInputNode.id,
            rowIndex: 0
        });
        expect(hoverState?.focusState?.rowSelections).toContainEqual({
            nodeId: projectionSourceNode.id,
            rowIndex: 0
        });
        expect(hoverState?.focusState?.rowSelections).toContainEqual({
            nodeId: preScoreNode.id,
            rowIndex: 0
        });
        expect(hoverState?.focusState?.rowSelections).toContainEqual({
            nodeId: maskedInputNode.id,
            rowIndex: 0
        });
        expect(hoverState?.focusState?.rowSelections).toContainEqual({
            nodeId: maskNode.id,
            rowIndex: 0
        });
        expect(hoverState?.focusState?.rowSelections).toContainEqual({
            nodeId: postNode.id,
            rowIndex: 0
        });
        expect(hoverState?.focusState?.rowSelections).toContainEqual({
            nodeId: postCopyNode.id,
            rowIndex: 0
        });
        expect(hoverState?.focusState?.rowSelections).toContainEqual({
            nodeId: headOutputNode.id,
            rowIndex: 0
        });
        expect(hoverState?.focusState?.activeNodeIds).toContain(headOutputNode.id);
        expect(hoverState?.focusState?.activeNodeIds).not.toContain(transposeNode.id);
    });

    it('mirrors copied K/V X_ln row hovers back to the shared parent X_ln row', () => {
        const {
            index,
            projectionSourceNode,
            keyInputNode,
            valueInputNode
        } = buildSceneFixtures();

        [keyInputNode, valueInputNode].forEach((node) => {
            const hoverState = resolveMhsaDetailHoverState(index, {
                node,
                rowHit: {
                    rowIndex: 1,
                    rowItem: node?.rowItems?.[1]
                }
            });

            expect(hoverState?.focusState?.rowSelections).toContainEqual({
                nodeId: node.id,
                rowIndex: 1
            });
            expect(hoverState?.focusState?.rowSelections).toContainEqual({
                nodeId: projectionSourceNode.id,
                rowIndex: 1
            });
            if (node === keyInputNode) {
                expect(hoverState?.focusState?.columnSelections || []).toHaveLength(0);
            }
        });
    });

    it('keeps the softmax closing parenthesis close to the trailing equals sign', () => {
        const scene = buildMhsaSceneModel({
            previewData: createPreviewData(6),
            layerIndex: 2,
            headIndex: 1
        });
        const layout = buildSceneLayout(scene);
        const nodes = flattenSceneNodes(scene);

        const closeEqualsGroupNode = nodes.find((node) => node.role === 'attention-softmax-close-equals-group') || null;
        const softmaxCloseNode = nodes.find((node) => node.role === 'attention-softmax-close') || null;
        const softmaxEqualsNode = nodes.find((node) => node.role === 'attention-softmax-equals') || null;
        const postNode = nodes.find((node) => node.role === 'attention-post') || null;

        const closeEntry = layout?.registry?.getNodeEntry(softmaxCloseNode?.id || '');
        const equalsEntry = layout?.registry?.getNodeEntry(softmaxEqualsNode?.id || '');
        const postEntry = layout?.registry?.getNodeEntry(postNode?.id || '');

        const closeEqualsGap = Math.round(
            (equalsEntry?.bounds?.x || 0) - ((closeEntry?.bounds?.x || 0) + (closeEntry?.bounds?.width || 0))
        );
        const equalsPostGap = Math.round(
            (postEntry?.bounds?.x || 0) - ((equalsEntry?.bounds?.x || 0) + (equalsEntry?.bounds?.width || 0))
        );

        expect(closeEqualsGroupNode?.metadata?.gapOverride).toBe(4);
        expect(closeEqualsGap).toBe(4);
        expect(equalsPostGap).toBeGreaterThan(closeEqualsGap);
    });

    it('increases the projection stack gap as more tokens are shown', () => {
        const smallScene = buildMhsaSceneModel({
            previewData: createPreviewData(2),
            layerIndex: 2,
            headIndex: 1
        });
        const largeScene = buildMhsaSceneModel({
            previewData: createPreviewData(8),
            layerIndex: 2,
            headIndex: 1
        });

        const smallProjectionStack = flattenSceneNodes(smallScene).find((node) => node.role === 'projection-stack');
        const largeProjectionStack = flattenSceneNodes(largeScene).find((node) => node.role === 'projection-stack');

        expect(smallProjectionStack?.metadata?.gapOverride).toBe(188);
        expect(largeProjectionStack?.metadata?.gapOverride).toBeGreaterThan(smallProjectionStack?.metadata?.gapOverride || 0);
        expect(largeProjectionStack?.metadata?.gapOverride).toBe(218);
    });

    it('shrinks detailed attention grid cells for larger token windows', () => {
        const smallScene = buildMhsaSceneModel({
            previewData: createPreviewData(2),
            layerIndex: 2,
            headIndex: 1
        });
        const largeScene = buildMhsaSceneModel({
            previewData: createPreviewData(25),
            layerIndex: 2,
            headIndex: 1
        });
        const smallLayout = buildSceneLayout(smallScene);
        const largeLayout = buildSceneLayout(largeScene);
        const findNodeByRole = (scene, role) => flattenSceneNodes(scene).find((node) => node.role === role) || null;

        const smallPreScoreNode = findNodeByRole(smallScene, 'attention-pre-score');
        const largePreScoreNode = findNodeByRole(largeScene, 'attention-pre-score');
        const smallPreScoreEntry = smallLayout?.registry?.getNodeEntry(smallPreScoreNode?.id || '');
        const largePreScoreEntry = largeLayout?.registry?.getNodeEntry(largePreScoreNode?.id || '');

        expect(smallPreScoreEntry?.layoutData?.cellSize).toBeLessThan(10);
        expect(largePreScoreEntry?.layoutData?.cellSize).toBeLessThanOrEqual(
            smallPreScoreEntry?.layoutData?.cellSize || 0
        );
        expect(largePreScoreEntry?.layoutData?.cellGap).toBeLessThanOrEqual(
            smallPreScoreEntry?.layoutData?.cellGap || 0
        );
    });

    it('keeps token-count attention grids more readable than the feature-dimension strip rows', () => {
        const mediumScene = buildMhsaSceneModel({
            previewData: createPreviewData(12),
            layerIndex: 2,
            headIndex: 1
        });
        const largeScene = buildMhsaSceneModel({
            previewData: createPreviewData(25),
            layerIndex: 2,
            headIndex: 1
        });
        const mediumLayout = buildSceneLayout(mediumScene);
        const largeLayout = buildSceneLayout(largeScene);

        const findProjectionOutput = (scene, kind) => flattenSceneNodes(scene).find((node) => (
            node.role === 'projection-output'
            && String(node.metadata?.kind || '').toLowerCase() === kind
        )) || null;
        const findNodeByRole = (scene, role) => flattenSceneNodes(scene).find((node) => node.role === role) || null;

        const mediumQueryNode = findNodeByRole(mediumScene, 'attention-query-source');
        const largeQueryNode = findNodeByRole(largeScene, 'attention-query-source');
        const largeXlnNode = flattenSceneNodes(largeScene).find((node) => (
            node.role === 'x-ln-copy'
            && String(node.semantic?.branchKey || '').toLowerCase() === 'q'
        )) || null;
        const mediumPreScoreNode = findNodeByRole(mediumScene, 'attention-pre-score');
        const largePreScoreNode = findNodeByRole(largeScene, 'attention-pre-score');
        const largeProjectionOutputNode = findProjectionOutput(largeScene, 'q');

        const mediumPreScoreEntry = mediumLayout?.registry?.getNodeEntry(mediumPreScoreNode?.id || '');
        const largePreScoreEntry = largeLayout?.registry?.getNodeEntry(largePreScoreNode?.id || '');

        expect(largeQueryNode?.metadata?.compactRows?.rowHeight).toBeLessThan(
            mediumQueryNode?.metadata?.compactRows?.rowHeight || 0
        );
        expect(largeQueryNode?.metadata?.compactRows?.compactWidth).toBe(
            mediumQueryNode?.metadata?.compactRows?.compactWidth
        );
        expect(largeProjectionOutputNode?.metadata?.compactRows?.rowHeight).toBe(
            largeQueryNode?.metadata?.compactRows?.rowHeight
        );
        expect(largeProjectionOutputNode?.metadata?.compactRows?.compactWidth).toBe(
            largeQueryNode?.metadata?.compactRows?.compactWidth
        );
        expect(mediumPreScoreEntry?.layoutData?.cellSize).toBeGreaterThanOrEqual(5);
        expect(mediumPreScoreEntry?.layoutData?.cellSize).toBeGreaterThan(
            mediumQueryNode?.metadata?.compactRows?.rowHeight || 0
        );
        expect(largeXlnNode?.metadata?.compactRows?.compactWidth).toBeGreaterThan(
            largeQueryNode?.metadata?.compactRows?.compactWidth || 0
        );
        expect(largePreScoreEntry?.layoutData?.cellSize).toBeGreaterThan(
            largeQueryNode?.metadata?.compactRows?.rowHeight || 0
        );
        expect(largePreScoreEntry?.contentBounds?.width).toBeGreaterThanOrEqual(
            mediumPreScoreEntry?.contentBounds?.width || 0
        );
        expect(largePreScoreEntry?.layoutData?.cellSize).toBeLessThan(
            mediumPreScoreEntry?.layoutData?.cellSize || 0
        );
    });

    it('scales attention caption bounds with larger token windows and routes connectors from those bounds', () => {
        const baseScene = buildMhsaSceneModel({
            previewData: createPreviewData(TOKEN_LABELS.length),
            layerIndex: 2,
            headIndex: 1
        });
        const baseLayout = buildSceneLayout(baseScene);
        const baseNodes = flattenSceneNodes(baseScene);
        const largeScene = buildMhsaSceneModel({
            previewData: createPreviewData(12),
            layerIndex: 2,
            headIndex: 1
        });
        const largeLayout = buildSceneLayout(largeScene);
        const nodes = flattenSceneNodes(largeScene);

        const basePreScoreNode = baseNodes.find((node) => node.role === 'attention-pre-score') || null;
        const baseValuePostNode = baseNodes.find((node) => node.role === 'attention-value-post') || null;

        const queryNode = nodes.find((node) => node.role === 'attention-query-source') || null;
        const transposeNode = nodes.find((node) => node.role === 'attention-key-transpose') || null;
        const preScoreNode = nodes.find((node) => node.role === 'attention-pre-score') || null;
        const maskedInputNode = nodes.find((node) => node.role === 'attention-masked-input') || null;
        const valueProjectionOutputNode = nodes.find((node) => (
            node.role === 'projection-output'
            && String(node.metadata?.kind || '').toLowerCase() === 'v'
        )) || null;
        const valuePostNode = nodes.find((node) => node.role === 'attention-value-post') || null;
        const connectorKNode = nodes.find((node) => node.role === 'connector-k') || null;
        const connectorPreNode = nodes.find((node) => node.role === 'connector-pre') || null;
        const connectorVNode = nodes.find((node) => node.role === 'connector-v') || null;

        const queryEntry = largeLayout?.registry?.getNodeEntry(queryNode?.id || '');
        const transposeEntry = largeLayout?.registry?.getNodeEntry(transposeNode?.id || '');
        const preScoreEntry = largeLayout?.registry?.getNodeEntry(preScoreNode?.id || '');
        const basePreScoreEntry = baseLayout?.registry?.getNodeEntry(basePreScoreNode?.id || '');
        const maskedInputEntry = largeLayout?.registry?.getNodeEntry(maskedInputNode?.id || '');
        const valueProjectionEntry = largeLayout?.registry?.getNodeEntry(valueProjectionOutputNode?.id || '');
        const valuePostEntry = largeLayout?.registry?.getNodeEntry(valuePostNode?.id || '');
        const baseValuePostEntry = baseLayout?.registry?.getNodeEntry(baseValuePostNode?.id || '');
        const connectorKEntry = largeLayout?.registry?.getConnectorEntry(connectorKNode?.id || '');
        const connectorPreEntry = largeLayout?.registry?.getConnectorEntry(connectorPreNode?.id || '');
        const connectorVEntry = largeLayout?.registry?.getConnectorEntry(connectorVNode?.id || '');
        expect(preScoreEntry?.labelBounds?.height).toBeGreaterThan(basePreScoreEntry?.labelBounds?.height || 0);
        expect(preScoreEntry?.dimensionBounds?.height).toBeGreaterThan(basePreScoreEntry?.dimensionBounds?.height || 0);
        expect(valuePostEntry?.labelBounds?.height).toBeGreaterThan(baseValuePostEntry?.labelBounds?.height || 0);
        expect(resolveEntryCaptionBottom(valuePostEntry)).toBeGreaterThan(resolveEntryCaptionBottom(baseValuePostEntry));
        expect(connectorKEntry?.pathPoints?.[2]?.y).toBeCloseTo(
            resolveEntryCaptionBottom(transposeEntry) + CONNECTOR_CAPTION_EXIT_GAP,
            4
        );
        expect(connectorPreEntry?.pathPoints?.[0]?.x).toBeCloseTo(preScoreEntry?.anchors?.bottom?.x || 0, 4);
        expect(connectorPreEntry?.pathPoints?.[0]?.y).toBeCloseTo(
            resolveEntryCaptionBottom(preScoreEntry) + CONNECTOR_CAPTION_EXIT_GAP + PRE_CONNECTOR_SOURCE_OFFSET_Y,
            4
        );
        expect(connectorPreEntry?.pathPoints?.[connectorPreEntry.pathPoints.length - 1]?.y).toBeLessThanOrEqual(
            maskedInputEntry?.contentBounds?.y || 0
        );
        expect(connectorVEntry?.pathPoints?.[0]?.x).toBeCloseTo(
            (valueProjectionEntry?.anchors?.right?.x || 0) + VALUE_CONNECTOR_SOURCE_GAP,
            4
        );
        expect(connectorVEntry?.pathPoints?.[0]?.y).toBeCloseTo(
            (valueProjectionEntry?.anchors?.right?.y || 0) + VALUE_CONNECTOR_SOURCE_OFFSET_Y,
            4
        );
        expect(connectorVEntry?.pathPoints?.[1]?.y).toBeCloseTo(
            connectorVEntry?.pathPoints?.[0]?.y || 0,
            4
        );
        expect(connectorVEntry?.pathPoints?.[1]?.x).toBeCloseTo(
            connectorVEntry?.pathPoints?.[2]?.x || 0,
            4
        );
        expect(Math.abs(
            (connectorVEntry?.pathPoints?.[2]?.x || 0) - (valuePostEntry?.anchors?.bottom?.x || 0)
        )).toBeLessThanOrEqual(4);
        expect(connectorVEntry?.pathPoints?.[2]?.y).toBeLessThan(
            connectorVEntry?.pathPoints?.[1]?.y || 0
        );
        expect(connectorVEntry?.pathPoints?.[2]?.y).toBeCloseTo(
            resolveEntryCaptionBottom(valuePostEntry) + VALUE_CONNECTOR_TARGET_GAP,
            4
        );
    });

    it('disables card surface effects on the W_Q / W_K / W_V projection weights', () => {
        const scene = buildMhsaSceneModel({
            previewData: createPreviewData(12),
            layerIndex: 2,
            headIndex: 1
        });
        const projectionWeightNodes = flattenSceneNodes(scene).filter((node) => node.role === 'projection-weight');

        expect(projectionWeightNodes).toHaveLength(3);
        projectionWeightNodes.forEach((node) => {
            expect(node.visual?.disableCardSurfaceEffects).toBe(true);
        });
    });
});
