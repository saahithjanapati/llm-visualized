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
const PRE_CONNECTOR_SOURCE_OFFSET_Y = 16;
const VALUE_CONNECTOR_SOURCE_OFFSET_Y = 14;
const VALUE_CONNECTOR_SOURCE_GAP = 8;
const VALUE_CONNECTOR_TARGET_GAP = 8;

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
    isSmallScreen = false
} = {}) {
    const previewData = createPreviewData(tokenCount);
    const scene = buildMhsaSceneModel({
        previewData,
        layerIndex: 2,
        headIndex: 1,
        isSmallScreen
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
    const keyProjectionOutputNode = nodes.find((node) => (
        node.role === 'projection-output'
        && String(node.metadata?.kind || '').toLowerCase() === 'k'
    )) || null;
    const qktEquationNode = nodes.find((node) => node.role === 'attention-qkt-equation') || null;
    const connectorKNode = nodes.find((node) => node.role === 'connector-k') || null;
    const connectorQNode = nodes.find((node) => node.role === 'connector-q') || null;
    const connectorPreNode = nodes.find((node) => node.role === 'connector-pre') || null;
    const connectorPostNode = nodes.find((node) => node.role === 'connector-post') || null;
    const connectorVNode = nodes.find((node) => node.role === 'connector-v') || null;
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
        qktEquationNode,
        connectorQNode,
        connectorKNode,
        connectorPreNode,
        connectorPostNode,
        connectorVNode,
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

    it('maps head-output hover rows to the Attention Weighted Sum tooltip payload', () => {
        const {
            index,
            headOutputNode
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
        expect(hoverState?.focusState?.activeNodeIds).not.toContain(postNode?.id);
        expect(hoverState?.focusState?.activeNodeIds).toContain(postCopyNode?.id);
        expect(hoverState?.focusState?.activeNodeIds).not.toContain(valuePostNode.id);
        expect(hoverState?.focusState?.activeConnectorIds).not.toContain(connectorVNode.id);
        expect(
            hoverState?.focusState?.columnSelections?.some((selection) => selection.nodeId === postNode?.id)
        ).toBe(false);
        expect(
            hoverState?.focusState?.cellSelections?.some((selection) => selection.nodeId === postNode?.id)
        ).toBe(false);
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

        expect(hoverState?.label).toBe('Post LayerNorm Residual Vector');
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
        expect(hoverState?.focusState?.columnSelections).toContainEqual({
            nodeId: transposeNode.id,
            colIndex: 1
        });
        expect(hoverState?.focusState?.activeConnectorIds).toHaveLength(3);
        projectionIngressConnectorNodes.forEach((connectorNode) => {
            expect(hoverState?.focusState?.activeConnectorIds).toContain(connectorNode.id);
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

        expect(hoverState?.label).toBe('Post LayerNorm Residual Vector');
        expect(hoverState?.info?.activationData?.label).toBe('Post LayerNorm Residual Vector');
        expect(hoverState?.info?.activationData?.stage).toBe('ln1.shift');
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
