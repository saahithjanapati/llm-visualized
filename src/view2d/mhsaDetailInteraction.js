import {
    flattenSceneNodes,
    VIEW2D_MATRIX_PRESENTATIONS,
    VIEW2D_NODE_KINDS
} from './schema/sceneTypes.js';
import {
    buildAttentionHoverInfo,
    normalizeAttentionHoverStageKey,
    resolveAttentionHoverLabel
} from '../ui/attentionHoverInfo.js';

const PROJECTION_KIND_LABELS = Object.freeze({
    q: 'Query',
    k: 'Key',
    v: 'Value'
});

const PRE_ATTENTION_ROLES = Object.freeze([
    'attention-open',
    'attention-query-source',
    'attention-multiply',
    'attention-key-transpose',
    'attention-close',
    'attention-divide',
    'attention-scale',
    'attention-equals'
]);

const SCORE_ATTENTION_ROLES = Object.freeze([
    'attention-pre-score',
    'attention-softmax-label',
    'attention-softmax-open',
    'attention-masked-input',
    'attention-softmax-plus',
    'attention-mask',
    'attention-softmax-close',
    'attention-softmax-equals',
    'attention-post'
]);

const WEIGHTED_OUTPUT_ROLES = Object.freeze([
    'attention-post-copy',
    'attention-head-output-multiply',
    'attention-value-post',
    'attention-head-output-equals',
    'attention-head-output'
]);

const MASK_STAGE_ATTENTION_ROLES = Object.freeze([
    'attention-pre-score',
    'attention-softmax-label',
    'attention-softmax-open',
    'attention-masked-input',
    'attention-softmax-plus',
    'attention-mask',
    'attention-softmax-close'
]);

function appendUnique(target, value) {
    if (typeof value !== 'string' || !value.length) return;
    if (!target.includes(value)) {
        target.push(value);
    }
}

function appendAllUnique(target, values = []) {
    values.forEach((value) => appendUnique(target, value));
}

function appendSelection(target, selection = null, key) {
    if (!selection || typeof selection !== 'object') return;
    const nodeId = typeof selection.nodeId === 'string' ? selection.nodeId : '';
    if (!nodeId.length) return;
    const indexValue = Number.isFinite(selection[key]) ? Math.max(0, Math.floor(selection[key])) : null;
    if (!Number.isFinite(indexValue)) return;
    const signature = `${nodeId}:${key}:${indexValue}`;
    if (target.some((entry) => `${entry.nodeId}:${key}:${entry[key]}` === signature)) {
        return;
    }
    target.push({
        nodeId,
        [key]: indexValue
    });
}

function appendCellSelection(target, selection = null) {
    if (!selection || typeof selection !== 'object') return;
    const nodeId = typeof selection.nodeId === 'string' ? selection.nodeId : '';
    const rowIndex = Number.isFinite(selection.rowIndex) ? Math.max(0, Math.floor(selection.rowIndex)) : null;
    const colIndex = Number.isFinite(selection.colIndex) ? Math.max(0, Math.floor(selection.colIndex)) : null;
    if (!nodeId.length || !Number.isFinite(rowIndex) || !Number.isFinite(colIndex)) return;
    const signature = `${nodeId}:${rowIndex}:${colIndex}`;
    if (target.some((entry) => `${entry.nodeId}:${entry.rowIndex}:${entry.colIndex}` === signature)) {
        return;
    }
    target.push({
        nodeId,
        rowIndex,
        colIndex
    });
}

function normalizeProjectionKind(value = '') {
    const safe = String(value || '').trim().toLowerCase();
    return safe === 'q' || safe === 'k' || safe === 'v' ? safe : '';
}

function resolveProjectionKindForNode(node = null) {
    const stageValue = String(node?.semantic?.stage || '').trim().toLowerCase();
    if (stageValue.startsWith('projection-')) {
        return normalizeProjectionKind(stageValue.slice('projection-'.length));
    }
    return normalizeProjectionKind(node?.metadata?.kind || '');
}

function resolveProjectionLabel(kind = '') {
    return PROJECTION_KIND_LABELS[normalizeProjectionKind(kind)] || 'Projection';
}

function normalizeSceneIndex(value = null) {
    return Number.isFinite(value) ? Math.max(0, Math.floor(value)) : null;
}

function buildProjectionHoverInfo(node = null, label = '', extraActivationData = {}) {
    const layerIndex = normalizeSceneIndex(node?.semantic?.layerIndex);
    const headIndex = normalizeSceneIndex(node?.semantic?.headIndex);
    const activationData = {
        label: String(label || '').trim(),
        ...(extraActivationData && typeof extraActivationData === 'object'
            ? extraActivationData
            : {})
    };

    if (Number.isFinite(layerIndex)) activationData.layerIndex = layerIndex;
    if (Number.isFinite(headIndex)) activationData.headIndex = headIndex;

    const info = {
        ...(Number.isFinite(layerIndex) ? { layerIndex } : {}),
        ...(Number.isFinite(headIndex) ? { headIndex } : {}),
        activationData
    };

    return Object.keys(info).length ? info : null;
}

function buildProjectionVectorHoverInfo(node = null, rowItem = null, kind = '') {
    const safeKind = normalizeProjectionKind(kind);
    if (!safeKind) return null;
    const label = `${resolveProjectionLabel(safeKind)} Vector`;
    const tokenInfo = createTokenInfo(rowItem) || {};
    const info = buildProjectionHoverInfo(node, label, {
        stage: `qkv.${safeKind}`,
        ...(Number.isFinite(tokenInfo.tokenIndex) ? { tokenIndex: tokenInfo.tokenIndex } : {}),
        ...(typeof tokenInfo.tokenLabel === 'string' && tokenInfo.tokenLabel.length
            ? { tokenLabel: tokenInfo.tokenLabel }
            : {})
    }) || {};

    return {
        ...tokenInfo,
        ...info,
        activationData: {
            ...(info.activationData || {}),
            ...(Number.isFinite(tokenInfo.tokenIndex) ? { tokenIndex: tokenInfo.tokenIndex } : {}),
            ...(typeof tokenInfo.tokenLabel === 'string' && tokenInfo.tokenLabel.length
                ? { tokenLabel: tokenInfo.tokenLabel }
                : {})
        }
    };
}

function buildPostLayerNormResidualHoverInfo(node = null, rowItem = null) {
    const tokenInfo = createTokenInfo(rowItem) || {};
    const info = buildProjectionHoverInfo(node, 'Post LayerNorm Residual Vector', {
        stage: 'ln1.shift',
        ...(Number.isFinite(tokenInfo.tokenIndex) ? { tokenIndex: tokenInfo.tokenIndex } : {}),
        ...(typeof tokenInfo.tokenLabel === 'string' && tokenInfo.tokenLabel.length
            ? { tokenLabel: tokenInfo.tokenLabel }
            : {})
    }) || {};

    return {
        ...tokenInfo,
        ...info,
        activationData: {
            ...(info.activationData || {}),
            stage: 'ln1.shift',
            ...(Number.isFinite(tokenInfo.tokenIndex) ? { tokenIndex: tokenInfo.tokenIndex } : {}),
            ...(typeof tokenInfo.tokenLabel === 'string' && tokenInfo.tokenLabel.length
                ? { tokenLabel: tokenInfo.tokenLabel }
                : {})
        }
    };
}

function buildProjectionColumnHoverInfo(node = null, columnItem = null, kind = '') {
    const safeKind = normalizeProjectionKind(kind);
    if (!safeKind) return null;
    const label = `${resolveProjectionLabel(safeKind)} Vector`;
    const tokenInfo = createTokenInfo(columnItem) || {};
    const info = buildProjectionHoverInfo(node, label, {
        stage: `qkv.${safeKind}`,
        ...(Number.isFinite(tokenInfo.tokenIndex) ? { tokenIndex: tokenInfo.tokenIndex } : {}),
        ...(typeof tokenInfo.tokenLabel === 'string' && tokenInfo.tokenLabel.length
            ? { tokenLabel: tokenInfo.tokenLabel }
            : {})
    }) || {};

    return {
        ...tokenInfo,
        ...info,
        activationData: {
            ...(info.activationData || {}),
            ...(Number.isFinite(tokenInfo.tokenIndex) ? { tokenIndex: tokenInfo.tokenIndex } : {}),
            ...(typeof tokenInfo.tokenLabel === 'string' && tokenInfo.tokenLabel.length
                ? { tokenLabel: tokenInfo.tokenLabel }
                : {})
        }
    };
}

function buildWeightedSumHoverInfo(node = null, rowItem = null) {
    const tokenInfo = createTokenInfo(rowItem) || {};
    const info = buildProjectionHoverInfo(node, 'Attention Weighted Sum', {
        stage: 'attention.weighted_sum',
        isWeightedSum: true,
        ...(Number.isFinite(tokenInfo.tokenIndex) ? { tokenIndex: tokenInfo.tokenIndex } : {}),
        ...(typeof tokenInfo.tokenLabel === 'string' && tokenInfo.tokenLabel.length
            ? { tokenLabel: tokenInfo.tokenLabel }
            : {})
    }) || {};

    return {
        ...tokenInfo,
        ...info,
        isWeightedSum: true,
        activationData: {
            ...(info.activationData || {}),
            stage: 'attention.weighted_sum',
            isWeightedSum: true,
            ...(Number.isFinite(tokenInfo.tokenIndex) ? { tokenIndex: tokenInfo.tokenIndex } : {}),
            ...(typeof tokenInfo.tokenLabel === 'string' && tokenInfo.tokenLabel.length
                ? { tokenLabel: tokenInfo.tokenLabel }
                : {})
        }
    };
}

function collectLeafDescendantIds(node = null, target = []) {
    if (!node || typeof node !== 'object') return target;
    if (!Array.isArray(node.children) || !node.children.length) {
        if (typeof node.id === 'string' && node.id.length && node.kind !== VIEW2D_NODE_KINDS.CONNECTOR) {
            appendUnique(target, node.id);
        }
        return target;
    }
    node.children.forEach((child) => {
        collectLeafDescendantIds(child, target);
    });
    return target;
}

function buildFocusSignature({
    activeNodeIds = [],
    activeConnectorIds = [],
    rowSelections = [],
    columnSelections = [],
    cellSelections = []
} = {}) {
    return JSON.stringify({
        activeNodeIds: [...activeNodeIds].sort(),
        activeConnectorIds: [...activeConnectorIds].sort(),
        rowSelections: [...rowSelections]
            .map((entry) => `${entry.nodeId}:${entry.rowIndex}`)
            .sort(),
        columnSelections: [...columnSelections]
            .map((entry) => `${entry.nodeId}:${entry.colIndex}`)
            .sort(),
        cellSelections: [...cellSelections]
            .map((entry) => `${entry.nodeId}:${entry.rowIndex}:${entry.colIndex}`)
            .sort()
    });
}

function buildFocusResult({
    label = '',
    info = null,
    activeNodeIds = [],
    activeConnectorIds = [],
    rowSelections = [],
    columnSelections = [],
    cellSelections = []
} = {}) {
    const normalizedNodeIds = [];
    const normalizedConnectorIds = [];
    const normalizedRowSelections = [];
    const normalizedColumnSelections = [];
    const normalizedCellSelections = [];

    appendAllUnique(normalizedNodeIds, activeNodeIds);
    appendAllUnique(normalizedConnectorIds, activeConnectorIds);
    rowSelections.forEach((selection) => appendSelection(normalizedRowSelections, selection, 'rowIndex'));
    columnSelections.forEach((selection) => appendSelection(normalizedColumnSelections, selection, 'colIndex'));
    cellSelections.forEach((selection) => appendCellSelection(normalizedCellSelections, selection));

    normalizedRowSelections.forEach((selection) => appendUnique(normalizedNodeIds, selection.nodeId));
    normalizedColumnSelections.forEach((selection) => appendUnique(normalizedNodeIds, selection.nodeId));
    normalizedCellSelections.forEach((selection) => appendUnique(normalizedNodeIds, selection.nodeId));

    const focusState = {
        activeNodeIds: normalizedNodeIds,
        activeConnectorIds: normalizedConnectorIds,
        rowSelections: normalizedRowSelections,
        columnSelections: normalizedColumnSelections,
        cellSelections: normalizedCellSelections
    };

    return {
        label: String(label || '').trim(),
        info,
        focusState,
        signature: buildFocusSignature(focusState)
    };
}

function resolveAttentionRoleIds(index = null, {
    query = false,
    transpose = false,
    score = false,
    includeWeightedOutput = true
} = {}) {
    if (!index) return [];
    const roles = [];
    if (query) {
        roles.push('attention-query-source');
    }
    if (transpose) {
        roles.push('attention-key-transpose');
    }
    if (query || transpose || score) {
        roles.push(...PRE_ATTENTION_ROLES);
    }
    if (score) {
        roles.push(...SCORE_ATTENTION_ROLES);
        if (includeWeightedOutput) {
            roles.push(...WEIGHTED_OUTPUT_ROLES);
        }
    }

    const nodeIds = [];
    roles.forEach((role) => {
        appendAllUnique(nodeIds, index.nodeIdsByRole.get(role) || []);
    });
    return nodeIds;
}

function resolveAttentionMaskStageRoleIds(index = null, {
    query = false,
    transpose = false
} = {}) {
    if (!index) return [];
    const nodeIds = resolveAttentionRoleIds(index, {
        query,
        transpose,
        score: false
    });
    MASK_STAGE_ATTENTION_ROLES.forEach((role) => {
        appendAllUnique(nodeIds, index.nodeIdsByRole.get(role) || []);
    });
    return nodeIds;
}

function appendProjectionStagePath(activeNodeIds, index = null, kinds = []) {
    kinds.forEach((kind) => {
        appendAllUnique(activeNodeIds, index?.projectionLeafIdsByKind?.[kind] || []);
    });
}

function appendConnectorKinds(activeConnectorIds, index = null, kinds = []) {
    kinds.forEach((kind) => {
        appendAllUnique(activeConnectorIds, index?.connectorIdsByKind?.[kind] || []);
    });
}

function createTokenInfo(rowItem = null) {
    const tokenLabel = typeof rowItem?.label === 'string' && rowItem.label.length
        ? rowItem.label
        : (typeof rowItem?.semantic?.tokenLabel === 'string' ? rowItem.semantic.tokenLabel : '');
    const tokenIndex = Number.isFinite(rowItem?.semantic?.tokenIndex)
        ? Math.max(0, Math.floor(rowItem.semantic.tokenIndex))
        : null;
    if (!tokenLabel.length && !Number.isFinite(tokenIndex)) {
        return null;
    }
    return {
        ...(tokenLabel.length ? { tokenLabel } : {}),
        ...(Number.isFinite(tokenIndex) ? { tokenIndex } : {})
    };
}

function hasNumericValue(value) {
    return typeof value === 'number' && !Number.isNaN(value);
}

function buildAttentionCellHoverInfo(node = null, cellItem = null, stageKey = '') {
    if (!cellItem || typeof cellItem !== 'object') return null;
    const semantic = cellItem.semantic && typeof cellItem.semantic === 'object'
        ? cellItem.semantic
        : (node?.semantic || null);
    return buildAttentionHoverInfo({
        stageKey,
        layerIndex: semantic?.layerIndex,
        headIndex: semantic?.headIndex,
        queryTokenIndex: cellItem.queryTokenIndex,
        queryTokenLabel: cellItem.queryTokenLabel || cellItem.rowLabel || '',
        keyTokenIndex: cellItem.keyTokenIndex,
        keyTokenLabel: cellItem.keyTokenLabel || cellItem.colLabel || '',
        preScore: hasNumericValue(cellItem.preScore) ? cellItem.preScore : null,
        postScore: hasNumericValue(cellItem.postScore) ? cellItem.postScore : null,
        maskValue: hasNumericValue(cellItem.maskValue) ? cellItem.maskValue : null,
        isMasked: cellItem.isMasked === true
    });
}

function resolveAttentionStageKeyForRole(role = '') {
    const safeRole = String(role || '').trim().toLowerCase();
    if (safeRole === 'attention-pre-score') return 'pre';
    if (safeRole === 'attention-masked-input') return 'masked-input';
    if (safeRole === 'attention-mask') return 'mask';
    if (safeRole === 'attention-post' || safeRole === 'attention-post-copy') return 'post';
    return '';
}

function buildAttentionStageRoleHoverInfo(node = null, stageKey = '') {
    const semantic = node?.semantic && typeof node.semantic === 'object'
        ? node.semantic
        : null;
    return buildAttentionHoverInfo({
        stageKey,
        layerIndex: semantic?.layerIndex,
        headIndex: semantic?.headIndex
    });
}

function resolveMatrixStageKey(node = null, hit = null) {
    const stage = String(hit?.cellItem?.semantic?.stage || node?.semantic?.stage || '').trim().toLowerCase();
    if (stage.includes('pre-score')) return 'pre-score';
    if (stage.includes('masked-input')) return 'masked-input';
    if (stage.includes('attention-mask')) return 'mask';
    if (stage.includes('attention-post-copy')) return 'post-copy';
    if (stage.includes('attention-post')) return 'post';
    return '';
}

function buildQueryRowSelections(index = null, rowIndex = null) {
    if (!Number.isFinite(rowIndex) || !index) return [];
    return [
        { nodeId: index.projectionInputIdsByKind.q, rowIndex },
        { nodeId: index.projectionOutputIdsByKind.q, rowIndex },
        { nodeId: index.singleNodeIds.attentionQuerySource, rowIndex },
        { nodeId: index.singleNodeIds.attentionHeadOutput, rowIndex }
    ];
}

function buildProjectionSourceRowSelections(index = null, rowIndex = null) {
    if (!Number.isFinite(rowIndex) || !index) return [];
    return [
        { nodeId: index.singleNodeIds.projectionSourceXln, rowIndex },
        { nodeId: index.projectionInputIdsByKind.q, rowIndex },
        { nodeId: index.projectionInputIdsByKind.k, rowIndex },
        { nodeId: index.projectionInputIdsByKind.v, rowIndex }
    ].filter((selection) => typeof selection.nodeId === 'string' && selection.nodeId.length);
}

function buildKeyRowSelections(index = null, rowIndex = null) {
    if (!Number.isFinite(rowIndex) || !index) return [];
    return [
        { nodeId: index.projectionInputIdsByKind.k, rowIndex },
        { nodeId: index.projectionOutputIdsByKind.k, rowIndex }
    ];
}

function buildValueRowSelections(index = null, rowIndex = null) {
    if (!Number.isFinite(rowIndex) || !index) return [];
    return [
        { nodeId: index.projectionInputIdsByKind.v, rowIndex },
        { nodeId: index.projectionOutputIdsByKind.v, rowIndex },
        { nodeId: index.singleNodeIds.attentionValuePost, rowIndex },
        { nodeId: index.singleNodeIds.attentionHeadOutput, rowIndex }
    ];
}

function usesRowSelectionsForAttentionKeyTranspose(index = null) {
    const transposeNodeId = index?.singleNodeIds?.attentionKeyTranspose || '';
    if (!transposeNodeId) return false;
    const transposeNode = index?.nodesById?.get(transposeNodeId) || null;
    return transposeNode?.presentation === VIEW2D_MATRIX_PRESENTATIONS.COMPACT_ROWS;
}

function buildAttentionKeyTransposeSelections(index = null, axisIndex = null) {
    if (!Number.isFinite(axisIndex) || !index) {
        return {
            rowSelections: [],
            columnSelections: []
        };
    }
    const transposeNodeId = index?.singleNodeIds?.attentionKeyTranspose || '';
    if (!transposeNodeId) {
        return {
            rowSelections: [],
            columnSelections: []
        };
    }
    if (usesRowSelectionsForAttentionKeyTranspose(index)) {
        return {
            rowSelections: [{ nodeId: transposeNodeId, rowIndex: axisIndex }],
            columnSelections: []
        };
    }
    return {
        rowSelections: [],
        columnSelections: [{ nodeId: transposeNodeId, colIndex: axisIndex }]
    };
}

function buildScoreAxisSelections(index = null, axisIndex = null, {
    includePost = true,
    includePostCopy = true
} = {}) {
    if (!Number.isFinite(axisIndex) || !index) {
        return {
            rowSelections: [],
            columnSelections: []
        };
    }
    const transposeSelections = buildAttentionKeyTransposeSelections(index, axisIndex);
    return {
        rowSelections: transposeSelections.rowSelections,
        columnSelections: [
            ...transposeSelections.columnSelections,
            { nodeId: index.singleNodeIds.attentionPreScore, colIndex: axisIndex },
            { nodeId: index.singleNodeIds.attentionMaskedInput, colIndex: axisIndex },
            { nodeId: index.singleNodeIds.attentionMask, colIndex: axisIndex },
            ...(includePost
                ? [{ nodeId: index.singleNodeIds.attentionPost, colIndex: axisIndex }]
                : []),
            ...(includePostCopy
                ? [{ nodeId: index.singleNodeIds.attentionPostCopy, colIndex: axisIndex }]
                : [])
        ].filter((selection) => typeof selection.nodeId === 'string' && selection.nodeId.length)
    };
}

function buildScoreAxisRowSelections(index = null, rowIndex = null) {
    if (!Number.isFinite(rowIndex) || !index) return [];
    return [
        { nodeId: index.singleNodeIds.attentionPreScore, rowIndex },
        { nodeId: index.singleNodeIds.attentionMaskedInput, rowIndex },
        { nodeId: index.singleNodeIds.attentionMask, rowIndex },
        { nodeId: index.singleNodeIds.attentionPost, rowIndex },
        { nodeId: index.singleNodeIds.attentionPostCopy, rowIndex }
    ];
}

function buildPreScoreCellResult(index = null, node = null, rowIndex = null, colIndex = null, cellItem = null) {
    const activeNodeIds = [];
    const activeConnectorIds = [];
    const scoreAxisSelections = buildScoreAxisSelections(index, colIndex);
    const info = buildAttentionCellHoverInfo(node, cellItem, 'pre');
    const rowSelections = [
        ...buildQueryRowSelections(index, rowIndex),
        ...buildKeyRowSelections(index, colIndex),
        ...scoreAxisSelections.rowSelections
    ];
    const columnSelections = scoreAxisSelections.columnSelections;
    const cellSelections = [
        { nodeId: index?.singleNodeIds?.attentionPreScore, rowIndex, colIndex },
        { nodeId: index?.singleNodeIds?.attentionMaskedInput, rowIndex, colIndex }
    ];

    appendProjectionStagePath(activeNodeIds, index, ['q', 'k']);
    appendAllUnique(activeNodeIds, resolveAttentionRoleIds(index, {
        query: true,
        transpose: true,
        score: false
    }));
    appendAllUnique(activeNodeIds, [
        index?.singleNodeIds?.attentionPreScore,
        index?.singleNodeIds?.attentionMaskedInput
    ]);
    appendConnectorKinds(activeConnectorIds, index, ['q', 'k', 'pre']);

    return buildFocusResult({
        label: info?.activationData?.label || resolveAttentionHoverLabel('pre'),
        info,
        activeNodeIds,
        activeConnectorIds,
        rowSelections,
        columnSelections,
        cellSelections
    });
}

function buildSoftmaxCellResult(index = null, node = null, rowIndex = null, colIndex = null, cellItem = null, {
    stageKey = 'post',
    includeWeightedOutput = false,
    includePostCopy = false
} = {}) {
    const activeNodeIds = [];
    const activeConnectorIds = [];
    const normalizedStageKey = normalizeAttentionHoverStageKey(stageKey);
    const isMaskStage = normalizedStageKey === 'mask';
    const includePostSelections = !isMaskStage;
    const scoreAxisSelections = buildScoreAxisSelections(index, colIndex, {
        includePost: includePostSelections,
        includePostCopy: includePostSelections && includePostCopy
    });
    const info = buildAttentionCellHoverInfo(node, cellItem, normalizedStageKey);
    const rowSelections = [
        ...buildQueryRowSelections(index, rowIndex),
        ...buildKeyRowSelections(index, colIndex),
        ...scoreAxisSelections.rowSelections,
        ...(includeWeightedOutput ? buildValueRowSelections(index, rowIndex) : [])
    ];
    const columnSelections = scoreAxisSelections.columnSelections;
    const cellSelections = [
        { nodeId: index?.singleNodeIds?.attentionPreScore, rowIndex, colIndex },
        { nodeId: index?.singleNodeIds?.attentionMaskedInput, rowIndex, colIndex },
        { nodeId: index?.singleNodeIds?.attentionMask, rowIndex, colIndex },
        ...(includePostSelections
            ? [{ nodeId: index?.singleNodeIds?.attentionPost, rowIndex, colIndex }]
            : []),
        ...(includePostSelections && includePostCopy
            ? [{ nodeId: index?.singleNodeIds?.attentionPostCopy, rowIndex, colIndex }]
            : [])
    ];

    appendProjectionStagePath(activeNodeIds, index, includeWeightedOutput ? ['q', 'k', 'v'] : ['q', 'k']);
    appendAllUnique(activeNodeIds, isMaskStage
        ? resolveAttentionMaskStageRoleIds(index, {
            query: true,
            transpose: true
        })
        : resolveAttentionRoleIds(index, {
            query: true,
            transpose: true,
            score: true,
            includeWeightedOutput
        }));
    appendConnectorKinds(activeConnectorIds, index, isMaskStage
        ? ['q', 'k', 'pre']
        : (
            includeWeightedOutput
                ? ['q', 'k', 'pre', 'post', 'v']
                : ['q', 'k', 'pre', 'post']
        ));

    return buildFocusResult({
        label: info?.activationData?.label || resolveAttentionHoverLabel(normalizedStageKey),
        info,
        activeNodeIds,
        activeConnectorIds,
        rowSelections,
        columnSelections,
        cellSelections
    });
}

function buildProjectionRowResult(index = null, node = null, kind = '', rowHit = null, role = '') {
    const safeKind = normalizeProjectionKind(kind);
    if (!safeKind || !rowHit) return null;
    const rowIndex = Number.isFinite(rowHit.rowIndex) ? Math.max(0, Math.floor(rowHit.rowIndex)) : null;
    const activeNodeIds = [];
    const activeConnectorIds = [];
    const rowSelections = [];
    const columnSelections = [];

    appendProjectionStagePath(activeNodeIds, index, [safeKind]);

    if (safeKind === 'q') {
        appendAllUnique(activeNodeIds, resolveAttentionRoleIds(index, {
            query: true,
            transpose: false,
            score: false
        }));
        appendConnectorKinds(activeConnectorIds, index, ['q']);
        rowSelections.push(...buildQueryRowSelections(index, rowIndex));
        if (role === 'attention-query-source' || role === 'projection-output') {
            rowSelections.push(...buildScoreAxisRowSelections(index, rowIndex));
        }
    } else if (safeKind === 'k') {
        const scoreAxisSelections = buildScoreAxisSelections(index, rowIndex);
        appendAllUnique(activeNodeIds, resolveAttentionRoleIds(index, {
            query: false,
            transpose: true,
            score: false
        }));
        appendConnectorKinds(activeConnectorIds, index, ['k']);
        rowSelections.push(...buildKeyRowSelections(index, rowIndex), ...scoreAxisSelections.rowSelections);
        columnSelections.push(...scoreAxisSelections.columnSelections);
    } else if (safeKind === 'v') {
        appendAllUnique(activeNodeIds, [
            ...(index?.projectionLeafIdsByKind?.v || []),
            ...(index?.nodeIdsByRole?.get('attention-post-copy') || []),
            ...(index?.nodeIdsByRole?.get('attention-value-post') || []),
            ...(index?.nodeIdsByRole?.get('attention-head-output') || [])
        ]);
        appendConnectorKinds(activeConnectorIds, index, ['v']);
        rowSelections.push(...buildValueRowSelections(index, rowIndex));
    }

    const result = buildFocusResult({
        label: `${resolveProjectionLabel(safeKind)} row`,
        info: createTokenInfo(rowHit.rowItem),
        activeNodeIds,
        activeConnectorIds,
        rowSelections,
        columnSelections
    });
    if (role === 'x-ln-copy') {
        return {
            ...result,
            label: 'Post LayerNorm Residual Vector',
            info: buildPostLayerNormResidualHoverInfo(node, rowHit.rowItem)
        };
    }
    if (role === 'projection-output' || role === 'attention-query-source') {
        return {
            ...result,
            label: `${resolveProjectionLabel(safeKind)} Vector`,
            info: buildProjectionVectorHoverInfo(node, rowHit.rowItem, safeKind)
        };
    }
    return result;
}

function buildTransposeAxisResult(index = null, axisHit = null) {
    if (!index || !axisHit) return null;
    const axisIndex = Number.isFinite(axisHit.rowIndex)
        ? Math.max(0, Math.floor(axisHit.rowIndex))
        : (Number.isFinite(axisHit.colIndex) ? Math.max(0, Math.floor(axisHit.colIndex)) : null);
    const activeNodeIds = [];
    const activeConnectorIds = [];
    const scoreAxisSelections = buildScoreAxisSelections(index, axisIndex);
    const rowSelections = [
        ...buildKeyRowSelections(index, axisIndex),
        ...scoreAxisSelections.rowSelections
    ];
    const columnSelections = scoreAxisSelections.columnSelections;

    appendProjectionStagePath(activeNodeIds, index, ['k']);
    appendAllUnique(activeNodeIds, resolveAttentionRoleIds(index, {
        query: false,
        transpose: true,
        score: false
    }));
    appendConnectorKinds(activeConnectorIds, index, ['k']);

    return buildFocusResult({
        label: 'Key Vector',
        info: axisHit.rowItem
            ? buildProjectionVectorHoverInfo(
                index?.nodesById?.get(index?.singleNodeIds?.attentionKeyTranspose || ''),
                axisHit.rowItem,
                'k'
            )
            : buildProjectionColumnHoverInfo(
                index?.nodesById?.get(index?.singleNodeIds?.attentionKeyTranspose || ''),
                axisHit.columnItem,
                'k'
            ),
        activeNodeIds,
        activeConnectorIds,
        rowSelections,
        columnSelections
    });
}

function buildWeightedOutputRowResult(index = null, node = null, rowHit = null, {
    includeProjection = false,
    includeConnector = false,
    label = 'Head output row',
    info = null
} = {}) {
    if (!index || !rowHit) return null;
    const rowIndex = Number.isFinite(rowHit.rowIndex) ? Math.max(0, Math.floor(rowHit.rowIndex)) : null;
    const activeNodeIds = [];
    const activeConnectorIds = [];
    const rowSelections = [
        { nodeId: index.singleNodeIds.attentionValuePost, rowIndex },
        { nodeId: index.singleNodeIds.attentionHeadOutput, rowIndex }
    ];

    appendAllUnique(activeNodeIds, [
        ...(index?.nodeIdsByRole?.get('attention-post-copy') || []),
        ...(index?.nodeIdsByRole?.get('attention-value-post') || []),
        ...(index?.nodeIdsByRole?.get('attention-head-output') || [])
    ]);
    if (includeProjection) {
        appendProjectionStagePath(activeNodeIds, index, ['v']);
        rowSelections.push(...buildValueRowSelections(index, rowIndex));
    }
    if (includeConnector) {
        appendConnectorKinds(activeConnectorIds, index, ['v']);
    }

    return buildFocusResult({
        label,
        info: info || createTokenInfo(rowHit.rowItem),
        activeNodeIds,
        activeConnectorIds,
        rowSelections
    });
}

function buildProjectionSourceRowResult(index = null, rowHit = null) {
    if (!index || !rowHit) return null;
    const rowIndex = Number.isFinite(rowHit.rowIndex) ? Math.max(0, Math.floor(rowHit.rowIndex)) : null;
    const activeNodeIds = [
        index?.singleNodeIds?.projectionSourceXln,
        index?.projectionInputIdsByKind?.q,
        index?.projectionInputIdsByKind?.k,
        index?.projectionInputIdsByKind?.v
    ].filter((nodeId) => typeof nodeId === 'string' && nodeId.length);
    const activeConnectorIds = [
        ...(index?.projectionIngressConnectorIdsByKind?.q || []),
        ...(index?.projectionIngressConnectorIdsByKind?.k || []),
        ...(index?.projectionIngressConnectorIdsByKind?.v || [])
    ];

    return buildFocusResult({
        label: 'Post LayerNorm Residual Vector',
        info: buildPostLayerNormResidualHoverInfo(
            index?.nodesById?.get(index?.singleNodeIds?.projectionSourceXln || ''),
            rowHit.rowItem
        ),
        activeNodeIds,
        activeConnectorIds,
        rowSelections: buildProjectionSourceRowSelections(index, rowIndex)
    });
}

function buildProjectionStageResult(index = null, kind = '') {
    const safeKind = normalizeProjectionKind(kind);
    if (!index || !safeKind) return null;
    const activeNodeIds = [];
    const activeConnectorIds = [];

    if (safeKind === 'q') {
        appendProjectionStagePath(activeNodeIds, index, ['q']);
        appendAllUnique(activeNodeIds, resolveAttentionRoleIds(index, {
            query: true,
            transpose: false,
            score: false
        }));
        appendConnectorKinds(activeConnectorIds, index, ['q']);
    } else if (safeKind === 'k') {
        appendProjectionStagePath(activeNodeIds, index, ['k']);
        appendAllUnique(activeNodeIds, resolveAttentionRoleIds(index, {
            query: false,
            transpose: true,
            score: false
        }));
        appendConnectorKinds(activeConnectorIds, index, ['k']);
    } else if (safeKind === 'v') {
        appendProjectionStagePath(activeNodeIds, index, ['q', 'k', 'v']);
        appendAllUnique(activeNodeIds, resolveAttentionRoleIds(index, {
            query: true,
            transpose: true,
            score: true,
            includeWeightedOutput: true
        }));
        appendConnectorKinds(activeConnectorIds, index, ['q', 'k', 'pre', 'post', 'v']);
    }

    return buildFocusResult({
        label: `${resolveProjectionLabel(safeKind)} projection`,
        activeNodeIds,
        activeConnectorIds
    });
}

function buildProjectionWeightResult(index = null, node = null, kind = '') {
    const stageResult = buildProjectionStageResult(index, kind);
    if (!stageResult) return null;

    const label = `${resolveProjectionLabel(kind)} Weight Matrix`;
    return {
        ...stageResult,
        label,
        info: buildProjectionHoverInfo(node, label)
    };
}

function buildProjectionBiasResult(index = null, node = null, kind = '') {
    const stageResult = buildProjectionStageResult(index, kind);
    if (!stageResult) return null;

    const label = `${resolveProjectionLabel(kind)} Bias Vector`;
    return {
        ...stageResult,
        label,
        info: buildProjectionHoverInfo(node, label)
    };
}

function buildAttentionRoleResult(index = null, node = null, role = '') {
    const safeRole = String(role || '').trim();
    const attentionStageKey = resolveAttentionStageKeyForRole(safeRole);
    const attentionStageInfo = attentionStageKey
        ? buildAttentionStageRoleHoverInfo(node, attentionStageKey)
        : null;
    if (!index || !safeRole.length) return null;
    const activeNodeIds = [];
    const activeConnectorIds = [];

    if (safeRole === 'attention-query-source') {
        appendProjectionStagePath(activeNodeIds, index, ['q']);
        appendAllUnique(activeNodeIds, resolveAttentionRoleIds(index, {
            query: true,
            transpose: false,
            score: false
        }));
        appendConnectorKinds(activeConnectorIds, index, ['q']);
        return buildFocusResult({
            label: 'Query path',
            activeNodeIds,
            activeConnectorIds
        });
    }

    if (safeRole === 'attention-key-transpose') {
        appendProjectionStagePath(activeNodeIds, index, ['k']);
        appendAllUnique(activeNodeIds, resolveAttentionRoleIds(index, {
            query: false,
            transpose: true,
            score: false
        }));
        appendConnectorKinds(activeConnectorIds, index, ['k']);
        return buildFocusResult({
            label: 'Key path',
            activeNodeIds,
            activeConnectorIds
        });
    }

    if (safeRole === 'attention-mask') {
        appendProjectionStagePath(activeNodeIds, index, ['q', 'k']);
        appendAllUnique(activeNodeIds, resolveAttentionMaskStageRoleIds(index, {
            query: true,
            transpose: true
        }));
        appendConnectorKinds(activeConnectorIds, index, ['q', 'k', 'pre']);
        return buildFocusResult({
            label: attentionStageInfo?.activationData?.label || resolveAttentionHoverLabel('mask'),
            info: attentionStageInfo,
            activeNodeIds,
            activeConnectorIds
        });
    }

    if (safeRole === 'attention-value-post') {
        appendProjectionStagePath(activeNodeIds, index, ['q', 'k', 'v']);
        appendAllUnique(activeNodeIds, resolveAttentionRoleIds(index, {
            query: true,
            transpose: true,
            score: true,
            includeWeightedOutput: true
        }));
        appendConnectorKinds(activeConnectorIds, index, ['q', 'k', 'pre', 'post', 'v']);
        return buildFocusResult({
            label: 'Value Vector',
            info: buildProjectionHoverInfo(node, 'Value Vector', {
                stage: 'qkv.v'
            }),
            activeNodeIds,
            activeConnectorIds
        });
    }

    if (safeRole === 'attention-post-copy') {
        appendProjectionStagePath(activeNodeIds, index, ['q', 'k', 'v']);
        appendAllUnique(activeNodeIds, resolveAttentionRoleIds(index, {
            query: true,
            transpose: true,
            score: true,
            includeWeightedOutput: true
        }));
        appendConnectorKinds(activeConnectorIds, index, ['q', 'k', 'pre', 'post', 'v']);
        return buildFocusResult({
            label: attentionStageInfo?.activationData?.label || resolveAttentionHoverLabel('post'),
            info: attentionStageInfo,
            activeNodeIds,
            activeConnectorIds
        });
    }

    if (WEIGHTED_OUTPUT_ROLES.includes(safeRole) && safeRole !== 'attention-head-output') {
        appendProjectionStagePath(activeNodeIds, index, ['q', 'k', 'v']);
        appendAllUnique(activeNodeIds, resolveAttentionRoleIds(index, {
            query: true,
            transpose: true,
            score: true,
            includeWeightedOutput: true
        }));
        appendConnectorKinds(activeConnectorIds, index, ['q', 'k', 'pre', 'post', 'v']);
        return buildFocusResult({
            label: 'Weighted output',
            activeNodeIds,
            activeConnectorIds
        });
    }

    if (safeRole === 'attention-head-output') {
        appendAllUnique(activeNodeIds, [
            ...(index?.nodeIdsByRole?.get('attention-post-copy') || []),
            ...(index?.nodeIdsByRole?.get('attention-value-post') || []),
            ...(index?.nodeIdsByRole?.get('attention-head-output') || [])
        ]);
        return buildFocusResult({
            label: 'Attention Weighted Sum',
            info: buildWeightedSumHoverInfo(node, null),
            activeNodeIds,
            activeConnectorIds
        });
    }

    const isScoreRole = PRE_ATTENTION_ROLES.includes(safeRole) || SCORE_ATTENTION_ROLES.includes(safeRole);
    if (isScoreRole) {
        appendProjectionStagePath(activeNodeIds, index, ['q', 'k', ...(safeRole === 'attention-post' ? ['v'] : [])]);
        appendAllUnique(activeNodeIds, resolveAttentionRoleIds(index, {
            query: true,
            transpose: true,
            score: true,
            includeWeightedOutput: safeRole !== 'attention-mask'
        }));
        appendConnectorKinds(activeConnectorIds, index,
            safeRole === 'attention-post'
                ? ['q', 'k', 'pre', 'post', 'v']
                : ['q', 'k', 'pre', 'post']
        );
        return buildFocusResult({
            label: attentionStageInfo?.activationData?.label || 'Attention score path',
            info: attentionStageInfo,
            activeNodeIds,
            activeConnectorIds
        });
    }

    return null;
}

export function createMhsaDetailSceneIndex(scene = null) {
    if (!scene?.nodes || !Array.isArray(scene.nodes)) return null;
    const allNodes = flattenSceneNodes(scene).filter(Boolean);
    const nodesById = new Map(allNodes.map((node) => [node.id, node]));
    const leafNodes = allNodes.filter((node) => (
        node.kind !== VIEW2D_NODE_KINDS.GROUP
        && node.kind !== VIEW2D_NODE_KINDS.CONNECTOR
    ));
    const nodeIdsByRole = new Map();
    leafNodes.forEach((node) => {
        const role = typeof node.role === 'string' ? node.role : '';
        if (!role.length) return;
        const next = nodeIdsByRole.get(role) || [];
        next.push(node.id);
        nodeIdsByRole.set(role, next);
    });

    const connectorIdsByKind = {
        q: [],
        k: [],
        pre: [],
        post: [],
        v: []
    };
    const projectionIngressConnectorIdsByKind = {
        q: [],
        k: [],
        v: []
    };
    allNodes
        .filter((node) => node.kind === VIEW2D_NODE_KINDS.CONNECTOR)
        .forEach((node) => {
            const connectorRole = String(node.role || '').trim().toLowerCase();
            let kind = '';
            if (connectorRole.startsWith('connector-xln-')) {
                kind = normalizeProjectionKind(connectorRole.slice('connector-xln-'.length));
                if (Object.prototype.hasOwnProperty.call(projectionIngressConnectorIdsByKind, kind)) {
                    appendUnique(projectionIngressConnectorIdsByKind[kind], node.id);
                }
            } else if (connectorRole.startsWith('connector-')) {
                kind = connectorRole.slice('connector-'.length);
            }
            if (!Object.prototype.hasOwnProperty.call(connectorIdsByKind, kind)) return;
            appendUnique(connectorIdsByKind[kind], node.id);
        });

    const projectionLeafIdsByKind = {
        q: [],
        k: [],
        v: []
    };
    const projectionInputIdsByKind = {
        q: null,
        k: null,
        v: null
    };
    const projectionOutputIdsByKind = {
        q: null,
        k: null,
        v: null
    };

    allNodes
        .filter((node) => node.role === 'projection-stage')
        .forEach((stageNode) => {
            const kind = normalizeProjectionKind(stageNode?.metadata?.kind || resolveProjectionKindForNode(stageNode));
            if (!kind) return;
            const leafIds = collectLeafDescendantIds(stageNode, []);
            projectionLeafIdsByKind[kind] = leafIds;
            leafIds.forEach((nodeId) => {
                const leafNode = nodesById.get(nodeId) || null;
                if (leafNode?.role === 'x-ln-copy') {
                    projectionInputIdsByKind[kind] = nodeId;
                }
                if (leafNode?.role === 'projection-output') {
                    projectionOutputIdsByKind[kind] = nodeId;
                }
            });
        });

    const singleNodeIds = {
        projectionSourceXln: (nodeIdsByRole.get('projection-source-xln') || [])[0] || null,
        attentionQuerySource: (nodeIdsByRole.get('attention-query-source') || [])[0] || null,
        attentionKeyTranspose: (nodeIdsByRole.get('attention-key-transpose') || [])[0] || null,
        attentionPreScore: (nodeIdsByRole.get('attention-pre-score') || [])[0] || null,
        attentionMaskedInput: (nodeIdsByRole.get('attention-masked-input') || [])[0] || null,
        attentionMask: (nodeIdsByRole.get('attention-mask') || [])[0] || null,
        attentionPost: (nodeIdsByRole.get('attention-post') || [])[0] || null,
        attentionPostCopy: (nodeIdsByRole.get('attention-post-copy') || [])[0] || null,
        attentionValuePost: (nodeIdsByRole.get('attention-value-post') || [])[0] || null,
        attentionHeadOutput: (nodeIdsByRole.get('attention-head-output') || [])[0] || null
    };

    return {
        scene,
        nodesById,
        nodeIdsByRole,
        connectorIdsByKind,
        projectionIngressConnectorIdsByKind,
        projectionLeafIdsByKind,
        projectionInputIdsByKind,
        projectionOutputIdsByKind,
        singleNodeIds
    };
}

export function resolveMhsaDetailHoverState(index = null, hit = null) {
    if (!index || !hit?.node) return null;

    if (hit.cellHit) {
        const stageKey = resolveMatrixStageKey(hit.node, hit.cellHit);
        if (stageKey === 'pre-score') {
            return buildPreScoreCellResult(
                index,
                hit.node,
                hit.cellHit.rowIndex,
                hit.cellHit.colIndex,
                hit.cellHit.cellItem
            );
        }
        if (stageKey === 'masked-input' || stageKey === 'mask') {
            return buildSoftmaxCellResult(
                index,
                hit.node,
                hit.cellHit.rowIndex,
                hit.cellHit.colIndex,
                hit.cellHit.cellItem,
                {
                    stageKey,
                    includeWeightedOutput: false,
                    includePostCopy: false
                }
            );
        }
        if (stageKey === 'post' || stageKey === 'post-copy') {
            return buildSoftmaxCellResult(
                index,
                hit.node,
                hit.cellHit.rowIndex,
                hit.cellHit.colIndex,
                hit.cellHit.cellItem,
                {
                    stageKey,
                    includeWeightedOutput: true,
                    includePostCopy: true
                }
            );
        }
    }

    if ((hit.rowHit || hit.columnHit) && hit.node.role === 'attention-key-transpose') {
        return buildTransposeAxisResult(index, hit.rowHit || hit.columnHit);
    }

    if (hit.rowHit) {
        const projectionKind = resolveProjectionKindForNode(hit.node);
        if (projectionKind) {
            if (hit.node.role === 'projection-bias') {
                return buildProjectionBiasResult(index, hit.node, projectionKind);
            }
            return buildProjectionRowResult(index, hit.node, projectionKind, hit.rowHit, hit.node.role);
        }
        if (hit.node.role === 'attention-query-source') {
            return buildProjectionRowResult(index, hit.node, 'q', hit.rowHit, hit.node.role);
        }
        if (hit.node.role === 'projection-source-xln') {
            return buildProjectionSourceRowResult(index, hit.rowHit);
        }
        if (hit.node.role === 'attention-value-post') {
            return buildWeightedOutputRowResult(index, hit.node, hit.rowHit, {
                includeProjection: true,
                includeConnector: true,
                label: 'Value Vector',
                info: buildProjectionVectorHoverInfo(hit.node, hit.rowHit.rowItem, 'v')
            });
        }
        if (hit.node.role === 'attention-head-output') {
            return buildWeightedOutputRowResult(index, hit.node, hit.rowHit, {
                includeProjection: false,
                includeConnector: false,
                label: 'Attention Weighted Sum',
                info: buildWeightedSumHoverInfo(hit.node, hit.rowHit.rowItem)
            });
        }
    }

    const projectionKind = resolveProjectionKindForNode(hit.node);
    if (projectionKind) {
        if (hit.node.role === 'projection-weight') {
            return buildProjectionWeightResult(index, hit.node, projectionKind);
        }
        if (hit.node.role === 'projection-bias') {
            return buildProjectionBiasResult(index, hit.node, projectionKind);
        }
        return buildProjectionStageResult(index, projectionKind);
    }

    return buildAttentionRoleResult(index, hit.node, hit.node.role);
}
