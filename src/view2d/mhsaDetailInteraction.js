import {
    flattenSceneNodes,
    VIEW2D_NODE_KINDS
} from './schema/sceneTypes.js';

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

function formatCellLabel(cellItem = null) {
    const rowLabel = typeof cellItem?.rowLabel === 'string' && cellItem.rowLabel.length
        ? cellItem.rowLabel
        : 'Query token';
    const colLabel = typeof cellItem?.colLabel === 'string' && cellItem.colLabel.length
        ? cellItem.colLabel
        : 'Key token';
    return `${rowLabel} -> ${colLabel}`;
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
        { nodeId: index.singleNodeIds.attentionQuerySource, rowIndex }
    ];
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

function buildScoreAxisColumnSelections(index = null, colIndex = null) {
    if (!Number.isFinite(colIndex) || !index) return [];
    return [
        { nodeId: index.singleNodeIds.attentionKeyTranspose, colIndex },
        { nodeId: index.singleNodeIds.attentionPreScore, colIndex },
        { nodeId: index.singleNodeIds.attentionMaskedInput, colIndex },
        { nodeId: index.singleNodeIds.attentionMask, colIndex },
        { nodeId: index.singleNodeIds.attentionPost, colIndex }
    ];
}

function buildScoreAxisRowSelections(index = null, rowIndex = null) {
    if (!Number.isFinite(rowIndex) || !index) return [];
    return [
        { nodeId: index.singleNodeIds.attentionPreScore, rowIndex },
        { nodeId: index.singleNodeIds.attentionMaskedInput, rowIndex },
        { nodeId: index.singleNodeIds.attentionMask, rowIndex },
        { nodeId: index.singleNodeIds.attentionPost, rowIndex }
    ];
}

function buildPreScoreCellResult(index = null, rowIndex = null, colIndex = null, cellItem = null) {
    const activeNodeIds = [];
    const activeConnectorIds = [];
    const rowSelections = [
        ...buildQueryRowSelections(index, rowIndex),
        ...buildKeyRowSelections(index, colIndex)
    ];
    const columnSelections = buildScoreAxisColumnSelections(index, colIndex);
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
        label: formatCellLabel(cellItem),
        activeNodeIds,
        activeConnectorIds,
        rowSelections,
        columnSelections,
        cellSelections
    });
}

function buildSoftmaxCellResult(index = null, rowIndex = null, colIndex = null, cellItem = null, {
    includeWeightedOutput = false,
    includePostCopy = false
} = {}) {
    const activeNodeIds = [];
    const activeConnectorIds = [];
    const rowSelections = [
        ...buildQueryRowSelections(index, rowIndex),
        ...buildKeyRowSelections(index, colIndex),
        ...(includeWeightedOutput ? buildValueRowSelections(index, rowIndex) : [])
    ];
    const columnSelections = buildScoreAxisColumnSelections(index, colIndex);
    const cellSelections = [
        { nodeId: index?.singleNodeIds?.attentionPreScore, rowIndex, colIndex },
        { nodeId: index?.singleNodeIds?.attentionMaskedInput, rowIndex, colIndex },
        { nodeId: index?.singleNodeIds?.attentionMask, rowIndex, colIndex },
        { nodeId: index?.singleNodeIds?.attentionPost, rowIndex, colIndex },
        ...(includePostCopy
            ? [{ nodeId: index?.singleNodeIds?.attentionPostCopy, rowIndex, colIndex }]
            : [])
    ];

    appendProjectionStagePath(activeNodeIds, index, includeWeightedOutput ? ['q', 'k', 'v'] : ['q', 'k']);
    appendAllUnique(activeNodeIds, resolveAttentionRoleIds(index, {
        query: true,
        transpose: true,
        score: true,
        includeWeightedOutput
    }));
    appendConnectorKinds(activeConnectorIds, index, includeWeightedOutput
        ? ['q', 'k', 'pre', 'post', 'v']
        : ['q', 'k', 'pre', 'post']);

    return buildFocusResult({
        label: formatCellLabel(cellItem),
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
        appendAllUnique(activeNodeIds, resolveAttentionRoleIds(index, {
            query: false,
            transpose: true,
            score: false
        }));
        appendConnectorKinds(activeConnectorIds, index, ['k']);
        rowSelections.push(...buildKeyRowSelections(index, rowIndex));
        columnSelections.push(...buildScoreAxisColumnSelections(index, rowIndex));
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
    if (role === 'projection-output' || role === 'attention-query-source') {
        return {
            ...result,
            label: `${resolveProjectionLabel(safeKind)} Vector`,
            info: buildProjectionVectorHoverInfo(node, rowHit.rowItem, safeKind)
        };
    }
    return result;
}

function buildTransposeColumnResult(index = null, columnHit = null) {
    if (!index || !columnHit) return null;
    const colIndex = Number.isFinite(columnHit.colIndex) ? Math.max(0, Math.floor(columnHit.colIndex)) : null;
    const activeNodeIds = [];
    const activeConnectorIds = [];
    const rowSelections = buildKeyRowSelections(index, colIndex);
    const columnSelections = buildScoreAxisColumnSelections(index, colIndex);

    appendProjectionStagePath(activeNodeIds, index, ['k']);
    appendAllUnique(activeNodeIds, resolveAttentionRoleIds(index, {
        query: false,
        transpose: true,
        score: false
    }));
    appendConnectorKinds(activeConnectorIds, index, ['k']);

    return buildFocusResult({
        label: 'Key Vector',
        info: buildProjectionColumnHoverInfo(
            index?.nodesById?.get(index?.singleNodeIds?.attentionKeyTranspose || ''),
            columnHit.columnItem,
            'k'
        ),
        activeNodeIds,
        activeConnectorIds,
        rowSelections,
        columnSelections
    });
}

function buildWeightedOutputRowResult(index = null, rowHit = null, {
    includeProjection = false,
    includeConnector = false,
    label = 'Head output row'
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
        info: createTokenInfo(rowHit.rowItem),
        activeNodeIds,
        activeConnectorIds,
        rowSelections
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

function buildAttentionRoleResult(index = null, role = '') {
    const safeRole = String(role || '').trim();
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
        appendAllUnique(activeNodeIds, resolveAttentionRoleIds(index, {
            query: true,
            transpose: true,
            score: true,
            includeWeightedOutput: false
        }));
        appendConnectorKinds(activeConnectorIds, index, ['q', 'k', 'pre', 'post']);
        return buildFocusResult({
            label: 'Causal mask',
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
            label: 'Head output',
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
            label: 'Attention score path',
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
    allNodes
        .filter((node) => node.kind === VIEW2D_NODE_KINDS.CONNECTOR)
        .forEach((node) => {
            const connectorRole = String(node.role || '').trim().toLowerCase();
            let kind = '';
            if (connectorRole.startsWith('connector-xln-')) {
                kind = normalizeProjectionKind(connectorRole.slice('connector-xln-'.length));
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
                hit.cellHit.rowIndex,
                hit.cellHit.colIndex,
                hit.cellHit.cellItem
            );
        }
        if (stageKey === 'masked-input' || stageKey === 'mask') {
            return buildSoftmaxCellResult(
                index,
                hit.cellHit.rowIndex,
                hit.cellHit.colIndex,
                hit.cellHit.cellItem,
                {
                    includeWeightedOutput: false,
                    includePostCopy: false
                }
            );
        }
        if (stageKey === 'post' || stageKey === 'post-copy') {
            return buildSoftmaxCellResult(
                index,
                hit.cellHit.rowIndex,
                hit.cellHit.colIndex,
                hit.cellHit.cellItem,
                {
                    includeWeightedOutput: true,
                    includePostCopy: true
                }
            );
        }
    }

    if (hit.columnHit && hit.node.role === 'attention-key-transpose') {
        return buildTransposeColumnResult(index, hit.columnHit);
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
        if (hit.node.role === 'attention-value-post') {
            return buildWeightedOutputRowResult(index, hit.rowHit, {
                includeProjection: true,
                includeConnector: true,
                label: 'Value row'
            });
        }
        if (hit.node.role === 'attention-head-output') {
            return buildWeightedOutputRowResult(index, hit.rowHit, {
                includeProjection: false,
                includeConnector: false,
                label: 'Head output row'
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

    return buildAttentionRoleResult(index, hit.node.role);
}
