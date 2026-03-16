import {
    normalizeAttentionHoverStageKey,
    resolveAttentionHoverLabel
} from '../ui/attentionHoverInfo.js';
import {
    MLP_ACTIVATION_TOOLTIP_LABEL,
    MLP_DOWN_BIAS_TOOLTIP_LABEL,
    MLP_DOWN_TOOLTIP_LABEL,
    MLP_UP_BIAS_TOOLTIP_LABEL
} from '../utils/mlpLabels.js';
import {
    PRE_ATTENTION_ROLES,
    SCORE_ATTENTION_ROLES,
    WEIGHTED_OUTPUT_ROLES,
    appendAttentionPostCopyNode,
    buildAttentionKeyTransposeSelections,
    buildFocusPathState,
    buildKeyRowSelections,
    buildProjectionSourceMirrorRowSelections,
    buildProjectionSourceRowSelections,
    buildQueryRowSelections,
    buildScoreAxisRowSelectionsWithOptions,
    buildScoreAxisSelections,
    buildValueRowSelections
} from './mhsaDetailPathGraph.js';
import {
    appendUnique,
    appendAllUnique,
    buildFocusResult
} from './mhsaDetailFocusResult.js';
import {
    buildAttentionCellHoverInfo,
    buildConcatenatedHeadOutputBandHoverInfo,
    buildConcatenatedHeadOutputHoverInfo,
    buildLayerNormActivationHoverInfo,
    buildLayerNormParamHoverInfo,
    buildMlpActivationHoverInfo,
    buildMlpDownBiasHoverInfo,
    buildMlpDownProjectionHoverInfo,
    buildMlpDownWeightHoverInfo,
    buildMlpUpBiasHoverInfo,
    buildMlpUpProjectionHoverInfo,
    buildMlpUpWeightHoverInfo,
    buildAttentionStageRoleHoverInfo,
    buildPostLayerNormResidualHoverInfo,
    buildProjectionBiasHoverInfo,
    buildProjectionColumnHoverInfo,
    buildProjectionHoverInfo,
    buildProjectionVectorHoverInfo,
    buildWeightedSumHoverInfo,
    createTokenInfo,
    normalizeProjectionKind,
    resolveAttentionStageKeyForRole,
    resolveProjectionLabel
} from './mhsaDetailHoverInfo.js';
import { resolvePostLayerNormResidualLabel } from '../utils/layerNormLabels.js';
import { resolveMhsaDetailHoverEntity } from './mhsaDetailHoverEntity.js';

const PROJECTION_STAGE_PATH_PRESETS = Object.freeze({
    q: 'queryStage',
    k: 'keyStage',
    v: 'valueStage'
});

const PROJECTION_SOURCE_PATH_PRESETS = Object.freeze({
    q: {
        projectionSourceKinds: ['q'],
        projectionInputKinds: ['q']
    },
    k: {
        projectionSourceKinds: ['k'],
        projectionInputKinds: ['k']
    },
    v: {
        projectionSourceKinds: ['v'],
        projectionInputKinds: ['v']
    }
});

const PROJECTION_TERMINAL_NODE_KEYS = Object.freeze({
    q: 'attentionQuerySource',
    k: 'attentionKeyTranspose',
    v: 'attentionValuePost'
});
const VALUE_BRANCH_EXTRA_DIM_PROJECTION_KINDS = Object.freeze(['q', 'k']);

function buildValueBranchExtraDimNodeIds(index = null) {
    if (!index) return [];
    const dimNodeIds = [];
    VALUE_BRANCH_EXTRA_DIM_PROJECTION_KINDS.forEach((kind) => {
        appendUnique(dimNodeIds, index?.projectionInputIdsByKind?.[kind] || '');
        appendUnique(dimNodeIds, index?.projectionOutputIdsByKind?.[kind] || '');
        appendUnique(dimNodeIds, index?.singleNodeIds?.[PROJECTION_TERMINAL_NODE_KEYS[kind]] || '');
        (index?.projectionLeafIdsByKind?.[kind] || []).forEach((nodeId) => {
            const node = index?.nodesById?.get(nodeId) || null;
            if (
                node?.role === 'projection-weight'
                || node?.role === 'projection-bias'
                || node?.role === 'projection-output'
            ) {
                appendUnique(dimNodeIds, nodeId);
            }
        });
    });
    return dimNodeIds;
}

function buildPathFocusResult(index = null, {
    label = '',
    info = null,
    descriptors = [],
    extraNodeIds = [],
    extraConnectorIds = [],
    dimNodeIds = [],
    rowSelections = [],
    columnSelections = [],
    cellSelections = []
} = {}, options = null) {
    const includeFocusState = options?.includeFocusState !== false;
    const { activeNodeIds, activeConnectorIds } = buildFocusPathState(index, descriptors, {
        extraNodeIds,
        extraConnectorIds
    });
    return buildFocusResult({
        label,
        info,
        activeNodeIds,
        activeConnectorIds,
        dimNodeIds,
        rowSelections,
        columnSelections,
        cellSelections,
        includeFocusState
    });
}

const LAYER_NORM_DIRECT_FOCUS_CONFIGS = Object.freeze({
    'layer-norm-input': {
        connectedNodeRoles: ['layer-norm-normalized'],
        connectorRoles: ['connector-layer-norm-normalization'],
        mirroredRowRoles: ['layer-norm-normalized']
    },
    'layer-norm-normalized': {
        connectedNodeRoles: ['layer-norm-input'],
        connectorRoles: ['connector-layer-norm-normalization'],
        mirroredRowRoles: ['layer-norm-input']
    },
    'layer-norm-normalized-copy': {
        connectedNodeRoles: ['layer-norm-normalized'],
        connectorRoles: ['connector-layer-norm-copy-normalized'],
        mirroredRowRoles: ['layer-norm-normalized']
    },
    'layer-norm-scale': {
        connectedNodeRoles: ['layer-norm-normalized-copy']
    },
    'layer-norm-scaled': {
        connectedNodeRoles: ['layer-norm-normalized-copy', 'layer-norm-scale'],
        mirroredRowRoles: ['layer-norm-normalized-copy'],
        fixedRowSelections: [{
            role: 'layer-norm-scale',
            rowIndex: 0
        }]
    },
    'layer-norm-scaled-copy': {
        connectedNodeRoles: ['layer-norm-scaled'],
        connectorRoles: ['connector-layer-norm-copy-scaled'],
        mirroredRowRoles: ['layer-norm-scaled']
    },
    'layer-norm-shift': {
        connectedNodeRoles: ['layer-norm-scaled-copy']
    },
    'layer-norm-output': {
        connectedNodeRoles: ['layer-norm-scaled-copy', 'layer-norm-shift'],
        mirroredRowRoles: ['layer-norm-scaled-copy'],
        fixedRowSelections: [{
            role: 'layer-norm-shift',
            rowIndex: 0
        }]
    }
});

function findSingleNodeIdByRole(index = null, role = '') {
    return (index?.nodeIdsByRole?.get(role) || [])[0] || '';
}

function findProjectionNodeIdByKind(index = null, role = '', kind = '') {
    const safeRole = String(role || '').trim();
    const safeKind = normalizeProjectionKind(kind);
    if (!index?.nodesById || !safeRole.length || !safeKind) return '';
    const nodeIds = index?.nodeIdsByRole?.get(safeRole) || [];
    for (const nodeId of nodeIds) {
        const node = index.nodesById.get(nodeId);
        const nodeKind = normalizeProjectionKind(node?.metadata?.kind || node?.semantic?.branchKey || '');
        if (nodeKind === safeKind) {
            return nodeId;
        }
    }
    return '';
}

function findNodeIdByRole(index = null, role = '') {
    const safeRole = String(role || '').trim();
    if (!index?.nodesById || !safeRole.length) return '';
    for (const node of index.nodesById.values()) {
        if (String(node?.role || '').trim() === safeRole) {
            return node.id;
        }
    }
    return '';
}

function buildDecodeProjectionSourceCurrentRowSelections(index = null, {
    includeSelection = false
} = {}) {
    if (!includeSelection) return [];
    const projectionSourceNodeId = index?.singleNodeIds?.projectionSourceXln || '';
    if (!projectionSourceNodeId) return [];
    return [{
        nodeId: projectionSourceNodeId,
        rowIndex: 0
    }];
}

function buildDecodeProjectionCurrentInputRowSelections(index = null, kind = '', {
    includeProjectionSource = true,
    rowIndex = 0
} = {}) {
    const safeKind = normalizeProjectionKind(kind);
    const safeRowIndex = Number.isFinite(rowIndex) ? Math.max(0, Math.floor(rowIndex)) : 0;
    if (!safeKind || !index) return [];
    const rowSelections = includeProjectionSource
        ? buildDecodeProjectionSourceCurrentRowSelections(index, {
            includeSelection: true
        })
        : [];
    const projectionInputNodeId = index?.projectionInputIdsByKind?.[safeKind] || '';
    if (projectionInputNodeId) {
        rowSelections.push({
            nodeId: projectionInputNodeId,
            rowIndex: safeRowIndex
        });
    }
    return rowSelections;
}

function buildScoreAxisSelectionsWithoutTranspose(index = null, axisIndex = null, options = null) {
    const selections = buildScoreAxisSelections(index, axisIndex, options);
    const transposeNodeId = index?.singleNodeIds?.attentionKeyTranspose || '';
    if (!transposeNodeId) return selections;
    return {
        rowSelections: (Array.isArray(selections?.rowSelections) ? selections.rowSelections : []).filter(
            (selection) => selection?.nodeId !== transposeNodeId
        ),
        columnSelections: (Array.isArray(selections?.columnSelections) ? selections.columnSelections : []).filter(
            (selection) => selection?.nodeId !== transposeNodeId
        )
    };
}

function buildDecodeClickedScoreCellKeyBackprop(index = null, axisIndex = null) {
    const safeAxisIndex = Number.isFinite(axisIndex) ? Math.max(0, Math.floor(axisIndex)) : null;
    if (!index || !Number.isFinite(safeAxisIndex)) {
        return {
            rowSelections: [],
            extraConnectorIds: []
        };
    }
    if (!index?.scene?.metadata?.kvCacheState?.kvCacheDecodeActive) {
        return {
            rowSelections: buildKeyRowSelections(index, safeAxisIndex),
            extraConnectorIds: []
        };
    }

    const liveDisplayRowIndex = resolveDecodeProjectionLiveDisplayRowIndex(index, 'k');
    const projectionOutputNodeId = index?.projectionOutputIdsByKind?.k || '';
    const projectionOutputCopyNodeId = findProjectionNodeIdByKind(index, 'projection-output-copy', 'k');
    const cacheNodeId = findProjectionNodeIdByKind(index, 'projection-cache', 'k');
    const cacheSourceNodeId = findProjectionNodeIdByKind(index, 'projection-cache-source', 'k');
    const concatResultNodeId = findProjectionNodeIdByKind(index, 'projection-cache-concat-result', 'k');
    const nextPreviewNodeId = findProjectionNodeIdByKind(index, 'projection-cache-next', 'k');
    const cacheSourceConnectorNodeId = findNodeIdByRole(index, 'connector-k-cache-source');
    const cacheCopyConnectorNodeId = findNodeIdByRole(index, 'connector-k-cache-copy');
    const nextPreviewConnectorNodeId = findNodeIdByRole(index, 'connector-k-cache-next');

    if (
        Number.isFinite(liveDisplayRowIndex)
        && safeAxisIndex === liveDisplayRowIndex
    ) {
        return {
            liveColumn: true,
            rowSelections: [
                ...buildDecodeProjectionCurrentInputRowSelections(index, 'k', {
                    includeProjectionSource: true,
                    rowIndex: 0
                }),
                ...(projectionOutputNodeId
                    ? [{
                        nodeId: projectionOutputNodeId,
                        rowIndex: 0
                    }]
                    : []),
                ...(projectionOutputCopyNodeId
                    ? [{
                        nodeId: projectionOutputCopyNodeId,
                        rowIndex: 0
                    }]
                    : []),
                ...(concatResultNodeId
                    ? [{
                        nodeId: concatResultNodeId,
                        rowIndex: liveDisplayRowIndex
                    }]
                    : []),
                ...(nextPreviewNodeId
                    ? [{
                        nodeId: nextPreviewNodeId,
                        rowIndex: liveDisplayRowIndex
                    }]
                    : [])
            ],
            extraConnectorIds: [
                cacheCopyConnectorNodeId,
                nextPreviewConnectorNodeId
            ].filter(Boolean)
        };
    }

    return {
        liveColumn: false,
        rowSelections: [
            ...(cacheNodeId
                ? [{
                    nodeId: cacheNodeId,
                    rowIndex: safeAxisIndex
                }]
                : []),
            ...(cacheSourceNodeId
                ? [{
                    nodeId: cacheSourceNodeId,
                    rowIndex: safeAxisIndex
                }]
                : []),
            ...(concatResultNodeId
                ? [{
                    nodeId: concatResultNodeId,
                    rowIndex: safeAxisIndex
                }]
                : [])
        ],
        extraConnectorIds: [cacheSourceConnectorNodeId].filter(Boolean)
    };
}

function buildDecodeClickedPreScoreDescriptors(keyBackprop = null) {
    const isLiveColumn = keyBackprop?.liveColumn === true;
    if (isLiveColumn) {
        return [
            {
                projectionKinds: ['q', 'k'],
                attentionOptions: {
                    query: true,
                    transpose: true,
                    score: false,
                    includeWeightedOutput: false
                },
                singleNodeKeys: ['attentionPreScore', 'attentionMaskedInput'],
                connectorKinds: ['q', 'k', 'pre']
            }
        ];
    }
    return [
        {
            projectionKinds: ['q'],
            attentionOptions: {
                query: true,
                transpose: true,
                score: false,
                includeWeightedOutput: false
            },
            singleNodeKeys: ['attentionPreScore', 'attentionMaskedInput'],
            connectorKinds: ['q', 'k', 'pre']
        }
    ];
}

function buildDecodeClickedSoftmaxDescriptors(keyBackprop = null, {
    isMaskStage = false
} = {}) {
    const isLiveColumn = keyBackprop?.liveColumn === true;
    if (isLiveColumn) {
        return [
            isMaskStage
                ? 'maskPath'
                : 'scorePath'
        ];
    }
    return [
        isMaskStage
            ? {
                projectionKinds: ['q'],
                attentionMaskOptions: {
                    query: true,
                    transpose: true
                },
                connectorKinds: ['q', 'k', 'pre']
            }
            : {
                projectionKinds: ['q'],
                attentionOptions: {
                    query: true,
                    transpose: true,
                    score: true,
                    includeWeightedOutput: false
                },
                connectorKinds: ['q', 'k', 'pre', 'post']
            }
    ];
}

function resolveDecodeProjectionLiveDisplayRowIndex(index = null, kind = '') {
    const concatResultNodeId = findProjectionNodeIdByKind(index, 'projection-cache-concat-result', kind);
    if (!concatResultNodeId) return null;
    const concatResultNode = index?.nodesById?.get(concatResultNodeId) || null;
    const rowCount = Number.isFinite(concatResultNode?.dimensions?.rows)
        ? Math.max(0, Math.floor(concatResultNode.dimensions.rows))
        : 0;
    if (rowCount > 0) {
        return rowCount - 1;
    }
    const rowItems = Array.isArray(concatResultNode?.rowItems) ? concatResultNode.rowItems : [];
    if (!rowItems.length) return null;
    return Math.max(0, rowItems.length - 1);
}

function appendLayerNormRowSelectionByRole(target = [], index = null, role = '', rowIndex = null) {
    if (!Array.isArray(target) || !Number.isFinite(rowIndex)) return;
    const nodeId = findSingleNodeIdByRole(index, role);
    if (!nodeId) return;
    target.push({
        nodeId,
        rowIndex: Math.max(0, Math.floor(rowIndex))
    });
}

function appendLayerNormNodeIdsByRoles(target = [], index = null, roles = []) {
    roles.forEach((role) => {
        appendUnique(target, findSingleNodeIdByRole(index, role));
    });
}

function appendLayerNormConnectorIdsByRoles(target = [], index = null, roles = []) {
    roles.forEach((role) => {
        appendUnique(target, findNodeIdByRole(index, role));
    });
}

function buildLayerNormDirectFocusResult(index = null, {
    label = '',
    info = null,
    hoveredRole = '',
    rowIndex = null,
    selfRowIndex = rowIndex
} = {}, options = null) {
    const config = LAYER_NORM_DIRECT_FOCUS_CONFIGS[String(hoveredRole || '').trim()] || {};
    const activeNodeIds = [];
    const activeConnectorIds = [];
    const rowSelections = [];

    appendLayerNormNodeIdsByRoles(activeNodeIds, index, [
        hoveredRole,
        ...(config.connectedNodeRoles || [])
    ]);
    appendLayerNormConnectorIdsByRoles(activeConnectorIds, index, config.connectorRoles || []);

    appendLayerNormRowSelectionByRole(rowSelections, index, hoveredRole, selfRowIndex);
    if (Number.isFinite(rowIndex)) {
        (config.mirroredRowRoles || []).forEach((role) => {
            appendLayerNormRowSelectionByRole(rowSelections, index, role, rowIndex);
        });
    }
    (config.fixedRowSelections || []).forEach((selection) => {
        appendLayerNormRowSelectionByRole(
            rowSelections,
            index,
            selection?.role || '',
            selection?.rowIndex
        );
    });

    return buildFocusResult({
        label,
        info,
        activeNodeIds,
        activeConnectorIds,
        rowSelections,
        includeFocusState: options?.includeFocusState !== false
    });
}

function buildPostCopyMirrorCellSelections(index = null, rowIndex = null, colIndex = null) {
    if (!index || !Number.isFinite(rowIndex) || !Number.isFinite(colIndex)) return [];
    const nodeId = index?.singleNodeIds?.attentionPostCopy;
    if (typeof nodeId !== 'string' || !nodeId.length) return [];
    return [{
        nodeId,
        rowIndex,
        colIndex
    }];
}

function buildPreScoreCellResult(index = null, node = null, rowIndex = null, colIndex = null, cellItem = null, options = null) {
    const interactionKind = String(options?.interactionKind || 'hover').trim().toLowerCase();
    const isClickInteraction = interactionKind === 'click';
    const keyBackprop = isClickInteraction
        ? buildDecodeClickedScoreCellKeyBackprop(index, colIndex)
        : {
            rowSelections: buildKeyRowSelections(index, colIndex),
            extraConnectorIds: []
        };
    const scoreAxisSelections = buildScoreAxisSelections(index, colIndex, {
        includePost: false,
        includePostCopy: false
        });
    const info = buildAttentionCellHoverInfo(node, cellItem, 'pre');
    const rowSelections = [
        ...buildQueryRowSelections(index, rowIndex),
        ...keyBackprop.rowSelections,
        ...scoreAxisSelections.rowSelections,
        { nodeId: index?.singleNodeIds?.attentionHeadOutput, rowIndex }
    ];
    const columnSelections = scoreAxisSelections.columnSelections;
    const cellSelections = [
        { nodeId: index?.singleNodeIds?.attentionPreScore, rowIndex, colIndex },
        { nodeId: index?.singleNodeIds?.attentionMaskedInput, rowIndex, colIndex },
        { nodeId: index?.singleNodeIds?.attentionMask, rowIndex, colIndex },
        { nodeId: index?.singleNodeIds?.attentionPost, rowIndex, colIndex },
        ...buildPostCopyMirrorCellSelections(index, rowIndex, colIndex)
    ];

    return buildPathFocusResult(index, {
        label: info?.activationData?.label || resolveAttentionHoverLabel('pre'),
        info,
        descriptors: isClickInteraction
            ? buildDecodeClickedPreScoreDescriptors(keyBackprop)
            : [
                {
                    projectionKinds: ['q', 'k'],
                    attentionOptions: {
                        query: true,
                        transpose: true,
                        score: false,
                        includeWeightedOutput: false
                    },
                    singleNodeKeys: ['attentionPreScore', 'attentionMaskedInput'],
                    connectorKinds: ['q', 'k', 'pre']
                }
            ],
        extraConnectorIds: keyBackprop.extraConnectorIds,
        rowSelections,
        columnSelections,
        cellSelections
    }, options);
}

function buildSoftmaxCellResult(index = null, node = null, rowIndex = null, colIndex = null, cellItem = null, {
    stageKey = 'post',
    includePostCopy = false,
    includeMirroredPostCopyCell = false,
    includePostForMask = false
} = {}, options = null) {
    const interactionKind = String(options?.interactionKind || 'hover').trim().toLowerCase();
    const isClickInteraction = interactionKind === 'click';
    const keyBackprop = isClickInteraction
        ? buildDecodeClickedScoreCellKeyBackprop(index, colIndex)
        : {
            rowSelections: buildKeyRowSelections(index, colIndex),
            extraConnectorIds: []
        };
    const normalizedStageKey = normalizeAttentionHoverStageKey(stageKey);
    const isMaskStage = normalizedStageKey === 'mask';
    const includePostSelections = !isMaskStage || includePostForMask;
    const scoreAxisSelections = buildScoreAxisSelections(index, colIndex, {
        includePost: includePostSelections,
        includePostCopy: includePostSelections && includePostCopy
    });
    const info = buildAttentionCellHoverInfo(node, cellItem, normalizedStageKey);
    const rowSelections = [
        ...buildQueryRowSelections(index, rowIndex),
        ...keyBackprop.rowSelections,
        ...scoreAxisSelections.rowSelections,
        { nodeId: index?.singleNodeIds?.attentionHeadOutput, rowIndex }
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
            : []),
        ...(includeMirroredPostCopyCell
            ? buildPostCopyMirrorCellSelections(index, rowIndex, colIndex)
            : [])
    ];

    return buildPathFocusResult(index, {
        label: info?.activationData?.label || resolveAttentionHoverLabel(normalizedStageKey),
        info,
        descriptors: isClickInteraction
            ? buildDecodeClickedSoftmaxDescriptors(keyBackprop, {
                isMaskStage
            })
            : [
                isMaskStage
                    ? 'maskPath'
                    : 'scorePath'
            ],
        extraConnectorIds: keyBackprop.extraConnectorIds,
        rowSelections,
        columnSelections,
        cellSelections
    }, options);
}

function buildProjectionRowResult(index = null, node = null, kind = '', rowHit = null, role = '', options = null) {
    const safeKind = normalizeProjectionKind(kind);
    if (!safeKind || !rowHit) return null;
    const rowIndex = Number.isFinite(rowHit.rowIndex) ? Math.max(0, Math.floor(rowHit.rowIndex)) : null;
    const interactionKind = String(options?.interactionKind || 'hover').trim().toLowerCase();
    const isClickInteraction = interactionKind === 'click';
    const projectionOutputNodeId = index?.projectionOutputIdsByKind?.[safeKind] || '';
    const dimNodeIds = safeKind === 'v'
        ? buildValueBranchExtraDimNodeIds(index)
        : [];
    const rowSelections = [];
    const columnSelections = [];
    const descriptors = [];
    const extraNodeIds = [];
    const extraConnectorIds = [];
    const projectionOutputCopyNodeId = findProjectionNodeIdByKind(index, 'projection-output-copy', safeKind);
    const cacheNodeId = findProjectionNodeIdByKind(index, 'projection-cache', safeKind);
    const cacheSourceNodeId = findProjectionNodeIdByKind(index, 'projection-cache-source', safeKind);
    const concatResultNodeId = findProjectionNodeIdByKind(index, 'projection-cache-concat-result', safeKind);
    const nextPreviewNodeId = findProjectionNodeIdByKind(index, 'projection-cache-next', safeKind);
    const cacheSourceConnectorNodeId = findNodeIdByRole(index, `connector-${safeKind}-cache-source`);
    const cacheCopyConnectorNodeId = findNodeIdByRole(index, `connector-${safeKind}-cache-copy`);
    const nextPreviewConnectorNodeId = findNodeIdByRole(index, `connector-${safeKind}-cache-next`);
    const isDecodeCurrentProjectionRow = (
        safeKind === 'k'
        && (role === 'projection-output' || role === 'projection-output-copy')
        && Number.isFinite(node?.dimensions?.rows)
        && Math.max(0, Math.floor(node.dimensions.rows)) === 1
        && !!concatResultNodeId
        && !!nextPreviewNodeId
    );
    const isDecodeCurrentValueProjectionRow = (
        safeKind === 'v'
        && (role === 'projection-output' || role === 'projection-output-copy')
        && Number.isFinite(node?.dimensions?.rows)
        && Math.max(0, Math.floor(node.dimensions.rows)) === 1
        && !!projectionOutputCopyNodeId
        && !!concatResultNodeId
        && !!nextPreviewNodeId
    );
    const isDecodeCurrentValueInputRow = (
        safeKind === 'v'
        && role === 'x-ln-copy'
        && !!index?.scene?.metadata?.kvCacheState?.kvCacheDecodeActive
        && Number.isFinite(node?.dimensions?.rows)
        && Math.max(0, Math.floor(node.dimensions.rows)) === 1
        && !!projectionOutputNodeId
        && !!projectionOutputCopyNodeId
        && !!concatResultNodeId
        && !!nextPreviewNodeId
    );

    if (isDecodeCurrentProjectionRow) {
        const liveDisplayRowIndex = resolveDecodeProjectionLiveDisplayRowIndex(index, safeKind);
        const scoreAxisSelections = buildScoreAxisSelections(index, liveDisplayRowIndex, {
            includePost: true,
            includePostCopy: true
        });
        return buildPathFocusResult(index, {
            label: `${resolveProjectionLabel(safeKind)} Vector`,
            info: buildProjectionVectorHoverInfo(node, rowHit.rowItem, safeKind),
            descriptors: ['keyScorePath'],
            extraConnectorIds: [cacheCopyConnectorNodeId, nextPreviewConnectorNodeId].filter(Boolean),
            rowSelections: [
                ...buildKeyRowSelections(index, rowIndex),
                {
                    nodeId: projectionOutputCopyNodeId,
                    rowIndex
                },
                {
                    nodeId: concatResultNodeId,
                    rowIndex: liveDisplayRowIndex
                },
                {
                    nodeId: nextPreviewNodeId,
                    rowIndex: liveDisplayRowIndex
                }
            ],
            columnSelections: scoreAxisSelections.columnSelections
        }, options);
    }

    if (isDecodeCurrentValueProjectionRow) {
        const liveDisplayRowIndex = resolveDecodeProjectionLiveDisplayRowIndex(index, safeKind);
        return buildPathFocusResult(index, {
            label: `${resolveProjectionLabel(safeKind)} Vector`,
            info: buildProjectionVectorHoverInfo(node, rowHit.rowItem, safeKind),
            descriptors: ['valueProjectionPath'],
            dimNodeIds,
            extraConnectorIds: [cacheCopyConnectorNodeId, nextPreviewConnectorNodeId].filter(Boolean),
            rowSelections: [
                ...buildDecodeProjectionCurrentInputRowSelections(index, safeKind, {
                    includeProjectionSource: true,
                    rowIndex
                }),
                {
                    nodeId: projectionOutputNodeId,
                    rowIndex
                },
                {
                    nodeId: projectionOutputCopyNodeId,
                    rowIndex
                },
                {
                    nodeId: concatResultNodeId,
                    rowIndex: liveDisplayRowIndex
                },
                {
                    nodeId: nextPreviewNodeId,
                    rowIndex: liveDisplayRowIndex
                },
                {
                    nodeId: index?.singleNodeIds?.attentionValuePost,
                    rowIndex: liveDisplayRowIndex
                }
            ]
        }, options);
    }

    if (isDecodeCurrentValueInputRow) {
        const liveDisplayRowIndex = resolveDecodeProjectionLiveDisplayRowIndex(index, safeKind);
        const info = buildPostLayerNormResidualHoverInfo(node, rowHit.rowItem);
        return buildPathFocusResult(index, {
            label: info?.activationData?.label || `${resolveProjectionLabel(safeKind)} row`,
            info,
            descriptors: ['valueProjectionPath'],
            dimNodeIds,
            extraConnectorIds: [cacheCopyConnectorNodeId, nextPreviewConnectorNodeId].filter(Boolean),
            rowSelections: [
                ...buildDecodeProjectionCurrentInputRowSelections(index, safeKind, {
                    includeProjectionSource: true,
                    rowIndex
                }),
                {
                    nodeId: projectionOutputNodeId,
                    rowIndex
                },
                {
                    nodeId: projectionOutputCopyNodeId,
                    rowIndex
                },
                {
                    nodeId: concatResultNodeId,
                    rowIndex: liveDisplayRowIndex
                },
                {
                    nodeId: nextPreviewNodeId,
                    rowIndex: liveDisplayRowIndex
                },
                {
                    nodeId: index?.singleNodeIds?.attentionValuePost,
                    rowIndex: liveDisplayRowIndex
                }
            ]
        }, options);
    }

    if (safeKind === 'k' && role === 'x-ln-copy' && !isClickInteraction) {
        const result = buildPathFocusResult(index, {
            label: `${resolveProjectionLabel(safeKind)} row`,
            info: createTokenInfo(rowHit.rowItem),
            descriptors: [{
                projectionKinds: ['k'],
                projectionInputKinds: ['k']
            }],
            rowSelections: [
                ...buildKeyRowSelections(index, rowIndex),
                ...buildProjectionSourceMirrorRowSelections(index, rowIndex)
            ]
        }, options);
        const info = buildPostLayerNormResidualHoverInfo(node, rowHit.rowItem);
        return {
            ...result,
            label: info?.activationData?.label || result.label,
            info
        };
    }

    if (safeKind === 'q') {
        const includeAttentionScorePath = (
            role === 'attention-query-source'
            || role === 'projection-output'
            || role === 'projection-output-copy'
            || role === 'x-ln-copy'
        );
        descriptors.push(includeAttentionScorePath ? 'queryScorePath' : 'queryStage');
        rowSelections.push(...buildQueryRowSelections(index, rowIndex));
        if (includeAttentionScorePath) {
            appendAttentionPostCopyNode(extraNodeIds, index);
            appendAllUnique(extraNodeIds, index?.nodeIdsByRole?.get('attention-head-output') || []);
            rowSelections.push(...buildScoreAxisRowSelectionsWithOptions(index, rowIndex, {
                includePost: true,
                includePostCopy: true
            }));
            rowSelections.push({
                nodeId: index?.singleNodeIds?.attentionHeadOutput,
                rowIndex
            });
        }
    } else if (safeKind === 'k') {
        const scoreAxisSelections = buildScoreAxisSelections(index, rowIndex, {
            includePost: true,
            includePostCopy: true
        });
        descriptors.push('keyScorePath');
        rowSelections.push(...buildKeyRowSelections(index, rowIndex), ...scoreAxisSelections.rowSelections);
        columnSelections.push(...scoreAxisSelections.columnSelections);
    } else if (safeKind === 'v') {
        descriptors.push('valueProjectionPath');
        rowSelections.push(...buildValueRowSelections(index, rowIndex));
        appendAllUnique(extraNodeIds, index?.nodeIdsByRole?.get('attention-head-output') || []);
    }

    if (role === 'x-ln-copy') {
        rowSelections.push(...buildProjectionSourceMirrorRowSelections(index, rowIndex));
    }
    if (role === 'projection-output' || role === 'projection-output-copy') {
        appendUnique(extraNodeIds, projectionOutputCopyNodeId);
        appendUnique(extraNodeIds, cacheNodeId);
        appendUnique(extraNodeIds, cacheSourceNodeId);
        appendUnique(extraConnectorIds, cacheSourceConnectorNodeId);
        appendUnique(extraConnectorIds, findNodeIdByRole(index, `connector-${safeKind}-cache`));
        appendUnique(extraConnectorIds, findNodeIdByRole(index, `connector-${safeKind}-cache-copy`));
    } else if (role === 'projection-cache' || role === 'projection-cache-source') {
        appendUnique(extraNodeIds, projectionOutputCopyNodeId);
        appendUnique(extraNodeIds, cacheNodeId);
        appendUnique(extraNodeIds, cacheSourceNodeId);
        appendUnique(extraConnectorIds, cacheSourceConnectorNodeId);
        appendUnique(extraConnectorIds, findNodeIdByRole(index, `connector-${safeKind}-cache-copy`));
    }

    const result = buildPathFocusResult(index, {
        label: `${resolveProjectionLabel(safeKind)} row`,
        info: createTokenInfo(rowHit.rowItem),
        descriptors,
        extraNodeIds,
        extraConnectorIds,
        dimNodeIds,
        rowSelections,
        columnSelections
    }, options);
    if (role === 'x-ln-copy') {
        const info = buildPostLayerNormResidualHoverInfo(node, rowHit.rowItem);
        return {
            ...result,
            label: info?.activationData?.label || result.label,
            info
        };
    }
    if (
        role === 'projection-output'
        || role === 'projection-output-copy'
        || role === 'attention-query-source'
    ) {
        return {
            ...result,
            label: `${resolveProjectionLabel(safeKind)} Vector`,
            info: buildProjectionVectorHoverInfo(node, rowHit.rowItem, safeKind)
        };
    }
    return result;
}

function buildProjectionCacheRowResult(index = null, node = null, kind = '', rowHit = null, options = null) {
    const safeKind = normalizeProjectionKind(kind);
    if (!index || !node?.id || !safeKind || !rowHit) return null;
    const rowIndex = Number.isFinite(rowHit.rowIndex) ? Math.max(0, Math.floor(rowHit.rowIndex)) : null;
    if (!Number.isFinite(rowIndex)) return null;

    const projectionOutputNodeId = index?.projectionOutputIdsByKind?.[safeKind] || '';
    const projectionOutputCopyNodeId = findProjectionNodeIdByKind(index, 'projection-output-copy', safeKind);
    const cacheNodeId = findProjectionNodeIdByKind(index, 'projection-cache', safeKind);
    const cacheSourceNodeId = findProjectionNodeIdByKind(index, 'projection-cache-source', safeKind);
    const concatResultNodeId = findProjectionNodeIdByKind(index, 'projection-cache-concat-result', safeKind);
    const nextPreviewNodeId = findProjectionNodeIdByKind(index, 'projection-cache-next', safeKind);
    const cacheConnectorNodeId = findNodeIdByRole(index, `connector-${safeKind}-cache`);
    const cacheSourceConnectorNodeId = findNodeIdByRole(index, `connector-${safeKind}-cache-source`);
    const cacheCopyConnectorNodeId = findNodeIdByRole(index, `connector-${safeKind}-cache-copy`);
    const nextPreviewConnectorNodeId = findNodeIdByRole(index, `connector-${safeKind}-cache-next`);
    const terminalConnectorNodeId = findNodeIdByRole(index, `connector-${safeKind}`);
    const preConnectorNodeId = findNodeIdByRole(index, 'connector-pre');
    const postConnectorNodeId = findNodeIdByRole(index, 'connector-post');
    const terminalNodeId = safeKind === 'v'
        ? index?.singleNodeIds?.attentionValuePost || ''
        : '';
    const isDecodeCacheBranch = String(node?.metadata?.kvCachePhase || '').trim().toLowerCase() === 'decode';
    const interactionKind = String(options?.interactionKind || 'hover').trim().toLowerCase();
    const isClickInteraction = interactionKind === 'click';
    const includeDecodeKeyScoreSelections = isDecodeCacheBranch && safeKind === 'k';
    const includeDecodeValueDownstreamSelections = isDecodeCacheBranch && safeKind === 'v';
    const scoreAxisSelections = includeDecodeKeyScoreSelections
        ? (
            isClickInteraction
                ? buildScoreAxisSelections(index, rowIndex, {
                    includePost: true,
                    includePostCopy: true
                })
                : buildScoreAxisSelectionsWithoutTranspose(index, rowIndex, {
                    includePost: true,
                    includePostCopy: true
                })
        )
        : {
            rowSelections: [],
            columnSelections: []
        };

    const activeNodeIds = isDecodeCacheBranch
        ? [
            cacheNodeId,
            cacheSourceNodeId,
            node.id
        ]
        : [
            projectionOutputNodeId,
            projectionOutputCopyNodeId,
            cacheNodeId,
            cacheSourceNodeId,
            node.id
        ];
    const activeConnectorIds = isDecodeCacheBranch
        ? [
            cacheSourceConnectorNodeId,
            cacheConnectorNodeId,
            ...(includeDecodeKeyScoreSelections
                ? [
                    ...(isClickInteraction ? [terminalConnectorNodeId] : []),
                    preConnectorNodeId,
                    postConnectorNodeId
                ]
                : []),
            ...(includeDecodeValueDownstreamSelections
                ? [nextPreviewConnectorNodeId, terminalConnectorNodeId]
                : [])
        ].filter(Boolean)
        : [cacheSourceConnectorNodeId, cacheConnectorNodeId, cacheCopyConnectorNodeId].filter(Boolean);
    const rowSelections = [
        ...(!isDecodeCacheBranch
            ? [{
                nodeId: projectionOutputNodeId,
                rowIndex
            }]
            : []),
        {
            nodeId: cacheNodeId,
            rowIndex
        },
        {
            nodeId: cacheSourceNodeId,
            rowIndex
        },
        ...(isDecodeCacheBranch && concatResultNodeId
            ? [{
                nodeId: concatResultNodeId,
                rowIndex
            }]
            : []),
        ...scoreAxisSelections.rowSelections,
        ...(includeDecodeValueDownstreamSelections
            ? [
                {
                    nodeId: concatResultNodeId,
                    rowIndex
                },
                {
                    nodeId: nextPreviewNodeId,
                    rowIndex
                },
                {
                    nodeId: terminalNodeId,
                    rowIndex
                }
            ]
            : []),
        {
            nodeId: node.id,
            rowIndex
        }
    ];

    return buildFocusResult({
        label: `Cached ${resolveProjectionLabel(safeKind)} Vector`,
        info: buildProjectionVectorHoverInfo(node, rowHit.rowItem, safeKind, {
            cachedKv: true
        }),
        activeNodeIds,
        activeConnectorIds,
        rowSelections,
        columnSelections: scoreAxisSelections.columnSelections,
        includeFocusState: options?.includeFocusState !== false
    });
}

function buildProjectionCacheConcatResultRowResult(index = null, node = null, kind = '', rowHit = null, options = null) {
    const safeKind = normalizeProjectionKind(kind);
    if (!index || !node?.id || !safeKind || !rowHit) return null;
    const displayRowIndex = Number.isFinite(rowHit.rowIndex) ? Math.max(0, Math.floor(rowHit.rowIndex)) : null;
    if (!Number.isFinite(displayRowIndex)) return null;
    const isNextCachePreviewNode = String(node?.role || '').trim() === 'projection-cache-next';

    const concatResultPart = String(rowHit?.rowItem?.semantic?.concatResultPart || '').trim().toLowerCase();
    const isLiveRow = concatResultPart === 'live';
    const sourceRowIndex = isLiveRow ? 0 : displayRowIndex;
    const projectionOutputNodeId = index?.projectionOutputIdsByKind?.[safeKind] || '';
    const projectionOutputCopyNodeId = findProjectionNodeIdByKind(index, 'projection-output-copy', safeKind);
    const cacheNodeId = findProjectionNodeIdByKind(index, 'projection-cache', safeKind);
    const cacheSourceNodeId = findProjectionNodeIdByKind(index, 'projection-cache-source', safeKind);
    const concatResultNodeId = findProjectionNodeIdByKind(index, 'projection-cache-concat-result', safeKind);
    const cacheSourceConnectorNodeId = findNodeIdByRole(index, `connector-${safeKind}-cache-source`);
    const cacheCopyConnectorNodeId = findNodeIdByRole(index, `connector-${safeKind}-cache-copy`);
    const nextPreviewConnectorNodeId = findNodeIdByRole(index, `connector-${safeKind}-cache-next`);
    const terminalConnectorNodeId = findNodeIdByRole(index, `connector-${safeKind}`);
    const preConnectorNodeId = findNodeIdByRole(index, 'connector-pre');
    const postConnectorNodeId = findNodeIdByRole(index, 'connector-post');
    const terminalNodeId = safeKind === 'k'
        ? index?.singleNodeIds?.attentionKeyTranspose || ''
        : (safeKind === 'v' ? index?.singleNodeIds?.attentionValuePost || '' : '');
    const cachedKeyScoreAxisSelections = !isLiveRow && safeKind === 'k'
        ? buildScoreAxisSelections(index, displayRowIndex, {
            includePost: true,
            includePostCopy: true
        })
        : {
            rowSelections: [],
            columnSelections: []
        };

    if (isNextCachePreviewNode) {
        return buildFocusResult({
            label: `Cached ${resolveProjectionLabel(safeKind)} Vector`,
            info: buildProjectionVectorHoverInfo(node, rowHit.rowItem, safeKind, {
                cachedKv: true
            }),
            activeConnectorIds: [
                isLiveRow ? cacheCopyConnectorNodeId : cacheSourceConnectorNodeId,
                nextPreviewConnectorNodeId,
                terminalConnectorNodeId
            ].filter(Boolean),
            dimNodeIds: safeKind === 'v'
                ? buildValueBranchExtraDimNodeIds(index)
                : [],
            columnSelections: safeKind === 'k' && terminalNodeId && terminalNodeId !== node.id
                ? [{
                    nodeId: terminalNodeId,
                    colIndex: displayRowIndex
                }]
                : [],
            rowSelections: [
                ...(isLiveRow
                    ? [
                        ...buildDecodeProjectionCurrentInputRowSelections(index, safeKind, {
                            includeProjectionSource: true,
                            rowIndex: sourceRowIndex
                        }),
                        {
                            nodeId: projectionOutputNodeId,
                            rowIndex: sourceRowIndex
                        },
                        {
                            nodeId: projectionOutputCopyNodeId,
                            rowIndex: sourceRowIndex
                        }
                    ]
                    : [
                        {
                            nodeId: cacheNodeId,
                            rowIndex: sourceRowIndex
                        },
                        {
                            nodeId: cacheSourceNodeId,
                            rowIndex: sourceRowIndex
                        }
                    ]),
                ...(concatResultNodeId && concatResultNodeId !== node.id
                    ? [{
                        nodeId: concatResultNodeId,
                        rowIndex: displayRowIndex
                    }]
                    : []),
                ...(safeKind === 'v' && terminalNodeId && terminalNodeId !== node.id
                    ? [{
                        nodeId: terminalNodeId,
                        rowIndex: displayRowIndex
                    }]
                    : []),
                {
                    nodeId: node.id,
                    rowIndex: displayRowIndex
                }
            ],
            includeFocusState: options?.includeFocusState !== false
        });
    }

    const activeNodeIds = isLiveRow
        ? [
            projectionOutputNodeId,
            projectionOutputCopyNodeId,
            terminalNodeId,
            node.id
        ]
        : [
            cacheNodeId,
            cacheSourceNodeId,
            terminalNodeId,
            node.id
        ];

    return buildFocusResult({
        label: `${isLiveRow ? '' : 'Cached '}${resolveProjectionLabel(safeKind)} Vector`,
        info: buildProjectionVectorHoverInfo(node, rowHit.rowItem, safeKind, {
            cachedKv: !isLiveRow
        }),
        activeNodeIds,
        activeConnectorIds: [
            isLiveRow ? cacheCopyConnectorNodeId : cacheSourceConnectorNodeId,
            terminalConnectorNodeId,
            ...(!isLiveRow && safeKind === 'k'
                ? [preConnectorNodeId, postConnectorNodeId]
                : [])
        ].filter(Boolean),
        dimNodeIds: safeKind === 'v'
            ? buildValueBranchExtraDimNodeIds(index)
            : [],
        columnSelections: isLiveRow
            ? (safeKind === 'k' && terminalNodeId && terminalNodeId !== node.id
                ? [{
                    nodeId: terminalNodeId,
                    colIndex: displayRowIndex
                }]
                : [])
            : cachedKeyScoreAxisSelections.columnSelections,
        rowSelections: [
            ...(isLiveRow
                ? [
                    ...buildDecodeProjectionCurrentInputRowSelections(index, safeKind, {
                        includeProjectionSource: true,
                        rowIndex: sourceRowIndex
                    }),
                    {
                        nodeId: projectionOutputNodeId,
                        rowIndex: sourceRowIndex
                    },
                    {
                        nodeId: projectionOutputCopyNodeId,
                        rowIndex: sourceRowIndex
                    }
                ]
                : [
                    {
                        nodeId: cacheNodeId,
                        rowIndex: sourceRowIndex
                    },
                    {
                        nodeId: cacheSourceNodeId,
                        rowIndex: sourceRowIndex
                    }
                ]),
            ...cachedKeyScoreAxisSelections.rowSelections,
            ...(safeKind === 'v' && terminalNodeId && terminalNodeId !== node.id
                ? [{
                    nodeId: terminalNodeId,
                    rowIndex: displayRowIndex
                }]
                : []),
            {
                nodeId: node.id,
                rowIndex: displayRowIndex
            }
        ],
        includeFocusState: options?.includeFocusState !== false
    });
}

function buildProjectionCacheStageResult(index = null, node = null, kind = '', options = null) {
    const safeKind = normalizeProjectionKind(kind);
    if (!index || !node?.id || !safeKind) return null;

    const projectionOutputNodeId = index?.projectionOutputIdsByKind?.[safeKind] || '';
    const projectionOutputCopyNodeId = findProjectionNodeIdByKind(index, 'projection-output-copy', safeKind);
    const cacheNodeId = findProjectionNodeIdByKind(index, 'projection-cache', safeKind);
    const cacheSourceNodeId = findProjectionNodeIdByKind(index, 'projection-cache-source', safeKind);
    const cacheConnectorNodeId = findNodeIdByRole(index, `connector-${safeKind}-cache`);
    const cacheSourceConnectorNodeId = findNodeIdByRole(index, `connector-${safeKind}-cache-source`);
    const cacheCopyConnectorNodeId = findNodeIdByRole(index, `connector-${safeKind}-cache-copy`);
    const representativeRowItem = Array.isArray(node.rowItems) && node.rowItems.length
        ? node.rowItems[0]
        : null;

    return buildFocusResult({
        label: `Cached ${resolveProjectionLabel(safeKind)} Vector`,
        info: buildProjectionVectorHoverInfo(node, representativeRowItem, safeKind, {
            cachedKv: true
        }),
        activeNodeIds: [
            projectionOutputNodeId,
            projectionOutputCopyNodeId,
            cacheNodeId,
            cacheSourceNodeId,
            node.id
        ],
        activeConnectorIds: [cacheSourceConnectorNodeId, cacheConnectorNodeId, cacheCopyConnectorNodeId].filter(Boolean),
        includeFocusState: options?.includeFocusState !== false
    });
}

function buildTransposeAxisResult(index = null, axisHit = null, options = null) {
    if (!index || !axisHit) return null;
    const axisIndex = Number.isFinite(axisHit.rowIndex)
        ? Math.max(0, Math.floor(axisHit.rowIndex))
        : (Number.isFinite(axisHit.colIndex) ? Math.max(0, Math.floor(axisHit.colIndex)) : null);
    const scoreAxisSelections = buildScoreAxisSelections(index, axisIndex, {
        includePost: true,
        includePostCopy: true
    });
    const rowSelections = [
        ...buildKeyRowSelections(index, axisIndex),
        ...scoreAxisSelections.rowSelections
    ];
    const columnSelections = scoreAxisSelections.columnSelections;

    return buildPathFocusResult(index, {
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
        descriptors: ['keyScorePath'],
        rowSelections,
        columnSelections
    }, options);
}

function buildWeightedOutputRowResult(index = null, node = null, rowHit = null, {
    includeProjection = false,
    label = 'Head output row',
    info = null
} = {}, options = null) {
    if (!index || !rowHit) return null;
    const rowIndex = Number.isFinite(rowHit.rowIndex) ? Math.max(0, Math.floor(rowHit.rowIndex)) : null;
    const dimNodeIds = includeProjection
        ? buildValueBranchExtraDimNodeIds(index)
        : [];
    const extraNodeIds = includeProjection
        ? [...(index?.nodeIdsByRole?.get('attention-head-output') || [])]
        : [];
    const rowSelections = includeProjection
        ? [{ nodeId: index.singleNodeIds.attentionValuePost, rowIndex }]
        : [
            { nodeId: index.singleNodeIds.attentionPost, rowIndex },
            { nodeId: index.singleNodeIds.attentionPostCopy, rowIndex },
            { nodeId: index.singleNodeIds.attentionHeadOutput, rowIndex }
        ];

    if (includeProjection) {
        rowSelections.push(...buildValueRowSelections(index, rowIndex));
    }

    return buildPathFocusResult(index, {
        label,
        info: info || createTokenInfo(rowHit.rowItem),
        descriptors: includeProjection
            ? ['valueProjectionPath']
            : ['weightedOutputOnly'],
        extraNodeIds,
        dimNodeIds,
        rowSelections
    }, options);
}

function findOutputProjectionHeadConnectorId(index = null, headIndex = null) {
    const safeHeadIndex = Number.isFinite(headIndex) ? Math.max(0, Math.floor(headIndex)) : null;
    if (!index?.nodesById || !Number.isFinite(safeHeadIndex)) return '';
    for (const node of index.nodesById.values()) {
        if (node?.role !== 'head-output-connector') continue;
        const nodeHeadIndex = Number.isFinite(node?.semantic?.headIndex)
            ? Math.max(0, Math.floor(node.semantic.headIndex))
            : null;
        if (nodeHeadIndex === safeHeadIndex) {
            return node.id;
        }
    }
    return '';
}

function findOutputProjectionConcatCopyConnectorId(index = null, headIndex = null) {
    const safeHeadIndex = Number.isFinite(headIndex) ? Math.max(0, Math.floor(headIndex)) : null;
    if (!index?.nodesById || !Number.isFinite(safeHeadIndex)) return '';
    for (const node of index.nodesById.values()) {
        if (node?.role !== 'concat-copy-connector') continue;
        const nodeHeadIndex = Number.isFinite(node?.semantic?.headIndex)
            ? Math.max(0, Math.floor(node.semantic.headIndex))
            : null;
        if (nodeHeadIndex === safeHeadIndex) {
            return node.id;
        }
    }
    return '';
}

function findOutputProjectionHeadMatrixIds(index = null, headIndex = null) {
    const safeHeadIndex = Number.isFinite(headIndex) ? Math.max(0, Math.floor(headIndex)) : null;
    if (!index?.nodesById || !Number.isFinite(safeHeadIndex)) {
        return {
            sourceNodeId: '',
            copyNodeId: ''
        };
    }
    let sourceNodeId = '';
    let copyNodeId = '';
    for (const node of index.nodesById.values()) {
        const nodeRole = String(node?.role || '').trim();
        if (nodeRole !== 'head-output-matrix' && nodeRole !== 'concat-head-copy-matrix') continue;
        const nodeHeadIndex = Number.isFinite(node?.semantic?.headIndex)
            ? Math.max(0, Math.floor(node.semantic.headIndex))
            : null;
        if (nodeHeadIndex !== safeHeadIndex) continue;
        if (nodeRole === 'head-output-matrix') {
            sourceNodeId = node.id;
        } else if (nodeRole === 'concat-head-copy-matrix') {
            copyNodeId = node.id;
        }
    }
    return {
        sourceNodeId,
        copyNodeId
    };
}

function findOutputProjectionConcatOutputNodeId(index = null) {
    return (index?.nodeIdsByRole?.get('concat-output-matrix') || [])[0] || '';
}

function findOutputProjectionConcatCopyNodeId(index = null) {
    return (index?.nodeIdsByRole?.get('concat-output-copy-matrix') || [])[0] || '';
}

function findOutputProjectionProjectedOutputNodeId(index = null) {
    return (index?.nodeIdsByRole?.get('projection-output') || [])[0] || '';
}

function collectOutputProjectionHeadMatrixNodeIds(index = null) {
    const nodeIds = [];
    appendAllUnique(nodeIds, index?.nodeIdsByRole?.get('head-output-matrix') || []);
    appendAllUnique(nodeIds, index?.nodeIdsByRole?.get('concat-head-copy-matrix') || []);
    return nodeIds;
}

function collectOutputProjectionConnectorIds(index = null) {
    const connectorIds = [];
    if (!index?.nodesById) return connectorIds;
    for (const node of index.nodesById.values()) {
        if (node?.role !== 'head-output-connector' && node?.role !== 'concat-copy-connector') continue;
        appendUnique(connectorIds, node.id);
    }
    return connectorIds;
}

function collectOutputProjectionEquationConnectorIds(index = null) {
    const connectorIds = [];
    if (!index?.nodesById) return connectorIds;
    for (const node of index.nodesById.values()) {
        if (
            node?.role !== 'concat-output-projection-connector'
            && node?.role !== 'projection-output-connector'
        ) {
            continue;
        }
        appendUnique(connectorIds, node.id);
    }
    return connectorIds;
}

function buildAttentionOutputProjectionHoverInfo(node = null, rowItem = null, {
    label = 'Attention Output Vector',
    extraActivationData = {}
} = {}) {
    const tokenInfo = createTokenInfo(rowItem) || {};
    const extraActivationDataObject = extraActivationData && typeof extraActivationData === 'object'
        ? extraActivationData
        : {};
    const explicitValues = Array.isArray(extraActivationDataObject?.values) || ArrayBuffer.isView(extraActivationDataObject?.values)
        ? Array.from(extraActivationDataObject.values).map((value) => (Number.isFinite(value) ? value : 0))
        : null;
    const rowValues = Array.isArray(rowItem?.rawValues) || ArrayBuffer.isView(rowItem?.rawValues)
        ? Array.from(rowItem.rawValues).map((value) => (Number.isFinite(value) ? value : 0))
        : null;
    const values = explicitValues?.length ? explicitValues : rowValues;
    const stage = typeof extraActivationDataObject.stage === 'string' && extraActivationDataObject.stage.trim().length
        ? extraActivationDataObject.stage.trim()
        : 'attention.output_projection';
    const info = buildProjectionHoverInfo(node, label, {
        stage,
        ...(values?.length ? { values } : {}),
        ...extraActivationDataObject,
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
            stage,
            ...(values?.length ? { values } : {}),
            ...extraActivationDataObject,
            ...(Number.isFinite(tokenInfo.tokenIndex) ? { tokenIndex: tokenInfo.tokenIndex } : {}),
            ...(typeof tokenInfo.tokenLabel === 'string' && tokenInfo.tokenLabel.length
                ? { tokenLabel: tokenInfo.tokenLabel }
                : {})
        }
    };
}

function buildOutputProjectionEquationResult(index = null, {
    label = '',
    info = null,
    rowIndex = null,
    includeOutputRow = false,
    extraRowSelections = []
} = {}, options = null) {
    if (!index) return null;
    const concatOutputNodeId = findOutputProjectionConcatOutputNodeId(index);
    const concatCopyNodeId = findOutputProjectionConcatCopyNodeId(index);
    const outputNodeId = findOutputProjectionProjectedOutputNodeId(index);
    const weightNodeId = (index?.nodeIdsByRole?.get('projection-weight') || [])[0] || '';
    const biasNodeId = (index?.nodeIdsByRole?.get('projection-bias') || [])[0] || '';
    const equationRowSelections = Number.isFinite(rowIndex)
        ? [
            ...(concatOutputNodeId ? [{ nodeId: concatOutputNodeId, rowIndex }] : []),
            ...(concatCopyNodeId ? [{ nodeId: concatCopyNodeId, rowIndex }] : []),
            ...(includeOutputRow && outputNodeId ? [{ nodeId: outputNodeId, rowIndex }] : []),
            ...(Array.isArray(extraRowSelections) ? extraRowSelections : [])
        ]
        : [];

    return buildPathFocusResult(index, {
        label,
        info,
        extraNodeIds: [
            concatOutputNodeId,
            concatCopyNodeId,
            weightNodeId,
            biasNodeId,
            outputNodeId
        ],
        extraConnectorIds: collectOutputProjectionEquationConnectorIds(index),
        rowSelections: equationRowSelections
    }, options);
}

function buildOutputProjectionHeadOutputRowResult(index = null, node = null, rowHit = null, options = null) {
    if (!index || !node || !rowHit) return null;
    const rowIndex = Number.isFinite(rowHit.rowIndex) ? Math.max(0, Math.floor(rowHit.rowIndex)) : null;
    if (!Number.isFinite(rowIndex)) return null;
    const headIndex = Number.isFinite(node?.semantic?.headIndex)
        ? Math.max(0, Math.floor(node.semantic.headIndex))
        : null;
    const concatOutputNodeId = findOutputProjectionConcatOutputNodeId(index);
    const concatOutputCopyNodeId = findOutputProjectionConcatCopyNodeId(index);
    const connectorId = findOutputProjectionHeadConnectorId(index, headIndex);
    const concatCopyConnectorId = findOutputProjectionConcatCopyConnectorId(index, headIndex);
    const {
        sourceNodeId,
        copyNodeId
    } = findOutputProjectionHeadMatrixIds(index, headIndex);
    return buildPathFocusResult(index, {
        label: 'Attention Weighted Sum',
        info: buildWeightedSumHoverInfo(node, rowHit.rowItem),
        extraConnectorIds: [
            ...(connectorId ? [connectorId] : []),
            ...(concatCopyConnectorId ? [concatCopyConnectorId] : [])
        ],
        rowSelections: [
            ...(sourceNodeId ? [{
                nodeId: sourceNodeId,
                rowIndex
            }] : []),
            ...(copyNodeId ? [{
                nodeId: copyNodeId,
                rowIndex
            }] : []),
            ...(concatOutputNodeId ? [{
                nodeId: concatOutputNodeId,
                rowIndex
            }] : []),
            ...(concatOutputCopyNodeId ? [{
                nodeId: concatOutputCopyNodeId,
                rowIndex
            }] : [])
        ],
        cellSelections: [
            ...(concatOutputNodeId && Number.isFinite(headIndex)
                ? [{
                    nodeId: concatOutputNodeId,
                    rowIndex,
                    colIndex: headIndex
                }]
                : []),
            ...(concatOutputCopyNodeId && Number.isFinite(headIndex)
                ? [{
                    nodeId: concatOutputCopyNodeId,
                    rowIndex,
                    colIndex: headIndex
                }]
                : [])
        ]
    }, options);
}

function buildOutputProjectionConcatOutputRowResult(index = null, node = null, rowHit = null, options = null) {
    if (!index || !node || !rowHit) return null;
    const rowIndex = Number.isFinite(rowHit.rowIndex) ? Math.max(0, Math.floor(rowHit.rowIndex)) : null;
    if (!Number.isFinite(rowIndex)) return null;
    const headMatrixNodeIds = collectOutputProjectionHeadMatrixNodeIds(index);
    const concatOutputNodeId = findOutputProjectionConcatOutputNodeId(index) || node.id;
    const concatCopyNodeId = findOutputProjectionConcatCopyNodeId(index);
    return buildPathFocusResult(index, {
        label: 'Concatenated Head Output',
        info: buildConcatenatedHeadOutputHoverInfo(node, rowHit.rowItem),
        extraConnectorIds: collectOutputProjectionConnectorIds(index),
        rowSelections: [
            ...headMatrixNodeIds.map((nodeId) => ({
                nodeId,
                rowIndex
            })),
            ...(concatOutputNodeId ? [{
                nodeId: concatOutputNodeId,
                rowIndex
            }] : []),
            ...(concatCopyNodeId ? [{
                nodeId: concatCopyNodeId,
                rowIndex
            }] : [])
        ]
    }, options);
}

function buildOutputProjectionConcatOutputBandResult(index = null, node = null, cellHit = null, options = null) {
    if (!index || !node || !cellHit) return null;
    const rowIndex = Number.isFinite(cellHit.rowIndex) ? Math.max(0, Math.floor(cellHit.rowIndex)) : null;
    const headIndex = Number.isFinite(cellHit.colIndex) ? Math.max(0, Math.floor(cellHit.colIndex)) : null;
    if (!Number.isFinite(rowIndex) || !Number.isFinite(headIndex)) return null;
    const connectorId = findOutputProjectionHeadConnectorId(index, headIndex);
    const concatCopyConnectorId = findOutputProjectionConcatCopyConnectorId(index, headIndex);
    const {
        sourceNodeId,
        copyNodeId
    } = findOutputProjectionHeadMatrixIds(index, headIndex);
    const concatOutputNodeId = findOutputProjectionConcatOutputNodeId(index);
    const concatCopyNodeId = findOutputProjectionConcatCopyNodeId(index);
    return buildPathFocusResult(index, {
        label: 'Attention Weighted Sum',
        info: buildConcatenatedHeadOutputBandHoverInfo(
            node,
            cellHit?.rowItem || cellHit?.cellItem?.rowItem || null,
            headIndex
        ),
        extraConnectorIds: [
            ...(connectorId ? [connectorId] : []),
            ...(concatCopyConnectorId ? [concatCopyConnectorId] : [])
        ],
        rowSelections: [
            ...(sourceNodeId ? [{
                nodeId: sourceNodeId,
                rowIndex
            }] : []),
            ...(copyNodeId ? [{
                nodeId: copyNodeId,
                rowIndex
            }] : [])
        ],
        cellSelections: [
            ...(concatOutputNodeId ? [{
                nodeId: concatOutputNodeId,
                rowIndex,
                colIndex: headIndex
            }] : []),
            ...(concatCopyNodeId ? [{
                nodeId: concatCopyNodeId,
                rowIndex,
                colIndex: headIndex
            }] : [])
        ]
    }, options);
}

function buildOutputProjectionWeightResult(index = null, node = null, options = null) {
    return buildOutputProjectionEquationResult(index, {
        label: 'Output Projection Matrix',
        info: buildAttentionOutputProjectionHoverInfo(node, null, {
            label: 'Output Projection Matrix',
            extraActivationData: {
                parameterType: 'weight'
            }
        })
    }, options);
}

function buildOutputProjectionBiasResult(index = null, node = null, rowHit = null, options = null) {
    const biasRowItem = rowHit?.rowItem || node?.rowItems?.[0] || null;
    const values = Array.isArray(biasRowItem?.rawValues) || ArrayBuffer.isView(biasRowItem?.rawValues)
        ? Array.from(biasRowItem.rawValues).map((value) => (Number.isFinite(value) ? value : 0))
        : null;
    return buildOutputProjectionEquationResult(index, {
        label: 'Output Projection Bias Vector',
        info: buildAttentionOutputProjectionHoverInfo(node, biasRowItem, {
            label: 'Output Projection Bias Vector',
            extraActivationData: {
                parameterType: 'bias',
                stage: 'attention.output_projection.bias',
                ...(values?.length ? { values } : {})
            }
        })
    }, options);
}

function buildOutputProjectionOutputRowResult(index = null, node = null, rowHit = null, options = null) {
    if (!index || !node || !rowHit) return null;
    const rowIndex = Number.isFinite(rowHit.rowIndex) ? Math.max(0, Math.floor(rowHit.rowIndex)) : null;
    if (!Number.isFinite(rowIndex)) return null;
    const headOutputMatrixNodeIds = collectOutputProjectionHeadMatrixNodeIds(index);
    const headOutputMatrixRowSelections = headOutputMatrixNodeIds.map((nodeId) => ({
        nodeId,
        rowIndex
    }));
    return buildOutputProjectionEquationResult(index, {
        label: 'Attention Output Vector',
        info: buildAttentionOutputProjectionHoverInfo(node, rowHit.rowItem, {
            label: 'Attention Output Vector',
            extraActivationData: {
                vectorCategory: 'attention-output'
            }
        }),
        rowIndex,
        includeOutputRow: true,
        extraRowSelections: headOutputMatrixRowSelections
    }, options);
}

function buildProjectionSourceRowResult(index = null, rowHit = null, options = null) {
    if (!index || !rowHit) return null;
    const rowIndex = Number.isFinite(rowHit.rowIndex) ? Math.max(0, Math.floor(rowHit.rowIndex)) : null;
    const interactionKind = String(options?.interactionKind || 'hover').trim().toLowerCase();
    const isClickInteraction = interactionKind === 'click';
    const isDecodeClickInteraction = isClickInteraction && !!index?.scene?.metadata?.kvCacheState?.kvCacheDecodeActive;
    const decodeKeyLiveDisplayRowIndex = isDecodeClickInteraction
        ? resolveDecodeProjectionLiveDisplayRowIndex(index, 'k')
        : null;
    const decodeValueLiveDisplayRowIndex = isDecodeClickInteraction
        ? resolveDecodeProjectionLiveDisplayRowIndex(index, 'v')
        : null;
    const keyProjectionOutputNodeId = index?.projectionOutputIdsByKind?.k || '';
    const keyProjectionOutputCopyNodeId = findProjectionNodeIdByKind(index, 'projection-output-copy', 'k');
    const keyConcatResultNodeId = findProjectionNodeIdByKind(index, 'projection-cache-concat-result', 'k');
    const keyNextPreviewNodeId = findProjectionNodeIdByKind(index, 'projection-cache-next', 'k');
    const valueProjectionOutputNodeId = index?.projectionOutputIdsByKind?.v || '';
    const valueProjectionOutputCopyNodeId = findProjectionNodeIdByKind(index, 'projection-output-copy', 'v');
    const valueConcatResultNodeId = findProjectionNodeIdByKind(index, 'projection-cache-concat-result', 'v');
    const valueNextPreviewNodeId = findProjectionNodeIdByKind(index, 'projection-cache-next', 'v');
    const decodeExtraConnectorIds = isDecodeClickInteraction
        ? [
            findNodeIdByRole(index, 'connector-k'),
            findNodeIdByRole(index, 'connector-k-cache-copy'),
            findNodeIdByRole(index, 'connector-k-cache-next'),
            findNodeIdByRole(index, 'connector-v'),
            findNodeIdByRole(index, 'connector-v-cache-copy'),
            findNodeIdByRole(index, 'connector-v-cache-next')
        ].filter(Boolean)
        : [];
    const transposeSelections = isClickInteraction
        ? buildAttentionKeyTransposeSelections(
            index,
            Number.isFinite(decodeKeyLiveDisplayRowIndex) ? decodeKeyLiveDisplayRowIndex : rowIndex
        )
        : {
            rowSelections: [],
            columnSelections: []
        };
    const mlpWeightNodeId = (index?.nodeIdsByRole?.get('mlp-up-weight') || [])[0] || '';
    const mlpBiasNodeId = (index?.nodeIdsByRole?.get('mlp-up-bias') || [])[0] || '';
    const mlpOutputNodeId = (index?.nodeIdsByRole?.get('mlp-up-output') || [])[0] || '';
    const mlpOutputCopyNodeId = (index?.nodeIdsByRole?.get('mlp-up-output-copy') || [])[0] || '';
    const mlpActivationNodeId = (index?.nodeIdsByRole?.get('mlp-activation-output') || [])[0] || '';
    const mlpActivationCopyNodeId = (index?.nodeIdsByRole?.get('mlp-activation-output-copy') || [])[0] || '';
    const mlpDownOutputNodeId = (index?.nodeIdsByRole?.get('mlp-down-output') || [])[0] || '';
    const info = buildPostLayerNormResidualHoverInfo(
        index?.nodesById?.get(index?.singleNodeIds?.projectionSourceXln || ''),
        rowHit.rowItem
    );
    return buildPathFocusResult(index, {
        label: info?.activationData?.label || resolvePostLayerNormResidualLabel({
            stage: info?.activationData?.stage || rowHit?.rowItem?.semantic?.stage || ''
        }),
        info,
        descriptors: ['projectionSourceAll'],
        extraNodeIds: [mlpWeightNodeId, mlpBiasNodeId],
        extraConnectorIds: decodeExtraConnectorIds,
        rowSelections: [
            ...buildProjectionSourceRowSelections(index, rowIndex),
            ...(mlpOutputNodeId ? [{ nodeId: mlpOutputNodeId, rowIndex }] : []),
            ...(mlpOutputCopyNodeId ? [{ nodeId: mlpOutputCopyNodeId, rowIndex }] : []),
            ...(mlpActivationNodeId ? [{ nodeId: mlpActivationNodeId, rowIndex }] : []),
            ...(mlpActivationCopyNodeId ? [{ nodeId: mlpActivationCopyNodeId, rowIndex }] : []),
            ...(mlpDownOutputNodeId ? [{ nodeId: mlpDownOutputNodeId, rowIndex }] : []),
            ...buildQueryRowSelections(index, rowIndex),
            ...(isDecodeClickInteraction
                ? [
                    ...(keyProjectionOutputNodeId
                        ? [{
                            nodeId: keyProjectionOutputNodeId,
                            rowIndex
                        }]
                        : []),
                    ...(keyProjectionOutputCopyNodeId
                        ? [{
                            nodeId: keyProjectionOutputCopyNodeId,
                            rowIndex
                        }]
                        : []),
                    ...(Number.isFinite(decodeKeyLiveDisplayRowIndex) && keyConcatResultNodeId
                        ? [{
                            nodeId: keyConcatResultNodeId,
                            rowIndex: decodeKeyLiveDisplayRowIndex
                        }]
                        : []),
                    ...(Number.isFinite(decodeKeyLiveDisplayRowIndex) && keyNextPreviewNodeId
                        ? [{
                            nodeId: keyNextPreviewNodeId,
                            rowIndex: decodeKeyLiveDisplayRowIndex
                        }]
                        : [])
                ]
                : buildKeyRowSelections(index, rowIndex)),
            ...(isDecodeClickInteraction
                ? [
                    ...(valueProjectionOutputNodeId
                        ? [{
                            nodeId: valueProjectionOutputNodeId,
                            rowIndex
                        }]
                        : []),
                    ...(valueProjectionOutputCopyNodeId
                        ? [{
                            nodeId: valueProjectionOutputCopyNodeId,
                            rowIndex
                        }]
                        : []),
                    ...(Number.isFinite(decodeValueLiveDisplayRowIndex) && valueConcatResultNodeId
                        ? [{
                            nodeId: valueConcatResultNodeId,
                            rowIndex: decodeValueLiveDisplayRowIndex
                        }]
                        : []),
                    ...(Number.isFinite(decodeValueLiveDisplayRowIndex) && valueNextPreviewNodeId
                        ? [{
                            nodeId: valueNextPreviewNodeId,
                            rowIndex: decodeValueLiveDisplayRowIndex
                        }]
                        : []),
                    ...(Number.isFinite(decodeValueLiveDisplayRowIndex) && index?.singleNodeIds?.attentionValuePost
                        ? [{
                            nodeId: index.singleNodeIds.attentionValuePost,
                            rowIndex: decodeValueLiveDisplayRowIndex
                        }]
                        : [])
                ]
                : buildValueRowSelections(index, rowIndex)),
            ...transposeSelections.rowSelections
        ],
        columnSelections: transposeSelections.columnSelections
    }, options);
}

function buildLayerNormInputRowResult(index = null, node = null, rowHit = null, options = null) {
    if (!index || !node || !rowHit) return null;
    const rowIndex = Number.isFinite(rowHit.rowIndex) ? Math.max(0, Math.floor(rowHit.rowIndex)) : null;
    const info = buildLayerNormActivationHoverInfo(node, rowHit.rowItem, {
        variant: 'input'
    });
    return buildLayerNormDirectFocusResult(index, {
        label: info?.activationData?.label || 'LayerNorm Input Vector',
        info,
        hoveredRole: String(node.role || '').trim(),
        rowIndex
    }, options);
}

function buildLayerNormNormalizedRowResult(index = null, node = null, rowHit = null, options = null) {
    if (!index || !node || !rowHit) return null;
    const rowIndex = Number.isFinite(rowHit.rowIndex) ? Math.max(0, Math.floor(rowHit.rowIndex)) : null;
    const info = buildLayerNormActivationHoverInfo(node, rowHit.rowItem, {
        variant: 'normalized'
    });
    return buildLayerNormDirectFocusResult(index, {
        label: info?.activationData?.label || 'LayerNorm Normalized Vector',
        info,
        hoveredRole: String(node.role || '').trim(),
        rowIndex
    }, options);
}

function buildLayerNormScaledRowResult(index = null, node = null, rowHit = null, options = null) {
    if (!index || !node || !rowHit) return null;
    const rowIndex = Number.isFinite(rowHit.rowIndex) ? Math.max(0, Math.floor(rowHit.rowIndex)) : null;
    const info = buildLayerNormActivationHoverInfo(node, rowHit.rowItem, {
        variant: 'scaled'
    });
    return buildLayerNormDirectFocusResult(index, {
        label: info?.activationData?.label || 'LayerNorm Product Vector',
        info,
        hoveredRole: String(node.role || '').trim(),
        rowIndex
    }, options);
}

function buildLayerNormOutputRowResult(index = null, node = null, rowHit = null, options = null) {
    if (!index || !node || !rowHit) return null;
    const rowIndex = Number.isFinite(rowHit.rowIndex) ? Math.max(0, Math.floor(rowHit.rowIndex)) : null;
    const info = buildLayerNormActivationHoverInfo(node, rowHit.rowItem, {
        variant: 'output'
    });
    return buildLayerNormDirectFocusResult(index, {
        label: info?.activationData?.label || 'LayerNorm Output Vector',
        info,
        hoveredRole: String(node.role || '').trim(),
        rowIndex
    }, options);
}

function buildLayerNormParamResult(index = null, node = null, {
    param = 'scale'
} = {}, options = null) {
    if (!index || !node) return null;
    const info = buildLayerNormParamHoverInfo(node, { param });
    return buildLayerNormDirectFocusResult(index, {
        label: info?.activationData?.label || 'LayerNorm Parameter',
        info,
        hoveredRole: String(node.role || '').trim(),
        selfRowIndex: 0
    }, options);
}

function buildMlpUpOutputRowResult(index = null, node = null, rowHit = null, options = null) {
    if (!index || !node || !rowHit) return null;
    const rowIndex = Number.isFinite(rowHit.rowIndex) ? Math.max(0, Math.floor(rowHit.rowIndex)) : null;
    if (!Number.isFinite(rowIndex)) return null;
    const info = buildMlpUpProjectionHoverInfo(node, rowHit.rowItem);
    const inputNodeId = index?.singleNodeIds?.projectionSourceXln || '';
    const weightNodeId = (index?.nodeIdsByRole?.get('mlp-up-weight') || [])[0] || '';
    const biasNodeId = (index?.nodeIdsByRole?.get('mlp-up-bias') || [])[0] || '';
    const outputNodeId = (index?.nodeIdsByRole?.get('mlp-up-output') || [])[0] || '';
    const outputCopyNodeId = (index?.nodeIdsByRole?.get('mlp-up-output-copy') || [])[0] || '';
    const activationNodeId = (index?.nodeIdsByRole?.get('mlp-activation-output') || [])[0] || '';
    const activationCopyNodeId = (index?.nodeIdsByRole?.get('mlp-activation-output-copy') || [])[0] || '';
    return buildPathFocusResult(index, {
        label: info?.activationData?.label || 'MLP Up Projection',
        info,
        extraNodeIds: [
            inputNodeId,
            weightNodeId,
            biasNodeId,
            outputNodeId,
            outputCopyNodeId,
            activationNodeId,
            activationCopyNodeId
        ],
        rowSelections: [
            ...(outputNodeId ? [{ nodeId: outputNodeId, rowIndex }] : []),
            ...(outputCopyNodeId ? [{ nodeId: outputCopyNodeId, rowIndex }] : []),
            ...(inputNodeId ? [{ nodeId: inputNodeId, rowIndex }] : []),
            ...(activationNodeId ? [{ nodeId: activationNodeId, rowIndex }] : []),
            ...(activationCopyNodeId ? [{ nodeId: activationCopyNodeId, rowIndex }] : [])
        ]
    }, options);
}

function buildMlpUpWeightResult(index = null, node = null, options = null) {
    if (!index || !node) return null;
    const inputNodeId = index?.singleNodeIds?.projectionSourceXln || '';
    const biasNodeId = (index?.nodeIdsByRole?.get('mlp-up-bias') || [])[0] || '';
    const outputNodeId = (index?.nodeIdsByRole?.get('mlp-up-output') || [])[0] || '';
    const outputCopyNodeId = (index?.nodeIdsByRole?.get('mlp-up-output-copy') || [])[0] || '';
    return buildPathFocusResult(index, {
        label: 'MLP Up Weight Matrix',
        info: buildMlpUpWeightHoverInfo(node),
        extraNodeIds: [node.id, inputNodeId, biasNodeId, outputNodeId, outputCopyNodeId]
    }, options);
}

function buildMlpActivationRowResult(index = null, node = null, rowHit = null, options = null) {
    if (!index || !node || !rowHit) return null;
    const rowIndex = Number.isFinite(rowHit.rowIndex) ? Math.max(0, Math.floor(rowHit.rowIndex)) : null;
    if (!Number.isFinite(rowIndex)) return null;
    const info = buildMlpActivationHoverInfo(node, rowHit.rowItem);
    const inputNodeId = index?.singleNodeIds?.projectionSourceXln || '';
    const outputNodeId = (index?.nodeIdsByRole?.get('mlp-up-output') || [])[0] || '';
    const outputCopyNodeId = (index?.nodeIdsByRole?.get('mlp-up-output-copy') || [])[0] || '';
    const activationNodeId = (index?.nodeIdsByRole?.get('mlp-activation-output') || [])[0] || '';
    const activationCopyNodeId = (index?.nodeIdsByRole?.get('mlp-activation-output-copy') || [])[0] || '';
    const downWeightNodeId = (index?.nodeIdsByRole?.get('mlp-down-weight') || [])[0] || '';
    const downBiasNodeId = (index?.nodeIdsByRole?.get('mlp-down-bias') || [])[0] || '';
    const downOutputNodeId = (index?.nodeIdsByRole?.get('mlp-down-output') || [])[0] || '';
    return buildPathFocusResult(index, {
        label: info?.activationData?.label || MLP_ACTIVATION_TOOLTIP_LABEL,
        info,
        extraNodeIds: [
            inputNodeId,
            outputNodeId,
            outputCopyNodeId,
            activationNodeId,
            activationCopyNodeId,
            downWeightNodeId,
            downBiasNodeId,
            downOutputNodeId
        ],
        rowSelections: [
            ...(inputNodeId ? [{ nodeId: inputNodeId, rowIndex }] : []),
            ...(outputNodeId ? [{ nodeId: outputNodeId, rowIndex }] : []),
            ...(outputCopyNodeId ? [{ nodeId: outputCopyNodeId, rowIndex }] : []),
            ...(activationNodeId ? [{ nodeId: activationNodeId, rowIndex }] : []),
            ...(activationCopyNodeId ? [{ nodeId: activationCopyNodeId, rowIndex }] : []),
            ...(downOutputNodeId ? [{ nodeId: downOutputNodeId, rowIndex }] : [])
        ]
    }, options);
}

function buildMlpUpBiasResult(index = null, node = null, rowHit = null, options = null) {
    if (!index || !node) return null;
    const inputNodeId = index?.singleNodeIds?.projectionSourceXln || '';
    const weightNodeId = (index?.nodeIdsByRole?.get('mlp-up-weight') || [])[0] || '';
    const outputNodeId = (index?.nodeIdsByRole?.get('mlp-up-output') || [])[0] || '';
    const outputCopyNodeId = (index?.nodeIdsByRole?.get('mlp-up-output-copy') || [])[0] || '';
    return buildPathFocusResult(index, {
        label: MLP_UP_BIAS_TOOLTIP_LABEL,
        info: buildMlpUpBiasHoverInfo(node, rowHit?.rowItem),
        extraNodeIds: [node.id, inputNodeId, weightNodeId, outputNodeId, outputCopyNodeId]
    }, options);
}

function buildMlpDownOutputRowResult(index = null, node = null, rowHit = null, options = null) {
    if (!index || !node || !rowHit) return null;
    const rowIndex = Number.isFinite(rowHit.rowIndex) ? Math.max(0, Math.floor(rowHit.rowIndex)) : null;
    if (!Number.isFinite(rowIndex)) return null;
    const info = buildMlpDownProjectionHoverInfo(node, rowHit.rowItem);
    const inputNodeId = index?.singleNodeIds?.projectionSourceXln || '';
    const upOutputNodeId = (index?.nodeIdsByRole?.get('mlp-up-output') || [])[0] || '';
    const upOutputCopyNodeId = (index?.nodeIdsByRole?.get('mlp-up-output-copy') || [])[0] || '';
    const activationNodeId = (index?.nodeIdsByRole?.get('mlp-activation-output') || [])[0] || '';
    const activationCopyNodeId = (index?.nodeIdsByRole?.get('mlp-activation-output-copy') || [])[0] || '';
    const weightNodeId = (index?.nodeIdsByRole?.get('mlp-down-weight') || [])[0] || '';
    const biasNodeId = (index?.nodeIdsByRole?.get('mlp-down-bias') || [])[0] || '';
    const outputNodeId = (index?.nodeIdsByRole?.get('mlp-down-output') || [])[0] || '';
    return buildPathFocusResult(index, {
        label: info?.activationData?.label || 'MLP Down Projection',
        info,
        extraNodeIds: [
            inputNodeId,
            upOutputNodeId,
            upOutputCopyNodeId,
            activationNodeId,
            activationCopyNodeId,
            weightNodeId,
            biasNodeId,
            outputNodeId
        ],
        rowSelections: [
            ...(inputNodeId ? [{ nodeId: inputNodeId, rowIndex }] : []),
            ...(upOutputNodeId ? [{ nodeId: upOutputNodeId, rowIndex }] : []),
            ...(upOutputCopyNodeId ? [{ nodeId: upOutputCopyNodeId, rowIndex }] : []),
            ...(activationNodeId ? [{ nodeId: activationNodeId, rowIndex }] : []),
            ...(activationCopyNodeId ? [{ nodeId: activationCopyNodeId, rowIndex }] : []),
            ...(outputNodeId ? [{ nodeId: outputNodeId, rowIndex }] : [])
        ]
    }, options);
}

function buildMlpDownWeightResult(index = null, node = null, options = null) {
    if (!index || !node) return null;
    const activationNodeId = (index?.nodeIdsByRole?.get('mlp-activation-output') || [])[0] || '';
    const activationCopyNodeId = (index?.nodeIdsByRole?.get('mlp-activation-output-copy') || [])[0] || '';
    const biasNodeId = (index?.nodeIdsByRole?.get('mlp-down-bias') || [])[0] || '';
    const outputNodeId = (index?.nodeIdsByRole?.get('mlp-down-output') || [])[0] || '';
    return buildPathFocusResult(index, {
        label: MLP_DOWN_TOOLTIP_LABEL,
        info: buildMlpDownWeightHoverInfo(node),
        extraNodeIds: [node.id, activationNodeId, activationCopyNodeId, biasNodeId, outputNodeId]
    }, options);
}

function buildMlpDownBiasResult(index = null, node = null, rowHit = null, options = null) {
    if (!index || !node) return null;
    const activationNodeId = (index?.nodeIdsByRole?.get('mlp-activation-output') || [])[0] || '';
    const activationCopyNodeId = (index?.nodeIdsByRole?.get('mlp-activation-output-copy') || [])[0] || '';
    const weightNodeId = (index?.nodeIdsByRole?.get('mlp-down-weight') || [])[0] || '';
    const outputNodeId = (index?.nodeIdsByRole?.get('mlp-down-output') || [])[0] || '';
    return buildPathFocusResult(index, {
        label: MLP_DOWN_BIAS_TOOLTIP_LABEL,
        info: buildMlpDownBiasHoverInfo(node, rowHit?.rowItem),
        extraNodeIds: [node.id, activationNodeId, activationCopyNodeId, weightNodeId, outputNodeId]
    }, options);
}

function buildProjectionStageResult(index = null, kind = '', options = null) {
    const safeKind = normalizeProjectionKind(kind);
    if (!index || !safeKind) return null;
    return buildPathFocusResult(index, {
        label: `${resolveProjectionLabel(safeKind)} projection`,
        dimNodeIds: safeKind === 'v'
            ? buildValueBranchExtraDimNodeIds(index)
            : [],
        descriptors: [PROJECTION_STAGE_PATH_PRESETS[safeKind]]
    }, options);
}

function buildProjectionWeightResult(index = null, node = null, kind = '', options = null) {
    const safeKind = normalizeProjectionKind(kind);
    if (!index || !safeKind) return null;
    const label = `${resolveProjectionLabel(safeKind)} Weight Matrix`;
    const extraExclusions = safeKind === 'k'
        ? {
            excludeProjectionKinds: ['q', 'v'],
            excludeProjectionInputKinds: ['q', 'v'],
            excludeRoleNames: ['attention-query-source', 'attention-value-post'],
            excludeConnectorKinds: ['q', 'v']
        }
        : {};
    return buildPathFocusResult(index, {
        label,
        info: buildProjectionHoverInfo(node, label),
        dimNodeIds: safeKind === 'v'
            ? buildValueBranchExtraDimNodeIds(index)
            : [],
        descriptors: [
            PROJECTION_SOURCE_PATH_PRESETS[safeKind],
            {
                projectionInputKinds: [safeKind],
                projectionOutputKinds: [safeKind],
                singleNodeKeys: [PROJECTION_TERMINAL_NODE_KEYS[safeKind]],
                connectorKinds: [safeKind],
                ...extraExclusions
            }
        ],
        extraNodeIds: [node?.id || '']
    }, options);
}

function buildProjectionBiasResult(index = null, node = null, kind = '', rowHit = null, options = null) {
    const safeKind = normalizeProjectionKind(kind);
    if (!index || !safeKind) return null;
    const label = `${resolveProjectionLabel(safeKind)} Bias Vector`;
    const biasRowItem = rowHit?.rowItem || node?.rowItems?.[0] || null;
    const weightNodeId = findProjectionNodeIdByKind(index, 'projection-weight', safeKind);
    return buildPathFocusResult(index, {
        label,
        info: buildProjectionBiasHoverInfo(node, biasRowItem, safeKind),
        dimNodeIds: safeKind === 'v'
            ? buildValueBranchExtraDimNodeIds(index)
            : [],
        descriptors: [
            PROJECTION_SOURCE_PATH_PRESETS[safeKind],
            {
                projectionInputKinds: [safeKind],
                projectionOutputKinds: [safeKind],
                singleNodeKeys: [PROJECTION_TERMINAL_NODE_KEYS[safeKind]],
                connectorKinds: [safeKind]
            }
        ],
        extraNodeIds: [node?.id || '', weightNodeId]
    }, options);
}

function buildAttentionRoleResult(index = null, node = null, role = '', options = null) {
    const safeRole = String(role || '').trim();
    const attentionStageKey = resolveAttentionStageKeyForRole(safeRole);
    const attentionStageInfo = attentionStageKey
        ? buildAttentionStageRoleHoverInfo(node, attentionStageKey)
        : null;
    if (!index || !safeRole.length) return null;

    if (safeRole === 'attention-query-source') {
        return buildPathFocusResult(index, {
            label: 'Query path',
            descriptors: ['queryStage']
        }, options);
    }

    if (safeRole === 'attention-key-transpose') {
        return buildPathFocusResult(index, {
            label: 'Key path',
            descriptors: ['keyStage']
        }, options);
    }

    if (safeRole === 'attention-mask') {
        return buildPathFocusResult(index, {
            label: attentionStageInfo?.activationData?.label || resolveAttentionHoverLabel('mask'),
            info: attentionStageInfo,
            descriptors: ['maskPath']
        }, options);
    }

    if (safeRole === 'attention-value-post') {
        return buildPathFocusResult(index, {
            label: 'Value Vector',
            info: buildProjectionHoverInfo(node, 'Value Vector', {
                stage: 'qkv.v'
            }),
            dimNodeIds: buildValueBranchExtraDimNodeIds(index),
            descriptors: ['valueProjectionPath']
        }, options);
    }

    if (safeRole === 'attention-post-copy') {
        return buildPathFocusResult(index, {
            label: attentionStageInfo?.activationData?.label || resolveAttentionHoverLabel('post'),
            info: attentionStageInfo,
            descriptors: ['weightedOutputPath']
        }, options);
    }

    if (WEIGHTED_OUTPUT_ROLES.includes(safeRole) && safeRole !== 'attention-head-output') {
        return buildPathFocusResult(index, {
            label: 'Weighted output',
            descriptors: ['weightedOutputPath']
        }, options);
    }

    if (safeRole === 'attention-head-output') {
        return buildPathFocusResult(index, {
            label: 'Attention Weighted Sum',
            info: buildWeightedSumHoverInfo(node, null),
            descriptors: ['weightedOutputOnly']
        }, options);
    }

    const isScoreRole = PRE_ATTENTION_ROLES.includes(safeRole) || SCORE_ATTENTION_ROLES.includes(safeRole);
    if (isScoreRole) {
        return buildPathFocusResult(index, {
            label: attentionStageInfo?.activationData?.label || 'Attention score path',
            info: attentionStageInfo,
            descriptors: [
                safeRole === 'attention-post'
                    ? 'weightedOutputPath'
                    : 'scorePath'
            ]
        }, options);
    }

    return null;
}

export function resolveMhsaDetailHoverState(index = null, hit = null, options = null) {
    if (!index || !hit?.node) return null;
    const entity = resolveMhsaDetailHoverEntity(hit);
    if (!entity) return null;

    if (entity.type === 'attention-cell') {
        if (entity.stageKey === 'pre-score') {
            return buildPreScoreCellResult(
                index,
                entity.node,
                entity.rowIndex,
                entity.colIndex,
                entity.cellItem,
                options
            );
        }
        if (entity.stageKey === 'masked-input' || entity.stageKey === 'mask') {
            return buildSoftmaxCellResult(
                index,
                entity.node,
                entity.rowIndex,
                entity.colIndex,
                entity.cellItem,
                {
                    stageKey: entity.stageKey,
                    includePostCopy: false,
                    includeMirroredPostCopyCell: true,
                    includePostForMask: entity.stageKey === 'mask'
                },
                options
            );
        }
        if (entity.stageKey === 'post' || entity.stageKey === 'post-copy') {
            return buildSoftmaxCellResult(
                index,
                entity.node,
                entity.rowIndex,
                entity.colIndex,
                entity.cellItem,
                {
                    stageKey: entity.stageKey,
                    includePostCopy: true
                },
                options
            );
        }
    }

    if (entity.type === 'transpose-axis') {
        return buildTransposeAxisResult(index, entity.axisHit, options);
    }

    if (entity.type === 'projection-row') {
        if (entity.role === 'projection-cache' || entity.role === 'projection-cache-source') {
            return buildProjectionCacheRowResult(index, entity.node, entity.projectionKind, entity.rowHit, options);
        }
        return buildProjectionRowResult(index, entity.node, entity.projectionKind, entity.rowHit, entity.role, options);
    }

    if (entity.type === 'projection-cache-concat-result-row') {
        return buildProjectionCacheConcatResultRowResult(
            index,
            entity.node,
            entity.projectionKind,
            entity.rowHit,
            options
        );
    }

    if (entity.type === 'projection-source-row') {
        return buildProjectionSourceRowResult(index, entity.rowHit, options);
    }

    if (entity.type === 'layer-norm-input-row') {
        return buildLayerNormInputRowResult(index, entity.node, entity.rowHit, options);
    }

    if (entity.type === 'layer-norm-normalized-row') {
        return buildLayerNormNormalizedRowResult(index, entity.node, entity.rowHit, options);
    }

    if (entity.type === 'layer-norm-scaled-row') {
        return buildLayerNormScaledRowResult(index, entity.node, entity.rowHit, options);
    }

    if (entity.type === 'layer-norm-output-row') {
        return buildLayerNormOutputRowResult(index, entity.node, entity.rowHit, options);
    }

    if (entity.type === 'layer-norm-scale') {
        return buildLayerNormParamResult(index, entity.node, {
            param: 'scale'
        }, options);
    }

    if (entity.type === 'layer-norm-shift') {
        return buildLayerNormParamResult(index, entity.node, {
            param: 'shift'
        }, options);
    }

    if (entity.type === 'mlp-up-output-row') {
        return buildMlpUpOutputRowResult(index, entity.node, entity.rowHit, options);
    }

    if (entity.type === 'mlp-activation-row') {
        return buildMlpActivationRowResult(index, entity.node, entity.rowHit, options);
    }

    if (entity.type === 'mlp-down-output-row') {
        return buildMlpDownOutputRowResult(index, entity.node, entity.rowHit, options);
    }

    if (entity.type === 'mlp-up-weight') {
        return buildMlpUpWeightResult(index, entity.node, options);
    }

    if (entity.type === 'mlp-up-bias') {
        return buildMlpUpBiasResult(index, entity.node, entity.rowHit, options);
    }

    if (entity.type === 'mlp-down-weight') {
        return buildMlpDownWeightResult(index, entity.node, options);
    }

    if (entity.type === 'mlp-down-bias') {
        return buildMlpDownBiasResult(index, entity.node, entity.rowHit, options);
    }

    if (entity.type === 'weighted-output-row') {
        if (entity.variant === 'value') {
            const concatResultPart = String(entity?.rowHit?.rowItem?.semantic?.concatResultPart || '').trim().toLowerCase();
            if (concatResultPart === 'cache' || concatResultPart === 'live') {
                return buildProjectionCacheConcatResultRowResult(index, entity.node, 'v', entity.rowHit, options);
            }
            return buildWeightedOutputRowResult(index, entity.node, entity.rowHit, {
                includeProjection: true,
                label: 'Value Vector',
                info: buildProjectionVectorHoverInfo(entity.node, entity.rowHit.rowItem, 'v')
            }, options);
        }
        return buildWeightedOutputRowResult(index, entity.node, entity.rowHit, {
            includeProjection: false,
            label: 'Attention Weighted Sum',
            info: buildWeightedSumHoverInfo(entity.node, entity.rowHit.rowItem)
        }, options);
    }

    if (entity.type === 'output-projection-head-output-row') {
        return buildOutputProjectionHeadOutputRowResult(index, entity.node, entity.rowHit, options);
    }

    if (entity.type === 'output-projection-concat-output-band') {
        return buildOutputProjectionConcatOutputBandResult(index, entity.node, entity.cellHit, options);
    }

    if (entity.type === 'output-projection-concat-output-row') {
        return buildOutputProjectionConcatOutputRowResult(index, entity.node, entity.rowHit, options);
    }

    if (entity.type === 'output-projection-weight') {
        return buildOutputProjectionWeightResult(index, entity.node, options);
    }

    if (entity.type === 'output-projection-bias') {
        return buildOutputProjectionBiasResult(index, entity.node, entity.rowHit, options);
    }

    if (entity.type === 'output-projection-output-row') {
        return buildOutputProjectionOutputRowResult(index, entity.node, entity.rowHit, options);
    }

    if (entity.type === 'projection-weight') {
        return buildProjectionWeightResult(index, entity.node, entity.projectionKind, options);
    }

    if (entity.type === 'projection-bias') {
        return buildProjectionBiasResult(index, entity.node, entity.projectionKind, entity.rowHit, options);
    }

    if (entity.type === 'projection-stage') {
        if (entity.node?.role === 'projection-cache' || entity.node?.role === 'projection-cache-source') {
            return buildProjectionCacheStageResult(index, entity.node, entity.projectionKind, options);
        }
        return buildProjectionStageResult(index, entity.projectionKind, options);
    }

    if (entity.type === 'projection-cache-concat-result') {
        return buildProjectionCacheStageResult(index, entity.node, entity.projectionKind, options);
    }

    return buildAttentionRoleResult(index, entity.node, entity.role, options);
}
