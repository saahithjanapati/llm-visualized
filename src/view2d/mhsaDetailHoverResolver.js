import {
    normalizeAttentionHoverStageKey,
    resolveAttentionHoverLabel
} from '../ui/attentionHoverInfo.js';
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
    buildAttentionStageRoleHoverInfo,
    buildPostLayerNormResidualHoverInfo,
    buildProjectionColumnHoverInfo,
    buildProjectionHoverInfo,
    buildProjectionVectorHoverInfo,
    buildWeightedSumHoverInfo,
    createTokenInfo,
    normalizeProjectionKind,
    resolveAttentionStageKeyForRole,
    resolveProjectionLabel
} from './mhsaDetailHoverInfo.js';
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
    const scoreAxisSelections = buildScoreAxisSelections(index, colIndex, {
        includePost: false,
        includePostCopy: false
    });
    const info = buildAttentionCellHoverInfo(node, cellItem, 'pre');
    const rowSelections = [
        ...buildQueryRowSelections(index, rowIndex),
        ...buildKeyRowSelections(index, colIndex),
        ...scoreAxisSelections.rowSelections,
        { nodeId: index?.singleNodeIds?.attentionHeadOutput, rowIndex }
    ];
    const columnSelections = scoreAxisSelections.columnSelections;
    const cellSelections = [
        { nodeId: index?.singleNodeIds?.attentionPreScore, rowIndex, colIndex },
        { nodeId: index?.singleNodeIds?.attentionMaskedInput, rowIndex, colIndex },
        ...buildPostCopyMirrorCellSelections(index, rowIndex, colIndex)
    ];

    return buildPathFocusResult(index, {
        label: info?.activationData?.label || resolveAttentionHoverLabel('pre'),
        info,
        descriptors: [
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
        rowSelections,
        columnSelections,
        cellSelections
    }, options);
}

function buildSoftmaxCellResult(index = null, node = null, rowIndex = null, colIndex = null, cellItem = null, {
    stageKey = 'post',
    includePostCopy = false,
    includeMirroredPostCopyCell = false
} = {}, options = null) {
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
        descriptors: [
            isMaskStage
                ? 'maskPath'
                : 'scorePath'
        ],
        rowSelections,
        columnSelections,
        cellSelections
    }, options);
}

function buildProjectionRowResult(index = null, node = null, kind = '', rowHit = null, role = '', options = null) {
    const safeKind = normalizeProjectionKind(kind);
    if (!safeKind || !rowHit) return null;
    const rowIndex = Number.isFinite(rowHit.rowIndex) ? Math.max(0, Math.floor(rowHit.rowIndex)) : null;
    const dimNodeIds = safeKind === 'v'
        ? buildValueBranchExtraDimNodeIds(index)
        : [];
    const rowSelections = [];
    const columnSelections = [];
    const descriptors = [];
    const extraNodeIds = [];

    if (safeKind === 'q') {
        const includeAttentionScorePath = (
            role === 'attention-query-source'
            || role === 'projection-output'
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

    const result = buildPathFocusResult(index, {
        label: `${resolveProjectionLabel(safeKind)} row`,
        info: createTokenInfo(rowHit.rowItem),
        descriptors,
        extraNodeIds,
        dimNodeIds,
        rowSelections,
        columnSelections
    }, options);
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
            { nodeId: index.singleNodeIds.attentionValuePost, rowIndex },
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

function buildProjectionSourceRowResult(index = null, rowHit = null, options = null) {
    if (!index || !rowHit) return null;
    const rowIndex = Number.isFinite(rowHit.rowIndex) ? Math.max(0, Math.floor(rowHit.rowIndex)) : null;
    const transposeSelections = buildAttentionKeyTransposeSelections(index, rowIndex);
    return buildPathFocusResult(index, {
        label: 'Post LayerNorm Residual Vector',
        info: buildPostLayerNormResidualHoverInfo(
            index?.nodesById?.get(index?.singleNodeIds?.projectionSourceXln || ''),
            rowHit.rowItem
        ),
        descriptors: ['projectionSourceAll'],
        rowSelections: [
            ...buildProjectionSourceRowSelections(index, rowIndex),
            ...buildQueryRowSelections(index, rowIndex),
            ...buildKeyRowSelections(index, rowIndex),
            ...buildValueRowSelections(index, rowIndex),
            ...transposeSelections.rowSelections
        ],
        columnSelections: transposeSelections.columnSelections
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

function buildProjectionBiasResult(index = null, node = null, kind = '', options = null) {
    const safeKind = normalizeProjectionKind(kind);
    if (!index || !safeKind) return null;
    const label = `${resolveProjectionLabel(safeKind)} Bias Vector`;
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
                connectorKinds: [safeKind]
            }
        ],
        extraNodeIds: [node?.id || '']
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
                    includeMirroredPostCopyCell: true
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
        return buildProjectionRowResult(index, entity.node, entity.projectionKind, entity.rowHit, entity.role, options);
    }

    if (entity.type === 'projection-source-row') {
        return buildProjectionSourceRowResult(index, entity.rowHit, options);
    }

    if (entity.type === 'weighted-output-row') {
        if (entity.variant === 'value') {
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

    if (entity.type === 'projection-weight') {
        return buildProjectionWeightResult(index, entity.node, entity.projectionKind, options);
    }

    if (entity.type === 'projection-bias') {
        return buildProjectionBiasResult(index, entity.node, entity.projectionKind, options);
    }

    if (entity.type === 'projection-stage') {
        return buildProjectionStageResult(index, entity.projectionKind, options);
    }

    return buildAttentionRoleResult(index, entity.node, entity.role, options);
}
