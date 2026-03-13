import { VIEW2D_MATRIX_PRESENTATIONS } from './schema/sceneTypes.js';

export const PRE_ATTENTION_ROLES = Object.freeze([
    'attention-open',
    'attention-query-source',
    'attention-multiply',
    'attention-key-transpose',
    'attention-close',
    'attention-divide',
    'attention-scale',
    'attention-equals'
]);

export const SCORE_ATTENTION_ROLES = Object.freeze([
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

export const WEIGHTED_OUTPUT_ROLES = Object.freeze([
    'attention-post-copy',
    'attention-head-output-multiply',
    'attention-value-post',
    'attention-head-output-equals',
    'attention-head-output'
]);

export const MHSA_DETAIL_FOCUS_PATH_PRESETS = Object.freeze({
    queryStage: {
        projectionKinds: ['q'],
        attentionOptions: {
            query: true,
            transpose: false,
            score: false,
            includeWeightedOutput: false
        },
        connectorKinds: ['q']
    },
    queryScorePath: {
        projectionKinds: ['q'],
        attentionOptions: {
            query: true,
            transpose: false,
            score: true,
            includeWeightedOutput: false
        },
        excludeProjectionKinds: ['k', 'v'],
        excludeProjectionInputKinds: ['k', 'v'],
        excludeRoleNames: ['attention-key-transpose', 'attention-value-post'],
        excludeConnectorKinds: ['k', 'v'],
        connectorKinds: ['q']
    },
    keyStage: {
        projectionKinds: ['k'],
        attentionOptions: {
            query: false,
            transpose: true,
            score: false,
            includeWeightedOutput: false
        },
        connectorKinds: ['k']
    },
    keyScorePath: {
        projectionKinds: ['k'],
        attentionOptions: {
            query: false,
            transpose: true,
            score: true,
            includeWeightedOutput: true
        },
        excludeProjectionKinds: ['q', 'v'],
        excludeProjectionInputKinds: ['q', 'v'],
        excludeRoleNames: ['attention-query-source', 'attention-value-post'],
        excludeConnectorKinds: ['v'],
        connectorKinds: ['k', 'pre', 'post', 'v']
    },
    maskPath: {
        projectionKinds: ['q', 'k'],
        attentionMaskOptions: {
            query: true,
            transpose: true
        },
        connectorKinds: ['q', 'k', 'pre']
    },
    scorePath: {
        projectionKinds: ['q', 'k'],
        attentionOptions: {
            query: true,
            transpose: true,
            score: true,
            includeWeightedOutput: false
        },
        connectorKinds: ['q', 'k', 'pre', 'post']
    },
    weightedOutputPath: {
        projectionKinds: ['q', 'k'],
        attentionOptions: {
            query: true,
            transpose: true,
            score: true,
            includeWeightedOutput: true
        },
        connectorKinds: ['q', 'k', 'pre', 'post', 'v']
    },
    valueStage: {
        projectionKinds: ['v']
    },
    valueProjectionPath: {
        projectionKinds: ['v'],
        roleNames: ['attention-value-post'],
        connectorKinds: ['v']
    },
    weightedProductPath: {
        projectionKinds: ['v'],
        roleNames: [
            'attention-post',
            'attention-post-copy',
            'attention-head-output-multiply',
            'attention-value-post',
            'attention-head-output-equals',
            'attention-head-output'
        ],
        connectorKinds: ['post', 'v']
    },
    weightedOutputOnly: {
        roleNames: ['attention-post-copy', 'attention-value-post', 'attention-head-output']
    },
    projectionSourceAll: {
        projectionSourceKinds: ['q', 'k', 'v'],
        projectionInputKinds: ['q', 'k', 'v']
    }
});

const MASK_STAGE_ATTENTION_ROLES = Object.freeze([
    'attention-pre-score',
    'attention-softmax-label',
    'attention-softmax-open',
    'attention-masked-input',
    'attention-softmax-plus',
    'attention-mask',
    'attention-softmax-close'
]);
const FOCUS_PATH_STATE_CACHE = new WeakMap();
const FOCUS_DESCRIPTOR_CACHE_FIELDS = Object.freeze([
    'projectionKinds',
    'projectionInputKinds',
    'projectionOutputKinds',
    'projectionSourceKinds',
    'singleNodeKeys',
    'roleNames',
    'connectorKinds',
    'excludeProjectionKinds',
    'excludeProjectionInputKinds',
    'excludeRoleNames',
    'excludeConnectorKinds',
    'attentionOptions',
    'attentionMaskOptions'
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

function removeValues(target = [], values = []) {
    if (!Array.isArray(target) || !target.length || !Array.isArray(values) || !values.length) return;
    const blocked = new Set(values.filter((value) => typeof value === 'string' && value.length));
    if (!blocked.size) return;
    for (let index = target.length - 1; index >= 0; index -= 1) {
        if (blocked.has(target[index])) {
            target.splice(index, 1);
        }
    }
}

export function resolveAttentionRoleIds(index = null, {
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

export function resolveAttentionMaskStageRoleIds(index = null, {
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

export function appendProjectionStagePath(activeNodeIds, index = null, kinds = []) {
    kinds.forEach((kind) => {
        appendAllUnique(activeNodeIds, index?.projectionLeafIdsByKind?.[kind] || []);
    });
}

export function removeProjectionStageKinds(activeNodeIds = [], index = null, kinds = []) {
    removeValues(activeNodeIds, kinds.flatMap((kind) => index?.projectionLeafIdsByKind?.[kind] || []));
}

export function appendAttentionPostCopyNode(activeNodeIds = [], index = null) {
    appendUnique(activeNodeIds, index?.singleNodeIds?.attentionPostCopy || '');
}

export function appendConnectorKinds(activeConnectorIds, index = null, kinds = []) {
    kinds.forEach((kind) => {
        appendAllUnique(activeConnectorIds, index?.connectorIdsByKind?.[kind] || []);
    });
}

export function removeConnectorKinds(activeConnectorIds = [], index = null, kinds = []) {
    removeValues(activeConnectorIds, kinds.flatMap((kind) => index?.connectorIdsByKind?.[kind] || []));
}

export function appendRoleNodeIds(activeNodeIds = [], index = null, roles = []) {
    roles.forEach((role) => {
        appendAllUnique(activeNodeIds, index?.nodeIdsByRole?.get(role) || []);
    });
}

export function removeRoleNodeIds(activeNodeIds = [], index = null, roles = []) {
    removeValues(activeNodeIds, roles.flatMap((role) => index?.nodeIdsByRole?.get(role) || []));
}

export function appendSingleNodeKeys(activeNodeIds = [], index = null, keys = []) {
    keys.forEach((key) => {
        appendUnique(activeNodeIds, index?.singleNodeIds?.[key] || '');
    });
}

export function appendProjectionInputKinds(activeNodeIds = [], index = null, kinds = []) {
    kinds.forEach((kind) => {
        appendUnique(activeNodeIds, index?.projectionInputIdsByKind?.[kind] || '');
    });
}

export function removeProjectionInputKinds(activeNodeIds = [], index = null, kinds = []) {
    removeValues(activeNodeIds, kinds.map((kind) => index?.projectionInputIdsByKind?.[kind] || ''));
}

export function appendProjectionOutputKinds(activeNodeIds = [], index = null, kinds = []) {
    kinds.forEach((kind) => {
        appendUnique(activeNodeIds, index?.projectionOutputIdsByKind?.[kind] || '');
    });
}

export function appendProjectionSourceKinds(activeNodeIds = [], activeConnectorIds = [], index = null, kinds = []) {
    if (!Array.isArray(kinds) || !kinds.length) return;
    appendUnique(activeNodeIds, index?.singleNodeIds?.projectionSourceXln || '');
    kinds.forEach((kind) => {
        appendAllUnique(activeConnectorIds, index?.projectionIngressConnectorIdsByKind?.[kind] || []);
    });
}

function resolveFocusPathDescriptor(descriptor = null) {
    if (!descriptor) return null;
    if (typeof descriptor === 'string') {
        return MHSA_DETAIL_FOCUS_PATH_PRESETS[descriptor] || null;
    }
    return typeof descriptor === 'object' ? descriptor : null;
}

function serializeDescriptorArray(values = []) {
    return Array.isArray(values) && values.length
        ? values.join(',')
        : '';
}

function serializeDescriptorFlags(value = null) {
    if (!value || typeof value !== 'object') return '';
    return Object.keys(value)
        .sort()
        .map((key) => `${key}:${value[key] ? 1 : 0}`)
        .join(',');
}

function buildFocusDescriptorCacheKey(descriptor = null) {
    if (!descriptor) return '';
    if (typeof descriptor === 'string') {
        return `preset:${descriptor}`;
    }
    const resolved = resolveFocusPathDescriptor(descriptor);
    if (!resolved) return '';
    return FOCUS_DESCRIPTOR_CACHE_FIELDS
        .map((field) => {
            const value = resolved[field];
            if (Array.isArray(value)) {
                const serialized = serializeDescriptorArray(value);
                return serialized.length ? `${field}=[${serialized}]` : '';
            }
            if (value && typeof value === 'object') {
                const serialized = serializeDescriptorFlags(value);
                return serialized.length ? `${field}={${serialized}}` : '';
            }
            return '';
        })
        .filter(Boolean)
        .join(';');
}

function applyFocusPathDescriptor(activeNodeIds = [], activeConnectorIds = [], index = null, descriptor = null) {
    const resolved = resolveFocusPathDescriptor(descriptor);
    if (!resolved || !index) return;

    appendProjectionStagePath(activeNodeIds, index, resolved.projectionKinds || []);
    appendProjectionInputKinds(activeNodeIds, index, resolved.projectionInputKinds || []);
    appendProjectionOutputKinds(activeNodeIds, index, resolved.projectionOutputKinds || []);
    appendProjectionSourceKinds(activeNodeIds, activeConnectorIds, index, resolved.projectionSourceKinds || []);
    appendSingleNodeKeys(activeNodeIds, index, resolved.singleNodeKeys || []);
    appendRoleNodeIds(activeNodeIds, index, resolved.roleNames || []);

    if (resolved.attentionOptions) {
        appendAllUnique(activeNodeIds, resolveAttentionRoleIds(index, resolved.attentionOptions));
    }
    if (resolved.attentionMaskOptions) {
        appendAllUnique(activeNodeIds, resolveAttentionMaskStageRoleIds(index, resolved.attentionMaskOptions));
    }

    appendConnectorKinds(activeConnectorIds, index, resolved.connectorKinds || []);
    removeProjectionStageKinds(activeNodeIds, index, resolved.excludeProjectionKinds || []);
    removeProjectionInputKinds(activeNodeIds, index, resolved.excludeProjectionInputKinds || []);
    removeRoleNodeIds(activeNodeIds, index, resolved.excludeRoleNames || []);
    removeConnectorKinds(activeConnectorIds, index, resolved.excludeConnectorKinds || []);
}

function resolveCachedFocusPathState(index = null, descriptors = []) {
    if (!index || !Array.isArray(descriptors) || !descriptors.length) return null;
    const cacheKey = descriptors
        .map((descriptor) => buildFocusDescriptorCacheKey(descriptor))
        .filter((value) => value.length)
        .join('||');
    if (!cacheKey.length) return null;
    let cache = FOCUS_PATH_STATE_CACHE.get(index);
    if (!cache) {
        cache = new Map();
        FOCUS_PATH_STATE_CACHE.set(index, cache);
    }
    const cachedState = cache.get(cacheKey) || null;
    if (cachedState) {
        return cachedState;
    }
    const activeNodeIds = [];
    const activeConnectorIds = [];
    descriptors.forEach((descriptor) => {
        applyFocusPathDescriptor(activeNodeIds, activeConnectorIds, index, descriptor);
    });
    const nextState = {
        activeNodeIds,
        activeConnectorIds
    };
    cache.set(cacheKey, nextState);
    return nextState;
}

export function buildFocusPathState(index = null, descriptors = [], {
    extraNodeIds = [],
    extraConnectorIds = []
} = {}) {
    const baseState = resolveCachedFocusPathState(index, descriptors) || {
        activeNodeIds: [],
        activeConnectorIds: []
    };
    if (!extraNodeIds.length && !extraConnectorIds.length) {
        return baseState;
    }
    const activeNodeIds = baseState.activeNodeIds.slice();
    const activeConnectorIds = baseState.activeConnectorIds.slice();
    appendAllUnique(activeNodeIds, extraNodeIds);
    appendAllUnique(activeConnectorIds, extraConnectorIds);
    return {
        activeNodeIds,
        activeConnectorIds
    };
}

export function buildQueryRowSelections(index = null, rowIndex = null) {
    if (!Number.isFinite(rowIndex) || !index) return [];
    return [
        ...buildProjectionSourceMirrorRowSelections(index, rowIndex),
        { nodeId: index.projectionInputIdsByKind.q, rowIndex },
        { nodeId: index.projectionOutputIdsByKind.q, rowIndex },
        { nodeId: index.singleNodeIds.attentionQuerySource, rowIndex }
    ];
}

export function buildProjectionSourceRowSelections(index = null, rowIndex = null) {
    if (!Number.isFinite(rowIndex) || !index) return [];
    return [
        { nodeId: index.singleNodeIds.projectionSourceXln, rowIndex },
        { nodeId: index.projectionInputIdsByKind.q, rowIndex },
        { nodeId: index.projectionInputIdsByKind.k, rowIndex },
        { nodeId: index.projectionInputIdsByKind.v, rowIndex }
    ].filter((selection) => typeof selection.nodeId === 'string' && selection.nodeId.length);
}

export function buildProjectionSourceMirrorRowSelections(index = null, rowIndex = null) {
    if (!Number.isFinite(rowIndex) || !index) return [];
    return [{
        nodeId: index.singleNodeIds.projectionSourceXln,
        rowIndex
    }].filter((selection) => typeof selection.nodeId === 'string' && selection.nodeId.length);
}

export function buildKeyRowSelections(index = null, rowIndex = null) {
    if (!Number.isFinite(rowIndex) || !index) return [];
    return [
        ...buildProjectionSourceMirrorRowSelections(index, rowIndex),
        { nodeId: index.projectionInputIdsByKind.k, rowIndex },
        { nodeId: index.projectionOutputIdsByKind.k, rowIndex }
    ];
}

export function buildValueRowSelections(index = null, rowIndex = null) {
    if (!Number.isFinite(rowIndex) || !index) return [];
    return [
        ...buildProjectionSourceMirrorRowSelections(index, rowIndex),
        { nodeId: index.projectionInputIdsByKind.v, rowIndex },
        { nodeId: index.projectionOutputIdsByKind.v, rowIndex },
        { nodeId: index.singleNodeIds.attentionValuePost, rowIndex }
    ];
}

function usesRowSelectionsForAttentionKeyTranspose(index = null) {
    const transposeNodeId = index?.singleNodeIds?.attentionKeyTranspose || '';
    if (!transposeNodeId) return false;
    const transposeNode = index?.nodesById?.get(transposeNodeId) || null;
    return transposeNode?.presentation === VIEW2D_MATRIX_PRESENTATIONS.COMPACT_ROWS;
}

export function buildAttentionKeyTransposeSelections(index = null, axisIndex = null) {
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

export function buildScoreAxisSelections(index = null, axisIndex = null, {
    includePost = true,
    includePostCopy = false
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

export function buildScoreAxisRowSelections(index = null, rowIndex = null) {
    return buildScoreAxisRowSelectionsWithOptions(index, rowIndex);
}

export function buildScoreAxisRowSelectionsWithOptions(index = null, rowIndex = null, {
    includePost = true,
    includePostCopy = false
} = {}) {
    if (!Number.isFinite(rowIndex) || !index) return [];
    return [
        { nodeId: index.singleNodeIds.attentionPreScore, rowIndex },
        { nodeId: index.singleNodeIds.attentionMaskedInput, rowIndex },
        { nodeId: index.singleNodeIds.attentionMask, rowIndex },
        ...(includePost
            ? [{ nodeId: index.singleNodeIds.attentionPost, rowIndex }]
            : []),
        ...(includePostCopy
            ? [{ nodeId: index.singleNodeIds.attentionPostCopy, rowIndex }]
            : [])
    ];
}
