import {
    flattenSceneNodes,
    VIEW2D_NODE_KINDS
} from './schema/sceneTypes.js';
import {
    appendAllUnique,
    appendUnique
} from './mhsaDetailFocusResult.js';
import {
    normalizeProjectionKind,
    resolveProjectionKindForNode
} from './mhsaDetailHoverInfo.js';
import { resolveMhsaDetailHoverState as resolveMhsaDetailHoverStateUncached } from './mhsaDetailHoverResolver.js';

const MHSA_DETAIL_HOVER_RESULT_CACHE = new WeakMap();

function buildMhsaDetailHoverCacheKey(hit = null, options = null) {
    const nodeId = typeof hit?.node?.id === 'string' ? hit.node.id : '';
    if (!nodeId.length) return '';
    const includeFocusState = options?.includeFocusState !== false;
    const interactionKind = String(options?.interactionKind || 'hover').trim().toLowerCase() || 'hover';
    const parts = [includeFocusState ? 'focus' : 'tooltip', interactionKind, nodeId];
    if (Number.isFinite(hit?.cellHit?.rowIndex) && Number.isFinite(hit?.cellHit?.colIndex)) {
        parts.push(`cell:${Math.max(0, Math.floor(hit.cellHit.rowIndex))}:${Math.max(0, Math.floor(hit.cellHit.colIndex))}`);
    } else if (Number.isFinite(hit?.rowHit?.rowIndex)) {
        parts.push(`row:${Math.max(0, Math.floor(hit.rowHit.rowIndex))}`);
    } else if (Number.isFinite(hit?.columnHit?.colIndex)) {
        parts.push(`col:${Math.max(0, Math.floor(hit.columnHit.colIndex))}`);
    } else {
        parts.push('node');
    }
    const stageKey = String(
        hit?.cellHit?.cellItem?.semantic?.stage
        || hit?.node?.semantic?.stage
        || ''
    ).trim().toLowerCase();
    if (stageKey.length) {
        parts.push(`stage:${stageKey}`);
    }
    return parts.join('|');
}

export function resolveMhsaDetailHoverState(index = null, hit = null, options = null) {
    if (!index || !hit?.node) return null;
    const cacheKey = buildMhsaDetailHoverCacheKey(hit, options);
    if (!cacheKey.length) {
        return resolveMhsaDetailHoverStateUncached(index, hit, options);
    }
    let cache = MHSA_DETAIL_HOVER_RESULT_CACHE.get(index);
    if (!cache) {
        cache = new Map();
        MHSA_DETAIL_HOVER_RESULT_CACHE.set(index, cache);
    }
    if (cache.has(cacheKey)) {
        return cache.get(cacheKey) || null;
    }
    const result = resolveMhsaDetailHoverStateUncached(index, hit, options);
    cache.set(cacheKey, result || null);
    return result;
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
