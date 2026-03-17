import {
    findUserDataNumber,
    findUserDataString,
    getActivationDataFromSelection,
    isLogitBarSelection,
    isQkvMatrixLabel,
    isSelfAttentionSelection,
    isValueSelection,
    isWeightedSumSelection
} from '../ui/selectionPanelSelectionUtils.js';
import {
    flattenSceneNodes,
    VIEW2D_NODE_KINDS
} from './schema/sceneTypes.js';
import { buildFocusResult } from './mhsaDetailFocusResult.js';
import {
    formatLayerNormLabel,
    isLayerNormNormalizedStage,
    normalizeLayerNormOutputStage,
    normalizeLayerNormProductStage,
    isPostLayerNormResidualStage,
    normalizePostLayerNormResidualStage,
    resolveLayerNormKind,
    resolvePostLayerNormResidualLabel
} from '../utils/layerNormLabels.js';
import { resolvePreferredTokenLabel } from '../utils/tokenLabelResolution.js';

export const TRANSFORMER_VIEW2D_OVERVIEW_LABEL = 'GPT-2 (124M)';
const OVERVIEW_HOVER_FOCUS_RESULT_CACHE = new WeakMap();

function isResidualOverviewStreamNode(node = null) {
    return (
        node?.kind === VIEW2D_NODE_KINDS.MATRIX
        && typeof node?.id === 'string'
        && node.id.length > 0
        && node?.semantic?.componentKind === 'residual'
        && (
            node?.role === 'module-card'
            || node?.role === 'add-circle'
        )
    );
}

function isResidualOverviewVectorNode(node = null) {
    return (
        isResidualOverviewStreamNode(node)
        && node?.role === 'module-card'
    );
}

export function normalizeOptionalIndex(value) {
    return Number.isFinite(value) ? Math.max(0, Math.floor(value)) : null;
}

export function buildSemanticTarget(rawTarget = null) {
    if (!rawTarget || typeof rawTarget !== 'object') return null;
    const target = Object.entries(rawTarget).reduce((acc, [key, value]) => {
        if (!key) return acc;
        if (typeof value === 'number') {
            if (Number.isFinite(value)) acc[key] = Math.floor(value);
            return acc;
        }
        if (typeof value === 'string') {
            const safe = value.trim();
            if (safe.length) acc[key] = safe;
            return acc;
        }
        if (typeof value === 'boolean') {
            acc[key] = value;
        }
        return acc;
    }, {});
    return Object.keys(target).length ? target : null;
}

export function resolveHeadDetailTarget(target = null) {
    if (!target || typeof target !== 'object') return null;
    const layerIndex = normalizeOptionalIndex(target.layerIndex);
    const headIndex = normalizeOptionalIndex(target.headIndex);
    if (!Number.isFinite(layerIndex) || !Number.isFinite(headIndex)) {
        return null;
    }
    return {
        layerIndex,
        headIndex
    };
}

export function resolveConcatDetailTarget(target = null) {
    if (!target || typeof target !== 'object') return null;
    const layerIndex = normalizeOptionalIndex(target.layerIndex);
    if (!Number.isFinite(layerIndex)) {
        return null;
    }
    return {
        layerIndex
    };
}

export function resolveOutputProjectionDetailTarget(target = null) {
    if (!target || typeof target !== 'object') return null;
    const layerIndex = normalizeOptionalIndex(target.layerIndex);
    if (!Number.isFinite(layerIndex)) {
        return null;
    }
    return {
        layerIndex
    };
}

export function resolveMlpDetailTarget(target = null) {
    if (!target || typeof target !== 'object') return null;
    const layerIndex = normalizeOptionalIndex(target.layerIndex);
    if (!Number.isFinite(layerIndex)) {
        return null;
    }
    return {
        layerIndex
    };
}

export function resolveLayerNormDetailTarget(target = null) {
    if (!target || typeof target !== 'object') return null;
    const explicitKind = String(target.layerNormKind || '').trim().toLowerCase();
    const layerNormKind = (
        explicitKind === 'ln1' || explicitKind === 'ln2' || explicitKind === 'final'
    )
        ? explicitKind
        : resolveLayerNormKindFromSemanticStage(target.stage);
    if (!layerNormKind) return null;
    const layerIndex = normalizeOptionalIndex(target.layerIndex);
    if (layerNormKind === 'final') {
        return {
            layerNormKind,
            ...(Number.isFinite(layerIndex) ? { layerIndex } : {})
        };
    }
    if (!Number.isFinite(layerIndex)) {
        return null;
    }
    return {
        layerNormKind,
        layerIndex
    };
}

function normalizeEmbeddingSemanticStage(stage = '') {
    const lower = String(stage || '').trim().toLowerCase();
    if (lower === 'token' || lower === 'embedding.token') return 'embedding.token';
    if (lower === 'position' || lower === 'embedding.position') return 'embedding.position';
    if (lower === 'sum' || lower === 'embedding.sum') return 'embedding.sum';
    return lower;
}

function normalizeResidualSemanticStage(stage = '') {
    const lower = String(stage || '').trim().toLowerCase();
    if (lower === 'incoming' || lower === 'layer.incoming') return 'incoming';
    if (
        lower === 'post-attn-add'
        || lower === 'post-attn-residual'
        || lower === 'residual.post_attention'
    ) {
        return 'post-attn-residual';
    }
    if (
        lower === 'post-mlp-add'
        || lower === 'post-mlp-residual'
        || lower === 'residual.post_mlp'
    ) {
        return 'post-mlp-residual';
    }
    if (lower === 'outgoing') return 'outgoing';
    return lower;
}

function buildLayerNormStageMap(layerNormKind = null) {
    if (layerNormKind === 'ln1') {
        return {
            inputStage: 'layer.incoming',
            normalizedStage: 'ln1.norm',
            scaledStage: 'ln1.scale',
            outputStage: 'ln1.output',
            paramScaleStage: 'ln1.param.scale',
            paramShiftStage: 'ln1.param.shift'
        };
    }
    if (layerNormKind === 'ln2') {
        return {
            inputStage: 'residual.post_attention',
            normalizedStage: 'ln2.norm',
            scaledStage: 'ln2.scale',
            outputStage: 'ln2.output',
            paramScaleStage: 'ln2.param.scale',
            paramShiftStage: 'ln2.param.shift'
        };
    }
    if (layerNormKind === 'final') {
        return {
            inputStage: 'residual.post_mlp',
            normalizedStage: 'final_ln.norm',
            scaledStage: 'final_ln.scale',
            outputStage: 'final_ln.output',
            paramScaleStage: 'final_ln.param.scale',
            paramShiftStage: 'final_ln.param.shift'
        };
    }
    return null;
}

function buildOutputProjectionOverviewSemanticTarget(layerIndex = null) {
    if (!Number.isFinite(layerIndex)) return null;
    return buildSemanticTarget({
        componentKind: 'output-projection',
        layerIndex,
        stage: 'attn-out',
        role: 'projection-weight'
    });
}

function buildMlpOverviewSemanticTarget(layerIndex = null) {
    if (!Number.isFinite(layerIndex)) return null;
    return buildSemanticTarget({
        componentKind: 'mlp',
        layerIndex,
        stage: 'mlp',
        role: 'module'
    });
}

function buildLayerNormOverviewSemanticTarget(layerNormKind = null, layerIndex = null) {
    const stage = resolveLayerNormStage(layerNormKind);
    if (!stage) return null;
    if (layerNormKind !== 'final' && !Number.isFinite(layerIndex)) {
        return null;
    }
    return buildSemanticTarget({
        componentKind: 'layer-norm',
        ...(layerNormKind === 'final' ? {} : { layerIndex }),
        stage,
        role: 'module'
    });
}

function resolveLayerNormDetailSemanticTargets(selectionInfo = null, normalizedLabel = '') {
    const label = String(normalizedLabel || selectionInfo?.label || '').trim();
    const activationData = getActivationDataFromSelection(selectionInfo);
    const stageLower = String(activationData?.stage || '').trim().toLowerCase();
    const explicitLayerNormKind = findUserDataString(selectionInfo, 'layerNormKind');
    const layerNormKind = resolveLayerNormKind({
        label,
        stage: stageLower,
        explicitKind: explicitLayerNormKind
    });
    const layerIndex = normalizeOptionalIndex(findUserDataNumber(selectionInfo, 'layerIndex'));
    const overviewTarget = buildLayerNormOverviewSemanticTarget(layerNormKind, layerIndex);
    const stageMap = buildLayerNormStageMap(layerNormKind);
    if (!overviewTarget || !stageMap) {
        return {
            semanticTarget: null,
            detailSemanticTargets: []
        };
    }

    const detailTargets = [];
    const appendTarget = (stage, role) => {
        appendUniqueSemanticTarget(detailTargets, buildSemanticTarget({
            componentKind: 'layer-norm',
            ...(layerNormKind === 'final' ? {} : { layerIndex }),
            stage,
            role
        }));
    };

    const lower = label.toLowerCase();
    if (
        stageLower === stageMap.paramScaleStage
        || lower.includes('layernorm') && (lower.includes('scale') || lower.includes('gamma'))
        || lower.includes('final layernorm') && (lower.includes('scale') || lower.includes('gamma'))
    ) {
        appendTarget(stageMap.paramScaleStage, 'layer-norm-scale');
    } else if (
        stageLower === stageMap.paramShiftStage
        || lower.includes('layernorm') && (lower.includes('shift') || lower.includes('beta'))
        || lower.includes('final layernorm') && (lower.includes('shift') || lower.includes('beta'))
    ) {
        appendTarget(stageMap.paramShiftStage, 'layer-norm-shift');
    } else if (isLayerNormNormalizedStage(stageLower) || lower.includes('normalized residual')) {
        appendTarget(stageMap.normalizedStage, 'layer-norm-normalized');
    } else if (
        normalizeLayerNormProductStage(stageLower, { preferLegacy: true }).length
        || lower.includes('product vector')
    ) {
        appendTarget(stageMap.scaledStage, 'layer-norm-scaled');
    } else if (
        normalizeLayerNormOutputStage(stageLower).length
        || lower.includes('post layernorm')
    ) {
        appendTarget(stageMap.outputStage, 'layer-norm-output');
    } else if (stageLower === stageMap.inputStage) {
        appendTarget(stageMap.inputStage, 'layer-norm-input');
    }

    return {
        semanticTarget: overviewTarget,
        detailSemanticTargets: detailTargets
    };
}

function resolveMlpDetailSemanticTargets(selectionInfo = null, normalizedLabel = '') {
    const label = String(normalizedLabel || selectionInfo?.label || '').trim();
    const lower = label.toLowerCase();
    const activationData = getActivationDataFromSelection(selectionInfo);
    const stageLower = String(activationData?.stage || '').trim().toLowerCase();
    const layerIndex = normalizeOptionalIndex(findUserDataNumber(selectionInfo, 'layerIndex'));
    const overviewTarget = buildMlpOverviewSemanticTarget(layerIndex);
    if (!overviewTarget) {
        return {
            semanticTarget: null,
            detailSemanticTargets: []
        };
    }

    const detailTargets = [];
    const appendTarget = (stage, role) => {
        appendUniqueSemanticTarget(detailTargets, buildSemanticTarget({
            componentKind: 'mlp',
            layerIndex,
            stage,
            role
        }));
    };

    if (stageLower === 'mlp.up.bias' || (lower.includes('bias') && lower.includes('mlp up'))) {
        appendTarget('mlp.up.bias', 'mlp-up-bias');
    } else if (lower.includes('mlp up weight matrix')) {
        appendTarget('mlp-up', 'mlp-up-weight');
    } else if (
        stageLower === 'mlp.up'
        || lower.includes('mlp up projection')
        || lower.includes('mlp expanded segments')
    ) {
        appendTarget('mlp-up', 'mlp-up-output');
    } else if (
        stageLower === 'mlp.activation'
        || lower.includes('gelu')
        || lower.includes('mlp activation')
    ) {
        appendTarget('mlp.activation', 'mlp-activation-output');
    } else if (stageLower === 'mlp.down.bias' || (lower.includes('bias') && lower.includes('mlp down'))) {
        appendTarget('mlp.down.bias', 'mlp-down-bias');
    } else if (lower.includes('mlp down weight matrix')) {
        appendTarget('mlp-down', 'mlp-down-weight');
    } else if (stageLower === 'mlp.down' || lower.includes('mlp down projection')) {
        appendTarget('mlp-down', 'mlp-down-output');
    }

    return {
        semanticTarget: overviewTarget,
        detailSemanticTargets: detailTargets
    };
}

function resolveOutputProjectionDetailSemanticTargets(selectionInfo = null, normalizedLabel = '') {
    const label = String(normalizedLabel || selectionInfo?.label || '').trim();
    const lower = label.toLowerCase();
    const activationData = getActivationDataFromSelection(selectionInfo);
    const stageLower = String(activationData?.stage || '').trim().toLowerCase();
    const layerIndex = normalizeOptionalIndex(findUserDataNumber(selectionInfo, 'layerIndex'));
    const overviewTarget = buildOutputProjectionOverviewSemanticTarget(layerIndex);
    if (!overviewTarget) {
        return {
            semanticTarget: null,
            detailSemanticTargets: []
        };
    }

    const detailTargets = [];
    const appendTarget = (role) => {
        appendUniqueSemanticTarget(detailTargets, buildSemanticTarget({
            componentKind: 'output-projection',
            layerIndex,
            stage: 'attn-out',
            role
        }));
    };

    if (
        stageLower === 'attention.concatenate'
        || lower === 'concatenate'
        || lower === 'concatenation'
        || lower.includes('concatenate heads')
        || lower.includes('head concatenation')
    ) {
        appendTarget('concat-output-copy-matrix');
    } else if (
        stageLower === 'attention.output_projection.bias'
        || lower.includes('output projection bias')
    ) {
        appendTarget('projection-bias');
    } else if (
        lower.includes('output projection matrix')
        || lower.includes('alpha projection matrix')
        || lower.includes('alpha projection')
    ) {
        appendTarget('projection-weight');
    } else if (
        stageLower === 'attention.output_projection'
        || lower.includes('attention output projection')
        || lower.includes('attention output vector')
    ) {
        appendTarget('projection-output');
    }

    return {
        semanticTarget: overviewTarget,
        detailSemanticTargets: detailTargets
    };
}

function resolveResidualHoverActivationStage(rowSemantic = null) {
    const stage = String(rowSemantic?.stage || '').trim().toLowerCase();
    if (stage === 'incoming') return 'layer.incoming';
    if (stage === 'post-attn-residual') return 'residual.post_attention';
    if (stage === 'post-mlp-residual') return 'residual.post_mlp';
    if (stage === 'outgoing') return 'residual.post_mlp';
    if (isPostLayerNormResidualStage(stage)) return normalizePostLayerNormResidualStage(stage);
    return '';
}

export function buildResidualRowHoverPayload(rowHit = null, activationSource = null) {
    const semantic = rowHit?.rowItem?.semantic || null;
    if (!semantic || semantic.componentKind !== 'residual') return null;
    const activationStage = resolveResidualHoverActivationStage(semantic);
    if (!activationStage) return null;

    const tokenIndex = normalizeOptionalIndex(semantic.tokenIndex);
    const layerIndex = normalizeOptionalIndex(semantic.layerIndex);
    const tokenId = Number.isFinite(tokenIndex) && typeof activationSource?.getTokenId === 'function'
        ? normalizeOptionalIndex(activationSource.getTokenId(tokenIndex))
        : null;
    const tokenLabel = resolvePreferredTokenLabel({
        tokenLabel: rowHit?.rowItem?.label,
        tokenIndex,
        activationSource
    });
    const isPostLayerNormResidual = isPostLayerNormResidualStage(activationStage);
    const headIndex = normalizeOptionalIndex(semantic.headIndex);
    const label = isPostLayerNormResidual
        ? resolvePostLayerNormResidualLabel({ stage: activationStage })
        : 'Residual Stream Vector';
    const sourceStage = isPostLayerNormResidual
        ? normalizePostLayerNormResidualStage(activationStage, { preferLegacy: true })
        : '';
    const info = {
        ...(Number.isFinite(layerIndex) ? { layerIndex } : {}),
        ...(Number.isFinite(headIndex) ? { headIndex } : {}),
        ...(Number.isFinite(tokenIndex) ? { tokenIndex } : {}),
        ...(Number.isFinite(tokenId) ? { tokenId } : {}),
        ...(tokenLabel.length ? { tokenLabel } : {}),
        activationData: {
            label,
            stage: activationStage,
            ...(sourceStage ? { sourceStage } : {}),
            ...(Number.isFinite(layerIndex) ? { layerIndex } : {}),
            ...(Number.isFinite(headIndex) ? { headIndex } : {}),
            ...(Number.isFinite(tokenIndex) ? { tokenIndex } : {}),
            ...(tokenLabel.length ? { tokenLabel } : {})
        }
    };
    return {
        label,
        info
    };
}

export function buildResidualRowSelectionFocusState(scene = null, hit = null) {
    const rowHit = hit?.rowHit || null;
    const semantic = rowHit?.rowItem?.semantic || null;
    const nodeId = typeof hit?.node?.id === 'string' ? hit.node.id : '';
    const rowIndex = Number.isFinite(rowHit?.rowIndex)
        ? Math.max(0, Math.floor(rowHit.rowIndex))
        : null;
    if (
        !scene
        || !nodeId.length
        || !Number.isFinite(rowIndex)
        || semantic?.componentKind !== 'residual'
    ) {
        return null;
    }

    const flattenedNodes = flattenSceneNodes(scene);
    const streamNodeIds = new Set();
    const residualVectorNodes = [];
    flattenedNodes.forEach((node) => {
        if (isResidualOverviewStreamNode(node)) {
            streamNodeIds.add(node.id);
            if (isResidualOverviewVectorNode(node)) {
                residualVectorNodes.push(node);
            }
        }
    });

    const activeConnectorIds = [];
    flattenedNodes.forEach((node) => {
        if (
            node?.kind !== VIEW2D_NODE_KINDS.CONNECTOR
            || typeof node.id !== 'string'
            || !node.id.length
        ) {
            return;
        }
        const sourceNodeId = typeof node?.source?.nodeId === 'string'
            ? node.source.nodeId
            : '';
        const targetNodeId = typeof node?.target?.nodeId === 'string'
            ? node.target.nodeId
            : '';
        if (
            streamNodeIds.has(sourceNodeId)
            && streamNodeIds.has(targetNodeId)
        ) {
            activeConnectorIds.push(node.id);
        }
    });

    const rowSelections = residualVectorNodes
        .filter((node) => {
            const rowItems = Array.isArray(node?.rowItems) ? node.rowItems : [];
            return rowIndex >= 0 && rowIndex < rowItems.length;
        })
        .map((node) => ({
            nodeId: node.id,
            rowIndex
        }));

    return buildFocusResult({
        activeNodeIds: streamNodeIds.size ? Array.from(streamNodeIds) : [nodeId],
        activeConnectorIds,
        rowSelections: rowSelections.length
            ? rowSelections
            : [{
                nodeId,
                rowIndex
            }]
    });
}

function resolveLayerNormKindFromSemanticStage(stage = '') {
    const lower = String(stage || '').trim().toLowerCase();
    if (lower === 'ln1') return 'ln1';
    if (lower === 'ln2') return 'ln2';
    if (lower === 'final-ln') return 'final';
    return null;
}

function buildSemanticHoverInfo({
    label = '',
    layerIndex = null,
    headIndex = null,
    activationStage = '',
    layerNormKind = null,
    suppressTokenChip = false
} = {}) {
    const info = {};
    if (Number.isFinite(layerIndex)) {
        info.layerIndex = Math.max(0, Math.floor(layerIndex));
    }
    if (Number.isFinite(headIndex)) {
        info.headIndex = Math.max(0, Math.floor(headIndex));
    }
    if (typeof layerNormKind === 'string' && layerNormKind.length) {
        info.layerNormKind = layerNormKind;
    }
    if (suppressTokenChip === true) {
        info.suppressTokenChip = true;
    }

    if (label || activationStage || Object.keys(info).length) {
        info.activationData = {
            ...(label ? { label } : {}),
            ...(activationStage ? { stage: activationStage } : {}),
            ...(Number.isFinite(layerIndex) ? { layerIndex: Math.max(0, Math.floor(layerIndex)) } : {}),
            ...(Number.isFinite(headIndex) ? { headIndex: Math.max(0, Math.floor(headIndex)) } : {}),
            ...(typeof layerNormKind === 'string' && layerNormKind.length ? { layerNormKind } : {}),
            ...(suppressTokenChip === true ? { suppressTokenChip: true } : {})
        };
    }

    return info;
}

function resolveSemanticTokenLabel(hit = null, tokenIndex = null) {
    return resolvePreferredTokenLabel({
        tokenLabel: hit?.entry?.metadata?.tokenLabel
            || hit?.node?.metadata?.tokenLabel
            || '',
        tokenIndex
    });
}

function resolveSemanticPositionIndex(hit = null, tokenIndex = null) {
    const rawPositionIndex = hit?.entry?.semantic?.positionIndex
        ?? hit?.node?.semantic?.positionIndex
        ?? hit?.entry?.metadata?.positionIndex
        ?? hit?.node?.metadata?.positionIndex;
    if (Number.isFinite(rawPositionIndex)) {
        return Math.max(1, Math.floor(rawPositionIndex));
    }
    return Number.isFinite(tokenIndex) ? tokenIndex + 1 : null;
}

function buildSemanticTokenChipLabel(tokenLabel = '', tokenIndex = null) {
    const safeTokenLabel = typeof tokenLabel === 'string' ? tokenLabel.trim() : '';
    const fallbackLabel = Number.isFinite(tokenIndex)
        ? `Token ${Math.floor(tokenIndex) + 1}`
        : '';
    const detailLabel = safeTokenLabel || fallbackLabel;
    return detailLabel ? `Token: ${detailLabel}` : 'Token';
}

function buildSemanticPositionChipLabel(positionIndex = null) {
    return Number.isFinite(positionIndex)
        ? `Position: ${Math.max(1, Math.floor(positionIndex))}`
        : 'Position';
}

function buildSemanticChosenTokenChipLabel(tokenLabel = '', tokenIndex = null) {
    const safeTokenLabel = typeof tokenLabel === 'string' ? tokenLabel.trim() : '';
    const fallbackLabel = Number.isFinite(tokenIndex)
        ? `Token ${Math.floor(tokenIndex) + 1}`
        : '';
    const detailLabel = safeTokenLabel || fallbackLabel;
    return detailLabel ? `Chosen Token: ${detailLabel}` : 'Chosen Token';
}

export function buildSemanticNodeHoverPayload(hit = null) {
    const entry = hit?.entry || null;
    const semantic = entry?.semantic || hit?.node?.semantic || null;
    const role = String(entry?.role || hit?.node?.role || semantic?.role || '').trim().toLowerCase();
    if (!semantic || typeof semantic !== 'object') return null;

    if (isMhsaHeadOverviewEntry(entry || hit?.node)) {
        const label = Number.isFinite(semantic.headIndex)
            ? `Attention Head ${Math.floor(semantic.headIndex) + 1}`
            : 'Attention Head';
        return {
            label,
            info: buildSemanticHoverInfo({
                label,
                layerIndex: semantic.layerIndex,
                headIndex: semantic.headIndex,
                activationStage: 'attention.head',
                suppressTokenChip: true
            })
        };
    }

    if (
        semantic.componentKind === 'layer-norm'
        && (role === 'module-card' || role === 'module-title' || role === 'module')
    ) {
        const layerNormKind = resolveLayerNormKindFromSemanticStage(semantic.stage);
        const label = formatLayerNormLabel(layerNormKind);
        return {
            label,
            info: buildSemanticHoverInfo({
                label,
                layerIndex: semantic.layerIndex,
                activationStage: semantic.stage,
                layerNormKind,
                suppressTokenChip: true
            })
        };
    }

    if (
        role === 'input-token-chip'
        || role === 'input-token-chip-label'
        || role === 'input-token-chip-group'
    ) {
        const tokenIndex = normalizeOptionalIndex(semantic.tokenIndex);
        const tokenLabel = resolveSemanticTokenLabel(hit, tokenIndex);
        const positionIndex = resolveSemanticPositionIndex(hit, tokenIndex);
        const label = buildSemanticTokenChipLabel(tokenLabel, tokenIndex);
        return {
            label,
            info: {
                ...(Number.isFinite(tokenIndex) ? { tokenIndex } : {}),
                ...(tokenLabel.length ? { tokenLabel } : {}),
                ...(Number.isFinite(positionIndex) ? { positionIndex } : {}),
                activationData: {
                    label,
                    stage: 'embedding.token',
                    ...(Number.isFinite(tokenIndex) ? { tokenIndex } : {}),
                    ...(tokenLabel.length ? { tokenLabel } : {}),
                    ...(Number.isFinite(positionIndex) ? { positionIndex } : {})
                }
            }
        };
    }

    if (
        role === 'input-position-chip'
        || role === 'input-position-chip-label'
        || role === 'input-position-chip-group'
    ) {
        const tokenIndex = normalizeOptionalIndex(semantic.tokenIndex);
        const tokenLabel = resolveSemanticTokenLabel(hit, tokenIndex);
        const positionIndex = resolveSemanticPositionIndex(hit, tokenIndex);
        const label = buildSemanticPositionChipLabel(positionIndex);
        return {
            label,
            info: {
                ...(Number.isFinite(tokenIndex) ? { tokenIndex } : {}),
                ...(tokenLabel.length ? { tokenLabel } : {}),
                ...(Number.isFinite(positionIndex) ? { positionIndex } : {}),
                activationData: {
                    label,
                    stage: 'embedding.position',
                    ...(Number.isFinite(tokenIndex) ? { tokenIndex } : {}),
                    ...(tokenLabel.length ? { tokenLabel } : {}),
                    ...(Number.isFinite(positionIndex) ? { positionIndex } : {})
                }
            }
        };
    }

    if (
        role === 'chosen-token-chip'
        || role === 'chosen-token-chip-label'
        || role === 'chosen-token-chip-group'
    ) {
        const tokenIndex = normalizeOptionalIndex(semantic.tokenIndex);
        const tokenLabel = resolveSemanticTokenLabel(hit, tokenIndex);
        const positionIndex = resolveSemanticPositionIndex(hit, tokenIndex);
        const label = buildSemanticChosenTokenChipLabel(tokenLabel, tokenIndex);
        return {
            label,
            info: {
                ...(Number.isFinite(tokenIndex) ? { tokenIndex } : {}),
                ...(tokenLabel.length ? { tokenLabel } : {}),
                ...(Number.isFinite(positionIndex) ? { positionIndex } : {}),
                activationData: {
                    label,
                    stage: 'generation.chosen',
                    ...(Number.isFinite(tokenIndex) ? { tokenIndex } : {}),
                    ...(tokenLabel.length ? { tokenLabel } : {}),
                    ...(Number.isFinite(positionIndex) ? { positionIndex } : {})
                }
            }
        };
    }

    if (
        semantic.componentKind === 'embedding'
        && semantic.stage === 'embedding.token'
        && (
            role === 'module'
            || role === 'module-title'
            || role === 'vocabulary-embedding-card'
        )
    ) {
        const label = 'Vocabulary Embedding Matrix';
        return {
            label,
            info: buildSemanticHoverInfo({
                label,
                activationStage: 'embedding.token',
                suppressTokenChip: true
            })
        };
    }

    if (
        semantic.componentKind === 'embedding'
        && semantic.stage === 'embedding.position'
        && (
            role === 'module'
            || role === 'module-title'
            || role === 'position-embedding-card'
        )
    ) {
        const label = 'Position Embedding Matrix';
        return {
            label,
            info: buildSemanticHoverInfo({
                label,
                activationStage: 'embedding.position',
                suppressTokenChip: true
            })
        };
    }

    if (
        semantic.componentKind === 'logits'
        && semantic.stage === 'unembedding'
        && (
            role === 'module'
            || role === 'module-title'
            || role === 'unembedding'
        )
    ) {
        const label = 'Vocabulary Unembedding Matrix';
        return {
            label,
            info: buildSemanticHoverInfo({
                label,
                activationStage: 'unembedding',
                suppressTokenChip: true
            })
        };
    }

    if (isMlpOverviewEntry(entry || hit?.node)) {
        const label = 'Multilayer Perceptron';
        return {
            label,
            info: buildSemanticHoverInfo({
                label,
                layerIndex: semantic.layerIndex,
                activationStage: 'mlp',
                suppressTokenChip: true
            })
        };
    }

    if (
        semantic.componentKind === 'output-projection'
        && (
            role === 'projection-weight'
            || role === 'module-card'
            || role === 'module-title'
            || role === 'module-title-top'
            || role === 'module-title-bottom'
            || role === 'module'
        )
    ) {
        const label = 'Output Projection Matrix';
        return {
            label,
            info: buildSemanticHoverInfo({
                label,
                layerIndex: semantic.layerIndex,
                activationStage: 'attention.output_projection',
                suppressTokenChip: true
            })
        };
    }

    return null;
}

function resolveOverviewComponentHoverMatchTarget(hit = null) {
    const entry = hit?.entry || hit?.node || null;
    const semantic = entry?.semantic || null;
    if (!semantic || typeof semantic !== 'object') return null;

    if (isMhsaHeadOverviewEntry(entry)) {
        return buildSemanticTarget({
            componentKind: 'mhsa',
            layerIndex: semantic.layerIndex,
            headIndex: semantic.headIndex,
            stage: 'attention'
        });
    }

    if (isOutputProjectionOverviewEntry(entry)) {
        return buildSemanticTarget({
            componentKind: 'output-projection',
            layerIndex: semantic.layerIndex,
            stage: semantic.stage || 'attn-out'
        });
    }

    if (isMlpOverviewEntry(entry)) {
        return buildSemanticTarget({
            componentKind: 'mlp',
            layerIndex: semantic.layerIndex,
            stage: semantic.stage || 'mlp'
        });
    }

    if (isLayerNormOverviewEntry(entry)) {
        return buildSemanticTarget({
            componentKind: 'layer-norm',
            ...(Number.isFinite(semantic.layerIndex) ? { layerIndex: semantic.layerIndex } : {}),
            stage: semantic.stage
        });
    }

    return null;
}

function buildOverviewHoverFocusCacheKey(target = null) {
    if (!target || typeof target !== 'object') return '';
    return [
        String(target.componentKind || ''),
        Number.isFinite(target.layerIndex) ? Math.floor(target.layerIndex) : '',
        Number.isFinite(target.headIndex) ? Math.floor(target.headIndex) : '',
        String(target.stage || '')
    ].join('|');
}

function overviewNodeMatchesHoverTarget(node = null, target = null) {
    if (!node || typeof node !== 'object' || !target || typeof target !== 'object') {
        return false;
    }
    if (node.kind === VIEW2D_NODE_KINDS.CONNECTOR) {
        return false;
    }

    const semantic = node.semantic || {};
    if (semantic.componentKind !== target.componentKind) {
        return false;
    }
    if (String(semantic.stage || '') !== String(target.stage || '')) {
        return false;
    }

    const targetLayerIndex = normalizeOptionalIndex(target.layerIndex);
    const nodeLayerIndex = normalizeOptionalIndex(semantic.layerIndex);
    if (Number.isFinite(targetLayerIndex) || Number.isFinite(nodeLayerIndex)) {
        if (targetLayerIndex !== nodeLayerIndex) {
            return false;
        }
    }

    const targetHeadIndex = normalizeOptionalIndex(target.headIndex);
    const nodeHeadIndex = normalizeOptionalIndex(semantic.headIndex);
    if (Number.isFinite(targetHeadIndex) || Number.isFinite(nodeHeadIndex)) {
        if (targetHeadIndex !== nodeHeadIndex) {
            return false;
        }
    }

    return true;
}

export function buildSemanticNodeHoverFocusState(scene = null, hit = null) {
    const target = resolveOverviewComponentHoverMatchTarget(hit);
    if (!scene || !target) {
        return null;
    }
    const cacheKey = buildOverviewHoverFocusCacheKey(target);
    let sceneCache = OVERVIEW_HOVER_FOCUS_RESULT_CACHE.get(scene);
    if (!sceneCache) {
        sceneCache = new Map();
        OVERVIEW_HOVER_FOCUS_RESULT_CACHE.set(scene, sceneCache);
    } else if (sceneCache.has(cacheKey)) {
        return sceneCache.get(cacheKey) || null;
    }

    const nodes = flattenSceneNodes(scene);
    if (!Array.isArray(nodes) || !nodes.length) {
        sceneCache.set(cacheKey, null);
        return null;
    }

    const activeNodeIds = [];
    const activeNodeIdSet = new Set();
    const connectorNodes = [];

    nodes.forEach((node) => {
        if (!node || typeof node !== 'object' || typeof node.id !== 'string' || !node.id.length) {
            return;
        }
        if (node.kind === VIEW2D_NODE_KINDS.CONNECTOR) {
            connectorNodes.push(node);
            return;
        }
        if (!overviewNodeMatchesHoverTarget(node, target)) {
            return;
        }
        activeNodeIds.push(node.id);
        activeNodeIdSet.add(node.id);
    });

    if (!activeNodeIds.length) {
        sceneCache.set(cacheKey, null);
        return null;
    }

    const activeConnectorIds = connectorNodes.reduce((acc, connectorNode) => {
        const sourceNodeId = typeof connectorNode?.source?.nodeId === 'string'
            ? connectorNode.source.nodeId
            : '';
        const targetNodeId = typeof connectorNode?.target?.nodeId === 'string'
            ? connectorNode.target.nodeId
            : '';
        if (activeNodeIdSet.has(sourceNodeId) || activeNodeIdSet.has(targetNodeId)) {
            acc.push(connectorNode.id);
        }
        return acc;
    }, []);

    const focusResult = buildFocusResult({
        activeNodeIds,
        activeConnectorIds
    });

    const resolvedFocusResult = focusResult?.focusState ? focusResult : null;
    sceneCache.set(cacheKey, resolvedFocusResult);
    return resolvedFocusResult;
}

export function buildHeadDetailSemanticTarget(target = null, role = 'head') {
    const resolvedTarget = resolveHeadDetailTarget(target);
    if (!resolvedTarget) return null;
    return buildSemanticTarget({
        componentKind: 'mhsa',
        layerIndex: resolvedTarget.layerIndex,
        headIndex: resolvedTarget.headIndex,
        stage: 'attention',
        role
    });
}

export function buildConcatDetailSemanticTarget(target = null, role = 'concat') {
    const resolvedTarget = resolveConcatDetailTarget(target);
    if (!resolvedTarget) return null;
    return buildSemanticTarget({
        componentKind: 'mhsa',
        layerIndex: resolvedTarget.layerIndex,
        stage: 'concatenate',
        role
    });
}

export function buildOutputProjectionDetailSemanticTarget(target = null, role = 'projection-weight') {
    const resolvedTarget = resolveOutputProjectionDetailTarget(target);
    if (!resolvedTarget) return null;
    return buildSemanticTarget({
        componentKind: 'output-projection',
        layerIndex: resolvedTarget.layerIndex,
        stage: 'attn-out',
        role
    });
}

export function buildMlpDetailSemanticTarget(target = null, role = 'module') {
    const resolvedTarget = resolveMlpDetailTarget(target);
    if (!resolvedTarget) return null;
    return buildSemanticTarget({
        componentKind: 'mlp',
        layerIndex: resolvedTarget.layerIndex,
        stage: 'mlp',
        role
    });
}

export function buildLayerNormDetailSemanticTarget(target = null, role = 'module') {
    const resolvedTarget = resolveLayerNormDetailTarget(target);
    if (!resolvedTarget) return null;
    return buildSemanticTarget({
        componentKind: 'layer-norm',
        ...(Number.isFinite(resolvedTarget.layerIndex) ? { layerIndex: resolvedTarget.layerIndex } : {}),
        stage: resolvedTarget.layerNormKind === 'final' ? 'final-ln' : resolvedTarget.layerNormKind,
        role
    });
}

function appendUniqueSemanticTarget(targets = [], target = null) {
    const safeTarget = buildSemanticTarget(target);
    if (!safeTarget) return targets;
    const nextKey = JSON.stringify(safeTarget);
    if (targets.some((candidate) => JSON.stringify(candidate) === nextKey)) {
        return targets;
    }
    targets.push(safeTarget);
    return targets;
}

function buildMhsaDetailSemanticTarget({
    layerIndex = null,
    headIndex = null,
    stage = '',
    role = ''
} = {}) {
    if (!Number.isFinite(layerIndex) || !Number.isFinite(headIndex)) {
        return null;
    }
    return buildSemanticTarget({
        componentKind: 'mhsa',
        layerIndex,
        headIndex,
        stage,
        role
    });
}

function resolveMhsaDetailFocusLabel(label = '', stageLower = '') {
    const lower = String(label || '').trim().toLowerCase();
    if (!lower.length) return '';
    if (lower.includes('query weight matrix')) return 'Query Weight Matrix';
    if (lower.includes('key weight matrix')) return 'Key Weight Matrix';
    if (lower.includes('value weight matrix')) return 'Value Weight Matrix';
    if (lower.includes('query bias')) return 'Query Bias Vector';
    if (lower.includes('key bias')) return 'Key Bias Vector';
    if (lower.includes('value bias')) return 'Value Bias Vector';
    if (lower.includes('query vector')) return 'Query Vector';
    if (lower.includes('key vector')) return 'Key Vector';
    if (lower.includes('weighted value vector')) return 'Weighted Value Vector';
    if (lower.includes('value vector')) return 'Value Vector';
    if (lower.includes('attention weighted sum') || stageLower === 'attention.weighted_sum') {
        return 'Attention Weighted Sum';
    }
    if (lower.includes('pre-softmax attention score') || stageLower === 'attention.pre') {
        return 'Pre-Softmax Attention Score';
    }
    if (lower.includes('post-softmax attention score') || stageLower === 'attention.post') {
        return 'Post-Softmax Attention Score';
    }
    if (lower.includes('attention mask')) return 'Attention Mask';
    if (lower.includes('softmax')) return 'Softmax';
    return String(label || '').trim();
}

export function resolveMhsaDetailSemanticTargets(selectionInfo = null, normalizedLabel = '') {
    const label = String(normalizedLabel || selectionInfo?.label || '').trim();
    if (!label.length) return [];

    const lower = label.toLowerCase();
    const activationData = getActivationDataFromSelection(selectionInfo);
    const stageLower = String(activationData?.stage || '').toLowerCase();
    const layerIndex = normalizeOptionalIndex(findUserDataNumber(selectionInfo, 'layerIndex'));
    const headIndex = normalizeOptionalIndex(findUserDataNumber(selectionInfo, 'headIndex'));
    if (!Number.isFinite(layerIndex) || !Number.isFinite(headIndex)) {
        return [];
    }

    const detailTargets = [];
    const appendMhsaTarget = (stage, role) => {
        appendUniqueSemanticTarget(detailTargets, buildMhsaDetailSemanticTarget({
            layerIndex,
            headIndex,
            stage,
            role
        }));
    };

    if (stageLower === 'attention.weighted_sum' || lower.includes('attention weighted sum')) {
        appendMhsaTarget('head-output', 'attention-head-output');
        return detailTargets;
    }

    if (stageLower === 'attention.weighted_value' || lower.includes('weighted value vector')) {
        appendMhsaTarget('head-output', 'attention-head-output-product');
        appendMhsaTarget('head-output', 'attention-value-post');
        return detailTargets;
    }

    if (stageLower === 'attention.pre' || lower.includes('pre-softmax attention score')) {
        appendMhsaTarget('attention', 'attention-pre-score');
        return detailTargets;
    }

    if (stageLower === 'attention.post' || lower.includes('post-softmax attention score')) {
        appendMhsaTarget('attention', 'attention-post');
        return detailTargets;
    }

    if (stageLower === 'attention.mask' || lower.includes('attention mask')) {
        appendMhsaTarget('attention', 'attention-mask');
        return detailTargets;
    }

    if (lower === 'softmax' || lower.includes('softmax(') || lower.includes('softmax')) {
        appendMhsaTarget('attention', 'attention-softmax-body');
        appendMhsaTarget('attention', 'attention-softmax-label');
        return detailTargets;
    }

    if (
        stageLower === 'qkv.q'
        || lower.includes('query ')
    ) {
        if (lower.includes('weight matrix')) {
            appendMhsaTarget('projection-q', 'projection-weight');
        } else if (lower.includes('bias')) {
            appendMhsaTarget('projection-q', 'projection-bias');
        } else {
            appendMhsaTarget('projection-q', 'projection-output');
            appendMhsaTarget('attention', 'attention-query-source');
        }
        return detailTargets;
    }

    if (
        stageLower === 'qkv.k'
        || lower.includes('key ')
    ) {
        if (lower.includes('weight matrix')) {
            appendMhsaTarget('projection-k', 'projection-weight');
        } else if (lower.includes('bias')) {
            appendMhsaTarget('projection-k', 'projection-bias');
        } else {
            appendMhsaTarget('projection-k', 'projection-output');
            appendMhsaTarget('attention', 'attention-key-transpose');
        }
        return detailTargets;
    }

    if (
        stageLower === 'qkv.v'
        || lower.includes('value ')
    ) {
        if (lower.includes('weight matrix')) {
            appendMhsaTarget('projection-v', 'projection-weight');
        } else if (lower.includes('bias')) {
            appendMhsaTarget('projection-v', 'projection-bias');
        } else {
            appendMhsaTarget('projection-v', 'projection-output');
            appendMhsaTarget('head-output', 'attention-value-post');
        }
        return detailTargets;
    }

    return detailTargets;
}

export function deriveBaseSemanticTarget(target = null) {
    const safeTarget = buildSemanticTarget(target);
    if (
        safeTarget?.componentKind === 'mhsa'
        && safeTarget?.stage === 'head-detail'
        && Number.isFinite(safeTarget?.layerIndex)
        && Number.isFinite(safeTarget?.headIndex)
    ) {
        return buildSemanticTarget({
            componentKind: 'mhsa',
            layerIndex: safeTarget.layerIndex,
            headIndex: safeTarget.headIndex,
            stage: 'attention',
            role: 'head'
        });
    }
    return safeTarget;
}

export function resolveDetailTargetsFromSemanticTarget(target = null) {
    const safeTarget = buildSemanticTarget(target);
    const headDetailTarget = (
        safeTarget?.componentKind === 'mhsa'
        && Number.isFinite(safeTarget?.headIndex)
    )
        ? resolveHeadDetailTarget(safeTarget)
        : null;
    const redirectedConcatDetailTarget = (
        !headDetailTarget
        && safeTarget?.componentKind === 'mhsa'
        && safeTarget?.stage === 'concatenate'
    )
        ? resolveConcatDetailTarget(safeTarget)
        : null;
    const outputProjectionDetailTarget = (
        !headDetailTarget
        && (
            redirectedConcatDetailTarget
            || safeTarget?.componentKind === 'output-projection'
        )
    )
        ? resolveOutputProjectionDetailTarget(
            redirectedConcatDetailTarget
                ? { layerIndex: redirectedConcatDetailTarget.layerIndex }
                : safeTarget
        )
        : null;
    const mlpDetailTarget = (
        !headDetailTarget
        && !outputProjectionDetailTarget
        && safeTarget?.componentKind === 'mlp'
    )
        ? resolveMlpDetailTarget(safeTarget)
        : null;
    const layerNormDetailTarget = (
        !headDetailTarget
        && !outputProjectionDetailTarget
        && !mlpDetailTarget
        && safeTarget?.componentKind === 'layer-norm'
    )
        ? resolveLayerNormDetailTarget({
            layerNormKind: resolveLayerNormKindFromSemanticStage(safeTarget?.stage),
            layerIndex: safeTarget?.layerIndex
        })
        : null;

    return {
        headDetailTarget,
        concatDetailTarget: null,
        outputProjectionDetailTarget,
        mlpDetailTarget,
        layerNormDetailTarget
    };
}

export function resolveTransformerView2dOpenTransitionMode({
    semanticTarget = null
} = {}) {
    const safeTarget = buildSemanticTarget(semanticTarget);
    if (!safeTarget) return '';

    const detailTargets = resolveDetailTargetsFromSemanticTarget(safeTarget);
    if (detailTargets.headDetailTarget) {
        return 'staged-head-detail';
    }
    if (
        detailTargets.concatDetailTarget
        || detailTargets.outputProjectionDetailTarget
        || detailTargets.mlpDetailTarget
        || detailTargets.layerNormDetailTarget
    ) {
        return '';
    }
    return 'staged-focus';
}

export function hasActiveDetailTarget({
    headDetailTarget = null,
    concatDetailTarget = null,
    outputProjectionDetailTarget = null,
    mlpDetailTarget = null,
    layerNormDetailTarget = null
} = {}) {
    return !!(
        headDetailTarget
        || concatDetailTarget
        || outputProjectionDetailTarget
        || mlpDetailTarget
        || layerNormDetailTarget
    );
}

export function resolveActiveSemanticTarget({
    baseSemanticTarget = null,
    headDetailTarget = null,
    concatDetailTarget = null,
    outputProjectionDetailTarget = null,
    mlpDetailTarget = null,
    layerNormDetailTarget = null
} = {}) {
    if (headDetailTarget) return buildHeadDetailSemanticTarget(headDetailTarget);
    if (concatDetailTarget) {
        return buildOutputProjectionDetailSemanticTarget({
            layerIndex: resolveConcatDetailTarget(concatDetailTarget)?.layerIndex
        });
    }
    if (outputProjectionDetailTarget) return buildOutputProjectionDetailSemanticTarget(outputProjectionDetailTarget);
    if (mlpDetailTarget) return buildMlpDetailSemanticTarget(mlpDetailTarget);
    if (layerNormDetailTarget) return buildLayerNormDetailSemanticTarget(layerNormDetailTarget);
    return buildSemanticTarget(baseSemanticTarget);
}

export function resolveActiveFocusLabel({
    baseSemanticTarget = null,
    baseFocusLabel = '',
    headDetailTarget = null,
    concatDetailTarget = null,
    outputProjectionDetailTarget = null,
    mlpDetailTarget = null,
    layerNormDetailTarget = null
} = {}) {
    const semanticTarget = resolveActiveSemanticTarget({
        baseSemanticTarget,
        headDetailTarget,
        concatDetailTarget,
        outputProjectionDetailTarget,
        mlpDetailTarget,
        layerNormDetailTarget
    });
    if (
        headDetailTarget
        || concatDetailTarget
        || outputProjectionDetailTarget
        || mlpDetailTarget
        || layerNormDetailTarget
    ) {
        return describeTransformerView2dTarget(semanticTarget);
    }
    return String(baseFocusLabel || '').trim() || describeTransformerView2dTarget(semanticTarget);
}

export function resolveFocusSemanticTargets({
    semanticTarget = null,
    baseSemanticTarget = null,
    headDetailTarget = null,
    concatDetailTarget = null,
    outputProjectionDetailTarget = null,
    mlpDetailTarget = null,
    layerNormDetailTarget = null
} = {}) {
    const candidates = [];
    if (headDetailTarget) {
        candidates.push(
            buildHeadDetailSemanticTarget(headDetailTarget, 'head-card'),
            buildHeadDetailSemanticTarget(headDetailTarget)
        );
    } else if (concatDetailTarget) {
        const redirectedTarget = resolveConcatDetailTarget(concatDetailTarget);
        candidates.push(
            buildOutputProjectionDetailSemanticTarget(redirectedTarget, 'projection-weight'),
            buildOutputProjectionDetailSemanticTarget(redirectedTarget, 'module')
        );
    } else if (outputProjectionDetailTarget) {
        candidates.push(
            buildOutputProjectionDetailSemanticTarget(outputProjectionDetailTarget, 'projection-weight'),
            buildOutputProjectionDetailSemanticTarget(outputProjectionDetailTarget, 'module')
        );
    } else if (mlpDetailTarget) {
        candidates.push(
            buildMlpDetailSemanticTarget(mlpDetailTarget, 'module-card'),
            buildMlpDetailSemanticTarget(mlpDetailTarget, 'module-title'),
            buildMlpDetailSemanticTarget(mlpDetailTarget, 'module')
        );
    } else if (layerNormDetailTarget) {
        candidates.push(
            buildLayerNormDetailSemanticTarget(layerNormDetailTarget, 'module-card'),
            buildLayerNormDetailSemanticTarget(layerNormDetailTarget, 'module-title'),
            buildLayerNormDetailSemanticTarget(layerNormDetailTarget, 'module')
        );
    } else {
        const activeTarget = buildSemanticTarget(semanticTarget)
            || resolveActiveSemanticTarget({
                baseSemanticTarget,
                headDetailTarget,
                concatDetailTarget,
                outputProjectionDetailTarget,
                mlpDetailTarget,
                layerNormDetailTarget
            });
        if (activeTarget?.componentKind === 'layer-norm') {
            candidates.push(
                buildSemanticTarget({
                    ...activeTarget,
                    role: 'module-card'
                }),
                buildSemanticTarget({
                    ...activeTarget,
                    role: 'module-title'
                }),
                buildSemanticTarget({
                    ...activeTarget,
                    role: 'module'
                })
            );
        } else {
            candidates.push(activeTarget);
        }
    }

    const seen = new Set();
    return candidates.filter((candidate) => {
        if (!candidate) return false;
        const key = JSON.stringify(candidate);
        if (seen.has(key)) return false;
        seen.add(key);
        return true;
    });
}

export function isMhsaHeadOverviewEntry(entry = null) {
    const semantic = entry?.semantic;
    if (!semantic || semantic.componentKind !== 'mhsa') return false;
    if (!Number.isFinite(semantic.layerIndex) || !Number.isFinite(semantic.headIndex)) return false;
    if (semantic.stage !== 'attention') return false;
    return entry.role === 'head' || entry.role === 'head-card' || entry.role === 'head-label';
}

export function isConcatOverviewEntry(entry = null) {
    const semantic = entry?.semantic;
    if (!semantic || semantic.componentKind !== 'mhsa') return false;
    if (!Number.isFinite(semantic.layerIndex) || Number.isFinite(semantic.headIndex)) return false;
    if (semantic.stage !== 'concatenate') return false;
    return entry.role === 'concat'
        || entry.role === 'concat-card'
        || entry.role === 'module-title';
}

export function isOutputProjectionOverviewEntry(entry = null) {
    const semantic = entry?.semantic;
    if (!semantic || semantic.componentKind !== 'output-projection') return false;
    if (!Number.isFinite(semantic.layerIndex)) return false;
    return entry.role === 'projection-weight'
        || entry.role === 'module-title'
        || entry.role === 'module-title-top'
        || entry.role === 'module-title-bottom';
}

export function isMlpOverviewEntry(entry = null) {
    const semantic = entry?.semantic;
    if (!semantic || semantic.componentKind !== 'mlp') return false;
    if (!Number.isFinite(semantic.layerIndex)) return false;
    if (semantic.stage !== 'mlp') return false;
    return entry.role === 'module-card'
        || entry.role === 'module-title'
        || entry.role === 'module-title-top'
        || entry.role === 'module-title-bottom'
        || entry.role === 'module';
}

export function isLayerNormOverviewEntry(entry = null) {
    const semantic = entry?.semantic;
    if (!semantic || semantic.componentKind !== 'layer-norm') return false;
    const stage = String(semantic.stage || '').trim();
    if (stage !== 'ln1' && stage !== 'ln2' && stage !== 'final-ln') return false;
    return entry.role === 'module-card'
        || entry.role === 'module-title'
        || entry.role === 'module';
}

function isTopUnembeddingLabel(lower = '') {
    return lower.includes('vocab embedding (top)')
        || lower.includes('vocabulary embedding (top)')
        || lower.includes('vocab unembedding')
        || lower.includes('vocabulary unembedding')
        || lower.includes('unembedding');
}

function resolveLayerNormStage(kind = null) {
    if (kind === 'ln1') return 'ln1';
    if (kind === 'ln2') return 'ln2';
    if (kind === 'final') return 'final-ln';
    return null;
}

function resolveLayerLabel(layerIndex = null) {
    return Number.isFinite(layerIndex) ? `Layer ${Math.floor(layerIndex) + 1}` : '';
}

function resolveTransformerView2dStageName(target = null) {
    if (!target) return TRANSFORMER_VIEW2D_OVERVIEW_LABEL;
    const embeddingStage = normalizeEmbeddingSemanticStage(target.stage);
    const residualStage = normalizeResidualSemanticStage(target.stage);

    if (target.componentKind === 'layer-norm') {
        if (target.stage === 'ln1') return 'LayerNorm 1';
        if (target.stage === 'ln2') return 'LayerNorm 2';
        if (target.stage === 'final-ln') return 'Final LayerNorm';
        return 'LayerNorm';
    }

    if (target.componentKind === 'mhsa') {
        if (Number.isFinite(target.headIndex)) {
            return `Attention Head ${Math.floor(target.headIndex) + 1}`;
        }
        if (target.stage === 'concatenate') return 'Output Projection';
        return 'Self-Attention';
    }

    if (target.componentKind === 'output-projection') {
        return 'Output Projection';
    }

    if (target.componentKind === 'mlp') {
        return 'Multilayer Perceptron';
    }

    if (target.componentKind === 'embedding') {
        if (embeddingStage === 'embedding.token') return 'Token Embeddings';
        if (embeddingStage === 'embedding.position') return 'Position Embeddings';
        if (embeddingStage === 'embedding.sum') return 'Embedding Sum';
        return 'Embeddings';
    }

    if (target.componentKind === 'residual') {
        if (residualStage === 'incoming') return 'Incoming Residual';
        if (residualStage === 'post-attn-residual') return 'Post-Attention Residual';
        if (residualStage === 'post-mlp-residual') return 'Post-MLP Residual';
        if (residualStage === 'outgoing') return 'Outgoing Residual';
        return 'Residual Stream';
    }

    if (target.componentKind === 'logits') {
        if (target.role === 'chosen-token-chip-group' || target.role === 'chosen-token-chip') {
            return 'Chosen Token';
        }
        if (target.stage === 'unembedding') return 'Unembedding';
        if (target.role === 'unembedding') return 'Unembedding';
        return 'Logits';
    }

    const description = describeTransformerView2dTarget(target);
    const layerLabel = resolveLayerLabel(target?.layerIndex);
    if (layerLabel && description.startsWith(`${layerLabel} `)) {
        return description.slice(layerLabel.length + 1);
    }
    return description;
}

export function resolveTransformerView2dStageHeader(target = null) {
    const safeTarget = buildSemanticTarget(target);
    const layerLabel = resolveLayerLabel(safeTarget?.layerIndex);
    const stageLabel = resolveTransformerView2dStageName(safeTarget);
    return {
        layerLabel,
        stageLabel,
        fullLabel: layerLabel ? `${layerLabel} ${stageLabel}` : stageLabel
    };
}

export function describeTransformerView2dTarget(target = null) {
    if (!target) return TRANSFORMER_VIEW2D_OVERVIEW_LABEL;
    const layerLabel = resolveLayerLabel(target.layerIndex);
    const embeddingStage = normalizeEmbeddingSemanticStage(target.stage);
    const residualStage = normalizeResidualSemanticStage(target.stage);
    if (target.componentKind === 'embedding') {
        if (embeddingStage === 'embedding.token') return 'Token embeddings';
        if (embeddingStage === 'embedding.position') return 'Position embeddings';
        if (embeddingStage === 'embedding.sum') return 'Embedding sum';
        return 'Embeddings';
    }
    if (target.componentKind === 'layer-norm') {
        if (target.stage === 'ln1') return layerLabel ? `${layerLabel} LayerNorm 1` : 'LayerNorm 1';
        if (target.stage === 'ln2') return layerLabel ? `${layerLabel} LayerNorm 2` : 'LayerNorm 2';
        return 'Final LayerNorm';
    }
    if (target.componentKind === 'mhsa') {
        if (Number.isFinite(target.headIndex)) {
            const headLabel = `Attention Head ${Math.floor(target.headIndex) + 1}`;
            return layerLabel ? `${layerLabel} ${headLabel}` : headLabel;
        }
        if (target.stage === 'concatenate') {
            return layerLabel ? `${layerLabel} Output Projection` : 'Output Projection';
        }
        return layerLabel ? `${layerLabel} Self-Attention` : 'Self-Attention';
    }
    if (target.componentKind === 'output-projection') {
        return layerLabel ? `${layerLabel} Output Projection` : 'Output Projection';
    }
    if (target.componentKind === 'mlp') {
        if (target.stage === 'mlp-up') {
            return layerLabel
                ? `${layerLabel} Multilayer Perceptron Up Projection`
                : 'Multilayer Perceptron Up Projection';
        }
        if (target.stage === 'mlp-down') {
            return layerLabel
                ? `${layerLabel} Multilayer Perceptron Down Projection`
                : 'Multilayer Perceptron Down Projection';
        }
        if (target.stage === 'mlp-activation') return layerLabel ? `${layerLabel} GELU Activation` : 'GELU Activation';
        return layerLabel ? `${layerLabel} Multilayer Perceptron` : 'Multilayer Perceptron';
    }
    if (target.componentKind === 'residual') {
        if (residualStage === 'incoming') return layerLabel ? `${layerLabel} incoming residual` : 'Incoming residual';
        if (residualStage === 'post-attn-residual') return layerLabel ? `${layerLabel} post-attention residual` : 'Post-attention residual';
        if (residualStage === 'post-mlp-residual') return layerLabel ? `${layerLabel} post-MLP residual` : 'Post-MLP residual';
        if (residualStage === 'outgoing') return layerLabel ? `${layerLabel} outgoing residual` : 'Outgoing residual';
        return layerLabel ? `${layerLabel} residual stream` : 'Residual stream';
    }
    if (target.componentKind === 'logits') {
        if (target.role === 'chosen-token-chip-group' || target.role === 'chosen-token-chip') {
            return 'Chosen Token';
        }
        if (target.stage === 'unembedding') return 'Unembedding';
        if (target.role === 'unembedding') return 'Unembedding';
        if (target.role === 'logits-topk') return 'Logits';
        return 'Logits';
    }
    return TRANSFORMER_VIEW2D_OVERVIEW_LABEL;
}

export function resolveTransformerView2dActionContext(selectionInfo = null, normalizedLabel = '') {
    const label = String(normalizedLabel || selectionInfo?.label || '').trim();
    if (!label.length) return null;

    const lower = label.toLowerCase();
    const activationData = getActivationDataFromSelection(selectionInfo);
    const stageLower = String(activationData?.stage || '').toLowerCase();
    const layerIndex = normalizeOptionalIndex(findUserDataNumber(selectionInfo, 'layerIndex'));
    const headIndex = normalizeOptionalIndex(findUserDataNumber(selectionInfo, 'headIndex'));
    const tokenIndex = normalizeOptionalIndex(findUserDataNumber(selectionInfo, 'tokenIndex'));
    const positionIndex = normalizeOptionalIndex(findUserDataNumber(selectionInfo, 'positionIndex'));
    const explicitLayerNormKind = findUserDataString(selectionInfo, 'layerNormKind');
    const layerNormKind = resolveLayerNormKind({
        label,
        stage: stageLower,
        explicitKind: explicitLayerNormKind
    });
    const topUnembeddingLabel = isTopUnembeddingLabel(lower);

    let semanticTarget = null;
    let detailSemanticTargets = [];
    let detailFocusLabel = '';

    if (layerNormKind) {
        const layerNormContext = resolveLayerNormDetailSemanticTargets(selectionInfo, label);
        semanticTarget = layerNormContext.semanticTarget;
        detailSemanticTargets = layerNormContext.detailSemanticTargets;
        detailFocusLabel = detailSemanticTargets.length ? label : '';
    } else if (
        isLogitBarSelection(label, selectionInfo)
        || lower.includes('top logit bars')
        || topUnembeddingLabel
        || lower === 'logit'
    ) {
        semanticTarget = buildSemanticTarget({
            componentKind: 'logits',
            stage: topUnembeddingLabel ? 'unembedding' : 'output',
            role: topUnembeddingLabel ? 'unembedding' : 'logits-topk'
        });
    } else if (
        stageLower.startsWith('generation.chosen')
        || lower.startsWith('chosen token:')
    ) {
        semanticTarget = buildSemanticTarget({
            componentKind: 'logits',
            stage: 'output',
            role: 'chosen-token-chip-group',
            ...(Number.isFinite(tokenIndex) ? { tokenIndex } : {}),
            ...(Number.isFinite(positionIndex) ? { positionIndex } : {})
        });
    } else if (
        stageLower.startsWith('embedding.token')
        || lower.startsWith('token:')
        || lower.includes('token embedding')
    ) {
        const role = (
            Number.isFinite(tokenIndex)
            && !lower.includes('vocabulary embedding')
            && !lower.includes('vocab embedding')
        )
            ? 'input-token-chip-group'
            : 'module';
        semanticTarget = buildSemanticTarget({
            componentKind: 'embedding',
            stage: 'embedding.token',
            role,
            ...(role === 'input-token-chip-group' && Number.isFinite(tokenIndex) ? { tokenIndex } : {}),
            ...(role === 'input-token-chip-group' && Number.isFinite(positionIndex) ? { positionIndex } : {})
        });
    } else if (
        stageLower.startsWith('embedding.position')
        || lower.startsWith('position:')
        || lower.includes('position embedding')
        || lower.includes('positional embedding')
    ) {
        const role = Number.isFinite(tokenIndex) ? 'input-position-chip-group' : 'module';
        semanticTarget = buildSemanticTarget({
            componentKind: 'embedding',
            stage: 'embedding.position',
            role,
            ...(role === 'input-position-chip-group' && Number.isFinite(tokenIndex) ? { tokenIndex } : {}),
            ...(role === 'input-position-chip-group' && Number.isFinite(positionIndex) ? { positionIndex } : {})
        });
    } else if (
        stageLower.startsWith('embedding.sum')
        || lower.includes('embedding sum')
        || (lower.includes('vocabulary embedding') && !topUnembeddingLabel)
        || (lower.includes('vocab embedding') && !topUnembeddingLabel)
    ) {
        semanticTarget = buildSemanticTarget({
            componentKind: 'embedding',
            stage: stageLower.startsWith('embedding.sum') || lower.includes('embedding sum')
                ? 'embedding.sum'
                : 'embedding.token',
            role: 'module'
        });
    } else if (
        stageLower === 'attention.concatenate'
        || lower === 'concatenate'
        || lower === 'concatenation'
        || lower.includes('concatenate heads')
        || lower.includes('head concatenation')
    ) {
        const outputProjectionContext = resolveOutputProjectionDetailSemanticTargets(selectionInfo, label);
        semanticTarget = outputProjectionContext.semanticTarget;
        detailSemanticTargets = outputProjectionContext.detailSemanticTargets;
        detailFocusLabel = detailSemanticTargets.length ? label : '';
    } else if (
        lower.includes('output projection matrix')
        || lower.includes('alpha projection matrix')
        || lower.includes('alpha projection')
        || lower.includes('output projection bias')
        || stageLower === 'attention.output_projection'
        || stageLower === 'attention.output_projection.bias'
        || lower.includes('attention output projection')
        || lower.includes('attention output vector')
    ) {
        const outputProjectionContext = resolveOutputProjectionDetailSemanticTargets(selectionInfo, label);
        semanticTarget = outputProjectionContext.semanticTarget;
        detailSemanticTargets = outputProjectionContext.detailSemanticTargets;
        detailFocusLabel = detailSemanticTargets.length ? label : '';
    } else if (
        lower.includes('mlp up weight matrix')
        || lower.includes('mlp up projection')
        || lower.includes('bias vector for mlp up matrix')
    ) {
        const mlpContext = resolveMlpDetailSemanticTargets(selectionInfo, label);
        semanticTarget = mlpContext.semanticTarget;
        detailSemanticTargets = mlpContext.detailSemanticTargets;
        detailFocusLabel = detailSemanticTargets.length ? label : '';
    } else if (
        lower.includes('mlp down weight matrix')
        || lower.includes('mlp down projection')
        || lower.includes('bias vector') && lower.includes('mlp down')
    ) {
        const mlpContext = resolveMlpDetailSemanticTargets(selectionInfo, label);
        semanticTarget = mlpContext.semanticTarget;
        detailSemanticTargets = mlpContext.detailSemanticTargets;
        detailFocusLabel = detailSemanticTargets.length ? label : '';
    } else if (
        lower.includes('gelu')
        || lower.includes('mlp activation')
    ) {
        const mlpContext = resolveMlpDetailSemanticTargets(selectionInfo, label);
        semanticTarget = mlpContext.semanticTarget;
        detailSemanticTargets = mlpContext.detailSemanticTargets;
        detailFocusLabel = detailSemanticTargets.length ? label : '';
    } else if (
        lower.includes('mlp')
        && Number.isFinite(layerIndex)
    ) {
        semanticTarget = buildMlpOverviewSemanticTarget(layerIndex);
    } else if (
        stageLower.startsWith('qkv.')
        || stageLower.startsWith('attention.')
        || isSelfAttentionSelection(label, selectionInfo)
        || isWeightedSumSelection(label, selectionInfo)
        || isValueSelection(label, selectionInfo)
        || isQkvMatrixLabel(label)
        || lower.includes('self-attention')
    ) {
        if (Number.isFinite(layerIndex)) {
            semanticTarget = buildSemanticTarget({
                componentKind: 'mhsa',
                layerIndex,
                headIndex,
                stage: 'attention',
                role: Number.isFinite(headIndex) ? 'head' : 'module'
            });
        }
    } else if (
        stageLower.startsWith('layer.incoming')
        || lower.includes('incoming residual')
    ) {
        if (Number.isFinite(layerIndex)) {
            semanticTarget = buildSemanticTarget({
                componentKind: 'residual',
                layerIndex,
                stage: 'incoming',
                role: 'module'
            });
        }
    } else if (
        stageLower.includes('post_attention')
        || lower.includes('post-attention residual')
        || lower.includes('post attention residual')
    ) {
        if (Number.isFinite(layerIndex)) {
            semanticTarget = buildSemanticTarget({
                componentKind: 'residual',
                layerIndex,
                stage: 'post-attn-residual',
                role: 'module'
            });
        }
    } else if (
        stageLower.includes('post_mlp')
        || lower.includes('post-mlp residual')
        || lower.includes('post mlp residual')
    ) {
        if (Number.isFinite(layerIndex)) {
            semanticTarget = buildSemanticTarget({
                componentKind: 'residual',
                layerIndex,
                stage: 'post-mlp-residual',
                role: 'module'
            });
        }
    } else if (lower.includes('residual stream vector') && Number.isFinite(layerIndex)) {
        semanticTarget = buildSemanticTarget({
            componentKind: 'residual',
            layerIndex,
            stage: 'incoming',
            role: 'module'
        });
    }

    if (!semanticTarget) return null;
    const mhsaDetailSemanticTargets = (
        semanticTarget?.componentKind === 'mhsa'
        && Number.isFinite(semanticTarget?.headIndex)
    )
        ? resolveMhsaDetailSemanticTargets(selectionInfo, label)
        : [];
    if (mhsaDetailSemanticTargets.length) {
        detailSemanticTargets = mhsaDetailSemanticTargets;
    }
    const resolvedDetailFocusLabel = detailSemanticTargets.length
        ? resolveMhsaDetailFocusLabel(label, stageLower)
        : detailFocusLabel;
    const transitionMode = (
        semanticTarget?.componentKind === 'mhsa'
        && Number.isFinite(semanticTarget?.headIndex)
    )
        ? 'staged-head-detail'
        : (
            semanticTarget?.componentKind === 'output-projection'
            || semanticTarget?.componentKind === 'mlp'
            || semanticTarget?.componentKind === 'layer-norm'
        )
            ? 'staged-detail'
            : resolveTransformerView2dOpenTransitionMode({
                semanticTarget
            });
    const focusLabel = describeTransformerView2dTarget(semanticTarget);
    return {
        semanticTarget,
        focusLabel,
        ...(detailSemanticTargets.length ? { detailSemanticTargets } : {}),
        ...(resolvedDetailFocusLabel.length ? { detailFocusLabel: resolvedDetailFocusLabel } : {}),
        ...(transitionMode ? { transitionMode } : {}),
        actionLabel: 'View in 2D / matrix form'
    };
}
