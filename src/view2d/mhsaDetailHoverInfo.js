import { buildAttentionHoverInfo } from '../ui/attentionHoverInfo.js';
import {
    formatLayerNormLabel,
    formatLayerNormParamLabel,
    normalizeLayerNormOutputStage,
    normalizePostLayerNormResidualStage,
    resolveLayerNormKind,
    resolvePostLayerNormResidualLabel
} from '../utils/layerNormLabels.js';
import {
    MLP_ACTIVATION_TOOLTIP_LABEL,
    MLP_DOWN_BIAS_TOOLTIP_LABEL,
    MLP_DOWN_TOOLTIP_LABEL
} from '../utils/mlpLabels.js';

const PROJECTION_KIND_LABELS = Object.freeze({
    q: 'Query',
    k: 'Key',
    v: 'Value'
});

export function normalizeProjectionKind(value = '') {
    const safe = String(value || '').trim().toLowerCase();
    return safe === 'q' || safe === 'k' || safe === 'v' ? safe : '';
}

export function resolveProjectionKindForNode(node = null) {
    const stageValue = String(node?.semantic?.stage || '').trim().toLowerCase();
    if (stageValue.startsWith('projection-')) {
        return normalizeProjectionKind(stageValue.slice('projection-'.length));
    }
    return normalizeProjectionKind(node?.metadata?.kind || '');
}

export function resolveProjectionLabel(kind = '') {
    return PROJECTION_KIND_LABELS[normalizeProjectionKind(kind)] || 'Projection';
}

function normalizeSceneIndex(value = null) {
    return Number.isFinite(value) ? Math.max(0, Math.floor(value)) : null;
}

export function createTokenInfo(rowItem = null) {
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

export function hasNumericValue(value) {
    return typeof value === 'number' && !Number.isNaN(value);
}

export function buildProjectionHoverInfo(node = null, label = '', extraActivationData = {}) {
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

export function buildProjectionVectorHoverInfo(node = null, rowItem = null, kind = '') {
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

export function buildMlpUpProjectionHoverInfo(node = null, rowItem = null) {
    const tokenInfo = createTokenInfo(rowItem) || {};
    const label = 'MLP Up Projection';
    const info = buildProjectionHoverInfo(node, label, {
        stage: 'mlp.up',
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
            stage: 'mlp.up',
            ...(Number.isFinite(tokenInfo.tokenIndex) ? { tokenIndex: tokenInfo.tokenIndex } : {}),
            ...(typeof tokenInfo.tokenLabel === 'string' && tokenInfo.tokenLabel.length
                ? { tokenLabel: tokenInfo.tokenLabel }
                : {})
        }
    };
}

export function buildMlpUpWeightHoverInfo(node = null) {
    const label = 'MLP Up Weight Matrix';
    const info = buildProjectionHoverInfo(node, label, {
        stage: 'mlp.up'
    }) || {};

    return {
        ...info,
        activationData: {
            ...(info.activationData || {}),
            stage: 'mlp.up'
        }
    };
}

export function buildMlpActivationHoverInfo(node = null, rowItem = null) {
    const tokenInfo = createTokenInfo(rowItem) || {};
    const label = MLP_ACTIVATION_TOOLTIP_LABEL;
    const info = buildProjectionHoverInfo(node, label, {
        stage: 'mlp.activation',
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
            stage: 'mlp.activation',
            ...(Number.isFinite(tokenInfo.tokenIndex) ? { tokenIndex: tokenInfo.tokenIndex } : {}),
            ...(typeof tokenInfo.tokenLabel === 'string' && tokenInfo.tokenLabel.length
                ? { tokenLabel: tokenInfo.tokenLabel }
                : {})
        }
    };
}

export function buildMlpDownProjectionHoverInfo(node = null, rowItem = null) {
    const tokenInfo = createTokenInfo(rowItem) || {};
    const label = 'MLP Down Projection';
    const info = buildProjectionHoverInfo(node, label, {
        stage: 'mlp.down',
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
            stage: 'mlp.down',
            ...(Number.isFinite(tokenInfo.tokenIndex) ? { tokenIndex: tokenInfo.tokenIndex } : {}),
            ...(typeof tokenInfo.tokenLabel === 'string' && tokenInfo.tokenLabel.length
                ? { tokenLabel: tokenInfo.tokenLabel }
                : {})
        }
    };
}

export function buildMlpUpBiasHoverInfo(node = null) {
    const label = 'Bias Vector for MLP Up Matrix';
    const info = buildProjectionHoverInfo(node, label, {
        stage: 'mlp.up.bias'
    }) || {};

    return {
        ...info,
        activationData: {
            ...(info.activationData || {}),
            stage: 'mlp.up.bias'
        }
    };
}

export function buildMlpDownWeightHoverInfo(node = null) {
    const label = MLP_DOWN_TOOLTIP_LABEL;
    const info = buildProjectionHoverInfo(node, label, {
        stage: 'mlp.down',
        suppressTokenChip: true
    }) || {};

    return {
        ...info,
        suppressTokenChip: true,
        activationData: {
            ...(info.activationData || {}),
            stage: 'mlp.down',
            suppressTokenChip: true
        }
    };
}

export function buildMlpDownBiasHoverInfo(node = null) {
    const label = MLP_DOWN_BIAS_TOOLTIP_LABEL;
    const info = buildProjectionHoverInfo(node, label, {
        stage: 'mlp.down.bias'
    }) || {};

    return {
        ...info,
        activationData: {
            ...(info.activationData || {}),
            stage: 'mlp.down.bias'
        }
    };
}

export function buildPostLayerNormResidualHoverInfo(node = null, rowItem = null) {
    const tokenInfo = createTokenInfo(rowItem) || {};
    const rowStage = String(rowItem?.semantic?.stage || '').trim().toLowerCase();
    const activationStage = normalizePostLayerNormResidualStage(
        rowStage === 'ln2.output' || rowStage === 'ln2.shift' ? rowStage : 'ln1.output'
    ) || 'ln1.output';
    const sourceStage = normalizePostLayerNormResidualStage(activationStage, { preferLegacy: true });
    const label = resolvePostLayerNormResidualLabel({ stage: activationStage });
    const info = buildProjectionHoverInfo(node, label, {
        stage: activationStage,
        sourceStage,
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
            stage: activationStage,
            sourceStage,
            ...(Number.isFinite(tokenInfo.tokenIndex) ? { tokenIndex: tokenInfo.tokenIndex } : {}),
            ...(typeof tokenInfo.tokenLabel === 'string' && tokenInfo.tokenLabel.length
                ? { tokenLabel: tokenInfo.tokenLabel }
                : {})
        }
    };
}

function resolveLayerNormStageKey(node = null, rowItem = null) {
    const rowStage = String(rowItem?.semantic?.stage || '').trim();
    if (rowStage.length) return rowStage;
    return String(node?.semantic?.stage || '').trim();
}

function resolveLayerNormHoverKind(node = null, rowItem = null) {
    const stage = resolveLayerNormStageKey(node, rowItem);
    return resolveLayerNormKind({
        stage,
        explicitKind: node?.semantic?.layerNormKind || rowItem?.semantic?.layerNormKind || null
    });
}

function buildLayerNormHoverInfo(node = null, rowItem = null, {
    label = '',
    stage = '',
    sourceStage = ''
} = {}) {
    const tokenInfo = createTokenInfo(rowItem) || {};
    const layerNormKind = resolveLayerNormHoverKind(node, rowItem);
    const resolvedStage = String(stage || resolveLayerNormStageKey(node, rowItem) || '').trim().toLowerCase();
    const resolvedSourceStage = String(sourceStage || resolveLayerNormStageKey(node, rowItem) || '').trim().toLowerCase();
    const info = buildProjectionHoverInfo(node, label, {
        stage: resolvedStage,
        ...(resolvedSourceStage.length ? { sourceStage: resolvedSourceStage } : {}),
        ...(layerNormKind ? { layerNormKind } : {}),
        ...(Number.isFinite(tokenInfo.tokenIndex) ? { tokenIndex: tokenInfo.tokenIndex } : {}),
        ...(typeof tokenInfo.tokenLabel === 'string' && tokenInfo.tokenLabel.length
            ? { tokenLabel: tokenInfo.tokenLabel }
            : {})
    }) || {};

    return {
        ...tokenInfo,
        ...info,
        ...(layerNormKind ? { layerNormKind } : {}),
        activationData: {
            ...(info.activationData || {}),
            stage: resolvedStage,
            ...(resolvedSourceStage.length ? { sourceStage: resolvedSourceStage } : {}),
            ...(layerNormKind ? { layerNormKind } : {}),
            ...(Number.isFinite(tokenInfo.tokenIndex) ? { tokenIndex: tokenInfo.tokenIndex } : {}),
            ...(typeof tokenInfo.tokenLabel === 'string' && tokenInfo.tokenLabel.length
                ? { tokenLabel: tokenInfo.tokenLabel }
                : {})
        }
    };
}

export function buildLayerNormActivationHoverInfo(node = null, rowItem = null, {
    variant = 'input'
} = {}) {
    const layerNormKind = resolveLayerNormHoverKind(node, rowItem);
    const layerNormLabel = formatLayerNormLabel(layerNormKind);
    const actualStage = resolveLayerNormStageKey(node, rowItem).toLowerCase();
    if (variant === 'output' && normalizePostLayerNormResidualStage(actualStage)) {
        return buildPostLayerNormResidualHoverInfo(node, rowItem);
    }

    let label = `${layerNormLabel} Input Vector`;
    let hoverStage = layerNormKind === 'final' ? 'final_ln.input' : `${layerNormKind || 'ln1'}.input`;
    if (variant === 'normalized') {
        label = `${layerNormLabel} Normalized Vector`;
        hoverStage = actualStage || (layerNormKind === 'final' ? 'final_ln.norm' : `${layerNormKind || 'ln1'}.norm`);
    } else if (variant === 'scaled') {
        label = `${layerNormLabel} Product Vector`;
        hoverStage = layerNormKind === 'final' ? 'final_ln.product' : `${layerNormKind || 'ln1'}.product`;
    } else if (variant === 'output') {
        label = `${layerNormLabel} Output Vector`;
        hoverStage = layerNormKind === 'final' ? 'final_ln.output' : `${layerNormKind || 'ln1'}.output`;
    }
    const sourceStage = variant === 'output'
        ? (normalizeLayerNormOutputStage(hoverStage, { preferLegacy: true }) || actualStage)
        : actualStage;

    return buildLayerNormHoverInfo(node, rowItem, {
        label,
        stage: hoverStage,
        sourceStage
    });
}

export function buildLayerNormParamHoverInfo(node = null, {
    param = 'scale'
} = {}) {
    const layerNormKind = resolveLayerNormHoverKind(node, null);
    const safeParam = String(param || '').trim().toLowerCase() === 'shift' ? 'shift' : 'scale';
    const stage = resolveLayerNormStageKey(node, null).toLowerCase();
    const label = formatLayerNormParamLabel(layerNormKind, safeParam);
    const info = buildProjectionHoverInfo(node, label, {
        stage,
        parameterType: safeParam,
        ...(layerNormKind ? { layerNormKind } : {})
    }) || {};

    return {
        ...info,
        ...(layerNormKind ? { layerNormKind } : {}),
        activationData: {
            ...(info.activationData || {}),
            stage,
            parameterType: safeParam,
            ...(layerNormKind ? { layerNormKind } : {})
        }
    };
}

export function buildProjectionColumnHoverInfo(node = null, columnItem = null, kind = '') {
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

export function buildWeightedSumHoverInfo(node = null, rowItem = null) {
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

export function buildConcatenatedHeadOutputHoverInfo(node = null, rowItem = null) {
    const tokenInfo = createTokenInfo(rowItem) || {};
    const info = buildProjectionHoverInfo(node, 'Concatenated Head Output', {
        stage: 'attention.concatenate',
        vectorCategory: 'concatenated-head-output',
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
            stage: 'attention.concatenate',
            vectorCategory: 'concatenated-head-output',
            ...(Number.isFinite(tokenInfo.tokenIndex) ? { tokenIndex: tokenInfo.tokenIndex } : {}),
            ...(typeof tokenInfo.tokenLabel === 'string' && tokenInfo.tokenLabel.length
                ? { tokenLabel: tokenInfo.tokenLabel }
                : {})
        }
    };
}

export function buildConcatenatedHeadOutputBandHoverInfo(node = null, rowItem = null, headIndex = null) {
    const tokenInfo = createTokenInfo(rowItem) || {};
    const layerIndex = normalizeSceneIndex(node?.semantic?.layerIndex);
    const safeHeadIndex = normalizeSceneIndex(headIndex);
    const baseInfo = buildProjectionHoverInfo(node, 'Attention Weighted Sum', {
        stage: 'attention.weighted_sum',
        isWeightedSum: true,
        ...(Number.isFinite(tokenInfo.tokenIndex) ? { tokenIndex: tokenInfo.tokenIndex } : {}),
        ...(typeof tokenInfo.tokenLabel === 'string' && tokenInfo.tokenLabel.length
            ? { tokenLabel: tokenInfo.tokenLabel }
            : {})
    }) || {};

    return {
        ...tokenInfo,
        ...(Number.isFinite(layerIndex) ? { layerIndex } : {}),
        ...(Number.isFinite(safeHeadIndex) ? { headIndex: safeHeadIndex } : {}),
        ...baseInfo,
        isWeightedSum: true,
        activationData: {
            ...(baseInfo.activationData || {}),
            stage: 'attention.weighted_sum',
            isWeightedSum: true,
            ...(Number.isFinite(layerIndex) ? { layerIndex } : {}),
            ...(Number.isFinite(safeHeadIndex) ? { headIndex: safeHeadIndex } : {}),
            ...(Number.isFinite(tokenInfo.tokenIndex) ? { tokenIndex: tokenInfo.tokenIndex } : {}),
            ...(typeof tokenInfo.tokenLabel === 'string' && tokenInfo.tokenLabel.length
                ? { tokenLabel: tokenInfo.tokenLabel }
                : {})
        }
    };
}

export function buildAttentionCellHoverInfo(node = null, cellItem = null, stageKey = '') {
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

export function resolveAttentionStageKeyForRole(role = '') {
    const safeRole = String(role || '').trim().toLowerCase();
    if (safeRole === 'attention-pre-score') return 'pre';
    if (safeRole === 'attention-masked-input') return 'masked-input';
    if (safeRole === 'attention-mask') return 'mask';
    if (safeRole === 'attention-post' || safeRole === 'attention-post-copy') return 'post';
    return '';
}

export function buildAttentionStageRoleHoverInfo(node = null, stageKey = '') {
    const semantic = node?.semantic && typeof node.semantic === 'object'
        ? node.semantic
        : null;
    return buildAttentionHoverInfo({
        stageKey,
        layerIndex: semantic?.layerIndex,
        headIndex: semantic?.headIndex
    });
}

export function resolveMatrixStageKey(node = null, hit = null) {
    const stage = String(hit?.cellItem?.semantic?.stage || node?.semantic?.stage || '').trim().toLowerCase();
    if (stage.includes('pre-score')) return 'pre-score';
    if (stage.includes('masked-input')) return 'masked-input';
    if (stage.includes('attention-mask')) return 'mask';
    if (stage.includes('attention-post-copy')) return 'post-copy';
    if (stage.includes('attention-post')) return 'post';
    return '';
}
