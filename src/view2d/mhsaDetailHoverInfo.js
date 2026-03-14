import { buildAttentionHoverInfo } from '../ui/attentionHoverInfo.js';
import { resolvePostLayerNormResidualLabel } from '../utils/layerNormLabels.js';
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
        suppressTokenChip: true,
        ...(Number.isFinite(tokenInfo.tokenIndex) ? { tokenIndex: tokenInfo.tokenIndex } : {}),
        ...(typeof tokenInfo.tokenLabel === 'string' && tokenInfo.tokenLabel.length
            ? { tokenLabel: tokenInfo.tokenLabel }
            : {})
    }) || {};

    return {
        ...tokenInfo,
        suppressTokenChip: true,
        ...info,
        activationData: {
            ...(info.activationData || {}),
            stage: 'mlp.down',
            suppressTokenChip: true,
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
        stage: 'mlp.down'
    }) || {};

    return {
        ...info,
        activationData: {
            ...(info.activationData || {}),
            stage: 'mlp.down'
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
    const activationStage = rowStage === 'ln2.shift' ? 'ln2.shift' : 'ln1.shift';
    const label = resolvePostLayerNormResidualLabel({ stage: activationStage });
    const info = buildProjectionHoverInfo(node, label, {
        stage: activationStage,
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
            ...(Number.isFinite(tokenInfo.tokenIndex) ? { tokenIndex: tokenInfo.tokenIndex } : {}),
            ...(typeof tokenInfo.tokenLabel === 'string' && tokenInfo.tokenLabel.length
                ? { tokenLabel: tokenInfo.tokenLabel }
                : {})
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
