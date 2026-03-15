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
    resolveLayerNormKind,
    resolvePostLayerNormResidualLabel
} from '../utils/layerNormLabels.js';
import { resolvePreferredTokenLabel } from '../utils/tokenLabelResolution.js';

export const TRANSFORMER_VIEW2D_OVERVIEW_LABEL = 'GPT-2 (124 M)';

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

function resolveResidualHoverActivationStage(rowSemantic = null) {
    const stage = String(rowSemantic?.stage || '').trim().toLowerCase();
    if (stage === 'incoming') return 'layer.incoming';
    if (stage === 'post-attn-residual') return 'residual.post_attention';
    if (stage === 'post-mlp-residual') return 'residual.post_mlp';
    if (stage === 'outgoing') return 'residual.post_mlp';
    if (stage === 'ln1.shift') return 'ln1.shift';
    if (stage === 'ln2.shift') return 'ln2.shift';
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
    const isPostLayerNormResidual = activationStage === 'ln1.shift' || activationStage === 'ln2.shift';
    const headIndex = normalizeOptionalIndex(semantic.headIndex);
    const label = isPostLayerNormResidual
        ? resolvePostLayerNormResidualLabel({ stage: activationStage })
        : 'Residual Stream Vector';
    const info = {
        ...(Number.isFinite(layerIndex) ? { layerIndex } : {}),
        ...(Number.isFinite(headIndex) ? { headIndex } : {}),
        ...(Number.isFinite(tokenIndex) ? { tokenIndex } : {}),
        ...(Number.isFinite(tokenId) ? { tokenId } : {}),
        ...(tokenLabel.length ? { tokenLabel } : {}),
        activationData: {
            label,
            stage: activationStage,
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
    activationStage = '',
    layerNormKind = null,
    suppressTokenChip = false
} = {}) {
    const info = {};
    if (Number.isFinite(layerIndex)) {
        info.layerIndex = Math.max(0, Math.floor(layerIndex));
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

export function buildSemanticNodeHoverPayload(hit = null) {
    const entry = hit?.entry || null;
    const semantic = entry?.semantic || hit?.node?.semantic || null;
    const role = String(entry?.role || hit?.node?.role || semantic?.role || '').trim().toLowerCase();
    if (!semantic || typeof semantic !== 'object') return null;

    if (
        semantic.componentKind === 'layer-norm'
        && (role === 'module-card' || role === 'module-title' || role === 'module')
    ) {
        const layerNormKind = resolveLayerNormKindFromSemanticStage(semantic.stage);
        const label = layerNormKind === 'final' ? 'LayerNorm (Top)' : 'LayerNorm';
        return {
            label,
            info: buildSemanticHoverInfo({
                label,
                layerIndex: semantic.layerIndex,
                activationStage: layerNormKind === 'final' ? 'final_ln.norm' : `${layerNormKind || 'layernorm'}.norm`,
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
        return {
            label: 'Input Token',
            info: {
                ...(Number.isFinite(tokenIndex) ? { tokenIndex } : {}),
                ...(tokenLabel.length ? { tokenLabel } : {}),
                ...(Number.isFinite(positionIndex) ? { positionIndex } : {}),
                activationData: {
                    label: 'Input Token',
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
        return {
            label: 'Input Position',
            info: {
                ...(Number.isFinite(tokenIndex) ? { tokenIndex } : {}),
                ...(tokenLabel.length ? { tokenLabel } : {}),
                ...(Number.isFinite(positionIndex) ? { positionIndex } : {}),
                activationData: {
                    label: 'Input Position',
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
        return {
            label: 'Chosen Token',
            info: {
                ...(Number.isFinite(tokenIndex) ? { tokenIndex } : {}),
                ...(tokenLabel.length ? { tokenLabel } : {}),
                ...(Number.isFinite(positionIndex) ? { positionIndex } : {}),
                activationData: {
                    label: 'Chosen Token',
                    stage: 'generation.chosen',
                    ...(Number.isFinite(tokenIndex) ? { tokenIndex } : {}),
                    ...(tokenLabel.length ? { tokenLabel } : {}),
                    ...(Number.isFinite(positionIndex) ? { positionIndex } : {})
                }
            }
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
        if (target.stage === 'token') return 'Token Embeddings';
        if (target.stage === 'position') return 'Position Embeddings';
        return 'Embeddings';
    }

    if (target.componentKind === 'residual') {
        if (target.stage === 'incoming') return 'Incoming Residual';
        if (target.stage === 'post-attn-add') return 'Post-Attention Residual';
        if (target.stage === 'post-mlp-add') return 'Post-MLP Residual';
        if (target.stage === 'outgoing') return 'Outgoing Residual';
        return 'Residual Stream';
    }

    if (target.componentKind === 'logits') {
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
    if (target.componentKind === 'embedding') {
        if (target.stage === 'token') return 'Token embeddings';
        if (target.stage === 'position') return 'Position embeddings';
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
        if (target.stage === 'incoming') return layerLabel ? `${layerLabel} incoming residual` : 'Incoming residual';
        if (target.stage === 'post-attn-add') return layerLabel ? `${layerLabel} post-attention residual` : 'Post-attention residual';
        if (target.stage === 'post-mlp-add') return layerLabel ? `${layerLabel} post-MLP residual` : 'Post-MLP residual';
        if (target.stage === 'outgoing') return layerLabel ? `${layerLabel} outgoing residual` : 'Outgoing residual';
        return layerLabel ? `${layerLabel} residual stream` : 'Residual stream';
    }
    if (target.componentKind === 'logits') {
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
    const explicitLayerNormKind = findUserDataString(selectionInfo, 'layerNormKind');
    const layerNormKind = resolveLayerNormKind({
        label,
        stage: stageLower,
        explicitKind: explicitLayerNormKind
    });
    const topUnembeddingLabel = isTopUnembeddingLabel(lower);

    let semanticTarget = null;

    if (layerNormKind) {
        const stage = resolveLayerNormStage(layerNormKind);
        if (stage && (layerNormKind === 'final' || Number.isFinite(layerIndex))) {
            semanticTarget = buildSemanticTarget({
                componentKind: 'layer-norm',
                layerIndex: layerNormKind === 'final' ? null : layerIndex,
                stage,
                role: 'module'
            });
        }
    } else if (
        isLogitBarSelection(label, selectionInfo)
        || lower.includes('top logit bars')
        || topUnembeddingLabel
        || lower === 'logit'
    ) {
        semanticTarget = buildSemanticTarget({
            componentKind: 'logits',
            stage: 'output',
            role: topUnembeddingLabel ? 'unembedding' : 'logits-topk'
        });
    } else if (
        stageLower.startsWith('embedding.token')
        || lower.includes('token embedding')
    ) {
        semanticTarget = buildSemanticTarget({
            componentKind: 'embedding',
            stage: 'token',
            role: 'token-embedding'
        });
    } else if (
        stageLower.startsWith('embedding.position')
        || lower.includes('position embedding')
        || lower.includes('positional embedding')
    ) {
        semanticTarget = buildSemanticTarget({
            componentKind: 'embedding',
            stage: 'position',
            role: 'position-embedding'
        });
    } else if (
        stageLower.startsWith('embedding.sum')
        || lower.includes('embedding sum')
        || (lower.includes('vocabulary embedding') && !topUnembeddingLabel)
        || (lower.includes('vocab embedding') && !topUnembeddingLabel)
    ) {
        semanticTarget = buildSemanticTarget({
            componentKind: 'embedding',
            stage: stageLower.startsWith('embedding.sum') || lower.includes('embedding sum') ? 'sum' : 'input',
            role: stageLower.startsWith('embedding.sum') || lower.includes('embedding sum') ? 'sum-output' : 'module'
        });
    } else if (
        stageLower === 'attention.concatenate'
        || lower === 'concatenate'
        || lower === 'concatenation'
        || lower.includes('concatenate heads')
        || lower.includes('head concatenation')
    ) {
        if (Number.isFinite(layerIndex)) {
            semanticTarget = buildSemanticTarget({
                componentKind: 'output-projection',
                layerIndex,
                stage: 'attn-out',
                role: 'projection-weight'
            });
        }
    } else if (
        lower.includes('output projection matrix')
        || lower.includes('alpha projection matrix')
        || lower.includes('alpha projection')
        || stageLower === 'attention.output_projection'
        || lower.includes('attention output projection')
    ) {
        if (Number.isFinite(layerIndex)) {
            semanticTarget = buildSemanticTarget({
                componentKind: 'output-projection',
                layerIndex,
                stage: 'attn-out',
                role: (
                    lower.includes('output projection matrix')
                    || lower.includes('alpha projection matrix')
                    || lower.includes('alpha projection')
                )
                    ? 'projection-weight'
                    : 'projection-output'
            });
        }
    } else if (
        lower.includes('mlp up weight matrix')
        || lower.includes('mlp up projection')
    ) {
        if (Number.isFinite(layerIndex)) {
            semanticTarget = buildSemanticTarget({
                componentKind: 'mlp',
                layerIndex,
                stage: 'mlp-up',
                role: 'mlp-up'
            });
        }
    } else if (
        lower.includes('mlp down weight matrix')
        || lower.includes('mlp down projection')
    ) {
        if (Number.isFinite(layerIndex)) {
            semanticTarget = buildSemanticTarget({
                componentKind: 'mlp',
                layerIndex,
                stage: 'mlp-down',
                role: 'mlp-down'
            });
        }
    } else if (
        lower.includes('gelu')
        || lower.includes('mlp activation')
    ) {
        if (Number.isFinite(layerIndex)) {
            semanticTarget = buildSemanticTarget({
                componentKind: 'mlp',
                layerIndex,
                stage: 'mlp-activation',
                role: 'mlp-activation'
            });
        }
    } else if (
        lower.includes('mlp')
        && Number.isFinite(layerIndex)
    ) {
        semanticTarget = buildSemanticTarget({
            componentKind: 'mlp',
            layerIndex,
            stage: 'mlp',
            role: 'module'
        });
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
                stage: 'post-attn-add',
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
                stage: 'post-mlp-add',
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
    const detailSemanticTargets = (
        semanticTarget?.componentKind === 'mhsa'
        && Number.isFinite(semanticTarget?.headIndex)
    )
        ? resolveMhsaDetailSemanticTargets(selectionInfo, label)
        : [];
    const detailFocusLabel = detailSemanticTargets.length
        ? resolveMhsaDetailFocusLabel(label, stageLower)
        : '';
    const transitionMode = resolveTransformerView2dOpenTransitionMode({
        semanticTarget
    });
    return {
        semanticTarget,
        focusLabel: describeTransformerView2dTarget(semanticTarget),
        ...(detailSemanticTargets.length ? { detailSemanticTargets } : {}),
        ...(detailFocusLabel.length ? { detailFocusLabel } : {}),
        ...(transitionMode ? { transitionMode } : {}),
        actionLabel: 'View in 2D / matrix form'
    };
}
