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
import { resolveLayerNormKind } from '../utils/layerNormLabels.js';
import { resolvePreferredTokenLabel } from '../utils/tokenLabelResolution.js';

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

function resolveResidualHoverActivationStage(rowSemantic = null) {
    const stage = String(rowSemantic?.stage || '').trim().toLowerCase();
    if (stage === 'incoming') return 'layer.incoming';
    if (stage === 'post-attn-residual') return 'residual.post_attention';
    if (stage === 'post-mlp-residual') return 'residual.post_mlp';
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
    const label = isPostLayerNormResidual ? 'Post LayerNorm Residual Vector' : 'Residual Stream Vector';
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
        semantic.componentKind === 'output-projection'
        && (
            role === 'projection-weight'
            || role === 'module-card'
            || role === 'module-title'
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
    const concatDetailTarget = (
        !headDetailTarget
        && safeTarget?.componentKind === 'mhsa'
        && safeTarget?.stage === 'concatenate'
    )
        ? resolveConcatDetailTarget(safeTarget)
        : null;
    const outputProjectionDetailTarget = (
        !headDetailTarget
        && !concatDetailTarget
        && safeTarget?.componentKind === 'output-projection'
    )
        ? resolveOutputProjectionDetailTarget(safeTarget)
        : null;

    return {
        headDetailTarget,
        concatDetailTarget,
        outputProjectionDetailTarget
    };
}

export function hasActiveDetailTarget({
    headDetailTarget = null,
    concatDetailTarget = null,
    outputProjectionDetailTarget = null
} = {}) {
    return !!(headDetailTarget || concatDetailTarget || outputProjectionDetailTarget);
}

export function resolveActiveSemanticTarget({
    baseSemanticTarget = null,
    headDetailTarget = null,
    concatDetailTarget = null,
    outputProjectionDetailTarget = null
} = {}) {
    if (headDetailTarget) return buildHeadDetailSemanticTarget(headDetailTarget);
    if (concatDetailTarget) return buildConcatDetailSemanticTarget(concatDetailTarget);
    if (outputProjectionDetailTarget) return buildOutputProjectionDetailSemanticTarget(outputProjectionDetailTarget);
    return buildSemanticTarget(baseSemanticTarget);
}

export function resolveActiveFocusLabel({
    baseSemanticTarget = null,
    baseFocusLabel = '',
    headDetailTarget = null,
    concatDetailTarget = null,
    outputProjectionDetailTarget = null
} = {}) {
    const semanticTarget = resolveActiveSemanticTarget({
        baseSemanticTarget,
        headDetailTarget,
        concatDetailTarget,
        outputProjectionDetailTarget
    });
    if (headDetailTarget || concatDetailTarget || outputProjectionDetailTarget) {
        return describeTransformerView2dTarget(semanticTarget);
    }
    return String(baseFocusLabel || '').trim() || describeTransformerView2dTarget(semanticTarget);
}

export function resolveFocusSemanticTargets({
    semanticTarget = null,
    baseSemanticTarget = null,
    headDetailTarget = null,
    concatDetailTarget = null,
    outputProjectionDetailTarget = null
} = {}) {
    const candidates = [];
    if (headDetailTarget) {
        candidates.push(
            buildHeadDetailSemanticTarget(headDetailTarget, 'head-card'),
            buildHeadDetailSemanticTarget(headDetailTarget)
        );
    } else if (concatDetailTarget) {
        candidates.push(
            buildConcatDetailSemanticTarget(concatDetailTarget, 'concat-card'),
            buildConcatDetailSemanticTarget(concatDetailTarget, 'concat')
        );
    } else if (outputProjectionDetailTarget) {
        candidates.push(
            buildOutputProjectionDetailSemanticTarget(outputProjectionDetailTarget, 'projection-weight'),
            buildOutputProjectionDetailSemanticTarget(outputProjectionDetailTarget, 'module')
        );
    } else {
        candidates.push(
            buildSemanticTarget(semanticTarget)
            || resolveActiveSemanticTarget({
                baseSemanticTarget,
                headDetailTarget,
                concatDetailTarget,
                outputProjectionDetailTarget
            })
        );
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
        || entry.role === 'module-title';
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

export function describeTransformerView2dTarget(target = null) {
    if (!target) return 'Transformer overview';
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
            return layerLabel ? `${layerLabel} Concatenate Heads` : 'Concatenate Heads';
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
    return 'Transformer overview';
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
                componentKind: 'mhsa',
                layerIndex,
                stage: 'concatenate',
                role: 'concat'
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
    return {
        semanticTarget,
        focusLabel: describeTransformerView2dTarget(semanticTarget),
        actionLabel: 'Open 2D canvas'
    };
}
