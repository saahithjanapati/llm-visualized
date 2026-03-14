import { resolveTokenChipLabel } from '../utils/tokenChipStyleUtils.js';
import {
    isPostLayerNormResidualSelection,
    resolveLayerNormKind,
    resolveLayerNormParamSpec
} from '../utils/layerNormLabels.js';
import { resolvePreferredTokenLabel } from '../utils/tokenLabelResolution.js';

function toFiniteTokenNumber(value) {
    const parsed = Number(value);
    if (!Number.isFinite(parsed)) return null;
    return Math.floor(parsed);
}

function toHoverNumber(value, { allowInfinity = false } = {}) {
    const parsed = Number(value);
    if (Number.isFinite(parsed)) return parsed;
    if (allowInfinity && (parsed === Number.POSITIVE_INFINITY || parsed === Number.NEGATIVE_INFINITY)) {
        return parsed;
    }
    return null;
}

function pushCandidate(candidates, seen, candidate) {
    if (!candidate || typeof candidate !== 'object') return;
    if (seen.has(candidate)) return;
    seen.add(candidate);
    candidates.push(candidate);
}

function collectHoverDataCandidates(info = null, object = null) {
    const candidates = [];
    const seen = new Set();

    pushCandidate(candidates, seen, info);
    pushCandidate(candidates, seen, info?.activationData);

    const vectorRef = info?.vectorRef || null;
    const vectorRefCandidates = [
        vectorRef?.userData,
        vectorRef?.userData?.parentLane,
        vectorRef?.userData?.activationData,
        vectorRef?.group?.userData,
        vectorRef?.group?.userData?.parentLane,
        vectorRef?.group?.userData?.activationData,
        vectorRef?.mesh?.userData,
        vectorRef?.mesh?.userData?.parentLane,
        vectorRef?.mesh?.userData?.activationData
    ];
    vectorRefCandidates.forEach((candidate) => pushCandidate(candidates, seen, candidate));

    let current = object;
    while (current && !current.isScene) {
        const userData = current.userData || null;
        pushCandidate(candidates, seen, userData);
        pushCandidate(candidates, seen, userData?.activationData);
        pushCandidate(candidates, seen, userData?.parentLane);
        current = current.parent || null;
    }

    return candidates;
}

export function findHoverTokenString(info = null, object = null, key = '') {
    const candidates = collectHoverDataCandidates(info, object);
    for (const candidate of candidates) {
        if (typeof candidate?.[key] === 'string' && candidate[key].length) {
            return candidate[key];
        }
    }
    return '';
}

export function findHoverTokenNumber(info = null, object = null, key = '') {
    const candidates = collectHoverDataCandidates(info, object);
    for (const candidate of candidates) {
        const value = toFiniteTokenNumber(candidate?.[key]);
        if (Number.isFinite(value)) return value;
    }
    return null;
}

function findHoverNumericValue(info = null, object = null, key = '', { allowInfinity = false } = {}) {
    const candidates = collectHoverDataCandidates(info, object);
    for (const candidate of candidates) {
        const value = toHoverNumber(candidate?.[key], { allowInfinity });
        if (value !== null) return value;
    }
    return null;
}

function findHoverBoolean(info = null, object = null, key = '') {
    const candidates = collectHoverDataCandidates(info, object);
    for (const candidate of candidates) {
        if (typeof candidate?.[key] === 'boolean') {
            return candidate[key];
        }
    }
    return null;
}

function isWeightMatrixHoverSelection(label = '') {
    return String(label || '').toLowerCase().includes('weight matrix');
}

function isBiasVectorHoverSelection(label = '') {
    return String(label || '').toLowerCase().includes('bias vector');
}

function isEmbeddingMatrixHoverSelection(label = '') {
    const lower = String(label || '').toLowerCase().trim();
    return lower === 'vocabulary embedding'
        || lower === 'positional embedding'
        || lower === 'vocab embedding (top)'
        || lower === 'vocabulary embedding (top)';
}

function isBottomTokenChipHoverSelection(label = '') {
    return String(label || '').toLowerCase().startsWith('token:');
}

function isBottomPositionChipHoverSelection(label = '') {
    return String(label || '').toLowerCase().startsWith('position:');
}

function isWeightedSumHoverSelection(label = '', info = null, object = null) {
    const lower = String(label || '').toLowerCase();
    if (lower.includes('weighted sum')) return true;
    if (info?.isWeightedSum === true) return true;
    const candidates = collectHoverDataCandidates(info, object);
    return candidates.some((candidate) => candidate?.isWeightedSum === true);
}

function isAttentionScoreHoverSelection(label = '', info = null, object = null) {
    const lower = String(label || '').toLowerCase();
    const stageLower = findHoverTokenString(info, object, 'stage')
        .toLowerCase()
        .replace(/-/g, '_');
    if (
        stageLower === 'attention.pre'
        || stageLower === 'attention.post'
        || stageLower === 'attention.masked_input'
        || stageLower === 'attention.mask'
        || stageLower === 'attention.post_copy'
    ) {
        return true;
    }

    const kindLower = String(info?.kind || object?.userData?.kind || '').toLowerCase();
    if (kindLower === 'attentionsphere') return true;

    if (!lower.includes('attention score')) return false;

    const keyTokenLabel = findHoverTokenString(info, object, 'keyTokenLabel');
    return Number.isFinite(findHoverTokenNumber(info, object, 'keyTokenIndex'))
        || (typeof keyTokenLabel === 'string' && keyTokenLabel.length > 0)
        || Number.isFinite(findHoverTokenNumber(info, object, 'preScore'))
        || Number.isFinite(findHoverTokenNumber(info, object, 'postScore'))
        || findHoverNumericValue(info, object, 'maskValue', { allowInfinity: true }) !== null;
}

function formatHeadLayerSubtitle(headIndex = null, layerIndex = null) {
    const parts = [];
    if (Number.isFinite(headIndex)) {
        parts.push(`Head ${Math.floor(headIndex) + 1}`);
    }
    if (Number.isFinite(layerIndex)) {
        parts.push(`Layer ${Math.floor(layerIndex) + 1}`);
    }
    return parts.join(' • ');
}

function resolveHoverPositionIndex(info = null, object = null) {
    const explicitPositionIndex = findHoverTokenNumber(info, object, 'positionIndex');
    if (Number.isFinite(explicitPositionIndex)) {
        return Math.max(1, Math.floor(explicitPositionIndex));
    }
    const tokenIndex = findHoverTokenNumber(info, object, 'tokenIndex');
    if (Number.isFinite(tokenIndex)) {
        return Math.max(1, Math.floor(tokenIndex) + 1);
    }
    return null;
}

function isQkvStage(stage = '') {
    return String(stage || '').toLowerCase().startsWith('qkv.');
}

function isResidualVectorStage(stage = '') {
    const lower = String(stage || '').toLowerCase();
    return lower.startsWith('layer.incoming') || lower.startsWith('residual.');
}

function isResidualStreamHoverSelection(label = '', stage = '') {
    const lower = String(label || '').toLowerCase();
    const stageLower = String(stage || '').toLowerCase();
    return lower.includes('residual stream vector')
        || stageLower === 'embedding.sum'
        || isResidualVectorStage(stageLower);
}

function isPostLayerNormResidualHoverSelection(label = '', stage = '') {
    return isPostLayerNormResidualSelection({ label, stage });
}

function isFinalLayerNormHoverSelection(label = '', info = null, object = null) {
    const stage = findHoverTokenString(info, object, 'stage');
    const explicitKind = findHoverTokenString(info, object, 'layerNormKind');
    return resolveLayerNormKind({
        label,
        stage,
        explicitKind
    }) === 'final';
}

function isLayerNormParamHoverSelection(label = '', info = null, object = null) {
    const stage = findHoverTokenString(info, object, 'stage');
    const explicitKind = findHoverTokenString(info, object, 'layerNormKind');
    return !!resolveLayerNormParamSpec({
        label,
        stage,
        explicitKind
    });
}

function isTokenVectorStage(stage = '') {
    const lower = String(stage || '').toLowerCase();
    if (!lower) return false;
    return lower === 'embedding.token'
        || lower === 'embedding.position'
        || lower === 'embedding.sum'
        || lower.startsWith('layer.incoming')
        || lower.startsWith('ln1.')
        || lower.startsWith('ln2.')
        || lower.startsWith('qkv.')
        || lower === 'attention.output_projection'
        || lower.startsWith('residual.')
        || lower.startsWith('mlp.up')
        || lower.startsWith('mlp.activation')
        || lower.startsWith('mlp.down');
}

function isVectorLikeCandidate(candidate = null) {
    if (!candidate || typeof candidate !== 'object') return false;
    if (candidate.isWeightedSum === true) return true;
    if (candidate.qkvProcessed === true) return true;
    if (candidate.cachedKv === true || candidate.kvCachePersistent === true) return true;
    if (typeof candidate.vectorCategory === 'string' && candidate.vectorCategory.length) return true;
    if (isTokenVectorStage(candidate.stage)) return true;
    const lane = candidate.parentLane;
    if (lane && typeof lane === 'object') {
        if (Number.isFinite(toFiniteTokenNumber(lane.tokenIndex))) return true;
        if (typeof lane.tokenLabel === 'string' && lane.tokenLabel.length) return true;
    }
    return false;
}

export function isVectorLikeHoverSelection(label = '', info = null, object = null) {
    if (isWeightMatrixHoverSelection(label)) return false;
    if (isBiasVectorHoverSelection(label)) return false;

    const lower = String(label || '').toLowerCase();
    if (lower.includes('vector') || lower.includes('weighted sum')) return true;
    if (info?.vectorRef) return true;

    const kind = String(info?.kind || '').toLowerCase();
    if (kind === 'mergedkv') return true;

    const candidates = collectHoverDataCandidates(info, object);
    return candidates.some((candidate) => isVectorLikeCandidate(candidate));
}

function shouldSuppressTokenChipHover(info = null, object = null) {
    const candidates = collectHoverDataCandidates(info, object);
    return candidates.some((candidate) => candidate?.suppressTokenChip === true);
}

function normalizeHoverTokenChipSyncEntry(entry = null) {
    if (!entry || typeof entry !== 'object') return null;
    const tokenIndex = toFiniteTokenNumber(entry.tokenIndex);
    const tokenId = toFiniteTokenNumber(entry.tokenId ?? entry.token_id);
    const tokenLabel = String(entry.tokenLabel || '').trim();
    if (!Number.isFinite(tokenIndex) && !Number.isFinite(tokenId) && !tokenLabel.length) {
        return null;
    }
    return {
        tokenIndex: Number.isFinite(tokenIndex) ? Math.floor(tokenIndex) : null,
        tokenId: Number.isFinite(tokenId) ? Math.floor(tokenId) : null,
        tokenLabel
    };
}

function hoverTokenChipSyncEntriesMatch(a = null, b = null) {
    const left = normalizeHoverTokenChipSyncEntry(a);
    const right = normalizeHoverTokenChipSyncEntry(b);
    if (!left || !right) return false;
    if (Number.isFinite(left.tokenIndex) && Number.isFinite(right.tokenIndex)) {
        return left.tokenIndex === right.tokenIndex;
    }
    if (Number.isFinite(left.tokenId) && Number.isFinite(right.tokenId)) {
        if (left.tokenId !== right.tokenId) return false;
        if (left.tokenLabel.length && right.tokenLabel.length) {
            return left.tokenLabel === right.tokenLabel;
        }
        return true;
    }
    return left.tokenLabel.length > 0
        && right.tokenLabel.length > 0
        && left.tokenLabel === right.tokenLabel;
}

function pushUniqueHoverTokenChipSyncEntry(entries, entry = null) {
    const normalizedEntry = normalizeHoverTokenChipSyncEntry(entry);
    if (!normalizedEntry) return;
    if (entries.some((candidate) => hoverTokenChipSyncEntriesMatch(candidate, normalizedEntry))) {
        return;
    }
    entries.push(normalizedEntry);
}

function resolveAttentionRowContext({
    roleLabel = 'Token',
    tokenIndex = null,
    tokenId = null,
    tokenLabel = '',
    activationSource = null
} = {}) {
    const safeTokenIndex = Number.isFinite(tokenIndex) ? Math.floor(tokenIndex) : null;
    let safeTokenId = Number.isFinite(tokenId) ? Math.floor(tokenId) : null;
    if (!Number.isFinite(safeTokenId)
        && Number.isFinite(safeTokenIndex)
        && typeof activationSource?.getTokenId === 'function') {
        const resolvedTokenId = activationSource.getTokenId(safeTokenIndex);
        if (Number.isFinite(resolvedTokenId)) safeTokenId = Math.floor(resolvedTokenId);
    }

    const preferredLabel = resolvePreferredTokenLabel({
        tokenLabel,
        tokenIndex: safeTokenIndex,
        activationSource
    });
    if (!preferredLabel && !Number.isFinite(safeTokenIndex)) return null;

    return {
        roleLabel,
        tokenLabel: preferredLabel,
        tokenIndex: safeTokenIndex,
        tokenId: safeTokenId,
        positionText: Number.isFinite(safeTokenIndex)
            ? `Position ${safeTokenIndex + 1}`
            : 'Position n/a'
    };
}

function resolveAttentionScoreHoverTokenEntries({
    label = '',
    info = null,
    object = null,
    activationSource = null
} = {}) {
    if (!isAttentionScoreHoverSelection(label, info, object)) return [];

    const sourceTokenIndex = (() => {
        const queryTokenIndex = findHoverTokenNumber(info, object, 'queryTokenIndex');
        if (Number.isFinite(queryTokenIndex)) return Math.floor(queryTokenIndex);
        const tokenIndex = findHoverTokenNumber(info, object, 'tokenIndex');
        return Number.isFinite(tokenIndex) ? Math.floor(tokenIndex) : null;
    })();
    const sourceTokenLabel = findHoverTokenString(info, object, 'queryTokenLabel')
        || findHoverTokenString(info, object, 'tokenLabel');
    const sourceTokenId = (() => {
        const queryTokenId = findHoverTokenNumber(info, object, 'queryTokenId');
        if (Number.isFinite(queryTokenId)) return Math.floor(queryTokenId);
        const tokenId = findHoverTokenNumber(info, object, 'tokenId');
        return Number.isFinite(tokenId) ? Math.floor(tokenId) : null;
    })();

    const targetTokenIndex = findHoverTokenNumber(info, object, 'keyTokenIndex');
    const targetTokenLabel = findHoverTokenString(info, object, 'keyTokenLabel');
    const targetTokenId = findHoverTokenNumber(info, object, 'keyTokenId');

    const entries = [];
    pushUniqueHoverTokenChipSyncEntry(entries, resolveAttentionRowContext({
        roleLabel: 'Source',
        tokenIndex: sourceTokenIndex,
        tokenId: sourceTokenId,
        tokenLabel: sourceTokenLabel,
        activationSource
    }));
    pushUniqueHoverTokenChipSyncEntry(entries, resolveAttentionRowContext({
        roleLabel: 'Target',
        tokenIndex: targetTokenIndex,
        tokenId: targetTokenId,
        tokenLabel: targetTokenLabel,
        activationSource
    }));
    return entries;
}

function resolveAttentionScoreHoverContext({
    label = '',
    info = null,
    object = null,
    activationSource = null
} = {}) {
    if (!isAttentionScoreHoverSelection(label, info, object)) return null;

    const sourceTokenIndex = (() => {
        const queryTokenIndex = findHoverTokenNumber(info, object, 'queryTokenIndex');
        if (Number.isFinite(queryTokenIndex)) return Math.floor(queryTokenIndex);
        const tokenIndex = findHoverTokenNumber(info, object, 'tokenIndex');
        return Number.isFinite(tokenIndex) ? Math.floor(tokenIndex) : null;
    })();
    const sourceTokenLabel = findHoverTokenString(info, object, 'queryTokenLabel')
        || findHoverTokenString(info, object, 'tokenLabel');
    const sourceTokenId = (() => {
        const queryTokenId = findHoverTokenNumber(info, object, 'queryTokenId');
        if (Number.isFinite(queryTokenId)) return Math.floor(queryTokenId);
        const tokenId = findHoverTokenNumber(info, object, 'tokenId');
        return Number.isFinite(tokenId) ? Math.floor(tokenId) : null;
    })();

    const targetTokenIndex = findHoverTokenNumber(info, object, 'keyTokenIndex');
    const targetTokenLabel = findHoverTokenString(info, object, 'keyTokenLabel');
    const targetTokenId = findHoverTokenNumber(info, object, 'keyTokenId');

    const attentionRows = [
        resolveAttentionRowContext({
            roleLabel: 'Source',
            tokenIndex: sourceTokenIndex,
            tokenId: sourceTokenId,
            tokenLabel: sourceTokenLabel,
            activationSource
        }),
        resolveAttentionRowContext({
            roleLabel: 'Target',
            tokenIndex: targetTokenIndex,
            tokenId: targetTokenId,
            tokenLabel: targetTokenLabel,
            activationSource
        })
    ];

    if (!attentionRows.some(Boolean)) return null;

    const preScore = findHoverNumericValue(info, object, 'preScore');
    const postScore = findHoverNumericValue(info, object, 'postScore');
    const maskValue = findHoverNumericValue(info, object, 'maskValue', { allowInfinity: true });
    const showMaskValue = findHoverBoolean(info, object, 'showMaskValue') === true;
    const attentionStage = findHoverTokenString(info, object, 'stage')
        .toLowerCase()
        .replace(/-/g, '_');
    const attentionMetrics = [];
    const formatMetricValue = (value) => {
        if (value === Number.POSITIVE_INFINITY) return '∞';
        if (value === Number.NEGATIVE_INFINITY) return '-∞';
        return Number.isFinite(value) ? value.toFixed(4) : 'n/a';
    };

    const preferredMetricKey = attentionStage === 'attention.post' || attentionStage === 'attention.post_copy'
        ? 'post'
        : (attentionStage === 'attention.mask' || attentionStage === 'attention.masked_input'
            ? 'mask'
            : 'pre');

    const pushPreMetric = () => {
        if (!Number.isFinite(preScore)) return false;
        attentionMetrics.push({
            roleLabel: 'Score:',
            valueText: formatMetricValue(preScore)
        });
        return true;
    };
    const pushPostMetric = () => {
        if (!Number.isFinite(postScore)) return false;
        attentionMetrics.push({
            roleLabel: 'Score:',
            valueText: formatMetricValue(postScore)
        });
        return true;
    };
    const pushMaskMetric = () => {
        if (!showMaskValue || maskValue === null) return false;
        attentionMetrics.push({
            roleLabel: 'Causal mask',
            valueText: formatMetricValue(maskValue)
        });
        return true;
    };

    if (preferredMetricKey === 'post') {
        pushPostMetric() || pushPreMetric() || pushMaskMetric();
    } else if (preferredMetricKey === 'mask') {
        pushMaskMetric() || pushPostMetric() || pushPreMetric();
    } else {
        pushPreMetric() || pushMaskMetric() || pushPostMetric();
    }

    return {
        suppressHoverLabel: false,
        showPrimaryLabel: true,
        detailKind: 'attention-token-pair',
        attentionRows,
        ...(attentionMetrics.length ? { attentionMetrics } : {})
    };
}

function resolveAssociatedHoverTokenIndex(info = null, object = null) {
    const tokenIndex = findHoverTokenNumber(info, object, 'tokenIndex');
    if (Number.isFinite(tokenIndex)) return Math.floor(tokenIndex);

    const queryTokenIndex = findHoverTokenNumber(info, object, 'queryTokenIndex');
    if (Number.isFinite(queryTokenIndex)) return Math.floor(queryTokenIndex);

    const keyTokenIndex = findHoverTokenNumber(info, object, 'keyTokenIndex');
    if (Number.isFinite(keyTokenIndex)) return Math.floor(keyTokenIndex);

    return null;
}

function resolveAssociatedHoverTokenId(info = null, object = null, tokenIndex = null, activationSource = null) {
    const tokenId = findHoverTokenNumber(info, object, 'tokenId');
    if (Number.isFinite(tokenId)) return Math.floor(tokenId);

    const queryTokenId = findHoverTokenNumber(info, object, 'queryTokenId');
    if (Number.isFinite(queryTokenId)) return Math.floor(queryTokenId);

    const keyTokenId = findHoverTokenNumber(info, object, 'keyTokenId');
    if (Number.isFinite(keyTokenId)) return Math.floor(keyTokenId);

    if (Number.isFinite(tokenIndex) && typeof activationSource?.getTokenId === 'function') {
        const resolvedTokenId = activationSource.getTokenId(tokenIndex);
        if (Number.isFinite(resolvedTokenId)) return Math.floor(resolvedTokenId);
    }

    return null;
}

function resolveAssociatedHoverTokenLabel(info = null, object = null, tokenIndex = null, activationSource = null) {
    return resolvePreferredTokenLabel({
        tokenLabel: findHoverTokenString(info, object, 'tokenLabel')
            || findHoverTokenString(info, object, 'queryTokenLabel')
            || findHoverTokenString(info, object, 'keyTokenLabel'),
        tokenIndex,
        activationSource
    });
}

function resolveAssociatedHoverTokenEntry({
    label = '',
    info = null,
    object = null,
    activationSource = null
} = {}) {
    if (isEmbeddingMatrixHoverSelection(label) || isWeightMatrixHoverSelection(label)) {
        return null;
    }
    if (isBiasVectorHoverSelection(label)) {
        return null;
    }
    if (isLayerNormParamHoverSelection(label, info, object)) {
        return null;
    }
    if (isBottomPositionChipHoverSelection(label)) return null;

    const isBottomTokenChip = isBottomTokenChipHoverSelection(label);
    if (!isBottomTokenChip && shouldSuppressTokenChipHover(info, object)) return null;
    if (!isBottomTokenChip && !isVectorLikeHoverSelection(label, info, object)) return null;

    const tokenIndex = resolveAssociatedHoverTokenIndex(info, object);
    const tokenId = resolveAssociatedHoverTokenId(info, object, tokenIndex, activationSource);
    const tokenLabel = resolveAssociatedHoverTokenLabel(info, object, tokenIndex, activationSource);

    if (!tokenLabel && !Number.isFinite(tokenIndex) && !Number.isFinite(tokenId)) {
        return null;
    }

    return {
        isBottomTokenChip,
        tokenIndex: Number.isFinite(tokenIndex) ? Math.floor(tokenIndex) : null,
        tokenId: Number.isFinite(tokenId) ? Math.floor(tokenId) : null,
        tokenLabel
    };
}

export function resolveHoverTokenChipSyncEntries({
    label = '',
    info = null,
    object = null,
    activationSource = null
} = {}) {
    const attentionEntries = resolveAttentionScoreHoverTokenEntries({
        label,
        info,
        object,
        activationSource
    });
    if (attentionEntries.length) return attentionEntries;

    const tokenEntry = resolveAssociatedHoverTokenEntry({
        label,
        info,
        object,
        activationSource
    });
    if (!tokenEntry) return [];

    return [{
        tokenIndex: tokenEntry.tokenIndex,
        tokenId: tokenEntry.tokenId,
        tokenLabel: tokenEntry.tokenLabel
    }];
}

export function resolveHoverTokenChipSyncEntry({
    label = '',
    info = null,
    object = null,
    activationSource = null
} = {}) {
    const tokenEntry = resolveAssociatedHoverTokenEntry({
        label,
        info,
        object,
        activationSource
    });
    if (!tokenEntry) return null;

    return {
        tokenIndex: tokenEntry.tokenIndex,
        tokenId: tokenEntry.tokenId,
        tokenLabel: tokenEntry.tokenLabel
    };
}

export function resolveHoverTokenContext({
    label = '',
    info = null,
    object = null,
    activationSource = null
} = {}) {
    const attentionScoreContext = resolveAttentionScoreHoverContext({
        label,
        info,
        object,
        activationSource
    });
    if (attentionScoreContext) return attentionScoreContext;

    const tokenEntry = resolveAssociatedHoverTokenEntry({
        label,
        info,
        object,
        activationSource
    });
    if (!tokenEntry) return null;
    if (isWeightedSumHoverSelection(label, info, object)) return null;
    if (isLayerNormParamHoverSelection(label, info, object)) return null;

    const resolvedLabel = resolveTokenChipLabel(tokenEntry.tokenLabel, tokenEntry.tokenIndex);
    if (!resolvedLabel) return null;

    return {
        suppressHoverLabel: false,
        showPrimaryLabel: true,
        primaryLabelText: tokenEntry.isBottomTokenChip ? 'Token' : '',
        detailKind: 'token-chip',
        detailText: resolvedLabel,
        tokenIndex: tokenEntry.tokenIndex,
        tokenId: tokenEntry.tokenId,
        tokenLabel: resolvedLabel
    };
}

export function resolveHoverLabelSubtitle({
    label = '',
    info = null,
    object = null
} = {}) {
    if (isBottomPositionChipHoverSelection(label)) {
        return '';
    }
    const positionIndex = resolveHoverPositionIndex(info, object);
    if (isBottomTokenChipHoverSelection(label)) {
        return Number.isFinite(positionIndex) ? `Position ${positionIndex}` : '';
    }
    if (isFinalLayerNormHoverSelection(label, info, object)) {
        return '';
    }
    const headIndex = findHoverTokenNumber(info, object, 'headIndex');
    const layerIndex = findHoverTokenNumber(info, object, 'layerIndex');
    const stage = findHoverTokenString(info, object, 'stage');
    const subtitle = formatHeadLayerSubtitle(headIndex, layerIndex);
    if (Number.isFinite(positionIndex)) {
        const positionText = `Position ${positionIndex}`;
        if (isResidualStreamHoverSelection(label, stage) || isPostLayerNormResidualHoverSelection(label, stage)) {
            return subtitle ? `${positionText} • ${subtitle}` : positionText;
        }
        if (isQkvStage(stage) || isWeightedSumHoverSelection(label, info, object)) {
            return subtitle ? `${positionText} • ${subtitle}` : positionText;
        }
    }
    if (!subtitle) {
        return '';
    }
    return subtitle;
}
