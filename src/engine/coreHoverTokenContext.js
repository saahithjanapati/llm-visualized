import { resolveTokenChipLabel } from '../utils/tokenChipStyleUtils.js';

function toFiniteTokenNumber(value) {
    const parsed = Number(value);
    if (!Number.isFinite(parsed)) return null;
    return Math.floor(parsed);
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

function isWeightMatrixHoverSelection(label = '') {
    return String(label || '').toLowerCase().includes('weight matrix');
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

    const lower = String(label || '').toLowerCase();
    if (lower.includes('vector') || lower.includes('weighted sum')) return true;
    if (info?.vectorRef) return true;

    const kind = String(info?.kind || '').toLowerCase();
    if (kind === 'mergedkv') return true;

    const candidates = collectHoverDataCandidates(info, object);
    return candidates.some((candidate) => isVectorLikeCandidate(candidate));
}

export function resolveHoverTokenContext({
    label = '',
    info = null,
    object = null,
    activationSource = null
} = {}) {
    if (isEmbeddingMatrixHoverSelection(label) || isWeightMatrixHoverSelection(label)) {
        return null;
    }

    if (isBottomPositionChipHoverSelection(label)) return null;

    const isBottomTokenChip = isBottomTokenChipHoverSelection(label);
    if (!isBottomTokenChip && !isVectorLikeHoverSelection(label, info, object)) return null;

    const tokenIndex = findHoverTokenNumber(info, object, 'tokenIndex');
    let tokenId = findHoverTokenNumber(info, object, 'tokenId');
    if (!Number.isFinite(tokenId) && Number.isFinite(tokenIndex) && typeof activationSource?.getTokenId === 'function') {
        const resolvedTokenId = activationSource.getTokenId(tokenIndex);
        if (Number.isFinite(resolvedTokenId)) tokenId = Math.floor(resolvedTokenId);
    }

    if (isWeightedSumHoverSelection(label, info, object)) {
        if (!Number.isFinite(tokenIndex)) return null;
        return {
            suppressHoverLabel: false,
            showPrimaryLabel: true,
            detailKind: 'position-text',
            detailText: `Position ${tokenIndex + 1}`,
            tokenIndex: Math.floor(tokenIndex),
            tokenId: Number.isFinite(tokenId) ? Math.floor(tokenId) : null,
            tokenLabel: ''
        };
    }

    let tokenLabel = findHoverTokenString(info, object, 'tokenLabel');
    if (!tokenLabel && Number.isFinite(tokenIndex) && typeof activationSource?.getTokenString === 'function') {
        const resolvedTokenLabel = activationSource.getTokenString(tokenIndex);
        if (typeof resolvedTokenLabel === 'string' && resolvedTokenLabel.length) {
            tokenLabel = resolvedTokenLabel;
        }
    }

    const resolvedLabel = resolveTokenChipLabel(tokenLabel, tokenIndex);
    if (!resolvedLabel) return null;

    return {
        suppressHoverLabel: false,
        showPrimaryLabel: true,
        primaryLabelText: isBottomTokenChip ? 'Token' : '',
        detailKind: 'token-chip',
        detailText: resolvedLabel,
        tokenIndex: Number.isFinite(tokenIndex) ? Math.floor(tokenIndex) : null,
        tokenId: Number.isFinite(tokenId) ? Math.floor(tokenId) : null,
        tokenLabel: resolvedLabel
    };
}
