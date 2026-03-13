function normalizeIndex(value = null) {
    return Number.isFinite(value) ? Math.max(0, Math.floor(value)) : null;
}

function normalizeLabel(value = '') {
    return typeof value === 'string' ? value : '';
}

function isNumericValue(value) {
    return typeof value === 'number' && !Number.isNaN(value);
}

export function normalizeAttentionHoverStageKey(stageKey = '') {
    const safeKey = String(stageKey || '').trim().toLowerCase();
    if (safeKey === 'pre' || safeKey === 'pre-score') return 'pre';
    if (safeKey === 'masked-input' || safeKey === 'masked_input') return 'masked-input';
    if (safeKey === 'mask' || safeKey === 'causal-mask' || safeKey === 'causal_mask') return 'mask';
    if (safeKey === 'post' || safeKey === 'post-copy' || safeKey === 'post_copy') return 'post';
    return 'pre';
}

export function resolveAttentionHoverLabel(stageKey = '') {
    switch (normalizeAttentionHoverStageKey(stageKey)) {
    case 'mask':
        return 'Causal Mask';
    case 'post':
        return 'Post-Softmax Attention Score';
    case 'masked-input':
        return 'Masked Attention Score';
    case 'pre':
    default:
        return 'Pre-Softmax Attention Score';
    }
}

export function resolveAttentionHoverActivationStage(stageKey = '') {
    switch (normalizeAttentionHoverStageKey(stageKey)) {
    case 'masked-input':
        return 'attention.masked_input';
    case 'mask':
        return 'attention.mask';
    case 'post':
        return 'attention.post';
    case 'pre':
    default:
        return 'attention.pre';
    }
}

export function buildAttentionHoverInfo({
    stageKey = 'pre',
    layerIndex = null,
    headIndex = null,
    queryTokenIndex = null,
    queryTokenLabel = '',
    keyTokenIndex = null,
    keyTokenLabel = '',
    preScore = null,
    postScore = null,
    maskValue = null,
    isMasked = false
} = {}) {
    const normalizedStageKey = normalizeAttentionHoverStageKey(stageKey);
    const safeLayerIndex = normalizeIndex(layerIndex);
    const safeHeadIndex = normalizeIndex(headIndex);
    const safeQueryTokenIndex = normalizeIndex(queryTokenIndex);
    const safeKeyTokenIndex = normalizeIndex(keyTokenIndex);
    const safeQueryTokenLabel = normalizeLabel(queryTokenLabel);
    const safeKeyTokenLabel = normalizeLabel(keyTokenLabel);
    const safePreScore = isNumericValue(preScore) ? preScore : null;
    const safePostScore = isNumericValue(postScore) ? postScore : null;
    const safeMaskValue = isNumericValue(maskValue) ? maskValue : null;
    const safeIsMasked = isMasked === true;
    const label = resolveAttentionHoverLabel(normalizedStageKey);
    const activationStage = resolveAttentionHoverActivationStage(normalizedStageKey);
    const showMaskValue = normalizedStageKey === 'mask'
        || normalizedStageKey === 'masked-input'
        || safeIsMasked;

    const activationData = {
        label,
        stage: activationStage,
        ...(Number.isFinite(safeLayerIndex) ? { layerIndex: safeLayerIndex } : {}),
        ...(Number.isFinite(safeHeadIndex) ? { headIndex: safeHeadIndex } : {}),
        ...(Number.isFinite(safeQueryTokenIndex) ? { tokenIndex: safeQueryTokenIndex, queryTokenIndex: safeQueryTokenIndex } : {}),
        ...(safeQueryTokenLabel.length ? { tokenLabel: safeQueryTokenLabel, queryTokenLabel: safeQueryTokenLabel } : {}),
        ...(Number.isFinite(safeKeyTokenIndex) ? { keyTokenIndex: safeKeyTokenIndex } : {}),
        ...(safeKeyTokenLabel.length ? { keyTokenLabel: safeKeyTokenLabel } : {}),
        ...(safePreScore !== null ? { preScore: safePreScore } : {}),
        ...(safePostScore !== null ? { postScore: safePostScore } : {}),
        ...(safeMaskValue !== null ? { maskValue: safeMaskValue } : {}),
        ...(safeIsMasked ? { isMasked: true } : {}),
        ...(showMaskValue ? { showMaskValue: true } : {})
    };

    return {
        ...(Number.isFinite(safeLayerIndex) ? { layerIndex: safeLayerIndex } : {}),
        ...(Number.isFinite(safeHeadIndex) ? { headIndex: safeHeadIndex } : {}),
        ...(Number.isFinite(safeQueryTokenIndex) ? { tokenIndex: safeQueryTokenIndex, queryTokenIndex: safeQueryTokenIndex } : {}),
        ...(safeQueryTokenLabel.length ? { tokenLabel: safeQueryTokenLabel, queryTokenLabel: safeQueryTokenLabel } : {}),
        ...(Number.isFinite(safeKeyTokenIndex) ? { keyTokenIndex: safeKeyTokenIndex } : {}),
        ...(safeKeyTokenLabel.length ? { keyTokenLabel: safeKeyTokenLabel } : {}),
        ...(safePreScore !== null ? { preScore: safePreScore } : {}),
        ...(safePostScore !== null ? { postScore: safePostScore } : {}),
        ...(safeMaskValue !== null ? { maskValue: safeMaskValue } : {}),
        ...(safeIsMasked ? { isMasked: true } : {}),
        ...(showMaskValue ? { showMaskValue: true } : {}),
        activationData
    };
}
