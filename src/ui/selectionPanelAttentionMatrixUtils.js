export function isCausalUpperAttentionCell(queryTokenIndex = null, keyTokenIndex = null) {
    const safeQueryTokenIndex = Number.isFinite(queryTokenIndex) ? Math.floor(queryTokenIndex) : null;
    const safeKeyTokenIndex = Number.isFinite(keyTokenIndex) ? Math.floor(keyTokenIndex) : null;
    if (!Number.isFinite(safeQueryTokenIndex) || !Number.isFinite(safeKeyTokenIndex)) {
        return false;
    }
    return safeKeyTokenIndex > safeQueryTokenIndex;
}

export function resolveAttentionMatrixCellValue({
    mode = 'pre',
    value = null,
    queryTokenIndex = null,
    keyTokenIndex = null
} = {}) {
    if (Number.isFinite(value)) return value;
    if (
        String(mode || '').toLowerCase() === 'post'
        && isCausalUpperAttentionCell(queryTokenIndex, keyTokenIndex)
    ) {
        return 0;
    }
    return null;
}

export function shouldMuteCausalUpperPreAttentionCell({
    mode = 'pre',
    queryTokenIndex = null,
    keyTokenIndex = null
} = {}) {
    return String(mode || '').toLowerCase() === 'pre'
        && isCausalUpperAttentionCell(queryTokenIndex, keyTokenIndex);
}

export function buildAttentionMatrixValues({
    activationSource,
    layerIndex,
    headIndex,
    tokenIndices,
    mode
} = {}) {
    if (!activationSource || !Array.isArray(tokenIndices) || !tokenIndices.length) return null;

    return tokenIndices.map((queryTokenIndex) => {
        const row = activationSource.getAttentionScoresRow
            ? activationSource.getAttentionScoresRow(layerIndex, mode, headIndex, queryTokenIndex)
            : null;

        return tokenIndices.map((keyTokenIndex) => resolveAttentionMatrixCellValue({
            mode,
            value: Array.isArray(row) ? row[keyTokenIndex] : null,
            queryTokenIndex,
            keyTokenIndex
        }));
    });
}

export function shouldClearPinnedAttentionOnDocumentPointerDown({
    isPinned = false,
    hitPanelTokenNavChip = false,
    hitPanelAttentionScoreLink = false,
    insideAttentionBody = false,
    insideAttentionMatrix = false,
    validMatrixCell = false,
    panelHit = false
} = {}) {
    if (!isPinned) return false;
    if (hitPanelTokenNavChip || hitPanelAttentionScoreLink) return false;
    if (insideAttentionBody) {
        if (insideAttentionMatrix) return !validMatrixCell;
        return true;
    }
    if (insideAttentionMatrix) return !validMatrixCell;
    if (panelHit) return false;
    return true;
}
