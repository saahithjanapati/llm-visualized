function clamp(value, min, max) {
    return Math.min(max, Math.max(min, value));
}

export function computeAttentionCellSize(count, { targetPx, minCell, maxCell }) {
    const safeCount = Math.max(1, Math.floor(count || 1));
    const size = targetPx / safeCount;
    return Math.max(minCell, Math.min(maxCell, size));
}

export function shouldRevealAttentionCell(progress, row, col, mode = 'pre', decodeProfile = null) {
    const decodeHighlightRow = Number.isFinite(decodeProfile?.highlightRow)
        ? Math.max(0, Math.floor(decodeProfile.highlightRow))
        : null;
    const decodeAnimating = !!(decodeProfile?.enabled && decodeProfile?.animating && decodeHighlightRow !== null);
    if (decodeAnimating) {
        // During KV-cache decode animation, preserve previously computed rows
        // and only allow the currently computed row to animate/reveal.
        if (row < decodeHighlightRow) return true;
        if (!progress) return false;
    }
    if (!progress) return true;
    if (mode === 'post') {
        const postCompletedRows = Number.isFinite(progress.postCompletedRows) ? progress.postCompletedRows : 0;
        return row < postCompletedRows;
    }
    const completedRows = Number.isFinite(progress.completedRows) ? progress.completedRows : 0;
    const activeRow = Number.isFinite(progress.activeRow) ? progress.activeRow : null;
    const activeCol = Number.isFinite(progress.activeCol) ? progress.activeCol : null;
    if (row < completedRows) return true;
    if (activeRow !== null && activeCol !== null && row === activeRow) {
        return col <= activeCol;
    }
    return false;
}

export function shouldAnimateAttentionCellReveal({
    mode = 'pre',
    wasEmpty = false,
    isMuted = false
} = {}) {
    if (!wasEmpty) return false;
    if (mode === 'post') return true;
    return !isMuted;
}

export function resolveMutedAttentionCellRevealDelay(
    row,
    col,
    count,
    staggerMs = 0,
    extraDelayMs = 0
) {
    const safeCount = Math.max(1, Math.floor(count || 1));
    const safeRow = clamp(Math.floor(row || 0), 0, safeCount - 1);
    const safeCol = clamp(Math.floor(col || 0), 0, safeCount - 1);
    const safeStagger = Number.isFinite(staggerMs) ? Math.max(0, Math.round(staggerMs)) : 0;
    const safeExtraDelay = Number.isFinite(extraDelayMs) ? Math.max(0, Math.round(extraDelayMs)) : 0;
    const completedPrefixCount = Math.max(1, Math.min(safeCount, safeRow + 1));
    const mutedOrder = Math.max(0, safeCol - completedPrefixCount);
    return (completedPrefixCount * safeStagger) + safeExtraDelay + (mutedOrder * safeStagger);
}

export function countVisibleAttentionCellsInRow(row, count, triangle = 'lower') {
    const safeCount = Math.max(1, Math.floor(count || 1));
    const safeRow = clamp(Math.floor(row || 0), 0, safeCount - 1);
    if (triangle === 'full') {
        return safeCount;
    }
    if (triangle === 'upper') {
        return Math.max(1, safeCount - safeRow);
    }
    return Math.max(1, safeRow + 1);
}

export function getAttentionRevealOrder(row, col, count, triangle = 'lower') {
    const safeCount = Math.max(1, Math.floor(count || 1));
    const safeRow = clamp(Math.floor(row || 0), 0, safeCount - 1);
    const safeCol = clamp(Math.floor(col || 0), 0, safeCount - 1);
    if (triangle === 'full') {
        return safeCol;
    }
    if (triangle === 'upper') {
        return Math.max(0, safeCol - safeRow);
    }
    return safeCol;
}
