import { normalizeTokenChipEntries } from './tokenChipHoverSync.js';

function toFiniteIndex(value) {
    if (value === null || value === undefined || value === '') return null;
    const parsed = Number(value);
    if (!Number.isFinite(parsed)) return null;
    return Math.floor(parsed);
}

function resolveRowTokenMetadata(rowIndex, {
    previewData = null,
    tokenIndices = null,
    tokenLabels = null,
    activationSource = null
} = {}) {
    const safeRowIndex = toFiniteIndex(rowIndex);
    if (!Number.isFinite(safeRowIndex)) return null;

    const rowData = Array.isArray(previewData?.rows) ? previewData.rows[safeRowIndex] : null;
    const tokenIndex = toFiniteIndex(
        rowData?.tokenIndex
            ?? (Array.isArray(tokenIndices) ? tokenIndices[safeRowIndex] : null)
    );

    let tokenLabel = typeof rowData?.tokenLabel === 'string' ? rowData.tokenLabel : '';
    if (!tokenLabel && Array.isArray(tokenLabels)) {
        const candidate = tokenLabels[safeRowIndex];
        tokenLabel = typeof candidate === 'string' ? candidate : '';
    }
    if (
        !tokenLabel
        && activationSource
        && typeof activationSource.getTokenString === 'function'
        && Number.isFinite(tokenIndex)
    ) {
        const candidate = activationSource.getTokenString(tokenIndex);
        tokenLabel = typeof candidate === 'string' ? candidate : '';
    }

    let tokenId = null;
    if (
        activationSource
        && typeof activationSource.getTokenId === 'function'
        && Number.isFinite(tokenIndex)
    ) {
        tokenId = toFiniteIndex(activationSource.getTokenId(tokenIndex));
    }

    return {
        tokenIndex,
        tokenId,
        tokenLabel
    };
}

export function resolveMhsaTokenMatrixHoverTokenEntries(targetInfo = null, options = {}) {
    if (!targetInfo || typeof targetInfo !== 'object') return [];

    switch (targetInfo.kind) {
    case 'row':
    case 'key-link':
        return normalizeTokenChipEntries([
            resolveRowTokenMetadata(targetInfo.rowIndex, options)
        ]);
    case 'score':
        return normalizeTokenChipEntries([
            resolveRowTokenMetadata(targetInfo.rowIndex, options),
            resolveRowTokenMetadata(targetInfo.colIndex, options)
        ]);
    default:
        return [];
    }
}

export function resolveMhsaTokenMatrixHoverTokenEntry(targetInfo = null, options = {}) {
    return resolveMhsaTokenMatrixHoverTokenEntries(targetInfo, options)[0] || null;
}
