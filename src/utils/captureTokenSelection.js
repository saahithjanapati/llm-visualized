function normalizeFiniteInteger(value) {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? Math.floor(parsed) : null;
}

function normalizeTokenText(value) {
    return typeof value === 'string' && value.length ? value : '';
}

function resolveVisibleTokenCount(activationSource = null) {
    const tokenCount = activationSource && typeof activationSource.getTokenCount === 'function'
        ? activationSource.getTokenCount()
        : null;
    return Number.isFinite(tokenCount) && tokenCount >= 0
        ? Math.floor(tokenCount)
        : 0;
}

function resolveBestLogitEntryIndex(logitRow = []) {
    if (!Array.isArray(logitRow) || !logitRow.length) return -1;
    let bestIdx = -1;
    let bestProb = -Infinity;
    for (let index = 0; index < logitRow.length; index += 1) {
        const probability = Number(logitRow[index]?.prob ?? logitRow[index]?.probability);
        if (!Number.isFinite(probability)) continue;
        if (probability > bestProb) {
            bestProb = probability;
            bestIdx = index;
        }
    }
    return bestIdx;
}

export function resolveHiddenTerminalToken(activationSource = null) {
    const rawToken = activationSource && typeof activationSource.getHiddenTerminalToken === 'function'
        ? activationSource.getHiddenTerminalToken()
        : activationSource?.meta?.hidden_terminal_token;
    if (!rawToken || typeof rawToken !== 'object') return null;

    const tokenId = normalizeFiniteInteger(rawToken.token_id ?? rawToken.tokenId);
    const tokenRaw = normalizeTokenText(
        rawToken.token
        ?? rawToken.token_raw
        ?? rawToken.tokenRaw
        ?? rawToken.token_string
        ?? rawToken.tokenString
    );
    const tokenDisplay = normalizeTokenText(rawToken.token_display ?? rawToken.tokenDisplay);
    const tokenHf = normalizeTokenText(rawToken.token_hf ?? rawToken.tokenHf);
    if (!Number.isFinite(tokenId) && !tokenRaw && !tokenDisplay && !tokenHf) {
        return null;
    }

    return {
        tokenId,
        tokenRaw: tokenRaw || tokenDisplay || tokenHf,
        tokenDisplay: tokenDisplay || tokenRaw || tokenHf,
        tokenHf
    };
}

export function findMatchingLogitEntryIndex(logitRow = [], {
    tokenRaw = null,
    tokenId = null
} = {}) {
    if (!Array.isArray(logitRow) || !logitRow.length) return -1;

    const resolvedTokenId = normalizeFiniteInteger(tokenId);
    if (Number.isFinite(resolvedTokenId)) {
        for (let index = 0; index < logitRow.length; index += 1) {
            const entryTokenId = normalizeFiniteInteger(logitRow[index]?.token_id ?? logitRow[index]?.tokenId);
            if (entryTokenId === resolvedTokenId) {
                return index;
            }
        }
    }

    const resolvedTokenRaw = normalizeTokenText(tokenRaw);
    if (resolvedTokenRaw) {
        for (let index = 0; index < logitRow.length; index += 1) {
            if (normalizeTokenText(logitRow[index]?.token) === resolvedTokenRaw) {
                return index;
            }
        }
    }

    return -1;
}

export function resolveChosenTokenCandidateForToken(
    activationSource = null,
    tokenIndex = null,
    { logitLimit = null } = {}
) {
    if (!activationSource || typeof activationSource.getLogitsForToken !== 'function') return null;
    const resolvedTokenIndex = normalizeFiniteInteger(tokenIndex);
    if (!Number.isFinite(resolvedTokenIndex)) return null;

    const logitRow = activationSource.getLogitsForToken(resolvedTokenIndex, logitLimit);
    if (!Array.isArray(logitRow) || !logitRow.length) return null;

    const tokenCount = resolveVisibleTokenCount(activationSource);
    const nextVisibleTokenIndex = resolvedTokenIndex + 1;
    const hasNextVisibleToken = Number.isFinite(tokenCount) && nextVisibleTokenIndex < tokenCount;

    if (hasNextVisibleToken) {
        const tokenRaw = typeof activationSource.getTokenString === 'function'
            ? normalizeTokenText(activationSource.getTokenString(nextVisibleTokenIndex))
            : '';
        const tokenId = typeof activationSource.getTokenId === 'function'
            ? normalizeFiniteInteger(activationSource.getTokenId(nextVisibleTokenIndex))
            : null;
        const logitEntryIndex = findMatchingLogitEntryIndex(logitRow, {
            tokenRaw,
            tokenId
        });
        return {
            resolution: 'next-visible-token',
            tokenIndex: nextVisibleTokenIndex,
            tokenId,
            tokenRaw,
            tokenDisplay: tokenRaw,
            logitRow,
            logitEntryIndex,
            logitEntry: logitEntryIndex >= 0 ? logitRow[logitEntryIndex] : null
        };
    }

    const hiddenTerminalToken = (
        Number.isFinite(tokenCount)
        && resolvedTokenIndex === Math.max(0, tokenCount - 1)
    )
        ? resolveHiddenTerminalToken(activationSource)
        : null;
    if (hiddenTerminalToken) {
        const logitEntryIndex = findMatchingLogitEntryIndex(logitRow, {
            tokenRaw: hiddenTerminalToken.tokenRaw,
            tokenId: hiddenTerminalToken.tokenId
        });
        return {
            resolution: 'hidden-terminal-token',
            tokenIndex: null,
            tokenId: hiddenTerminalToken.tokenId,
            tokenRaw: hiddenTerminalToken.tokenRaw,
            tokenDisplay: hiddenTerminalToken.tokenDisplay,
            logitRow,
            logitEntryIndex,
            logitEntry: logitEntryIndex >= 0 ? logitRow[logitEntryIndex] : null
        };
    }

    const bestLogitEntryIndex = resolveBestLogitEntryIndex(logitRow);
    if (bestLogitEntryIndex < 0) return null;
    const bestEntry = logitRow[bestLogitEntryIndex];
    const bestTokenId = normalizeFiniteInteger(bestEntry?.token_id ?? bestEntry?.tokenId);
    const bestTokenRaw = normalizeTokenText(bestEntry?.token);
    const bestTokenDisplay = normalizeTokenText(bestEntry?.token_display ?? bestEntry?.tokenDisplay);

    return {
        resolution: 'top-logit',
        tokenIndex: null,
        tokenId: bestTokenId,
        tokenRaw: bestTokenRaw || bestTokenDisplay,
        tokenDisplay: bestTokenDisplay || bestTokenRaw,
        logitRow,
        logitEntryIndex: bestLogitEntryIndex,
        logitEntry: bestEntry
    };
}
