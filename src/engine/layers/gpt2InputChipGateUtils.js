function resolveTokenKey(tokenIndex) {
    return Number.isFinite(tokenIndex) ? String(Math.max(0, Math.floor(tokenIndex))) : null;
}

function resolveGateReleaseAt(gate, tokenKey) {
    if (!gate) return NaN;
    const releaseByToken = gate.releaseByToken;
    if (
        tokenKey !== null
        && releaseByToken
        && Object.prototype.hasOwnProperty.call(releaseByToken, tokenKey)
        && Number.isFinite(releaseByToken[tokenKey])
    ) {
        return releaseByToken[tokenKey];
    }
    return Number.isFinite(gate.defaultReleaseAt) ? gate.defaultReleaseAt : NaN;
}

function shouldWaitForReleaseAfterInside(gate) {
    return !!(gate && gate.waitForReleaseAfterInside === true);
}

export function shouldWaitForInputChipGate(gate, tokenIndex, nowMs = NaN) {
    if (!gate || gate.enabled === false) return false;

    if (gate.pending) {
        const pendingFallbackAt = Number.isFinite(gate.pendingFallbackAt) ? gate.pendingFallbackAt : NaN;
        if (Number.isFinite(nowMs) && Number.isFinite(pendingFallbackAt) && nowMs >= pendingFallbackAt) {
            return false;
        }
        return true;
    }

    const tokenKey = resolveTokenKey(tokenIndex);
    const insideByToken = gate.insideByToken;
    if (
        tokenKey !== null
        && insideByToken
        && Object.prototype.hasOwnProperty.call(insideByToken, tokenKey)
    ) {
        const tokenReleaseAt = resolveGateReleaseAt(gate, tokenKey);
        if (insideByToken[tokenKey] === true) {
            if (
                shouldWaitForReleaseAfterInside(gate)
                && Number.isFinite(nowMs)
                && Number.isFinite(tokenReleaseAt)
            ) {
                return nowMs < tokenReleaseAt;
            }
            return false;
        }
        if (Number.isFinite(nowMs) && Number.isFinite(tokenReleaseAt) && nowMs >= tokenReleaseAt) {
            return false;
        }
        return true;
    }

    const defaultReleaseAt = resolveGateReleaseAt(gate, tokenKey);
    if (Number.isFinite(nowMs) && Number.isFinite(defaultReleaseAt)) {
        return nowMs < defaultReleaseAt;
    }
    return false;
}
