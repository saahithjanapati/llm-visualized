export const TOKEN_CHIP_HOVER_SYNC_EVENT = 'tokenChipHoverSync';

function toFiniteIndex(value) {
    if (value === null || value === undefined || value === '') return null;
    const parsed = Number(value);
    if (!Number.isFinite(parsed)) return null;
    return Math.floor(parsed);
}

function toTokenLabel(value) {
    if (value === null || value === undefined) return '';
    return String(value);
}

export function normalizeTokenChipEntry(entry = null) {
    if (!entry || typeof entry !== 'object') return null;
    const tokenIndex = toFiniteIndex(entry.tokenIndex);
    const tokenId = toFiniteIndex(entry.tokenId ?? entry.token_id);
    const tokenLabel = toTokenLabel(entry.tokenLabel ?? entry.tokenText ?? entry.token);
    if (!Number.isFinite(tokenIndex) && !Number.isFinite(tokenId) && !tokenLabel.length) {
        return null;
    }
    return {
        tokenIndex,
        tokenId,
        tokenLabel
    };
}

export function tokenChipEntriesMatch(a, b) {
    const left = normalizeTokenChipEntry(a);
    const right = normalizeTokenChipEntry(b);
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

export function dispatchTokenChipHoverSync(entry = null, {
    active = true,
    source = ''
} = {}) {
    if (typeof window === 'undefined' || typeof window.dispatchEvent !== 'function') return;
    const normalized = normalizeTokenChipEntry(entry);
    const isActive = !!active && !!normalized;
    window.dispatchEvent(new CustomEvent(TOKEN_CHIP_HOVER_SYNC_EVENT, {
        detail: {
            active: isActive,
            source: String(source || ''),
            tokenIndex: isActive ? normalized.tokenIndex : null,
            tokenId: isActive ? normalized.tokenId : null,
            tokenLabel: isActive ? normalized.tokenLabel : ''
        }
    }));
}
