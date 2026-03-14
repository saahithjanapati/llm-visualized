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

export function normalizeTokenChipEntries(entries = null) {
    const rawEntries = Array.isArray(entries)
        ? entries
        : Array.isArray(entries?.entries)
            ? entries.entries
            : [entries];
    const normalizedEntries = [];
    rawEntries.forEach((entry) => {
        const normalizedEntry = normalizeTokenChipEntry(entry);
        if (!normalizedEntry) return;
        if (normalizedEntries.some((candidate) => tokenChipEntriesMatch(candidate, normalizedEntry))) {
            return;
        }
        normalizedEntries.push(normalizedEntry);
    });
    return normalizedEntries;
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

export function tokenChipEntryListsMatch(a, b) {
    const left = normalizeTokenChipEntries(a);
    const right = normalizeTokenChipEntries(b);
    if (left.length !== right.length) return false;
    return left.every((entry) => right.some((candidate) => tokenChipEntriesMatch(entry, candidate)));
}

export function matchesFocusVisibleTarget(target = null) {
    if (!target || typeof target.matches !== 'function') return false;
    try {
        return target.matches(':focus-visible');
    } catch (_) {
        return false;
    }
}

export function dispatchTokenChipHoverSync(entry = null, {
    active = true,
    source = ''
} = {}) {
    if (typeof window === 'undefined' || typeof window.dispatchEvent !== 'function') return;
    const normalizedEntries = !!active ? normalizeTokenChipEntries(entry) : [];
    const primaryEntry = normalizedEntries[0] || null;
    const isActive = !!active && normalizedEntries.length > 0;
    window.dispatchEvent(new CustomEvent(TOKEN_CHIP_HOVER_SYNC_EVENT, {
        detail: {
            active: isActive,
            source: String(source || ''),
            entries: isActive ? normalizedEntries : [],
            tokenIndex: isActive ? primaryEntry.tokenIndex : null,
            tokenId: isActive ? primaryEntry.tokenId : null,
            tokenLabel: isActive ? primaryEntry.tokenLabel : ''
        }
    }));
}
