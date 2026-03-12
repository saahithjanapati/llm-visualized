import { formatTokenLabel } from '../app/gpt-tower/tokenLabels.js';

function toFiniteTokenIndex(value) {
    const parsed = Number(value);
    if (!Number.isFinite(parsed)) return null;
    return Math.floor(parsed);
}

export function isPlaceholderTokenLabel(label = '') {
    const text = typeof label === 'string' ? label.trim() : '';
    if (!text.length) return false;
    return /^token\s+\d+$/i.test(text);
}

export function resolvePreferredTokenLabel({
    tokenLabel = '',
    tokenIndex = null,
    activationSource = null
} = {}) {
    const rawFallback = typeof tokenLabel === 'string' ? tokenLabel : '';
    const normalizedFallback = rawFallback.trim();
    const safeTokenIndex = toFiniteTokenIndex(tokenIndex);

    if (normalizedFallback.length && !isPlaceholderTokenLabel(normalizedFallback)) {
        return formatTokenLabel(rawFallback);
    }

    if (rawFallback.length && !normalizedFallback.length) {
        return formatTokenLabel(rawFallback);
    }

    if (Number.isFinite(safeTokenIndex) && typeof activationSource?.getTokenString === 'function') {
        const sourceLabel = activationSource.getTokenString(safeTokenIndex);
        if (typeof sourceLabel === 'string' && sourceLabel.length) {
            return formatTokenLabel(sourceLabel);
        }
    }

    return rawFallback.length ? formatTokenLabel(rawFallback) : '';
}
