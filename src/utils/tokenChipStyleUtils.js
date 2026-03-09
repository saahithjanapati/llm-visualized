import { formatTokenLabel } from '../app/gpt-tower/tokenLabels.js';
import { resolveTokenChipColors } from '../ui/tokenChipColorUtils.js';

function toFiniteIndex(value) {
    if (value === null || value === undefined || value === '') return null;
    const parsed = Number(value);
    if (!Number.isFinite(parsed)) return null;
    return Math.floor(parsed);
}

export function resolveTokenChipLabel(tokenText, tokenIndex = null) {
    const raw = tokenText === null || tokenText === undefined ? '' : String(tokenText);
    if (raw.length) return formatTokenLabel(raw);
    const safeTokenIndex = toFiniteIndex(tokenIndex);
    return Number.isFinite(safeTokenIndex) ? `Token ${safeTokenIndex + 1}` : '';
}

export function formatTokenChipDisplayText(tokenText, tokenIndex = null) {
    const label = resolveTokenChipLabel(tokenText, tokenIndex);
    if (!label) return '';
    return label
        .replace(/\r/g, '')
        .replace(/\n/g, '\u21b5')
        .replace(/\t/g, '\u21e5')
        .replace(/ /g, '\u00A0');
}

export function applyPromptTokenChipColors(element, {
    tokenText = '',
    tokenIndex = null,
    tokenId = null,
    fallbackIndex = null
} = {}) {
    if (!element || !element.style || typeof element.style.setProperty !== 'function') {
        return null;
    }

    const safeTokenIndex = toFiniteIndex(tokenIndex);
    const safeFallbackIndex = toFiniteIndex(fallbackIndex);
    const resolvedLabel = resolveTokenChipLabel(tokenText, safeTokenIndex);
    const colors = resolveTokenChipColors(
        {
            tokenId: toFiniteIndex(tokenId),
            tokenLabel: resolvedLabel
        },
        Number.isFinite(safeTokenIndex)
            ? safeTokenIndex
            : (Number.isFinite(safeFallbackIndex) ? safeFallbackIndex : 0)
    );

    element.style.setProperty('--token-color-border', colors.border);
    element.style.setProperty('--token-color-fill', colors.fill);
    element.style.setProperty('--token-color-fill-hover', colors.fillHover);
    return colors.colorKey;
}
