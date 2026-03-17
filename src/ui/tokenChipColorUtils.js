import {
    getLogitTokenChipColorCss,
    resolveAdjacentLogitTokenChipColorKeys,
    resolveLogitTokenChipColorKey
} from '../app/gpt-tower/logitColor.js';
import { normalizeTokenChipEntry, tokenChipEntriesMatch } from './tokenChipHoverSync.js';

let activePromptTokenChipLookup = new Map();

function toFiniteInteger(value) {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? Math.floor(parsed) : null;
}

function toTokenLabel(value) {
    if (value === null || value === undefined) return '';
    return String(value);
}

function buildTokenChipLookupKeys(entry = null) {
    const normalized = normalizeTokenChipEntry(entry);
    if (!normalized) return [];

    const keys = [];
    if (Number.isFinite(normalized.tokenIndex)) {
        keys.push(`index:${normalized.tokenIndex}`);
    }
    if (Number.isFinite(normalized.tokenId) && normalized.tokenLabel.length > 0) {
        keys.push(`id:${normalized.tokenId}|label:${normalized.tokenLabel}`);
    }
    if (Number.isFinite(normalized.tokenId)) {
        keys.push(`id:${normalized.tokenId}`);
    }
    if (normalized.tokenLabel.length > 0) {
        keys.push(`label:${normalized.tokenLabel}`);
    }
    return keys;
}

function resolveStoredTokenChipColorKey(entry = null, lookup = null) {
    if (!(lookup instanceof Map)) return null;
    const keys = buildTokenChipLookupKeys(entry);
    for (let i = 0; i < keys.length; i += 1) {
        const colorKey = lookup.get(keys[i]);
        if (Number.isFinite(colorKey)) {
            return colorKey;
        }
    }
    return null;
}

export function buildPromptTokenChipEntries({
    tokenLabels = [],
    tokenIndices = null,
    tokenIds = null,
    generatedToken = null
} = {}) {
    const labels = Array.isArray(tokenLabels) ? tokenLabels : [];
    const indices = Array.isArray(tokenIndices) ? tokenIndices : [];
    const ids = Array.isArray(tokenIds) ? tokenIds : [];

    const promptEntries = labels
        .map((tokenLabel, laneIndex) => {
            const label = toTokenLabel(tokenLabel);
            if (!label.length) return null;
            return {
                entryType: 'prompt',
                laneIndex,
                tokenIndex: toFiniteInteger(indices[laneIndex]),
                tokenId: toFiniteInteger(ids[laneIndex]),
                tokenLabel: label
            };
        })
        .filter(Boolean);

    const generatedLabel = toTokenLabel(generatedToken?.tokenLabel);
    const generatedEntry = generatedLabel
        ? {
            entryType: 'generated',
            laneIndex: toFiniteInteger(generatedToken?.laneIndex),
            tokenIndex: toFiniteInteger(generatedToken?.tokenIndex),
            tokenId: toFiniteInteger(generatedToken?.tokenId),
            tokenLabel: generatedLabel,
            selectionLabel: (typeof generatedToken?.selectionLabel === 'string' && generatedToken.selectionLabel.length)
                ? generatedToken.selectionLabel
                : null,
            logitEntry: (generatedToken?.logitEntry && typeof generatedToken.logitEntry === 'object')
                ? generatedToken.logitEntry
                : null,
            seed: toFiniteInteger(generatedToken?.seed)
        }
        : null;

    const generatedAlreadyPresent = generatedEntry
        ? promptEntries.some((entry) => tokenChipEntriesMatch(entry, generatedEntry))
        : false;
    return (generatedEntry && !generatedAlreadyPresent)
        ? [...promptEntries, generatedEntry]
        : promptEntries;
}

export function resolvePromptTokenChipColorState(entries = [], { previousLookup = null } = {}) {
    const safeEntries = Array.isArray(entries) ? entries.filter(Boolean) : [];
    const effectivePreviousLookup = previousLookup instanceof Map ? previousLookup : null;
    const preferredColorKeys = effectivePreviousLookup
        ? safeEntries.map((entry) => resolveStoredTokenChipColorKey(entry, effectivePreviousLookup))
        : null;
    const colorKeys = resolveAdjacentLogitTokenChipColorKeys(safeEntries, { preferredColorKeys });
    const lookup = new Map();

    safeEntries.forEach((entry, index) => {
        const colorKey = colorKeys[index] ?? resolveLogitTokenChipColorKey(entry, index);
        buildTokenChipLookupKeys(entry).forEach((key) => {
            if (!lookup.has(key)) {
                lookup.set(key, colorKey);
            }
        });
    });

    return {
        entries: safeEntries,
        colorKeys,
        lookup
    };
}

export function setActivePromptTokenChipEntries(entries = []) {
    const state = resolvePromptTokenChipColorState(entries, {
        previousLookup: activePromptTokenChipLookup
    });
    activePromptTokenChipLookup = state.lookup;
    return state;
}

export function getActivePromptTokenChipLookup() {
    return new Map(activePromptTokenChipLookup);
}

export function resolveTokenChipColorKey(entry = null, fallbackIndex = 0, { lookup = null } = {}) {
    const explicitColorKey = Number(entry?.colorKey ?? entry?.color_key);
    if (Number.isFinite(explicitColorKey)) {
        return Math.floor(explicitColorKey);
    }
    const effectiveLookup = lookup instanceof Map ? lookup : activePromptTokenChipLookup;
    const normalized = normalizeTokenChipEntry(entry);
    if (effectiveLookup && normalized) {
        const keys = buildTokenChipLookupKeys(normalized);
        for (let i = 0; i < keys.length; i += 1) {
            const colorKey = effectiveLookup.get(keys[i]);
            if (Number.isFinite(colorKey)) {
                return colorKey;
            }
        }
    }
    return resolveLogitTokenChipColorKey(normalized || entry, fallbackIndex);
}

export function resolveTokenChipColors(entry = null, fallbackIndex = 0, options = {}) {
    const colorKey = resolveTokenChipColorKey(entry, fallbackIndex, options);
    return {
        colorKey,
        border: getLogitTokenChipColorCss(colorKey, 0.92),
        fill: getLogitTokenChipColorCss(colorKey, 0.2),
        fillHover: getLogitTokenChipColorCss(colorKey, 0.28),
        fillActive: getLogitTokenChipColorCss(colorKey, 0.42)
    };
}

export function applyTokenChipColors(element, entry = null, fallbackIndex = 0, options = {}) {
    if (!element) return null;
    const colors = resolveTokenChipColors(entry, fallbackIndex, options);
    element.style.setProperty('--token-color-border', colors.border);
    element.style.setProperty('--token-color-fill', colors.fill);
    element.style.setProperty('--token-color-fill-hover', colors.fillHover);
    element.style.setProperty('--token-color-fill-active', colors.fillActive);
    return colors;
}
