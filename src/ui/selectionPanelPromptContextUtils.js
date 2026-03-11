import { formatTokenLabel } from '../app/gpt-tower/tokenLabels.js';
import { getIncompleteUtf8TokenDisplay } from '../utils/tokenEncodingNotes.js';
import { normalizeTokenChipEntry, tokenChipEntriesMatch } from './tokenChipHoverSync.js';

function normalizePromptChipText(token) {
    if (token === null || token === undefined) return '';
    const raw = String(token);
    if (!raw.length) return '';
    return raw
        .replace(/\r/g, '')
        .replace(/\n/g, '\u21b5')
        .replace(/\t/g, '\u21e5')
        .replace(/ /g, '\u00A0');
}

export function buildSelectionPromptContext({
    activationSource = null,
    laneTokenIndices = null,
    tokenLabels = null,
    selectedTokenIndex = null,
    selectedTokenId = null,
    selectedTokenText = ''
} = {}) {
    const labels = Array.isArray(tokenLabels) ? tokenLabels : [];
    const indices = Array.isArray(laneTokenIndices) ? laneTokenIndices : [];
    const entries = labels
        .map((tokenLabel, laneIndex) => {
            const label = (tokenLabel === null || tokenLabel === undefined)
                ? ''
                : String(tokenLabel);
            if (!label.length) return null;
            const formattedLabel = formatTokenLabel(label);
            const tokenIndex = Number.isFinite(indices[laneIndex]) ? Math.floor(indices[laneIndex]) : null;
            let tokenId = null;
            if (
                Number.isFinite(tokenIndex)
                && activationSource
                && typeof activationSource.getTokenId === 'function'
            ) {
                const resolvedTokenId = activationSource.getTokenId(tokenIndex);
                tokenId = Number.isFinite(resolvedTokenId) ? Math.floor(resolvedTokenId) : null;
            }
            const incompleteDisplay = getIncompleteUtf8TokenDisplay(tokenId);
            const resolvedLabel = incompleteDisplay || formattedLabel;
            return {
                laneIndex,
                tokenIndex,
                tokenId,
                tokenLabel: resolvedLabel,
                displayText: normalizePromptChipText(resolvedLabel),
                titleText: resolvedLabel
            };
        })
        .filter(Boolean);

    const selectedDisplayText = getIncompleteUtf8TokenDisplay(selectedTokenId)
        || formatTokenLabel(selectedTokenText);
    const selectedEntry = normalizeTokenChipEntry({
        tokenIndex: selectedTokenIndex,
        tokenId: selectedTokenId,
        tokenLabel: selectedDisplayText
    });
    const activeIndex = selectedEntry
        ? entries.findIndex((entry) => tokenChipEntriesMatch(entry, selectedEntry))
        : -1;

    return {
        entries,
        activeIndex
    };
}
