import { getIncompleteUtf8TokenDisplay } from './tokenEncodingNotes.js';

export function resolveLogitEntryText(entry = null) {
    if (!entry || typeof entry !== 'object') return '';

    const rawTokenId = Number(entry.tokenId ?? entry.token_id);
    const tokenId = Number.isFinite(rawTokenId) ? Math.floor(rawTokenId) : null;
    const incompleteDisplay = getIncompleteUtf8TokenDisplay(tokenId);
    if (incompleteDisplay) return incompleteDisplay;

    const candidates = [
        entry.tokenLabel,
        entry.token,
        entry.tokenDisplay,
        entry.token_display
    ];

    for (let i = 0; i < candidates.length; i += 1) {
        const candidate = candidates[i];
        if (typeof candidate === 'string' && candidate.length) {
            return candidate;
        }
    }

    return '';
}
