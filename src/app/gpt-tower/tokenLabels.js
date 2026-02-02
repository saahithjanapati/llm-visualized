import { SPACE_TOKEN_DISPLAY } from './config.js';

export function formatTokenLabel(token) {
    if (token === null || token === undefined) return SPACE_TOKEN_DISPLAY;
    const raw = String(token);
    const normalized = raw.replace(/^\u0120+/, (match) => ' '.repeat(match.length));
    if (!normalized.length) return SPACE_TOKEN_DISPLAY;
    if (normalized.trim().length === 0) return SPACE_TOKEN_DISPLAY;
    return normalized;
}
