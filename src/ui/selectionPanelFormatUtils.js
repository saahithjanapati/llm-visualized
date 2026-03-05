import {
    ATTENTION_SCORE_DECIMALS,
    ATTENTION_VALUE_PLACEHOLDER,
    SPACE_TOKEN_DISPLAY
} from './selectionPanelConstants.js';

export function formatValues(values, perLine = 8) {
    if (!values || typeof values.length !== 'number' || values.length === 0) return '(empty)';
    let result = '';
    for (let idx = 0; idx < values.length; idx += 1) {
        const num = Number(values[idx]);
        const formatted = Number.isFinite(num) ? num.toFixed(4) : '0.0000';
        const sep = idx === 0 ? '' : idx % perLine === 0 ? '\n' : ', ';
        result += sep + formatted;
    }
    return result;
}

export function normalizeAttentionValuePart(value, fallback = ATTENTION_VALUE_PLACEHOLDER) {
    const text = typeof value === 'string' ? value.trim() : '';
    return text || fallback;
}

export function formatTokenWithIndex(index, label, fallback = 'Token') {
    const tokenText = formatTokenLabelForPreview(label);
    if (Number.isFinite(index) && tokenText) return `${index + 1} (${tokenText})`;
    if (Number.isFinite(index)) return String(index + 1);
    return tokenText || fallback;
}

export function formatAttentionSubtitleTokenPart(label, tokenIndex, roleLabel) {
    const tokenText = normalizeAttentionValuePart(formatTokenLabelForPreview(label));
    const positionText = Number.isFinite(tokenIndex)
        ? `Position ${Math.floor(tokenIndex) + 1}`
        : 'Position n/a';
    return `${roleLabel} ${tokenText} (${positionText})`;
}

export function formatActivationData(data) {
    if (!data || typeof data !== 'object') return 'No activation data.';
    const lines = [];
    const stage = data.stage ? String(data.stage) : '';
    const stageLower = stage.toLowerCase();
    const isAttentionScore = stageLower.startsWith('attention.');
    if (stage) lines.push(`Stage: ${stage}`);
    if (Number.isFinite(data.layerIndex)) lines.push(`Layer: ${data.layerIndex + 1}`);
    if (isAttentionScore) {
        if (Number.isFinite(data.tokenIndex) || data.tokenLabel) {
            lines.push(`Source token: ${formatTokenWithIndex(data.tokenIndex, data.tokenLabel, 'Source')}`);
        }
        if (Number.isFinite(data.keyTokenIndex) || data.keyTokenLabel) {
            lines.push(`Target token: ${formatTokenWithIndex(data.keyTokenIndex, data.keyTokenLabel, 'Target')}`);
        }
    } else {
        if (Number.isFinite(data.tokenIndex)) {
            const tokenText = data.tokenLabel ? ` (${formatTokenLabelForPreview(data.tokenLabel)})` : '';
            lines.push(`Token: ${data.tokenIndex + 1}${tokenText}`);
        }
        if (Number.isFinite(data.keyTokenIndex)) {
            const keyText = data.keyTokenLabel ? ` (${formatTokenLabelForPreview(data.keyTokenLabel)})` : '';
            lines.push(`Key: ${data.keyTokenIndex + 1}${keyText}`);
        }
    }
    if (Number.isFinite(data.headIndex)) lines.push(`Head: ${data.headIndex + 1}`);
    if (Number.isFinite(data.segmentIndex)) lines.push(`Segment: ${data.segmentIndex + 1}`);
    if (Number.isFinite(data.preScore) || Number.isFinite(data.postScore)) {
        if (isAttentionScore) {
            const selectedMode = stageLower.includes('post') ? 'post' : 'pre';
            const selectedScore = selectedMode === 'post' ? data.postScore : data.preScore;
            if (Number.isFinite(selectedScore)) {
                lines.push(`Attention score (${selectedMode}-softmax): ${selectedScore.toFixed(ATTENTION_SCORE_DECIMALS)}`);
            }
        }
        if (Number.isFinite(data.preScore)) lines.push(`Pre-softmax: ${data.preScore.toFixed(ATTENTION_SCORE_DECIMALS)}`);
        if (Number.isFinite(data.postScore)) lines.push(`Post-softmax: ${data.postScore.toFixed(ATTENTION_SCORE_DECIMALS)}`);
    }
    if (data.values && typeof data.values.length === 'number') {
        lines.push(`Values (${data.values.length}):`);
        lines.push(formatValues(data.values));
    }
    if (data.notes) lines.push(String(data.notes));
    return lines.join('\n');
}

export function formatTokenLabelForPreview(label) {
    if (typeof label !== 'string') return '';
    const normalized = label.replace(/^\u0120+/, (match) => ' '.repeat(match.length));
    if (!normalized.length) return SPACE_TOKEN_DISPLAY;
    const collapsed = normalized.replace(/\s+/g, ' ');
    const trimmed = collapsed.trim();
    return trimmed.length ? trimmed : SPACE_TOKEN_DISPLAY;
}
