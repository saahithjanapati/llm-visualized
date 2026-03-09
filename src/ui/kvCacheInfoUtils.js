export const KV_CACHE_INFO_REQUEST_EVENT = 'llm-visualized:open-kv-cache-info';

export function normalizeKvCachePhase(phase = 'prefill') {
    return String(phase || '').toLowerCase() === 'decode' ? 'decode' : 'prefill';
}

export function formatKvCachePhaseLabel(phase = 'prefill') {
    return normalizeKvCachePhase(phase) === 'decode'
        ? 'Decode'
        : 'Pre-Fill';
}

export function buildKvCacheOverlayBadgeText(phase = 'prefill') {
    return `KV Cache Enabled \u00b7 ${formatKvCachePhaseLabel(phase)}`;
}

export function buildKvCacheInfoSelection({ phase = 'prefill' } = {}) {
    const safePhase = normalizeKvCachePhase(phase);
    const phaseLabel = formatKvCachePhaseLabel(safePhase);
    return {
        label: `KV Cache: ${phaseLabel}`,
        kind: 'kvCacheInfo',
        info: {
            kvCachePhase: safePhase,
            kvCachePhaseLabel: phaseLabel
        }
    };
}

export function isKvCacheInfoSelection(label = '', selectionInfo = null) {
    if (selectionInfo?.kind === 'kvCacheInfo') return true;
    return String(label || '').toLowerCase().startsWith('kv cache');
}
