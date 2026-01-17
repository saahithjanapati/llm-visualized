import { ActivationUtils } from './CaptureActivationSource.js';
import rawParams from './gpt2_layernorm_params.json';

const PARAM_CACHE = new Map();

function getRawParam(layerIndex, lnKind, param) {
    if (!rawParams || typeof rawParams !== 'object') return null;
    if (lnKind === 'final') {
        return rawParams.final && rawParams.final[param] ? rawParams.final[param] : null;
    }
    const layer = rawParams.layers && rawParams.layers[layerIndex];
    if (!layer || !layer[lnKind] || !layer[lnKind][param]) return null;
    return layer[lnKind][param];
}

export function getLayerNormParamData(layerIndex, lnKind, param, targetLength) {
    const length = Number.isFinite(targetLength) ? Math.max(1, Math.floor(targetLength)) : null;
    const cacheKey = `${layerIndex}|${lnKind}|${param}|${length ?? 'auto'}`;
    if (PARAM_CACHE.has(cacheKey)) return PARAM_CACHE.get(cacheKey);
    const raw = getRawParam(layerIndex, lnKind, param);
    if (!raw || !Array.isArray(raw) || raw.length === 0) return null;
    const normalized = ActivationUtils.normalizeVector(raw, length, true);
    PARAM_CACHE.set(cacheKey, normalized);
    return normalized;
}

export function getLayerNormParamMeta() {
    if (!rawParams || typeof rawParams !== 'object') return {};
    return rawParams.meta || {};
}
