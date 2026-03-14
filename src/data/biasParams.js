import rawParams from './gpt2_bias_params.json';

function clampIndex(value, max) {
    if (!Number.isFinite(value)) return 0;
    return Math.max(0, Math.min(max, Math.floor(value)));
}

function getAttentionBiasSamples(layerIndex, kind = 'query') {
    if (!rawParams || typeof rawParams !== 'object') return null;
    const layers = Array.isArray(rawParams.layers) ? rawParams.layers : null;
    if (!layers || !layers.length) return null;
    const safeLayerIndex = clampIndex(layerIndex, layers.length - 1);
    const layer = layers[safeLayerIndex];
    const attention = layer && typeof layer === 'object' ? layer.attention : null;
    const samples = attention && Array.isArray(attention[kind]) ? attention[kind] : null;
    return samples && samples.length ? samples : null;
}

export function getAttentionBiasHeadSample(layerIndex, kind = 'query', headIndex = 0) {
    const samples = getAttentionBiasSamples(layerIndex, kind);
    if (!samples || !samples.length) return null;
    const safeHeadIndex = clampIndex(headIndex, samples.length - 1);
    const value = samples[safeHeadIndex];
    return Number.isFinite(value) ? value : null;
}

export function getMlpBiasVectorSample(layerIndex, kind = 'up') {
    if (!rawParams || typeof rawParams !== 'object') return null;
    const layers = Array.isArray(rawParams.layers) ? rawParams.layers : null;
    if (!layers || !layers.length) return null;
    const safeLayerIndex = clampIndex(layerIndex, layers.length - 1);
    const layer = layers[safeLayerIndex];
    const mlp = layer && typeof layer === 'object' ? layer.mlp : null;
    const samples = mlp && Array.isArray(mlp[kind]) ? mlp[kind] : null;
    return samples && samples.length ? [...samples] : null;
}

export function getBiasParamsMeta() {
    if (!rawParams || typeof rawParams !== 'object') return {};
    return rawParams.meta || {};
}
