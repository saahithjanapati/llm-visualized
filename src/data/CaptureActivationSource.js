import { VECTOR_LENGTH_PRISM } from '../utils/constants.js';

function cleanArray(values) {
    return values.map((val) => (Number.isFinite(val) ? val : 0));
}

function downsample(values, targetLength) {
    const ratio = values.length / targetLength;
    const result = new Array(targetLength).fill(0);
    for (let i = 0; i < targetLength; i++) {
        const start = i * ratio;
        const end = (i + 1) * ratio;
        let sum = 0;
        let weight = 0;
        let idx = Math.floor(start);
        while (idx < end && idx < values.length) {
            const left = Math.max(start, idx);
            const right = Math.min(end, idx + 1);
            const w = Math.max(0, right - left);
            sum += values[idx] * w;
            weight += w;
            idx += 1;
        }
        result[i] = weight ? sum / weight : 0;
    }
    return result;
}

function upsample(values, targetLength) {
    if (values.length === 1) return new Array(targetLength).fill(values[0]);
    const result = new Array(targetLength).fill(0);
    for (let i = 0; i < targetLength; i++) {
        const t = targetLength === 1 ? 0 : i / (targetLength - 1);
        const idx = t * (values.length - 1);
        const lo = Math.floor(idx);
        const hi = Math.min(values.length - 1, lo + 1);
        const frac = idx - lo;
        result[i] = values[lo] * (1 - frac) + values[hi] * frac;
    }
    return result;
}

function normalizeVector(values, targetLength = VECTOR_LENGTH_PRISM, assumeClean = false) {
    const cleaned = assumeClean ? values : cleanArray(values);
    const length = Math.max(1, Math.floor(targetLength || VECTOR_LENGTH_PRISM));
    if (cleaned.length === length) return cleaned;
    if (cleaned.length === 0) return new Array(length).fill(0);
    if (cleaned.length > length) {
        return downsample(cleaned, length);
    }
    return upsample(cleaned, length);
}

function decodeVector(entry) {
    if (!entry) return [];
    if (Array.isArray(entry)) return cleanArray(entry);
    if (entry && typeof entry === 'object' && Array.isArray(entry.v)) {
        const scale = typeof entry.s === 'number' ? entry.s : 1;
        return entry.v.map((val) => {
            const num = Number(val);
            if (!Number.isFinite(num)) return 0;
            return num * scale;
        });
    }
    return [];
}

function clampIndex(value, max) {
    if (!Number.isFinite(value)) return 0;
    return Math.max(0, Math.min(max, Math.floor(value)));
}

function buildCacheKey(parts) {
    return parts.map((part) => String(part ?? '')).join('|');
}

function getVectorLengthFromEntry(entry) {
    if (!entry) return 0;
    if (Array.isArray(entry)) return entry.length;
    if (entry && typeof entry === 'object') {
        if (Array.isArray(entry.v)) return entry.v.length;
        if (Array.isArray(entry.values)) return entry.values.length;
    }
    return 0;
}

export class CaptureActivationSource {
    constructor(data = {}) {
        this.data = data || {};
        this.meta = this.data.meta || {};
        this.activations = this.data.activations || {};
        this.embeddings = this.activations.embeddings || {};
        this.layers = Array.isArray(this.activations.layers) ? this.activations.layers : [];
        this.tokenStrings = Array.isArray(this.meta.token_strings) ? this.meta.token_strings : [];
        this._laneTokenCache = new Map();
        this._vectorCache = new Map();
        this._attentionRowCache = new Map();
        this._qkvScalarCache = new Map();
        this._baseVectorLength = null;
    }

    _getCached(cache, key, compute) {
        if (cache.has(key)) return cache.get(key);
        const value = compute();
        cache.set(key, value);
        return value;
    }

    static async load(url) {
        const response = await fetch(url);
        if (!response.ok) {
            throw new Error(`Failed to load capture data from ${url} (${response.status})`);
        }
        const data = await response.json();
        return new CaptureActivationSource(data);
    }

    getTokenCount() {
        if (this.tokenStrings.length) return this.tokenStrings.length;
        const embedTokens = Array.isArray(this.embeddings.token) ? this.embeddings.token.length : 0;
        if (embedTokens) return embedTokens;
        return Array.isArray(this.layers[0]?.incoming) ? this.layers[0].incoming.length : 0;
    }

    getTokenString(tokenIndex) {
        if (!this.tokenStrings.length) return null;
        const idx = clampIndex(tokenIndex, this.tokenStrings.length - 1);
        return this.tokenStrings[idx] ?? null;
    }

    getBaseVectorLength() {
        if (Number.isFinite(this._baseVectorLength)) return this._baseVectorLength;
        const candidates = [
            this.embeddings?.token,
            this.embeddings?.position,
            this.layers[0]?.incoming,
            this.layers[0]?.ln1?.norm,
            this.layers[0]?.ln2?.norm,
            this.layers[0]?.post_attn_residual,
            this.layers[0]?.post_mlp_residual,
        ];
        for (const list of candidates) {
            if (!Array.isArray(list) || !list.length) continue;
            const length = getVectorLengthFromEntry(list[0]);
            if (length > 0) {
                this._baseVectorLength = length;
                return length;
            }
        }
        this._baseVectorLength = VECTOR_LENGTH_PRISM;
        return this._baseVectorLength;
    }

    getLaneTokenIndices(laneCount) {
        const count = Math.max(1, laneCount || 1);
        if (this._laneTokenCache.has(count)) return this._laneTokenCache.get(count);
        const tokenCount = this.getTokenCount();
        const indices = [];
        if (tokenCount <= 0) {
            for (let i = 0; i < count; i++) indices.push(i);
            this._laneTokenCache.set(count, indices);
            return indices;
        }
        const limit = Math.min(count, tokenCount);
        for (let i = 0; i < limit; i++) indices.push(i);
        while (indices.length < count) indices.push(tokenCount - 1);
        this._laneTokenCache.set(count, indices);
        return indices;
    }

    getLaneTokenIndex(laneIndex, laneCount) {
        const indices = this.getLaneTokenIndices(laneCount);
        const idx = clampIndex(laneIndex, indices.length - 1);
        return indices[idx] ?? idx;
    }

    getEmbedding(kind, tokenIndex, targetLength = VECTOR_LENGTH_PRISM) {
        const list = this.embeddings?.[kind];
        if (!Array.isArray(list) || !list.length) return null;
        const idx = clampIndex(tokenIndex, list.length - 1);
        const cacheKey = buildCacheKey(['embedding', kind, idx, targetLength]);
        return this._getCached(this._vectorCache, cacheKey, () => {
            const decoded = decodeVector(list[idx]);
            return normalizeVector(decoded, targetLength, true);
        });
    }

    getLayerIncoming(layerIndex, tokenIndex, targetLength = VECTOR_LENGTH_PRISM) {
        const layer = this.layers[layerIndex];
        if (!layer || !Array.isArray(layer.incoming)) return null;
        const idx = clampIndex(tokenIndex, layer.incoming.length - 1);
        const cacheKey = buildCacheKey(['incoming', layerIndex, idx, targetLength]);
        return this._getCached(this._vectorCache, cacheKey, () => {
            const decoded = decodeVector(layer.incoming[idx]);
            return normalizeVector(decoded, targetLength, true);
        });
    }

    getLayerLn1(layerIndex, stage, tokenIndex, targetLength = VECTOR_LENGTH_PRISM) {
        const layer = this.layers[layerIndex];
        const ln1 = layer && layer.ln1 ? layer.ln1 : null;
        const arr = ln1 && Array.isArray(ln1[stage]) ? ln1[stage] : null;
        if (!arr || !arr.length) return null;
        const idx = clampIndex(tokenIndex, arr.length - 1);
        const cacheKey = buildCacheKey(['ln1', stage, layerIndex, idx, targetLength]);
        return this._getCached(this._vectorCache, cacheKey, () => (
            normalizeVector(decodeVector(arr[idx]), targetLength, true)
        ));
    }

    getLayerLn2(layerIndex, stage, tokenIndex, targetLength = VECTOR_LENGTH_PRISM) {
        const layer = this.layers[layerIndex];
        const ln2 = layer && layer.ln2 ? layer.ln2 : null;
        const arr = ln2 && Array.isArray(ln2[stage]) ? ln2[stage] : null;
        if (!arr || !arr.length) return null;
        const idx = clampIndex(tokenIndex, arr.length - 1);
        const cacheKey = buildCacheKey(['ln2', stage, layerIndex, idx, targetLength]);
        return this._getCached(this._vectorCache, cacheKey, () => (
            normalizeVector(decodeVector(arr[idx]), targetLength, true)
        ));
    }

    getLayerQKVScalar(layerIndex, kind, headIndex, tokenIndex) {
        const layer = this.layers[layerIndex];
        const qkv = layer && layer.qkv ? layer.qkv : null;
        const heads = qkv && Array.isArray(qkv[kind]) ? qkv[kind] : null;
        if (!heads || !heads.length) return null;
        const hIdx = clampIndex(headIndex, heads.length - 1);
        const tokens = Array.isArray(heads[hIdx]) ? heads[hIdx] : null;
        if (!tokens || !tokens.length) return null;
        const tIdx = clampIndex(tokenIndex, tokens.length - 1);
        const cacheKey = buildCacheKey(['qkv', kind, layerIndex, hIdx, tIdx]);
        return this._getCached(this._qkvScalarCache, cacheKey, () => {
            const decoded = decodeVector(tokens[tIdx]);
            return decoded.length ? decoded[0] : null;
        });
    }

    getAttentionScoresRow(layerIndex, mode, headIndex, tokenIndex) {
        const layer = this.layers[layerIndex];
        const scores = layer && layer.attention_scores ? layer.attention_scores : null;
        const heads = scores && Array.isArray(scores[mode]) ? scores[mode] : null;
        if (!heads || !heads.length) return null;
        const hIdx = clampIndex(headIndex, heads.length - 1);
        const tokens = Array.isArray(heads[hIdx]) ? heads[hIdx] : null;
        if (!tokens || !tokens.length) return null;
        const tIdx = clampIndex(tokenIndex, tokens.length - 1);
        const cacheKey = buildCacheKey(['attn', mode, layerIndex, hIdx, tIdx]);
        return this._getCached(this._attentionRowCache, cacheKey, () => decodeVector(tokens[tIdx]));
    }

    getAttentionScore(layerIndex, mode, headIndex, queryTokenIndex, keyTokenIndex) {
        const row = this.getAttentionScoresRow(layerIndex, mode, headIndex, queryTokenIndex);
        if (!row || !row.length) return null;
        const idx = clampIndex(keyTokenIndex, row.length - 1);
        return row[idx] ?? null;
    }

    getAttentionOutputProjection(layerIndex, tokenIndex, targetLength = VECTOR_LENGTH_PRISM) {
        const layer = this.layers[layerIndex];
        const arr = layer && Array.isArray(layer.attn_output_proj) ? layer.attn_output_proj : null;
        if (!arr || !arr.length) return null;
        const idx = clampIndex(tokenIndex, arr.length - 1);
        const cacheKey = buildCacheKey(['attn_output_proj', layerIndex, idx, targetLength]);
        return this._getCached(this._vectorCache, cacheKey, () => (
            normalizeVector(decodeVector(arr[idx]), targetLength, true)
        ));
    }

    getPostAttentionResidual(layerIndex, tokenIndex, targetLength = VECTOR_LENGTH_PRISM) {
        const layer = this.layers[layerIndex];
        const arr = layer && Array.isArray(layer.post_attn_residual) ? layer.post_attn_residual : null;
        if (!arr || !arr.length) return null;
        const idx = clampIndex(tokenIndex, arr.length - 1);
        const cacheKey = buildCacheKey(['post_attn_residual', layerIndex, idx, targetLength]);
        return this._getCached(this._vectorCache, cacheKey, () => (
            normalizeVector(decodeVector(arr[idx]), targetLength, true)
        ));
    }

    getMlpUp(layerIndex, tokenIndex, targetLength) {
        const layer = this.layers[layerIndex];
        const arr = layer && Array.isArray(layer.mlp_up) ? layer.mlp_up : null;
        if (!arr || !arr.length) return null;
        const idx = clampIndex(tokenIndex, arr.length - 1);
        const cacheKey = buildCacheKey(['mlp_up', layerIndex, idx, targetLength]);
        return this._getCached(this._vectorCache, cacheKey, () => (
            normalizeVector(decodeVector(arr[idx]), targetLength, true)
        ));
    }

    getMlpActivation(layerIndex, tokenIndex, targetLength) {
        const layer = this.layers[layerIndex];
        const arr = layer && Array.isArray(layer.mlp_act) ? layer.mlp_act : null;
        if (!arr || !arr.length) return null;
        const idx = clampIndex(tokenIndex, arr.length - 1);
        const cacheKey = buildCacheKey(['mlp_act', layerIndex, idx, targetLength]);
        return this._getCached(this._vectorCache, cacheKey, () => (
            normalizeVector(decodeVector(arr[idx]), targetLength, true)
        ));
    }

    getMlpDown(layerIndex, tokenIndex, targetLength = VECTOR_LENGTH_PRISM) {
        const layer = this.layers[layerIndex];
        const arr = layer && Array.isArray(layer.mlp_down) ? layer.mlp_down : null;
        if (!arr || !arr.length) return null;
        const idx = clampIndex(tokenIndex, arr.length - 1);
        const cacheKey = buildCacheKey(['mlp_down', layerIndex, idx, targetLength]);
        return this._getCached(this._vectorCache, cacheKey, () => (
            normalizeVector(decodeVector(arr[idx]), targetLength, true)
        ));
    }

    getPostMlpResidual(layerIndex, tokenIndex, targetLength = VECTOR_LENGTH_PRISM) {
        const layer = this.layers[layerIndex];
        const arr = layer && Array.isArray(layer.post_mlp_residual) ? layer.post_mlp_residual : null;
        if (!arr || !arr.length) return null;
        const idx = clampIndex(tokenIndex, arr.length - 1);
        const cacheKey = buildCacheKey(['post_mlp_residual', layerIndex, idx, targetLength]);
        return this._getCached(this._vectorCache, cacheKey, () => (
            normalizeVector(decodeVector(arr[idx]), targetLength)
        ));
    }
}

export const ActivationUtils = {
    decodeVector,
    normalizeVector,
};
