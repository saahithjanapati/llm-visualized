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

function getArrayValueAtIndex(values, index) {
    if (!Array.isArray(values) || !values.length || !Number.isFinite(index)) return null;
    const safeIndex = Math.floor(index);
    if (safeIndex < 0 || safeIndex >= values.length) return null;
    const value = values[safeIndex];
    return Number.isFinite(value) ? value : null;
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

function normalizeTokenIdArray(values) {
    if (!Array.isArray(values) || !values.length) return [];
    return values.map((value) => {
        const num = Number(value);
        return Number.isFinite(num) ? Math.floor(num) : null;
    });
}

function inferTriangularTokenCount(flatLength) {
    const length = Number(flatLength);
    if (!Number.isFinite(length) || length <= 0) return 0;
    const n = Math.floor((Math.sqrt((8 * length) + 1) - 1) / 2);
    return (n * (n + 1)) / 2 === length ? n : 0;
}

function decodeScaledValues(values, scale = 1) {
    if (!Array.isArray(values) || !values.length) return [];
    const safeScale = Number.isFinite(scale) ? scale : 1;
    return values.map((value) => {
        const num = Number(value);
        return Number.isFinite(num) ? num * safeScale : 0;
    });
}

function decodeAttentionRowEntry(entry, rowIndex, tokenCount) {
    if (Array.isArray(entry)) return cleanArray(entry);
    if (!entry || typeof entry !== 'object' || !Array.isArray(entry.v)) return null;
    const lower = decodeScaledValues(entry.v, entry.s);
    const upper = Array.isArray(entry.u) ? decodeScaledValues(entry.u, entry.us) : [];
    const upperLength = Math.max(0, tokenCount - rowIndex - 1);
    if (!upperLength || !upper.length) return lower;
    return lower.concat(upper.slice(0, upperLength));
}

function getStrictUpperRowStart(rowIndex, tokenCount) {
    const safeRowIndex = Number.isFinite(rowIndex) ? Math.max(0, Math.floor(rowIndex)) : 0;
    const safeTokenCount = Number.isFinite(tokenCount) ? Math.max(0, Math.floor(tokenCount)) : 0;
    return (safeRowIndex * ((2 * safeTokenCount) - safeRowIndex - 1)) / 2;
}

function addVectors(a, b) {
    const left = Array.isArray(a) ? a : null;
    const right = Array.isArray(b) ? b : null;
    if (!left && !right) return null;
    if (!left) return right.slice();
    if (!right) return left.slice();
    const length = Math.max(left.length, right.length);
    const out = new Array(length).fill(0);
    for (let i = 0; i < length; i += 1) {
        const lhs = Number.isFinite(left[i]) ? left[i] : 0;
        const rhs = Number.isFinite(right[i]) ? right[i] : 0;
        out[i] = lhs + rhs;
    }
    return out;
}

export class CaptureActivationSource {
    constructor(data = {}) {
        this.data = data || {};
        this.meta = this.data.meta || {};
        this.activations = this.data.activations || {};
        this.embeddings = this.activations.embeddings || {};
        this.layers = Array.isArray(this.activations.layers) ? this.activations.layers : [];
        this.finalLayerNorm = this.activations && this.activations.final_layernorm
            ? this.activations.final_layernorm
            : null;
        this.tokenStrings = Array.isArray(this.meta.token_strings) ? this.meta.token_strings : [];
        this.tokenHfStrings = Array.isArray(this.meta.token_hf_strings) ? this.meta.token_hf_strings : [];
        const explicitTokenIds = normalizeTokenIdArray(this.meta.token_ids);
        const promptTokenIds = normalizeTokenIdArray(this.meta.prompt_tokens);
        const completionTokenIds = normalizeTokenIdArray(this.meta.completion_tokens);
        const combinedTokenIds = (promptTokenIds.length || completionTokenIds.length)
            ? [...promptTokenIds, ...completionTokenIds]
            : [];
        this.tokenIds = explicitTokenIds.length ? explicitTokenIds : combinedTokenIds;
        this.logits = Array.isArray(this.data.logits) ? this.data.logits : [];
        this._laneTokenCache = new Map();
        this._vectorCache = new Map();
        this._attentionRowCache = new Map();
        this._attentionHeadCache = new Map();
        this._qkvScalarCache = new Map();
        this._qkvVectorCache = new Map();
        this._attentionWeightedSumCache = new Map();
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
        if (this.tokenHfStrings.length) return this.tokenHfStrings.length;
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

    getTokenRawString(tokenIndex) {
        if (!this.tokenHfStrings.length) return null;
        const idx = clampIndex(tokenIndex, this.tokenHfStrings.length - 1);
        return this.tokenHfStrings[idx] ?? null;
    }

    getTokenId(tokenIndex) {
        if (!this.tokenIds.length) return null;
        if (!Number.isFinite(tokenIndex)) return null;
        const idx = Math.floor(tokenIndex);
        if (idx < 0 || idx >= this.tokenIds.length) return null;
        const tokenId = this.tokenIds[idx];
        return Number.isFinite(tokenId) ? tokenId : null;
    }

    getLogitTopK() {
        const metaValue = Number(this.meta?.logit_top_k);
        if (Number.isFinite(metaValue) && metaValue > 0) return Math.floor(metaValue);
        if (this.logits.length && Array.isArray(this.logits[0])) return this.logits[0].length;
        return 0;
    }

    getLogitsForToken(tokenIndex, limit = null) {
        if (!this.logits.length) return null;
        const idx = clampIndex(tokenIndex, this.logits.length - 1);
        const row = Array.isArray(this.logits[idx]) ? this.logits[idx] : null;
        if (!row) return null;
        if (limit == null) return row;
        const safeLimit = Math.max(0, Math.floor(limit));
        return row.slice(0, safeLimit);
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
        const safeTokenIndex = Number.isFinite(tokenIndex) ? Math.floor(tokenIndex) : 0;
        if (kind === 'sum') {
            const cacheKey = buildCacheKey(['embedding', kind, safeTokenIndex, targetLength]);
            return this._getCached(this._vectorCache, cacheKey, () => {
                const token = this.getEmbedding('token', tokenIndex, targetLength);
                const position = this.getEmbedding('position', tokenIndex, targetLength);
                const combined = addVectors(token, position);
                return combined ? normalizeVector(combined, targetLength, true) : null;
            });
        }
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

    getFinalLayerNorm(stage, tokenIndex, targetLength = VECTOR_LENGTH_PRISM) {
        const finalLn = this.finalLayerNorm;
        const arr = finalLn && Array.isArray(finalLn[stage]) ? finalLn[stage] : null;
        if (!arr || !arr.length) return null;
        const idx = clampIndex(tokenIndex, arr.length - 1);
        const cacheKey = buildCacheKey(['final_ln', stage, idx, targetLength]);
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

    getLayerQKVVector(layerIndex, kind, headIndex, tokenIndex, targetLength = null) {
        const layer = this.layers[layerIndex];
        const qkv = layer && layer.qkv ? layer.qkv : null;
        const heads = qkv && Array.isArray(qkv[kind]) ? qkv[kind] : null;
        if (!heads || !heads.length) return null;
        const hIdx = clampIndex(headIndex, heads.length - 1);
        const tokens = Array.isArray(heads[hIdx]) ? heads[hIdx] : null;
        if (!tokens || !tokens.length) return null;
        const tIdx = clampIndex(tokenIndex, tokens.length - 1);
        const normalizedLength = Number.isFinite(targetLength) && targetLength > 0
            ? Math.max(1, Math.floor(targetLength))
            : null;
        const cacheKey = buildCacheKey([
            'qkv_vector',
            kind,
            layerIndex,
            hIdx,
            tIdx,
            normalizedLength === null ? 'raw' : normalizedLength
        ]);
        return this._getCached(this._qkvVectorCache, cacheKey, () => {
            const decoded = decodeVector(tokens[tIdx]);
            if (!decoded.length) return null;
            if (normalizedLength === null) return decoded;
            return normalizeVector(decoded, normalizedLength, true);
        });
    }

    getAttentionScoresRow(layerIndex, mode, headIndex, tokenIndex) {
        const layer = this.layers[layerIndex];
        const scores = layer && layer.attention_scores ? layer.attention_scores : null;
        const heads = scores && Array.isArray(scores[mode]) ? scores[mode] : null;
        if (!heads || !heads.length) return null;
        const hIdx = clampIndex(headIndex, heads.length - 1);
        const headEntry = heads[hIdx];
        const tIdx = clampIndex(tokenIndex, Number.MAX_SAFE_INTEGER);
        const cacheKey = buildCacheKey(['attn', mode, layerIndex, hIdx, tIdx]);
        return this._getCached(this._attentionRowCache, cacheKey, () => {
            if (Array.isArray(headEntry)) {
                if (!headEntry.length) return null;
                const rowIdx = clampIndex(tIdx, headEntry.length - 1);
                return decodeAttentionRowEntry(headEntry[rowIdx], rowIdx, headEntry.length);
            }
            if (!headEntry || typeof headEntry !== 'object' || !Array.isArray(headEntry.v)) return null;
            const packedCacheKey = buildCacheKey(['attn_packed', mode, layerIndex, hIdx]);
            const packedHead = this._getCached(this._attentionHeadCache, packedCacheKey, () => {
                const values = cleanArray(headEntry.v);
                const explicitCount = Number(headEntry.n);
                const tokenCount = Number.isFinite(explicitCount) && explicitCount > 0
                    ? Math.floor(explicitCount)
                    : inferTriangularTokenCount(values.length);
                if (!tokenCount) return null;
                const globalScale = Number(headEntry.s);
                const scaledValues = Number.isFinite(globalScale) && globalScale !== 1
                    ? values.map((value) => value * globalScale)
                    : values;
                const rowScales = Array.isArray(headEntry.rs) ? cleanArray(headEntry.rs) : null;
                const upperValues = Array.isArray(headEntry.u) ? cleanArray(headEntry.u) : null;
                const upperRowScales = Array.isArray(headEntry.urs) ? cleanArray(headEntry.urs) : null;
                return {
                    values: scaledValues,
                    tokenCount,
                    rowScales,
                    upperValues,
                    upperRowScales,
                };
            });
            if (!packedHead || !packedHead.values.length) return null;
            const rowIdx = clampIndex(tIdx, packedHead.tokenCount - 1);
            const start = (rowIdx * (rowIdx + 1)) / 2;
            const end = start + rowIdx + 1;
            if (!Number.isFinite(start) || !Number.isFinite(end)) return null;
            if (start < 0 || end > packedHead.values.length) return null;
            const row = packedHead.values.slice(start, end);
            if (packedHead.rowScales && packedHead.rowScales.length) {
                const scaleIdx = clampIndex(rowIdx, packedHead.rowScales.length - 1);
                const rowScale = packedHead.rowScales[scaleIdx];
                if (Number.isFinite(rowScale) && rowScale !== 1) {
                    for (let i = 0; i < row.length; i += 1) row[i] *= rowScale;
                }
            }
            const upperLength = Math.max(0, packedHead.tokenCount - rowIdx - 1);
            if (!upperLength || !packedHead.upperValues?.length) {
                return row;
            }
            const upperStart = getStrictUpperRowStart(rowIdx, packedHead.tokenCount);
            const upperEnd = upperStart + upperLength;
            if (!Number.isFinite(upperStart) || !Number.isFinite(upperEnd)) return row;
            if (upperStart < 0 || upperEnd > packedHead.upperValues.length) return row;
            const upperRow = packedHead.upperValues.slice(upperStart, upperEnd);
            if (packedHead.upperRowScales && packedHead.upperRowScales.length) {
                const upperScaleIdx = clampIndex(rowIdx, packedHead.upperRowScales.length - 1);
                const upperRowScale = packedHead.upperRowScales[upperScaleIdx];
                if (Number.isFinite(upperRowScale) && upperRowScale !== 1) {
                    for (let i = 0; i < upperRow.length; i += 1) upperRow[i] *= upperRowScale;
                }
            }
            return row.concat(upperRow);
        });
    }

    getAttentionScore(layerIndex, mode, headIndex, queryTokenIndex, keyTokenIndex) {
        const row = this.getAttentionScoresRow(layerIndex, mode, headIndex, queryTokenIndex);
        return getArrayValueAtIndex(row, keyTokenIndex);
    }

    getAttentionWeightedSum(layerIndex, headIndex, queryTokenIndex, targetLength = null) {
        const normalizedLength = Number.isFinite(targetLength) && targetLength > 0
            ? Math.max(1, Math.floor(targetLength))
            : null;
        const cacheKey = buildCacheKey([
            'attention_weighted_sum',
            layerIndex,
            headIndex,
            queryTokenIndex,
            normalizedLength === null ? 'raw' : normalizedLength
        ]);
        return this._getCached(this._attentionWeightedSumCache, cacheKey, () => {
            const weightRow = this.getAttentionScoresRow(layerIndex, 'post', headIndex, queryTokenIndex);
            if (!Array.isArray(weightRow) || !weightRow.length) return null;

            let result = null;
            let usedWeight = false;

            for (let keyTokenIndex = 0; keyTokenIndex < weightRow.length; keyTokenIndex += 1) {
                const weight = weightRow[keyTokenIndex];
                if (!Number.isFinite(weight)) continue;
                const valueVector = this.getLayerQKVVector(
                    layerIndex,
                    'v',
                    headIndex,
                    keyTokenIndex,
                    normalizedLength
                );
                if (!Array.isArray(valueVector) || !valueVector.length) continue;
                if (!Array.isArray(result)) {
                    result = new Array(valueVector.length).fill(0);
                }
                usedWeight = true;
                for (let i = 0; i < valueVector.length; i += 1) {
                    const value = Number.isFinite(valueVector[i]) ? valueVector[i] : 0;
                    result[i] += weight * value;
                }
            }

            if (!usedWeight || !Array.isArray(result)) return null;
            return result;
        });
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
        const idx = arr && arr.length ? clampIndex(tokenIndex, arr.length - 1) : clampIndex(tokenIndex, Number.MAX_SAFE_INTEGER);
        const cacheKey = buildCacheKey(['post_attn_residual', layerIndex, idx, targetLength]);
        return this._getCached(this._vectorCache, cacheKey, () => {
            if (arr && arr.length) {
                return normalizeVector(decodeVector(arr[idx]), targetLength, true);
            }
            const incoming = this.getLayerIncoming(layerIndex, tokenIndex, targetLength);
            const attnOutput = this.getAttentionOutputProjection(layerIndex, tokenIndex, targetLength);
            const combined = addVectors(incoming, attnOutput);
            return combined ? normalizeVector(combined, targetLength, true) : null;
        });
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
        const idx = arr && arr.length ? clampIndex(tokenIndex, arr.length - 1) : clampIndex(tokenIndex, Number.MAX_SAFE_INTEGER);
        const cacheKey = buildCacheKey(['post_mlp_residual', layerIndex, idx, targetLength]);
        return this._getCached(this._vectorCache, cacheKey, () => {
            if (arr && arr.length) {
                return normalizeVector(decodeVector(arr[idx]), targetLength);
            }
            const postAttention = this.getPostAttentionResidual(layerIndex, tokenIndex, targetLength);
            const mlpDown = this.getMlpDown(layerIndex, tokenIndex, targetLength);
            const combined = addVectors(postAttention, mlpDown);
            return combined ? normalizeVector(combined, targetLength, true) : null;
        });
    }
}

export const ActivationUtils = {
    decodeVector,
    normalizeVector,
};
