import { VECTOR_LENGTH_PRISM } from '../utils/constants.js';

export function isArrayLike(values) {
    return Array.isArray(values) || ArrayBuffer.isView(values);
}

export function toMutableArray(values, targetLength, fallbackValue = 0) {
    const length = Math.max(1, Math.floor(targetLength || 1));
    if (!isArrayLike(values) || values.length === 0) {
        return new Array(length).fill(fallbackValue);
    }
    const source = Array.isArray(values) ? values : Array.from(values);
    if (source.length === length) return source.slice();
    if (source.length > length) return source.slice(0, length);
    const out = source.slice();
    const padValue = Number.isFinite(source[source.length - 1]) ? source[source.length - 1] : fallbackValue;
    while (out.length < length) out.push(padValue);
    return out;
}

export function recolorVectorFromData(vec, values, colorOptions = null) {
    if (!vec || !isArrayLike(values) || values.length === 0) return;
    vec.rawData = Array.isArray(values) ? values.slice() : Array.from(values);
    const numKeyColors = Math.min(30, Math.max(1, vec.rawData.length || 1));
    vec.updateKeyColorsFromData(vec.rawData, numKeyColors, colorOptions, values);
}

export function buildElementwiseSum(lhs, rhs, targetLength) {
    const length = Math.max(1, Math.floor(targetLength || 1));
    const out = new Array(length);
    for (let i = 0; i < length; i++) {
        const left = isArrayLike(lhs) ? (lhs[i] || 0) : 0;
        const right = isArrayLike(rhs) ? (rhs[i] || 0) : 0;
        out[i] = left + right;
    }
    return out;
}

export function simplePrismMultiply(srcVec, tgtVec, onComplete) {
    const srcCount = srcVec && Number.isFinite(srcVec.instanceCount) ? srcVec.instanceCount : VECTOR_LENGTH_PRISM;
    const tgtCount = tgtVec && Number.isFinite(tgtVec.instanceCount) ? tgtVec.instanceCount : VECTOR_LENGTH_PRISM;
    const srcLen = srcVec && Array.isArray(srcVec.rawData) ? srcVec.rawData.length : srcCount;
    const tgtLen = tgtVec && Array.isArray(tgtVec.rawData) ? tgtVec.rawData.length : tgtCount;
    const length = Math.min(srcCount, tgtCount, srcLen, tgtLen);
    for (let i = 0; i < length; i++) {
        tgtVec.rawData[i] = (srcVec.rawData[i] || 0) * (tgtVec.rawData[i] || 0);
    }
    const numKeyColors = Math.min(30, Math.max(1, tgtVec.rawData.length || 1));
    tgtVec.updateKeyColorsFromData(tgtVec.rawData, numKeyColors);
    if (onComplete) onComplete();
}

