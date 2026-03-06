import { LN2_PHASE } from './gpt2LanePhases.js';

function quantizeLaneCoord(value, scale = 100) {
    if (!Number.isFinite(value)) return 0x7fffffff;
    return Math.round(value * scale) | 0;
}

function mixLaneHash(hash, value) {
    const normalized = Number.isFinite(value) ? (value | 0) : 0;
    return Math.imul((hash ^ (normalized >>> 0)) >>> 0, 16777619) >>> 0;
}

function mixLanePhase(hash, phase) {
    const phaseText = (typeof phase === 'string') ? phase : '';
    let nextHash = hash;
    for (let i = 0; i < phaseText.length; i++) {
        nextHash = mixLaneHash(nextHash, phaseText.charCodeAt(i));
    }
    return mixLaneHash(nextHash, 255);
}

function mixLaneVector(hash, vec) {
    if (!vec || !vec.group || !vec.group.position) {
        return mixLaneHash(hash, 0x9e3779b9);
    }
    const p = vec.group.position;
    let nextHash = mixLaneHash(hash, quantizeLaneCoord(p.x));
    nextHash = mixLaneHash(nextHash, quantizeLaneCoord(p.y));
    nextHash = mixLaneHash(nextHash, quantizeLaneCoord(p.z));
    return nextHash;
}

function mixLaneActiveBranchVectors(hash, lane) {
    if (!lane) return hash;

    switch (lane.ln2Phase) {
        case LN2_PHASE.PRE_RISE:
            return mixLaneVector(hash, lane.postAdditionVec || lane.originalVec);
        case LN2_PHASE.RIGHT: {
            let nextHash = mixLaneVector(hash, lane.movingVecLN2);
            nextHash = mixLaneVector(nextHash, lane.resultVecLN2);
            return nextHash;
        }
        case LN2_PHASE.INSIDE_LN: {
            let nextHash = mixLaneVector(hash, lane.movingVecLN2);
            nextHash = mixLaneVector(nextHash, lane.resultVecLN2);
            return nextHash;
        }
        case LN2_PHASE.MLP_READY: {
            let nextHash = mixLaneVector(hash, lane.resultVecLN2);
            nextHash = mixLaneVector(nextHash, lane.finalVecAfterMlp);
            return nextHash;
        }
        case LN2_PHASE.DONE:
            return mixLaneVector(hash, lane.finalVecAfterMlp);
        case LN2_PHASE.NOT_STARTED:
        default: {
            let nextHash = mixLaneVector(hash, lane.originalVec);
            nextHash = mixLaneVector(nextHash, lane.postAdditionVec);
            nextHash = mixLaneVector(nextHash, lane.movingVecLN2);
            nextHash = mixLaneVector(nextHash, lane.resultVecLN2);
            nextHash = mixLaneVector(nextHash, lane.finalVecAfterMlp);
            return nextHash;
        }
    }
}

export function getLaneProgressSignature(lane) {
    if (!lane) return 0;
    let hash = 2166136261 >>> 0;
    hash = mixLanePhase(hash, lane.horizPhase);
    hash = mixLanePhase(hash, lane.ln2Phase);
    hash = mixLaneHash(hash, lane.stopRise ? 1 : 0);
    hash = mixLaneHash(hash, lane.ln1AddStarted ? 1 : 0);
    hash = mixLaneHash(hash, lane.ln1AddComplete ? 1 : 0);
    hash = mixLaneHash(hash, lane.ln2AddStarted ? 1 : 0);
    hash = mixLaneHash(hash, lane.ln2AddComplete ? 1 : 0);
    hash = mixLaneHash(hash, lane.mlpUpStarted ? 1 : 0);
    hash = mixLaneHash(hash, lane.mlpDownStarted ? 1 : 0);
    hash = mixLaneHash(hash, lane.mlpDownComplete ? 1 : 0);
    hash = mixLaneHash(hash, quantizeLaneCoord(lane.ln1ShiftProgress));
    hash = mixLaneHash(hash, quantizeLaneCoord(lane.mhsaResidualAddProgress));
    hash = mixLaneHash(hash, quantizeLaneCoord(lane.ln2ShiftProgress));
    hash = mixLaneActiveBranchVectors(hash, lane);
    return hash;
}

export function toDebugArray(values) {
    if (Array.isArray(values)) return values.slice();
    if (ArrayBuffer.isView(values)) return Array.from(values);
    return null;
}

export function buildDebugVectorSum(lhs, rhs, fallbackLength = 0) {
    const left = toDebugArray(lhs);
    const right = toDebugArray(rhs);
    const fallback = Number.isFinite(fallbackLength) ? Math.max(0, Math.floor(fallbackLength)) : 0;
    const length = Math.max(
        fallback,
        left ? left.length : 0,
        right ? right.length : 0
    );
    if (length <= 0) return null;
    const sum = new Array(length);
    for (let i = 0; i < length; i++) {
        const l = left && Number.isFinite(left[i]) ? left[i] : 0;
        const r = right && Number.isFinite(right[i]) ? right[i] : 0;
        sum[i] = l + r;
    }
    return sum;
}
