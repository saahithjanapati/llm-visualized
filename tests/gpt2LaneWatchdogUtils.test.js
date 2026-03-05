import { describe, it, expect } from 'vitest';
import {
    buildDebugVectorSum,
    getLaneProgressSignature,
    toDebugArray
} from '../src/engine/layers/gpt2LaneWatchdogUtils.js';

function makeVec(x, y, z) {
    return {
        group: {
            position: { x, y, z }
        }
    };
}

function makeLane() {
    return {
        horizPhase: 'to_ln1',
        ln2Phase: 'waiting',
        stopRise: false,
        ln1AddStarted: true,
        ln1AddComplete: false,
        ln2AddStarted: false,
        ln2AddComplete: false,
        mlpUpStarted: false,
        mlpDownStarted: false,
        mlpDownComplete: false,
        ln1ShiftProgress: 0.2,
        mhsaResidualAddProgress: 0.4,
        ln2ShiftProgress: 0.0,
        originalVec: makeVec(1, 2, 3),
        postAdditionVec: makeVec(4, 5, 6),
        movingVecLN2: makeVec(7, 8, 9),
        resultVecLN2: makeVec(10, 11, 12),
    };
}

describe('gpt2LaneWatchdogUtils', () => {
    it('returns deterministic lane signatures for identical lane state', () => {
        const laneA = makeLane();
        const laneB = makeLane();
        expect(getLaneProgressSignature(laneA)).toBe(getLaneProgressSignature(laneB));
    });

    it('changes lane signature when lane state changes', () => {
        const lane = makeLane();
        const before = getLaneProgressSignature(lane);
        lane.originalVec.group.position.x += 1;
        const after = getLaneProgressSignature(lane);
        expect(after).not.toBe(before);
    });

    it('normalizes array-like values for debug dumps', () => {
        expect(toDebugArray([1, 2, 3])).toEqual([1, 2, 3]);
        expect(toDebugArray(new Float32Array([4, 5]))).toEqual([4, 5]);
        expect(toDebugArray('x')).toBeNull();
    });

    it('builds debug vector sum with fallback length', () => {
        expect(buildDebugVectorSum([1, 2], [3, 4])).toEqual([4, 6]);
        expect(buildDebugVectorSum([1], null, 3)).toEqual([1, 0, 0]);
        expect(buildDebugVectorSum(null, null, 0)).toBeNull();
    });
});
