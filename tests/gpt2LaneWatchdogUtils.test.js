import { describe, it, expect } from 'vitest';
import {
    buildDebugVectorSum,
    getLaneProgressSignature,
    toDebugArray
} from '../src/engine/layers/gpt2LaneWatchdogUtils.js';
import { LN2_PHASE } from '../src/engine/layers/gpt2LanePhases.js';

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
        normStarted: false,
        normApplied: false,
        multStarted: false,
        ln1AddStarted: true,
        ln1AddComplete: false,
        normStartedLN2: false,
        normAppliedLN2: false,
        multDoneLN2: false,
        ln2AddStarted: false,
        ln2AddComplete: false,
        mlpUpStarted: false,
        mlpDownStarted: false,
        mlpDownComplete: false,
        mlpReturnStarted: false,
        ln1ShiftProgress: 0.2,
        mhsaResidualAddProgress: 0.4,
        ln2ShiftProgress: 0.0,
        originalVec: makeVec(1, 2, 3),
        dupVec: makeVec(2, 3, 4),
        resultVec: makeVec(3, 4, 5),
        travellingVec: makeVec(4, 5, 6),
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

    it('tracks pre-attention branch vectors before LN2 starts', () => {
        const lane = makeLane();
        lane.ln2Phase = LN2_PHASE.NOT_STARTED;

        const before = getLaneProgressSignature(lane);
        lane.dupVec.group.position.x += 1;
        const afterDupMove = getLaneProgressSignature(lane);
        expect(afterDupMove).not.toBe(before);

        lane.dupVec.group.position.x -= 1;
        lane.travellingVec.group.position.y += 1;
        const afterTravelMove = getLaneProgressSignature(lane);
        expect(afterTravelMove).not.toBe(before);
    });

    it('ignores residual-stream motion once the LN2 branch is active', () => {
        const lane = makeLane();
        lane.ln2Phase = LN2_PHASE.RIGHT;

        const before = getLaneProgressSignature(lane);
        lane.originalVec.group.position.y += 25;
        lane.postAdditionVec.group.position.y += 25;
        const afterResidualRise = getLaneProgressSignature(lane);
        expect(afterResidualRise).toBe(before);

        lane.movingVecLN2.group.position.x += 1;
        const afterBranchMove = getLaneProgressSignature(lane);
        expect(afterBranchMove).not.toBe(before);
    });

    it('still tracks residual motion during LN2 pre-rise staging', () => {
        const lane = makeLane();
        lane.ln2Phase = LN2_PHASE.PRE_RISE;

        const before = getLaneProgressSignature(lane);
        lane.postAdditionVec.group.position.y += 1;
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
