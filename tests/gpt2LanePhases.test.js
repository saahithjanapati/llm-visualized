import { describe, expect, it } from 'vitest';
import {
    HORIZ_PHASE,
    LEGACY_HORIZ_PHASE,
    LN2_PHASE,
    isAllowedHorizPhaseTransition,
    isAllowedLanePhaseTransition,
    isAllowedLn2PhaseTransition,
    isLn2PrimedPhase,
    primeLaneForLn2Fallback,
    shouldAdvanceHorizPhaseToLn2,
} from '../src/engine/layers/gpt2LanePhases.js';

describe('gpt2LanePhases transition guards', () => {
    it('allows the expected horiz-phase forward transitions', () => {
        expect(isAllowedHorizPhaseTransition(HORIZ_PHASE.WAITING, HORIZ_PHASE.RIGHT)).toBe(true);
        expect(isAllowedHorizPhaseTransition(HORIZ_PHASE.RIGHT, HORIZ_PHASE.INSIDE_LN)).toBe(true);
        expect(isAllowedHorizPhaseTransition(HORIZ_PHASE.INSIDE_LN, HORIZ_PHASE.RISE_ABOVE_LN)).toBe(true);
        expect(isAllowedHorizPhaseTransition(HORIZ_PHASE.RISE_ABOVE_LN, HORIZ_PHASE.READY_MHSA)).toBe(true);
        expect(isAllowedHorizPhaseTransition(HORIZ_PHASE.READY_MHSA, HORIZ_PHASE.TRAVEL_MHSA)).toBe(true);
        expect(isAllowedHorizPhaseTransition(HORIZ_PHASE.TRAVEL_MHSA, HORIZ_PHASE.POST_MHSA_ADDITION)).toBe(true);
        expect(isAllowedHorizPhaseTransition(HORIZ_PHASE.POST_MHSA_ADDITION, HORIZ_PHASE.WAITING_FOR_LN2)).toBe(true);
    });

    it('rejects invalid horiz-phase jumps', () => {
        expect(isAllowedHorizPhaseTransition(HORIZ_PHASE.WAITING, HORIZ_PHASE.TRAVEL_MHSA)).toBe(false);
        expect(isAllowedHorizPhaseTransition(HORIZ_PHASE.RIGHT, HORIZ_PHASE.WAITING_FOR_LN2)).toBe(false);
        expect(isAllowedLanePhaseTransition('horizPhase', HORIZ_PHASE.WAITING, HORIZ_PHASE.POST_MHSA_ADDITION)).toBe(false);
    });

    it('allows the expected ln2-phase forward transitions', () => {
        expect(isAllowedLn2PhaseTransition(LN2_PHASE.NOT_STARTED, LN2_PHASE.PRE_RISE)).toBe(true);
        expect(isAllowedLn2PhaseTransition(LN2_PHASE.PRE_RISE, LN2_PHASE.RIGHT)).toBe(true);
        expect(isAllowedLn2PhaseTransition(LN2_PHASE.RIGHT, LN2_PHASE.INSIDE_LN)).toBe(true);
        expect(isAllowedLn2PhaseTransition(LN2_PHASE.INSIDE_LN, LN2_PHASE.MLP_READY)).toBe(true);
        expect(isAllowedLn2PhaseTransition(LN2_PHASE.MLP_READY, LN2_PHASE.DONE)).toBe(true);
    });

    it('rejects invalid ln2-phase jumps', () => {
        expect(isAllowedLn2PhaseTransition(LN2_PHASE.NOT_STARTED, LN2_PHASE.INSIDE_LN)).toBe(false);
        expect(isAllowedLn2PhaseTransition(LN2_PHASE.RIGHT, LN2_PHASE.DONE)).toBe(false);
        expect(isAllowedLanePhaseTransition('ln2Phase', LN2_PHASE.PRE_RISE, LN2_PHASE.DONE)).toBe(false);
    });
});

describe('gpt2LanePhases fallback priming', () => {
    it('identifies ln2 primed phases', () => {
        expect(isLn2PrimedPhase(LN2_PHASE.PRE_RISE)).toBe(true);
        expect(isLn2PrimedPhase(LN2_PHASE.RIGHT)).toBe(true);
        expect(isLn2PrimedPhase(LN2_PHASE.INSIDE_LN)).toBe(true);
        expect(isLn2PrimedPhase(LN2_PHASE.MLP_READY)).toBe(true);
        expect(isLn2PrimedPhase(LN2_PHASE.DONE)).toBe(true);
        expect(isLn2PrimedPhase(LN2_PHASE.NOT_STARTED)).toBe(false);
    });

    it('marks only expected horiz phases as requiring ln2 advancement', () => {
        expect(shouldAdvanceHorizPhaseToLn2(HORIZ_PHASE.READY_MHSA)).toBe(true);
        expect(shouldAdvanceHorizPhaseToLn2(HORIZ_PHASE.TRAVEL_MHSA)).toBe(true);
        expect(shouldAdvanceHorizPhaseToLn2(HORIZ_PHASE.POST_MHSA_ADDITION)).toBe(true);
        expect(shouldAdvanceHorizPhaseToLn2(LEGACY_HORIZ_PHASE.FINISHED_HEADS)).toBe(true);
        expect(shouldAdvanceHorizPhaseToLn2(HORIZ_PHASE.WAITING)).toBe(false);
    });

    it('primes fallback lane state for ln2 handoff', () => {
        const fallbackVec = { group: { position: { y: 10 } } };
        const lane = {
            postAdditionVec: fallbackVec,
            originalVec: null,
            resultVec: null,
            travellingVec: null,
            ln2Phase: LN2_PHASE.NOT_STARTED,
            horizPhase: HORIZ_PHASE.TRAVEL_MHSA,
            stopRise: true,
            stopRiseTarget: 123,
            additionTargetData: [1, 2, 3],
        };

        const mutated = primeLaneForLn2Fallback(lane);

        expect(mutated).toBe(true);
        expect(lane.originalVec).toBe(fallbackVec);
        expect(lane.postAdditionVec).toBe(fallbackVec);
        expect(lane.ln2Phase).toBe(LN2_PHASE.PRE_RISE);
        expect(lane.horizPhase).toBe(HORIZ_PHASE.WAITING_FOR_LN2);
        expect('stopRise' in lane).toBe(false);
        expect('stopRiseTarget' in lane).toBe(false);
        expect('additionTargetData' in lane).toBe(false);
    });

    it('returns false for already completed lanes', () => {
        const lane = {
            ln2Phase: LN2_PHASE.DONE,
            horizPhase: HORIZ_PHASE.WAITING_FOR_LN2,
        };
        expect(primeLaneForLn2Fallback(lane)).toBe(false);
        expect(lane.ln2Phase).toBe(LN2_PHASE.DONE);
    });
});

