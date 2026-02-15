const HORIZ_PHASE_VALUES = {
    WAITING: 'waiting',
    RIGHT: 'right',
    INSIDE_LN: 'insideLN',
    RISE_ABOVE_LN: 'riseAboveLN',
    READY_MHSA: 'readyMHSA',
    TRAVEL_MHSA: 'travelMHSA',
    POST_MHSA_ADDITION: 'postMHSAAddition',
    WAITING_FOR_LN2: 'waitingForLN2'
};

const LEGACY_HORIZ_PHASE_VALUES = {
    FINISHED_HEADS: 'finishedHeads',
};

const LN2_PHASE_VALUES = {
    NOT_STARTED: 'notStarted',
    PRE_RISE: 'preRise',
    RIGHT: 'right',
    INSIDE_LN: 'insideLN',
    MLP_READY: 'mlpReady',
    DONE: 'done'
};

export const HORIZ_PHASE = Object.freeze(HORIZ_PHASE_VALUES);
export const LEGACY_HORIZ_PHASE = Object.freeze(LEGACY_HORIZ_PHASE_VALUES);
export const LN2_PHASE = Object.freeze(LN2_PHASE_VALUES);

const HORIZ_TRANSITIONS = Object.freeze({
    [HORIZ_PHASE.WAITING]: Object.freeze(new Set([HORIZ_PHASE.RIGHT])),
    [HORIZ_PHASE.RIGHT]: Object.freeze(new Set([HORIZ_PHASE.INSIDE_LN])),
    [HORIZ_PHASE.INSIDE_LN]: Object.freeze(new Set([HORIZ_PHASE.RISE_ABOVE_LN])),
    [HORIZ_PHASE.RISE_ABOVE_LN]: Object.freeze(new Set([HORIZ_PHASE.READY_MHSA])),
    [HORIZ_PHASE.READY_MHSA]: Object.freeze(new Set([HORIZ_PHASE.TRAVEL_MHSA, HORIZ_PHASE.WAITING_FOR_LN2])),
    [HORIZ_PHASE.TRAVEL_MHSA]: Object.freeze(new Set([HORIZ_PHASE.POST_MHSA_ADDITION, HORIZ_PHASE.WAITING_FOR_LN2])),
    [HORIZ_PHASE.POST_MHSA_ADDITION]: Object.freeze(new Set([HORIZ_PHASE.WAITING_FOR_LN2])),
    [HORIZ_PHASE.WAITING_FOR_LN2]: Object.freeze(new Set()),
});

const LN2_TRANSITIONS = Object.freeze({
    [LN2_PHASE.NOT_STARTED]: Object.freeze(new Set([LN2_PHASE.PRE_RISE])),
    [LN2_PHASE.PRE_RISE]: Object.freeze(new Set([LN2_PHASE.RIGHT, LN2_PHASE.MLP_READY])),
    [LN2_PHASE.RIGHT]: Object.freeze(new Set([LN2_PHASE.INSIDE_LN])),
    [LN2_PHASE.INSIDE_LN]: Object.freeze(new Set([LN2_PHASE.MLP_READY])),
    [LN2_PHASE.MLP_READY]: Object.freeze(new Set([LN2_PHASE.DONE])),
    [LN2_PHASE.DONE]: Object.freeze(new Set()),
});

const LN2_PRIMED_PHASES = Object.freeze(new Set([
    LN2_PHASE.PRE_RISE,
    LN2_PHASE.RIGHT,
    LN2_PHASE.INSIDE_LN,
    LN2_PHASE.MLP_READY,
    LN2_PHASE.DONE
]));

const HORIZ_PHASES_ADVANCING_TO_LN2 = Object.freeze(new Set([
    HORIZ_PHASE.READY_MHSA,
    HORIZ_PHASE.TRAVEL_MHSA,
    HORIZ_PHASE.POST_MHSA_ADDITION,
    LEGACY_HORIZ_PHASE.FINISHED_HEADS
]));

export function isLn2PrimedPhase(phase) {
    return LN2_PRIMED_PHASES.has(phase);
}

export function shouldAdvanceHorizPhaseToLn2(phase) {
    return HORIZ_PHASES_ADVANCING_TO_LN2.has(phase);
}

export function isAllowedHorizPhaseTransition(fromPhase, toPhase) {
    if (fromPhase === toPhase) return true;
    const allowed = HORIZ_TRANSITIONS[fromPhase];
    if (!allowed) return true;
    return allowed.has(toPhase);
}

export function isAllowedLn2PhaseTransition(fromPhase, toPhase) {
    if (fromPhase === toPhase) return true;
    const allowed = LN2_TRANSITIONS[fromPhase];
    if (!allowed) return true;
    return allowed.has(toPhase);
}

export function isAllowedLanePhaseTransition(key, fromPhase, toPhase) {
    if (key === 'horizPhase') return isAllowedHorizPhaseTransition(fromPhase, toPhase);
    if (key === 'ln2Phase') return isAllowedLn2PhaseTransition(fromPhase, toPhase);
    return true;
}

export function primeLaneForLn2Fallback(lane) {
    if (!lane || lane.ln2Phase === LN2_PHASE.DONE) return false;
    const fallbackVec = (lane.postAdditionVec && lane.postAdditionVec.group)
        ? lane.postAdditionVec
        : (lane.originalVec && lane.originalVec.group)
            ? lane.originalVec
            : (lane.resultVec && lane.resultVec.group)
                ? lane.resultVec
                : (lane.travellingVec && lane.travellingVec.group)
                    ? lane.travellingVec
                    : null;
    if (!fallbackVec || !fallbackVec.group) return false;

    let mutated = false;
    if (!lane.originalVec) {
        lane.originalVec = fallbackVec;
        mutated = true;
    }
    if (lane.postAdditionVec !== fallbackVec) {
        lane.postAdditionVec = fallbackVec;
        mutated = true;
    }
    if (!isLn2PrimedPhase(lane.ln2Phase)) {
        lane.ln2Phase = LN2_PHASE.PRE_RISE;
        mutated = true;
    }
    if (shouldAdvanceHorizPhaseToLn2(lane.horizPhase)) {
        lane.horizPhase = HORIZ_PHASE.WAITING_FOR_LN2;
        mutated = true;
    }
    if (lane.stopRise || lane.stopRiseTarget) {
        delete lane.stopRise;
        delete lane.stopRiseTarget;
        mutated = true;
    }
    if (lane.additionTargetData) {
        delete lane.additionTargetData;
        mutated = true;
    }
    return mutated;
}

export function getLanePhaseTransitionMaps() {
    return {
        horiz: HORIZ_TRANSITIONS,
        ln2: LN2_TRANSITIONS,
    };
}

