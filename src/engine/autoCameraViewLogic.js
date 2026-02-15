import { HORIZ_PHASE, LEGACY_HORIZ_PHASE, LN2_PHASE } from './layers/gpt2LanePhases.js';

export function resolveAutoCameraViewState({
    pipeline = null,
    layers = [],
    currentLayerIdx = 0,
    priorViewKey = 'default',
    nowMs = 0,
    isLargeDesktopViewport = false,
    isTopLayerNormCameraPhase = () => false,
} = {}) {
    if (!Array.isArray(layers) || !layers.length) {
        return { rawKey: 'default', viewContext: null };
    }

    const layerIndex = Math.min(Math.max(0, currentLayerIdx), layers.length - 1);
    const layer = layers[layerIndex];
    const mhsa = layer?.mhsaAnimation;
    const lanes = Array.isArray(layer?.lanes) ? layer.lanes : [];
    const laneCount = lanes.length;
    const laneIndex = laneCount ? Math.min(laneCount - 1, Math.floor(laneCount / 2)) : -1;
    const lane = laneIndex >= 0 ? lanes[laneIndex] : null;
    const inLaneLn = !!(lane
        && (lane.horizPhase === HORIZ_PHASE.INSIDE_LN || lane.ln2Phase === LN2_PHASE.INSIDE_LN));
    const inTopLn = !!isTopLayerNormCameraPhase(layer, lanes);
    const inLayerHandoff = !!(layerIndex > 0
        && lane
        && lane.horizPhase === HORIZ_PHASE.WAITING
        && lane.ln2Phase === LN2_PHASE.NOT_STARTED);
    const inLn = inLaneLn || inTopLn;
    const holdViewBeforeLn2 = lanes.some((candidate) => candidate
        && (candidate.horizPhase === HORIZ_PHASE.POST_MHSA_ADDITION
            || candidate.horizPhase === HORIZ_PHASE.WAITING_FOR_LN2
            || candidate.ln2Phase === LN2_PHASE.PRE_RISE
            || candidate.ln2Phase === LN2_PHASE.RIGHT));
    const anyResidualAddActive = lanes.some((candidate) => candidate
        && candidate.stopRise
        && candidate.stopRiseTarget);
    const anyResidualAddReleaseHold = lanes.some((candidate) => candidate
        && Number.isFinite(candidate.__cameraHoldAfterAddUntil)
        && candidate.__cameraHoldAfterAddUntil > nowMs);
    const residualAddHoldUntilMs = lanes.reduce((max, candidate) => {
        const untilMs = Number.isFinite(candidate?.__cameraHoldAfterAddUntil)
            ? candidate.__cameraHoldAfterAddUntil
            : 0;
        return untilMs > max ? untilMs : max;
    }, 0);
    const inResidualAdd = anyResidualAddActive || anyResidualAddReleaseHold;
    const holdViewDuringResidualAdd = !!(inResidualAdd
        && priorViewKey !== 'final'
        && priorViewKey !== 'layer-end-desktop');
    const holdViewUntilLn2Inside = !!(holdViewBeforeLn2
        && priorViewKey !== 'ln'
        && priorViewKey !== 'final'
        && priorViewKey !== 'layer-end-desktop');
    const holdViewThroughLayerHandoff = !!(inLayerHandoff
        && priorViewKey !== 'layer-end-desktop'
        && priorViewKey !== 'final');
    const forwardComplete = (typeof pipeline?.isForwardPassComplete === 'function')
        ? pipeline.isForwardPassComplete()
        : false;
    const viewContext = {
        lane,
        laneIndex,
        laneCount,
        inLn,
        inTopLn,
        inLayerHandoff,
        inResidualAdd,
        anyResidualAddActive,
        anyResidualAddReleaseHold,
        residualAddHoldUntilMs,
        holdViewBeforeLn2,
        holdViewUntilLn2Inside,
        holdViewDuringResidualAdd,
        holdViewThroughLayerHandoff,
        priorViewKey,
        forwardComplete
    };
    if (forwardComplete) {
        return { rawKey: 'final', viewContext };
    }

    const passPhase = mhsa?.mhaPassThroughPhase || 'positioning_mha_vectors';
    const inTravel = !!(lane && lane.horizPhase === HORIZ_PHASE.TRAVEL_MHSA
        && passPhase === 'positioning_mha_vectors');
    const inCopyStage = !!(lane
        && lane.horizPhase === LEGACY_HORIZ_PHASE.FINISHED_HEADS
        && (passPhase === 'positioning_mha_vectors' || passPhase === 'ready_for_parallel_pass_through'));
    if (!mhsa) {
        if (inLn) return { rawKey: 'ln', viewContext };
        if (holdViewThroughLayerHandoff) return { rawKey: priorViewKey, viewContext };
        if (inLayerHandoff && isLargeDesktopViewport) return { rawKey: 'layer-end-desktop', viewContext };
        return { rawKey: 'default', viewContext };
    }

    if (inLn) return { rawKey: 'ln', viewContext };
    if (holdViewUntilLn2Inside) return { rawKey: priorViewKey, viewContext };
    if (holdViewDuringResidualAdd) return { rawKey: priorViewKey, viewContext };
    if (holdViewThroughLayerHandoff) return { rawKey: priorViewKey, viewContext };
    if (inLayerHandoff && isLargeDesktopViewport) return { rawKey: 'layer-end-desktop', viewContext };
    if (inTravel || inCopyStage) return { rawKey: 'travel', viewContext };

    const outputPhase = mhsa.outputProjMatrixAnimationPhase || 'waiting';
    const rowPhase = mhsa.rowMergePhase || 'not_started';
    const outputReturnComplete = mhsa.outputProjMatrixReturnComplete === true;
    const concatActive = (rowPhase === 'merging' || rowPhase === 'merged')
        && outputPhase === 'waiting'
        && !outputReturnComplete;
    if (concatActive) return { rawKey: 'concat', viewContext };

    const mhsaGate = rowPhase === 'not_started' && outputPhase === 'waiting';
    const mhsaActive = mhsaGate && (passPhase === 'parallel_pass_through_active'
        || passPhase === 'mha_pass_through_complete'
        || (passPhase === 'ready_for_parallel_pass_through' && !inCopyStage));
    if (mhsaActive) return { rawKey: 'mhsa', viewContext };
    return { rawKey: 'default', viewContext };
}

export function getAutoCameraViewSwitchHoldMs({
    fromKey,
    toKey,
    viewContext = null,
    baseHoldMs = 90
} = {}) {
    if (toKey === fromKey) return 0;
    let holdMs = baseHoldMs;
    if (viewContext && viewContext.holdViewUntilLn2Inside) {
        holdMs = Math.max(holdMs, 200);
    }
    if (viewContext && viewContext.holdViewDuringResidualAdd) {
        holdMs = Math.max(holdMs, 220);
    }
    if (viewContext && viewContext.inLayerHandoff) {
        holdMs = Math.max(holdMs, 120);
    }
    if (viewContext && viewContext.lane
        && (viewContext.lane.stopRise || viewContext.lane.__followStopRiseReleaseFrom)) {
        holdMs = Math.max(holdMs, 110);
    }
    if (toKey === 'ln') {
        holdMs = Math.min(holdMs, 48);
    } else if (toKey === 'final') {
        holdMs = Math.min(holdMs, 20);
    }
    if (fromKey === 'ln' && toKey === 'default') {
        holdMs = Math.max(holdMs, 72);
    }
    if (toKey === 'layer-end-desktop' || fromKey === 'layer-end-desktop') {
        holdMs = Math.max(holdMs, 130);
    }
    return Math.max(0, holdMs);
}

export function resolveStableAutoCameraViewKey({
    rawKey,
    currentKey = 'default',
    pendingKey = 'default',
    pendingSinceMs = 0,
    nowMs = 0,
    baseHoldMs = 90,
    viewContext = null
} = {}) {
    if (rawKey === currentKey) {
        return {
            key: rawKey,
            pendingKey: rawKey,
            pendingSinceMs: 0
        };
    }

    if (pendingKey !== rawKey) {
        return {
            key: currentKey,
            pendingKey: rawKey,
            pendingSinceMs: nowMs
        };
    }

    const pendingSince = Number.isFinite(pendingSinceMs) ? pendingSinceMs : nowMs;
    const elapsedMs = Math.max(0, nowMs - pendingSince);
    const holdMs = getAutoCameraViewSwitchHoldMs({
        fromKey: currentKey,
        toKey: rawKey,
        viewContext,
        baseHoldMs
    });
    if (elapsedMs < holdMs) {
        return {
            key: currentKey,
            pendingKey: rawKey,
            pendingSinceMs: pendingSince
        };
    }

    return {
        key: rawKey,
        pendingKey: rawKey,
        pendingSinceMs: 0
    };
}

