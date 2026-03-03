import { HORIZ_PHASE, LEGACY_HORIZ_PHASE, LN2_PHASE } from './layers/gpt2LanePhases.js';

const EMBED_VIEW_KEY_VOCAB = 'embed-vocab';
const EMBED_VIEW_KEY_POSITION = 'embed-position';
const EMBED_VIEW_KEY_ADD = 'embed-add';

const isEmbedBottomViewKey = (key) => (
    key === EMBED_VIEW_KEY_VOCAB
    || key === EMBED_VIEW_KEY_POSITION
    || key === EMBED_VIEW_KEY_ADD
);

function gateHasStarted(gate, nowMs) {
    if (!gate || gate.enabled === false) return false;
    if (gate.pending) return false;
    const startByToken = gate.startByToken;
    if (startByToken && typeof startByToken === 'object') {
        const starts = Object.values(startByToken).filter((value) => Number.isFinite(value));
        if (starts.length) {
            return nowMs >= Math.min(...starts);
        }
    }
    if (Number.isFinite(gate.defaultStartAt)) {
        return nowMs >= gate.defaultStartAt;
    }
    return false;
}

function gateHasPendingEntries(gate, nowMs) {
    if (!gate || gate.enabled === false) return false;
    if (gate.pending) return true;
    const insideByToken = gate.insideByToken;
    if (insideByToken && typeof insideByToken === 'object') {
        const states = Object.values(insideByToken).filter((value) => typeof value === 'boolean');
        if (states.length) return states.some((value) => value === false);
    }
    return Number.isFinite(gate.defaultReleaseAt) && nowMs < gate.defaultReleaseAt;
}

function resolveBottomEmbeddingViewKey({ pipeline = null, layer = null, lanes = [], nowMs = 0 } = {}) {
    if (!pipeline || !layer || layer.index !== 0 || !Array.isArray(lanes) || !lanes.length) return null;
    const anyPosWorkRemaining = lanes.some((candidate) => candidate && !candidate.posAddComplete);
    if (!anyPosWorkRemaining) return null;

    const anyVectorsInsideVocab = lanes.some((candidate) => {
        const y = candidate?.originalVec?.group?.position?.y;
        const exitY = Number.isFinite(candidate?.vocabEmbeddingExitY) ? candidate.vocabEmbeddingExitY : NaN;
        return Number.isFinite(y) && Number.isFinite(exitY) && y < (exitY - 0.01);
    });
    const anyAwaitingPosPass = lanes.some((candidate) => (
        candidate?.posVec
        && !candidate.__posPassStarted
        && !candidate.posAddStarted
        && !candidate.posAddComplete
    ));

    const anyResidualAddPreview = lanes.some((candidate) => candidate
        && candidate.__posPreAddApproach
        && !candidate.posAddStarted
        && !candidate.posAddComplete);
    const anyResidualAddActive = lanes.some((candidate) => candidate?.posAddStarted && !candidate.posAddComplete);
    if (anyResidualAddPreview) return EMBED_VIEW_KEY_ADD;
    if (anyResidualAddActive) return EMBED_VIEW_KEY_ADD;

    const positionGate = pipeline.__inputPositionChipGate;
    const anyPositionPassActive = lanes.some((candidate) => (
        candidate
        && !candidate.posAddComplete
        && (
            candidate.__posPassStarted
            || candidate.posAddStarted
        )
    ));
    if (anyPositionPassActive || gateHasStarted(positionGate, nowMs)) return EMBED_VIEW_KEY_POSITION;

    const vocabGate = pipeline.__inputVocabChipGate;
    if (
        anyVectorsInsideVocab
        || gateHasPendingEntries(vocabGate, nowMs)
        || anyAwaitingPosPass
        || anyPosWorkRemaining
    ) {
        return EMBED_VIEW_KEY_VOCAB;
    }
    return null;
}

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
        && priorViewKey !== 'layer-end-desktop'
        // Once concat hands off to output projection/residual-add, allow
        // follow mode to move to the default (MLP-style) framing.
        && priorViewKey !== 'concat');
    const holdViewUntilLn2Inside = !!(holdViewBeforeLn2
        && priorViewKey !== 'ln'
        && priorViewKey !== 'final'
        && priorViewKey !== 'layer-end-desktop'
        // Do not keep concat framing pinned while entering LN2/MLP.
        && priorViewKey !== 'concat');
    const holdViewThroughLayerHandoff = !!(inLayerHandoff
        && priorViewKey !== 'layer-end-desktop'
        && priorViewKey !== 'final');
    const bottomEmbeddingViewKey = resolveBottomEmbeddingViewKey({
        pipeline,
        layer,
        lanes,
        nowMs
    });
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
        bottomEmbeddingViewKey,
        priorViewKey,
        forwardComplete
    };
    if (forwardComplete) {
        return { rawKey: 'final', viewContext };
    }
    if (bottomEmbeddingViewKey) {
        return { rawKey: bottomEmbeddingViewKey, viewContext };
    }

    const firstLayerEmbeddingToLnHandoff = !!(
        layer.index === 0
        && lanes.some((candidate) => !!candidate?.posVec)
        && lanes.every((candidate) => !candidate?.posVec || candidate.posAddComplete === true)
        && lanes.some((candidate) => candidate
            && (
                candidate.horizPhase === HORIZ_PHASE.WAITING
                || candidate.horizPhase === HORIZ_PHASE.RIGHT
                || candidate.horizPhase === HORIZ_PHASE.RISE_ABOVE_LN
            ))
    );
    if (firstLayerEmbeddingToLnHandoff) {
        return { rawKey: 'ln', viewContext };
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
    if (fromKey === 'concat' && toKey === 'default') {
        holdMs = Math.min(holdMs, 36);
    }
    if (fromKey === EMBED_VIEW_KEY_VOCAB && toKey === EMBED_VIEW_KEY_POSITION) {
        holdMs = 0;
    }
    if (isEmbedBottomViewKey(fromKey) || isEmbedBottomViewKey(toKey)) {
        holdMs = Math.min(holdMs, 24);
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
