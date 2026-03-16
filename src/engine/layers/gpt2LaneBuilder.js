import * as THREE from 'three';
import { StraightLineTrail } from '../../utils/trailUtils.js';
import { TRAIL_MIN_SEGMENT_DISTANCE } from '../../utils/trailConstants.js';
import { PrismLayerNormAnimation } from '../../animations/PrismLayerNormAnimation.js';
import { startPrismAdditionAnimation } from '../../utils/additionUtils.js';
import {
    applyVectorData,
    cloneVectorKeyColors,
    copyVectorAppearance,
    LN_INTERNAL_TRAIL_MIN_SEGMENT
} from './gpt2LayerUtils.js';
import { BatchedPrismVectorSet } from '../../components/BatchedPrismVectorSet.js';
import { logRandomColorDebug } from '../../utils/randomColorDebug.js';
import { LAYER_NORM_PARAM_COLOR_OPTIONS } from '../../utils/layerNormParamColorOptions.js';
import {
    LN_PARAMS,
    LAYER_NORM_1_Y_POS,
    EMBEDDING_BOTTOM_TOP_ALIGN_OFFSET_FROM_LN1_BOTTOM,
    EMBEDDING_BOTTOM_Y_ADJUST,
    EMBEDDING_MATRIX_PARAMS_POSITION,
    EMBEDDING_MATRIX_PARAMS_VOCAB,
    EMBEDDING_BOTTOM_PAIR_GAP_X,
    EMBEDDING_BOTTOM_POS_X_OFFSET,
    EMBEDDING_BOTTOM_VOCAB_X_OFFSET,
    POS_VEC_Y_OFFSET_ABOVE_VOCAB,
    POS_VEC_VERTICAL_SPEED_MULT,
    POS_VEC_HORIZONTAL_SPEED_MULT,
    ANIM_RISE_SPEED_ORIGINAL,
    GLOBAL_ANIM_SPEED_MULT,
    ANIM_HORIZ_SPEED
} from '../../utils/constants.js';
import { HORIZ_PHASE, LN2_PHASE } from './gpt2LanePhases.js';

const TMP_WORLD_POS = new THREE.Vector3();
const LN_ADD_VECTOR_OFFSET_FRACTION = 0.25; // fraction of LN height above centre for bias addition
export const LN_PARAM_MONOCHROME = {
    ...LAYER_NORM_PARAM_COLOR_OPTIONS
};

function getPrismVectorHeight(vec) {
    const halfPrismHeight = Number.isFinite(vec?._basePrismCenterY)
        ? vec._basePrismCenterY
        : 0;
    return halfPrismHeight > 0 ? halfPrismHeight * 2 : 10.5;
}

function getInputVocabSpawnLayout(startY, vec) {
    const prismHeight = getPrismVectorHeight(vec);
    const prismHalfHeight = prismHeight / 2;
    const matrixBottomY = startY - EMBEDDING_MATRIX_PARAMS_VOCAB.height;
    const travelStartY = matrixBottomY - prismHalfHeight;
    const entryY = matrixBottomY + prismHalfHeight;
    return {
        travelStartY,
        entryY,
        visibleStartY: travelStartY + prismHeight * 0.2,
        revealY: startY - prismHeight
    };
}

export function createFreshLanes(layer, offsetX, ln1CenterY, ln2CenterY, ln1TopY) {
    const layoutCount = (typeof layer._getLaneLayoutCount === 'function')
        ? layer._getLaneLayoutCount()
        : layer._laneCount;
    const activeLaneLayoutIndices = (typeof layer._getActiveLaneLayoutIndices === 'function')
        ? layer._getActiveLaneLayoutIndices()
        : Array.from({ length: layer._laneCount }, (_, idx) => idx);
    const slitSpacing = LN_PARAMS.depth / (layoutCount + 1);
    // Anchor the first-layer vocab vectors to the bottom embedding geometry;
    // per-lane setup may place them below the block for an upward entry.
    const startY = (LAYER_NORM_1_Y_POS - LN_PARAMS.height / 2 + EMBEDDING_BOTTOM_TOP_ALIGN_OFFSET_FROM_LN1_BOTTOM) + EMBEDDING_BOTTOM_Y_ADJUST;
    const meetY = ln1TopY + 5;
    for (let localLaneIdx = 0; localLaneIdx < layer._laneCount; localLaneIdx++) {
        const laneLayoutIdx = Number.isFinite(activeLaneLayoutIndices[localLaneIdx])
            ? activeLaneLayoutIndices[localLaneIdx]
            : localLaneIdx;
        buildSingleLane(
            layer,
            null,
            offsetX,
            ln1CenterY,
            ln2CenterY,
            startY,
            meetY,
            laneLayoutIdx,
            slitSpacing,
            localLaneIdx
        );
    }
    layer._flushDirtyLayerNormParamBanks?.();
    if (layer._ln1AddPlaceholders && layer._ln1AddPlaceholders.every(p => !p)) {
        layer._ln1AddPlaceholders = [];
    }
    if (layer._ln2AddPlaceholders && layer._ln2AddPlaceholders.every(p => !p)) {
        layer._ln2AddPlaceholders = [];
    }
    if (layer._ln1ScalePlaceholders && layer._ln1ScalePlaceholders.every(p => !p)) {
        layer._ln1ScalePlaceholders = [];
    }
    if (layer._ln2ScalePlaceholders && layer._ln2ScalePlaceholders.every(p => !p)) {
        layer._ln2ScalePlaceholders = [];
    }
}

export function createAdditionPlaceholders(layer, offsetX, ln1CenterY, ln2CenterY) {
    try {
        const laneCount = Math.max(1, layer._laneCount || 1);
        const layoutCount = (typeof layer._getLaneLayoutCount === 'function')
            ? layer._getLaneLayoutCount()
            : laneCount;
        const activeLaneLayoutIndices = (typeof layer._getActiveLaneLayoutIndices === 'function')
            ? layer._getActiveLaneLayoutIndices()
            : Array.from({ length: laneCount }, (_, idx) => idx);
        const slitSpacing = LN_PARAMS.depth / (layoutCount + 1);
        const addYOffset = LN_PARAMS.height * LN_ADD_VECTOR_OFFSET_FRACTION;
        const raycastRoot = layer.raycastRoot || layer.root;
        const prismCount = typeof layer._getBaseVectorLength === 'function'
            ? layer._getBaseVectorLength()
            : undefined;

        const makeBank = (label) => new BatchedPrismVectorSet({
            vectorCount: laneCount,
            prismCount,
            parentGroup: raycastRoot,
            label,
        });

        if (!layer._lnParamBanks || layer._lnParamBanks._laneCount !== laneCount) {
            layer._lnParamBanks = {
                _laneCount: laneCount,
                ln1Scale: makeBank('LN1 Scale Params'),
                ln1Shift: makeBank('LN1 Shift Params'),
                ln2Scale: makeBank('LN2 Scale Params'),
                ln2Shift: makeBank('LN2 Shift Params'),
            };
        }

        const banks = layer._lnParamBanks;
        for (let localLaneIdx = 0; localLaneIdx < laneCount; localLaneIdx++) {
            const laneLayoutIdx = Number.isFinite(activeLaneLayoutIndices[localLaneIdx])
                ? activeLaneLayoutIndices[localLaneIdx]
                : localLaneIdx;
            const zPos = -LN_PARAMS.depth / 2 + slitSpacing * (laneLayoutIdx + 1);

            const ln1ScaleRef = layer._registerLayerNormParamBankRef(
                banks.ln1Scale.getVectorRef(localLaneIdx),
                'ln1Scale'
            );
            layer._setLayerNormParamRefLayout(ln1ScaleRef, {
                x: offsetX,
                y: ln1CenterY + 3.3,
                z: zPos,
                visible: true
            });
            layer._applyLayerNormParamVector(ln1ScaleRef, 'ln1', 'scale', LN_PARAM_MONOCHROME);

            const ln1ShiftRef = layer._registerLayerNormParamBankRef(
                banks.ln1Shift.getVectorRef(localLaneIdx),
                'ln1Shift'
            );
            layer._setLayerNormParamRefLayout(ln1ShiftRef, {
                x: offsetX,
                y: ln1CenterY + addYOffset,
                z: zPos,
                visible: true
            });
            layer._applyLayerNormParamVector(ln1ShiftRef, 'ln1', 'shift', LN_PARAM_MONOCHROME);

            const ln2ScaleRef = layer._registerLayerNormParamBankRef(
                banks.ln2Scale.getVectorRef(localLaneIdx),
                'ln2Scale'
            );
            layer._setLayerNormParamRefLayout(ln2ScaleRef, {
                x: offsetX,
                y: ln2CenterY + 3.3,
                z: zPos,
                visible: true
            });
            layer._applyLayerNormParamVector(ln2ScaleRef, 'ln2', 'scale', LN_PARAM_MONOCHROME);

            const ln2ShiftRef = layer._registerLayerNormParamBankRef(
                banks.ln2Shift.getVectorRef(localLaneIdx),
                'ln2Shift'
            );
            layer._setLayerNormParamRefLayout(ln2ShiftRef, {
                x: offsetX,
                y: ln2CenterY + addYOffset,
                z: zPos,
                visible: true
            });
            layer._applyLayerNormParamVector(ln2ShiftRef, 'ln2', 'shift', LN_PARAM_MONOCHROME);
        }

        banks.ln1Scale.syncAll();
        banks.ln1Shift.syncAll();
        banks.ln2Scale.syncAll();
        banks.ln2Shift.syncAll();
        layer._clearDirtyLayerNormParamBanks?.();
    } catch (_) {
        // Placeholders are a visual aid only - failures shouldn't stop the demo.
    }
}

export function createLanesFromExternal(layer, externalLanes, offsetX, ln1CenterY, ln2CenterY, ln1TopY) {
    const meetY = ln1TopY + 5; // where original vectors pause just above LN1

    // DON'T reset position - let vectors continue from where they are after layer 1.
    const activeLaneLayoutIndices = (typeof layer._getActiveLaneLayoutIndices === 'function')
        ? layer._getActiveLaneLayoutIndices()
        : [];
    externalLanes.forEach((oldLane, localLaneIdx) => {
        const laneLayoutIdx = Number.isFinite(oldLane?.laneLayoutIndex)
            ? oldLane.laneLayoutIndex
            : (Number.isFinite(activeLaneLayoutIndices[localLaneIdx])
                ? activeLaneLayoutIndices[localLaneIdx]
                : localLaneIdx);
        buildSingleLane(
            layer,
            oldLane,
            offsetX,
            ln1CenterY,
            ln2CenterY,
            null,
            meetY,
            laneLayoutIdx,
            null,
            localLaneIdx
        );
    });
    layer._flushDirtyLayerNormParamBanks?.();
    if (layer._ln1AddPlaceholders && layer._ln1AddPlaceholders.every(p => !p)) {
        layer._ln1AddPlaceholders = [];
    }
    if (layer._ln2AddPlaceholders && layer._ln2AddPlaceholders.every(p => !p)) {
        layer._ln2AddPlaceholders = [];
    }
    if (layer._ln1ScalePlaceholders && layer._ln1ScalePlaceholders.every(p => !p)) {
        layer._ln1ScalePlaceholders = [];
    }
    if (layer._ln2ScalePlaceholders && layer._ln2ScalePlaceholders.every(p => !p)) {
        layer._ln2ScalePlaceholders = [];
    }
}

export function buildSingleLane(layer, oldLane, offsetX, ln1CenterY, ln2CenterY, startY_override, meetY, laneLayoutIdx, slitSpacing, laneLocalIdx = 0) {
    const raycastRoot = layer.raycastRoot || layer.root;
    // Reuse existing trail when lanes are passed from a lower layer.
    let trailFromPrev = oldLane && oldLane.originalTrail ? oldLane.originalTrail : null;
    const resolvedLayoutLaneIdx = Number.isFinite(laneLayoutIdx) ? Math.floor(laneLayoutIdx) : laneLocalIdx;
    const laneTokenIndex = (oldLane && Number.isFinite(oldLane.tokenIndex))
        ? oldLane.tokenIndex
        : layer._getTokenIndexForLane(laneLocalIdx, resolvedLayoutLaneIdx);
    const laneTokenLabel = (oldLane && oldLane.tokenLabel)
        ? oldLane.tokenLabel
        : layer._getTokenLabel(laneTokenIndex);
    let originalVec, zPos, startY, trail;
    let inputVocabLayout = null;
    let inputVocabSpawnLowered = false;

    if (oldLane && oldLane.originalVec) {
        originalVec = oldLane.originalVec;
        raycastRoot.attach(originalVec.group);
        zPos = originalVec.group.position.z;
        startY = originalVec.group.position.y; // Keep current position.
        // Prefer to carry over the existing residual-stream trail so it
        // remains a single continuous line across layer boundaries.
        if (trailFromPrev) {
            trail = trailFromPrev;
            if (typeof trail.reparent === 'function') {
                trail.reparent(layer._globalScene);
            }
            originalVec.userData = originalVec.userData || {};
            originalVec.userData.trail = trail;
            originalVec.userData.trailWorld = true;
        }
    } else {
        zPos = -LN_PARAMS.depth / 2 + slitSpacing * (resolvedLayoutLaneIdx + 1);
        let data = layer.random.nextVector(layer._getBaseVectorLength());
        let tokenData = null;
        if (layer.activationSource) {
            tokenData = layer._getEmbeddingData({ tokenIndex: laneTokenIndex }, 'token');
            if (tokenData) data = tokenData;
        }
        if (!tokenData) {
            logRandomColorDebug('LaneBuilder.tokenEmbedding.randomVectorUsed', {
                layerIndex: layer.index,
                laneIndex: laneLocalIdx,
                tokenIndex: laneTokenIndex,
                length: layer._getBaseVectorLength()
            });
        }
        startY = startY_override;
        originalVec = layer._createPrismVector(
            data,
            new THREE.Vector3(0, startY, zPos),
            30,
            layer._getInstanceCountFromData(data)
        );
        if (layer.index === 0) {
            // Start first-layer vocab vectors below the bottom face so they
            // visibly enter the embedding before emerging from the top slit.
            inputVocabLayout = getInputVocabSpawnLayout(startY, originalVec);
            originalVec.group.position.y = inputVocabLayout.travelStartY;
            originalVec.group.visible = false;
            inputVocabSpawnLowered = true;
        }
        raycastRoot.add(originalVec.group);
        applyVectorData(
            originalVec,
            data,
            laneTokenLabel ? `Token Embedding - ${laneTokenLabel}` : 'Token Embedding',
            layer._getLaneMeta({ tokenIndex: laneTokenIndex, tokenLabel: laneTokenLabel }, 'embedding.token')
        );

        // Trail for the ORIGINAL vector.
        // Attach to the GLOBAL scene and record WORLD positions so the trail
        // remains continuous across layers as lanes are transferred upwards.
        trail = new StraightLineTrail(layer._globalScene, 0xffffff, 1, undefined, undefined, TRAIL_MIN_SEGMENT_DISTANCE);
        originalVec.group.getWorldPosition(TMP_WORLD_POS);
        trail.start(TMP_WORLD_POS);
        originalVec.userData = originalVec.userData || {};
        originalVec.userData.trail = trail;
        originalVec.userData.trailWorld = true;
    }

    if (oldLane && layer.activationSource) {
        const incomingData = layer._getLayerIncomingData({ tokenIndex: laneTokenIndex });
        if (incomingData) {
            applyVectorData(
                originalVec,
                incomingData,
                laneTokenLabel ? `Incoming Residual (Pre-LN1) - ${laneTokenLabel}` : 'Incoming Residual (Pre-LN1)',
                layer._getLaneMeta({ tokenIndex: laneTokenIndex, tokenLabel: laneTokenLabel }, 'layer.incoming')
            );
        }
    }

    // Spawn the LN-1 duplicate at the staging height (bottom + 5) so that
    // when it becomes visible and starts the 'right' phase it travels purely
    // horizontally, matching LN-2 behaviour.
    const dupStartPos = new THREE.Vector3(
        originalVec.group.position.x,
        ln1CenterY - LN_PARAMS.height / 2 + 5,
        originalVec.group.position.z
    );
    const dupVec = layer._createPrismVector(
        originalVec.rawData.slice(),
        dupStartPos,
        30,
        originalVec.instanceCount
    );
    dupVec.group.visible = false;
    copyVectorAppearance(dupVec, originalVec);
    raycastRoot.add(dupVec.group);
    // Trail for duplicate vector inside LN1.
    const dupTrail = new StraightLineTrail(layer.root, 0xffffff, 1, undefined, undefined, LN_INTERNAL_TRAIL_MIN_SEGMENT);
    // Preserve exact right-angle LN entry geometry during skip: internal LN trails
    // should not be throttled by global trail-step clamping.
    if (typeof dupTrail.setMaxStepDistance === 'function') {
        dupTrail.setMaxStepDistance(1e9);
    }
    dupTrail.start(dupVec.group.position);
    dupVec.userData = dupVec.userData || {};
    dupVec.userData.trail = dupTrail;
    const normAnim = new PrismLayerNormAnimation(dupVec);

    // If we're reusing an existing lane we may not have created the trail yet.
    if (!trail) {
        trail = new StraightLineTrail(layer._globalScene, 0xffffff, 1, undefined, undefined, TRAIL_MIN_SEGMENT_DISTANCE);
        originalVec.group.getWorldPosition(TMP_WORLD_POS);
        trail.start(TMP_WORLD_POS);
    }

    const addYOffset = LN_PARAMS.height * LN_ADD_VECTOR_OFFSET_FRACTION;
    const paramBanks = layer._lnParamBanks || null;
    const multTarget = paramBanks && paramBanks.ln1Scale
        ? layer._registerLayerNormParamBankRef(paramBanks.ln1Scale.getVectorRef(laneLocalIdx), 'ln1Scale')
        : null;
    const addTarget = paramBanks && paramBanks.ln1Shift
        ? layer._registerLayerNormParamBankRef(paramBanks.ln1Shift.getVectorRef(laneLocalIdx), 'ln1Shift')
        : null;
    const multTargetLN2 = paramBanks && paramBanks.ln2Scale
        ? layer._registerLayerNormParamBankRef(paramBanks.ln2Scale.getVectorRef(laneLocalIdx), 'ln2Scale')
        : null;
    const addTargetLN2 = paramBanks && paramBanks.ln2Shift
        ? layer._registerLayerNormParamBankRef(paramBanks.ln2Shift.getVectorRef(laneLocalIdx), 'ln2Shift')
        : null;

    layer._setLayerNormParamRefLayout(multTarget, {
        x: offsetX,
        y: ln1CenterY + 3.3,
        z: zPos,
        visible: true
    });
    layer._setLayerNormParamRefLayout(addTarget, {
        x: offsetX,
        y: ln1CenterY + addYOffset,
        z: zPos,
        visible: true
    });
    layer._setLayerNormParamRefLayout(multTargetLN2, {
        x: offsetX,
        y: ln2CenterY + 3.3,
        z: zPos,
        visible: true
    });
    layer._setLayerNormParamRefLayout(addTargetLN2, {
        x: offsetX,
        y: ln2CenterY + addYOffset,
        z: zPos,
        visible: true
    });

    // Fallback to previous trail if a new one wasn't created in this constructor.
    if (!trail && trailFromPrev) trail = trailFromPrev;

    // Ensure originalVec always has a trail reference.
    originalVec.userData = originalVec.userData || {};
    if (!originalVec.userData.trail) {
        originalVec.userData.trail = trail;
    }
    // Always mark world-space trail semantics so updates use world coords.
    originalVec.userData.trailWorld = true;

    layer.lanes.push({
        layer,
        laneIndex: laneLocalIdx,
        laneLayoutIndex: resolvedLayoutLaneIdx,
        tokenIndex: laneTokenIndex,
        tokenLabel: laneTokenLabel,
        originalVec,
        originalTrail: trail,
        dupVec,
        multTarget,
        multTargetLN2,
        addTarget,
        addTargetLN2,
        normAnim,
        horizPhase: HORIZ_PHASE.WAITING,
        branchStartY: ln1CenterY - LN_PARAMS.height / 2 + 5,
        ln1MidY: ln1CenterY,
        normStarted: false,
        normApplied: false,
        pendingNormData: null,
        pendingNormLabel: null,
        pendingNormMeta: null,
        multStarted: false,
        ln1AddStarted: false,
        ln1AddComplete: false,
        ln1ShiftProgress: 0,
        ln1ParamColored: false,
        resultVec: null,
        targetY: meetY,
        travellingVec: null,
        upwardCopies: [],
        sideCopies: [],
        headIndex: 0,
        finalAscend: false,
        ln2Phase: LN2_PHASE.NOT_STARTED,
        postAdditionVec: null,
        movingVecLN2: null,
        normAnimationLN2: null,
        normStartedLN2: false,
        normAppliedLN2: false,
        pendingNormDataLN2: null,
        pendingNormLabelLN2: null,
        pendingNormMetaLN2: null,
        multDoneLN2: false,
        ln2ParamColored: false,
        ln2AddStarted: false,
        ln2AddComplete: false,
        ln2ShiftProgress: 0,
        resultVecLN2: null,
        mlpUpStarted: false,
        mlpGeluActive: false,
        mlpGeluComplete: false,
        expandedVecGroup: null,
        expandedVecSegments: null,
        finalVecAfterMlp: null,
        expandedVecTrail: null,
        mlpDownStarted: false,
        mlpDownComplete: false,
        mlpReturnStarted: false,
        // Top Y of the bottom vocab embedding matrix; used to detect when the
        // residual vector has exited the embedding block.
        vocabEmbeddingExitY: Number.isFinite(startY_override) ? startY_override : startY,
        vocabEmbeddingEntryY: inputVocabLayout ? inputVocabLayout.entryY : NaN,
        vocabEmbeddingVisibleStartY: inputVocabLayout ? inputVocabLayout.visibleStartY : NaN,
        vocabEmbeddingRevealY: inputVocabLayout
            ? inputVocabLayout.revealY
            : (Number.isFinite(startY_override) ? startY_override : startY) - getPrismVectorHeight(originalVec),
        vocabEmbeddingTravelStartY: inputVocabLayout ? inputVocabLayout.travelStartY : NaN,
        vocabEmbeddingBaseColors: inputVocabLayout ? cloneVectorKeyColors(originalVec) : null,
        __inputVocabGateAdjustedStartY: inputVocabSpawnLowered,
        zPos,
        __residualMaxY: (function(){ originalVec.group.getWorldPosition(TMP_WORLD_POS); return TMP_WORLD_POS.y; })()
    });

    // ------------------------------------------------------------
    // Initial positional-embedding vector (first layer only)
    // ------------------------------------------------------------
    if (layer.index === 0) {
        const lane = layer.lanes[layer.lanes.length - 1];

        try {
            lane.posAddComplete = false;
            lane.posAddStarted = false;
            lane.__posPassStarted = false;
            lane.__posPreAddApproach = false;
            // Start at the TOP of the bottom positional embedding, horizontally to the right.
            const residualTopY = (LAYER_NORM_1_Y_POS - LN_PARAMS.height / 2 + EMBEDDING_BOTTOM_TOP_ALIGN_OFFSET_FROM_LN1_BOTTOM) + EMBEDDING_BOTTOM_Y_ADJUST;
            // The positional embedding matrix is shorter than the vocab matrix. Drop the
            // starting Y so the trail originates from the actual top of the positional
            // matrix rather than the top of the taller vocab matrix.
            const posStartY = residualTopY - (EMBEDDING_MATRIX_PARAMS_VOCAB.height - EMBEDDING_MATRIX_PARAMS_POSITION.height);
            const posStartX = (EMBEDDING_MATRIX_PARAMS_VOCAB.width / 2)
                            + (EMBEDDING_MATRIX_PARAMS_POSITION.width / 2)
                            + EMBEDDING_BOTTOM_PAIR_GAP_X
                            + EMBEDDING_BOTTOM_POS_X_OFFSET
                            + EMBEDDING_BOTTOM_VOCAB_X_OFFSET;

            // Give positional a distinct random pattern.
            let posData = layer.random.nextVector(layer._getBaseVectorLength());
            let posEmbedding = null;
            if (layer.activationSource) {
                posEmbedding = layer._getEmbeddingData(lane, 'position');
                if (posEmbedding) posData = posEmbedding;
            }
            if (!posEmbedding) {
                logRandomColorDebug('LaneBuilder.positionEmbedding.randomVectorUsed', {
                    layerIndex: layer.index,
                    laneIndex: laneLocalIdx,
                    tokenIndex: laneTokenIndex,
                    length: layer._getBaseVectorLength()
                });
            }
            const posVec = layer._createPrismVector(
                posData,
                new THREE.Vector3(posStartX, posStartY, zPos),
                30,
                layer._getInstanceCountFromData(posData)
            );
            // Match vocab behaviour: begin slightly inside the positional
            // embedding so the first visible motion is an upward emerge.
            posVec.group.position.y -= getPrismVectorHeight(posVec);
            raycastRoot.add(posVec.group);
            // Keep positional vectors hidden until the deferred positional pass-through starts.
            posVec.group.visible = false;
            applyVectorData(
                posVec,
                posData,
                lane.tokenLabel ? `Position Embedding - ${lane.tokenLabel}` : 'Position Embedding',
                layer._getLaneMeta(lane, 'embedding.position')
            );
            // Trail (local to this layer) - enabled only until it reaches residual stream.
            const posTrail = new StraightLineTrail(layer.root, 0xffffff, 1, undefined, undefined, TRAIL_MIN_SEGMENT_DISTANCE);
            // Skip mode globally clamps trail advance per update; disable that
            // for positional pass-through so explicit corner waypoints
            // (into gap, then up, then left) are preserved instead of
            // collapsing into diagonals.
            if (typeof posTrail.setMaxStepDistance === 'function') {
                posTrail.setMaxStepDistance(1e9);
            }
            posTrail.start(posVec.group.position);
            posVec.userData = posVec.userData || {};
            posVec.userData.trail = posTrail;
            let posTrailDisposed = false;
            const retirePosTrail = () => {
                if (posTrailDisposed) return;
                posTrailDisposed = true;
                try { if (posVec.userData) delete posVec.userData.trail; } catch (_) {}
                lane.posTrail = null;
            };

            lane.posVec = posVec;
            lane.posTrail = posTrail;
            lane.__manualPosTrail = false;
            lane.__posPassAdjustedStartY = true;

            const fasterRise = ANIM_RISE_SPEED_ORIGINAL * POS_VEC_VERTICAL_SPEED_MULT;
            const horizSpeed = ANIM_HORIZ_SPEED * POS_VEC_HORIZONTAL_SPEED_MULT * GLOBAL_ANIM_SPEED_MULT;
            // Preserve the original rise target baseline (top-of-vocab stage),
            // even though positional motion now starts later.
            const vocabRiseReferenceY = (startY_override != null ? startY_override : originalVec.group.position.y);
            const triggerPositionalAddition = () => {
                lane.__posPreAddApproach = false;
                if (posTrail && !posTrailDisposed) {
                    posTrail.snapLastPointTo(posVec.group.position);
                }
                lane.__manualPosTrail = false;
                // Stop extending trail once we arrive at residual stream.
                retirePosTrail();
                // Trigger addition: positional (above) travels DOWN into vocab (rising).
                try {
                    const sumData = layer._getEmbeddingData(lane, 'sum');
                    lane.posAddStarted = true;
                    startPrismAdditionAnimation(posVec, originalVec, null, () => {
                        if (sumData) {
                            applyVectorData(
                                originalVec,
                                sumData,
                                lane.tokenLabel ? `Embedding Sum - ${lane.tokenLabel}` : 'Embedding Sum',
                                layer._getLaneMeta(lane, 'embedding.sum')
                            );
                        }
                        lane.posAddComplete = true;
                    }, { finalData: sumData });
                } catch (_) {
                    lane.__manualPosTrail = false;
                    lane.posAddComplete = true;
                }
            };

            // Defer positional pass-through start until Gpt2Layer releases the
            // per-lane position-chip gate for this token.
            lane.startPositionalPassThrough = ({ immediate = false } = {}) => {
                if (lane.__posPassStarted || lane.posAddComplete) return;
                lane.__posPassStarted = true;
                lane.__posPreAddApproach = false;
                lane.__manualPosTrail = true;
                const syncPosTrailToCurrent = ({ append = false } = {}) => {
                    if (!posTrail || posTrailDisposed || !posVec || !posVec.group) return;
                    if (append && typeof posTrail.update === 'function') {
                        posTrail.update(posVec.group.position);
                    } else if (typeof posTrail.snapLastPointTo === 'function') {
                        posTrail.snapLastPointTo(posVec.group.position);
                    } else if (typeof posTrail.update === 'function') {
                        posTrail.update(posVec.group.position);
                    }
                };
                if (!lane.__posPassAdjustedStartY && posVec && posVec.group) {
                    const halfPrismHeight = Number.isFinite(posVec._basePrismCenterY)
                        ? posVec._basePrismCenterY
                        : 0;
                    const prismHeight = halfPrismHeight > 0 ? halfPrismHeight * 2 : 10.5;
                    posVec.group.position.y -= prismHeight;
                    lane.__posPassAdjustedStartY = true;
                    syncPosTrailToCurrent({ append: true });
                }
                if (posVec && posVec.group) {
                    posVec.group.visible = true;
                }
                const passStartX = posVec && posVec.group ? posVec.group.position.x : posStartX;
                const passStartY = posVec && posVec.group ? posVec.group.position.y : posStartY;
                const targetYAbove = vocabRiseReferenceY + POS_VEC_Y_OFFSET_ABOVE_VOCAB;
                const riseDist = Math.max(0, targetYAbove - posVec.group.position.y);
                const riseMs = (riseDist / (fasterRise * GLOBAL_ANIM_SPEED_MULT)) * 1000;
                const mergeDist = Math.abs(passStartX - 0);
                const mergeMs = (mergeDist / horizSpeed) * 1000;

                if (typeof TWEEN !== 'undefined' && !immediate) {
                    const startMergeToResidual = () => {
                        lane.__posPreAddApproach = true;
                        const finishMergeToResidual = () => {
                            posVec.group.position.y = targetYAbove;
                            syncPosTrailToCurrent({ append: true });
                            triggerPositionalAddition();
                        };
                        if (mergeDist <= 1e-4) {
                            posVec.group.position.x = 0;
                            finishMergeToResidual();
                            return;
                        }
                        new TWEEN.Tween(posVec.group.position)
                            .to({ x: 0, y: targetYAbove }, Math.max(100, mergeMs))
                            .easing(TWEEN.Easing.Quadratic.InOut)
                            .onStart(() => {
                                // Hard-lock Y during horizontal travel to ensure a perfectly straight path.
                                posVec.group.position.y = targetYAbove;
                                syncPosTrailToCurrent({ append: true });
                            })
                            .onUpdate(() => {
                                // Maintain Y lock during horizontal interpolation.
                                posVec.group.position.y = targetYAbove;
                                syncPosTrailToCurrent({ append: true });
                            })
                            .onComplete(finishMergeToResidual)
                            .start();
                    };
                    const finishVerticalRise = () => {
                        // Force a corner sample so fast transitions preserve
                        // the right-angle path (up first, then left).
                        posVec.group.position.x = passStartX;
                        posVec.group.position.y = targetYAbove;
                        syncPosTrailToCurrent({ append: true });
                        startMergeToResidual();
                    };
                    if (riseDist <= 1e-4) {
                        finishVerticalRise();
                    } else {
                        new TWEEN.Tween(posVec.group.position)
                            .to({ y: targetYAbove }, Math.max(100, riseMs))
                            .easing(TWEEN.Easing.Quadratic.InOut)
                            .onUpdate(() => {
                                // Hard-lock X during the vertical stage.
                                posVec.group.position.x = passStartX;
                                syncPosTrailToCurrent({ append: true });
                            })
                            .onComplete(finishVerticalRise)
                            .start();
                    }
                    return;
                }

                // Preserve the same right-angle trail geometry during skip:
                // rise first, then merge to x=0.
                if (posVec && posVec.group) {
                    lane.__posPreAddApproach = true;
                    if (Math.abs(posVec.group.position.y - targetYAbove) > 1e-4) {
                        posVec.group.position.x = passStartX;
                        posVec.group.position.y = targetYAbove;
                        syncPosTrailToCurrent({ append: true });
                    }
                    if (Math.abs(posVec.group.position.x) > 1e-4) {
                        posVec.group.position.x = 0;
                        posVec.group.position.y = targetYAbove;
                        syncPosTrailToCurrent({ append: true });
                    }
                    posVec.group.position.y = targetYAbove;
                    posVec.group.position.z = zPos;
                }
                triggerPositionalAddition();
            };
        } catch (_) {
            // Non-fatal - positional addition is a visual enhancement only.
        }
    }
}
