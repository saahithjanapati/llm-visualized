import * as THREE from 'three';
import { StraightLineTrail } from '../../utils/trailUtils.js';
import { TRAIL_MIN_SEGMENT_DISTANCE } from '../../utils/trailConstants.js';
import { PrismLayerNormAnimation } from '../../animations/PrismLayerNormAnimation.js';
import { startPrismAdditionAnimation } from '../../utils/additionUtils.js';
import { applyVectorData, copyVectorAppearance, LN_INTERNAL_TRAIL_MIN_SEGMENT } from './gpt2LayerUtils.js';
import { BatchedPrismVectorSet } from '../../components/BatchedPrismVectorSet.js';
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

const TMP_WORLD_POS = new THREE.Vector3();
const LN_ADD_VECTOR_OFFSET_FRACTION = 0.25; // fraction of LN height above centre for bias addition
export const LN_PARAM_MONOCHROME = {
    type: 'monochromatic',
    baseHue: 0,
    saturation: 0,
    minLightness: 0.03,
    maxLightness: 0.88,
    useData: true,
    valueMin: -1.8,
    valueMax: 1.8
};

export function createFreshLanes(layer, offsetX, ln1CenterY, ln2CenterY, ln1TopY) {
    const slitSpacing = LN_PARAMS.depth / (layer._laneCount + 1);
    // Start vectors at the TOP of the bottom embedding matrix.
    const startY = (LAYER_NORM_1_Y_POS - LN_PARAMS.height / 2 + EMBEDDING_BOTTOM_TOP_ALIGN_OFFSET_FROM_LN1_BOTTOM) + EMBEDDING_BOTTOM_Y_ADJUST;
    const meetY = ln1TopY + 5;
    for (let laneIdx = 0; laneIdx < layer._laneCount; laneIdx++) {
        buildSingleLane(layer, null, offsetX, ln1CenterY, ln2CenterY, startY, meetY, laneIdx, slitSpacing);
    }
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
        const slitSpacing = LN_PARAMS.depth / (laneCount + 1);
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
        for (let laneIdx = 0; laneIdx < laneCount; laneIdx++) {
            const zPos = -LN_PARAMS.depth / 2 + slitSpacing * (laneIdx + 1);

            const ln1ScaleRef = banks.ln1Scale.getVectorRef(laneIdx);
            ln1ScaleRef.group.position.set(offsetX, ln1CenterY + 3.3, zPos);
            ln1ScaleRef.group.visible = true;
            layer._applyLayerNormParamVector(ln1ScaleRef, 'ln1', 'scale', LN_PARAM_MONOCHROME);

            const ln1ShiftRef = banks.ln1Shift.getVectorRef(laneIdx);
            ln1ShiftRef.group.position.set(offsetX, ln1CenterY + addYOffset, zPos);
            ln1ShiftRef.group.visible = true;
            layer._applyLayerNormParamVector(ln1ShiftRef, 'ln1', 'shift', LN_PARAM_MONOCHROME);

            const ln2ScaleRef = banks.ln2Scale.getVectorRef(laneIdx);
            ln2ScaleRef.group.position.set(offsetX, ln2CenterY + 3.3, zPos);
            ln2ScaleRef.group.visible = true;
            layer._applyLayerNormParamVector(ln2ScaleRef, 'ln2', 'scale', LN_PARAM_MONOCHROME);

            const ln2ShiftRef = banks.ln2Shift.getVectorRef(laneIdx);
            ln2ShiftRef.group.position.set(offsetX, ln2CenterY + addYOffset, zPos);
            ln2ShiftRef.group.visible = true;
            layer._applyLayerNormParamVector(ln2ShiftRef, 'ln2', 'shift', LN_PARAM_MONOCHROME);
        }

        banks.ln1Scale.syncAll();
        banks.ln1Shift.syncAll();
        banks.ln2Scale.syncAll();
        banks.ln2Shift.syncAll();
    } catch (_) {
        // Placeholders are a visual aid only - failures shouldn't stop the demo.
    }
}

export function createLanesFromExternal(layer, externalLanes, offsetX, ln1CenterY, ln2CenterY, ln1TopY) {
    const meetY = ln1TopY + 5; // where original vectors pause just above LN1

    // DON'T reset position - let vectors continue from where they are after layer 1.
    externalLanes.forEach((oldLane, laneIdx) => {
        buildSingleLane(layer, oldLane, offsetX, ln1CenterY, ln2CenterY, null, meetY, laneIdx, null);
    });
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

export function buildSingleLane(layer, oldLane, offsetX, ln1CenterY, ln2CenterY, startY_override, meetY, laneIdx, slitSpacing) {
    const raycastRoot = layer.raycastRoot || layer.root;
    // Reuse existing trail when lanes are passed from a lower layer.
    let trailFromPrev = oldLane && oldLane.originalTrail ? oldLane.originalTrail : null;
    const laneTokenIndex = (oldLane && Number.isFinite(oldLane.tokenIndex))
        ? oldLane.tokenIndex
        : layer._getTokenIndexForLane(laneIdx);
    const laneTokenLabel = (oldLane && oldLane.tokenLabel)
        ? oldLane.tokenLabel
        : layer._getTokenLabel(laneTokenIndex);
    let originalVec, zPos, startY, trail;

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
        zPos = -LN_PARAMS.depth / 2 + slitSpacing * (laneIdx + 1);
        let data = layer.random.nextVector(layer._getBaseVectorLength());
        if (layer.activationSource) {
            const tokenData = layer._getEmbeddingData({ tokenIndex: laneTokenIndex }, 'token');
            if (tokenData) data = tokenData;
        }
        startY = startY_override;
        originalVec = layer._createPrismVector(
            data,
            new THREE.Vector3(0, startY, zPos),
            30,
            layer._getInstanceCountFromData(data)
        );
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
    const multTarget = paramBanks && paramBanks.ln1Scale ? paramBanks.ln1Scale.getVectorRef(laneIdx) : null;
    const addTarget = paramBanks && paramBanks.ln1Shift ? paramBanks.ln1Shift.getVectorRef(laneIdx) : null;
    const multTargetLN2 = paramBanks && paramBanks.ln2Scale ? paramBanks.ln2Scale.getVectorRef(laneIdx) : null;
    const addTargetLN2 = paramBanks && paramBanks.ln2Shift ? paramBanks.ln2Shift.getVectorRef(laneIdx) : null;

    if (multTarget && multTarget.group) {
        multTarget.group.position.set(offsetX, ln1CenterY + 3.3, zPos);
        multTarget.group.visible = true;
    }
    if (addTarget && addTarget.group) {
        addTarget.group.position.set(offsetX, ln1CenterY + addYOffset, zPos);
        addTarget.group.visible = true;
    }
    if (multTargetLN2 && multTargetLN2.group) {
        multTargetLN2.group.position.set(offsetX, ln2CenterY + 3.3, zPos);
        multTargetLN2.group.visible = true;
    }
    if (addTargetLN2 && addTargetLN2.group) {
        addTargetLN2.group.position.set(offsetX, ln2CenterY + addYOffset, zPos);
        addTargetLN2.group.visible = true;
    }

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
        laneIndex: laneIdx,
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
        horizPhase: 'waiting',
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
        ln2Phase: 'notStarted',
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
            if (layer.activationSource) {
                const posEmbedding = layer._getEmbeddingData(lane, 'position');
                if (posEmbedding) posData = posEmbedding;
            }
            const posVec = layer._createPrismVector(
                posData,
                new THREE.Vector3(posStartX, posStartY, zPos),
                30,
                layer._getInstanceCountFromData(posData)
            );
            raycastRoot.add(posVec.group);
            applyVectorData(
                posVec,
                posData,
                lane.tokenLabel ? `Position Embedding - ${lane.tokenLabel}` : 'Position Embedding',
                layer._getLaneMeta(lane, 'embedding.position')
            );
            // Trail (local to this layer) - enabled only until it reaches residual stream.
            const posTrail = new StraightLineTrail(layer.root, 0xffffff, 1, undefined, undefined, TRAIL_MIN_SEGMENT_DISTANCE);
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

            // Two-phase motion: vertical rise, then perfectly horizontal slide.
            const targetYAbove = (startY_override != null ? startY_override : originalVec.group.position.y) + POS_VEC_Y_OFFSET_ABOVE_VOCAB;
            const fasterRise = ANIM_RISE_SPEED_ORIGINAL * POS_VEC_VERTICAL_SPEED_MULT;
            const riseDist = Math.max(0, targetYAbove - posStartY);
            const riseMs = (riseDist / (fasterRise * GLOBAL_ANIM_SPEED_MULT)) * 1000;

            const horizDist = Math.abs(posStartX - 0);
            const horizSpeed = ANIM_HORIZ_SPEED * POS_VEC_HORIZONTAL_SPEED_MULT * GLOBAL_ANIM_SPEED_MULT;
            const horizMs = (horizDist / horizSpeed) * 1000;
            if (typeof TWEEN !== 'undefined') {
                new TWEEN.Tween(posVec.group.position)
                    .to({ y: targetYAbove }, Math.max(100, riseMs))
                    .easing(TWEEN.Easing.Quadratic.InOut)
                    .onComplete(() => {
                        new TWEEN.Tween(posVec.group.position)
                            .to({ x: 0, y: targetYAbove }, Math.max(100, horizMs))
                            .easing(TWEEN.Easing.Quadratic.InOut)
                            .onStart(() => {
                                // Hard-lock Y during horizontal travel to ensure a perfectly straight path.
                                posVec.group.position.y = targetYAbove;
                            })
                            .onUpdate(() => {
                                // Maintain Y lock during horizontal interpolation.
                                posVec.group.position.y = targetYAbove;
                            })
                            .onComplete(() => {
                                if (posTrail && !posTrailDisposed) {
                                    posTrail.snapLastPointTo(posVec.group.position);
                                }
                                // Stop extending trail once we arrive at residual stream.
                                retirePosTrail();
                                // Trigger addition: positional (above) travels DOWN into vocab (rising).
                                try {
                                    const sumData = layer._getEmbeddingData(lane, 'sum');
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
                                    lane.posAddComplete = true;
                                }
                            })
                            .start();
                    })
                    .start();
            }
        } catch (_) {
            // Non-fatal - positional addition is a visual enhancement only.
        }
    }
}
