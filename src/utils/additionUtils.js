// Utility: Addition animation between two VectorVisualizationInstancedPrism vectors
// -----------------------------------------------------------------------------
// This helper consolidates the formerly duplicated "addition" logic so any
// module can trigger a residual-style add (vec1 += vec2) with trails and
// InstancedMesh visuals.  It is effectively a verbatim extraction of
// MHSAAnimation._startAdditionAnimation but parameterised to avoid a hard
// dependency on the MHSAAnimation class.
// -----------------------------------------------------------------------------
import * as THREE from 'three';



import {
    VECTOR_LENGTH_PRISM,
    PRISM_ADD_ANIM_BASE_DURATION,
    PRISM_ADD_ANIM_BASE_FLASH_DURATION,
    PRISM_ADD_ANIM_BASE_DELAY_BETWEEN_PRISMS,
    PRISM_ADD_ANIM_SPEED_MULT,
    HIDE_INSTANCE_Y_OFFSET,
} from './constants.js';

const ANTICIPATION_PORTION = 0.2;
const FOLLOW_THROUGH_PORTION = 0.25;
const TRAVEL_PORTION = Math.max(0.0001, 1 - ANTICIPATION_PORTION - FOLLOW_THROUGH_PORTION);
const BASE_SQUASH_FACTOR = 0.28;
const MIN_HEIGHT_SCALE = 0.55;
const ARC_SWAY_NOISE = 0.35;
const SECONDARY_WIGGLE_MULT = 0.28;
const ADDITION_ACCENT_COLOR = new THREE.Color(1.0, 0.92, 0.6);

/**
 * Animate element-wise addition of two instanced-prism vectors, visually moving
 * each prism from `sourceVec` into `targetVec` while updating colours & data.
 *
 * @param {VectorVisualizationInstancedPrism} sourceVec – travelling vector
 * @param {VectorVisualizationInstancedPrism} targetVec – stationary vector that will hold the sum
 * @param {Object} [lane]             – optional lane object; if provided the helper
*                                      will update lane fields
 */
export function startPrismAdditionAnimation(sourceVec, targetVec, lane, onComplete) {
    if (!sourceVec || !targetVec || !sourceVec.mesh || !targetVec.mesh) return;
    if (typeof TWEEN === 'undefined') {
        console.warn('TWEEN not available – addition animation skipped');
        return;
    }

    // Ensure a metadata container exists so we can store residual trail state
    // even when no lane object is available (e.g. LayerNorm additions create
    // temporary vectors without full lane context).
    sourceVec.userData = sourceVec.userData || {};

    // Freeze upward movement of the source so its group position remains static.
    if (lane) {
        lane.stopRise = true;
        lane.stopRiseTarget = targetVec.group;
        // Keep residual trail brightness unchanged during addition
        // Reset residual trail monotonic tracker to the centre prism's current Y
        // so the trail extends immediately as the middle unit begins to rise.
        try {
            const centreIndex = Math.floor(VECTOR_LENGTH_PRISM / 2);
            const instMat = new THREE.Matrix4();
            sourceVec.mesh.getMatrixAt(centreIndex, instMat);
            const centreWorld = new THREE.Vector3().setFromMatrixPosition(instMat).applyMatrix4(sourceVec.group.matrixWorld);
            lane.__residualMaxY = (typeof centreWorld.y === 'number') ? centreWorld.y - 0.001 : undefined;
        } catch (err) {
            console.warn('Failed to init residual trail:', err);
        }
    } else {
        sourceVec.group.userData = sourceVec.group.userData || {};
        const svUD = sourceVec.group.userData;
        svUD.stopRise = true;
        svUD.stopRiseTarget = targetVec.group;
        try {
            const centreIndex = Math.floor(VECTOR_LENGTH_PRISM / 2);
            const instMat = new THREE.Matrix4();
            sourceVec.mesh.getMatrixAt(centreIndex, instMat);
            const centreWorld = new THREE.Vector3().setFromMatrixPosition(instMat).applyMatrix4(sourceVec.group.matrixWorld);
            sourceVec.userData.__residualMaxY =
                (typeof centreWorld.y === 'number') ? centreWorld.y - 0.001 : undefined;
        } catch (err) {
            console.warn('Failed to init residual trail:', err);
        }
    }

    const vectorLength = VECTOR_LENGTH_PRISM;
    const centreIndex = Math.floor(vectorLength / 2);





    // Timing params (scale by dedicated multiplier so we can tune independently).
    const duration      = PRISM_ADD_ANIM_BASE_DURATION            / PRISM_ADD_ANIM_SPEED_MULT;
    const flashDuration = PRISM_ADD_ANIM_BASE_FLASH_DURATION      / PRISM_ADD_ANIM_SPEED_MULT;
    const delayBetween  = PRISM_ADD_ANIM_BASE_DELAY_BETWEEN_PRISMS/ PRISM_ADD_ANIM_SPEED_MULT;

    const basePrismCenterY = sourceVec.getUniformHeight() / 2;

    for (let i = 0; i < vectorLength; i++) {
        const srcLocalMatrix = new THREE.Matrix4();
        sourceVec.mesh.getMatrixAt(i, srcLocalMatrix);
        const srcLocalPos = new THREE.Vector3().setFromMatrixPosition(srcLocalMatrix);
        const srcBaseScale = new THREE.Vector3();
        srcLocalMatrix.decompose(new THREE.Vector3(), new THREE.Quaternion(), srcBaseScale);

        const gradCol = new THREE.Color();
        targetVec.mesh.getColorAt(i, gradCol);
        const highlightColor = gradCol.clone().lerp(ADDITION_ACCENT_COLOR, 0.45);

        const initialYOffset = srcLocalPos.y - basePrismCenterY;
        sourceVec.setInstanceAppearance(i, initialYOffset, highlightColor);

        const targetInitialMatrix = new THREE.Matrix4();
        targetVec.mesh.getMatrixAt(i, targetInitialMatrix);
        const targetInitialWorld = new THREE.Vector3().setFromMatrixPosition(targetInitialMatrix).applyMatrix4(targetVec.group.matrixWorld);
        const targetInitialLocal = targetInitialWorld.clone();
        sourceVec.group.worldToLocal(targetInitialLocal);

        const baseTravelDistance = targetInitialLocal.y - srcLocalPos.y;
        let travelDirection = Math.sign(baseTravelDistance);
        if (travelDirection === 0) travelDirection = 1;

        const baseHeight = srcBaseScale.y || sourceVec.getUniformHeight();
        const anticipationMagnitude = Math.min(baseHeight * 0.5, Math.max(baseHeight * 0.08, Math.abs(baseTravelDistance) * 0.18 + baseHeight * 0.02));
        const overshootMagnitudeAbs = Math.min(baseHeight * 0.45, Math.max(baseHeight * 0.08, Math.abs(baseTravelDistance) * 0.14 + baseHeight * 0.01));
        const arcMagnitude = Math.min(baseHeight * 0.35, Math.max(baseHeight * 0.05, Math.abs(baseTravelDistance) * 0.12 + baseHeight * 0.01));

        const anticipationOffset = -travelDirection * anticipationMagnitude;
        const overshootOffset = travelDirection * overshootMagnitudeAbs;
        const travelStartY = srcLocalPos.y + anticipationOffset;
        const arcPhase = (i / Math.max(1, vectorLength - 1)) * Math.PI;
        const arcSign = (i % 2 === 0) ? 1 : -1;

        const dynamicColor = new THREE.Color();
        const dynamicScale = new THREE.Vector3();
        const offsetVec = new THREE.Vector3();
        const targetMatrixDyn = new THREE.Matrix4();
        const targetWorldPos = new THREE.Vector3();
        const targetLocalPos = new THREE.Vector3();

        const tweenState = { t: 0 };

        new TWEEN.Tween(tweenState)
            .to({ t: 1 }, duration)
            .delay(i * delayBetween)
            .easing(TWEEN.Easing.Quadratic.InOut)
            .onUpdate(obj => {
                targetVec.mesh.getMatrixAt(i, targetMatrixDyn);
                targetWorldPos.setFromMatrixPosition(targetMatrixDyn).applyMatrix4(targetVec.group.matrixWorld);
                targetLocalPos.copy(targetWorldPos);
                sourceVec.group.worldToLocal(targetLocalPos);

                const targetY = targetLocalPos.y;
                const overshootTargetY = targetY + overshootOffset;
                const finalTargetY = targetY;

                let desiredY = srcLocalPos.y;
                let arcOffsetZ = 0;
                let arcOffsetX = 0;
                let heightScale = 1;
                let colorBlend = 0;

                if (obj.t < ANTICIPATION_PORTION) {
                    const stageT = obj.t / ANTICIPATION_PORTION;
                    const ease = TWEEN.Easing.Quadratic.Out(stageT);
                    desiredY = srcLocalPos.y + anticipationOffset * ease;
                    heightScale = Math.max(MIN_HEIGHT_SCALE, 1 - BASE_SQUASH_FACTOR * ease);
                    const swing = Math.sin(ease * Math.PI);
                    arcOffsetZ = swing * arcMagnitude * 0.2 * arcSign;
                    arcOffsetX = Math.sin((ease + arcPhase) * Math.PI) * arcMagnitude * 0.15 * arcSign;
                    colorBlend = 0.25 + 0.35 * ease;
                } else if (obj.t < ANTICIPATION_PORTION + TRAVEL_PORTION) {
                    const stageT = (obj.t - ANTICIPATION_PORTION) / TRAVEL_PORTION;
                    const ease = TWEEN.Easing.Cubic.InOut(stageT);
                    desiredY = THREE.MathUtils.lerp(travelStartY, overshootTargetY, ease);
                    heightScale = Math.max(MIN_HEIGHT_SCALE, 1 + BASE_SQUASH_FACTOR * TWEEN.Easing.Quadratic.Out(stageT));
                    const swing = Math.sin(ease * Math.PI);
                    arcOffsetZ = swing * arcMagnitude * arcSign;
                    arcOffsetX = Math.sin((ease + ARC_SWAY_NOISE + arcPhase) * Math.PI) * arcMagnitude * 0.35 * arcSign;
                    colorBlend = 0.55 + 0.35 * ease;
                } else {
                    const stageT = (obj.t - ANTICIPATION_PORTION - TRAVEL_PORTION) / FOLLOW_THROUGH_PORTION;
                    const ease = TWEEN.Easing.Bounce.Out(stageT);
                    desiredY = THREE.MathUtils.lerp(overshootTargetY, finalTargetY, ease);
                    heightScale = Math.max(MIN_HEIGHT_SCALE, THREE.MathUtils.lerp(1 + BASE_SQUASH_FACTOR * 0.2, 1, ease));
                    const wiggle = Math.sin((stageT * (3 + (i % 3))) * Math.PI + arcPhase) * (1 - ease) * arcMagnitude * SECONDARY_WIGGLE_MULT * arcSign;
                    arcOffsetZ = wiggle;
                    arcOffsetX = Math.sin((stageT + arcPhase) * Math.PI * 2) * (1 - ease) * arcMagnitude * 0.2 * arcSign;
                    colorBlend = (1 - ease) * 0.4;
                }

                const widthFactor = 1 / Math.sqrt(Math.max(0.01, heightScale));
                dynamicScale.set(
                    srcBaseScale.x * widthFactor,
                    srcBaseScale.y * heightScale,
                    srcBaseScale.z * widthFactor
                );

                const offsetY = desiredY - basePrismCenterY;
                const colorMix = Math.min(1, Math.max(0, colorBlend));
                dynamicColor.copy(gradCol).lerp(highlightColor, colorMix);
                offsetVec.set(arcOffsetX, 0, arcOffsetZ);

                sourceVec.setInstanceAppearance(i, offsetY, dynamicColor, dynamicScale, offsetVec);

                if (i === centreIndex) {
                    // Live-update the residual trail from the bottom vector while the
                    // centre prism rises toward the top vector. This mirrors the
                    // behaviour users expect: the connecting line grows as the
                    // middle unit moves, rather than appearing only after addition.
                    try {
                        const instMat = new THREE.Matrix4();
                        sourceVec.mesh.getMatrixAt(centreIndex, instMat);
                        const wPos = new THREE.Vector3().setFromMatrixPosition(instMat).applyMatrix4(sourceVec.group.matrixWorld);

                        // Skip if the prism is effectively hidden far below
                        const hideThreshold = HIDE_INSTANCE_Y_OFFSET / 10;
                        if (wPos.y >= hideThreshold) {
                            // Update the residual trail continuously as the prism rises all the
                            // way to the target vector. Previously, a muted band near the
                            // merge point prevented trailing up to the very top, leaving a
                            // visible gap.
                            const residualTrail = (lane && lane.originalTrail)
                                || (sourceVec && sourceVec.userData && sourceVec.userData.trail)
                                || null;
                            const residualOwner = lane
                                || (sourceVec && sourceVec.userData)
                                || null;
                            if (residualTrail && residualOwner && typeof residualTrail.update === 'function') {
                                if (typeof residualOwner.__residualMaxY !== 'number') {
                                    residualOwner.__residualMaxY = wPos.y - 0.001;
                                }
                                if (wPos.y >= residualOwner.__residualMaxY) {
                                    let localPos = wPos;
                                    try {
                                        const parentObject = (residualTrail._line && residualTrail._line.parent)
                                            || residualTrail._scene
                                            || null;
                                        if (parentObject && typeof parentObject.worldToLocal === 'function') {
                                            localPos = parentObject.worldToLocal(wPos.clone());
                                        }
                                    } catch (conversionErr) {
                                        console.warn('Residual trail coordinate conversion failed:', conversionErr);
                                        localPos = wPos;
                                    }
                                    residualTrail.update(localPos);
                                    residualOwner.__residualMaxY = wPos.y;
                                }
                            }
                        }
                    } catch (err) {
                        console.warn('Residual trail update failed:', err);
                    }
                }
            })
            .onComplete(() => {
                targetVec.setInstanceAppearance(i, 0, new THREE.Color(0xffffff));
                new TWEEN.Tween({})
                    .to({}, flashDuration)
                    .onComplete(() => {
                        const sum = (sourceVec.rawData[i] ?? 0) + (targetVec.rawData[i] ?? 0);
                        targetVec.rawData[i] = sum;
                        targetVec.setInstanceAppearance(i, 0, gradCol);
                        sourceVec.setInstanceAppearance(i, HIDE_INSTANCE_Y_OFFSET, null);
                    })
                    .start();
            })
            .start();
    }

    const totalAnimTime = duration + flashDuration + vectorLength * delayBetween;
    setTimeout(() => {
        if (lane) {
            delete lane.stopRise;
            delete lane.stopRiseTarget;
        } else if (sourceVec && sourceVec.group && sourceVec.group.userData) {
            delete sourceVec.group.userData.stopRise;
            delete sourceVec.group.userData.stopRiseTarget;
        }

        if (lane) {
            lane.originalVec    = targetVec;
            lane.postAdditionVec= targetVec;
            if (lane.ln2Phase !== 'done') {
                lane.ln2Phase = 'preRise';
                if (lane.layer && typeof lane.layer._emitProgress === 'function') lane.layer._emitProgress();
                // Set horizPhase to trigger LN2 pipeline
                if (lane.horizPhase === 'travelMHSA' || lane.horizPhase === 'finishedHeads') {
                    lane.horizPhase = 'postMHSAAddition';
                    if (lane.layer && typeof lane.layer._emitProgress === 'function') lane.layer._emitProgress();
                }
            }
            const topY = targetVec.group.position.y;
        }
        if (typeof onComplete === 'function') {
            try {
                onComplete();
            } catch (err) {
                console.warn('Addition completion callback failed:', err);
            }
        }
    }, totalAnimTime + 100);
}