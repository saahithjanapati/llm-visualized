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

// Animation style constants inspired by the 12 principles of animation.
const ADDITION_ANTICIPATION_RATIO = 0.28;          // Portion of travel spent building anticipation
const ADDITION_FOLLOW_THROUGH_RATIO = 0.4;         // Portion reserved for settle / follow-through
const ADDITION_SQUASH_INTENSITY = 0.22;            // Squash factor during anticipation
const ADDITION_STRETCH_INTENSITY = 0.3;            // Stretch factor during main motion
const ADDITION_ARC_BASE = 12;                      // Base horizontal sway magnitude (world units)
const ADDITION_ARC_VARIANCE = 6;                   // Extra sway towards the edges for stronger arcs
const ADDITION_ROTATION_MAX_RAD = THREE.MathUtils.degToRad(7.5);
const ADDITION_SECONDARY_COLOR = new THREE.Color(1.0, 0.9, 0.7); // Warm highlight for staging/appeal
const ADDITION_FLASH_COLOR = new THREE.Color(1.0, 0.98, 0.92);

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

    const anticipationDuration   = Math.max(80, duration * ADDITION_ANTICIPATION_RATIO);
    const followThroughDuration  = Math.max(140, duration * ADDITION_FOLLOW_THROUGH_RATIO);
    const holdDuration           = Math.max(60, flashDuration);

    const basePrismCenterY = sourceVec.getUniformHeight() / 2;
    const defaultScale = new THREE.Vector3(
        typeof sourceVec.getWidthScale === 'function' ? sourceVec.getWidthScale() : 1,
        typeof sourceVec.getUniformHeight === 'function' ? sourceVec.getUniformHeight() : 1,
        typeof sourceVec.getDepthScale === 'function' ? sourceVec.getDepthScale() : 1
    );
    const targetDefaultScale = new THREE.Vector3(
        typeof targetVec.getWidthScale === 'function' ? targetVec.getWidthScale() : defaultScale.x,
        typeof targetVec.getUniformHeight === 'function' ? targetVec.getUniformHeight() : defaultScale.y,
        typeof targetVec.getDepthScale === 'function' ? targetVec.getDepthScale() : defaultScale.z
    );

    for (let i = 0; i < vectorLength; i++) {
        const srcLocalMatrix = new THREE.Matrix4();
        sourceVec.mesh.getMatrixAt(i, srcLocalMatrix);
        const srcLocalPos = new THREE.Vector3().setFromMatrixPosition(srcLocalMatrix);

        // Capture target gradient colour so we can flash & restore
        const gradCol = new THREE.Color();
        targetVec.mesh.getColorAt(i, gradCol);

        // Match travelling prism colour to destination gradient for initial staging
        const startOffset = srcLocalPos.y - basePrismCenterY;
        sourceVec.setInstanceAppearance(i, startOffset, gradCol);

        const arcDirection = (i - centreIndex) / Math.max(1, centreIndex);
        const swayDirection = arcDirection === 0 ? (i % 2 === 0 ? 1 : -1) : Math.sign(arcDirection);
        const arcMagnitude = swayDirection * (ADDITION_ARC_BASE + Math.abs(arcDirection) * ADDITION_ARC_VARIANCE);
        const anticipationDip = defaultScale.y * 0.08;

        const anticipationState = { t: 0 };
        const anticipationScale = new THREE.Vector3();
        const anticipationColor = gradCol.clone();
        const anticipationEuler = new THREE.Euler();

        const runMainTween = () => {
            const mainState = { t: 0 };
            const stretchScale = new THREE.Vector3();
            const rotationEuler = new THREE.Euler();

                    new TWEEN.Tween(mainState)
                .to({ t: 1 }, duration)
                .easing(TWEEN.Easing.Cubic.InOut)
                .onStart(() => {
                    // Restore base colour as the prism launches upward
                    sourceVec.setInstanceAppearance(i, startOffset, gradCol, defaultScale);
                })
                .onUpdate(obj => {
                    const trgLocalMatrixDyn = new THREE.Matrix4();
                    targetVec.mesh.getMatrixAt(i, trgLocalMatrixDyn);
                    const trgWorld = new THREE.Vector3().setFromMatrixPosition(trgLocalMatrixDyn).applyMatrix4(targetVec.group.matrixWorld);
                    const trgLocal = sourceVec.group.worldToLocal(trgWorld.clone());

                    const eased = TWEEN.Easing.Back.Out(obj.t);
                    const interpY = THREE.MathUtils.lerp(srcLocalPos.y, trgLocal.y, eased);
                    const offsetY = interpY - basePrismCenterY;

                    const stretchAmount = Math.sin(obj.t * Math.PI);
                    stretchScale.copy(defaultScale);
                    stretchScale.x = defaultScale.x * (1 - ADDITION_STRETCH_INTENSITY * 0.55 * stretchAmount);
                    stretchScale.y = defaultScale.y * (1 + ADDITION_STRETCH_INTENSITY * stretchAmount);
                    stretchScale.z = defaultScale.z * (1 - ADDITION_STRETCH_INTENSITY * 0.35 * stretchAmount);

                    const arcT = Math.sin(obj.t * Math.PI);
                    const xOffset = arcMagnitude * arcT;
                    const tilt = ADDITION_ROTATION_MAX_RAD * arcT * swayDirection;
                    rotationEuler.set(0, 0, tilt);

                    sourceVec.setInstanceAppearance(i, offsetY, null, stretchScale, { xOffset, rotationEuler });

                    if (i === centreIndex) {
                        try {
                            const instMat = new THREE.Matrix4();
                            sourceVec.mesh.getMatrixAt(centreIndex, instMat);
                            const wPos = new THREE.Vector3().setFromMatrixPosition(instMat).applyMatrix4(sourceVec.group.matrixWorld);

                            const hideThreshold = HIDE_INSTANCE_Y_OFFSET / 10;
                            if (wPos.y >= hideThreshold) {
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
                    const flashState = { t: 0 };
                    const flashScale = new THREE.Vector3();
                    const followScale = new THREE.Vector3();
                    const settleEuler = new THREE.Euler();

                    new TWEEN.Tween(flashState)
                        .to({ t: 1 }, holdDuration)
                        .easing(TWEEN.Easing.Quadratic.Out)
                        .onStart(() => {
                            flashScale.copy(targetDefaultScale);
                            targetVec.setInstanceAppearance(i, 0, ADDITION_FLASH_COLOR, flashScale);
                        })
                        .onUpdate(state => {
                            const shimmer = Math.sin(state.t * Math.PI * 2) * 0.05;
                            flashScale.copy(targetDefaultScale);
                            flashScale.y = targetDefaultScale.y * (1 + shimmer);
                            flashScale.x = targetDefaultScale.x * (1 - shimmer * 0.4);
                            targetVec.setInstanceAppearance(i, 0, ADDITION_FLASH_COLOR, flashScale);
                        })
                        .onComplete(() => {
                            const sum = (sourceVec.rawData[i] ?? 0) + (targetVec.rawData[i] ?? 0);
                            targetVec.rawData[i] = sum;

                            new TWEEN.Tween({ t: 0 })
                                .to({ t: 1 }, followThroughDuration)
                                .easing(TWEEN.Easing.Elastic.Out)
                                .onStart(() => {
                                    sourceVec.setInstanceAppearance(i, HIDE_INSTANCE_Y_OFFSET, null, defaultScale);
                                })
                                .onUpdate(({ t }) => {
                                    const settleStretch = (1 - TWEEN.Easing.Cubic.Out(t)) * 0.35;
                                    followScale.copy(targetDefaultScale);
                                    followScale.x = targetDefaultScale.x * (1 - settleStretch * 0.5);
                                    followScale.y = targetDefaultScale.y * (1 + settleStretch);
                                    followScale.z = targetDefaultScale.z * (1 - settleStretch * 0.3);
                                    settleEuler.set(0, 0, 0);
                                    const settleColor = gradCol.clone().lerp(ADDITION_SECONDARY_COLOR, (1 - t) * 0.4);
                                    targetVec.setInstanceAppearance(i, 0, settleColor, followScale, { rotationEuler: settleEuler });
                                })
                                .onComplete(() => {
                                    targetVec.setInstanceAppearance(i, 0, gradCol, targetDefaultScale);
                                })
                                .start();
                        })
                        .start();
                })
                .start();
        };

        new TWEEN.Tween(anticipationState)
            .to({ t: 1 }, anticipationDuration)
            .delay(i * delayBetween)
            .easing(TWEEN.Easing.Quadratic.Out)
            .onUpdate(obj => {
                const squash = TWEEN.Easing.Quadratic.Out(obj.t);
                anticipationScale.copy(defaultScale);
                anticipationScale.x = defaultScale.x * (1 + ADDITION_SQUASH_INTENSITY * squash);
                anticipationScale.y = defaultScale.y * (1 - ADDITION_SQUASH_INTENSITY * squash);
                anticipationScale.z = defaultScale.z * (1 + ADDITION_SQUASH_INTENSITY * 0.6 * squash);

                const dip = anticipationDip * squash;
                anticipationColor.copy(gradCol).lerp(ADDITION_SECONDARY_COLOR, 0.35 * squash);
                anticipationEuler.set(0, 0, 0);
                sourceVec.setInstanceAppearance(i, startOffset - dip, anticipationColor, anticipationScale, { rotationEuler: anticipationEuler });
            })
            .onComplete(() => {
                runMainTween();
            })
            .start();
    }

    const totalAnimTime = anticipationDuration + duration + holdDuration + followThroughDuration + vectorLength * delayBetween;
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