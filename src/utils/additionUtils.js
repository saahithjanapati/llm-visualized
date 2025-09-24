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

const ADDITION_ACCENT_COLORS = [
    new THREE.Color(0xffc857),
    new THREE.Color(0xff6f91),
    new THREE.Color(0x4ecdc4),
    new THREE.Color(0x9c6efb),
];
const WHITE = new THREE.Color(0xffffff);

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
    const baseWidthScale   = sourceVec.getWidthScale();
    const baseHeightScale  = sourceVec.getUniformHeight();
    const baseDepthScale   = sourceVec.getDepthScale();
    const baseScaleVec     = new THREE.Vector3(baseWidthScale, baseHeightScale, baseDepthScale);

    const targetWidthScale  = targetVec.getWidthScale();
    const targetHeightScale = targetVec.getUniformHeight();
    const targetDepthScale  = targetVec.getDepthScale();
    const targetBaseScale   = new THREE.Vector3(targetWidthScale, targetHeightScale, targetDepthScale);

    const anticipationDuration = duration * 0.28;
    const travelDuration       = duration - anticipationDuration;

    for (let i = 0; i < vectorLength; i++) {
        const srcLocalMatrix = new THREE.Matrix4();
        sourceVec.mesh.getMatrixAt(i, srcLocalMatrix);
        const srcLocalPos = new THREE.Vector3().setFromMatrixPosition(srcLocalMatrix);

        const initialOffset = srcLocalPos.y - basePrismCenterY;
        const gradCol = new THREE.Color();
        targetVec.mesh.getColorAt(i, gradCol);

        const accent = ADDITION_ACCENT_COLORS[i % ADDITION_ACCENT_COLORS.length];
        const anticipationColor = gradCol.clone().lerp(new THREE.Color(0xffffff), 0.35);
        const travelColor = gradCol.clone().lerp(accent, 0.45);
        const settleColor = gradCol.clone();

        const tempColor = new THREE.Color();
        const tempScale = new THREE.Vector3();
        const tempEuler = new THREE.Euler();
        const flashColor = new THREE.Color();
        const targetScale = new THREE.Vector3();
        const targetEuler = new THREE.Euler();

        const targetMatrix = new THREE.Matrix4();
        targetVec.mesh.getMatrixAt(i, targetMatrix);
        const targetWorld = new THREE.Vector3().setFromMatrixPosition(targetMatrix).applyMatrix4(targetVec.group.matrixWorld);
        const initialTargetLocal = sourceVec.group.worldToLocal(targetWorld.clone());
        let finalTargetY = initialTargetLocal.y;
        const direction = initialTargetLocal.y >= srcLocalPos.y ? 1 : -1;

        const anticipationOffset = -direction * baseHeightScale * 0.18;
        const overshootDistance = direction * baseHeightScale * 0.22;
        const followThroughDrift = direction * baseHeightScale * 0.12;

        const arcSeed = Math.random() * Math.PI * 2;
        const arcMagnitude = baseWidthScale * (0.35 + 0.15 * Math.sin(arcSeed + i));
        const swayMagnitude = baseWidthScale * 0.22;
        const swaySign = Math.random() > 0.5 ? 1 : -1;

        let launchY = srcLocalPos.y;

        const applyResidualTrailUpdate = () => {
            if (i !== centreIndex) return;
            try {
                const instMat = new THREE.Matrix4();
                sourceVec.mesh.getMatrixAt(centreIndex, instMat);
                const wPos = new THREE.Vector3().setFromMatrixPosition(instMat).applyMatrix4(sourceVec.group.matrixWorld);
                const hideThreshold = HIDE_INSTANCE_Y_OFFSET / 10;
                if (wPos.y < hideThreshold) return;

                const residualTrail = (lane && lane.originalTrail)
                    || (sourceVec && sourceVec.userData && sourceVec.userData.trail)
                    || null;
                const residualOwner = lane
                    || (sourceVec && sourceVec.userData)
                    || null;
                if (!residualTrail || !residualOwner || typeof residualTrail.update !== 'function') return;

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
            } catch (err) {
                console.warn('Residual trail update failed:', err);
            }
        };

        const anticipationTween = new TWEEN.Tween({ t: 0 })
            .to({ t: 1 }, anticipationDuration)
            .delay(i * delayBetween)
            .easing(TWEEN.Easing.Quadratic.Out)
            .onStart(() => {
                sourceVec.setInstanceAppearance(i, initialOffset, gradCol, baseScaleVec, { rotationEuler: tempEuler.set(0, 0, 0) });
            })
            .onUpdate(({ t }) => {
                const eased = TWEEN.Easing.Sinusoidal.Out(t);
                const anticipationY = srcLocalPos.y + anticipationOffset * eased;
                launchY = anticipationY;
                const offsetY = anticipationY - basePrismCenterY;

                const squashStretch = THREE.MathUtils.lerp(1, 0.72, eased);
                const preserve = 1 / Math.sqrt(Math.max(squashStretch, 0.001));
                tempScale.set(baseWidthScale * preserve, baseHeightScale * squashStretch, baseDepthScale * preserve);

                const sway = Math.sin(eased * Math.PI) * swayMagnitude * 0.35 * swaySign;
                const arc = Math.cos(eased * Math.PI) * arcMagnitude * 0.25;
                tempEuler.set(direction * 0.25 * eased, 0, swaySign * 0.2 * eased);

                tempColor.copy(gradCol).lerp(anticipationColor, eased);
                sourceVec.setInstanceAppearance(i, offsetY, tempColor, tempScale, {
                    xOffset: sway,
                    zOffset: arc,
                    rotationEuler: tempEuler,
                });
                applyResidualTrailUpdate();
            });

        const moveTween = new TWEEN.Tween({ t: 0 })
            .to({ t: 1 }, travelDuration)
            .delay(i * delayBetween + anticipationDuration)
            .easing(TWEEN.Easing.Cubic.Out)
            .onUpdate(({ t }) => {
                const trgLocalMatrixDyn = new THREE.Matrix4();
                targetVec.mesh.getMatrixAt(i, trgLocalMatrixDyn);
                const trgWorld = new THREE.Vector3().setFromMatrixPosition(trgLocalMatrixDyn).applyMatrix4(targetVec.group.matrixWorld);
                const trgLocal = sourceVec.group.worldToLocal(trgWorld.clone());

                const pathT = TWEEN.Easing.Sinusoidal.InOut(t);
                const targetY = trgLocal.y + overshootDistance;
                const interpY = THREE.MathUtils.lerp(launchY, targetY, pathT);
                const offsetY = interpY - basePrismCenterY;

                const stretch = THREE.MathUtils.lerp(1, 1.32, pathT);
                const preserve = 1 / Math.sqrt(Math.max(stretch, 0.001));
                tempScale.set(baseWidthScale * preserve, baseHeightScale * stretch, baseDepthScale * preserve);

                const sway = Math.sin(pathT * Math.PI * 1.5 + arcSeed) * swayMagnitude * 0.9 * swaySign;
                const arc = Math.sin(pathT * Math.PI) * arcMagnitude;
                tempEuler.set(direction * 0.18 * Math.sin(pathT * Math.PI * 0.9), 0, swaySign * 0.25 * Math.sin(pathT * Math.PI));

                tempColor.copy(gradCol).lerp(travelColor, pathT);
                sourceVec.setInstanceAppearance(i, offsetY, tempColor, tempScale, {
                    xOffset: sway,
                    zOffset: arc,
                    rotationEuler: tempEuler,
                });

                finalTargetY = trgLocal.y;
                applyResidualTrailUpdate();
            })
            .onComplete(() => {
                const settleTween = new TWEEN.Tween({ t: 0 })
                    .to({ t: 1 }, flashDuration)
                    .easing(TWEEN.Easing.Sinusoidal.Out)
                    .onUpdate(({ t }) => {
                        const trgLocalMatrixDyn = new THREE.Matrix4();
                        targetVec.mesh.getMatrixAt(i, trgLocalMatrixDyn);
                        const trgWorld = new THREE.Vector3().setFromMatrixPosition(trgLocalMatrixDyn).applyMatrix4(targetVec.group.matrixWorld);
                        const trgLocal = sourceVec.group.worldToLocal(trgWorld.clone());

                        const eased = TWEEN.Easing.Sinusoidal.Out(t);
                        const bounce = Math.sin((1 - eased) * Math.PI) * followThroughDrift * (1 - eased);
                        const currentY = THREE.MathUtils.lerp(trgLocal.y + overshootDistance, trgLocal.y, eased) + bounce;
                        const offsetY = currentY - basePrismCenterY;

                        const stretch = THREE.MathUtils.lerp(1.15, 0.9, eased);
                        const preserve = 1 / Math.sqrt(Math.max(stretch, 0.001));
                        tempScale.set(baseWidthScale * preserve, baseHeightScale * stretch, baseDepthScale * preserve);

                        const sway = swayMagnitude * 0.6 * (1 - eased) * swaySign;
                        const arc = arcMagnitude * 0.3 * (1 - eased);
                        tempEuler.set(direction * 0.12 * (1 - eased), 0, swaySign * 0.18 * (1 - eased));

                        tempColor.copy(travelColor).lerp(settleColor, eased);
                        sourceVec.setInstanceAppearance(i, offsetY, tempColor, tempScale, {
                            xOffset: sway,
                            zOffset: arc,
                            rotationEuler: tempEuler,
                        });
                        finalTargetY = trgLocal.y;
                        applyResidualTrailUpdate();
                    })
                    .onComplete(() => {
                        tempEuler.set(0, 0, 0);
                        sourceVec.setInstanceAppearance(i, finalTargetY - basePrismCenterY, gradCol, baseScaleVec, {
                            xOffset: 0,
                            zOffset: 0,
                            rotationEuler: tempEuler,
                        });
                    });

                const flashTween = new TWEEN.Tween({ t: 0 })
                    .to({ t: 1 }, flashDuration)
                    .easing(TWEEN.Easing.Sinusoidal.InOut)
                    .onStart(() => {
                        flashColor.copy(WHITE);
                        targetVec.setInstanceAppearance(i, 0, flashColor, targetBaseScale, {
                            rotationEuler: targetEuler.set(0, 0, 0),
                        });
                    })
                    .onUpdate(({ t }) => {
                        const pulse = 1 + 0.3 * Math.sin(t * Math.PI);
                        const preserve = 1 / Math.sqrt(Math.max(pulse, 0.001));
                        targetScale.set(targetWidthScale * preserve, targetHeightScale * pulse, targetDepthScale * preserve);
                        flashColor.copy(WHITE).lerp(accent, t * 0.6);
                        targetEuler.set(0, 0, swaySign * 0.18 * Math.sin(t * Math.PI));
                        targetVec.setInstanceAppearance(i, 0, flashColor, targetScale, {
                            rotationEuler: targetEuler,
                        });
                    })
                    .onComplete(() => {
                        const sum = (sourceVec.rawData[i] ?? 0) + (targetVec.rawData[i] ?? 0);
                        targetVec.rawData[i] = sum;
                        targetVec.setInstanceAppearance(i, 0, gradCol, targetBaseScale, {
                            rotationEuler: targetEuler.set(0, 0, 0),
                        });
                        sourceVec.setInstanceAppearance(i, HIDE_INSTANCE_Y_OFFSET, null);
                    });

                settleTween.start();
                flashTween.start();
            });

        anticipationTween.start();
        moveTween.start();
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