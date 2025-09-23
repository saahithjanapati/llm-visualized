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

    const uniformHeight = (typeof sourceVec.getUniformHeight === 'function')
        ? sourceVec.getUniformHeight()
        : 0;
    const basePrismCenterY = uniformHeight / 2;
    const baseWidthScale = (typeof sourceVec.getWidthScale === 'function')
        ? sourceVec.getWidthScale()
        : 1;
    const baseDepthScale = (typeof sourceVec.getDepthScale === 'function')
        ? sourceVec.getDepthScale()
        : 1;
    const baseWidthConstant = (typeof sourceVec.getBaseWidthConstant === 'function')
        ? sourceVec.getBaseWidthConstant()
        : 1;
    const targetUniformHeight = (typeof targetVec.getUniformHeight === 'function')
        ? targetVec.getUniformHeight()
        : uniformHeight;
    const targetWidthScale = (typeof targetVec.getWidthScale === 'function')
        ? targetVec.getWidthScale()
        : baseWidthScale;
    const targetDepthScale = (typeof targetVec.getDepthScale === 'function')
        ? targetVec.getDepthScale()
        : baseDepthScale;

    const warmHighlight = new THREE.Color(0xfff2cc);
    const whiteColor = new THREE.Color(0xffffff);

    for (let i = 0; i < vectorLength; i++) {
        const isCentre = i === centreIndex;

        // Grab starting local Y offset of each instance
        const srcLocalMatrix = new THREE.Matrix4();
        sourceVec.mesh.getMatrixAt(i, srcLocalMatrix);
        const srcLocalPos = new THREE.Vector3().setFromMatrixPosition(srcLocalMatrix);

        // Capture target gradient colour so we can flash & restore
        const gradCol = new THREE.Color();
        targetVec.mesh.getColorAt(i, gradCol);

        // Match travelling prism colour to destination gradient
        sourceVec.setInstanceAppearance(i, srcLocalPos.y, gradCol);

        const arcDirection = i % 2 === 0 ? 1 : -1;
        const arcMagnitude = baseWidthConstant * baseWidthScale * (0.45 + Math.random() * 0.35);
        const anticipationDrop = Math.min(uniformHeight * (0.18 + Math.random() * 0.08), uniformHeight * 0.45);
        const startDownY = srcLocalPos.y - anticipationDrop;
        const wobblePhase = Math.random() * Math.PI * 2;

        const accentColor = gradCol.clone().lerp(warmHighlight, 0.55);
        const displayColor = accentColor.clone();
        const sparkleColor = new THREE.Color().copy(whiteColor);

        const animatedScale = new THREE.Vector3(baseWidthScale, uniformHeight, baseDepthScale);
        const targetBaseScale = new THREE.Vector3(targetWidthScale, targetUniformHeight, targetDepthScale);
        const targetPulseScale = targetBaseScale.clone();
        const flashColor = new THREE.Color();

        const trgLocalMatrixDyn = new THREE.Matrix4();
        const trgWorld = new THREE.Vector3();
        const trgLocal = new THREE.Vector3();
        const residualInstMat = new THREE.Matrix4();
        const residualWorldPos = new THREE.Vector3();
        const residualLocalPos = new THREE.Vector3();

        const getDynamicTargetLocalY = () => {
            targetVec.mesh.getMatrixAt(i, trgLocalMatrixDyn);
            trgWorld.setFromMatrixPosition(trgLocalMatrixDyn).applyMatrix4(targetVec.group.matrixWorld);
            trgLocal.copy(trgWorld);
            sourceVec.group.worldToLocal(trgLocal);
            return trgLocal.y;
        };

        const updateResidualTrail = () => {
            if (!isCentre) return;
            try {
                sourceVec.mesh.getMatrixAt(centreIndex, residualInstMat);
                residualWorldPos.setFromMatrixPosition(residualInstMat).applyMatrix4(sourceVec.group.matrixWorld);

                const hideThreshold = HIDE_INSTANCE_Y_OFFSET / 10;
                if (residualWorldPos.y >= hideThreshold) {
                    const residualTrail = (lane && lane.originalTrail)
                        || (sourceVec && sourceVec.userData && sourceVec.userData.trail)
                        || null;
                    const residualOwner = lane
                        || (sourceVec && sourceVec.userData)
                        || null;
                    if (residualTrail && residualOwner && typeof residualTrail.update === 'function') {
                        if (typeof residualOwner.__residualMaxY !== 'number') {
                            residualOwner.__residualMaxY = residualWorldPos.y - 0.001;
                        }
                        if (residualWorldPos.y >= residualOwner.__residualMaxY) {
                            residualLocalPos.copy(residualWorldPos);
                            try {
                                const parentObject = (residualTrail._line && residualTrail._line.parent)
                                    || residualTrail._scene
                                    || null;
                                if (parentObject && typeof parentObject.worldToLocal === 'function') {
                                    parentObject.worldToLocal(residualLocalPos);
                                }
                            } catch (conversionErr) {
                                console.warn('Residual trail coordinate conversion failed:', conversionErr);
                            }
                            residualTrail.update(residualLocalPos.clone());
                            residualOwner.__residualMaxY = residualWorldPos.y;
                        }
                    }
                }
            } catch (err) {
                console.warn('Residual trail update failed:', err);
            }
        };

        const timelineState = { progress: 0 };

        new TWEEN.Tween(timelineState)
            .to({ progress: 1 }, duration)
            .delay(i * delayBetween)
            .easing(TWEEN.Easing.Linear.None)
            .onUpdate(({ progress }) => {
                const anticipationPortion = 0.22;
                const clampedProgress = THREE.MathUtils.clamp(progress, 0, 1);

                let currentY = srcLocalPos.y;
                let xOffset = 0;
                let tilt = 0;

                if (clampedProgress < anticipationPortion) {
                    const localT = clampedProgress / anticipationPortion;
                    const eased = TWEEN.Easing.Sine.InOut(localT);
                    currentY = srcLocalPos.y - anticipationDrop * eased;

                    const widthPulse = 1 + 0.25 * eased;
                    const heightSquash = 1 - 0.35 * eased;
                    animatedScale.set(
                        baseWidthScale * widthPulse,
                        uniformHeight * heightSquash,
                        baseDepthScale * widthPulse
                    );

                    xOffset = arcDirection * arcMagnitude * eased * 0.35;
                    tilt = arcDirection * 0.18 * eased;

                    displayColor.copy(accentColor);
                    const sparkle = Math.sin(eased * Math.PI) * 0.2;
                    if (sparkle > 0) {
                        displayColor.lerp(sparkleColor, sparkle * 0.25);
                    }
                } else {
                    const travelT = (clampedProgress - anticipationPortion) / (1 - anticipationPortion);
                    const easedBase = TWEEN.Easing.Sine.InOut(travelT);
                    const overshootEase = TWEEN.Easing.Back.Out(easedBase);
                    const targetY = getDynamicTargetLocalY();
                    currentY = THREE.MathUtils.lerp(startDownY, targetY, overshootEase);

                    const arc = Math.sin(easedBase * Math.PI);
                    const sway = Math.sin(easedBase * Math.PI + wobblePhase);
                    xOffset = arcDirection * arcMagnitude * arc;
                    tilt = arcDirection * (0.28 * arc + 0.1 * sway);

                    const stretch = 1 + 0.25 * Math.sin(easedBase * Math.PI);
                    const squash = 1 - 0.18 * Math.sin(easedBase * Math.PI);
                    animatedScale.set(
                        baseWidthScale * squash,
                        uniformHeight * stretch,
                        baseDepthScale * squash
                    );

                    const fadeBack = Math.pow(travelT, 1.4);
                    displayColor.copy(accentColor).lerp(gradCol, fadeBack);
                    const sparkle = Math.sin(easedBase * Math.PI) * 0.18;
                    if (sparkle > 0) {
                        displayColor.lerp(sparkleColor, sparkle * 0.35);
                    }
                }

                const offsetY = currentY - basePrismCenterY;

                sourceVec.setInstanceAppearance(i, offsetY, displayColor, {
                    scale: animatedScale,
                    xOffset,
                    tilt
                });

                updateResidualTrail();
            })
            .onComplete(() => {
                targetVec.setInstanceAppearance(i, 0, whiteColor);

                const flashState = { pulse: 0 };
                new TWEEN.Tween(flashState)
                    .to({ pulse: 1 }, flashDuration)
                    .easing(TWEEN.Easing.Sine.InOut)
                    .onUpdate(({ pulse }) => {
                        const wave = Math.sin(pulse * Math.PI);
                        targetPulseScale.set(
                            targetBaseScale.x * (1 - 0.1 * wave),
                            targetBaseScale.y * (1 + 0.12 * wave),
                            targetBaseScale.z * (1 - 0.1 * wave)
                        );
                        flashColor.copy(whiteColor).lerp(gradCol, pulse * 0.6);
                        targetVec.setInstanceAppearance(i, 0, flashColor, { scale: targetPulseScale });
                    })
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