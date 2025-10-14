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





    const baseHeight       = typeof sourceVec.getUniformHeight === 'function' ? sourceVec.getUniformHeight() : 1;
    const baseWidthScale   = typeof sourceVec.getWidthScale    === 'function' ? sourceVec.getWidthScale()    : 1;
    const baseDepthScale   = typeof sourceVec.getDepthScale    === 'function' ? sourceVec.getDepthScale()    : 1;
    const targetBaseHeight = typeof targetVec.getUniformHeight === 'function' ? targetVec.getUniformHeight() : baseHeight;
    const targetBaseWidth  = typeof targetVec.getWidthScale    === 'function' ? targetVec.getWidthScale()    : baseWidthScale;
    const targetBaseDepth  = typeof targetVec.getDepthScale    === 'function' ? targetVec.getDepthScale()    : baseDepthScale;

    const buildScaleVector = (width, height, depth, stretch) => {
        const safeStretch = THREE.MathUtils.clamp(stretch, 0.45, 1.55);
        const lateralFactor = 1 / Math.sqrt(Math.max(0.0001, safeStretch));
        return new THREE.Vector3(
            width  * lateralFactor,
            height * safeStretch,
            depth  * lateralFactor,
        );
    };

    const targetBaseScaleVec = buildScaleVector(targetBaseWidth, targetBaseHeight, targetBaseDepth, 1);

    // Timing params (scale by dedicated multiplier so we can tune independently).
    const duration      = PRISM_ADD_ANIM_BASE_DURATION            / PRISM_ADD_ANIM_SPEED_MULT;
    const flashDuration = PRISM_ADD_ANIM_BASE_FLASH_DURATION      / PRISM_ADD_ANIM_SPEED_MULT;
    const delayBetween  = PRISM_ADD_ANIM_BASE_DELAY_BETWEEN_PRISMS/ PRISM_ADD_ANIM_SPEED_MULT;

    const anticipationDuration = duration * 0.2;
    const travelDuration       = duration * 0.55;
    const settleDuration       = duration * 0.25;

    const basePrismCenterY = baseHeight / 2;

    for (let i = 0; i < vectorLength; i++) {
        const srcLocalMatrix = new THREE.Matrix4();
        sourceVec.mesh.getMatrixAt(i, srcLocalMatrix);
        const srcLocalPos = new THREE.Vector3().setFromMatrixPosition(srcLocalMatrix);

        const gradCol = new THREE.Color();
        targetVec.mesh.getColorAt(i, gradCol);
        sourceVec.setInstanceAppearance(i, srcLocalPos.y, gradCol);

        const travelStartY = srcLocalPos.y;
        let lastTravelY = travelStartY;
        let settleStartStretch = 1;

        const applyAppearance = (newY, stretchFactor) => {
            const offsetY = newY - basePrismCenterY;
            const scaleVec = buildScaleVector(baseWidthScale, baseHeight, baseDepthScale, stretchFactor);
            sourceVec.setInstanceAppearance(i, offsetY, null, scaleVec);

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
        };

        const getDynamicTargetLocalY = () => {
            const trgLocalMatrixDyn = new THREE.Matrix4();
            targetVec.mesh.getMatrixAt(i, trgLocalMatrixDyn);
            const trgWorld = new THREE.Vector3().setFromMatrixPosition(trgLocalMatrixDyn).applyMatrix4(targetVec.group.matrixWorld);
            const trgLocal = sourceVec.group.worldToLocal(trgWorld.clone());
            return trgLocal.y;
        };

        const anticipationDip = baseHeight * 0.18;
        const anticipationState = { t: 0 };
        const travelState = { t: 0 };
        const settleState = { t: 0 };

        const anticipationTween = new TWEEN.Tween(anticipationState)
            .to({ t: 1 }, anticipationDuration)
            .delay(i * delayBetween)
            .easing(TWEEN.Easing.Quadratic.InOut)
            .onUpdate(obj => {
                const squashFactor = 1 - 0.35 * obj.t;
                const newY = travelStartY - anticipationDip * obj.t;
                applyAppearance(newY, squashFactor);
            });

        const travelTween = new TWEEN.Tween(travelState)
            .to({ t: 1 }, travelDuration)
            .easing(TWEEN.Easing.Cubic.Out)
            .onUpdate(obj => {
                const targetLocalY = getDynamicTargetLocalY();
                const overshoot = (targetLocalY - (travelStartY - anticipationDip)) * 0.12;
                const destY = targetLocalY + overshoot;
                const newY = THREE.MathUtils.lerp(travelStartY - anticipationDip, destY, obj.t);
                const stretchFactor = 1 + 0.25 * Math.sin(obj.t * Math.PI);
                settleStartStretch = stretchFactor;
                lastTravelY = newY;
                applyAppearance(newY, stretchFactor);
            });

        const settleTween = new TWEEN.Tween(settleState)
            .to({ t: 1 }, settleDuration)
            .easing(TWEEN.Easing.Back.Out)
            .onUpdate(obj => {
                const targetLocalY = getDynamicTargetLocalY();
                const startY = lastTravelY;
                const newY = THREE.MathUtils.lerp(startY, targetLocalY, obj.t);
                const settleSquash = 1 - 0.12 * Math.sin(obj.t * Math.PI) * (1 - obj.t);
                const combined = THREE.MathUtils.lerp(settleStartStretch, settleSquash, obj.t);
                applyAppearance(newY, combined);
            })
            .onComplete(() => {
                const flashState = { t: 0 };
                const bounceState = { t: 0 };
                const bounceDuration = flashDuration * 1.5;
                const targetHighlightColor = new THREE.Color(1, 1, 1);
                const targetColorTemp = gradCol.clone();

                new TWEEN.Tween(flashState)
                    .to({ t: 1 }, flashDuration)
                    .easing(TWEEN.Easing.Sine.Out)
                    .onUpdate(flash => {
                        const stretch = 1 + 0.08 * Math.sin(flash.t * Math.PI);
                        const scaleVec = buildScaleVector(targetBaseWidth, targetBaseHeight, targetBaseDepth, stretch);
                        targetColorTemp.copy(gradCol).lerp(targetHighlightColor, 1 - flash.t * 0.6);
                        targetVec.setInstanceAppearance(i, 0, targetColorTemp, scaleVec);
                    })
                    .onComplete(() => {
                        const sum = (sourceVec.rawData[i] ?? 0) + (targetVec.rawData[i] ?? 0);
                        targetVec.rawData[i] = sum;
                        sourceVec.setInstanceAppearance(i, HIDE_INSTANCE_Y_OFFSET, null);

                        new TWEEN.Tween(bounceState)
                            .to({ t: 1 }, bounceDuration)
                            .easing(TWEEN.Easing.Sine.Out)
                            .onUpdate(bounce => {
                                const settleStretch = 1 - 0.15 * Math.sin(bounce.t * Math.PI) * (1 - bounce.t * 0.6);
                                const scaleVec = buildScaleVector(targetBaseWidth, targetBaseHeight, targetBaseDepth, settleStretch);
                                targetColorTemp.copy(gradCol).lerp(targetHighlightColor, (1 - bounce.t) * 0.25);
                                targetVec.setInstanceAppearance(i, 0, targetColorTemp, scaleVec);
                            })
                            .onComplete(() => {
                                targetVec.setInstanceAppearance(i, 0, gradCol, targetBaseScaleVec);
                            })
                            .start();
                    })
                    .start();
            });

        anticipationTween.chain(travelTween);
        travelTween.chain(settleTween);
        anticipationTween.start();
    }

    const totalAnimTime = duration + flashDuration + vectorLength * delayBetween;
    const finishAddition = () => {
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
    };

    if (typeof TWEEN !== 'undefined') {
        new TWEEN.Tween({ progress: 0 })
            .to({ progress: 1 }, totalAnimTime + 100)
            .onComplete(finishAddition)
            .start();
    } else {
        setTimeout(finishAddition, totalAnimTime + 100);
    }
}
