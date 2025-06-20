// Utility: Addition animation between two VectorVisualizationInstancedPrism vectors
// -----------------------------------------------------------------------------
// This helper consolidates the formerly duplicated "addition" logic so any
// module can trigger a residual-style add (vec1 += vec2) with trails and
// InstancedMesh visuals.  It is effectively a verbatim extraction of
// MHSAAnimation._startAdditionAnimation but parameterised to avoid a hard
// dependency on the MHSAAnimation class.
// -----------------------------------------------------------------------------
import * as THREE from 'three';
import { createTrailLine, updateTrail } from './trailUtils.js';
import { TRAIL_LINE_COLOR } from '../animations/LayerAnimationConstants.js';
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
 * @param {THREE.Scene} scene         – the active Three.js scene (for trail lines)
 * @param {VectorVisualizationInstancedPrism} sourceVec – travelling vector
 * @param {VectorVisualizationInstancedPrism} targetVec – stationary vector that will hold the sum
 * @param {Object} [lane]             – optional lane object; if provided the helper
 *                                      will update lane.origTrail & lane fields
 */
export function startPrismAdditionAnimation(scene, sourceVec, targetVec, lane) {
    if (!sourceVec || !targetVec || !sourceVec.mesh || !targetVec.mesh) return;
    if (typeof TWEEN === 'undefined') {
        console.warn('TWEEN not available – addition animation skipped');
        return;
    }

    // Freeze upward movement of the source so its group position remains static.
    sourceVec.group.userData = sourceVec.group.userData || {};
    const svUD = sourceVec.group.userData;
    svUD.stopRise = true;
    svUD.stopRiseTarget = targetVec.group;

    const vectorLength = VECTOR_LENGTH_PRISM;
    const centreIndex = Math.floor(vectorLength / 2);

    // Prepare / re-use a vertical trail following the centre prism.
    let additionTrail = lane && lane.origTrail;
    if (!additionTrail) {
        additionTrail = createTrailLine(scene, TRAIL_LINE_COLOR);
        if (lane) lane.origTrail = additionTrail;
    }
    svUD.additionTrail = additionTrail;

    // Seed trail with current centre-prism world-pos.
    const initMat = new THREE.Matrix4();
    sourceVec.mesh.getMatrixAt(centreIndex, initMat);
    const initPos = new THREE.Vector3().setFromMatrixPosition(initMat).applyMatrix4(sourceVec.group.matrixWorld);
    updateTrail(additionTrail, initPos);

    // Timing params (scale by dedicated multiplier so we can tune independently).
    const duration      = PRISM_ADD_ANIM_BASE_DURATION            / PRISM_ADD_ANIM_SPEED_MULT;
    const flashDuration = PRISM_ADD_ANIM_BASE_FLASH_DURATION      / PRISM_ADD_ANIM_SPEED_MULT;
    const delayBetween  = PRISM_ADD_ANIM_BASE_DELAY_BETWEEN_PRISMS/ PRISM_ADD_ANIM_SPEED_MULT;

    const basePrismCenterY = sourceVec.getUniformHeight() / 2;

    for (let i = 0; i < vectorLength; i++) {
        // Grab starting local Y offset of each instance
        const srcLocalMatrix = new THREE.Matrix4();
        sourceVec.mesh.getMatrixAt(i, srcLocalMatrix);
        const srcLocalPos = new THREE.Vector3().setFromMatrixPosition(srcLocalMatrix);

        // Capture target gradient colour so we can flash & restore
        const gradCol = new THREE.Color();
        targetVec.mesh.getColorAt(i, gradCol);

        // Match travelling prism colour to destination gradient
        sourceVec.setInstanceAppearance(i, srcLocalPos.y, gradCol);

        const tweenState = { t: 0 };

        new TWEEN.Tween(tweenState)
            .to({ t: 1 }, duration)
            .delay(i * delayBetween)
            .easing(TWEEN.Easing.Quadratic.InOut)
            .onUpdate(obj => {
                // Re-compute dynamic target position each frame (target may move)
                const trgLocalMatrixDyn = new THREE.Matrix4();
                targetVec.mesh.getMatrixAt(i, trgLocalMatrixDyn);
                const trgWorld = new THREE.Vector3().setFromMatrixPosition(trgLocalMatrixDyn).applyMatrix4(targetVec.group.matrixWorld);
                const trgLocal = sourceVec.group.worldToLocal(trgWorld.clone());

                let interpY = THREE.MathUtils.lerp(srcLocalPos.y, trgLocal.y, obj.t);
                interpY = trgLocal.y >= srcLocalPos.y ? Math.min(interpY, trgLocal.y) : Math.max(interpY, trgLocal.y);
                const offsetY = interpY - basePrismCenterY;

                sourceVec.setInstanceAppearance(i, offsetY, null);

                // Update trail for centre prism only
                if (i === centreIndex) {
                    const m = new THREE.Matrix4();
                    sourceVec.mesh.getMatrixAt(i, m);
                    const worldP = new THREE.Vector3().setFromMatrixPosition(m).applyMatrix4(sourceVec.group.matrixWorld);
                    updateTrail(additionTrail, worldP);
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
        if (sourceVec && sourceVec.group && sourceVec.group.userData) {
            delete sourceVec.group.userData.stopRise;
            delete sourceVec.group.userData.stopRiseTarget;
        }

        if (lane) {
            lane.originalVec    = targetVec;
            lane.postAdditionVec= targetVec;
            if (lane.ln2Phase !== 'done') {
                lane.ln2Phase = 'preRise';
                // Set horizPhase to trigger LN2 pipeline
                if (lane.horizPhase === 'travelMHSA' || lane.horizPhase === 'finishedHeads') {
                    lane.horizPhase = 'postMHSAAddition';
                }
            }
            if (sourceVec && sourceVec.group) {
                const topY = targetVec.group.position.y;
                sourceVec.group.userData.skipTrailResumeY = topY + 0.01;
            }
        }
    }, totalAnimTime + 100);
} 