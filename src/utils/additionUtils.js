// Utility: Addition animation between two VectorVisualizationInstancedPrism vectors
// -----------------------------------------------------------------------------
// This helper consolidates the formerly duplicated "addition" logic so any
// module can trigger a residual-style add (vec1 += vec2) with trails and
// InstancedMesh visuals.  It is effectively a verbatim extraction of
// MHSAAnimation._startAdditionAnimation but parameterised to avoid a hard
// dependency on the MHSAAnimation class.
// -----------------------------------------------------------------------------
import * as THREE from 'three';
import TWEEN from '@tweenjs/tween.js';



import {
    VECTOR_LENGTH_PRISM,
    PRISM_ADD_ANIM_BASE_DURATION,
    PRISM_ADD_ANIM_BASE_FLASH_DURATION,
    PRISM_ADD_ANIM_BASE_DELAY_BETWEEN_PRISMS,
    PRISM_ADD_ANIM_SPEED_MULT,
    HIDE_INSTANCE_Y_OFFSET,
} from './constants.js';
import { getCentralPrismIndices } from './prismLayout.js';

const TMP_MATRIX_A = new THREE.Matrix4();
const TMP_MATRIX_B = new THREE.Matrix4();
const TMP_WORLD_A = new THREE.Vector3();
const TMP_WORLD_B = new THREE.Vector3();
const TMP_WORLD_AVG = new THREE.Vector3();
const TMP_WORLD_ANCHOR = new THREE.Vector3();
const TMP_WORLD_SNAP = new THREE.Vector3();
const TMP_LOCAL_SNAP = new THREE.Vector3();

function computeMidlineWorldPosition(vec, length = VECTOR_LENGTH_PRISM, out = new THREE.Vector3()) {
    if (!vec || !vec.mesh || typeof vec.mesh.getMatrixAt !== 'function' || !vec.group) {
        return out.set(0, 0, 0);
    }

    const centreIndices = getCentralPrismIndices(length);
    if (!centreIndices.length) {
        return out.set(0, 0, 0);
    }

    vec.mesh.getMatrixAt(centreIndices[0], TMP_MATRIX_A);
    TMP_WORLD_A.setFromMatrixPosition(TMP_MATRIX_A).applyMatrix4(vec.group.matrixWorld);

    if (centreIndices.length === 1) {
        return out.copy(TMP_WORLD_A);
    }

    vec.mesh.getMatrixAt(centreIndices[1], TMP_MATRIX_B);
    TMP_WORLD_B.setFromMatrixPosition(TMP_MATRIX_B).applyMatrix4(vec.group.matrixWorld);

    return out.copy(TMP_WORLD_A).add(TMP_WORLD_B).multiplyScalar(0.5);
}

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

    const vectorLength = VECTOR_LENGTH_PRISM;
    const centreIndices = getCentralPrismIndices(vectorLength);

    // Freeze upward movement of the source so its group position remains static.
    if (lane) {
        lane.stopRise = true;
        lane.stopRiseTarget = targetVec.group;
        // Keep residual trail brightness unchanged during addition
        // Reset residual trail monotonic tracker to the centre prism's current Y
        // so the trail extends immediately as the middle unit begins to rise.
        try {
            const centreWorld = computeMidlineWorldPosition(sourceVec, vectorLength, TMP_WORLD_AVG);
            lane.__residualMaxY = Number.isFinite(centreWorld.y) ? centreWorld.y - 0.001 : undefined;
        } catch (err) {
            console.warn('Failed to init residual trail:', err);
        }
        try {
            if (lane.originalVec && lane.originalVec.group) {
                lane.originalVec.group.getWorldPosition(TMP_WORLD_ANCHOR);
                lane.__residualTrailAnchor = lane.__residualTrailAnchor || { x: undefined, z: undefined };
                lane.__residualTrailAnchor.x = TMP_WORLD_ANCHOR.x;
                lane.__residualTrailAnchor.z = TMP_WORLD_ANCHOR.z;
            } else {
                delete lane.__residualTrailAnchor;
            }
        } catch (err) {
            console.warn('Failed to cache residual trail anchor:', err);
            delete lane.__residualTrailAnchor;
        }
    } else {
        sourceVec.group.userData = sourceVec.group.userData || {};
        const svUD = sourceVec.group.userData;
        svUD.stopRise = true;
        svUD.stopRiseTarget = targetVec.group;
        try {
            const centreWorld = computeMidlineWorldPosition(sourceVec, vectorLength, TMP_WORLD_AVG);
            sourceVec.userData.__residualMaxY =
                Number.isFinite(centreWorld.y) ? centreWorld.y - 0.001 : undefined;
        } catch (err) {
            console.warn('Failed to init residual trail:', err);
        }
        try {
            if (sourceVec.group) {
                sourceVec.group.getWorldPosition(TMP_WORLD_ANCHOR);
                sourceVec.userData.__residualTrailAnchor = sourceVec.userData.__residualTrailAnchor || { x: undefined, z: undefined };
                sourceVec.userData.__residualTrailAnchor.x = TMP_WORLD_ANCHOR.x;
                sourceVec.userData.__residualTrailAnchor.z = TMP_WORLD_ANCHOR.z;
            }
        } catch (err) {
            console.warn('Failed to cache residual trail anchor:', err);
            delete sourceVec.userData.__residualTrailAnchor;
        }
    }

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

                if (centreIndices.includes(i)) {
                    // When the animation is running within a lane (e.g. inside the
                    // MHSA pipeline) the owning animation loop already updates the
                    // residual trail in world-space each frame.  Duplicating those
                    // updates here introduces slightly different sample points which
                    // produce small horizontal kinks in the polyline.  Rely on the
                    // lane-managed updates instead to keep the trail perfectly
                    // vertical.
                    if (!lane) {
                        // For standalone additions (no lane object) we own the trail updates.
                        // Live-update the residual trail from the bottom vector while the
                        // centre prism rises toward the top vector. This mirrors the
                        // behaviour users expect: the connecting line grows as the
                        // middle unit moves, rather than appearing only after addition.
                        try {
                            const wPos = computeMidlineWorldPosition(sourceVec, vectorLength, TMP_WORLD_AVG);

                            // Skip if the prism is effectively hidden far below
                            const hideThreshold = HIDE_INSTANCE_Y_OFFSET / 10;
                            if (wPos.y >= hideThreshold) {
                                // Update the residual trail continuously as the prism rises all the
                                // way to the target vector. Previously, a muted band near the
                                // merge point prevented trailing up to the very top, leaving a
                                // visible gap.
                                const residualTrail = (sourceVec && sourceVec.userData && sourceVec.userData.trail)
                                    || null;
                                const residualOwner = (sourceVec && sourceVec.userData) || null;
                                if (residualTrail && residualOwner && typeof residualTrail.update === 'function') {
                                    if (!Number.isFinite(residualOwner.__residualMaxY)) {
                                        residualOwner.__residualMaxY = wPos.y - 0.001;
                                    }
                                    if (wPos.y >= residualOwner.__residualMaxY) {
                                        const anchor = residualOwner.__residualTrailAnchor;
                                        if (anchor) {
                                            if (Number.isFinite(anchor.x)) wPos.x = anchor.x;
                                            if (Number.isFinite(anchor.z)) wPos.z = anchor.z;
                                        }
                                        let localPos = wPos;
                                        const expectsWorldSpace = Boolean(
                                            (sourceVec && sourceVec.userData && sourceVec.userData.trailWorld)
                                            || (residualOwner && residualOwner.trailWorld)
                                        );
                                        if (!expectsWorldSpace) {
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
    const snapResidualTrailEndpoint = (trail, expectsWorldSpace, ownerVec) => {
        if (!trail || !ownerVec || !ownerVec.group) return;
        try {
            ownerVec.group.getWorldPosition(TMP_WORLD_SNAP);
            if (expectsWorldSpace) {
                if (typeof trail.snapLastPointTo === 'function') {
                    trail.snapLastPointTo(TMP_WORLD_SNAP);
                } else if (typeof trail.update === 'function') {
                    trail.update(TMP_WORLD_SNAP);
                }
            } else {
                const parentObject = (trail._line && trail._line.parent) || trail._scene || null;
                if (parentObject && typeof parentObject.worldToLocal === 'function') {
                    TMP_LOCAL_SNAP.copy(TMP_WORLD_SNAP);
                    parentObject.worldToLocal(TMP_LOCAL_SNAP);
                    if (typeof trail.snapLastPointTo === 'function') {
                        trail.snapLastPointTo(TMP_LOCAL_SNAP);
                    } else if (typeof trail.update === 'function') {
                        trail.update(TMP_LOCAL_SNAP);
                    }
                }
            }
            return TMP_WORLD_SNAP.y;
        } catch (_) {
            return undefined;
        }
    };

    const finishAddition = () => {
        if (lane) {
            delete lane.stopRise;
            delete lane.stopRiseTarget;
            delete lane.__residualTrailAnchor;
        } else if (sourceVec && sourceVec.group && sourceVec.group.userData) {
            delete sourceVec.group.userData.stopRise;
            delete sourceVec.group.userData.stopRiseTarget;
        }

        if (sourceVec && sourceVec.userData) {
            delete sourceVec.userData.__residualTrailAnchor;
        }

        if (lane) {
            const ownerVec = lane.originalVec || targetVec;
            const attachedTrail = ownerVec && ownerVec.userData && ownerVec.userData.trail;
            const fallbackTrail = lane.originalTrail;
            const trail = attachedTrail || fallbackTrail;
            if (trail && ownerVec) {
                const expectsWorldSpace = Boolean(
                    (ownerVec.userData && ownerVec.userData.trailWorld) ||
                    (trail._scene && trail._scene.isScene)
                );
                const snappedY = snapResidualTrailEndpoint(trail, expectsWorldSpace, ownerVec);
                if (Number.isFinite(snappedY)) {
                    lane.__residualMaxY = snappedY;
                    lane.__residualTrailAnchor = lane.__residualTrailAnchor || { x: undefined, z: undefined };
                    lane.__residualTrailAnchor.x = TMP_WORLD_SNAP.x;
                    lane.__residualTrailAnchor.z = TMP_WORLD_SNAP.z;
                }
            }
        } else if (targetVec && targetVec.userData) {
            const trail = targetVec.userData.trail;
            if (trail) {
                const expectsWorldSpace = Boolean(
                    targetVec.userData.trailWorld ||
                    (trail._scene && trail._scene.isScene)
                );
                const snappedY = snapResidualTrailEndpoint(trail, expectsWorldSpace, targetVec);
                if (Number.isFinite(snappedY)) {
                    targetVec.userData.__residualMaxY = snappedY;
                    targetVec.userData.__residualTrailAnchor = targetVec.userData.__residualTrailAnchor || { x: undefined, z: undefined };
                    targetVec.userData.__residualTrailAnchor.x = TMP_WORLD_SNAP.x;
                    targetVec.userData.__residualTrailAnchor.z = TMP_WORLD_SNAP.z;
                }
            }
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
