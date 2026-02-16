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
import { getCentralPrismIndices } from './prismLayout.js';
import { mapValueToColor } from './colors.js';

const TMP_MATRIX_A = new THREE.Matrix4();
const TMP_MATRIX_B = new THREE.Matrix4();
const TMP_WORLD_A = new THREE.Vector3();
const TMP_WORLD_B = new THREE.Vector3();
const TMP_WORLD_AVG = new THREE.Vector3();
const TMP_WORLD_ANCHOR = new THREE.Vector3();
const TMP_WORLD_SNAP = new THREE.Vector3();
const TMP_LOCAL_SNAP = new THREE.Vector3();
const TMP_ADD_TARGET_MATRIX = new THREE.Matrix4();
const TMP_ADD_TARGET_WORLD = new THREE.Vector3();
const TMP_ADD_TARGET_LOCAL = new THREE.Vector3();
const COLOR_WHITE = new THREE.Color(0xffffff);
const TMP_COLOR_A = new THREE.Color();
const TMP_COLOR_B = new THREE.Color();
const TMP_COLOR_C = new THREE.Color();
const CAMERA_HOLD_AFTER_ADDITION_MS = 0;

function isArrayLike(value) {
    return Array.isArray(value) || ArrayBuffer.isView(value);
}

function buildKeyColorsFromData(data, numKeyColors) {
    const keyColors = [];
    const dataLength = data.length || 0;
    if (numKeyColors <= 1) {
        const midIndex = dataLength > 0 ? Math.floor(dataLength / 2) : 0;
        const midValue = dataLength > 0 ? data[midIndex] : 0;
        keyColors.push(mapValueToColor(midValue));
        return keyColors;
    }

    const step = dataLength > 1 ? (dataLength - 1) / (numKeyColors - 1) : 0;
    for (let i = 0; i < numKeyColors; i++) {
        const sampleIndex = dataLength > 0
            ? Math.min(Math.round(i * step), dataLength - 1)
            : 0;
        const value = dataLength > 0 ? data[sampleIndex] : 0;
        keyColors.push(mapValueToColor(value));
    }
    return keyColors;
}

function getGradientColorForIndex(index, instanceCount, keyColors, numSubsections, outColor) {
    if (!keyColors || keyColors.length === 0) {
        return outColor.setRGB(0.5, 0.5, 0.5);
    }
    if (instanceCount <= 1) {
        return outColor.copy(keyColors[0]);
    }
    const currentNumSubsections = Math.max(1, numSubsections);
    const globalProgress = index / (instanceCount - 1);
    const segmentProgress = globalProgress * currentNumSubsections;
    const idx1 = Math.floor(segmentProgress);
    const safeIdx1 = Math.min(idx1, keyColors.length - 1);
    const safeIdx2 = Math.min(idx1 + 1, keyColors.length - 1);
    const localT = segmentProgress - idx1;
    return outColor.copy(keyColors[safeIdx1]).lerp(keyColors[safeIdx2], localT);
}

function buildFinalColorBuffers(finalData, instanceCount) {
    if (!isArrayLike(finalData) || finalData.length === 0) {
        return null;
    }
    const count = Math.max(1, Math.floor(instanceCount));
    const numKeyColors = Math.min(30, Math.max(1, finalData.length || 1));
    const keyColors = buildKeyColorsFromData(finalData, numKeyColors);
    const numSubsections = Math.max(0, numKeyColors - 1);

    const colorStart = new Float32Array(count * 3);
    const colorEnd = new Float32Array(count * 3);
    const instanceColors = new Float32Array(count * 3);
    for (let i = 0; i < count; i++) {
        const midColor = getGradientColorForIndex(i, count, keyColors, numSubsections, TMP_COLOR_A);
        const leftColor = getGradientColorForIndex(Math.max(0, i - 1), count, keyColors, numSubsections, TMP_COLOR_B);
        const rightColor = getGradientColorForIndex(Math.min(count - 1, i + 1), count, keyColors, numSubsections, TMP_COLOR_C);
        const i3 = i * 3;
        instanceColors[i3] = midColor.r;
        instanceColors[i3 + 1] = midColor.g;
        instanceColors[i3 + 2] = midColor.b;
        colorStart[i3] = leftColor.r;
        colorStart[i3 + 1] = leftColor.g;
        colorStart[i3 + 2] = leftColor.b;
        colorEnd[i3] = rightColor.r;
        colorEnd[i3 + 1] = rightColor.g;
        colorEnd[i3 + 2] = rightColor.b;
    }

    return {
        colorStart,
        colorEnd,
        instanceColors,
        dataLength: finalData.length,
        instanceCount: count,
    };
}

function getMappedDataValue(data, index, instanceCount) {
    if (!isArrayLike(data) || data.length === 0) {
        return undefined;
    }
    if (data.length === 1) {
        return data[0];
    }
    const t = instanceCount > 1 ? index / (instanceCount - 1) : 0;
    const mappedIndex = Math.min(data.length - 1, Math.max(0, Math.round(t * (data.length - 1))));
    return data[mappedIndex];
}

function applyFinalColorAtIndex(targetVec, index, finalBuffers) {
    if (!targetVec || !targetVec.mesh || !finalBuffers) return;
    const i3 = index * 3;
    const colorStartAttr = targetVec.mesh.geometry?.getAttribute?.('colorStart');
    const colorEndAttr = targetVec.mesh.geometry?.getAttribute?.('colorEnd');

    if (colorStartAttr && finalBuffers.colorStart) {
        colorStartAttr.array[i3] = finalBuffers.colorStart[i3];
        colorStartAttr.array[i3 + 1] = finalBuffers.colorStart[i3 + 1];
        colorStartAttr.array[i3 + 2] = finalBuffers.colorStart[i3 + 2];
        colorStartAttr.needsUpdate = true;
    }
    if (colorEndAttr && finalBuffers.colorEnd) {
        colorEndAttr.array[i3] = finalBuffers.colorEnd[i3];
        colorEndAttr.array[i3 + 1] = finalBuffers.colorEnd[i3 + 1];
        colorEndAttr.array[i3 + 2] = finalBuffers.colorEnd[i3 + 2];
        colorEndAttr.needsUpdate = true;
    }

    if (!targetVec.mesh.instanceColor) {
        targetVec.mesh.instanceColor = new THREE.InstancedBufferAttribute(
            new Float32Array(targetVec.instanceCount * 3),
            3
        );
    }
    if (finalBuffers.instanceColors && targetVec.mesh.instanceColor) {
        const instanceColors = targetVec.mesh.instanceColor.array;
        instanceColors[i3] = finalBuffers.instanceColors[i3];
        instanceColors[i3 + 1] = finalBuffers.instanceColors[i3 + 1];
        instanceColors[i3 + 2] = finalBuffers.instanceColors[i3 + 2];
        targetVec.mesh.instanceColor.needsUpdate = true;
    }
}

function applySingleColorAtIndex(targetVec, index, color) {
    if (!targetVec || !targetVec.mesh || !(color instanceof THREE.Color)) return;
    const i3 = index * 3;
    const colorStartAttr = targetVec.mesh.geometry?.getAttribute?.('colorStart');
    const colorEndAttr = targetVec.mesh.geometry?.getAttribute?.('colorEnd');

    if (colorStartAttr) {
        colorStartAttr.array[i3] = color.r;
        colorStartAttr.array[i3 + 1] = color.g;
        colorStartAttr.array[i3 + 2] = color.b;
        colorStartAttr.needsUpdate = true;
    }
    if (colorEndAttr) {
        colorEndAttr.array[i3] = color.r;
        colorEndAttr.array[i3 + 1] = color.g;
        colorEndAttr.array[i3 + 2] = color.b;
        colorEndAttr.needsUpdate = true;
    }

    if (!targetVec.mesh.instanceColor) {
        targetVec.mesh.instanceColor = new THREE.InstancedBufferAttribute(
            new Float32Array(targetVec.instanceCount * 3),
            3
        );
    }
    if (targetVec.mesh.instanceColor) {
        const instanceColors = targetVec.mesh.instanceColor.array;
        instanceColors[i3] = color.r;
        instanceColors[i3 + 1] = color.g;
        instanceColors[i3 + 2] = color.b;
        targetVec.mesh.instanceColor.needsUpdate = true;
    }
}

function applySingleColorAtIndexRGB(targetVec, index, r, g, b) {
    if (!targetVec || !targetVec.mesh) return;
    const i3 = index * 3;
    const colorStartAttr = targetVec.mesh.geometry?.getAttribute?.('colorStart');
    const colorEndAttr = targetVec.mesh.geometry?.getAttribute?.('colorEnd');

    if (colorStartAttr) {
        colorStartAttr.array[i3] = r;
        colorStartAttr.array[i3 + 1] = g;
        colorStartAttr.array[i3 + 2] = b;
        colorStartAttr.needsUpdate = true;
    }
    if (colorEndAttr) {
        colorEndAttr.array[i3] = r;
        colorEndAttr.array[i3 + 1] = g;
        colorEndAttr.array[i3 + 2] = b;
        colorEndAttr.needsUpdate = true;
    }

    if (!targetVec.mesh.instanceColor) {
        targetVec.mesh.instanceColor = new THREE.InstancedBufferAttribute(
            new Float32Array(targetVec.instanceCount * 3),
            3
        );
    }
    if (targetVec.mesh.instanceColor) {
        const instanceColors = targetVec.mesh.instanceColor.array;
        instanceColors[i3] = r;
        instanceColors[i3 + 1] = g;
        instanceColors[i3 + 2] = b;
        targetVec.mesh.instanceColor.needsUpdate = true;
    }
}

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

function getVectorInstanceCount(vec, fallback = VECTOR_LENGTH_PRISM) {
    if (vec && Number.isFinite(vec.instanceCount)) {
        return Math.max(1, Math.floor(vec.instanceCount));
    }
    return Math.max(1, Math.floor(fallback));
}

/**
 * Animate element-wise addition of two instanced-prism vectors, visually moving
 * each prism from `sourceVec` into `targetVec` while updating colours & data.
 *
 * @param {VectorVisualizationInstancedPrism} sourceVec – travelling vector
 * @param {VectorVisualizationInstancedPrism} targetVec – stationary vector that will hold the sum
 * @param {Object} [lane]             – optional lane object; if provided the helper
 *                                     will update lane fields
 * @param {Object} [options]
 * @param {boolean} [options.suppressResidualTrailUpdates=false] – skip residual
 *                                     trail updates from per-instance motion
 *                                     (useful when a higher-level controller
 *                                     already manages the trail).
 */
export function startPrismAdditionAnimation(sourceVec, targetVec, lane, onComplete, options = null) {
    const hasRenderableSource = !!(sourceVec && sourceVec.mesh && sourceVec.group);
    const hasRenderableTarget = !!(targetVec && targetVec.mesh && targetVec.group);

    const suppressResidualTrailUpdates = options && options.suppressResidualTrailUpdates === true;

    // Ensure a metadata container exists so we can store residual trail state
    // even when no lane object is available (e.g. LayerNorm additions create
    // temporary vectors without full lane context).
    if (sourceVec) {
        sourceVec.userData = sourceVec.userData || {};
    }

    const sourceCount = getVectorInstanceCount(sourceVec, VECTOR_LENGTH_PRISM);
    const targetCount = getVectorInstanceCount(targetVec, VECTOR_LENGTH_PRISM);
    const sourceLen = Array.isArray(sourceVec && sourceVec.rawData) ? sourceVec.rawData.length : sourceCount;
    const targetLen = Array.isArray(targetVec && targetVec.rawData) ? targetVec.rawData.length : targetCount;
    const vectorLength = Math.min(sourceCount, targetCount, sourceLen, targetLen);
    const centreIndices = getCentralPrismIndices(vectorLength);
    const finalDataCandidate = (options && isArrayLike(options.finalData))
        ? options.finalData
        : (lane && isArrayLike(lane.additionTargetData))
            ? lane.additionTargetData
            : null;
    const finalBuffers = finalDataCandidate
        ? buildFinalColorBuffers(finalDataCandidate, vectorLength)
        : null;
    const preserveSourceColors = options ? options.preserveSourceColors !== false : true;
    const progressTarget = options && options.progressTarget ? options.progressTarget : null;
    const progressKey = options && typeof options.progressKey === 'string' ? options.progressKey : null;
    const setProgress = (value) => {
        if (!progressTarget || !progressKey) return;
        const next = Math.max(0, Math.min(1, Number(value) || 0));
        progressTarget[progressKey] = next;
    };
    const invokeOnComplete = () => {
        if (typeof onComplete === 'function') {
            try {
                onComplete();
            } catch (err) {
                console.warn('Addition completion callback failed:', err);
            }
        }
    };
    const finalizeLaneState = () => {
        if (lane) {
            delete lane.stopRise;
            delete lane.stopRiseTarget;
            delete lane.__residualTrailAnchor;
            const nowMs = (typeof performance !== 'undefined' && typeof performance.now === 'function')
                ? performance.now()
                : Date.now();
            lane.__cameraHoldAfterAddUntil = nowMs + CAMERA_HOLD_AFTER_ADDITION_MS;
            const fallbackTarget = targetVec || lane.postAdditionVec || lane.originalVec || sourceVec || null;
            if (fallbackTarget) {
                lane.originalVec = fallbackTarget;
                lane.postAdditionVec = fallbackTarget;
            }
            if (lane.ln2Phase !== 'done') {
                lane.ln2Phase = 'preRise';
                if (lane.layer && typeof lane.layer._emitProgress === 'function') lane.layer._emitProgress();
                if (lane.horizPhase === 'travelMHSA' || lane.horizPhase === 'finishedHeads') {
                    lane.horizPhase = 'postMHSAAddition';
                    if (lane.layer && typeof lane.layer._emitProgress === 'function') lane.layer._emitProgress();
                }
            }
        } else if (sourceVec && sourceVec.group && sourceVec.group.userData) {
            delete sourceVec.group.userData.stopRise;
            delete sourceVec.group.userData.stopRiseTarget;
        }
        if (sourceVec && sourceVec.userData) {
            delete sourceVec.userData.__residualTrailAnchor;
        }
    };
    const failForward = (reason) => {
        console.warn(`Addition animation fallback: ${reason}`);
        setProgress(1);
        finalizeLaneState();
        invokeOnComplete();
    };

    if (!hasRenderableSource || !hasRenderableTarget) {
        failForward('missing source/target vector references');
        return;
    }
    if (typeof TWEEN === 'undefined') {
        failForward('TWEEN unavailable');
        return;
    }

    // Freeze upward movement of the source so its group position remains static.
    if (lane) {
        delete lane.__cameraHoldAfterAddUntil;
        lane.stopRise = true;
        lane.stopRiseTarget = targetVec.group;
        if (lane.layer && typeof lane.layer._emitProgress === 'function') {
            lane.layer._emitProgress();
        }
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
    let completedPrisms = 0;
    let finishOnce = null;

    for (let i = 0; i < vectorLength; i++) {
        // Grab starting local Y offset of each instance
        sourceVec.mesh.getMatrixAt(i, TMP_MATRIX_A);
        TMP_WORLD_A.setFromMatrixPosition(TMP_MATRIX_A);
        const srcLocalY = TMP_WORLD_A.y;

        // Capture target gradient colour so we can flash & restore
        targetVec.mesh.getColorAt(i, TMP_COLOR_A);
        const gradR = TMP_COLOR_A.r;
        const gradG = TMP_COLOR_A.g;
        const gradB = TMP_COLOR_A.b;

        // Optionally match travelling prism colour to destination gradient.
        if (!preserveSourceColors) {
            const initialOffsetY = srcLocalY - basePrismCenterY;
            TMP_COLOR_B.setRGB(gradR, gradG, gradB);
            sourceVec.setInstanceAppearance(i, initialOffsetY, TMP_COLOR_B);
        }

        const tweenState = { t: 0 };

        new TWEEN.Tween(tweenState)
            .to({ t: 1 }, duration)
            .delay(i * delayBetween)
            .easing(TWEEN.Easing.Quadratic.InOut)
            .onUpdate(obj => {
                // Re-compute dynamic target position each frame (target may move)
                targetVec.mesh.getMatrixAt(i, TMP_ADD_TARGET_MATRIX);
                TMP_ADD_TARGET_WORLD.setFromMatrixPosition(TMP_ADD_TARGET_MATRIX).applyMatrix4(targetVec.group.matrixWorld);
                TMP_ADD_TARGET_LOCAL.copy(TMP_ADD_TARGET_WORLD);
                sourceVec.group.worldToLocal(TMP_ADD_TARGET_LOCAL);

                const targetLocalY = TMP_ADD_TARGET_LOCAL.y;
                let interpY = THREE.MathUtils.lerp(srcLocalY, targetLocalY, obj.t);
                interpY = targetLocalY >= srcLocalY ? Math.min(interpY, targetLocalY) : Math.max(interpY, targetLocalY);
                const offsetY = interpY - basePrismCenterY;

                sourceVec.setInstanceAppearance(i, offsetY, null);

                if (centreIndices.includes(i)) {
                    // When running inside a lane (MHSA/MLP pipelines) the owning animation loop
                    // already updates the residual trail to avoid redundant sample points. Only
                    // handle standalone additions here.
                    if (!lane && !suppressResidualTrailUpdates) {
                        try {
                            const wPos = computeMidlineWorldPosition(sourceVec, vectorLength, TMP_WORLD_AVG);

                            const hideThreshold = HIDE_INSTANCE_Y_OFFSET / 10;
                            if (wPos.y >= hideThreshold) {
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
                                                    TMP_LOCAL_SNAP.copy(wPos);
                                                    parentObject.worldToLocal(TMP_LOCAL_SNAP);
                                                    localPos = TMP_LOCAL_SNAP;
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
                targetVec.setInstanceAppearance(i, 0, COLOR_WHITE);
                new TWEEN.Tween({})
                    .to({}, flashDuration)
                    .onComplete(() => {
                        const sum = (sourceVec.rawData[i] ?? 0) + (targetVec.rawData[i] ?? 0);
                        const finalValue = finalDataCandidate
                            ? getMappedDataValue(finalDataCandidate, i, vectorLength)
                            : sum;
                        if (Number.isFinite(finalValue)) {
                            targetVec.rawData[i] = finalValue;
                        } else {
                            targetVec.rawData[i] = sum;
                        }

                        if (finalBuffers) {
                            applyFinalColorAtIndex(targetVec, i, finalBuffers);
                        } else {
                            applySingleColorAtIndexRGB(targetVec, i, gradR, gradG, gradB);
                        }
                        sourceVec.setInstanceAppearance(i, HIDE_INSTANCE_Y_OFFSET, null);
                        completedPrisms += 1;
                        if (completedPrisms >= vectorLength && typeof finishOnce === 'function') {
                            finishOnce();
                        }
                    })
                    .start();
            })
            .start();
    }

    const totalAnimTime = duration + flashDuration + Math.max(0, (vectorLength - 1) * delayBetween);
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
        setProgress(1);
        if (lane) {
            delete lane.stopRise;
            delete lane.stopRiseTarget;
            delete lane.__residualTrailAnchor;
            const nowMs = (typeof performance !== 'undefined' && typeof performance.now === 'function')
                ? performance.now()
                : Date.now();
            lane.__cameraHoldAfterAddUntil = nowMs + CAMERA_HOLD_AFTER_ADDITION_MS;
        } else if (sourceVec && sourceVec.group && sourceVec.group.userData) {
            delete sourceVec.group.userData.stopRise;
            delete sourceVec.group.userData.stopRiseTarget;
        }

        if (sourceVec && sourceVec.userData) {
            delete sourceVec.userData.__residualTrailAnchor;
        }

        if (lane) {
            const trailOwner = lane.originalVec || targetVec || sourceVec;
            const attachedTrail = trailOwner && trailOwner.userData && trailOwner.userData.trail;
            const fallbackTrail = lane.originalTrail;
            const trail = attachedTrail || fallbackTrail;
            if (trail && trailOwner) {
                const expectsWorldSpace = Boolean(
                    (trailOwner.userData && trailOwner.userData.trailWorld) ||
                    (trail._scene && trail._scene.isScene)
                );
                const snapTargetVec = targetVec || trailOwner;
                const snappedY = snapResidualTrailEndpoint(trail, expectsWorldSpace, snapTargetVec);
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
        }
        invokeOnComplete();
    };

    let additionFinalized = false;
    finishOnce = () => {
        if (additionFinalized) return;
        additionFinalized = true;
        finishAddition();
    };

    if (vectorLength <= 0) {
        finishOnce();
        return;
    }

    setProgress(0);
    if (typeof TWEEN !== 'undefined') {
        new TWEEN.Tween({ progress: 0 })
            .to({ progress: 1 }, totalAnimTime)
            .onUpdate(obj => {
                if (additionFinalized) return;
                setProgress(obj.progress);
            })
            .onComplete(finishOnce)
            .start();
    } else {
        setProgress(1);
        setTimeout(finishOnce, totalAnimTime);
    }
}
