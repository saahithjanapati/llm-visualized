import * as THREE from 'three';
import { VectorVisualizationInstancedPrism } from '../../components/VectorVisualizationInstancedPrism.js';


import {
    MHA_MATRIX_PARAMS,
    VECTOR_LENGTH_PRISM,
} from '../../utils/constants.js';
import { MHSA_PASS_THROUGH_BRIGHTEN_RATIO, MHSA_PASS_THROUGH_DIM_RATIO, MHSA_MATRIX_MAX_EMISSIVE_INTENSITY } from '../../utils/constants.js';
import { buildMonochromeOptions, mapValueToMonochrome } from '../../utils/colors.js';
import { buildActivationData, applyActivationDataToVector } from '../../utils/activationMetadata.js';
import { MHA_VALUE_SPECTRUM_COLOR } from '../LayerAnimationConstants.js';

const _trailScratch = new THREE.Vector3();

/**
 * Animate a vector passing vertically through its corresponding weight matrix.
 * The heavy 768-dimensional vector is swapped for a lightweight 64-dimensional
 * version once it enters the matrix to save GPU resources.
 *
 * NOTE: This helper is completely stateless – all references to scene objects
 *       and runtime flags are resolved from the provided `ctx` (the
 *       MHSAAnimation instance).
 */
export function animateVectorMatrixPassThrough(
    ctx,
    vector,
    matrix,
    brightMatrixColor,
    darkTintedMatrixColor,
    finalVectorColor,
    passThroughY,
    duration,
    riseOffset,
    riseDurationVal, // Unused but kept for API parity
    outLength,
    animationCompletionCallback,
    vectorCategory = 'K',
) {
    if (typeof TWEEN === 'undefined') {
        console.error('Global TWEEN object not loaded for MHSAAnimation!');
        if (animationCompletionCallback) animationCompletionCallback();
        return;
    }

    if (!vector || !matrix) {
        console.warn('Missing vector or matrix for pass-through animation in MHSA.');
        if (animationCompletionCallback) animationCompletionCallback();
        return;
    }

    const alignVectorToMatrixLane = (vec, mat) => {
        if (!vec || !mat) return;
        const lane = vec.userData ? vec.userData.parentLane : null;
        let depth = Number.isFinite(mat.depth) ? mat.depth : MHA_MATRIX_PARAMS.depth;
        const count = Number.isFinite(mat.numberOfSlits) ? mat.numberOfSlits : MHA_MATRIX_PARAMS.numberOfSlits;
        if (!Number.isFinite(depth) || !Number.isFinite(count) || count <= 0) return;
        let centerZ = mat.group && Number.isFinite(mat.group.position.z) ? mat.group.position.z : 0;
        if (mat.mesh && mat.mesh.geometry && !mat.mesh.isInstancedMesh) {
            const geom = mat.mesh.geometry;
            if (!geom.boundingBox) geom.computeBoundingBox();
            if (geom.boundingBox) {
                const bb = geom.boundingBox;
                depth = Math.max(1e-6, bb.max.z - bb.min.z);
                const localCenterZ = (bb.min.z + bb.max.z) * 0.5;
                const meshZ = mat.mesh.position && Number.isFinite(mat.mesh.position.z) ? mat.mesh.position.z : 0;
                const meshScaleZ = mat.mesh.scale && Number.isFinite(mat.mesh.scale.z) ? mat.mesh.scale.z : 1;
                centerZ = (mat.group && Number.isFinite(mat.group.position.z) ? mat.group.position.z : 0)
                    + meshZ + localCenterZ * meshScaleZ;
            }
        }
        const spacing = depth / (count + 1);
        let laneIndex = lane && Number.isFinite(lane.laneIndex) ? lane.laneIndex : null;
        if (!Number.isFinite(laneIndex)) {
            const refZ = lane && Number.isFinite(lane.zPos) ? lane.zPos : vec.group.position.z;
            laneIndex = Math.round((refZ - centerZ + depth / 2) / spacing - 1);
        }
        laneIndex = Math.max(0, Math.min(count - 1, Math.floor(laneIndex)));
        const targetZ = centerZ - depth / 2 + spacing * (laneIndex + 1);
        vec.group.position.z = targetZ;
        if (mat.group && Number.isFinite(mat.group.position.x)) {
            vec.group.position.x = mat.group.position.x;
        }
    };

    alignVectorToMatrixLane(vector, matrix);

    // ------------------------------------------------------------------
    const matrixBottomY = ctx.mhsa_matrix_center_y - MHA_MATRIX_PARAMS.height / 2;

    // ------------------------------------------------------------------
    //  Prepare tween state helpers
    // ------------------------------------------------------------------
    const originalMatrixIntensity = matrix.mesh.material.emissiveIntensity;
    let finalVisualsApplied = false;
    let initialDimensionChangeApplied = false;

    // ------------------------------
    // Adjust rise height – custom tweak for heads-only test
    // Lower overall rise by 20 units. Additional rise for V vectors is now handled
    // by SelfAttentionAnimator for above-matrix animations.
    // ------------------------------
    const BASE_RISE_ADJUST = -30; // lower all coloured vectors
    const targetRiseY = passThroughY + riseOffset + BASE_RISE_ADJUST;

    const tweenState = {
        y: vector.group.position.y,
        progress: 0,
        matrixEmissiveIntensity: originalMatrixIntensity,
    };

    // Cache starting colour of the prism for smooth transition
    const initialVecColor = new THREE.Color();
    if (vector.mesh.instanceColor) {
        vector.mesh.getColorAt(0, initialVecColor);
    } else {
        initialVecColor.setRGB(0.5, 0.5, 0.5);
    }

    tweenState.colorR = initialVecColor.r;
    tweenState.colorG = initialVecColor.g;
    tweenState.colorB = initialVecColor.b;

    new TWEEN.Tween(tweenState)
        .to(
            {
                y: targetRiseY,
                progress: 1.0,
                colorR: 1.0,
                colorG: 1.0,
                colorB: 1.0,
                matrixEmissiveIntensity: MHSA_MATRIX_MAX_EMISSIVE_INTENSITY,
            },
            duration,
        )
        .easing(TWEEN.Easing.Quadratic.InOut)
        .onUpdate(() => {
            // --------------------------------------------------------------
            //  Vector motion       
            // --------------------------------------------------------------
            vector.group.position.y = tweenState.y;
            // Update trails only while BELOW the matrix; no trails above matrices
            try {
                const ud = vector.userData || {};
                const trail = ud.trail;
                if (trail) {
                    // Clamp to just below the matrix bottom so skip-to-end jumps
                    // still leave a visible vertical trail segment.
                    const clampY = Math.min(tweenState.y, matrixBottomY - 0.001);
                    if (ud.trailWorld) {
                        vector.group.getWorldPosition(_trailScratch);
                        const deltaY = tweenState.y - clampY;
                        if (deltaY !== 0) _trailScratch.y -= deltaY;
                        trail.update(_trailScratch);
                    } else {
                        _trailScratch.copy(vector.group.position);
                        _trailScratch.y = clampY;
                        trail.update(_trailScratch);
                    }
                }
            } catch (_) { /* no-op */ }
        

            // --------------------------------------------------------------
            //  Lightweight 64-dimensional swap as soon as we touch matrix
            // --------------------------------------------------------------
            if (!initialDimensionChangeApplied && tweenState.y >= matrixBottomY) {
                const heavyVec = vector;
                const smallVec = new VectorVisualizationInstancedPrism(
                    heavyVec.rawData.slice(0, outLength),
                    heavyVec.group.position.clone(),
                    3,
                    heavyVec.instanceCount || ctx.vectorPrismCount,
                );

                ctx.parentGroup.add(smallVec.group);
                vector = smallVec; // continue animating this handle
                // Preserve metadata such as headIndex for downstream alignment
                vector.userData = heavyVec.userData ? { ...heavyVec.userData } : {};
                if (vector.group) {
                    vector.group.userData = vector.group.userData || {};
                    if (Number.isFinite(vector.userData?.headIndex)) {
                        vector.group.userData.headIndex = vector.userData.headIndex;
                    }
                    if (Number.isFinite(ctx?.layerIndex)) {
                        vector.group.userData.layerIndex = ctx.layerIndex;
                    }
                }
                // Preserve and refine hover label for clarity
                try {
                    const cat = vectorCategory === 'K' ? 'Key Vector (Green)'
                              : vectorCategory === 'Q' ? 'Query Vector (Blue)'
                              : 'Value Vector (Orange)';
                    vector.group.userData.label = cat;
                    if (vector.mesh) vector.mesh.userData = { ...(vector.mesh.userData||{}), label: cat };
                } catch (_) {}
                // Ensure trails do NOT continue above matrices for any category
                if (vector.userData) {
                    delete vector.userData.trail;
                    delete vector.userData.trailWorld;
                }
                // If this is a green (K) vector, update its parent lane reference
                if (vectorCategory === 'K' && vector.userData.parentLane) {
                    const pl = vector.userData.parentLane;
                    const hIdx = vector.userData.headIndex;
                    if (pl && typeof hIdx === 'number') {
                        pl.upwardCopies[hIdx] = vector;
                    }
                }
                // If this is a side copy (Q or V), update the lane.sideCopies entry to
                // reference the new lightweight 64-dim vector that will be raised/animated above the matrices.
                if ((vectorCategory === 'Q' || vectorCategory === 'V') && vector.userData.parentLane) {
                    const pl = vector.userData.parentLane;
                    const hIdx = vector.userData.headIndex;
                    if (pl && Array.isArray(pl.sideCopies) && typeof hIdx === 'number') {
                        const entry = pl.sideCopies.find(sc => sc && sc.headIndex === hIdx && sc.type === vectorCategory);
                        if (entry) {
                            entry.vec = vector;
                        }
                    }
                }
                // heavyVec trail remains in scene as a static line below; do not update above
                initialDimensionChangeApplied = true;
                ctx.parentGroup.remove(heavyVec.group);
                if (typeof heavyVec.dispose === 'function') heavyVec.dispose();

                const activationSource = ctx && ctx.activationSource ? ctx.activationSource : null;
                const layerIndex = Number.isFinite(ctx?.layerIndex) ? ctx.layerIndex : null;
                const headIndex = vector?.userData?.headIndex;
                const tokenIndex = vector?.userData?.parentLane?.tokenIndex;
                const tokenLabel = vector?.userData?.parentLane?.tokenLabel;
                const kind = vectorCategory === 'Q' ? 'q' : vectorCategory === 'K' ? 'k' : 'v';
                const scalar = activationSource && Number.isFinite(layerIndex)
                    ? activationSource.getLayerQKVScalar(layerIndex, kind, headIndex, tokenIndex)
                    : null;
                const data = Number.isFinite(scalar)
                    ? [scalar]
                    : vector.rawData.slice(0, outLength);
                // Map values into a monochrome spectrum derived from the head's final tint.
                // Use an orange-leaning spectrum for V outputs without changing head colours.
                const monoBase = vectorCategory === 'V' ? MHA_VALUE_SPECTRUM_COLOR : finalVectorColor;
                const monoOptions = buildMonochromeOptions(monoBase);
                const numKeyColors = Number.isFinite(scalar) ? 1 : 3;
                vector.applyProcessedVisuals(
                    data,
                    outLength,
                    { numKeyColors, generationOptions: monoOptions },
                    { setHiddenToBlack: false },
                );
                if (Number.isFinite(scalar) && typeof vector.setUniformColor === 'function') {
                    // Scalar case: map the single value to a uniform mono tint.
                    const monoColor = mapValueToMonochrome(scalar, monoOptions);
                    vector.setUniformColor(monoColor);
                }
                finalVisualsApplied = true;
                if (Number.isFinite(scalar)) {
                    const label = vectorCategory === 'K'
                        ? 'Key Vector (Green)'
                        : vectorCategory === 'Q'
                            ? 'Query Vector (Blue)'
                            : 'Value Vector (Orange)';
                    const activationData = buildActivationData({
                        label,
                        values: data,
                        stage: `qkv.${kind}`,
                        layerIndex,
                        tokenIndex,
                        tokenLabel,
                        headIndex,
                    });
                    applyActivationDataToVector(vector, activationData, label);
                }
            }

            // --------------------------------------------------------------
            //  Matrix colour & emissive pulsing (throttled)
            //  Skip per-vector updates if a global pulse is active to reduce
            //  material updates per frame.
            // --------------------------------------------------------------
            if (!ctx._mhaPulseActive) {
                let currentMatrixTargetColor;
                let currentEmissiveIntensity;
                const p = tweenState.progress;
                if (p < MHSA_PASS_THROUGH_BRIGHTEN_RATIO) {
                    const t = THREE.MathUtils.smoothstep(p / MHSA_PASS_THROUGH_BRIGHTEN_RATIO, 0, 1);
                    currentMatrixTargetColor = ctx.matrixInitialRestingColor.clone().lerp(brightMatrixColor, t);
                    currentEmissiveIntensity = THREE.MathUtils.lerp(
                        ctx.matrixRestingEmissiveIntensity,
                        MHSA_MATRIX_MAX_EMISSIVE_INTENSITY,
                        t,
                    );
                } else if (p < MHSA_PASS_THROUGH_BRIGHTEN_RATIO + MHSA_PASS_THROUGH_DIM_RATIO) {
                    const t = THREE.MathUtils.smoothstep(
                        (p - MHSA_PASS_THROUGH_BRIGHTEN_RATIO) / MHSA_PASS_THROUGH_DIM_RATIO,
                        0,
                        1,
                    );
                    currentMatrixTargetColor = brightMatrixColor.clone().lerp(darkTintedMatrixColor, t);
                    currentEmissiveIntensity = THREE.MathUtils.lerp(
                        MHSA_MATRIX_MAX_EMISSIVE_INTENSITY,
                        ctx.matrixRestingEmissiveIntensity,
                        t,
                    );
                } else {
                    currentMatrixTargetColor = darkTintedMatrixColor.clone();
                    currentEmissiveIntensity = ctx.matrixRestingEmissiveIntensity;
                }
                matrix.setColor(currentMatrixTargetColor);
                matrix.setEmissive(currentMatrixTargetColor, currentEmissiveIntensity);
            }
        })
        .onComplete(() => {
            // Ensure the matrix ends dimmed if no global pulse managed it
            if (!ctx._mhaPulseActive) {
                matrix.setColor(darkTintedMatrixColor);
                matrix.setEmissive(darkTintedMatrixColor, ctx.matrixRestingEmissiveIntensity);
                matrix.setOpacity(ctx.matrixRestingOpacity);
            }

            // Guarantee processed visuals in case tween never hit swap point
            if (!finalVisualsApplied) {
                const processedData = vector.rawData.slice(0, outLength);
                const monoBase = vectorCategory === 'V' ? MHA_VALUE_SPECTRUM_COLOR : finalVectorColor;
                const monoOptions = buildMonochromeOptions(monoBase);
                vector.applyProcessedVisuals(processedData, outLength, {
                    numKeyColors: 3,
                    generationOptions: monoOptions,
                });
                if (processedData.length === 1 && typeof vector.setUniformColor === 'function') {
                    const monoColor = mapValueToMonochrome(processedData[0], monoOptions);
                    vector.setUniformColor(monoColor);
                }
                finalVisualsApplied = true;
            }

            // Delegate above-matrix animations to SelfAttentionAnimator
            if (ctx.selfAttentionAnimator) {
                ctx.selfAttentionAnimator.start(vector, vectorCategory, () => {
                    // Continue with any additional post-animation logic if needed
                });
            }
            // --------------------------------------------------------------
            //  Temp-mode bookkeeping identical to legacy implementation
            // --------------------------------------------------------------
            if (ctx.mode === 'temp') {
                ctx._tempAllOutputVectors.push(vector);
                if (vectorCategory === 'K') ctx._tempKOutputVectors.push(vector);
            }

            // --------------------------------------------------------------
            //  Notify upstream logic that this pass-through animation is done
            // --------------------------------------------------------------
            if (animationCompletionCallback) {
                animationCompletionCallback();
            }
        })
        .start();
}
