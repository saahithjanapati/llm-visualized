import * as THREE from 'three';
import { VectorVisualizationInstancedPrism } from '../../components/VectorVisualizationInstancedPrism.js';
import { createTrailLine, updateTrail } from '../../utils/trailUtils.js';
import {
    MHA_MATRIX_PARAMS,
    VECTOR_LENGTH_PRISM,
} from '../../utils/constants.js';
import {
    TRAIL_LINE_COLOR,
    MHSA_MATRIX_MAX_EMISSIVE_INTENSITY,
    MHSA_PASS_THROUGH_BRIGHTEN_RATIO,
    MHSA_PASS_THROUGH_DIM_RATIO,
} from '../LayerAnimationConstants.js';

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
    finalVectorHue,
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

    // ------------------------------------------------------------------
    //  Trail line that follows the vector until it enters the matrix
    // ------------------------------------------------------------------
    const passThroughTrail = createTrailLine(ctx.parentGroup, TRAIL_LINE_COLOR);
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
            //  Vector motion & trail update
            // --------------------------------------------------------------
            vector.group.position.y = tweenState.y;
            if (vector.group.position.y < matrixBottomY) {
                updateTrail(passThroughTrail, vector.group.position);
            }

            // --------------------------------------------------------------
            //  Lightweight 64-dimensional swap as soon as we touch matrix
            // --------------------------------------------------------------
            if (!initialDimensionChangeApplied && tweenState.y >= matrixBottomY) {
                const smallVec = new VectorVisualizationInstancedPrism(
                    vector.rawData.slice(0, outLength),
                    vector.group.position.clone(),
                    3,
                );

                ctx.parentGroup.add(smallVec.group);
                const heavyVec = vector;
                vector = smallVec; // continue animating this handle
                // Preserve metadata such as headIndex for downstream alignment
                vector.userData = heavyVec.userData ? { ...heavyVec.userData } : {};
                // If this is a green (K) vector, update its parent lane reference
                if (vectorCategory === 'K' && vector.userData.parentLane) {
                    const pl = vector.userData.parentLane;
                    const hIdx = vector.userData.headIndex;
                    if (pl && typeof hIdx === 'number') {
                        pl.upwardCopies[hIdx] = vector;
                    }
                }
                initialDimensionChangeApplied = true;
                ctx.parentGroup.remove(heavyVec.group);
                if (typeof heavyVec.dispose === 'function') heavyVec.dispose();

                vector.applyProcessedVisuals(
                    vector.rawData.slice(0, outLength),
                    outLength,
                    {
                        numKeyColors: 3,
                        generationOptions: {
                            type: 'monochromatic',
                            baseHue: finalVectorHue,
                            saturation: 0.9,
                            minLightness: 0.4,
                            maxLightness: 0.8,
                        },
                    },
                    { setHiddenToBlack: false },
                );
            }

            // --------------------------------------------------------------
            //  Matrix colour & emissive pulsing
            // --------------------------------------------------------------
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
        })
        .onComplete(() => {
            // Ensure the matrix ends dimmed
            matrix.setColor(darkTintedMatrixColor);
            matrix.setEmissive(darkTintedMatrixColor, ctx.matrixRestingEmissiveIntensity);
            matrix.setOpacity(ctx.matrixRestingOpacity);

            // Guarantee processed visuals in case tween never hit swap point
            if (!finalVisualsApplied) {
                const processedData = vector.rawData.slice(0, outLength);
                vector.applyProcessedVisuals(processedData, outLength, {
                    numKeyColors: 3,
                    generationOptions: {
                        type: 'monochromatic',
                        baseHue: finalVectorHue,
                        saturation: 0.9,
                        minLightness: 0.4,
                        maxLightness: 0.8,
                    },
                });
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