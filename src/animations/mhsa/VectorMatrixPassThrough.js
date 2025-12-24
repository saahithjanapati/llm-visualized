import * as THREE from 'three';
import { VectorVisualizationInstancedPrism } from '../../components/VectorVisualizationInstancedPrism.js';


import {
    MHA_MATRIX_PARAMS,
    VECTOR_LENGTH_PRISM,
} from '../../utils/constants.js';
import { MHSA_PASS_THROUGH_BRIGHTEN_RATIO, MHSA_PASS_THROUGH_DIM_RATIO, MHSA_MATRIX_MAX_EMISSIVE_INTENSITY } from '../../utils/constants.js';

const _matrixWorldScratch = new THREE.Vector3();

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

    if (vector.userData) {
        vector.userData.vectorCategory = vectorCategory;
    }

    tweenState.colorR = initialVecColor.r;
    tweenState.colorG = initialVecColor.g;
    tweenState.colorB = initialVecColor.b;

    const alignVectorHorizontallyWithMatrix = (vec) => {
        if (!vec || !vec.group || !matrix || !matrix.group) return;
        try {
            const parent = vec.group.parent;
            if (!parent) return;
            const lane = vec.userData ? vec.userData.parentLane : null;
            const headIdx = vec.userData ? vec.userData.headIndex : null;
            let desiredX = null;

            if (lane && typeof headIdx === 'number') {
                if (vectorCategory === 'K') {
                    if (Array.isArray(ctx.headsCentersX) && headIdx < ctx.headsCentersX.length) {
                        desiredX = ctx.headsCentersX[headIdx];
                    }
                } else if (vectorCategory === 'Q' || vectorCategory === 'V') {
                    if (Array.isArray(lane.sideCopies)) {
                        const entry = lane.sideCopies.find(sc => sc && sc.vec === vec);
                        if (entry && typeof entry.targetX === 'number') {
                            desiredX = entry.targetX;
                        }
                    }
                }
            }

            if (typeof desiredX === 'number' && Number.isFinite(desiredX)) {
                // Convert the desired X (expressed in ctx.parentGroup space) into the vector's parent space.
                if (ctx && ctx.parentGroup && parent !== ctx.parentGroup) {
                    _matrixWorldScratch.set(desiredX, 0, 0);
                    ctx.parentGroup.localToWorld(_matrixWorldScratch);
                    parent.worldToLocal(_matrixWorldScratch);
                    vec.group.position.x = _matrixWorldScratch.x;
                } else {
                    vec.group.position.x = desiredX;
                }
                return;
            }

            matrix.group.getWorldPosition(_matrixWorldScratch);
            parent.worldToLocal(_matrixWorldScratch);
            vec.group.position.x = _matrixWorldScratch.x;
        } catch (_) { /* no-op */ }
    };

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
            if (!initialDimensionChangeApplied && tweenState.y >= matrixBottomY) {
                alignVectorHorizontallyWithMatrix(vector);
            }
            // Update trails only while BELOW the matrix; no trails above matrices
            try {
                const ud = vector.userData || {};
                const trail = ud.trail;
                if (trail && tweenState.y < matrixBottomY) {
                    if (ud.trailWorld) {
                        const wp = new THREE.Vector3();
                        vector.group.getWorldPosition(wp);
                        trail.update(wp);
                    } else {
                        trail.update(vector.group.position);
                    }
                }
            } catch (_) { /* no-op */ }
        

            // --------------------------------------------------------------
            //  Lightweight 64-dimensional swap as soon as we touch matrix
            // --------------------------------------------------------------
            if (!initialDimensionChangeApplied && tweenState.y >= matrixBottomY) {
                alignVectorHorizontallyWithMatrix(vector);
                const smallVec = new VectorVisualizationInstancedPrism(
                    vector.rawData.slice(0, outLength),
                    vector.group.position.clone(),
                    3,
                );

                ctx.parentGroup.add(smallVec.group);
                const heavyVec = vector;
                vector = smallVec; // continue animating this handle
                alignVectorHorizontallyWithMatrix(vector);
                // Preserve metadata such as headIndex for downstream alignment
                vector.userData = heavyVec.userData ? { ...heavyVec.userData } : {};
                vector.userData.vectorCategory = vectorCategory;
                // Preserve and refine hover label for clarity
                try {
                    const cat = vectorCategory === 'K' ? 'Key Vector (Green)'
                              : vectorCategory === 'Q' ? 'Query Vector (Blue)'
                              : 'Value Vector (Red)';
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