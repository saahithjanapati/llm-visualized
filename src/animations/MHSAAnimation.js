import * as THREE from 'three';
import { WeightMatrixVisualization } from '../components/WeightMatrixVisualization.js';
import { VectorVisualizationInstancedPrism } from '../components/VectorVisualizationInstancedPrism.js';
import { createTrailLine, updateTrail } from '../utils/trailUtils.js';
import { MHSA_DUPLICATE_VECTOR_RISE_SPEED, MHSA_PASS_THROUGH_TOTAL_DURATION_MS, MHSA_PASS_THROUGH_BRIGHTEN_RATIO, MHSA_PASS_THROUGH_DIM_RATIO, MHSA_MATRIX_MAX_EMISSIVE_INTENSITY, MHSA_MATRIX_INITIAL_RESTING_COLOR, MHSA_BRIGHT_GREEN, MHSA_DARK_TINTED_GREEN, MHSA_BRIGHT_BLUE, MHSA_DARK_TINTED_BLUE, MHSA_BRIGHT_RED, MHSA_DARK_TINTED_RED, MHSA_RESULT_RISE_OFFSET_Y, MHSA_HEAD_VECTOR_STOP_BELOW, TRAIL_LINE_COLOR, TRAIL_LINE_OPACITY } from './LayerAnimationConstants.js';
import {
    // Constants needed for setup & animation
    MHA_MATRIX_PARAMS,
    NUM_HEAD_SETS_LAYER,
    HEAD_SET_GAP_LAYER,
    MHA_INTERNAL_MATRIX_SPACING,
    HEAD_VECTOR_STOP_BELOW,
    ANIM_HORIZ_SPEED,
    GLOBAL_ANIM_SPEED_MULT,
    SIDE_COPY_DELAY_MS,
    SIDE_COPY_HORIZ_SPEED,
    ROW_MERGE_HORIZ_SPEED,
    ROW_SEGMENT_SPACING,
    VECTOR_LENGTH_PRISM,
    HIDE_INSTANCE_Y_OFFSET,
} from '../utils/constants.js';

// Define speed multiplier
const SPEED_MULT = GLOBAL_ANIM_SPEED_MULT;

export class MHSAAnimation {
    constructor(scene, branchX, mhsaBaseY, clock, mode = 'temp') {
        this.scene = scene;
        this.branchX = branchX;
        this.mhsaBaseY = mhsaBaseY;
        this.clock = clock;

        this.mhaVisualizations = [];
        this.headsCentersX = [];
        this.headCoords = [];
        this.mhaPassThroughPhase = 'positioning_mha_vectors';

        this.mhsa_matrix_center_y = this.mhsaBaseY + MHA_MATRIX_PARAMS.height / 2;
        this.headStopY = this.mhsa_matrix_center_y - MHSA_HEAD_VECTOR_STOP_BELOW;
        this.mhaPassThroughTargetY = this.mhsa_matrix_center_y + MHA_MATRIX_PARAMS.height / 2 + 20;
        this.outputVectorLength = 64;
        this.mhaResultRiseOffsetY = MHSA_RESULT_RISE_OFFSET_Y;
        this.mhaResultRiseDuration = 500 / SPEED_MULT;
        this.mhaPassThroughDuration = MHSA_PASS_THROUGH_TOTAL_DURATION_MS / SPEED_MULT;

        this.matrixInitialRestingColor = new THREE.Color(MHSA_MATRIX_INITIAL_RESTING_COLOR);
        this.matrixRestingEmissiveIntensity = 0.1; // Default low emissive intensity
        this.matrixRestingOpacity = 0.7; // Default opacity for resting matrices

        this.brightGreen = new THREE.Color(MHSA_BRIGHT_GREEN);
        this.darkTintedGreen = new THREE.Color(MHSA_DARK_TINTED_GREEN);
        this.brightBlue = new THREE.Color(MHSA_BRIGHT_BLUE);
        this.darkTintedBlue = new THREE.Color(MHSA_DARK_TINTED_BLUE);
        this.brightRed = new THREE.Color(MHSA_BRIGHT_RED);
        this.darkTintedRed = new THREE.Color(MHSA_DARK_TINTED_RED);

        // Mode control (e.g., 'temp', 'perm', etc.)
        this.mode = mode;

        // Temp-mode bookkeeping
        this._tempModeCompleted = false;
        this._tempAllOutputVectors = []; // K,Q,V combined
        this._tempKOutputVectors = [];   // Only central K vectors

        this._setupMHSAVisualizations();
    }

    _setupMHSAVisualizations() {
        const darkGrayColor = new THREE.Color(0x404040);
        const matrixOpacity = 0.7;
        const matrixCenterY = this.mhsaBaseY + MHA_MATRIX_PARAMS.height / 2;

        for (let i = 0; i < NUM_HEAD_SETS_LAYER; i++) {
            const headSetWidth = MHA_INTERNAL_MATRIX_SPACING * 2 + MHA_MATRIX_PARAMS.width;
            const currentHeadSetBaseX = this.branchX - MHA_INTERNAL_MATRIX_SPACING + i * (headSetWidth + HEAD_SET_GAP_LAYER);

            const x_q = currentHeadSetBaseX;
            const x_k = currentHeadSetBaseX + MHA_INTERNAL_MATRIX_SPACING;
            const x_v = currentHeadSetBaseX + MHA_INTERNAL_MATRIX_SPACING * 2;

            const queryMatrix = new WeightMatrixVisualization(
                null, new THREE.Vector3(x_q, matrixCenterY, 0),
                MHA_MATRIX_PARAMS.width, MHA_MATRIX_PARAMS.height, MHA_MATRIX_PARAMS.depth,
                MHA_MATRIX_PARAMS.topWidthFactor, MHA_MATRIX_PARAMS.cornerRadius, MHA_MATRIX_PARAMS.numberOfSlits,
                MHA_MATRIX_PARAMS.slitWidth, MHA_MATRIX_PARAMS.slitDepthFactor,
                MHA_MATRIX_PARAMS.slitBottomWidthFactor, MHA_MATRIX_PARAMS.slitTopWidthFactor
            );
            queryMatrix.setColor(darkGrayColor);
            queryMatrix.group.children.forEach(child => {
                if (child.material) {
                    child.material.transparent = true;
                    child.material.opacity = matrixOpacity;
                }
            });
            this.scene.add(queryMatrix.group);
            this.mhaVisualizations.push(queryMatrix);

            const keyMatrix = new WeightMatrixVisualization(
                null, new THREE.Vector3(x_k, matrixCenterY, 0),
                MHA_MATRIX_PARAMS.width, MHA_MATRIX_PARAMS.height, MHA_MATRIX_PARAMS.depth,
                MHA_MATRIX_PARAMS.topWidthFactor, MHA_MATRIX_PARAMS.cornerRadius, MHA_MATRIX_PARAMS.numberOfSlits,
                MHA_MATRIX_PARAMS.slitWidth, MHA_MATRIX_PARAMS.slitDepthFactor,
                MHA_MATRIX_PARAMS.slitBottomWidthFactor, MHA_MATRIX_PARAMS.slitTopWidthFactor
            );
            keyMatrix.setColor(darkGrayColor);
            keyMatrix.group.children.forEach(child => {
                if (child.material) {
                    child.material.transparent = true;
                    child.material.opacity = matrixOpacity;
                }
            });
            this.scene.add(keyMatrix.group);
            this.mhaVisualizations.push(keyMatrix);

            const valueMatrix = new WeightMatrixVisualization(
                null, new THREE.Vector3(x_v, matrixCenterY, 0),
                MHA_MATRIX_PARAMS.width, MHA_MATRIX_PARAMS.height, MHA_MATRIX_PARAMS.depth,
                MHA_MATRIX_PARAMS.topWidthFactor, MHA_MATRIX_PARAMS.cornerRadius, MHA_MATRIX_PARAMS.numberOfSlits,
                MHA_MATRIX_PARAMS.slitWidth, MHA_MATRIX_PARAMS.slitDepthFactor,
                MHA_MATRIX_PARAMS.slitBottomWidthFactor, MHA_MATRIX_PARAMS.slitTopWidthFactor
            );
            valueMatrix.setColor(darkGrayColor);
            valueMatrix.group.children.forEach(child => {
                if (child.material) {
                    child.material.transparent = true;
                    child.material.opacity = matrixOpacity;
                }
            });
            this.scene.add(valueMatrix.group);
            this.mhaVisualizations.push(valueMatrix);

            this.headsCentersX.push(x_k);
            this.headCoords.push({ q: x_q, k: x_k, v: x_v });
        }
    }

    areAllMHAVectorsInPosition(lanes) {
        if (!lanes || !lanes.length) return false;

        for (const lane of lanes) {
            if (!lane.upwardCopies || lane.upwardCopies.length !== NUM_HEAD_SETS_LAYER) {
                return false; 
            }

            for (let headIdx = 0; headIdx < NUM_HEAD_SETS_LAYER; headIdx++) {
                const kVec = lane.upwardCopies[headIdx];
                if (!kVec || Math.abs(kVec.group.position.y - this.headStopY) > 0.1) {
                    return false; 
                }
                if (!kVec.userData.sideSpawned) {
                    return false; 
                }
            }

            if (!lane.sideCopies || lane.sideCopies.length !== NUM_HEAD_SETS_LAYER * 2) {
                return false;
            }

            for (const sideCopyObj of lane.sideCopies) {
                if (!sideCopyObj || !sideCopyObj.vec) {
                    return false;
                }
                if (Math.abs(sideCopyObj.vec.group.position.y - this.headStopY) > 0.1) {
                    return false; 
                }
                if (Math.abs(sideCopyObj.vec.group.position.x - sideCopyObj.targetX) > 0.1) {
                    return false; 
                }
            }
        }
        return true;
    }

    animateVectorMatrixPassThrough(vector, matrix, brightMatrixColor, darkTintedMatrixColor, finalVectorHue, passThroughY, duration, riseOffset, riseDurationVal, outLength, animationCompletionCallback, vectorCategory = 'K') {
        if (typeof TWEEN === 'undefined') {
            console.error("Global TWEEN object not loaded for MHSAAnimation!");
            if (animationCompletionCallback) animationCompletionCallback();
            return;
        }
        if (!vector || !matrix) {
            console.warn("Missing vector or matrix for pass-through animation in MHSA.");
            if (animationCompletionCallback) animationCompletionCallback();
            return;
        }

        // Create a trail line that follows the vector only inside the matrix
        const passThroughTrail = createTrailLine(this.scene, TRAIL_LINE_COLOR);
        const matrixBottomY = this.mhsa_matrix_center_y - MHA_MATRIX_PARAMS.height / 2;
        const matrixTopY = this.mhsa_matrix_center_y + MHA_MATRIX_PARAMS.height / 2;

        const originalMatrixEmissive = matrix.mesh.material.emissive.clone();
        const originalMatrixIntensity = matrix.mesh.material.emissiveIntensity;
        let finalVisualsApplied = false; // Flag to ensure processed visuals are applied once
        let initialDimensionChangeApplied = false; // Flag for early dimension change
        const tweenState = { y: vector.group.position.y, progress: 0, colorR: 1, colorG: 1, colorB: 1, matrixEmissiveIntensity: originalMatrixIntensity };
        const initialVecColor = new THREE.Color();
        if(vector.mesh.instanceColor) { vector.mesh.getColorAt(0, initialVecColor); } else { initialVecColor.setRGB(0.5,0.5,0.5); }
        tweenState.colorR = initialVecColor.r; tweenState.colorG = initialVecColor.g; tweenState.colorB = initialVecColor.b;

        new TWEEN.Tween(tweenState)
            .to({ y: passThroughY + riseOffset, progress: 1.0, colorR: 1.0, colorG: 1.0, colorB: 1.0, matrixEmissiveIntensity: MHSA_MATRIX_MAX_EMISSIVE_INTENSITY }, duration)
            .easing(TWEEN.Easing.Quadratic.InOut)
            .onUpdate(() => {
                // Move vector
                vector.group.position.y = tweenState.y;
                // Only update trail while the vector is inside the matrix region
                const y = vector.group.position.y;

                // Apply 64-dimensional conversion as vector enters the matrix bottom
                const matrixBottomEntryY = this.mhsa_matrix_center_y - MHA_MATRIX_PARAMS.height / 2;
                if (!initialDimensionChangeApplied && y >= matrixBottomEntryY) {
                    // ------------------------------------------------------------------
                    // 1) Create new lightweight vector (still uses the same class)
                    // ------------------------------------------------------------------
                    const smallVec = new VectorVisualizationInstancedPrism(
                        vector.rawData.slice(0, outLength), // supply d_model slice
                        vector.group.position.clone(),
                        3 // subsection count for colour gradient
                    );

                    // ------------------------------------------------------------------
                    // 2) Swap out the heavyweight 768-unit vector for the lightweight one
                    // ------------------------------------------------------------------
                    this.scene.add(smallVec.group);

                    // Keep a handle to the original (large) vector before we overwrite
                    const heavyVec = vector;

                    // From now on animate using the lightweight reference
                    vector = smallVec;
                    initialDimensionChangeApplied = true;

                    // Remove & dispose the heavyweight vector to free GPU/CPU resources
                    this.scene.remove(heavyVec.group);
                    if (typeof heavyVec.dispose === 'function') heavyVec.dispose();

                    // ------------------------------------------------------------------
                    // Apply final visuals immediately to the lightweight vector
                    // ------------------------------------------------------------------
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
                                maxLightness: 0.8
                            }
                        },
                        { setHiddenToBlack: false }
                    );

                    // ------------------------------------------------------------------
                    // END heavy → lightweight swap optimisation
                    // ------------------------------------------------------------------
                }

                // Only update the trail while the vector is rising towards the matrix (below it)
                if (y < matrixBottomY) {
                    updateTrail(passThroughTrail, vector.group.position);
                }

                let currentMatrixTargetColor = new THREE.Color();
                let currentEmissiveIntensity = this.matrixRestingEmissiveIntensity;
                let t = 0;
                let emissiveTargetColorForMatrix = new THREE.Color();

                // Use smoothstep for a softer transition from dark → bright.
                if (tweenState.progress < MHSA_PASS_THROUGH_BRIGHTEN_RATIO) {
                    const raw = tweenState.progress / MHSA_PASS_THROUGH_BRIGHTEN_RATIO;
                    t = THREE.MathUtils.smoothstep(raw, 0, 1);
                    currentMatrixTargetColor = this.matrixInitialRestingColor.clone().lerp(brightMatrixColor, t);
                    currentEmissiveIntensity = THREE.MathUtils.lerp(this.matrixRestingEmissiveIntensity, MHSA_MATRIX_MAX_EMISSIVE_INTENSITY, t);
                    matrix.setOpacity(THREE.MathUtils.lerp(this.matrixRestingOpacity, 1.0, t));
                    emissiveTargetColorForMatrix = currentMatrixTargetColor.clone();
                } else if (tweenState.progress < MHSA_PASS_THROUGH_BRIGHTEN_RATIO + MHSA_PASS_THROUGH_DIM_RATIO) {
                    const raw = (tweenState.progress - MHSA_PASS_THROUGH_BRIGHTEN_RATIO) / MHSA_PASS_THROUGH_DIM_RATIO;
                    t = THREE.MathUtils.smoothstep(raw, 0, 1);
                    currentMatrixTargetColor = brightMatrixColor.clone().lerp(darkTintedMatrixColor, t);
                    currentEmissiveIntensity = THREE.MathUtils.lerp(MHSA_MATRIX_MAX_EMISSIVE_INTENSITY, this.matrixRestingEmissiveIntensity, t);
                    matrix.setOpacity(1.0);
                    emissiveTargetColorForMatrix = currentMatrixTargetColor.clone();
                } else {
                    currentMatrixTargetColor = darkTintedMatrixColor.clone();
                    currentEmissiveIntensity = this.matrixRestingEmissiveIntensity;
                    matrix.setOpacity(this.matrixRestingOpacity);
                    emissiveTargetColorForMatrix = currentMatrixTargetColor.clone();
                }
                matrix.setColor(currentMatrixTargetColor);
                matrix.setEmissive(emissiveTargetColorForMatrix, currentEmissiveIntensity);

                const numCentralUnits = outLength;
                const startVisibleIndex = Math.floor((VECTOR_LENGTH_PRISM - numCentralUnits) / 2);
                const endVisibleIndex = startVisibleIndex + numCentralUnits - 1;
 
                 // Progressive shrink animation for outer prisms, only if the initial snap to 64-dim hasn't happened yet.
                 if (!initialDimensionChangeApplied) {
                     // Content of this block (progressive shrink loop) is removed for optimization.
                     // The vector will maintain its full appearance until it hits the matrix boundary,
                     // at which point applyProcessedVisuals handles the instantaneous change.
                 }
            })
            .onComplete(() => {
                matrix.setColor(darkTintedMatrixColor);
                matrix.setEmissive(darkTintedMatrixColor, this.matrixRestingEmissiveIntensity);
                matrix.setOpacity(1.0);

                // Ensure final visuals are applied at least once (in case progress never reached threshold).
                if (!finalVisualsApplied) {
                    const processedData = vector.rawData.slice(0, outLength);
                    vector.applyProcessedVisuals(processedData, outLength, {
                        numKeyColors: 3,
                        generationOptions: {
                            type: 'monochromatic',
                            baseHue: finalVectorHue,
                            saturation: 0.9,
                            minLightness: 0.4,
                            maxLightness: 0.8
                        }
                    });
                    finalVisualsApplied = true;
                }

                // No additional rise tween needed – the vector is already at its
                // final Y.

                // ------------------------------------------------------------------
                //  Temp-mode collection of finished vectors
                // ------------------------------------------------------------------
                if (this.mode === 'temp') {
                    this._tempAllOutputVectors.push(vector);
                    if (vectorCategory === 'K') {
                        this._tempKOutputVectors.push(vector);
                    }
                }

                // Invoke caller-supplied callback last so that state above is ready.
                if (animationCompletionCallback) animationCompletionCallback();
            })
            .start();
    }

    initiateParallelHeadPassThroughAnimations(allLanes) {
        if (this.mhaPassThroughPhase !== 'ready_for_parallel_pass_through') return;
        console.log("MHSAAnimation: Initiating Parallel MHSA Head Pass-Through Animations...");
        this.mhaPassThroughPhase = 'parallel_pass_through_active';

        let totalAnimationsToComplete = allLanes.length * NUM_HEAD_SETS_LAYER * 3;
        let animationsCompleted = 0;

        const singleAnimationDone = () => {
            animationsCompleted++;
            if (animationsCompleted >= totalAnimationsToComplete) {
                console.log("MHSAAnimation: All MHSA parallel pass-through animations complete.");
                this.mhaPassThroughPhase = 'mha_pass_through_complete';

                // ---------------------------------------------------------------
                //  TEMP MODE: post pass-through behaviour
                // ---------------------------------------------------------------
                if (this.mode === 'temp' && !this._tempModeCompleted) {
                    this._applyTempModeBehaviour();
                    this._tempModeCompleted = true;
                }
            }
        };

        allLanes.forEach((lane) => {
            for (let headIdx = 0; headIdx < NUM_HEAD_SETS_LAYER; headIdx++) {
                const kVec = lane.upwardCopies[headIdx];
                const kMatrix = this.mhaVisualizations[headIdx * 3 + 1];
                this.animateVectorMatrixPassThrough(kVec, kMatrix, this.brightGreen, this.darkTintedGreen, 0.333, this.mhaPassThroughTargetY, this.mhaPassThroughDuration, this.mhaResultRiseOffsetY, this.mhaResultRiseDuration, this.outputVectorLength, singleAnimationDone, 'K');

                const qSideCopy = lane.sideCopies.find(sc => sc.headIndex === headIdx && sc.type === 'Q');
                if (qSideCopy && qSideCopy.vec) {
                    this.animateVectorMatrixPassThrough(qSideCopy.vec, qSideCopy.matrixRef, this.brightBlue, this.darkTintedBlue, 0.666, this.mhaPassThroughTargetY, this.mhaPassThroughDuration, this.mhaResultRiseOffsetY, this.mhaResultRiseDuration, this.outputVectorLength, singleAnimationDone, 'Q');
                } else { totalAnimationsToComplete--; }

                const vSideCopy = lane.sideCopies.find(sc => sc.headIndex === headIdx && sc.type === 'V');
                if (vSideCopy && vSideCopy.vec) {
                    this.animateVectorMatrixPassThrough(vSideCopy.vec, vSideCopy.matrixRef, this.brightRed, this.darkTintedRed, 0.0, this.mhaPassThroughTargetY, this.mhaPassThroughDuration, this.mhaResultRiseOffsetY, this.mhaResultRiseDuration, this.outputVectorLength, singleAnimationDone, 'V');
                } else { totalAnimationsToComplete--; }
            }
        });
        if (totalAnimationsToComplete === 0 && allLanes.length > 0) {
             console.log("MHSAAnimation: No valid K,Q,V vectors found to animate for parallel pass-through.");
             this.mhaPassThroughPhase = 'mha_pass_through_complete';
        }
    }
    
    update(deltaTime, timeNow, lanes) {
        lanes.forEach((lane, idx) => {
            if (lane.horizPhase === 'travelMHSA') {
                const tVec = lane.travellingVec;
                if (!tVec) return;

                const targetHeadIdx = lane.headIndex || 0;
                if (targetHeadIdx >= this.headsCentersX.length) {
                    tVec.group.visible = false;
                    lane.horizPhase = 'finishedHeads';
                    return; 
                }
                const targetX = this.headsCentersX[targetHeadIdx];
                const dx = ANIM_HORIZ_SPEED * SPEED_MULT * deltaTime;

                if (tVec.group.position.x < targetX - 0.01) {
                    tVec.group.position.x = Math.min(targetX, tVec.group.position.x + dx);
                } else {
                    const dupeData = [...tVec.rawData];
                    const upVec = new VectorVisualizationInstancedPrism(dupeData, tVec.group.position.clone());
                    this.scene.add(upVec.group);
                    upVec.userData = { headIndex: targetHeadIdx, sideSpawned: false, sideSpawnRequested: false, sideSpawnTime: 0 };
                    lane.upwardCopies.push(upVec);
                    
                    const upTrail = createTrailLine(this.scene, TRAIL_LINE_COLOR);
                    updateTrail(upTrail, upVec.group.position);
                    lane.upwardTrails = lane.upwardTrails || [];
                    lane.upwardTrails.push(upTrail);

                    lane.headIndex = targetHeadIdx + 1;
                    if (lane.headIndex >= NUM_HEAD_SETS_LAYER) {
                        tVec.group.visible = false;
                        lane.horizPhase = 'finishedHeads';
                    }
                }
            } else if (lane.horizPhase === 'finishedHeads') {
                // No-op
            }

            if (lane.upwardCopies && lane.upwardCopies.length) {
                lane.upwardCopies.forEach((upVec, trailIdx) => {
                    if (upVec.group.position.y < this.headStopY) {
                        upVec.group.position.y = Math.min(this.headStopY, upVec.group.position.y + MHSA_DUPLICATE_VECTOR_RISE_SPEED * SPEED_MULT * deltaTime);
                        if (lane.upwardTrails && lane.upwardTrails[trailIdx]) {
                            updateTrail(lane.upwardTrails[trailIdx], upVec.group.position);
                        }
                    }
                });
            }

            if (lane.upwardCopies) {
                lane.upwardCopies.forEach(centerVec => {
                    if (!centerVec.userData.sideSpawnRequested && Math.abs(centerVec.group.position.y - this.headStopY) < 0.1) {
                        centerVec.userData.sideSpawnRequested = true;
                        centerVec.userData.sideSpawnTime = timeNow + SIDE_COPY_DELAY_MS / SPEED_MULT;
                    }
                    if (centerVec.userData.sideSpawnRequested && !centerVec.userData.sideSpawned && timeNow >= centerVec.userData.sideSpawnTime) {
                        const hIdx = centerVec.userData.headIndex;
                        const coord = this.headCoords[hIdx];
                        if (coord) {
                            const qMatrixForHead = this.mhaVisualizations[hIdx * 3];
                            const vMatrixForHead = this.mhaVisualizations[hIdx * 3 + 2];

                            const qVec = new VectorVisualizationInstancedPrism(centerVec.rawData.slice(), centerVec.group.position.clone());
                            const vVec = new VectorVisualizationInstancedPrism(centerVec.rawData.slice(), centerVec.group.position.clone());
                            this.scene.add(qVec.group);
                            this.scene.add(vVec.group);
                            
                            lane.sideCopies = lane.sideCopies || [];
                            lane.sideCopies.push({ vec: qVec, targetX: coord.q, type: 'Q', matrixRef: qMatrixForHead, headIndex: hIdx });
                            lane.sideCopies.push({ vec: vVec, targetX: coord.v, type: 'V', matrixRef: vMatrixForHead, headIndex: hIdx });
                            
                            lane.sideTrails = lane.sideTrails || [];
                            lane.sideTrails.push(createTrailLine(this.scene, TRAIL_LINE_COLOR));
                            lane.sideTrails.push(createTrailLine(this.scene, TRAIL_LINE_COLOR));
                            
                            updateTrail(lane.sideTrails[lane.sideTrails.length-2], qVec.group.position);
                            updateTrail(lane.sideTrails[lane.sideTrails.length-1], vVec.group.position);
                            centerVec.userData.sideSpawned = true;
                        }
                    }
                });
            }

            if (this.mhaPassThroughPhase === 'positioning_mha_vectors' && lane.sideCopies && lane.sideCopies.length) {
                lane.sideCopies.forEach((obj, trailIdx) => {
                    const v = obj.vec;
                    const dx = SIDE_COPY_HORIZ_SPEED * SPEED_MULT * deltaTime;
                    if (Math.abs(v.group.position.x - obj.targetX) > 0.01) {
                        const dir = v.group.position.x < obj.targetX ? 1 : -1;
                        v.group.position.x += dir * dx;
                        if ((dir === 1 && v.group.position.x > obj.targetX) || (dir === -1 && v.group.position.x < obj.targetX))
                            v.group.position.x = obj.targetX;
                    }
                    v.group.position.y = this.headStopY;
                    
                    if (lane.sideTrails && lane.sideTrails[trailIdx]) {
                         updateTrail(lane.sideTrails[trailIdx], v.group.position);
                    }
                });
            }
        });

        if (this.mhaPassThroughPhase === 'positioning_mha_vectors') {
            if (this.areAllMHAVectorsInPosition(lanes)) {
                this.mhaPassThroughPhase = 'ready_for_parallel_pass_through';
                console.log("MHSAAnimation: All MHSA vectors are in position. Ready for PARALLEL pass-through.");
                this.initiateParallelHeadPassThroughAnimations(lanes);
            }
        }

        // Update merge trails
        if (this._mergeLaneTrails) {
            // Lane trails are updated during tweens; no continuous endpoint updates needed.
        }
    }

    dispose() {
        // Standard THREE.js objects added to scene are usually handled by scene traversal on global cleanup.
    }

    _applyTempModeBehaviour() {
        const grayColor = new THREE.Color(0x606060);
        const startVisibleIdx = Math.floor((VECTOR_LENGTH_PRISM - this.outputVectorLength) / 2);
        const endVisibleIdx = startVisibleIdx + this.outputVectorLength - 1;

        this._tempAllOutputVectors.forEach(vec => {
            if (!vec || !vec.mesh) return;

            // Gray-out only the visible 64-dim region so outer hidden prisms stay hidden
            for (let i = startVisibleIdx; i <= endVisibleIdx; i++) {
                vec.setInstanceAppearance(i, 0, grayColor);
            }

            // Lower emissive intensity on the shared material (if present)
            if (vec.mesh.material && typeof vec.mesh.material.emissiveIntensity === 'number') {
                vec.mesh.material.emissiveIntensity = 0.05;
            }

            // Ensure material supports transparency
            if (vec.mesh.material) {
                vec.mesh.material.transparent = true;
                vec.mesh.material.opacity = 1.0; // start fully opaque gray
                vec.mesh.material.needsUpdate = true;
            }

            // Fade out the gray vectors to make them less visible
            if (vec.mesh.material && typeof TWEEN !== 'undefined') {
                new TWEEN.Tween({ op: 1.0 })
                    .to({ op: 0.2 }, 600)
                    .easing(TWEEN.Easing.Quadratic.Out)
                    .onUpdate(function(obj){
                        vec.mesh.material.opacity = obj.op;
                        vec.mesh.material.needsUpdate = true;
                    })
                    .start();
            } else if (vec.mesh.material) {
                vec.mesh.material.opacity = 0.2;
            }
        });

        // 2) Spawn new decorative vectors (64-dim) above each central K vector and fade them in
        const verticalOffset = 60; // world units above existing vector

        this._tempDecorativeVecs = []; // store objects {vec, laneZ} for later merge

        this._tempKOutputVectors.forEach(kVec => {
            if (!kVec || !kVec.group) return;

            // Build raw 768-dim data with 3 random switch points for varied gradient
            const rawData = [];
            const switchPoints = new Set();
            while (switchPoints.size < 3) {
                const idx = Math.floor(Math.random() * VECTOR_LENGTH_PRISM);
                switchPoints.add(idx);
            }
            const sortedSwitch = Array.from(switchPoints).sort((a, b) => a - b);
            let curVal = Math.random() * 2 - 1;
            let nextSwitch = sortedSwitch.shift();
            for (let i = 0; i < VECTOR_LENGTH_PRISM; i++) {
                if (i === nextSwitch) {
                    curVal = Math.random() * 2 - 1;
                    nextSwitch = sortedSwitch.shift();
                }
                rawData.push(curVal);
            }

            const spawnPos = kVec.group.position.clone().add(new THREE.Vector3(0, verticalOffset, 0));
            const decoVec = new VectorVisualizationInstancedPrism(rawData, spawnPos, 3);

            // Hide outer prisms and snap decorative vec to 64-dim geometry
            decoVec.applyProcessedVisuals(
                rawData.slice(startVisibleIdx, startVisibleIdx + this.outputVectorLength),
                this.outputVectorLength,
                { numKeyColors: this.outputVectorLength },
                { setHiddenToBlack: true }
            );

            // Override visible region with a random two-color gradient
            const startColor = new THREE.Color().setHSL(Math.random(), 0.9, 0.6);
            const endColor = new THREE.Color().setHSL(Math.random(), 0.9, 0.6);
            const visibleCount = this.outputVectorLength;
            for (let vi = 0; vi < visibleCount; vi++) {
                const idx = startVisibleIdx + vi;
                const t = visibleCount > 1 ? vi / (visibleCount - 1) : 0;
                const col = startColor.clone().lerp(endColor, t);
                decoVec.setInstanceAppearance(idx, 0, col);
            }

            this.scene.add(decoVec.group);

            // Keep reference for merge phase
            this._tempDecorativeVecs.push({ vec: decoVec, laneZ: kVec.group.position.z });

            // Create a trail line connecting the grayed-out vector to its colored vector above
            const connectionTrail = createTrailLine(this.scene, TRAIL_LINE_COLOR);
            // Add the starting point (gray vector position)
            updateTrail(connectionTrail, kVec.group.position);
            // Add the ending point (colored vector position)
            updateTrail(connectionTrail, spawnPos);

            // Start invisible: set material opacity to 0
            if (decoVec.mesh && decoVec.mesh.material) {
                decoVec.mesh.material.transparent = true;
                decoVec.mesh.material.opacity = 0.0;
            }

            if (typeof TWEEN !== 'undefined') {
                const mat = decoVec.mesh.material;
                new TWEEN.Tween({ op: 0.0 })
                    .to({ op: 1.0 }, 800)
                    .easing(TWEEN.Easing.Quadratic.InOut)
                    .onUpdate(function(o){
                        mat.opacity = o.op;
                        mat.needsUpdate = true;
                    })
                    .start();
            }
        });

        // After decorative vectors begin fading in, further dim gray vectors for subtlety
        if (typeof TWEEN !== 'undefined') {
            this._tempAllOutputVectors.forEach(vec => {
                if (!vec || !vec.mesh || !vec.mesh.material) return;
                const mat = vec.mesh.material;
                new TWEEN.Tween({ op: mat.opacity })
                    .to({ op: 0.05 }, 800)
                    .delay(800)
                    .easing(TWEEN.Easing.Quadratic.Out)
                    .onUpdate(function(o){
                        mat.opacity = o.op;
                        mat.needsUpdate = true;
                    })
                    .start();
            });
        }

        // ------------------------------------------------------
        //   Begin merge-to-row-vector phase after fade-in delay
        // ------------------------------------------------------
        if (typeof TWEEN !== 'undefined') {
            // Start after decorative fade-in completes (800 ms)
            setTimeout(() => {
                this._startMergeToRowVectors();
            }, 900); // small extra buffer
        }
    }

    _startMergeToRowVectors() {
        if (!this._tempDecorativeVecs || this._tempDecorativeVecs.length === 0) return;

        this._mergeLaneTrails = new Map(); // laneZ -> trailObj

        // Build map laneZ -> array of decorative vectors
        const laneVectors = new Map();
        this._tempDecorativeVecs.forEach(obj => {
            const laneZ = obj.laneZ;
            if (!laneVectors.has(laneZ)) laneVectors.set(laneZ, []);
            laneVectors.get(laneZ).push(obj.vec);
        });

        const targetX = this.headsCentersX.length ? this.headsCentersX[0] : 0;

        laneVectors.forEach((vecList, laneZ) => {
            // create trail per lane
            const laneTrail = createTrailLine(this.scene, TRAIL_LINE_COLOR);
            if (laneTrail.line && laneTrail.line.material) {
                laneTrail.line.material.opacity = TRAIL_LINE_OPACITY / NUM_HEAD_SETS_LAYER;
                laneTrail.line.material.needsUpdate = true;
            }
            this._mergeLaneTrails.set(laneZ, laneTrail);

            // sort vectors by original x for consistent ordering
            vecList.sort((a, b) => a.group.position.x - b.group.position.x);

            vecList.forEach((vec, idx) => {
                const destX = targetX + (idx - (NUM_HEAD_SETS_LAYER - 1) / 2) * ROW_SEGMENT_SPACING;

                const distance = Math.abs(vec.group.position.x - destX);
                const durationMs = (distance / (ROW_MERGE_HORIZ_SPEED * SPEED_MULT)) * 1000;

                if (typeof TWEEN !== 'undefined') {
                    new TWEEN.Tween(vec.group.position)
                        .to({ x: destX }, durationMs)
                        .easing(TWEEN.Easing.Quadratic.InOut)
                        .onUpdate(() => {
                            updateTrail(laneTrail, vec.group.position);
                        })
                        .start();
                } else {
                    vec.group.position.x = destX;
                    updateTrail(laneTrail, vec.group.position);
                }
            });
        });
    }
} 