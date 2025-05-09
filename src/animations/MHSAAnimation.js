import * as THREE from 'three';
import { WeightMatrixVisualization } from '../components/WeightMatrixVisualization.js';
import { VectorVisualizationInstancedPrism } from '../components/VectorVisualizationInstancedPrism.js';
import { createTrailLine, updateTrail } from '../utils/trailUtils.js';
import {
    // Constants needed for setup & animation
    MHA_MATRIX_PARAMS,
    NUM_HEAD_SETS_LAYER,
    HEAD_SET_GAP_LAYER,
    MHA_INTERNAL_MATRIX_SPACING,
    HEAD_VECTOR_STOP_BELOW,
    ANIM_RISE_SPEED_HEAD,
    ANIM_HORIZ_SPEED,
    GLOBAL_ANIM_SPEED_MULT,
    SIDE_COPY_DELAY_MS,
    SIDE_COPY_HORIZ_SPEED,
    VECTOR_LENGTH_PRISM,
    HIDE_INSTANCE_Y_OFFSET,
} from '../utils/constants.js';

// Define speed multiplier
const SPEED_MULT = GLOBAL_ANIM_SPEED_MULT;

export class MHSAAnimation {
    constructor(scene, branchX, mhsaBaseY, clock) {
        this.scene = scene;
        this.branchX = branchX;
        this.mhsaBaseY = mhsaBaseY;
        this.clock = clock;

        this.mhaVisualizations = [];
        this.headsCentersX = [];
        this.headCoords = [];
        this.mhaPassThroughPhase = 'positioning_mha_vectors';

        this.mhsa_matrix_center_y = this.mhsaBaseY + MHA_MATRIX_PARAMS.height / 2;
        this.headStopY = this.mhsa_matrix_center_y - HEAD_VECTOR_STOP_BELOW;
        this.mhaPassThroughTargetY = this.mhsa_matrix_center_y + MHA_MATRIX_PARAMS.height / 2 + 20;
        this.mhaPassThroughDuration = 2000 / SPEED_MULT;
        this.outputVectorLength = 64;
        this.mhaResultRiseOffsetY = 50;
        this.mhaResultRiseDuration = 500 / SPEED_MULT;

        this.brightGreen = new THREE.Color(0x33FF33);
        this.darkTintedGreen = new THREE.Color(0x002200);
        this.brightBlue = new THREE.Color(0x6666FF);
        this.darkTintedBlue = new THREE.Color(0x000022);
        this.brightRed = new THREE.Color(0xFF3333);
        this.darkTintedRed = new THREE.Color(0x220000);

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

    animateVectorMatrixPassThrough(vector, matrix, brightMatrixColor, darkTintedMatrixColor, finalVectorHue, passThroughY, duration, riseOffset, riseDurationVal, outLength, animationCompletionCallback) {
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

        const originalMatrixEmissive = matrix.mesh.material.emissive.clone();
        const originalMatrixIntensity = matrix.mesh.material.emissiveIntensity;
        const tweenState = { y: vector.group.position.y, progress: 0, colorR: 1, colorG: 1, colorB: 1, matrixEmissiveIntensity: originalMatrixIntensity };
        const initialVecColor = new THREE.Color();
        if(vector.mesh.instanceColor) { vector.mesh.getColorAt(0, initialVecColor); } else { initialVecColor.setRGB(0.5,0.5,0.5); }
        tweenState.colorR = initialVecColor.r; tweenState.colorG = initialVecColor.g; tweenState.colorB = initialVecColor.b;

        new TWEEN.Tween(tweenState)
            .to({ y: passThroughY, progress: 1.0, colorR: 1.0, colorG: 1.0, colorB: 1.0, matrixEmissiveIntensity: 1.5 }, duration)
            .easing(TWEEN.Easing.Quadratic.InOut)
            .onUpdate(() => {
                vector.group.position.y = tweenState.y;
                const numCentralUnits = outLength;
                const startVisibleIndex = Math.floor((VECTOR_LENGTH_PRISM - numCentralUnits) / 2);
                const endVisibleIndex = startVisibleIndex + numCentralUnits - 1;
                const currentWhite = new THREE.Color(tweenState.colorR, tweenState.colorG, tweenState.colorB);
                for (let i = 0; i < VECTOR_LENGTH_PRISM; i++) {
                    let targetScaleY = vector.getUniformHeight();
                    let instanceYOffset = 0;
                    if (i < startVisibleIndex || i > endVisibleIndex) {
                        targetScaleY = THREE.MathUtils.lerp(vector.getUniformHeight(), 0.001, tweenState.progress);
                        if (targetScaleY < 0.01 && tweenState.progress > 0.5) { instanceYOffset = HIDE_INSTANCE_Y_OFFSET - vector.group.position.y; }
                    }
                    vector.setInstanceAppearance(i, instanceYOffset, currentWhite, new THREE.Vector3(vector.getWidthScale(), targetScaleY, vector.getDepthScale()));
                }
                matrix.setColor(brightMatrixColor); matrix.setEmissive(brightMatrixColor, tweenState.matrixEmissiveIntensity); matrix.setOpacity(1.0);
            })
            .onComplete(() => {
                const processedData = vector.rawData.slice(0, outLength);
                vector.applyProcessedVisuals(processedData, outLength, { numKeyColors: 3, generationOptions: { type: 'monochromatic', baseHue: finalVectorHue, saturation: 0.9, minLightness: 0.4, maxLightness: 0.8 }});
                vector.group.position.set(vector.group.position.x, passThroughY, vector.group.position.z);
                matrix.setColor(darkTintedMatrixColor); matrix.setEmissive(darkTintedMatrixColor, 0.1); matrix.setOpacity(1.0);

                new TWEEN.Tween(vector.group.position)
                    .to({ y: passThroughY + riseOffset }, riseDurationVal)
                    .easing(TWEEN.Easing.Cubic.Out)
                    .onComplete(animationCompletionCallback)
                    .start();
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
            }
        };

        allLanes.forEach((lane) => {
            for (let headIdx = 0; headIdx < NUM_HEAD_SETS_LAYER; headIdx++) {
                const kVec = lane.upwardCopies[headIdx];
                const kMatrix = this.mhaVisualizations[headIdx * 3 + 1];
                this.animateVectorMatrixPassThrough(kVec, kMatrix, this.brightGreen, this.darkTintedGreen, 0.333, this.mhaPassThroughTargetY, this.mhaPassThroughDuration, this.mhaResultRiseOffsetY, this.mhaResultRiseDuration, this.outputVectorLength, singleAnimationDone);

                const qSideCopy = lane.sideCopies.find(sc => sc.headIndex === headIdx && sc.type === 'Q');
                if (qSideCopy && qSideCopy.vec) {
                    this.animateVectorMatrixPassThrough(qSideCopy.vec, qSideCopy.matrixRef, this.brightBlue, this.darkTintedBlue, 0.666, this.mhaPassThroughTargetY, this.mhaPassThroughDuration, this.mhaResultRiseOffsetY, this.mhaResultRiseDuration, this.outputVectorLength, singleAnimationDone);
                } else { totalAnimationsToComplete--; }

                const vSideCopy = lane.sideCopies.find(sc => sc.headIndex === headIdx && sc.type === 'V');
                if (vSideCopy && vSideCopy.vec) {
                    this.animateVectorMatrixPassThrough(vSideCopy.vec, vSideCopy.matrixRef, this.brightRed, this.darkTintedRed, 0.0, this.mhaPassThroughTargetY, this.mhaPassThroughDuration, this.mhaResultRiseOffsetY, this.mhaResultRiseDuration, this.outputVectorLength, singleAnimationDone);
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
                    
                    const upTrail = createTrailLine(this.scene, 0xffffff);
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
                        upVec.group.position.y = Math.min(this.headStopY, upVec.group.position.y + ANIM_RISE_SPEED_HEAD * SPEED_MULT * deltaTime);
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
                            lane.sideTrails.push(createTrailLine(this.scene, 0xffffff));
                            lane.sideTrails.push(createTrailLine(this.scene, 0xffffff));
                            
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
    }

    dispose() {
        // Standard THREE.js objects added to scene are usually handled by scene traversal on global cleanup.
    }
} 