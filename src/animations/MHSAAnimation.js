import * as THREE from 'three';
import { WeightMatrixVisualization } from '../components/WeightMatrixVisualization.js';
import { VectorVisualizationInstancedPrism } from '../components/VectorVisualizationInstancedPrism.js';
// Trail functionality removed – no-ops keep API intact
function createTrailLine() {
  return {
    line: { material: { opacity: 0, needsUpdate: false } },
    geometry: {
      attributes: { position: { setXYZ: () => {}, needsUpdate: false } },
      setDrawRange: () => {},
      computeBoundingSphere: () => {},
    },
    positions: [],
    points: [],
    isFrozen: false,
  };
}
function updateTrail() {}

import { mapValueToColor } from '../utils/colors.js';
import { MHSA_DUPLICATE_VECTOR_RISE_SPEED, MHSA_PASS_THROUGH_TOTAL_DURATION_MS, MHSA_PASS_THROUGH_BRIGHTEN_RATIO, MHSA_PASS_THROUGH_DIM_RATIO, MHSA_MATRIX_MAX_EMISSIVE_INTENSITY, MHSA_MATRIX_INITIAL_RESTING_COLOR, MHSA_BRIGHT_GREEN, MHSA_DARK_TINTED_GREEN, MHSA_BRIGHT_BLUE, MHSA_DARK_TINTED_BLUE, MHSA_BRIGHT_RED, MHSA_DARK_TINTED_RED, MHSA_RESULT_RISE_OFFSET_Y, MHSA_HEAD_VECTOR_STOP_BELOW,  MHA_FINAL_Q_COLOR, MHA_FINAL_K_COLOR, MHA_FINAL_V_COLOR, MHA_OUTPUT_PROJECTION_MATRIX_Y_OFFSET_ABOVE_ROW, MHA_OUTPUT_PROJECTION_MATRIX_PARAMS, MHA_OUTPUT_PROJECTION_MATRIX_COLOR } from './LayerAnimationConstants.js';
import { INACTIVE_COMPONENT_COLOR } from '../utils/constants.js';
import {
    // Constants needed for setup & animation
    MHA_MATRIX_PARAMS,
    NUM_VECTOR_LANES,
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
    ANIM_RISE_SPEED_ORIGINAL,
    ANIM_RISE_SPEED_POST_SPLIT_LN1,
    ANIM_RISE_SPEED_POST_SPLIT_LN2,
    ORIGINAL_TO_PROCESSED_GAP,
    PRISM_DIMENSIONS_PER_UNIT,
    BRANCH_X
} from '../utils/constants.js';
import { startPrismAdditionAnimation } from '../utils/additionUtils.js';
import { buildMHAVisuals, VectorRouter, PassThroughAnimator, SelfAttentionAnimator } from './mhsa/index.js';
import { animateVectorMatrixPassThrough as animateVectorMatrixPassThroughExternal } from './mhsa/VectorMatrixPassThrough.js';

// Define speed multiplier
const SPEED_MULT = GLOBAL_ANIM_SPEED_MULT;

export class MHSAAnimation {
    /**
     * Global toggle.  Set `MHSAAnimation.ENABLE_SELF_ATTENTION = true` **before**
     * constructing an instance to activate the (placeholder) self-attention
     * sub-animation.  The big 12-layer demo leaves this `false` so nothing runs.
     */
    static ENABLE_SELF_ATTENTION = false;

    constructor(parentGroup, branchX, mhsaBaseY, clock, mode = 'temp', opts = {}) {
        this.parentGroup = parentGroup;
        this.branchX = branchX;
        this.mhsaBaseY = mhsaBaseY;
        this.clock = clock;

        // Speed at which residual-stream vectors rise while branched
        // during MHSA/MLP processing. Starts with the LN1 value.
        this.postSplitRiseSpeed = ANIM_RISE_SPEED_POST_SPLIT_LN1;

        // Core positional helpers & state flags
        this.mhaPassThroughPhase = 'positioning_mha_vectors';

        this.mhsa_matrix_center_y = this.mhsaBaseY + MHA_MATRIX_PARAMS.height / 2;
        this.headStopY            = this.mhsa_matrix_center_y - MHSA_HEAD_VECTOR_STOP_BELOW;
        this.mhaPassThroughTargetY = this.mhsa_matrix_center_y + MHA_MATRIX_PARAMS.height / 2 + 20;

        // Durations & dimensional constants
        this.outputVectorLength      = 64;
        this.mhaResultRiseOffsetY    = MHSA_RESULT_RISE_OFFSET_Y;
        this.mhaResultRiseDuration   = 500 / SPEED_MULT;
        this.mhaPassThroughDuration  = MHSA_PASS_THROUGH_TOTAL_DURATION_MS / SPEED_MULT;

        // Colours & material defaults
        this.matrixInitialRestingColor     = new THREE.Color(MHSA_MATRIX_INITIAL_RESTING_COLOR);
        this.matrixRestingEmissiveIntensity = 0.1;
        this.matrixRestingOpacity           = 1.0;

        this.brightGreen      = new THREE.Color(MHSA_BRIGHT_GREEN);
        this.darkTintedGreen  = new THREE.Color(MHSA_DARK_TINTED_GREEN);
        this.brightBlue       = new THREE.Color(MHSA_BRIGHT_BLUE);
        this.darkTintedBlue   = new THREE.Color(MHSA_DARK_TINTED_BLUE);
        this.brightRed        = new THREE.Color(MHSA_BRIGHT_RED);
        this.darkTintedRed    = new THREE.Color(MHSA_DARK_TINTED_RED);

        // --------------------------------------------------------------
        //   Build static visuals via new refactored helper
        // --------------------------------------------------------------
        const visuals = buildMHAVisuals(this.parentGroup, {
            branchX: this.branchX,
            mhsaBaseY: this.mhsaBaseY,
            matrixRestingOpacity: 1.0, // retains original behaviour
        });

        this.mhaVisualizations           = visuals.mhaVisualizations;
        this.headsCentersX               = visuals.headsCentersX;
        this.headCoords                  = visuals.headCoords;
        this.outputProjectionMatrix      = visuals.outputProjectionMatrix;
        this.outputProjMatrixCenterY     = visuals.outputProjMatrixCenterY;
        this.outputProjMatrixHeight      = visuals.outputProjMatrixHeight;
        this.outputProjMatrixDefaultColor = visuals.outputProjMatrixDefaultColor;
        this.outputProjMatrixActiveColor  = visuals.outputProjMatrixActiveColor;
        this.finalCombinedY              = visuals.finalCombinedY;
        this.finalOriginalY              = visuals.finalOriginalY;

        // Additional arrays required by later stages
        this.outputProjMatrixAnimationPhase = 'waiting';
        this.outputProjMatrixTrails         = [];
        this.outputProjMatrixVectors        = [];

        // Mode control (e.g., 'temp', 'perm', etc.)
        this.mode = mode;

        // --------------------------------------------------------------
        //  Self-attention toggle (defaults to global static value)
        // --------------------------------------------------------------
        this.enableSelfAttentionAnimation =
            opts.enableSelfAttention ?? MHSAAnimation.ENABLE_SELF_ATTENTION;

        // Self-attention helper (placeholder)
        this.selfAttentionAnimator = new SelfAttentionAnimator(this);

        // Temp-mode bookkeeping
        this._tempModeCompleted = false;
        this._tempAllOutputVectors = []; // K,Q,V combined
        this._tempKOutputVectors   = []; // Only central K vectors

        // --------------------------------------------------------------
        //   Vector router: handles all positioning before pass-through
        // --------------------------------------------------------------
        this.vectorRouter = new VectorRouter(this.parentGroup, this.headsCentersX, this.headCoords, this.headStopY, this.mhaVisualizations);
        this.vectorRouter.onReady(() => {
            this.mhaPassThroughPhase = 'ready_for_parallel_pass_through';
            console.log("MHSAAnimation: All MHSA vectors are in position. Ready for PARALLEL pass-through.");
            this.passThroughAnimator = new PassThroughAnimator(this);
            this.passThroughAnimator.start(this.currentLanes);
        });

        // ----------------------------------------------
        //  Stage pipeline scaffolding (legacy)
        // ----------------------------------------------
        // These two helper calls are replaced by the new builder.
        // Keeping methods defined below for now so update() logic referring to
        // them compiles, but we skip execution to avoid duplicate visuals.
        // this._setupMHSAVisualizations();
        // this._setupOutputProjectionMatrix();
    }

    // ------------------------------------------------------------------
    //  Placeholder self-attention phase – waits 3 s before continuing
    // ------------------------------------------------------------------
    _runSelfAttentionPhase(onDone) {
        if (this.selfAttentionAnimator) {
            this.selfAttentionAnimator.start(onDone);
        } else if (onDone) {
            onDone();
        }
    }

    _setupMHSAVisualizations() {
        const darkGrayColor = new THREE.Color(0x404040);
        const matrixOpacity = this.matrixRestingOpacity;
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
            {
                const lbl = 'Query Weight Matrix';
                queryMatrix.group.userData.label = lbl;
                if (queryMatrix.mesh) queryMatrix.mesh.userData.label = lbl;
                if (queryMatrix.frontCapMesh) queryMatrix.frontCapMesh.userData.label = lbl;
                if (queryMatrix.backCapMesh)  queryMatrix.backCapMesh.userData.label  = lbl;
            }
            queryMatrix.group.children.forEach(child => {
                if (child.material) {
                    child.material.transparent = matrixOpacity < 1.0;
                    child.material.opacity = matrixOpacity;
                }
            });
            this.parentGroup.add(queryMatrix.group);
            this.mhaVisualizations.push(queryMatrix);

            const keyMatrix = new WeightMatrixVisualization(
                null, new THREE.Vector3(x_k, matrixCenterY, 0),
                MHA_MATRIX_PARAMS.width, MHA_MATRIX_PARAMS.height, MHA_MATRIX_PARAMS.depth,
                MHA_MATRIX_PARAMS.topWidthFactor, MHA_MATRIX_PARAMS.cornerRadius, MHA_MATRIX_PARAMS.numberOfSlits,
                MHA_MATRIX_PARAMS.slitWidth, MHA_MATRIX_PARAMS.slitDepthFactor,
                MHA_MATRIX_PARAMS.slitBottomWidthFactor, MHA_MATRIX_PARAMS.slitTopWidthFactor
            );
            keyMatrix.setColor(darkGrayColor);
            {
                const lbl = 'Key Weight Matrix';
                keyMatrix.group.userData.label = lbl;
                if (keyMatrix.mesh) keyMatrix.mesh.userData.label = lbl;
                if (keyMatrix.frontCapMesh) keyMatrix.frontCapMesh.userData.label = lbl;
                if (keyMatrix.backCapMesh)  keyMatrix.backCapMesh.userData.label  = lbl;
            }
            keyMatrix.group.children.forEach(child => {
                if (child.material) {
                    child.material.transparent = matrixOpacity < 1.0;
                    child.material.opacity = matrixOpacity;
                }
            });
            this.parentGroup.add(keyMatrix.group);
            this.mhaVisualizations.push(keyMatrix);

            const valueMatrix = new WeightMatrixVisualization(
                null, new THREE.Vector3(x_v, matrixCenterY, 0),
                MHA_MATRIX_PARAMS.width, MHA_MATRIX_PARAMS.height, MHA_MATRIX_PARAMS.depth,
                MHA_MATRIX_PARAMS.topWidthFactor, MHA_MATRIX_PARAMS.cornerRadius, MHA_MATRIX_PARAMS.numberOfSlits,
                MHA_MATRIX_PARAMS.slitWidth, MHA_MATRIX_PARAMS.slitDepthFactor,
                MHA_MATRIX_PARAMS.slitBottomWidthFactor, MHA_MATRIX_PARAMS.slitTopWidthFactor
            );
            valueMatrix.setColor(darkGrayColor);
            {
                const lbl = 'Value Weight Matrix';
                valueMatrix.group.userData.label = lbl;
                if (valueMatrix.mesh) valueMatrix.mesh.userData.label = lbl;
                if (valueMatrix.frontCapMesh) valueMatrix.frontCapMesh.userData.label = lbl;
                if (valueMatrix.backCapMesh)  valueMatrix.backCapMesh.userData.label  = lbl;
            }
            valueMatrix.group.children.forEach(child => {
                if (child.material) {
                    child.material.transparent = matrixOpacity < 1.0;
                    child.material.opacity = matrixOpacity;
                }
            });
            this.parentGroup.add(valueMatrix.group);
            this.mhaVisualizations.push(valueMatrix);

            this.headsCentersX.push(x_k);
            this.headCoords.push({ q: x_q, k: x_k, v: x_v });
        }
    }

    _setupOutputProjectionMatrix() {
        // Positioned above the merged row, aligned with the first head's K matrix X-coordinate
        const firstHeadKMatrixX = this.headCoords.length > 0 ? this.headCoords[0].k : this.branchX; // Fallback to branchX if no heads

        // Y position calculation:
        // Base Y of K vectors after passing through heads and initial rise:
        const postPassThroughBaseY = this.mhaPassThroughTargetY + this.mhaResultRiseOffsetY;
        // Y of the decorative vectors that form the merged row:
        const decorativeVectorsY = postPassThroughBaseY + 60; // 60 is the verticalOffset from _applyTempModeBehaviour
        // Center Y for the new output projection matrix:
        const matrixHeight = MHA_MATRIX_PARAMS.height * MHA_OUTPUT_PROJECTION_MATRIX_PARAMS.heightFactor;
        const outputProjMatrixCenterY = decorativeVectorsY + MHA_OUTPUT_PROJECTION_MATRIX_Y_OFFSET_ABOVE_ROW + matrixHeight / 2;

        // Use same depth as other matrices for consistency
        const inputDepth = MHA_MATRIX_PARAMS.depth;

        this.outputProjectionMatrix = new WeightMatrixVisualization(
            null, // No specific data array needed for this visualization
            new THREE.Vector3(firstHeadKMatrixX, outputProjMatrixCenterY, 0),
            MHA_OUTPUT_PROJECTION_MATRIX_PARAMS.width,
            matrixHeight,
            inputDepth,
            MHA_OUTPUT_PROJECTION_MATRIX_PARAMS.topWidthFactor,
            MHA_OUTPUT_PROJECTION_MATRIX_PARAMS.cornerRadius,
            NUM_VECTOR_LANES,
            MHA_OUTPUT_PROJECTION_MATRIX_PARAMS.slitWidth,
            MHA_OUTPUT_PROJECTION_MATRIX_PARAMS.slitDepthFactor,
            MHA_OUTPUT_PROJECTION_MATRIX_PARAMS.slitBottomWidthFactor,
            MHA_OUTPUT_PROJECTION_MATRIX_PARAMS.slitTopWidthFactor
        );

        // Initialise pitch-black; will brighten once vectors pass through
        const initDarkColor = new THREE.Color(INACTIVE_COMPONENT_COLOR);
        this.outputProjectionMatrix.setColor(initDarkColor);
        {
            const lbl = 'Output Projection Matrix';
            this.outputProjectionMatrix.group.userData.label = lbl;
            if (this.outputProjectionMatrix.mesh) this.outputProjectionMatrix.mesh.userData.label = lbl;
            if (this.outputProjectionMatrix.frontCapMesh) this.outputProjectionMatrix.frontCapMesh.userData.label = lbl;
            if (this.outputProjectionMatrix.backCapMesh)  this.outputProjectionMatrix.backCapMesh.userData.label  = lbl;
        }
        this.outputProjectionMatrix.group.children.forEach(child => {
            if (child.material) {
                // Ensure the output projection matrix starts fully opaque
                child.material.transparent = false;
                child.material.opacity = 1.0;
                child.material.emissive = initDarkColor;
                child.material.emissiveIntensity = 0.1; // Low initial emissive intensity
            }
        });
        this.parentGroup.add(this.outputProjectionMatrix.group);
        
        // Store the matrix's Y position for later animations
        this.outputProjMatrixCenterY = outputProjMatrixCenterY;
        this.outputProjMatrixBasePosition = new THREE.Vector3(firstHeadKMatrixX, outputProjMatrixCenterY, 0);
        this.outputProjMatrixHeight = matrixHeight;
        
        // Store default and target colors for animation
        this.outputProjMatrixDefaultColor = initDarkColor;
        this.outputProjMatrixActiveColor = new THREE.Color(MHA_OUTPUT_PROJECTION_MATRIX_COLOR);
        
        // Animation state
        this.outputProjMatrixAnimationPhase = 'waiting'; // 'waiting', 'vectors_entering', 'vectors_inside', 'completed'
        this.outputProjMatrixTrails = [];
        this.outputProjMatrixVectors = [];
        
        // Log the matrix dimensions to confirm they match desired specifications
        console.log(`MHSAAnimation: Output Projection Matrix added - Width: ${MHA_OUTPUT_PROJECTION_MATRIX_PARAMS.width}, Height: ${matrixHeight}, Depth: ${inputDepth}`);

        // ------------------------------------------------------------------
        //  Pre-compute the Y coordinate where the processed vectors will finish
        //  ("finalCombinedY"), and where the original residual-stream vectors
        //  should end up ("finalOriginalY").  These values let us start moving
        //  the original vectors immediately rather than waiting until the
        //  Output-Projection phase kicks in.
        // ------------------------------------------------------------------

        // The processed vectors rise to:  matrixTopY + 60.
        const matrixTopY = outputProjMatrixCenterY + matrixHeight / 2;
        this.finalCombinedY = matrixTopY + 60; // 30 (rise to matrix) + 30 (extraRise)
        this.finalOriginalY = this.finalCombinedY - ORIGINAL_TO_PROCESSED_GAP;
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
        // Thin wrapper delegating to extracted helper for maintainability.
        return animateVectorMatrixPassThroughExternal(
            this,
            vector,
            matrix,
            brightMatrixColor,
            darkTintedMatrixColor,
            finalVectorHue,
            passThroughY,
            duration,
            riseOffset,
            riseDurationVal,
            outLength,
            animationCompletionCallback,
            vectorCategory
        );
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
                } else if (this.mode !== 'temp') {
                    // For perm mode, trigger final color transition here
                    this._transitionHeadColorsToFinal(1000); // 1 second duration
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
        // Keep a reference to the latest lanes array so that other internal
        // methods (triggered asynchronously) can access the original vectors.
        this.currentLanes = lanes;

        // ---------------- Vector routing (refactored) ----------------
        if (this.vectorRouter) {
            this.vectorRouter.update(deltaTime, timeNow, lanes);
        }

        // ---------------- End VectorRouter section -------------------
        /* Legacy inline routing logic has been moved to VectorRouter.js
           and will be removed once the migration is complete. */

        // ------------------------------------------------------------------
        //  CONTINUOUSLY MOVE ORIGINAL RESIDUAL-STREAM VECTORS UPWARDS
        // ------------------------------------------------------------------
        if (this.finalOriginalY !== undefined) {
            const riseStep = this.postSplitRiseSpeed * SPEED_MULT * deltaTime;
            lanes.forEach(lane => {
                if (!lane || !lane.originalVec || !lane.originalVec.group) return;

                const curY = lane.originalVec.group.position.y;
                let targetY = this.finalOriginalY;
                let shouldMove = true;

                // Enforce suspension during addition animation
                if (lane.stopRise) {
                    if (lane.stopRiseTarget) {
                        // Allow drifting up to just below the target vector
                        targetY = Math.min(targetY, lane.stopRiseTarget.position.y - ORIGINAL_TO_PROCESSED_GAP);
                    } else {
                        // If no target, freeze completely
                        shouldMove = false;
                    }
                }
                
                // Move the vector if not frozen
                if (shouldMove && curY < targetY) {
                    lane.originalVec.group.position.y = Math.min(curY + riseStep, targetY);
                }

                // Update the trail unless it's explicitly suspended
                const trailSuspended = lane.stopRise || (typeof lane.skipTrailResumeY === 'number' && curY <= lane.skipTrailResumeY);

                if (!trailSuspended && lane.origTrail) {
                    updateTrail(lane.origTrail, lane.originalVec.group.position);
                    // Clean up the suspension flag once we've passed the resume threshold
                    if (typeof lane.skipTrailResumeY === 'number' && curY > lane.skipTrailResumeY) {
                        delete lane.skipTrailResumeY;
                    }
                }
            });
        }

        // Update merge trails
        if (this._mergeLaneTrails) {
            // Lane trails are updated during tweens; no continuous endpoint updates needed.
        }

        // ------------------------------------------------------------------
        //  Ensure addition trail follows centre prism even before tween starts
        // ------------------------------------------------------------------
        lanes.forEach(lane => {
            if (!lane || !lane.originalVec) return;

            const trailObj = lane.additionTrail;
            if (!trailObj) return; // no active addition trail

            // Only follow while the addition animation is active (stopRise flag present)
            if (!lane.stopRise) {
                // Once the addition is complete, we may need to clean up the
                // reference on the lane to prevent this logic from running again.
                if (lane.additionTrail) delete lane.additionTrail;
                return;
            }

            const centreIdx = Math.floor(VECTOR_LENGTH_PRISM / 2);
            const instMat = new THREE.Matrix4();

            lane.originalVec.mesh.getMatrixAt(centreIdx, instMat);
            const wPos = new THREE.Vector3().setFromMatrixPosition(instMat);
            wPos.applyMatrix4(lane.originalVec.group.matrixWorld);

            // Do not continue the trail once the instance has been "hidden"
            // far away (HIDE_INSTANCE_Y_OFFSET ≈ 10000). Assuming the scene's
            // meaningful Y range is < 2 000 units.
            if (Math.abs(wPos.y) < 2000) {
                // Convert world-space centre-prism position to the local
                // coordinate space of the layer's root group so the trail
                // points remain consistent across stacked layers.
                const localPos = this.parentGroup.worldToLocal(wPos.clone());
                updateTrail(trailObj, localPos);
            }
        });
    }

    dispose() {
        // Standard THREE.js objects added to scene are usually handled by scene traversal on global cleanup.
    }

    _applyTempModeBehaviour() {
        const grayColor = new THREE.Color(0x606060);
        // Visible prism window for gray-out and gradient calculations
        const visiblePrismCountTemp = Math.min(VECTOR_LENGTH_PRISM, Math.ceil(this.outputVectorLength / PRISM_DIMENSIONS_PER_UNIT));
        const startVisibleIdx = Math.max(0, Math.floor((VECTOR_LENGTH_PRISM - visiblePrismCountTemp) / 2));
        const endVisibleIdx = startVisibleIdx + visiblePrismCountTemp - 1;

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

            // Build raw 768-dim data with 30 random switch points for varied gradient
            const rawData = [];
            const desiredSwitches = Math.min(30, VECTOR_LENGTH_PRISM);
            const switchPoints = new Set();
            while (switchPoints.size < desiredSwitches) {
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
                rawData.slice(startVisibleIdx, startVisibleIdx + visiblePrismCountTemp),
                this.outputVectorLength,
                { numKeyColors: this.outputVectorLength },
                { setHiddenToBlack: true }
            );

            // Colour the visible prism(s) with a smooth two-colour gradient so
            // each 64-D decorative vector has its own distinctive colour.
            const startColor = new THREE.Color().setHSL(Math.random(), 0.9, 0.6);
            const endColor   = new THREE.Color().setHSL(Math.random(), 0.9, 0.6);
            const visibleCount = visiblePrismCountTemp;
            for (let vi = 0; vi < visibleCount; vi++) {
                const idx = startVisibleIdx + vi;
                const t   = visibleCount > 1 ? vi / (visibleCount - 1) : 0;
                const col = startColor.clone().lerp(endColor, t);
                decoVec.setInstanceAppearance(idx, 0, col);
            }

            this.parentGroup.add(decoVec.group);

            // Keep reference for merge phase
            this._tempDecorativeVecs.push({ vec: decoVec, laneZ: kVec.group.position.z });

            // Create a trail line connecting the grayed-out vector to its colored vector above
            const connectionTrail = createTrailLine(this.parentGroup);
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
                    .delay(800)
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
                    .onUpdate(function(o){
                        mat.opacity = o.op;
                        mat.needsUpdate = true;
                    })
                    .onComplete(() => {
                        // Once the gray vector has fully faded, remove it from the scene
                        if (vec && vec.group) {
                            // Hide, detach, and dispose to reclaim resources
                            vec.group.visible = false;
                            this.parentGroup.remove(vec.group);
                            if (typeof vec.dispose === 'function') {
                                vec.dispose();
                            }
                        }
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

        // Build map laneZ -> array of decorative vectors
        const laneVectors = new Map();
        this._tempDecorativeVecs.forEach(obj => {
            const laneZ = obj.laneZ;
            if (!laneVectors.has(laneZ)) laneVectors.set(laneZ, []);
            laneVectors.get(laneZ).push(obj.vec);
        });

        // ------------------------------------------------------------------
        //  NEW: static reference trail per lane (from last → first head set)
        // ------------------------------------------------------------------
        const firstHeadCenterX = this.headsCentersX.length ? this.headsCentersX[0] : 0;
        const lastHeadCenterX  = this.headsCentersX.length ? this.headsCentersX[this.headsCentersX.length - 1] : 0;

        const targetX = firstHeadCenterX; // Existing merge target for centralised row
        let maxDurationMs = 0;

        laneVectors.forEach((vecList, laneZ) => {
            // Ensure vecList ordered so we can grab a representative Y coordinate
            vecList.sort((a, b) => a.group.position.x - b.group.position.x);
            const yPos = vecList.length ? vecList[0].group.position.y : 0;

            // Create a simple two-point trail line across the lane
            const laneTrail = createTrailLine(this.parentGroup);
            
            laneTrail.line.material.needsUpdate = true;
            updateTrail(laneTrail, new THREE.Vector3(lastHeadCenterX, yPos, laneZ));
            updateTrail(laneTrail, new THREE.Vector3(firstHeadCenterX, yPos, laneZ));
            // Intentionally DO NOT update this trail afterwards – it is static.

            // --------------------------------------------------------------
            //  Launch horizontal merge tweens for the decorative vectors
            // --------------------------------------------------------------
            vecList.forEach((vec, idx) => {
                const destX = targetX + (idx - (NUM_HEAD_SETS_LAYER - 1) / 2) * ROW_SEGMENT_SPACING;
                const distance = Math.abs(vec.group.position.x - destX);
                const durationMs = (distance / (ROW_MERGE_HORIZ_SPEED * SPEED_MULT)) * 1000;
                if (durationMs > maxDurationMs) maxDurationMs = durationMs;

                if (typeof TWEEN !== 'undefined') {
                    new TWEEN.Tween(vec.group.position)
                        .to({ x: destX }, durationMs)
                        .easing(TWEEN.Easing.Quadratic.InOut)
                        .start();
                } else {
                    vec.group.position.x = destX;
                }
            });
        });

        // After all merge tweens are initiated, schedule the next phases.
        if (typeof TWEEN !== 'undefined') {
            setTimeout(() => {
                this._transitionHeadColorsToFinal(1000);
                setTimeout(() => {
                    this._startVectorsThroughOutputProjection(laneVectors);
                }, 1000);
            }, maxDurationMs + 200);
        } else {
            this._transitionHeadColorsToFinal(0);
        }
    }
    
    _startVectorsThroughOutputProjection(laneVectors) {
        // Combine decorative vectors in each lane into a single vector, then animate those combined vectors
        this.outputProjMatrixAnimationPhase = 'vectors_entering';

        const combinedVectors = [];
        const combinedTrails = [];

        // Central X coordinate for combined vector (align with first head center)
        const centerX = this.headsCentersX.length ? this.headsCentersX[0] : 0;

        // Determine central prism range for visible region.  Each prism now
        // represents 64 real dimensions, so calculate required prism count
        // and clamp to avoid negative indices.
        const visiblePrismCount = Math.min(VECTOR_LENGTH_PRISM, Math.ceil(this.outputVectorLength / PRISM_DIMENSIONS_PER_UNIT));
        const startVisibleIdx = Math.max(0, Math.floor((VECTOR_LENGTH_PRISM - visiblePrismCount) / 2));
        const endVisibleIdx = startVisibleIdx + visiblePrismCount - 1;

        laneVectors.forEach((vecList, laneZ) => {
            if (!vecList || vecList.length === 0) return;

            // Ensure vecList is sorted by X position
            vecList.sort((a, b) => a.group.position.x - b.group.position.x);

            // Build combined raw data by concatenating the 64-dim slices of each decorative vector (preserves lane-specific data)
            const combinedRaw = [];
            vecList.forEach(v => {
                const slice = v.rawData.slice(startVisibleIdx, startVisibleIdx + visiblePrismCount);
                combinedRaw.push(...slice);
            });

            // Ensure the final length is exactly VECTOR_LENGTH_PRISM (768)
            if (combinedRaw.length < VECTOR_LENGTH_PRISM) {
                while (combinedRaw.length < VECTOR_LENGTH_PRISM) combinedRaw.push(0);
            } else if (combinedRaw.length > VECTOR_LENGTH_PRISM) {
                combinedRaw.length = VECTOR_LENGTH_PRISM;
            }

            const spawnPos = new THREE.Vector3(centerX, vecList[0].group.position.y, laneZ);
            const combinedVec = new VectorVisualizationInstancedPrism(combinedRaw, spawnPos, 3);

            // ------------------------------------------------------------------
            //  Re-colour combined vector with smooth gradient across its 12 prisms
            // ------------------------------------------------------------------
            // Copy colours from each decorative vector's visible prism so the
            // combined vector visually matches its inputs (avoids a colour
            // shift before the Output-Projection matrix).
            const tmpColor = new THREE.Color();
            vecList.forEach((srcVec, i) => {
                const srcPrismIdx  = startVisibleIdx;      // only one visible prism
                const destPrismIdx = startVisibleIdx + i;  // prism position in combined vector
                if (srcVec.mesh && srcVec.mesh.getColorAt) {
                    srcVec.mesh.getColorAt(srcPrismIdx, tmpColor);
                    combinedVec.mesh.setColorAt(destPrismIdx, tmpColor);
                }
            });
            if (combinedVec.mesh.instanceColor) combinedVec.mesh.instanceColor.needsUpdate = true;

            this.parentGroup.add(combinedVec.group);
            combinedVectors.push({ vec: combinedVec, laneZ });

            // Hide original decorative vectors
            vecList.forEach(v => { v.group.visible = false; });

            // Create a dedicated trail for the combined vector
            const trail = createTrailLine(this.parentGroup);
            // Use the base trail opacity so the path through the output-projection
            // matrix is clearly visible to the viewer.
            
            trail.line.material.needsUpdate = true;
            // Seed trail with current position
            updateTrail(trail, combinedVec.group.position);
            combinedTrails.push(trail);
        });

        if (combinedVectors.length === 0) {
            console.warn("No combined vectors created for output projection animation");
            return;
        }

        // Store for later reference
        this.outputProjMatrixVectors = combinedVectors.map(obj => obj.vec);
        this.outputProjMatrixTrails = combinedTrails;

        // Matrix positions
        const matrixBottomY = this.outputProjMatrixCenterY - this.outputProjMatrixHeight / 2;
        const matrixTopY = this.outputProjMatrixCenterY + this.outputProjMatrixHeight / 2;
        const targetYAboveMatrix = matrixTopY + 30;

        // Durations
        const duration1 = 1000;
        const duration2 = 1000;
        const duration3 = 500;

        if (typeof TWEEN === 'undefined') {
            console.warn("TWEEN not available for output projection matrix animation");
            return;
        }

        combinedVectors.forEach((vecObj, idx) => {
            const vec = vecObj.vec;
            const laneZ = vecObj.laneZ;

            new TWEEN.Tween(vec.group.position)
                .to({ y: matrixBottomY }, duration1)
                .easing(TWEEN.Easing.Quadratic.InOut)
                .onUpdate(() => {
                    if (this.outputProjMatrixTrails[idx]) updateTrail(this.outputProjMatrixTrails[idx], vec.group.position);
                })
                .onComplete(() => {
                    if (idx === 0) {
                        this._animateOutputMatrixBrightening(duration2);
                    }
                    new TWEEN.Tween(vec.group.position)
                        .to({ y: matrixTopY }, duration2)
                        .onStart(() => {
                            // Apply transformation INSIDE the projection matrix
                            const newRaw = this._generateRawDataWithSwitchPoints(30);
                            vec.applyProcessedVisuals(
                                newRaw.slice(),
                                NUM_HEAD_SETS_LAYER * this.outputVectorLength, // 12 * 64  = 768 visible output units
                                { numKeyColors: 30, generationOptions: null },
                                { setHiddenToBlack: false }
                            );

                            // Regenerate random key colors (similar to initial vectors)
                            if (typeof vec._generateKeyColors === 'function' && typeof vec._updateInstanceColors === 'function') {
                                vec._generateKeyColors();
                                vec._updateInstanceColors();
                            }
                        })
                        .onUpdate(() => {
                            if (this.outputProjMatrixTrails[idx]) updateTrail(this.outputProjMatrixTrails[idx], vec.group.position);
                        })
                        .onComplete(() => {
                            const extraRise = 30; // additional upward distance
                            const finalCombinedY = targetYAboveMatrix + extraRise;

                            // Final rise after transformation done
                            new TWEEN.Tween(vec.group.position)
                                .to({ y: finalCombinedY }, duration3)
                                .easing(TWEEN.Easing.Quadratic.InOut)
                                .onUpdate(() => {
                                    if (this.outputProjMatrixTrails[idx]) updateTrail(this.outputProjMatrixTrails[idx], vec.group.position);
                                })
                                .onComplete(() => {
                                    // Horizontal move back to residual stream centre (x = 0),
                                    // then perform the addition with the lane's original vector
                                    const horizDistance = Math.abs(vec.group.position.x);
                                    const horizDur = (horizDistance / (ANIM_HORIZ_SPEED * SPEED_MULT)) * 1000;

                                    new TWEEN.Tween(vec.group.position)
                                        .to({ x: 0 }, horizDur)
                                        .easing(TWEEN.Easing.Quadratic.InOut)
                                        .onUpdate(() => {
                                            if (this.outputProjMatrixTrails[idx]) updateTrail(this.outputProjMatrixTrails[idx], vec.group.position);
                                        })
                                        .onComplete(() => {
                                            if (this.currentLanes) {
                                                const matchingLane = this.currentLanes.find(l => Math.abs(l.zPos - laneZ) < 0.1);
                                                if (matchingLane && matchingLane.originalVec) {
                                                    this._startAdditionAnimation(matchingLane.originalVec, vec, matchingLane);
                                                }
                                            }
                                        })
                                        .start();
                                })
                                .start();
                        })
                        .start();
                })
                .start();
        });

        console.log("Starting animation of combined lane vectors through output projection matrix");
    }
    
    _animateOutputMatrixBrightening(duration) {
        if (typeof TWEEN === 'undefined') return;
        
        this.outputProjMatrixAnimationPhase = 'vectors_inside';
        
        // Animation parameters
        const startColor = this.outputProjMatrixDefaultColor.clone();
        const brightColor = this.outputProjMatrixActiveColor.clone();
        const startEmissiveIntensity = 0.1;
        const peakEmissiveIntensity = 0.8;
        const endEmissiveIntensity = 0.3;
        
        // First brighten the matrix
        const state = { 
            r: startColor.r, 
            g: startColor.g, 
            b: startColor.b,
            emissiveIntensity: startEmissiveIntensity
        };
        
        new TWEEN.Tween(state)
            .to({ 
                r: brightColor.r, 
                g: brightColor.g, 
                b: brightColor.b,
                emissiveIntensity: peakEmissiveIntensity
            }, duration * 0.6) // 60% of the total duration
            .easing(TWEEN.Easing.Quadratic.InOut)
            .onUpdate(() => {
                const currentColor = new THREE.Color(state.r, state.g, state.b);
                this.outputProjectionMatrix.setColor(currentColor);
                this.outputProjectionMatrix.setEmissive(currentColor, state.emissiveIntensity);
            })
            .onComplete(() => {
                // Then dim slightly to the final state
                new TWEEN.Tween(state)
                    .to({ emissiveIntensity: endEmissiveIntensity }, duration * 0.4) // 40% of the total duration
                    .easing(TWEEN.Easing.Quadratic.InOut)
                    .onUpdate(() => {
                        this.outputProjectionMatrix.setEmissive(brightColor, state.emissiveIntensity);
                    })
                    .onComplete(() => {
                        this.outputProjMatrixAnimationPhase = 'completed';
                        // Ensure final opacity is fully opaque
                        this.outputProjectionMatrix.setMaterialProperties({ opacity: 1.0, transparent: false });
                    })
                    .start();
            })
            .start();
    }

    _transitionHeadColorsToFinal(duration) {
        if (typeof TWEEN === 'undefined') {
            console.warn("TWEEN not available for final head color transition.");
            // Set colors directly if TWEEN is not available
            for (let i = 0; i < NUM_HEAD_SETS_LAYER; i++) {
                const qMatrix = this.mhaVisualizations[i * 3];
                const kMatrix = this.mhaVisualizations[i * 3 + 1];
                const vMatrix = this.mhaVisualizations[i * 3 + 2];

                if (qMatrix) qMatrix.setColor(new THREE.Color(MHA_FINAL_Q_COLOR));
                if (kMatrix) kMatrix.setColor(new THREE.Color(MHA_FINAL_K_COLOR));
                if (vMatrix) vMatrix.setColor(new THREE.Color(MHA_FINAL_V_COLOR));
            }
            return;
        }

        for (let i = 0; i < NUM_HEAD_SETS_LAYER; i++) {
            const qMatrix = this.mhaVisualizations[i * 3];
            const kMatrix = this.mhaVisualizations[i * 3 + 1];
            const vMatrix = this.mhaVisualizations[i * 3 + 2];

            // Ensure matrices are fully opaque before colour tween begins.
            [qMatrix, kMatrix, vMatrix].forEach(m => {
                if (m) m.setMaterialProperties({ opacity: 1.0, transparent: false });
            });

            const finalQColor = new THREE.Color(MHA_FINAL_Q_COLOR);
            const finalKColor = new THREE.Color(MHA_FINAL_K_COLOR);
            const finalVColor = new THREE.Color(MHA_FINAL_V_COLOR);

            if (qMatrix && qMatrix.mesh && qMatrix.mesh.material) {
                const initialQColor = qMatrix.mesh.material.color.clone();
                new TWEEN.Tween(initialQColor)
                    .to(finalQColor, duration)
                    .easing(TWEEN.Easing.Quadratic.InOut)
                    .onUpdate(() => {
                        qMatrix.setColor(initialQColor);
                         qMatrix.setEmissive(initialQColor, 0.3); // Add some emissiveness
                    })
                    .start();
            }

            if (kMatrix && kMatrix.mesh && kMatrix.mesh.material) {
                const initialKColor = kMatrix.mesh.material.color.clone();
                new TWEEN.Tween(initialKColor)
                    .to(finalKColor, duration)
                    .easing(TWEEN.Easing.Quadratic.InOut)
                    .onUpdate(() => {
                        kMatrix.setColor(initialKColor);
                        kMatrix.setEmissive(initialKColor, 0.3); // Add some emissiveness
                    })
                    .start();
            }

            if (vMatrix && vMatrix.mesh && vMatrix.mesh.material) {
                const initialVColor = vMatrix.mesh.material.color.clone();
                new TWEEN.Tween(initialVColor)
                    .to(finalVColor, duration)
                    .easing(TWEEN.Easing.Quadratic.InOut)
                    .onUpdate(() => {
                        vMatrix.setColor(initialVColor);
                        vMatrix.setEmissive(initialVColor, 0.3); // Add some emissiveness
                    })
                    .start();
            }
        }
        console.log("MHSAAnimation: Initiated final head color transitions.");
    }

    // ----------------------------------------------------------------------
    // Helper: Generate raw data with switch points (similar to earlier)
    // ----------------------------------------------------------------------
    _generateRawDataWithSwitchPoints(numSwitchPoints = 30) {
        const raw = [];
        // Clamp switch point count to available prisms to avoid infinite loop.
        numSwitchPoints = Math.min(numSwitchPoints, VECTOR_LENGTH_PRISM);
        const switchPoints = new Set();
        while (switchPoints.size < numSwitchPoints) {
            const idx = Math.floor(Math.random() * VECTOR_LENGTH_PRISM);
            switchPoints.add(idx);
        }
        const sortedSwitches = Array.from(switchPoints).sort((a, b) => a - b);
        let curVal = Math.random() * 2 - 1;
        let nextSwitch = sortedSwitches.shift();
        for (let i = 0; i < VECTOR_LENGTH_PRISM; i++) {
            if (i === nextSwitch) {
                curVal = Math.random() * 2 - 1;
                nextSwitch = sortedSwitches.shift();
            }
            raw.push(curVal);
        }
        return raw;
    }

    // ----------------------------------------------------------------------
    // Helper: Addition animation between two InstancedPrism vectors
    // ----------------------------------------------------------------------
    _startAdditionAnimation(sourceVec, targetVec, lane) {
        startPrismAdditionAnimation(this.parentGroup, sourceVec, targetVec, lane);
        // Don't force position - let vectors maintain their natural flow
    }
} 