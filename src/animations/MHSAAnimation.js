import * as THREE from 'three';
import { WeightMatrixVisualization } from '../components/WeightMatrixVisualization.js';
import { VectorVisualizationInstancedPrism } from '../components/VectorVisualizationInstancedPrism.js';
import { StraightLineTrail, mergeTrailsIntoLineSegments } from '../utils/trailUtils.js';



import { mapValueToColor } from '../utils/colors.js';
import { MHSA_MATRIX_INITIAL_RESTING_COLOR, MHSA_BRIGHT_GREEN, MHSA_DARK_TINTED_GREEN, MHSA_BRIGHT_BLUE, MHSA_DARK_TINTED_BLUE, MHSA_BRIGHT_RED, MHSA_DARK_TINTED_RED, MHA_FINAL_Q_COLOR, MHA_FINAL_K_COLOR, MHA_FINAL_V_COLOR, MHA_OUTPUT_PROJECTION_MATRIX_Y_OFFSET_ABOVE_ROW, MHA_OUTPUT_PROJECTION_MATRIX_PARAMS, MHA_OUTPUT_PROJECTION_MATRIX_COLOR } from './LayerAnimationConstants.js';
import { INACTIVE_COMPONENT_COLOR, MHSA_DUPLICATE_VECTOR_RISE_SPEED, MHSA_PASS_THROUGH_TOTAL_DURATION_MS, MHSA_RESULT_RISE_OFFSET_Y, MHSA_HEAD_VECTOR_STOP_BELOW, MHA_RESULT_RISE_DURATION_BASE_MS, DECORATIVE_FADE_MS, DECORATIVE_FADE_DELAY_MS, MERGE_TO_ROW_DELAY_AFTER_FADE_MS, HEAD_COLOR_TRANSITION_MS, MERGE_POST_COLOR_TRANSITION_DELAY_MS, MERGE_EXTRA_BUFFER_MS, OUTPUT_PROJ_STAGE1_MS, OUTPUT_PROJ_STAGE2_MS, OUTPUT_PROJ_STAGE3_MS, GLOBAL_ANIM_SPEED_MULT, MHSA_PASS_THROUGH_BRIGHTEN_RATIO, MHSA_PASS_THROUGH_DIM_RATIO, MHSA_MATRIX_MAX_EMISSIVE_INTENSITY } from '../utils/constants.js';
import {
    // Constants needed for setup & animation
    MHA_MATRIX_PARAMS,
    NUM_VECTOR_LANES,
    NUM_HEAD_SETS_LAYER,
    HEAD_SET_GAP_LAYER,
    MHA_INTERNAL_MATRIX_SPACING,
    HEAD_VECTOR_STOP_BELOW,
    ANIM_HORIZ_SPEED,
    SIDE_COPY_DELAY_MS,
    SIDE_COPY_HORIZ_SPEED,
    ROW_MERGE_HORIZ_SPEED,
    ROW_SEGMENT_SPACING,
    VECTOR_LENGTH_PRISM,
    PRISM_BASE_WIDTH,
    PRISM_BASE_DEPTH,
    PRISM_MAX_HEIGHT,
    PRISM_HEIGHT_SCALE_FACTOR,
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

// Use live binding of GLOBAL_ANIM_SPEED_MULT at each use; do not cache

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
        // Track the merge-to-row phase so UI can switch equations as soon as
        // vectors start travelling back toward the output projection matrix.
        // Values: 'not_started' | 'merging' | 'merged'
        this.rowMergePhase = 'not_started';

        this.mhsa_matrix_center_y = this.mhsaBaseY + MHA_MATRIX_PARAMS.height / 2;
        this.headStopY            = this.mhsa_matrix_center_y - MHSA_HEAD_VECTOR_STOP_BELOW;
        this.mhaPassThroughTargetY = this.mhsa_matrix_center_y + MHA_MATRIX_PARAMS.height / 2 + 20;

        // Durations & dimensional constants
        this.outputVectorLength      = 64;
        this.mhaResultRiseOffsetY    = MHSA_RESULT_RISE_OFFSET_Y;
        // Note: durations are exposed as getters to reflect runtime speed changes

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
        this.outputProjMatrixVectors        = [];

        // Mode control (e.g., 'temp', 'perm', etc.)
        this.mode = mode;

        // Flag to indicate a global matrix pulse is active (throttles per-vector updates)
        this._mhaPulseActive = false;

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

        // Track per-head merged fixed vectors so we only merge once.
        this._mergedHeads = new Set();
        this._mergedGroupsByHead = new Map(); // headIdx -> { K: Group, V: Group }
        this._allFixedMerged = false; // single merged K and V meshes across all heads created

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

    // Dynamic durations that adapt to GLOBAL_ANIM_SPEED_MULT at access time
    get mhaResultRiseDuration() {
        return MHA_RESULT_RISE_DURATION_BASE_MS / GLOBAL_ANIM_SPEED_MULT;
    }
    get mhaPassThroughDuration() {
        return MHSA_PASS_THROUGH_TOTAL_DURATION_MS / GLOBAL_ANIM_SPEED_MULT;
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

        // Start one shared pulse per matrix instead of per-vector material updates
        try { this._startMatrixPulseDuringPassThrough(this.mhaPassThroughDuration); } catch (_) { /* optional */ }

        // Before launching pass-through tweens, merge parked K/Q/V copy trails
        // into a single static LineSegments to reduce draw calls and skip their
        // per-frame updates while vectors move through matrices.
        try { this._mergeCopyTrailsBeforePassThrough(); } catch (_) { /* optional */ }

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
                    this._transitionHeadColorsToFinal(HEAD_COLOR_TRANSITION_MS); // 1 second duration
                }

                // After pass-through completes for all vectors, try merging all fixed K/V across heads
                try { this._mergeAllFixedKVIfReady(); } catch (_) { /* optional */ }

                // Schedule disposal of merged K/V groups only AFTER the very last
                // blue (Q) vector completes its conveyor-belt traversal.
                try { this._waitAndDisposeMergedKVWhenBlueConveyorComplete(); } catch (_) { /* optional */ }
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

    // ------------------------------------------------------------------
    //  One shared pulse per Q/K/V matrix during pass-through (perf)
    // ------------------------------------------------------------------
    _startMatrixPulseDuringPassThrough(totalDurationMs) {
        if (typeof TWEEN === 'undefined') return;
        this._mhaPulseActive = true;

        const restingColor = this.matrixInitialRestingColor.clone();
        const restIntensity = this.matrixRestingEmissiveIntensity;

        const makePulse = (matrix, brightCol, darkTintedCol) => {
            if (!matrix || !matrix.mesh || !matrix.mesh.material) return null;
            const state = { p: 0 };
            return new TWEEN.Tween(state)
                .to({ p: 1 }, totalDurationMs)
                .easing(TWEEN.Easing.Quadratic.InOut)
                .onUpdate(() => {
                    const p = state.p;
                    let currentColor;
                    let currentEmissive;
                    if (p < MHSA_PASS_THROUGH_BRIGHTEN_RATIO) {
                        const t = THREE.MathUtils.smoothstep(p / MHSA_PASS_THROUGH_BRIGHTEN_RATIO, 0, 1);
                        currentColor = restingColor.clone().lerp(brightCol, t);
                        currentEmissive = THREE.MathUtils.lerp(restIntensity, MHSA_MATRIX_MAX_EMISSIVE_INTENSITY, t);
                    } else if (p < MHSA_PASS_THROUGH_BRIGHTEN_RATIO + MHSA_PASS_THROUGH_DIM_RATIO) {
                        const t = THREE.MathUtils.smoothstep(
                            (p - MHSA_PASS_THROUGH_BRIGHTEN_RATIO) / MHSA_PASS_THROUGH_DIM_RATIO,
                            0, 1
                        );
                        currentColor = brightCol.clone().lerp(darkTintedCol, t);
                        currentEmissive = THREE.MathUtils.lerp(MHSA_MATRIX_MAX_EMISSIVE_INTENSITY, restIntensity, t);
                    } else {
                        currentColor = darkTintedCol.clone();
                        currentEmissive = restIntensity;
                    }
                    matrix.setColor(currentColor);
                    matrix.setEmissive(currentColor, currentEmissive);
                })
                .onComplete(() => {
                    matrix.setColor(darkTintedCol);
                    matrix.setEmissive(darkTintedCol, restIntensity);
                })
                .start();
        };

        let pulsesStarted = 0;
        const totalMatrices = NUM_HEAD_SETS_LAYER * 3;
        for (let i = 0; i < NUM_HEAD_SETS_LAYER; i++) {
            const qMatrix = this.mhaVisualizations[i * 3];
            const kMatrix = this.mhaVisualizations[i * 3 + 1];
            const vMatrix = this.mhaVisualizations[i * 3 + 2];
            if (makePulse(qMatrix, this.brightBlue, this.darkTintedBlue)) pulsesStarted++;
            if (makePulse(kMatrix, this.brightGreen, this.darkTintedGreen)) pulsesStarted++;
            if (makePulse(vMatrix, this.brightRed, this.darkTintedRed)) pulsesStarted++;
        }

        // Safely clear the pulse flag after the pulses end
        setTimeout(() => { this._mhaPulseActive = false; }, totalDurationMs + 50);
        if (pulsesStarted !== totalMatrices) {
            // If some matrices missing, still clear after duration to avoid stuck flag
            setTimeout(() => { this._mhaPulseActive = false; }, totalDurationMs + 60);
        }
    }
    
    update(deltaTime, timeNow, lanes) {
        // Increment a lightweight frame counter to de-duplicate residual trail writes
        if (typeof this._frameCounter !== 'number') this._frameCounter = 0;
        this._frameCounter++;
        // Keep a reference to the latest lanes array so that other internal
        // methods (triggered asynchronously) can access the original vectors.
        this.currentLanes = lanes;

        // ---------------- Vector routing (refactored) ----------------
        if (this.vectorRouter) {
            this.vectorRouter.update(deltaTime, timeNow, lanes);
        }

        // ---- Update trails for combined vectors through Output Projection ----
        if (this.outputProjMatrixVectors && this.outputProjMatrixVectors.length) {
            this.outputProjMatrixVectors.forEach(v => {
                if (v && v.userData && v.userData.trail) {
                    v.userData.trail.update(v.group.position);
                }
            });
        }

        // ---------------- End VectorRouter section -------------------
        /* Legacy inline routing logic has been moved to VectorRouter.js
           and will be removed once the migration is complete. */

        // ------------------------------------------------------------------
        //  CONTINUOUSLY MOVE ORIGINAL RESIDUAL-STREAM VECTORS UPWARDS
        // ------------------------------------------------------------------
        if (this.finalOriginalY !== undefined) {
            const riseStep = this.postSplitRiseSpeed * GLOBAL_ANIM_SPEED_MULT * deltaTime;
            lanes.forEach(lane => {
                if (!lane || !lane.originalVec || !lane.originalVec.group) return;

                const curY = lane.originalVec.group.position.y;
                let targetY = this.finalOriginalY;
                // Hard clamp: once the pipeline signals the top embedding stop height,
                // never allow the residual vectors to rise past that entrance.
                if (typeof this.topEmbeddingStopY === 'number') {
                    targetY = Math.min(targetY, this.topEmbeddingStopY);
                }
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
                
                // Only allow upward movement toward the target/clamp; never force downward snaps
                if (shouldMove && curY < targetY) {
                    lane.originalVec.group.position.y = Math.min(curY + riseStep, targetY);
                }

                // Update trail after movement – respect world-space trails.
                // During residual addition we update the trail using the centre prism's
                // world position instead (handled just below), so skip the group-centre
                // update while lane.stopRise is active to avoid double-writing.
                if (!lane.stopRise) {
                    try {
                        // Prefer attached trail; fall back to lane.originalTrail if not attached yet
                        const attached = lane.originalVec && lane.originalVec.userData && lane.originalVec.userData.trail;
                        const residualTrail = attached || lane.originalTrail;
                        if (residualTrail && typeof residualTrail.update === 'function') {
                            const wp = new THREE.Vector3();
                            lane.originalVec.group.getWorldPosition(wp);
                            // Monotonic Y clamp: only extend the trail upwards
                            if (typeof lane.__residualMaxY !== 'number') lane.__residualMaxY = wp.y;
                            if (wp.y >= lane.__residualMaxY) {
                                if (lane.__lastResidualTrailFrame !== this._frameCounter) {
                                    residualTrail.update(wp);
                                    lane.__lastResidualTrailFrame = this._frameCounter;
                                }
                                lane.__residualMaxY = wp.y;
                            }
                        }
                    } catch (_) { /* defensive */ }
                }

            });
        }


        lanes.forEach(lane => {
            if (!lane || !lane.originalVec) return;

            // Only follow while the addition animation is active (stopRise flag present)
            if (!lane.stopRise) return;

            try {
                // Compute world position of the centre prism for the ORIGINAL (source) vector
                // so the residual trail extends continuously as prisms rise during addition.
                const centreIdx = Math.floor(VECTOR_LENGTH_PRISM / 2);
                const instMat = new THREE.Matrix4();
                lane.originalVec.mesh.getMatrixAt(centreIdx, instMat);
                const wPos = new THREE.Vector3().setFromMatrixPosition(instMat);
                wPos.applyMatrix4(lane.originalVec.group.matrixWorld);

                // Skip bogus updates when the centre prism is hidden far below the scene
                // during/addition (it is moved to HIDE_INSTANCE_Y_OFFSET to disappear).
                const hideThreshold = HIDE_INSTANCE_Y_OFFSET / 10; // e.g. -5000 for -50000 offset
                if (wPos.y < hideThreshold) return;

                // Update trail all the way to the top vector so there is no visible gap.
                // Prefer the dedicated world-space residual trail reference carried by the lane.
                const residualTrail = (lane.originalTrail)
                    || (lane.originalVec && lane.originalVec.userData && lane.originalVec.userData.trail)
                    || (lane.postAdditionVec && lane.postAdditionVec.userData && lane.postAdditionVec.userData.trail);
                if (residualTrail && typeof residualTrail.update === 'function') {
                    // Guard against accidental double-write in the same frame and enforce monotonic Y extension
                    // Allow immediate extension from the centre prism upward when addition begins.
                    if (typeof lane.__residualMaxY !== 'number') lane.__residualMaxY = wPos.y - 0.001;
                    if (wPos.y >= lane.__residualMaxY) {
                        if (lane.__lastResidualTrailFrame !== this._frameCounter) {
                            residualTrail.update(wPos);
                            lane.__lastResidualTrailFrame = this._frameCounter;
                        }
                        lane.__residualMaxY = wPos.y;
                    }
                }
            } catch (_) { /* defensive */ }
        });
    }

    // ------------------------------------------------------------------
    //  Merge faint trails drawn by the parked K/Q/V copies (pre pass-through)
    // ------------------------------------------------------------------
    _mergeCopyTrailsBeforePassThrough() {
        const lanes = this.currentLanes || [];
        if (!lanes.length) return;

        /** @type {import('../utils/trailUtils.js').StraightLineTrail[]} */
        const trailsToMerge = [];
        const vectorsWithTrails = [];

        lanes.forEach(lane => {
            // Upward K copies
            if (Array.isArray(lane.upwardCopies)) {
                lane.upwardCopies.forEach(v => {
                    const t = v && v.userData && v.userData.trail;
                    if (t) {
                        trailsToMerge.push(t);
                        vectorsWithTrails.push(v);
                    }
                });
            }
            // Side Q/V copies
            if (Array.isArray(lane.sideCopies)) {
                lane.sideCopies.forEach(obj => {
                    const v = obj && obj.vec;
                    const t = v && v.userData && v.userData.trail;
                    if (t) {
                        trailsToMerge.push(t);
                        vectorsWithTrails.push(v);
                    }
                });
            }
            // Travelling vector trail inside MHSA branch (not the residual stream)
            const tv = lane && lane.travellingVec;
            const tvTrail = tv && tv.userData && tv.userData.trail;
            if (tvTrail) {
                trailsToMerge.push(tvTrail);
                vectorsWithTrails.push(tv);
            }
        });

        if (!trailsToMerge.length) return;

        // Preserve appearance based on first trail
        let colorHex = undefined;
        let baseOpacity = undefined;
        const first = trailsToMerge[0];
        if (first) {
            if (first._material && first._material.color) colorHex = first._material.color.getHex();
            else if (typeof first._color !== 'undefined') colorHex = first._color;
            if (typeof first._opacity === 'number') baseOpacity = first._opacity;
        }

        // Build merged static segments and dispose originals
        mergeTrailsIntoLineSegments(trailsToMerge, this.parentGroup, colorHex, undefined, baseOpacity);

        // Remove trail refs from vectors to avoid updating disposed trails
        vectorsWithTrails.forEach(v => { if (v && v.userData) delete v.userData.trail; });
    }

    dispose() {
        // Standard THREE.js objects added to scene are usually handled by scene traversal on global cleanup.
    }

    // ------------------------------------------------------------------
    //  Merge fixed K (green) and V (red) vectors for a head into instanced
    //  batches to reduce draw calls once alignment is complete. Original
    //  VectorVisualizationInstancedPrism meshes are disposed, but their
    //  empty groups remain for positional queries used by the conveyor.
    // ------------------------------------------------------------------
    _mergeFixedVectorsForHead(headIdx) {
        if (this._mergedHeads.has(headIdx)) return; // already merged
        if (!Array.isArray(this.currentLanes) || this.currentLanes.length === 0) return;

        const greens = [];
        const reds = [];

        // Collect K and V vectors for this head across all lanes
        this.currentLanes.forEach((lane) => {
            if (!lane) return;
            const kVec = lane.upwardCopies && lane.upwardCopies[headIdx];
            if (kVec && kVec.mesh) greens.push(kVec);
            if (Array.isArray(lane.sideCopies)) {
                const vObj = lane.sideCopies.find(sc => sc && sc.headIndex === headIdx && sc.type === 'V');
                if (vObj && vObj.vec && vObj.vec.mesh) reds.push(vObj.vec);
            }
        });

        // Build merged instanced meshes (one for K, one for V) and add to scene
        const mergedGroups = { K: null, V: null };
        try {
            if (greens.length) mergedGroups.K = this._buildMergedPrismsFromVectors(greens, `MergedK_head${headIdx}`);
        } catch (_) { /* non-fatal */ }
        try {
            if (reds.length) mergedGroups.V = this._buildMergedPrismsFromVectors(reds, `MergedV_head${headIdx}`);
        } catch (_) { /* non-fatal */ }

        if (mergedGroups.K) this.parentGroup.add(mergedGroups.K);
        if (mergedGroups.V) this.parentGroup.add(mergedGroups.V);

        // For temp mode, simply hide original meshes to avoid double-drawing,
        // but keep them around for any logic that still references them.
        // For other modes, dispose meshes to reclaim resources.
        const retireOriginalMesh = (vec) => {
            try {
                if (!vec || !vec.mesh) return;
                if (this.mode === 'temp') {
                    vec.mesh.visible = false;
                } else {
                    if (vec.mesh.material) vec.mesh.material.dispose();
                    if (vec.mesh.geometry) vec.mesh.geometry.dispose();
                    if (vec.group) vec.group.remove(vec.mesh);
                    vec.mesh = null;
                }
            } catch (_) { /* ignore */ }
        };
        greens.forEach(retireOriginalMesh);
        reds.forEach(retireOriginalMesh);

        this._mergedHeads.add(headIdx);
        this._mergedGroupsByHead.set(headIdx, mergedGroups);
    }

    // Helper to create a single InstancedMesh representing all prisms from a list of vectors
    _buildMergedPrismsFromVectors(vecList, debugName = 'MergedKV') {
        if (!Array.isArray(vecList) || vecList.length === 0) return null;

        // Mirror VectorVisualizationInstancedPrism internal constants
        const PRISM_WIDTH_SCALE = 1.5;
        const PRISM_DEPTH_SCALE = 1.5;
        const uniformCalculatedHeight = Math.max(0.01, PRISM_MAX_HEIGHT * PRISM_HEIGHT_SCALE_FACTOR * 2.0);
        const baseWidth = PRISM_BASE_WIDTH;
        const baseDepth = PRISM_BASE_DEPTH;
        const hideY = HIDE_INSTANCE_Y_OFFSET;

        // Determine if this merged group represents V (red) vectors.
        // If so, we anchor them at the canonical above-matrix height rather than
        // whatever transient Y the fixed copies happen to have when we merge.
        const isRedCategory = /V(\b|_|head|$)/.test(debugName);
        const extraRise = (this.selfAttentionAnimator && this.selfAttentionAnimator.RED_EXTRA_RISE) || 75;
        // BASE_RISE_ADJUST from VectorMatrixPassThrough is -30; replicate here to match final resting height
        const canonicalRaisedBaseY = this.mhaPassThroughTargetY + this.mhaResultRiseOffsetY - 30 + extraRise;

        const totalInstances = vecList.length * VECTOR_LENGTH_PRISM;
        const baseGeo = new THREE.BoxGeometry(baseWidth, 1, baseDepth);
        const material = new THREE.MeshBasicMaterial({ color: 0xffffff });
        // Shader patch to support per-instance left→right gradient identical to VectorVisualizationInstancedPrism
        material.customProgramCacheKey = () => 'InstancedPrismGradientV1';
        material.onBeforeCompile = (shader) => {
            shader.uniforms.prismHalfWidth = { value: (baseWidth * PRISM_WIDTH_SCALE) / 2 };
            shader.vertexShader = shader.vertexShader.replace(
                '#include <common>',
                `#include <common>\nattribute vec3 colorStart;\nattribute vec3 colorEnd;\nvarying vec3 vColorStart;\nvarying vec3 vColorEnd;\nvarying float vGradientT;\nuniform float prismHalfWidth;`
            );
            shader.vertexShader = shader.vertexShader.replace(
                '#include <begin_vertex>',
                `#include <begin_vertex>\n    vColorStart = colorStart;\n    vColorEnd   = colorEnd;\n    vGradientT  = clamp( (position.x + prismHalfWidth) / (2.0 * prismHalfWidth), 0.0, 1.0 );`
            );
            shader.fragmentShader = shader.fragmentShader.replace(
                '#include <common>',
                `#include <common>\nvarying vec3 vColorStart;\nvarying vec3 vColorEnd;\nvarying float vGradientT;`
            );
            shader.fragmentShader = shader.fragmentShader.replace(
                'vec4 diffuseColor = vec4( diffuse, opacity );',
                `vec3 grad = mix( vColorStart, vColorEnd, vGradientT );\n    vec4 diffuseColor = vec4( grad, opacity );`
            );
        };

        const instanced = new THREE.InstancedMesh(baseGeo, material, totalInstances);
        instanced.instanceMatrix.setUsage(THREE.DynamicDrawUsage);

        // Gradient buffers (per instance)
        const colorStartArr = new Float32Array(totalInstances * 3);
        const colorEndArr   = new Float32Array(totalInstances * 3);
        const colorStartAttr = new THREE.InstancedBufferAttribute(colorStartArr, 3);
        const colorEndAttr   = new THREE.InstancedBufferAttribute(colorEndArr, 3);
        instanced.geometry.setAttribute('colorStart', colorStartAttr);
        instanced.geometry.setAttribute('colorEnd',   colorEndAttr);

        const tmpMatrix = new THREE.Matrix4();
        const tmpPos = new THREE.Vector3();
        const tmpQuat = new THREE.Quaternion();
        const tmpScale = new THREE.Vector3();
        const dummy = new THREE.Object3D();

        // Populate instances by copying per-prism transforms and colours from the original vectors
        let instIndex = 0;
        // Build mapping so raycasts (instanceId) can be decoded back to vector/prism/head/lane
        const vectorRefs = vecList.slice();
        const headIdxs = vecList.map(v => (v && v.userData && typeof v.userData.headIndex === 'number') ? v.userData.headIndex : null);
        const laneIdxs = vecList.map(v => {
            const z = v && v.group ? v.group.position.z : NaN;
            const lanes = this.currentLanes || [];
            let best = -1, bestDist = Infinity;
            for (let i = 0; i < lanes.length; i++) {
                const d = Math.abs(lanes[i].zPos - z);
                if (d < bestDist) { best = i; bestDist = d; }
            }
            return best;
        });
        for (let vIdx = 0; vIdx < vecList.length; vIdx++) {
            const vec = vecList[vIdx];
            if (!vec || !vec.group) { instIndex += VECTOR_LENGTH_PRISM; continue; }

            // Access original gradient attrs if available
            const srcCS = vec.mesh && vec.mesh.geometry && vec.mesh.geometry.getAttribute ? vec.mesh.geometry.getAttribute('colorStart') : null;
            const srcCE = vec.mesh && vec.mesh.geometry && vec.mesh.geometry.getAttribute ? vec.mesh.geometry.getAttribute('colorEnd')   : null;

            for (let i = 0; i < VECTOR_LENGTH_PRISM; i++, instIndex++) {
                // Determine if this prism is visible or hidden in the source vector by reading its matrix
                let hidden = false;
                if (vec.mesh && typeof vec.mesh.getMatrixAt === 'function') {
                    vec.mesh.getMatrixAt(i, tmpMatrix);
                    tmpMatrix.decompose(tmpPos, tmpQuat, tmpScale);
                    // Hidden prisms are scaled down and translated to a large negative Y
                    if (tmpPos.y < hideY * 0.5 || tmpScale.y < 0.01) hidden = true;
                }

                // Compute world position for this prism
                const baseX = (i - VECTOR_LENGTH_PRISM / 2) * (baseWidth * PRISM_WIDTH_SCALE);
                const worldX = vec.group.position.x + baseX;
                const baseYForCategory = isRedCategory ? canonicalRaisedBaseY : (vec.group && vec.group.position ? vec.group.position.y : 0);
                const worldY = baseYForCategory + (hidden ? hideY : uniformCalculatedHeight / 2);
                const worldZ = vec.group.position.z;

                dummy.position.set(worldX, worldY, worldZ);
                // Apply fixed uniform dimensions (mirrors original visual size)
                dummy.scale.set(PRISM_WIDTH_SCALE, hidden ? 0.001 : uniformCalculatedHeight, PRISM_DEPTH_SCALE);
                dummy.updateMatrix();
                instanced.setMatrixAt(instIndex, dummy.matrix);

                // Copy gradient colours from source if present; otherwise derive from vector helper
                if (srcCS && srcCE) {
                    const i3 = i * 3;
                    colorStartArr[instIndex * 3 + 0] = srcCS.array[i3 + 0] || 0.5;
                    colorStartArr[instIndex * 3 + 1] = srcCS.array[i3 + 1] || 0.5;
                    colorStartArr[instIndex * 3 + 2] = srcCS.array[i3 + 2] || 0.5;
                    colorEndArr[instIndex * 3 + 0] = srcCE.array[i3 + 0] || 0.5;
                    colorEndArr[instIndex * 3 + 1] = srcCE.array[i3 + 1] || 0.5;
                    colorEndArr[instIndex * 3 + 2] = srcCE.array[i3 + 2] || 0.5;
                } else if (typeof vec.getDefaultColorForIndex === 'function') {
                    const leftColor  = vec.getDefaultColorForIndex(Math.max(0, i - 1));
                    const rightColor = vec.getDefaultColorForIndex(Math.min(VECTOR_LENGTH_PRISM - 1, i + 1));
                    colorStartArr[instIndex * 3 + 0] = leftColor.r;  colorStartArr[instIndex * 3 + 1] = leftColor.g;  colorStartArr[instIndex * 3 + 2] = leftColor.b;
                    colorEndArr[instIndex * 3 + 0]   = rightColor.r; colorEndArr[instIndex * 3 + 1]   = rightColor.g; colorEndArr[instIndex * 3 + 2]   = rightColor.b;
                } else {
                    // Neutral grey fallback
                    colorStartArr[instIndex * 3 + 0] = 0.5; colorStartArr[instIndex * 3 + 1] = 0.5; colorStartArr[instIndex * 3 + 2] = 0.5;
                    colorEndArr[instIndex * 3 + 0]   = 0.5; colorEndArr[instIndex * 3 + 1]   = 0.5; colorEndArr[instIndex * 3 + 2]   = 0.5;
                }
            }
        }

        instanced.instanceMatrix.needsUpdate = true;
        colorStartAttr.needsUpdate = true;
        colorEndAttr.needsUpdate = true;

        const group = new THREE.Group();
        group.name = debugName;
        group.add(instanced);
        // Attach metadata for decoding instanceId on raycast
        const category = /V(\b|_|head|$)/.test(debugName) ? 'V' : 'K';
        // Assign descriptive label so hover shows full text for merged groups
        group.userData.label = (category === 'V') ? 'Merged Value Vectors (Red)' : 'Merged Key Vectors (Green)';
        const mergedKVMeta = {
            category,
            vectorPrismCount: VECTOR_LENGTH_PRISM,
            vectorRefs,
            headIdxs,
            laneIdxs
        };
        group.userData.mergedKVMeta = mergedKVMeta;
        instanced.userData.mergedKVMeta = mergedKVMeta;
        // Also set label on mesh for direct hits
        instanced.userData.label = (category === 'V') ? 'Merged Value Vectors (Red)' : 'Merged Key Vectors (Green)';
        return group;
    }

    /** Decode a THREE.Raycaster intersection against a merged K/V InstancedMesh. */
    decodeMergedKVIntersection(intersection) {
        if (!intersection || typeof intersection.instanceId !== 'number') return null;
        const mesh = intersection.object;
        if (!mesh || !mesh.isInstancedMesh) return null;
        const meta = (mesh.userData && mesh.userData.mergedKVMeta)
            || (mesh.parent && mesh.parent.userData && mesh.parent.userData.mergedKVMeta)
            || null;
        if (!meta) return null;
        const instanceId = intersection.instanceId;
        const vectorIndex = Math.floor(instanceId / meta.vectorPrismCount);
        const prismIndex = instanceId % meta.vectorPrismCount;
        const headIndex = Array.isArray(meta.headIdxs) ? meta.headIdxs[vectorIndex] : null;
        const laneIndex = Array.isArray(meta.laneIdxs) ? meta.laneIdxs[vectorIndex] : null;
        const vectorRef = Array.isArray(meta.vectorRefs) ? meta.vectorRefs[vectorIndex] : null;
        return {
            category: meta.category, // 'K' or 'V'
            vectorIndex,
            prismIndex,
            headIndex,
            laneIndex,
            vectorRef
        };
    }

    // Merge ALL fixed K and V vectors across heads into two meshes
    _mergeAllFixedKVIfReady() {
        if (this._allFixedMerged) return;
        if (!Array.isArray(this.currentLanes) || this.currentLanes.length === 0) return;
        if (this.mhaPassThroughPhase !== 'mha_pass_through_complete') return;
        // Ensure all heads are aligned before collapsing geometry
        for (let h = 0; h < NUM_HEAD_SETS_LAYER; h++) {
            if (!this.selfAttentionAnimator || this.selfAttentionAnimator.greensAligned[h] !== true) {
                return;
            }
        }

        const allGreens = [];
        const allReds = [];
        this.currentLanes.forEach(lane => {
            if (!lane) return;
            if (Array.isArray(lane.upwardCopies)) {
                for (let h = 0; h < NUM_HEAD_SETS_LAYER; h++) {
                    const kVec = lane.upwardCopies[h];
                    if (kVec && kVec.mesh) allGreens.push(kVec);
                }
            }
            if (Array.isArray(lane.sideCopies)) {
                lane.sideCopies.forEach(sc => {
                    if (sc && sc.type === 'V' && sc.vec && sc.vec.mesh) allReds.push(sc.vec);
                });
            }
        });
        if (!allGreens.length && !allReds.length) return;

        let mergedKGroup = null;
        let mergedVGroup = null;
        try { if (allGreens.length) mergedKGroup = this._buildMergedPrismsFromVectors(allGreens, 'MergedAllK'); } catch (_) {}
        try { if (allReds.length)   mergedVGroup = this._buildMergedPrismsFromVectors(allReds,   'MergedAllV'); } catch (_) {}
        if (mergedKGroup) this.parentGroup.add(mergedKGroup);
        if (mergedVGroup) this.parentGroup.add(mergedVGroup);

        const stripMeshKeepGroup = (vec) => {
            try {
                if (vec && vec.mesh) {
                    if (vec.mesh.material) vec.mesh.material.dispose();
                    if (vec.mesh.geometry) vec.mesh.geometry.dispose();
                    if (vec.group) vec.group.remove(vec.mesh);
                    vec.mesh = null;
                }
            } catch (_) {}
        };
        allGreens.forEach(stripMeshKeepGroup);
        allReds.forEach(stripMeshKeepGroup);
        this._allFixedMerged = true;
        console.log('MHSAAnimation: Merged all fixed K and V vectors into two instanced meshes.');
    }

    // Fade any existing merged K/V instanced meshes to a target opacity
    _fadeMergedKVOpacity(targetOpacity, durationMs, delayMs = 0) {
        const doFade = (group) => {
            if (!group) return;
            const inst = group.children && group.children[0];
            if (!inst || !inst.isMesh) return;
            const mat = inst.material;
            if (!mat) return;
            mat.transparent = true;
            const state = { op: typeof mat.opacity === 'number' ? mat.opacity : 1.0 };
            if (typeof TWEEN !== 'undefined') {
                new TWEEN.Tween(state)
                    .to({ op: targetOpacity }, durationMs)
                    .delay(delayMs)
                    .easing(TWEEN.Easing.Quadratic.InOut)
                    .onUpdate(() => { mat.opacity = state.op; mat.needsUpdate = true; })
                    .start();
            } else {
                mat.opacity = targetOpacity;
                mat.needsUpdate = true;
            }
        };
        this._mergedGroupsByHead.forEach((grp) => {
            if (grp && grp.K) doFade(grp.K);
            if (grp && grp.V) doFade(grp.V);
        });
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
                    .to({ op: 0.2 }, DECORATIVE_FADE_MS)
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



            // Start invisible: set material opacity to 0
            if (decoVec.mesh && decoVec.mesh.material) {
                decoVec.mesh.material.transparent = true;
                decoVec.mesh.material.opacity = 0.0;
            }

            if (typeof TWEEN !== 'undefined') {
                const mat = decoVec.mesh.material;
                new TWEEN.Tween({ op: 0.0 })
                    .to({ op: 1.0 }, DECORATIVE_FADE_MS)
                    .easing(TWEEN.Easing.Quadratic.InOut)
                    .onUpdate(function(o){
                        mat.opacity = o.op;
                        mat.needsUpdate = true;
                    })
                    .delay(DECORATIVE_FADE_DELAY_MS)
                    .start();
            }
        });

        // After decorative vectors begin fading in, further dim gray vectors for subtlety
        if (typeof TWEEN !== 'undefined') {
            this._tempAllOutputVectors.forEach(vec => {
                if (!vec || !vec.mesh || !vec.mesh.material) return;
                const mat = vec.mesh.material;
                new TWEEN.Tween({ op: mat.opacity })
                    .to({ op: 0.05 }, DECORATIVE_FADE_MS)
                    .delay(DECORATIVE_FADE_DELAY_MS)
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
            // Start after decorative fade-in completes
            setTimeout(() => {
                this._startMergeToRowVectors();
            }, MERGE_TO_ROW_DELAY_AFTER_FADE_MS / GLOBAL_ANIM_SPEED_MULT);
        }
    }

    _startMergeToRowVectors() {
        if (!this._tempDecorativeVecs || this._tempDecorativeVecs.length === 0) return;

        // Mark that decorative vectors are now travelling horizontally back
        // toward the output projection alignment row.
        this.rowMergePhase = 'merging';

        // Build map laneZ -> array of decorative vectors
        const laneVectors = new Map();
        this._tempDecorativeVecs.forEach(obj => {
            const laneZ = obj.laneZ;
            if (!laneVectors.has(laneZ)) laneVectors.set(laneZ, []);
            laneVectors.get(laneZ).push(obj.vec);
        });

        const firstHeadCenterX = this.headsCentersX.length ? this.headsCentersX[0] : 0; 
        const targetX = firstHeadCenterX; // Existing merge target for centralised row
        let maxDurationMs = 0;

        laneVectors.forEach((vecList, laneZ) => {
            // Ensure vecList ordered so we can grab a representative Y coordinate
            vecList.sort((a, b) => a.group.position.x - b.group.position.x);
            const yPos = vecList.length ? vecList[0].group.position.y : 0;



            // --------------------------------------------------------------
            //  Launch horizontal merge tweens for the decorative vectors
            // --------------------------------------------------------------
            vecList.forEach((vec, idx) => {
                const destX = targetX + (idx - (NUM_HEAD_SETS_LAYER - 1) / 2) * ROW_SEGMENT_SPACING;
                const distance = Math.abs(vec.group.position.x - destX);
                const durationMs = (distance / (ROW_MERGE_HORIZ_SPEED * GLOBAL_ANIM_SPEED_MULT)) * 1000;
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
                this._transitionHeadColorsToFinal(HEAD_COLOR_TRANSITION_MS);
                setTimeout(() => {
                    this._startVectorsThroughOutputProjection(laneVectors);
                }, MERGE_POST_COLOR_TRANSITION_DELAY_MS);
            }, maxDurationMs + MERGE_EXTRA_BUFFER_MS);
        } else {
            this._transitionHeadColorsToFinal(0);
        }
    }
    
    _startVectorsThroughOutputProjection(laneVectors) {
        // Combine decorative vectors in each lane into a single vector, then animate those combined vectors
        // Mark merge as complete prior to entering the output projection stage
        this.rowMergePhase = 'merged';
        this.outputProjMatrixAnimationPhase = 'vectors_entering';

        // Keep K/V visuals present during concatenation; they will be disposed
        // once the very last blue (Q) vector finishes its conveyor belt.

        const combinedVectors = [];

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
            // Do NOT attach a trail yet. We only want a horizontal trail
            // when returning to the residual stream (no vertical segments).

            combinedVectors.push({ vec: combinedVec, laneZ });

            // Hide original decorative vectors
            vecList.forEach(v => { v.group.visible = false; });


        });

        if (combinedVectors.length === 0) {
            console.warn("No combined vectors created for output projection animation");
            return;
        }

        // Store for later reference
        this.outputProjMatrixVectors = combinedVectors.map(obj => obj.vec);

        // Matrix positions
        const matrixBottomY = this.outputProjMatrixCenterY - this.outputProjMatrixHeight / 2;
        const matrixTopY = this.outputProjMatrixCenterY + this.outputProjMatrixHeight / 2;
        const targetYAboveMatrix = matrixTopY + 30;

        // Durations
        const duration1 = OUTPUT_PROJ_STAGE1_MS / GLOBAL_ANIM_SPEED_MULT;
        const duration2 = OUTPUT_PROJ_STAGE2_MS / GLOBAL_ANIM_SPEED_MULT;
        const duration3 = OUTPUT_PROJ_STAGE3_MS / GLOBAL_ANIM_SPEED_MULT;

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
                    // Keep trail brightness consistent during stage 1 rise
                    if (vec && vec.userData && vec.userData.trail) {
                        vec.userData.trail.update(vec.group.position);
                    }
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
                            // Continue updating trail through matrix travel
                            if (vec && vec.userData && vec.userData.trail) {
                                vec.userData.trail.update(vec.group.position);
                            }
                        })
                        .onComplete(() => {
                            const extraRise = 30; // additional upward distance
                            const finalCombinedY = targetYAboveMatrix + extraRise;

                            // Final rise after transformation done
                            new TWEEN.Tween(vec.group.position)
                                .to({ y: finalCombinedY }, duration3)
                                .easing(TWEEN.Easing.Quadratic.InOut)
                                .onStart(() => {
                                    // Start a trail as the vector exits the output-projection matrix
                                    try {
                                        vec.userData = vec.userData || {};
                                        if (!vec.userData.trail) {
                                            const tr = new StraightLineTrail(this.parentGroup);
                                            tr.start(vec.group.position.clone());
                                            vec.userData.trail = tr;
                                        }
                                    } catch (_) { /* optional visual */ }
                                })
                                .onUpdate(() => {
                                    if (vec && vec.userData && vec.userData.trail) {
                                        vec.userData.trail.update(vec.group.position);
                                    }
                                })
                                .onComplete(() => {
                                    // Horizontal move back to residual stream centre (x = 0),
                                    // then perform the addition with the lane's original vector
                                    const horizDistance = Math.abs(vec.group.position.x);
                                    const horizDur = (horizDistance / (ANIM_HORIZ_SPEED * GLOBAL_ANIM_SPEED_MULT)) * 1000;

                                    new TWEEN.Tween(vec.group.position)
                                        .to({ x: 0 }, horizDur)
                                        .easing(TWEEN.Easing.Quadratic.InOut)
                                        .onUpdate(() => {
                                            // Continue updating the same trail during horizontal travel
                                            const tr = vec && vec.userData && vec.userData.trail;
                                            if (tr) tr.update(vec.group.position);
                                        })

                                        .onComplete(() => {
                                            // Freeze the trail into static segments so the horizontal
                                            // return path remains visible, then remove live trail ref
                                            try {
                                                const tr = vec && vec.userData && vec.userData.trail;
                                                if (tr) {
                                                    mergeTrailsIntoLineSegments([tr], this.parentGroup);
                                                    if (vec && vec.userData) delete vec.userData.trail;
                                                }
                                            } catch (_) { /* optional visual */ }
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

    // Hide all merged K/V instanced groups (both per-head and global-all)
    _hideMergedKVGroups() {
        // Per-head merged groups
        if (this._mergedGroupsByHead && typeof this._mergedGroupsByHead.forEach === 'function') {
            this._mergedGroupsByHead.forEach((grp) => {
                if (grp && grp.K) grp.K.visible = false;
                if (grp && grp.V) grp.V.visible = false;
            });
        }
        // Global merged groups (if created)
        if (this.parentGroup && typeof this.parentGroup.getObjectByName === 'function') {
            const allK = this.parentGroup.getObjectByName('MergedAllK');
            const allV = this.parentGroup.getObjectByName('MergedAllV');
            if (allK) allK.visible = false;
            if (allV) allV.visible = false;
        }
    }

    // Dispose and remove any merged K/V instanced groups (both per-head and global)
    _disposeMergedKVGroups() {
        const safeDisposeGroup = (group) => {
            if (!group) return;
            try {
                const inst = group.children && group.children[0];
                if (inst && inst.isMesh) {
                    if (inst.material) {
                        if (Array.isArray(inst.material)) {
                            inst.material.forEach((m) => { if (m && m.dispose) m.dispose(); });
                        } else if (inst.material.dispose) {
                            inst.material.dispose();
                        }
                    }
                    if (inst.geometry && inst.geometry.dispose) inst.geometry.dispose();
                }
                if (group.parent) {
                    group.parent.remove(group);
                } else if (this.parentGroup) {
                    this.parentGroup.remove(group);
                }
            } catch (_) { /* ignore */ }
        };

        // Per-head merged groups
        if (this._mergedGroupsByHead && typeof this._mergedGroupsByHead.forEach === 'function') {
            this._mergedGroupsByHead.forEach((grp) => {
                if (grp && grp.K) safeDisposeGroup(grp.K);
                if (grp && grp.V) safeDisposeGroup(grp.V);
            });
            try { this._mergedGroupsByHead.clear(); } catch (_) { /* ignore */ }
        }

        // Global merged groups (if created)
        if (this.parentGroup && typeof this.parentGroup.getObjectByName === 'function') {
            const allK = this.parentGroup.getObjectByName('MergedAllK');
            const allV = this.parentGroup.getObjectByName('MergedAllV');
            if (allK) safeDisposeGroup(allK);
            if (allV) safeDisposeGroup(allV);
        }

        // Reset tracking flags/collections
        try { if (this._mergedHeads && this._mergedHeads.clear) this._mergedHeads.clear(); } catch (_) { /* ignore */ }
        this._allFixedMerged = false;
    }

    // Poll self-attention animator until all blue (Q) conveyors across heads
    // have completed, then dispose merged K/V groups so they disappear exactly
    // after the last blue finishes its path.
    _waitAndDisposeMergedKVWhenBlueConveyorComplete() {
        // Prefer a direct completion callback from SelfAttentionAnimator
        try {
            if (this.selfAttentionAnimator && typeof this.selfAttentionAnimator.start === 'function') {
                this.selfAttentionAnimator.start(() => {
                    try { this._disposeMergedKVGroups(); } catch (_) {}
                });
                return;
            }
        } catch (_) { /* ignore and fall back to polling */ }

        // Fallback: poll phase with a safety timeout
        const checkIntervalMs = 100;
        const maxWaitMs = 120000;
        let waited = 0;
        const tick = () => {
            try {
                if (!this.selfAttentionAnimator || this.selfAttentionAnimator.phase === 'complete') {
                    this._disposeMergedKVGroups();
                    return;
                }
            } catch (_) { /* ignore */ }
            waited += checkIntervalMs;
            if (waited >= maxWaitMs) {
                try { this._disposeMergedKVGroups(); } catch (_) {}
                return;
            }
            setTimeout(tick, checkIntervalMs);
        };
        setTimeout(tick, checkIntervalMs);
    }

    // Immediately hide all K (green) and V (red) vectors, including any merged
    // instanced groups and any leftover individual K/V copies in lanes.
    _hideAllKandVVectorsImmediately() {
        try { this._hideMergedKVGroups(); } catch (_) {}
        const lanes = this.currentLanes || [];
        lanes.forEach((lane) => {
            // Hide K upward copies for every head
            if (Array.isArray(lane.upwardCopies)) {
                lane.upwardCopies.forEach((kVec) => {
                    if (!kVec) return;
                    try {
                        if (kVec.mesh) kVec.mesh.visible = false;
                        if (kVec.group) kVec.group.visible = false;
                    } catch (_) {}
                });
            }
            // Hide V side copies
            if (Array.isArray(lane.sideCopies)) {
                lane.sideCopies.forEach((sc) => {
                    if (!sc || sc.type !== 'V') return;
                    const v = sc.vec;
                    if (!v) return;
                    try {
                        if (v.mesh) v.mesh.visible = false;
                        if (v.group) v.group.visible = false;
                    } catch (_) {}
                });
            }
        });
    }

    // Permanently dispose any remaining individual K (green) and V (red) vectors
    // across all lanes. Keeps the empty groups (for positional queries) but
    // removes meshes and their GPU resources so nothing remains visible.
    _disposeAllIndividualKandVVectorsImmediately() {
        const stripMeshKeepGroup = (vec) => {
            try {
                if (!vec) return;
                if (vec.mesh) {
                    try { if (vec.mesh.material) vec.mesh.material.dispose(); } catch (_) {}
                    try { if (vec.mesh.geometry) vec.mesh.geometry.dispose(); } catch (_) {}
                    if (vec.group) vec.group.remove(vec.mesh);
                    vec.mesh = null;
                }
                if (vec.group) vec.group.visible = false;
            } catch (_) { /* ignore */ }
        };
        const lanes = this.currentLanes || [];
        lanes.forEach((lane) => {
            // Upward K copies for each head
            if (Array.isArray(lane.upwardCopies)) {
                lane.upwardCopies.forEach((kVec, idx) => {
                    stripMeshKeepGroup(kVec);
                });
            }
            // Side V copies for each head
            if (Array.isArray(lane.sideCopies)) {
                lane.sideCopies.forEach((sc) => {
                    if (!sc || sc.type !== 'V') return;
                    stripMeshKeepGroup(sc.vec);
                });
            }
        });
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

        // Ensure matrix begins in its dark resting state
        this.outputProjectionMatrix.setColor(startColor);
        this.outputProjectionMatrix.setEmissive(startColor, startEmissiveIntensity);

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
        // Initiate prism-by-prism addition animation where prisms from sourceVec
        // move into their corresponding positions in targetVec.
        // The lane object is forwarded so the helper can update lane state
        // (stopRise flags, phase transitions, etc.).
        startPrismAdditionAnimation(sourceVec, targetVec, lane);
        // Don't force absolute positions here – vectors should keep their
        // natural flow handled by the tween callbacks inside the helper.
    }
} 
