import * as THREE from 'three';
import BaseLayer from '../BaseLayer.js';
import { LayerNormalizationVisualization } from '../../components/LayerNormalizationVisualization.js';
import { WeightMatrixVisualization } from '../../components/WeightMatrixVisualization.js';
import { VectorVisualizationInstancedPrism } from '../../components/VectorVisualizationInstancedPrism.js';
import { StraightLineTrail, buildMergedLineSegmentsFromSegments, collectTrailsUnder, mergeTrailsIntoLineSegments } from '../../utils/trailUtils.js';
import {
    LN_PARAMS,
    LN_TO_MHA_GAP,
    BRANCH_X,
    LN2_TO_MLP_GAP,
    LAYER_NORM_1_Y_POS,
    LAYER_NORM_2_Y_POS,
    MLP_MATRIX_PARAMS_UP,
    MLP_MATRIX_PARAMS_DOWN,
    MLP_INTER_MATRIX_GAP,
    MHA_MATRIX_PARAMS,
    NUM_HEAD_SETS_LAYER,
    HEAD_SET_GAP_LAYER,
    MHA_INTERNAL_MATRIX_SPACING,
    NUM_VECTOR_LANES,
    VECTOR_DEPTH_SPACING,
    ANIM_OFFSET_Y_ORIGINAL_SPAWN,
    ANIM_RISE_SPEED_ORIGINAL,
    VECTOR_LENGTH_PRISM,
    GLOBAL_ANIM_SPEED_MULT,
    ANIM_RISE_SPEED_INSIDE_LN,
    ANIM_RISE_SPEED_POST_SPLIT_LN1,
    ANIM_RISE_SPEED_POST_SPLIT_LN2,
    ORIGINAL_TO_PROCESSED_GAP,
    ANIM_HORIZ_SPEED,
    INACTIVE_COMPONENT_COLOR,
    PRISM_ADD_ANIM_BASE_DURATION,
    PRISM_ADD_ANIM_BASE_FLASH_DURATION,
    PRISM_ADD_ANIM_BASE_DELAY_BETWEEN_PRISMS,
    PRISM_ADD_ANIM_SPEED_MULT
} from '../../utils/constants.js';
import {
    MHA_FINAL_Q_COLOR,
    MHA_FINAL_K_COLOR,
    MHA_FINAL_V_COLOR,
    MHA_OUTPUT_PROJECTION_MATRIX_PARAMS,
    MHA_OUTPUT_PROJECTION_MATRIX_COLOR,
    MHA_OUTPUT_PROJECTION_MATRIX_Y_OFFSET_ABOVE_ROW,
} from '../../animations/LayerAnimationConstants.js';
import { PrismLayerNormAnimation } from '../../animations/PrismLayerNormAnimation.js';
import { MHSAAnimation } from '../../animations/MHSAAnimation.js';


const VERTICAL_SPACING = 1500; // matches LayerAnimation.js vertical extent

function simplePrismMultiply(srcVec, tgtVec, onComplete) {
    // instant product; flash white then call onComplete
    for (let i=0;i<VECTOR_LENGTH_PRISM;i++) {
        tgtVec.rawData[i] = srcVec.rawData[i]*tgtVec.rawData[i];
    }
    tgtVec.updateKeyColorsFromData(tgtVec.rawData,30);
    if (onComplete) onComplete();
}


export default class Gpt2Layer extends BaseLayer {
    /**
     * @param {number} index – 0-based index of this layer in the transformer.
     * @param {object} random – object returned by createRandomSource().
     * @param {number} yOffset – Optional y-offset for this layer.
     * @param {Layer} waitForLayer – Optional layer to wait for before starting.
     * @param {Array} externalLanes – Optional array of lanes to re-use.
     * @param {Function} onFinished – Optional callback to invoke when all lanes finish.
     */
    constructor(index, random, yOffset = 0, externalLanes = null, onFinished = null, isActive = true) {
        super(index);
        this.random = random;
        this.yOffset = yOffset;
        this.externalLanes = externalLanes;
        this.onFinished = typeof onFinished === 'function' ? onFinished : null;
        this.isActive = isActive;
        this._completed = false;
        this._pendingAdditions = 0; // Track ongoing addition animations
        // Synchronisation flag – ensures all lanes begin the LN-2 branch at the exact same time.
        this._ln2Ready = false;
        // Flag to indicate that all lanes have reached MLP readiness.
        this._mlpStart = false;
        // NEW: Barrier flags
        this._ln1Start = false;        // start LN-1 branch
        this._mhsaStart = false;       // start horizontal travel to heads
    }

    init(scene) {
        super.init(scene);

        // Keep a reference to the global scene so certain visuals (like the
        // continuous residual-stream trails) can render in world-space and
        // survive lane hand-offs between stacked layers.
        this._globalScene = scene;

        // Offset root vertically for stack layout
        this.root.position.y = this.index * VERTICAL_SPACING + this.yOffset;

        const offsetX = BRANCH_X; // all branched components share this X

        // ────────────────────────────────────────────────────────────────
        // 1) LayerNorm 1
        // ────────────────────────────────────────────────────────────────
        const ln1CenterY = LAYER_NORM_1_Y_POS;
        const ln1 = new LayerNormalizationVisualization(
            new THREE.Vector3(offsetX, ln1CenterY, 0),
            LN_PARAMS.width,
            LN_PARAMS.height,
            LN_PARAMS.depth,
            LN_PARAMS.wallThickness,
            LN_PARAMS.numberOfHoles,
            LN_PARAMS.holeWidth,
            LN_PARAMS.holeWidthFactor
        );
        // Start LN1 in the same "inactive" palette used for LN2 so it only lights up once
        // vectors begin to enter the ring.
        const inactiveDark = new THREE.Color(INACTIVE_COMPONENT_COLOR);
        ln1.setColor(inactiveDark);
        // Start fully opaque to avoid early depth-sorting costs.
        ln1.setMaterialProperties({ opacity: 1.0, transparent: false, emissiveIntensity: 0.05 });
        this.root.add(ln1.group);

        const ln1TopY = ln1CenterY + LN_PARAMS.height / 2;
        this.ln1TopY = ln1TopY;

        // ────────────────────────────────────────────────────────────────
        // 2) Multi-Head Self-Attention matrices
        // ────────────────────────────────────────────────────────────────
        // Skipped: Legacy per-layer Q/K/V matrix group construction.
        // MHSA visuals are now created exclusively by MHSAAnimation to avoid
        // duplicate geometry and unnecessary CPU/GPU overhead.

        // ────────────────────────────────────────────────────────────────
        //  MHSAAnimation controller – handles vector routing through the
        //  12-head attention block and subsequent merging logic.
        // ────────────────────────────────────────────────────────────────

        const mhaBaseY_local = ln1TopY + LN_TO_MHA_GAP;
        const mhaBaseY = mhaBaseY_local;
        // Create an internal clock for sub-animations handled by the MHSA helper
        this._mhsaClock = new THREE.Clock();
        // Pass empty opts – twelve-layer stack keeps self-attention disabled via global static flag.
        this.mhsaAnimation = new MHSAAnimation(this.root, BRANCH_X, mhaBaseY, this._mhsaClock, 'temp', {});

        // ────────────────────────────────────────────────────────────────
        // 2.5) Output-projection matrix
        //      Now created solely inside MHSAAnimation.  The former static
        //      placeholder has been removed to stop overlapping transparent
        //      meshes that caused colour flickering.

        // ────────────────────────────────────────────────────────────────
        // 3) LayerNorm 2
        // ────────────────────────────────────────────────────────────────
        const ln2CenterY = LAYER_NORM_2_Y_POS;
        const ln2 = new LayerNormalizationVisualization(
            new THREE.Vector3(offsetX, ln2CenterY, 0),
            LN_PARAMS.width,
            LN_PARAMS.height,
            LN_PARAMS.depth,
            LN_PARAMS.wallThickness,
            LN_PARAMS.numberOfHoles,
            LN_PARAMS.holeWidth,
            LN_PARAMS.holeWidthFactor
        );
        ln2.setColor(inactiveDark);
        ln2.setMaterialProperties({ opacity: 1.0, transparent: false, emissiveIntensity: 0.05 });
        this.root.add(ln2.group);

        const ln2TopY = ln2CenterY + LN_PARAMS.height / 2;

        // ────────────────────────────────────────────────────────────────
        // 4) MLP Up-projection matrix (orange)
        // ────────────────────────────────────────────────────────────────
        const mlpUpCenterY = ln2TopY + LN2_TO_MLP_GAP + MLP_MATRIX_PARAMS_UP.height / 2;
        const mlpUp = new WeightMatrixVisualization(
            null,
            new THREE.Vector3(offsetX, mlpUpCenterY, 0),
            MLP_MATRIX_PARAMS_UP.width,
            MLP_MATRIX_PARAMS_UP.height,
            MLP_MATRIX_PARAMS_UP.depth,
            MLP_MATRIX_PARAMS_UP.topWidthFactor,
            MLP_MATRIX_PARAMS_UP.cornerRadius,
            MLP_MATRIX_PARAMS_UP.numberOfSlits,
            MLP_MATRIX_PARAMS_UP.slitWidth,
            MLP_MATRIX_PARAMS_UP.slitDepthFactor,
            MLP_MATRIX_PARAMS_UP.slitBottomWidthFactor,
            MLP_MATRIX_PARAMS_UP.slitTopWidthFactor
        );
        mlpUp.setColor(inactiveDark.clone());
        mlpUp.setMaterialProperties({ opacity: 1.0, transparent: false });
        {
            const lbl = 'MLP Up Weight Matrix';
            mlpUp.group.userData.label = lbl;
            if (mlpUp.mesh) mlpUp.mesh.userData.label = lbl;
            if (mlpUp.frontCapMesh) mlpUp.frontCapMesh.userData.label = lbl;
            if (mlpUp.backCapMesh)  mlpUp.backCapMesh.userData.label  = lbl;
        }
        this.root.add(mlpUp.group);

        // 5) MLP Down-projection matrix (same orange)
        const mlpDownCenterY = mlpUpCenterY + MLP_MATRIX_PARAMS_UP.height / 2 + MLP_INTER_MATRIX_GAP + MLP_MATRIX_PARAMS_DOWN.height / 2;
        const mlpDown = new WeightMatrixVisualization(
            null,
            new THREE.Vector3(offsetX, mlpDownCenterY, 0),
            MLP_MATRIX_PARAMS_DOWN.width,
            MLP_MATRIX_PARAMS_DOWN.height,
            MLP_MATRIX_PARAMS_DOWN.depth,
            MLP_MATRIX_PARAMS_DOWN.topWidthFactor,
            MLP_MATRIX_PARAMS_DOWN.cornerRadius,
            MLP_MATRIX_PARAMS_DOWN.numberOfSlits,
            MLP_MATRIX_PARAMS_DOWN.slitWidth,
            MLP_MATRIX_PARAMS_DOWN.slitDepthFactor,
            MLP_MATRIX_PARAMS_DOWN.slitBottomWidthFactor,
            MLP_MATRIX_PARAMS_DOWN.slitTopWidthFactor
        );
        mlpDown.setColor(inactiveDark.clone());
        mlpDown.setMaterialProperties({ opacity: 1.0, transparent: false });
        {
            const lbl = 'MLP Down Weight Matrix';
            mlpDown.group.userData.label = lbl;
            if (mlpDown.mesh) mlpDown.mesh.userData.label = lbl;
            if (mlpDown.frontCapMesh) mlpDown.frontCapMesh.userData.label = lbl;
            if (mlpDown.backCapMesh)  mlpDown.backCapMesh.userData.label  = lbl;
        }
        this.root.add(mlpDown.group);

        // ---------- Residual vectors (original stream) ----------
        this.lanes = [];
        if (this.isActive) {
            if (this.externalLanes && this.externalLanes.length) {
                this._createLanesFromExternal(this.externalLanes, offsetX, ln1CenterY, ln2CenterY, ln1TopY);
            } else {
                this._createFreshLanes(offsetX, ln1CenterY, ln2CenterY, ln1TopY);
            }
        }

        // Keep references for per-frame updates
        this.ln1 = ln1;
        this.ln2 = ln2;
        this.mlpUp = mlpUp;
        this.mlpDown = mlpDown;
    }

    update(dt) {
        // ──────────────── Update straight-line trails ────────────────
        if (this.lanes && this.lanes.length) {
            this.lanes.forEach(lane => {
                const vecsToCheck = [
                    lane.originalVec,
                    lane.dupVec,
                    lane.travellingVec,
                    lane.resultVec,
                    lane.postAdditionVec,
                    lane.movingVecLN2,
                    lane.resultVecLN2,
                    lane.finalVecAfterMlp
                ];

                // Ensure we only update each unique trail once per frame per lane
                const updatedTrailRefs = new Set();

                vecsToCheck.forEach(v => {
                    if (!v || !v.userData || !v.userData.trail) return;
                    const trailRef = v.userData.trail;
                    if (updatedTrailRefs.has(trailRef)) return;

                    // During residual addition, let MHSAAnimation drive world-space
                    // residual trail updates (it follows the centre prism). Skip here
                    // to avoid double-writing the same world trail in the same frame.
                    if (lane.stopRise && v.userData.trailWorld) return;

                    if (v.userData.trailWorld) {
                        const wp = new THREE.Vector3();
                        v.group.getWorldPosition(wp);
                        trailRef.update(wp);
                    } else {
                        trailRef.update(v.group.position);
                    }
                    updatedTrailRefs.add(trailRef);
                });

                // Expanded 4× vector group trail
                if (lane.expandedVecGroup && lane.expandedVecTrail) {
                    lane.expandedVecTrail.update(lane.expandedVecGroup.position);
                }

                // Trails for K/Q/V copies are updated inside VectorRouter.
                // Avoid double-updating here to reduce CPU work.
            });
        }
        // Handle transition phase - wait for vectors to reach position
        if (this._transitionPhase === 'positioning') {
            const allVectorsInPosition = this.lanes.every(lane => {
                const targetY = lane.branchStartY;
                return lane.originalVec.group.position.y >= targetY - 5; // small tolerance
            });
            
            if (allVectorsInPosition) {
                this._transitionPhase = 'complete';
                this.isActive = true; // now start the actual animation
                console.log(`Layer ${this.index}: All vectors in position, starting animation`);
            } else {
                // Keep vectors rising toward the target
                this.lanes.forEach(lane => {
                    const targetY = lane.branchStartY;
                    if (lane.originalVec.group.position.y < targetY) {
                        lane.originalVec.group.position.y = Math.min(targetY, 
                            lane.originalVec.group.position.y + ANIM_RISE_SPEED_ORIGINAL * GLOBAL_ANIM_SPEED_MULT * dt);
                    }
                });
                return; // keep waiting
            }
        }
        
        if (!this.isActive) return; // Skip processing when inactive / placeholder
        // ────────────────────────────────────────────────────────────
        // Dynamic colour / opacity transition for the FIRST LayerNorm
        // ────────────────────────────────────────────────────────────
        const darkGray = new THREE.Color(0x333333);
        const lightYellow = new THREE.Color(0xFFFFFF); // transition target updated to white
        const brightYellow = new THREE.Color(0xFFFFFF);
        const opaqueOpacity = 1.0;
        const semiTransparentOpacity = 0.6;
        const exitTransitionRange = 5; // world–unit distance for final fade

        const bottomY_ln1_abs = LAYER_NORM_1_Y_POS - LN_PARAMS.height / 2;
        const midY_ln1_abs    = LAYER_NORM_1_Y_POS;
        const topY_ln1_abs    = LAYER_NORM_1_Y_POS + LN_PARAMS.height / 2;

        // Determine the highest Y position of any vector still interacting with LN-1
        let highestLN1VecY = -Infinity;
        let anyVectorInLN1 = false;

        this.lanes.forEach(lane => {
            // Duplicate vector that travels through LN-1
            if (lane.dupVec && lane.dupVec.group.visible) {
                const y = lane.dupVec.group.position.y;
                highestLN1VecY = Math.max(highestLN1VecY, y);
                if (y >= bottomY_ln1_abs - exitTransitionRange) anyVectorInLN1 = true;
            }
            // Result vector that rises out of LN-1
            if (lane.resultVec && lane.resultVec.group.visible) {
                const y = lane.resultVec.group.position.y;
                highestLN1VecY = Math.max(highestLN1VecY, y);
                if (y >= bottomY_ln1_abs - exitTransitionRange) anyVectorInLN1 = true;
            }
        });

        let targetColor = darkGray.clone();
        let targetOpacity = opaqueOpacity;

        if (anyVectorInLN1 && highestLN1VecY > -Infinity) {
            if (highestLN1VecY >= bottomY_ln1_abs && highestLN1VecY < midY_ln1_abs) {
                // Entering LN-1
                const t = (highestLN1VecY - bottomY_ln1_abs) / (midY_ln1_abs - bottomY_ln1_abs);
                targetColor = darkGray.clone().lerp(lightYellow, t);
                targetOpacity = THREE.MathUtils.lerp(opaqueOpacity, semiTransparentOpacity, t);
            } else if (highestLN1VecY >= midY_ln1_abs && highestLN1VecY < topY_ln1_abs) {
                // Inside LN-1
                targetColor = lightYellow.clone();
                targetOpacity = semiTransparentOpacity;
            } else if (highestLN1VecY >= topY_ln1_abs) {
                // Exiting LN-1
                const tRaw = (highestLN1VecY - topY_ln1_abs) / exitTransitionRange;
                const t = Math.min(1, Math.max(0, tRaw));
                targetColor = lightYellow.clone().lerp(brightYellow, t);
                targetOpacity = THREE.MathUtils.lerp(semiTransparentOpacity, opaqueOpacity, t);
            }
        }

        // -------------------------------------------------------------
        // Once a vector has risen sufficiently above LN-1 we want to
        // "bake" the bright colour so the ring doesn't revert to the
        // inactive palette when no vectors are nearby (e.g. while the
        // MHSA animation runs).  We do this by latching a flag the first
        // frame the exit transition completes.
        if (!this._ln1ColorLocked) {
            this._ln1ColorLocked = false; // ensure property exists
        }

        if (!this._ln1ColorLocked && highestLN1VecY >= topY_ln1_abs + exitTransitionRange) {
            this._ln1ColorLocked = true;
            this._ln1LockedColor = brightYellow.clone();
        }

        if (this._ln1ColorLocked) {
            targetColor = this._ln1LockedColor.clone();
            targetOpacity = opaqueOpacity;
        }

        // Apply to mesh material(s)
        if (this.ln1 && this.ln1.group) {
            this.ln1.group.children.forEach(child => {
                if (child instanceof THREE.Mesh && child.material) {
                    child.material.color.copy(targetColor);
                    child.material.emissive.copy(targetColor);
                    child.material.transparent = targetOpacity < 1.0;
                    child.material.opacity = targetOpacity;
                    child.material.needsUpdate = true;
                }
            });
        }

        // ────────────────────────────────────────────────────────────
        // Dynamic colour / opacity transition for the SECOND LayerNorm
        // ────────────────────────────────────────────────────────────
        const bottomY_ln2_abs = LAYER_NORM_2_Y_POS - LN_PARAMS.height / 2;
        const midY_ln2_abs = LAYER_NORM_2_Y_POS;
        const topY_ln2_abs = LAYER_NORM_2_Y_POS + LN_PARAMS.height / 2;

        // Find the highest Y position of any vector moving through LN2
        let highestLN2VecY = -Infinity;
        let anyVectorInLN2 = false;

        this.lanes.forEach(lane => {
            if (lane.movingVecLN2 && lane.movingVecLN2.group.visible) {
                const y = lane.movingVecLN2.group.position.y;
                highestLN2VecY = Math.max(highestLN2VecY, y);
                if (y >= bottomY_ln2_abs - exitTransitionRange) {
                    anyVectorInLN2 = true;
                }
            } else if (lane.resultVecLN2 && lane.resultVecLN2.group.visible) {
                const y = lane.resultVecLN2.group.position.y;
                highestLN2VecY = Math.max(highestLN2VecY, y);
                if (y >= bottomY_ln2_abs - exitTransitionRange) {
                    anyVectorInLN2 = true;
                }
            }
        });

        if (anyVectorInLN2 && highestLN2VecY > -Infinity) {
            let ln2TargetColor = darkGray.clone();
            let ln2TargetOpacity = opaqueOpacity;

            if (highestLN2VecY >= bottomY_ln2_abs && highestLN2VecY < midY_ln2_abs) {
                // Entering LN2
                const t = (highestLN2VecY - bottomY_ln2_abs) / (midY_ln2_abs - bottomY_ln2_abs);
                ln2TargetColor = darkGray.clone().lerp(lightYellow, t);
                ln2TargetOpacity = THREE.MathUtils.lerp(opaqueOpacity, semiTransparentOpacity, t);
            } else if (highestLN2VecY >= midY_ln2_abs && highestLN2VecY < topY_ln2_abs) {
                // Inside LN2
                ln2TargetColor = lightYellow.clone();
                ln2TargetOpacity = semiTransparentOpacity;
            } else if (highestLN2VecY >= topY_ln2_abs) {
                // Exiting LN2
                const tRaw = (highestLN2VecY - topY_ln2_abs) / exitTransitionRange;
                const t = Math.min(1, Math.max(0, tRaw));
                ln2TargetColor = lightYellow.clone().lerp(brightYellow, t);
                ln2TargetOpacity = THREE.MathUtils.lerp(semiTransparentOpacity, opaqueOpacity, t);
            }

            // Apply to LN2
            if (this.ln2 && this.ln2.group) {
                this.ln2.group.children.forEach(child => {
                    if (child instanceof THREE.Mesh && child.material) {
                        child.material.color.copy(ln2TargetColor);
                        child.material.emissive.copy(ln2TargetColor);
                        child.material.transparent = ln2TargetOpacity < 1.0;
                        child.material.opacity = ln2TargetOpacity;
                        child.material.needsUpdate = true;
                    }
                });
            }
        }

        // ────────────────────────────────────────────────────────────
        // LN-2 synchronisation check – only once all lanes have reached
        // the staging height (ln2Phase === 'preRise' and held position)
        // do we allow any of them to continue into the branch.
        // ────────────────────────────────────────────────────────────
        if (!this._ln2Ready) {
            const syncY = bottomY_ln2_abs + 5; // target height used in 'preRise'
            const allLn2Ready = this.lanes.length && this.lanes.every(l =>
                l.ln2Phase === 'preRise' &&
                l.postAdditionVec && l.postAdditionVec.group &&
                l.postAdditionVec.group.position.y >= syncY - 0.01);

            if (allLn2Ready) {
                this._ln2Ready = true;
                console.log(`Layer ${this.index}: All lanes ready – starting LN2 simultaneously`);
            }
        }

        // ----------------------------------------------------------------
        // MLP synchronisation: wait until every lane has completed LN-2 and
        // is marked as 'mlpReady' before triggering the up-projection.
        // ----------------------------------------------------------------
        if (!this._mlpStart) {
            const allMlpReady = this.lanes.length && this.lanes.every(l => l.ln2Phase === 'mlpReady');
            if (allMlpReady) {
                this._mlpStart = true;
                console.log(`Layer ${this.index}: All lanes ready – starting MLP up-projection simultaneously`);
            }
        }

        const speedMult = GLOBAL_ANIM_SPEED_MULT;

        // ────────────────────────────────────────────────────────────────
        //  NEW: LayerNorm-1 synchronisation barrier – wait until EVERY
        //  lane's original residual-stream vector has reached the branching
        //  height before triggering the horizontal duplicate move.  This
        //  guarantees that all lanes enter LN-1 in perfect lock-step.
        // ────────────────────────────────────────────────────────────────
        if (!this._ln1Start) {
            const allLn1Ready = this.lanes.length && this.lanes.every(l => l.originalVec.group.position.y >= l.branchStartY);
            if (allLn1Ready) {
                this._ln1Start = true;
                console.log(`Layer ${this.index}: All lanes ready – starting LN-1 branch simultaneously`);

                // Kick every lane out of the waiting state together.
                this.lanes.forEach(l => {
                    if (l.horizPhase === 'waiting') {
                        l.horizPhase = 'right';
                        l.dupVec.group.visible = true;
                        // Snap duplicate to the LN-1 branch staging height to avoid
                        // any vertical drift while moving horizontally into the ring.
                        if (typeof l.branchStartY === 'number') {
                            l.dupVec.group.position.y = l.branchStartY;
                        } else {
                            l.dupVec.group.position.y = l.originalVec.group.position.y;
                        }
                    }
                });
            }
        }

        //  NEW: MHSA travel synchronisation barrier – wait until every lane
        //  has its duplicate result vector staged above LN-1 before letting
        //  them begin horizontal travel to the attention heads.
        // ────────────────────────────────────────────────────────────────
        if (!this._mhsaStart) {
            const allMHSAReady = this.lanes.length && this.lanes.every(l => l.horizPhase === 'readyMHSA');
            if (allMHSAReady) {
                this._mhsaStart = true;
                console.log(`Layer ${this.index}: All lanes ready – starting travel to MHSA heads simultaneously`);
                this.lanes.forEach(l => {
                    if (l.horizPhase === 'readyMHSA') l.horizPhase = 'travelMHSA';
                });
            }
        }

        this.lanes.forEach(lane => {
            const { originalVec, dupVec } = lane;
            
            // The Gpt2Layer's direct update logic is now ONLY responsible for
            // handling the initial branching toward the first LayerNorm.
            // ALL subsequent movement, including the continuous rise of the
            // original residual-stream vector, is now managed by the
            // dedicated MHSAAnimation controller. This prevents conflicting
            // updates.

            // branch logic
            switch (lane.horizPhase) {
                case 'waiting':
                    // Hold at the branching height until the global LN-1
                    // barrier is released.
                    if (originalVec.group.position.y >= lane.branchStartY) {
                        // Clamp position so early-arriving lanes don’t drift.
                        originalVec.group.position.y = lane.branchStartY;
                    } else {
                        // Continue rising towards the branching height.
                        originalVec.group.position.y = Math.min(lane.branchStartY, originalVec.group.position.y + ANIM_RISE_SPEED_ORIGINAL * speedMult * dt);
                    }
                    break;
                case 'right':
                    // Mirror LN-2: lock Y at staging height and move X only.
                    if (typeof lane.branchStartY === 'number') {
                        dupVec.group.position.y = lane.branchStartY;
                    }
                    dupVec.group.position.x = Math.min(BRANCH_X, dupVec.group.position.x + ANIM_HORIZ_SPEED * speedMult * dt);
                    if (dupVec.group.position.x >= BRANCH_X - 0.01) {
                        // Ensure alignment with LN-1 centre
                        dupVec.group.position.x = BRANCH_X;
                        // Show the multiplication target inside LN-1 (parity with LN-2 behaviour)
                        if (lane.multTarget && lane.multTarget.group) {
                            lane.multTarget.group.visible = true;
                        }
                        lane.horizPhase = 'insideLN';
                    }
                    break;
                case 'insideLN':
                    // start norm animation after entering 35% of LN
                    const normStartY = lane.ln1MidY - LN_PARAMS.height * 0.15;
                    if (!lane.normStarted && dupVec.group.position.y >= normStartY) {
                        lane.normAnim.start(dupVec.rawData.slice());
                        lane.normStarted = true;
                    }
                    if (lane.normStarted) {
                        lane.normAnim.update(dt);
                    }
                    const normAnimating = lane.normStarted && lane.normAnim.isAnimating;
                    if (!normAnimating) {
                        dupVec.group.position.y = Math.min(lane.ln1MidY, dupVec.group.position.y + ANIM_RISE_SPEED_INSIDE_LN * speedMult * dt);
                    }
                    // --- NEW POST-MOVE SAFETY CHECK -----------------------------------
                    // If the frame delta is large enough that the vector skipped over
                    // the trigger height in a single update, the earlier check will
                    // have missed it.  Run an extra guard here to ensure the
                    // normalisation animation still begins.
                    if (!lane.normStarted && dupVec.group.position.y >= normStartY) {
                        lane.normAnim.start(dupVec.rawData.slice());
                        lane.normStarted = true;
                    }
                    // -----------------------------------------------------------------
                    if (dupVec.group.position.y >= lane.ln1MidY - 0.01) {
                        lane.multStarted = true;
                        if (lane.multTarget) {
                            lane.multTarget.group.visible = true;
                            simplePrismMultiply(dupVec, lane.multTarget, () => {
                                dupVec.group.visible = false;
                                lane.multTarget.group.visible = false;
                                const res = new VectorVisualizationInstancedPrism(lane.multTarget.rawData.slice(), lane.multTarget.group.position.clone());
                                // Trail for the result vector that will travel to MHSA
                                const resTrail = new StraightLineTrail(this.root, 0xffffff, 1);
                                resTrail.start(res.group.position);
                                res.userData = res.userData || {};
                                res.userData.trail = resTrail;
                                this.root.add(res.group);
                                lane.resultVec = res;
                                lane.horizPhase = 'riseAboveLN';
                            });
                        }
                    }
                    break;
                case 'riseAboveLN':
                    // Rise to just above LN1 before starting horizontal travel
                    const rv = lane.resultVec;
                    if (rv) {
                        const targetY = this.ln1TopY + 5; // Same as meetY in original
                        if (rv.group.position.y < targetY) {
                            rv.group.position.y = Math.min(targetY, rv.group.position.y + ANIM_RISE_SPEED_INSIDE_LN * speedMult * dt);
                        } else {
                            // Now that we're above LN1, mark lane ready for MHSA travel.
                            lane.travellingVec = rv;
                            lane.headIndex = 0;
                            lane.horizPhase = 'readyMHSA'; // wait for global barrier
                        }
                    }
                    break;
                case 'readyMHSA':
                    // Hold at staging height until global _mhsaStart flag triggers.
                    // Ensure vector stays exactly at meetY.
                    if (lane.travellingVec) {
                        lane.travellingVec.group.position.y = this.ln1TopY + 5;
                    }
                    break;
                case 'travelMHSA':
                    // MHSAAnimation will handle the horizontal movement
                    break;
                case 'postMHSAAddition':
                    // After MHSA addition completes, start LN2 phase
                    // This state is set by MHSAAnimation._startAdditionAnimation
                    if (lane.ln2Phase === 'preRise' && lane.postAdditionVec) {
                        lane.horizPhase = 'waitingForLN2'; // Move to next state
                    }
                    break;
                case 'waitingForLN2':
                    // Just a placeholder state while LN2 animation runs
                    break;
                default:
                    break;
            }

            // ─────────────────────────────────────────────────────────────
            // LayerNorm2 / MLP Pipeline
            // ─────────────────────────────────────────────────────────────
            switch (lane.ln2Phase) {
                case 'preRise': {
                    // Rise the post-addition vector before branching to LN2
                    const v = lane.postAdditionVec;
                    if (!v) break;
                    
                    // Stage the vector at the same relative position used for LayerNorm-1
                    // (5 units above the bottom of the norm ring) so that the ensuing
                    // normalisation begins at a consistent height across both LayerNorms.
                    const targetY = bottomY_ln2_abs + 5; // align with LN1 offset
                    if (v.group.position.y < targetY) {
                        v.group.position.y = Math.min(targetY, v.group.position.y + ANIM_RISE_SPEED_POST_SPLIT_LN2 * speedMult * dt);
                    } else {
                        if (!this._ln2Ready) {
                            // Wait here until every lane reaches the staging height.
                            break;
                        }

                        // ────────────────────────────────────────────────
                        //  Reached staging height – begin LN-2 branch
                        // ────────────────────────────────────────────────

                        // Allow residual stream to keep rising while the
                        // duplicate goes through LN-2/MLP.
                        if (this.mhsaAnimation && typeof this.mhsaAnimation.finalOriginalY === 'number') {
                            const newTarget = this.mlpUp.group.position.y + MLP_MATRIX_PARAMS_UP.height / 2 - ORIGINAL_TO_PROCESSED_GAP;
                            if (newTarget > this.mhsaAnimation.finalOriginalY) {
                                this.mhsaAnimation.finalOriginalY = newTarget;
                            }
                            this.mhsaAnimation.postSplitRiseSpeed = ANIM_RISE_SPEED_POST_SPLIT_LN2;
                        }

                        // Spawn duplicate vector that will travel into LN-2
                        const mv = new VectorVisualizationInstancedPrism(v.rawData.slice(), v.group.position.clone());
                        this.root.add(mv.group);
                        // ---- Trail for LN2 moving vector ----
                        const mvTrail = new StraightLineTrail(this.root, 0xffffff, 1);
                        mvTrail.start(mv.group.position);
                        mv.userData = mv.userData || {};
                        mv.userData.trail = mvTrail;
                        lane.movingVecLN2 = mv;
                        lane.normAnimationLN2 = new PrismLayerNormAnimation(mv);

                        lane.ln2Phase = 'right';
                    }
                    break;
                }
                
                case 'right': {
                    // Move horizontally to LN2
                    const mv = lane.movingVecLN2;
                    if (!mv) break;
                    
                    mv.group.visible = true;
                    const dx = ANIM_HORIZ_SPEED * speedMult * dt;
                    mv.group.position.x = Math.min(BRANCH_X, mv.group.position.x + dx);
                    
                    if (mv.group.position.x >= BRANCH_X - 0.01) {
                        mv.group.position.x = BRANCH_X;
                        if (lane.multTargetLN2 && lane.multTargetLN2.group) {
                            lane.multTargetLN2.group.visible = true;
                        }
                        lane.ln2Phase = 'insideLN';
                    }
                    
                    break;
                }
                
                case 'insideLN': {
                    // Inside LayerNorm2 - normalize and multiply
                    const mv = lane.movingVecLN2;
                    if (!mv) break;
                    
                    // Use the same threshold formula as LayerNorm-1 so the
                    // normalisation animation begins at the identical
                    // relative height inside the ring (centre − 15 % of the
                    // LayerNorm’s height).
                    const normStartY2 = midY_ln2_abs - LN_PARAMS.height * 0.15;
                    const normAnimating2 = lane.normStartedLN2 && lane.normAnimationLN2 && lane.normAnimationLN2.isAnimating;
                    if (!lane.normStartedLN2 && mv.group.position.y >= normStartY2) {
                        lane.normAnimationLN2.start(mv.rawData.slice());
                        lane.normStartedLN2 = true;
                    }
                    if (lane.normStartedLN2 && lane.normAnimationLN2) {
                        lane.normAnimationLN2.update(dt);
                    }
                    if (!lane.multDoneLN2 && !normAnimating2) {
                        mv.group.position.y += ANIM_RISE_SPEED_INSIDE_LN * speedMult * dt;
                    }
                    // --- NEW POST-MOVE SAFETY CHECK --------------------------------
                    if (!lane.normStartedLN2 && mv.group.position.y >= normStartY2) {
                        lane.normAnimationLN2.start(mv.rawData.slice());
                        lane.normStartedLN2 = true;
                    }
                    // ----------------------------------------------------------------
                    
                    // Trigger multiplication at center of LN2
                    if (!lane.multDoneLN2 && mv.group.position.y >= midY_ln2_abs) {
                        lane.multDoneLN2 = true;
                        if (lane.multTargetLN2) {
                            simplePrismMultiply(mv, lane.multTargetLN2, () => {
                                mv.group.visible = false;
                                if (lane.multTargetLN2 && lane.multTargetLN2.group) {
                                    lane.multTargetLN2.group.visible = false;
                                }
                                
                                // Create result vector
                                const sourceRaw = lane.multTargetLN2 ? lane.multTargetLN2.rawData.slice() : mv.rawData.slice();
                                const sourcePos = lane.multTargetLN2 && lane.multTargetLN2.group ? lane.multTargetLN2.group.position.clone() : mv.group.position.clone();
                                const resVec = new VectorVisualizationInstancedPrism(sourceRaw, sourcePos);
                                this.root.add(resVec.group);
                                // ---- Trail for LN2 result vector ----
                                const resTrail = new StraightLineTrail(this.root, 0xffffff, 1);
                                resTrail.start(resVec.group.position);
                                resVec.userData = resVec.userData || {};
                                resVec.userData.trail = resTrail;
                                lane.resultVecLN2 = resVec;
                                
                                // Rise to MLP
                                const destY = this.mlpUp.group.position.y - MLP_MATRIX_PARAMS_UP.height / 2 - 10;
                                const dist = destY - resVec.group.position.y;
                                const dur = (dist / (ANIM_RISE_SPEED_INSIDE_LN * speedMult)) * 1000;
                                
                                if (typeof TWEEN !== 'undefined') {
                                    new TWEEN.Tween(resVec.group.position)
                                        .to({ y: destY }, dur)
                                        .easing(TWEEN.Easing.Linear.None)
                                        .onUpdate(() => {
                                        })
                                        .onComplete(() => {
                                            lane.ln2Phase = 'mlpReady';
                                        })
                                        .start();
                                }
                            });
                        }
                    }
                    

                    break;
                }
                
                case 'mlpReady':
                    // Ready for MLP animation – begin only once every lane
                    // has reached this state to maintain synchronisation.
                    if (!this._mlpStart) {
                        break; // hold until global start flag set
                    }

                    if (!lane.mlpUpStarted && lane.resultVecLN2) {
                        lane.mlpUpStarted = true;
                        this._animateMlpUpProjection(lane);
                    }
                    break;
                    
                case 'done':
                default:
                    break;
            }
        });

        // Update the MHSA controller so vectors travel to attention heads
        if (this.mhsaAnimation) {
            this.mhsaAnimation.update(dt, performance.now(), this.lanes);
        }

        // ----------------------------------------------------------
        // Notify LayerPipeline once **all** lanes have finished AND all additions complete
        // ----------------------------------------------------------
        if (this.isActive && !this._completed && this.lanes.length && 
            this.lanes.every(l => l.ln2Phase === 'done') && this._pendingAdditions === 0) {
            this._completed = true;
            this._makeLayerOpaque(); // disable expensive transparency; dynamic vectors culled later by pipeline
            if (this.onFinished) this.onFinished();
        }
    }

    /**
     * Animate vector through MLP up-projection (768 → 3072 dimensions)
     */
    _animateMlpUpProjection(lane) {
        const vec = lane.resultVecLN2;
        if (!vec || typeof TWEEN === 'undefined') return;

        const bottomY = this.mlpUp.group.position.y - MLP_MATRIX_PARAMS_UP.height / 2;
        const topY = this.mlpUp.group.position.y + MLP_MATRIX_PARAMS_UP.height / 2;
        const distance = topY - vec.group.position.y;
        const duration = (distance / (ANIM_RISE_SPEED_INSIDE_LN * GLOBAL_ANIM_SPEED_MULT)) * 1000;
        
        const matrixStartColor = new THREE.Color(INACTIVE_COMPONENT_COLOR);
        const matrixEndColor = new THREE.Color(0xb07c13); // orange
        
        // Animate matrix color
        new TWEEN.Tween({ t: 0 })
            .to({ t: 1 }, duration)
            .easing(TWEEN.Easing.Quadratic.InOut)
            .onUpdate(o => {
                const col = matrixStartColor.clone().lerp(matrixEndColor, o.t);
                this.mlpUp.setColor(col);
                this.mlpUp.setEmissive(col, 0.5);
            })
            .start();
            
        // Move vector through matrix
        new TWEEN.Tween(vec.group.position)
            .to({ y: topY }, duration)
            .easing(TWEEN.Easing.Linear.None)
            .onUpdate(() => {
            })
            .onStart(() => {
                // Shrink to fit in narrowing matrix
                vec.group.scale.setScalar(0.6);
            })
            .onComplete(() => {
                // Restore scale
                vec.group.scale.setScalar(0.6);
                this.mlpUp.setColor(matrixEndColor);
                this.mlpUp.setEmissive(matrixEndColor, 0.5);
                
                // Expand to 4x width (3072 dimensions)
                this._expandTo4x(lane, vec);
            })
            .start();
    }

    /**
     * Expand vector to 4x width representing 3072 dimensions
     */
    _expandTo4x(lane, vec) {
        const segments = 4;
        const segWidth = vec.getBaseWidthConstant() * vec.getWidthScale() * VECTOR_LENGTH_PRISM;
        const expandedGroup = new THREE.Group();
        const segmentVecs = [];

        // Create 4 segments side by side
        for (let s = 0; s < segments; s++) {
            const segVec = new VectorVisualizationInstancedPrism(vec.rawData.slice(), new THREE.Vector3());
            
            // Copy color gradient
            if (Array.isArray(vec.currentKeyColors) && vec.currentKeyColors.length) {
                segVec.currentKeyColors = vec.currentKeyColors.map(c => c.clone());
                segVec.updateInstanceGeometryAndColors();
            }
            
            // Position segments horizontally
            const localX = (s - (segments - 1) / 2) * segWidth;
            segVec.group.position.set(localX, 0, 0);
            expandedGroup.add(segVec.group);
            segmentVecs.push(segVec);
        }

        expandedGroup.position.copy(vec.group.position);
        this.root.add(expandedGroup);
        vec.group.visible = false;

        // ---- Trail for expanded 4x vector group ----
        const expTrail = new StraightLineTrail(this.root, 0xffffff, 1);
        expTrail.start(expandedGroup.position);
        lane.expandedVecTrail = expTrail;
        
        lane.expandedVecGroup = expandedGroup;
        lane.expandedVecSegments = segmentVecs;
        
        // Rise and pause before down-projection
        const extraRise = 20;
        const pauseMs = 0;
        
        new TWEEN.Tween(expandedGroup.position)
            .to({ y: expandedGroup.position.y + extraRise }, 500)
            .easing(TWEEN.Easing.Quadratic.InOut)
            .onUpdate(() => {
            })
            .onComplete(() => {
                this._animateMlpDownProjection(lane);
            })
            .start();
    }

    /**
     * Animate through MLP down-projection (3072 → 768 dimensions)
     */
    _animateMlpDownProjection(lane) {
        const expandedGroup = lane.expandedVecGroup;
        if (!expandedGroup || typeof TWEEN === 'undefined') return;
        
        const orangeColor = new THREE.Color(0xb07c13);
        const downBottomY = this.mlpDown.group.position.y - MLP_MATRIX_PARAMS_DOWN.height / 2;
        const downTopY = this.mlpDown.group.position.y + MLP_MATRIX_PARAMS_DOWN.height / 2;
        
        const startY = expandedGroup.position.y;
        const totalDist = downTopY - startY;
        const durationDown = (Math.abs(totalDist) / (ANIM_RISE_SPEED_INSIDE_LN * GLOBAL_ANIM_SPEED_MULT)) * 1000;
        
        // Matrix color animation
        new TWEEN.Tween({ t: 0 })
            .to({ t: 1 }, durationDown)
            .easing(TWEEN.Easing.Quadratic.InOut)
            .onUpdate(o => {
                const col = new THREE.Color(INACTIVE_COMPONENT_COLOR).lerp(orangeColor, o.t);
                this.mlpDown.setColor(col);
                this.mlpDown.setEmissive(col, 0.5);
            })
            .start();
            
        // Move expanded vector through matrix
        new TWEEN.Tween(expandedGroup.position)
            .to({ y: downTopY }, durationDown)
            .easing(TWEEN.Easing.Linear.None)
            .onUpdate(() => {

                // Once the vector has fully entered the down-projection matrix, shrink its overall width
                if (!lane.shrunkInsideDown && expandedGroup.position.y >= downBottomY) {
                    lane.shrunkInsideDown = true;
                    if (typeof TWEEN !== 'undefined') {
                        new TWEEN.Tween(expandedGroup.scale)
                            .to({ x:0.25, y:0.25, z:0.25 }, 300)
                            .easing(TWEEN.Easing.Quadratic.InOut)
                            .start();
                    } else {
                        expandedGroup.scale.setScalar(0.25);
                    }
                }

                // Check if we've reached the middle of the down-projection matrix
                const midY = this.mlpDown.group.position.y;
                if (!lane.collapsedInMatrix && expandedGroup.position.y >= midY) {
                    lane.collapsedInMatrix = true;
                    
                    // Create collapsed vector at current position
                    const collapseVec = new VectorVisualizationInstancedPrism(
                        lane.expandedVecSegments[0].rawData.slice(), 
                        expandedGroup.position.clone()
                    );
                    // Do not start a local trail yet; we'll create a clean path trail
                    // when rising above the MLP to avoid zig-zag artifacts.
                    
                    // Copy gradient colors
                    if (Array.isArray(lane.expandedVecSegments[0].currentKeyColors) && lane.expandedVecSegments[0].currentKeyColors.length) {
                        collapseVec.currentKeyColors = lane.expandedVecSegments[0].currentKeyColors.map(c => c.clone());
                        collapseVec.updateInstanceGeometryAndColors();
                    }
                    
                    this.root.add(collapseVec.group);
                    expandedGroup.visible = false;
                    
                    lane.finalVecAfterMlp = collapseVec;
                    
                    // Continue animating the collapsed vector for the rest of the journey
                    new TWEEN.Tween(collapseVec.group.position)
                        .to({ y: downTopY }, durationDown / 2) // Half duration since we're halfway
                        .easing(TWEEN.Easing.Linear.None)
                        .onUpdate(() => {
                        })
                        .start();
                }
            })
            .onComplete(() => {
                this.mlpDown.setColor(orangeColor);
                this.mlpDown.setEmissive(orangeColor, 0.5);
                
                // Ensure both MLP matrices are fully opaque at the end
                this.mlpUp.setMaterialProperties({ opacity: 1.0, transparent: false });
                this.mlpDown.setMaterialProperties({ opacity: 1.0, transparent: false });
                
                // If we haven't collapsed yet (shouldn't happen), do it now
                if (!lane.collapsedInMatrix) {
                    this._collapseToSingle(lane);
                } else {
                    // Continue with the rise animation
                    this._riseAfterMlp(lane);
                }
            })
            .start();
    }

    /**
     * Collapse expanded vector back to single 768-dim vector
     */
    _collapseToSingle(lane) {
        const expandedGroup = lane.expandedVecGroup;
        const segmentVecs = lane.expandedVecSegments;
        if (!expandedGroup || !segmentVecs || segmentVecs.length === 0) return;
        
        // Create collapsed vector
        const collapseVec = new VectorVisualizationInstancedPrism(
            segmentVecs[0].rawData.slice(), 
            expandedGroup.position.clone()
        );
        // Defer trail creation until after the rise above the MLP.
        
        // Copy gradient colors
        if (Array.isArray(segmentVecs[0].currentKeyColors) && segmentVecs[0].currentKeyColors.length) {
            collapseVec.currentKeyColors = segmentVecs[0].currentKeyColors.map(c => c.clone());
            collapseVec.updateInstanceGeometryAndColors();
        }
        
        this.root.add(collapseVec.group);
        expandedGroup.visible = false;
        
        lane.finalVecAfterMlp = collapseVec;
        
        this._riseAfterMlp(lane);
    }
    
    /**
     * Rise above matrix after MLP processing
     */
    _riseAfterMlp(lane) {
        const vec = lane.finalVecAfterMlp;
        if (!vec) return;
        
        // Rise above matrix
        const riseAbove = 40;
        const riseDur = (riseAbove / (ANIM_RISE_SPEED_INSIDE_LN * GLOBAL_ANIM_SPEED_MULT)) * 1000;
        
        new TWEEN.Tween(vec.group.position)
            .to({ y: vec.group.position.y + riseAbove }, riseDur)
            .easing(TWEEN.Easing.Quadratic.InOut)
            .onStart(() => {
                // Start a dedicated post-MLP path trail for a clean rise-then-right
                try {
                    vec.userData = vec.userData || {};
                    if (!vec.userData.mlpTrail) {
                        const pathTrail = new StraightLineTrail(this.root, 0xffffff, 1);
                        pathTrail.start(vec.group.position.clone());
                        vec.userData.mlpTrail = pathTrail;
                    }
                } catch (_) { /* optional visual */ }
            })
            .onUpdate(() => {
                const t = vec && vec.userData && vec.userData.mlpTrail;
                if (t && typeof t.update === 'function') {
                    t.update(vec.group.position);
                }
            })
            .onComplete(() => {
                // Update residual stream target height
                if (this.mhsaAnimation) {
                    this.mhsaAnimation.finalOriginalY = vec.group.position.y - ORIGINAL_TO_PROCESSED_GAP;
                }
                
                // Move back to residual stream
                this._returnToResidualStream(lane, vec);
            })
            .start();
    }

    /**
     * Move processed vector back to residual stream for final addition
     */
    _returnToResidualStream(lane, vec) {
        const horizDist = Math.abs(vec.group.position.x);
        const horizDur = (horizDist / (ANIM_HORIZ_SPEED * GLOBAL_ANIM_SPEED_MULT)) * 1000;
        // Lock Y to eliminate tiny easing jitter at the corner so the trail
        // forms a clean right-angle when switching from vertical to horizontal.
        const lockedY = vec.group.position.y;

        new TWEEN.Tween(vec.group.position)
            .to({ x: 0, y: lockedY }, horizDur)
            .easing(TWEEN.Easing.Quadratic.InOut)
            .onStart(() => {
                vec.group.position.y = lockedY;
            })
            .onUpdate(() => {
                // Ensure Y stays perfectly constant during horizontal travel
                // to avoid any up-then-down wobble in the trail.
                vec.group.position.y = lockedY;
                // Continue updating the post-MLP path trail during horizontal move
                try {
                    const t = vec && vec.userData && vec.userData.mlpTrail;
                    if (t && typeof t.update === 'function') t.update(vec.group.position);
                } catch (_) { /* optional */ }
            })
            .onComplete(() => {
                // Freeze the post-MLP path trail into static segments
                try {
                    const t = vec && vec.userData && vec.userData.mlpTrail;
                    if (t) {
                        mergeTrailsIntoLineSegments([t], this.root);
                        if (vec.userData) delete vec.userData.mlpTrail;
                    }
                } catch (_) { /* optional */ }
                // Prevent double-drawing along the residual stream: retire the local collapse
                // trail (attached to this.root) and switch this vector to the world-space
                // residual trail so only one trail continues at x=0.
                try {
                    // 1) Freeze the local collapse trail into a static segments object
                    const localTrail = vec && vec.userData && vec.userData.trail;
                    if (localTrail && localTrail._scene === this.root) {
                        const colorHex = (localTrail._material && localTrail._material.color)
                            ? localTrail._material.color.getHex() : undefined;
                        // Preserve original trail appearance (opacity/linewidth) when freezing
                        const frozenOpacity = (typeof localTrail._opacity === 'number') ? localTrail._opacity : undefined;
                        mergeTrailsIntoLineSegments([localTrail], this.root, colorHex, undefined, frozenOpacity);
                        if (vec.userData) delete vec.userData.trail;
                    }

                    // 2) Reassign the world-space residual trail to the collapsed vector
                    const worldTrail = (lane && lane.originalTrail)
                        || (lane && lane.originalVec && lane.originalVec.userData && lane.originalVec.userData.trail);
                    if (worldTrail) {
                        if (lane && lane.originalVec && lane.originalVec.userData) {
                            delete lane.originalVec.userData.trail;
                            delete lane.originalVec.userData.trailWorld;
                        }
                        vec.userData = vec.userData || {};
                        vec.userData.trail = worldTrail;
                        vec.userData.trailWorld = true;
                    }
                } catch (_) { /* no-op */ }

                // Perform final addition with original vector
                if (this.mhsaAnimation && lane.originalVec) {
                    // Track this addition animation
                    this._pendingAdditions++;
                    
                    // Trigger the final addition animation (originalVec ➔ vec)
                    // Prisms should rise from the lower original vector up into the processed one.
                    this.mhsaAnimation._startAdditionAnimation(lane.originalVec, vec, lane);
                    
                    // Set up completion callback for when addition finishes
                    this._scheduleAdditionCompletion(lane);
                }
                lane.ln2Phase = 'done';
            })
            .start();
    }

    // ------------------------------------------------------------
    // Public helpers
    // ------------------------------------------------------------

    /** Inject external lanes from the previous layer and activate animation */
    activateWithLanes(externalLanes) {
        if (this.isActive) return; // already active
        
        // Set up lanes but don't start animation yet - wait for positioning
        this.isActive = false; // keep inactive during transition
        this._transitionPhase = 'positioning';
        this._createLanesFromExternal(externalLanes, BRANCH_X, LAYER_NORM_1_Y_POS, LAYER_NORM_2_Y_POS, LAYER_NORM_1_Y_POS + LN_PARAMS.height/2);

        // MHSAAnimation's constructor already sets the correct initial rise speed.
        // No override is needed here.
    }

    /** Replace/assign onFinished callback after construction */
    setOnFinished(cb) { this.onFinished = typeof cb === 'function' ? cb : null; }

    // ------------------------------------------------------------
    // Internal lane creation helpers (extracted from init)
    // ------------------------------------------------------------

    _createFreshLanes(offsetX, ln1CenterY, ln2CenterY, ln1TopY) {
        const slitSpacing = LN_PARAMS.depth / (NUM_VECTOR_LANES + 1);
        const startY = LAYER_NORM_1_Y_POS - LN_PARAMS.height / 2 - ANIM_OFFSET_Y_ORIGINAL_SPAWN;
        const meetY  = ln1TopY + 5;
        for (let laneIdx = 0; laneIdx < NUM_VECTOR_LANES; laneIdx++) {
            this._buildSingleLane(null, offsetX, ln1CenterY, ln2CenterY, startY, meetY, laneIdx, slitSpacing);
        }
    }

    _createLanesFromExternal(externalLanes, offsetX, ln1CenterY, ln2CenterY, ln1TopY) {
        const meetY = ln1TopY + 5; // where original vectors pause just above LN1

        // DON'T reset position - let vectors continue from where they are after layer 1
        externalLanes.forEach((oldLane, laneIdx) => {
            this._buildSingleLane(oldLane, offsetX, ln1CenterY, ln2CenterY, null, meetY, laneIdx, null);
        });
    }

    _buildSingleLane(oldLane, offsetX, ln1CenterY, ln2CenterY, startY_override, meetY, laneIdx, slitSpacing) {
        // Reuse existing trail when lanes are passed from a lower layer
        let trailFromPrev = oldLane && oldLane.originalTrail ? oldLane.originalTrail : null;
        let originalVec, zPos, startY, trail; // trail will be reused or created anew
        if (oldLane && oldLane.originalVec) {
            originalVec = oldLane.originalVec;
            this.root.attach(originalVec.group);
            zPos   = originalVec.group.position.z;
            startY = originalVec.group.position.y; // Keep current position
            // Prefer to carry over the existing residual-stream trail so it
            // remains a single continuous line across layer boundaries.
            if (trailFromPrev) {
                trail = trailFromPrev;
                if (typeof trail.reparent === 'function') {
                    trail.reparent(this._globalScene);
                }
                originalVec.userData = originalVec.userData || {};
                originalVec.userData.trail = trail;
                originalVec.userData.trailWorld = true;
            }
        } else {
            zPos = -LN_PARAMS.depth / 2 + slitSpacing * (laneIdx + 1);
            const data = this.random.nextVector(VECTOR_LENGTH_PRISM);
            startY = startY_override;
            originalVec = new VectorVisualizationInstancedPrism(data, new THREE.Vector3(0, startY, zPos));
            this.root.add(originalVec.group);

        // ────────────── Trail for the ORIGINAL vector ──────────────
        // Attach to the GLOBAL scene and record WORLD positions so the trail
        // remains continuous across layers as lanes are transferred upwards.
        trail = new StraightLineTrail(this._globalScene, 0xffffff, 1);
        {
            const wp = new THREE.Vector3();
            originalVec.group.getWorldPosition(wp);
            trail.start(wp);
        }
        originalVec.userData = originalVec.userData || {};
        originalVec.userData.trail = trail;
        originalVec.userData.trailWorld = true; // mark as world-space trail

        }

        // Spawn the LN-1 duplicate at the staging height (bottom + 5) so that
        // when it becomes visible and starts the 'right' phase it travels purely
        // horizontally, matching LN-2 behaviour.
        const dupStartPos = new THREE.Vector3(
            originalVec.group.position.x,
            ln1CenterY - LN_PARAMS.height / 2 + 5,
            originalVec.group.position.z
        );
        const dupVec = new VectorVisualizationInstancedPrism(originalVec.rawData.slice(), dupStartPos);
        dupVec.group.visible = false;
        this.root.add(dupVec.group);
        // Trail for duplicate vector inside LN1
        const dupTrail = new StraightLineTrail(this.root, 0xffffff, 1);
        dupTrail.start(dupVec.group.position);
        dupVec.userData = dupVec.userData || {};
        dupVec.userData.trail = dupTrail;
        const normAnim = new PrismLayerNormAnimation(dupVec);

        // If we're reusing an existing lane we may not have created the trail yet
        if (!trail) {
            trail = new StraightLineTrail(this._globalScene, 0xffffff, 1);
            const wp = new THREE.Vector3();
            originalVec.group.getWorldPosition(wp);
            trail.start(wp);
        }

        const multTarget = new VectorVisualizationInstancedPrism(originalVec.rawData.slice(), new THREE.Vector3(offsetX, ln1CenterY + 3.3, zPos));
        this.root.add(multTarget.group);
        multTarget.group.visible = false;

        const multTargetLN2 = new VectorVisualizationInstancedPrism(originalVec.rawData.slice(), new THREE.Vector3(offsetX, ln2CenterY + 3.3, zPos));
        this.root.add(multTargetLN2.group);
        multTargetLN2.group.visible = false;

        // Fallback to previous trail if a new one wasn't created in this constructor
        if (!trail && trailFromPrev) trail = trailFromPrev;

        // Ensure originalVec always has a trail reference
        originalVec.userData = originalVec.userData || {};
        if (!originalVec.userData.trail) {
            originalVec.userData.trail = trail;
        }
        // Always mark world-space trail semantics so updates use world coords
        originalVec.userData.trailWorld = true;

        this.lanes.push({
            originalVec,
            originalTrail: trail,
            dupVec,
            multTarget,
            multTargetLN2,
            normAnim,
            horizPhase: 'waiting',
            branchStartY: ln1CenterY - LN_PARAMS.height / 2 + 5,
            ln1MidY: ln1CenterY,
            normStarted:false,
            multStarted:false,
            resultVec:null,
            targetY: meetY,
            travellingVec: null,
            upwardCopies: [],
            sideCopies: [],
            headIndex: 0,
            finalAscend: false,
            ln2Phase: 'notStarted',
            postAdditionVec: null,
            movingVecLN2: null,
            normAnimationLN2: null,
            normStartedLN2: false,
            multDoneLN2: false,
            resultVecLN2: null,
            mlpUpStarted: false,
            expandedVecGroup: null,
            expandedVecSegments: null,
            finalVecAfterMlp: null,
            expandedVecTrail: null,
            zPos
        });
    }

    /**
     * Schedule callback for when addition animation completes
     */
    _scheduleAdditionCompletion(lane) {
        // Mirror timings from additionUtils to ensure consistent completion detection
        const duration      = PRISM_ADD_ANIM_BASE_DURATION             / PRISM_ADD_ANIM_SPEED_MULT;
        const flashDuration = PRISM_ADD_ANIM_BASE_FLASH_DURATION       / PRISM_ADD_ANIM_SPEED_MULT;
        const delayBetween  = PRISM_ADD_ANIM_BASE_DELAY_BETWEEN_PRISMS / PRISM_ADD_ANIM_SPEED_MULT;
        const vectorLength  = VECTOR_LENGTH_PRISM;
        const totalAnimTime = duration + flashDuration + vectorLength * delayBetween;

        setTimeout(() => {
            this._pendingAdditions--;
            lane.additionComplete = true;
        }, totalAnimTime + 100);
    }

    /**
     * Walk every mesh in this layer and turn off Three.js transparency for
     * materials that are already fully opaque.  This removes them from the
     * per-frame depth-sorting list, improving performance once the layer is
     * finished and static.
     */
    _makeLayerOpaque() {
        this.root.traverse(obj => {
            if (obj.isMesh && obj.material) {
                const mats = Array.isArray(obj.material) ? obj.material : [obj.material];
                mats.forEach(mat => {
                    // Only change materials that use blending but are visually opaque
                    const fullyOpaque = (mat.opacity === undefined) || (mat.opacity >= 0.99);
                    if (mat.transparent && fullyOpaque) {
                        mat.transparent = false;
                        mat.depthWrite = true;
                        mat.needsUpdate = true;
                    }
                });
            }
        });

        // After making meshes opaque, fuse the numerous per-lane residual
        // trails into a single static LineSegments and trim the live trails so
        // they can continue seamlessly in higher layers with minimal cost.
        try {
            // 1) Freeze residual trails completed so far into a static merged line
            if (Array.isArray(this.lanes) && this.lanes.length) {
                const segmentsList = [];
                const origTrailSet = new Set();
                let residualColor = null;
                this.lanes.forEach(l => {
                    // Prefer the dedicated world-space residual trail reference if available
                    const t = (l && l.originalTrail)
                        || (l && l.originalVec && l.originalVec.userData && l.originalVec.userData.trail);
                    if (t) {
                        origTrailSet.add(t);
                        // Capture the color/opacity from the first trail so the merged line matches brightness
                        if (residualColor === null && t._material && t._material.color) {
                            residualColor = t._material.color.getHex();
                        } else if (residualColor === null && typeof t._color !== 'undefined') {
                            residualColor = t._color;
                        }
                    }
                    if (t && typeof t.extractSegmentsAndTrim === 'function') {
                        const seg = t.extractSegmentsAndTrim();
                        if (seg && seg.length) segmentsList.push(seg);
                    }
                });
                if (segmentsList.length) {
                    // Preserve brightness by using the same color as the live trails (default to white if unknown)
                    const colorToUse = residualColor != null ? residualColor : 0xffffff;
                    // Use the same effective opacity as the live trails to avoid brightening
                    let residualOpacity = null;
                    if (origTrailSet.size) {
                        const first = origTrailSet.values().next().value;
                        if (first && typeof first._opacity === 'number') residualOpacity = first._opacity;
                    }
                    buildMergedLineSegmentsFromSegments(
                        segmentsList,
                        this._globalScene || this.root,
                        colorToUse,
                        undefined,
                        residualOpacity != null ? residualOpacity : undefined
                    );
                }

                // 2) Merge all other per-layer (non-residual) trails under this.root
                const allLayerTrails = collectTrailsUnder(this.root);
                const otherTrails = allLayerTrails.filter(t => !origTrailSet.has(t));
                if (otherTrails.length) {
                    // Try to preserve their appearance by copying the first trail's color
                    let otherColor = null;
                    const firstTrail = otherTrails[0];
                    if (firstTrail && firstTrail._material && firstTrail._material.color) {
                        otherColor = firstTrail._material.color.getHex();
                    } else if (firstTrail && typeof firstTrail._color !== 'undefined') {
                        otherColor = firstTrail._color;
                    }
                    mergeTrailsIntoLineSegments(otherTrails, this.root, otherColor != null ? otherColor : undefined);
                }
            }
        } catch (e) {
            // Best-effort optimisation; ignore failures to avoid breaking the demo.
        }
    }

    /**
     * Remove heavy dynamic geometry (vectors) after the layer has
     * handed its residual stream to the next layer.  This is more aggressive
     * than the previous implementation: we now detach the objects from the
     * scene graph and dispose of their GPU resources so they no longer add
     * draw-calls, memory pressure or ray-casting traversal cost.
     */
    hideDynamicGeometry() {
        const disposeObj = (obj) => {
            if (!obj) return;

            // Dispose materials
            if (obj.material) {
                const materials = Array.isArray(obj.material) ? obj.material : [obj.material];
                materials.forEach(mat => mat && mat.dispose && mat.dispose());
            }

            // Dispose geometry
            if (obj.geometry) {
                obj.geometry.dispose();
            }

            // Recursively dispose children
            if (obj.children && obj.children.length) {
                [...obj.children].forEach(child => disposeObj(child));
            }

            // Finally, detach from parent to remove from scene traversal
            if (obj.parent) {
                obj.parent.remove(obj);
            }
        };

        // We walk a *copy* of the children array because we'll be mutating
        // the hierarchy while iterating.
        [...this.root.children].forEach(obj => {
            if (!obj) return;

            const label = obj.userData && obj.userData.label;
            const isVector = label === 'Vector' || label === 'Vector24';

            if (isVector) {
                disposeObj(obj);
            }
        });
    }
} 