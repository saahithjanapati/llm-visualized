import * as THREE from 'three';

import { VectorVisualizationInstancedPrism } from '../../components/VectorVisualizationInstancedPrism.js';
import { StraightLineTrail } from '../../utils/trailUtils.js';
import { TRAIL_COLOR } from '../../utils/trailConstants.js';
import {
    ANIM_HORIZ_SPEED,
    GLOBAL_ANIM_SPEED_MULT,
    SIDE_COPY_DELAY_MS,
    SIDE_COPY_HORIZ_SPEED,
    NUM_HEAD_SETS_LAYER,
    VECTOR_LENGTH_PRISM,
} from '../../utils/constants.js';
// Speed-related constants centralised in utils/constants
import { MHSA_DUPLICATE_VECTOR_RISE_SPEED } from '../../utils/constants.js';
import { computeArcOffset, applySquashStretchFromProgress, resetJoyfulTransform } from '../joyfulMotion.js';

// Add configurable opacity for trails specific to vector copies under Q/K/V to reduce visual prominence
const FAINT_TRAIL_OPACITY = 0.13; // must be < default 0.1


// Read live binding each use to reflect UI changes at runtime

function ensureArcState(vector, baseYOverride = null) {
    if (!vector || !vector.group) return null;
    vector.userData = vector.userData || {};
    let state = vector.userData.__joyfulArcState;
    if (!state) {
        state = {
            startX: vector.group.position.x,
            baseY: baseYOverride ?? vector.group.position.y,
            amplitude: 14 + Math.random() * 8,
            rotationAmplitude: 0.12 + Math.random() * 0.05,
            intensity: 0.16 + Math.random() * 0.05,
        };
        vector.userData.__joyfulArcState = state;
    } else {
        if (state.startX == null) state.startX = vector.group.position.x;
        if (state.amplitude == null) state.amplitude = 14 + Math.random() * 8;
        if (state.rotationAmplitude == null) state.rotationAmplitude = 0.12 + Math.random() * 0.05;
        if (state.intensity == null) state.intensity = 0.16 + Math.random() * 0.05;
    }
    if (baseYOverride != null) state.baseY = baseYOverride;
    if (state.baseY == null) state.baseY = baseYOverride ?? vector.group.position.y;
    return state;
}

function updateHorizontalJoyfulMotion(vector, targetX, state) {
    if (!vector || !vector.group || !state) return 0;
    const startX = state.startX;
    const dist = Math.max(1e-4, Math.abs(targetX - startX));
    const travelled = Math.abs(vector.group.position.x - startX);
    const progress = THREE.MathUtils.clamp(travelled / dist, 0, 1);
    const direction = targetX >= startX ? 1 : -1;
    const offset = computeArcOffset(progress, state.amplitude);
    vector.group.position.y = state.baseY + offset;
    applySquashStretchFromProgress(vector.group, progress, {
        intensity: state.intensity,
        rotationAmplitude: state.rotationAmplitude * direction,
    });
    if (progress >= 0.999) {
        vector.group.position.y = state.baseY;
        resetJoyfulTransform(vector.group);
    }
    return progress;
}

function updateVerticalJoyfulRise(vector, targetY) {
    if (!vector || !vector.group) return 0;
    vector.userData = vector.userData || {};
    let state = vector.userData.__joyfulVerticalState;
    if (!state) {
        state = {
            startY: vector.group.position.y,
            intensity: 0.15 + Math.random() * 0.04,
            rotationAmplitude: 0.1 + Math.random() * 0.04,
        };
        vector.userData.__joyfulVerticalState = state;
    }
    const dist = Math.max(1e-4, targetY - state.startY);
    const travelled = vector.group.position.y - state.startY;
    const progress = dist <= 0 ? 1 : THREE.MathUtils.clamp(travelled / dist, 0, 1);
    applySquashStretchFromProgress(vector.group, progress, {
        intensity: state.intensity,
        rotationAmplitude: state.rotationAmplitude,
    });
    if (progress >= 0.999) {
        resetJoyfulTransform(vector.group);
    }
    return progress;
}

/**
 * Responsible only for moving vectors to their parking positions
 * under Q/K/V matrices (horizontal travel, upward copies, side copies).
 * Emits a callback once *every* K/Q/V copy is under its matrix so the
 * rest of the MHSA pipeline can begin the pass-through phase.
 */
export class VectorRouter {
    constructor(parentGroup, headsCentersX, headCoords, headStopY, mhaVisualizations) {
        this.parentGroup   = parentGroup;
        this.headsCentersX = headsCentersX;
        this.headCoords    = headCoords;
        this.headStopY     = headStopY;
        this.mhaVisualizations = mhaVisualizations;

        this._readyEmitted = false;
        this._callbacks    = new Set();
    }

    onReady(cb) {
        if (typeof cb === 'function') this._callbacks.add(cb);
    }

    /**
     * Main per-frame update.
     * @param {number} deltaTime – seconds since last frame
     * @param {number} timeNow   – absolute clock time (ms)
     * @param {Array}  lanes     – residual-stream lanes (mutable objects)
     */
    update(deltaTime, timeNow, lanes) {
        if (!lanes || !lanes.length) return;
        // Once all vectors are in their parking positions and the callback has
        // fired, this router should stop mutating side-copy positions. This
        // avoids fighting with above-matrix animations that take over later.
        if (this._readyEmitted) return;

        lanes.forEach(lane => {
            // ------------------------------------------------------------------
            // 1) Horizontal travel of the duplicated vector toward each head
            // ------------------------------------------------------------------
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
                const dx      = ANIM_HORIZ_SPEED * GLOBAL_ANIM_SPEED_MULT * deltaTime;
                const arcState = ensureArcState(tVec);

                if (arcState && arcState.startX == null) {
                    arcState.startX = tVec.group.position.x;
                }

                const direction = targetX >= tVec.group.position.x ? 1 : -1;
                if (Math.abs(tVec.group.position.x - targetX) > 0.01) {
                    tVec.group.position.x += direction * dx;
                    if ((direction === 1 && tVec.group.position.x > targetX) || (direction === -1 && tVec.group.position.x < targetX)) {
                        tVec.group.position.x = targetX;
                    }
                    if (tVec.userData && tVec.userData.trail) {
                        tVec.userData.trail.update(tVec.group.position);
                    }
                } else {
                    tVec.group.position.x = targetX;
                }

                updateHorizontalJoyfulMotion(tVec, targetX, arcState);

                if (Math.abs(tVec.group.position.x - targetX) <= 0.01) {
                    tVec.group.position.x = targetX;
                    if (arcState) arcState.startX = targetX;
                    resetJoyfulTransform(tVec.group);

                    // Arrived: spawn upward copy used for K
                    const upVec = new VectorVisualizationInstancedPrism([...tVec.rawData], tVec.group.position.clone());
                    this.parentGroup.add(upVec.group);
                    // Trail for upward K copy
                    const upTrail = new StraightLineTrail(this.parentGroup, TRAIL_COLOR, 1, undefined, FAINT_TRAIL_OPACITY);
                    upTrail.start(upVec.group.position);
                    upVec.userData = upVec.userData || {};
                    upVec.userData.trail = upTrail;
                    Object.assign(upVec.userData, { headIndex: targetHeadIdx, sideSpawned: false, sideSpawnRequested: false, sideSpawnTime: 0, parentLane: lane });
                    // Label for hover – Key vector (green)
                    try {
                        const lbl = `Key Vector (Green)`;
                        upVec.group.userData.label = lbl;
                        if (upVec.mesh) upVec.mesh.userData = { ...(upVec.mesh.userData||{}), label: lbl };
                    } catch (_) {}
                    lane.upwardCopies.push(upVec);

                    lane.headIndex = targetHeadIdx + 1;
                    if (lane.headIndex >= NUM_HEAD_SETS_LAYER) {
                        tVec.group.visible = false;
                        lane.horizPhase = 'finishedHeads';
                    }
                }
            }

            // ------------------------------------------------------------------
            // 2) Vertical rise of the upward K copies
            // ------------------------------------------------------------------
            if (lane.upwardCopies && lane.upwardCopies.length) {
                lane.upwardCopies.forEach((upVec) => {
                    if (upVec.group.position.y < this.headStopY) {
                        upVec.group.position.y = Math.min(
                            this.headStopY,
                            upVec.group.position.y + MHSA_DUPLICATE_VECTOR_RISE_SPEED * GLOBAL_ANIM_SPEED_MULT * deltaTime,
                        );
                        if (upVec.userData && upVec.userData.trail) {
                            const ud = upVec.userData;
                            if (ud.trailWorld) {
                                const wp = new THREE.Vector3();
                                upVec.group.getWorldPosition(wp);
                                ud.trail.update(wp);
                            } else {
                                ud.trail.update(upVec.group.position);
                            }
                        }
                    } else {
                        upVec.group.position.y = this.headStopY;
                    }
                    updateVerticalJoyfulRise(upVec, this.headStopY);
                });
            }

            // ------------------------------------------------------------------
            // 3) Spawn sideways Q / V copies once an upward copy is parked
            // ------------------------------------------------------------------
            if (lane.upwardCopies) {
                lane.upwardCopies.forEach(centerVec => {
                    if (!centerVec.userData.sideSpawnRequested && Math.abs(centerVec.group.position.y - this.headStopY) < 0.1) {
                        centerVec.userData.sideSpawnRequested = true;
                        centerVec.userData.sideSpawnTime = timeNow + SIDE_COPY_DELAY_MS / GLOBAL_ANIM_SPEED_MULT;
                    }
                    if (centerVec.userData.sideSpawnRequested && !centerVec.userData.sideSpawned && timeNow >= centerVec.userData.sideSpawnTime) {
                        const hIdx  = centerVec.userData.headIndex;
                        const coord = this.headCoords[hIdx];
                        if (coord) {
                            const qVec = new VectorVisualizationInstancedPrism(centerVec.rawData.slice(), centerVec.group.position.clone());
                            const qTrail = new StraightLineTrail(this.parentGroup, TRAIL_COLOR, 1, undefined, FAINT_TRAIL_OPACITY);
                            qTrail.start(qVec.group.position);
                            qVec.userData = qVec.userData || {};
                            qVec.userData.trail = qTrail;
                            Object.assign(qVec.userData, { headIndex: hIdx, parentLane: lane });
                            try {
                                const lblQ = `Query Vector (Blue)`;
                                qVec.group.userData.label = lblQ;
                                if (qVec.mesh) qVec.mesh.userData = { ...(qVec.mesh.userData||{}), label: lblQ };
                            } catch (_) {}
                            const vVec = new VectorVisualizationInstancedPrism(centerVec.rawData.slice(), centerVec.group.position.clone());
                            const vTrail = new StraightLineTrail(this.parentGroup, TRAIL_COLOR, 1, undefined, FAINT_TRAIL_OPACITY);
                            vTrail.start(vVec.group.position);
                            vVec.userData = vVec.userData || {};
                            vVec.userData.trail = vTrail;
                            Object.assign(vVec.userData, { headIndex: hIdx, parentLane: lane });
                            try {
                                const lblV = `Value Vector (Red)`;
                                vVec.group.userData.label = lblV;
                                if (vVec.mesh) vVec.mesh.userData = { ...(vVec.mesh.userData||{}), label: lblV };
                            } catch (_) {}
                            this.parentGroup.add(qVec.group);
                            this.parentGroup.add(vVec.group);

                            lane.sideCopies = lane.sideCopies || [];
                            const qMatrixForHead = this.mhaVisualizations[hIdx * 3];
                            const vMatrixForHead = this.mhaVisualizations[hIdx * 3 + 2];
                            lane.sideCopies.push({ vec: qVec, targetX: coord.q, type: 'Q', matrixRef: qMatrixForHead, headIndex: hIdx });
                            lane.sideCopies.push({ vec: vVec, targetX: coord.v, type: 'V', matrixRef: vMatrixForHead, headIndex: hIdx });

                            centerVec.userData.sideSpawned = true;
                        }
                    }
                });
            }

            // ------------------------------------------------------------------
            // 4) Slide the side copies horizontally toward their Q / V matrices
            // ------------------------------------------------------------------
            if (lane.sideCopies && lane.sideCopies.length) {
                lane.sideCopies.forEach((obj) => {
                    const v  = obj.vec;
                    if (!v || !v.group) return;
                    const dx = SIDE_COPY_HORIZ_SPEED * GLOBAL_ANIM_SPEED_MULT * deltaTime;
                    const arcState = ensureArcState(v, this.headStopY);
                    if (arcState && arcState.startX == null) {
                        arcState.startX = v.group.position.x;
                    }

                    if (Math.abs(v.group.position.x - obj.targetX) > 0.01) {
                        const dir = v.group.position.x < obj.targetX ? 1 : -1;
                        v.group.position.x += dir * dx;
                        if ((dir === 1 && v.group.position.x > obj.targetX) || (dir === -1 && v.group.position.x < obj.targetX)) {
                            v.group.position.x = obj.targetX;
                        }
                    } else {
                        v.group.position.x = obj.targetX;
                    }

                    const progress = updateHorizontalJoyfulMotion(v, obj.targetX, arcState);
                    if (progress >= 0.999 && arcState) {
                        arcState.startX = v.group.position.x;
                    }

                    if (v.userData && v.userData.trail) {
                        const ud = v.userData;
                        const matrixBottomY = this.headStopY;
                        if (v.group.position.y < matrixBottomY) {
                            if (ud.trailWorld) {
                                const wp = new THREE.Vector3();
                                v.group.getWorldPosition(wp);
                                ud.trail.update(wp);
                            } else {
                                ud.trail.update(v.group.position);
                            }
                        }
                    }
                });
            }
        });

        // After processing all lanes, check readiness once per frame.
        if (!this._readyEmitted && this._areAllVectorsInPosition(lanes)) {
            this._readyEmitted = true;
            this._callbacks.forEach(cb => cb());
        }
    }

    // ------------------------------------------------------------------
    // Internal helper – identical logic to original areAllMHAVectorsInPosition
    // ------------------------------------------------------------------
    _areAllVectorsInPosition(lanes) {
        if (!lanes || !lanes.length) return false;

        for (const lane of lanes) {
            if (!lane.upwardCopies || lane.upwardCopies.length !== NUM_HEAD_SETS_LAYER) return false;
            for (let headIdx = 0; headIdx < NUM_HEAD_SETS_LAYER; headIdx++) {
                const kVec = lane.upwardCopies[headIdx];
                if (!kVec || Math.abs(kVec.group.position.y - this.headStopY) > 0.1) return false;
                if (!kVec.userData.sideSpawned) return false;
            }
            if (!lane.sideCopies || lane.sideCopies.length !== NUM_HEAD_SETS_LAYER * 2) return false;
            for (const side of lane.sideCopies) {
                if (!side || !side.vec) return false;
                if (Math.abs(side.vec.group.position.y - this.headStopY) > 0.1) return false;
                if (Math.abs(side.vec.group.position.x - side.targetX) > 0.1) return false;
            }
        }
        return true;
    }
}