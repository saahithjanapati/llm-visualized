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
import { easeInOutSine, easeOutBack, easeOutCubic, clamp } from '../../utils/animationCurves.js';

// Add configurable opacity for trails specific to vector copies under Q/K/V to reduce visual prominence
const FAINT_TRAIL_OPACITY = 0.13; // must be < default 0.1


// Read live binding each use to reflect UI changes at runtime

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

                const headIdx = lane.headIndex || 0;
                if (headIdx >= this.headsCentersX.length) {
                    tVec.group.visible = false;
                    lane.horizPhase = 'finishedHeads';
                    return;
                }

                const targetX = this.headsCentersX[headIdx];
                const speedScale = Math.max(GLOBAL_ANIM_SPEED_MULT, 1e-3);

                if (!lane.travelMotion || lane.travelMotion.headIndex !== headIdx) {
                    const startX = tVec.group.position.x;
                    const baseY = tVec.group.position.y;
                    const distanceX = Math.abs(targetX - startX);
                    lane.travelMotion = {
                        headIndex,
                        phase: 'anticipate',
                        elapsed: 0,
                        startX,
                        baseY,
                        targetX,
                        launchStartX: startX,
                        backDistance: Math.min(180, distanceX * 0.2 + 30),
                        arcHeight: 110 + headIdx * 8,
                        baseAnticipation: 0.16,
                        baseTravel: Math.max(0.25, distanceX / Math.max(ANIM_HORIZ_SPEED, 1e-3)) * 1.35,
                        completed: false,
                    };
                    if (tVec.group.scale) tVec.group.scale.set(1, 1, 1);
                }

                const motion = lane.travelMotion;
                const anticipationDuration = motion.baseAnticipation / speedScale;
                const travelDuration = motion.baseTravel / speedScale;

                if (motion.phase === 'anticipate') {
                    motion.elapsed += deltaTime;
                    const t = anticipationDuration > 0 ? clamp(motion.elapsed / anticipationDuration, 0, 1) : 1;
                    const eased = easeInOutSine(t);
                    const drop = Math.sin(t * Math.PI) * 40;
                    tVec.group.position.x = motion.startX - motion.backDistance * eased;
                    tVec.group.position.y = motion.baseY - drop;
                    const squash = eased * 0.2;
                    tVec.group.scale.set(1 + squash, 1 - squash, 1 + squash);
                    if (tVec.userData && tVec.userData.trail) tVec.userData.trail.update(tVec.group.position);
                    if (t >= 1) {
                        motion.phase = 'launch';
                        motion.elapsed = 0;
                        motion.launchStartX = tVec.group.position.x;
                    }
                } else if (!motion.completed) {
                    motion.elapsed += deltaTime;
                    const t = travelDuration > 0 ? clamp(motion.elapsed / travelDuration, 0, 1) : 1;
                    const eased = easeOutBack(t, 1.35);
                    const arc = Math.sin(t * Math.PI) * motion.arcHeight;
                    const currentX = THREE.MathUtils.lerp(motion.launchStartX, motion.targetX, eased);
                    tVec.group.position.x = currentX;
                    tVec.group.position.y = motion.baseY + arc;
                    const stretch = Math.sin(t * Math.PI) * 0.18;
                    tVec.group.scale.set(1 - stretch * 0.4, 1 + stretch, 1 - stretch * 0.4);
                    if (tVec.userData && tVec.userData.trail) tVec.userData.trail.update(tVec.group.position);

                    if (t >= 1) {
                        motion.completed = true;
                        tVec.group.position.x = motion.targetX;
                        tVec.group.position.y = motion.baseY;
                        tVec.group.scale.set(1, 1, 1);

                        const upVec = new VectorVisualizationInstancedPrism([...tVec.rawData], tVec.group.position.clone());
                        this.parentGroup.add(upVec.group);
                        const upTrail = new StraightLineTrail(this.parentGroup, TRAIL_COLOR, 1, undefined, FAINT_TRAIL_OPACITY);
                        upTrail.start(upVec.group.position);
                        upVec.userData = upVec.userData || {};
                        upVec.userData.trail = upTrail;
                        Object.assign(upVec.userData, {
                            headIndex: headIdx,
                            sideSpawned: false,
                            sideSpawnRequested: false,
                            sideSpawnTime: 0,
                            parentLane: lane,
                            riseMeta: {
                                startY: upVec.group.position.y,
                                elapsed: 0,
                                baseDuration: Math.max(
                                    0.35,
                                    Math.abs(this.headStopY - upVec.group.position.y) / Math.max(MHSA_DUPLICATE_VECTOR_RISE_SPEED, 1e-3)
                                ),
                            },
                        });
                        try {
                            const lbl = `Key Vector (Green)`;
                            upVec.group.userData.label = lbl;
                            if (upVec.mesh) upVec.mesh.userData = { ...(upVec.mesh.userData || {}), label: lbl };
                        } catch (_) {}
                        lane.upwardCopies[headIdx] = upVec;

                        lane.headIndex = headIdx + 1;
                        lane.travelMotion = null;
                        if (lane.headIndex >= NUM_HEAD_SETS_LAYER) {
                            tVec.group.visible = false;
                            lane.horizPhase = 'finishedHeads';
                        }
                    }
                }
            }

            // ------------------------------------------------------------------
            // 2) Vertical rise of the upward K copies
            // ------------------------------------------------------------------
            if (lane.upwardCopies && lane.upwardCopies.length) {
                lane.upwardCopies.forEach((upVec) => {
                    if (!upVec) return;
                    const meta = (upVec.userData && upVec.userData.riseMeta) || null;
                    if (!meta) {
                        if (upVec.group.position.y < this.headStopY) {
                            upVec.group.position.y = Math.min(this.headStopY, upVec.group.position.y + MHSA_DUPLICATE_VECTOR_RISE_SPEED * GLOBAL_ANIM_SPEED_MULT * deltaTime);
                        }
                    } else if (!meta.done) {
                        const duration = meta.baseDuration / Math.max(GLOBAL_ANIM_SPEED_MULT, 1e-3);
                        meta.elapsed += deltaTime;
                        const t = duration > 0 ? clamp(meta.elapsed / duration, 0, 1) : 1;
                        const eased = easeOutBack(t, 1.2);
                        upVec.group.position.y = THREE.MathUtils.lerp(meta.startY, this.headStopY, eased);
                        const bounce = Math.sin(t * Math.PI) * 0.14;
                        upVec.group.scale.set(1 - bounce * 0.45, 1 + bounce, 1 - bounce * 0.45);
                        if (t >= 1) {
                            upVec.group.position.y = this.headStopY;
                            upVec.group.scale.set(1, 1, 1);
                            meta.done = true;
                        }
                    }

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
                            qVec.group.scale.set(0.9, 1.15, 0.9);
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
                            vVec.group.scale.set(0.9, 1.15, 0.9);
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
                            lane.sideCopies.push({ vec: qVec, targetX: coord.q, type: 'Q', matrixRef: qMatrixForHead, headIndex: hIdx, arcHeight: -60, motion: null });
                            lane.sideCopies.push({ vec: vVec, targetX: coord.v, type: 'V', matrixRef: vMatrixForHead, headIndex: hIdx, arcHeight: -60, motion: null });

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
                    if (!v) return;
                    if (!obj.motion) {
                        obj.motion = {
                            startX: v.group.position.x,
                            baseY: v.group.position.y,
                            elapsed: 0,
                            baseDuration: Math.max(0.22, Math.abs(obj.targetX - v.group.position.x) / Math.max(SIDE_COPY_HORIZ_SPEED, 1e-3)) * 1.3,
                            arcHeight: obj.arcHeight ?? -60,
                        };
                    }

                    const motion = obj.motion;
                    const duration = motion.baseDuration / Math.max(GLOBAL_ANIM_SPEED_MULT, 1e-3);
                    motion.elapsed += deltaTime;
                    const t = duration > 0 ? clamp(motion.elapsed / duration, 0, 1) : 1;
                    const eased = easeOutCubic(t);
                    v.group.position.x = THREE.MathUtils.lerp(motion.startX, obj.targetX, eased);
                    const arc = Math.sin(t * Math.PI) * motion.arcHeight;
                    v.group.position.y = motion.baseY + arc;
                    const wobble = Math.sin(t * Math.PI) * 0.12;
                    v.group.scale.set(0.9 + wobble * 0.15, 1.05 + wobble * 0.35, 0.9 + wobble * 0.15);

                    if (v.userData && v.userData.trail) {
                        const ud = v.userData;
                        const matrixBottomY = this.headStopY;
                        if (v.group.position.y <= matrixBottomY) {
                            if (ud.trailWorld) {
                                const wp = new THREE.Vector3();
                                v.group.getWorldPosition(wp);
                                ud.trail.update(wp);
                            } else {
                                ud.trail.update(v.group.position);
                            }
                        }
                    }

                    if (t >= 1) {
                        v.group.position.x = obj.targetX;
                        v.group.position.y = motion.baseY;
                        v.group.scale.set(1, 1, 1);
                        obj.motion = null;
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