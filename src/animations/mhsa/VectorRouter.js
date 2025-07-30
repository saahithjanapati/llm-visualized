import { VectorVisualizationInstancedPrism } from '../../components/VectorVisualizationInstancedPrism.js';
import { createTrailLine, updateTrail } from '../../utils/trailUtils.js';
import {
    ANIM_HORIZ_SPEED,
    GLOBAL_ANIM_SPEED_MULT,
    SIDE_COPY_DELAY_MS,
    SIDE_COPY_HORIZ_SPEED,
    NUM_HEAD_SETS_LAYER,
    VECTOR_LENGTH_PRISM,
} from '../../utils/constants.js';
import {
    MHSA_DUPLICATE_VECTOR_RISE_SPEED,
    TRAIL_LINE_COLOR,
} from '../LayerAnimationConstants.js';

const SPEED_MULT = GLOBAL_ANIM_SPEED_MULT;

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
                const dx      = ANIM_HORIZ_SPEED * SPEED_MULT * deltaTime;

                if (tVec.group.position.x < targetX - 0.01) {
                    tVec.group.position.x = Math.min(targetX, tVec.group.position.x + dx);
                    if (lane.dupTrail) updateTrail(lane.dupTrail, tVec.group.position);
                } else {
                    // Arrived: spawn upward copy used for K
                    if (lane.dupTrail) updateTrail(lane.dupTrail, tVec.group.position);

                    const upVec = new VectorVisualizationInstancedPrism([...tVec.rawData], tVec.group.position.clone());
                    this.parentGroup.add(upVec.group);
                    upVec.userData = { headIndex: targetHeadIdx, sideSpawned: false, sideSpawnRequested: false, sideSpawnTime: 0 };
                    lane.upwardCopies.push(upVec);

                    const upTrail = createTrailLine(this.parentGroup, TRAIL_LINE_COLOR);
                    updateTrail(upTrail, upVec.group.position);
                    lane.upwardTrails = lane.upwardTrails || [];
                    lane.upwardTrails.push(upTrail);

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
                lane.upwardCopies.forEach((upVec, trailIdx) => {
                    if (upVec.group.position.y < this.headStopY) {
                        upVec.group.position.y = Math.min(this.headStopY, upVec.group.position.y + MHSA_DUPLICATE_VECTOR_RISE_SPEED * SPEED_MULT * deltaTime);
                        if (lane.upwardTrails && lane.upwardTrails[trailIdx]) {
                            updateTrail(lane.upwardTrails[trailIdx], upVec.group.position);
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
                        centerVec.userData.sideSpawnTime = timeNow + SIDE_COPY_DELAY_MS / SPEED_MULT;
                    }
                    if (centerVec.userData.sideSpawnRequested && !centerVec.userData.sideSpawned && timeNow >= centerVec.userData.sideSpawnTime) {
                        const hIdx  = centerVec.userData.headIndex;
                        const coord = this.headCoords[hIdx];
                        if (coord) {
                            const qVec = new VectorVisualizationInstancedPrism(centerVec.rawData.slice(), centerVec.group.position.clone());
                            const vVec = new VectorVisualizationInstancedPrism(centerVec.rawData.slice(), centerVec.group.position.clone());
                            this.parentGroup.add(qVec.group);
                            this.parentGroup.add(vVec.group);

                            lane.sideCopies = lane.sideCopies || [];
                            const qMatrixForHead = this.mhaVisualizations[hIdx * 3];
                            const vMatrixForHead = this.mhaVisualizations[hIdx * 3 + 2];
                            lane.sideCopies.push({ vec: qVec, targetX: coord.q, type: 'Q', matrixRef: qMatrixForHead, headIndex: hIdx });
                            lane.sideCopies.push({ vec: vVec, targetX: coord.v, type: 'V', matrixRef: vMatrixForHead, headIndex: hIdx });

                            const qTrail = createTrailLine(this.parentGroup, TRAIL_LINE_COLOR);
                            const vTrail = createTrailLine(this.parentGroup, TRAIL_LINE_COLOR);
                            updateTrail(qTrail, qVec.group.position);
                            updateTrail(vTrail, vVec.group.position);
                            lane.sideTrails = lane.sideTrails || [];
                            lane.sideTrails.push(qTrail, vTrail);

                            centerVec.userData.sideSpawned = true;
                        }
                    }
                });
            }

            // ------------------------------------------------------------------
            // 4) Slide the side copies horizontally toward their Q / V matrices
            // ------------------------------------------------------------------
            if (lane.sideCopies && lane.sideCopies.length) {
                lane.sideCopies.forEach((obj, trailIdx) => {
                    const v  = obj.vec;
                    const dx = SIDE_COPY_HORIZ_SPEED * SPEED_MULT * deltaTime;
                    if (Math.abs(v.group.position.x - obj.targetX) > 0.01) {
                        const dir = v.group.position.x < obj.targetX ? 1 : -1;
                        v.group.position.x += dir * dx;
                        if ((dir === 1 && v.group.position.x > obj.targetX) || (dir === -1 && v.group.position.x < obj.targetX)) v.group.position.x = obj.targetX;
                    }
                    v.group.position.y = this.headStopY;
                    if (lane.sideTrails && lane.sideTrails[trailIdx]) {
                        updateTrail(lane.sideTrails[trailIdx], v.group.position);
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