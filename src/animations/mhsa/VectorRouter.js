import * as THREE from 'three';

import { VectorVisualizationInstancedPrism } from '../../components/VectorVisualizationInstancedPrism.js';
import { StraightLineTrail } from '../../utils/trailUtils.js';
import { TRAIL_COLOR, TRAIL_MIN_SEGMENT_DISTANCE } from '../../utils/trailConstants.js';
import {
    ANIM_HORIZ_SPEED,
    GLOBAL_ANIM_SPEED_MULT,
    SIDE_COPY_DELAY_MS,
    SIDE_COPY_HORIZ_SPEED,
    NUM_HEAD_SETS_LAYER,
} from '../../utils/constants.js';
// Speed-related constants centralised in utils/constants
import { MHSA_DUPLICATE_VECTOR_RISE_SPEED } from '../../utils/constants.js';
import { setSideCopyEntry } from './laneIndex.js';

const _scratchWorld = new THREE.Vector3();
const _scratchWorld2 = new THREE.Vector3();
const _scratchSpawn = new THREE.Vector3();

// Add configurable opacity for trails specific to vector copies under Q/K/V to reduce visual prominence
const FAINT_TRAIL_OPACITY = 0.08;
const PRE_QKV_RESIDUAL_LABEL = 'Post-layernorm residual';


// Read live binding each use to reflect UI changes at runtime

/**
 * Responsible only for moving vectors to their parking positions
 * under Q/K/V matrices (horizontal travel, upward copies, side copies).
 * Emits a callback once *every* K/Q/V copy is under its matrix so the
 * rest of the MHSA pipeline can begin the pass-through phase.
 */
export class VectorRouter {
    constructor(parentGroup, headsCentersX, headCoords, headStopY, mhaVisualizations, opts = {}) {
        this.parentGroup   = parentGroup;
        this.headsCentersX = headsCentersX;
        this.headCoords    = headCoords;
        this.headStopY     = headStopY;
        this.mhaVisualizations = mhaVisualizations;
        this._acquireVector = typeof opts.acquireVector === 'function' ? opts.acquireVector : null;
        this._shareVectorData = !!opts.shareVectorData;

        this._readyEmitted = false;
        this._callbacks    = new Set();
        this._skipToEndActive = false;
    }

    onReady(cb) {
        if (typeof cb === 'function') this._callbacks.add(cb);
    }

    setSkipToEndMode(enabled = false) {
        this._skipToEndActive = !!enabled;
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

                let remaining = ANIM_HORIZ_SPEED * GLOBAL_ANIM_SPEED_MULT * deltaTime;
                let safety    = 0; // prevent runaway loops in pathological cases

                while (remaining > 0.0001 && lane.horizPhase === 'travelMHSA') {
                    if (++safety > 32) break;

                    const targetHeadIdx = lane.headIndex || 0;
                    if (targetHeadIdx >= this.headsCentersX.length) {
                        tVec.group.visible = false;
                        lane.horizPhase = 'finishedHeads';
                        break;
                    }

                    const targetX     = this.headsCentersX[targetHeadIdx];
                    const currentX    = tVec.group.position.x;
                    const deltaToHead = targetX - currentX;
                    const absDelta    = Math.abs(deltaToHead);

                    // Already within tolerance of the head location – treat as arrival.
                    if (absDelta <= 0.0005) {
                        this._spawnUpwardCopyForHead(lane, targetHeadIdx, tVec);
                        lane.headIndex = targetHeadIdx + 1;
                        if (lane.headIndex >= NUM_HEAD_SETS_LAYER) {
                            tVec.group.visible = false;
                            lane.horizPhase = 'finishedHeads';
                        }
                        if (tVec.userData && tVec.userData.trail) tVec.userData.trail.update(tVec.group.position);
                        continue;
                    }

                    const dir     = Math.sign(deltaToHead) || 1;
                    const step    = Math.min(absDelta, remaining);
                    const newX    = currentX + dir * step;
                    const arrived = Math.abs(targetX - newX) <= 0.0005;

                    tVec.group.position.x = newX;
                    remaining -= step;

                    if (arrived) {
                        this._spawnUpwardCopyForHead(lane, targetHeadIdx, tVec);
                        lane.headIndex = targetHeadIdx + 1;
                        if (lane.headIndex >= NUM_HEAD_SETS_LAYER) {
                            tVec.group.visible = false;
                            lane.horizPhase = 'finishedHeads';
                        }
                    }

                    // Update trail for travelling vector after movement (even if arrived)
                    if (tVec.userData && tVec.userData.trail) tVec.userData.trail.update(tVec.group.position);

                    if (!arrived) break; // Still en route this frame
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
                            upVec.group.position.y + MHSA_DUPLICATE_VECTOR_RISE_SPEED * GLOBAL_ANIM_SPEED_MULT * deltaTime
                        );
                    }
                    if (upVec.userData && upVec.userData.trail) {
                        const ud = upVec.userData;
                        if (ud.trailWorld) {
                            upVec.group.getWorldPosition(_scratchWorld);
                            ud.trail.update(_scratchWorld);
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
                            const qVec = this._createVectorCopy(centerVec, _scratchSpawn.copy(centerVec.group.position), 30);
                            if (typeof qVec.copyColorsFrom === 'function') {
                                qVec.copyColorsFrom(centerVec);
                            }
                            const qTrail = new StraightLineTrail(this.parentGroup, TRAIL_COLOR, 1, undefined, FAINT_TRAIL_OPACITY, TRAIL_MIN_SEGMENT_DISTANCE);
                            qTrail.start(qVec.group.position);
                            qVec.userData = qVec.userData || {};
                            qVec.userData.trail = qTrail;
                            Object.assign(qVec.userData, { headIndex: hIdx, parentLane: lane });
                            qVec.group.userData = qVec.group.userData || {};
                            qVec.group.userData.headIndex = hIdx;
                            const parentLayerIndex = Number.isFinite(this.parentGroup?.userData?.layerIndex)
                                ? this.parentGroup.userData.layerIndex
                                : null;
                            if (Number.isFinite(parentLayerIndex)) qVec.group.userData.layerIndex = parentLayerIndex;
                            try {
                                qVec.group.userData.label = PRE_QKV_RESIDUAL_LABEL;
                                if (qVec.mesh) qVec.mesh.userData = { ...(qVec.mesh.userData||{}), label: PRE_QKV_RESIDUAL_LABEL };
                            } catch (_) {}
                            const vVec = this._createVectorCopy(centerVec, _scratchSpawn.copy(centerVec.group.position), 30);
                            if (typeof vVec.copyColorsFrom === 'function') {
                                vVec.copyColorsFrom(centerVec);
                            }
                            const vTrail = new StraightLineTrail(this.parentGroup, TRAIL_COLOR, 1, undefined, FAINT_TRAIL_OPACITY, TRAIL_MIN_SEGMENT_DISTANCE);
                            vTrail.start(vVec.group.position);
                            vVec.userData = vVec.userData || {};
                            vVec.userData.trail = vTrail;
                            Object.assign(vVec.userData, { headIndex: hIdx, parentLane: lane });
                            vVec.group.userData = vVec.group.userData || {};
                            vVec.group.userData.headIndex = hIdx;
                            const parentLayerIndexV = Number.isFinite(this.parentGroup?.userData?.layerIndex)
                                ? this.parentGroup.userData.layerIndex
                                : null;
                            if (Number.isFinite(parentLayerIndexV)) vVec.group.userData.layerIndex = parentLayerIndexV;
                            try {
                                vVec.group.userData.label = PRE_QKV_RESIDUAL_LABEL;
                                if (vVec.mesh) vVec.mesh.userData = { ...(vVec.mesh.userData||{}), label: PRE_QKV_RESIDUAL_LABEL };
                            } catch (_) {}
                            lane.sideCopies = lane.sideCopies || [];
                            const qMatrixForHead = this.mhaVisualizations[hIdx * 3];
                            const vMatrixForHead = this.mhaVisualizations[hIdx * 3 + 2];
                            lane.sideCopies.push({ vec: qVec, targetX: coord.q, type: 'Q', matrixRef: qMatrixForHead, headIndex: hIdx });
                            lane.sideCopies.push({ vec: vVec, targetX: coord.v, type: 'V', matrixRef: vMatrixForHead, headIndex: hIdx });
                            setSideCopyEntry(lane, hIdx, 'Q', lane.sideCopies[lane.sideCopies.length - 2]);
                            setSideCopyEntry(lane, hIdx, 'V', lane.sideCopies[lane.sideCopies.length - 1]);

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
                    const dx = SIDE_COPY_HORIZ_SPEED * GLOBAL_ANIM_SPEED_MULT * deltaTime;
                    const deltaToTarget = obj.targetX - v.group.position.x;
                    if (Math.abs(deltaToTarget) > 0.01) {
                        const step = Math.min(Math.abs(deltaToTarget), dx);
                        v.group.position.x += Math.sign(deltaToTarget || 1) * step;
                    }
                    v.group.position.y = this.headStopY;
                    if (v.userData && v.userData.trail) {
                        // Only update trail while below matrix level (consistent with other vectors)
                        const matrixBottomY = this.headStopY; // Q/V vectors stop at headStopY which is just below matrices
                        if (v.group.position.y <= matrixBottomY + 0.001 || this._skipToEndActive) {
                            const ud = v.userData;
                            if (ud.trailWorld) {
                                v.group.getWorldPosition(_scratchWorld2);
                                ud.trail.update(_scratchWorld2);
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

    _spawnUpwardCopyForHead(lane, targetHeadIdx, sourceVec) {
        if (!lane || !sourceVec) return null;
        lane.upwardCopies = lane.upwardCopies || [];
        if (lane.upwardCopies[targetHeadIdx]) return lane.upwardCopies[targetHeadIdx];

        const upVec = this._createVectorCopy(sourceVec, _scratchSpawn.copy(sourceVec.group.position), 30);
        if (typeof upVec.copyColorsFrom === 'function') {
            upVec.copyColorsFrom(sourceVec);
        }

        const upTrail = new StraightLineTrail(this.parentGroup, TRAIL_COLOR, 1, undefined, FAINT_TRAIL_OPACITY, TRAIL_MIN_SEGMENT_DISTANCE);
        upTrail.start(upVec.group.position);
        upVec.userData = upVec.userData || {};
        upVec.userData.trail = upTrail;
        Object.assign(upVec.userData, {
            headIndex: targetHeadIdx,
            sideSpawned: false,
            sideSpawnRequested: false,
            sideSpawnTime: 0,
            parentLane: lane,
        });
        upVec.group.userData = upVec.group.userData || {};
        upVec.group.userData.headIndex = targetHeadIdx;
        const parentLayerIndex = Number.isFinite(this.parentGroup?.userData?.layerIndex)
            ? this.parentGroup.userData.layerIndex
            : null;
        if (Number.isFinite(parentLayerIndex)) upVec.group.userData.layerIndex = parentLayerIndex;

        try {
            upVec.group.userData.label = PRE_QKV_RESIDUAL_LABEL;
            if (upVec.mesh) upVec.mesh.userData = { ...(upVec.mesh.userData || {}), label: PRE_QKV_RESIDUAL_LABEL };
        } catch (_) {}

        lane.upwardCopies[targetHeadIdx] = upVec;
        return upVec;
    }

    _createVectorCopy(sourceVec, position, numSubsections = 30) {
        const rawData = this._shareVectorData ? sourceVec.rawData : (sourceVec.rawData ? sourceVec.rawData.slice() : []);
        const instanceCount = sourceVec.instanceCount;
        if (this._acquireVector) {
            return this._acquireVector({
                rawData,
                position,
                instanceCount,
                numSubsections,
                shareData: this._shareVectorData
            });
        }
        const vec = new VectorVisualizationInstancedPrism(
            rawData,
            position ? position.clone() : new THREE.Vector3(),
            numSubsections,
            instanceCount,
            { copyData: !this._shareVectorData }
        );
        this.parentGroup.add(vec.group);
        return vec;
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
