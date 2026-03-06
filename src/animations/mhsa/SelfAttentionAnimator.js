import * as THREE from 'three';
import { VECTOR_LENGTH_PRISM, SA_RED_EXTRA_RISE, SA_V_RISE_DURATION_MS, SA_K_ALIGN_DURATION_MS, SA_BLUE_HORIZ_DURATION_MS, SA_BLUE_VERT_DURATION_MS, SA_BLUE_PAUSE_MS, SA_BLUE_PREPASS_SLOW_MULT, SA_DUPLICATE_POP_IN_MS, SA_DUPLICATE_TRAVEL_MERGE_MS, SA_DUPLICATE_POP_OUT_MS, SA_DUPLICATE_TO_SCORE_TRAVEL_FRACTION, SA_DUPLICATE_TO_SUM_TRAVEL_FRACTION, SA_DUPLICATE_SCORE_COLLISION_PULSE_MS, SA_DUPLICATE_SCORE_COLLISION_PULSE_MIN, SA_DUPLICATE_SCORE_COLLISION_PULSE_MAX, SA_DUPLICATE_SCORE_COLLISION_HALO_OPACITY, SA_DUPLICATE_SCORE_COLLISION_HALO_START_SCALE, SA_DUPLICATE_SCORE_COLLISION_HALO_END_SCALE, SA_DUPLICATE_SCORE_COLLISION_HALO_COLOR, SA_DUPLICATE_SCORE_COLLISION_HALO_DURATION_MULT, ATTENTION_POST_SOFTMAX_GRAYSCALE_MIN, GLOBAL_ANIM_SPEED_MULT, SELF_ATTENTION_TIME_MULT } from '../../utils/constants.js';
import { VectorVisualizationInstancedPrism } from '../../components/VectorVisualizationInstancedPrism.js';
import { mapValueToColor, mapValueToGrayscale, buildHueRangeOptions, mapValueToHueRange } from '../../utils/colors.js';
import { buildActivationData, applyActivationDataToVector } from '../../utils/activationMetadata.js';
import { logRandomColorDebug } from '../../utils/randomColorDebug.js';
import {
    MHA_VALUE_SPECTRUM_COLOR,
    MHA_VALUE_HUE_SPREAD,
    MHA_VALUE_LIGHTNESS_MIN,
    MHA_VALUE_LIGHTNESS_MAX,
    MHA_VALUE_RANGE_MIN,
    MHA_VALUE_RANGE_MAX,
    MHA_VALUE_CLAMP_MAX,
    MHA_VALUE_KEY_COLOR_COUNT,
    MHA_WEIGHTED_SUM_DOCK_OFFSET
} from '../LayerAnimationConstants.js';
import { getSideCopyEntry } from './laneIndex.js';

function logMhsaDebug(...args) {
    if (typeof window === 'undefined' || window.__MHSA_DEBUG !== true) return;
    console.log(...args);
}

// Shared lightweight geometry for self-attention highlight spheres
const SHARED_SPHERE_GEOMETRY = new THREE.SphereGeometry(10, 12, 12);
const BLUE_ENTRY_DURATION_MULT = 1.75;

/**
 * Self-attention specific, above-matrix animations.
 *
 * Responsibilities:
 * 1. Handle the legacy placeholder flow (for backwards compatibility)
 * 2. Extra rise for Value (red) vectors + horizontal alignment of
 *    corresponding Key (green) vectors underneath them.
 * 3. NEW – "conveyor-belt" style animation for Query (blue) vectors:
 *    • After the V/K alignment for a given head finishes we begin processing
 *      that head's queue of blue vectors.
 *    • The blue vectors are processed one-by-one in their ORIGINAL order
 *      (top → bottom along the slit index / lane order).
 *    • For the i-th blue vector we:
 *        a) Slide horizontally so it sits over the green matrix (x = K)
 *        b) Travel i positions along the green vectors (positive z direction
 *           – lanes are sorted by z). We pause briefly at each K vector.
 *        c) Slide horizontally so it sits over the red matrix (x = V)
 *        d) Travel i positions visiting the red vectors, pausing at each.
 *        e) Fade the vector out (dispose + remove from scene).
 *    • Blue vectors stay parked at their original slit positions until called.
 *      When it is a vector's turn, that vector alone moves into the front slot
 *      before starting the lane traversal.
 *    • The same logic runs in parallel for every attention head.
 */
export class SelfAttentionAnimator {
    /**
     * @param {MHSAAnimation} ctx – parent MHSAAnimation instance.
     */
    constructor(ctx) {
        this.ctx = ctx;
        this.phase = 'waiting'; // 'waiting' | 'running' | 'complete'
        this._callbacks = [];

        // ------------------------------------------------------------------
        // V/K constants (distances remain static; durations scale via getters)
        // ------------------------------------------------------------------
        this.RED_EXTRA_RISE   = SA_RED_EXTRA_RISE;   // additional rise for red (V) vectors

        // Internal bookkeeping (per-head)
        this.blueQueues        = {};  // { headIdx: [vec, ...] }
        this.blueProcessing    = {};  // { headIdx: boolean }
        this.blueProcessedCount = {}; // { headIdx: number }
        this.greensAligned     = {};  // { headIdx: boolean } – flagged once K vectors are in place
        this.skipRequested     = false;
        this._activeBlueVectors = {}; // { headIdx: Vector }
        this._pendingTimeouts   = new Set();
        this._spawnedSpheres    = new Set(); // stores active attention-sphere instance ids
        this._sphereMesh        = null;
        this._sphereCapacity    = 0;
        this._sphereNextIndex   = 0;
        this._sphereFreeList    = [];
        this._sphereInstances   = new Map(); // instanceId -> { position, scale, activationData }
        this._sphereLabels      = [];
        this._sphereEntries     = [];
        this._sphereDummy       = new THREE.Object3D();
        this._sphereColorTmp    = new THREE.Color();
        this._collisionHaloColorTmp = new THREE.Color(SA_DUPLICATE_SCORE_COLLISION_HALO_COLOR);
        this._activeSphereHalos = new Set();
        this._dupVecPool        = [];
        this._dupVecPoolLimit   = 32;
        this._spawnedTempVectors = new Set();
        this._pendingDockCount  = 0;
        this.attentionProgress = {};
        this.attentionCompletedRows = {};
        this.attentionPostCompletedRows = {};
        this._attentionRowCache = new Map();
        this._weightedSumCache = new Map();
        this._tmpMidpoint = new THREE.Vector3();
        this._tmpCtxPosA = new THREE.Vector3();
        this._tmpCtxPosB = new THREE.Vector3();
        this._valueHueRangeOptions = buildHueRangeOptions(MHA_VALUE_SPECTRUM_COLOR, {
            hueSpread: MHA_VALUE_HUE_SPREAD,
            minLightness: MHA_VALUE_LIGHTNESS_MIN,
            maxLightness: MHA_VALUE_LIGHTNESS_MAX,
            valueMin: MHA_VALUE_RANGE_MIN,
            valueMax: MHA_VALUE_RANGE_MAX,
            valueClampMax: MHA_VALUE_CLAMP_MAX,
        });
    }

    // Durations decoupled from GLOBAL_ANIM_SPEED_MULT to make presets clearly visible
    get V_RISE_DURATION() { return SA_V_RISE_DURATION_MS * SELF_ATTENTION_TIME_MULT; }
    get K_ALIGN_DURATION() { return SA_K_ALIGN_DURATION_MS * SELF_ATTENTION_TIME_MULT; }
    get BLUE_HORIZ_DURATION() { return SA_BLUE_HORIZ_DURATION_MS * SELF_ATTENTION_TIME_MULT; }
    get BLUE_VERT_DURATION() { return SA_BLUE_VERT_DURATION_MS * SELF_ATTENTION_TIME_MULT; }
    get BLUE_PAUSE_MS() { return SA_BLUE_PAUSE_MS * SELF_ATTENTION_TIME_MULT; }
    get DUPLICATE_POP_IN_MS() { return SA_DUPLICATE_POP_IN_MS * SELF_ATTENTION_TIME_MULT; }
    get DUPLICATE_TRAVEL_MERGE_MS() { return SA_DUPLICATE_TRAVEL_MERGE_MS * SELF_ATTENTION_TIME_MULT; }
    get DUPLICATE_POP_OUT_MS() { return SA_DUPLICATE_POP_OUT_MS * SELF_ATTENTION_TIME_MULT; }
    get DUPLICATE_SCORE_COLLISION_PULSE_MS() { return SA_DUPLICATE_SCORE_COLLISION_PULSE_MS * SELF_ATTENTION_TIME_MULT; }

    // ------------------------------------------------------------------
    // Skip / cleanup helpers
    // ------------------------------------------------------------------
    _registerTimeout(id) {
        if (!id) return;
        this._pendingTimeouts.add(id);
    }

    _clearPendingTimeouts() {
        this._pendingTimeouts.forEach((cancel) => {
            try {
                if (typeof cancel === 'function') {
                    cancel();
                } else {
                    clearTimeout(cancel);
                }
            } catch (_) { /* ignore */ }
        });
        this._pendingTimeouts.clear();
    }

    _scheduleAfterDelay(callback, delayMs) {
        if (typeof callback !== 'function') return null;
        let cancelFn = null;
        const wrapped = () => {
            if (cancelFn) {
                this._pendingTimeouts.delete(cancelFn);
            }
            try {
                callback();
            } catch (_) { /* ignore */ }
        };
        if (this.ctx && typeof this.ctx._scheduleAfterDelay === 'function') {
            cancelFn = this.ctx._scheduleAfterDelay(wrapped, delayMs);
        } else {
            const timeoutId = setTimeout(wrapped, delayMs);
            cancelFn = () => clearTimeout(timeoutId);
        }
        this._registerTimeout(cancelFn);
        return cancelFn;
    }

    _markVectorLayoutDirty(vec) {
        if (!this.ctx || typeof this.ctx._markBatchedVectorLayoutDirty !== 'function') return false;
        return this.ctx._markBatchedVectorLayoutDirty(vec) === true;
    }

    _cleanupAttentionScoreMeshes() {
        // Clear instanced attention spheres regardless of scene labels.
        this._clearAttentionSphereInstances();
        this._clearAttentionSphereHalos();
        const root = this.ctx && this.ctx.parentGroup;
        if (!root || typeof root.traverse !== 'function') return;
        const toRemove = [];
        root.traverse((obj) => {
            if (!obj || !obj.isMesh) return;
            const ud = obj.userData || {};
            const label = typeof ud.label === 'string' ? ud.label : '';
            const stage = ud.activationData && ud.activationData.stage;
            if (label.includes('Attention Score') || stage === 'attention.pre' || stage === 'attention.post') {
                toRemove.push(obj);
            }
        });
        toRemove.forEach((obj) => {
            if (obj.userData && obj.userData._attentionSphereInstanced) {
                this._clearAttentionSphereInstances();
                return;
            }
            try { if (obj.parent) obj.parent.remove(obj); } catch (_) { /* optional cleanup */ }
            try { if (obj.material && typeof obj.material.dispose === 'function') obj.material.dispose(); } catch (_) { /* optional cleanup */ }
            try {
                if (obj.geometry && obj.geometry !== SHARED_SPHERE_GEOMETRY && typeof obj.geometry.dispose === 'function') {
                    obj.geometry.dispose();
                }
            } catch (_) { /* optional cleanup */ }
        });
    }

    _ensureAttentionSphereMesh() {
        if (this._sphereMesh || !this.ctx || !this.ctx.parentGroup) return;
        const capacity = 512;
        const material = new THREE.MeshBasicMaterial({ color: 0xffffff });
        const mesh = new THREE.InstancedMesh(SHARED_SPHERE_GEOMETRY, material, capacity);
        mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
        mesh.userData._attentionSphereInstanced = true;
        mesh.userData.instanceKind = 'attentionSphere';
        mesh.userData.instanceLabels = new Array(capacity).fill(null);
        mesh.userData.instanceEntries = new Array(capacity).fill(null);
        // Initialize all instances hidden.
        const dummy = this._sphereDummy;
        for (let i = 0; i < capacity; i += 1) {
            dummy.position.set(0, -9999, 0);
            dummy.scale.set(0.001, 0.001, 0.001);
            dummy.updateMatrix();
            mesh.setMatrixAt(i, dummy.matrix);
            mesh.setColorAt(i, this._sphereColorTmp.setRGB(0.1, 0.1, 0.1));
        }
        mesh.instanceMatrix.needsUpdate = true;
        if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
        this.ctx.parentGroup.add(mesh);
        this._sphereMesh = mesh;
        this._sphereCapacity = capacity;
        this._sphereNextIndex = 0;
        this._sphereLabels = mesh.userData.instanceLabels;
        this._sphereEntries = mesh.userData.instanceEntries;
    }

    _updateAttentionSphereInstance(instanceId) {
        if (!this._sphereMesh || !Number.isFinite(instanceId)) return;
        const data = this._sphereInstances.get(instanceId);
        if (!data) return;
        const dummy = this._sphereDummy;
        dummy.position.copy(data.position);
        dummy.scale.setScalar(data.scale);
        dummy.updateMatrix();
        this._sphereMesh.setMatrixAt(instanceId, dummy.matrix);
        this._sphereMesh.instanceMatrix.needsUpdate = true;
    }

    _setAttentionSphereColor(instanceId, color) {
        if (!this._sphereMesh || !Number.isFinite(instanceId) || !color) return;
        this._sphereMesh.setColorAt(instanceId, color);
        if (this._sphereMesh.instanceColor) {
            this._sphereMesh.instanceColor.needsUpdate = true;
        }
    }

    _acquireAttentionSphere(position, color, activationData) {
        this._ensureAttentionSphereMesh();
        if (!this._sphereMesh) return null;
        let instanceId = this._sphereFreeList.pop();
        if (!Number.isFinite(instanceId)) {
            if (this._sphereNextIndex < this._sphereCapacity) {
                instanceId = this._sphereNextIndex;
                this._sphereNextIndex += 1;
            } else {
                // Recycle the oldest active sphere if we exceed capacity.
                const iter = this._spawnedSpheres.values().next();
                if (iter.done) return null;
                instanceId = iter.value;
                this._releaseAttentionSphere(instanceId);
            }
        }
        const data = {
            position: position.clone(),
            scale: 0.001,
            activationData: activationData || null
        };
        this._sphereInstances.set(instanceId, data);
        this._spawnedSpheres.add(instanceId);
        this._sphereLabels[instanceId] = 'Pre-Softmax Attention Score';
        this._sphereEntries[instanceId] = { activationData: activationData || null };
        this._setAttentionSphereColor(instanceId, color);
        this._updateAttentionSphereInstance(instanceId);
        return instanceId;
    }

    _releaseAttentionSphere(instanceId) {
        if (!this._sphereMesh || !Number.isFinite(instanceId)) return;
        this._spawnedSpheres.delete(instanceId);
        this._sphereInstances.delete(instanceId);
        this._sphereLabels[instanceId] = null;
        this._sphereEntries[instanceId] = null;
        this._sphereFreeList.push(instanceId);
        const dummy = this._sphereDummy;
        dummy.position.set(0, -9999, 0);
        dummy.scale.set(0.001, 0.001, 0.001);
        dummy.updateMatrix();
        this._sphereMesh.setMatrixAt(instanceId, dummy.matrix);
        this._sphereMesh.instanceMatrix.needsUpdate = true;
    }

    _clearAttentionSphereInstances() {
        this._spawnedSpheres.forEach((instanceId) => {
            this._releaseAttentionSphere(instanceId);
        });
        this._spawnedSpheres.clear();
        this._sphereInstances.clear();
        this._sphereFreeList = [];
        this._sphereNextIndex = 0;
    }

    _getAttentionSphereData(instanceId) {
        if (!Number.isFinite(instanceId)) return null;
        return this._sphereInstances.get(instanceId) || null;
    }

    _disposeAttentionSphereHalo(mesh) {
        if (!mesh) return;
        this._activeSphereHalos.delete(mesh);
        try { if (mesh.parent) mesh.parent.remove(mesh); } catch (_) { /* optional cleanup */ }
        try { if (mesh.material && typeof mesh.material.dispose === 'function') mesh.material.dispose(); } catch (_) { /* optional cleanup */ }
    }

    _clearAttentionSphereHalos() {
        if (!this._activeSphereHalos || this._activeSphereHalos.size === 0) return;
        this._activeSphereHalos.forEach((mesh) => {
            this._disposeAttentionSphereHalo(mesh);
        });
        this._activeSphereHalos.clear();
    }

    _spawnAttentionSphereHalo(instanceId, durationMs = 120, colorOverride = null) {
        const sphereData = this._getAttentionSphereData(instanceId);
        if (!sphereData || !this.ctx || !this.ctx.parentGroup) return;
        const baseScale = Number.isFinite(sphereData.scale) ? Math.max(0.001, sphereData.scale) : 0.8;
        const startScale = baseScale * SA_DUPLICATE_SCORE_COLLISION_HALO_START_SCALE;
        const endScale = baseScale * SA_DUPLICATE_SCORE_COLLISION_HALO_END_SCALE;
        const startOpacity = THREE.MathUtils.clamp(SA_DUPLICATE_SCORE_COLLISION_HALO_OPACITY, 0, 1);
        if (startOpacity <= 0) return;
        const haloColor = colorOverride && colorOverride.isColor
            ? colorOverride
            : SA_DUPLICATE_SCORE_COLLISION_HALO_COLOR;

        const haloMaterial = new THREE.MeshBasicMaterial({
            color: haloColor,
            transparent: true,
            opacity: startOpacity,
            // Keep halo flashes occluded by foreground geometry.
            depthTest: true,
            depthWrite: false,
            blending: THREE.AdditiveBlending
        });
        const halo = new THREE.Mesh(SHARED_SPHERE_GEOMETRY, haloMaterial);
        halo.position.copy(sphereData.position);
        halo.scale.setScalar(startScale);
        halo.userData = halo.userData || {};
        halo.userData.label = 'Attention Score Halo';
        halo.userData.activationData = sphereData.activationData || null;
        this.ctx.parentGroup.add(halo);
        this._activeSphereHalos.add(halo);

        const duration = Number.isFinite(durationMs) ? Math.max(1, durationMs) : 120;
        const state = { s: startScale, o: startOpacity };
        new TWEEN.Tween(state)
            .to({ s: endScale, o: 0 }, duration)
            .easing(TWEEN.Easing.Cubic.Out)
            .onUpdate(() => {
                if (!this._activeSphereHalos.has(halo)) return;
                const liveSphere = this._getAttentionSphereData(instanceId);
                if (liveSphere) halo.position.copy(liveSphere.position);
                halo.scale.setScalar(state.s);
                if (halo.material) {
                    halo.material.opacity = state.o;
                    halo.material.needsUpdate = true;
                }
            })
            .onComplete(() => {
                this._disposeAttentionSphereHalo(halo);
            })
            .start();
    }

    _tweenSphereScale(instanceId, targetScale, duration, options = {}) {
        const data = this._getAttentionSphereData(instanceId);
        if (!data) return;
        const startScale = Number.isFinite(data.scale) ? data.scale : 0.001;
        const state = { s: startScale };
        const tween = new TWEEN.Tween(state)
            .to({ s: targetScale }, duration)
            .easing(options.easing || TWEEN.Easing.Quadratic.Out)
            .onUpdate(() => {
                data.scale = state.s;
                this._updateAttentionSphereInstance(instanceId);
            });
        if (Number.isFinite(options.delay) && options.delay > 0) {
            tween.delay(options.delay);
        }
        if (options.yoyo) {
            tween.yoyo(true);
            tween.repeat(Number.isFinite(options.repeat) ? options.repeat : 1);
        }
        if (typeof options.onComplete === 'function') {
            tween.onComplete(options.onComplete);
        }
        tween.start();
    }

    _acquireDuplicateVector(sourceVec) {
        const desiredCount = Number.isFinite(sourceVec?.instanceCount)
            ? sourceVec.instanceCount
            : (Number.isFinite(this.ctx?.vectorPrismCount) ? this.ctx.vectorPrismCount : VECTOR_LENGTH_PRISM);
        let dupVec = null;
        while (this._dupVecPool.length) {
            const candidate = this._dupVecPool.pop();
            if (candidate && candidate.instanceCount === desiredCount) {
                dupVec = candidate;
                break;
            }
            try { if (typeof candidate?.dispose === 'function') candidate.dispose(); } catch (_) { /* ignore */ }
        }
        if (!dupVec) {
            const startPos = sourceVec?.group?.position ? sourceVec.group.position.clone() : new THREE.Vector3();
            dupVec = new VectorVisualizationInstancedPrism(
                sourceVec?.rawData ? sourceVec.rawData.slice() : [],
                startPos,
                3,
                desiredCount
            );
        } else {
            dupVec.group.visible = true;
            dupVec.group.scale.set(1, 1, 1);
            if (sourceVec?.group?.position) {
                dupVec.group.position.copy(sourceVec.group.position);
            }
            if (typeof dupVec.updateDataInternal === 'function' && Array.isArray(sourceVec?.rawData)) {
                dupVec.updateDataInternal(sourceVec.rawData.slice());
            }
            if (dupVec.mesh?.material) {
                dupVec.mesh.material.transparent = false;
                dupVec.mesh.material.opacity = 1;
                dupVec.mesh.material.needsUpdate = true;
            }
        }
        dupVec.userData = {
            ...(dupVec.userData || {}),
            isPooledDuplicate: true,
            __releasedToPool: false,
        };
        return dupVec;
    }

    _releaseDuplicateVector(vec) {
        if (!vec) return;
        vec.userData = vec.userData || {};
        if (vec.userData.__releasedToPool) return;
        vec.userData.__releasedToPool = true;
        try { if (vec.group?.parent) vec.group.parent.remove(vec.group); } catch (_) { /* ignore */ }
        try {
            vec.group.visible = false;
            vec.group.scale.set(0.001, 0.001, 0.001);
        } catch (_) { /* ignore */ }
        if (this._dupVecPool.length < this._dupVecPoolLimit) {
            this._dupVecPool.push(vec);
        } else {
            try { if (typeof vec.dispose === 'function') vec.dispose(); } catch (_) { /* ignore */ }
        }
    }

    _getLaneForZ(zPos) {
        if (this.ctx && typeof this.ctx.getLaneForZ === 'function') {
            const lane = this.ctx.getLaneForZ(zPos);
            if (lane) return lane;
        }
        const lanes = this.ctx && Array.isArray(this.ctx.currentLanes) ? this.ctx.currentLanes : null;
        if (!lanes || !Number.isFinite(zPos)) return null;
        let bestLane = null;
        let bestDist = Infinity;
        for (let i = 0; i < lanes.length; i++) {
            const lane = lanes[i];
            if (!lane || !Number.isFinite(lane.zPos)) continue;
            const dist = Math.abs(lane.zPos - zPos);
            if (dist < bestDist) {
                bestDist = dist;
                bestLane = lane;
            }
        }
        return bestDist <= 0.25 ? bestLane : null;
    }

    _getVectorPositionInContextSpace(vec, out = null) {
        if (!vec || !vec.group) return null;
        const dst = out || this._tmpCtxPosA;
        const ctxGroup = this.ctx && this.ctx.parentGroup ? this.ctx.parentGroup : null;
        if (vec.isBatchedVectorRef) {
            const isKvPersistent = !!(
                vec.userData?.kvCachePersistent
                || vec.userData?.cachedKv
                || vec.group?.userData?.kvCachePersistent
                || vec.group?.userData?.cachedKv
            );
            if (isKvPersistent && ctxGroup) {
                if (vec.group.parent && vec.group.parent !== ctxGroup && typeof vec.group.getWorldPosition === 'function') {
                    vec.group.getWorldPosition(dst);
                    try {
                        if (typeof ctxGroup.worldToLocal === 'function') {
                            ctxGroup.worldToLocal(dst);
                        }
                    } catch (_) { /* fallback to world position */ }
                    return dst;
                }
                if (!vec.group.parent && vec._batch?.mesh?.parent) {
                    dst.copy(vec.group.position);
                    try {
                        vec._batch.mesh.parent.localToWorld(dst);
                        if (typeof ctxGroup.worldToLocal === 'function') {
                            ctxGroup.worldToLocal(dst);
                        }
                    } catch (_) { /* fallback to local copy below */ }
                    return dst;
                }
            }
            // Non-persistent batched vectors keep local pose in the helper group.
            dst.copy(vec.group.position);
            return dst;
        }
        if (!ctxGroup || vec.group.parent === ctxGroup || !vec.group.parent) {
            dst.copy(vec.group.position);
            return dst;
        }
        vec.group.getWorldPosition(dst);
        try {
            if (typeof ctxGroup.worldToLocal === 'function') {
                ctxGroup.worldToLocal(dst);
            }
        } catch (_) { /* fallback to world position */ }
        return dst;
    }

    _getAttentionScoresRow(mode, layerIndex, headIdx, queryTokenIndex) {
        if (!Number.isFinite(layerIndex) || !Number.isFinite(headIdx) || !Number.isFinite(queryTokenIndex)) return null;
        const key = `${mode}|${layerIndex}|${headIdx}|${queryTokenIndex}`;
        if (this._attentionRowCache.has(key)) {
            return this._attentionRowCache.get(key);
        }
        const activationSource = this.ctx && this.ctx.activationSource ? this.ctx.activationSource : null;
        const row = activationSource && typeof activationSource.getAttentionScoresRow === 'function'
            ? activationSource.getAttentionScoresRow(layerIndex, mode, headIdx, queryTokenIndex)
            : null;
        if (row) this._attentionRowCache.set(key, row);
        return row;
    }

    _isHeadActive(headIdx) {
        if (!Number.isFinite(headIdx)) return false;
        const queued = this.blueQueues && this.blueQueues[headIdx];
        return !!(this.blueProcessing && this.blueProcessing[headIdx])
            || !!(this._activeBlueVectors && this._activeBlueVectors[headIdx])
            || (Array.isArray(queued) && queued.length > 0);
    }

    _setAttentionProgress(headIdx, rowIndex, colIndex) {
        if (!Number.isFinite(headIdx) || !Number.isFinite(rowIndex)) return;
        const prev = this.attentionProgress[headIdx] || {};
        const next = {
            activeRow: rowIndex,
            activeCol: Number.isFinite(colIndex) ? colIndex : prev.activeCol
        };
        this.attentionProgress[headIdx] = next;
    }

    _markAttentionRowComplete(headIdx, rowIndex) {
        if (!Number.isFinite(headIdx) || !Number.isFinite(rowIndex)) return;
        const completed = this.attentionCompletedRows[headIdx] || 0;
        this.attentionCompletedRows[headIdx] = Math.max(completed, rowIndex + 1);
        const progress = this.attentionProgress[headIdx];
        if (progress && progress.activeRow === rowIndex) {
            progress.activeRow = null;
            progress.activeCol = null;
        }
    }

    _markAttentionPostRowComplete(headIdx, rowIndex) {
        if (!Number.isFinite(headIdx) || !Number.isFinite(rowIndex)) return;
        const completed = this.attentionPostCompletedRows[headIdx] || 0;
        this.attentionPostCompletedRows[headIdx] = Math.max(completed, rowIndex + 1);
    }

    _markAttentionHeadComplete(headIdx) {
        if (!Number.isFinite(headIdx)) return;
        const conveyorLanes = (this.ctx && typeof this.ctx.getAttentionConveyorLanes === 'function')
            ? this.ctx.getAttentionConveyorLanes()
            : null;
        const laneCount = Array.isArray(conveyorLanes) && conveyorLanes.length
            ? conveyorLanes.length
            : (Array.isArray(this.ctx?.currentLanes) ? this.ctx.currentLanes.length : 0);
        if (laneCount > 0) {
            const completed = this.attentionCompletedRows[headIdx] || 0;
            this.attentionCompletedRows[headIdx] = Math.max(completed, laneCount);
            const postCompleted = this.attentionPostCompletedRows[headIdx] || 0;
            this.attentionPostCompletedRows[headIdx] = Math.max(postCompleted, laneCount);
        }
        const progress = this.attentionProgress[headIdx];
        if (progress) {
            progress.activeRow = null;
            progress.activeCol = null;
        }
    }

    getAttentionProgress(headIdx) {
        if (!this._isHeadActive(headIdx)) return null;
        const completedRows = this.attentionCompletedRows[headIdx] || 0;
        const postCompletedRows = this.attentionPostCompletedRows[headIdx] || 0;
        const progress = this.attentionProgress[headIdx] || {};
        const activeRow = Number.isFinite(progress.activeRow) ? progress.activeRow : null;
        const activeCol = Number.isFinite(progress.activeCol) ? progress.activeCol : null;
        return { completedRows, postCompletedRows, activeRow, activeCol };
    }

    _retireVector(vec, { preserveTrail = false } = {}) {
        if (!vec) return;
        // Batched vector refs are owned by MHSAAnimation; hand them back to release/hide.
        if (vec.isBatchedVectorRef) {
            try {
                if (this.ctx && typeof this.ctx._releaseVectorCopy === 'function') {
                    this.ctx._releaseVectorCopy(vec);
                } else if (vec.group) {
                    vec.group.visible = false;
                }
            } catch (_) { /* optional cleanup */ }
            return;
        }
        try {
            if (!preserveTrail && vec.userData && vec.userData.trail && typeof vec.userData.trail.dispose === 'function') {
                vec.userData.trail.dispose();
                delete vec.userData.trail;
            }
        } catch (_) { /* optional cleanup */ }
        try {
            if (vec.group) {
                if (vec.group.parent) vec.group.parent.remove(vec.group);
                vec.group.visible = false;
                vec.group.scale.set(0.001, 0.001, 0.001);
            }
        } catch (_) { /* optional cleanup */ }
        try {
            if (typeof vec.dispose === 'function') vec.dispose();
        } catch (_) { /* optional cleanup */ }
        if (this._spawnedTempVectors) this._spawnedTempVectors.delete(vec);
    }

    _finishBlueImmediately(vector, headIdx, doneCb, options = null) {
        this._retireVector(vector, options || undefined);
        if (typeof headIdx === 'number' && this._activeBlueVectors) {
            delete this._activeBlueVectors[headIdx];
        }
        if (typeof doneCb === 'function') doneCb();
    }

    /**
     * Immediately complete the conveyor belt, clear queues, and fire callbacks.
     */
    forceComplete(options = {}) {
        if (this.phase === 'complete') {
            this._flushCallbacks();
            return;
        }
        const preserveTrails = !!options.preserveTrails;
        const createWeightedSums = !!options.createWeightedSums;
        const replaceWeightedSums = options.replaceWeightedSums !== false;
        this.skipRequested = true;
        this._clearPendingTimeouts();
        if (createWeightedSums) {
            try { this._createWeightedSumsImmediate({ replaceExisting: replaceWeightedSums }); } catch (_) {}
        }
        // Retire any active vectors and queued blues
        Object.values(this._activeBlueVectors || {}).forEach((vec) => this._retireVector(vec, { preserveTrail: preserveTrails }));
        Object.values(this.blueQueues || {}).forEach((queue) => {
            if (Array.isArray(queue)) queue.forEach((vec) => this._retireVector(vec, { preserveTrail: preserveTrails }));
        });
        this.blueQueues = {};
        this.blueProcessing = {};
        this.blueProcessedCount = {};
        this._pendingDockCount = 0;
        this.phase = 'complete';
        // Clean up any lingering highlight spheres
        this._spawnedSpheres.forEach((sp) => {
            this._releaseAttentionSphere(sp);
        });
        this._spawnedSpheres.clear();
        this._cleanupAttentionScoreMeshes();
        this.attentionProgress = {};
        this.attentionCompletedRows = {};
        this.attentionPostCompletedRows = {};
        this._attentionRowCache.clear();
        this._weightedSumCache.clear();
        // Dispose any transient attention vectors (duplicates / travellers)
        this._spawnedTempVectors.forEach((vec) => {
            if (vec && vec.userData && vec.userData.isPooledDuplicate) {
                this._releaseDuplicateVector(vec);
            } else {
                this._retireVector(vec, { preserveTrail: preserveTrails });
            }
        });
        this._spawnedTempVectors.clear();
        // Finalize K/V visuals after skip completion.
        try {
            if (this.ctx && typeof this.ctx._shouldPreserveKVCacheVectors === 'function' && this.ctx._shouldPreserveKVCacheVectors()) {
                this.ctx._preserveKVVectorsForCache?.();
            } else {
                this.ctx && this.ctx._disposeMergedKVGroups && this.ctx._disposeMergedKVGroups();
                this.ctx && this.ctx._disposeAllIndividualKandVVectorsImmediately && this.ctx._disposeAllIndividualKandVVectorsImmediately();
            }
            this.ctx && this.ctx._hideAllQVectorsImmediately && this.ctx._hideAllQVectorsImmediately();
        } catch (_) { /* optional */ }
        this._flushCallbacks();
    }

    /**
     * Public helper so UI can decide whether a skip button should be visible.
     */
    isConveyorActive() {
        return !this._isConveyorComplete();
    }

    /**
     * Public entry – dual signature for backwards compatibility.
     * 1. start(onDone)                              – legacy placeholder
     * 2. start(vector, vectorCategory, onDone)      – per-vector handling
     */
    start(vectorOrCallback = null, vectorCategory = null, onDone = null) {
        // Global completion callback registration (legacy signature)
        if (typeof vectorOrCallback === 'function') {
            this._callbacks.push(vectorOrCallback);
            // If the conveyor is already finished, flush immediately.
            if (this._isConveyorComplete()) {
                this._flushCallbacks();
                this.phase = 'complete';
            }
            return;
        }

        // Vector-aware mode
        const vector = vectorOrCallback;
        this._handleAboveMatrixAnimations(vector, vectorCategory, onDone);
    }

    // ------------------------------------------------------------------
    // Placeholder mode (kept for compatibility with older tests)
    // ------------------------------------------------------------------
    _runPlaceholderMode() {
        if (this.phase === 'complete') {
            this._flushCallbacks();
            return;
        }
        if (this.phase === 'running') {
            return;
        }

        this.phase = 'running';
        logMhsaDebug('SelfAttentionAnimator: placeholder phase started');

        this._scheduleAfterDelay(() => {
            this.phase = 'complete';
            logMhsaDebug('SelfAttentionAnimator: placeholder phase complete');
            this._flushCallbacks();
        }, 3000);
    }

    // ------------------------------------------------------------------
    // Per-vector dispatch
    // ------------------------------------------------------------------
    _handleAboveMatrixAnimations(vector, vectorCategory, onDone) {
        if (this.skipRequested) {
            onDone && onDone();
            return;
        }
        if (this.phase === 'waiting') this.phase = 'running';
        if (typeof TWEEN === 'undefined') {
            console.error('Global TWEEN object not loaded for SelfAttentionAnimator!');
            onDone && onDone();
            return;
        }

        if (vectorCategory === 'V') {
            // Value (red) vector: extra rise then trigger K alignment.
            this._animateVVectorRise(vector, onDone);
            return;
        }

        if (vectorCategory === 'Q') {
            // Query (blue) vector: enqueue for conveyor-belt processing.
            this._enqueueBlueVector(vector);
            onDone && onDone(); // Immediately release the caller – we manage blue vectors asynchronously.
            return;
        }

        // Other vector categories – nothing special for now.
        onDone && onDone();
    }

    // ------------------------------------------------------------------
    // V / K alignment helpers
    // ------------------------------------------------------------------
    /**
     * Fixed Value (red) vectors rise in TWO separate stages:
     * 1. VectorMatrixPassThrough.js – while passing vertically through the V-matrix
     *    each red copy is tweened from its parking height (`headStopY`) up to
     *    `ctx.mhaPassThroughTargetY + ctx.mhaResultRiseOffsetY - 30`.
     *    This is the same base rise distance Q and K experience.
     * 2. SelfAttentionAnimator.js – this method then adds an additional
     *    `RED_EXTRA_RISE` (75 world-units) so the fixed V copies end up sitting
     *    above the green K vectors.  At the end of this tween each fixed V copy
     *    is located at the canonical "raised-V" height:
     *        ctx.mhaPassThroughTargetY + ctx.mhaResultRiseOffsetY - 30 + RED_EXTRA_RISE
     *    All subsequent duplicates and travelling red vectors snap to / match
     *    this same Y so they remain perfectly level.
     */
    _animateVVectorRise(vector, onDone) {
        if (this.skipRequested) {
            this._retireVector(vector);
            onDone && onDone();
            return;
        }
        new TWEEN.Tween({ y: vector.group.position.y })
            .to({ y: vector.group.position.y + this.RED_EXTRA_RISE }, this.V_RISE_DURATION)
            .easing(TWEEN.Easing.Quadratic.Out)
            .onUpdate(obj => {
                vector.group.position.y = obj.y;
                this._markVectorLayoutDirty(vector);
            })
            .onComplete(() => {
                this._alignKVectorsUnderV(vector, onDone);
            })
            .start();
    }

    _alignKVectorsUnderV(redVector, onDone) {
        if (this.skipRequested) {
            onDone && onDone();
            return;
        }
        const redX = redVector.group.position.x;
        const redZ = redVector.group.position.z;
        const headIdx = (redVector.userData && typeof redVector.userData.headIndex === 'number')
            ? redVector.userData.headIndex : null;

        const conveyorLanes = (this.ctx && typeof this.ctx.getAttentionConveyorLanes === 'function')
            ? this.ctx.getAttentionConveyorLanes()
            : this.ctx.currentLanes;
        if (headIdx === null || !Array.isArray(conveyorLanes) || conveyorLanes.length === 0) {
            onDone && onDone();
            return;
        }

        let alignmentsInProgress = 0;
        let alignmentsCompleted  = 0;

        const laneHit = this._getLaneForZ(redZ);
        // In decode/KV-cache passes, only one live red vector animates, but we still
        // need cached K vectors to align under their V columns for conveyor visuals.
        // Prefer full conveyor lanes (cached + live). Fall back to laneHit only if needed.
        const lanesToAlign = Array.isArray(conveyorLanes) && conveyorLanes.length
            ? conveyorLanes
            : (laneHit ? [laneHit] : this.ctx.currentLanes);
        lanesToAlign.forEach(lane => {
            if (lane.upwardCopies && lane.upwardCopies[headIdx]) {
                const green = lane.upwardCopies[headIdx];
                alignmentsInProgress++;

                new TWEEN.Tween(green.group.position)
                    .to({ x: redX }, this.K_ALIGN_DURATION)
                    .easing(TWEEN.Easing.Quadratic.Out)
                    .onUpdate(() => {
                        this._markVectorLayoutDirty(green);
                    })
                    .onComplete(() => {
                        alignmentsCompleted++;
                        if (alignmentsCompleted >= alignmentsInProgress) {
                            // Mark this head as ready for blue-vector conveyor processing.
                            this.greensAligned[headIdx] = true;
                            // Merge fixed K/V vectors for this head to reduce draw calls
                            try { this.ctx && this.ctx._mergeFixedVectorsForHead && this.ctx._mergeFixedVectorsForHead(headIdx); } catch (_) { /* optional */ }
                            this._kickoffBlueConveyor(headIdx);
                            onDone && onDone();
                        }
                    })
                    .start();
            }
        });

        // No greens to align – still flag readiness so blue vectors can run.
        if (alignmentsInProgress === 0) {
            this.greensAligned[headIdx] = true;
                // Merge fixed K/V vectors for this head immediately if nothing to align
                try { this.ctx && this.ctx._mergeFixedVectorsForHead && this.ctx._mergeFixedVectorsForHead(headIdx); } catch (_) { /* optional */ }
            this._kickoffBlueConveyor(headIdx);
            onDone && onDone();
        }
    }

    // ------------------------------------------------------------------
    // Blue (Query) conveyor-belt logic
    // ------------------------------------------------------------------
    _enqueueBlueVector(vector) {
        if (this.skipRequested) return;
        if (this.phase === 'waiting') this.phase = 'running';
        const headIdx = (vector.userData && typeof vector.userData.headIndex === 'number')
            ? vector.userData.headIndex : null;
        if (headIdx === null) return;

        if (!this.blueQueues[headIdx]) this.blueQueues[headIdx] = [];
        const queue = this.blueQueues[headIdx];

        queue.push(vector);
        // Keep queue ordered by lane z (ascending) so index == lane order (top → bottom).
        // Using the parent lane's zPos avoids jitter while vectors are mid-shift.
        queue.sort((a, b) => {
            const az = (a.userData && a.userData.parentLane && typeof a.userData.parentLane.zPos === 'number')
                ? a.userData.parentLane.zPos
                : a.group.position.z;
            const bz = (b.userData && b.userData.parentLane && typeof b.userData.parentLane.zPos === 'number')
                ? b.userData.parentLane.zPos
                : b.group.position.z;
            return az - bz;
        });

        // If greens are already in position we can start processing immediately.
        if (this.greensAligned[headIdx]) {
            this._kickoffBlueConveyor(headIdx);
        }
    }

    _kickoffBlueConveyor(headIdx) {
        if (this.skipRequested) return;
        if (this.blueProcessing[headIdx]) return; // already running
        if (!this.blueQueues[headIdx] || this.blueQueues[headIdx].length === 0) {
            // Nothing queued – might be done already
            this._checkGlobalCompletion();
            return;
        }

        this.blueProcessing[headIdx]     = true;
        if (typeof this.blueProcessedCount[headIdx] !== 'number') this.blueProcessedCount[headIdx] = 0;

        this._processNextBlueVector(headIdx);
    }

    _processNextBlueVector(headIdx) {
        const queue = this.blueQueues[headIdx];
        if (this.skipRequested) {
            if (Array.isArray(queue)) queue.length = 0;
            this.blueProcessing[headIdx] = false;
            this._markAttentionHeadComplete(headIdx);
            this._checkGlobalCompletion();
            return;
        }
        if (!queue || queue.length === 0) {
            // Finished all blues for this head
            this.blueProcessing[headIdx] = false;
            this._markAttentionHeadComplete(headIdx);
            this._checkGlobalCompletion();
            return;
        }

        // Pop the first (top-most) blue vector
        const vector = queue.shift();

        // i == 1-based index of this vector in processing order (1 for first vector, 2 for second, ...)
        const i = this.blueProcessedCount[headIdx] + 1; // shift to 1-based
        this.blueProcessedCount[headIdx] += 1;
        // Pre-compute sorted lane z positions (top → bottom)
        const laneZs = Array.isArray(this.ctx?.sortedLaneZs) && this.ctx.sortedLaneZs.length
            ? this.ctx.sortedLaneZs
            : (this.ctx.currentLanes || []).map(l => l.zPos).sort((a, b) => a - b);
        const kvCacheDecodeTraversal = !!(
            this.ctx
            && typeof this.ctx._isKvCacheModeEnabled === 'function'
            && this.ctx._isKvCacheModeEnabled()
            && Array.isArray(this.ctx.currentLanes)
            && this.ctx.currentLanes.length === 1
            && laneZs.length > 1
        );
        const hopCount = kvCacheDecodeTraversal ? laneZs.length : i;
        const rowIndex = hopCount - 1;
        if (kvCacheDecodeTraversal && rowIndex > 0) {
            // KV-cache decode should preserve previously completed rows in the
            // attention matrix and only "re-open" the newest row.
            const completed = this.attentionCompletedRows[headIdx] || 0;
            const postCompleted = this.attentionPostCompletedRows[headIdx] || 0;
            this.attentionCompletedRows[headIdx] = Math.max(completed, rowIndex);
            this.attentionPostCompletedRows[headIdx] = Math.max(postCompleted, rowIndex);
        }
        if (vector) {
            vector.userData = vector.userData || {};
            vector.userData.attnRowIndex = rowIndex;
        }
        this._setAttentionProgress(headIdx, rowIndex, -1);

        this._activeBlueVectors[headIdx] = vector;
        this._animateBlueVector(vector, headIdx, hopCount, laneZs, () => {
            delete this._activeBlueVectors[headIdx];
            // Recursive continuation
            this._processNextBlueVector(headIdx);
        });
    }

    _riseSpheres(spheresArr, rowIndex = null, headIdx = null) {
        if (this.skipRequested) return;
        if (!Array.isArray(spheresArr) || spheresArr.length === 0) return;
        const resolvedHeadIdx = Number.isFinite(headIdx)
            ? headIdx
            : (() => {
                const firstId = spheresArr[0];
                const data = this._getAttentionSphereData(firstId);
                const activationData = data ? data.activationData : null;
                return activationData && Number.isFinite(activationData.headIndex) ? activationData.headIndex : null;
            })();
        if (Number.isFinite(resolvedHeadIdx) && Number.isFinite(rowIndex)) {
            this._markAttentionPostRowComplete(resolvedHeadIdx, rowIndex);
        }

        spheresArr.forEach((instanceId) => {
            const data = this._getAttentionSphereData(instanceId);
            if (!data) return;
            const startY = data.position.y;
            const posState = { y: startY };
            new TWEEN.Tween(posState)
                .to({ y: startY + this.RED_EXTRA_RISE }, this.V_RISE_DURATION)
                .easing(TWEEN.Easing.Quadratic.Out)
                .onUpdate(() => {
                    data.position.y = posState.y;
                    this._updateAttentionSphereInstance(instanceId);
                })
                .start();

            const activationData = data.activationData || null;
            const layerIndex = activationData && Number.isFinite(activationData.layerIndex) ? activationData.layerIndex : null;
            const headIdx = activationData && Number.isFinite(activationData.headIndex) ? activationData.headIndex : null;
            const queryTokenIndex = activationData && Number.isFinite(activationData.tokenIndex) ? activationData.tokenIndex : null;
            const keyTokenIndex = activationData && Number.isFinite(activationData.keyTokenIndex) ? activationData.keyTokenIndex : null;
            let postScore = null;
            if (this.ctx && this.ctx.activationSource
                && Number.isFinite(layerIndex) && Number.isFinite(headIdx)
                && Number.isFinite(queryTokenIndex) && Number.isFinite(keyTokenIndex)) {
                const postRow = this._getAttentionScoresRow('post', layerIndex, headIdx, queryTokenIndex);
                postScore = Array.isArray(postRow)
                    ? postRow[Math.max(0, Math.min(postRow.length - 1, keyTokenIndex))]
                    : this.ctx.activationSource.getAttentionScore(layerIndex, 'post', headIdx, queryTokenIndex, keyTokenIndex);
            }
            let targetColor;
            if (Number.isFinite(postScore)) {
                targetColor = mapValueToGrayscale(postScore, { minValue: ATTENTION_POST_SOFTMAX_GRAYSCALE_MIN });
            } else {
                const fallbackLightness = THREE.MathUtils.lerp(0.2, 0.9, Math.random());
                logRandomColorDebug('SelfAttentionAnimator.postScore.randomFallbackColor', {
                    layerIndex,
                    headIndex: headIdx,
                    queryTokenIndex,
                    keyTokenIndex,
                    fallbackLightness
                });
                targetColor = this._sphereColorTmp.setHSL(0, 0, fallbackLightness);
            }

            // Capture current color from the instance buffer if present.
            let curR = 1, curG = 1, curB = 1;
            if (this._sphereMesh && this._sphereMesh.instanceColor && this._sphereMesh.instanceColor.array) {
                const arr = this._sphereMesh.instanceColor.array;
                const idx3 = instanceId * 3;
                curR = arr[idx3] ?? 1;
                curG = arr[idx3 + 1] ?? 1;
                curB = arr[idx3 + 2] ?? 1;
            }
            const colorState = { r: curR, g: curG, b: curB };
            new TWEEN.Tween(colorState)
                .to({ r: targetColor.r, g: targetColor.g, b: targetColor.b }, this.V_RISE_DURATION)
                .easing(TWEEN.Easing.Quadratic.Out)
                .onStart(() => {
                    if (activationData) {
                        activationData.stage = 'attention.post';
                        if (Number.isFinite(postScore)) activationData.postScore = postScore;
                    }
                    data.activationData = activationData;
                    if (this._sphereEntries[instanceId]) {
                        this._sphereEntries[instanceId].activationData = activationData;
                    } else {
                        this._sphereEntries[instanceId] = { activationData };
                    }
                    if (Array.isArray(this._sphereLabels)) {
                        this._sphereLabels[instanceId] = 'Post-Softmax Attention Score';
                    }
                })
                .onUpdate(() => {
                    this._sphereColorTmp.setRGB(colorState.r, colorState.g, colorState.b);
                    this._setAttentionSphereColor(instanceId, this._sphereColorTmp);
                })
                .start();
        });
    }

    /**
     * Helper to uniformly recolor a vector (all prisms) to the given THREE.Color.
     */
    _setVectorColor(vector, color) {
        if (!vector || !vector.mesh) return;
        const col = color instanceof THREE.Color ? color : new THREE.Color(color);
        const count = vector.instanceCount || VECTOR_LENGTH_PRISM;
        for (let idx = 0; idx < count; idx++) {
            if (vector.mesh.setColorAt) {
                vector.mesh.setColorAt(idx, col);
            }
        }
        if (vector.mesh.instanceColor) {
            vector.mesh.instanceColor.needsUpdate = true;
        }
        const csAttr = vector.mesh.geometry.getAttribute('colorStart');
        const ceAttr = vector.mesh.geometry.getAttribute('colorEnd');
        if (csAttr && ceAttr) {
            for (let idx = 0; idx < count; idx++) {
                csAttr.setXYZ(idx, col.r, col.g, col.b);
                ceAttr.setXYZ(idx, col.r, col.g, col.b);
            }
            csAttr.needsUpdate = true;
            ceAttr.needsUpdate = true;
        }
    }

    _sampleVectorFlashColor(vector, out = this._collisionHaloColorTmp) {
        const target = out || new THREE.Color();
        target.set(SA_DUPLICATE_SCORE_COLLISION_HALO_COLOR);

        const mesh = vector && vector.mesh;
        if (!mesh) return target;

        const sampleColorBuffer = (arr) => {
            if (!arr || !arr.length) return false;
            const available = Math.floor(arr.length / 3);
            const requestedCount = Number.isFinite(vector?.instanceCount) ? vector.instanceCount : available;
            const count = Math.min(available, requestedCount);
            if (count <= 0) return false;

            let sumR = 0;
            let sumG = 0;
            let sumB = 0;
            let used = 0;
            for (let i = 0; i < count; i += 1) {
                const i3 = i * 3;
                const r = arr[i3];
                const g = arr[i3 + 1];
                const b = arr[i3 + 2];
                if (!Number.isFinite(r) || !Number.isFinite(g) || !Number.isFinite(b)) continue;
                // Skip hidden/black prisms so the flash follows visible value-vector color.
                if (Math.max(r, g, b) < 0.03) continue;
                sumR += r;
                sumG += g;
                sumB += b;
                used += 1;
            }
            if (used > 0) {
                target.setRGB(sumR / used, sumG / used, sumB / used);
                return true;
            }
            const mid = Math.max(0, Math.min(count - 1, Math.floor(count * 0.5)));
            const i3 = mid * 3;
            const r = arr[i3];
            const g = arr[i3 + 1];
            const b = arr[i3 + 2];
            if (!Number.isFinite(r) || !Number.isFinite(g) || !Number.isFinite(b)) return false;
            target.setRGB(r, g, b);
            return true;
        };

        if (sampleColorBuffer(mesh.instanceColor?.array)) return target;
        if (sampleColorBuffer(mesh.geometry?.getAttribute?.('colorStart')?.array)) return target;
        if (sampleColorBuffer(mesh.geometry?.getAttribute?.('colorEnd')?.array)) return target;
        if (mesh.material?.color?.isColor) {
            target.copy(mesh.material.color);
            return target;
        }
        if (mesh.material?.emissive?.isColor) {
            target.copy(mesh.material.emissive);
            return target;
        }
        return target;
    }

    _applyValueVectorScheme(vector, sourceData = null, options = {}) {
        if (!vector) return;
        const outputLength = (this.ctx && this.ctx.outputVectorLength) ? this.ctx.outputVectorLength : 64;
        const setHiddenToBlack = options && typeof options.setHiddenToBlack === 'boolean'
            ? options.setHiddenToBlack
            : true;
        const disableCache = options && options.disableCache === true;
        const cacheKeyData = options && Object.prototype.hasOwnProperty.call(options, 'cacheKeyData')
            ? options.cacheKeyData
            : null;
        const raw = Array.isArray(sourceData)
            ? sourceData.slice(0, outputLength)
            : (vector.rawData ? vector.rawData.slice(0, outputLength) : []);
        const rangeOptions = this._valueHueRangeOptions || buildHueRangeOptions(MHA_VALUE_SPECTRUM_COLOR, {
            hueSpread: MHA_VALUE_HUE_SPREAD,
            minLightness: MHA_VALUE_LIGHTNESS_MIN,
            maxLightness: MHA_VALUE_LIGHTNESS_MAX,
            valueMin: MHA_VALUE_RANGE_MIN,
            valueMax: MHA_VALUE_RANGE_MAX,
            valueClampMax: MHA_VALUE_CLAMP_MAX,
        });
        const numKeyColors = raw.length <= 1
            ? 1
            : Math.min(MHA_VALUE_KEY_COLOR_COUNT, raw.length);
        vector.applyProcessedVisuals(
            raw,
            outputLength,
            { numKeyColors, generationOptions: rangeOptions },
            { setHiddenToBlack },
            disableCache ? null : (cacheKeyData !== null ? cacheKeyData : raw)
        );
        if (raw.length === 1 && typeof vector.setUniformColor === 'function') {
            const rangeColor = mapValueToHueRange(raw[0], rangeOptions);
            vector.setUniformColor(rangeColor);
        }
        vector.userData = vector.userData || {};
        vector.userData.weightedSumVisuals = {
            numKeyColors,
            generationOptions: rangeOptions,
            outputLength
        };
    }

    _applyQueryVectorScheme(vector, sourceData = null) {
        if (!vector || typeof vector.applyProcessedVisuals !== 'function') return;
        const outputLength = (this.ctx && this.ctx.outputVectorLength) ? this.ctx.outputVectorLength : 64;
        const rawSource = (Array.isArray(sourceData) || ArrayBuffer.isView(sourceData))
            ? sourceData
            : ((Array.isArray(vector.rawData) || ArrayBuffer.isView(vector.rawData)) ? vector.rawData : []);
        const raw = typeof rawSource.slice === 'function'
            ? rawSource.slice(0, outputLength)
            : Array.from(rawSource).slice(0, outputLength);
        const baseColor = this.ctx?.finalHeadColors?.Q
            || this.ctx?.brightBlue
            || new THREE.Color(0x4b9bff);
        const rangeOptions = this._queryHueRangeOptions || buildHueRangeOptions(baseColor, {
            hueSpread: MHA_VALUE_HUE_SPREAD,
            minLightness: MHA_VALUE_LIGHTNESS_MIN,
            maxLightness: MHA_VALUE_LIGHTNESS_MAX,
            valueMin: MHA_VALUE_RANGE_MIN,
            valueMax: MHA_VALUE_RANGE_MAX,
            valueClampMax: MHA_VALUE_CLAMP_MAX,
        });
        const numKeyColors = raw.length <= 1
            ? 1
            : Math.min(MHA_VALUE_KEY_COLOR_COUNT, raw.length);
        vector.applyProcessedVisuals(
            raw,
            outputLength,
            { numKeyColors, generationOptions: rangeOptions },
            {
                setHiddenToBlack: false,
                hideByScaleOnly: true
            },
            raw
        );
        if (raw.length === 1 && typeof vector.setUniformColor === 'function') {
            const rangeColor = mapValueToHueRange(raw[0], rangeOptions);
            vector.setUniformColor(rangeColor);
        }
        vector.userData = vector.userData || {};
        vector.userData.qkvProcessed = true;
        vector.userData.qkvOutputLength = outputLength;
        vector.userData.qkvProcessedCategory = 'Q';
        vector.userData.vectorCategory = 'Q';
        if (vector.group) {
            vector.group.userData = vector.group.userData || {};
            vector.group.userData.label = 'Query Vector';
            if (Number.isFinite(vector.userData?.headIndex)) {
                vector.group.userData.headIndex = vector.userData.headIndex;
            }
            if (Number.isFinite(this.ctx?.layerIndex)) {
                vector.group.userData.layerIndex = this.ctx.layerIndex;
            }
        }
        if (vector.mesh) {
            vector.mesh.userData = {
                ...(vector.mesh.userData || {}),
                label: 'Query Vector'
            };
        }
    }

    _applyWeightedSumScheme(vector, sourceData = null, options = {}) {
        const mergedOptions = {
            ...options,
            disableCache: true,
        };
        this._applyValueVectorScheme(vector, sourceData, mergedOptions);
    }

    _tagWeightedSumVector(vector, lane = null) {
        if (!vector) return;
        vector.userData = vector.userData || {};
        vector.userData.isWeightedSum = true;
        const tokenIndex = Number.isFinite(lane?.tokenIndex) ? Math.floor(lane.tokenIndex) : null;
        const tokenLabel = (lane && typeof lane.tokenLabel === 'string') ? lane.tokenLabel : null;
        if (Number.isFinite(tokenIndex)) vector.userData.tokenIndex = tokenIndex;
        else delete vector.userData.tokenIndex;
        if (tokenLabel) vector.userData.tokenLabel = tokenLabel;
        else delete vector.userData.tokenLabel;
        const label = 'Attention Weighted Sum';
        if (vector.group) {
            vector.group.userData = vector.group.userData || {};
            vector.group.userData.label = label;
            vector.group.userData.isWeightedSum = true;
            if (Number.isFinite(tokenIndex)) vector.group.userData.tokenIndex = tokenIndex;
            else delete vector.group.userData.tokenIndex;
            if (tokenLabel) vector.group.userData.tokenLabel = tokenLabel;
            else delete vector.group.userData.tokenLabel;
        }
        if (vector.mesh) {
            vector.mesh.userData = vector.mesh.userData || {};
            vector.mesh.userData.label = label;
            vector.mesh.userData.isWeightedSum = true;
            if (Number.isFinite(tokenIndex)) vector.mesh.userData.tokenIndex = tokenIndex;
            else delete vector.mesh.userData.tokenIndex;
            if (tokenLabel) vector.mesh.userData.tokenLabel = tokenLabel;
            else delete vector.mesh.userData.tokenLabel;
        }
    }

    _resolveOutputVectorLength() {
        const length = Number.isFinite(this.ctx?.outputVectorLength)
            ? this.ctx.outputVectorLength
            : 64;
        return Math.max(1, Math.floor(length));
    }

    _sliceVectorData(sourceData, outputLength = null) {
        const isArrayLike = Array.isArray(sourceData) || ArrayBuffer.isView(sourceData);
        if (!isArrayLike) return null;
        const arr = Array.from(sourceData).map((value) => (Number.isFinite(value) ? value : 0));
        if (!Number.isFinite(outputLength)) return arr;
        return arr.slice(0, Math.max(1, Math.floor(outputLength)));
    }

    _buildWeightedValueData(sourceData, weight, outputLength = null) {
        const base = this._sliceVectorData(sourceData, outputLength);
        if (!Array.isArray(base)) return null;
        const scalar = Number.isFinite(weight) ? weight : 1;
        return base.map((value) => value * scalar);
    }

    _tagConveyorValueVector(vector, {
        lane = null,
        queryLane = null,
        headIdx = null,
        weighted = false,
        sourceData = null,
        valuesOverride = null,
        postScore = null,
    } = {}) {
        if (!vector) return;
        const outputLength = this._resolveOutputVectorLength();
        const valueTokenIndex = Number.isFinite(lane?.tokenIndex) ? lane.tokenIndex : null;
        const valueTokenLabel = (lane && typeof lane.tokenLabel === 'string') ? lane.tokenLabel : null;
        const queryTokenIndex = Number.isFinite(queryLane?.tokenIndex) ? queryLane.tokenIndex : null;
        const queryTokenLabel = (queryLane && typeof queryLane.tokenLabel === 'string') ? queryLane.tokenLabel : null;
        const layerIndex = Number.isFinite(this.ctx?.layerIndex) ? this.ctx.layerIndex : null;
        const labelBase = weighted ? 'Weighted Value Vector' : 'Value Vector';
        const label = valueTokenLabel ? `${labelBase} - ${valueTokenLabel}` : labelBase;
        const stage = weighted ? 'attention.weighted_value' : 'qkv.v';
        const values = Array.isArray(valuesOverride)
            ? valuesOverride.slice()
            : this._sliceVectorData(sourceData || vector.rawData, outputLength);
        const activationData = buildActivationData({
            label,
            stage,
            layerIndex,
            tokenIndex: valueTokenIndex,
            tokenLabel: valueTokenLabel,
            headIndex: Number.isFinite(headIdx) ? headIdx : undefined,
            keyTokenIndex: valueTokenIndex,
            keyTokenLabel: valueTokenLabel,
            postScore,
            values,
            copyValues: false,
        });
        if (activationData) {
            if (Number.isFinite(queryTokenIndex)) activationData.queryTokenIndex = queryTokenIndex;
            if (queryTokenLabel) activationData.queryTokenLabel = queryTokenLabel;
        }
        applyActivationDataToVector(vector, activationData, label);
    }

    _buildWeightedSumData(headIdx, queryLane, lanes, outputLength) {
        const activationSource = this.ctx && this.ctx.activationSource ? this.ctx.activationSource : null;
        const layerIndex = Number.isFinite(this.ctx?.layerIndex) ? this.ctx.layerIndex : null;
        const queryTokenIndex = queryLane && Number.isFinite(queryLane.tokenIndex) ? queryLane.tokenIndex : null;
        if (Number.isFinite(layerIndex) && Number.isFinite(headIdx) && Number.isFinite(queryTokenIndex)) {
            const cacheKey = `${layerIndex}|${headIdx}|${queryTokenIndex}|${outputLength}`;
            const cached = this._weightedSumCache.get(cacheKey);
            if (cached) return cached;
        }
        const data = new Array(outputLength).fill(0);
        let usedWeight = false;
        let missingData = false;

        if (activationSource && Number.isFinite(layerIndex) && Number.isFinite(queryTokenIndex)) {
            const weightRow = this._getAttentionScoresRow('post', layerIndex, headIdx, queryTokenIndex);
            lanes.forEach((keyLane) => {
                const keyTokenIndex = keyLane && Number.isFinite(keyLane.tokenIndex) ? keyLane.tokenIndex : null;
                if (!Number.isFinite(keyTokenIndex)) return;
                const weight = Array.isArray(weightRow)
                    ? weightRow[keyTokenIndex]
                    : (activationSource.getAttentionScore
                        ? activationSource.getAttentionScore(layerIndex, 'post', headIdx, queryTokenIndex, keyTokenIndex)
                        : null);
                if (!Number.isFinite(weight)) return;
                const vObj = getSideCopyEntry(keyLane, headIdx, 'V');
                const vVec = vObj && vObj.vec ? vObj.vec : null;
                const vData = vVec && Array.isArray(vVec.rawData) ? vVec.rawData : null;
                if (!vData || !vData.length) {
                    missingData = true;
                    return;
                }
                usedWeight = true;
                for (let i = 0; i < outputLength; i++) {
                    data[i] += weight * (vData[i] ?? 0);
                }
            });
        }

        if (!usedWeight) {
            const fallbackObj = getSideCopyEntry(queryLane, headIdx, 'V');
            const fallbackVec = fallbackObj && fallbackObj.vec ? fallbackObj.vec : null;
            const fallbackData = fallbackVec && Array.isArray(fallbackVec.rawData) ? fallbackVec.rawData : null;
            if (fallbackData && fallbackData.length) {
                return fallbackData.slice(0, outputLength);
            }
            return data;
        }

        if (usedWeight && !missingData && Number.isFinite(layerIndex) && Number.isFinite(headIdx) && Number.isFinite(queryTokenIndex)) {
            const cacheKey = `${layerIndex}|${headIdx}|${queryTokenIndex}|${outputLength}`;
            this._weightedSumCache.set(cacheKey, data);
        }

        return data;
    }

    _ensureRunningSumData(vector, outputLength) {
        if (!vector) return null;
        const length = Number.isFinite(outputLength)
            ? Math.max(1, Math.floor(outputLength))
            : (vector.rawData ? vector.rawData.length : 0);
        if (length <= 0) return null;
        vector.userData = vector.userData || {};
        const current = vector.userData.runningSumData;
        if (!Array.isArray(current) || current.length !== length) {
            vector.userData.runningSumData = new Array(length).fill(0);
        }
        return vector.userData.runningSumData;
    }

    _accumulateWeightedSum(vector, sourceData, weight, outputLength) {
        const isArrayLike = Array.isArray(sourceData) || ArrayBuffer.isView(sourceData);
        if (!vector || !isArrayLike || !Number.isFinite(weight)) return;
        const len = Number.isFinite(outputLength)
            ? Math.max(1, Math.floor(outputLength))
            : Math.min(sourceData.length, vector.rawData ? vector.rawData.length : sourceData.length);
        if (len <= 0) return;
        const running = this._ensureRunningSumData(vector, len);
        if (!running) return;
        for (let i = 0; i < len; i++) {
            running[i] += weight * (sourceData[i] ?? 0);
        }
        this._applyWeightedSumScheme(vector, running, { setHiddenToBlack: true });
    }

    _createWeightedSumsImmediate(options = {}) {
        const ctx = this.ctx;
        const queryLanes = Array.isArray(ctx?.currentLanes) ? ctx.currentLanes : [];
        if (!queryLanes.length) return 0;
        const conveyorLanes = (ctx && typeof ctx.getAttentionConveyorLanes === 'function')
            ? ctx.getAttentionConveyorLanes()
            : queryLanes;
        const kvLanes = Array.isArray(conveyorLanes) && conveyorLanes.length
            ? conveyorLanes
            : queryLanes;
        const headCount = Array.isArray(ctx?.headCoords) ? ctx.headCoords.length : 0;
        if (!headCount) return 0;
        const outputLength = Number.isFinite(ctx?.outputVectorLength) ? ctx.outputVectorLength : 64;
        const replaceExisting = options && options.replaceExisting !== false;

        if (replaceExisting) {
            try { ctx && ctx._clearTempDecoratives && ctx._clearTempDecoratives(); } catch (_) {}
            try { ctx && ctx._clearWeightedSumVectors && ctx._clearWeightedSumVectors(); } catch (_) {}
        }

        let created = 0;
        const dockOffset = Number.isFinite(ctx?.weightedSumDockOffset) ? ctx.weightedSumDockOffset : MHA_WEIGHTED_SUM_DOCK_OFFSET;
        const fallbackBaseY = Number.isFinite(ctx?.mhaPassThroughTargetY) && Number.isFinite(ctx?.mhaResultRiseOffsetY)
            ? ctx.mhaPassThroughTargetY + ctx.mhaResultRiseOffsetY - 30 + this.RED_EXTRA_RISE
            : (this.RED_EXTRA_RISE || 0);

        queryLanes.forEach((lane) => {
            const laneZ = Number.isFinite(lane?.zPos) ? lane.zPos : null;
            for (let headIdx = 0; headIdx < headCount; headIdx++) {
                const vObj = getSideCopyEntry(lane, headIdx, 'V');
                const vVec = vObj && vObj.vec ? vObj.vec : null;
                const instanceCount = Number.isFinite(vVec?.instanceCount)
                    ? vVec.instanceCount
                    : (Number.isFinite(ctx?.vectorPrismCount) ? ctx.vectorPrismCount : undefined);
                const targetX = Number.isFinite(vVec?.group?.position?.x)
                    ? vVec.group.position.x
                    : (ctx?.headCoords && ctx.headCoords[headIdx] ? ctx.headCoords[headIdx].v : 0);
                const baseY = Number.isFinite(vVec?.group?.position?.y)
                    ? vVec.group.position.y
                    : fallbackBaseY;
                const targetY = baseY + dockOffset;
                const zPos = Number.isFinite(laneZ)
                    ? laneZ
                    : (Number.isFinite(vVec?.group?.position?.z) ? vVec.group.position.z : 0);

                const data = this._buildWeightedSumData(headIdx, lane, kvLanes, outputLength);
                const spawnPos = new THREE.Vector3(targetX, targetY, zPos);
                const wsVec = new VectorVisualizationInstancedPrism(
                    data.slice(),
                    spawnPos,
                    3,
                    instanceCount
                );
                wsVec.userData = wsVec.userData || {};
                wsVec.userData.headIndex = headIdx;
                wsVec.userData.parentLane = lane || null;
                wsVec.userData.weightedSumLaneZ = zPos;
                wsVec.userData.weightedSumReadyForConcat = true;
                wsVec.userData.weightedSumDocked = true;
                this._tagWeightedSumVector(wsVec, lane);
                if (ctx && ctx.parentGroup) {
                    ctx.parentGroup.add(wsVec.group);
                }
                this._applyWeightedSumScheme(wsVec, data);
                try {
                    if (ctx && typeof ctx.registerWeightedSumVector === 'function') {
                        ctx.registerWeightedSumVector(wsVec, zPos, headIdx, lane);
                    }
                } catch (_) { /* optional */ }
                created += 1;
            }
        });

        return created;
    }

    _copyVectorAppearance(target, source) {
        if (!target || !target.mesh || !source || !source.mesh) return;
        const srcMesh = source.mesh;
        const dstMesh = target.mesh;
        const tmpMat = new THREE.Matrix4();
        const instCount = Math.min(target.instanceCount || 0, source.instanceCount || 0);
        for (let i = 0; i < instCount; i++) {
            srcMesh.getMatrixAt(i, tmpMat);
            dstMesh.setMatrixAt(i, tmpMat);
        }
        if (dstMesh.instanceMatrix) dstMesh.instanceMatrix.needsUpdate = true;
        const srcCS = srcMesh.geometry?.getAttribute?.('colorStart');
        const srcCE = srcMesh.geometry?.getAttribute?.('colorEnd');
        const dstCS = dstMesh.geometry?.getAttribute?.('colorStart');
        const dstCE = dstMesh.geometry?.getAttribute?.('colorEnd');
        if (srcCS && dstCS && srcCS.array && dstCS.array) {
            dstCS.array.set(srcCS.array);
            dstCS.needsUpdate = true;
        }
        if (srcCE && dstCE && srcCE.array && dstCE.array) {
            dstCE.array.set(srcCE.array);
            dstCE.needsUpdate = true;
        }
        if (srcMesh.instanceColor) {
            if (!dstMesh.instanceColor) {
                dstMesh.instanceColor = new THREE.InstancedBufferAttribute(
                    new Float32Array(target.instanceCount * 3),
                    3
                );
            }
            dstMesh.instanceColor.array.set(srcMesh.instanceColor.array);
            dstMesh.instanceColor.needsUpdate = true;
        }
        if (srcMesh.material && dstMesh.material) {
            dstMesh.material.transparent = srcMesh.material.transparent;
            dstMesh.material.opacity = srcMesh.material.opacity;
            if ('emissiveIntensity' in dstMesh.material && 'emissiveIntensity' in srcMesh.material) {
                dstMesh.material.emissiveIntensity = srcMesh.material.emissiveIntensity;
            }
            if (dstMesh.material.emissive && srcMesh.material.emissive) {
                dstMesh.material.emissive.copy(srcMesh.material.emissive);
            }
            dstMesh.material.needsUpdate = true;
        }
    }

    _createQueryShrinkStandIn(source) {
        if (!source || !source.group) return null;
        const standIn = this._acquireDuplicateVector(source);
        if (!standIn || !standIn.group) return null;
        standIn.group.visible = true;
        standIn.group.scale.set(1, 1, 1);
        standIn.group.position.copy(source.group.position);
        standIn.userData = {
            ...(source.userData || {}),
            isPooledDuplicate: true,
        };
        standIn.group.userData = {
            ...(standIn.group.userData || {}),
            ...(source.group?.userData || {}),
            label: 'Query Vector'
        };
        if (source.mesh) {
            this._copyVectorAppearance(standIn, source);
        } else {
            this._applyQueryVectorScheme(standIn, source.rawData);
        }
        if (this.ctx?.parentGroup && standIn.group.parent !== this.ctx.parentGroup) {
            this.ctx.parentGroup.add(standIn.group);
        }
        this._spawnedTempVectors.add(standIn);
        return standIn;
    }

    _animateQueryVectorShrinkOut(vector, onComplete) {
        if (!vector) {
            onComplete && onComplete();
            return;
        }
        if (typeof TWEEN === 'undefined') {
            this._retireVector(vector);
            onComplete && onComplete();
            return;
        }
        const duration = Math.max(120, Math.floor(this.BLUE_HORIZ_DURATION * 0.45));
        const liftY = 10;
        const easing = TWEEN.Easing.Cubic.In;
        const runShrinkTween = (targetVector, finish) => {
            if (!targetVector || !targetVector.group) {
                finish && finish();
                return;
            }
            const startY = targetVector.group.position.y;
            const state = {
                s: Number.isFinite(targetVector.group.scale?.x) ? Math.max(0.001, targetVector.group.scale.x) : 1,
                y: startY
            };
            new TWEEN.Tween(state)
                .to({ s: 0.001, y: startY + liftY }, duration)
                .easing(easing)
                .onUpdate(() => {
                    if (!targetVector.group) return;
                    targetVector.group.scale.set(state.s, state.s, state.s);
                    targetVector.group.position.y = state.y;
                })
                .onComplete(() => {
                    finish && finish();
                })
                .start();
        };

        if (vector.isBatchedVectorRef) {
            const standIn = this._createQueryShrinkStandIn(vector);
            this._retireVector(vector);
            if (!standIn) {
                onComplete && onComplete();
                return;
            }
            runShrinkTween(standIn, () => {
                this._spawnedTempVectors.delete(standIn);
                this._releaseDuplicateVector(standIn);
                onComplete && onComplete();
            });
            return;
        }

        runShrinkTween(vector, () => {
            this._retireVector(vector);
            onComplete && onComplete();
        });
    }

    _animateBlueVector(vector, headIdx, i, laneZs, allDoneCb) {
        if (this.skipRequested) {
            this._finishBlueImmediately(vector, headIdx, allDoneCb);
            return;
        }
        // Note: indices now start at 1, so the former i==0 no-movement case will not occur.
        const horizontalToK = this.ctx.headCoords && this.ctx.headCoords[headIdx]
            ? this.ctx.headCoords[headIdx].k
            : vector.group.position.x;
        const firstLaneZ = Array.isArray(laneZs) && laneZs.length
            ? laneZs[0]
            : vector.group.position.z;


        // Convenience alias for durations / easing
        const QEasing = TWEEN.Easing.Quadratic.InOut;

        const startLaneTraversal = () => {
            if (this.skipRequested) {
                this._finishBlueImmediately(vector, headIdx, allDoneCb);
                return;
            }
            // 3. Traverse along K vectors i times
            const spheres = [];
            this._traverseLanes(vector, laneZs, i, spheres, true, () => {
                this._markAttentionRowComplete(headIdx, i - 1);
                // Lift spheres upward to align with red vectors
                this._riseSpheres(spheres, i - 1, headIdx);
                // 4. Shrink the query vector away at the last K stop, then
                // continue the weighted-sum traversal over the V column.
                this._animateQueryVectorShrinkOut(vector, () => {
                    this._startRedTraversalFromFirstCopy(headIdx, i, laneZs, spheres, allDoneCb);
                });
            });
        };

        // ------------------------------------------------------------------
        // 1. Move into the first lane depth, then slide across to the K column.
        //    Drive the full L path with one tween so the corner does not pause.
        // ------------------------------------------------------------------
        const startX = vector.group.position.x;
        const startZ = vector.group.position.z;
        const needsVerticalEntry = Math.abs(startZ - firstLaneZ) > 0.0005;
        const needsHorizontalEntry = Math.abs(startX - horizontalToK) > 0.0005;
        if (!needsVerticalEntry && !needsHorizontalEntry) {
            startLaneTraversal();
            return;
        }

        const verticalDuration = needsVerticalEntry ? this.BLUE_VERT_DURATION * BLUE_ENTRY_DURATION_MULT : 0;
        const horizontalDuration = needsHorizontalEntry ? this.BLUE_HORIZ_DURATION * BLUE_ENTRY_DURATION_MULT : 0;
        const totalEntryDuration = Math.max(1, verticalDuration + horizontalDuration);
        const cornerProgress = totalEntryDuration > 0 ? (verticalDuration / totalEntryDuration) : 0;
        const entryState = { progress: 0 };

        new TWEEN.Tween(entryState)
            .to({ progress: 1 }, totalEntryDuration)
            .easing(QEasing)
            .onUpdate(() => {
                const progress = THREE.MathUtils.clamp(entryState.progress, 0, 1);
                if (!needsVerticalEntry) {
                    vector.group.position.z = firstLaneZ;
                } else if (progress <= cornerProgress) {
                    const verticalT = cornerProgress > 0 ? (progress / cornerProgress) : 1;
                    vector.group.position.z = THREE.MathUtils.lerp(startZ, firstLaneZ, verticalT);
                } else {
                    vector.group.position.z = firstLaneZ;
                }

                if (!needsHorizontalEntry) {
                    vector.group.position.x = horizontalToK;
                } else if (!needsVerticalEntry) {
                    vector.group.position.x = THREE.MathUtils.lerp(startX, horizontalToK, progress);
                } else if (progress > cornerProgress) {
                    const horizontalSpan = Math.max(1e-6, 1 - cornerProgress);
                    const horizontalT = (progress - cornerProgress) / horizontalSpan;
                    vector.group.position.x = THREE.MathUtils.lerp(startX, horizontalToK, horizontalT);
                } else {
                    vector.group.position.x = startX;
                }
                this._markVectorLayoutDirty(vector);
            })
            .onComplete(() => {
                vector.group.position.z = firstLaneZ;
                vector.group.position.x = horizontalToK;
                this._markVectorLayoutDirty(vector);
                if (this.skipRequested) {
                    this._finishBlueImmediately(vector, headIdx, allDoneCb);
                    return;
                }
                startLaneTraversal();
            })
            .start();
    }

    _startRedTraversalFromFirstCopy(headIdx, hopCount, laneZs, spheresArr, doneCb) {
        if (this.skipRequested) {
            doneCb && doneCb();
            return;
        }
        if (!this.ctx.currentLanes || laneZs.length === 0) {
            doneCb && doneCb();
            return;
        }
        const queryLaneZ = laneZs[Math.min(Math.max(0, hopCount - 1), laneZs.length - 1)];
        const queryLane = this._getLaneForZ(queryLaneZ);
        const topLaneZ = laneZs[0];
        const topLane = this._getLaneForZ(topLaneZ);
        if (!topLane || !Array.isArray(topLane.sideCopies)) {
            doneCb && doneCb();
            return;
        }
        const fixedObj = getSideCopyEntry(topLane, headIdx, 'V');
        if (!fixedObj || !fixedObj.vec) {
            doneCb && doneCb();
            return;
        }
        const fixedVec = fixedObj.vec;
        const fixedPosInCtx = this._getVectorPositionInContextSpace(fixedVec, this._tmpCtxPosA);
        // Spawn travelling red vector OVER K column (horizontally offset) and ABOVE green vectors.
        const kX = (this.ctx.headCoords && this.ctx.headCoords[headIdx]) ? this.ctx.headCoords[headIdx].k : fixedVec.group.position.x;
        // Set vertical position to the **canonical** raised-V height so it always matches
        // fixed red vectors and highlight spheres, even if the fixed copy is still
        // mid-animation.
        const spawnY = this.ctx.mhaPassThroughTargetY + this.ctx.mhaResultRiseOffsetY - 30 + this.RED_EXTRA_RISE;
        const spawnPos = new THREE.Vector3(
            kX,
            spawnY,
            Number.isFinite(fixedPosInCtx?.z) ? fixedPosInCtx.z : fixedVec.group.position.z
        );
        const travellingVec = new VectorVisualizationInstancedPrism(
            fixedVec.rawData.slice(),
            spawnPos,
            3,
            fixedVec.instanceCount
        );
        travellingVec.userData = { headIndex: headIdx, parentLane: queryLane, attnRowIndex: hopCount - 1 };
        this._tagWeightedSumVector(travellingVec, queryLane);
        travellingVec.userData.weightedSumLaneZ = queryLaneZ;
        travellingVec.userData.weightedSumReadyForConcat = false;
        travellingVec.userData.weightedSumDocked = false;
        this._activeBlueVectors[headIdx] = travellingVec;
        this.ctx.parentGroup.add(travellingVec.group);
        this._spawnedTempVectors.add(travellingVec);
        const outputLength = Number.isFinite(this.ctx?.outputVectorLength) ? this.ctx.outputVectorLength : 64;
        const runningSum = this._ensureRunningSumData(travellingVec, outputLength);
        if (runningSum) {
            this._applyWeightedSumScheme(travellingVec, runningSum);
        } else {
            this._applyWeightedSumScheme(travellingVec, fixedVec.rawData);
        }
        // Start invisible – will be revealed on first merge
        travellingVec.group.scale.set(0.001, 0.001, 0.001);

        // Begin traversal over red vectors
        this._traverseLanes(travellingVec, laneZs, hopCount, spheresArr, false, () => {
            this._parkWeightedSumVector(travellingVec, headIdx, queryLane, queryLaneZ);
            doneCb && doneCb();
        });
    }

    _parkWeightedSumVector(vector, headIdx, lane, laneZ) {
        if (!vector || !vector.group) return;
        const resolvedLaneZ = Number.isFinite(lane?.zPos) ? lane.zPos : (Number.isFinite(laneZ) ? laneZ : vector.group.position.z);
        const vObj = getSideCopyEntry(lane, headIdx, 'V');
        const vVec = vObj && vObj.vec ? vObj.vec : null;
        const vPosInCtx = this._getVectorPositionInContextSpace(vVec, this._tmpCtxPosA);
        const targetX = Number.isFinite(vPosInCtx?.x)
            ? vPosInCtx.x
            : (this.ctx.headCoords && this.ctx.headCoords[headIdx] ? this.ctx.headCoords[headIdx].v : vector.group.position.x);
        const baseY = Number.isFinite(vPosInCtx?.y) ? vPosInCtx.y : vector.group.position.y;
        const dockOffset = Number.isFinite(this.ctx?.weightedSumDockOffset) ? this.ctx.weightedSumDockOffset : MHA_WEIGHTED_SUM_DOCK_OFFSET;
        const targetY = baseY + dockOffset;
        vector.group.scale.set(1, 1, 1);

        vector.userData = vector.userData || {};
        vector.userData.headIndex = headIdx;
        vector.userData.parentLane = lane || vector.userData.parentLane || null;
        vector.userData.weightedSumLaneZ = resolvedLaneZ;
        vector.userData.weightedSumReadyForConcat = false;
        vector.userData.weightedSumDocked = false;
        this._tagWeightedSumVector(vector, lane);

        this._applyWeightedSumScheme(vector);

        // Ensure weighted sums survive cleanup of temp conveyor vectors
        this._spawnedTempVectors.delete(vector);
        if (this._activeBlueVectors && this._activeBlueVectors[headIdx] === vector) {
            delete this._activeBlueVectors[headIdx];
        }

        // Register with parent so it can drive concatenation later
        try {
            if (this.ctx && typeof this.ctx.registerWeightedSumVector === 'function') {
                this.ctx.registerWeightedSumVector(vector, resolvedLaneZ, headIdx, lane);
            }
        } catch (_) { /* optional */ }

        this._pendingDockCount += 1;

        const finalizeDock = () => {
            if (vector.userData) {
                vector.userData.weightedSumDocked = true;
            }
            this._pendingDockCount = Math.max(0, this._pendingDockCount - 1);
            this._checkGlobalCompletion();
        };

        if (typeof TWEEN !== 'undefined') {
            new TWEEN.Tween(vector.group.position)
                .to({ x: targetX, y: targetY, z: resolvedLaneZ }, this.DUPLICATE_TRAVEL_MERGE_MS)
                .easing(TWEEN.Easing.Quadratic.Out)
                .onComplete(() => {
                    finalizeDock();
                })
                .start();
        } else {
            vector.group.position.set(targetX, targetY, resolvedLaneZ);
            finalizeDock();
        }
    }

    _traverseLanes(vector, laneZs, count, spheresArr, createSpheres, doneCb, stepIdx = 0) {
        if (this.skipRequested) {
            return;
        }
        if (count === 0 || stepIdx >= count || laneZs.length === 0) {
            // Nothing to traverse
            doneCb && doneCb();
            return;
        }

        const targetZ = laneZs[stepIdx < laneZs.length ? stepIdx : laneZs.length - 1];
        const prepassSlowMult = createSpheres ? SA_BLUE_PREPASS_SLOW_MULT : 1;
        const laneHopDuration = this.BLUE_VERT_DURATION * prepassSlowMult;
        const lanePauseDuration = this.BLUE_PAUSE_MS * prepassSlowMult;
        new TWEEN.Tween(vector.group.position)
            .to({ z: targetZ }, laneHopDuration)
            .easing(TWEEN.Easing.Quadratic.InOut)
            .onUpdate(() => {
                this._markVectorLayoutDirty(vector);
            })
            .onComplete(() => {
                if (this.skipRequested) return;
                const headIdx = (vector.userData && typeof vector.userData.headIndex === 'number') ? vector.userData.headIndex : null;
                if (createSpheres && Number.isFinite(headIdx)) {
                    const rowIndex = (vector.userData && Number.isFinite(vector.userData.attnRowIndex))
                        ? vector.userData.attnRowIndex
                        : null;
                    if (Number.isFinite(rowIndex)) {
                        this._setAttentionProgress(headIdx, rowIndex, stepIdx);
                    }
                }
                // Create a sphere between blue (vector) and corresponding green vector if enabled
                if (createSpheres) {
                    const headIdx = (vector.userData && typeof vector.userData.headIndex === 'number') ? vector.userData.headIndex : null;
                    if (headIdx !== null && this.ctx.currentLanes) {
                        const lane = this._getLaneForZ(targetZ);
                        if (lane && lane.upwardCopies && lane.upwardCopies[headIdx]) {
                            const greenVec = lane.upwardCopies[headIdx];
                            if (greenVec && greenVec.group) {
                                const greenPos = this._getVectorPositionInContextSpace(greenVec, this._tmpCtxPosA);
                                if (!greenPos) {
                                    this._scheduleAfterDelay(() => {
                                        if (this.skipRequested) {
                                            doneCb && doneCb();
                                            return;
                                        }
                                        this._traverseLanes(vector, laneZs, count, spheresArr, createSpheres, doneCb, stepIdx + 1);
                                    }, lanePauseDuration);
                                    return;
                                }
                                const midPoint = this._tmpMidpoint
                                    .addVectors(vector.group.position, greenPos)
                                    .multiplyScalar(0.5);
                                const queryLane = vector.userData ? vector.userData.parentLane : null;
                                const queryTokenIndex = queryLane && Number.isFinite(queryLane.tokenIndex) ? queryLane.tokenIndex : null;
                                const keyTokenIndex = lane && Number.isFinite(lane.tokenIndex) ? lane.tokenIndex : null;
                                const layerIndex = Number.isFinite(this.ctx?.layerIndex) ? this.ctx.layerIndex : null;
                                const preRow = (this.ctx && this.ctx.activationSource && Number.isFinite(layerIndex) && Number.isFinite(headIdx))
                                    ? this._getAttentionScoresRow('pre', layerIndex, headIdx, queryTokenIndex)
                                    : null;
                                const preScore = Array.isArray(preRow) && preRow.length && Number.isFinite(keyTokenIndex)
                                    ? preRow[Math.max(0, Math.min(preRow.length - 1, keyTokenIndex))]
                                    : ((this.ctx && this.ctx.activationSource && Number.isFinite(layerIndex) && Number.isFinite(headIdx))
                                        ? this.ctx.activationSource.getAttentionScore(layerIndex, 'pre', headIdx, queryTokenIndex, keyTokenIndex)
                                        : null);
                                let baseColor;
                                if (Number.isFinite(preScore)) {
                                    baseColor = mapValueToColor(preScore, { clampMax: 5 }, this._sphereColorTmp);
                                } else {
                                    const hue = Math.random();
                                    const saturation = THREE.MathUtils.lerp(0.85, 1.0, Math.random());
                                    const lightness = THREE.MathUtils.lerp(0.45, 0.6, Math.random());
                                    logRandomColorDebug('SelfAttentionAnimator.preScore.randomFallbackColor', {
                                        layerIndex,
                                        headIndex: headIdx,
                                        queryTokenIndex,
                                        keyTokenIndex,
                                        hue,
                                        saturation,
                                        lightness
                                    });
                                    baseColor = this._sphereColorTmp.setHSL(hue, saturation, lightness);
                                }
                                const activationData = buildActivationData({
                                    label: 'Pre-Softmax Attention Score',
                                    stage: 'attention.pre',
                                    layerIndex,
                                    tokenIndex: queryTokenIndex,
                                    tokenLabel: queryLane ? queryLane.tokenLabel : null,
                                    headIndex: headIdx,
                                    preScore,
                                });
                                if (activationData && Number.isFinite(keyTokenIndex)) {
                                    activationData.keyTokenIndex = keyTokenIndex;
                                    activationData.keyTokenLabel = lane ? lane.tokenLabel : null;
                                }
                                const sphereId = this._acquireAttentionSphere(midPoint, baseColor, activationData);
                                if (Number.isFinite(sphereId)) {
                                    this._tweenSphereScale(sphereId, 0.8, 350, { easing: TWEEN.Easing.Quadratic.Out });
                                    if (Array.isArray(spheresArr)) spheresArr.push(sphereId);
                                }
                            }
                        }
                    }
                }
                // Handle sphere removal during red-vector traversal
                if (!createSpheres && Array.isArray(spheresArr)) {
                    const idx = spheresArr.findIndex((id) => {
                        const data = this._getAttentionSphereData(id);
                        return data && Math.abs(data.position.z - targetZ) < 0.1;
                    });
                    if (idx >= 0) {
                        const sphereId = spheresArr[idx];
                        const spData = this._getAttentionSphereData(sphereId);
                        const spPos = spData ? spData.position : null;
                        const ContinueTraversal = () => {
                            this._traverseLanes(vector, laneZs, count, spheresArr, createSpheres, doneCb, stepIdx + 1);
                        };
                        // ------------------------------------------------------------------
                        //  Spawn duplicate red vector from the fixed V vector and animate it
                        //  through the sphere and into the moving red vector.
                        // ------------------------------------------------------------------
                        const headIdx = (vector.userData && typeof vector.userData.headIndex === 'number') ? vector.userData.headIndex : null;
                        const lane = this._getLaneForZ(targetZ);
                        if (lane && headIdx !== null && Array.isArray(lane.sideCopies)) {
                            const fixedObj = getSideCopyEntry(lane, headIdx, 'V');
                            if (fixedObj && fixedObj.vec) {
                                const fixedVec = fixedObj.vec;
                                const queryLane = vector.userData ? vector.userData.parentLane : null;
                                const fixedPosInCtx = this._getVectorPositionInContextSpace(fixedVec, this._tmpCtxPosB);
                                // Ensure duplicates spawn at the **raised** red-vector height (match the highlight spheres).
                                const raisedY = spPos ? spPos.y : vector.group.position.y;
                                const activationData = spData ? spData.activationData : null;
                                const postScore = activationData && Number.isFinite(activationData.postScore)
                                    ? activationData.postScore
                                    : null;
                                const weight = Number.isFinite(postScore)
                                    ? THREE.MathUtils.clamp(postScore, 0, 1)
                                    : null;
                                const weightScale = 1.0;
                                const weightOpacity = Number.isFinite(weight)
                                    ? THREE.MathUtils.lerp(0.65, 1.0, weight)
                                    : 1.0;
                                const dupVec = this._acquireDuplicateVector(fixedVec);
                                const startPos = fixedPosInCtx
                                    ? fixedPosInCtx.clone()
                                    : fixedVec.group.position.clone();
                                dupVec.group.position.copy(startPos);
                                // Start with the ORIGINAL value-vector look.
                                this._copyVectorAppearance(dupVec, fixedVec);
                                // If the fixed V vector hasn't been collapsed to output length yet (first row can race),
                                // shrink the duplicate to match processed V visuals so it doesn't look oversized.
                                const outputLength = Number.isFinite(this.ctx?.outputVectorLength)
                                    ? this.ctx.outputVectorLength
                                    : 64;
                                const fixedRaw = fixedVec.rawData;
                                const weightedData = this._buildWeightedValueData(fixedRaw, weight, outputLength);
                                const fixedProcessed = !!(fixedVec.userData && fixedVec.userData.qkvProcessed);
                                const hasMesh = !!(fixedVec.mesh && fixedVec.mesh.isMesh);
                                if (!fixedProcessed || !hasMesh || (fixedRaw && fixedRaw.length > outputLength)) {
                                    this._applyValueVectorScheme(dupVec, fixedRaw, {
                                        setHiddenToBlack: false,
                                        cacheKeyData: fixedRaw,
                                    });
                                }
                                this._tagConveyorValueVector(dupVec, {
                                    lane,
                                    queryLane,
                                    headIdx,
                                    weighted: false,
                                    sourceData: fixedRaw
                                });
                                this.ctx.parentGroup.add(dupVec.group);
                                this._spawnedTempVectors.add(dupVec);
                                dupVec.group.scale.set(0.001, 0.001, 0.001);
                                // Optional quick pop-in (scale encodes weight)
                                new TWEEN.Tween(dupVec.group.scale)
                                    .to({ x: 1, y: 1, z: 1 }, this.DUPLICATE_POP_IN_MS)
                                    .easing(TWEEN.Easing.Quadratic.Out)
                                    .start();
                                // Pulse the post-softmax score to visualize weighting
                                if (Number.isFinite(sphereId) && spData) {
                                    const baseScale = Number.isFinite(spData.scale) ? spData.scale : 1;
                                    const pulse = Number.isFinite(weight)
                                        ? THREE.MathUtils.lerp(1.15, 1.9, weight)
                                        : 1.4;
                                    this._tweenSphereScale(sphereId, baseScale * pulse, 180, {
                                        easing: TWEEN.Easing.Quadratic.Out,
                                        yoyo: true,
                                        repeat: 1
                                    });
                                }
                                const sumTarget = { x: vector.group.position.x, y: raisedY, z: vector.group.position.z };
                                const sumDuration = spData
                                    ? this.DUPLICATE_TRAVEL_MERGE_MS * SA_DUPLICATE_TO_SUM_TRAVEL_FRACTION
                                    : this.DUPLICATE_TRAVEL_MERGE_MS;
                                const applyWeightedLook = () => {
                                    const weightedVisualData = Array.isArray(weightedData) ? weightedData : fixedRaw;
                                    this._applyWeightedSumScheme(dupVec, weightedVisualData, { setHiddenToBlack: true });
                                    this._tagConveyorValueVector(dupVec, {
                                        lane,
                                        queryLane,
                                        headIdx,
                                        weighted: true,
                                        sourceData: fixedRaw,
                                        valuesOverride: weightedData,
                                        postScore
                                    });
                                    if (dupVec.mesh && dupVec.mesh.material) {
                                        dupVec.mesh.material.transparent = true;
                                        dupVec.mesh.material.opacity = weightOpacity;
                                        dupVec.mesh.material.needsUpdate = true;
                                    }
                                    // Keep size identical; only opacity changes for weighting.
                                    dupVec.group.scale.set(weightScale, weightScale, weightScale);
                                };
                                const flyToSum = () => {
                                    new TWEEN.Tween(dupVec.group.position)
                                        .to(sumTarget, sumDuration)
                                        .easing(TWEEN.Easing.Quadratic.InOut)
                                        .onComplete(() => {
                                            const outputLength = Number.isFinite(this.ctx?.outputVectorLength)
                                                ? this.ctx.outputVectorLength
                                                : (fixedVec.rawData ? fixedVec.rawData.length : 0);
                                            if (Number.isFinite(weight) && fixedVec.rawData) {
                                                this._accumulateWeightedSum(vector, fixedVec.rawData, weight, outputLength);
                                            }
                                            const pulseSum = () => {
                                                const baseScale = Math.max(1, vector.group.scale.x);
                                                const pulseFactor = Number.isFinite(weight)
                                                    ? THREE.MathUtils.lerp(1.04, 1.14, weight)
                                                    : 1.08;
                                                const state = { s: baseScale };
                                                new TWEEN.Tween(state)
                                                    .to({ s: baseScale * pulseFactor }, 140)
                                                    .easing(TWEEN.Easing.Quadratic.Out)
                                                    .yoyo(true)
                                                    .repeat(1)
                                                    .onUpdate(() => {
                                                        vector.group.scale.set(state.s, state.s, state.s);
                                                    })
                                                    .start();
                                            };
                                        // Reveal travelling red vector on its first merge
                                        if (vector.group.scale.x < 0.5) {
                                            new TWEEN.Tween(vector.group.scale)
                                                .to({ x: 1, y: 1, z: 1 }, this.DUPLICATE_POP_IN_MS)
                                                .easing(TWEEN.Easing.Quadratic.Out)
                                                .onComplete(pulseSum)
                                                .start();
                                        } else {
                                            pulseSum();
                                        }
                                        // Fade / dispose duplicate
                                        new TWEEN.Tween(dupVec.group.scale)
                                            .to({ x: 0.001, y: 0.001, z: 0.001 }, this.DUPLICATE_POP_OUT_MS)
                                            .onComplete(() => {
                                                this._spawnedTempVectors.delete(dupVec);
                                                this._releaseDuplicateVector(dupVec);
                                                // Continue traversal AFTER merge completes
                                                ContinueTraversal();
                                            })
                                            .start();
                                    }).start();
                                };

                                if (spData && spPos) {
                                    const scoreTarget = { x: spPos.x, y: spPos.y, z: spPos.z };
                                    new TWEEN.Tween(dupVec.group.position)
                                        .to(scoreTarget, this.DUPLICATE_TRAVEL_MERGE_MS * SA_DUPLICATE_TO_SCORE_TRAVEL_FRACTION)
                                        .easing(TWEEN.Easing.Quadratic.Out)
                                        .onComplete(() => {
                                            const baseDupScale = Math.max(0.001, Number.isFinite(dupVec.group.scale.x) ? dupVec.group.scale.x : 1);
                                            const collisionPulseFactor = Number.isFinite(weight)
                                                ? THREE.MathUtils.lerp(
                                                    SA_DUPLICATE_SCORE_COLLISION_PULSE_MIN,
                                                    SA_DUPLICATE_SCORE_COLLISION_PULSE_MAX,
                                                    weight
                                                )
                                                : (SA_DUPLICATE_SCORE_COLLISION_PULSE_MIN + SA_DUPLICATE_SCORE_COLLISION_PULSE_MAX) * 0.5;
                                            const collisionPulseDuration = this.DUPLICATE_SCORE_COLLISION_PULSE_MS;
                                            const pulseScaleState = { s: baseDupScale };
                                            new TWEEN.Tween(pulseScaleState)
                                                .to({ s: baseDupScale * collisionPulseFactor }, collisionPulseDuration)
                                                .easing(TWEEN.Easing.Quadratic.Out)
                                                .yoyo(true)
                                                .repeat(1)
                                                .onUpdate(() => {
                                                    dupVec.group.scale.set(pulseScaleState.s, pulseScaleState.s, pulseScaleState.s);
                                                })
                                                .start();
                                            if (Number.isFinite(sphereId)) {
                                                const haloColor = this._sampleVectorFlashColor(dupVec, this._collisionHaloColorTmp);
                                                this._spawnAttentionSphereHalo(
                                                    sphereId,
                                                    collisionPulseDuration * SA_DUPLICATE_SCORE_COLLISION_HALO_DURATION_MULT,
                                                    haloColor
                                                );
                                            }
                                            // Recolor at collision to reflect value * post-softmax weight.
                                            applyWeightedLook();
                                            // Brief linger at the post-softmax score before merging into the running sum
                                            this._scheduleAfterDelay(() => {
                                                if (this.skipRequested) return;
                                                flyToSum();
                                            }, Math.max(80, collisionPulseDuration * 2));
                                        })
                                        .start();
                                } else {
                                    applyWeightedLook();
                                    flyToSum();
                                }
                            }
                        }

                        // Shrink & remove sphere after the weighting merge completes
                        const shrinkDelay = Math.max(250, this.DUPLICATE_TRAVEL_MERGE_MS * 0.8);
                        if (Number.isFinite(sphereId)) {
                            this._tweenSphereScale(sphereId, 0.001, 250, {
                                delay: shrinkDelay,
                                easing: TWEEN.Easing.Quadratic.In,
                                onComplete: () => {
                                    this._releaseAttentionSphere(sphereId);
                                }
                            });
                        }
                        spheresArr.splice(idx, 1);
                        // IMPORTANT: Return here so we do NOT recurse twice.
                        return;
                    }
                }
                // Pause briefly, then recurse
                this._scheduleAfterDelay(() => {
                    if (this.skipRequested) {
                        doneCb && doneCb();
                        return;
                    }
                    this._traverseLanes(vector, laneZs, count, spheresArr, createSpheres, doneCb, stepIdx + 1);
                }, lanePauseDuration);
            })
            .start();
    }

    // ------------------------------------------------------------------
    // Utility helpers for global completion detection
    // ------------------------------------------------------------------
    _isConveyorComplete() {
        const anyProcessing = Object.values(this.blueProcessing).some(v => v);
        const anyQueued = Object.values(this.blueQueues).some(q => q && q.length > 0);
        return !anyProcessing && !anyQueued && this._pendingDockCount === 0;
    }

    _checkGlobalCompletion() {
        if (this.phase === 'complete') return;
        if (this._isConveyorComplete()) {
            this.phase = 'complete';
            this._cleanupAttentionScoreMeshes();
            this.attentionProgress = {};
            this.attentionCompletedRows = {};
            this.attentionPostCompletedRows = {};
            this._attentionRowCache.clear();
            this._weightedSumCache.clear();
            // Notify parent to finalize K/V visuals immediately after the
            // last blue vector finishes its conveyor belt.
            try {
                if (this.ctx && typeof this.ctx._shouldPreserveKVCacheVectors === 'function' && this.ctx._shouldPreserveKVCacheVectors()) {
                    this.ctx._preserveKVVectorsForCache?.();
                } else {
                    this.ctx && this.ctx._disposeMergedKVGroups && this.ctx._disposeMergedKVGroups();
                    this.ctx && this.ctx._disposeAllIndividualKandVVectorsImmediately && this.ctx._disposeAllIndividualKandVVectorsImmediately();
                }
                this.ctx && this.ctx._hideAllQVectorsImmediately && this.ctx._hideAllQVectorsImmediately();
            } catch (_) { /* optional */ }
            this._flushCallbacks();
        }
    }

    // ------------------------------------------------------------------
    // Callback handling utilities
    // ------------------------------------------------------------------
    _flushCallbacks() {
        const list = this._callbacks.splice(0, this._callbacks.length);
        list.forEach(cb => {
            try { cb && cb(); } catch (err) { console.error(err); }
        });
    }
}
