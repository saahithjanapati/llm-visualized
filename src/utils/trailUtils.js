import * as THREE from 'three';
import { Line2 } from 'three/examples/jsm/lines/Line2.js';
import { LineGeometry } from 'three/examples/jsm/lines/LineGeometry.js';
import { LineMaterial } from 'three/examples/jsm/lines/LineMaterial.js';
import { LineSegments2 } from 'three/examples/jsm/lines/LineSegments2.js';
import { LineSegmentsGeometry } from 'three/examples/jsm/lines/LineSegmentsGeometry.js';
import { TRAIL_COLOR, TRAIL_LINE_WIDTH, TRAIL_OPACITY, TRAIL_MAX_SEGMENTS, scaleOpacityForDisplay, scaleLineWidthForDisplay } from './trailConstants.js';
import { HIDE_INSTANCE_Y_OFFSET } from './constants.js';
import { perfStats } from './perfStats.js';

/**
 * StraightLineTrail – memory-efficient trail renderer.
 * ---------------------------------------------------
 * Keeps ONE geometry + ONE material per trail.  As the owning object moves the
 * class appends vertices to a single BufferGeometry instead of allocating a
 * new THREE.Line for every segment.  That eliminates thousands of meshes &
 * materials and dramatically reduces RAM / draw-calls.
 *
 * API is unchanged from the earlier version so existing integration continues
 * to work:  constructor → start(pos) → update(pos) → dispose().
 */
const _snapPrev = new THREE.Vector3();
const _snapDir = new THREE.Vector3();
const _trimLast = new THREE.Vector3();
const _stepTarget = new THREE.Vector3();
const WIDE_LINE_HIDDEN_POSITIONS = new Float32Array([
    0, HIDE_INSTANCE_Y_OFFSET, 0,
    0, HIDE_INSTANCE_Y_OFFSET, 0
]);

let GLOBAL_MAX_STEP_DISTANCE = 0;
const USE_WIDE_TRAIL_LINES = false;

function getWideLineResolution() {
    const width = (typeof window !== 'undefined' && Number.isFinite(window.innerWidth) && window.innerWidth > 0)
        ? window.innerWidth
        : 1920;
    const height = (typeof window !== 'undefined' && Number.isFinite(window.innerHeight) && window.innerHeight > 0)
        ? window.innerHeight
        : 1080;
    return { width, height };
}

function updateWideLineMaterialResolution(material) {
    if (!material || !material.resolution || typeof material.resolution.set !== 'function') return;
    const { width, height } = getWideLineResolution();
    material.resolution.set(width, height);
}

function createTrailMaterial(color, effectiveWidth, effectiveOpacity) {
    if (!USE_WIDE_TRAIL_LINES) {
        return new THREE.LineBasicMaterial({
            color,
            linewidth: effectiveWidth,
            transparent: effectiveOpacity < 1.0,
            opacity: effectiveOpacity,
            depthWrite: false,
            fog: false,
            toneMapped: false
        });
    }
    const material = new LineMaterial({
        color,
        linewidth: effectiveWidth,
        transparent: effectiveOpacity < 1.0,
        opacity: effectiveOpacity,
        depthWrite: false,
        fog: false,
        toneMapped: false
    });
    updateWideLineMaterialResolution(material);
    return material;
}

function applyTrailMaterialScale(material, effectiveOpacity, effectiveWidth) {
    if (!material) return;
    material.opacity = effectiveOpacity;
    material.transparent = effectiveOpacity < 1.0;
    if (Number.isFinite(effectiveWidth) && typeof material.linewidth === 'number') {
        material.linewidth = effectiveWidth;
    }
    updateWideLineMaterialResolution(material);
    material.needsUpdate = true;
}

function resolveTrailPassId(root) {
    let node = root;
    while (node) {
        const passId = node.userData && node.userData.trailPassId;
        if (Number.isFinite(passId)) return passId;
        node = node.parent || null;
    }
    return null;
}

export function setGlobalTrailMaxStepDistance(distance) {
    const next = Number(distance);
    GLOBAL_MAX_STEP_DISTANCE = Number.isFinite(next) && next > 0 ? next : 0;
}

export class StraightLineTrail {
    /**
     * @param {THREE.Object3D} scene        Scene (or Group) to attach the trail.
     * @param {number}         color        Hex colour.
     * @param {number}         lineWidth    Width in pixels (ignored on most HW).
     * @param {number}         maxSegments  Preallocated straight-line segments.
     * @param {number}         opacity      Base opacity before DPR scaling.
     * @param {number}         minSegmentDistance Minimum distance between trail points before recording.
     */
    constructor(scene, color = TRAIL_COLOR, lineWidth = TRAIL_LINE_WIDTH, maxSegments = TRAIL_MAX_SEGMENTS, opacity = TRAIL_OPACITY, minSegmentDistance = 0) {
        this._scene = scene;
        this._color = color;
        this._opacity = opacity;
        this._lineWidth = Number.isFinite(lineWidth) ? lineWidth : TRAIL_LINE_WIDTH;
        this._useWideLine = USE_WIDE_TRAIL_LINES;
        const maxSegmentsSafe = Number.isFinite(maxSegments)
            ? Math.max(1, Math.floor(maxSegments))
            : TRAIL_MAX_SEGMENTS;
        const clampedMin = Math.max(0, minSegmentDistance);
        this._minSegmentDistanceSq = clampedMin * clampedMin;
        this._maxStepDistance = 0;

        // Preallocate vertex buffer (N segments ⇒ N+1 vertices; we duplicate the
        // first vertex to create a zero-length segment so drawRange ≥2).  Each
        // vertex is 3 floats.
        const maxVertices = maxSegmentsSafe + 1;
        this._positions = new Float32Array(maxVertices * 3);
        this._attr = null;
        if (this._useWideLine) {
            this._geometry = new LineGeometry();
            this._geometry.setPositions(WIDE_LINE_HIDDEN_POSITIONS);
        } else {
            this._attr = new THREE.BufferAttribute(this._positions, 3).setUsage(THREE.DynamicDrawUsage);
            this._geometry = new THREE.BufferGeometry();
            this._geometry.setAttribute('position', this._attr);
            this._geometry.setDrawRange(0, 0); // nothing yet
        }

        const effectiveOpacity = scaleOpacityForDisplay(this._opacity);
        const effectiveWidth = scaleLineWidthForDisplay(this._lineWidth);
        this._material = createTrailMaterial(this._color, effectiveWidth, effectiveOpacity);
        // Ensure trails stay visible during camera fly-throughs by bypassing frustum
        // culling; depth testing remains enabled so solids continue to occlude trails.
        this._line = this._useWideLine
            ? new Line2(this._geometry, this._material)
            : new THREE.Line(this._geometry, this._material);
        this._line.frustumCulled = false;
        // Tag for discovery and back-reference
        this._line.userData.isTrail = true;
        this._line.userData.trailBaseOpacity = this._opacity;
        this._line.userData.trailBaseLineWidth = this._lineWidth;
        const passId = resolveTrailPassId(scene);
        if (Number.isFinite(passId)) {
            this._line.userData.trailPassId = passId;
        }
        this._line.userData.trailRef = this;
        // Trails are decorative; skip raycast intersection work.
        this._line.raycast = () => {};
        scene.add(this._line);

        this._vertexCount = 0;
        this._prevPos = new THREE.Vector3();
        this._currentDir = null;
        this._tmpDir = new THREE.Vector3();
        this._updateCounter = 0; // for deferred bounding-sphere update
    }

    /**
     * Reparent the trail's THREE.Line to a new scene/group so future updates
     * continue in the correct local coordinate space (useful when vectors are
     * transferred between layer roots).
     * @param {THREE.Object3D} newScene
     */
    reparent(newScene) {
        if (!newScene || newScene === this._scene) return;
        if (this._line && this._line.parent) {
            this._line.parent.remove(this._line);
        }
        newScene.add(this._line);
        this._scene = newScene;
    }

    // ---------------------------------------------------------------------
    // Public API – same as previous implementation
    // ---------------------------------------------------------------------

    start(pos) {
        if (this._vertexCount) return; // already started
        // Duplicate first vertex so we have at least 2 for Line rendering
        this._writeVertex(0, pos);
        this._writeVertex(1, pos);
        this._vertexCount = 2;
        this._syncGeometryFromPositions();
        this._prevPos.copy(pos);
        this._currentDir = null;
    }

    /**
     * Clear existing segments and restart the trail from the supplied position.
     * Useful when reassigning a trail to a different vector so history does not
     * overlap and appear brighter.
     */
    resetToPosition(pos) {
        if (!pos) return;
        this._vertexCount = 0;
        this._currentDir = null;
        this._prevPos.copy(pos);
        this._writeVertex(0, pos);
        this._writeVertex(1, pos);
        this._vertexCount = 2;
        this._syncGeometryFromPositions();
    }

    update(pos) {
        if (this._vertexCount === 0) return; // not started yet
        if (!pos) return;

        const maxStep = this._maxStepDistance > 0 ? this._maxStepDistance : GLOBAL_MAX_STEP_DISTANCE;
        let targetPos = pos;
        if (maxStep > 0) {
            const dx = pos.x - this._prevPos.x;
            const dy = pos.y - this._prevPos.y;
            const dz = pos.z - this._prevPos.z;
            const distSq = dx * dx + dy * dy + dz * dz;
            if (distSq > maxStep * maxStep) {
                const invLen = 1 / Math.sqrt(distSq);
                _stepTarget.set(
                    this._prevPos.x + dx * invLen * maxStep,
                    this._prevPos.y + dy * invLen * maxStep,
                    this._prevPos.z + dz * invLen * maxStep
                );
                targetPos = _stepTarget;
            }
        }

        if (targetPos.equals(this._prevPos)) return; // no movement

        const dir = this._tmpDir;
        dir.subVectors(targetPos, this._prevPos);
        const lenSq = dir.lengthSq();
        if (lenSq === 0) return;
        if (this._minSegmentDistanceSq > 0 && lenSq < this._minSegmentDistanceSq) return;
        dir.multiplyScalar(1 / Math.sqrt(lenSq)); // normalise

        const DOT_THRESHOLD = 0.999; // ~2.5° angle tolerance
        if (!this._currentDir) {
            // first real move – simply update last vertex
            this._currentDir = new THREE.Vector3();
            this._currentDir.copy(dir);
            this._writeVertex(this._vertexCount - 1, targetPos);
        } else if (dir.dot(this._currentDir) > DOT_THRESHOLD) {
            // still same direction – extend last vertex
            this._writeVertex(this._vertexCount - 1, targetPos);
        } else {
            // direction changed – append new vertex
            this._currentDir.copy(dir);
            // Ensure capacity; if we exceed, silently skip to avoid crashes.
            if (!this._useWideLine && this._vertexCount >= this._attr.count) return;
            this._writeVertex(this._vertexCount, targetPos);
            this._vertexCount += 1;
        }

        this._syncGeometryFromPositions();
        if (perfStats.enabled) {
            perfStats.inc('trailUpdates');
        }

        // Update bounding sphere every ~50 modifications to avoid CPU cost
        if (++this._updateCounter % 50 === 0) {
            this._geometry.computeBoundingSphere();
        }

        this._prevPos.copy(targetPos);
    }

    /**
     * Force the most recent vertex to match the provided position. Useful when
     * a consumer needs to "snap" the live trail endpoint to a final resting
     * coordinate (e.g. after a tween finishes) to avoid tiny gaps or overlaps
     * when the trail is later frozen into a static mesh.
     * @param {THREE.Vector3} pos
     */
    snapLastPointTo(pos) {
        if (!pos || this._vertexCount === 0) return;
        this._writeVertex(this._vertexCount - 1, pos);
        if (this._vertexCount >= 2) {
            const prevIdx = (this._vertexCount - 2) * 3;
            _snapPrev.set(
                this._positions[prevIdx],
                this._positions[prevIdx + 1],
                this._positions[prevIdx + 2]
            );
            _snapDir.subVectors(pos, _snapPrev);
            const lenSq = _snapDir.lengthSq();
            if (lenSq > 0) {
                _snapDir.multiplyScalar(1 / Math.sqrt(lenSq));
                if (!this._currentDir) this._currentDir = new THREE.Vector3();
                this._currentDir.copy(_snapDir);
            }
        }
        this._prevPos.copy(pos);
        this._syncGeometryFromPositions();
    }

    dispose() {
        this._scene.remove(this._line);
        this._geometry.dispose();
        this._material.dispose();
    }

    /** Adjust base opacity at runtime and update underlying material accordingly. */
    setBaseOpacity(newBaseOpacity) {
        if (typeof newBaseOpacity !== 'number' || !isFinite(newBaseOpacity)) return;
        this._opacity = Math.max(0, Math.min(1, newBaseOpacity));
        if (this._line && this._line.userData) {
            this._line.userData.trailBaseOpacity = this._opacity;
        }
        this.refreshDisplayScale();
    }

    /** Adjust base line width at runtime and update underlying material accordingly. */
    setBaseLineWidth(newBaseLineWidth) {
        if (typeof newBaseLineWidth !== 'number' || !isFinite(newBaseLineWidth)) return;
        this._lineWidth = Math.max(0.1, newBaseLineWidth);
        if (this._line && this._line.userData) {
            this._line.userData.trailBaseLineWidth = this._lineWidth;
        }
        this.refreshDisplayScale();
    }

    /** Re-apply runtime/DPR scaling to opacity + line width. */
    refreshDisplayScale() {
        const effOpacity = scaleOpacityForDisplay(this._opacity);
        const effWidth = scaleLineWidthForDisplay(this._lineWidth);
        applyTrailMaterialScale(this._material, effOpacity, effWidth);
    }

    /** Return current base opacity prior to DPR scaling. */
    getBaseOpacity() {
        return this._opacity;
    }

    /** Return current base line width prior to DPR scaling. */
    getBaseLineWidth() {
        return this._lineWidth;
    }

    /** Clamp how far the trail can advance per update (0 disables). */
    setMaxStepDistance(distance) {
        const next = Number(distance);
        this._maxStepDistance = Number.isFinite(next) && next > 0 ? next : 0;
    }

    /** Return a shallow copy of currently used positions (vertexCount * 3). */
    copyUsedPositions() {
        const n = Math.max(0, this._vertexCount);
        const out = new Float32Array(n * 3);
        out.set(this._positions.subarray(0, n * 3));
        return out;
    }

    /** Convert current polyline vertices into LineSegments vertex pairs. */
    toSegmentsFloat32() {
        const n = this._vertexCount;
        if (n < 2) return new Float32Array(0);
        const segs = (n - 1);
        const out = new Float32Array(segs * 2 * 3);
        let o = 0;
        for (let i = 0; i < n - 1; i++) {
            const i0 = i * 3;
            const i1 = (i + 1) * 3;
            // v0
            out[o++] = this._positions[i0 + 0];
            out[o++] = this._positions[i0 + 1];
            out[o++] = this._positions[i0 + 2];
            // v1
            out[o++] = this._positions[i1 + 0];
            out[o++] = this._positions[i1 + 1];
            out[o++] = this._positions[i1 + 2];
        }
        return out;
    }

    /**
     * Extract current segments and trim the live trail so future updates pick
     * up from the same endpoint. Optionally keep the most recent N segments
     * attached to the live trail to avoid visible gaps during hand-offs.
     * @param {object} [options]
     * @param {number} [options.preserveSegments=0] – number of newest segments to keep live.
     */
    extractSegmentsAndTrim(options = {}) {
        const { preserveSegments = 0 } = options;
        if (this._vertexCount < 2) return new Float32Array(0);

        const totalSegments = this._vertexCount - 1;
        const clampedPreserve = Math.max(
            0,
            Math.min(preserveSegments, totalSegments - 1)
        );
        const segmentsToExtract = totalSegments - clampedPreserve;
        if (segmentsToExtract <= 0) return new Float32Array(0);

        const segFloats = segmentsToExtract * 2 * 3;
        const seg = new Float32Array(segFloats);
        let o = 0;
        for (let i = 0; i < segmentsToExtract; i++) {
            const i0 = i * 3;
            const i1 = (i + 1) * 3;
            seg[o++] = this._positions[i0 + 0];
            seg[o++] = this._positions[i0 + 1];
            seg[o++] = this._positions[i0 + 2];
            seg[o++] = this._positions[i1 + 0];
            seg[o++] = this._positions[i1 + 1];
            seg[o++] = this._positions[i1 + 2];
        }

        if (clampedPreserve === 0) {
            const lastIdx = (this._vertexCount - 1) * 3;
            _trimLast.set(
                this._positions[lastIdx + 0],
                this._positions[lastIdx + 1],
                this._positions[lastIdx + 2]
            );
            this.resetToPosition(_trimLast);
            return seg;
        }

        const verticesToKeep = clampedPreserve + 1;
        const startVertex = this._vertexCount - verticesToKeep;
        for (let i = 0; i < verticesToKeep; i++) {
            const srcIdx = (startVertex + i) * 3;
            const dstIdx = i * 3;
            this._positions[dstIdx + 0] = this._positions[srcIdx + 0];
            this._positions[dstIdx + 1] = this._positions[srcIdx + 1];
            this._positions[dstIdx + 2] = this._positions[srcIdx + 2];
        }
        this._vertexCount = verticesToKeep;
        this._syncGeometryFromPositions();
        const lastIdx = (this._vertexCount - 1) * 3;
        this._prevPos.set(
            this._positions[lastIdx + 0],
            this._positions[lastIdx + 1],
            this._positions[lastIdx + 2]
        );
        this._currentDir = null;

        return seg;
    }

    // ------------------------------------------------------------------
    // Internal helpers
    // ------------------------------------------------------------------

    _syncGeometryFromPositions() {
        if (!this._geometry) return;
        if (this._useWideLine) {
            updateWideLineMaterialResolution(this._material);
            if (typeof this._geometry.setPositions === 'function') {
                this._geometry.setPositions(this._positions.subarray(0, this._vertexCount * 3));
            }
            return;
        }
        this._geometry.setDrawRange(0, this._vertexCount);
        if (this._attr) {
            this._attr.needsUpdate = true;
        }
    }

    _writeVertex(index, vec3) {
        const i3 = index * 3;
        this._positions[i3] = vec3.x;
        this._positions[i3 + 1] = vec3.y;
        this._positions[i3 + 2] = vec3.z;
    }
}

// ------------------------------------------------------------
// Batched straight-line trails (one segment per trail)
// ------------------------------------------------------------

export class SegmentTrailBatch {
    constructor(scene, capacity = 1, color = TRAIL_COLOR, lineWidth = TRAIL_LINE_WIDTH, opacity = TRAIL_OPACITY) {
        this._scene = scene;
        this._capacity = Math.max(1, Math.floor(capacity || 1));
        this._lineWidth = Number.isFinite(lineWidth) ? lineWidth : TRAIL_LINE_WIDTH;
        this._opacity = Number.isFinite(opacity) ? opacity : TRAIL_OPACITY;
        this._positions = new Float32Array(this._capacity * 2 * 3);
        this._attr = new THREE.BufferAttribute(this._positions, 3).setUsage(THREE.DynamicDrawUsage);
        this._geometry = new THREE.BufferGeometry();
        this._geometry.setAttribute('position', this._attr);
        this._geometry.setDrawRange(0, this._capacity * 2);

        const effectiveOpacity = scaleOpacityForDisplay(this._opacity);
        const effectiveWidth = scaleLineWidthForDisplay(this._lineWidth);
        this._material = new THREE.LineBasicMaterial({
            color,
            linewidth: effectiveWidth,
            transparent: effectiveOpacity < 1.0,
            opacity: effectiveOpacity,
            depthWrite: false,
            fog: false,
            toneMapped: false,
        });

        this._line = new THREE.LineSegments(this._geometry, this._material);
        this._line.frustumCulled = false;
        this._line.userData.isTrail = true;
        this._line.userData.trailBatch = true;
        this._line.userData.trailBatchRef = this;
        this._line.userData.trailBaseOpacity = this._opacity;
        this._line.userData.trailBaseLineWidth = this._lineWidth;
        const passId = resolveTrailPassId(scene);
        if (Number.isFinite(passId)) {
            this._line.userData.trailPassId = passId;
        }
        this._line.raycast = () => {};
        if (scene) scene.add(this._line);

        this._nextIndex = 0;
    }

    refreshDisplayScale() {
        if (!this._material) return;
        const effOpacity = scaleOpacityForDisplay(this._opacity);
        const effWidth = scaleLineWidthForDisplay(this._lineWidth);
        this._material.opacity = effOpacity;
        this._material.transparent = effOpacity < 1.0;
        if (Number.isFinite(effWidth)) {
            this._material.linewidth = effWidth;
        }
        this._material.needsUpdate = true;
    }

    acquireTrail() {
        if (this._nextIndex >= this._capacity) return null;
        const idx = this._nextIndex++;
        return new BatchedSegmentTrail(this, idx);
    }

    _setSegment(index, start, end) {
        if (index < 0 || index >= this._capacity) return;
        const i = index * 2 * 3;
        this._positions[i] = start.x;
        this._positions[i + 1] = start.y;
        this._positions[i + 2] = start.z;
        this._positions[i + 3] = end.x;
        this._positions[i + 4] = end.y;
        this._positions[i + 5] = end.z;
        this._attr.needsUpdate = true;
    }
}

export class BatchedSegmentTrail {
    constructor(batch, index) {
        this._batch = batch;
        this._index = index;
        this._start = new THREE.Vector3();
        this.isBatchedTrail = true;
    }

    start(pos) {
        if (!pos) return;
        this._start.copy(pos);
        this._batch._setSegment(this._index, pos, pos);
    }

    update(pos) {
        if (!pos) return;
        this._batch._setSegment(this._index, this._start, pos);
    }

    snapLastPointTo(pos) {
        this.update(pos);
    }

    dispose() {
        const hide = new THREE.Vector3(0, HIDE_INSTANCE_Y_OFFSET, 0);
        this._batch._setSegment(this._index, hide, hide);
    }
}

/** Traverse an Object3D subtree and collect StraightLineTrail refs. */
export function collectTrailsUnder(root) {
    const trails = [];
    if (!root) return trails;
    root.traverse(obj => {
        if (obj && obj.userData && obj.userData.isTrail && obj.userData.trailRef) {
            trails.push(obj.userData.trailRef);
        }
    });
    return trails;
}

function applyFadeIn(material, targetOpacity, fadeInMs) {
    if (!material) return;
    if (!Number.isFinite(fadeInMs) || fadeInMs <= 0 || typeof TWEEN === 'undefined') {
        material.opacity = targetOpacity;
        material.transparent = targetOpacity < 1.0;
        material.needsUpdate = true;
        return;
    }
    const state = { t: 0 };
    material.opacity = 0;
    material.transparent = true;
    material.needsUpdate = true;
    new TWEEN.Tween(state)
        .to({ t: 1 }, fadeInMs)
        .easing(TWEEN.Easing.Quadratic.Out)
        .onUpdate(() => {
            material.opacity = targetOpacity * state.t;
            material.needsUpdate = true;
        })
        .onComplete(() => {
            material.opacity = targetOpacity;
            material.transparent = targetOpacity < 1.0;
            material.needsUpdate = true;
        })
        .start();
}

/** Merge multiple StraightLineTrail polylines into a single LineSegments. */
export function mergeTrailsIntoLineSegments(trails, scene, color = TRAIL_COLOR, lineWidth = TRAIL_LINE_WIDTH, opacity = TRAIL_OPACITY, options = null) {
    if (!Array.isArray(trails) || trails.length === 0 || !scene) return null;

    // Concatenate segments for all trails
    let totalFloats = 0;
    const chunks = [];
    for (const t of trails) {
        if (!t || typeof t.toSegmentsFloat32 !== 'function') continue;
        const seg = t.toSegmentsFloat32();
        if (seg.length === 0) continue;
        chunks.push(seg);
        totalFloats += seg.length;
    }
    if (totalFloats === 0) return null;

    const positions = new Float32Array(totalFloats);
    let offset = 0;
    for (const c of chunks) {
        positions.set(c, offset);
        offset += c.length;
    }

    const effOpacity = scaleOpacityForDisplay(opacity);
    const effWidth = scaleLineWidthForDisplay(lineWidth);
    let geometry;
    let merged;
    if (USE_WIDE_TRAIL_LINES) {
        geometry = new LineSegmentsGeometry();
        geometry.setPositions(positions);
        const material = createTrailMaterial(color, effWidth, effOpacity);
        if (options && Number.isFinite(options.fadeInMs) && options.fadeInMs > 0) {
            applyFadeIn(material, effOpacity, options.fadeInMs);
        }
        merged = new LineSegments2(geometry, material);
    } else {
        geometry = new THREE.BufferGeometry();
        geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        geometry.setDrawRange(0, positions.length / 3);
        // Keep depthWrite disabled for static segments as well to prevent occlusion artifacts
        const material = createTrailMaterial(color, effWidth, effOpacity);
        if (options && Number.isFinite(options.fadeInMs) && options.fadeInMs > 0) {
            applyFadeIn(material, effOpacity, options.fadeInMs);
        }
        merged = new THREE.LineSegments(geometry, material);
    }
    merged.frustumCulled = false;
    merged.userData.isTrail = true;
    merged.userData.trailMerged = true;
    merged.userData.trailBaseOpacity = opacity;
    merged.userData.trailBaseLineWidth = lineWidth;
    const passId = resolveTrailPassId(scene);
    if (Number.isFinite(passId)) {
        merged.userData.trailPassId = passId;
    }
    merged.raycast = () => {};
    // Intentionally omit a hover label so merged trail lines remain non-interactive
    // in raycast tooltips (they are purely decorative).
    scene.add(merged);

    // Dispose original individual trails
    trails.forEach(t => t && typeof t.dispose === 'function' && t.dispose());

    return merged;
}

/** Build one LineSegments object from a list of Float32Array segment buffers. */
export function buildMergedLineSegmentsFromSegments(segmentsList, scene, color = TRAIL_COLOR, lineWidth = TRAIL_LINE_WIDTH, opacity = TRAIL_OPACITY, options = null) {
    if (!Array.isArray(segmentsList) || segmentsList.length === 0 || !scene) return null;
    let totalFloats = 0;
    for (const seg of segmentsList) {
        if (seg && seg.length) totalFloats += seg.length;
    }
    if (totalFloats === 0) return null;
    const positions = new Float32Array(totalFloats);
    let offset = 0;
    for (const seg of segmentsList) {
        if (!seg || !seg.length) continue;
        positions.set(seg, offset);
        offset += seg.length;
    }
    const effOpacity2 = scaleOpacityForDisplay(opacity);
    const effWidth2 = scaleLineWidthForDisplay(lineWidth);
    let geometry;
    let merged;
    if (USE_WIDE_TRAIL_LINES) {
        geometry = new LineSegmentsGeometry();
        geometry.setPositions(positions);
        const material = createTrailMaterial(color, effWidth2, effOpacity2);
        if (options && Number.isFinite(options.fadeInMs) && options.fadeInMs > 0) {
            applyFadeIn(material, effOpacity2, options.fadeInMs);
        }
        merged = new LineSegments2(geometry, material);
    } else {
        geometry = new THREE.BufferGeometry();
        geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        geometry.setDrawRange(0, positions.length / 3);
        const material = createTrailMaterial(color, effWidth2, effOpacity2);
        if (options && Number.isFinite(options.fadeInMs) && options.fadeInMs > 0) {
            applyFadeIn(material, effOpacity2, options.fadeInMs);
        }
        merged = new THREE.LineSegments(geometry, material);
    }
    merged.frustumCulled = false;
    merged.userData.isTrail = true;
    merged.userData.trailMerged = true;
    merged.userData.trailBaseOpacity = opacity;
    merged.userData.trailBaseLineWidth = lineWidth;
    const passId = resolveTrailPassId(scene);
    if (Number.isFinite(passId)) {
        merged.userData.trailPassId = passId;
    }
    merged.raycast = () => {};
    // Intentionally omit a hover label so merged trail lines remain non-interactive
    // in raycast tooltips (they are purely decorative).
    scene.add(merged);
    return merged;
}

/**
 * Re-apply runtime/DPR trail scaling to already-created trail materials.
 * Useful when runtime multipliers change mid-pass (e.g. while skipping).
 */
export function refreshTrailDisplayScales(root) {
    if (!root || typeof root.traverse !== 'function') return 0;
    let refreshed = 0;
    root.traverse((obj) => {
        if (!obj || !obj.userData) return;
        const isTrail = !!(obj.userData.isTrail || obj.userData.trailMerged || obj.userData.trailBatch);
        if (!isTrail) return;

        const trailRef = obj.userData.trailRef;
        if (trailRef && typeof trailRef.refreshDisplayScale === 'function') {
            trailRef.refreshDisplayScale();
            refreshed += 1;
            return;
        }
        const trailBatchRef = obj.userData.trailBatchRef;
        if (trailBatchRef && typeof trailBatchRef.refreshDisplayScale === 'function') {
            trailBatchRef.refreshDisplayScale();
            refreshed += 1;
            return;
        }

        const material = obj.material;
        if (!material) return;
        const materials = Array.isArray(material) ? material : [material];
        const baseOpacity = Number.isFinite(obj.userData.trailBaseOpacity)
            ? obj.userData.trailBaseOpacity
            : TRAIL_OPACITY;
        const baseLineWidth = Number.isFinite(obj.userData.trailBaseLineWidth)
            ? obj.userData.trailBaseLineWidth
            : TRAIL_LINE_WIDTH;
        const effOpacity = scaleOpacityForDisplay(baseOpacity);
        const effWidth = scaleLineWidthForDisplay(baseLineWidth);
        materials.forEach((mat) => {
            applyTrailMaterialScale(mat, effOpacity, effWidth);
        });
        refreshed += 1;
    });
    return refreshed;
}

export function clearTrailsFromScene(scene, { includeAllLines = false, passId = null } = {}) {
    if (!scene || typeof scene.traverse !== 'function') return 0;
    const targetPass = Number.isFinite(passId) ? passId : null;
    const targets = [];
    scene.traverse((obj) => {
        if (!obj) return;
        const isTrail = !!(obj.userData && (obj.userData.isTrail || obj.userData.trailMerged || obj.userData.trailBatch));
        const isLine = !!(obj.isLine || obj.isLineSegments);
        if (!isTrail && !(includeAllLines && isLine)) return;
        if (targetPass !== null) {
            const objPass = obj.userData && obj.userData.trailPassId;
            if (Number.isFinite(objPass) && objPass === targetPass) return;
        }
        targets.push(obj);
    });
    targets.forEach((obj) => {
        if (obj.parent) obj.parent.remove(obj);
        if (obj.geometry) obj.geometry.dispose();
        if (obj.material) {
            const materials = Array.isArray(obj.material) ? obj.material : [obj.material];
            materials.forEach((mat) => mat && mat.dispose && mat.dispose());
        }
    });
    return targets.length;
}
