import * as THREE from 'three';
import { TRAIL_COLOR, TRAIL_LINE_WIDTH, TRAIL_OPACITY, TRAIL_MAX_SEGMENTS, scaleOpacityForDisplay, scaleLineWidthForDisplay } from './trailConstants.js';
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
        const clampedMin = Math.max(0, minSegmentDistance);
        this._minSegmentDistanceSq = clampedMin * clampedMin;

        // Preallocate vertex buffer (N segments ⇒ N+1 vertices; we duplicate the
        // first vertex to create a zero-length segment so drawRange ≥2).  Each
        // vertex is 3 floats.
        const maxVertices = maxSegments + 1;
        this._positions = new Float32Array(maxVertices * 3);
        this._attr = new THREE.BufferAttribute(this._positions, 3).setUsage(THREE.DynamicDrawUsage);

        this._geometry = new THREE.BufferGeometry();
        this._geometry.setAttribute('position', this._attr);
        this._geometry.setDrawRange(0, 0); // nothing yet

        const effectiveOpacity = scaleOpacityForDisplay(this._opacity);
        const effectiveWidth = scaleLineWidthForDisplay(lineWidth);
        // Keep depthWrite disabled for transparent lines to avoid occluding scene content
        this._material = new THREE.LineBasicMaterial({ color: this._color, linewidth: effectiveWidth, transparent: effectiveOpacity < 1.0, opacity: effectiveOpacity, depthWrite: false, fog: false, toneMapped: false });
        // Ensure trails stay visible during camera fly-throughs by bypassing frustum
        // culling; depth testing remains enabled so solids continue to occlude trails.
        this._line = new THREE.Line(this._geometry, this._material);
        this._line.frustumCulled = false;
        // Tag for discovery and back-reference
        this._line.userData.isTrail = true;
        this._line.userData.trailRef = this;
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
        this._geometry.setDrawRange(0, this._vertexCount);
        this._attr.needsUpdate = true;
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
        this._geometry.setDrawRange(0, this._vertexCount);
        this._attr.needsUpdate = true;
    }

    update(pos) {
        if (this._vertexCount === 0) return; // not started yet
        if (pos.equals(this._prevPos)) return; // no movement

        const dir = this._tmpDir;
        dir.subVectors(pos, this._prevPos);
        const lenSq = dir.lengthSq();
        if (lenSq === 0) return;
        if (this._minSegmentDistanceSq > 0 && lenSq < this._minSegmentDistanceSq) return;
        dir.multiplyScalar(1 / Math.sqrt(lenSq)); // normalise

        const DOT_THRESHOLD = 0.999; // ~2.5° angle tolerance
        if (!this._currentDir) {
            // first real move – simply update last vertex
            this._currentDir = new THREE.Vector3();
            this._currentDir.copy(dir);
            this._writeVertex(this._vertexCount - 1, pos);
        } else if (dir.dot(this._currentDir) > DOT_THRESHOLD) {
            // still same direction – extend last vertex
            this._writeVertex(this._vertexCount - 1, pos);
        } else {
            // direction changed – append new vertex
            this._currentDir.copy(dir);
            // Ensure capacity; if we exceed, silently skip to avoid crashes.
            if (this._vertexCount >= this._attr.count) return;
            this._writeVertex(this._vertexCount, pos);
            this._vertexCount += 1;
            this._geometry.setDrawRange(0, this._vertexCount);
        }

        this._attr.needsUpdate = true;
        if (perfStats.enabled) {
            perfStats.inc('trailUpdates');
        }

        // Update bounding sphere every ~50 modifications to avoid CPU cost
        if (++this._updateCounter % 50 === 0) {
            this._geometry.computeBoundingSphere();
        }

        this._prevPos.copy(pos);
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
        this._attr.needsUpdate = true;
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
        const eff = scaleOpacityForDisplay(this._opacity);
        if (this._material) {
            this._material.opacity = eff;
            this._material.transparent = eff < 1.0;
            this._material.needsUpdate = true;
        }
    }

    /** Return current base opacity prior to DPR scaling. */
    getBaseOpacity() {
        return this._opacity;
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
        this._geometry.setDrawRange(0, this._vertexCount);
        this._attr.needsUpdate = true;
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

    _writeVertex(index, vec3) {
        const i3 = index * 3;
        this._positions[i3] = vec3.x;
        this._positions[i3 + 1] = vec3.y;
        this._positions[i3 + 2] = vec3.z;
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

/** Merge multiple StraightLineTrail polylines into a single LineSegments. */
export function mergeTrailsIntoLineSegments(trails, scene, color = TRAIL_COLOR, lineWidth = TRAIL_LINE_WIDTH, opacity = TRAIL_OPACITY) {
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

    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geometry.setDrawRange(0, positions.length / 3);

    const effOpacity = scaleOpacityForDisplay(opacity);
    const effWidth = scaleLineWidthForDisplay(lineWidth);
    // Keep depthWrite disabled for static segments as well to prevent occlusion artifacts
    const material = new THREE.LineBasicMaterial({ color, linewidth: effWidth, transparent: effOpacity < 1.0, opacity: effOpacity, depthWrite: false, fog: false, toneMapped: false });
    const merged = new THREE.LineSegments(geometry, material);
    // Intentionally omit a hover label so merged trail lines remain non-interactive
    // in raycast tooltips (they are purely decorative).
    scene.add(merged);

    // Dispose original individual trails
    trails.forEach(t => t && typeof t.dispose === 'function' && t.dispose());

    return merged;
}

/** Build one LineSegments object from a list of Float32Array segment buffers. */
export function buildMergedLineSegmentsFromSegments(segmentsList, scene, color = TRAIL_COLOR, lineWidth = TRAIL_LINE_WIDTH, opacity = TRAIL_OPACITY) {
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
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geometry.setDrawRange(0, positions.length / 3);
    const effOpacity2 = scaleOpacityForDisplay(opacity);
    const effWidth2 = scaleLineWidthForDisplay(lineWidth);
    const material = new THREE.LineBasicMaterial({ color, linewidth: effWidth2, transparent: effOpacity2 < 1.0, opacity: effOpacity2, depthWrite: false, fog: false, toneMapped: false });
    const merged = new THREE.LineSegments(geometry, material);
    // Intentionally omit a hover label so merged trail lines remain non-interactive
    // in raycast tooltips (they are purely decorative).
    scene.add(merged);
    return merged;
}
