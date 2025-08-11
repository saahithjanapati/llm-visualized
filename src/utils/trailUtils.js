import * as THREE from 'three';
import { TRAIL_COLOR, TRAIL_LINE_WIDTH, TRAIL_OPACITY, TRAIL_MAX_SEGMENTS } from './trailConstants.js';

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
export class StraightLineTrail {
    /**
     * @param {THREE.Object3D} scene        Scene (or Group) to attach the trail.
     * @param {number}         color        Hex colour.
     * @param {number}         lineWidth    Width in pixels (ignored on most HW).
     * @param {number}         maxSegments  Preallocated straight-line segments.
     */
    constructor(scene, color = TRAIL_COLOR, lineWidth = TRAIL_LINE_WIDTH, maxSegments = TRAIL_MAX_SEGMENTS, opacity = TRAIL_OPACITY) {
        this._scene = scene;
        this._color = color;
        this._opacity = opacity;

        // Preallocate vertex buffer (N segments ⇒ N+1 vertices; we duplicate the
        // first vertex to create a zero-length segment so drawRange ≥2).  Each
        // vertex is 3 floats.
        const maxVertices = maxSegments + 1;
        this._positions = new Float32Array(maxVertices * 3);
        this._attr = new THREE.BufferAttribute(this._positions, 3).setUsage(THREE.DynamicDrawUsage);

        this._geometry = new THREE.BufferGeometry();
        this._geometry.setAttribute('position', this._attr);
        this._geometry.setDrawRange(0, 0); // nothing yet

        this._material = new THREE.LineBasicMaterial({ color: this._color, linewidth: lineWidth, transparent: this._opacity < 1.0, opacity: this._opacity });
        this._line = new THREE.Line(this._geometry, this._material);
        scene.add(this._line);

        this._vertexCount = 0;
        this._prevPos = new THREE.Vector3();
        this._currentDir = null;
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

    update(pos) {
        if (this._vertexCount === 0) return; // not started yet
        if (pos.equals(this._prevPos)) return; // no movement

        const dir = new THREE.Vector3().subVectors(pos, this._prevPos);
        const lenSq = dir.lengthSq();
        if (lenSq === 0) return;
        dir.multiplyScalar(1 / Math.sqrt(lenSq)); // normalise

        const DOT_THRESHOLD = 0.999; // ~2.5° angle tolerance
        if (!this._currentDir) {
            // first real move – simply update last vertex
            this._currentDir = dir.clone();
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

        // Update bounding sphere every ~50 modifications to avoid CPU cost
        if (++this._updateCounter % 50 === 0) {
            this._geometry.computeBoundingSphere();
        }

        this._prevPos.copy(pos);
    }

    dispose() {
        this._scene.remove(this._line);
        this._geometry.dispose();
        this._material.dispose();
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
