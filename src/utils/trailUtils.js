import * as THREE from 'three';
import { MAX_TRAIL_POINTS, GLOBAL_ANIM_SPEED_MULT } from './constants.js'; // Assuming MAX_TRAIL_POINTS is in constants.js
import { TRAIL_LINE_OPACITY, TRAIL_LINE_COLOR } from '../animations/LayerAnimationConstants.js';
import { QUALITY_PRESET } from './constants.js';

// ────────────────────────────────────────────────────────────────────────────
// Global toggle – set to `false` to completely disable trail line creation
// and updates without touching calling code elsewhere.  This avoids large
// ArrayBuffer allocations (MAX_TRAIL_POINTS * 3) when profiling memory.
// ---------------------------------------------------------------------------
// Enable actual trail rendering (set to false when profiling memory)
export const TRAILS_ENABLED = true;

const SPEED_MULT = GLOBAL_ANIM_SPEED_MULT; // If SPEED_MULT is used by updateTrail

export function createTrailLine(scene, color = TRAIL_LINE_COLOR) {
    if (!TRAILS_ENABLED) {
        // Provide a minimal placeholder line so callers can still tweak
        // opacity/material without throwing, but allocate only 2 vertices.
        const minimalGeometry = new THREE.BufferGeometry();
        const pos = new Float32Array(6); // two 3-component points
        minimalGeometry.setAttribute('position', new THREE.BufferAttribute(pos, 3));
        minimalGeometry.setDrawRange(0, 2);
        const material = new THREE.LineBasicMaterial({ 
            color,
            transparent: true,
            opacity: TRAIL_LINE_OPACITY,
            depthTest: true,
            depthWrite: true,
            polygonOffset: true,
            polygonOffsetFactor: -2,
            polygonOffsetUnits: -2,
            linewidth: 1
        });
        const line = new THREE.Line(minimalGeometry, material);
        line.frustumCulled = false;
        scene.add(line);
        return { line, geometry: minimalGeometry, positions: pos, points: [] };
    }
    const geometry = new THREE.BufferGeometry();
    const positions = new Float32Array(MAX_TRAIL_POINTS * 3);
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geometry.setDrawRange(0, 0);
    const material = new THREE.LineBasicMaterial({ 
        color, 
        transparent: true, 
        opacity: TRAIL_LINE_OPACITY,
        depthTest: true,
        depthWrite: true,
        polygonOffset: true,
        polygonOffsetFactor: -2,
        polygonOffsetUnits: -2,
        linewidth: 1
    });
    
    const line = new THREE.Line(geometry, material);
    line.frustumCulled = false;
    line.renderOrder = 15; // Ensure trails render on top of other objects
    
    // Apply matrix transformations once to avoid floating point errors on update
    line.matrixAutoUpdate = true;
    
    geometry.computeBoundingSphere();
    scene.add(line); // Add to scene here
    return { line, geometry, positions, points: [] };
}

export function updateTrail(trailObj, pos) {
    // Early-out when trails globally disabled or no geometry present
    if (!TRAILS_ENABLED || !trailObj || !trailObj.geometry) return;

    // Performance: skip every other frame on low-quality preset
    if (QUALITY_PRESET === 'low') {
        updateTrail._frame = (updateTrail._frame || 0) + 1;
        if ((updateTrail._frame % 2) === 0) return;
    }

    if (!pos || typeof pos.x !== 'number' || typeof pos.y !== 'number' || typeof pos.z !== 'number') {
        return;
    }

    const pts = trailObj.points;
    let needsToPushNewPoint = false;

    // Minimum distance threshold to add a new point (helps prevent flickering from tiny movements)
    const MIN_DISTANCE_THRESHOLD = 0.05;

    if (pts.length === 0) {
        needsToPushNewPoint = true;
    } else {
        const lastPt = pts[pts.length - 1];
        if (Array.isArray(lastPt) && lastPt.length === 3 &&
            typeof lastPt[0] === 'number' && typeof lastPt[1] === 'number' && typeof lastPt[2] === 'number') {
            
            // Calculate distance to last point
            const dx = pos.x - lastPt[0];
            const dy = pos.y - lastPt[1];
            const dz = pos.z - lastPt[2];
            const distSquared = dx*dx + dy*dy + dz*dz;
            
            // Only add new point if it's meaningfully different from the last point
            if (distSquared > MIN_DISTANCE_THRESHOLD * MIN_DISTANCE_THRESHOLD) {
                needsToPushNewPoint = true;
            }
        } else {
            needsToPushNewPoint = true;
        }
    }

    if (needsToPushNewPoint) {
        if (pts.length < MAX_TRAIL_POINTS) {
            // Store rounded coordinates to reduce floating point errors
            const roundedX = Math.round(pos.x * 1000) / 1000;
            const roundedY = Math.round(pos.y * 1000) / 1000;
            const roundedZ = Math.round(pos.z * 1000) / 1000;

            //------------------------------------------------------------------
            //  High-speed interpolation (ported from now-removed helper in
            //  LayerAnimation.js). When the global SPEED_MULT is large and the
            //  current movement step covers a long distance, inject up to four
            //  intermediate points so the line appears continuous instead of
            //  jumpy/segmented.
            //------------------------------------------------------------------
            const lastPosArr = pts.length > 0 ? pts[pts.length - 1] : null;
            if (lastPosArr && SPEED_MULT > 10) {
                const dx = roundedX - lastPosArr[0];
                const dy = roundedY - lastPosArr[1];
                const dz = roundedZ - lastPosArr[2];
                const distance = Math.sqrt(dx * dx + dy * dy + dz * dz);

                // Keep the interpolation count modest so we don't flood the
                // buffer with too many points in a single frame.
                const steps = Math.min(4, Math.ceil(distance / 10));

                for (let i = 1; i < steps; i++) {
                    const t = i / steps;
                    const ix = lastPosArr[0] + dx * t;
                    const iy = lastPosArr[1] + dy * t;
                    const iz = lastPosArr[2] + dz * t;

                    if (pts.length >= MAX_TRAIL_POINTS) break;

                    pts.push([ix, iy, iz]);
                    const interpolatedIdx = pts.length - 1;
                    trailObj.geometry.attributes.position.setXYZ(interpolatedIdx, ix, iy, iz);
                }
            }

            // ------------------------------------------------------------------
            //  Colinear-segment collapse
            //  If the new point lies on the same straight line as the previous
            //  segment, overwrite the tail vertex instead of pushing a new one.
            // ------------------------------------------------------------------
            if (pts.length >= 2) {
                const pPrevPrev = pts[pts.length - 2];
                const pPrev     = pts[pts.length - 1];

                const ax = pPrev[0] - pPrevPrev[0];
                const ay = pPrev[1] - pPrevPrev[1];
                const az = pPrev[2] - pPrevPrev[2];

                const bx = roundedX - pPrev[0];
                const by = roundedY - pPrev[1];
                const bz = roundedZ - pPrev[2];

                // Cross-product magnitude squared – ~0 means colinear
                const cx = ay * bz - az * by;
                const cy = az * bx - ax * bz;
                const cz = ax * by - ay * bx;
                const crossMagSq = cx * cx + cy * cy + cz * cz;

                // If almost colinear (threshold tuned for world-unit scale)
                if (crossMagSq < 1e-10) {
                    pPrev[0] = roundedX; pPrev[1] = roundedY; pPrev[2] = roundedZ;
                    const lastIdx = pts.length - 1;
                    trailObj.geometry.attributes.position.setXYZ(lastIdx, roundedX, roundedY, roundedZ);
                    trailObj.geometry.attributes.position.needsUpdate = true;
                    return; // skip push → no new vertex allocated
                }
            }

            // Add the actual final position (non-colinear case)
            pts.push([roundedX, roundedY, roundedZ]);
            const idx = pts.length - 1;
            trailObj.geometry.attributes.position.setXYZ(idx, roundedX, roundedY, roundedZ);
            trailObj.geometry.setDrawRange(0, pts.length);
            trailObj.geometry.attributes.position.needsUpdate = true;

            // Compute bounding sphere less frequently to improve performance
            if (idx === 0 || idx % 200 === 0) {
                trailObj.geometry.computeBoundingSphere();
            }
        }
    }
} 