import * as THREE from 'three';
import { MAX_TRAIL_POINTS, GLOBAL_ANIM_SPEED_MULT } from './constants.js'; // Assuming MAX_TRAIL_POINTS is in constants.js
import { TRAIL_LINE_OPACITY, TRAIL_LINE_COLOR } from '../animations/LayerAnimationConstants.js';

const SPEED_MULT = GLOBAL_ANIM_SPEED_MULT; // If SPEED_MULT is used by updateTrail

export function createTrailLine(scene, color = TRAIL_LINE_COLOR) { // Added scene parameter and default color
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

            // Add the actual final position
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