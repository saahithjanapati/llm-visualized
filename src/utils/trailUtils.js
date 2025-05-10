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
        opacity: TRAIL_LINE_OPACITY 
    });
    const line = new THREE.Line(geometry, material);
    line.frustumCulled = false;
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

    if (pts.length === 0) {
        needsToPushNewPoint = true;
    } else {
        const lastPt = pts[pts.length - 1];
        if (Array.isArray(lastPt) && lastPt.length === 3 &&
            typeof lastPt[0] === 'number' && typeof lastPt[1] === 'number' && typeof lastPt[2] === 'number') {
            if (pos.x !== lastPt[0] || pos.y !== lastPt[1] || pos.z !== lastPt[2]) {
                needsToPushNewPoint = true;
            }
        } else {
            needsToPushNewPoint = true;
        }
    }

    if (needsToPushNewPoint) {
        if (pts.length < MAX_TRAIL_POINTS) {
            const lastPos = pts.length > 0 ? pts[pts.length - 1] : null;
            
            if (lastPos && SPEED_MULT > 10) { // SPEED_MULT needs to be defined or passed
                const dx = pos.x - lastPos[0];
                const dy = pos.y - lastPos[1];
                const dz = pos.z - lastPos[2];
                const distance = Math.sqrt(dx*dx + dy*dy + dz*dz);
                
                if (distance > 5) {
                    const steps = Math.min(10, Math.ceil(distance / 5));
                    for (let i = 1; i < steps; i++) {
                        const t = i / steps;
                        const ix = lastPos[0] + dx * t;
                        const iy = lastPos[1] + dy * t;
                        const iz = lastPos[2] + dz * t;
                        
                        if (pts.length < MAX_TRAIL_POINTS) { // Check again before pushing intermediate points
                            pts.push([ix, iy, iz]);
                            const interpolatedIdx = pts.length - 1;
                            trailObj.geometry.attributes.position.setXYZ(interpolatedIdx, ix, iy, iz);
                        } else {
                            break; 
                        }
                    }
                }
            }
            
            if (pts.length < MAX_TRAIL_POINTS) { // Check again before pushing the main point
                pts.push([pos.x, pos.y, pos.z]);
                const idx = pts.length - 1;
                trailObj.geometry.attributes.position.setXYZ(idx, pos.x, pos.y, pos.z);
                trailObj.geometry.setDrawRange(0, pts.length);
                trailObj.geometry.attributes.position.needsUpdate = true;
                if (idx === 0 || idx % 100 === 0) trailObj.geometry.computeBoundingSphere();
            }
        }
    }
} 