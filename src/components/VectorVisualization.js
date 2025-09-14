import * as THREE from 'three';
import { mapValueToColor_LOG, mapValueToColor } from '../utils/colors.js';
import { uniformRandom } from '../utils/mathUtils.js';
import { VECTOR_LENGTH, SPHERE_RADIUS, EPSILON, SPHERE_DIAMETER } from '../utils/constants.js';

// Generate a smoother, fully spherical geometry for each bead.  We increase
// the width/height segments for better roundness and remove the previous
// Y/Z stretching so the beads appear nearly perfect circles instead of tall
// ellipsoids.
const baseSphereGeometry = new THREE.SphereGeometry(SPHERE_RADIUS, 16, 16);
// Preserve the original elongated stone/ellipse look by stretching along the
// Y and Z axes **after** generating a smoother base sphere.
const yScale = 4;
const zScale = yScale;
const scaleMatrix = new THREE.Matrix4().makeScale(1, yScale, zScale);
baseSphereGeometry.applyMatrix4(scaleMatrix);

export class VectorVisualization {
    constructor(initialData = null, initialPosition = new THREE.Vector3(0, 0, 0)) {
        this.group = new THREE.Group();
        this.group.position.copy(initialPosition);
        // Label for raycasting hover info
        this.group.userData.label = 'Vector';
        this.ellipses = []; // Keep track of meshes for potential updates
        // Assign a default speed, can be overridden later
        this.speed = 0; // Initialize speed, will be set externally in main.js

        const vectorData = initialData || this.generateTestData();
        // const normalizedVectorData = this.layerNormalize(vectorData); // REMOVED: Do not normalize for initial coloring

        for (let i = 0; i < VECTOR_LENGTH; i++) {
            const value = vectorData[i]; // NEW: Use raw initialData for initial coloring
            const color = mapValueToColor_LOG(value, i); // Call logging version
            // const fixedColor = new THREE.Color(0x00ff00); // TEST: Use fixed bright green

            const material = new THREE.MeshStandardMaterial({
                color: color, // Original
                // color: fixedColor, // TEST
                metalness: 0.3,
                roughness: 0.5,
                emissive: color, // Original
                // emissive: fixedColor, // TEST
                // Increase base emissive intensity so vectors glow a bit more
                emissiveIntensity: 0.6
             });

            // Re-use the single shared geometry instance instead of cloning it for
            // every mesh.  This saves GPU memory and upload time because the
            // vertex data now exists only once.
            const ellipse = new THREE.Mesh(baseSphereGeometry, material);
            // Set label for raycasting hover identification
            ellipse.userData.label = 'Vector';
            ellipse.position.x = (i - VECTOR_LENGTH / 2) * SPHERE_DIAMETER;
            ellipse.position.y = 0; // Relative to the group
            this.group.add(ellipse);
            this.ellipses.push(ellipse);
        }
    }

    generateTestData() {
        const data = [];
        for (let i = 0; i < VECTOR_LENGTH; i++) {
            data.push(uniformRandom());
        }
        return data;
    }

    layerNormalize(vectorData) {
        const sum = vectorData.reduce((acc, val) => acc + val, 0);
        const mean = sum / VECTOR_LENGTH;
        const varianceSum = vectorData.reduce((acc, val) => acc + Math.pow(val - mean, 2), 0);
        const variance = varianceSum / VECTOR_LENGTH;
        const stdDev = Math.sqrt(variance + EPSILON);
        return vectorData.map(val => (val - mean) / stdDev);
    }

    updateData(newData) {
        const normalizedVectorData = this.layerNormalize(newData);
        for (let i = 0; i < VECTOR_LENGTH; i++) {
            const value = normalizedVectorData[i];
            const color = mapValueToColor_LOG(value, i); // Also use logging version here for consistency during updates if any
            const ellipse = this.ellipses[i];
            if (ellipse && ellipse.material) { // Check if ellipse and material exist
                ellipse.material.color.copy(color);
                ellipse.material.emissive.copy(color);
            }
        }
    }

    dispose() {
        // All ellipses share the same Geometry instance, so we must NOT dispose
        // it individually here (it will be released once when the whole app
        // tears down).  Only dispose the per-sphere materials.
        this.ellipses.forEach(ellipse => {
            if (ellipse.material) ellipse.material.dispose();
        });
        this.ellipses = []; // Clear the array
    }
}

// Consider disposing the shared baseSphereGeometry when the app exits if necessary.
// For simple examples, it might be okay, but in larger apps, manage shared resources.
// export function disposeSharedGeometry() {
//     baseSphereGeometry.dispose();
// }
