import * as THREE from 'three';
import { mapValueToColor } from '../utils/colors.js';
import { uniformRandom } from '../utils/mathUtils.js';
import { VECTOR_LENGTH, SPHERE_RADIUS, EPSILON, SPHERE_DIAMETER } from '../utils/constants.js';

// Prepare base geometry
const baseSphereGeometry = new THREE.SphereGeometry(SPHERE_RADIUS, 8, 8);
const yScale = 4;
const zScale = yScale;
const scaleMatrix = new THREE.Matrix4().makeScale(1, yScale, zScale);
baseSphereGeometry.applyMatrix4(scaleMatrix);

export class VectorVisualization {
    constructor(initialData = null, initialPosition = new THREE.Vector3(0, 0, 0)) {
        this.group = new THREE.Group();
        this.group.position.copy(initialPosition);
        this.ellipses = []; // Keep track of meshes for potential updates
        // Assign a default speed, can be overridden later
        this.speed = 0; // Initialize speed, will be set externally in main.js

        const vectorData = initialData || this.generateTestData();
        const normalizedVectorData = this.layerNormalize(vectorData);

        for (let i = 0; i < VECTOR_LENGTH; i++) {
            const value = normalizedVectorData[i];
            const color = mapValueToColor(value);
            const material = new THREE.MeshStandardMaterial({
                color: color,
                metalness: 0.3,
                roughness: 0.5,
                emissive: color,
                emissiveIntensity: 0.3
             });

            const ellipseGeometry = baseSphereGeometry.clone();
            const ellipse = new THREE.Mesh(ellipseGeometry, material);
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
            const color = mapValueToColor(value);
            const ellipse = this.ellipses[i];
            if (ellipse && ellipse.material) { // Check if ellipse and material exist
                ellipse.material.color.copy(color);
                ellipse.material.emissive.copy(color);
            }
        }
    }

    dispose() {
         this.ellipses.forEach(ellipse => {
            if (ellipse.geometry) ellipse.geometry.dispose();
            if (ellipse.material) ellipse.material.dispose();
        });
        this.ellipses = []; // Clear the array
        // Note: We don't dispose the shared baseSphereGeometry here
    }
}

// Consider disposing the shared baseSphereGeometry when the app exits if necessary.
// For simple examples, it might be okay, but in larger apps, manage shared resources.
// export function disposeSharedGeometry() {
//     baseSphereGeometry.dispose();
// }
