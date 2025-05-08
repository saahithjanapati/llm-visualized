import * as THREE from 'three';
import { uniformRandom } from '../utils/mathUtils.js';
import { 
    VECTOR_LENGTH_PRISM, 
    PRISM_BASE_WIDTH,
    PRISM_BASE_DEPTH,
    PRISM_MAX_HEIGHT,
    PRISM_HEIGHT_SCALE_FACTOR,
} from '../utils/constants.js';

// Generate the base geometry once
// The prism's origin will be at its center. We'll adjust its y-position when setting instance matrices.
const basePrismGeometry = new THREE.BoxGeometry(PRISM_BASE_WIDTH, 1, PRISM_BASE_DEPTH); // Height is 1, will be scaled

export class VectorVisualizationInstancedPrism {
    constructor(initialData = null, initialPosition = new THREE.Vector3(0, 0, 0)) {
        this.group = new THREE.Group();
        this.group.position.copy(initialPosition);

        this.rawData = initialData || this.generateTestData();
        this.normalizedData = this.layerNormalize(this.rawData);

        // Using MeshBasicMaterial to rule out lighting issues
        const material = new THREE.MeshBasicMaterial({ 
            color: new THREE.Color(0xffffff) // Base color white, will be tinted by instance color
        });

        const instancedGeometry = basePrismGeometry.clone();
        this.mesh = new THREE.InstancedMesh(instancedGeometry, material, VECTOR_LENGTH_PRISM);
        this.mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);

        this.updateInstanceGeometryAndColors();
        this.group.add(this.mesh);
    }

    generateTestData() {
        const data = [];
        for (let i = 0; i < VECTOR_LENGTH_PRISM; i++) {
            // Generate values between 0 and 1 for simplicity
            data.push(Math.random()); 
        }
        return data;
    }

    // Rewritten layerNormalize for simplicity and robustness
    layerNormalize(vectorData) {
        if (!vectorData || vectorData.length === 0) {
            console.warn("LayerNormalize: Input data is empty or null.");
            return [];
        }

        const finiteData = vectorData.filter(val => typeof val === 'number' && isFinite(val));
        if (finiteData.length === 0) {
            console.warn("LayerNormalize: No finite numbers in input data. Defaulting to all 0.5.");
            return vectorData.map(() => 0.5); // Return array of 0.5 of original length
        }

        let minVal = finiteData[0];
        let maxVal = finiteData[0];
        for (let i = 1; i < finiteData.length; i++) {
            if (finiteData[i] < minVal) minVal = finiteData[i];
            if (finiteData[i] > maxVal) maxVal = finiteData[i];
        }

        const range = maxVal - minVal;

        // If all values are the same (or very close), normalize all to 0.5
        if (range < 1e-7) { // Using a small epsilon for float comparison
            return vectorData.map(val => (typeof val === 'number' && isFinite(val) ? 0.5 : 0.5));
        }

        // Normalize to 0-1 range
        return vectorData.map(val => {
            if (typeof val === 'number' && isFinite(val)) {
                return (val - minVal) / range;
            } else {
                // Handle non-finite numbers in original array by mapping them to 0.5 (mid-point)
                return 0.5; 
            }
        });
    }
    
    updateInstanceGeometryAndColors() {
        const dummy = new THREE.Object3D();
        let uniformHeight = PRISM_MAX_HEIGHT * PRISM_HEIGHT_SCALE_FACTOR;
        uniformHeight *= 2.0; // Make prisms twice as tall
        
        const prismWidthScale = 1.5; // Make prisms 50% wider
        const prismDepthScale = 1.5; // Make prisms 50% thicker (depth)

        const color = new THREE.Color();

        for (let i = 0; i < VECTOR_LENGTH_PRISM; i++) {
            // Position and Scale
            const x = (i - VECTOR_LENGTH_PRISM / 2) * (PRISM_BASE_WIDTH * prismWidthScale);
            dummy.scale.set(prismWidthScale, uniformHeight, prismDepthScale); // Apply new width, height, and depth scale
            dummy.position.set(x, uniformHeight / 2, 0); 
            dummy.updateMatrix();
            this.mesh.setMatrixAt(i, dummy.matrix);

            // Set color to form a smooth gradient across instances
            const hue = i / VECTOR_LENGTH_PRISM; // Hue from 0.0 to 1.0 across all prisms
            const saturation = 1.0; // Full saturation for vibrancy
            const lightness = 0.5;  // Standard lightness for vibrant hues
            color.setHSL(hue, saturation, lightness);
            this.mesh.setColorAt(i, color);
        }

        this.mesh.instanceMatrix.needsUpdate = true;
        if (this.mesh.instanceColor) {
            this.mesh.instanceColor.needsUpdate = true;
            // console.log("instanceColor buffer exists and needsUpdate set to true.");
        } else {
            // console.warn("instanceColor buffer does NOT exist on the mesh after setColorAt!");
        }
        // console.log("Finished attempting to set all instances to RED.");
    }

    updateData(newData) {
        if (!newData || newData.length !== VECTOR_LENGTH_PRISM) {
            console.warn(`updateData: New data length (${newData ? newData.length : 'null'}) does not match VECTOR_LENGTH_PRISM (${VECTOR_LENGTH_PRISM}). Using old data or re-generating.`);
            this.rawData = this.generateTestData(); // Fallback to new test data if bad input
        } else {
            this.rawData = newData;
        }
        this.normalizedData = this.layerNormalize(this.rawData);
        this.updateInstanceGeometryAndColors();
    }

    dispose() {
        if (this.mesh.geometry) this.mesh.geometry.dispose();
        if (this.mesh.material) this.mesh.material.dispose();
    }
}