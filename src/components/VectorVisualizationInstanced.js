import * as THREE from 'three';
import { mapValueToColor } from '../utils/colors.js';
import { uniformRandom } from '../utils/mathUtils.js';
import { VECTOR_LENGTH, SPHERE_RADIUS, EPSILON, SPHERE_DIAMETER } from '../utils/constants.js';

// Generate the base geometry once
const baseSphereGeometry = new THREE.SphereGeometry(SPHERE_RADIUS, 16, 16);
const yScale = 4;
const zScale = yScale;
const scaleMatrix = new THREE.Matrix4().makeScale(1, yScale, zScale);
baseSphereGeometry.applyMatrix4(scaleMatrix);

export class VectorVisualizationInstanced {
    constructor(initialData = null, initialPosition = new THREE.Vector3(0, 0, 0)) {
        // Wrapper group to allow positioning the entire vector easily
        this.group = new THREE.Group();
        this.group.position.copy(initialPosition);

        // Prepare data arrays
        this.rawData = initialData || this.generateTestData();
        this.normalizedData = this.layerNormalize(this.rawData);

        // Material – shared for all instances. Emissive/color will be overridden per instance via vertex colors.
        const material = new THREE.MeshStandardMaterial({
            metalness: 0.3,
            roughness: 0.5,
            emissive: new THREE.Color(0xffffff),
            emissiveIntensity: 0.2
        });

        // Create InstancedMesh
        this.mesh = new THREE.InstancedMesh(baseSphereGeometry, material, VECTOR_LENGTH);
        this.mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage); // we will animate

        // Populate initial transforms/colors
        const dummy = new THREE.Object3D();
        for (let i = 0; i < VECTOR_LENGTH; i++) {
            const x = (i - VECTOR_LENGTH / 2) * SPHERE_DIAMETER;
            dummy.position.set(x, 0, 0);
            dummy.updateMatrix();
            this.mesh.setMatrixAt(i, dummy.matrix);

            const col = mapValueToColor(this.normalizedData[i]);
            this.mesh.setColorAt(i, col);
        }
        this.mesh.instanceColor.needsUpdate = true;

        this.group.add(this.mesh);
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

    // Update colors with new data, keeping transforms unchanged
    updateData(newData) {
        this.rawData = newData;
        this.normalizedData = this.layerNormalize(newData);
        for (let i = 0; i < VECTOR_LENGTH; i++) {
            const col = mapValueToColor(this.normalizedData[i]);
            this.mesh.setColorAt(i, col);
        }
        this.mesh.instanceColor.needsUpdate = true;
    }

    // Utility to update one instance's color quickly
    setInstanceColor(index, color) {
        this.mesh.setColorAt(index, color);
        this.mesh.instanceColor.needsUpdate = true;
    }

    // Utility to update one instance's position (y coordinate) quickly
    setInstanceYOffset(index, y) {
        const dummy = new THREE.Object3D();
        // Compute X based on index
        const x = (index - VECTOR_LENGTH / 2) * SPHERE_DIAMETER;
        dummy.position.set(x, y, 0);
        dummy.updateMatrix();
        this.mesh.setMatrixAt(index, dummy.matrix);
        this.mesh.instanceMatrix.needsUpdate = true;
    }

    dispose() {
        this.mesh.geometry.dispose();
        this.mesh.material.dispose();
    }
} 