import * as THREE from 'three';
import { mapValueToColor_LOG } from '../utils/colors.js';
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

        // Material – shared for all instances.
        const material = new THREE.MeshBasicMaterial({ // MeshBasicMaterial for diagnostics
            color: new THREE.Color(0xffffff),      // Base color white, to be tinted by instance color
            opacity: 1.0,
            transparent: false
        });

        // TEST: Clone the geometry for the InstancedMesh
        const instancedGeometry = baseSphereGeometry.clone();

        // Create InstancedMesh
        this.mesh = new THREE.InstancedMesh(instancedGeometry, material, VECTOR_LENGTH); // Use cloned geometry
        this.mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage); // we will animate

        // Populate initial transforms/colors
        const dummy = new THREE.Object3D();
        const redColor = new THREE.Color(0xff0000); // TEST: Force all to red

        for (let i = 0; i < VECTOR_LENGTH; i++) {
            const x = (i - VECTOR_LENGTH / 2) * SPHERE_DIAMETER;
            dummy.position.set(x, 0, 0);
            dummy.updateMatrix();
            this.mesh.setMatrixAt(i, dummy.matrix);

            this.mesh.setColorAt(i, redColor); // TEST: Set to red
        }
        this.mesh.instanceColor.needsUpdate = true; // Crucial for instance colors

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

    updateData(newData) {
        this.rawData = newData;
        this.normalizedData = this.layerNormalize(newData);
        const redColor = new THREE.Color(0xff0000); // TEST: Force all to red for updates too

        for (let i = 0; i < VECTOR_LENGTH; i++) {
            this.mesh.setColorAt(i, redColor); // TEST: Set to red
        }
        this.mesh.instanceColor.needsUpdate = true; // Crucial for instance colors
    }

    setInstanceColor(index, color) {
        this.mesh.setColorAt(index, color);
        this.mesh.instanceColor.needsUpdate = true; // Crucial for instance colors
    }

    setInstanceYOffset(index, y) {
        const dummy = new THREE.Object3D();
        const x = (index - VECTOR_LENGTH / 2) * SPHERE_DIAMETER;
        dummy.position.set(x, y, 0);
        dummy.updateMatrix();
        this.mesh.setMatrixAt(index, dummy.matrix);
        this.mesh.instanceMatrix.needsUpdate = true;
    }

    dispose() {
        this.mesh.geometry.dispose(); // Dispose the cloned geometry
        this.mesh.material.dispose();
    }
} 