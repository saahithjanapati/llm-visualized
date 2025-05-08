import * as THREE from 'three';

export class PrismRowVisualization {
    constructor(instanceCount = 768) {
        this.instanceCount = instanceCount;
        this.mesh = this._createInstancedMesh();
    }

    _createInstancedMesh() {
        const geometry = new THREE.BoxGeometry(0.1, 0.1, 0.5); // Skinny rectangular prisms
        const material = new THREE.MeshBasicMaterial({
            color: 0xffffff // Default white color, instance colors should override
        });

        const mesh = new THREE.InstancedMesh(geometry, material, this.instanceCount);
        const dummy = new THREE.Object3D();
        const color = new THREE.Color();

        for (let i = 0; i < this.instanceCount; i++) {
            // Position prisms next to each other
            dummy.position.set(i * 0.12 - (this.instanceCount * 0.12) / 2, 0, 0);
            dummy.updateMatrix();
            mesh.setMatrixAt(i, dummy.matrix);

            // Assign a unique color to each instance
            // Cycle through hues, with full saturation and lightness
            color.setHSL(i / this.instanceCount, 1.0, 0.5);
            mesh.setColorAt(i, color);
        }

        if (mesh.instanceColor) {
            mesh.instanceColor.needsUpdate = true;
        } else {
            console.error("mesh.instanceColor is null! Ensure Three.js version supports it or material is configured correctly.");
        }
        
        return mesh;
    }

    getMesh() {
        return this.mesh;
    }

    // Optional: Add methods to update colors, positions, etc. later if needed
} 