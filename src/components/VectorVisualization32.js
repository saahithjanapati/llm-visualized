import * as THREE from 'three';
import { PRISM_BASE_WIDTH, PRISM_BASE_DEPTH, PRISM_MAX_HEIGHT, PRISM_HEIGHT_SCALE_FACTOR } from '../utils/constants.js';

// Number of grouped components and how many original prism-units each covers
const GROUP_COUNT = 32;
const UNITS_PER_GROUP = 24; // 24 * 32 = 768 (original d_model)

// Visual scaling factors (tweaked for readability – feel free to adjust)
const WIDTH_SCALE  = 1.5;   // Same width scale used by existing InstancedPrism viz
const DEPTH_SCALE  = 1.5;   // Slightly thicker so gradients are visible

// Pre-compute the uniform height we use for all prisms (match InstancedPrism double-height visual)
const UNIFORM_HEIGHT = PRISM_MAX_HEIGHT * PRISM_HEIGHT_SCALE_FACTOR * 2.0;

// Helper to generate an array of random HSL colours (fully saturated, medium lightness)
function generateRandomKeyColors(num) {
    const arr = [];
    for (let i = 0; i < num; i++) {
        arr.push(new THREE.Color().setHSL(Math.random(), 1.0, 0.5));
    }
    return arr;
}

// Create a **shared** geometry that already contains vertex colors encoding a left-to-right gradient.
// The same geo is reused for all 32 meshes.
function createGradientGeometry(startColor, endColor) {
    // Width covers 24 original units (times the width scale)
    const width  = PRISM_BASE_WIDTH * UNITS_PER_GROUP * WIDTH_SCALE;
    const height = 1; // Will be scaled on the Y-axis later
    const depth  = PRISM_BASE_DEPTH * DEPTH_SCALE;

    // Subdivide the box along X so the face interpolation produces a smoother gradient
    const segmentsX = UNITS_PER_GROUP; // 24 subdivisions → plenty of gradient steps
    const geometry = new THREE.BoxGeometry(width, height, depth, segmentsX, 1, 1);

    // Add vertex colours (THREE.BoxGeometry produces a non-indexed BufferGeometry)
    const position = geometry.attributes.position;
    const colorArr = new Float32Array(position.count * 3);

    const colA = new THREE.Color(startColor);
    const colB = new THREE.Color(endColor);

    const halfW = width / 2;

    for (let i = 0; i < position.count; i++) {
        const x = position.getX(i);
        // Normalised progress across width: 0 at left, 1 at right
        const t = THREE.MathUtils.clamp((x + halfW) / width, 0, 1);
        const c = colA.clone().lerp(colB, t);
        colorArr[i * 3 + 0] = c.r;
        colorArr[i * 3 + 1] = c.g;
        colorArr[i * 3 + 2] = c.b;
    }

    geometry.setAttribute('color', new THREE.BufferAttribute(colorArr, 3));
    return geometry;
}

export class VectorVisualization32 {
    /**
     * @param {THREE.Color|number|string} startColor – Leftmost colour of each component's gradient.
     * @param {THREE.Color|number|string} endColor   – Rightmost colour of each component's gradient.
     * @param {THREE.Vector3} [initialPosition]      – Position of the entire vector group.
     */
    constructor(startColor = null, endColor = null, initialPosition = new THREE.Vector3()) {
        this.group = new THREE.Group();
        this.group.position.copy(initialPosition);
        this.group.userData.label = 'Vector32';

        // ------------------------------------------------------------------
        // Generate key colours for 32 switches. 33 colours → 32 transitions.
        // If caller provided explicit colours, use them for first/last keys.
        // ------------------------------------------------------------------
        this.keyColors = generateRandomKeyColors(GROUP_COUNT + 1);
        if (startColor !== null) this.keyColors[0]               = new THREE.Color(startColor);
        if (endColor   !== null) this.keyColors[this.keyColors.length - 1] = new THREE.Color(endColor);

        // Width of one component for positioning
        const compWidth = PRISM_BASE_WIDTH * UNITS_PER_GROUP * WIDTH_SCALE;
        const gap       = PRISM_BASE_WIDTH * WIDTH_SCALE * 0.2;

        // Build each component with its own gradient (keyColors[i] → keyColors[i+1])
        for (let i = 0; i < GROUP_COUNT; i++) {
            const geometry = createGradientGeometry(this.keyColors[i], this.keyColors[i + 1]);
            const material = new THREE.MeshBasicMaterial({ vertexColors: true });
            const mesh = new THREE.Mesh(geometry, material);
            mesh.scale.set(1, UNIFORM_HEIGHT, 1);
            const x = (i - GROUP_COUNT / 2 + 0.5) * (compWidth + gap);
            mesh.position.set(x, UNIFORM_HEIGHT / 2, 0);
            this.group.add(mesh);
        }
    }

    /** Clean up GPU resources */
    dispose() {
        this.group.children.forEach(child => {
            if (child.geometry) child.geometry.dispose();
            if (child.material) child.material.dispose();
        });
    }
} 