import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { VectorNormalizationVisualization } from '/src/components/VectorNormalizationVisualization.js';

// Scene setup
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
camera.position.set(0, 5, 20); // Position camera to view the vector

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(window.devicePixelRatio);
document.body.appendChild(renderer.domElement);

// Basic lighting
const ambientLight = new THREE.AmbientLight(0xffffff, 1.0);
scene.add(ambientLight);
const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
directionalLight.position.set(0, 1, 1);
scene.add(directionalLight);

// Controls
const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(0, 0, 0);
controls.enableDamping = true;
controls.dampingFactor = 0.05;

// Keyboard controls
const keyboardState = {};
const panSpeed = 0.5;

document.addEventListener('keydown', (event) => {
    keyboardState[event.code] = true;
});

document.addEventListener('keyup', (event) => {
    keyboardState[event.code] = false;
});

// Create the vector normalization visualization
const vectorNormalization = new VectorNormalizationVisualization(new THREE.Vector3(0, 0, 0));
scene.add(vectorNormalization.group);

// UI controls
document.getElementById('startAnimation').addEventListener('click', () => {
    vectorNormalization.startAnimation();
});

document.getElementById('reset').addEventListener('click', () => {
    vectorNormalization.reset();
});

document.getElementById('generateNewData').addEventListener('click', () => {
    vectorNormalization.reset();
    vectorNormalization.generateNewData();
});

// Animation loop
function animate() {
    requestAnimationFrame(animate);

    // Handle keyboard controls
    const forward = new THREE.Vector3();
    camera.getWorldDirection(forward);
    const right = new THREE.Vector3().crossVectors(camera.up, forward).normalize();
    const up = new THREE.Vector3().crossVectors(forward, right).normalize();

    let didPan = false;
    const panVector = new THREE.Vector3();

    if (keyboardState['KeyW'] || keyboardState['ArrowUp']) {
        panVector.add(up);
        didPan = true;
    }
    if (keyboardState['KeyS'] || keyboardState['ArrowDown']) {
        panVector.sub(up);
        didPan = true;
    }
    if (keyboardState['KeyA'] || keyboardState['ArrowLeft']) {
        panVector.sub(right);
        didPan = true;
    }
    if (keyboardState['KeyD'] || keyboardState['ArrowRight']) {
        panVector.add(right);
        didPan = true;
    }

    if (didPan) {
        panVector.normalize().multiplyScalar(panSpeed);
        camera.position.add(panVector);
        controls.target.add(panVector);
    }

    // Update the vector normalization animation
    vectorNormalization.update(performance.now());

    controls.update();
    renderer.render(scene, camera);
}

// Handle window resize
window.addEventListener('resize', () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
});

// Start animation loop
animate(); 