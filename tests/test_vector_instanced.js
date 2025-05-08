import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
// Adjust path to import the component relative to the project root
// Use absolute path from root for robustness
import { VectorVisualizationInstanced } from '/src/components/VectorVisualizationInstanced.js';

// --- Minimal Scene Setup ---
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
camera.position.set(0, 5, 20); // Position camera to view the single vector

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(window.devicePixelRatio);
document.body.appendChild(renderer.domElement);

// Basic lighting
const ambientLight = new THREE.AmbientLight(0xffffff, 1.0);
scene.add(ambientLight);
const directionalLight = new THREE.DirectionalLight( 0xffffff, 0.8 );
directionalLight.position.set(0, 1, 1);
scene.add( directionalLight );

// Controls to inspect it
const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(0, 0, 0); // Target the origin
controls.enableDamping = true;
controls.dampingFactor = 0.05;

// --- Keyboard Controls Setup ---
const keyboardState = {};
const panSpeed = 0.5; // Adjust panning speed as needed

document.addEventListener('keydown', (event) => {
    keyboardState[event.code] = true;
});

document.addEventListener('keyup', (event) => {
    keyboardState[event.code] = false;
});

// --- Instantiate and Add the Component ---

// Create a vector at the origin using its default random data generation
const testVector = new VectorVisualizationInstanced(null, new THREE.Vector3(0, 0, 0));
scene.add(testVector.group); // Add the vector's group to the scene

// --- Minimal Animation Loop ---
function animate() {
    requestAnimationFrame(animate);

    // --- Handle Keyboard Panning ---
    const forward = new THREE.Vector3();
    camera.getWorldDirection(forward);
    const right = new THREE.Vector3().crossVectors(camera.up, forward).normalize();
    const up = new THREE.Vector3().crossVectors(forward, right).normalize(); // Ensure 'up' is orthogonal

    let didPan = false;
    const panVector = new THREE.Vector3();

    if (keyboardState['KeyW'] || keyboardState['ArrowUp']) {
        panVector.add(up); // Pan up
        didPan = true;
    }
    if (keyboardState['KeyS'] || keyboardState['ArrowDown']) {
        panVector.sub(up); // Pan down
        didPan = true;
    }
    if (keyboardState['KeyA'] || keyboardState['ArrowLeft']) {
        panVector.sub(right); // Pan left
        didPan = true;
    }
    if (keyboardState['KeyD'] || keyboardState['ArrowRight']) {
        panVector.add(right); // Pan right
        didPan = true;
    }

    if (didPan) {
        panVector.normalize().multiplyScalar(panSpeed);
        camera.position.add(panVector);
        controls.target.add(panVector); // Move the target along with the camera
    }
    // --- End Keyboard Panning ---

    controls.update();

    // Add any test animations here if desired
    // testVector.animatePulse(performance.now() / 1000); // Example, actual animation method might differ for Instanced version

    renderer.render(scene, camera);
}

// Handle Resize
window.addEventListener('resize', () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
});

animate(); // Start the loop 