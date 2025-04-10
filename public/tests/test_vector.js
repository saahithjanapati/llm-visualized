import * as THREE from '/node_modules/three/build/three.module.js';
import { OrbitControls } from '/node_modules/three/examples/jsm/controls/OrbitControls.js';
// Adjust path to import the component relative to the project root
// Since this file is in public/tests, we go up two levels then into src
import { VectorVisualization } from '../../src/components/VectorVisualization.js';

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

// --- Instantiate and Add the Component ---

// Create a vector at the origin using its default random data generation
const testVector = new VectorVisualization(null, new THREE.Vector3(0, 0, 0));
scene.add(testVector.group); // Add the vector's group to the scene

// --- Minimal Animation Loop ---
function animate() {
    requestAnimationFrame(animate);
    controls.update();

    // Add any test animations here if desired
    // testVector.animatePulse(performance.now() / 1000);

    renderer.render(scene, camera);
}

// Handle Resize
window.addEventListener('resize', () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
});

animate(); // Start the loop 