import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { EffectComposer } from 'three/examples/jsm/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/examples/jsm/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/examples/jsm/postprocessing/UnrealBloomPass.js';

// Import components and utils
import { VectorVisualization } from '/src/components/VectorVisualization.js';
import {
    SPAWN_Y,
    DESPAWN_Y,
    SPAWN_X_RANGE,
    SPAWN_Z_RANGE,
    VECTOR_SPEED,
    SPAWN_INTERVAL
} from '/src/utils/constants.js';

// Scene setup
const scene = new THREE.Scene();
// Remove Fog for now
// scene.fog = new THREE.Fog(0x000000, 200, 600);

const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
camera.position.set(0, 20, 60); // Pulled camera back slightly for wider view
camera.lookAt(0, 20, 0); // Keep looking at the same height

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(window.devicePixelRatio);
document.body.appendChild(renderer.domElement);

// Post-processing Setup
const composer = new EffectComposer(renderer);
const renderPass = new RenderPass(scene, camera);
composer.addPass(renderPass);

const bloomPass = new UnrealBloomPass(
    new THREE.Vector2(window.innerWidth, window.innerHeight),
    1.2, // strength
    0.4, // radius
    0.85 // threshold - pixels brighter than this value will bloom
);
composer.addPass(bloomPass);

// Lighting
const ambientLight = new THREE.AmbientLight(0xffffff, 1.0);
scene.add(ambientLight);
const directionalLight = new THREE.DirectionalLight( 0xffffff, 0.8 );
directionalLight.position.set(0, 1, 1);
scene.add( directionalLight );

// Controls
const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(0, 20, 0); // Keep target
controls.enableDamping = true;
controls.dampingFactor = 0.05;
controls.screenSpacePanning = false;
controls.minDistance = 10;
controls.maxDistance = 180; // Slightly increased max distance

// --- GPT-2 Vector Visualization --- - Now managed via class

// Removed constants - now in src/utils/constants.js
// Removed helper functions - now in src/utils/

const vectors = []; // Store instances of VectorVisualization
let spawnTimer = 0;
// SPAWN_INTERVAL imported from constants

// Removed mapValueToColor - now in src/utils/colors.js

// Removed createVectorVisualization function - logic moved to VectorVisualization class

// Removed the single test cube

// Handle window resize
window.addEventListener('resize', () => {
    const width = window.innerWidth;
    const height = window.innerHeight;

    camera.aspect = width / height;
    camera.updateProjectionMatrix();

    renderer.setSize(width, height);
    composer.setSize(width, height);
    renderer.setPixelRatio(window.devicePixelRatio);
});

// Animation Loop
function animate() {
    requestAnimationFrame(animate);

    // Spawn new vectors periodically
    spawnTimer++;
    if (spawnTimer > SPAWN_INTERVAL) {
        spawnTimer = 0;

        // Create an instance of VectorVisualization
        const initialPosition = new THREE.Vector3(
            (Math.random() - 0.5) * SPAWN_X_RANGE,
            SPAWN_Y,
            (Math.random() - 0.5) * SPAWN_Z_RANGE
        );
        const newVectorVis = new VectorVisualization(null, initialPosition);
        newVectorVis.speed = VECTOR_SPEED; // Set the speed for this instance

        scene.add(newVectorVis.group); // Add the group to the scene
        vectors.push(newVectorVis); // Store the instance
    }

    // Animate existing vectors
    for (let i = vectors.length - 1; i >= 0; i--) {
        const vectorVis = vectors[i];
        vectorVis.group.position.y += vectorVis.speed; // Move the group

        // Remove vectors that have moved off screen
        if (vectorVis.group.position.y > DESPAWN_Y) {
            scene.remove(vectorVis.group); // Remove the group from the scene
            vectorVis.dispose(); // Call dispose to clean up geometries/materials
            vectors.splice(i, 1); // Remove from the array
        }
    }

    controls.update();

    composer.render();
}

animate();
