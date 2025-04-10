import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';

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

// --- GPT-2 Vector Visualization ---

const VECTOR_LENGTH = 100; // Reverted length
const SPHERE_RADIUS = 0.15; // Reverted radius
const SPHERE_DIAMETER = SPHERE_RADIUS * 2;
const VECTOR_VISUAL_WIDTH = VECTOR_LENGTH * SPHERE_DIAMETER; // Reverted width: ~30
const SPAWN_Y = -50; // Start lower, below view
const DESPAWN_Y = 90; // End higher, above view
const SPAWN_X_RANGE = 70; // Increased horizontal spawn range significantly
const SPAWN_Z_RANGE = 10; // Increased Z range for more depth variation (was 5)
const VECTOR_SPEED = 0.1;
const EPSILON = 1e-5; // Small value for Layer Norm stability


const vectors = [];
let spawnTimer = 0;
const SPAWN_INTERVAL = 120; // Faster spawning (was 360)

// Use Uniform random number generator [-1, 1)
function uniformRandom() {
    return Math.random() * 2 - 1;
}

// Map a value (normalized, potentially outside -1 to 1) to a rainbow color (HSL)
function mapValueToColor(value) {
    // Clamp or scale the normalized value to the -1 to 1 range for hue mapping
    // Simple clamping for now:
    const clampedValue = Math.max(-1, Math.min(1, value / 2)); // Divide by 2 assuming norm keeps most data in [-2, 2]

    const hue = (clampedValue + 1) / 2; // Full hue range based on clamped value
    const saturation = 1.0;
    // Use constant lightness for now after normalization
    const lightness = 0.55;
    return new THREE.Color().setHSL(hue, saturation, lightness);
}

// Create a single vector visualization with Layer Normalization
function createVectorVisualization() {
    const vectorData = [];
    for (let i = 0; i < VECTOR_LENGTH; i++) {
        vectorData.push(uniformRandom());
    }

    // --- Apply Layer Normalization --- (within this specific vector)
    // 1. Calculate mean
    const sum = vectorData.reduce((acc, val) => acc + val, 0);
    const mean = sum / VECTOR_LENGTH;

    // 2. Calculate variance
    const varianceSum = vectorData.reduce((acc, val) => acc + Math.pow(val - mean, 2), 0);
    const variance = varianceSum / VECTOR_LENGTH;

    // 3. Calculate standard deviation
    const stdDev = Math.sqrt(variance + EPSILON); // Add epsilon for stability

    // 4. Normalize data
    const normalizedVectorData = vectorData.map(val => (val - mean) / stdDev);
    // --- End Layer Normalization ---

    const vectorGroup = new THREE.Group();
    const baseSphereGeometry = new THREE.SphereGeometry(SPHERE_RADIUS, 8, 8);
    const yScale = 4;
    const zScale = yScale;
    const scaleMatrix = new THREE.Matrix4().makeScale(1, yScale, zScale);
    baseSphereGeometry.applyMatrix4(scaleMatrix);

    for (let i = 0; i < VECTOR_LENGTH; i++) {
        // Use the NORMALIZED value for color mapping
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
        ellipse.position.y = 0;
        vectorGroup.add(ellipse);
    }
    baseSphereGeometry.dispose();

    // Set initial position using new SPAWN_Y and wider SPAWN_X_RANGE
    vectorGroup.position.x = (Math.random() - 0.5) * SPAWN_X_RANGE;
    vectorGroup.position.y = SPAWN_Y;
    vectorGroup.position.z = (Math.random() - 0.5) * SPAWN_Z_RANGE;

    // Assign fixed speed
    vectorGroup.speed = VECTOR_SPEED;

    return vectorGroup;
}

// Removed the single test cube
// const geometry = new THREE.BoxGeometry(1, 1, 1);
// const material = new THREE.MeshStandardMaterial({ color: 0x00ff00, metalness: 0.3, roughness: 0.5 });
// const cube = new THREE.Mesh(geometry, material);
// scene.add(cube);

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
        const newVector = createVectorVisualization();
        scene.add(newVector);
        vectors.push(newVector);
    }

    // Animate existing vectors
    for (let i = vectors.length - 1; i >= 0; i--) {
        const vector = vectors[i];
        vector.position.y += vector.speed; // Use fixed speed

        // Remove vectors that have moved off screen (uses new DESPAWN_Y)
        if (vector.position.y > DESPAWN_Y) {
            scene.remove(vector);
            // Dispose cloned geometry and materials
            vector.children.forEach(child => {
                if (child.geometry) child.geometry.dispose();
                if (child.material) child.material.dispose();
            });
            vectors.splice(i, 1);
        }
    }

    controls.update();

    composer.render();
}

animate();
