import * as THREE from '/node_modules/three/build/three.module.js';
import { OrbitControls } from '/node_modules/three/examples/jsm/controls/OrbitControls.js';
import { GUI } from '/node_modules/lil-gui/dist/lil-gui.esm.js'; // Import GUI from node_modules
// Adjust path to import the component relative to the project root
import { WeightMatrixVisualization } from '../../src/components/WeightMatrixVisualization.js';

// --- Minimal Scene Setup --- 
const scene = new THREE.Scene();
// Add a soft background color to make the bright plane more visible
scene.background = new THREE.Color(0x111122);

const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
camera.position.set(0, 8, 60); // Move camera significantly further back

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(window.devicePixelRatio);
document.body.appendChild(renderer.domElement);

// Basic lighting
const ambientLight = new THREE.AmbientLight(0xffffff, 0.5); // Reduce intensity
scene.add(ambientLight);

// Main directional light (from top-right-front)
const directionalLight1 = new THREE.DirectionalLight( 0xffffff, 0.8 );
directionalLight1.position.set(5, 10, 7);
scene.add( directionalLight1 );

// Add a second directional light (from top-left-front) to fill shadows
const directionalLight2 = new THREE.DirectionalLight( 0xffffff, 0.4 );
directionalLight2.position.set(-5, 5, 7); 
scene.add( directionalLight2 );

// Add a third light from below to illuminate the bottom face
const bottomLight = new THREE.DirectionalLight( 0xffffff, 0.3 ); // Subtle intensity
bottomLight.position.set(0, -10, 5); // Position below and slightly in front
scene.add( bottomLight );

// Add a bright plane below to see through the slits
function addBrightBackPlane() {
    // Create a wide, high plane to ensure it's visible through the slits
    const planeGeometry = new THREE.PlaneGeometry(50, 50);
    
    // Create a bright material that emits light (will be visible even in shadow)
    const planeMaterial = new THREE.MeshStandardMaterial({
        color: 0xffff00, // Bright yellow
        emissive: 0xffff00, // Self-illuminating
        emissiveIntensity: 0.5, // Moderate emission
        side: THREE.DoubleSide // Visible from both sides
    });
    
    const plane = new THREE.Mesh(planeGeometry, planeMaterial);
    
    // Position the plane below and behind the trapezoid
    // Assumed trapezoid is centered at origin
    plane.position.set(0, -10, -15);
    
    // Rotate the plane to face upward
    plane.rotation.x = -Math.PI / 2;
    
    scene.add(plane);
    return plane;
}

// Add the bright plane
const brightPlane = addBrightBackPlane();

// Controls to inspect it
const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(0, 0, 0); // Target the origin
controls.enableDamping = true;
controls.dampingFactor = 0.05;

// --- Instantiate and Add the Component ---
const initialParams = {
    width: 8,
    height: 4,
    depth: 30,
    topWidthFactor: 0.7,
    cornerRadius: 0.8,
    numberOfSlits: 10, // Start with some slits visible
    slitWidth: 0.2,
    slitDepthFactor: 1.0, // Add new depth factor (1.0 = full depth)
    slitWidthFactor: 0.9
};

const testMatrix = new WeightMatrixVisualization(null, new THREE.Vector3(0, 0, 0),
    initialParams.width,
    initialParams.height,
    initialParams.depth,
    initialParams.topWidthFactor,
    initialParams.cornerRadius,
    initialParams.numberOfSlits,
    initialParams.slitWidth,
    initialParams.slitDepthFactor,
    initialParams.slitWidthFactor
);
scene.add(testMatrix.group);

// --- Setup GUI ---
const gui = new GUI();
const guiParams = { ...initialParams }; // Clone initial params for GUI
delete guiParams.slitColor; // Remove from guiParams
delete guiParams.slitOpacity; // Remove from guiParams
delete guiParams.slitHeight; // Remove old height param
guiParams.metalness = 0.1; // Add initial material params to guiParams
guiParams.roughness = 0.7;

function updateMatrixGeometry() {
    testMatrix.updateGeometry(guiParams);
}

function updateMatrixMaterial() {
    testMatrix.setMaterialProperties(guiParams); 
}

// Geometry Folder
const geometryFolder = gui.addFolder('Geometry');
geometryFolder.add(guiParams, 'width', 1, 20, 0.1).name('Width').onChange(updateMatrixGeometry);
geometryFolder.add(guiParams, 'height', 1, 10, 0.1).name('Height').onChange(updateMatrixGeometry);
geometryFolder.add(guiParams, 'depth', 1, 50, 0.5).name('Depth').onChange(updateMatrixGeometry);
geometryFolder.add(guiParams, 'topWidthFactor', 0.1, 1.0, 0.05).name('Top Width Factor').onChange(updateMatrixGeometry);
geometryFolder.add(guiParams, 'cornerRadius', 0.1, 2.0, 0.05).name('Corner Radius').onChange(updateMatrixGeometry);
geometryFolder.close(); // Start closed

// Slits Folder
const slitsFolder = gui.addFolder('Slits');
slitsFolder.add(guiParams, 'numberOfSlits', 0, 15, 1).name('Number').onChange(updateMatrixGeometry);
slitsFolder.add(guiParams, 'slitWidth', 0.05, 1.0, 0.05).name('Thickness').onChange(updateMatrixGeometry);
slitsFolder.add(guiParams, 'slitWidthFactor', 0.1, 1.5, 0.05).name('Width Factor').onChange(updateMatrixGeometry);

// Add control for slit depth factor
slitsFolder.add(guiParams, 'slitDepthFactor', 0, 1, 0.01)
    .name('Depth Factor')
    .onChange(updateMatrixGeometry);

slitsFolder.open(); // Start open

// Material Folder
const materialFolder = gui.addFolder('Material');
materialFolder.add(guiParams, 'metalness', 0, 1, 0.01).name('Metalness').onChange(updateMatrixMaterial);
materialFolder.add(guiParams, 'roughness', 0, 1, 0.01).name('Roughness').onChange(updateMatrixMaterial);
materialFolder.close(); // Start closed

// --- Animation Loop ---
const darkBlue = new THREE.Color(0x00008B); // Dark blue
const lightBlue = new THREE.Color(0xADD8E6); // Light blue
const pulseColor = new THREE.Color(); // Color to lerp into

function animate() {
    requestAnimationFrame(animate);
    controls.update();

    // Color Pulsation
    const time = performance.now() / 1000; // Time in seconds
    // Use Math.sin for smooth oscillation between -1 and 1, then map to 0-1
    const pulseFactor = (Math.sin(time * Math.PI * 0.5) + 1) / 2; // Oscillates over 4 seconds (PI * 0.5)
    pulseColor.lerpColors(darkBlue, lightBlue, pulseFactor);
    testMatrix.setColor(pulseColor);

    // Add any other test animations here if desired
    // testMatrix.animatePulse(time);

    renderer.render(scene, camera);
}

// Handle Resize
window.addEventListener('resize', () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
});

animate(); // Start the loop 