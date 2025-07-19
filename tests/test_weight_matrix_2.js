import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { GUI } from 'lil-gui';
import { WeightMatrixVisualization } from '/src/components/WeightMatrixVisualization.js';
import { MHA_OUTPUT_PROJECTION_MATRIX_COLOR } from '/src/animations/LayerAnimationConstants.js';

// --- Minimal Scene Setup --- 
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x111122);

const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
camera.position.set(0, 15, 150); // Adjusted for larger matrix

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(window.devicePixelRatio);
document.body.appendChild(renderer.domElement);

// Basic lighting
const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
scene.add(ambientLight);

const directionalLight1 = new THREE.DirectionalLight(0xffffff, 0.8);
directionalLight1.position.set(5, 10, 7);
scene.add(directionalLight1);

const directionalLight2 = new THREE.DirectionalLight(0xffffff, 0.4);
directionalLight2.position.set(-5, 5, 7);
scene.add(directionalLight2);

const bottomLight = new THREE.DirectionalLight(0xffffff, 0.3);
bottomLight.position.set(0, -10, 5);
scene.add(bottomLight);

// Add a bright plane below to see through the slits
function addBrightBackPlane() {
    const planeGeometry = new THREE.PlaneGeometry(300, 300);
    const planeMaterial = new THREE.MeshStandardMaterial({
        color: 0xffff00,
        emissive: 0xffff00,
        emissiveIntensity: 0.5,
        side: THREE.DoubleSide
    });
    const plane = new THREE.Mesh(planeGeometry, planeMaterial);
    plane.position.set(0, -50, -50);
    plane.rotation.x = -Math.PI / 2;
    scene.add(plane);
    return plane;
}

// COMMENT_OUT_PLANE
// const brightPlane = addBrightBackPlane(); // Removed bright background plane

// Controls
const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(0, 0, 0);
controls.enableDamping = true;
controls.dampingFactor = 0.05;

// --- Keyboard Controls Setup ---
const keyboardState = {};
const panSpeed = 0.5;

document.addEventListener('keydown', (event) => {
    keyboardState[event.code] = true;
});

document.addEventListener('keyup', (event) => {
    keyboardState[event.code] = false;
});

// --- Instantiate and Add the Component ---
const initialParams = {
    bottomWidth: 150,          // bottom width of the trapezoid
    topWidth:   150,          // top width starts equal to bottom (rectangular)
    height:     30,
    depth:      100,
    lanes:      5,            // corresponds to numberOfSlits
    cornerRadius: 20,
    slitWidth: 20,
    slitDepthFactor: 1.0,
    slitBottomWidthFactor: 0.92,
    slitTopWidthFactor: 0.92
};

const topWidthFactor = initialParams.topWidth / initialParams.bottomWidth;
const testMatrix = new WeightMatrixVisualization(
    null,
    new THREE.Vector3(0, 0, 0),
    initialParams.bottomWidth,
    initialParams.height,
    initialParams.depth,
    topWidthFactor,
    initialParams.cornerRadius,
    initialParams.lanes, // numberOfSlits
    initialParams.slitWidth,
    initialParams.slitDepthFactor,
    initialParams.slitBottomWidthFactor,
    initialParams.slitTopWidthFactor
);
// Immediately set the correct colour for the output-projection matrix
// to match the final colour used in the MHSA animation.
const matrixColor = new THREE.Color(MHA_OUTPUT_PROJECTION_MATRIX_COLOR);
testMatrix.setColor(matrixColor);
// Ensure caps and walls are fully opaque so top/bottom faces are clearly visible
testMatrix.setMaterialProperties({ transparent: false, opacity: 1 });
scene.add(testMatrix.group);

// --- Setup GUI ---
const gui = new GUI();
const guiParams = { ...initialParams };
// Keep material params separate so we don't expose them as geometry controls
const materialParams = {
    metalness: 0.1,
    roughness: 0.7,
    opacity: 1.0,
    emissiveIntensity: 0.0,
    color: '#ff99ff' // initial colour matches constant
};

// Now that materialParams is defined, apply initial material settings
updateMatrixMaterial();

function updateMatrixGeometry() {
    // Derive topWidthFactor from the two independent width values
    const topWidthFactorComputed = guiParams.topWidth / guiParams.bottomWidth;
    const updateObj = {
        width: guiParams.bottomWidth,
        topWidthFactor: topWidthFactorComputed,
        numberOfSlits: guiParams.lanes,
        depth: guiParams.depth,
        height: guiParams.height,
        cornerRadius: guiParams.cornerRadius,
        slitWidth: guiParams.slitWidth,
        slitDepthFactor: guiParams.slitDepthFactor
    };
    testMatrix.updateGeometry(updateObj);
}

function updateMatrixMaterial() {
    // Apply base colour
    testMatrix.setColor(new THREE.Color(materialParams.color));
    // Update physical material properties
    testMatrix.setMaterialProperties({
        metalness: materialParams.metalness,
        roughness: materialParams.roughness,
        opacity: materialParams.opacity,
        transparent: materialParams.opacity < 1.0
    });
    // Update emissive channel
    testMatrix.setEmissive(new THREE.Color(materialParams.color), materialParams.emissiveIntensity);
}

// Geometry Folder
const geometryFolder = gui.addFolder('Geometry');
geometryFolder.add(guiParams, 'bottomWidth', 10, 300, 1).name('Bottom Width').onChange(updateMatrixGeometry);
geometryFolder.add(guiParams, 'topWidth', 10, 300, 1).name('Top Width').onChange(updateMatrixGeometry);
geometryFolder.add(guiParams, 'height', 1, 100, 1).name('Height').onChange(updateMatrixGeometry);
geometryFolder.add(guiParams, 'depth', 1, 200, 1).name('Depth').onChange(updateMatrixGeometry);
geometryFolder.add(guiParams, 'cornerRadius', 0, 50, 1).name('Corner Radius').onChange(updateMatrixGeometry);
geometryFolder.open();

// Slits Folder
const slitsFolder = gui.addFolder('Lanes (Slits)');
slitsFolder.add(guiParams, 'lanes', 0, 20, 1).name('Lane Count').onChange(updateMatrixGeometry);
slitsFolder.add(guiParams, 'slitWidth', 0.05, 50, 0.05).name('Slit Thickness').onChange(updateMatrixGeometry);
slitsFolder.add(guiParams, 'slitDepthFactor', 0, 1, 0.01).name('Slit Depth Factor').onChange(updateMatrixGeometry);
slitsFolder.open();

// Material Folder
const materialFolder = gui.addFolder('Material');
materialFolder.addColor(materialParams, 'color').name('Color').onChange(updateMatrixMaterial);
materialFolder.add(materialParams, 'opacity', 0, 1, 0.01).name('Opacity').onChange(updateMatrixMaterial);
materialFolder.add(materialParams, 'emissiveIntensity', 0, 5, 0.05).name('Emissive Intensity').onChange(updateMatrixMaterial);
materialFolder.add(materialParams, 'metalness', 0, 1, 0.01).name('Metalness').onChange(updateMatrixMaterial);
materialFolder.add(materialParams, 'roughness', 0, 1, 0.01).name('Roughness').onChange(updateMatrixMaterial);
materialFolder.close();

// --- Animation Loop ---
function animate() {
    requestAnimationFrame(animate);

    // --- Handle Keyboard Panning ---
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

    controls.update();

    renderer.render(scene, camera);
}

// Handle Resize
window.addEventListener('resize', () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
});

animate();