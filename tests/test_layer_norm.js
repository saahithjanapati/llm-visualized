import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { GUI } from 'lil-gui';
import { LayerNormalizationVisualization } from '/src/components/LayerNormalizationVisualization.js';

// ---- Scene setup ----
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x111122);

const camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 1000);
camera.position.set(0, 6, 25);

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(window.devicePixelRatio);
document.body.appendChild(renderer.domElement);

// Lighting
scene.add(new THREE.AmbientLight(0xffffff, 0.5));
const dirLight = new THREE.DirectionalLight(0xffffff, 0.8);
dirLight.position.set(5, 10, 7);
scene.add(dirLight);

// Controls
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.05;

// ---- Instantiate the layer norm component ----
const params = {
    width: 8,
    height: 4,
    depth: 10,
    wallThickness: 0.5,
    numberOfHoles: 8,
    holeWidth: 0.4,
    holeWidthFactor: 0.8,
    metalness: 0.15,
    roughness: 0.6
};

const layerNormVis = new LayerNormalizationVisualization(
    new THREE.Vector3(0, 0, 0),
    params.width,
    params.height,
    params.depth,
    params.wallThickness,
    params.numberOfHoles,
    params.holeWidth,
    params.holeWidthFactor
);
scene.add(layerNormVis.group);

// ---- GUI ----
const gui = new GUI();
const geomFolder = gui.addFolder('Geometry');
geomFolder.add(params, 'width', 2, 1000, 0.1).onChange(() => {
    layerNormVis.updateGeometry(params);
});
geomFolder.add(params, 'height', 1, 1000, 0.1).onChange(() => {
    layerNormVis.updateGeometry(params);
});
geomFolder.add(params, 'depth', 1, 1000, 0.1).onChange(() => {
    layerNormVis.updateGeometry(params);
});
geomFolder.add(params, 'wallThickness', 0.1, 3, 0.05).onChange(() => {
    layerNormVis.updateGeometry(params);
});
geomFolder.add(params, 'numberOfHoles', 0, 20, 1).onChange(() => {
    layerNormVis.updateGeometry(params);
});
geomFolder.add(params, 'holeWidth', 0.05, 2, 0.05).onChange(() => {
    layerNormVis.updateGeometry(params);
});
geomFolder.add(params, 'holeWidthFactor', 0.1, 1.5, 0.05).onChange(() => {
    layerNormVis.updateGeometry(params);
});
geomFolder.close();

const matFolder = gui.addFolder('Material');
matFolder.add(params, 'metalness', 0, 1, 0.01).onChange(() => {
    layerNormVis.setMaterialProperties(params);
});
matFolder.add(params, 'roughness', 0, 1, 0.01).onChange(() => {
    layerNormVis.setMaterialProperties(params);
});
matFolder.close();

// ---- Handle resize ----
window.addEventListener('resize', () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
});

// ---- Animation loop ----
function animate() {
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
}
animate(); 