import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { WeightMatrixVisualization } from '../../components/WeightMatrixVisualization.js';
import { WeightMatrixVisualizationNoCSG } from '../../components/WeightMatrixVisualizationNoCSG.js';
import { MHA_FINAL_Q_COLOR, MHA_FINAL_K_COLOR, MHA_FINAL_V_COLOR } from '../../animations/LayerAnimationConstants.js';
import { MHA_MATRIX_PARAMS } from '../../utils/constants.js';

const canvas = document.getElementById('comparisonCanvas');
const laneLegend = document.getElementById('laneLegend');

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x060a12);

const camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 1, 6000);
camera.position.set(0, 70, 2800);

const renderer = new THREE.WebGLRenderer({
    canvas,
    antialias: true,
    alpha: false
});
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(window.innerWidth, window.innerHeight);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.minDistance = 120;
controls.maxDistance = 9000;
controls.target.set(0, 0, 0);

scene.add(new THREE.AmbientLight(0xffffff, 0.65));

const keyLight = new THREE.DirectionalLight(0xaed2ff, 0.95);
keyLight.position.set(180, 300, 260);
scene.add(keyLight);

const fillLight = new THREE.DirectionalLight(0xffd9a8, 0.6);
fillLight.position.set(-240, -120, 200);
scene.add(fillLight);

const laneRows = [
    { label: 'Q final', color: MHA_FINAL_Q_COLOR, y: 220 },
    { label: 'K final', color: MHA_FINAL_K_COLOR, y: 0 },
    { label: 'V final', color: MHA_FINAL_V_COLOR, y: -220 }
];

const TEST_LANE_COUNT = 3;

const matrixParams = {
    width: MHA_MATRIX_PARAMS.width,
    height: MHA_MATRIX_PARAMS.height,
    depth: MHA_MATRIX_PARAMS.depth,
    topWidthFactor: MHA_MATRIX_PARAMS.topWidthFactor,
    cornerRadius: MHA_MATRIX_PARAMS.cornerRadius,
    // Keep the comparison fixed at 3 lanes for this test view.
    numberOfSlits: TEST_LANE_COUNT,
    slitWidth: MHA_MATRIX_PARAMS.slitWidth,
    slitDepthFactor: MHA_MATRIX_PARAMS.slitDepthFactor,
    slitBottomWidthFactor: MHA_MATRIX_PARAMS.slitBottomWidthFactor,
    slitTopWidthFactor: MHA_MATRIX_PARAMS.slitTopWidthFactor
};

const objects = [];
const rowHtml = [];
laneRows.forEach((row) => {
    const leftMatrix = new WeightMatrixVisualization(
        null,
        new THREE.Vector3(-290, row.y, 0),
        matrixParams.width,
        matrixParams.height,
        matrixParams.depth,
        matrixParams.topWidthFactor,
        matrixParams.cornerRadius,
        matrixParams.numberOfSlits,
        matrixParams.slitWidth,
        matrixParams.slitDepthFactor,
        matrixParams.slitBottomWidthFactor,
        matrixParams.slitTopWidthFactor,
        false
    );
    leftMatrix.setColor(new THREE.Color(row.color));
    leftMatrix.setMaterialProperties({ opacity: 0.97, transparent: false, emissiveIntensity: 0.2 });
    scene.add(leftMatrix.group);

    const rightMatrix = new WeightMatrixVisualizationNoCSG(
        null,
        new THREE.Vector3(290, row.y, 0),
        matrixParams.width,
        matrixParams.height,
        matrixParams.depth,
        matrixParams.topWidthFactor,
        matrixParams.cornerRadius,
        matrixParams.numberOfSlits,
        matrixParams.slitWidth,
        matrixParams.slitDepthFactor,
        matrixParams.slitBottomWidthFactor,
        matrixParams.slitTopWidthFactor
    );
    rightMatrix.setColor(new THREE.Color(row.color));
    rightMatrix.setMaterialProperties({ opacity: 0.97, transparent: false, emissiveIntensity: 0.2 });
    scene.add(rightMatrix.group);

    objects.push(leftMatrix, rightMatrix);
    rowHtml.push(
        `<li><span class="swatch" style="background:#${new THREE.Color(row.color).getHexString()}"></span>${row.label}</li>`
    );
});

laneLegend.innerHTML = rowHtml.join('');

const dividerGeometry = new THREE.PlaneGeometry(2, 760);
const dividerMaterial = new THREE.MeshBasicMaterial({
    color: 0x3c4a62,
    transparent: true,
    opacity: 0.45,
    side: THREE.DoubleSide
});
const divider = new THREE.Mesh(dividerGeometry, dividerMaterial);
divider.position.set(0, 0, 0);
scene.add(divider);

function onResize() {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
}
window.addEventListener('resize', onResize);

function animate() {
    controls.update();
    renderer.render(scene, camera);
    requestAnimationFrame(animate);
}
animate();

window.addEventListener('beforeunload', () => {
    objects.forEach((obj) => obj.dispose && obj.dispose());
    dividerGeometry.dispose();
    dividerMaterial.dispose();
    controls.dispose();
    renderer.dispose();
});
