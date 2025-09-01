import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { WeightMatrixVisualization } from '/src/components/WeightMatrixVisualization.js';
import {
  LN_PARAMS,
  MLP_MATRIX_PARAMS_UP,
  NUM_VECTOR_LANES,
  EMBEDDING_MATRIX_PARAMS_VOCAB,
  EMBEDDING_MATRIX_PARAMS_POSITION
} from '/src/utils/constants.js';

// Basic scene
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0e0e0e);

const camera = new THREE.PerspectiveCamera(55, window.innerWidth / window.innerHeight, 0.1, 50000);
camera.position.set(0, 800, 6000);

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
document.body.appendChild(renderer.domElement);

// Lights
scene.add(new THREE.AmbientLight(0xffffff, 0.7));
const dir = new THREE.DirectionalLight(0xffffff, 0.9);
dir.position.set(1000, 2000, 1500);
scene.add(dir);

// Controls
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.06;

// Use lane-based depth so each matrix renders as 5-lane instanced slices
const zDepth = LN_PARAMS.depth; // NUM_VECTOR_LANES dependent

// Helper to build an embedding-like matrix
function makeEmbeddingMatrix(params, position) {
  return new WeightMatrixVisualization(
    null,
    position,
    params.width,
    params.height,
    zDepth,
    params.topWidthFactor,
    params.cornerRadius,
    NUM_VECTOR_LANES,
    params.slitWidth,
    params.slitDepthFactor,
    params.slitBottomWidthFactor,
    params.slitTopWidthFactor
  );
}

// Reference MLP up-projection (narrow → wide) using 5 lanes
const mlpUp = new WeightMatrixVisualization(
  null,
  new THREE.Vector3(0, 0, 0),
  MLP_MATRIX_PARAMS_UP.width,
  MLP_MATRIX_PARAMS_UP.height,
  zDepth,
  MLP_MATRIX_PARAMS_UP.topWidthFactor,
  MLP_MATRIX_PARAMS_UP.cornerRadius,
  NUM_VECTOR_LANES,
  MLP_MATRIX_PARAMS_UP.slitWidth,
  MLP_MATRIX_PARAMS_UP.slitDepthFactor,
  MLP_MATRIX_PARAMS_UP.slitBottomWidthFactor,
  MLP_MATRIX_PARAMS_UP.slitTopWidthFactor
);
scene.add(mlpUp.group);

// Token/Vocab embedding
const vocab = makeEmbeddingMatrix(EMBEDDING_MATRIX_PARAMS_VOCAB, new THREE.Vector3(0, -900, 0));
scene.add(vocab.group);

// Positional embedding
const pos = makeEmbeddingMatrix(EMBEDDING_MATRIX_PARAMS_POSITION, new THREE.Vector3(0, -1900, 0));
scene.add(pos.group);

// Subtle base colors
const inactive = new THREE.Color(0xffffff);
mlpUp.setColor(inactive);
mlpUp.setMaterialProperties({ opacity: 1.0, transparent: false });
vocab.setColor(inactive.clone());
vocab.setMaterialProperties({ opacity: 1.0, transparent: false });
pos.setColor(inactive.clone());
pos.setMaterialProperties({ opacity: 1.0, transparent: false });

// Render loop
function animate() {
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
}

window.addEventListener('resize', () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

animate();
