import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { EffectComposer } from 'three/examples/jsm/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/examples/jsm/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/examples/jsm/postprocessing/UnrealBloomPass.js';
import { GUI } from 'lil-gui';
import { WeightMatrixVisualization } from '/src/components/WeightMatrixVisualization.js';
import {
  MLP_MATRIX_PARAMS_UP,
  MLP_MATRIX_PARAMS_DOWN,
  LN_PARAMS,
  BRANCH_X,
  MLP_INTER_MATRIX_GAP,
  MHA_MATRIX_PARAMS,
  MHA_INTERNAL_MATRIX_SPACING
} from '/src/utils/constants.js';
import {
  MHA_FINAL_Q_COLOR,
  MHA_FINAL_K_COLOR,
  MHA_FINAL_V_COLOR,
  MHA_OUTPUT_PROJECTION_MATRIX_PARAMS,
  MHA_OUTPUT_PROJECTION_MATRIX_COLOR
} from '/src/animations/LayerAnimationConstants.js';

// Basic scene
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x111111);

const camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 100000);
camera.position.set(0, 500, 2000);

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
document.body.appendChild(renderer.domElement);

// Post-processing: Bloom (for glow)
const composer = new EffectComposer(renderer);
const renderPass = new RenderPass(scene, camera);
composer.addPass(renderPass);
const bloomPass = new UnrealBloomPass(new THREE.Vector2(window.innerWidth, window.innerHeight), 1.2, 0.4, 0.2);
composer.addPass(bloomPass);

// Lights
scene.add(new THREE.AmbientLight(0xffffff, 0.6));
const dir = new THREE.DirectionalLight(0xffffff, 0.8);
dir.position.set(500, 1000, 750);
scene.add(dir);

// Controls
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.06;

// Create two MLP matrices replicating the stack setup
// Use same Z-depth as LN_PARAMS.depth so slits align with lanes like the stack
const zDepth = LN_PARAMS.depth;

// Up-projection matrix (768 → 3072)
const upMatrix = new WeightMatrixVisualization(
  null,
  new THREE.Vector3(BRANCH_X, 0, 0),
  MLP_MATRIX_PARAMS_UP.width,
  MLP_MATRIX_PARAMS_UP.height,
  zDepth,
  MLP_MATRIX_PARAMS_UP.topWidthFactor,
  MLP_MATRIX_PARAMS_UP.cornerRadius,
  MLP_MATRIX_PARAMS_UP.numberOfSlits,
  MLP_MATRIX_PARAMS_UP.slitWidth,
  MLP_MATRIX_PARAMS_UP.slitDepthFactor,
  MLP_MATRIX_PARAMS_UP.slitBottomWidthFactor,
  MLP_MATRIX_PARAMS_UP.slitTopWidthFactor
);
scene.add(upMatrix.group);

// Down-projection matrix (3072 → 768)
const downMatrix = new WeightMatrixVisualization(
  null,
  new THREE.Vector3(BRANCH_X, 0, 0),
  MLP_MATRIX_PARAMS_DOWN.width,
  MLP_MATRIX_PARAMS_DOWN.height,
  zDepth,
  MLP_MATRIX_PARAMS_DOWN.topWidthFactor,
  MLP_MATRIX_PARAMS_DOWN.cornerRadius,
  MLP_MATRIX_PARAMS_DOWN.numberOfSlits,
  MLP_MATRIX_PARAMS_DOWN.slitWidth,
  MLP_MATRIX_PARAMS_DOWN.slitDepthFactor,
  MLP_MATRIX_PARAMS_DOWN.slitBottomWidthFactor,
  MLP_MATRIX_PARAMS_DOWN.slitTopWidthFactor
);
scene.add(downMatrix.group);

// Stack vertically like the layer
const centerYUp = 0;
const centerYDown = centerYUp + (MLP_MATRIX_PARAMS_UP.height / 2) + MLP_INTER_MATRIX_GAP + (MLP_MATRIX_PARAMS_DOWN.height / 2);
upMatrix.group.position.set(BRANCH_X, centerYUp, 0);
downMatrix.group.position.set(BRANCH_X, centerYDown, 0);

// ─────────────────────────────────────────────────────────────
// Add a single Q, K, V head set next to the MLP matrices
// ─────────────────────────────────────────────────────────────
const qkvY = centerYUp; // align to bottom MLP for simplicity
const qBaseX = BRANCH_X + 600; // offset to the right of MLP block

const qX = qBaseX;
const kX = qBaseX + MHA_INTERNAL_MATRIX_SPACING;
const vX = kX + MHA_INTERNAL_MATRIX_SPACING;

const qMatrix = new WeightMatrixVisualization(
  null,
  new THREE.Vector3(qX, qkvY, 0),
  MHA_MATRIX_PARAMS.width,
  MHA_MATRIX_PARAMS.height,
  zDepth,
  MHA_MATRIX_PARAMS.topWidthFactor,
  MHA_MATRIX_PARAMS.cornerRadius,
  MHA_MATRIX_PARAMS.numberOfSlits,
  MHA_MATRIX_PARAMS.slitWidth,
  MHA_MATRIX_PARAMS.slitDepthFactor,
  MHA_MATRIX_PARAMS.slitBottomWidthFactor,
  MHA_MATRIX_PARAMS.slitTopWidthFactor
);
scene.add(qMatrix.group);

const kMatrix = new WeightMatrixVisualization(
  null,
  new THREE.Vector3(kX, qkvY, 0),
  MHA_MATRIX_PARAMS.width,
  MHA_MATRIX_PARAMS.height,
  zDepth,
  MHA_MATRIX_PARAMS.topWidthFactor,
  MHA_MATRIX_PARAMS.cornerRadius,
  MHA_MATRIX_PARAMS.numberOfSlits,
  MHA_MATRIX_PARAMS.slitWidth,
  MHA_MATRIX_PARAMS.slitDepthFactor,
  MHA_MATRIX_PARAMS.slitBottomWidthFactor,
  MHA_MATRIX_PARAMS.slitTopWidthFactor
);
scene.add(kMatrix.group);

const vMatrix = new WeightMatrixVisualization(
  null,
  new THREE.Vector3(vX, qkvY, 0),
  MHA_MATRIX_PARAMS.width,
  MHA_MATRIX_PARAMS.height,
  zDepth,
  MHA_MATRIX_PARAMS.topWidthFactor,
  MHA_MATRIX_PARAMS.cornerRadius,
  MHA_MATRIX_PARAMS.numberOfSlits,
  MHA_MATRIX_PARAMS.slitWidth,
  MHA_MATRIX_PARAMS.slitDepthFactor,
  MHA_MATRIX_PARAMS.slitBottomWidthFactor,
  MHA_MATRIX_PARAMS.slitTopWidthFactor
);
scene.add(vMatrix.group);

qMatrix.setColor(new THREE.Color(MHA_FINAL_Q_COLOR));
kMatrix.setColor(new THREE.Color(MHA_FINAL_K_COLOR));
vMatrix.setColor(new THREE.Color(MHA_FINAL_V_COLOR));
qMatrix.setMaterialProperties({ opacity: 1.0, transparent: false });
kMatrix.setMaterialProperties({ opacity: 1.0, transparent: false });
vMatrix.setMaterialProperties({ opacity: 1.0, transparent: false });

// Ensure unique geometries so gradients can be distinct per matrix
function ensureUniqueGeometries(matrixObj) {
  if (!matrixObj) return;
  if (matrixObj.mesh && matrixObj.mesh.geometry) {
    matrixObj.mesh.geometry = matrixObj.mesh.geometry.clone();
  }
  if (matrixObj.frontCapMesh && matrixObj.frontCapMesh.geometry) {
    matrixObj.frontCapMesh.geometry = matrixObj.frontCapMesh.geometry.clone();
  }
  if (matrixObj.backCapMesh && matrixObj.backCapMesh.geometry) {
    matrixObj.backCapMesh.geometry = matrixObj.backCapMesh.geometry.clone();
  }
}
ensureUniqueGeometries(qMatrix);
ensureUniqueGeometries(kMatrix);
ensureUniqueGeometries(vMatrix);

// ─────────────────────────────────────────────────────────────
// Additive glow overlays to make the whole matrix bloom evenly
// ─────────────────────────────────────────────────────────────
function buildGlowMaterial(opacity = 0.5) {
  return new THREE.MeshBasicMaterial({
    vertexColors: true,
    transparent: true,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
    opacity,
    toneMapped: false
  });
}

function addGlowOverlayForMesh(baseMesh, opacity) {
  if (!baseMesh) return null;
  const mat = buildGlowMaterial(opacity);
  let overlay = null;
  if (baseMesh.isInstancedMesh) {
    overlay = new THREE.InstancedMesh(baseMesh.geometry, mat, baseMesh.count);
    const tmp = new THREE.Matrix4();
    for (let i = 0; i < baseMesh.count; i++) {
      baseMesh.getMatrixAt(i, tmp);
      overlay.setMatrixAt(i, tmp);
    }
    overlay.instanceMatrix.needsUpdate = true;
  } else {
    overlay = new THREE.Mesh(baseMesh.geometry, mat);
  }
  overlay.renderOrder = (baseMesh.renderOrder || 0) + 10;
  return overlay;
}

function addGlowOverlay(matrixObj, opacity = 0.6, scale = 1.001) {
  if (!matrixObj || !matrixObj.group) return;
  const overlays = {};
  overlays.body = addGlowOverlayForMesh(matrixObj.mesh, opacity);
  overlays.front = addGlowOverlayForMesh(matrixObj.frontCapMesh, opacity);
  overlays.back  = addGlowOverlayForMesh(matrixObj.backCapMesh, opacity);
  // Attach to the matrix group so it follows transforms
  if (overlays.body) {
    overlays.body.scale.setScalar(scale);
    matrixObj.group.add(overlays.body);
  }
  if (overlays.front) {
    overlays.front.scale.setScalar(scale);
    matrixObj.group.add(overlays.front);
  }
  if (overlays.back) {
    overlays.back.scale.setScalar(scale);
    matrixObj.group.add(overlays.back);
  }
  matrixObj.group.userData = matrixObj.group.userData || {};
  matrixObj.group.userData.glowOverlays = overlays;
}

function setGlowOverlayVisibility(matrixObj, visible) {
  const go = matrixObj?.group?.userData?.glowOverlays;
  if (!go) return;
  ['body','front','back'].forEach(k => { if (go[k]) go[k].visible = visible; });
}

function setGlowOverlayOpacity(matrixObj, opacity) {
  const go = matrixObj?.group?.userData?.glowOverlays;
  if (!go) return;
  ['body','front','back'].forEach(k => { if (go[k] && go[k].material) { go[k].material.opacity = opacity; go[k].material.needsUpdate = true; } });
}

function setGlowOverlayScale(matrixObj, scale) {
  const go = matrixObj?.group?.userData?.glowOverlays;
  if (!go) return;
  ['body','front','back'].forEach(k => { if (go[k]) go[k].scale.setScalar(scale); });
}

// ─────────────────────────────────────────────────────────────
// Output Projection Matrix (rectangular)
// ─────────────────────────────────────────────────────────────
const outHeight = (MHA_OUTPUT_PROJECTION_MATRIX_PARAMS.heightFactor || 1) * MHA_MATRIX_PARAMS.height;
const QKV_TO_OUT_INITIAL_GAP = MHA_INTERNAL_MATRIX_SPACING * 2; // extra horizontal spacing beyond V
const outX = vX + MHA_INTERNAL_MATRIX_SPACING + QKV_TO_OUT_INITIAL_GAP; // place after V with extra gap

const outMatrix = new WeightMatrixVisualization(
  null,
  new THREE.Vector3(outX, qkvY, 0),
  MHA_OUTPUT_PROJECTION_MATRIX_PARAMS.width,
  outHeight,
  zDepth,
  MHA_OUTPUT_PROJECTION_MATRIX_PARAMS.topWidthFactor,
  MHA_OUTPUT_PROJECTION_MATRIX_PARAMS.cornerRadius,
  MHA_OUTPUT_PROJECTION_MATRIX_PARAMS.numberOfSlits,
  MHA_OUTPUT_PROJECTION_MATRIX_PARAMS.slitWidth,
  MHA_OUTPUT_PROJECTION_MATRIX_PARAMS.slitDepthFactor,
  MHA_OUTPUT_PROJECTION_MATRIX_PARAMS.slitBottomWidthFactor,
  MHA_OUTPUT_PROJECTION_MATRIX_PARAMS.slitTopWidthFactor
);
scene.add(outMatrix.group);
outMatrix.setColor(new THREE.Color(MHA_OUTPUT_PROJECTION_MATRIX_COLOR));
outMatrix.setMaterialProperties({ opacity: 1.0, transparent: false });

// Base color white so vertex colors (gradient) are visible
const inactive = new THREE.Color(0xffffff);
upMatrix.setColor(inactive);
upMatrix.setMaterialProperties({ opacity: 1.0, transparent: false });
downMatrix.setColor(inactive);
downMatrix.setMaterialProperties({ opacity: 1.0, transparent: false });

// Helper: apply per-vertex vertical gradient to a BufferGeometry
function applyVerticalGradientToGeometry(geometry, bottomHex, topHex) {
  if (!geometry || !geometry.attributes || !geometry.attributes.position) return;
  geometry.computeBoundingBox();
  const bb = geometry.boundingBox;
  if (!bb) return;
  const minY = bb.min.y;
  const maxY = bb.max.y;
  const range = Math.max(1e-6, maxY - minY);
  const pos = geometry.attributes.position;
  const count = pos.count;
  let colorAttr = geometry.getAttribute('color');
  if (!colorAttr || colorAttr.count !== count) {
    colorAttr = new THREE.BufferAttribute(new Float32Array(count * 3), 3);
    geometry.setAttribute('color', colorAttr);
  }
  const cBottom = new THREE.Color(bottomHex);
  const cTop = new THREE.Color(topHex);
  const c = new THREE.Color();
  for (let i = 0; i < count; i++) {
    const y = pos.getY(i);
    const t = (y - minY) / range;
    c.copy(cBottom).lerp(cTop, t);
    colorAttr.setXYZ(i, c.r, c.g, c.b);
  }
  colorAttr.needsUpdate = true;
}

// Helper: enable vertex colors on a material or array of materials
function enableVertexColors(mat) {
  const apply = (m) => { if (!m) return; m.vertexColors = true; m.color.set(0xffffff); m.needsUpdate = true; };
  if (Array.isArray(mat)) mat.forEach(apply); else apply(mat);
}

// Apply gradients:
// - Bottom matrix (upMatrix): bottom #c78b2f → top #e9d04c
// - Top matrix (downMatrix): bottom #e9d04c → top #c78b2f (reverse)
const BOTTOM_COL = '#c78b2f';
const TOP_COL = '#e9d04c';

// upMatrix
if (upMatrix.mesh && upMatrix.mesh.geometry) {
  applyVerticalGradientToGeometry(upMatrix.mesh.geometry, BOTTOM_COL, TOP_COL);
  enableVertexColors(upMatrix.mesh.material);
}
if (upMatrix.frontCapMesh && upMatrix.frontCapMesh.geometry) {
  applyVerticalGradientToGeometry(upMatrix.frontCapMesh.geometry, BOTTOM_COL, TOP_COL);
  enableVertexColors(upMatrix.frontCapMesh.material);
}
if (upMatrix.backCapMesh && upMatrix.backCapMesh.geometry) {
  applyVerticalGradientToGeometry(upMatrix.backCapMesh.geometry, BOTTOM_COL, TOP_COL);
  enableVertexColors(upMatrix.backCapMesh.material);
}

// downMatrix (reversed)
if (downMatrix.mesh && downMatrix.mesh.geometry) {
  applyVerticalGradientToGeometry(downMatrix.mesh.geometry, TOP_COL, BOTTOM_COL);
  enableVertexColors(downMatrix.mesh.material);
}
if (downMatrix.frontCapMesh && downMatrix.frontCapMesh.geometry) {
  applyVerticalGradientToGeometry(downMatrix.frontCapMesh.geometry, TOP_COL, BOTTOM_COL);
  enableVertexColors(downMatrix.frontCapMesh.material);
}
if (downMatrix.backCapMesh && downMatrix.backCapMesh.geometry) {
  applyVerticalGradientToGeometry(downMatrix.backCapMesh.geometry, TOP_COL, BOTTOM_COL);
  enableVertexColors(downMatrix.backCapMesh.material);
}

// GUI for quick tweaks
const gui = new GUI();
const groupFolder = gui.addFolder('Selection');
const moveFolder = gui.addFolder('Movement');
const gradientFolder = gui.addFolder('Gradient');
const bloomFolder = gui.addFolder('Glow / Bloom');
const layoutFolder = gui.addFolder('Layout');
const qkvFolder = gui.addFolder('Q/K/V Gradients');
const qFolder = qkvFolder.addFolder('Q Gradient');
const kFolder = qkvFolder.addFolder('K Gradient');
const vFolder = qkvFolder.addFolder('V Gradient');
const outFolder = gui.addFolder('Output Projection Gradient');

const state = {
  selection: 'both', // both | up | down
  moveStepXY: 25,
  moveStepZ: 50,
  emissive: '#000000',
  // Gradient colors
  bottomMatrixBottom: '#c78b2f', // upMatrix bottom color
  bottomMatrixTop: '#e9d04c',    // upMatrix top color
  topMatrixBottom: '#e9d04c',    // downMatrix bottom color
  topMatrixTop: '#c78b2f',       // downMatrix top color
  // Q/K/V gradients (default to solid via same color top/bottom)
  qBottom: `#${new THREE.Color(MHA_FINAL_Q_COLOR).getHexString()}`,
  qTop: `#${new THREE.Color(MHA_FINAL_Q_COLOR).getHexString()}`,
  kBottom: `#${new THREE.Color(MHA_FINAL_K_COLOR).getHexString()}`,
  kTop: `#${new THREE.Color(MHA_FINAL_K_COLOR).getHexString()}`,
  vBottom: `#${new THREE.Color(MHA_FINAL_V_COLOR).getHexString()}`,
  vTop: `#${new THREE.Color(MHA_FINAL_V_COLOR).getHexString()}`,
  // Output projection gradient (default to solid via same color)
  outBottom: `#${new THREE.Color(MHA_OUTPUT_PROJECTION_MATRIX_COLOR).getHexString()}`,
  outTop: `#${new THREE.Color(MHA_OUTPUT_PROJECTION_MATRIX_COLOR).getHexString()}`,
  // Layout
  qkvToOutGap: QKV_TO_OUT_INITIAL_GAP,
  // Bloom controls
  enableBloom: true,
  bloomStrength: 1.2,
  bloomRadius: 0.4,
  bloomThreshold: 0.2,
  // Emissive intensity (applied to all matrices)
  emissiveIntensity: 0.6,
  glowOverlayOpacity: 0.35,
  glowOverlayScale: 1.002,
  resetPositions: () => {
    upMatrix.group.position.set(BRANCH_X, centerYUp, 0);
    downMatrix.group.position.set(BRANCH_X, centerYDown, 0);
  }
};

groupFolder.add(state, 'selection', ['both', 'up', 'down']).name('Active');
moveFolder.add(state, 'moveStepXY', 1, 200, 1).name('XY Step');
moveFolder.add(state, 'moveStepZ', 1, 400, 1).name('Z Step');
moveFolder.addColor(state, 'emissive').name('Emissive').onChange(() => {
  const c = new THREE.Color(state.emissive);
  upMatrix.setEmissive(c, state.emissiveIntensity);
  downMatrix.setEmissive(c, state.emissiveIntensity);
  qMatrix.setEmissive(c, state.emissiveIntensity);
  kMatrix.setEmissive(c, state.emissiveIntensity);
  vMatrix.setEmissive(c, state.emissiveIntensity);
  outMatrix.setEmissive(c, state.emissiveIntensity);
});
moveFolder.add(state, 'resetPositions').name('Reset');

// Layout controls
layoutFolder.add(state, 'qkvToOutGap', 0, MHA_INTERNAL_MATRIX_SPACING * 8, 1)
  .name('QKV → Output gap')
  .onChange(() => {
    outMatrix.group.position.x = vX + MHA_INTERNAL_MATRIX_SPACING + state.qkvToOutGap;
  });

// Bloom controls
bloomFolder.add(state, 'enableBloom').name('Enable Bloom');
bloomFolder.add(state, 'bloomStrength', 0, 5, 0.01).name('Strength').onChange(() => {
  bloomPass.strength = state.bloomStrength;
});
bloomFolder.add(state, 'bloomRadius', 0, 2, 0.01).name('Radius').onChange(() => {
  bloomPass.radius = state.bloomRadius;
});
bloomFolder.add(state, 'bloomThreshold', 0, 1, 0.001).name('Threshold').onChange(() => {
  bloomPass.threshold = state.bloomThreshold;
});
bloomFolder.add(state, 'emissiveIntensity', 0, 3, 0.01).name('Emissive Intensity').onChange(() => {
  const c = new THREE.Color(state.emissive);
  upMatrix.setEmissive(c, state.emissiveIntensity);
  downMatrix.setEmissive(c, state.emissiveIntensity);
  qMatrix.setEmissive(c, state.emissiveIntensity);
  kMatrix.setEmissive(c, state.emissiveIntensity);
  vMatrix.setEmissive(c, state.emissiveIntensity);
  outMatrix.setEmissive(c, state.emissiveIntensity);
});
bloomFolder.add(state, 'glowOverlayOpacity', 0, 1, 0.01).name('Overlay Opacity').onChange(() => {
  [upMatrix, downMatrix, qMatrix, kMatrix, vMatrix, outMatrix].forEach(m => setGlowOverlayOpacity(m, state.glowOverlayOpacity));
});
bloomFolder.add(state, 'glowOverlayScale', 1.0, 1.02, 0.0005).name('Overlay Scale').onChange(() => {
  [upMatrix, downMatrix, qMatrix, kMatrix, vMatrix, outMatrix].forEach(m => setGlowOverlayScale(m, state.glowOverlayScale));
});
bloomFolder.open();

// Helper to (re)apply gradients based on current state
function applyAllGradients() {
  // upMatrix (bottomMatrix)
  if (upMatrix.mesh && upMatrix.mesh.geometry) {
    applyVerticalGradientToGeometry(upMatrix.mesh.geometry, state.bottomMatrixBottom, state.bottomMatrixTop);
    enableVertexColors(upMatrix.mesh.material);
  }
  if (upMatrix.frontCapMesh && upMatrix.frontCapMesh.geometry) {
    applyVerticalGradientToGeometry(upMatrix.frontCapMesh.geometry, state.bottomMatrixBottom, state.bottomMatrixTop);
    enableVertexColors(upMatrix.frontCapMesh.material);
  }
  if (upMatrix.backCapMesh && upMatrix.backCapMesh.geometry) {
    applyVerticalGradientToGeometry(upMatrix.backCapMesh.geometry, state.bottomMatrixBottom, state.bottomMatrixTop);
    enableVertexColors(upMatrix.backCapMesh.material);
  }

  // downMatrix (topMatrix)
  if (downMatrix.mesh && downMatrix.mesh.geometry) {
    applyVerticalGradientToGeometry(downMatrix.mesh.geometry, state.topMatrixBottom, state.topMatrixTop);
    enableVertexColors(downMatrix.mesh.material);
  }
  if (downMatrix.frontCapMesh && downMatrix.frontCapMesh.geometry) {
    applyVerticalGradientToGeometry(downMatrix.frontCapMesh.geometry, state.topMatrixBottom, state.topMatrixTop);
    enableVertexColors(downMatrix.frontCapMesh.material);
  }
  if (downMatrix.backCapMesh && downMatrix.backCapMesh.geometry) {
    applyVerticalGradientToGeometry(downMatrix.backCapMesh.geometry, state.topMatrixBottom, state.topMatrixTop);
    enableVertexColors(downMatrix.backCapMesh.material);
  }
}

// Gradient UI
gradientFolder.addColor(state, 'bottomMatrixBottom').name('Bottom matrix: bottom').onChange(applyAllGradients);
gradientFolder.addColor(state, 'bottomMatrixTop').name('Bottom matrix: top').onChange(applyAllGradients);
gradientFolder.addColor(state, 'topMatrixBottom').name('Top matrix: bottom').onChange(applyAllGradients);
gradientFolder.addColor(state, 'topMatrixTop').name('Top matrix: top').onChange(applyAllGradients);
gradientFolder.open();

// Initial application (in case defaults changed above)
applyAllGradients();

// Helper: generic matrix gradient applier
function applyGradientToMatrix(matrixObj, bottomHex, topHex) {
  if (!matrixObj) return;
  if (matrixObj.mesh && matrixObj.mesh.geometry) {
    applyVerticalGradientToGeometry(matrixObj.mesh.geometry, bottomHex, topHex);
    enableVertexColors(matrixObj.mesh.material);
  }
  if (matrixObj.frontCapMesh && matrixObj.frontCapMesh.geometry) {
    applyVerticalGradientToGeometry(matrixObj.frontCapMesh.geometry, bottomHex, topHex);
    enableVertexColors(matrixObj.frontCapMesh.material);
  }
  if (matrixObj.backCapMesh && matrixObj.backCapMesh.geometry) {
    applyVerticalGradientToGeometry(matrixObj.backCapMesh.geometry, bottomHex, topHex);
    enableVertexColors(matrixObj.backCapMesh.material);
  }
}

function applyQGradient() { applyGradientToMatrix(qMatrix, state.qBottom, state.qTop); }
function applyKGradient() { applyGradientToMatrix(kMatrix, state.kBottom, state.kTop); }
function applyVGradient() { applyGradientToMatrix(vMatrix, state.vBottom, state.vTop); }

// Q/K/V gradient controls split per-matrix
qFolder.addColor(state, 'qBottom').name('Bottom').onChange(applyQGradient);
qFolder.addColor(state, 'qTop').name('Top').onChange(applyQGradient);
kFolder.addColor(state, 'kBottom').name('Bottom').onChange(applyKGradient);
kFolder.addColor(state, 'kTop').name('Top').onChange(applyKGradient);
vFolder.addColor(state, 'vBottom').name('Bottom').onChange(applyVGradient);
vFolder.addColor(state, 'vTop').name('Top').onChange(applyVGradient);
qkvFolder.open();
qFolder.open();
kFolder.open();
vFolder.open();

// Output projection gradient controls
function applyOutGradient() { applyGradientToMatrix(outMatrix, state.outBottom, state.outTop); }
outFolder.addColor(state, 'outBottom').name('Bottom').onChange(applyOutGradient);
outFolder.addColor(state, 'outTop').name('Top').onChange(applyOutGradient);
outFolder.open();

// Initial application for Q/K/V and Output gradients
applyQGradient();
applyKGradient();
applyVGradient();
applyOutGradient();

// Build glow overlays for all matrices (scaled slightly to avoid z-fighting)
addGlowOverlay(upMatrix, 0.35, 1.002);
addGlowOverlay(downMatrix, 0.35, 1.002);
addGlowOverlay(qMatrix, 0.35, 1.002);
addGlowOverlay(kMatrix, 0.35, 1.002);
addGlowOverlay(vMatrix, 0.35, 1.002);
addGlowOverlay(outMatrix, 0.35, 1.002);

// Helper: which objects are currently controlled
function getTargets() {
  if (state.selection === 'up') return [upMatrix.group];
  if (state.selection === 'down') return [downMatrix.group];
  return [upMatrix.group, downMatrix.group];
}

// Keyboard movement: WASD / arrows for X/Y, Q/E for Z.
const help = [
  'MLP Weight Matrices Mover',
  '',
  '1/2/3: Select Up / Down / Both',
  'W/S or ArrowUp/ArrowDown: Move +Y / -Y',
  'A/D or ArrowLeft/ArrowRight: Move -X / +X',
  'Q/E: Move -Z / +Z',
  'R: Reset positions',
].join('\n');
const helpDiv = document.getElementById('helpOverlay');
if (helpDiv) helpDiv.textContent = help;

function move(dx, dy, dz) {
  const targets = getTargets();
  for (const g of targets) {
    g.position.x += dx;
    g.position.y += dy;
    g.position.z += dz;
  }
}

window.addEventListener('keydown', (e) => {
  const stepXY = state.moveStepXY;
  const stepZ = state.moveStepZ;

  switch (e.key.toLowerCase()) {
    case '1': state.selection = 'up'; break;
    case '2': state.selection = 'down'; break;
    case '3': state.selection = 'both'; break;
    case 'w': move(0, +stepXY, 0); break;
    case 's': move(0, -stepXY, 0); break;
    case 'a': move(-stepXY, 0, 0); break;
    case 'd': move(+stepXY, 0, 0); break;
    case 'arrowup': move(0, +stepXY, 0); break;
    case 'arrowdown': move(0, -stepXY, 0); break;
    case 'arrowleft': move(-stepXY, 0, 0); break;
    case 'arrowright': move(+stepXY, 0, 0); break;
    case 'q': move(0, 0, -stepZ); break;
    case 'e': move(0, 0, +stepZ); break;
    case 'r': state.resetPositions(); break;
    default: break;
  }
});

// Render loop
function animate() {
  requestAnimationFrame(animate);
  controls.update();
  if (state.enableBloom) {
    composer.render();
  } else {
    renderer.render(scene, camera);
  }
}
animate();

// Resize
window.addEventListener('resize', () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});


