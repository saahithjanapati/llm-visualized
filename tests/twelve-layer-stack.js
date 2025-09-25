import * as THREE from 'three';
import { MHSAAnimation } from '../src/animations/MHSAAnimation.js';
import { loadPrecomputedGeometries } from '../src/utils/precomputedGeometryLoader.js';
import { LayerPipeline } from '../src/engine/LayerPipeline.js';
import {
    CAPTION_TEXT_Y_POS,
    setPlaybackSpeed,
    EMBEDDING_MATRIX_PARAMS_VOCAB,
    EMBEDDING_MATRIX_PARAMS_POSITION,
    LN_PARAMS,
    LAYER_NORM_1_Y_POS,
    MLP_MATRIX_PARAMS_DOWN,
    INACTIVE_COMPONENT_COLOR,
    EMBEDDING_BOTTOM_TOP_ALIGN_OFFSET_FROM_LN1_BOTTOM,
    EMBEDDING_BOTTOM_Y_ADJUST,
    EMBEDDING_BOTTOM_VOCAB_X_OFFSET,
    EMBEDDING_BOTTOM_PAIR_GAP_X,
    EMBEDDING_BOTTOM_POS_X_OFFSET,
    TOP_EMBED_VOCAB_X_OFFSET,
    TOP_EMBED_Y_GAP_ABOVE_TOWER,
    TOP_EMBED_Y_ADJUST,
    TOP_LN_TO_TOP_EMBED_GAP
} from '../src/utils/constants.js';
import { WeightMatrixVisualization } from '../src/components/WeightMatrixVisualization.js';
import { LayerNormalizationVisualization } from '../src/components/LayerNormalizationVisualization.js';
import { MHA_FINAL_Q_COLOR, MHA_FINAL_K_COLOR } from '../src/animations/LayerAnimationConstants.js';
import { appState } from '../src/state/appState.js';
import { initIntroAnimation } from '../src/ui/introAnimation.js';
import { initStatusOverlay } from '../src/ui/statusOverlay.js';
import { initSettingsModal } from '../src/ui/settingsModal.js';
import { initPauseButton } from '../src/ui/pauseButton.js';
import { FontLoader } from 'three/examples/jsm/loaders/FontLoader.js';
import { TextGeometry } from 'three/examples/jsm/geometries/TextGeometry.js';

// Optionally load pre-baked geometries; returns instantly if disabled
await loadPrecomputedGeometries('../precomputed_components.glb');

// Skip intro typing screen for direct animation entry
appState.skipIntro = true;

// Set default playback speed to fast on load
try { setPlaybackSpeed('fast'); } catch (_) { /* no-op */ }

// GPT-2 tower – initialise immediately
MHSAAnimation.ENABLE_SELF_ATTENTION = true;
const gptCanvas = document.getElementById('gptCanvas');
const NUM_LAYERS = 12;
const camPos    = new THREE.Vector3(0, 11000, 16000);
const camTarget = new THREE.Vector3(0, 9000, 0);
const pipeline = new LayerPipeline(gptCanvas, NUM_LAYERS, {
    cameraPosition: camPos,
    cameraTarget: camTarget
});

const scene = pipeline.engine.scene;
const renderer = pipeline.engine.renderer;

// Futuristic tone mapping & atmosphere -------------------------------------------------
renderer.setClearColor(0x040612, 1);
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.08;
scene.background = null;
scene.fog = new THREE.FogExp2(0x040612, 0.000035);

const stackBounds = new THREE.Box3();
for (const layer of pipeline._layers) {
    if (layer?.root) {
        stackBounds.expandByObject(layer.root);
    }
}
const fallbackHeight = NUM_LAYERS * 1600;
const stackHeight = stackBounds.isEmpty() ? fallbackHeight : Math.max(fallbackHeight * 0.45, stackBounds.max.y - stackBounds.min.y);
const stackCenterY = stackBounds.isEmpty() ? 0 : (stackBounds.max.y + stackBounds.min.y) / 2;
const stackBottomY = stackBounds.isEmpty() ? CAPTION_TEXT_Y_POS + 1400 : stackBounds.min.y;
const stackTopY = stackBottomY + stackHeight;

const ambientLight = scene.children.find(child => child?.isAmbientLight);
if (ambientLight) {
    ambientLight.color.set(0x1a7dff);
    ambientLight.intensity = 0.45;
}
const keyLight = scene.children.find(child => child?.isDirectionalLight);
if (keyLight) {
    keyLight.color.set(0xff66ff);
    keyLight.intensity = 0.75;
    keyLight.position.set(9000, stackTopY + 5000, 6000);
}

// Neon starfield backdrop ---------------------------------------------------------------
const starGeometry = new THREE.BufferGeometry();
const STAR_COUNT = 2200;
const starPositions = new Float32Array(STAR_COUNT * 3);
for (let i = 0; i < STAR_COUNT; i++) {
    const radius = 42000 + Math.random() * 22000;
    const theta = Math.random() * Math.PI * 2;
    const phi = Math.acos(THREE.MathUtils.randFloatSpread(2));
    const x = radius * Math.sin(phi) * Math.cos(theta);
    const y = radius * Math.cos(phi);
    const z = radius * Math.sin(phi) * Math.sin(theta);
    starPositions[i * 3] = x;
    starPositions[i * 3 + 1] = y;
    starPositions[i * 3 + 2] = z;
}
starGeometry.setAttribute('position', new THREE.BufferAttribute(starPositions, 3));
const starMaterial = new THREE.PointsMaterial({
    color: new THREE.Color(0x59e5ff),
    size: 220,
    sizeAttenuation: true,
    transparent: true,
    opacity: 0.7,
    blending: THREE.AdditiveBlending,
    depthWrite: false
});
const starField = new THREE.Points(starGeometry, starMaterial);
starField.name = 'SciFiStarfield';
scene.add(starField);

// Energy platform & holographic rings ---------------------------------------------------
const platformMaterial = new THREE.MeshStandardMaterial({
    color: 0x060b17,
    metalness: 0.85,
    roughness: 0.25,
    emissive: new THREE.Color(0x08204b),
    emissiveIntensity: 0.9
});
const platformHeight = 380;
const platformGeometry = new THREE.CylinderGeometry(6200, 6600, platformHeight, 48, 1, false);
const basePlatform = new THREE.Mesh(platformGeometry, platformMaterial);
basePlatform.position.set(0, stackBottomY - platformHeight * 0.4, 0);
basePlatform.receiveShadow = false;
basePlatform.castShadow = false;
scene.add(basePlatform);

const hologramMaterial = new THREE.MeshBasicMaterial({
    color: 0x49f5ff,
    transparent: true,
    opacity: 0.32,
    side: THREE.DoubleSide,
    blending: THREE.AdditiveBlending,
    depthWrite: false
});
const hologramMaterialTop = hologramMaterial.clone();
hologramMaterialTop.color = new THREE.Color(0xff4df2);
const ringInner = 3200;
const ringOuter = 4100;
const bottomRingGeo = new THREE.RingGeometry(ringInner, ringOuter, 128, 1);
const bottomRing = new THREE.Mesh(bottomRingGeo, hologramMaterial);
bottomRing.rotation.x = Math.PI / 2;
bottomRing.position.y = stackBottomY + 400;
scene.add(bottomRing);

const topRing = new THREE.Mesh(new THREE.RingGeometry(ringInner * 0.7, ringOuter * 0.72, 128, 1), hologramMaterialTop);
topRing.rotation.x = Math.PI / 2;
topRing.position.y = stackTopY - 1200;
scene.add(topRing);

const energyHeight = stackHeight + 5200;
const energyMaterial = new THREE.MeshBasicMaterial({
    color: 0x1fe4ff,
    transparent: true,
    opacity: 0.16,
    blending: THREE.AdditiveBlending,
    side: THREE.DoubleSide,
    depthWrite: false
});
const energyColumn = new THREE.Mesh(new THREE.CylinderGeometry(1900, 1900, energyHeight, 80, 1, true), energyMaterial);
energyColumn.position.set(0, stackCenterY + 600, 0);
scene.add(energyColumn);

// Vertical holo pylons for extra framing
const pylonMaterial = new THREE.MeshBasicMaterial({
    color: 0x00f0ff,
    transparent: true,
    opacity: 0.24,
    blending: THREE.AdditiveBlending,
    depthWrite: false,
    side: THREE.DoubleSide
});
const pylonGeometry = new THREE.CylinderGeometry(120, 200, stackHeight + 3600, 12, 1, true);
const pylons = [];
const pylonRadius = 4800;
for (let i = 0; i < 4; i++) {
    const angle = (i / 4) * Math.PI * 2;
    const pylon = new THREE.Mesh(pylonGeometry, pylonMaterial);
    pylon.position.set(Math.cos(angle) * pylonRadius, stackCenterY, Math.sin(angle) * pylonRadius);
    pylon.rotation.y = angle;
    scene.add(pylon);
    pylons.push(pylon);
}

// Infinite neon grid floor ----------------------------------------------------------------
const grid = new THREE.GridHelper(64000, 140, 0x06d8ff, 0x0834ff);
grid.position.y = stackBottomY - 800;
if (Array.isArray(grid.material)) {
    grid.material.forEach(mat => {
        mat.opacity = 0.18;
        mat.transparent = true;
        mat.depthWrite = false;
    });
} else if (grid.material) {
    grid.material.opacity = 0.18;
    grid.material.transparent = true;
    grid.material.depthWrite = false;
}
scene.add(grid);

// Dynamic accent lights -------------------------------------------------------------------
const cyanLight = new THREE.PointLight(0x39e0ff, 180, 0, 2);
cyanLight.position.set(0, stackTopY + 2000, 8000);
scene.add(cyanLight);
const magentaLight = new THREE.PointLight(0xff4cf5, 160, 0, 2);
magentaLight.position.set(0, stackBottomY + 3000, -7600);
scene.add(magentaLight);

const neonElements = {
    starField,
    rings: [bottomRing, topRing],
    energyColumn,
    pylons,
    lights: [cyanLight, magentaLight]
};

const prevOnBeforeRender = scene.onBeforeRender;
const animationClock = new THREE.Clock();
scene.onBeforeRender = function onBeforeRender(rendererInstance, sceneInstance, cameraInstance, geometry, material, group) {
    if (typeof prevOnBeforeRender === 'function') {
        prevOnBeforeRender.call(this, rendererInstance, sceneInstance, cameraInstance, geometry, material, group);
    }
    const t = animationClock.getElapsedTime();
    if (neonElements.starField) {
        neonElements.starField.rotation.y += 0.00025;
        neonElements.starField.rotation.x = Math.sin(t * 0.07) * 0.08;
    }
    neonElements.rings.forEach((ring, idx) => {
        if (!ring) return;
        const scale = 1 + Math.sin(t * 1.25 + idx) * 0.08;
        ring.scale.set(scale, scale, scale);
        const pulse = 0.22 + Math.sin(t * 3.4 + idx * 1.7) * 0.12;
        ring.material.opacity = THREE.MathUtils.clamp(pulse, 0.08, 0.45);
        ring.rotation.z += 0.0008 * (idx % 2 === 0 ? 1 : -1);
    });
    if (neonElements.energyColumn) {
        const flicker = 0.16 + Math.sin(t * 2.8) * 0.04;
        neonElements.energyColumn.material.opacity = THREE.MathUtils.clamp(flicker, 0.08, 0.28);
        neonElements.energyColumn.scale.x = 1 + Math.sin(t * 1.9) * 0.02;
        neonElements.energyColumn.scale.z = neonElements.energyColumn.scale.x;
    }
    neonElements.pylons.forEach((pylon, idx) => {
        const offset = Math.sin(t * 1.6 + idx * Math.PI * 0.5) * 0.12;
        pylon.material.opacity = THREE.MathUtils.clamp(0.16 + offset, 0.08, 0.32);
    });
    neonElements.lights.forEach((light, idx) => {
        if (!light) return;
        const radius = idx === 0 ? 13200 : 9800;
        const speed = idx === 0 ? 0.28 : -0.34;
        light.position.x = Math.cos(t * speed + idx) * radius;
        light.position.z = Math.sin(t * speed + idx) * radius;
        light.position.y = stackCenterY + Math.sin(t * 0.7 + idx) * 3600;
        light.intensity = idx === 0 ? 160 + Math.sin(t * 1.4) * 20 : 140 + Math.cos(t * 1.2) * 18;
    });
};

// Show GPT canvas immediately
gptCanvas.style.display = 'block';
try {
    const eng = pipeline.engine;
    eng.renderer.shadowMap.enabled = false;
    eng.scene.traverse((obj) => {
        if (obj.isMesh) { obj.castShadow = false; obj.receiveShadow = false; }
        if (obj.isLight) { obj.castShadow = false; }
    });
} catch (_) {}

// Embedding matrices (static visuals)
try {
    const headBlue = new THREE.Color(MHA_FINAL_Q_COLOR);
    const headGreen = new THREE.Color(MHA_FINAL_K_COLOR);
    const residualYBase = LAYER_NORM_1_Y_POS - LN_PARAMS.height / 2 + EMBEDDING_BOTTOM_TOP_ALIGN_OFFSET_FROM_LN1_BOTTOM;
    const bottomVocabCenterY = residualYBase - EMBEDDING_MATRIX_PARAMS_VOCAB.height / 2 + EMBEDDING_BOTTOM_Y_ADJUST;
    const vocabBottom = new WeightMatrixVisualization(
        null,
        new THREE.Vector3(0 + EMBEDDING_BOTTOM_VOCAB_X_OFFSET, bottomVocabCenterY, 0),
        EMBEDDING_MATRIX_PARAMS_VOCAB.width,
        EMBEDDING_MATRIX_PARAMS_VOCAB.height,
        EMBEDDING_MATRIX_PARAMS_VOCAB.depth,
        EMBEDDING_MATRIX_PARAMS_VOCAB.topWidthFactor,
        EMBEDDING_MATRIX_PARAMS_VOCAB.cornerRadius,
        EMBEDDING_MATRIX_PARAMS_VOCAB.numberOfSlits,
        EMBEDDING_MATRIX_PARAMS_VOCAB.slitWidth,
        EMBEDDING_MATRIX_PARAMS_VOCAB.slitDepthFactor,
        EMBEDDING_MATRIX_PARAMS_VOCAB.slitBottomWidthFactor,
        EMBEDDING_MATRIX_PARAMS_VOCAB.slitTopWidthFactor
    );
    vocabBottom.group.userData.label = 'Vocab Embedding';
    vocabBottom.setColor(headBlue);
    vocabBottom.setMaterialProperties({
        metalness: 0.65,
        roughness: 0.35,
        emissive: new THREE.Color(0x0b4dff),
        emissiveIntensity: 0.7,
        opacity: 0.95,
        transparent: true
    });
    pipeline.engine.scene.add(vocabBottom.group);
    if (pipeline.engine && typeof pipeline.engine.registerRaycastRoot === 'function') {
        pipeline.engine.registerRaycastRoot(vocabBottom.group);
    }

    const gapX = EMBEDDING_BOTTOM_PAIR_GAP_X;
    const posX = (EMBEDDING_MATRIX_PARAMS_VOCAB.width / 2) + (EMBEDDING_MATRIX_PARAMS_POSITION.width / 2) + gapX + EMBEDDING_BOTTOM_POS_X_OFFSET + EMBEDDING_BOTTOM_VOCAB_X_OFFSET;
    const vocabBottomY = bottomVocabCenterY - EMBEDDING_MATRIX_PARAMS_VOCAB.height / 2;
    const bottomPosCenterY = vocabBottomY + EMBEDDING_MATRIX_PARAMS_POSITION.height / 2;
    const posBottom = new WeightMatrixVisualization(
        null,
        new THREE.Vector3(posX, bottomPosCenterY, 0),
        EMBEDDING_MATRIX_PARAMS_POSITION.width,
        EMBEDDING_MATRIX_PARAMS_POSITION.height,
        EMBEDDING_MATRIX_PARAMS_POSITION.depth,
        EMBEDDING_MATRIX_PARAMS_POSITION.topWidthFactor,
        EMBEDDING_MATRIX_PARAMS_POSITION.cornerRadius,
        EMBEDDING_MATRIX_PARAMS_POSITION.numberOfSlits,
        EMBEDDING_MATRIX_PARAMS_POSITION.slitWidth,
        EMBEDDING_MATRIX_PARAMS_POSITION.slitDepthFactor,
        EMBEDDING_MATRIX_PARAMS_POSITION.slitBottomWidthFactor,
        EMBEDDING_MATRIX_PARAMS_POSITION.slitTopWidthFactor
    );
    posBottom.group.userData.label = 'Positional Embedding';
    posBottom.setColor(headGreen);
    posBottom.setMaterialProperties({
        metalness: 0.6,
        roughness: 0.3,
        emissive: new THREE.Color(0x12ffdc),
        emissiveIntensity: 0.65,
        opacity: 0.95,
        transparent: true
    });
    pipeline.engine.scene.add(posBottom.group);
    if (pipeline.engine && typeof pipeline.engine.registerRaycastRoot === 'function') {
        pipeline.engine.registerRaycastRoot(posBottom.group);
    }

    const lastLayer = pipeline._layers[NUM_LAYERS - 1];
    if (lastLayer && lastLayer.mlpDown && lastLayer.mlpDown.group) {
        const tmp = new THREE.Vector3();
        lastLayer.mlpDown.group.getWorldPosition(tmp);
        const towerTopY = tmp.y + MLP_MATRIX_PARAMS_DOWN.height / 2;
        const topGap = TOP_EMBED_Y_GAP_ABOVE_TOWER;
        const topLnCenterY = towerTopY + topGap + LN_PARAMS.height / 2;
        const lnTop = new LayerNormalizationVisualization(
            new THREE.Vector3(0 + TOP_EMBED_VOCAB_X_OFFSET, topLnCenterY, 0),
            LN_PARAMS.width,
            LN_PARAMS.height,
            LN_PARAMS.depth,
            LN_PARAMS.wallThickness,
            LN_PARAMS.numberOfHoles,
            LN_PARAMS.holeWidth,
            LN_PARAMS.holeWidthFactor
        );
        lnTop.group.userData.label = 'LayerNorm (Top)';
        lnTop.setColor(new THREE.Color(INACTIVE_COMPONENT_COLOR));
        lnTop.setMaterialProperties({
            metalness: 0.55,
            roughness: 0.2,
            emissive: new THREE.Color(0x1a9fff),
            emissiveIntensity: 0.55,
            opacity: 0.98,
            transparent: true
        });
        pipeline.engine.scene.add(lnTop.group);
        if (pipeline.engine && typeof pipeline.engine.registerRaycastRoot === 'function') {
            pipeline.engine.registerRaycastRoot(lnTop.group);
        }

        const topVocabCenterY = topLnCenterY + (LN_PARAMS.height / 2) + TOP_LN_TO_TOP_EMBED_GAP + (EMBEDDING_MATRIX_PARAMS_VOCAB.height / 2) + TOP_EMBED_Y_ADJUST;
        const vocabTop = new WeightMatrixVisualization(
            null,
            new THREE.Vector3(0 + TOP_EMBED_VOCAB_X_OFFSET, topVocabCenterY, 0),
            EMBEDDING_MATRIX_PARAMS_VOCAB.width,
            EMBEDDING_MATRIX_PARAMS_VOCAB.height,
            EMBEDDING_MATRIX_PARAMS_VOCAB.depth,
            EMBEDDING_MATRIX_PARAMS_VOCAB.topWidthFactor,
            EMBEDDING_MATRIX_PARAMS_VOCAB.cornerRadius,
            EMBEDDING_MATRIX_PARAMS_VOCAB.numberOfSlits,
            EMBEDDING_MATRIX_PARAMS_VOCAB.slitWidth,
            EMBEDDING_MATRIX_PARAMS_VOCAB.slitDepthFactor,
            EMBEDDING_MATRIX_PARAMS_VOCAB.slitBottomWidthFactor,
            EMBEDDING_MATRIX_PARAMS_VOCAB.slitTopWidthFactor
        );
        vocabTop.group.rotation.z = Math.PI;
        vocabTop.group.userData.label = 'Vocab Embedding (Top)';
        vocabTop.setColor(new THREE.Color(0x0c162a));
        vocabTop.setMaterialProperties({
            metalness: 0.7,
            roughness: 0.25,
            emissive: new THREE.Color(0x45f2ff),
            emissiveIntensity: 0.6,
            opacity: 0.9,
            transparent: true
        });
        appState.vocabTopRef = vocabTop;
        pipeline.engine.scene.add(vocabTop.group);
        if (pipeline.engine && typeof pipeline.engine.registerRaycastRoot === 'function') {
            pipeline.engine.registerRaycastRoot(vocabTop.group);
        }
    }
} catch (_) { /* optional – embedding visuals are non-critical */ }

// Initialise UI modules
initIntroAnimation(pipeline, gptCanvas);
initStatusOverlay(pipeline, NUM_LAYERS);
initPauseButton(pipeline);
initSettingsModal(pipeline);

// Typewriter caption underneath GPT tower
const TYPE_TEXT = 'Can machines think?';
const TYPE_DELAY_CAP = 120; // ms
const captionLoader = new FontLoader();
captionLoader.load('https://threejs.org/examples/fonts/helvetiker_regular.typeface.json', (font) => {
    const charGroup = new THREE.Group();
    pipeline.engine.scene.add(charGroup);
    const charMeshes = [];
    const charWidths = [];
    let xOffset = 0;
    const charSpacing = 100;
    for (const ch of TYPE_TEXT) {
        if (ch === ' ') {
            xOffset += 400;
            charMeshes.push(null);
            charWidths.push(400);
            continue;
        }
        const geo = new TextGeometry(ch, {
            font,
            size: 800,
            height: 80,
            curveSegments: 6,
            bevelEnabled: true,
            bevelThickness: 4,
            bevelSize: 3,
            bevelOffset: 0,
            bevelSegments: 1
        });
        geo.computeBoundingBox();
        const width = geo.boundingBox.max.x - geo.boundingBox.min.x;
        const mat = new THREE.MeshStandardMaterial({
            color: 0xa8f3ff,
            transparent: true,
            opacity: 0,
            emissive: new THREE.Color(0x27dcff),
            emissiveIntensity: 0.9,
            metalness: 0.35,
            roughness: 0.2
        });
        const mesh = new THREE.Mesh(geo, mat);
        mesh.position.x = xOffset;
        mesh.castShadow = false;
        mesh.receiveShadow = false;
        charGroup.add(mesh);
        charMeshes.push(mesh);
        charWidths.push(width);
        xOffset += width + charSpacing;
    }
    charGroup.position.set(-xOffset / 2 + charSpacing / 2, CAPTION_TEXT_Y_POS, 0);
    const revealChar = (index) => {
        if (index >= charMeshes.length) return;
        const mesh = charMeshes[index];
        if (mesh) {
            new TWEEN.Tween(mesh.material).to({ opacity: 1 }, TYPE_DELAY_CAP).start();
        }
        setTimeout(() => revealChar(index + 1), TYPE_DELAY_CAP);
    };
    setTimeout(() => revealChar(0), 500);
    const gif = document.getElementById('loadingGif');
    if (gif) gif.style.display = 'none';
});
