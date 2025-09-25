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
    cameraTarget: camTarget,
    enableBloom: true
});

// Show GPT canvas immediately
gptCanvas.style.display = 'block';
try {
    const eng = pipeline.engine;
    eng.renderer.shadowMap.enabled = false;
    eng.scene.traverse((obj) => {
        if (obj.isMesh) { obj.castShadow = false; obj.receiveShadow = false; }
        if (obj.isLight) { obj.castShadow = false; }
    });

    // Transform the base scene into a neon sci-fi space dock aesthetic.
    const scene = eng.scene;
    const renderer = eng.renderer;
    scene.background = new THREE.Color(0x020010);
    scene.fog = new THREE.FogExp2(0x020010, 0.000018);
    renderer.setClearColor(0x020010, 1);
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.35;
    if (eng.composer) {
        eng.composer.passes.forEach((pass) => {
            if (pass && pass.constructor && pass.constructor.name === 'UnrealBloomPass') {
                pass.strength = 1.05;
                pass.radius = 0.65;
                pass.threshold = 0.52;
            }
        });
    }

    const ambient = scene.children.find((c) => c.isAmbientLight);
    if (ambient) {
        ambient.color.set(0x2060ff);
        ambient.intensity = 0.42;
    }
    const mainDir = scene.children.find((c) => c.isDirectionalLight);
    if (mainDir) {
        mainDir.color.set(0x5cdfff);
        mainDir.intensity = 1.15;
        mainDir.position.set(18000, 22000, 12000);
    }

    const rimLight = new THREE.DirectionalLight(0xff67ff, 0.55);
    rimLight.position.set(-16000, 14000, -9000);
    scene.add(rimLight);

    const hemi = new THREE.HemisphereLight(0x2af3ff, 0x050011, 0.55);
    scene.add(hemi);

    const starGeometry = new THREE.BufferGeometry();
    const STAR_COUNT = 2500;
    const starPositions = new Float32Array(STAR_COUNT * 3);
    const radiusMin = 28000;
    const radiusMax = 120000;
    for (let i = 0; i < STAR_COUNT; i++) {
        const theta = Math.random() * Math.PI * 2;
        const phi = Math.acos(THREE.MathUtils.randFloatSpread(2));
        const radius = THREE.MathUtils.lerp(radiusMin, radiusMax, Math.random() ** 0.6);
        const sinPhi = Math.sin(phi);
        const x = Math.cos(theta) * sinPhi * radius;
        const y = Math.cos(phi) * radius;
        const z = Math.sin(theta) * sinPhi * radius;
        starPositions[i * 3] = x;
        starPositions[i * 3 + 1] = y;
        starPositions[i * 3 + 2] = z;
    }
    starGeometry.setAttribute('position', new THREE.BufferAttribute(starPositions, 3));
    const starMaterial = new THREE.PointsMaterial({
        color: 0x74faff,
        size: 280,
        sizeAttenuation: true,
        transparent: true,
        opacity: 0.45,
        blending: THREE.AdditiveBlending,
        depthWrite: false
    });
    const starField = new THREE.Points(starGeometry, starMaterial);
    starField.name = 'StarfieldBackdrop';
    starField.renderOrder = -10;
    scene.add(starField);

    const grid = new THREE.GridHelper(80000, 60, 0x1fefff, 0x1fefff);
    grid.position.y = -4500;
    const gridMaterials = Array.isArray(grid.material) ? grid.material : [grid.material];
    gridMaterials.forEach((mat, idx) => {
        mat.transparent = true;
        mat.opacity = idx === 0 ? 0.18 : 0.07;
        mat.depthWrite = false;
        mat.color.set(idx === 0 ? 0x19f0ff : 0x5b8cff);
    });
    grid.renderOrder = -9;
    scene.add(grid);

    const ringGeometry = new THREE.RingGeometry(18000, 26000, 128, 1);
    const ringMaterial = new THREE.MeshBasicMaterial({
        color: 0x18f6ff,
        side: THREE.DoubleSide,
        transparent: true,
        opacity: 0.22,
        blending: THREE.AdditiveBlending,
        depthWrite: false
    });
    const energyRing = new THREE.Mesh(ringGeometry, ringMaterial);
    energyRing.rotation.x = Math.PI / 2;
    energyRing.position.y = -2500;
    energyRing.name = 'EnergyRing';
    scene.add(energyRing);

    const haloGeometry = new THREE.CylinderGeometry(0, 18000, 2800, 36, 1, true);
    const haloMaterial = new THREE.MeshBasicMaterial({
        color: 0x944dff,
        transparent: true,
        opacity: 0.16,
        blending: THREE.AdditiveBlending,
        side: THREE.DoubleSide,
        depthWrite: false
    });
    const holoCone = new THREE.Mesh(haloGeometry, haloMaterial);
    holoCone.position.y = -2500;
    holoCone.name = 'HoloCone';
    scene.add(holoCone);

    const startTime = performance.now();
    const priorOnBeforeRender = scene.onBeforeRender;
    scene.onBeforeRender = function (...args) {
        if (typeof priorOnBeforeRender === 'function') {
            priorOnBeforeRender.apply(this, args);
        }
        const elapsed = (performance.now() - startTime) * 0.001;
        starField.rotation.y = elapsed * 0.04;
        starField.rotation.x = Math.sin(elapsed * 0.12) * 0.05;
        energyRing.rotation.z = elapsed * 0.18;
        energyRing.material.opacity = 0.18 + Math.sin(elapsed * 2.2) * 0.05;
        haloMaterial.opacity = 0.12 + Math.cos(elapsed * 1.6) * 0.04;
        gridMaterials.forEach((mat, idx) => {
            const base = idx === 0 ? 0.16 : 0.06;
            mat.opacity = base + Math.sin(elapsed * 0.7 + idx) * 0.015;
        });
    };
} catch (_) {}

// Embedding matrices (static visuals)
try {
    const headBlue = new THREE.Color(0x18f6ff);
    const headGreen = new THREE.Color(0x944dff);
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
        opacity: 1.0,
        transparent: false,
        emissive: new THREE.Color(0x047f8c),
        emissiveIntensity: 0.8,
        metalness: 0.75,
        roughness: 0.25
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
        opacity: 1.0,
        transparent: false,
        emissive: new THREE.Color(0x340079),
        emissiveIntensity: 0.9,
        metalness: 0.78,
        roughness: 0.22
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
        lnTop.setColor(new THREE.Color(0x0d1027));
        lnTop.setMaterialProperties({
            opacity: 1.0,
            transparent: false,
            emissive: new THREE.Color(0x1c2cff),
            emissiveIntensity: 0.45,
            metalness: 0.65,
            roughness: 0.35
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
        vocabTop.setColor(new THREE.Color(0x18f6ff));
        vocabTop.setMaterialProperties({
            opacity: 0.92,
            transparent: true,
            emissive: new THREE.Color(0x0d9be0),
            emissiveIntensity: 1.1,
            metalness: 0.5,
            roughness: 0.18
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
const TYPE_TEXT = '// INITIALIZING GPT-2 CORE';
const TYPE_DELAY_CAP = 110; // ms
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
            color: 0x18f6ff,
            emissive: new THREE.Color(0x0d9be0),
            emissiveIntensity: 1.2,
            metalness: 0.5,
            roughness: 0.25,
            transparent: true,
            opacity: 0
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
