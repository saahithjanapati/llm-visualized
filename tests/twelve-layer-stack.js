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
import { FontLoader } from 'three/examples/jsm/loaders/FontLoader.js';
import { TextGeometry } from 'three/examples/jsm/geometries/TextGeometry.js';
import { createSciFiPanelMaterial } from '../src/utils/sciFiMaterials.js';

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

// Show GPT canvas immediately
gptCanvas.style.display = 'block';
try {
    const eng = pipeline.engine;
    eng.renderer.shadowMap.enabled = false;
    const scene = eng.scene;
    scene.traverse((obj) => {
        if (obj.isMesh) { obj.castShadow = false; obj.receiveShadow = false; }
        if (obj.isLight) { obj.castShadow = false; }
    });

    scene.background = new THREE.Color(0x020718);
    scene.fog = new THREE.FogExp2(0x020718, 0.00011);

    if (eng.renderer) {
        eng.renderer.toneMapping = THREE.ACESFilmicToneMapping;
        eng.renderer.toneMappingExposure = 1.08;
    }

    const ambient = scene.children.find(child => child.isAmbientLight);
    if (ambient) {
        ambient.color.set(0x3f5fff);
        ambient.intensity = 0.55;
    }
    const keyLight = scene.children.find(child => child.isDirectionalLight);
    if (keyLight) {
        keyLight.color.set(0x8bd7ff);
        keyLight.intensity = 1.15;
        keyLight.position.set(4200, 9200, 3600);
        keyLight.castShadow = false;
    }

    const rimLight = new THREE.PointLight(0x1b8eff, 2.4, 0, 2);
    rimLight.position.set(-14000, 9000, 16000);
    rimLight.castShadow = false;
    scene.add(rimLight);

    const accentLight = new THREE.PointLight(0x22ffd2, 2.0, 0, 2);
    accentLight.position.set(15000, 7000, -13000);
    accentLight.castShadow = false;
    scene.add(accentLight);

    const floorGlow = new THREE.PointLight(0x4f7bff, 1.35, 0, 2);
    floorGlow.position.set(0, 2200, 0);
    floorGlow.castShadow = false;
    scene.add(floorGlow);
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
        opacity: 0.92,
        transparent: true,
        emissiveIntensity: 1.15,
        metalness: 0.7,
        roughness: 0.22,
        clearcoat: 0.85,
        clearcoatRoughness: 0.12,
        envMapIntensity: 1.4,
        transmission: 0.08,
        thickness: 1.45
    });
    pipeline.engine.scene.add(vocabBottom.group);

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
        opacity: 0.9,
        transparent: true,
        emissiveIntensity: 1.05,
        metalness: 0.68,
        roughness: 0.24,
        clearcoat: 0.82,
        clearcoatRoughness: 0.14,
        envMapIntensity: 1.35,
        transmission: 0.06,
        thickness: 1.35
    });
    pipeline.engine.scene.add(posBottom.group);

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
        const lnHue = new THREE.Color(INACTIVE_COMPONENT_COLOR).lerp(new THREE.Color(0x26ffc5), 0.35);
        lnTop.setColor(lnHue);
        lnTop.setMaterialProperties({
            opacity: 0.95,
            transparent: true,
            emissiveIntensity: 0.55,
            metalness: 0.6,
            roughness: 0.28,
            clearcoat: 0.78,
            clearcoatRoughness: 0.16,
            envMapIntensity: 1.25,
            transmission: 0.04,
            thickness: 1.2
        });
        pipeline.engine.scene.add(lnTop.group);

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
        vocabTop.setColor(new THREE.Color(0x060c33));
        vocabTop.setMaterialProperties({
            opacity: 0.95,
            transparent: true,
            emissiveIntensity: 0.65,
            metalness: 0.74,
            roughness: 0.2,
            clearcoat: 0.88,
            clearcoatRoughness: 0.1,
            envMapIntensity: 1.55,
            transmission: 0.07,
            thickness: 1.5
        });
        appState.vocabTopRef = vocabTop;
        pipeline.engine.scene.add(vocabTop.group);
    }
} catch (_) { /* optional – embedding visuals are non-critical */ }

// Initialise UI modules
initIntroAnimation(pipeline, gptCanvas);
initStatusOverlay(pipeline, NUM_LAYERS);
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
        const mat = createSciFiPanelMaterial({
            baseColor: 0xffffff,
            emissiveColor: 0x7acbff,
            opacity: 0,
            mapRepeat: new THREE.Vector2(1.6, 0.6),
            scanStrength: 0.55,
            scanSpeed: 1.6,
            fresnelPower: 2.8,
            fresnelIntensity: 0.75
        });
        mat.transparent = true;
        mat.emissiveIntensity = 0.9;
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
