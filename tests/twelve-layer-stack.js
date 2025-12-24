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
    cameraTarget: camTarget,
    enableBloom: true
});

// Show GPT canvas immediately
gptCanvas.style.display = 'block';
applySciFiEnvironment(pipeline);
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
        opacity: 0.92,
        transparent: true,
        emissive: new THREE.Color(0x0bcfff),
        emissiveIntensity: 0.45,
        metalness: 0.2,
        roughness: 0.38
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
        opacity: 0.9,
        transparent: true,
        emissive: new THREE.Color(0x2bffb0),
        emissiveIntensity: 0.42,
        metalness: 0.18,
        roughness: 0.4
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
            opacity: 0.95,
            transparent: true,
            emissive: new THREE.Color(0x0ebaff),
            emissiveIntensity: 0.2,
            metalness: 0.16,
            roughness: 0.38
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
        vocabTop.setColor(new THREE.Color(0x041c33));
        vocabTop.setMaterialProperties({
            opacity: 0.9,
            transparent: true,
            emissive: new THREE.Color(0x12d4ff),
            emissiveIntensity: 0.26,
            metalness: 0.22,
            roughness: 0.35
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
            color: 0x7ffcff,
            transparent: true,
            opacity: 0,
            emissive: new THREE.Color(0x118aff),
            emissiveIntensity: 2.4,
            metalness: 0.35,
            roughness: 0.18
        });
        const mesh = new THREE.Mesh(geo, mat);
        mesh.position.x = xOffset;
        mesh.castShadow = false;
        mesh.receiveShadow = false;
        const emissivePhase = Math.random() * Math.PI * 2;
        mesh.userData.emissivePhase = emissivePhase;
        mesh.onBeforeRender = () => {
            const t = performance.now() * 0.0022;
            mesh.material.emissiveIntensity = 1.6 + Math.sin(t + emissivePhase) * 0.7;
            mesh.rotation.y = Math.sin(t * 0.08 + emissivePhase) * 0.05;
        };
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

function applySciFiEnvironment(pipeline) {
    if (!pipeline || !pipeline.engine) return;
    const engine = pipeline.engine;
    const scene = engine.scene;
    if (!scene) return;

    scene.userData = scene.userData || {};
    if (scene.userData.__sciFiDecorated) return;
    scene.userData.__sciFiDecorated = true;

    try {
        if (engine.renderer) {
            engine.renderer.setClearColor(0x020812, 1);
            if (typeof engine.renderer.toneMappingExposure === 'number') {
                engine.renderer.toneMappingExposure = Math.max(engine.renderer.toneMappingExposure, 1.0) * 1.18;
            }
        }
    } catch (_) { /* optional renderer tuning */ }

    const nebulaTexture = createNebulaTexture();
    if (nebulaTexture) {
        nebulaTexture.colorSpace = THREE.SRGBColorSpace;
        nebulaTexture.anisotropy = 4;
        scene.background = nebulaTexture;
    }

    scene.fog = new THREE.FogExp2(0x020516, 0.00008);

    retuneBaseLights(scene);
    addAccentLights(scene);
    addStarfield(scene);
    addNeonGround(scene);
    addEnergyColumn(scene);
    addScanningPlane(scene);
}

function createNebulaTexture() {
    const size = 1024;
    const canvas = document.createElement('canvas');
    canvas.width = canvas.height = size;
    const ctx = canvas.getContext('2d');
    if (!ctx) return null;

    const gradient = ctx.createRadialGradient(size * 0.5, size * 0.45, size * 0.15, size * 0.5, size * 0.5, size * 0.75);
    gradient.addColorStop(0, '#0b2a4f');
    gradient.addColorStop(0.4, '#050d1f');
    gradient.addColorStop(1, '#000107');
    ctx.fillStyle = gradient;
    ctx.fillRect(0, 0, size, size);

    ctx.globalAlpha = 0.55;
    for (let i = 0; i < 220; i++) {
        const x = Math.random() * size;
        const y = Math.random() * size;
        const radius = Math.random() * 2.4 + 0.6;
        const glow = ctx.createRadialGradient(x, y, 0, x, y, radius);
        if (Math.random() > 0.5) {
            glow.addColorStop(0, 'rgba(0, 255, 255, 1)');
            glow.addColorStop(0.6, 'rgba(0, 255, 255, 0.35)');
            glow.addColorStop(1, 'rgba(0, 255, 255, 0)');
        } else {
            glow.addColorStop(0, 'rgba(173, 60, 255, 1)');
            glow.addColorStop(0.6, 'rgba(173, 60, 255, 0.35)');
            glow.addColorStop(1, 'rgba(173, 60, 255, 0)');
        }
        ctx.fillStyle = glow;
        ctx.beginPath();
        ctx.arc(x, y, radius, 0, Math.PI * 2);
        ctx.fill();
    }
    ctx.globalAlpha = 1;

    return new THREE.CanvasTexture(canvas);
}

function retuneBaseLights(scene) {
    const ambient = scene.children.find(obj => obj && obj.isAmbientLight);
    if (ambient) {
        ambient.color.set(0x1b3f88);
        ambient.intensity = 0.85;
    }
    const directional = scene.children.find(obj => obj && obj.isDirectionalLight);
    if (directional) {
        directional.color.set(0x6bc9ff);
        directional.intensity = 1.35;
        directional.position.set(-16000, 22000, 14000);
    }
}

function addAccentLights(scene) {
    if (!scene.getObjectByName('NeonRimLight')) {
        const rimLight = new THREE.PointLight(0x30faff, 2.6, 60000, 2);
        rimLight.name = 'NeonRimLight';
        rimLight.position.set(16000, 14000, -16000);
        scene.add(rimLight);
    }
    if (!scene.getObjectByName('MagentaPulseLight')) {
        const magentaLight = new THREE.PointLight(0xff35d8, 2.1, 52000, 2);
        magentaLight.name = 'MagentaPulseLight';
        magentaLight.position.set(-22000, 8000, 18000);
        scene.add(magentaLight);
    }
    if (!scene.getObjectByName('CyanSpotLight')) {
        const spot = new THREE.SpotLight(0x53f1ff, 1.15, 70000, Math.PI / 3.2, 0.6, 1.2);
        spot.name = 'CyanSpotLight';
        spot.position.set(0, 26000, 12000);
        spot.target.position.set(0, 6000, 0);
        spot.target.name = 'CyanSpotLightTarget';
        scene.add(spot);
        scene.add(spot.target);
    }
}

function addStarfield(scene) {
    if (scene.getObjectByName('SciFiStarfield')) return;

    const starCount = 1500;
    const radius = 52000;
    const positions = new Float32Array(starCount * 3);
    const colors = new Float32Array(starCount * 3);
    const colorA = new THREE.Color(0x18b7ff);
    const colorB = new THREE.Color(0xad3bff);
    const tempColor = new THREE.Color();

    for (let i = 0; i < starCount; i++) {
        const theta = Math.acos(2 * Math.random() - 1);
        const phi = Math.random() * Math.PI * 2;
        const r = radius * (0.65 + Math.random() * 0.35);
        const sinTheta = Math.sin(theta);
        positions[i * 3] = r * sinTheta * Math.cos(phi);
        positions[i * 3 + 1] = (Math.random() - 0.3) * radius * 0.4;
        positions[i * 3 + 2] = r * sinTheta * Math.sin(phi);

        tempColor.copy(colorA).lerp(colorB, Math.random());
        colors[i * 3] = tempColor.r;
        colors[i * 3 + 1] = tempColor.g;
        colors[i * 3 + 2] = tempColor.b;
    }

    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));

    const material = new THREE.PointsMaterial({
        size: 420,
        transparent: true,
        opacity: 0.85,
        depthWrite: false,
        vertexColors: true,
        blending: THREE.AdditiveBlending,
        sizeAttenuation: true
    });

    const starfield = new THREE.Points(geometry, material);
    starfield.name = 'SciFiStarfield';
    starfield.renderOrder = -10;
    starfield.onBeforeRender = () => {
        starfield.rotation.y += 0.00035;
    };
    scene.add(starfield);
}

function addNeonGround(scene) {
    if (!scene.getObjectByName('NeonGroundGrid')) {
        const grid = new THREE.GridHelper(36000, 72, 0x00d2ff, 0x003866);
        grid.name = 'NeonGroundGrid';
        grid.position.y = CAPTION_TEXT_Y_POS - 360;
        grid.rotation.y = Math.PI / 4;
        if (Array.isArray(grid.material)) {
            grid.material.forEach(mat => {
                mat.transparent = true;
                mat.opacity = 0.24;
                mat.depthWrite = false;
                if (mat.color) mat.color.setHex(0x00baff);
            });
        } else {
            grid.material.transparent = true;
            grid.material.opacity = 0.24;
            grid.material.depthWrite = false;
            if (grid.material.color) grid.material.color.setHex(0x00baff);
        }
        scene.add(grid);
    }

    if (!scene.getObjectByName('GroundEnergyRing')) {
        const ring = new THREE.Mesh(
            new THREE.TorusGeometry(6500, 160, 48, 180),
            new THREE.MeshBasicMaterial({
                color: 0x0ff6ff,
                transparent: true,
                opacity: 0.22,
                blending: THREE.AdditiveBlending,
                depthWrite: false
            })
        );
        ring.name = 'GroundEnergyRing';
        ring.rotation.x = Math.PI / 2;
        ring.position.y = CAPTION_TEXT_Y_POS + 420;
        ring.onBeforeRender = () => {
            const t = performance.now() * 0.0012;
            const pulse = 1 + Math.sin(t) * 0.04;
            ring.scale.set(pulse, pulse, pulse);
            ring.material.opacity = 0.18 + (Math.sin(t * 2.0) + 1) * 0.1;
        };
        scene.add(ring);
    }

    if (!scene.getObjectByName('GroundGlowPlate')) {
        const plate = new THREE.Mesh(
            new THREE.CircleGeometry(15000, 64),
            new THREE.MeshBasicMaterial({
                color: 0x02c7ff,
                transparent: true,
                opacity: 0.12,
                side: THREE.DoubleSide,
                blending: THREE.AdditiveBlending,
                depthWrite: false
            })
        );
        plate.name = 'GroundGlowPlate';
        plate.rotation.x = -Math.PI / 2;
        plate.position.y = CAPTION_TEXT_Y_POS + 220;
        plate.onBeforeRender = () => {
            const t = performance.now() * 0.0015;
            plate.material.opacity = 0.1 + (Math.sin(t * 1.6) + 1) * 0.08;
        };
        scene.add(plate);
    }
}

function addEnergyColumn(scene) {
    if (!scene.getObjectByName('TowerEnergyColumn')) {
        const column = new THREE.Mesh(
            new THREE.CylinderGeometry(6800, 6800, 28000, 64, 1, true),
            new THREE.MeshBasicMaterial({
                color: 0x0efcff,
                transparent: true,
                opacity: 0.04,
                side: THREE.DoubleSide,
                depthWrite: false,
                blending: THREE.AdditiveBlending
            })
        );
        column.name = 'TowerEnergyColumn';
        column.position.y = 7200;
        column.onBeforeRender = () => {
            const t = performance.now() * 0.00065;
            column.rotation.y = t;
            column.material.opacity = 0.03 + (Math.sin(t * 6.0) + 1) * 0.02;
        };
        scene.add(column);
    }

    if (!scene.getObjectByName('TowerCorePulse')) {
        const core = new THREE.Mesh(
            new THREE.SphereGeometry(1400, 32, 32),
            new THREE.MeshBasicMaterial({
                color: 0x12f6ff,
                transparent: true,
                opacity: 0.22,
                blending: THREE.AdditiveBlending,
                depthWrite: false
            })
        );
        core.name = 'TowerCorePulse';
        core.position.y = 11500;
        core.onBeforeRender = () => {
            const t = performance.now() * 0.0012;
            const scale = 1 + Math.sin(t * 2.4) * 0.08;
            core.scale.set(scale, scale, scale);
            core.material.opacity = 0.16 + (Math.sin(t * 3.1) + 1) * 0.12;
        };
        scene.add(core);
    }
}

function addScanningPlane(scene) {
    if (scene.getObjectByName('TowerScanner')) return;

    const scanner = new THREE.Mesh(
        new THREE.PlaneGeometry(20000, 20000, 1, 1),
        new THREE.MeshBasicMaterial({
            color: 0x1ffbff,
            transparent: true,
            opacity: 0.08,
            side: THREE.DoubleSide,
            depthWrite: false,
            blending: THREE.AdditiveBlending
        })
    );
    scanner.name = 'TowerScanner';
    scanner.rotation.x = Math.PI / 2;
    const baseY = CAPTION_TEXT_Y_POS + 1800;
    scanner.onBeforeRender = () => {
        const t = performance.now() * 0.00085;
        scanner.position.y = baseY + Math.sin(t) * 3200;
        scanner.material.opacity = 0.05 + (Math.sin(t * 3.4) + 1) * 0.05;
    };
    scene.add(scanner);
}
