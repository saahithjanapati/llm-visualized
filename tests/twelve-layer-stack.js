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

const STARFIELD_LAYERS = [
    { count: 1400, radius: 90000, size: 240, color: 0x4acfff, opacity: 0.85 },
    { count: 900, radius: 65000, size: 320, color: 0xff63e6, opacity: 0.75 }
];

function createSpaceGradientTexture() {
    try {
        const size = 1024;
        const canvas = document.createElement('canvas');
        canvas.width = size;
        canvas.height = size;
        const ctx = canvas.getContext('2d');
        const radial = ctx.createRadialGradient(size * 0.55, size * 0.35, size * 0.15, size * 0.5, size * 0.65, size * 0.75);
        radial.addColorStop(0, '#1a3a7d');
        radial.addColorStop(0.28, '#0c1a3c');
        radial.addColorStop(0.55, '#050b23');
        radial.addColorStop(1, '#010208');
        ctx.fillStyle = radial;
        ctx.fillRect(0, 0, size, size);

        const tex = new THREE.CanvasTexture(canvas);
        tex.colorSpace = THREE.SRGBColorSpace;
        tex.magFilter = THREE.LinearFilter;
        tex.minFilter = THREE.LinearMipMapLinearFilter;
        tex.generateMipmaps = true;
        return tex;
    } catch (err) {
        console.warn('Failed to build gradient texture', err);
        return null;
    }
}

function createStarfieldLayer({ count, radius, size, color, opacity }) {
    const geometry = new THREE.BufferGeometry();
    const positions = new Float32Array(count * 3);
    for (let i = 0; i < count; i++) {
        const phi = Math.acos(THREE.MathUtils.randFloatSpread(2));
        const theta = THREE.MathUtils.randFloatSpread(2) * Math.PI;
        const r = radius * (0.65 + Math.random() * 0.35);
        const x = r * Math.sin(phi) * Math.cos(theta);
        const y = r * Math.cos(phi);
        const z = r * Math.sin(phi) * Math.sin(theta);
        positions[i * 3] = x;
        positions[i * 3 + 1] = y;
        positions[i * 3 + 2] = z;
    }
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    const material = new THREE.PointsMaterial({
        color,
        size,
        sizeAttenuation: true,
        transparent: true,
        opacity,
        depthWrite: false,
        blending: THREE.AdditiveBlending
    });
    const points = new THREE.Points(geometry, material);
    points.userData.isStarfield = true;
    points.userData.baseOpacity = opacity;
    points.userData.twinklePhase = Math.random() * Math.PI * 2;
    return points;
}

function createHolographicGrid(size, divisions, majorColor, minorColor) {
    const grid = new THREE.GridHelper(size, divisions, majorColor, minorColor);
    const materials = Array.isArray(grid.material) ? grid.material : [grid.material];
    materials.forEach((mat) => {
        mat.transparent = true;
        mat.opacity = 0.32;
        mat.depthWrite = false;
    });
    grid.renderOrder = -2;
    return grid;
}

function createNebulaPlane(width, height, color, opacity, offset) {
    const geo = new THREE.PlaneGeometry(width, height, 1, 1);
    const mat = new THREE.MeshBasicMaterial({
        color,
        transparent: true,
        opacity,
        depthWrite: false,
        blending: THREE.AdditiveBlending
    });
    const mesh = new THREE.Mesh(geo, mat);
    mesh.position.copy(offset);
    mesh.userData.baseOpacity = opacity;
    return mesh;
}

function addEnergyColumn(scene, bounds) {
    if (!scene || !bounds || bounds.isEmpty?.()) return null;
    const height = bounds.max.y - bounds.min.y + 6000;
    if (!Number.isFinite(height) || height <= 0) return null;
    const center = bounds.getCenter(new THREE.Vector3());
    const columnGroup = new THREE.Group();
    const outer = new THREE.Mesh(
        new THREE.CylinderGeometry(2600, 2200, height, 64, 1, true),
        new THREE.MeshBasicMaterial({
            color: 0x3df6ff,
            transparent: true,
            opacity: 0.08,
            side: THREE.DoubleSide,
            depthWrite: false,
            blending: THREE.AdditiveBlending
        })
    );
    const inner = new THREE.Mesh(
        new THREE.CylinderGeometry(1200, 1200, height * 1.05, 48, 1, true),
        new THREE.MeshBasicMaterial({
            color: 0x7b5bff,
            transparent: true,
            opacity: 0.12,
            side: THREE.DoubleSide,
            depthWrite: false,
            blending: THREE.AdditiveBlending
        })
    );
    const core = new THREE.Mesh(
        new THREE.CylinderGeometry(360, 360, height * 1.08, 24, 1, true),
        new THREE.MeshBasicMaterial({
            color: 0xffffff,
            transparent: true,
            opacity: 0.28,
            side: THREE.DoubleSide,
            depthWrite: false,
            blending: THREE.AdditiveBlending
        })
    );
    outer.rotation.y = Math.PI / 8;
    inner.rotation.y = Math.PI / 4;
    columnGroup.add(outer, inner, core);
    columnGroup.position.set(0, center.y + 1200, 0);
    scene.add(columnGroup);
    return columnGroup;
}

function createPulseRing(scene, radius, y, color, duration) {
    const geo = new THREE.TorusGeometry(radius, 260, 24, 160);
    const mat = new THREE.MeshBasicMaterial({
        color,
        transparent: true,
        opacity: 0.14,
        blending: THREE.AdditiveBlending,
        depthWrite: false
    });
    const ring = new THREE.Mesh(geo, mat);
    ring.rotation.x = Math.PI / 2;
    ring.position.y = y;
    ring.renderOrder = -1;
    scene.add(ring);
    const pulse = { scale: 0.9 };
    new TWEEN.Tween(pulse)
        .to({ scale: 1.45 }, duration)
        .yoyo(true)
        .repeat(Infinity)
        .easing(TWEEN.Easing.Sinusoidal.InOut)
        .onUpdate(() => {
            ring.scale.setScalar(pulse.scale);
            ring.material.opacity = 0.08 + (1.45 - pulse.scale) * 0.1;
        })
        .start();
    return ring;
}

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
    enableBloom: true,
    cameraFar: 260000
});

// Show GPT canvas immediately
gptCanvas.style.display = 'block';
try {
    const eng = pipeline.engine;
    const { scene, renderer } = eng;
    if (renderer) {
        renderer.shadowMap.enabled = false;
        if ('outputColorSpace' in renderer && renderer.outputColorSpace !== THREE.SRGBColorSpace) {
            renderer.outputColorSpace = THREE.SRGBColorSpace;
        }
        renderer.toneMapping = THREE.ACESFilmicToneMapping;
        renderer.toneMappingExposure = 1.35;
        renderer.setClearColor(0x03040d, 1);
    }

    const gradientTexture = createSpaceGradientTexture();
    if (gradientTexture) {
        scene.background = gradientTexture;
        scene.environment = gradientTexture;
    } else {
        scene.background = new THREE.Color(0x03040d);
    }

    scene.fog = new THREE.FogExp2(0x02020a, 0.00006);

    scene.traverse((obj) => {
        if (obj.isMesh) {
            obj.castShadow = false;
            obj.receiveShadow = false;
        }
        if (obj.isLight) {
            obj.castShadow = false;
            if (obj.isDirectionalLight) {
                obj.color.setHex(0x7fdfff);
                obj.intensity = 1.35;
                obj.position.set(16000, 26000, 14000);
            }
            if (obj.isAmbientLight) {
                obj.color.setHex(0x0d1635);
                obj.intensity = 0.6;
            }
        }
    });

    const nebulaBack = createNebulaPlane(180000, 120000, 0x1638ff, 0.16, new THREE.Vector3(-32000, 18000, -140000));
    nebulaBack.rotation.y = Math.PI * 0.08;
    nebulaBack.rotation.x = Math.PI * -0.04;
    scene.add(nebulaBack);

    const nebulaRight = createNebulaPlane(150000, 100000, 0x4ff0ff, 0.12, new THREE.Vector3(160000, 9000, -60000));
    nebulaRight.rotation.y = -Math.PI / 2.1;
    scene.add(nebulaRight);

    const nebulaPlanes = [nebulaBack, nebulaRight];

    const starfieldMeshes = STARFIELD_LAYERS.map(createStarfieldLayer).filter(Boolean);
    starfieldMeshes.forEach(layer => scene.add(layer));

    const gridPrimary = createHolographicGrid(220000, 90, 0x2cf9ff, 0x3a6cff);
    gridPrimary.position.y = -3800;
    scene.add(gridPrimary);

    const gridSecondary = createHolographicGrid(160000, 45, 0x6b4fff, 0x1df5ff);
    gridSecondary.position.y = -2500;
    const secondaryMaterials = Array.isArray(gridSecondary.material)
        ? gridSecondary.material
        : [gridSecondary.material];
    secondaryMaterials.forEach((mat) => { mat.opacity = 0.18; });
    scene.add(gridSecondary);

    const towerBounds = new THREE.Box3();
    const tmpBox = new THREE.Box3();
    pipeline._layers.forEach((layer) => {
        if (layer?.root) {
            tmpBox.setFromObject(layer.root);
            towerBounds.union(tmpBox);
        }
    });

    const energyColumn = addEnergyColumn(scene, towerBounds);
    if (energyColumn) {
        const baseY = towerBounds.min.y - 1200;
        createPulseRing(scene, 9000, baseY, 0x54f5ff, 4200);
        createPulseRing(scene, 14000, towerBounds.max.y + 2200, 0xff6be9, 5200);
    }

    const rimLight = new THREE.PointLight(0x5ce7ff, 220, 220000, 1.8);
    rimLight.position.set(28000, 24000, 20000);
    scene.add(rimLight);

    const magentaLight = new THREE.PointLight(0xff5be6, 160, 180000, 2.2);
    magentaLight.position.set(-26000, 18000, -22000);
    scene.add(magentaLight);

    if (!towerBounds.isEmpty()) {
        const center = towerBounds.getCenter(new THREE.Vector3());
        const apex = towerBounds.max.y + 16000;
        const spotlight = new THREE.SpotLight(0x72fffd, 2.4, 260000, Math.PI / 3.4, 0.45, 1.25);
        spotlight.position.set(0, apex, 16000);
        spotlight.target.position.copy(center);
        scene.add(spotlight, spotlight.target);
    }

    const dynamicUpdaters = [];
    scene.userData.dynamicUpdaters = dynamicUpdaters;

    scene.onBeforeRender = () => {
        const now = (typeof performance !== 'undefined' && typeof performance.now === 'function')
            ? performance.now()
            : Date.now();
        starfieldMeshes.forEach((mesh, idx) => {
            mesh.rotation.y += 0.00002 * (idx + 1);
            mesh.rotation.x += 0.00001 * (idx + 1);
            const base = mesh.userData?.baseOpacity ?? 0.8;
            const material = mesh.material;
            if (material) {
                const flicker = Math.sin(now * 0.0003 * (idx + 1) + (mesh.userData?.twinklePhase ?? 0)) * 0.05;
                material.opacity = Math.max(0, base + flicker);
            }
        });
        nebulaPlanes.forEach((plane, idx) => {
            if (!plane || !plane.material) return;
            const baseOpacity = plane.userData?.baseOpacity ?? 0.1;
            const pulse = Math.sin(now * 0.00022 + idx) * 0.025;
            plane.material.opacity = Math.max(0, baseOpacity + pulse);
            plane.rotation.z = Math.sin(now * 0.00004 + idx * 0.4) * 0.05;
        });
        dynamicUpdaters.forEach((fn) => {
            try { fn(now); } catch (updateErr) { /* ignore individual updater failures */ }
        });
    };
} catch (err) {
    console.warn('Failed to initialise sci-fi environment', err);
}

// Embedding matrices (static visuals)
try {
    const neonVocabColor = new THREE.Color(0x31f0ff);
    const neonPosColor = new THREE.Color(0xff68f4);
    const vocabEmissive = new THREE.Color(0x0c4a6d);
    const posEmissive = new THREE.Color(0x521c5d);
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
    vocabBottom.setColor(neonVocabColor);
    vocabBottom.setMaterialProperties({
        opacity: 0.92,
        transparent: true,
        emissive: vocabEmissive,
        emissiveIntensity: 0.9,
        metalness: 0.65,
        roughness: 0.22
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
    posBottom.setColor(neonPosColor);
    posBottom.setMaterialProperties({
        opacity: 0.9,
        transparent: true,
        emissive: posEmissive,
        emissiveIntensity: 0.85,
        metalness: 0.7,
        roughness: 0.2
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
        lnTop.setColor(new THREE.Color(INACTIVE_COMPONENT_COLOR).offsetHSL(0.02, 0.15, 0.05));
        lnTop.setMaterialProperties({
            opacity: 0.94,
            transparent: true,
            emissive: new THREE.Color(0x3046ff),
            emissiveIntensity: 0.55,
            metalness: 0.85,
            roughness: 0.16
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
        vocabTop.setColor(new THREE.Color(0x09121f));
        vocabTop.setMaterialProperties({
            opacity: 0.9,
            transparent: true,
            emissive: new THREE.Color(0x52fff7),
            emissiveIntensity: 1.25,
            metalness: 0.78,
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
            color: 0x8cfff7,
            emissive: new THREE.Color(0x1b314f),
            emissiveIntensity: 0.25,
            metalness: 0.6,
            roughness: 0.28,
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
    const baseCaptionY = charGroup.position.y;
    const updaters = pipeline?.engine?.scene?.userData?.dynamicUpdaters;
    if (Array.isArray(updaters)) {
        updaters.push((now) => {
            const t = now * 0.00018;
            charGroup.position.y = baseCaptionY + Math.sin(t * 1.6) * 140;
            charGroup.rotation.y = Math.sin(t * 0.9) * 0.12;
            charGroup.rotation.x = Math.cos(t * 0.7) * 0.04;
        });
    }
    const revealChar = (index) => {
        if (index >= charMeshes.length) return;
        const mesh = charMeshes[index];
        if (mesh) {
            const tweenState = { opacity: 0, glow: mesh.material.emissiveIntensity };
            new TWEEN.Tween(tweenState)
                .to({ opacity: 1, glow: 1.2 }, TYPE_DELAY_CAP * 2)
                .easing(TWEEN.Easing.Quadratic.Out)
                .onUpdate(() => {
                    mesh.material.opacity = tweenState.opacity;
                    mesh.material.emissiveIntensity = tweenState.glow;
                })
                .start();
        }
        setTimeout(() => revealChar(index + 1), TYPE_DELAY_CAP);
    };
    setTimeout(() => revealChar(0), 500);
    const gif = document.getElementById('loadingGif');
    if (gif) gif.style.display = 'none';
});
