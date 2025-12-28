import * as THREE from 'three';
import { MHSAAnimation } from '../src/animations/MHSAAnimation.js';
import { loadPrecomputedGeometries } from '../src/utils/precomputedGeometryLoader.js';
import { LayerPipeline } from '../src/engine/LayerPipeline.js';
import {
    setPlaybackSpeed,
    setNumVectorLanes,
    EMBEDDING_MATRIX_PARAMS_VOCAB,
    EMBEDDING_MATRIX_PARAMS_POSITION,
    LN_PARAMS,
    NUM_VECTOR_LANES,
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
import {
    MHA_FINAL_Q_COLOR,
    MHA_FINAL_K_COLOR,
    setAnimationLaneCount
} from '../src/animations/LayerAnimationConstants.js';
import { appState } from '../src/state/appState.js';
import { initIntroAnimation } from '../src/ui/introAnimation.js';
import { initStatusOverlay } from '../src/ui/statusOverlay.js';
import { initSettingsModal } from '../src/ui/settingsModal.js';
import { initPauseButton } from '../src/ui/pauseButton.js';
import { initConveyorSkipButton } from '../src/ui/conveyorSkipButton.js';
import { initSelectionPanel } from '../src/ui/selectionPanel.js';
import { initPerfOverlay } from '../src/ui/perfOverlay.js';
import { FontLoader } from 'three/examples/jsm/loaders/FontLoader.js';
import { TextGeometry } from 'three/examples/jsm/geometries/TextGeometry.js';
import { CaptureActivationSource } from '../src/data/CaptureActivationSource.js';
import { precomputeActivationCaches } from '../src/utils/activationPrecompute.js';

const NUM_LAYERS = 12;
const PROMPT_TOKENS = ['Can', '\u0120machines', '\u0120think', '?', '\u0120'];
const POSITION_TOKENS = ['1', '2', '3', '4', '5'];
const TOKEN_CHIP_STYLE = {
    padding: 140,
    minWidth: 440,
    minHeight: 150,
    height: 170,
    cornerRadius: 24,
    depth: 12,
    textSize: 90,
    textDepth: 0, // unused for flat text
    textOffset: 0.6,
    riseDistance: 220,
    riseDelay: 120,
    riseDuration: 1200,
    vocabSlowdown: 1.6,
    positionSlowdown: 1.0,
    inset: 0,
    zOffset: 0,
    scale: 2.6,
    staticGap: 200,
    staticZOffset: 0,
    cameraHoldMs: 800,
    cameraReturnMs: 0
};
const POSITION_CHIP_STYLE = {
    ...TOKEN_CHIP_STYLE,
    padding: 80,
    minWidth: 260,
    minHeight: 120,
    height: 130,
    textSize: 70,
    scale: 2.0
};

function formatTokenLabel(token) {
    return token.replace(/^\u0120/, ' ');
}

function buildRoundedRectShape(width, height, radius) {
    const clampedRadius = Math.max(0, Math.min(radius, Math.min(width, height) / 2 - 1));
    const halfW = width / 2;
    const halfH = height / 2;
    const shape = new THREE.Shape();
    shape.moveTo(-halfW + clampedRadius, -halfH);
    shape.lineTo(halfW - clampedRadius, -halfH);
    shape.quadraticCurveTo(halfW, -halfH, halfW, -halfH + clampedRadius);
    shape.lineTo(halfW, halfH - clampedRadius);
    shape.quadraticCurveTo(halfW, halfH, halfW - clampedRadius, halfH);
    shape.lineTo(-halfW + clampedRadius, halfH);
    shape.quadraticCurveTo(-halfW, halfH, -halfW, halfH - clampedRadius);
    shape.lineTo(-halfW, -halfH + clampedRadius);
    shape.quadraticCurveTo(-halfW, -halfH, -halfW + clampedRadius, -halfH);
    shape.closePath();
    return shape;
}

function createTokenChip(label, font, style) {
    let textGeo = null;
    let bounds = null;
    if (font && label.trim().length) {
        const shapes = font.generateShapes(label, style.textSize, 2);
        textGeo = new THREE.ShapeGeometry(shapes);
        textGeo.computeBoundingBox();
        textGeo.computeVertexNormals();
        bounds = textGeo.boundingBox;
    }
    let textWidth = 0;
    let textHeight = 0;
    if (bounds && Number.isFinite(bounds.max.x) && Number.isFinite(bounds.min.x)) {
        textWidth = Math.max(0, bounds.max.x - bounds.min.x);
        textHeight = Math.max(0, bounds.max.y - bounds.min.y);
    }

    const chipWidth = Math.max(style.minWidth, textWidth + style.padding);
    const chipHeight = typeof style.height === 'number' && Number.isFinite(style.height)
        ? style.height
        : Math.max(style.minHeight, textHeight + style.padding);
    const chipRadius = Math.min(style.cornerRadius, Math.min(chipWidth, chipHeight) / 2 - 1);
    const chipShape = buildRoundedRectShape(chipWidth, chipHeight, chipRadius);
    const chipGeo = new THREE.ExtrudeGeometry(chipShape, {
        depth: style.depth,
        bevelEnabled: false
    });
    chipGeo.translate(0, 0, -style.depth / 2);
    chipGeo.computeVertexNormals();

    const chipMat = new THREE.MeshStandardMaterial({
        color: 0xf2e8d5,
        roughness: 0.35,
        metalness: 0.15,
        side: THREE.DoubleSide
    });
    const chipMesh = new THREE.Mesh(chipGeo, chipMat);

    const group = new THREE.Group();
    group.add(chipMesh);

    const capMat = chipMat.clone();
    capMat.polygonOffset = true;
    capMat.polygonOffsetFactor = -1;
    capMat.polygonOffsetUnits = -2;
    const capGeo = new THREE.ShapeGeometry(chipShape);
    capGeo.computeVertexNormals();
    const capOffset = 0.05;
    const frontCap = new THREE.Mesh(capGeo, capMat);
    frontCap.position.z = style.depth / 2 + capOffset;
    const backCap = new THREE.Mesh(capGeo, capMat);
    backCap.position.z = -style.depth / 2 - capOffset;
    backCap.rotation.y = Math.PI;
    group.add(frontCap, backCap);

    if (textGeo && textWidth > 0 && textHeight > 0) {
        const textMat = new THREE.MeshBasicMaterial({
            color: 0xffffff,
            side: THREE.DoubleSide,
            depthWrite: true,
            depthTest: true,
            polygonOffset: true,
            polygonOffsetFactor: -0.5,
            polygonOffsetUnits: -0.5
        });
        const textMesh = new THREE.Mesh(textGeo, textMat);
        const centerX = (bounds.min.x + bounds.max.x) / 2;
        const centerY = (bounds.min.y + bounds.max.y) / 2;
        const textFrontOffset = style.depth / 2 + style.textOffset;
        textMesh.position.set(-centerX, -centerY, textFrontOffset);
        group.add(textMesh);
    } else if (textGeo) {
        textGeo.dispose();
    }

    if (typeof style.scale === 'number' && Number.isFinite(style.scale) && style.scale > 0) {
        group.scale.setScalar(style.scale);
    }
    const scaleFactor = (typeof style.scale === 'number' && Number.isFinite(style.scale) && style.scale > 0)
        ? style.scale
        : 1;
    group.userData.size = { width: chipWidth * scaleFactor, height: chipHeight * scaleFactor };
    return group;
}

function stageChipCamera(pipeline, startPos, startTarget, endPos, endTarget, holdMs, returnMs) {
    const engine = pipeline?.engine;
    if (!engine || !engine.camera || !engine.controls) return;
    const hadAutoFollow = typeof pipeline.isAutoCameraFollowEnabled === 'function'
        ? pipeline.isAutoCameraFollowEnabled()
        : false;
    if (hadAutoFollow && typeof pipeline.setAutoCameraFollow === 'function') {
        pipeline.setAutoCameraFollow(false, { immediate: true });
    }

    engine.camera.position.copy(startPos);
    engine.controls.target.copy(startTarget);
    engine.notifyCameraUpdated();
    engine.controls.update();

    const restoreAutoFollow = () => {
        if (hadAutoFollow && typeof pipeline.setAutoCameraFollow === 'function') {
            pipeline.setAutoCameraFollow(true, { immediate: true });
        }
    };

    const shouldReturn = Number.isFinite(returnMs) && returnMs > 0 && endPos && endTarget;
    if (!shouldReturn) {
        if (hadAutoFollow) {
            const delayMs = Math.max(0, holdMs);
            if (delayMs > 0) {
                setTimeout(restoreAutoFollow, delayMs);
            } else {
                restoreAutoFollow();
            }
        }
        return;
    }

    if (typeof TWEEN !== 'undefined') {
        new TWEEN.Tween(engine.camera.position)
            .to({ x: endPos.x, y: endPos.y, z: endPos.z }, returnMs)
            .delay(holdMs)
            .easing(TWEEN.Easing.Quadratic.InOut)
            .onUpdate(() => engine.notifyCameraUpdated())
            .start();

        new TWEEN.Tween(engine.controls.target)
            .to({ x: endTarget.x, y: endTarget.y, z: endTarget.z }, returnMs)
            .delay(holdMs)
            .easing(TWEEN.Easing.Quadratic.InOut)
            .onUpdate(() => {
                engine.notifyCameraUpdated();
                engine.controls.update();
            })
            .onComplete(restoreAutoFollow)
            .start();
    } else {
        setTimeout(() => {
            engine.camera.position.copy(endPos);
            engine.controls.target.copy(endTarget);
            engine.notifyCameraUpdated();
            engine.controls.update();
            restoreAutoFollow();
        }, holdMs);
    }
}

// Optionally load pre-baked geometries; returns instantly if disabled
await loadPrecomputedGeometries('../precomputed_components.glb');

let activationSource = null;
let laneTokenIndices = null;
let laneCount = NUM_VECTOR_LANES;
const MAX_CAPTURE_LANES = 5;
const statusDiv = document.getElementById('statusOverlay');
const setLoadingStatus = (text) => {
    if (statusDiv) statusDiv.textContent = text;
};
try {
    const params = new URLSearchParams(window.location.search);
    const captureFile = params.get('capture') || params.get('file') || 'capture.json';
    const captureUrl = captureFile.startsWith('http')
        ? captureFile
        : `/${captureFile.replace(/^\/+/, '')}`;
    activationSource = await CaptureActivationSource.load(captureUrl);
    const tokensInCapture = typeof activationSource.getTokenCount === 'function'
        ? activationSource.getTokenCount()
        : 0;
    const desiredLanes = Math.max(1, tokensInCapture || laneCount);
    laneCount = Math.min(MAX_CAPTURE_LANES, desiredLanes);
    laneTokenIndices = activationSource.getLaneTokenIndices(laneCount);
} catch (err) {
    console.warn('Capture data unavailable; falling back to random vectors.', err);
}
if (!activationSource) {
    laneCount = Math.max(1, PROMPT_TOKENS.length);
}

setNumVectorLanes(laneCount);
setAnimationLaneCount(laneCount);

if (activationSource && laneTokenIndices) {
    try {
        setLoadingStatus('Preparing activation cache...');
        await precomputeActivationCaches(activationSource, {
            layerCount: NUM_LAYERS,
            laneTokenIndices,
            onProgress: ({ message }) => {
                if (message) {
                    setLoadingStatus(message);
                }
            }
        });
    } catch (err) {
        console.warn('Activation cache precompute failed; continuing without warm cache.', err);
    }
}

const tokenLabelsFromCapture = activationSource && laneTokenIndices
    ? laneTokenIndices.map((idx) => activationSource.getTokenString(idx) || '')
    : PROMPT_TOKENS;
const positionLabelsFromCapture = activationSource && laneTokenIndices
    ? laneTokenIndices.map((idx) => String(idx + 1))
    : POSITION_TOKENS;

// Skip intro typing screen for direct animation entry
appState.skipIntro = true;

// Set default playback speed to fast on load
try { setPlaybackSpeed('fast'); } catch (_) { /* no-op */ }

// GPT-2 tower – initialise immediately
MHSAAnimation.ENABLE_SELF_ATTENTION = true;
const gptCanvas = document.getElementById('gptCanvas');
const camPos    = new THREE.Vector3(0, 11000, 16000);
const camTarget = new THREE.Vector3(0, 9000, 0);
const pipeline = new LayerPipeline(gptCanvas, NUM_LAYERS, {
    cameraPosition: camPos,
    cameraTarget: camTarget,
    activationSource,
    laneCount
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
} catch (_) {}

// Embedding matrices (static visuals)
try {
    const headBlue = new THREE.Color(MHA_FINAL_Q_COLOR);
    const headGreen = new THREE.Color(MHA_FINAL_K_COLOR);
    const residualYBase = LAYER_NORM_1_Y_POS - LN_PARAMS.height / 2 + EMBEDDING_BOTTOM_TOP_ALIGN_OFFSET_FROM_LN1_BOTTOM;
    const bottomVocabCenterY = residualYBase - EMBEDDING_MATRIX_PARAMS_VOCAB.height / 2 + EMBEDDING_BOTTOM_Y_ADJUST;
    const vocabX = 0 + EMBEDDING_BOTTOM_VOCAB_X_OFFSET;
    const vocabBottom = new WeightMatrixVisualization(
        null,
        new THREE.Vector3(vocabX, bottomVocabCenterY, 0),
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
    vocabBottom.setMaterialProperties({ opacity: 1.0, transparent: false, emissiveIntensity: 0.05 });
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
    posBottom.setMaterialProperties({ opacity: 1.0, transparent: false, emissiveIntensity: 0.05 });
    pipeline.engine.scene.add(posBottom.group);
    if (pipeline.engine && typeof pipeline.engine.registerRaycastRoot === 'function') {
        pipeline.engine.registerRaycastRoot(posBottom.group);
    }

    const laneSpacing = LN_PARAMS.depth / (laneCount + 1);
    const laneZs = [];
    for (let i = 0; i < laneCount; i++) {
        laneZs.push(-LN_PARAMS.depth / 2 + laneSpacing * (i + 1));
    }

    const vocabMatrixBottomY = bottomVocabCenterY - EMBEDDING_MATRIX_PARAMS_VOCAB.height / 2;
    const posMatrixBottomY = bottomPosCenterY - EMBEDDING_MATRIX_PARAMS_POSITION.height / 2;
    const chipStartZOffset = TOKEN_CHIP_STYLE.zOffset;
    const chipFontLoader = new FontLoader();
    const registerChip = (chip) => {
        if (pipeline.engine && typeof pipeline.engine.registerRaycastRoot === 'function') {
            pipeline.engine.registerRaycastRoot(chip);
        }
    };
    const spawnTokenChips = (font) => {
        const vocabRiseDuration = TOKEN_CHIP_STYLE.riseDuration * (TOKEN_CHIP_STYLE.vocabSlowdown || 1);
        const posRiseDuration = TOKEN_CHIP_STYLE.riseDuration * (TOKEN_CHIP_STYLE.positionSlowdown || 1);
        const laneSpanMs = TOKEN_CHIP_STYLE.riseDelay * Math.max(0, laneCount - 1);
        const maxRiseDuration = Math.max(vocabRiseDuration, posRiseDuration) + laneSpanMs;
        const cameraStartPos = new THREE.Vector3(
            vocabX + EMBEDDING_MATRIX_PARAMS_VOCAB.width * 0.4,
            bottomVocabCenterY - EMBEDDING_MATRIX_PARAMS_VOCAB.height * 0.8,
            EMBEDDING_MATRIX_PARAMS_VOCAB.depth * 1.8
        );
        const cameraStartTarget = new THREE.Vector3(
            vocabX,
            bottomVocabCenterY + EMBEDDING_MATRIX_PARAMS_VOCAB.height * 0.2,
            0
        );
        const adjustedHoldMs = maxRiseDuration + TOKEN_CHIP_STYLE.cameraHoldMs;
        stageChipCamera(
            pipeline,
            cameraStartPos,
            cameraStartTarget,
            camPos,
            camTarget,
            adjustedHoldMs,
            TOKEN_CHIP_STYLE.cameraReturnMs
        );

        const tokenLabels = tokenLabelsFromCapture.slice(0, laneCount).map(formatTokenLabel);
        tokenLabels.forEach((label, idx) => {
            const chip = createTokenChip(label, font, TOKEN_CHIP_STYLE);
            const chipLabel = `Token: ${label}`;
            chip.userData.label = chipLabel;
            chip.name = chipLabel;
            const chipHeight = chip.userData.size.height;
            const targetY = vocabMatrixBottomY + chipHeight / 2 + TOKEN_CHIP_STYLE.inset;
            const targetZ = laneZs[idx];
            const startZ = targetZ + chipStartZOffset;

            const staticY = vocabMatrixBottomY - chipHeight / 2 - TOKEN_CHIP_STYLE.staticGap;
            const staticZ = targetZ + TOKEN_CHIP_STYLE.staticZOffset;
            const startY = staticY;

            chip.position.set(vocabX, startY, startZ);
            pipeline.engine.scene.add(chip);
            registerChip(chip);

            const staticChip = createTokenChip(label, font, TOKEN_CHIP_STYLE);
            staticChip.userData.label = chipLabel;
            staticChip.name = chipLabel;
            staticChip.position.set(vocabX, staticY, staticZ);
            pipeline.engine.scene.add(staticChip);
            registerChip(staticChip);

            if (typeof TWEEN !== 'undefined') {
                new TWEEN.Tween(chip.position)
                    .to({ y: targetY, z: targetZ }, vocabRiseDuration)
                    .delay(idx * TOKEN_CHIP_STYLE.riseDelay)
                    .easing(TWEEN.Easing.Quadratic.Out)
                    .start();
            } else {
                chip.position.y = targetY;
                chip.position.z = targetZ;
            }
        });
    };
    const spawnPositionChips = (font) => {
        const posRiseDuration = TOKEN_CHIP_STYLE.riseDuration * (TOKEN_CHIP_STYLE.positionSlowdown || 1);
        const style = POSITION_CHIP_STYLE;
        const positionLabels = positionLabelsFromCapture.slice(0, laneCount);
        positionLabels.forEach((label, idx) => {
            const chip = createTokenChip(label, font, style);
            const chipLabel = `Position: ${label}`;
            chip.userData.label = chipLabel;
            chip.name = chipLabel;
            const chipHeight = chip.userData.size.height;
            const targetY = posMatrixBottomY + chipHeight / 2 + style.inset;
            const targetZ = laneZs[idx];
            const startZ = targetZ + chipStartZOffset;

            const staticY = posMatrixBottomY - chipHeight / 2 - style.staticGap;
            const staticZ = targetZ + style.staticZOffset;
            const startY = staticY;

            chip.position.set(posX, startY, startZ);
            pipeline.engine.scene.add(chip);
            registerChip(chip);

            const staticChip = createTokenChip(label, font, style);
            staticChip.userData.label = chipLabel;
            staticChip.name = chipLabel;
            staticChip.position.set(posX, staticY, staticZ);
            pipeline.engine.scene.add(staticChip);
            registerChip(staticChip);

            if (typeof TWEEN !== 'undefined') {
                new TWEEN.Tween(chip.position)
                    .to({ y: targetY, z: targetZ }, posRiseDuration)
                    .delay(idx * TOKEN_CHIP_STYLE.riseDelay)
                    .easing(TWEEN.Easing.Quadratic.Out)
                    .start();
            } else {
                chip.position.y = targetY;
                chip.position.z = targetZ;
            }
        });
    };
    chipFontLoader.load(
        'https://threejs.org/examples/fonts/helvetiker_regular.typeface.json',
        (font) => {
            spawnTokenChips(font);
            spawnPositionChips(font);
        },
        undefined,
        (err) => {
            console.warn('Token chip font failed to load, rendering without labels.', err);
            spawnTokenChips(null);
            spawnPositionChips(null);
        }
    );

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
        lnTop.setMaterialProperties({ opacity: 1.0, transparent: false, emissiveIntensity: 0.05 });
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
        vocabTop.setColor(new THREE.Color(0x000000));
        vocabTop.setMaterialProperties({ opacity: 1.0, transparent: false, emissiveIntensity: 0.0 });
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
initConveyorSkipButton(pipeline);
initSettingsModal(pipeline);
initPerfOverlay();

const selectionPanel = initSelectionPanel();
if (pipeline.engine && typeof pipeline.engine.setRaycastSelectionHandler === 'function') {
    pipeline.engine.setRaycastSelectionHandler(selection => {
        selectionPanel.handleSelection(selection);
    });
}
