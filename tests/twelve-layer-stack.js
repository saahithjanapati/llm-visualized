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
    TOP_LN_TO_TOP_EMBED_GAP,
    TOP_LOGIT_BAR_MAX_COUNT,
    TOP_LOGIT_BAR_MIN_HEIGHT,
    TOP_LOGIT_BAR_MAX_HEIGHT,
    TOP_LOGIT_BAR_HEIGHT_GAMMA,
    TOP_LOGIT_BAR_LOW_SPLIT,
    TOP_LOGIT_BAR_LOW_GAMMA,
    TOP_LOGIT_BAR_WIDTH_SCALE,
    TOP_LOGIT_BAR_GAP_FRACTION,
    TOP_LOGIT_BAR_DEPTH_SCALE,
    TOP_LOGIT_BAR_INSET_X,
    TOP_LOGIT_BAR_Y_OFFSET,
    TOP_LOGIT_BAR_OPACITY,
    TOP_LOGIT_BAR_RISE_DURATION_MS,
    TOP_LOGIT_BAR_RISE_STAGGER_MS
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
import { initParameterCounter } from '../src/ui/parameterCounter.js';
import { initSettingsModal } from '../src/ui/settingsModal.js';
import { initPauseButton } from '../src/ui/pauseButton.js';
import { initConveyorSkipButton } from '../src/ui/conveyorSkipButton.js';
import { initSkipToEndButton } from '../src/ui/skipToEndButton.js';
import { initSkipMenu } from '../src/ui/skipMenu.js';
import { initSelectionPanel } from '../src/ui/selectionPanel.js';
import { FontLoader } from 'three/examples/jsm/loaders/FontLoader.js';
import { CaptureActivationSource } from '../src/data/CaptureActivationSource.js';
import { precomputeActivationCaches } from '../src/utils/activationPrecompute.js';

// -----------------------------------------------------------------------------
// Demo configuration for the 12-layer GPT-2 stack entrypoint.
// -----------------------------------------------------------------------------
const NUM_LAYERS = 12;
const PROMPT_TOKENS = ['Can', '\u0120machines', '\u0120think', '?', '\u0120'];
const POSITION_TOKENS = ['1', '2', '3', '4', '5'];
const DEFAULT_PROMPT_LANES = Math.max(1, PROMPT_TOKENS.length);
// Visual config for the floating "token chip" labels at the base of the stack.
const TOKEN_CHIP_STYLE = {
    padding: 140,
    minWidth: 440,
    minHeight: 150,
    height: 170,
    cornerRadius: 24,
    depth: 12,
    textSize: 90,
    textDepth: 16,
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

// GPT-2 BPE uses a leading U+0120 to indicate a space; render as a normal space.
function formatTokenLabel(token) {
    return token.replace(/^\u0120/, ' ');
}

// Build a rounded rectangle shape used for the token chip body.
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

// Create a 3D chip group with optional text mesh, plus a cached size in userData.
function createTokenChip(label, font, style) {
    let textGeo = null;
    let textShapes = null;
    let textDepth = 0;
    let bounds = null;
    const capOffset = 0.05;
    if (font && label.trim().length) {
        const desiredDepth = Number.isFinite(style.textDepth) ? style.textDepth : 0;
        const chipDepth = Number.isFinite(style.depth) ? style.depth : desiredDepth;
        textDepth = Number.isFinite(chipDepth) ? chipDepth + capOffset * 2 : desiredDepth;
        textShapes = font.generateShapes(label, style.textSize, 2);
        textGeo = new THREE.ExtrudeGeometry(textShapes, {
            depth: textDepth,
            curveSegments: 4,
            bevelEnabled: false
        });
        textGeo.computeBoundingBox();
        textGeo.computeVertexNormals();
        textGeo.translate(0, 0, -textDepth / 2);
        textGeo.computeBoundingBox();
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

    // Add thin caps to avoid transparent edges when viewed from the sides.
    const capMat = chipMat.clone();
    capMat.polygonOffset = false;
    capMat.polygonOffsetFactor = 0;
    capMat.polygonOffsetUnits = 0;
    const capGeo = new THREE.ShapeGeometry(chipShape);
    capGeo.computeVertexNormals();
    const frontCap = new THREE.Mesh(capGeo, capMat);
    frontCap.position.z = style.depth / 2 + capOffset;
    const backCap = new THREE.Mesh(capGeo, capMat);
    backCap.position.z = -style.depth / 2 - capOffset;
    backCap.rotation.y = Math.PI;
    group.add(frontCap, backCap);

    // Optional 3D text mesh, centered to span the chip depth.
    if (textGeo && textWidth > 0 && textHeight > 0) {
        const textBaseMat = new THREE.MeshBasicMaterial({
            color: 0xffffff,
            side: THREE.DoubleSide,
            depthWrite: true,
            depthTest: true,
            polygonOffset: true,
            polygonOffsetFactor: -0.5,
            polygonOffsetUnits: -0.5
        });
        const textCullMat = textBaseMat.clone();
        textCullMat.colorWrite = false;
        textCullMat.depthWrite = false;
        textCullMat.transparent = true;
        textCullMat.opacity = 0;
        const textGroup = new THREE.Group();
        const textMesh = new THREE.Mesh(textGeo, [textCullMat, textBaseMat]);
        textGroup.add(textMesh);
        if (textShapes) {
            const faceGeo = new THREE.ShapeGeometry(textShapes);
            faceGeo.computeVertexNormals();
            const faceOffset = 0.02;
            const frontFace = new THREE.Mesh(faceGeo, textBaseMat);
            frontFace.position.z = textDepth / 2 + faceOffset;
            const backFace = new THREE.Mesh(faceGeo, textBaseMat);
            backFace.position.z = -textDepth / 2 - faceOffset;
            textGroup.add(frontFace, backFace);
        }
        const centerX = (bounds.min.x + bounds.max.x) / 2;
        const centerY = (bounds.min.y + bounds.max.y) / 2;
        textGroup.position.set(-centerX, -centerY, 0);
        group.add(textGroup);
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

function hashStringToSeed(value) {
    if (!value) return 0;
    let hash = 0;
    for (let i = 0; i < value.length; i += 1) {
        hash = ((hash << 5) - hash + value.charCodeAt(i)) | 0;
    }
    return hash >>> 0;
}

function hashToUnit(seed) {
    let x = seed >>> 0;
    x ^= x >>> 16;
    x = Math.imul(x, 0x7feb352d);
    x ^= x >>> 15;
    x = Math.imul(x, 0x846ca68b);
    x ^= x >>> 16;
    return (x >>> 0) / 4294967295;
}

function resolveTokenSeed(entry, fallbackIndex) {
    if (entry && Number.isFinite(entry.token_id)) {
        return Math.floor(entry.token_id) >>> 0;
    }
    if (entry && typeof entry.token === 'string') {
        return hashStringToSeed(entry.token);
    }
    return (fallbackIndex ?? 0) >>> 0;
}

function getBrightTokenColor(seed, cache) {
    if (cache.has(seed)) return cache.get(seed);
    const hue = hashToUnit(seed);
    const saturation = 0.78 + 0.18 * hashToUnit(seed ^ 0x9e3779b9);
    const lightness = 0.5 + 0.18 * hashToUnit(seed ^ 0x85ebca6b);
    const color = new THREE.Color().setHSL(hue, saturation, lightness);
    cache.set(seed, color);
    return color;
}

function computeLogitBarHeight(prob, maxProb) {
    const minHeight = TOP_LOGIT_BAR_MIN_HEIGHT;
    const maxHeight = Math.max(minHeight + 1, TOP_LOGIT_BAR_MAX_HEIGHT);
    const maxProbSafe = Number.isFinite(maxProb) ? maxProb : 0;
    if (maxProbSafe <= 0) return minHeight;
    const linearT = Math.min(1, Math.max(0, Math.max(0, prob) / maxProbSafe));
    const split = Math.min(0.9, Math.max(0.05, TOP_LOGIT_BAR_LOW_SPLIT));
    let t = 0;
    if (linearT <= split) {
        const localT = split > 0 ? (linearT / split) : 0;
        t = Math.pow(localT, TOP_LOGIT_BAR_LOW_GAMMA) * split;
    } else {
        const localT = (linearT - split) / (1 - split);
        t = split + Math.pow(localT, TOP_LOGIT_BAR_HEIGHT_GAMMA) * (1 - split);
    }
    return minHeight + t * (maxHeight - minHeight);
}

function addTopLogitBars({ activationSource, laneTokenIndices, laneZs, vocabCenter, scene, engine }) {
    if (!activationSource || !Array.isArray(laneZs) || !laneZs.length) return;
    if (typeof activationSource.getLogitsForToken !== 'function') return;

    const logitTopK = typeof activationSource.getLogitTopK === 'function'
        ? activationSource.getLogitTopK()
        : 0;
    const barCount = Math.min(TOP_LOGIT_BAR_MAX_COUNT, logitTopK || TOP_LOGIT_BAR_MAX_COUNT);
    if (!barCount) return;

    const bottomWidth = EMBEDDING_MATRIX_PARAMS_VOCAB.width;
    const topWidth = bottomWidth * EMBEDDING_MATRIX_PARAMS_VOCAB.topWidthFactor;
    const useTopWidth = topWidth >= bottomWidth;
    const surfaceWidth = useTopWidth ? topWidth : bottomWidth;
    const slitWidthFactor = useTopWidth
        ? (EMBEDDING_MATRIX_PARAMS_VOCAB.slitTopWidthFactor ?? 1)
        : (EMBEDDING_MATRIX_PARAMS_VOCAB.slitBottomWidthFactor ?? 1);
    const usableWidth = Math.max(0, surfaceWidth * slitWidthFactor - TOP_LOGIT_BAR_INSET_X * 2);
    if (!usableWidth) return;

    const barSpacing = usableWidth / barCount;
    const maxBarWidth = Math.max(0.1, barSpacing * Math.max(0.1, 1 - TOP_LOGIT_BAR_GAP_FRACTION));
    const barWidth = Math.max(0.5, Math.min(barSpacing * TOP_LOGIT_BAR_WIDTH_SCALE, maxBarWidth));
    const barDepth = Math.max(0.5, EMBEDDING_MATRIX_PARAMS_VOCAB.slitWidth * TOP_LOGIT_BAR_DEPTH_SCALE);
    const baseX = vocabCenter.x - usableWidth / 2 + barSpacing / 2;
    const baseY = vocabCenter.y + EMBEDDING_MATRIX_PARAMS_VOCAB.height / 2 + TOP_LOGIT_BAR_Y_OFFSET;

    const barGeometry = new THREE.BoxGeometry(1, 1, 1);
    const barMaterial = new THREE.MeshStandardMaterial({
        color: 0xffffff,
        roughness: 0.35,
        metalness: 0.1,
        emissive: new THREE.Color(0x111111),
        emissiveIntensity: 0.2,
        transparent: TOP_LOGIT_BAR_OPACITY < 1,
        opacity: TOP_LOGIT_BAR_OPACITY,
        vertexColors: true
    });

    const barGroup = new THREE.Group();
    barGroup.name = 'TopLogitBars';
    barGroup.visible = false;
    barGroup.userData.revealed = false;

    const colorCache = new Map();
    const instances = [];
    const instanceLabels = [];
    const instanceEntries = [];

    const tokenIndices = Array.isArray(laneTokenIndices)
        ? laneTokenIndices
        : laneZs.map((_, idx) => idx);

    const laneRows = [];
    let globalMaxProb = 0;

    for (let laneIdx = 0; laneIdx < laneZs.length; laneIdx += 1) {
        const tokenIndex = tokenIndices[laneIdx] ?? laneIdx;
        const logitRow = activationSource.getLogitsForToken(tokenIndex, barCount);
        if (!Array.isArray(logitRow) || !logitRow.length) continue;
        laneRows.push({ laneIdx, logitRow });
        logitRow.forEach(entry => {
            const prob = Number(entry?.prob);
            if (Number.isFinite(prob) && prob > globalMaxProb) {
                globalMaxProb = prob;
            }
        });
    }

    for (let rowIdx = 0; rowIdx < laneRows.length; rowIdx += 1) {
        const { laneIdx, logitRow } = laneRows[rowIdx];
        const laneZ = laneZs[laneIdx] ?? 0;

        for (let i = 0; i < Math.min(barCount, logitRow.length); i += 1) {
            const entry = logitRow[i];
            const prob = Number(entry?.prob);
            if (!Number.isFinite(prob)) continue;
            const height = computeLogitBarHeight(prob, globalMaxProb);
            const seed = resolveTokenSeed(entry, i);
            const barColor = getBrightTokenColor(seed, colorCache);
            const startHeight = Math.max(0.1, TOP_LOGIT_BAR_MIN_HEIGHT * 0.15);
            const xPos = baseX + i * barSpacing;
            if (entry) {
                const tokenText = typeof entry.token === 'string'
                    ? formatTokenLabel(entry.token.replace(/\n/g, '\\n').replace(/\t/g, '\\t'))
                    : '';
                const tokenId = Number.isFinite(entry.token_id) ? entry.token_id : null;
                const labelParts = [];
                if (tokenText) labelParts.push(`token \"${tokenText}\"`);
                if (tokenId !== null) labelParts.push(`id ${tokenId}`);
                if (Number.isFinite(prob)) labelParts.push(`p ${prob.toFixed(3)}`);
                const label = labelParts.length ? `Logit ${labelParts.join(' | ')}` : 'Logit';
                const instanceIndex = instances.length;
                instanceLabels[instanceIndex] = label;
                instanceEntries[instanceIndex] = entry;
                instances.push({
                    x: xPos,
                    z: laneZ,
                    baseY,
                    startHeight,
                    targetHeight: height,
                    color: barColor
                });
            } else {
                const instanceIndex = instances.length;
                instanceLabels[instanceIndex] = 'Logit';
                instanceEntries[instanceIndex] = entry;
                instances.push({
                    x: xPos,
                    z: laneZ,
                    baseY,
                    startHeight,
                    targetHeight: height,
                    color: barColor
                });
            }
        }
    }

    if (!instances.length) return;
    const instanced = new THREE.InstancedMesh(barGeometry, barMaterial, instances.length);
    instanced.name = 'TopLogitBarsMesh';
    instanced.frustumCulled = false; // enable once bounds are computed after the reveal
    instanced.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    instanced.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(instances.length * 3), 3);
    instanced.instanceColor.setUsage(THREE.StaticDrawUsage);
    const dummy = new THREE.Object3D();
    instances.forEach((instance, idx) => {
        dummy.position.set(instance.x, instance.baseY + instance.startHeight / 2, instance.z);
        dummy.scale.set(barWidth, instance.startHeight, barDepth);
        dummy.updateMatrix();
        instanced.setMatrixAt(idx, dummy.matrix);
        instanced.setColorAt(idx, instance.color);
    });
    instanced.instanceMatrix.needsUpdate = true;
    if (instanced.instanceColor) instanced.instanceColor.needsUpdate = true;
    instanced.userData.label = 'Top Logit Bars';
    instanced.userData.instanceLabels = instanceLabels;
    instanced.userData.instanceEntries = instanceEntries;
    instanced.userData.instanceKind = 'logitBar';

    barGroup.userData.instancedMesh = instanced;
    barGroup.userData.instances = instances;
    barGroup.userData.barWidth = barWidth;
    barGroup.userData.barDepth = barDepth;
    barGroup.add(instanced);

    scene.add(barGroup);
    if (engine && typeof engine.registerRaycastRoot === 'function') {
        engine.registerRaycastRoot(barGroup);
    }
    return barGroup;
}

function revealTopLogitBars(barGroup, { immediate = false } = {}) {
    if (!barGroup || barGroup.userData.revealed) return;
    barGroup.userData.revealed = true;
    barGroup.visible = true;

    const instanced = barGroup.userData.instancedMesh;
    const instances = barGroup.userData.instances;
    if (!instanced || !Array.isArray(instances) || !instances.length) return;

    const barWidth = Number.isFinite(barGroup.userData.barWidth) ? barGroup.userData.barWidth : 1;
    const barDepth = Number.isFinite(barGroup.userData.barDepth) ? barGroup.userData.barDepth : 1;
    const dummy = new THREE.Object3D();
    const finalizeInstancing = () => {
        instanced.frustumCulled = true;
        if (typeof instanced.computeBoundingBox === 'function') instanced.computeBoundingBox();
        if (typeof instanced.computeBoundingSphere === 'function') instanced.computeBoundingSphere();
        instanced.instanceMatrix.setUsage(THREE.StaticDrawUsage);
    };
    const applyHeight = (idx, height) => {
        const instance = instances[idx];
        if (!instance) return;
        dummy.position.set(instance.x, instance.baseY + height / 2, instance.z);
        dummy.scale.set(barWidth, height, barDepth);
        dummy.updateMatrix();
        instanced.setMatrixAt(idx, dummy.matrix);
    };

    if (immediate || typeof requestAnimationFrame !== 'function') {
        instances.forEach((instance, idx) => {
            applyHeight(idx, instance.targetHeight);
        });
        instanced.instanceMatrix.needsUpdate = true;
        finalizeInstancing();
        return;
    }

    const startTime = performance.now();
    const duration = TOP_LOGIT_BAR_RISE_DURATION_MS;
    const stagger = TOP_LOGIT_BAR_RISE_STAGGER_MS;
    const easeOutQuad = (t) => 1 - (1 - t) * (1 - t);

    const animate = (now) => {
        let anyActive = false;
        for (let i = 0; i < instances.length; i += 1) {
            const instance = instances[i];
            if (!instance) continue;
            const localStart = startTime + i * stagger;
            const elapsed = now - localStart;
            let height = instance.startHeight;
            if (elapsed > 0) {
                const t = Math.min(1, elapsed / duration);
                const eased = easeOutQuad(t);
                height = instance.startHeight + (instance.targetHeight - instance.startHeight) * eased;
                if (t < 1) anyActive = true;
            } else {
                anyActive = true;
            }
            applyHeight(i, height);
        }
        instanced.instanceMatrix.needsUpdate = true;
        if (anyActive) {
            requestAnimationFrame(animate);
        } else {
            finalizeInstancing();
        }
    };
    requestAnimationFrame(animate);
}

// Temporarily stage the camera near the chips, then optionally return to tower view.
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

// Optionally load pre-baked geometries to skip heavy procedural work.
await loadPrecomputedGeometries('../precomputed_components_slice.glb');

// Activation data + lane count selection (defaults to static prompt if no capture).
let activationSource = null;
let laneTokenIndices = null;
let laneCount = NUM_VECTOR_LANES;
// Special "/full" path shows all capture tokens instead of the prompt subset.
const isFullTokenMode = (() => {
    const path = window.location.pathname.replace(/\/+$/, '');
    return path.endsWith('/full');
})();
const statusDiv = document.getElementById('statusOverlay');
const setLoadingStatus = (text) => {
    if (statusDiv) statusDiv.textContent = text;
};
try {
    // Load activation capture data from ?capture= or ?file= query param.
    const params = new URLSearchParams(window.location.search);
    const captureFile = params.get('capture') || params.get('file') || 'capture.json';
    const captureUrl = captureFile.startsWith('http')
        ? captureFile
        : `/${captureFile.replace(/^\/+/, '')}`;
    activationSource = await CaptureActivationSource.load(captureUrl);
    const tokensInCapture = typeof activationSource.getTokenCount === 'function'
        ? activationSource.getTokenCount()
        : 0;
    const safeTokenCount = Math.max(0, tokensInCapture || 0);
    const desiredLanes = Math.max(1, safeTokenCount || laneCount);
    if (isFullTokenMode) {
        laneCount = desiredLanes;
    } else {
        const cappedLanes = safeTokenCount ? Math.min(safeTokenCount, DEFAULT_PROMPT_LANES) : DEFAULT_PROMPT_LANES;
        laneCount = Math.min(desiredLanes, cappedLanes);
    }
    laneTokenIndices = activationSource.getLaneTokenIndices(laneCount);
} catch (err) {
    console.warn('Capture data unavailable; falling back to random vectors.', err);
}
if (!activationSource) {
    laneCount = DEFAULT_PROMPT_LANES;
}

// Sync global lane counts so component spacing/animation widths align.
setNumVectorLanes(laneCount);
setAnimationLaneCount(laneCount);

// Warm activation caches so lane startup avoids async stalls.
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

// Labels for the token/position chips at the base of the stack.
const tokenLabelsFromCapture = isFullTokenMode && activationSource && laneTokenIndices
    ? laneTokenIndices.map((idx) => activationSource.getTokenString(idx) || '')
    : PROMPT_TOKENS;
const positionLabelsFromCapture = isFullTokenMode && activationSource && laneTokenIndices
    ? laneTokenIndices.map((idx) => String(idx + 1))
    : POSITION_TOKENS;

// Skip intro typing screen for direct animation entry.
appState.skipIntro = true;

// Set default playback speed to fast on load.
try { setPlaybackSpeed('fast'); } catch (_) { /* no-op */ }

// GPT-2 tower - initialize immediately.
MHSAAnimation.ENABLE_SELF_ATTENTION = true;
const gptCanvas = document.getElementById('gptCanvas');
const camPos    = new THREE.Vector3(0, 11000, 16000);
const camTarget = new THREE.Vector3(0, 9000, 0);
const targetClampRadius = Math.max(8000, NUM_LAYERS * 900);
const autoCameraHeadBias = 0.0;
const followDefaultCameraOffset = new THREE.Vector3(-1215.87, 465.86, 3350.33);
const followDefaultTargetOffset = new THREE.Vector3(1675.46, 227.33, -469.85);
const followMhsaCameraOffset = new THREE.Vector3(1366.76, 1062.82, 1936.74);
const followMhsaTargetOffset = new THREE.Vector3(3699.19, 110.55, 268.33);
const followConcatCameraOffset = new THREE.Vector3(403.43, -14.39, 7.47);
const followConcatTargetOffset = new THREE.Vector3(3383.52, 26.22, 364.05);
const followLnCameraOffset = new THREE.Vector3(605.51, -78.03, 2433.13);
const followLnTargetOffset = new THREE.Vector3(1026.71, 144.37, -607.81);
const followTravelCameraOffset = new THREE.Vector3(1106.53, -860.48, 1389.16);
const followTravelTargetOffset = new THREE.Vector3(4038.68, -398.41, 601.18);
const followTravelMobileCameraOffset = new THREE.Vector3(650.00, -731.41, 1165.51);
const followTravelMobileTargetOffset = new THREE.Vector3(2339.23, -550.29, 738.49);
// LayerPipeline builds all static visuals first, then advances active lanes upward.
const pipeline = new LayerPipeline(gptCanvas, NUM_LAYERS, {
    cameraPosition: camPos,
    cameraTarget: camTarget,
    targetClampCenter: camTarget,
    targetClampRadius,
    autoCameraHeadBias,
    autoCameraDefaultCameraOffset: followDefaultCameraOffset,
    autoCameraDefaultTargetOffset: followDefaultTargetOffset,
    autoCameraMhsaCameraOffset: followMhsaCameraOffset,
    autoCameraMhsaTargetOffset: followMhsaTargetOffset,
    autoCameraConcatCameraOffset: followConcatCameraOffset,
    autoCameraConcatTargetOffset: followConcatTargetOffset,
    autoCameraLnCameraOffset: followLnCameraOffset,
    autoCameraLnTargetOffset: followLnTargetOffset,
    autoCameraTravelCameraOffset: followTravelCameraOffset,
    autoCameraTravelTargetOffset: followTravelTargetOffset,
    autoCameraTravelMobileCameraOffset: followTravelMobileCameraOffset,
    autoCameraTravelMobileTargetOffset: followTravelMobileTargetOffset,
    autoCameraMobileScale: 1.8,
    autoCameraMobileShiftX: -600,
    autoCameraMhsaMobileShiftX: -2000,
    autoCameraTravelMobileShiftX: -4500,
    autoCameraScaleMinWidth: 360,
    autoCameraScaleMaxWidth: 980,
    autoCameraSmoothAlpha: 0.06,
    autoCameraOffsetLerpAlpha: 0.06,
    autoCameraViewBlendAlpha: 0.05,
    activationSource,
    laneCount
});

// Show GPT canvas immediately.
gptCanvas.style.display = 'block';
try {
    const eng = pipeline.engine;
    // Keep shadows off for this demo to reduce GPU cost.
    eng.renderer.shadowMap.enabled = false;
    eng.scene.traverse((obj) => {
        if (obj.isMesh) { obj.castShadow = false; obj.receiveShadow = false; }
        if (obj.isLight) { obj.castShadow = false; }
    });
} catch (_) {}

// Embedding matrices + token chips (static visuals at tower base).
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

    // Precompute Z positions for each lane so chips align with vector lanes.
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
        // Animate the token chips upward from a resting "stash" position.
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

            // Keep a static chip below the stack for context once the animated one rises.
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

            // Static chip for the position stream, matching the token chips below.
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

    // Top-of-tower LayerNorm + vocab projection marker.
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
        const vocabTopPos = new THREE.Vector3();
        vocabTop.group.getWorldPosition(vocabTopPos);
        const topLogitBars = addTopLogitBars({
            activationSource,
            laneTokenIndices,
            laneZs,
            vocabCenter: vocabTopPos,
            scene: pipeline.engine.scene,
            engine: pipeline.engine
        });
        if (topLogitBars && pipeline && typeof pipeline.addEventListener === 'function') {
            const maybeReveal = () => {
                if (typeof pipeline.isForwardPassComplete !== 'function' || !pipeline.isForwardPassComplete()) {
                    return false;
                }
                const immediate = typeof pipeline.isSkipToEndActive === 'function'
                    ? pipeline.isSkipToEndActive()
                    : false;
                revealTopLogitBars(topLogitBars, { immediate });
                return true;
            };
            const onProgress = () => {
                if (maybeReveal()) {
                    pipeline.removeEventListener('progress', onProgress);
                }
            };
            pipeline.addEventListener('progress', onProgress);
            onProgress();
        }
    }
} catch (_) { /* optional – embedding visuals are non-critical */ }

// Initialize UI modules (status, settings, pause/skip, selection panel).
initIntroAnimation(pipeline, gptCanvas);
initStatusOverlay(pipeline, NUM_LAYERS);
initParameterCounter(pipeline, NUM_LAYERS);
initPauseButton(pipeline);
initConveyorSkipButton(pipeline);
initSkipToEndButton(pipeline);
initSkipMenu();
initSettingsModal(pipeline);

const followModeBtn = document.getElementById('followModeBtn');
const followSettingsToggle = document.getElementById('toggleAutoCamera');
const updateFollowButton = (enabled) => {
    if (!followModeBtn) return;
    const isOn = !!enabled;
    followModeBtn.dataset.state = isOn ? 'enabled' : 'disabled';
    followModeBtn.setAttribute('aria-pressed', String(isOn));
    followModeBtn.textContent = isOn ? 'Follow mode on' : 'Enable Follow Mode';
    followModeBtn.setAttribute('aria-label', isOn ? 'Follow mode enabled' : 'Enable follow mode');
    followModeBtn.setAttribute('title', isOn ? 'Follow mode enabled' : 'Enable follow mode');
    followModeBtn.disabled = isOn;
};

const setFollowMode = (enabled, { resetView = false } = {}) => {
    const next = !!enabled;
    if (appState.autoCameraFollow === next && pipeline?.isAutoCameraFollowEnabled?.() === next) {
        updateFollowButton(next);
        if (followSettingsToggle) followSettingsToggle.checked = next;
        return;
    }
    appState.autoCameraFollow = next;
    pipeline?.setAutoCameraFollow?.(next, { immediate: next, resetView: next && resetView, smoothReset: next && resetView });
    updateFollowButton(next);
    if (followSettingsToggle) followSettingsToggle.checked = next;
};

if (followModeBtn) {
    followModeBtn.addEventListener('click', (event) => {
        event.preventDefault();
        setFollowMode(true, { resetView: true });
    });
}

if (followSettingsToggle) {
    followSettingsToggle.addEventListener('change', () => {
        updateFollowButton(!!followSettingsToggle.checked);
    });
}

if (pipeline?.engine?.controls?.addEventListener) {
    pipeline.engine.controls.addEventListener('start', () => {
        if (pipeline?.isAutoCameraFollowEnabled?.()) {
            setFollowMode(false);
        }
    });
}

if (typeof window !== 'undefined') {
    window.addEventListener('autoCameraFollowRequest', (event) => {
        const enabled = !!event?.detail?.enabled;
        if (!enabled) {
            setFollowMode(false);
        }
    });
}

updateFollowButton(pipeline?.isAutoCameraFollowEnabled?.());

const topControls = document.getElementById('topControls');
const isSkinnyScreen = () => window.matchMedia('(max-aspect-ratio: 1/1), (max-width: 880px)').matches;
let topControlsHideTimer = null;

const showTopControls = () => {
    if (!topControls) return;
    topControls.removeAttribute('data-auto-hidden');
    if (isSkinnyScreen()) {
        clearTimeout(topControlsHideTimer);
        topControlsHideTimer = setTimeout(() => {
            if (isSkinnyScreen()) {
                topControls.setAttribute('data-auto-hidden', 'true');
            }
        }, 5000);
    }
};

const handleViewportChange = () => {
    if (!topControls) return;
    if (isSkinnyScreen()) {
        showTopControls();
    } else {
        clearTimeout(topControlsHideTimer);
        topControls.removeAttribute('data-auto-hidden');
    }
};

handleViewportChange();
window.addEventListener('resize', handleViewportChange);

const selectionPanel = initSelectionPanel();
if (pipeline.engine && typeof pipeline.engine.setRaycastSelectionHandler === 'function') {
    pipeline.engine.setRaycastSelectionHandler(selection => {
        if (!selection || !selection.label) {
            showTopControls();
            return;
        }
        selectionPanel.handleSelection(selection);
    });
}
