import * as THREE from 'three';
import { FontLoader } from 'three/examples/jsm/loaders/FontLoader.js';
import {
    EMBEDDING_MATRIX_PARAMS_VOCAB,
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
    TOP_LOGIT_BAR_RISE_DURATION_MS
} from '../../utils/constants.js';
import { resolveLogitEntryText } from '../../utils/logitTokenText.js';
import { scaleGlobalEmissiveIntensity } from '../../utils/materialUtils.js';
import { isIncompleteUtf8TokenId } from '../../utils/tokenEncodingNotes.js';
import {
    buildPromptTokenChipEntries,
    getActivePromptTokenChipLookup,
    resolvePromptTokenChipColorState,
    resolveTokenChipColorKey
} from '../../ui/tokenChipColorUtils.js';
import { formatTokenLabel } from './tokenLabels.js';
import {
    getLogitTokenChipColorHex,
    getLogitTokenColorUnit,
    resolveLogitTokenSeed
} from './logitColor.js';
import { resolveChosenTokenCandidateForToken } from '../../utils/captureTokenSelection.js';

const LOGIT_LABEL_FONT_URL = 'https://threejs.org/examples/fonts/helvetiker_regular.typeface.json';
const LOGIT_LABEL_TEXT_SIZE_MIN = 320;
const LOGIT_LABEL_TEXT_SIZE_MAX = 1200;
const LOGIT_LABEL_TEXT_SIZE_SCALE = 11.0;
const LOGIT_LABEL_DEPTH_SCALE = 0.3;
const LOGIT_LABEL_GAP_Y = 88;
const LOGIT_LABEL_EXTRA_Y = 300;
const LOGIT_LABEL_SIDE_GAP = 130;
const LOGIT_LABEL_MAX_WIDTH_MULTIPLIER = 40;
const LOGIT_LABEL_MIN_WIDTH = 1800;
const LOGIT_LABEL_LINE_OPACITY = 0.65;
const LOGIT_LABEL_LINE_INSET = 6;

let cachedLogitFont = null;
let cachedLogitFontPromise = null;

function loadLogitLabelFont() {
    if (cachedLogitFont) return Promise.resolve(cachedLogitFont);
    if (cachedLogitFontPromise) return cachedLogitFontPromise;
    cachedLogitFontPromise = new Promise((resolve, reject) => {
        const loader = new FontLoader();
        loader.load(
            LOGIT_LABEL_FONT_URL,
            (font) => {
                cachedLogitFont = font;
                resolve(font);
            },
            undefined,
            (err) => {
                cachedLogitFontPromise = null;
                reject(err);
            }
        );
    });
    return cachedLogitFontPromise;
}

function sanitizeLogitToken(token) {
    if (token === null || token === undefined) return '';
    const raw = String(token);
    if (!raw.length) return '';
    return raw.replace(/\n/g, '\\n').replace(/\t/g, '\\t');
}

function resolveLogitLabelText(entry) {
    if (!entry || typeof entry !== 'object') return '';
    const tokenText = resolveLogitEntryText(entry);
    if (tokenText) {
        return formatTokenLabel(sanitizeLogitToken(tokenText));
    }
    const tokenId = resolveChosenLogitTokenId(entry);
    if (Number.isFinite(tokenId)) {
        return `#${tokenId}`;
    }
    return '';
}

function resolveChosenLogitTokenId(entry) {
    const rawTokenId = Number(entry?.token_id ?? entry?.tokenId);
    return Number.isFinite(rawTokenId) ? Math.floor(rawTokenId) : null;
}

function resolveChosenLogitProbability(entry) {
    const rawProbability = Number(entry?.prob ?? entry?.probability);
    return Number.isFinite(rawProbability) ? rawProbability : null;
}

function applyChosenLogitSelectionMetadata(object, chosen, labelText, { setName = true } = {}) {
    if (!object) return;
    object.userData = object.userData || {};
    object.userData.label = `Chosen token: ${labelText}`;
    if (setName) object.name = `Chosen token: ${labelText}`;
    const tokenLabel = (typeof chosen?.tokenLabel === 'string' && chosen.tokenLabel.length)
        ? chosen.tokenLabel
        : ((typeof labelText === 'string' && labelText.length) ? labelText : '');
    if (tokenLabel) {
        object.userData.tokenLabel = tokenLabel;
    }
    object.userData.logitEntry = chosen.entry;
    if (Number.isFinite(chosen.tokenIndex)) {
        object.userData.tokenIndex = Math.floor(chosen.tokenIndex);
    }
    const tokenId = resolveChosenLogitTokenId(chosen.entry);
    if (Number.isFinite(tokenId)) {
        object.userData.tokenId = tokenId;
    }
    const probability = resolveChosenLogitProbability(chosen.entry);
    if (Number.isFinite(probability)) {
        object.userData.probability = probability;
    }
}

function resolveChosenLogitTokenLabel(entry) {
    const tokenText = resolveLogitEntryText(entry);
    return tokenText ? formatTokenLabel(sanitizeLogitToken(tokenText)) : '';
}

function resolvePromptContextTokenLabel(activationSource, tokenIndex) {
    if (!Number.isFinite(tokenIndex) || !activationSource || typeof activationSource.getTokenString !== 'function') {
        return '';
    }
    return formatTokenLabel(activationSource.getTokenString(Math.floor(tokenIndex)) ?? '');
}

export function resolveChosenLogitDisplayColorKey({
    activationSource = null,
    laneTokenIndices = null,
    chosenEntry = null,
    chosenTokenIndex = null,
    fallbackIndex = 0
} = {}) {
    const tokenIndices = Array.isArray(laneTokenIndices) ? laneTokenIndices.slice() : [];
    const tokenLabels = tokenIndices.map((tokenIndex) => resolvePromptContextTokenLabel(activationSource, tokenIndex));
    const tokenIds = tokenIndices.map((tokenIndex) => {
        if (!Number.isFinite(tokenIndex) || !activationSource || typeof activationSource.getTokenId !== 'function') {
            return null;
        }
        return activationSource.getTokenId(Math.floor(tokenIndex));
    });
    const chosenTokenLabel = resolveChosenLogitTokenLabel(chosenEntry);
    const previousLookup = getActivePromptTokenChipLookup();
    const promptEntries = buildPromptTokenChipEntries({
        tokenLabels,
        tokenIndices,
        tokenIds,
        generatedToken: chosenTokenLabel
            ? {
                tokenIndex: chosenTokenIndex,
                tokenId: resolveChosenLogitTokenId(chosenEntry),
                tokenLabel: chosenTokenLabel,
                logitEntry: chosenEntry
            }
            : null
    });
    const promptColorState = resolvePromptTokenChipColorState(promptEntries, { previousLookup });
    return resolveTokenChipColorKey({
        tokenIndex: chosenTokenIndex,
        tokenId: resolveChosenLogitTokenId(chosenEntry),
        tokenLabel: chosenTokenLabel
    }, fallbackIndex, { lookup: promptColorState.lookup });
}

function resolveChosenLogitDisplayColor({
    activationSource = null,
    laneTokenIndices = null,
    chosenEntry = null,
    chosenTokenIndex = null,
    fallbackIndex = 0,
    fallbackColor = 0xffffff
} = {}) {
    try {
        const colorKey = resolveChosenLogitDisplayColorKey({
            activationSource,
            laneTokenIndices,
            chosenEntry,
            chosenTokenIndex,
            fallbackIndex
        });
        if (Number.isFinite(colorKey)) {
            return new THREE.Color(getLogitTokenChipColorHex(colorKey));
        }
    } catch (_) {
        // Fall back to the logit-bar hue if prompt-strip color resolution fails.
    }
    return fallbackColor instanceof THREE.Color
        ? fallbackColor.clone()
        : new THREE.Color(fallbackColor);
}

function createExtrudedTextGroup(label, font, { size, depth, color }) {
    if (!font || !label || !label.trim()) return null;
    const shapes = font.generateShapes(label, size, 2);
    if (!shapes || !shapes.length) return null;
    const geometry = new THREE.ExtrudeGeometry(shapes, {
        depth,
        curveSegments: 4,
        bevelEnabled: false
    });
    geometry.computeBoundingBox();
    geometry.computeVertexNormals();
    geometry.translate(0, 0, -depth / 2);
    geometry.computeBoundingBox();

    const bounds = geometry.boundingBox;
    const width = bounds ? Math.max(0, bounds.max.x - bounds.min.x) : 0;
    const height = bounds ? Math.max(0, bounds.max.y - bounds.min.y) : 0;
    const centerX = bounds ? (bounds.min.x + bounds.max.x) / 2 : 0;
    const centerY = bounds ? (bounds.min.y + bounds.max.y) / 2 : 0;
    if (bounds) {
        geometry.translate(-centerX, -centerY, 0);
    }

    const matColor = color instanceof THREE.Color ? color.clone() : new THREE.Color(color ?? 0xffffff);
    const material = new THREE.MeshStandardMaterial({
        color: matColor,
        roughness: 0.32,
        metalness: 0.12,
        emissive: matColor.clone().multiplyScalar(0.15),
        emissiveIntensity: scaleGlobalEmissiveIntensity(0.9),
        side: THREE.DoubleSide
    });
    const mesh = new THREE.Mesh(geometry, material);
    const group = new THREE.Group();
    const textGroup = new THREE.Group();
    textGroup.add(mesh);

    const faceGeo = new THREE.ShapeGeometry(shapes);
    faceGeo.computeVertexNormals();
    if (bounds) {
        faceGeo.translate(-centerX, -centerY, 0);
    }
    const faceOffset = 0.02;
    const frontFace = new THREE.Mesh(faceGeo, material);
    frontFace.position.z = depth / 2 + faceOffset;
    const backFace = new THREE.Mesh(faceGeo, material);
    backFace.position.z = -depth / 2 - faceOffset;
    textGroup.add(frontFace, backFace);

    group.add(textGroup);
    group.userData.size = { width, height, depth };
    return group;
}

function applyChosenLabelColorToObject(object, color) {
    if (!object || !color) return;
    const applyToMaterial = (material) => {
        if (!material) return;
        if (material.color?.copy) material.color.copy(color);
        if (material.emissive?.copy) {
            material.emissive.copy(color).multiplyScalar(0.15);
        }
        material.needsUpdate = true;
    };

    if (object.material) {
        const materials = Array.isArray(object.material) ? object.material : [object.material];
        materials.forEach((material) => applyToMaterial(material));
    }
    if (typeof object.traverse === 'function') {
        object.traverse((node) => {
            if (!node || node === object || !node.material) return;
            const materials = Array.isArray(node.material) ? node.material : [node.material];
            materials.forEach((material) => applyToMaterial(material));
        });
    }
}

function refreshChosenLabelGroupColors(barGroup) {
    const labelGroup = barGroup?.userData?.chosenLabelGroup;
    const chosenEntries = Array.isArray(barGroup?.userData?.chosenEntries)
        ? barGroup.userData.chosenEntries
        : [];
    if (!labelGroup || !chosenEntries.length) return;

    const activationSource = barGroup?.userData?.activationSource ?? null;
    const laneTokenIndices = Array.isArray(barGroup?.userData?.laneTokenIndices)
        ? barGroup.userData.laneTokenIndices
        : [];

    labelGroup.children.forEach((child) => {
        const chosenEntryIndex = Number(child?.userData?.chosenEntryIndex);
        if (!Number.isFinite(chosenEntryIndex)) return;
        const chosen = chosenEntries[chosenEntryIndex];
        if (!chosen) return;
        const color = resolveChosenLogitDisplayColor({
            activationSource,
            laneTokenIndices,
            chosenEntry: chosen.entry,
            chosenTokenIndex: chosen.tokenIndex,
            fallbackIndex: chosen.fallbackIndex ?? chosenEntryIndex,
            fallbackColor: chosen.fallbackColor ?? 0xffffff
        });
        applyChosenLabelColorToObject(child, color);
    });
}

function refreshChosenBarGroupColors(barGroup) {
    const instancedMesh = barGroup?.userData?.instancedMesh;
    const chosenEntries = Array.isArray(barGroup?.userData?.chosenEntries)
        ? barGroup.userData.chosenEntries
        : [];
    if (!instancedMesh || !instancedMesh.isInstancedMesh || !chosenEntries.length) return;

    const activationSource = barGroup?.userData?.activationSource ?? null;
    const laneTokenIndices = Array.isArray(barGroup?.userData?.laneTokenIndices)
        ? barGroup.userData.laneTokenIndices
        : [];
    let changed = false;

    chosenEntries.forEach((chosen, chosenEntryIndex) => {
        if (chosen?.useChipColor !== true) return;
        const instanceIndex = Number(chosen?.instanceIndex);
        if (!Number.isFinite(instanceIndex) || instanceIndex < 0) return;
        const color = resolveChosenLogitDisplayColor({
            activationSource,
            laneTokenIndices,
            chosenEntry: chosen.entry,
            chosenTokenIndex: chosen.tokenIndex,
            fallbackIndex: chosen.fallbackIndex ?? chosenEntryIndex,
            fallbackColor: chosen.fallbackColor ?? 0xffffff
        });
        instancedMesh.setColorAt(Math.floor(instanceIndex), color);
        changed = true;
    });

    if (changed && instancedMesh.instanceColor) {
        instancedMesh.instanceColor.needsUpdate = true;
    }
}

function buildChosenLogitLabelGroup(barGroup, font) {
    const chosenEntries = barGroup?.userData?.chosenEntries;
    if (!Array.isArray(chosenEntries) || !chosenEntries.length) return null;
    const activationSource = barGroup?.userData?.activationSource ?? null;
    const laneTokenIndices = Array.isArray(barGroup?.userData?.laneTokenIndices)
        ? barGroup.userData.laneTokenIndices
        : [];
    const barWidth = Number.isFinite(barGroup.userData.barWidth) ? barGroup.userData.barWidth : 1;
    const barDepth = Number.isFinite(barGroup.userData.barDepth) ? barGroup.userData.barDepth : 0;
    const baseX = Number.isFinite(barGroup.userData.baseX) ? barGroup.userData.baseX : null;
    const barSpacing = Number.isFinite(barGroup.userData.barSpacing) ? barGroup.userData.barSpacing : null;
    const barCount = Number.isFinite(barGroup.userData.barCount) ? barGroup.userData.barCount : null;
    const leftEdge = baseX !== null ? baseX - barWidth / 2 : null;
    const rightEdge = (baseX !== null && barSpacing !== null && barCount !== null)
        ? baseX + Math.max(0, barCount - 1) * barSpacing + barWidth / 2
        : null;
    const midX = (leftEdge !== null && rightEdge !== null)
        ? (leftEdge + rightEdge) / 2
        : 0;
    const textSize = Math.max(
        LOGIT_LABEL_TEXT_SIZE_MIN,
        Math.min(LOGIT_LABEL_TEXT_SIZE_MAX, barWidth * LOGIT_LABEL_TEXT_SIZE_SCALE)
    );
    const baseTextDepth = Math.max(8, textSize * LOGIT_LABEL_DEPTH_SCALE);
    const maxLabelWidth = Math.max(LOGIT_LABEL_MIN_WIDTH, barWidth * LOGIT_LABEL_MAX_WIDTH_MULTIPLIER);
    const labelGroup = new THREE.Group();
    labelGroup.name = 'TopLogitChosenTokens';
    labelGroup.visible = false;

    const gapY = LOGIT_LABEL_GAP_Y;
    const extraY = LOGIT_LABEL_EXTRA_Y;
    const sideGap = LOGIT_LABEL_SIDE_GAP;

    chosenEntries.forEach((chosen, chosenEntryIndex) => {
        const labelText = (
            typeof chosen?.tokenLabel === 'string'
            && chosen.tokenLabel.length
        )
            ? chosen.tokenLabel
            : resolveLogitLabelText(chosen.entry);
        if (!labelText) return;
        const labelColor = resolveChosenLogitDisplayColor({
            activationSource,
            laneTokenIndices,
            chosenEntry: chosen.entry,
            chosenTokenIndex: chosen.tokenIndex,
            fallbackIndex: chosen.fallbackIndex ?? chosenEntryIndex,
            fallbackColor: chosen.fallbackColor ?? 0xffffff
        });
        const textGroup = createExtrudedTextGroup(labelText, font, {
            size: textSize,
            depth: baseTextDepth,
            color: labelColor
        });
        if (!textGroup) return;

        let { width, height } = textGroup.userData.size || { width: 0, height: 0 };
        if (width > maxLabelWidth && width > 0) {
            const scale = maxLabelWidth / width;
            textGroup.scale.setScalar(scale);
            width *= scale;
            height *= scale;
        }

        const barTopY = chosen.baseY + chosen.targetHeight;
        const labelY = barTopY + gapY + height / 2 + extraY;
        const baseOffset = barWidth / 2 + sideGap + width / 2;
        const sideDir = chosen.x >= midX ? 1 : -1;
        let labelX = chosen.x + baseOffset * sideDir;
        if (rightEdge !== null && labelX + width / 2 > rightEdge + sideGap) {
            labelX = rightEdge + sideGap + width / 2;
        }
        if (leftEdge !== null && labelX - width / 2 < leftEdge - sideGap) {
            labelX = leftEdge - sideGap - width / 2;
        }
        const labelZ = chosen.z;
        textGroup.position.set(labelX, labelY, labelZ);
        textGroup.userData = textGroup.userData || {};
        textGroup.userData.chosenEntryIndex = chosenEntryIndex;
        applyChosenLogitSelectionMetadata(textGroup, chosen, labelText);
        textGroup.traverse((node) => {
            if (node === textGroup) return;
            applyChosenLogitSelectionMetadata(node, chosen, labelText, { setName: false });
        });

        const lineStartX = chosen.x;
        const lineStartZ = chosen.z;
        const lineStart = new THREE.Vector3(lineStartX, barTopY, lineStartZ);
        const labelCenter = new THREE.Vector3(labelX, labelY, labelZ);
        const lineDir = labelCenter.clone().sub(lineStart);
        if (lineDir.lengthSq() < 1e-6) lineDir.set(1, 0, 0);
        lineDir.normalize();
        const halfW = (width * textGroup.scale.x) / 2;
        const halfH = (height * textGroup.scale.y) / 2;
        const textDepthWorld = (textGroup.userData.size?.depth ?? baseTextDepth) * textGroup.scale.z;
        const halfD = textDepthWorld / 2;
        const textBounds = new THREE.Box3(
            new THREE.Vector3(labelX - halfW, labelY - halfH, labelZ - halfD),
            new THREE.Vector3(labelX + halfW, labelY + halfH, labelZ + halfD)
        );
        const hit = new THREE.Vector3();
        const hasHit = new THREE.Ray(lineStart.clone(), lineDir).intersectBox(textBounds, hit);
        const maxInset = Math.max(0, Math.min(halfW, halfH, halfD) * 0.8);
        const lineInset = Math.min(LOGIT_LABEL_LINE_INSET, maxInset);
        const lineEnd = hasHit ? hit.addScaledVector(lineDir, lineInset) : labelCenter;
        const lineGeometry = new THREE.BufferGeometry().setFromPoints([lineStart, lineEnd]);
        const lineMaterial = new THREE.LineBasicMaterial({
            color: labelColor || 0xffffff,
            transparent: true,
            opacity: LOGIT_LABEL_LINE_OPACITY
        });
        const line = new THREE.Line(lineGeometry, lineMaterial);
        line.userData = line.userData || {};
        line.userData.chosenEntryIndex = chosenEntryIndex;
        applyChosenLogitSelectionMetadata(line, chosen, labelText, { setName: false });

        labelGroup.add(line);
        labelGroup.add(textGroup);
    });

    if (!labelGroup.children.length) return null;
    return labelGroup;
}

function ensureChosenLabelGroup(barGroup) {
    if (!barGroup || barGroup.userData?.chosenLabelLoading) return;
    if (!Array.isArray(barGroup.userData?.chosenEntries) || !barGroup.userData.chosenEntries.length) return;
    if (barGroup.userData.chosenLabelGroup) return;

    barGroup.userData.chosenLabelLoading = true;
    loadLogitLabelFont()
        .then((font) => {
            if (!barGroup || !barGroup.parent) return;
            const labelGroup = buildChosenLogitLabelGroup(barGroup, font);
            if (!labelGroup) return;
            barGroup.userData.chosenLabelGroup = labelGroup;
            barGroup.add(labelGroup);
            if (barGroup.userData.revealChosenLabels) {
                refreshChosenLabelGroupColors(barGroup);
                labelGroup.visible = true;
            }
        })
        .catch((err) => {
            console.warn('Top logit label font failed to load; skipping chosen-token labels.', err);
        })
        .finally(() => {
            if (barGroup?.userData) barGroup.userData.chosenLabelLoading = false;
        });
}

function revealChosenLabelGroup(barGroup) {
    if (!barGroup || !barGroup.userData) return;
    barGroup.userData.revealChosenLabels = true;
    refreshChosenBarGroupColors(barGroup);
    const labelGroup = barGroup.userData.chosenLabelGroup;
    if (labelGroup) {
        refreshChosenLabelGroupColors(barGroup);
        labelGroup.visible = true;
    }
}

function queueRevealCompleteCallback(barGroup, callback) {
    if (!barGroup || !barGroup.userData || typeof callback !== 'function') return;
    if (barGroup.userData.revealComplete) {
        try { callback(); } catch (_) { /* callback errors are non-fatal */ }
        return;
    }
    if (!Array.isArray(barGroup.userData.revealCompleteCallbacks)) {
        barGroup.userData.revealCompleteCallbacks = [];
    }
    barGroup.userData.revealCompleteCallbacks.push(callback);
}

function markRevealComplete(barGroup) {
    if (!barGroup || !barGroup.userData || barGroup.userData.revealComplete) return;
    barGroup.userData.revealComplete = true;
    const callbacks = Array.isArray(barGroup.userData.revealCompleteCallbacks)
        ? barGroup.userData.revealCompleteCallbacks.splice(0)
        : [];
    callbacks.forEach((callback) => {
        try { callback(); } catch (_) { /* callback errors are non-fatal */ }
    });
}

function getBrightTokenColor(seed, cache) {
    if (cache.has(seed)) return cache.get(seed);
    const { h, s, l } = getLogitTokenColorUnit(seed);
    const color = new THREE.Color().setHSL(h, s, l);
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

function formatHoverProbabilityPercentage(prob) {
    if (!Number.isFinite(prob)) return '';
    const percentage = prob * 100;
    const abs = Math.abs(percentage);
    if (abs === 0) return '0%';
    if (abs < 0.0001) return `${percentage.toExponential(2)}%`;
    if (abs < 0.01) return `${percentage.toFixed(4).replace(/\.?0+$/, '')}%`;
    return `${percentage.toFixed(2).replace(/\.?0+$/, '')}%`;
}

export function addTopLogitBars({ activationSource, laneTokenIndices, laneZs, vocabCenter, scene, engine }) {
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
        emissiveIntensity: scaleGlobalEmissiveIntensity(0.2),
        transparent: TOP_LOGIT_BAR_OPACITY < 1,
        opacity: TOP_LOGIT_BAR_OPACITY,
        vertexColors: true
    });

    const barGroup = new THREE.Group();
    barGroup.name = 'TopLogitBars';
    barGroup.visible = false;
    barGroup.userData.revealed = false;
    barGroup.userData.revealComplete = false;
    barGroup.userData.revealCompleteCallbacks = [];

    const colorCache = new Map();
    const instances = [];
    const instanceLabels = [];
    const instanceEntries = [];
    const chosenEntries = [];

    const tokenIndices = Array.isArray(laneTokenIndices)
        ? laneTokenIndices
        : laneZs.map((_, idx) => idx);

    const laneRows = [];
    let globalMaxProb = 0;
    const lastLaneIdx = laneZs.length - 1;

    for (let laneIdx = 0; laneIdx < laneZs.length; laneIdx += 1) {
        const tokenIndex = tokenIndices[laneIdx] ?? laneIdx;
        const logitRow = activationSource.getLogitsForToken(tokenIndex, barCount);
        if (!Array.isArray(logitRow) || !logitRow.length) continue;
        const limit = Math.min(barCount, logitRow.length);
        let bestIdx = -1;
        let bestProb = -Infinity;
        for (let i = 0; i < limit; i += 1) {
            const entry = logitRow[i];
            const prob = Number(entry?.prob);
            if (!Number.isFinite(prob)) continue;
            if (prob > globalMaxProb) globalMaxProb = prob;
            if (prob > bestProb) {
                bestProb = prob;
                bestIdx = i;
            }
        }
        const chosenToken = resolveChosenTokenCandidateForToken(activationSource, tokenIndex, {
            logitLimit: barCount
        });
        const hasVisibleChosenEntry = Number.isFinite(chosenToken?.logitEntryIndex) && chosenToken.logitEntryIndex >= 0;
        const chosenIdx = hasVisibleChosenEntry ? chosenToken.logitEntryIndex : bestIdx;
        const chosenTokenLabel = hasVisibleChosenEntry
            ? formatTokenLabel(
                sanitizeLogitToken(
                    chosenToken?.tokenDisplay
                    || chosenToken?.tokenRaw
                    || ''
                )
            )
            : '';
        laneRows.push({
            laneIdx,
            logitRow,
            bestIdx: chosenIdx,
            chosenTokenLabel,
            useChosenColor: hasVisibleChosenEntry,
            chosenTokenIndex: Number.isFinite(chosenToken?.tokenIndex)
                ? Math.floor(chosenToken.tokenIndex)
                : null
        });
    }

    for (let rowIdx = 0; rowIdx < laneRows.length; rowIdx += 1) {
        const {
            laneIdx,
            logitRow,
            bestIdx,
            chosenTokenIndex,
            chosenTokenLabel,
            useChosenColor
        } = laneRows[rowIdx];
        const laneZ = laneZs[laneIdx] ?? 0;

        for (let i = 0; i < Math.min(barCount, logitRow.length); i += 1) {
            const entry = logitRow[i];
            const prob = resolveChosenLogitProbability(entry);
            if (!Number.isFinite(prob)) continue;
            const height = computeLogitBarHeight(prob, globalMaxProb);
            const seed = resolveLogitTokenSeed(entry, i);
            let barColor = getBrightTokenColor(seed, colorCache);
            const startHeight = Math.max(0.1, TOP_LOGIT_BAR_MIN_HEIGHT * 0.15);
            const xPos = baseX + i * barSpacing;
            if (entry) {
                const isChosenBar = i === bestIdx && laneIdx === lastLaneIdx;
                if (isChosenBar && useChosenColor) {
                    barColor = resolveChosenLogitDisplayColor({
                        activationSource,
                        laneTokenIndices: tokenIndices,
                        chosenEntry: entry,
                        chosenTokenIndex,
                        fallbackIndex: i,
                        fallbackColor: barColor
                    });
                }
                const tokenTextRaw = resolveLogitEntryText(entry);
                const tokenText = tokenTextRaw
                    ? formatTokenLabel(tokenTextRaw.replace(/\n/g, '\\n').replace(/\t/g, '\\t'))
                    : '';
                const tokenId = resolveChosenLogitTokenId(entry);
                const isIncompleteToken = tokenId !== null && isIncompleteUtf8TokenId(tokenId);
                const labelLines = ['Logit'];
                if (tokenText && !isIncompleteToken) labelLines.push(`Token "${tokenText}"`);
                if (tokenId !== null) labelLines.push(`ID ${tokenId}`);
                if (isIncompleteToken) {
                    labelLines.push('Incomplete UTF-8 byte fragment');
                }
                if (Number.isFinite(prob)) labelLines.push(formatHoverProbabilityPercentage(prob));
                const label = labelLines.join('\n');
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
                if (i === bestIdx && laneIdx === lastLaneIdx) {
                    chosenEntries.push({
                        laneIdx,
                        entry,
                        instanceIndex,
                        tokenLabel: chosenTokenLabel,
                        tokenIndex: chosenTokenIndex,
                        useChipColor: useChosenColor,
                        x: xPos,
                        z: laneZ,
                        baseY,
                        targetHeight: height,
                        fallbackColor: barColor.clone(),
                        fallbackIndex: i
                    });
                }
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
                if (i === bestIdx && laneIdx === lastLaneIdx) {
                    chosenEntries.push({
                        laneIdx,
                        entry,
                        instanceIndex,
                        tokenLabel: chosenTokenLabel,
                        tokenIndex: chosenTokenIndex,
                        useChipColor: useChosenColor,
                        x: xPos,
                        z: laneZ,
                        baseY,
                        targetHeight: height,
                        fallbackColor: barColor.clone(),
                        fallbackIndex: i
                    });
                }
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
    barGroup.userData.baseX = baseX;
    barGroup.userData.barSpacing = barSpacing;
    barGroup.userData.barCount = barCount;
    barGroup.userData.activationSource = activationSource;
    barGroup.userData.laneTokenIndices = tokenIndices.slice();
    barGroup.userData.chosenEntries = chosenEntries;
    barGroup.add(instanced);

    scene.add(barGroup);
    if (engine && typeof engine.registerRaycastRoot === 'function') {
        engine.registerRaycastRoot(barGroup);
    }
    ensureChosenLabelGroup(barGroup);
    return barGroup;
}

export function revealTopLogitBars(barGroup, { immediate = false, onComplete = null } = {}) {
    if (!barGroup || !barGroup.userData) return;
    queueRevealCompleteCallback(barGroup, onComplete);
    if (barGroup.userData.revealed) return;
    barGroup.userData.revealed = true;
    barGroup.visible = true;

    const instanced = barGroup.userData.instancedMesh;
    const instances = barGroup.userData.instances;
    if (!instanced || !Array.isArray(instances) || !instances.length) {
        markRevealComplete(barGroup);
        return;
    }

    const barWidth = Number.isFinite(barGroup.userData.barWidth) ? barGroup.userData.barWidth : 1;
    const barDepth = Number.isFinite(barGroup.userData.barDepth) ? barGroup.userData.barDepth : 1;
    const dummy = new THREE.Object3D();
    const finalizeInstancing = () => {
        instanced.frustumCulled = true;
        if (typeof instanced.computeBoundingBox === 'function') instanced.computeBoundingBox();
        if (typeof instanced.computeBoundingSphere === 'function') instanced.computeBoundingSphere();
        instanced.instanceMatrix.setUsage(THREE.StaticDrawUsage);
        revealChosenLabelGroup(barGroup);
        markRevealComplete(barGroup);
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
    const stagger = 0; // rise all top logit prisms at the same time
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
