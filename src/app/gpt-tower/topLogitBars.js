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
import { scaleGlobalEmissiveIntensity } from '../../utils/materialUtils.js';
import { isIncompleteUtf8TokenId } from '../../utils/tokenEncodingNotes.js';
import { formatTokenLabel } from './tokenLabels.js';
import { getLogitTokenColorUnit, resolveLogitTokenSeed } from './logitColor.js';

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
    if (typeof entry.token === 'string') {
        return formatTokenLabel(sanitizeLogitToken(entry.token));
    }
    if (Number.isFinite(entry.token_id)) {
        return `#${Math.floor(entry.token_id)}`;
    }
    return '';
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

function buildChosenLogitLabelGroup(barGroup, font) {
    const chosenEntries = barGroup?.userData?.chosenEntries;
    if (!Array.isArray(chosenEntries) || !chosenEntries.length) return null;
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

    chosenEntries.forEach((chosen) => {
        const labelText = resolveLogitLabelText(chosen.entry);
        if (!labelText) return;
        const textGroup = createExtrudedTextGroup(labelText, font, {
            size: textSize,
            depth: baseTextDepth,
            color: chosen.color
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
        textGroup.userData.label = `Chosen token: ${labelText}`;
        textGroup.name = `Chosen token: ${labelText}`;

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
            color: chosen.color || 0xffffff,
            transparent: true,
            opacity: LOGIT_LABEL_LINE_OPACITY
        });
        const line = new THREE.Line(lineGeometry, lineMaterial);
        line.userData.label = `Chosen token: ${labelText}`;

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
    const labelGroup = barGroup.userData.chosenLabelGroup;
    if (labelGroup) labelGroup.visible = true;
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
    const tokenCount = typeof activationSource.getTokenCount === 'function'
        ? activationSource.getTokenCount()
        : 0;
    const lastLaneIdx = laneZs.length - 1;

    for (let laneIdx = 0; laneIdx < laneZs.length; laneIdx += 1) {
        const tokenIndex = tokenIndices[laneIdx] ?? laneIdx;
        const logitRow = activationSource.getLogitsForToken(tokenIndex, barCount);
        if (!Array.isArray(logitRow) || !logitRow.length) continue;
        const limit = Math.min(barCount, logitRow.length);
        let bestIdx = -1;
        let bestProb = -Infinity;
        let pickedIdx = -1;
        const nextTokenRaw = (Number.isFinite(tokenCount) && tokenIndex + 1 < tokenCount)
            ? activationSource.getTokenString(tokenIndex + 1)
            : null;
        for (let i = 0; i < limit; i += 1) {
            const entry = logitRow[i];
            const prob = Number(entry?.prob);
            if (!Number.isFinite(prob)) continue;
            if (prob > globalMaxProb) globalMaxProb = prob;
            if (prob > bestProb) {
                bestProb = prob;
                bestIdx = i;
            }
            if (pickedIdx === -1 && nextTokenRaw && typeof entry?.token === 'string' && entry.token === nextTokenRaw) {
                pickedIdx = i;
            }
        }
        const chosenIdx = pickedIdx !== -1 ? pickedIdx : bestIdx;
        laneRows.push({ laneIdx, logitRow, bestIdx: chosenIdx });
    }

    for (let rowIdx = 0; rowIdx < laneRows.length; rowIdx += 1) {
        const { laneIdx, logitRow, bestIdx } = laneRows[rowIdx];
        const laneZ = laneZs[laneIdx] ?? 0;

        for (let i = 0; i < Math.min(barCount, logitRow.length); i += 1) {
            const entry = logitRow[i];
            const prob = Number(entry?.prob);
            if (!Number.isFinite(prob)) continue;
            const height = computeLogitBarHeight(prob, globalMaxProb);
            const seed = resolveLogitTokenSeed(entry, i);
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
                if (tokenId !== null && isIncompleteUtf8TokenId(tokenId)) {
                    labelParts.push('incomplete UTF-8 byte fragment');
                }
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
                if (i === bestIdx && laneIdx === lastLaneIdx) {
                    chosenEntries.push({
                        laneIdx,
                        entry,
                        x: xPos,
                        z: laneZ,
                        baseY,
                        targetHeight: height,
                        color: barColor
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
                        x: xPos,
                        z: laneZ,
                        baseY,
                        targetHeight: height,
                        color: barColor
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
    barGroup.userData.chosenEntries = chosenEntries;
    barGroup.add(instanced);

    scene.add(barGroup);
    if (engine && typeof engine.registerRaycastRoot === 'function') {
        engine.registerRaycastRoot(barGroup);
    }
    ensureChosenLabelGroup(barGroup);
    return barGroup;
}

export function revealTopLogitBars(barGroup, { immediate = false } = {}) {
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
        revealChosenLabelGroup(barGroup);
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
