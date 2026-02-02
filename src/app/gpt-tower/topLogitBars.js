import * as THREE from 'three';
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
import { formatTokenLabel } from './tokenLabels.js';

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
