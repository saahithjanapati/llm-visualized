import * as THREE from 'three';
import { VECTOR_LENGTH_PRISM } from '../../utils/constants.js';
import { buildActivationData, applyActivationDataToVector } from '../../utils/activationMetadata.js';
import { perfStats } from '../../utils/perfStats.js';
import { updateSciFiMaterialColor } from '../../utils/sciFiMaterial.js';

const LN_MATERIAL_EPSILON = 1e-4;

// Minimum segment length for LN-internal trails to reduce overdraw.
export const LN_INTERNAL_TRAIL_MIN_SEGMENT = 0.15;

// GPT-2 BPE uses a leading U+0120 to indicate a space; render pure-space tokens visibly.
const SPACE_TOKEN_DISPLAY = '" "';

export function formatTokenLabel(token) {
    if (token === null || token === undefined) return null;
    const raw = String(token);
    const normalized = raw.replace(/^\u0120+/, (match) => ' '.repeat(match.length));
    if (!normalized.length) return SPACE_TOKEN_DISPLAY;
    if (normalized.trim().length === 0) return SPACE_TOKEN_DISPLAY;
    return normalized;
}

function getKeyColorCount(values) {
    const length = Array.isArray(values) ? values.length : 0;
    return Math.min(30, Math.max(1, length || 1));
}

export function applyVectorData(vec, values, label, meta, colorOptions = null) {
    const isArrayLike = Array.isArray(values) || ArrayBuffer.isView(values);
    if (!vec || !isArrayLike || values.length === 0) return false;
    if (perfStats.enabled) {
        perfStats.inc('vectorUpdates');
    }
    vec.rawData = values.slice();
    const numKeyColors = getKeyColorCount(vec.rawData);
    if (colorOptions) {
        vec.updateKeyColorsFromData(vec.rawData, numKeyColors, colorOptions, values);
    } else {
        vec.updateKeyColorsFromData(vec.rawData, numKeyColors, null, values);
    }
    const activationData = buildActivationData({
        label,
        values,
        copyValues: false,
        ...meta
    });
    applyActivationDataToVector(vec, activationData, label);
    return true;
}

export function copyVectorAppearance(targetVec, sourceVec, fallbackLabel = null, fallbackMeta = null) {
    if (!targetVec || !sourceVec) return false;
    const activation = sourceVec.userData && sourceVec.userData.activationData;
    const label = (sourceVec.group && sourceVec.group.userData && sourceVec.group.userData.label) || fallbackLabel;
    const meta = activation
        ? {
            stage: activation.stage,
            layerIndex: activation.layerIndex,
            tokenIndex: activation.tokenIndex,
            tokenLabel: activation.tokenLabel,
            headIndex: activation.headIndex,
            keyTokenIndex: activation.keyTokenIndex,
            keyTokenLabel: activation.keyTokenLabel,
            segmentIndex: activation.segmentIndex,
            preScore: activation.preScore,
            postScore: activation.postScore,
            notes: activation.notes
        }
        : fallbackMeta;
    const values = (activation && (Array.isArray(activation.values) || ArrayBuffer.isView(activation.values)))
        ? activation.values
        : sourceVec.rawData;
    return applyVectorData(targetVec, values, label, meta || null);
}

export function cloneVectorKeyColors(vec) {
    const keyColors = vec && Array.isArray(vec.currentKeyColors) ? vec.currentKeyColors : null;
    if (!keyColors || !keyColors.length) return [];
    return keyColors.map((color) => (color && typeof color.clone === 'function')
        ? color.clone()
        : new THREE.Color(0.5, 0.5, 0.5));
}

export function blendVectorKeyColors(vec, baseColors, tintColor, tintAlpha = 0) {
    if (!vec || typeof vec._buildColorBuffers !== 'function' || typeof vec._applyColorBuffers !== 'function') {
        return false;
    }
    if (!Array.isArray(baseColors) || !baseColors.length) return false;

    const safeAlpha = THREE.MathUtils.clamp(Number.isFinite(tintAlpha) ? tintAlpha : 0, 0, 1);
    const resolvedTint = (tintColor && tintColor.isColor) ? tintColor : new THREE.Color(0xffffff);
    if (!Array.isArray(vec.currentKeyColors) || vec.currentKeyColors.length !== baseColors.length) {
        vec.currentKeyColors = baseColors.map((color) => color.clone());
    }

    for (let i = 0; i < baseColors.length; i++) {
        const baseColor = baseColors[i];
        if (!baseColor) continue;
        if (!(vec.currentKeyColors[i] instanceof THREE.Color)) {
            vec.currentKeyColors[i] = baseColor.clone();
        }
        vec.currentKeyColors[i].copy(baseColor).lerp(resolvedTint, safeAlpha);
    }

    const buffers = vec._buildColorBuffers();
    vec._applyColorBuffers(buffers);
    return true;
}

export function applyVectorKeyColors(vec, keyColors) {
    if (!vec || !Array.isArray(keyColors) || !keyColors.length) return false;
    const colors = keyColors
        .map((color) => {
            if (color && color.isColor) return color.clone();
            try {
                return new THREE.Color(color);
            } catch (_) {
                return null;
            }
        })
        .filter(Boolean);
    if (!colors.length) return false;

    vec.currentKeyColors = colors;
    vec.numSubsections = Math.max(0, colors.length - 1);

    if (typeof vec._buildColorBuffers === 'function' && typeof vec._applyColorBuffers === 'function') {
        vec._applyColorBuffers(vec._buildColorBuffers());
        return true;
    }
    if (typeof vec.updateInstanceGeometryAndColors === 'function') {
        vec.updateInstanceGeometryAndColors();
        return true;
    }
    return false;
}

export function interpolateVectorKeyColors(vec, fromColors, toColors, progress) {
    if (!Array.isArray(fromColors) || !fromColors.length || !Array.isArray(toColors) || !toColors.length) {
        return false;
    }
    const targetCount = Math.max(fromColors.length, toColors.length);
    const clampedProgress = THREE.MathUtils.clamp(Number.isFinite(progress) ? progress : 0, 0, 1);
    const sampleColor = (colors, index, scratch) => {
        if (colors.length === 1 || targetCount <= 1) {
            return scratch.copy(colors[0]);
        }
        const sourceT = index / (targetCount - 1);
        const sourceIndex = sourceT * (colors.length - 1);
        const low = Math.floor(sourceIndex);
        const high = Math.min(colors.length - 1, low + 1);
        const localT = sourceIndex - low;
        return scratch.copy(colors[low]).lerp(colors[high], localT);
    };
    const blended = [];
    const fromScratch = new THREE.Color();
    const toScratch = new THREE.Color();
    for (let i = 0; i < targetCount; i++) {
        const from = sampleColor(fromColors, i, fromScratch);
        const to = sampleColor(toColors, i, toScratch);
        blended.push(from.clone().lerp(to, clampedProgress));
    }
    return applyVectorKeyColors(vec, blended);
}

export function setObjectTreeOpacity(root, opacity) {
    if (!root || typeof root.traverse !== 'function') return false;
    const clampedOpacity = THREE.MathUtils.clamp(Number.isFinite(opacity) ? opacity : 1, 0, 1);
    const shouldBeTransparent = clampedOpacity < 0.999;
    let changed = false;
    root.traverse((obj) => {
        if (!obj || !obj.isMesh || !obj.material) return;
        const mats = Array.isArray(obj.material) ? obj.material : [obj.material];
        mats.forEach((mat) => {
            if (!mat) return;
            if (mat.opacity !== clampedOpacity) {
                mat.opacity = clampedOpacity;
                changed = true;
            }
            if (mat.transparent !== shouldBeTransparent) {
                mat.transparent = shouldBeTransparent;
                mat.needsUpdate = true;
                changed = true;
            }
            const nextDepthWrite = !shouldBeTransparent;
            if (mat.depthWrite !== nextDepthWrite) {
                mat.depthWrite = nextDepthWrite;
                mat.needsUpdate = true;
                changed = true;
            }
        });
    });
    return changed;
}

export function geluApprox(x) {
    const coeff = Math.sqrt(2 / Math.PI);
    return 0.5 * x * (1 + Math.tanh(coeff * (x + 0.044715 * Math.pow(x, 3))));
}

export function freezeStaticTransforms(object3d, includeChildren = false) {
    if (!object3d) return;
    object3d.matrixAutoUpdate = false;
    object3d.updateMatrix();
    if (!includeChildren || typeof object3d.traverse !== 'function') return;
    object3d.traverse(child => {
        if (child === object3d) return;
        child.matrixAutoUpdate = false;
        child.updateMatrix();
    });
}

export function applyLayerNormMaterial(group, targetColor, targetOpacity, state) {
    if (!group || !state) return;
    const transparent = targetOpacity < 1.0;
    const hasState = state.initialized === true;
    const colorChanged = !hasState || !state.color.equals(targetColor);
    const opacityChanged = !hasState || Math.abs((state.opacity ?? 0) - targetOpacity) > LN_MATERIAL_EPSILON;
    const transparentChanged = !hasState || state.transparent !== transparent;

    if (!colorChanged && !opacityChanged && !transparentChanged) return;

    state.color.copy(targetColor);
    state.opacity = targetOpacity;
    state.transparent = transparent;
    state.initialized = true;

    const children = group.children || [];
    for (let i = 0; i < children.length; i++) {
        const child = children[i];
        if (!(child instanceof THREE.Mesh) || !child.material) continue;
        if (Array.isArray(child.material)) {
            child.material.forEach(mat => {
                if (colorChanged) {
                    const prevEmissiveIntensity = (typeof mat.emissiveIntensity === 'number')
                        ? mat.emissiveIntensity
                        : null;
                    updateSciFiMaterialColor(mat, targetColor);
                    // Preserve LN emissive level; LN color transitions should not
                    // implicitly boost bloom intensity.
                    if (prevEmissiveIntensity !== null) {
                        mat.emissiveIntensity = prevEmissiveIntensity;
                    }
                }
                if (opacityChanged) {
                    mat.opacity = targetOpacity;
                }
                if (transparentChanged) {
                    mat.transparent = transparent;
                    mat.needsUpdate = true;
                }
            });
        } else {
            const mat = child.material;
            if (colorChanged) {
                const prevEmissiveIntensity = (typeof mat.emissiveIntensity === 'number')
                    ? mat.emissiveIntensity
                    : null;
                updateSciFiMaterialColor(mat, targetColor);
                // Preserve LN emissive level; LN color transitions should not
                // implicitly boost bloom intensity.
                if (prevEmissiveIntensity !== null) {
                    mat.emissiveIntensity = prevEmissiveIntensity;
                }
            }
            if (opacityChanged) {
                mat.opacity = targetOpacity;
            }
            if (transparentChanged) {
                mat.transparent = transparent;
                mat.needsUpdate = true;
            }
        }
    }
}

export function simplePrismMultiply(srcVec, tgtVec, onComplete) {
    const srcCount = srcVec && Number.isFinite(srcVec.instanceCount) ? srcVec.instanceCount : VECTOR_LENGTH_PRISM;
    const tgtCount = tgtVec && Number.isFinite(tgtVec.instanceCount) ? tgtVec.instanceCount : VECTOR_LENGTH_PRISM;
    const srcLen = srcVec && Array.isArray(srcVec.rawData) ? srcVec.rawData.length : srcCount;
    const tgtLen = tgtVec && Array.isArray(tgtVec.rawData) ? tgtVec.rawData.length : tgtCount;
    const length = Math.min(srcCount, tgtCount, srcLen, tgtLen);
    for (let i = 0; i < length; i++) {
        tgtVec.rawData[i] = (srcVec.rawData[i] || 0) * (tgtVec.rawData[i] || 0);
    }
    const numKeyColors = Math.min(30, Math.max(1, tgtVec.rawData.length || 1));
    tgtVec.updateKeyColorsFromData(tgtVec.rawData, numKeyColors);
    if (onComplete) onComplete();
}
