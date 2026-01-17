import * as THREE from 'three';
import { VECTOR_LENGTH_PRISM } from '../../utils/constants.js';
import { buildActivationData, applyActivationDataToVector } from '../../utils/activationMetadata.js';
import { perfStats } from '../../utils/perfStats.js';
import { updateSciFiMaterialColor } from '../../utils/sciFiMaterial.js';

const LN_MATERIAL_EPSILON = 1e-4;

// Minimum segment length for LN-internal trails to reduce overdraw.
export const LN_INTERNAL_TRAIL_MIN_SEGMENT = 0.15;

// GPT-2 BPE uses a leading U+0120 to indicate a space; render as a normal space.
export function formatTokenLabel(token) {
    if (!token) return null;
    return token.replace(/^\u0120/, ' ');
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
                    updateSciFiMaterialColor(mat, targetColor);
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
                updateSciFiMaterialColor(mat, targetColor);
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
