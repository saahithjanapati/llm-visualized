import * as THREE from 'three';
import { FontLoader } from 'three/examples/jsm/loaders/FontLoader.js';
import { WeightMatrixVisualization } from '../components/WeightMatrixVisualization.js';
import { VectorVisualizationInstancedPrism } from '../components/VectorVisualizationInstancedPrism.js';
import { LayerNormalizationVisualization } from '../components/LayerNormalizationVisualization.js';
import { appState } from '../state/appState.js';
import { createSciFiMaterial, updateSciFiMaterialColor } from '../utils/sciFiMaterial.js';
import { mapValueToColor, mapValueToGrayscale } from '../utils/colors.js';
import {
    MHA_MATRIX_PARAMS,
    MLP_MATRIX_PARAMS_UP,
    MLP_MATRIX_PARAMS_DOWN,
    EMBEDDING_MATRIX_PARAMS_VOCAB,
    EMBEDDING_MATRIX_PARAMS_POSITION,
    LAYER_NORM_FINAL_COLOR,
    VECTOR_LENGTH_PRISM,
    HIDE_INSTANCE_Y_OFFSET
} from '../utils/constants.js';
import {
    MHA_FINAL_Q_COLOR,
    MHA_FINAL_K_COLOR,
    MHA_FINAL_V_COLOR,
    MHA_OUTPUT_PROJECTION_MATRIX_COLOR,
    MHA_OUTPUT_PROJECTION_MATRIX_PARAMS,
    LN_PARAMS
} from '../animations/LayerAnimationConstants.js';

const PREVIEW_LANES = 3;
const PREVIEW_SOLID_LANES = 5;
const PREVIEW_TOKEN_LANES = 1;
const PREVIEW_MATRIX_DEPTH = 320;
const PREVIEW_LANE_SPACING = 80;
const PREVIEW_TARGET_SIZE = 140;
// Base framing for most objects; vector previews can request additional padding.
const PREVIEW_FRAME_PADDING = 1.25;
const PREVIEW_BASE_DISTANCE_MULT = 1.15;
const PREVIEW_VECTOR_PADDING_MULT = 3.0;
const PREVIEW_VECTOR_DISTANCE_MULT = 2.4;
const PREVIEW_ROTATION_SPEED = 0.0035;
const PREVIEW_BASE_TILT_X = -0.12;
const PREVIEW_BASE_ROTATION_Y = 0.38;
const PREVIEW_TILT_AMPLITUDE = 0.02;
const PREVIEW_TILT_OSC_SPEED = 0.32;
const FINAL_MLP_COLOR = 0xc07a12;
const FINAL_VOCAB_TOP_COLOR = 0x000000;
const PREVIEW_QKV_LANES = 3;
const PREVIEW_QKV_LANE_SPACING = 360;
const PREVIEW_VECTOR_LARGE_SCALE = 1.0;
const PREVIEW_VECTOR_SMALL_SCALE = 0.38;
const PREVIEW_TRAIL_COLOR = 0x6ea0ff;
const PREVIEW_QKV_X_SPREAD = 72;
const PREVIEW_QKV_START_Y = -150;
const PREVIEW_QKV_MATRIX_Y = -15;
const PREVIEW_QKV_OUTPUT_Y = 95;
const PREVIEW_QKV_EXIT_Y = PREVIEW_QKV_OUTPUT_Y + 60;
const PREVIEW_QKV_RISE_DURATION = 900;
const PREVIEW_QKV_CONVERT_DURATION = 420;
const PREVIEW_QKV_HOLD_DURATION = 520;
const PREVIEW_QKV_EXIT_DURATION = 420;
const PREVIEW_QKV_IDLE_DURATION = 260;
const PREVIEW_QKV_LANE_STAGGER = 0;
const PREVIEW_VECTOR_HEAD_INSTANCES = 1;
const PREVIEW_VECTOR_BODY_INSTANCES = VECTOR_LENGTH_PRISM;
const ATTENTION_PREVIEW_MAX_TOKENS = 16;
const ATTENTION_PREVIEW_TARGET_PX = 320;
const ATTENTION_PREVIEW_MIN_CELL = 4;
const ATTENTION_PREVIEW_MAX_CELL = 24;
const ATTENTION_PREVIEW_GAP = 4;
const ATTENTION_PREVIEW_TRIANGLE = 'lower';
const ATTENTION_PREVIEW_GRID_GAP = 8; // matches .detail-attention-grid column gap in CSS
const SPACE_TOKEN_DISPLAY = '" "';

const TOKEN_CHIP_STYLE = {
    padding: 80,
    minWidth: 220,
    minHeight: 100,
    height: 120,
    cornerRadius: 18,
    depth: 12,
    textDepth: 16,
    textSize: 52,
    textOffset: 1.2
};

const TOKEN_CHIP_FONT_URL = 'https://threejs.org/examples/fonts/helvetiker_regular.typeface.json';
let tokenChipFont = null;
let tokenChipFontPromise = null;

function requestTokenChipFont() {
    if (tokenChipFont) return Promise.resolve(tokenChipFont);
    if (tokenChipFontPromise) return tokenChipFontPromise;
    const loader = new FontLoader();
    tokenChipFontPromise = new Promise((resolve) => {
        loader.load(
            TOKEN_CHIP_FONT_URL,
            (font) => {
                tokenChipFont = font;
                resolve(font);
            },
            undefined,
            (err) => {
                console.warn('Selection token font failed to load, falling back to canvas text.', err);
                resolve(null);
            }
        );
    });
    return tokenChipFontPromise;
}

const D_MODEL = 768;
const VOCAB_SIZE = 50257;
const CONTEXT_LEN = 1024;

function formatNumber(value) {
    if (!Number.isFinite(value)) return 'TBD';
    return Math.round(value).toLocaleString('en-US');
}

function formatDims(rows, cols) {
    if (!Number.isFinite(rows) || !Number.isFinite(cols)) return 'TBD';
    return `${rows} x ${cols}`;
}

function formatValues(values, perLine = 8) {
    if (!values || typeof values.length !== 'number' || values.length === 0) return '(empty)';
    let result = '';
    for (let idx = 0; idx < values.length; idx += 1) {
        const num = Number(values[idx]);
        const formatted = Number.isFinite(num) ? num.toFixed(4) : '0.0000';
        const sep = idx === 0 ? '' : idx % perLine === 0 ? '\n' : ', ';
        result += sep + formatted;
    }
    return result;
}

function formatActivationData(data) {
    if (!data || typeof data !== 'object') return 'No activation data.';
    const lines = [];
    if (data.stage) lines.push(`Stage: ${data.stage}`);
    if (Number.isFinite(data.layerIndex)) lines.push(`Layer: ${data.layerIndex + 1}`);
    if (Number.isFinite(data.tokenIndex)) {
        const tokenText = data.tokenLabel ? ` (${data.tokenLabel})` : '';
        lines.push(`Token: ${data.tokenIndex + 1}${tokenText}`);
    }
    if (Number.isFinite(data.keyTokenIndex)) {
        const keyText = data.keyTokenLabel ? ` (${data.keyTokenLabel})` : '';
        lines.push(`Key: ${data.keyTokenIndex + 1}${keyText}`);
    }
    if (Number.isFinite(data.headIndex)) lines.push(`Head: ${data.headIndex + 1}`);
    if (Number.isFinite(data.segmentIndex)) lines.push(`Segment: ${data.segmentIndex + 1}`);
    if (Number.isFinite(data.preScore) || Number.isFinite(data.postScore)) {
        if (Number.isFinite(data.preScore)) lines.push(`Pre-softmax: ${data.preScore.toFixed(4)}`);
        if (Number.isFinite(data.postScore)) lines.push(`Post-softmax: ${data.postScore.toFixed(4)}`);
    }
    if (data.values && typeof data.values.length === 'number') {
        lines.push(`Values (${data.values.length}):`);
        lines.push(formatValues(data.values));
    }
    if (data.notes) lines.push(String(data.notes));
    return lines.join('\n');
}

function colorToCss(color) {
    if (!color) return 'transparent';
    const target = color.isColor ? color : new THREE.Color(color);
    return `#${target.getHexString()}`;
}

function formatTokenLabelForPreview(label) {
    if (typeof label !== 'string') return '';
    const normalized = label.replace(/^\u0120+/, (match) => ' '.repeat(match.length));
    if (!normalized.length) return SPACE_TOKEN_DISPLAY;
    const collapsed = normalized.replace(/\s+/g, ' ');
    const trimmed = collapsed.trim();
    return trimmed.length ? trimmed : SPACE_TOKEN_DISPLAY;
}

function getActivationDataFromSelection(selectionInfo) {
    return selectionInfo?.info?.activationData
        || selectionInfo?.object?.userData?.activationData
        || selectionInfo?.hit?.object?.userData?.activationData
        || null;
}

function findUserDataNumber(selectionInfo, key) {
    const direct = selectionInfo?.info?.[key];
    if (Number.isFinite(direct)) return direct;
    const infoActivation = selectionInfo?.info?.activationData?.[key];
    if (Number.isFinite(infoActivation)) return infoActivation;
    const candidates = [selectionInfo?.object, selectionInfo?.hit?.object];
    for (const obj of candidates) {
        let current = obj;
        while (current && !current.isScene) {
            const ud = current.userData;
            if (ud && Number.isFinite(ud[key])) return ud[key];
            if (ud?.activationData && Number.isFinite(ud.activationData[key])) return ud.activationData[key];
            current = current.parent;
        }
    }
    return null;
}

function resolveAttentionModeFromSelection(selectionInfo) {
    const stage = getActivationDataFromSelection(selectionInfo)?.stage;
    if (stage === 'attention.post') return 'post';
    if (stage === 'attention.pre') return 'pre';
    return null;
}

function isSelfAttentionSelection(label, selectionInfo) {
    const lower = (label || '').toLowerCase();
    if (isAttentionScoreSelection(label, selectionInfo)) return true;
    if (selectionInfo?.kind === 'mergedKV') return true;
    if (lower.includes('query vector') || lower.includes('key vector') || lower.includes('value vector')) return true;
    if (lower.includes('query weight matrix') || lower.includes('key weight matrix') || lower.includes('value weight matrix')) return true;
    if (lower.includes('merged key vectors') || lower.includes('merged value vectors')) return true;
    const stage = getActivationDataFromSelection(selectionInfo)?.stage;
    if (stage && (stage.startsWith('attention.') || stage.startsWith('qkv.'))) return true;
    return false;
}

function computeAttentionCellSize(count) {
    const safeCount = Math.max(1, Math.floor(count || 1));
    const size = ATTENTION_PREVIEW_TARGET_PX / safeCount;
    return Math.max(ATTENTION_PREVIEW_MIN_CELL, Math.min(ATTENTION_PREVIEW_MAX_CELL, size));
}

function getContentWidth(el) {
    if (!el || typeof window === 'undefined') return 0;
    const style = window.getComputedStyle(el);
    const paddingLeft = parseFloat(style.paddingLeft) || 0;
    const paddingRight = parseFloat(style.paddingRight) || 0;
    const width = (el.clientWidth || 0) - paddingLeft - paddingRight;
    return Math.max(0, width);
}

function measureMaxTokenLabelWidth(labels, referenceEl) {
    if (!Array.isArray(labels) || labels.length === 0 || !referenceEl || typeof window === 'undefined') return 0;
    const style = window.getComputedStyle(referenceEl);
    const fontStyle = style.fontStyle || 'normal';
    const fontWeight = style.fontWeight || '400';
    const fontSize = style.fontSize || '10px';
    const fontFamily = style.fontFamily || 'monospace';
    const font = `${fontStyle} ${fontWeight} ${fontSize} ${fontFamily}`;
    const canvas = measureMaxTokenLabelWidth._canvas || (measureMaxTokenLabelWidth._canvas = document.createElement('canvas'));
    const ctx = canvas.getContext('2d');
    if (!ctx) return 0;
    ctx.font = font;
    let maxWidth = 0;
    for (let i = 0; i < labels.length; i += 1) {
        const text = typeof labels[i] === 'string' ? labels[i] : String(labels[i] ?? '');
        const width = ctx.measureText(text).width;
        if (width > maxWidth) maxWidth = width;
    }
    const padding = 4; // subtle breathing room; padding is handled separately
    return maxWidth + padding;
}

function buildAttentionMatrixValues({ activationSource, layerIndex, headIndex, tokenIndices, mode }) {
    if (!activationSource || !Array.isArray(tokenIndices) || !tokenIndices.length) return null;
    const values = [];
    for (let i = 0; i < tokenIndices.length; i += 1) {
        const queryTokenIndex = tokenIndices[i];
        const row = activationSource.getAttentionScoresRow
            ? activationSource.getAttentionScoresRow(layerIndex, mode, headIndex, queryTokenIndex)
            : null;
        const rowValues = [];
        for (let j = 0; j < tokenIndices.length; j += 1) {
            const keyTokenIndex = tokenIndices[j];
            const value = Array.isArray(row) ? row[keyTokenIndex] : null;
            rowValues.push(Number.isFinite(value) ? value : null);
        }
        values.push(rowValues);
    }
    return values;
}

function clamp01(value) {
    if (!Number.isFinite(value)) return 0;
    return Math.min(1, Math.max(0, value));
}

function easeInOutCubic(t) {
    const clamped = clamp01(t);
    return clamped < 0.5
        ? 4 * clamped * clamped * clamped
        : 1 - Math.pow(-2 * clamped + 2, 3) / 2;
}

function inferQkvType(label, selectionInfo) {
    const lower = (label || '').toLowerCase();
    if (selectionInfo?.info?.category === 'V') return 'V';
    if (selectionInfo?.info?.category === 'Q') return 'Q';
    if (selectionInfo?.info?.category === 'K') return 'K';
    if (lower.includes('value')) return 'V';
    if (lower.includes('query')) return 'Q';
    if (lower.includes('key')) return 'K';
    if (selectionInfo?.kind === 'mergedKV') return 'K';
    return 'K';
}

const TMP_BOX = new THREE.Box3();
const TMP_MATRIX = new THREE.Matrix4();
const TMP_POS = new THREE.Vector3();
const TMP_QUAT = new THREE.Quaternion();
const TMP_SCALE = new THREE.Vector3();

function isWeightMatrixLabel(label) {
    const lower = (label || '').toLowerCase();
    return lower.includes('weight matrix')
        || lower.includes('embedding')
        || lower.includes('output projection matrix');
}

function isQkvMatrixLabel(label) {
    const lower = (label || '').toLowerCase();
    return lower.includes('query weight matrix')
        || lower.includes('key weight matrix')
        || lower.includes('value weight matrix');
}

function isParameterSelection(label) {
    const lower = (label || '').toLowerCase();
    if (isWeightMatrixLabel(lower)) return true;
    if (lower.includes('layernorm') || lower.includes('layer norm')) {
        if (lower.includes('scale') || lower.includes('shift') || lower.includes('gamma') || lower.includes('beta')) {
            return true;
        }
    }
    return false;
}

function isAttentionScoreSelection(label, selectionInfo) {
    const lower = (label || '').toLowerCase();
    if (lower.includes('attention score')) return true;
    const stage = selectionInfo?.info?.activationData?.stage
        || selectionInfo?.object?.userData?.activationData?.stage
        || selectionInfo?.hit?.object?.userData?.activationData?.stage;
    if (typeof stage === 'string' && stage.startsWith('attention.')) return true;
    const obj = selectionInfo?.object || selectionInfo?.hit?.object;
    return !!(obj && obj.isMesh && obj.geometry && obj.geometry.type === 'SphereGeometry');
}

function resolveFinalPreviewColor(label) {
    const lower = (label || '').toLowerCase();
    if (lower.includes('query weight matrix')) return MHA_FINAL_Q_COLOR;
    if (lower.includes('key weight matrix')) return MHA_FINAL_K_COLOR;
    if (lower.includes('value weight matrix')) return MHA_FINAL_V_COLOR;
    if (lower.includes('output projection matrix')) return MHA_OUTPUT_PROJECTION_MATRIX_COLOR;
    if (lower.includes('mlp up weight matrix')) return FINAL_MLP_COLOR;
    if (lower.includes('mlp down weight matrix')) return FINAL_MLP_COLOR;
    if (lower.includes('vocab embedding')) {
        return lower.includes('top') ? FINAL_VOCAB_TOP_COLOR : MHA_FINAL_Q_COLOR;
    }
    if (lower.includes('positional embedding')) return MHA_FINAL_K_COLOR;
    if (lower.includes('layernorm') || lower.includes('layer norm')) return LAYER_NORM_FINAL_COLOR;
    return null;
}

function clonePreviewMaterial(material) {
    if (!material) return material;
    const uniforms = material.userData?.sciFiUniforms;
    if (uniforms) {
        const clone = createSciFiMaterial({
            baseColor: material.color,
            accentColor: uniforms.uAccentColor?.value,
            secondaryColor: uniforms.uSecondaryColor?.value,
            edgeColor: uniforms.uEdgeColor?.value,
            emissiveColor: material.emissive,
            emissiveIntensity: material.emissiveIntensity,
            metalness: material.metalness,
            roughness: material.roughness,
            clearcoat: material.clearcoat,
            clearcoatRoughness: material.clearcoatRoughness,
            transmission: material.transmission,
            thickness: material.thickness,
            iridescence: material.iridescence,
            iridescenceIOR: material.iridescenceIOR,
            sheen: material.sheen,
            sheenColor: material.sheenColor,
            sheenRoughness: material.sheenRoughness,
            envMapIntensity: material.envMapIntensity,
            transparent: material.transparent,
            opacity: material.opacity,
            side: material.side,
            dimensions: uniforms.uDimensions?.value,
            stripeFrequency: uniforms.uStripeFrequency?.value,
            stripeStrength: uniforms.uStripeStrength?.value,
            rimIntensity: uniforms.uRimIntensity?.value,
            gradientSharpness: uniforms.uGradientSharpness?.value,
            gradientBias: uniforms.uGradientBias?.value,
            fresnelBoost: uniforms.uFresnelBoost?.value,
            accentMix: uniforms.uAccentMix?.value,
            glowFalloff: uniforms.uGlowFalloff?.value,
            depthAccentStrength: uniforms.uDepthAccentStrength?.value,
            scanlineFrequency: uniforms.uScanlineFrequency?.value,
            scanlineStrength: uniforms.uScanlineStrength?.value,
            glintStrength: uniforms.uGlintStrength?.value,
            noiseStrength: uniforms.uNoiseStrength?.value
        });
        clone.polygonOffset = material.polygonOffset;
        clone.polygonOffsetFactor = material.polygonOffsetFactor;
        clone.polygonOffsetUnits = material.polygonOffsetUnits;
        clone.depthWrite = material.depthWrite;
        clone.depthTest = material.depthTest;
        clone.alphaTest = material.alphaTest;
        clone.colorWrite = material.colorWrite;
        clone.toneMapped = material.toneMapped;
        clone.visible = material.visible;
        clone.envMap = material.envMap;
        clone.map = material.map;
        clone.normalMap = material.normalMap;
        clone.roughnessMap = material.roughnessMap;
        clone.metalnessMap = material.metalnessMap;
        clone.emissiveMap = material.emissiveMap;
        clone.alphaMap = material.alphaMap;
        return clone;
    }
    const clone = material.clone();
    if (material.onBeforeCompile) clone.onBeforeCompile = material.onBeforeCompile;
    if (material.customProgramCacheKey) clone.customProgramCacheKey = material.customProgramCacheKey;
    return clone;
}

function cloneMaterialsForPreview(object) {
    const materials = [];
    object.traverse((child) => {
        if (!child.material) return;
        if (Array.isArray(child.material)) {
            const cloned = child.material.map((mat) => clonePreviewMaterial(mat));
            child.material = cloned;
            materials.push(...cloned.filter(Boolean));
            return;
        }
        const clone = clonePreviewMaterial(child.material);
        child.material = clone;
        if (clone) materials.push(clone);
    });
    return materials;
}

function cloneGeometriesForPreview(object) {
    const geometries = [];
    object.traverse((child) => {
        if (!child.geometry || typeof child.geometry.clone !== 'function') return;
        const clonedGeo = child.geometry.clone();
        child.geometry = clonedGeo;
        geometries.push(clonedGeo);

        if (child.isInstancedMesh) {
            if (child.instanceMatrix && typeof child.instanceMatrix.clone === 'function') {
                child.instanceMatrix = child.instanceMatrix.clone();
                child.instanceMatrix.needsUpdate = true;
            }
            if (child.instanceColor && typeof child.instanceColor.clone === 'function') {
                child.instanceColor = child.instanceColor.clone();
                child.instanceColor.needsUpdate = true;
            }
        }
    });
    return geometries;
}

function applyLaneOverrideToInstancedMeshes(object, laneCount, laneSpacing) {
    if (!object || !Number.isFinite(laneCount) || laneCount < 1) return;
    const spacing = Number.isFinite(laneSpacing) ? laneSpacing : PREVIEW_QKV_LANE_SPACING;
    const mtx = new THREE.Matrix4();
    object.traverse((child) => {
        if (!child?.isInstancedMesh) return;
        child.count = laneCount;
        for (let i = 0; i < laneCount; i++) {
            const z = (i - (laneCount - 1) / 2) * spacing;
            mtx.makeTranslation(0, 0, z);
            child.setMatrixAt(i, mtx);
        }
        child.instanceMatrix.needsUpdate = true;
    });
}

function applyFinalColorToObject(object, color) {
    if (!object || color === null || color === undefined) return;
    object.traverse((child) => {
        if (!child.material) return;
        const materials = Array.isArray(child.material) ? child.material : [child.material];
        materials.forEach((mat) => {
            if (!mat) return;
            const prevIntensity = mat.userData?.sciFiUniforms && typeof mat.emissiveIntensity === 'number'
                ? mat.emissiveIntensity
                : null;
            updateSciFiMaterialColor(mat, color);
            if (prevIntensity !== null) {
                mat.emissiveIntensity = prevIntensity;
            }
        });
    });
}

function buildSelectionClonePreview(selectionInfo, label) {
    const source = selectionInfo?.object || selectionInfo?.hit?.object;
    if (!source || !label) return null;
    let match = null;
    let current = source;
    while (current) {
        if (current.userData?.label === label) {
            match = current;
        }
        current = current.parent;
    }
    const root = match || source;
    if (!root || root.isScene) return null;
    const clone = root.clone(true);
    clone.traverse((child) => {
        child.matrixAutoUpdate = true;
    });
    const previewGeometries = cloneGeometriesForPreview(clone);
    const previewMaterials = cloneMaterialsForPreview(clone);
    if (isQkvMatrixLabel(label)) {
        applyLaneOverrideToInstancedMeshes(clone, PREVIEW_QKV_LANES, PREVIEW_QKV_LANE_SPACING);
    }
    const finalColor = resolveFinalPreviewColor(label);
    if (!isLayerNormLabel(label)) {
        applyFinalColorToObject(clone, finalColor);
    }
    return {
        object: clone,
        dispose: () => {
            previewGeometries.forEach((geo) => geo && geo.dispose && geo.dispose());
            previewMaterials.forEach((mat) => mat && mat.dispose && mat.dispose());
        }
    };
}

function buildSharedClonePreview(selectionInfo, label) {
    const source = selectionInfo?.object || selectionInfo?.hit?.object;
    if (!source || !label) return null;
    let match = null;
    let current = source;
    while (current) {
        if (current.userData?.label === label) {
            match = current;
        }
        current = current.parent;
    }
    const root = match || source;
    if (!root || root.isScene) return null;
    const clone = root.clone(true);
    clone.traverse((child) => {
        child.matrixAutoUpdate = true;
        if (child.isInstancedMesh) {
            if (child.instanceMatrix) child.instanceMatrix.needsUpdate = true;
            if (child.instanceColor) child.instanceColor.needsUpdate = true;
        }
    });
    return {
        object: clone,
        dispose: () => {}
    };
}

function buildDirectClonePreview(selectionInfo) {
    const source = selectionInfo?.object || selectionInfo?.hit?.object;
    if (!source || source.isScene) return null;
    const clone = source.clone(true);
    clone.traverse((child) => {
        child.matrixAutoUpdate = true;
    });
    const previewGeometries = cloneGeometriesForPreview(clone);
    const previewMaterials = cloneMaterialsForPreview(clone);
    return {
        object: clone,
        dispose: () => {
            previewGeometries.forEach((geo) => geo && geo.dispose && geo.dispose());
            previewMaterials.forEach((mat) => mat && mat.dispose && mat.dispose());
        }
    };
}

// Compute bounds with instanced meshes included (Box3.setFromObject skips instances).
function getObjectBounds(object) {
    const bounds = new THREE.Box3();
    if (!object) return bounds;
    object.updateWorldMatrix(true, true);
    object.traverse((child) => {
        if (!child.geometry) return;
        if (child.isInstancedMesh) {
            if (!child.geometry.boundingBox) child.geometry.computeBoundingBox();
            if (!child.geometry.boundingBox) return;
            const instanceCount = Number.isFinite(child.count)
                ? child.count
                : (child.instanceMatrix?.count ?? 0);
            for (let i = 0; i < instanceCount; i++) {
                child.getMatrixAt(i, TMP_MATRIX);
                TMP_MATRIX.decompose(TMP_POS, TMP_QUAT, TMP_SCALE);
                // Skip hidden instances (moved to sentinel Y or shrunk)
                if (TMP_POS.y <= HIDE_INSTANCE_Y_OFFSET * 0.5 || TMP_SCALE.y < 0.01) continue;
                TMP_MATRIX.compose(TMP_POS, TMP_QUAT, TMP_SCALE);
                TMP_MATRIX.multiplyMatrices(child.matrixWorld, TMP_MATRIX);
                TMP_BOX.copy(child.geometry.boundingBox).applyMatrix4(TMP_MATRIX);
                bounds.union(TMP_BOX);
            }
            return;
        }
        if (!child.geometry.boundingBox) child.geometry.computeBoundingBox();
        if (!child.geometry.boundingBox) return;
        TMP_BOX.copy(child.geometry.boundingBox).applyMatrix4(child.matrixWorld);
        bounds.union(TMP_BOX);
    });
    return bounds;
}

function resolveMetadata(label, kind = null) {
    const lower = (label || '').toLowerCase();
    if (lower.startsWith('token:') || lower.startsWith('position:')) {
        return { params: 'TBD', dims: 'TBD' };
    }
    if (lower.includes('query weight matrix')) {
        return { params: formatNumber(D_MODEL * D_MODEL), dims: formatDims(D_MODEL, D_MODEL) };
    }
    if (lower.includes('key weight matrix')) {
        return { params: formatNumber(D_MODEL * D_MODEL), dims: formatDims(D_MODEL, D_MODEL) };
    }
    if (lower.includes('value weight matrix')) {
        return { params: formatNumber(D_MODEL * D_MODEL), dims: formatDims(D_MODEL, D_MODEL) };
    }
    if (lower.includes('output projection matrix')) {
        return { params: formatNumber(D_MODEL * D_MODEL), dims: formatDims(D_MODEL, D_MODEL) };
    }
    if (lower.includes('mlp up weight matrix')) {
        return { params: formatNumber(D_MODEL * D_MODEL * 4), dims: formatDims(D_MODEL, D_MODEL * 4) };
    }
    if (lower.includes('mlp down weight matrix')) {
        return { params: formatNumber(D_MODEL * D_MODEL * 4), dims: formatDims(D_MODEL * 4, D_MODEL) };
    }
    if (lower.includes('vocab embedding')) {
        return { params: formatNumber(VOCAB_SIZE * D_MODEL), dims: formatDims(VOCAB_SIZE, D_MODEL) };
    }
    if (lower.includes('positional embedding')) {
        return { params: formatNumber(CONTEXT_LEN * D_MODEL), dims: formatDims(CONTEXT_LEN, D_MODEL) };
    }
    if (lower.includes('query vector') || lower.includes('key vector') || lower.includes('value vector')) {
        return { params: 'TBD', dims: 'TBD' };
    }
    if (lower.includes('attention') || (kind === 'mergedKV')) {
        return { params: 'TBD', dims: 'TBD' };
    }
    return { params: 'TBD', dims: 'TBD' };
}

function resolveDescription(label, kind = null, selectionInfo = null) {
    const lower = (label || '').toLowerCase();
    const activation = getActivationDataFromSelection(selectionInfo);
    const stage = activation?.stage || '';

    if (lower.startsWith('token:')) {
        return 'This is a prompt token, the raw discrete symbol GPT-2 reads. It gets embedded into a dense vector and then combined with a positional embedding before flowing through the stack. The same token can influence many later tokens via attention.';
    }
    if (lower.startsWith('position:')) {
        return 'This is the position embedding for a specific index in the sequence. It is added to the token embedding so the model can reason about order. Without this signal, GPT-2 would treat the sequence as a bag of tokens.';
    }
    if (lower.includes('vocab embedding (top)')) {
        return 'This is the unembedding matrix used at the top of the model. It maps hidden states back into vocabulary logits to choose the next token. In GPT‑2 it is tied to the input token embedding weights.';
    }
    if (lower.includes('vocab embedding')) {
        return 'This matrix maps discrete token ids into continuous vectors. It is the first learned projection the model applies to the prompt. Those vectors carry semantic and syntactic information into the residual stream.';
    }
    if (lower.includes('positional embedding')) {
        return 'This adds position information so each token knows where it sits in the sequence. It is summed with the token embedding at the bottom of the model. This allows attention to distinguish “first” versus “last” occurrences.';
    }
    if (lower.includes('query weight matrix')) {
        return 'This matrix projects residual vectors into query space for self‑attention. Queries determine what each token is looking for. They are matched against keys to produce attention scores.';
    }
    if (lower.includes('key weight matrix')) {
        return 'This matrix projects residual vectors into key space for self‑attention. Keys represent what each token offers to be attended to. Queries score against keys to decide which tokens matter most.';
    }
    if (lower.includes('value weight matrix')) {
        return 'This matrix projects residual vectors into value space for self‑attention. Values are the content that gets mixed together according to attention weights. The weighted sum of values becomes the attention output.';
    }
    if (lower.includes('output projection matrix')) {
        return 'After heads are concatenated, this matrix projects them back to model dimension. It lets the model recombine head information into a single residual stream vector. This is the final linear step of attention.';
    }
    if (lower.includes('mlp up weight matrix')) {
        return 'This matrix expands the hidden state into a larger MLP dimension. The expansion gives the model more capacity for nonlinear transformations. It precedes the GELU activation.';
    }
    if (lower.includes('mlp down weight matrix')) {
        return 'This matrix projects the MLP hidden state back to model dimension. It compresses the nonlinear features into the residual stream. Together with the up‑projection it forms the feed‑forward block.';
    }
    if (lower.includes('layernorm') || lower.includes('layer norm')) {
        if (lower.includes('scale') || lower.includes('gamma')) {
            return 'This is the LayerNorm scale (gamma) vector. It rescales each normalized feature to restore useful magnitudes. Its values are learned per feature and are shared across tokens.';
        }
        if (lower.includes('shift') || lower.includes('beta')) {
            return 'This is the LayerNorm shift (beta) vector. It offsets each normalized feature after scaling. It lets the model re‑center activations in a learned way.';
        }
        if (lower.includes('normed') || lower.includes('normalized')) {
            return 'This is the LayerNorm output for a token after normalization. It has zero‑mean, unit‑variance statistics per token (before scale/shift). This stabilized vector feeds into attention or the MLP.';
        }
        return 'LayerNorm normalizes each token’s features and then applies learned scale and shift. It stabilizes training and keeps activations in a usable range. GPT‑2 uses pre‑LayerNorm, so it happens before each sublayer.';
    }
    if (lower.includes('merged key vectors')) {
        return 'These are the key vectors from all heads stacked together. They represent what each token offers to be matched. The attention mechanism compares queries to these keys.';
    }
    if (lower.includes('merged value vectors')) {
        return 'These are the value vectors from all heads stacked together. They hold the content that will be mixed by attention weights. The weighted mix becomes the attended representation.';
    }
    if (lower.includes('query vector')) {
        return 'This is a query vector for a specific token. It encodes what the token is looking for in the rest of the sequence. It is used to score all key vectors.';
    }
    if (lower.includes('key vector')) {
        return 'This is a key vector for a specific token. It encodes what the token offers to be attended to. Queries compare against it to compute attention scores.';
    }
    if (lower.includes('value vector')) {
        return 'This is a value vector for a specific token. It is the information that gets mixed when other tokens attend to it. The final attention output is a weighted sum of values.';
    }
    if (lower.includes('attention score') || stage.startsWith('attention.')) {
        if (stage === 'attention.pre') {
            return 'This is a raw attention score from a source token to a target token. It is computed by a scaled dot‑product of query and key. A causal mask is added before softmax.';
        }
        if (stage === 'attention.post') {
            return 'This is a normalized attention weight after softmax. It tells how much the source token reads from the target token. The weights sum to 1 for each source token.';
        }
        return 'This is an attention score connecting a source token to a target token. Scores determine how much information flows between tokens. They are computed per head.';
    }
    if (lower.includes('attention')) {
        return 'Self‑attention lets tokens read from other tokens in the sequence. Queries, keys, and values are computed per head and combined. The result is added back into the residual stream.';
    }
    if (lower.includes('top logit bars')) {
        return 'These are logits over the vocabulary before softmax. Higher values indicate more likely next tokens. The model samples from these to produce the next token.';
    }
    if (lower.includes('incoming residual') || lower.includes('residual')) {
        return 'This is the residual stream vector for a token at this point in the model. It carries forward all information accumulated so far. Sub‑layers (attention, MLP) add their outputs back into it.';
    }
    if (lower.includes('mlp')) {
        return 'The MLP is a token‑wise feed‑forward network. It expands, applies a nonlinearity, and compresses back to model dimension. This adds nonlinear capacity separate from attention.';
    }
    if (lower.includes('vector')) {
        return 'This vector represents a token’s state at a particular stage. It is a slice of the residual stream or a derived representation (Q/K/V). It will be transformed or mixed by later layers.';
    }
    if (lower.includes('weight matrix')) {
        return 'This is a learned linear transformation. It projects token representations into a new space. Different matrices specialize for attention or MLP roles.';
    }
    if (kind) {
        return 'This selection is a GPT‑2 component involved in transforming token representations. It participates in the residual stream, attention, or MLP pipeline. Its role depends on the stage of the forward pass.';
    }
    return 'This selection is part of the GPT‑2 visualization. It represents a structure used to transform, normalize, or route token information. Look at nearby labels to see how it fits into the layer.';
}

function extractTokenText(label) {
    if (!label) return '';
    const match = label.match(/^(token|position)\s*:\s*(.*)$/i);
    if (!match) {
        const trimmed = label.trim();
        return trimmed.length ? trimmed : SPACE_TOKEN_DISPLAY;
    }
    const extracted = match[2] || '';
    const trimmed = extracted.trim();
    return trimmed.length ? trimmed : SPACE_TOKEN_DISPLAY;
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

function createTokenChipShared(labelText) {
    const rawText = (typeof labelText === 'string') ? labelText : '';
    const text = rawText.trim().length ? rawText : SPACE_TOKEN_DISPLAY;
    const font = tokenChipFont;
    let textGeo = null;
    let textMat = null;
    let textCullMat = null;
    let textMesh = null;
    let textGroup = null;
    let textTexture = null;
    let textPlaneAspect = 1;
    let textShapes = null;
    let textDepth = 0;
    let textFaceGeo = null;
    let bounds = null;
    let textWidth = 0;
    let textHeight = 0;
    let useGeometryText = false;
    const capOffset = 0.05;

    if (font && text.trim().length) {
        const desiredDepth = Number.isFinite(TOKEN_CHIP_STYLE.textDepth) ? TOKEN_CHIP_STYLE.textDepth : 0;
        const chipDepth = Number.isFinite(TOKEN_CHIP_STYLE.depth) ? TOKEN_CHIP_STYLE.depth : desiredDepth;
        textDepth = Number.isFinite(chipDepth) ? chipDepth + capOffset * 2 : desiredDepth;
        textShapes = font.generateShapes(text, TOKEN_CHIP_STYLE.textSize, 2);
        textGeo = new THREE.ExtrudeGeometry(textShapes, {
            depth: textDepth,
            curveSegments: 4,
            bevelEnabled: false
        });
        textGeo.computeBoundingBox();
        textGeo.computeVertexNormals();
        const textBounds = textGeo.boundingBox;
        if (textBounds && Number.isFinite(textBounds.max.x) && Number.isFinite(textBounds.min.x)) {
            textWidth = Math.max(0, textBounds.max.x - textBounds.min.x);
            textHeight = Math.max(0, textBounds.max.y - textBounds.min.y);
        }
        textGeo.translate(0, 0, -textDepth / 2);
        textGeo.computeBoundingBox();
        bounds = textGeo.boundingBox;
        useGeometryText = true;
    } else {
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        const fontSize = TOKEN_CHIP_STYLE.textSize;
        ctx.font = `600 ${fontSize}px ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial`;
        const textMetrics = ctx.measureText(text);
        textWidth = Math.ceil(textMetrics.width);
        textHeight = Math.ceil(fontSize * 1.15);
        canvas.width = Math.max(256, textWidth + 80);
        canvas.height = Math.max(128, textHeight + 60);
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.fillStyle = '#ffffff';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.font = `600 ${fontSize}px ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial`;
        ctx.fillText(text, canvas.width / 2, canvas.height / 2);

        textTexture = new THREE.CanvasTexture(canvas);
        textTexture.minFilter = THREE.LinearFilter;
        textTexture.magFilter = THREE.LinearFilter;
        textTexture.needsUpdate = true;
        textPlaneAspect = canvas.width / canvas.height;
    }

    const chipWidth = Math.max(TOKEN_CHIP_STYLE.minWidth, textWidth + TOKEN_CHIP_STYLE.padding);
    const chipHeight = typeof TOKEN_CHIP_STYLE.height === 'number'
        ? TOKEN_CHIP_STYLE.height
        : Math.max(TOKEN_CHIP_STYLE.minHeight, textHeight + TOKEN_CHIP_STYLE.padding);
    const chipRadius = Math.min(TOKEN_CHIP_STYLE.cornerRadius, Math.min(chipWidth, chipHeight) / 2 - 1);
    const chipShape = buildRoundedRectShape(chipWidth, chipHeight, chipRadius);
    const chipGeo = new THREE.ExtrudeGeometry(chipShape, { depth: TOKEN_CHIP_STYLE.depth, bevelEnabled: false });
    chipGeo.translate(0, 0, -TOKEN_CHIP_STYLE.depth / 2);
    chipGeo.computeVertexNormals();

    const chipMat = new THREE.MeshStandardMaterial({
        color: 0xf2e8d5,
        roughness: 0.35,
        metalness: 0.15,
        side: THREE.DoubleSide
    });
    const chipMesh = new THREE.Mesh(chipGeo, chipMat);

    const capMat = chipMat.clone();
    capMat.polygonOffset = false;
    capMat.polygonOffsetFactor = 0;
    capMat.polygonOffsetUnits = 0;
    const capGeo = new THREE.ShapeGeometry(chipShape);
    capGeo.computeVertexNormals();
    const frontCap = new THREE.Mesh(capGeo, capMat);
    frontCap.position.z = TOKEN_CHIP_STYLE.depth / 2 + capOffset;
    const backCap = new THREE.Mesh(capGeo, capMat);
    backCap.position.z = -TOKEN_CHIP_STYLE.depth / 2 - capOffset;
    backCap.rotation.y = Math.PI;

    if (useGeometryText) {
        if (textGeo && textWidth > 0 && textHeight > 0) {
            textMat = new THREE.MeshBasicMaterial({
                color: 0xffffff,
                side: THREE.DoubleSide,
                depthWrite: true,
                depthTest: true,
                polygonOffset: true,
                polygonOffsetFactor: -0.5,
                polygonOffsetUnits: -0.5
            });
            textCullMat = textMat.clone();
            textCullMat.colorWrite = false;
            textCullMat.depthWrite = false;
            textCullMat.transparent = true;
            textCullMat.opacity = 0;
            textGroup = new THREE.Group();
            textMesh = new THREE.Mesh(textGeo, [textCullMat, textMat]);
            textGroup.add(textMesh);
            if (textShapes) {
                const faceGeo = new THREE.ShapeGeometry(textShapes);
                faceGeo.computeVertexNormals();
                textFaceGeo = faceGeo;
                const faceOffset = 0.02;
                const frontFace = new THREE.Mesh(faceGeo, textMat);
                frontFace.position.z = textDepth / 2 + faceOffset;
                const backFace = new THREE.Mesh(faceGeo, textMat);
                backFace.position.z = -textDepth / 2 - faceOffset;
                textGroup.add(frontFace, backFace);
            }
            if (bounds) {
                const centerX = (bounds.min.x + bounds.max.x) / 2;
                const centerY = (bounds.min.y + bounds.max.y) / 2;
                textGroup.position.set(-centerX, -centerY, 0);
            }
        }
    } else if (textTexture) {
        let textPlaneHeight = chipHeight * 0.38;
        let textPlaneWidth = textPlaneHeight * textPlaneAspect;
        const maxTextWidth = chipWidth * 0.8;
        if (textPlaneWidth > maxTextWidth) {
            textPlaneWidth = maxTextWidth;
            textPlaneHeight = textPlaneWidth / textPlaneAspect;
        }
        textGeo = new THREE.PlaneGeometry(textPlaneWidth, textPlaneHeight);
        textMat = new THREE.MeshBasicMaterial({
            map: textTexture,
            transparent: true,
            depthWrite: true,
            depthTest: true,
            polygonOffset: true,
            polygonOffsetFactor: -0.5,
            polygonOffsetUnits: -0.5,
            side: THREE.DoubleSide
        });
        textMesh = new THREE.Mesh(textGeo, textMat);
        textMesh.position.z = TOKEN_CHIP_STYLE.depth / 2 + TOKEN_CHIP_STYLE.textOffset;
    }

    const group = new THREE.Group();
    group.add(chipMesh, frontCap, backCap);
    if (textGroup) {
        group.add(textGroup);
    } else if (textMesh) {
        group.add(textMesh);
    }

    return {
        group,
        dispose: () => {
            chipGeo.dispose();
            chipMat.dispose();
            capGeo.dispose();
            capMat.dispose();
            if (textGeo) textGeo.dispose();
            if (textFaceGeo) textFaceGeo.dispose();
            if (textMat) textMat.dispose();
            if (textCullMat) textCullMat.dispose();
            if (textTexture) textTexture.dispose();
        }
    };
}

function buildTokenChipPreview(labelText) {
    const shared = createTokenChipShared(labelText);
    const group = new THREE.Group();
    const laneCount = Math.max(1, Math.floor(PREVIEW_TOKEN_LANES));
    for (let i = 0; i < laneCount; i++) {
        const chip = (i === 0) ? shared.group : shared.group.clone(true);
        chip.position.z = (i - (laneCount - 1) / 2) * PREVIEW_LANE_SPACING * 0.6;
        group.add(chip);
    }
    return { object: group, dispose: shared.dispose };
}

function buildWeightMatrixPreview(params, colorHex) {
    const depth = PREVIEW_MATRIX_DEPTH;
    const slitCount = PREVIEW_SOLID_LANES;
    const matrix = new WeightMatrixVisualization(
        null,
        new THREE.Vector3(0, 0, 0),
        params.width,
        params.height,
        depth,
        params.topWidthFactor,
        params.cornerRadius,
        slitCount,
        params.slitWidth,
        params.slitDepthFactor,
        params.slitBottomWidthFactor,
        params.slitTopWidthFactor,
        true
    );
    if (colorHex !== null && colorHex !== undefined) {
        matrix.setColor(new THREE.Color(colorHex));
    }
    matrix.setMaterialProperties({ opacity: 0.98, transparent: false, emissiveIntensity: 0.18 });
    return {
        object: matrix.group,
        dispose: () => {
            const meshes = [matrix.mesh, matrix.frontCapMesh, matrix.backCapMesh];
            meshes.forEach(mesh => {
                if (!mesh || !mesh.material) return;
                if (Array.isArray(mesh.material)) {
                    mesh.material.forEach(mat => mat && mat.dispose && mat.dispose());
                } else {
                    mesh.material.dispose();
                }
            });
        }
    };
}

function extractPreviewVectorData(selectionInfo) {
    const candidates = [
        selectionInfo?.info?.activationData?.values,
        selectionInfo?.object?.userData?.activationData?.values,
        selectionInfo?.hit?.object?.userData?.activationData?.values,
        selectionInfo?.info?.vectorData,
        selectionInfo?.info?.values
    ];
    for (const arr of candidates) {
        if (Array.isArray(arr) && arr.length > 0) {
            return Array.from(arr).map((v) => Number.isFinite(v) ? v : 0);
        }
    }
    return null;
}

function mapDataToInstanceCount(data, instanceCount) {
    if (!Array.isArray(data) || !Number.isFinite(instanceCount) || instanceCount < 1) return null;
    const count = Math.max(1, Math.floor(instanceCount));
    if (data.length === count) return data.slice();
    const buckets = [];
    const step = data.length / count;
    for (let i = 0; i < count; i++) {
        const start = Math.floor(i * step);
        const end = Math.floor((i + 1) * step);
        if (end <= start) {
            buckets.push(data[Math.min(start, data.length - 1)] ?? 0);
            continue;
        }
        let sum = 0;
        let n = 0;
        for (let j = start; j < end; j++) {
            sum += data[j];
            n += 1;
        }
        buckets.push(n > 0 ? sum / n : 0);
    }
    return buckets;
}

function applyDataToPreviewVector(vec, data) {
    if (!vec || !Array.isArray(data) || data.length === 0) return;
    vec.updateDataAndSnapVisuals(data.slice());
    const numKeyColors = Math.max(2, Math.min(12, data.length));
    vec.updateKeyColorsFromData(data, numKeyColors, null, data);
}

function createPreviewVector(options = {}) {
    const { colorHex, data = null, instanceCount = PREVIEW_VECTOR_BODY_INSTANCES } = options;
    const vec = new VectorVisualizationInstancedPrism(null, new THREE.Vector3(0, 0, 0), 1, instanceCount);
    vec.numSubsections = 1;
    if (Array.isArray(data) && data.length > 0) {
        const mapped = mapDataToInstanceCount(data, instanceCount) || data;
        applyDataToPreviewVector(vec, mapped);
    } else {
        const color = new THREE.Color(colorHex || 0xffffff);
        vec.currentKeyColors = [color.clone(), color.clone()];
        vec.updateInstanceGeometryAndColors();
    }
    return vec;
}

function buildVectorPreview(colorHex, selectionInfo = null) {
    const group = new THREE.Group();
    const vectors = [];
    const data = extractPreviewVectorData(selectionInfo);
    for (let i = 0; i < PREVIEW_LANES; i++) {
        const vec = createPreviewVector({ colorHex, data });
        vec.group.position.z = (i - (PREVIEW_LANES - 1) / 2) * PREVIEW_LANE_SPACING;
        group.add(vec.group);
        vectors.push(vec);
    }
    return {
        object: group,
        dispose: () => {
            vectors.forEach(vec => {
                if (vec.mesh?.geometry) vec.mesh.geometry.dispose();
                if (vec.mesh?.material) vec.mesh.material.dispose();
            });
        }
    };
}

function createTrailLine(colorHex) {
    const points = [new THREE.Vector3(), new THREE.Vector3()];
    const geometry = new THREE.BufferGeometry().setFromPoints(points);
    const material = new THREE.LineBasicMaterial({
        color: colorHex || PREVIEW_TRAIL_COLOR,
        transparent: true,
        opacity: 0.85,
        depthWrite: false,
        depthTest: true
    });
    const line = new THREE.Line(geometry, material);
    return {
        line,
        update: (start, end, opacity = 1.0) => {
            const pos = geometry.attributes.position.array;
            pos[0] = start.x; pos[1] = start.y; pos[2] = start.z;
            pos[3] = end.x; pos[4] = end.y; pos[5] = end.z;
            geometry.attributes.position.needsUpdate = true;
            if (material) {
                material.opacity = THREE.MathUtils.clamp(opacity, 0, 1);
            }
        },
        dispose: () => {
            geometry.dispose();
            material.dispose();
        }
    };
}

function buildQkvFlowPreview(highlightType, selectionInfo = null) {
    // Animated depiction of post-LN vectors rising into the Q/K/V projection,
    // leaving a trail, and emitting a smaller head vector above.
    const group = new THREE.Group();
    const lanes = [];
    const laneSpacing = PREVIEW_QKV_LANE_SPACING;
    const totalDuration = PREVIEW_QKV_RISE_DURATION + PREVIEW_QKV_CONVERT_DURATION
        + PREVIEW_QKV_HOLD_DURATION + PREVIEW_QKV_EXIT_DURATION + PREVIEW_QKV_IDLE_DURATION;
    const startTime = performance.now();
    const neutralColor = 0xa7b3c2;
    const highlightKey = typeof highlightType === 'string' ? highlightType.toUpperCase() : '';
    const highlightScale = (type) => (highlightKey === type ? 1.1 : 1.0);
    const highlightColor = (type) => {
        if (type === 'Q') return MHA_FINAL_Q_COLOR;
        if (type === 'K') return MHA_FINAL_K_COLOR;
        return MHA_FINAL_V_COLOR;
    };
    const baseData = extractPreviewVectorData(selectionInfo);
    const headData = Array.isArray(baseData) && baseData.length > 0
        ? [baseData.reduce((sum, v) => sum + v, 0) / baseData.length]
        : baseData;

    for (let i = 0; i < PREVIEW_QKV_LANES; i++) {
        const z = (i - (PREVIEW_QKV_LANES - 1) / 2) * laneSpacing;
        const x = (highlightKey === 'Q') ? -PREVIEW_QKV_X_SPREAD : (highlightKey === 'V' ? PREVIEW_QKV_X_SPREAD : 0);
        const incoming = createPreviewVector({
            colorHex: neutralColor,
            data: baseData,
            instanceCount: PREVIEW_VECTOR_BODY_INSTANCES
        });
        const outgoing = createPreviewVector({
            colorHex: highlightColor(highlightKey || 'K'),
            data: headData,
            instanceCount: PREVIEW_VECTOR_HEAD_INSTANCES
        });
        const trail = createTrailLine(PREVIEW_TRAIL_COLOR);

        incoming.group.position.set(x, PREVIEW_QKV_START_Y, z);
        outgoing.group.position.set(x, PREVIEW_QKV_START_Y, z);
        incoming.group.scale.setScalar(PREVIEW_VECTOR_LARGE_SCALE);
        outgoing.group.scale.setScalar(PREVIEW_VECTOR_SMALL_SCALE * highlightScale(highlightKey || 'K'));

        group.add(incoming.group, outgoing.group, trail.line);
        lanes.push({ incoming, outgoing, trail, x, z });
        incoming.group.visible = false;
        outgoing.group.visible = false;
        trail.line.visible = false;
    }

    const disposeVector = (vec) => {
        if (vec?.dispose) vec.dispose();
    };

    const dispose = () => {
        lanes.forEach((lane) => {
            disposeVector(lane.incoming);
            disposeVector(lane.outgoing);
            if (lane.trail?.dispose) lane.trail.dispose();
        });
    };

    const updateLane = (lane, localTime) => {
        const { incoming, outgoing, trail, x, z } = lane;
        const endRise = PREVIEW_QKV_RISE_DURATION;
        const endConvert = endRise + PREVIEW_QKV_CONVERT_DURATION;
        const endHold = endConvert + PREVIEW_QKV_HOLD_DURATION;
        const endExit = endHold + PREVIEW_QKV_EXIT_DURATION;

        if (localTime < 0) {
            incoming.group.visible = false;
            outgoing.group.visible = false;
            trail.line.visible = false;
            incoming.group.position.set(x, PREVIEW_QKV_START_Y, z);
            outgoing.group.position.set(x, PREVIEW_QKV_START_Y, z);
            return;
        }

        if (localTime <= endRise) {
            const t = easeInOutCubic(localTime / PREVIEW_QKV_RISE_DURATION);
            incoming.group.visible = true;
            outgoing.group.visible = false;
            trail.line.visible = true;
            const y = THREE.MathUtils.lerp(PREVIEW_QKV_START_Y, PREVIEW_QKV_MATRIX_Y, t);
            incoming.group.position.set(x, y, z);
            trail.update(new THREE.Vector3(x, PREVIEW_QKV_START_Y, z), new THREE.Vector3(x, y, z), 0.85);
            return;
        }

        if (localTime <= endConvert) {
            const t = easeInOutCubic((localTime - endRise) / PREVIEW_QKV_CONVERT_DURATION);
            incoming.group.visible = true;
            outgoing.group.visible = true;
            trail.line.visible = true;

            const incomingScale = THREE.MathUtils.lerp(PREVIEW_VECTOR_LARGE_SCALE, PREVIEW_VECTOR_LARGE_SCALE * 0.45, t);
            incoming.group.scale.setScalar(incomingScale);
            incoming.group.position.set(x, PREVIEW_QKV_MATRIX_Y, z);

            const y = THREE.MathUtils.lerp(PREVIEW_QKV_MATRIX_Y, PREVIEW_QKV_OUTPUT_Y, t);
            outgoing.group.position.set(x, y, z);
            trail.update(new THREE.Vector3(x, PREVIEW_QKV_START_Y, z), new THREE.Vector3(x, y, z), 1.0 - t * 0.4);
            return;
        }

        if (localTime <= endHold) {
            incoming.group.visible = false;
            outgoing.group.visible = true;
            trail.line.visible = true;
            outgoing.group.position.set(x, PREVIEW_QKV_OUTPUT_Y, z);
            trail.update(new THREE.Vector3(x, PREVIEW_QKV_START_Y, z), new THREE.Vector3(x, PREVIEW_QKV_OUTPUT_Y, z), 0.55);
            return;
        }

        if (localTime <= endExit) {
            incoming.group.visible = false;
            const t = easeInOutCubic((localTime - endHold) / PREVIEW_QKV_EXIT_DURATION);
            const y = THREE.MathUtils.lerp(PREVIEW_QKV_OUTPUT_Y, PREVIEW_QKV_EXIT_Y, t);
            outgoing.group.position.set(x, y, z);
            const visible = t < 0.96;
            outgoing.group.visible = visible;
            trail.update(new THREE.Vector3(x, PREVIEW_QKV_START_Y, z), new THREE.Vector3(x, y, z), visible ? 0.35 : 0.0);
            return;
        }

        incoming.group.visible = false;
        outgoing.group.visible = false;
        trail.line.visible = false;
        incoming.group.position.set(x, PREVIEW_QKV_START_Y, z);
        outgoing.group.position.set(x, PREVIEW_QKV_START_Y, z);
    };

    const animate = (_, nowMs) => {
        const elapsed = ((nowMs - startTime) % totalDuration + totalDuration) % totalDuration;
        lanes.forEach((lane, idx) => {
            const laneTime = elapsed - idx * PREVIEW_QKV_LANE_STAGGER;
            updateLane(lane, laneTime);
        });
    };

    return { object: group, dispose, animate };
}

function buildQkvMatrixFlowPreview(label, selectionInfo) {
    const type = inferQkvType(label, selectionInfo);
    const matrixColor = type === 'Q' ? MHA_FINAL_Q_COLOR : (type === 'V' ? MHA_FINAL_V_COLOR : MHA_FINAL_K_COLOR);
    const matrixPreview = buildWeightMatrixPreview(MHA_MATRIX_PARAMS, matrixColor);
    const x = (type === 'Q') ? -PREVIEW_QKV_X_SPREAD : (type === 'V' ? PREVIEW_QKV_X_SPREAD : 0);

    if (matrixPreview?.object) {
        matrixPreview.object.position.x = x;
        matrixPreview.object.position.y = PREVIEW_QKV_MATRIX_Y;
        return { ...matrixPreview, animate: null };
    }

    return buildStackedBoxPreview(matrixColor);
}

function buildStackedBoxPreview(colorHex) {
    const group = new THREE.Group();
    const geometry = new THREE.BoxGeometry(140, 140, 8);
    const meshes = [];
    for (let i = 0; i < PREVIEW_SOLID_LANES; i++) {
        const material = new THREE.MeshStandardMaterial({
            color: colorHex || 0x1f1f1f,
            metalness: 0.25,
            roughness: 0.65,
            emissive: new THREE.Color(0x060606),
            emissiveIntensity: 0.3
        });
        const mesh = new THREE.Mesh(geometry, material);
        mesh.position.z = (i - (PREVIEW_SOLID_LANES - 1) / 2) * 18;
        group.add(mesh);
        meshes.push(mesh);
    }
    return {
        object: group,
        dispose: () => {
            geometry.dispose();
            meshes.forEach(mesh => mesh.material && mesh.material.dispose());
        }
    };
}

function findVectorLikeObject(selectionInfo) {
    const directRef = selectionInfo?.info?.vectorRef;
    if (directRef?.group?.isObject3D) return directRef.group;
    if (directRef?.isObject3D) return directRef;
    const sources = [];
    if (selectionInfo?.hit?.object) sources.push(selectionInfo.hit.object);
    if (selectionInfo?.object) sources.push(selectionInfo.object);
    if (!sources.length) return null;

    const hasVectorAttributes = (obj) => {
        const geo = obj?.geometry;
        if (!geo || typeof geo.getAttribute !== 'function') return false;
        return !!(
            geo.getAttribute('colorStart')
            || geo.getAttribute('colorEnd')
            || geo.getAttribute('instanceColor')
            || obj.instanceColor
        );
    };

    const findInHierarchy = (root) => {
        let found = null;
        root.traverse((child) => {
            if (found) return;
            if (hasVectorAttributes(child) && (child.isMesh || child.isInstancedMesh)) {
                found = child;
            }
        });
        return found;
    };

    let vectorMesh = null;
    for (const src of sources) {
        vectorMesh = findInHierarchy(src);
        if (vectorMesh) break;
    }
    if (!vectorMesh) return null;

    // Prefer the nearest ancestor labeled as a vector if present.
    let candidate = vectorMesh;
    let walker = vectorMesh.parent;
    while (walker && !walker.isScene) {
        const lbl = walker.userData?.label;
        if (lbl && String(lbl).toLowerCase().includes('vector')) {
            candidate = walker;
        }
        walker = walker.parent;
    }
    return candidate;
}

function extractMaterialSnapshot(selectionInfo) {
    const root = selectionInfo?.object || selectionInfo?.hit?.object;
    if (!root) return null;
    let material = null;
    let current = root;
    while (current && !material) {
        if (current.material) {
            material = Array.isArray(current.material) ? current.material.find(Boolean) : current.material;
        }
        current = current.parent;
    }
    if (!material) return null;
    return {
        color: material.color ? material.color.clone() : null,
        emissiveIntensity: material.emissiveIntensity,
        opacity: material.opacity,
        transparent: material.transparent,
        metalness: material.metalness,
        roughness: material.roughness,
        clearcoat: material.clearcoat,
        clearcoatRoughness: material.clearcoatRoughness,
        transmission: material.transmission,
        thickness: material.thickness,
        iridescence: material.iridescence,
        sheen: material.sheen,
        sheenColor: material.sheenColor && material.sheenColor.clone ? material.sheenColor.clone() : material.sheenColor,
        envMapIntensity: material.envMapIntensity
    };
}

function applyMaterialSnapshot(object, snapshot) {
    if (!object || !snapshot) return;
    object.traverse((child) => {
        if (!child.material) return;
        const materials = Array.isArray(child.material) ? child.material : [child.material];
        materials.forEach((mat) => {
            if (!mat) return;
            if (snapshot.color) {
                updateSciFiMaterialColor(mat, snapshot.color);
            }
            if (Number.isFinite(snapshot.emissiveIntensity)) mat.emissiveIntensity = snapshot.emissiveIntensity;
            if (Number.isFinite(snapshot.opacity)) mat.opacity = snapshot.opacity;
            if (typeof snapshot.transparent === 'boolean') mat.transparent = snapshot.transparent;
            if (Number.isFinite(snapshot.metalness)) mat.metalness = snapshot.metalness;
            if (Number.isFinite(snapshot.roughness)) mat.roughness = snapshot.roughness;
            if (Number.isFinite(snapshot.clearcoat)) mat.clearcoat = snapshot.clearcoat;
            if (Number.isFinite(snapshot.clearcoatRoughness)) mat.clearcoatRoughness = snapshot.clearcoatRoughness;
            if (Number.isFinite(snapshot.transmission)) mat.transmission = snapshot.transmission;
            if (Number.isFinite(snapshot.thickness)) mat.thickness = snapshot.thickness;
            if (Number.isFinite(snapshot.iridescence)) mat.iridescence = snapshot.iridescence;
            if (Number.isFinite(snapshot.sheen)) mat.sheen = snapshot.sheen;
            if (snapshot.sheenColor && mat.sheenColor && mat.sheenColor.copy) {
                mat.sheenColor.copy(snapshot.sheenColor);
            }
            if (Number.isFinite(snapshot.envMapIntensity)) mat.envMapIntensity = snapshot.envMapIntensity;
        });
    });
}

function buildVectorClonePreview(selectionInfo) {
    const vectorObject = findVectorLikeObject(selectionInfo);
    if (!vectorObject || typeof vectorObject.clone !== 'function' || !vectorObject.isObject3D) return null;
    const clone = vectorObject.clone(true);
    clone.traverse((child) => {
        child.matrixAutoUpdate = true;
        child.visible = true;
        if (child.isInstancedMesh && child.instanceMatrix) {
            // Ensure instanced meshes render at least one instance.
            const instanceCount = child.instanceMatrix.count || child.count || 1;
            child.count = Math.max(1, instanceCount);
        }
    });
    const previewGeometries = cloneGeometriesForPreview(clone);
    const previewMaterials = cloneMaterialsForPreview(clone);
    return {
        object: clone,
        dispose: () => {
            previewGeometries.forEach((geo) => geo && geo.dispose && geo.dispose());
            previewMaterials.forEach((mat) => mat && mat.dispose && mat.dispose());
        }
    };
}

function buildLayerNormPreview(label, selectionInfo) {
    const clonePreview = buildSelectionClonePreview(selectionInfo, label)
        || buildDirectClonePreview(selectionInfo);
    if (clonePreview) {
        applyFinalColorToObject(clonePreview.object, 0xffffff);
        return clonePreview;
    }

    const baseHoles = Number.isFinite(LN_PARAMS.numberOfHoles) ? LN_PARAMS.numberOfHoles : PREVIEW_SOLID_LANES;
    const depthScale = PREVIEW_SOLID_LANES / Math.max(1, baseHoles);
    const previewDepth = Math.max(
        180,
        Math.min(PREVIEW_MATRIX_DEPTH, LN_PARAMS.depth * depthScale)
    );
    const params = { ...LN_PARAMS, numberOfHoles: PREVIEW_SOLID_LANES, depth: previewDepth };
    const ln = new LayerNormalizationVisualization(
        new THREE.Vector3(0, 0, 0),
        params.width,
        params.height,
        params.depth,
        params.wallThickness,
        params.numberOfHoles,
        params.holeWidth,
        params.holeWidthFactor,
        undefined,
        true
    );
    ln.setColor(new THREE.Color(0xffffff));
    return {
        object: ln.group,
        dispose: () => ln.dispose()
    };
}

function isLikelyVectorSelection(label, selectionInfo) {
    const lower = (label || '').toLowerCase();
    if (lower.includes('vector') || lower.includes('residual')) return true;
    const cat = selectionInfo?.info?.category;
    if (cat && ['q', 'k', 'v', 'vector', 'residual'].includes(String(cat).toLowerCase())) return true;
    const kind = selectionInfo?.kind;
    if (kind && ['vector', 'residual', 'mergedkv'].includes(String(kind).toLowerCase())) return true;
    if (findVectorLikeObject(selectionInfo)) return true;
    return false;
}

function isLayerNormLabel(label) {
    const lower = (label || '').toLowerCase();
    return lower.includes('layernorm') || lower.includes('layer norm');
}


function resolvePreviewObject(label, selectionInfo) {
    const lower = (label || '').toLowerCase();
    const isVectorSelection = isLikelyVectorSelection(label, selectionInfo);
    if (isVectorSelection) {
        const vectorClone = buildVectorClonePreview(selectionInfo);
        if (vectorClone) return vectorClone;
    }
    if (isQkvMatrixLabel(lower)) {
        const type = inferQkvType(label, selectionInfo);
        const matrixColor = type === 'Q' ? MHA_FINAL_Q_COLOR : (type === 'V' ? MHA_FINAL_V_COLOR : MHA_FINAL_K_COLOR);
        const sharedClone = buildSharedClonePreview(selectionInfo, label);
        if (sharedClone) return sharedClone;
        const preview = buildWeightMatrixPreview(MHA_MATRIX_PARAMS, matrixColor);
        const snapshot = extractMaterialSnapshot(selectionInfo);
        if (snapshot) {
            applyMaterialSnapshot(preview.object, snapshot);
        }
        return preview;
    }
    const clonePreview = buildSelectionClonePreview(selectionInfo, label)
        || buildDirectClonePreview(selectionInfo);
    if (clonePreview) {
        if (isLayerNormLabel(label)) {
            applyFinalColorToObject(clonePreview.object, 0xffffff);
        }
        return clonePreview;
    }
    if (lower.startsWith('token:') || lower.startsWith('position:')) {
        const clonePreview = buildSelectionClonePreview(selectionInfo, label);
        if (clonePreview) return clonePreview;
        return buildTokenChipPreview(extractTokenText(label));
    }
    if (lower.includes('output projection matrix')) {
        const height = MHA_MATRIX_PARAMS.height * MHA_OUTPUT_PROJECTION_MATRIX_PARAMS.heightFactor;
        const params = {
            ...MHA_MATRIX_PARAMS,
            width: MHA_OUTPUT_PROJECTION_MATRIX_PARAMS.width,
            height,
            topWidthFactor: MHA_OUTPUT_PROJECTION_MATRIX_PARAMS.topWidthFactor,
            cornerRadius: MHA_OUTPUT_PROJECTION_MATRIX_PARAMS.cornerRadius,
            slitWidth: MHA_OUTPUT_PROJECTION_MATRIX_PARAMS.slitWidth,
            slitDepthFactor: MHA_OUTPUT_PROJECTION_MATRIX_PARAMS.slitDepthFactor,
            slitBottomWidthFactor: MHA_OUTPUT_PROJECTION_MATRIX_PARAMS.slitBottomWidthFactor,
            slitTopWidthFactor: MHA_OUTPUT_PROJECTION_MATRIX_PARAMS.slitTopWidthFactor
        };
        return buildWeightMatrixPreview(params, MHA_OUTPUT_PROJECTION_MATRIX_COLOR);
    }
    if (lower.includes('mlp up weight matrix')) {
        return buildWeightMatrixPreview(MLP_MATRIX_PARAMS_UP, FINAL_MLP_COLOR);
    }
    if (lower.includes('mlp down weight matrix')) {
        return buildWeightMatrixPreview(MLP_MATRIX_PARAMS_DOWN, FINAL_MLP_COLOR);
    }
    if (lower.includes('vocab embedding')) {
        const clonePreview = buildSelectionClonePreview(selectionInfo, label)
            || buildDirectClonePreview(selectionInfo);
        if (clonePreview) return clonePreview;
        const color = lower.includes('top') ? FINAL_VOCAB_TOP_COLOR : MHA_FINAL_Q_COLOR;
        return buildWeightMatrixPreview(EMBEDDING_MATRIX_PARAMS_VOCAB, color);
    }
    if (lower.includes('positional embedding')) {
        const clonePreview = buildSelectionClonePreview(selectionInfo, label)
            || buildDirectClonePreview(selectionInfo);
        if (clonePreview) return clonePreview;
        return buildWeightMatrixPreview(EMBEDDING_MATRIX_PARAMS_POSITION, MHA_FINAL_K_COLOR);
    }
    if (isWeightMatrixLabel(lower)) {
        const color = resolveFinalPreviewColor(label);
        return buildWeightMatrixPreview(MHA_MATRIX_PARAMS, color);
    }

    if (lower.includes('query vector')) {
        return buildDirectClonePreview(selectionInfo)
            || buildSelectionClonePreview(selectionInfo, label)
            || buildVectorPreview(MHA_FINAL_Q_COLOR, selectionInfo);
    }
    if (lower.includes('key vector')) {
        return buildDirectClonePreview(selectionInfo)
            || buildSelectionClonePreview(selectionInfo, label)
            || buildVectorPreview(MHA_FINAL_K_COLOR, selectionInfo);
    }
    if (lower.includes('value vector')) {
        return buildDirectClonePreview(selectionInfo)
            || buildSelectionClonePreview(selectionInfo, label)
            || buildVectorPreview(MHA_FINAL_V_COLOR, selectionInfo);
    }
    if (selectionInfo?.kind === 'mergedKV') {
        const category = (selectionInfo.info?.category === 'V') ? 'V' : 'K';
        return buildDirectClonePreview(selectionInfo)
            || buildSelectionClonePreview(selectionInfo, label)
            || buildVectorPreview(category === 'V' ? MHA_FINAL_V_COLOR : MHA_FINAL_K_COLOR, selectionInfo);
    }
    if (isLikelyVectorSelection(label, selectionInfo)) {
        return buildDirectClonePreview(selectionInfo)
            || buildSelectionClonePreview(selectionInfo, label)
            || buildVectorPreview(null, selectionInfo);
    }

    if (lower.includes('layernorm') || lower.includes('layer norm')) {
        return buildLayerNormPreview(label, selectionInfo);
    }

    if (isAttentionScoreSelection(label, selectionInfo)) {
        return buildDirectClonePreview(selectionInfo)
            || buildSelectionClonePreview(selectionInfo, label)
            || buildStackedBoxPreview(0x1b1b1b);
    }

    if (lower.includes('attention')) {
        return buildStackedBoxPreview(0x1b1b1b);
    }

    return buildStackedBoxPreview(0x202020);
}

function fitObjectToView(object, camera, options = {}) {
    if (!object) return;
    const box = getObjectBounds(object);
    if (box.isEmpty()) return;
    const size = new THREE.Vector3();
    const center = new THREE.Vector3();
    box.getSize(size);
    box.getCenter(center);
    object.position.sub(center);
    const paddingMult = Number.isFinite(options.paddingMultiplier) ? options.paddingMultiplier : 1;
    const distanceMult = Number.isFinite(options.distanceMultiplier) ? options.distanceMultiplier : 1;
    const maxDim = Math.max(size.x, size.y, size.z);
    const scale = maxDim > 0 ? PREVIEW_TARGET_SIZE / (maxDim * PREVIEW_FRAME_PADDING * paddingMult) : 1;
    object.scale.setScalar(scale);

    const scaledBox = getObjectBounds(object);
    const scaledSize = new THREE.Vector3();
    scaledBox.getSize(scaledSize);
    const scaledMax = Math.max(scaledSize.x, scaledSize.y, scaledSize.z);
    const fov = THREE.MathUtils.degToRad(camera.fov);
    const distance = ((scaledMax / 2) / Math.tan(fov / 2)) * PREVIEW_BASE_DISTANCE_MULT * distanceMult;

    camera.near = Math.max(0.1, distance / 50);
    camera.far = distance * 20;
    camera.position.set(0, 0, distance * 1.6);
    camera.lookAt(0, 0, 0);
    camera.updateProjectionMatrix();
}

class SelectionPanel {
    constructor(options = {}) {
        this.panel = document.getElementById('detailPanel');
        this.hudPanel = document.getElementById('hudPanel');
        this.title = document.getElementById('detailTitle');
        this.params = document.getElementById('detailParams');
        this.dims = document.getElementById('detailDims');
        this.metaSection = document.getElementById('detailMeta');
        this.closeBtn = document.getElementById('detailClose');
        this.canvas = document.getElementById('detailCanvas');
        this.description = document.getElementById('detailDescription');
        this.dataEl = document.getElementById('detailData');
        this.dataSection = document.getElementById('detailDataSection');
        this.attentionRoot = document.getElementById('detailAttention');
        this.attentionToggle = document.getElementById('detailAttentionToggle');
        this.attentionTokensTop = document.getElementById('detailAttentionTokensTop');
        this.attentionTokensLeft = document.getElementById('detailAttentionTokensLeft');
        this.attentionMatrix = document.getElementById('detailAttentionMatrix');
        this.attentionEmpty = document.getElementById('detailAttentionEmpty');
        this.attentionNote = document.getElementById('detailAttentionNote');
        this.attentionValue = document.getElementById('detailAttentionValue');
        this.attentionLegend = document.getElementById('detailAttentionLegend');
        this.attentionLegendLow = document.getElementById('detailAttentionLegendLow');
        this.attentionLegendHigh = document.getElementById('detailAttentionLegendHigh');
        this.engine = options.engine || null;

        if (!this.panel || !this.canvas || !this.title) {
            this.isReady = false;
            return;
        }

        this.isReady = true;
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0x000000);

        this.camera = new THREE.PerspectiveCamera(35, 1, 0.1, 1000);
        this.camera.position.set(0, 0, 220);

        this.renderer = new THREE.WebGLRenderer({ canvas: this.canvas, antialias: true, alpha: false });
        this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
        this.renderer.setClearColor(0x000000, 1);

        this._ambientBaseIntensity = 0.7;
        this.ambientLight = new THREE.AmbientLight(0xffffff, this._ambientBaseIntensity);
        this.keyLight = new THREE.DirectionalLight(0xffffff, 0.9);
        this.keyLight.position.set(25, 40, 40);
        this.scene.add(this.ambientLight, this.keyLight);
        this._environmentTexture = null;
        this._syncEnvironment();

        this.currentPreview = null;
        this.currentDispose = null;
        this.currentAnimator = null;
        this.isOpen = false;
        this._lastFrameTime = performance.now();
        this._mobilePauseActive = false;
        this._mobileFocusActive = false;

        this._animate = this._animate.bind(this);
        this._onResize = this._onResize.bind(this);
        this._onKeydown = this._onKeydown.bind(this);
        this._onClosePointerDown = this._onClosePointerDown.bind(this);
        this._onDocumentPointerDown = this._onDocumentPointerDown.bind(this);
        this._blockPreviewGesture = this._blockPreviewGesture.bind(this);
        this._onAttentionPointerMove = this._onAttentionPointerMove.bind(this);
        this._onAttentionPointerDown = this._onAttentionPointerDown.bind(this);
        this._clearAttentionHover = this._clearAttentionHover.bind(this);
        this._onPanelPointerEnter = this._onPanelPointerEnter.bind(this);
        this._onPanelPointerLeave = this._onPanelPointerLeave.bind(this);
        this._startLoop();

        this.activationSource = options.activationSource || null;
        this.laneTokenIndices = Array.isArray(options.laneTokenIndices) ? options.laneTokenIndices.slice() : null;
        this.tokenLabels = Array.isArray(options.tokenLabels) ? options.tokenLabels.slice() : null;
        this.maxAttentionTokens = Number.isFinite(options.maxAttentionTokens)
            ? Math.max(1, Math.floor(options.maxAttentionTokens))
            : ATTENTION_PREVIEW_MAX_TOKENS;
        this.attentionMode = this.attentionToggle?.checked ? 'post' : 'pre';
        this._attentionContext = null;
        this._attentionTokenElsTop = [];
        this._attentionTokenElsLeft = [];
        this._attentionHoverCell = null;
        this._attentionHoverRow = null;
        this._attentionHoverCol = null;
        this._attentionValueDefault = '';
        this._attentionPinned = false;
        this._attentionPinnedRow = null;
        this._attentionPinnedCol = null;

        this.closeBtn?.addEventListener('click', () => this.close());
        this.closeBtn?.addEventListener('pointerdown', this._onClosePointerDown);
        this.canvas.addEventListener('pointerdown', this._blockPreviewGesture, { passive: false });
        this.canvas.addEventListener('pointermove', this._blockPreviewGesture, { passive: false });
        this.canvas.addEventListener('pointerup', this._blockPreviewGesture, { passive: false });
        this.canvas.addEventListener('wheel', this._blockPreviewGesture, { passive: false });
        this.canvas.addEventListener('touchstart', this._blockPreviewGesture, { passive: false });
        this.canvas.addEventListener('touchmove', this._blockPreviewGesture, { passive: false });
        this.canvas.addEventListener('touchend', this._blockPreviewGesture, { passive: false });
        if (this.attentionToggle) {
            this.attentionToggle.addEventListener('change', () => {
                this.attentionMode = this.attentionToggle.checked ? 'post' : 'pre';
                this._renderAttentionPreview();
            });
        }
        if (this.attentionMatrix) {
            this.attentionMatrix.addEventListener('pointermove', this._onAttentionPointerMove);
            this.attentionMatrix.addEventListener('pointerdown', this._onAttentionPointerDown);
            this.attentionMatrix.addEventListener('pointerleave', this._clearAttentionHover);
        }
        this.panel.addEventListener('pointerenter', this._onPanelPointerEnter);
        this.panel.addEventListener('pointerleave', this._onPanelPointerLeave);
        window.addEventListener('resize', this._onResize);
        document.addEventListener('keydown', this._onKeydown);
        document.addEventListener('pointerdown', this._onDocumentPointerDown, { capture: true });
        this._observeResize();
        this._onResize();
    }

    _observeResize() {
        if (!('ResizeObserver' in window) || !this.canvas?.parentElement) return;
        this._resizeObserver = new ResizeObserver(() => this._onResize());
        this._resizeObserver.observe(this.canvas.parentElement);
    }

    _onResize() {
        if (!this.isReady) return;
        const rect = this.canvas.getBoundingClientRect();
        const width = Math.max(1, Math.floor(rect.width));
        const height = Math.max(1, Math.floor(rect.height));
        this.renderer.setSize(width, height, false);
        this.camera.aspect = width / height;
        this.camera.updateProjectionMatrix();
        this._updateMobileState();
    }

    _onKeydown(event) {
        if (event.key === 'Escape' && this.isOpen) {
            this.close();
        }
    }

    _onClosePointerDown(event) {
        if (!this.isOpen) return;
        event.preventDefault();
        event.stopPropagation();
        this.close();
    }

    _blockPreviewGesture(event) {
        const isTouch = event.pointerType === 'touch' || event.type.startsWith('touch');
        if (!isTouch) return;
        if (event.cancelable) event.preventDefault();
        event.stopPropagation();
    }

    _setHoverLabelSuppression(suppressed) {
        if (this.engine && typeof this.engine.setHoverLabelsSuppressed === 'function') {
            this.engine.setHoverLabelsSuppressed(!!suppressed);
        }
    }

    _isSmallScreen() {
        if (typeof window === 'undefined') return false;
        if (typeof window.matchMedia === 'function') {
            return window.matchMedia('(max-aspect-ratio: 1/1), (max-width: 880px)').matches;
        }
        return window.innerWidth <= 880 || window.innerHeight <= window.innerWidth;
    }

    _updateMobileState() {
        const shouldFocus = this.isOpen && this._isSmallScreen();
        if (shouldFocus !== this._mobilePauseActive) {
            this._mobilePauseActive = shouldFocus;
            if (this.engine) {
                if (shouldFocus) {
                    this.engine.pause?.('detail-mobile');
                } else {
                    this.engine.resume?.('detail-mobile');
                }
            }
        }
        if (shouldFocus === this._mobileFocusActive) return;
        this._mobileFocusActive = shouldFocus;
        if (typeof document === 'undefined' || !document.body) return;
        if (shouldFocus) {
            document.body.classList.add('detail-mobile-focus');
        } else {
            document.body.classList.remove('detail-mobile-focus');
        }
    }

    _onPanelPointerEnter() {
        this._setHoverLabelSuppression(true);
    }

    _onPanelPointerLeave() {
        this._setHoverLabelSuppression(false);
    }

    _setAttentionVisibility(visible) {
        if (!this.attentionRoot) return;
        if (visible) {
            this.attentionRoot.classList.add('is-visible');
            this.attentionRoot.setAttribute('aria-hidden', 'false');
        } else {
            this.attentionRoot.classList.remove('is-visible');
            this.attentionRoot.setAttribute('aria-hidden', 'true');
        }
    }

    _resolveAttentionContext(selection) {
        const label = selection?.label || '';
        if (!isSelfAttentionSelection(label, selection)) return null;
        const headIndex = findUserDataNumber(selection, 'headIndex');
        const layerIndex = findUserDataNumber(selection, 'layerIndex');
        if (!Number.isFinite(headIndex) || !Number.isFinite(layerIndex)) return null;

        let tokenIndices = Array.isArray(this.laneTokenIndices) ? this.laneTokenIndices.slice() : null;
        if (!tokenIndices || !tokenIndices.length) {
            const tokenCount = this.activationSource && typeof this.activationSource.getTokenCount === 'function'
                ? this.activationSource.getTokenCount()
                : 0;
            const labelCount = Array.isArray(this.tokenLabels) ? this.tokenLabels.length : 0;
            const fallbackCount = Math.max(
                0,
                Math.min(this.maxAttentionTokens, tokenCount || labelCount || this.maxAttentionTokens)
            );
            tokenIndices = Array.from({ length: fallbackCount }, (_, idx) => idx);
        }
        if (!tokenIndices.length) return null;

        const totalCount = tokenIndices.length;
        tokenIndices = tokenIndices.slice(0, this.maxAttentionTokens);
        const trimmed = totalCount > tokenIndices.length;

        const tokenLabels = tokenIndices.map((tokenIndex, idx) => {
            let labelText = Array.isArray(this.tokenLabels) ? this.tokenLabels[idx] : null;
            if (!labelText && this.activationSource && typeof this.activationSource.getTokenString === 'function') {
                labelText = this.activationSource.getTokenString(tokenIndex);
            }
            const formatted = formatTokenLabelForPreview(labelText);
            if (formatted) return formatted;
            // If capture data exists, keep blanks as a single space instead of "Token N".
            if (this.activationSource) return SPACE_TOKEN_DISPLAY;
            return `Token ${tokenIndex + 1}`;
        });

        return {
            headIndex,
            layerIndex,
            tokenIndices,
            tokenLabels,
            trimmed,
            totalCount,
            hasSource: !!this.activationSource
        };
    }

    _updateAttentionPreview(selection) {
        if (!this.attentionRoot) return;
        const context = this._resolveAttentionContext(selection);
        this._attentionContext = context;
        const preferredMode = resolveAttentionModeFromSelection(selection);
        if (preferredMode && preferredMode !== this.attentionMode) {
            this.attentionMode = preferredMode;
            if (this.attentionToggle) {
                this.attentionToggle.checked = preferredMode === 'post';
            }
        }
        this._renderAttentionPreview();
    }

    _renderAttentionPreview() {
        if (!this.attentionRoot || !this.attentionMatrix || !this.attentionTokensTop || !this.attentionTokensLeft) return;
        const context = this._attentionContext;
        if (!context || !this.activationSource) {
            this._setAttentionVisibility(false);
            this._clearPinnedAttention();
            return;
        }

        const { tokenIndices, tokenLabels, headIndex, layerIndex, trimmed, totalCount, hasSource } = context;
        const mode = this.attentionMode === 'post' ? 'post' : 'pre';
        this._updateAttentionLegend(mode);
        const values = hasSource
            ? buildAttentionMatrixValues({
                activationSource: this.activationSource,
                layerIndex,
                headIndex,
                tokenIndices,
                mode
            })
            : null;

        this._setAttentionVisibility(true);
        if (this.attentionEmpty) this.attentionEmpty.style.display = 'none';
        if (this.attentionNote) {
            if (!hasSource) {
                this.attentionNote.textContent = 'Attention scores unavailable (no capture loaded).';
            } else {
                this.attentionNote.textContent = trimmed
                    ? `Showing first ${tokenIndices.length} of ${totalCount} tokens`
                    : '';
            }
        }
        if (this.attentionValue) {
            this._attentionValueDefault = hasSource
                ? 'Tap or hover a square to see its score.'
                : '';
            this.attentionValue.textContent = this._attentionValueDefault;
        }

        const count = tokenIndices.length;
        let cellSize = computeAttentionCellSize(count);
        const densityScale = Math.min(1, Math.max(0.35, 8 / Math.max(1, count)));
        const gap = Math.max(1, Math.round(ATTENTION_PREVIEW_GAP * densityScale));
        const gridGap = Math.max(2, Math.round(ATTENTION_PREVIEW_GRID_GAP * densityScale));
        const leftPad = Math.max(2, Math.round(4 * densityScale));
        const leftTokenPad = Math.max(2, Math.round(6 * densityScale));
        const labelWidth = measureMaxTokenLabelWidth(tokenLabels, this.attentionTokensLeft);
        const availableWidth = getContentWidth(this.attentionRoot);
        if (labelWidth > 0 && availableWidth > 0) {
            const usable = availableWidth - labelWidth - gridGap - leftPad - leftTokenPad;
            if (usable > 0) {
                const maxCellByWidth = (usable - (count - 1) * gap) / count;
                if (Number.isFinite(maxCellByWidth) && maxCellByWidth > 0) {
                    cellSize = Math.min(cellSize, maxCellByWidth);
                }
            }
        }
        cellSize = Math.max(ATTENTION_PREVIEW_MIN_CELL, Math.min(ATTENTION_PREVIEW_MAX_CELL, Math.floor(cellSize)));
        this.attentionRoot.style.setProperty('--cell-size', `${cellSize}px`);
        this.attentionRoot.style.setProperty('--cell-gap', `${gap}px`);
        this.attentionRoot.style.setProperty('--attention-grid-gap', `${gridGap}px`);
        this.attentionRoot.style.setProperty('--attention-left-padding', `${leftPad}px`);
        this.attentionRoot.style.setProperty('--attention-left-token-padding', `${leftTokenPad}px`);
        this.attentionRoot.style.setProperty('--attention-matrix-justify', 'center');
        this.attentionTokensTop.style.gridTemplateColumns = `repeat(${count}, ${cellSize}px)`;
        this.attentionTokensTop.style.gap = `${gap}px`;
        this.attentionTokensLeft.style.gridTemplateRows = `repeat(${count}, ${cellSize}px)`;
        this.attentionTokensLeft.style.gap = `${gap}px`;
        this.attentionMatrix.style.gridTemplateColumns = `repeat(${count}, ${cellSize}px)`;
        this.attentionMatrix.style.gridTemplateRows = `repeat(${count}, ${cellSize}px)`;
        this.attentionMatrix.style.gap = `${gap}px`;

        this._attentionTokenElsTop = [];
        this._attentionTokenElsLeft = [];

        this.attentionTokensTop.innerHTML = '';
        this.attentionTokensLeft.innerHTML = '';
        this.attentionMatrix.innerHTML = '';

        const topFrag = document.createDocumentFragment();
        const leftFrag = document.createDocumentFragment();
        const matrixFrag = document.createDocumentFragment();

        for (let i = 0; i < count; i += 1) {
            const topToken = document.createElement('div');
            topToken.className = 'attention-token attention-token-top';
            topToken.textContent = tokenLabels[i];
            topToken.title = tokenLabels[i];
            topFrag.appendChild(topToken);
            this._attentionTokenElsTop.push(topToken);

            const leftToken = document.createElement('div');
            leftToken.className = 'attention-token attention-token-left';
            leftToken.textContent = tokenLabels[i];
            leftToken.title = tokenLabels[i];
            leftFrag.appendChild(leftToken);
            this._attentionTokenElsLeft.push(leftToken);
        }

        let hasAnyValue = false;
        for (let row = 0; row < count; row += 1) {
            for (let col = 0; col < count; col += 1) {
                const cell = document.createElement('div');
                cell.className = 'attention-cell';
                cell.dataset.row = String(row);
                cell.dataset.col = String(col);
                const isVisible = ATTENTION_PREVIEW_TRIANGLE === 'upper'
                    ? col >= row
                    : col <= row;
                const value = values ? values[row]?.[col] : null;
                if (!isVisible) {
                    cell.classList.add('is-hidden');
                } else if (value === null) {
                    cell.classList.add('is-empty');
                } else {
                    const color = mode === 'post' ? mapValueToGrayscale(value) : mapValueToColor(value);
                    cell.style.backgroundColor = colorToCss(color);
                    cell.title = `${tokenLabels[row]} → ${tokenLabels[col]} (${mode === 'post' ? 'post' : 'pre'}): ${value.toFixed(4)}`;
                    cell.dataset.value = String(value);
                    cell.dataset.rowLabel = tokenLabels[row] || '';
                    cell.dataset.colLabel = tokenLabels[col] || '';
                    hasAnyValue = true;
                }
                matrixFrag.appendChild(cell);
            }
        }

        this.attentionTokensTop.appendChild(topFrag);
        this.attentionTokensLeft.appendChild(leftFrag);
        this.attentionMatrix.appendChild(matrixFrag);
        if (this.attentionEmpty) {
            this.attentionEmpty.style.display = hasAnyValue ? 'none' : 'block';
        }
        if (!this._restorePinnedAttentionCell()) {
            this._clearAttentionHover(true);
        }
    }

    _updateAttentionLegend(mode) {
        if (!this.attentionLegend || !this.attentionLegendLow || !this.attentionLegendHigh) return;
        const safeMode = mode === 'post' ? 'post' : 'pre';
        if (this.attentionRoot) {
            this.attentionRoot.dataset.attnMode = safeMode;
        }

        if (safeMode === 'post') {
            const low = colorToCss(mapValueToGrayscale(0));
            const high = colorToCss(mapValueToGrayscale(1));
            this.attentionLegend.style.setProperty('--attention-legend-gradient', `linear-gradient(90deg, ${low}, ${high})`);
            this.attentionLegend.style.setProperty('--attention-legend-mid-opacity', '0');
            this.attentionLegend.dataset.mid = '';
            this.attentionLegendLow.textContent = '0';
            this.attentionLegendHigh.textContent = '1';
            return;
        }

        const low = colorToCss(mapValueToColor(-2));
        const mid = colorToCss(mapValueToColor(0));
        const high = colorToCss(mapValueToColor(2));
        this.attentionLegend.style.setProperty('--attention-legend-gradient', `linear-gradient(90deg, ${low}, ${mid}, ${high})`);
        this.attentionLegend.style.setProperty('--attention-legend-mid-opacity', '1');
        this.attentionLegend.dataset.mid = '0';
        this.attentionLegendLow.textContent = '-2';
        this.attentionLegendHigh.textContent = '+2';
    }

    _setAttentionHoverFromCell(cell, { force = false } = {}) {
        if (!cell || cell.classList.contains('is-empty') || cell.classList.contains('is-hidden')) {
            this._clearAttentionHover(force);
            return;
        }
        const row = Number(cell.dataset.row);
        const col = Number(cell.dataset.col);
        if (row === this._attentionHoverRow && col === this._attentionHoverCol) return;

        this._clearAttentionHover(force);
        this._attentionHoverCell = cell;
        this._attentionHoverRow = row;
        this._attentionHoverCol = col;
        cell.classList.add('is-hovered');
        const leftToken = this._attentionTokenElsLeft[row];
        const topToken = this._attentionTokenElsTop[col];
        if (leftToken) leftToken.classList.add('is-highlighted');
        if (topToken) topToken.classList.add('is-highlighted');
        if (this.attentionValue) {
            const rawValue = cell.dataset.value;
            const valueNum = Number(rawValue);
            const rowLabel = cell.dataset.rowLabel || '';
            const colLabel = cell.dataset.colLabel || '';
            const label = rowLabel || colLabel
                ? `${rowLabel || 'Token'} → ${colLabel || 'Token'}`
                : 'Score';
            const scoreText = Number.isFinite(valueNum) ? valueNum.toFixed(4) : String(rawValue || '');
            this.attentionValue.textContent = `${label}: ${scoreText}`;
        }
    }

    _onAttentionPointerMove(event) {
        if (this._attentionPinned) return;
        const target = event.target;
        const cell = target && typeof target.closest === 'function'
            ? target.closest('.attention-cell')
            : null;
        if (!cell || !this.attentionMatrix || !this.attentionMatrix.contains(cell)) {
            this._clearAttentionHover();
            return;
        }
        this._setAttentionHoverFromCell(cell);
    }

    _onAttentionPointerDown(event) {
        const target = event.target;
        const cell = target && typeof target.closest === 'function'
            ? target.closest('.attention-cell')
            : null;
        const isDirectManipulation = event.pointerType === 'touch' || event.pointerType === 'pen';
        if (!cell || !this.attentionMatrix || !this.attentionMatrix.contains(cell)) {
            if (isDirectManipulation) {
                this._clearPinnedAttention();
                return;
            }
            this._clearAttentionHover();
            return;
        }
        if (isDirectManipulation) {
            const row = Number(cell.dataset.row);
            const col = Number(cell.dataset.col);
            if (this._attentionPinned && row === this._attentionPinnedRow && col === this._attentionPinnedCol) {
                this._clearPinnedAttention();
                return;
            }
            this._attentionPinned = true;
            this._attentionPinnedRow = row;
            this._attentionPinnedCol = col;
            this._setAttentionHoverFromCell(cell, { force: true });
            return;
        }
        this._setAttentionHoverFromCell(cell);
    }

    _clearAttentionHover(force = false) {
        const forceFlag = force === true;
        if (this._attentionPinned && !forceFlag) return;
        if (this._attentionHoverCell) {
            this._attentionHoverCell.classList.remove('is-hovered');
        }
        if (Number.isFinite(this._attentionHoverRow)) {
            const leftToken = this._attentionTokenElsLeft[this._attentionHoverRow];
            if (leftToken) leftToken.classList.remove('is-highlighted');
        }
        if (Number.isFinite(this._attentionHoverCol)) {
            const topToken = this._attentionTokenElsTop[this._attentionHoverCol];
            if (topToken) topToken.classList.remove('is-highlighted');
        }
        this._attentionHoverCell = null;
        this._attentionHoverRow = null;
        this._attentionHoverCol = null;
        if (this.attentionValue) {
            this.attentionValue.textContent = this._attentionValueDefault || '';
        }
    }

    _clearPinnedAttention() {
        this._attentionPinned = false;
        this._attentionPinnedRow = null;
        this._attentionPinnedCol = null;
        this._clearAttentionHover(true);
    }

    _restorePinnedAttentionCell() {
        if (!this._attentionPinned || !this.attentionMatrix) return false;
        if (!Number.isFinite(this._attentionPinnedRow) || !Number.isFinite(this._attentionPinnedCol)) return false;
        const selector = `.attention-cell[data-row="${this._attentionPinnedRow}"][data-col="${this._attentionPinnedCol}"]`;
        const cell = this.attentionMatrix.querySelector(selector);
        if (!cell || cell.classList.contains('is-empty') || cell.classList.contains('is-hidden')) {
            this._clearPinnedAttention();
            return false;
        }
        this._setAttentionHoverFromCell(cell, { force: true });
        return true;
    }

    _onDocumentPointerDown(event) {
        if (!this.isOpen || !this.closeBtn) return;
        if (!Number.isFinite(event.clientX) || !Number.isFinite(event.clientY)) return;
        if (event.target === this.closeBtn) return;
        const hit = document.elementFromPoint(event.clientX, event.clientY);
        if (!hit || typeof hit.closest !== 'function') return;
        if (hit.closest('#detailClose') !== this.closeBtn) return;
        // Close even if the canvas captured the pointer event.
        event.preventDefault();
        event.stopPropagation();
        this.close();
    }

    _syncEnvironment() {
        const env = appState.environmentTexture;
        if (env && this._environmentTexture !== env) {
            this.scene.environment = env;
            this._environmentTexture = env;
            if (this.ambientLight) this.ambientLight.intensity = 0.0;
            this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
            this.renderer.toneMappingExposure = 1.0;
        } else if (!env && this._environmentTexture) {
            this.scene.environment = null;
            this._environmentTexture = null;
            if (this.ambientLight) this.ambientLight.intensity = this._ambientBaseIntensity;
            this.renderer.toneMapping = THREE.NoToneMapping;
            this.renderer.toneMappingExposure = 1.0;
        }
    }

    _startLoop() {
        if (this._loopStarted) return;
        this._loopStarted = true;
        requestAnimationFrame(this._animate);
    }

    _animate(time) {
        requestAnimationFrame(this._animate);
        const now = (typeof time === 'number') ? time : performance.now();
        const deltaMs = this._lastFrameTime ? (now - this._lastFrameTime) : 16.6667;
        this._lastFrameTime = now;

        if (!this.isReady || !this.isOpen || !this.currentPreview) return;
        this._syncEnvironment();

        if (typeof this.currentAnimator === 'function') {
            try {
                this.currentAnimator(deltaMs, now);
            } catch (err) {
                // Keep selection preview resilient to animation errors.
                console.warn('Selection preview animation error:', err);
            }
        }

        const rotationStep = PREVIEW_ROTATION_SPEED * (deltaMs / 16.6667);
        this.currentPreview.rotation.y += rotationStep;
        const timeSeconds = now * 0.001;
        this.currentPreview.rotation.x = PREVIEW_BASE_TILT_X
            + Math.sin(timeSeconds * PREVIEW_TILT_OSC_SPEED) * PREVIEW_TILT_AMPLITUDE;
        this.currentPreview.rotation.z = 0;
        this.renderer.render(this.scene, this.camera);
    }

    open() {
        if (!this.isReady) return;
        this.isOpen = true;
        this.panel.classList.add('is-open');
        this.hudPanel?.classList.add('detail-open');
        this.panel.setAttribute('aria-hidden', 'false');
        this._updateMobileState();
    }

    close() {
        if (!this.isReady) return;
        this.isOpen = false;
        this.panel.classList.remove('is-open');
        this.hudPanel?.classList.remove('detail-open');
        this.panel.setAttribute('aria-hidden', 'true');
        this._setHoverLabelSuppression(false);
        this._updateMobileState();
        if (this.description) this.description.textContent = '';
    }

    showSelection(selection) {
        if (!this.isReady || !selection || !selection.label) return;

        const label = selection.label;
        const metadata = resolveMetadata(label, selection.kind);
        this.title.textContent = label;
        if (this.params) this.params.textContent = metadata.params;
        if (this.dims) this.dims.textContent = metadata.dims;
        if (this.description) {
            const desc = resolveDescription(label, selection.kind, selection);
            this.description.textContent = desc || '';
        }
        const isParam = isParameterSelection(label);
        if (this.dataSection) {
            this.dataSection.style.display = isParam ? 'none' : '';
        }
        if (this.dataEl && isParam) {
            this.dataEl.textContent = '';
        }
        if (this.metaSection && this.attentionRoot && this.panel) {
            if (isQkvMatrixLabel(label)) {
                if (this.attentionRoot.parentElement === this.panel) {
                    this.panel.insertBefore(this.metaSection, this.attentionRoot);
                }
            } else if (this.dataSection && this.dataSection.parentElement === this.panel) {
                this.panel.insertBefore(this.metaSection, this.dataSection);
            }
        }
        if (this.dataEl) {
            const activationData = (selection.object && selection.object.userData && selection.object.userData.activationData)
                || (selection.info && selection.info.activationData)
                || (selection.hit && selection.hit.object && selection.hit.object.userData && selection.hit.object.userData.activationData)
                || null;
            if (!isParam) {
                this.dataEl.textContent = formatActivationData(activationData);
            }
        }

        if (this.currentPreview) {
            this.scene.remove(this.currentPreview);
            if (this.currentDispose) {
                try { this.currentDispose(); } catch (_) { /* no-op */ }
            }
            this.currentPreview = null;
            this.currentDispose = null;
            this.currentAnimator = null;
        }

        const preview = resolvePreviewObject(label, selection);
        this.currentPreview = preview.object;
        this.currentDispose = preview.dispose;
        this.currentAnimator = preview.animate || null;
        const desiredRotation = new THREE.Euler(PREVIEW_BASE_TILT_X, PREVIEW_BASE_ROTATION_Y, 0);
        if (this.currentPreview?.rotation) {
            this.currentPreview.rotation.set(0, 0, 0);
        }
        this._lastFrameTime = performance.now();
        const isVectorPreview = isLikelyVectorSelection(label, selection);
        const isQkvPreview = isQkvMatrixLabel(label);
        const paddingMultiplier = isVectorPreview
            ? PREVIEW_VECTOR_PADDING_MULT
            : (isQkvPreview ? 0.75 : 1);
        const distanceMultiplier = isVectorPreview
            ? PREVIEW_VECTOR_DISTANCE_MULT
            : (isQkvPreview ? 0.75 : 1);
        fitObjectToView(this.currentPreview, this.camera, { paddingMultiplier, distanceMultiplier });
        if (this.currentPreview?.rotation) {
            this.currentPreview.rotation.copy(desiredRotation);
        }
        this.scene.add(this.currentPreview);

        this._updateAttentionPreview(selection);
        this.open();
    }
}

export function initSelectionPanel(options = {}) {
    requestTokenChipFont();
    const panel = new SelectionPanel(options);
    if (!panel.isReady) {
        return { handleSelection: () => {}, close: () => {} };
    }
    return {
        handleSelection: (selection) => panel.showSelection(selection),
        close: () => panel.close()
    };
}
