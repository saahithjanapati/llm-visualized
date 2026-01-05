import * as THREE from 'three';
import { FontLoader } from 'three/examples/jsm/loaders/FontLoader.js';
import { WeightMatrixVisualization } from '../components/WeightMatrixVisualization.js';
import { VectorVisualizationInstancedPrism } from '../components/VectorVisualizationInstancedPrism.js';
import { LayerNormalizationVisualization } from '../components/LayerNormalizationVisualization.js';
import { appState } from '../state/appState.js';
import { createSciFiMaterial, updateSciFiMaterialColor } from '../utils/sciFiMaterial.js';
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
    applyFinalColorToObject(clone, finalColor);
    return {
        object: clone,
        dispose: () => {
            previewGeometries.forEach((geo) => geo && geo.dispose && geo.dispose());
            previewMaterials.forEach((mat) => mat && mat.dispose && mat.dispose());
        }
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

function extractTokenText(label) {
    if (!label) return '';
    const match = label.match(/^(token|position)\s*:\s*(.*)$/i);
    if (!match) return label.trim();
    return (match[2] || '').trim();
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
    const text = rawText.trim().length ? rawText : 'Token';
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
    const depth = Number.isFinite(params.depth) ? params.depth : PREVIEW_MATRIX_DEPTH;
    const slitCount = Number.isFinite(params.numberOfSlits) ? params.numberOfSlits : PREVIEW_LANES;
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
        params.slitTopWidthFactor
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
    const matrixPreview = buildSelectionClonePreview(selectionInfo, label)
        || buildWeightMatrixPreview(MHA_MATRIX_PARAMS, matrixColor);
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
    for (let i = 0; i < PREVIEW_LANES; i++) {
        const material = new THREE.MeshStandardMaterial({
            color: colorHex || 0x1f1f1f,
            metalness: 0.25,
            roughness: 0.65,
            emissive: new THREE.Color(0x060606),
            emissiveIntensity: 0.3
        });
        const mesh = new THREE.Mesh(geometry, material);
        mesh.position.z = (i - (PREVIEW_LANES - 1) / 2) * 18;
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
    const clonePreview = buildSelectionClonePreview(selectionInfo, label);
    if (clonePreview?.object) {
        applyFinalColorToObject(clonePreview.object, 0xffffff);
        return clonePreview;
    }

    const params = LN_PARAMS;
    const ln = new LayerNormalizationVisualization(
        new THREE.Vector3(0, 0, 0),
        params.width,
        params.height,
        params.depth,
        params.wallThickness,
        params.numberOfHoles,
        params.holeWidth,
        params.holeWidthFactor
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

function resolvePreviewObject(label, selectionInfo) {
    const lower = (label || '').toLowerCase();
    const vectorClone = buildVectorClonePreview(selectionInfo);
    if (vectorClone) return vectorClone;
    if (lower.startsWith('token:') || lower.startsWith('position:')) {
        const clonePreview = buildSelectionClonePreview(selectionInfo, label);
        if (clonePreview) return clonePreview;
        return buildTokenChipPreview(extractTokenText(label));
    }
    if (isQkvMatrixLabel(lower)) {
        return buildQkvMatrixFlowPreview(label, selectionInfo);
    }
    if (isWeightMatrixLabel(lower)) {
        const clonePreview = buildSelectionClonePreview(selectionInfo, label);
        if (clonePreview) return clonePreview;
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
        const color = lower.includes('top') ? FINAL_VOCAB_TOP_COLOR : MHA_FINAL_Q_COLOR;
        return buildWeightMatrixPreview(EMBEDDING_MATRIX_PARAMS_VOCAB, color);
    }
    if (lower.includes('positional embedding')) {
        return buildWeightMatrixPreview(EMBEDDING_MATRIX_PARAMS_POSITION, MHA_FINAL_K_COLOR);
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
    constructor() {
        this.panel = document.getElementById('detailPanel');
        this.hudPanel = document.getElementById('hudPanel');
        this.title = document.getElementById('detailTitle');
        this.params = document.getElementById('detailParams');
        this.dims = document.getElementById('detailDims');
        this.closeBtn = document.getElementById('detailClose');
        this.canvas = document.getElementById('detailCanvas');
        this.dataEl = document.getElementById('detailData');

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

        this._animate = this._animate.bind(this);
        this._onResize = this._onResize.bind(this);
        this._onKeydown = this._onKeydown.bind(this);
        this._startLoop();

        this.closeBtn?.addEventListener('click', () => this.close());
        window.addEventListener('resize', this._onResize);
        document.addEventListener('keydown', this._onKeydown);
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
    }

    _onKeydown(event) {
        if (event.key === 'Escape' && this.isOpen) {
            this.close();
        }
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
    }

    close() {
        if (!this.isReady) return;
        this.isOpen = false;
        this.panel.classList.remove('is-open');
        this.hudPanel?.classList.remove('detail-open');
        this.panel.setAttribute('aria-hidden', 'true');
    }

    showSelection(selection) {
        if (!this.isReady || !selection || !selection.label) return;

        const label = selection.label;
        const metadata = resolveMetadata(label, selection.kind);
        this.title.textContent = label;
        if (this.params) this.params.textContent = metadata.params;
        if (this.dims) this.dims.textContent = metadata.dims;
        if (this.dataEl) {
            const activationData = (selection.object && selection.object.userData && selection.object.userData.activationData)
                || (selection.info && selection.info.activationData)
                || (selection.hit && selection.hit.object && selection.hit.object.userData && selection.hit.object.userData.activationData)
                || null;
            this.dataEl.textContent = formatActivationData(activationData);
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
        const paddingMultiplier = isVectorPreview ? PREVIEW_VECTOR_PADDING_MULT : 1;
        const distanceMultiplier = isVectorPreview ? PREVIEW_VECTOR_DISTANCE_MULT : 1;
        fitObjectToView(this.currentPreview, this.camera, { paddingMultiplier, distanceMultiplier });
        if (this.currentPreview?.rotation) {
            this.currentPreview.rotation.copy(desiredRotation);
        }
        this.scene.add(this.currentPreview);

        this.open();
    }
}

export function initSelectionPanel() {
    requestTokenChipFont();
    const panel = new SelectionPanel();
    if (!panel.isReady) {
        return { handleSelection: () => {}, close: () => {} };
    }
    return {
        handleSelection: (selection) => panel.showSelection(selection),
        close: () => panel.close()
    };
}
