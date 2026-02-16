import * as THREE from 'three';

// Visual dependencies
import { WeightMatrixVisualization } from '../../components/WeightMatrixVisualization.js';
import { updateSciFiMaterialUniforms } from '../../utils/sciFiMaterial.js';
import { scaleGlobalEmissiveIntensity } from '../../utils/materialUtils.js';

// Animation constants
import {
    MHA_MATRIX_PARAMS,
    NUM_HEAD_SETS_LAYER,
    HEAD_SET_GAP_LAYER,
    MHA_INTERNAL_MATRIX_SPACING,
    NUM_VECTOR_LANES,
    PRISM_DIMENSIONS_PER_UNIT,
    ORIGINAL_TO_PROCESSED_GAP,
    MHSA_RESIDUAL_ADDITION_EXTRA_GAP,
} from '../../utils/constants.js';

import {
    MHSA_MATRIX_INITIAL_RESTING_COLOR,
    MHA_OUTPUT_PROJECTION_MATRIX_Y_OFFSET_ABOVE_ROW,
    MHA_OUTPUT_PROJECTION_MATRIX_PARAMS,
    MHA_OUTPUT_PROJECTION_MATRIX_COLOR,
} from '../LayerAnimationConstants.js';

import { INACTIVE_COMPONENT_COLOR } from '../../utils/constants.js';

const OUTPUT_PROJECTION_LABEL = 'Output Projection Matrix';
const QKV_LABELS = Object.freeze({
    q: 'Query Weight Matrix',
    k: 'Key Weight Matrix',
    v: 'Value Weight Matrix',
});

function withMaterialArray(material, callback) {
    if (!material || typeof callback !== 'function') return;
    const mats = Array.isArray(material) ? material : [material];
    mats.forEach((mat) => {
        if (!mat) return;
        callback(mat);
    });
}

function forEachMatrixMaterial(matrix, callback) {
    if (!matrix || typeof callback !== 'function') return;
    withMaterialArray(matrix.mesh?.material, callback);
    withMaterialArray(matrix.frontCapMesh?.material, callback);
    withMaterialArray(matrix.backCapMesh?.material, callback);
}

function forEachGroupChildMaterial(group, callback) {
    if (!group || !Array.isArray(group.children) || typeof callback !== 'function') return;
    group.children.forEach((child) => {
        withMaterialArray(child?.material, callback);
    });
}

function applyMatrixUserData(matrix, data) {
    if (!matrix || !data) return;
    const targets = [matrix.group, matrix.mesh, matrix.frontCapMesh, matrix.backCapMesh];
    targets.forEach((target) => {
        if (!target) return;
        target.userData = target.userData || {};
        Object.assign(target.userData, data);
    });
}

function createWeightMatrixAt(position, {
    width,
    height,
    depth,
    topWidthFactor,
    cornerRadius,
    numberOfSlits,
    slitWidth,
    slitDepthFactor,
    slitBottomWidthFactor,
    slitTopWidthFactor
}) {
    return new WeightMatrixVisualization(
        null,
        position,
        width,
        height,
        depth,
        topWidthFactor,
        cornerRadius,
        numberOfSlits,
        slitWidth,
        slitDepthFactor,
        slitBottomWidthFactor,
        slitTopWidthFactor
    );
}

/**
 * Build all static visual elements required for a Multi-Head Self-Attention (MHSA) layer.
 * Returns an object with references that the rest of the animation pipeline will need.
 *
 * NOTE: This module is *purely* about construction – no tweens, no per-frame updates.
 */
export function buildMHAVisuals(parentGroup, {
    branchX = 0,
    mhsaBaseY = 0,
    matrixRestingOpacity = 1.0,
    layerIndex = null,
} = {}) {
    const mhaVisualizations = [];
    const headsCentersX      = [];
    const headCoords         = [];

    // ------------------------------------------------------------
    // 1) Build Q / K / V weight matrices for every head set
    // ------------------------------------------------------------
    const matrixCenterY = mhsaBaseY + MHA_MATRIX_PARAMS.height / 2;
    const inactiveMatrixColor = new THREE.Color(INACTIVE_COMPONENT_COLOR);

    const inactiveMatrixUniforms = {
        stripeStrength: 0.0,
        scanlineStrength: 0.0,
        rimIntensity: 0.28,
        depthAccentStrength: 0.08,
        glintStrength: 0.04
    };
    const tuneInactiveMatrix = (matrix) => {
        if (!matrix) return;
        updateSciFiMaterialUniforms(matrix.mesh?.material, inactiveMatrixUniforms);
        updateSciFiMaterialUniforms(matrix.frontCapMesh?.material, inactiveMatrixUniforms);
        updateSciFiMaterialUniforms(matrix.backCapMesh?.material, inactiveMatrixUniforms);
    };

    const softenedMatrixUniforms = {
        stripeStrength: 0.0,
        scanlineStrength: 0.0,
        glintStrength: 0.0,
        noiseStrength: 0.0,
        rimIntensity: 0.42,
        depthAccentStrength: 0.12
    };

    const softenMatrixSurface = (matrix, {
        roughnessMin = 0.24,
        metalnessMax = null,
        clearcoatMax = 0.65,
        clearcoatRoughnessMin = 0.22,
        iridescenceMax = 0.4,
        envMapIntensityMax = 1.3
    } = {}) => {
        if (!matrix) return;
        forEachMatrixMaterial(matrix, (mat) => {
            if (!mat) return;
            if (typeof mat.roughness === 'number') mat.roughness = Math.max(mat.roughness, roughnessMin);
            if (typeof metalnessMax === 'number' && typeof mat.metalness === 'number') {
                mat.metalness = Math.min(mat.metalness, metalnessMax);
            }
            if (typeof mat.clearcoat === 'number') mat.clearcoat = Math.min(mat.clearcoat, clearcoatMax);
            if (typeof mat.clearcoatRoughness === 'number') {
                mat.clearcoatRoughness = Math.max(mat.clearcoatRoughness, clearcoatRoughnessMin);
            }
            if (typeof mat.iridescence === 'number') mat.iridescence = Math.min(mat.iridescence, iridescenceMax);
            if (typeof mat.envMapIntensity === 'number') {
                mat.envMapIntensity = Math.min(mat.envMapIntensity, envMapIntensityMax);
            }
        });
        updateSciFiMaterialUniforms(matrix.mesh?.material, softenedMatrixUniforms);
        updateSciFiMaterialUniforms(matrix.frontCapMesh?.material, softenedMatrixUniforms);
        updateSciFiMaterialUniforms(matrix.backCapMesh?.material, softenedMatrixUniforms);
    };

    const QKV_SURFACE_TWEAKS = {
        roughnessMin: 0.5,
        metalnessMax: 0.6,
        clearcoatMax: 0.45,
        clearcoatRoughnessMin: 0.45,
        iridescenceMax: 0.22,
        envMapIntensityMax: 0.85
    };

    const buildHeadMatrix = (xPos, label, headIndex) => {
        const matrix = createWeightMatrixAt(
            new THREE.Vector3(xPos, matrixCenterY, 0),
            MHA_MATRIX_PARAMS
        );

        matrix.setColor(inactiveMatrixColor);
        applyMatrixUserData(matrix, {
            label,
            headIndex
        });
        if (Number.isFinite(layerIndex)) {
            applyMatrixUserData(matrix, { layerIndex });
        }

        const wantsTransparency = matrixRestingOpacity < 1.0;
        matrix.setMaterialProperties({
            opacity: matrixRestingOpacity,
            transparent: wantsTransparency,
            emissiveIntensity: 0.08,
        });

        parentGroup.add(matrix.group);
        mhaVisualizations.push(matrix);
        tuneInactiveMatrix(matrix);
        softenMatrixSurface(matrix, QKV_SURFACE_TWEAKS);
        return matrix;
    };

    for (let i = 0; i < NUM_HEAD_SETS_LAYER; i++) {
        const headSetWidth       = MHA_INTERNAL_MATRIX_SPACING * 2 + MHA_MATRIX_PARAMS.width;
        const currentHeadSetX    = branchX - MHA_INTERNAL_MATRIX_SPACING + i * (headSetWidth + HEAD_SET_GAP_LAYER);

        const x_q = currentHeadSetX;
        const x_k = currentHeadSetX + MHA_INTERNAL_MATRIX_SPACING;
        const x_v = currentHeadSetX + MHA_INTERNAL_MATRIX_SPACING * 2;

        buildHeadMatrix(x_q, QKV_LABELS.q, i);
        buildHeadMatrix(x_k, QKV_LABELS.k, i);
        buildHeadMatrix(x_v, QKV_LABELS.v, i);

        headsCentersX.push(x_k);
        headCoords.push({ q: x_q, k: x_k, v: x_v });
    }

    // ------------------------------------------------------------
    // 2) Build the Output-Projection matrix (sits above merged row)
    // ------------------------------------------------------------
    const firstHeadKMatrixX = headCoords.length ? headCoords[0].k : branchX;

    const postPassThroughBaseY = mhsaBaseY + MHA_MATRIX_PARAMS.height + 20; // reused from original calculation
    const decorativeVectorsY   = postPassThroughBaseY + 60;
    const matrixHeight         = MHA_MATRIX_PARAMS.height * MHA_OUTPUT_PROJECTION_MATRIX_PARAMS.heightFactor;
    const outputProjMatrixCenterY = decorativeVectorsY + MHA_OUTPUT_PROJECTION_MATRIX_Y_OFFSET_ABOVE_ROW + matrixHeight / 2;

    const outputProjectionMatrix = createWeightMatrixAt(
        new THREE.Vector3(firstHeadKMatrixX, outputProjMatrixCenterY, 0),
        {
            ...MHA_OUTPUT_PROJECTION_MATRIX_PARAMS,
            height: matrixHeight,
            depth: MHA_MATRIX_PARAMS.depth,
            numberOfSlits: NUM_VECTOR_LANES,
        }
    );

    const initDarkColor = new THREE.Color(MHSA_MATRIX_INITIAL_RESTING_COLOR);
    outputProjectionMatrix.setColor(initDarkColor);
    applyMatrixUserData(outputProjectionMatrix, { label: OUTPUT_PROJECTION_LABEL });

    forEachGroupChildMaterial(outputProjectionMatrix.group, (material) => {
        material.opacity = 1.0;
        material.transparent = false;
        material.emissive = initDarkColor;
        material.emissiveIntensity = scaleGlobalEmissiveIntensity(0.16);
    });
    // Keep output-projection reflectivity consistent with Q/K/V (and MLP) matrices.
    softenMatrixSurface(outputProjectionMatrix, QKV_SURFACE_TWEAKS);

    parentGroup.add(outputProjectionMatrix.group);

    const matrixTopY     = outputProjMatrixCenterY + matrixHeight / 2;
    const finalCombinedY = matrixTopY + 60; // matches legacy comment
    const finalOriginalY = finalCombinedY - ORIGINAL_TO_PROCESSED_GAP - MHSA_RESIDUAL_ADDITION_EXTRA_GAP;

    return {
        // raw visual arrays & helpers
        mhaVisualizations,
        headsCentersX,
        headCoords,

        // Output-projection specifics
        outputProjectionMatrix,
        outputProjMatrixCenterY,
        outputProjMatrixHeight: matrixHeight,
        outputProjMatrixDefaultColor : initDarkColor,
        outputProjMatrixActiveColor  : new THREE.Color(MHA_OUTPUT_PROJECTION_MATRIX_COLOR),

        // Pre-computed Y positions useful for later stages
        finalCombinedY,
        finalOriginalY,
    };
}
