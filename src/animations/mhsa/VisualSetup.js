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
} from '../../utils/constants.js';

import {
    MHSA_MATRIX_INITIAL_RESTING_COLOR,
    MHA_OUTPUT_PROJECTION_MATRIX_Y_OFFSET_ABOVE_ROW,
    MHA_OUTPUT_PROJECTION_MATRIX_PARAMS,
    MHA_OUTPUT_PROJECTION_MATRIX_COLOR,
} from '../LayerAnimationConstants.js';

import { INACTIVE_COMPONENT_COLOR } from '../../utils/constants.js';

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
        const mats = [matrix.mesh?.material, matrix.frontCapMesh?.material, matrix.backCapMesh?.material];
        mats.forEach((mat) => {
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

    for (let i = 0; i < NUM_HEAD_SETS_LAYER; i++) {
        const headSetWidth       = MHA_INTERNAL_MATRIX_SPACING * 2 + MHA_MATRIX_PARAMS.width;
        const currentHeadSetX    = branchX - MHA_INTERNAL_MATRIX_SPACING + i * (headSetWidth + HEAD_SET_GAP_LAYER);

        const x_q = currentHeadSetX;
        const x_k = currentHeadSetX + MHA_INTERNAL_MATRIX_SPACING;
        const x_v = currentHeadSetX + MHA_INTERNAL_MATRIX_SPACING * 2;

        const buildMatrix = (xPos, label, headIndex) => {
            const mat = new WeightMatrixVisualization(
                null,
                new THREE.Vector3(xPos, matrixCenterY, 0),
                MHA_MATRIX_PARAMS.width,
                MHA_MATRIX_PARAMS.height,
                MHA_MATRIX_PARAMS.depth,
                MHA_MATRIX_PARAMS.topWidthFactor,
                MHA_MATRIX_PARAMS.cornerRadius,
                MHA_MATRIX_PARAMS.numberOfSlits,
                MHA_MATRIX_PARAMS.slitWidth,
                MHA_MATRIX_PARAMS.slitDepthFactor,
                MHA_MATRIX_PARAMS.slitBottomWidthFactor,
                MHA_MATRIX_PARAMS.slitTopWidthFactor
            );
            mat.setColor(inactiveMatrixColor);
            mat.group.userData.label = label;
            mat.group.userData.headIndex = headIndex;
            if (Number.isFinite(layerIndex)) mat.group.userData.layerIndex = layerIndex;
            if (mat.mesh)         mat.mesh.userData.label        = label;
            if (mat.mesh) {
                mat.mesh.userData.headIndex = headIndex;
                if (Number.isFinite(layerIndex)) mat.mesh.userData.layerIndex = layerIndex;
            }
            if (mat.frontCapMesh) {
                mat.frontCapMesh.userData.label = label;
                mat.frontCapMesh.userData.headIndex = headIndex;
                if (Number.isFinite(layerIndex)) mat.frontCapMesh.userData.layerIndex = layerIndex;
            }
            if (mat.backCapMesh)  {
                mat.backCapMesh.userData.label  = label;
                mat.backCapMesh.userData.headIndex = headIndex;
                if (Number.isFinite(layerIndex)) mat.backCapMesh.userData.layerIndex = layerIndex;
            }

            // Make heavy matrix materials opaque by default when fully opaque to avoid sorting
            const wantsTransparency = matrixRestingOpacity < 1.0;
            mat.setMaterialProperties({
                opacity: matrixRestingOpacity,
                transparent: wantsTransparency,
                emissiveIntensity: 0.08,
            });

            parentGroup.add(mat.group);
            mhaVisualizations.push(mat);
            return mat;
        };

        const qMatrix = buildMatrix(x_q, 'Query Weight Matrix', i);
        const kMatrix = buildMatrix(x_k, 'Key Weight Matrix', i);
        const vMatrix = buildMatrix(x_v, 'Value Weight Matrix', i);

        tuneInactiveMatrix(qMatrix);
        tuneInactiveMatrix(kMatrix);
        tuneInactiveMatrix(vMatrix);
        softenMatrixSurface(qMatrix, QKV_SURFACE_TWEAKS);
        softenMatrixSurface(kMatrix, QKV_SURFACE_TWEAKS);
        softenMatrixSurface(vMatrix, QKV_SURFACE_TWEAKS);

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

    const outputProjectionMatrix = new WeightMatrixVisualization(
        null,
        new THREE.Vector3(firstHeadKMatrixX, outputProjMatrixCenterY, 0),
        MHA_OUTPUT_PROJECTION_MATRIX_PARAMS.width,
        matrixHeight,
        MHA_MATRIX_PARAMS.depth,
        MHA_OUTPUT_PROJECTION_MATRIX_PARAMS.topWidthFactor,
        MHA_OUTPUT_PROJECTION_MATRIX_PARAMS.cornerRadius,
        NUM_VECTOR_LANES,
        MHA_OUTPUT_PROJECTION_MATRIX_PARAMS.slitWidth,
        MHA_OUTPUT_PROJECTION_MATRIX_PARAMS.slitDepthFactor,
        MHA_OUTPUT_PROJECTION_MATRIX_PARAMS.slitBottomWidthFactor,
        MHA_OUTPUT_PROJECTION_MATRIX_PARAMS.slitTopWidthFactor
    );

    const initDarkColor = new THREE.Color(MHSA_MATRIX_INITIAL_RESTING_COLOR);
    outputProjectionMatrix.setColor(initDarkColor);
    outputProjectionMatrix.group.userData.label = 'Output Projection Matrix';
    if (outputProjectionMatrix.mesh)         outputProjectionMatrix.mesh.userData.label        = 'Output Projection Matrix';
    if (outputProjectionMatrix.frontCapMesh) outputProjectionMatrix.frontCapMesh.userData.label = 'Output Projection Matrix';
    if (outputProjectionMatrix.backCapMesh)  outputProjectionMatrix.backCapMesh.userData.label  = 'Output Projection Matrix';

    outputProjectionMatrix.group.children.forEach(child => {
        if (child.material) {
            child.material.opacity          = 1.0;
            child.material.transparent      = false;
            child.material.emissive         = initDarkColor;
            child.material.emissiveIntensity = scaleGlobalEmissiveIntensity(0.16);
        }
    });
    softenMatrixSurface(outputProjectionMatrix);

    parentGroup.add(outputProjectionMatrix.group);

    const matrixTopY     = outputProjMatrixCenterY + matrixHeight / 2;
    const finalCombinedY = matrixTopY + 60; // matches legacy comment
    const finalOriginalY = finalCombinedY - ORIGINAL_TO_PROCESSED_GAP;

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
