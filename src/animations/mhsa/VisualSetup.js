import * as THREE from 'three';

// Visual dependencies
import { WeightMatrixVisualization } from '../../components/WeightMatrixVisualization.js';
import { updateSciFiMaterialUniforms } from '../../utils/sciFiMaterial.js';

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

    for (let i = 0; i < NUM_HEAD_SETS_LAYER; i++) {
        const headSetWidth       = MHA_INTERNAL_MATRIX_SPACING * 2 + MHA_MATRIX_PARAMS.width;
        const currentHeadSetX    = branchX - MHA_INTERNAL_MATRIX_SPACING + i * (headSetWidth + HEAD_SET_GAP_LAYER);

        const x_q = currentHeadSetX;
        const x_k = currentHeadSetX + MHA_INTERNAL_MATRIX_SPACING;
        const x_v = currentHeadSetX + MHA_INTERNAL_MATRIX_SPACING * 2;

        const buildMatrix = (xPos, label) => {
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
                MHA_MATRIX_PARAMS.slitTopWidthFactor,
            );
            mat.setColor(inactiveMatrixColor);
            mat.group.userData.label = label;
            if (mat.mesh)         mat.mesh.userData.label        = label;
            if (mat.frontCapMesh) mat.frontCapMesh.userData.label = label;
            if (mat.backCapMesh)  mat.backCapMesh.userData.label  = label;

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

        const qMatrix = buildMatrix(x_q, 'Query Weight Matrix');
        const kMatrix = buildMatrix(x_k, 'Key Weight Matrix');
        const vMatrix = buildMatrix(x_v, 'Value Weight Matrix');

        tuneInactiveMatrix(qMatrix);
        tuneInactiveMatrix(kMatrix);
        tuneInactiveMatrix(vMatrix);

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
        MHA_OUTPUT_PROJECTION_MATRIX_PARAMS.slitTopWidthFactor,
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
            child.material.emissiveIntensity = 0.16;
        }
    });

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
