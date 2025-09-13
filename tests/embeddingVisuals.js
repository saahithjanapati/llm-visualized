import * as THREE from 'three';
import { WeightMatrixVisualization } from '../src/components/WeightMatrixVisualization.js';
import { LayerNormalizationVisualization } from '../src/components/LayerNormalizationVisualization.js';
import {
    LAYER_NORM_1_Y_POS,
    LN_PARAMS,
    MLP_MATRIX_PARAMS_DOWN,
    EMBEDDING_MATRIX_PARAMS_VOCAB,
    EMBEDDING_MATRIX_PARAMS_POSITION,
    INACTIVE_COMPONENT_COLOR,
    EMBEDDING_BOTTOM_TOP_ALIGN_OFFSET_FROM_LN1_BOTTOM,
    EMBEDDING_BOTTOM_Y_ADJUST,
    EMBEDDING_BOTTOM_VOCAB_X_OFFSET,
    EMBEDDING_BOTTOM_PAIR_GAP_X,
    EMBEDDING_BOTTOM_POS_X_OFFSET,
    TOP_EMBED_VOCAB_X_OFFSET,
    TOP_EMBED_Y_GAP_ABOVE_TOWER,
    TOP_EMBED_Y_ADJUST,
    TOP_LN_TO_TOP_EMBED_GAP
} from '../src/utils/constants.js';
import { MHA_FINAL_Q_COLOR, MHA_FINAL_K_COLOR } from '../src/animations/LayerAnimationConstants.js';

export function setupEmbeddingVisuals(pipeline, NUM_LAYERS) {
    let vocabTopRef = null;
    let __topEmbedActivated = false;
    try {
        const headBlue = new THREE.Color(MHA_FINAL_Q_COLOR);
        const headGreen = new THREE.Color(MHA_FINAL_K_COLOR);

        const residualYBase = LAYER_NORM_1_Y_POS - LN_PARAMS.height / 2 + EMBEDDING_BOTTOM_TOP_ALIGN_OFFSET_FROM_LN1_BOTTOM;

        const bottomVocabCenterY = residualYBase - EMBEDDING_MATRIX_PARAMS_VOCAB.height / 2 + EMBEDDING_BOTTOM_Y_ADJUST;
        const vocabBottom = new WeightMatrixVisualization(
            null,
            new THREE.Vector3(0 + EMBEDDING_BOTTOM_VOCAB_X_OFFSET, bottomVocabCenterY, 0),
            EMBEDDING_MATRIX_PARAMS_VOCAB.width,
            EMBEDDING_MATRIX_PARAMS_VOCAB.height,
            EMBEDDING_MATRIX_PARAMS_VOCAB.depth,
            EMBEDDING_MATRIX_PARAMS_VOCAB.topWidthFactor,
            EMBEDDING_MATRIX_PARAMS_VOCAB.cornerRadius,
            EMBEDDING_MATRIX_PARAMS_VOCAB.numberOfSlits,
            EMBEDDING_MATRIX_PARAMS_VOCAB.slitWidth,
            EMBEDDING_MATRIX_PARAMS_VOCAB.slitDepthFactor,
            EMBEDDING_MATRIX_PARAMS_VOCAB.slitBottomWidthFactor,
            EMBEDDING_MATRIX_PARAMS_VOCAB.slitTopWidthFactor
        );
        vocabBottom.group.userData.label = 'Vocab Embedding';
        vocabBottom.setColor(headBlue);
        vocabBottom.setMaterialProperties({ opacity: 1.0, transparent: false, emissiveIntensity: 0.05 });
        pipeline.engine.scene.add(vocabBottom.group);

        const gapX = EMBEDDING_BOTTOM_PAIR_GAP_X;
        const posX = (EMBEDDING_MATRIX_PARAMS_VOCAB.width / 2) + (EMBEDDING_MATRIX_PARAMS_POSITION.width / 2) + gapX + EMBEDDING_BOTTOM_POS_X_OFFSET + EMBEDDING_BOTTOM_VOCAB_X_OFFSET;
        const vocabBottomY = bottomVocabCenterY - EMBEDDING_MATRIX_PARAMS_VOCAB.height / 2;
        const bottomPosCenterY = vocabBottomY + EMBEDDING_MATRIX_PARAMS_POSITION.height / 2;
        const posBottom = new WeightMatrixVisualization(
            null,
            new THREE.Vector3(posX, bottomPosCenterY, 0),
            EMBEDDING_MATRIX_PARAMS_POSITION.width,
            EMBEDDING_MATRIX_PARAMS_POSITION.height,
            EMBEDDING_MATRIX_PARAMS_POSITION.depth,
            EMBEDDING_MATRIX_PARAMS_POSITION.topWidthFactor,
            EMBEDDING_MATRIX_PARAMS_POSITION.cornerRadius,
            EMBEDDING_MATRIX_PARAMS_POSITION.numberOfSlits,
            EMBEDDING_MATRIX_PARAMS_POSITION.slitWidth,
            EMBEDDING_MATRIX_PARAMS_POSITION.slitDepthFactor,
            EMBEDDING_MATRIX_PARAMS_POSITION.slitBottomWidthFactor,
            EMBEDDING_MATRIX_PARAMS_POSITION.slitTopWidthFactor
        );
        posBottom.group.userData.label = 'Positional Embedding';
        posBottom.setColor(headGreen);
        posBottom.setMaterialProperties({ opacity: 1.0, transparent: false, emissiveIntensity: 0.05 });
        pipeline.engine.scene.add(posBottom.group);

        const lastLayer = pipeline._layers[NUM_LAYERS - 1];
        if (lastLayer && lastLayer.mlpDown && lastLayer.mlpDown.group) {
            const tmp = new THREE.Vector3();
            lastLayer.mlpDown.group.getWorldPosition(tmp);
            const towerTopY = tmp.y + MLP_MATRIX_PARAMS_DOWN.height / 2;
            const topGap = TOP_EMBED_Y_GAP_ABOVE_TOWER;

            const topLnCenterY = towerTopY + topGap + LN_PARAMS.height / 2;
            const lnTop = new LayerNormalizationVisualization(
                new THREE.Vector3(0 + TOP_EMBED_VOCAB_X_OFFSET, topLnCenterY, 0),
                LN_PARAMS.width,
                LN_PARAMS.height,
                LN_PARAMS.depth,
                LN_PARAMS.wallThickness,
                LN_PARAMS.numberOfHoles,
                LN_PARAMS.holeWidth,
                LN_PARAMS.holeWidthFactor
            );
            lnTop.group.userData.label = 'LayerNorm (Top)';
            lnTop.setColor(new THREE.Color(INACTIVE_COMPONENT_COLOR));
            lnTop.setMaterialProperties({ opacity: 1.0, transparent: false, emissiveIntensity: 0.05 });
            pipeline.engine.scene.add(lnTop.group);

            const topVocabCenterY = topLnCenterY + (LN_PARAMS.height / 2) + TOP_LN_TO_TOP_EMBED_GAP + (EMBEDDING_MATRIX_PARAMS_VOCAB.height / 2) + TOP_EMBED_Y_ADJUST;

            const vocabTop = new WeightMatrixVisualization(
                null,
                new THREE.Vector3(0 + TOP_EMBED_VOCAB_X_OFFSET, topVocabCenterY, 0),
                EMBEDDING_MATRIX_PARAMS_VOCAB.width,
                EMBEDDING_MATRIX_PARAMS_VOCAB.height,
                EMBEDDING_MATRIX_PARAMS_VOCAB.depth,
                EMBEDDING_MATRIX_PARAMS_VOCAB.topWidthFactor,
                EMBEDDING_MATRIX_PARAMS_VOCAB.cornerRadius,
                EMBEDDING_MATRIX_PARAMS_VOCAB.numberOfSlits,
                EMBEDDING_MATRIX_PARAMS_VOCAB.slitWidth,
                EMBEDDING_MATRIX_PARAMS_VOCAB.slitDepthFactor,
                EMBEDDING_MATRIX_PARAMS_VOCAB.slitBottomWidthFactor,
                EMBEDDING_MATRIX_PARAMS_VOCAB.slitTopWidthFactor
            );
            vocabTop.group.rotation.z = Math.PI;
            vocabTop.group.userData.label = 'Vocab Embedding (Top)';
            vocabTop.setColor(new THREE.Color(0x000000));
            vocabTop.setMaterialProperties({ opacity: 1.0, transparent: false, emissiveIntensity: 0.0 });
            vocabTopRef = vocabTop;
            pipeline.engine.scene.add(vocabTop.group);
        }
    } catch (_) { /* optional – embedding visuals are non-critical */ }

    function checkTopEmbeddingActivation() {
        try {
            if (__topEmbedActivated) return;
            const lastLayer = pipeline._layers[NUM_LAYERS - 1];
            if (!lastLayer || !vocabTopRef) return;
            const stopY = lastLayer.__topEmbedStopYLocal;
            if (typeof stopY !== 'number') return;
            const lanes = Array.isArray(lastLayer.lanes) ? lastLayer.lanes : [];
            for (const lane of lanes) {
                const v = lane && lane.originalVec;
                if (v && v.group && v.group.position.y >= stopY - 0.01) {
                    const headBlue = new THREE.Color(MHA_FINAL_Q_COLOR);
                    vocabTopRef.setColor(headBlue);
                    vocabTopRef.setMaterialProperties({ emissiveIntensity: 0.05 });
                    __topEmbedActivated = true;
                    break;
                }
            }
        } catch (_) { /* ignore */ }
    }

    return { checkTopEmbeddingActivation };
}
