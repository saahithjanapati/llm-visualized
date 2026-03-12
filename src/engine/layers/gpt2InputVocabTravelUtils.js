import * as THREE from 'three';

const INPUT_VOCAB_RISE_SPEED_PRE_REVEAL = 4.1;
const INPUT_VOCAB_RISE_SPEED_AT_REVEAL = 3.15;
const INPUT_VOCAB_RISE_SPEED_NEAR_EXIT = 2.8;
const INPUT_VOCAB_RISE_SPEED_POST_EXIT = 3.25;

export function getFirstLayerInputVocabRiseSpeedMult({
    layerIndex = -1,
    lane = null,
    currentY = NaN,
    skipActive = false
} = {}) {
    if (skipActive || layerIndex !== 0 || !lane || !Number.isFinite(lane.vocabEmbeddingTravelStartY)) {
        return 1;
    }

    const travelStartY = lane.vocabEmbeddingTravelStartY;
    const revealY = Number.isFinite(lane.vocabEmbeddingRevealY) ? lane.vocabEmbeddingRevealY : NaN;
    const exitY = Number.isFinite(lane.vocabEmbeddingExitY) ? lane.vocabEmbeddingExitY : NaN;
    if (!Number.isFinite(currentY) || !Number.isFinite(exitY) || exitY <= travelStartY + 1e-6) {
        return INPUT_VOCAB_RISE_SPEED_POST_EXIT;
    }

    if (currentY >= exitY - 0.01) {
        return INPUT_VOCAB_RISE_SPEED_POST_EXIT;
    }

    if (Number.isFinite(revealY) && revealY > travelStartY + 1e-6 && currentY < revealY - 0.01) {
        const preRevealProgress = THREE.MathUtils.clamp(
            (currentY - travelStartY) / Math.max(1e-6, revealY - travelStartY),
            0,
            1
        );
        return THREE.MathUtils.lerp(
            INPUT_VOCAB_RISE_SPEED_PRE_REVEAL,
            INPUT_VOCAB_RISE_SPEED_AT_REVEAL,
            THREE.MathUtils.smoothstep(preRevealProgress, 0, 1)
        );
    }

    const nearExitStartY = (Number.isFinite(revealY) && exitY > revealY + 1e-6)
        ? revealY
        : travelStartY;
    const progress = THREE.MathUtils.clamp(
        (currentY - nearExitStartY) / Math.max(1e-6, exitY - nearExitStartY),
        0,
        1
    );

    return THREE.MathUtils.lerp(
        INPUT_VOCAB_RISE_SPEED_AT_REVEAL,
        INPUT_VOCAB_RISE_SPEED_NEAR_EXIT,
        THREE.MathUtils.smoothstep(progress, 0, 1)
    );
}
