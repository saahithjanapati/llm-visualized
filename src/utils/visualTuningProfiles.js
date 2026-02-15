export const GPT2_LAYER_VISUAL_TUNING = Object.freeze({
    layerNorm: Object.freeze({
        inactiveColor: 0x333333,
        activeColor: 0xffffff,
        finalColor: 0xffffff,
        opaqueOpacity: 1.0,
        activeOpacity: 0.6,
        exitTransitionRange: 5,
    }),
    mlp: Object.freeze({
        postPassFinalEmissiveIntensity: 0.22,
    }),
    mhsa: Object.freeze({
        qkvFinalMatrixEmissiveIntensity: 0.13,
        outputProjection: Object.freeze({
            startEmissiveIntensity: 0.12,
            peakEmissiveIntensity: 0.30,
            endEmissiveIntensity: 0.30,
        }),
    }),
});

