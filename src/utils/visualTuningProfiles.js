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
        // Keep MLP pass-through flashes visible at high playback speeds.
        postPassFinalEmissiveIntensity: 0.22,
        flashStartEmissiveIntensity: 0.04,
        flashPeakEmissiveIntensity: 0.62,
        // Allow separate tuning for up/down projection flashes.
        flashPeakEmissiveIntensityUp: 0.78,
        flashPeakEmissiveIntensityDown: 0.62,
        flashMinDurationMs: 560,
    }),
    mhsa: Object.freeze({
        qkvFinalMatrixEmissiveIntensity: 0.13,
        outputProjection: Object.freeze({
            // Output projection uses very short travel durations by default,
            // so enforce a minimum pulse window and stronger contrast.
            startEmissiveIntensity: 0.05,
            peakEmissiveIntensity: 0.78,
            endEmissiveIntensity: 0.34,
            minStageEnterDurationMs: 140,
            minStageThroughDurationMs: 190,
            minStageExitDurationMs: 140,
            minFlashDurationMs: 260,
        }),
    }),
});
