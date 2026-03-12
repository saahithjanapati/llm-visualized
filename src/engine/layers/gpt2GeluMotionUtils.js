import * as THREE from 'three';

const DEFAULT_HARMONIC_BLEND = 0.28;
const DEFAULT_WHIP_STRENGTH = 0.4;
const DEFAULT_ENVELOPE_POWER = 0.72;
const DEFAULT_TRAVEL_POWER = 1.12;
const DEFAULT_ROOT_GAIN = 0.92;
const DEFAULT_HARMONIC_TRAVEL_BLEND = 0.45;
const DEFAULT_TIP_BIAS_POWER = 1.7;

export function buildGeluWaveField(segmentCounts, waveCycles = 1) {
    const safeCounts = Array.isArray(segmentCounts)
        ? segmentCounts.map((count) => Math.max(1, Math.floor(Number.isFinite(count) ? count : 1)))
        : [];
    const totalCount = safeCounts.reduce((sum, count) => sum + count, 0);
    const phaseSpan = Math.PI * 2 * Math.max(0, Number.isFinite(waveCycles) ? waveCycles : 1);
    const phaseOffsets = safeCounts.map((count) => new Float32Array(count));
    const normalizedIndices = safeCounts.map((count) => new Float32Array(count));
    let globalIndex = 0;

    for (let s = 0; s < safeCounts.length; s++) {
        const count = safeCounts[s];
        const phaseRow = phaseOffsets[s];
        const normalizedRow = normalizedIndices[s];
        for (let i = 0; i < count; i++) {
            const normalizedIndex = totalCount > 1 ? globalIndex / (totalCount - 1) : 0;
            phaseRow[i] = normalizedIndex * phaseSpan;
            normalizedRow[i] = normalizedIndex;
            globalIndex++;
        }
    }

    return {
        phaseOffsets,
        normalizedIndices,
        totalCount
    };
}

export function computeGeluWhipOffset({
    phaseOffset = 0,
    normalizedIndex = 0,
    progress = 0,
    waveHeight = 1,
    waveTravelCycles = 1,
    harmonicBlend = DEFAULT_HARMONIC_BLEND,
    whipStrength = DEFAULT_WHIP_STRENGTH
} = {}) {
    const safeProgress = THREE.MathUtils.clamp(Number.isFinite(progress) ? progress : 0, 0, 1);
    const safeNormalizedIndex = THREE.MathUtils.clamp(
        Number.isFinite(normalizedIndex) ? normalizedIndex : 0,
        0,
        1
    );
    const safeWaveHeight = Number.isFinite(waveHeight) ? waveHeight : 1;
    const safeTravelCycles = Number.isFinite(waveTravelCycles) ? waveTravelCycles : 1;
    const safeHarmonicBlend = Math.max(0, Number.isFinite(harmonicBlend) ? harmonicBlend : DEFAULT_HARMONIC_BLEND);
    const safeWhipStrength = Math.max(0, Number.isFinite(whipStrength) ? whipStrength : DEFAULT_WHIP_STRENGTH);

    const envelopeBase = Math.max(0, Math.sin(Math.PI * safeProgress));
    const envelope = Math.pow(envelopeBase, DEFAULT_ENVELOPE_POWER);
    const travelT = Math.pow(safeProgress, DEFAULT_TRAVEL_POWER);
    const phaseAdvance = travelT * Math.PI * 2 * safeTravelCycles;
    const phase = (Number.isFinite(phaseOffset) ? phaseOffset : 0) - phaseAdvance;

    // Mix in a light harmonic and a tip-biased gain so the ripple reads
    // as a snapping whip instead of a single soft bulge.
    const fundamental = Math.sin(phase);
    const harmonic = Math.sin((phase * 2) - (phaseAdvance * DEFAULT_HARMONIC_TRAVEL_BLEND));
    const carrier = (fundamental + safeHarmonicBlend * harmonic) / (1 + safeHarmonicBlend);
    const tipBias = Math.pow(safeNormalizedIndex, DEFAULT_TIP_BIAS_POWER);
    const whipGain = THREE.MathUtils.lerp(DEFAULT_ROOT_GAIN, 1 + safeWhipStrength, tipBias);

    return safeWaveHeight * envelope * whipGain * carrier;
}
