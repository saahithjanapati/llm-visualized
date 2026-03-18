function clamp(value, min, max) {
    return Math.min(max, Math.max(min, value));
}

export function resolveFrameRateIndependentDamping(
    baseDampingFactor,
    deltaSeconds,
    referenceStepSeconds = (1 / 60)
) {
    const safeBaseDampingFactor = clamp(
        Number.isFinite(baseDampingFactor) ? baseDampingFactor : 0,
        0,
        1
    );
    if (safeBaseDampingFactor === 0 || safeBaseDampingFactor === 1) {
        return safeBaseDampingFactor;
    }

    const safeReferenceStepSeconds = Number.isFinite(referenceStepSeconds) && referenceStepSeconds > 0
        ? referenceStepSeconds
        : (1 / 60);
    if (!(Number.isFinite(deltaSeconds) && deltaSeconds > 0)) {
        return 0;
    }

    const normalizedStepCount = deltaSeconds / safeReferenceStepSeconds;
    return 1 - ((1 - safeBaseDampingFactor) ** normalizedStepCount);
}
