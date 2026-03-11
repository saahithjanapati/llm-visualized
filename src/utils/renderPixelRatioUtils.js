export function resolveZoomOutSupersampleCeiling({
    baseRatio,
    liveDpr = null,
    maxMultiplier = 1,
    maxDpr = null
} = {}) {
    const safeBaseRatio = (Number.isFinite(baseRatio) && baseRatio > 0) ? baseRatio : 1;
    const safeLiveDpr = (Number.isFinite(liveDpr) && liveDpr > 0) ? liveDpr : null;
    const safeMaxMultiplier = (Number.isFinite(maxMultiplier) && maxMultiplier >= 1) ? maxMultiplier : 1;
    const safeMaxDpr = (Number.isFinite(maxDpr) && maxDpr > 0)
        ? maxDpr
        : safeBaseRatio * safeMaxMultiplier;

    const boostedCeiling = safeBaseRatio * safeMaxMultiplier;
    const dprAwareCeiling = (safeLiveDpr !== null)
        ? Math.max(safeLiveDpr, boostedCeiling)
        : boostedCeiling;

    return Math.max(safeBaseRatio, Math.min(safeMaxDpr, dprAwareCeiling));
}
