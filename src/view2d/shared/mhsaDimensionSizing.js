const MHSA_DIMENSION_BASE_COUNT = 64;
const MHSA_DIMENSION_SCALE_EXPONENT = 0.24;
const MHSA_TOKEN_BASE_COUNT = 25;
const MHSA_TOKEN_SCALE_EXPONENT = 0.18;

const DESKTOP_MHSA_DIMENSION_EXTENT_RULES = Object.freeze({
    baseExtentPx: 72,
    minExtentPx: 30,
    maxExtentPx: 132
});

const MOBILE_MHSA_DIMENSION_EXTENT_RULES = Object.freeze({
    baseExtentPx: 62,
    minExtentPx: 26,
    maxExtentPx: 112
});

const DESKTOP_MHSA_TOKEN_EXTENT_RULES = Object.freeze({
    baseExtentPx: 60,
    minExtentPx: 30,
    maxExtentPx: 60
});

const MOBILE_MHSA_TOKEN_EXTENT_RULES = Object.freeze({
    baseExtentPx: 54,
    minExtentPx: 26,
    maxExtentPx: 54
});

function clamp(value, min, max) {
    return Math.max(min, Math.min(max, value));
}

export function resolveMhsaDimensionVisualExtent(dimensionCount = 1, {
    isSmallScreen = false,
    baseDimensionCount = MHSA_DIMENSION_BASE_COUNT,
    exponent = MHSA_DIMENSION_SCALE_EXPONENT,
    baseExtentPx = null,
    minExtentPx = null,
    maxExtentPx = null
} = {}) {
    const safeDimensionCount = Number.isFinite(dimensionCount)
        ? Math.max(1, Number(dimensionCount))
        : 1;
    const safeBaseDimensionCount = Number.isFinite(baseDimensionCount)
        ? Math.max(1, Number(baseDimensionCount))
        : MHSA_DIMENSION_BASE_COUNT;
    const rules = isSmallScreen
        ? MOBILE_MHSA_DIMENSION_EXTENT_RULES
        : DESKTOP_MHSA_DIMENSION_EXTENT_RULES;
    const resolvedBaseExtentPx = Number.isFinite(baseExtentPx)
        ? Math.max(1, Number(baseExtentPx))
        : rules.baseExtentPx;
    const resolvedMinExtentPx = Number.isFinite(minExtentPx)
        ? Math.max(1, Number(minExtentPx))
        : rules.minExtentPx;
    const resolvedMaxExtentPx = Number.isFinite(maxExtentPx)
        ? Math.max(resolvedMinExtentPx, Number(maxExtentPx))
        : rules.maxExtentPx;
    const safeExponent = Number.isFinite(exponent) && exponent > 0
        ? Number(exponent)
        : MHSA_DIMENSION_SCALE_EXPONENT;
    const scaledExtentPx = resolvedBaseExtentPx
        * Math.pow(safeDimensionCount / safeBaseDimensionCount, safeExponent);

    return Math.round(clamp(
        scaledExtentPx,
        resolvedMinExtentPx,
        resolvedMaxExtentPx
    ));
}

export function resolveMhsaTokenVisualExtent(tokenCount = 1, {
    isSmallScreen = false,
    baseTokenCount = MHSA_TOKEN_BASE_COUNT,
    exponent = MHSA_TOKEN_SCALE_EXPONENT,
    baseExtentPx = null,
    minExtentPx = null,
    maxExtentPx = null
} = {}) {
    const rules = isSmallScreen
        ? MOBILE_MHSA_TOKEN_EXTENT_RULES
        : DESKTOP_MHSA_TOKEN_EXTENT_RULES;

    return resolveMhsaDimensionVisualExtent(tokenCount, {
        isSmallScreen,
        baseDimensionCount: baseTokenCount,
        exponent,
        baseExtentPx: Number.isFinite(baseExtentPx) ? baseExtentPx : rules.baseExtentPx,
        minExtentPx: Number.isFinite(minExtentPx) ? minExtentPx : rules.minExtentPx,
        maxExtentPx: Number.isFinite(maxExtentPx) ? maxExtentPx : rules.maxExtentPx
    });
}
