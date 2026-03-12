export const VIEW2D_VECTOR_STRIP_VARIANT = 'vector-strip';

export const VIEW2D_VECTOR_STRIP_DEFAULTS = Object.freeze({
    rowGap: 1,
    paddingX: 0,
    paddingY: 0,
    cornerRadius: 10
});

function normalizePositiveInt(value, fallback = null) {
    return Number.isFinite(value) && value > 0
        ? Math.max(1, Math.floor(value))
        : fallback;
}

function normalizeNonNegativeInt(value, fallback = null) {
    return Number.isFinite(value) && value >= 0
        ? Math.max(0, Math.floor(value))
        : fallback;
}

export function createView2dVectorStripMetadata({
    compactWidth = null,
    rowHeight = null,
    rowGap = VIEW2D_VECTOR_STRIP_DEFAULTS.rowGap,
    paddingX = VIEW2D_VECTOR_STRIP_DEFAULTS.paddingX,
    paddingY = VIEW2D_VECTOR_STRIP_DEFAULTS.paddingY,
    cornerRadius = VIEW2D_VECTOR_STRIP_DEFAULTS.cornerRadius,
    hideSurface = false,
    collapsedBinCount = null
} = {}) {
    const compactRows = {
        variant: VIEW2D_VECTOR_STRIP_VARIANT,
        rowGap: normalizeNonNegativeInt(rowGap, VIEW2D_VECTOR_STRIP_DEFAULTS.rowGap),
        paddingX: normalizeNonNegativeInt(paddingX, VIEW2D_VECTOR_STRIP_DEFAULTS.paddingX),
        paddingY: normalizeNonNegativeInt(paddingY, VIEW2D_VECTOR_STRIP_DEFAULTS.paddingY)
    };
    const normalizedCompactWidth = normalizePositiveInt(compactWidth, null);
    const normalizedRowHeight = normalizePositiveInt(rowHeight, null);
    const normalizedCornerRadius = normalizeNonNegativeInt(cornerRadius, VIEW2D_VECTOR_STRIP_DEFAULTS.cornerRadius);
    const normalizedCollapsedBinCount = normalizePositiveInt(collapsedBinCount, null);

    if (normalizedCompactWidth !== null) compactRows.compactWidth = normalizedCompactWidth;
    if (normalizedRowHeight !== null) compactRows.rowHeight = normalizedRowHeight;
    if (hideSurface) compactRows.hideSurface = true;
    if (normalizedCollapsedBinCount !== null) compactRows.collapsedBinCount = normalizedCollapsedBinCount;

    const metadata = { compactRows };
    if (normalizedCornerRadius !== null) {
        metadata.card = {
            cornerRadius: normalizedCornerRadius
        };
    }
    return metadata;
}
