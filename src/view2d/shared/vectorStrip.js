export const VIEW2D_VECTOR_STRIP_VARIANT = 'vector-strip';
export const VIEW2D_VECTOR_STRIP_STYLE_VARIANTS = Object.freeze({
    STANDARD: 'standard'
});

export const VIEW2D_VECTOR_STRIP_DEFAULTS = Object.freeze({
    rowGap: 0,
    paddingX: 0,
    paddingY: 0,
    cornerRadius: 8,
    bandCount: 12,
    bandSeparatorOpacity: 0,
    hoverScaleY: 1.16,
    hoverScaleX: 1.06,
    hoverGlowColor: 'rgba(255,255,255,0.08)',
    hoverGlowBlur: 12,
    hoverStrokeColor: 'rgba(255,255,255,0.10)',
    dimmedRowOpacity: 0.18
});

const VIEW2D_VECTOR_STRIP_STYLE_PRESETS = Object.freeze({
    [VIEW2D_VECTOR_STRIP_STYLE_VARIANTS.STANDARD]: Object.freeze({
        rowGap: VIEW2D_VECTOR_STRIP_DEFAULTS.rowGap,
        paddingX: VIEW2D_VECTOR_STRIP_DEFAULTS.paddingX,
        paddingY: VIEW2D_VECTOR_STRIP_DEFAULTS.paddingY,
        cornerRadius: VIEW2D_VECTOR_STRIP_DEFAULTS.cornerRadius,
        bandCount: VIEW2D_VECTOR_STRIP_DEFAULTS.bandCount,
        bandSeparatorOpacity: VIEW2D_VECTOR_STRIP_DEFAULTS.bandSeparatorOpacity,
        hoverScaleY: VIEW2D_VECTOR_STRIP_DEFAULTS.hoverScaleY,
        hoverScaleX: VIEW2D_VECTOR_STRIP_DEFAULTS.hoverScaleX,
        hoverGlowColor: VIEW2D_VECTOR_STRIP_DEFAULTS.hoverGlowColor,
        hoverGlowBlur: VIEW2D_VECTOR_STRIP_DEFAULTS.hoverGlowBlur,
        hoverStrokeColor: VIEW2D_VECTOR_STRIP_DEFAULTS.hoverStrokeColor,
        dimmedRowOpacity: VIEW2D_VECTOR_STRIP_DEFAULTS.dimmedRowOpacity
    })
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

function normalizePositiveNumber(value, fallback = null) {
    return Number.isFinite(value) && value > 0
        ? Math.max(0.001, Number(value))
        : fallback;
}

function normalizeUnitInterval(value, fallback = null) {
    return Number.isFinite(value)
        ? Math.max(0, Math.min(1, Number(value)))
        : fallback;
}

export function resolveView2dVectorStripStyle(styleVariant = VIEW2D_VECTOR_STRIP_STYLE_VARIANTS.STANDARD) {
    return VIEW2D_VECTOR_STRIP_STYLE_PRESETS[styleVariant]
        || VIEW2D_VECTOR_STRIP_STYLE_PRESETS[VIEW2D_VECTOR_STRIP_STYLE_VARIANTS.STANDARD];
}

export function createView2dVectorStripMetadata({
    compactWidth = null,
    rowHeight = null,
    rowGap = null,
    paddingX = null,
    paddingY = null,
    cornerRadius = null,
    styleVariant = VIEW2D_VECTOR_STRIP_STYLE_VARIANTS.STANDARD,
    bandCount = null,
    bandSeparatorOpacity = null,
    hoverScaleY = null,
    hoverGlowColor = null,
    hoverGlowBlur = null,
    hoverStrokeColor = null,
    dimmedRowOpacity = null,
    hideSurface = false,
    collapsedBinCount = null
} = {}) {
    const resolvedStyle = resolveView2dVectorStripStyle(styleVariant);
    const compactRows = {
        variant: VIEW2D_VECTOR_STRIP_VARIANT,
        styleVariant,
        rowGap: normalizeNonNegativeInt(rowGap, resolvedStyle.rowGap),
        paddingX: normalizeNonNegativeInt(paddingX, resolvedStyle.paddingX),
        paddingY: normalizeNonNegativeInt(paddingY, resolvedStyle.paddingY),
        bandCount: normalizePositiveInt(bandCount, resolvedStyle.bandCount),
        bandSeparatorOpacity: normalizeUnitInterval(bandSeparatorOpacity, resolvedStyle.bandSeparatorOpacity),
        hoverScaleY: normalizePositiveNumber(hoverScaleY, resolvedStyle.hoverScaleY),
        hoverGlowBlur: normalizePositiveNumber(hoverGlowBlur, resolvedStyle.hoverGlowBlur),
        dimmedRowOpacity: normalizeUnitInterval(dimmedRowOpacity, resolvedStyle.dimmedRowOpacity)
    };
    const normalizedCompactWidth = normalizePositiveInt(compactWidth, null);
    const normalizedRowHeight = normalizePositiveInt(rowHeight, null);
    const normalizedCornerRadius = normalizeNonNegativeInt(cornerRadius, resolvedStyle.cornerRadius);
    const normalizedCollapsedBinCount = normalizePositiveInt(collapsedBinCount, null);

    if (normalizedCompactWidth !== null) compactRows.compactWidth = normalizedCompactWidth;
    if (normalizedRowHeight !== null) compactRows.rowHeight = normalizedRowHeight;
    if (hideSurface) compactRows.hideSurface = true;
    if (normalizedCollapsedBinCount !== null) compactRows.collapsedBinCount = normalizedCollapsedBinCount;
    compactRows.hoverGlowColor = typeof hoverGlowColor === 'string' && hoverGlowColor.length
        ? hoverGlowColor
        : resolvedStyle.hoverGlowColor;
    compactRows.hoverStrokeColor = typeof hoverStrokeColor === 'string' && hoverStrokeColor.length
        ? hoverStrokeColor
        : resolvedStyle.hoverStrokeColor;

    const metadata = { compactRows };
    if (normalizedCornerRadius !== null) {
        metadata.card = {
            cornerRadius: normalizedCornerRadius
        };
    }
    return metadata;
}

export function createView2dTransposeStripMetadata({
    colWidth = null,
    colGap = null,
    colHeight = null,
    paddingX = null,
    paddingY = null,
    cornerRadius = null,
    styleVariant = VIEW2D_VECTOR_STRIP_STYLE_VARIANTS.STANDARD,
    hoverScaleX = null,
    hoverGlowColor = null,
    hoverGlowBlur = null,
    hoverStrokeColor = null,
    dimmedColumnOpacity = null,
    hideSurface = false
} = {}) {
    const resolvedStyle = resolveView2dVectorStripStyle(styleVariant);
    const columnStrip = {
        variant: VIEW2D_VECTOR_STRIP_VARIANT,
        styleVariant,
        colGap: normalizeNonNegativeInt(colGap, resolvedStyle.rowGap),
        paddingX: normalizeNonNegativeInt(paddingX, resolvedStyle.paddingX),
        paddingY: normalizeNonNegativeInt(paddingY, resolvedStyle.paddingY),
        hoverScaleX: normalizePositiveNumber(hoverScaleX, resolvedStyle.hoverScaleX),
        hoverGlowBlur: normalizePositiveNumber(hoverGlowBlur, resolvedStyle.hoverGlowBlur),
        dimmedColumnOpacity: normalizeUnitInterval(dimmedColumnOpacity, resolvedStyle.dimmedRowOpacity)
    };
    const normalizedColWidth = normalizePositiveInt(colWidth, null);
    const normalizedColHeight = normalizePositiveInt(colHeight, null);
    const normalizedCornerRadius = normalizeNonNegativeInt(cornerRadius, resolvedStyle.cornerRadius);

    if (normalizedColWidth !== null) columnStrip.colWidth = normalizedColWidth;
    if (normalizedColHeight !== null) columnStrip.colHeight = normalizedColHeight;
    if (hideSurface) columnStrip.hideSurface = true;
    columnStrip.hoverGlowColor = typeof hoverGlowColor === 'string' && hoverGlowColor.length
        ? hoverGlowColor
        : resolvedStyle.hoverGlowColor;
    columnStrip.hoverStrokeColor = typeof hoverStrokeColor === 'string' && hoverStrokeColor.length
        ? hoverStrokeColor
        : resolvedStyle.hoverStrokeColor;

    const metadata = { columnStrip };
    if (normalizedCornerRadius !== null) {
        metadata.card = {
            cornerRadius: normalizedCornerRadius
        };
    }
    return metadata;
}
