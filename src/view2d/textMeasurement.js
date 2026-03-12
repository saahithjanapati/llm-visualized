import { resolveSimpleTexPlainText } from './simpleTex.js';

const DEFAULT_FONT_FAMILY = 'ui-monospace, SFMono-Regular, Menlo, monospace';
const DEFAULT_LABEL_FONT_WEIGHT = 500;
const FALLBACK_WIDTH_FACTOR = 0.58;
const TEXT_SIDE_PADDING_FACTOR = 0.16;
const MIN_TEXT_SIDE_PADDING_PX = 2;
const TEXT_MEASUREMENT_CACHE_LIMIT = 4096;

let cachedMeasureContext = undefined;
const measuredTextCache = new Map();

function clampPositive(value, fallback = 0) {
    return Number.isFinite(value) && value > 0 ? value : fallback;
}

function resolveMeasureContext() {
    if (cachedMeasureContext !== undefined) {
        return cachedMeasureContext || null;
    }

    let ctx = null;
    if (typeof OffscreenCanvas !== 'undefined') {
        ctx = new OffscreenCanvas(1, 1).getContext('2d');
    }
    if (!ctx && typeof document !== 'undefined' && typeof document.createElement === 'function') {
        ctx = document.createElement('canvas')?.getContext?.('2d') || null;
    }

    cachedMeasureContext = ctx || null;
    return cachedMeasureContext;
}

function buildMeasurementCacheKey(text = '', font = '') {
    return `${font}__${text}`;
}

export function resolveView2dTextFont({
    fontSize = 12,
    fontWeight = DEFAULT_LABEL_FONT_WEIGHT,
    fontFamily = DEFAULT_FONT_FAMILY
} = {}) {
    const safeFontSize = clampPositive(fontSize, 12);
    const safeFontWeight = Number.isFinite(fontWeight) ? Math.max(1, Math.floor(fontWeight)) : DEFAULT_LABEL_FONT_WEIGHT;
    const safeFontFamily = typeof fontFamily === 'string' && fontFamily.trim().length
        ? fontFamily.trim()
        : DEFAULT_FONT_FAMILY;
    return `${safeFontWeight} ${safeFontSize}px ${safeFontFamily}`;
}

function resolveFallbackWidth(text = '', fontSize = 12) {
    const safeText = resolveSimpleTexPlainText(text);
    const safeFontSize = clampPositive(fontSize, 12);
    return Math.max(0, (safeText.length || 1) * safeFontSize * FALLBACK_WIDTH_FACTOR);
}

export function resolveView2dTextPadding(fontSize = 12) {
    const safeFontSize = clampPositive(fontSize, 12);
    return Math.max(
        MIN_TEXT_SIDE_PADDING_PX,
        Math.ceil(safeFontSize * TEXT_SIDE_PADDING_FACTOR)
    );
}

export function measureView2dText(text = '', {
    fontSize = 12,
    fontWeight = DEFAULT_LABEL_FONT_WEIGHT,
    fontFamily = DEFAULT_FONT_FAMILY
} = {}) {
    const safeText = resolveSimpleTexPlainText(text);
    const safeFontSize = clampPositive(fontSize, 12);
    const font = resolveView2dTextFont({
        fontSize: safeFontSize,
        fontWeight,
        fontFamily
    });
    const cacheKey = buildMeasurementCacheKey(safeText, font);
    if (measuredTextCache.has(cacheKey)) {
        return measuredTextCache.get(cacheKey);
    }

    const fallbackWidth = resolveFallbackWidth(safeText, safeFontSize);
    const fallbackHeight = safeFontSize * 1.1;
    const ctx = resolveMeasureContext();

    let measurement = {
        width: fallbackWidth,
        inkWidth: fallbackWidth,
        left: 0,
        right: fallbackWidth,
        ascent: safeFontSize * 0.8,
        descent: safeFontSize * 0.3,
        height: fallbackHeight
    };

    if (ctx && typeof ctx.measureText === 'function') {
        const previousFont = ctx.font;
        ctx.font = font;
        const metrics = ctx.measureText(safeText || ' ');
        ctx.font = previousFont;

        const measuredWidth = Number.isFinite(metrics?.width) ? metrics.width : fallbackWidth;
        const left = Number.isFinite(metrics?.actualBoundingBoxLeft) ? metrics.actualBoundingBoxLeft : 0;
        const right = Number.isFinite(metrics?.actualBoundingBoxRight) ? metrics.actualBoundingBoxRight : measuredWidth;
        const ascent = Number.isFinite(metrics?.actualBoundingBoxAscent) ? metrics.actualBoundingBoxAscent : (safeFontSize * 0.8);
        const descent = Number.isFinite(metrics?.actualBoundingBoxDescent) ? metrics.actualBoundingBoxDescent : (safeFontSize * 0.3);
        const inkWidth = Math.max(measuredWidth, left + right, fallbackWidth);

        measurement = {
            width: measuredWidth,
            inkWidth,
            left,
            right,
            ascent,
            descent,
            height: Math.max(ascent + descent, fallbackHeight)
        };
    }

    if (measuredTextCache.size >= TEXT_MEASUREMENT_CACHE_LIMIT) {
        measuredTextCache.clear();
    }
    measuredTextCache.set(cacheKey, measurement);
    return measurement;
}

export function fitView2dText(text = '', {
    baseFontSize = 12,
    maxWidth = null,
    fontWeight = DEFAULT_LABEL_FONT_WEIGHT,
    fontFamily = DEFAULT_FONT_FAMILY
} = {}) {
    const safeBaseFontSize = clampPositive(baseFontSize, 12);
    const safeMaxWidth = Number.isFinite(maxWidth) && maxWidth > 0
        ? Math.floor(maxWidth)
        : null;

    let fontSize = safeBaseFontSize;
    let paddingX = resolveView2dTextPadding(fontSize);
    let measurement = measureView2dText(text, {
        fontSize,
        fontWeight,
        fontFamily
    });

    if (safeMaxWidth) {
        for (let iteration = 0; iteration < 3; iteration += 1) {
            const availableTextWidth = Math.max(1, safeMaxWidth - (paddingX * 2));
            const requiredWidth = measurement.inkWidth;
            if (requiredWidth <= availableTextWidth) {
                break;
            }

            fontSize = Math.max(1, fontSize * (availableTextWidth / Math.max(1, requiredWidth)));
            paddingX = resolveView2dTextPadding(fontSize);
            measurement = measureView2dText(text, {
                fontSize,
                fontWeight,
                fontFamily
            });
        }
    }

    const fittedWidth = measurement.inkWidth + (paddingX * 2);
    return {
        width: safeMaxWidth ? Math.min(safeMaxWidth, fittedWidth) : fittedWidth,
        textWidth: measurement.inkWidth,
        fontSize,
        maxWidth: safeMaxWidth,
        paddingX,
        height: Math.max(fontSize * 1.4, measurement.height),
        measurement
    };
}

export function resetView2dTextMeasurementCache() {
    cachedMeasureContext = undefined;
    measuredTextCache.clear();
}
