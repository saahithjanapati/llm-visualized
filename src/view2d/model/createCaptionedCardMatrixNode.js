import { D_MODEL } from '../../ui/selectionPanelConstants.js';
import {
    createMatrixNode,
    VIEW2D_MATRIX_PRESENTATIONS,
    VIEW2D_MATRIX_SHAPES
} from '../schema/sceneTypes.js';
import { formatView2dMatrixDimensions } from '../shared/formatMatrixDimensions.js';
import { VIEW2D_STYLE_KEYS } from '../theme/visualTokens.js';

const DEFAULT_CARD_REFERENCE_EXTENT = 72;
const DEFAULT_CARD_DIMENSION_EXPONENT = 0.5;

function buildLabel(tex = '', text = '') {
    return {
        tex: typeof tex === 'string' ? tex : '',
        text: typeof text === 'string' && text.length ? text : tex
    };
}

function mergeMetadata(...parts) {
    const merged = parts.reduce((acc, part) => {
        if (!part || typeof part !== 'object') return acc;
        return {
            ...acc,
            ...part
        };
    }, {});
    return Object.keys(merged).length ? merged : null;
}

function createMeasureMetadata(cols, rows = null) {
    const measure = {};
    if (Number.isFinite(cols) && cols > 0) measure.cols = Math.floor(cols);
    if (Number.isFinite(rows) && rows > 0) measure.rows = Math.floor(rows);
    return Object.keys(measure).length ? { measure } : null;
}

function createCardMetadata(width = null, height = null, {
    cornerRadius = null
} = {}) {
    const card = {};
    if (Number.isFinite(width) && width > 0) card.width = Math.floor(width);
    if (Number.isFinite(height) && height > 0) card.height = Math.floor(height);
    if (Number.isFinite(cornerRadius) && cornerRadius >= 0) card.cornerRadius = Math.floor(cornerRadius);
    return Object.keys(card).length ? { card } : null;
}

function createCaptionMetadata({
    position = 'bottom',
    styleKey = null,
    dimensionsTex = '',
    dimensionsText = '',
    minScreenHeightPx = null,
    renderMode = 'dom-katex',
    scaleWithNode = false,
    labelScale = null,
    dimensionsScale = null
} = {}) {
    const caption = {};
    const safePosition = String(position || '').trim().toLowerCase();
    if (
        safePosition === 'top'
        || safePosition === 'bottom'
        || safePosition === 'inside-top'
        || safePosition === 'float-top'
    ) {
        caption.position = safePosition;
    }
    if (typeof styleKey === 'string' && styleKey.length) {
        caption.styleKey = styleKey;
    }
    if (typeof dimensionsTex === 'string' && dimensionsTex.trim().length) {
        caption.dimensionsTex = dimensionsTex.trim();
    }
    if (typeof dimensionsText === 'string' && dimensionsText.trim().length) {
        caption.dimensionsText = dimensionsText.trim();
    }
    if (Number.isFinite(minScreenHeightPx) && minScreenHeightPx > 0) {
        caption.minScreenHeightPx = Math.max(1, Math.floor(minScreenHeightPx));
    }
    const safeRenderMode = String(renderMode || '').trim().toLowerCase();
    if (safeRenderMode.length) {
        caption.renderMode = safeRenderMode;
    }
    if (scaleWithNode === true) {
        caption.scaleWithNode = true;
    }
    if (Number.isFinite(labelScale) && labelScale > 0) {
        caption.labelScale = Number(labelScale);
    }
    if (Number.isFinite(dimensionsScale) && dimensionsScale > 0) {
        caption.dimensionsScale = Number(dimensionsScale);
    }
    return Object.keys(caption).length ? { caption } : null;
}

function clampDimension(value, min, max) {
    const safeValue = Number.isFinite(value) ? value : min;
    return Math.max(min, Math.min(max, safeValue));
}

function resolveRelativeExtent(
    count,
    referenceCount,
    referenceExtent,
    exponent,
    minExtent,
    maxExtent
) {
    const safeCount = Number.isFinite(count) && count > 0 ? count : 1;
    const safeReferenceCount = Number.isFinite(referenceCount) && referenceCount > 0 ? referenceCount : D_MODEL;
    const safeReferenceExtent = Number.isFinite(referenceExtent) && referenceExtent > 0
        ? referenceExtent
        : DEFAULT_CARD_REFERENCE_EXTENT;
    const safeExponent = Number.isFinite(exponent) && exponent > 0
        ? exponent
        : DEFAULT_CARD_DIMENSION_EXPONENT;
    const relative = Math.pow(safeCount / safeReferenceCount, safeExponent);
    return clampDimension(
        Math.round(safeReferenceExtent * relative),
        minExtent,
        maxExtent
    );
}

export function resolveRelativeCardSize({
    rows = 1,
    cols = 1,
    referenceCount = D_MODEL,
    referenceExtent = DEFAULT_CARD_REFERENCE_EXTENT,
    exponent = DEFAULT_CARD_DIMENSION_EXPONENT,
    minWidth = 36,
    maxWidth = 132,
    minHeight = 48,
    maxHeight = 132
} = {}) {
    return {
        width: resolveRelativeExtent(cols, referenceCount, referenceExtent, exponent, minWidth, maxWidth),
        height: resolveRelativeExtent(rows, referenceCount, referenceExtent, exponent, minHeight, maxHeight)
    };
}

export function createCaptionedCardMatrixNode({
    role = 'module-card',
    semantic = {},
    labelTex = '',
    labelText = '',
    rowCount = 1,
    columnCount = 1,
    cardWidth = null,
    cardHeight = null,
    cardCornerRadius = 16,
    captionPosition = 'bottom',
    captionStyleKey = VIEW2D_STYLE_KEYS.LABEL,
    captionMinScreenHeightPx = 20,
    captionDimensionsTex = null,
    captionDimensionsText = null,
    captionScaleWithNode = false,
    captionLabelScale = null,
    captionDimensionsScale = null,
    visualStyleKey = VIEW2D_STYLE_KEYS.MATRIX_WEIGHT,
    background = null,
    accent = null,
    stroke = null,
    metadata = null
} = {}) {
    const resolvedRows = Number.isFinite(rowCount) ? Math.max(1, Math.floor(rowCount)) : 1;
    const resolvedCols = Number.isFinite(columnCount) ? Math.max(1, Math.floor(columnCount)) : 1;
    const dimensionCaption = formatView2dMatrixDimensions(resolvedRows, resolvedCols);
    const visual = {
        styleKey: visualStyleKey
    };
    if (typeof background === 'string' && background.length) {
        visual.background = background;
    }
    if (typeof accent === 'string' && accent.length) {
        visual.accent = accent;
    }
    if (typeof stroke === 'string' && stroke.length) {
        visual.stroke = stroke;
    }

    return createMatrixNode({
        role,
        semantic,
        label: buildLabel(labelTex, labelText),
        dimensions: {
            rows: resolvedRows,
            cols: resolvedCols
        },
        presentation: VIEW2D_MATRIX_PRESENTATIONS.CARD,
        shape: VIEW2D_MATRIX_SHAPES.MATRIX,
        visual,
        metadata: mergeMetadata(
            createMeasureMetadata(resolvedCols, resolvedRows),
            createCardMetadata(cardWidth, cardHeight, {
                cornerRadius: cardCornerRadius
            }),
            createCaptionMetadata({
                position: captionPosition,
                styleKey: captionStyleKey,
                dimensionsTex: typeof captionDimensionsTex === 'string' && captionDimensionsTex.length
                    ? captionDimensionsTex
                    : dimensionCaption.tex,
                dimensionsText: typeof captionDimensionsText === 'string' && captionDimensionsText.length
                    ? captionDimensionsText
                    : dimensionCaption.text,
                minScreenHeightPx: captionMinScreenHeightPx,
                renderMode: 'dom-katex',
                scaleWithNode: captionScaleWithNode,
                labelScale: captionLabelScale,
                dimensionsScale: captionDimensionsScale
            }),
            metadata
        )
    });
}
