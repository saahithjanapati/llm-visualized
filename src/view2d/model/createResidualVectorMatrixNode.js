import { D_MODEL } from '../../ui/selectionPanelConstants.js';
import {
    createMatrixNode,
    VIEW2D_MATRIX_PRESENTATIONS,
    VIEW2D_MATRIX_SHAPES
} from '../schema/sceneTypes.js';
import {
    createView2dTransposeStripMetadata,
    createView2dVectorStripMetadata,
    VIEW2D_VECTOR_STRIP_STYLE_VARIANTS
} from '../shared/vectorStrip.js';
import { formatView2dMatrixDimensions } from '../shared/formatMatrixDimensions.js';
import { VIEW2D_STYLE_KEYS } from '../theme/visualTokens.js';

export const RESIDUAL_VECTOR_MEASURE_COLS = 12;
export const RESIDUAL_VECTOR_STRIP_UNIT = 6;
const MIN_VECTOR_STRIP_COMPACT_WIDTH = RESIDUAL_VECTOR_STRIP_UNIT * 4;

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

function createCaptionMetadata({
    position = 'float-top',
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

function resolvePositiveInt(value, fallback = null) {
    return Number.isFinite(value) && value > 0
        ? Math.max(1, Math.floor(value))
        : fallback;
}

function resolveNonNegativeInt(value, fallback = null) {
    return Number.isFinite(value) && value >= 0
        ? Math.max(0, Math.floor(value))
        : fallback;
}

function resolveVectorStripMeasureCols(columnCount = D_MODEL, measureCols = null) {
    const safeMeasureCols = resolvePositiveInt(measureCols, null);
    if (safeMeasureCols !== null) {
        return safeMeasureCols;
    }
    const safeColumnCount = resolvePositiveInt(columnCount, D_MODEL);
    return Math.max(
        1,
        Math.round((safeColumnCount / D_MODEL) * RESIDUAL_VECTOR_MEASURE_COLS)
    );
}

function resolveVectorStripCompactWidth(compactWidth = null, measureCols = RESIDUAL_VECTOR_MEASURE_COLS) {
    const safeCompactWidth = resolvePositiveInt(compactWidth, null);
    if (safeCompactWidth !== null) {
        return safeCompactWidth;
    }
    return Math.max(
        MIN_VECTOR_STRIP_COMPACT_WIDTH,
        resolvePositiveInt(measureCols, RESIDUAL_VECTOR_MEASURE_COLS) * RESIDUAL_VECTOR_STRIP_UNIT
    );
}

export function createVectorStripMatrixNode({
    role = 'module-card',
    semantic = {},
    labelTex = '\\mathrm{X}',
    labelText = 'X',
    rowItems = [],
    rowCount = null,
    columnCount = D_MODEL,
    measureCols = null,
    compactWidth = null,
    rowHeight = RESIDUAL_VECTOR_STRIP_UNIT,
    captionPosition = 'float-top',
    captionStyleKey = VIEW2D_STYLE_KEYS.LABEL,
    captionMinScreenHeightPx = 28,
    captionDimensionsTex = null,
    captionDimensionsText = null,
    captionScaleWithNode = false,
    captionLabelScale = null,
    captionDimensionsScale = null,
    visualStyleKey = VIEW2D_STYLE_KEYS.RESIDUAL,
    stripStyleVariant = VIEW2D_VECTOR_STRIP_STYLE_VARIANTS.STANDARD,
    stripMetadata = null,
    metadata = null
} = {}) {
    const resolvedRowCount = Number.isFinite(rowCount)
        ? Math.max(1, Math.floor(rowCount))
        : Math.max(1, Array.isArray(rowItems) ? rowItems.length : 0);
    const resolvedColumnCount = resolvePositiveInt(columnCount, D_MODEL);
    const resolvedMeasureCols = resolveVectorStripMeasureCols(resolvedColumnCount, measureCols);
    const resolvedCompactWidth = resolveVectorStripCompactWidth(compactWidth, resolvedMeasureCols);
    const resolvedRowHeight = resolvePositiveInt(rowHeight, RESIDUAL_VECTOR_STRIP_UNIT);
    const dimensionCaption = formatView2dMatrixDimensions(resolvedRowCount, resolvedColumnCount);
    return createMatrixNode({
        role,
        semantic,
        label: buildLabel(labelTex, labelText),
        dimensions: {
            rows: resolvedRowCount,
            cols: resolvedColumnCount
        },
        presentation: VIEW2D_MATRIX_PRESENTATIONS.COMPACT_ROWS,
        shape: VIEW2D_MATRIX_SHAPES.MATRIX,
        rowItems: Array.isArray(rowItems) ? rowItems : [],
        visual: {
            styleKey: visualStyleKey
        },
        metadata: mergeMetadata(
            createMeasureMetadata(resolvedMeasureCols, resolvedRowCount),
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
            createView2dVectorStripMetadata({
                compactWidth: resolvedCompactWidth,
                rowHeight: resolvedRowHeight,
                styleVariant: stripStyleVariant,
                hideSurface: true
            }),
            stripMetadata,
            metadata
        )
    });
}

export function createTransposeVectorStripMatrixNode({
    role = 'module-card',
    semantic = {},
    labelTex = '\\mathrm{X}^{\\mathsf{T}}',
    labelText = 'X^T',
    columnItems = [],
    rowCount = 1,
    columnCount = null,
    compactHeight = null,
    columnWidth = null,
    columnGap = null,
    paddingX = null,
    paddingY = null,
    cornerRadius = null,
    captionPosition = 'bottom',
    captionStyleKey = VIEW2D_STYLE_KEYS.LABEL,
    captionMinScreenHeightPx = 28,
    captionDimensionsTex = null,
    captionDimensionsText = null,
    captionScaleWithNode = false,
    captionLabelScale = null,
    captionDimensionsScale = null,
    visualStyleKey = VIEW2D_STYLE_KEYS.RESIDUAL,
    stripStyleVariant = VIEW2D_VECTOR_STRIP_STYLE_VARIANTS.STANDARD,
    stripMetadata = null,
    metadata = null
} = {}) {
    const resolvedRowCount = resolvePositiveInt(rowCount, 1);
    const resolvedColumnCount = Number.isFinite(columnCount)
        ? Math.max(1, Math.floor(columnCount))
        : Math.max(1, Array.isArray(columnItems) ? columnItems.length : 0);
    const resolvedCompactHeight = resolvePositiveInt(compactHeight, null);
    const resolvedColumnWidth = resolvePositiveInt(columnWidth, null);
    const dimensionCaption = formatView2dMatrixDimensions(resolvedRowCount, resolvedColumnCount);

    return createMatrixNode({
        role,
        semantic,
        label: buildLabel(labelTex, labelText),
        dimensions: {
            rows: resolvedRowCount,
            cols: resolvedColumnCount
        },
        presentation: VIEW2D_MATRIX_PRESENTATIONS.COLUMN_STRIP,
        shape: VIEW2D_MATRIX_SHAPES.MATRIX,
        columnItems: Array.isArray(columnItems) ? columnItems : [],
        visual: {
            styleKey: visualStyleKey
        },
        metadata: mergeMetadata(
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
            createView2dTransposeStripMetadata({
                colWidth: resolvedColumnWidth,
                colGap: resolveNonNegativeInt(columnGap, null),
                colHeight: resolvedCompactHeight,
                paddingX: resolveNonNegativeInt(paddingX, null),
                paddingY: resolveNonNegativeInt(paddingY, null),
                cornerRadius: resolveNonNegativeInt(cornerRadius, null),
                styleVariant: stripStyleVariant
            }),
            stripMetadata,
            metadata
        )
    });
}

export function createResidualVectorMatrixNode(options = {}) {
    return createVectorStripMatrixNode({
        ...options,
        visualStyleKey: options.visualStyleKey || VIEW2D_STYLE_KEYS.RESIDUAL,
        stripStyleVariant: options.stripStyleVariant || VIEW2D_VECTOR_STRIP_STYLE_VARIANTS.STANDARD
    });
}
