import { mapValueToColor } from '../../utils/colors.js';
import {
    D_MODEL,
    FINAL_MLP_COLOR,
    RESIDUAL_COLOR_CLAMP
} from '../../ui/selectionPanelConstants.js';
import { getMlpBiasVectorSample } from '../../data/biasParams.js';
import {
    buildSceneNodeId,
    createAnchorRef,
    createOperatorNode,
    createConnectorNode,
    createGroupNode,
    createMatrixNode,
    createSceneModel,
    createTextNode,
    VIEW2D_ANCHOR_SIDES,
    VIEW2D_CONNECTOR_ROUTES,
    VIEW2D_LAYOUT_DIRECTIONS,
    VIEW2D_MATRIX_PRESENTATIONS,
    VIEW2D_MATRIX_SHAPES,
    VIEW2D_TEXT_PRESENTATIONS
} from '../schema/sceneTypes.js';
import {
    resolveView2dVisualTokens,
    VIEW2D_STYLE_KEYS
} from '../theme/visualTokens.js';
import {
    createCaptionedCardMatrixNode,
    resolveRelativeCardSize
} from './createCaptionedCardMatrixNode.js';
import { createVectorStripMatrixNode } from './createResidualVectorMatrixNode.js';
import { createView2dVectorStripMetadata } from '../shared/vectorStrip.js';
import {
    resolveMhsaDimensionVisualExtent,
    resolveMhsaTokenVisualExtent
} from '../shared/mhsaDimensionSizing.js';
import { resolveMhsaTokenMatrixLayoutMetrics } from '../../ui/selectionPanelMhsaLayoutUtils.js';

const INPUT_MEASURE_COLS = 12;
const MLP_INTERMEDIATE_SIZE = D_MODEL * 4;
const BASE_VECTOR_ROW_HEIGHT = 8;
const BASE_VECTOR_ROW_HEIGHT_SMALL = 7;
const BASE_VECTOR_ROW_GAP = 0;
const BASE_VECTOR_PADDING_Y = 0;
const BASE_VECTOR_PADDING_X = 0;
const INPUT_CAPTION_LABEL_SCALE = 0.9;
const MLP_UP_STAGE_INLINE_GAP = 12;
const MLP_UP_STAGE_INLINE_GAP_SMALL = 10;
const MLP_UP_EQUATION_GAP = 12;
const MLP_UP_EQUATION_GAP_SMALL = 10;
const MLP_GELU_STAGE_GAP = 72;
const MLP_GELU_STAGE_GAP_SMALL = 56;
const MLP_GELU_LABEL_GAP = 6;
const MLP_GELU_LABEL_GAP_SMALL = 5;
const MLP_GELU_COPY_GROUP_GAP = 4;
const MLP_GELU_COPY_GROUP_GAP_SMALL = 3;
const MLP_GELU_TEXT_FONT_SCALE = 1.28;
const MLP_GELU_GROUPING_OPERATOR_SCALE = 1.34;
const INCOMING_ARROW_SPACER_WIDTH = 60;
const INCOMING_ARROW_SPACER_WIDTH_SMALL = 52;
const OUTGOING_ARROW_SPACER_WIDTH = 44;
const OUTGOING_ARROW_SPACER_WIDTH_SMALL = 38;
const EDGE_CONNECTOR_STROKE_WIDTH_SCALE = 0.88;
const INPUT_CONNECTOR_TARGET_GAP = 8;
const GELU_CONNECTOR_SOURCE_GAP = 6;
const GELU_CONNECTOR_TARGET_GAP = 12;
const MLP_DOWN_CONNECTOR_SOURCE_GAP = 8;
const MLP_DOWN_CONNECTOR_TARGET_GAP = 8;
const MLP_DOWN_OUTPUT_EDGE_SOURCE_GAP = 8;
const MLP_DOWN_OUTPUT_EDGE_ARROW_HEAD_LENGTH_SCREEN_PX = 6;
const MLP_DOWN_OUTPUT_EDGE_ARROW_HEAD_WING_SCREEN_PX = 3.2;
const MLP_ACTIVATION_COPY_OFFSET = 40;
const MLP_ACTIVATION_COPY_OFFSET_SMALL = 28;
const MLP_WEIGHT_MIN_WIDTH = 108;
const MLP_WEIGHT_MAX_WIDTH = 168;
const MLP_WEIGHT_MIN_HEIGHT = 60;
const MLP_WEIGHT_MAX_HEIGHT = 112;
const MLP_WEIGHT_MIN_WIDTH_SMALL = 94;
const MLP_WEIGHT_MAX_WIDTH_SMALL = 148;
const MLP_WEIGHT_MIN_HEIGHT_SMALL = 54;
const MLP_WEIGHT_MAX_HEIGHT_SMALL = 98;
const MLP_WEIGHT_REFERENCE_EXTENT_SCALE = 0.88;
const MLP_BIAS_ROW_HEIGHT = 14;
const MLP_BIAS_ROW_HEIGHT_SMALL = 12;
const MLP_BIAS_CORNER_RADIUS = 5;
const MLP_GELU_STAGE_GAP_PER_EXTRA_ROW = 4;
const MLP_GELU_STAGE_GAP_PER_EXTRA_ROW_SMALL = 3;
const MLP_GELU_STAGE_GAP_MAX_EXTRA = 28;
const MLP_GELU_STAGE_GAP_MAX_EXTRA_SMALL = 20;

function normalizeIndex(value) {
    return Number.isFinite(value) ? Math.max(0, Math.floor(value)) : null;
}

function buildSemantic(baseSemantic = {}, extra = {}) {
    return {
        ...baseSemantic,
        ...extra
    };
}

function cleanNumberArray(values = [], fallbackLength = 0) {
    if (!Array.isArray(values) || !values.length) {
        return fallbackLength > 0 ? new Array(fallbackLength).fill(0) : [];
    }
    return values.map((value) => (Number.isFinite(value) ? value : 0));
}

function sampleVector(values = [], targetLength = INPUT_MEASURE_COLS) {
    const safeValues = cleanNumberArray(values);
    const length = Math.max(1, Math.floor(targetLength || INPUT_MEASURE_COLS));
    if (!safeValues.length) return new Array(length).fill(0);
    if (safeValues.length <= length) {
        return [...safeValues];
    }
    return Array.from({ length }, (_, index) => {
        const start = Math.floor((index * safeValues.length) / length);
        const end = Math.max(start + 1, Math.floor(((index + 1) * safeValues.length) / length));
        let sum = 0;
        for (let cursor = start; cursor < end; cursor += 1) {
            sum += safeValues[cursor];
        }
        return sum / Math.max(1, end - start);
    });
}

function colorToCss(color) {
    return color?.isColor ? `#${color.getHexString()}` : 'transparent';
}

function hexToRgb(hexValue = 0xFFFFFF) {
    const safe = Number.isFinite(hexValue)
        ? Math.max(0, Math.min(0xFFFFFF, Math.floor(hexValue)))
        : 0xFFFFFF;
    return [
        (safe >> 16) & 0xFF,
        (safe >> 8) & 0xFF,
        safe & 0xFF
    ];
}

function hexToCss(hexValue = 0xFFFFFF, alpha = 1) {
    const [r, g, b] = hexToRgb(hexValue);
    if (alpha >= 1) {
        return `rgb(${r}, ${g}, ${b})`;
    }
    return `rgba(${r}, ${g}, ${b}, ${Math.max(0, Math.min(1, alpha)).toFixed(3)})`;
}

function hexToKatexColor(hexValue = 0xFFFFFF) {
    return `#${Math.max(0, Math.min(0xFFFFFF, Math.floor(
        Number.isFinite(hexValue) ? hexValue : 0xFFFFFF
    ))).toString(16).padStart(6, '0')}`;
}

function mixHexValues(fromHex = 0xFFFFFF, toHex = 0xFFFFFF, amount = 0.5) {
    const t = Math.max(0, Math.min(1, Number.isFinite(amount) ? amount : 0.5));
    const [fromR, fromG, fromB] = hexToRgb(fromHex);
    const [toR, toG, toB] = hexToRgb(toHex);
    const mix = (from, to) => Math.round(from + ((to - from) * t));
    return (mix(fromR, toR) << 16) | (mix(fromG, toG) << 8) | mix(fromB, toB);
}

function buildMlpBiasGradientCss() {
    return `linear-gradient(138deg, ${
        hexToCss(mixHexValues(FINAL_MLP_COLOR, 0xFFFFFF, 0.28), 0.82)
    } 0%, ${
        hexToCss(FINAL_MLP_COLOR, 0.98)
    } 56%, ${
        hexToCss(mixHexValues(FINAL_MLP_COLOR, 0x000000, 0.06), 0.84)
    } 100%)`;
}

function buildGradientCss(values = []) {
    const safeValues = cleanNumberArray(values);
    if (!safeValues.length) return 'none';
    const fillColors = safeValues.map((value) => colorToCss(mapValueToColor(value, {
        clampMax: RESIDUAL_COLOR_CLAMP
    })));
    const stops = fillColors.map((fillColor, index) => {
        const ratio = fillColors.length > 1 ? index / (fillColors.length - 1) : 0;
        return `${fillColor} ${(ratio * 100).toFixed(4)}%`;
    });
    if (stops.length === 1) {
        return `linear-gradient(90deg, ${stops[0].replace(' 0.0000%', ' 0%')}, ${stops[0].replace(' 0.0000%', ' 100%')})`;
    }
    return `linear-gradient(90deg, ${stops.join(', ')})`;
}

function resolveMeasureCols(columnCount = D_MODEL) {
    const dimensionRatio = Math.max(1, Number(columnCount) / D_MODEL);
    return Math.max(
        INPUT_MEASURE_COLS,
        Math.round(INPUT_MEASURE_COLS * Math.pow(dimensionRatio, 0.44))
    );
}

function resolveCompactWidth(columnCount = D_MODEL, {
    isSmallScreen = false
} = {}) {
    const baseExtentPx = isSmallScreen ? 112 : 126;
    const minExtentPx = isSmallScreen ? 84 : 96;
    const maxExtentPx = isSmallScreen ? 196 : 232;
    return Math.max(
        68,
        Math.round(resolveMhsaDimensionVisualExtent(columnCount, {
            isSmallScreen,
            baseDimensionCount: D_MODEL,
            exponent: 0.34,
            baseExtentPx,
            minExtentPx,
            maxExtentPx
        }))
    );
}

function resolveVectorMetrics(rowCount = 1, {
    isSmallScreen = false,
    columnCount = D_MODEL
} = {}) {
    const safeRowCount = Math.max(1, Math.floor(rowCount || 1));
    const baseRowHeight = isSmallScreen ? BASE_VECTOR_ROW_HEIGHT_SMALL : BASE_VECTOR_ROW_HEIGHT;
    const rowTargetExtent = resolveMhsaTokenVisualExtent(safeRowCount, {
        isSmallScreen
    });
    const minRowHeight = safeRowCount >= 20 ? 2 : (safeRowCount >= 10 ? 3 : 4);
    const rowHeight = Math.max(
        minRowHeight,
        Math.min(
            baseRowHeight,
            Math.floor(rowTargetExtent / safeRowCount)
        )
    );
    return {
        rowHeight,
        rowGap: BASE_VECTOR_ROW_GAP,
        paddingX: BASE_VECTOR_PADDING_X,
        paddingY: BASE_VECTOR_PADDING_Y,
        compactWidth: resolveCompactWidth(columnCount, {
            isSmallScreen
        }),
        measureCols: resolveMeasureCols(columnCount),
        rowCount: safeRowCount
    };
}

function resolveMlpGeluStageGap(extraRows = 0, isSmallScreen = false) {
    const safeExtraRows = Number.isFinite(extraRows) ? Math.max(0, Math.floor(extraRows)) : 0;
    const baseGap = isSmallScreen ? MLP_GELU_STAGE_GAP_SMALL : MLP_GELU_STAGE_GAP;
    const perExtraRow = isSmallScreen ? MLP_GELU_STAGE_GAP_PER_EXTRA_ROW_SMALL : MLP_GELU_STAGE_GAP_PER_EXTRA_ROW;
    const maxExtra = isSmallScreen ? MLP_GELU_STAGE_GAP_MAX_EXTRA_SMALL : MLP_GELU_STAGE_GAP_MAX_EXTRA;
    return baseGap + Math.min(maxExtra, safeExtraRows * perExtraRow);
}

function buildResidualRowItems(tokenRefs = [], {
    layerIndex = null,
    measureCols = INPUT_MEASURE_COLS,
    getVector = () => null
} = {}) {
    return tokenRefs.map((tokenRef) => {
        const values = sampleVector(getVector(tokenRef) || [], measureCols);
        const semantic = {
            componentKind: 'residual',
            layerIndex,
            stage: 'ln2.shift',
            role: 'x-ln-row',
            rowIndex: tokenRef.rowIndex,
            tokenIndex: tokenRef.tokenIndex
        };
        const label = typeof tokenRef?.tokenLabel === 'string' && tokenRef.tokenLabel.length
            ? tokenRef.tokenLabel
            : `Token ${tokenRef.rowIndex + 1}`;
        return {
            id: buildSceneNodeId(semantic),
            index: tokenRef.rowIndex,
            label,
            semantic,
            rawValues: values,
            gradientCss: buildGradientCss(values),
            title: label
        };
    });
}

function buildMlpUpRowItems(tokenRefs = [], {
    layerIndex = null,
    measureCols = INPUT_MEASURE_COLS,
    stage = 'mlp-up',
    rowRole = 'mlp-up-row',
    getVector = () => null
} = {}) {
    return tokenRefs.map((tokenRef) => {
        const values = sampleVector(getVector(tokenRef) || [], measureCols);
        const semantic = {
            componentKind: 'mlp',
            layerIndex,
            stage,
            role: rowRole,
            rowIndex: tokenRef.rowIndex,
            tokenIndex: tokenRef.tokenIndex
        };
        const label = typeof tokenRef?.tokenLabel === 'string' && tokenRef.tokenLabel.length
            ? tokenRef.tokenLabel
            : `Token ${tokenRef.rowIndex + 1}`;
        return {
            id: buildSceneNodeId(semantic),
            index: tokenRef.rowIndex,
            label,
            semantic,
            rawValues: values,
            gradientCss: buildGradientCss(values),
            title: label
        };
    });
}

function buildMlpBiasRowItems(layerIndex = null, {
    kind = 'up',
    measureCols = INPUT_MEASURE_COLS
} = {}) {
    const values = sampleVector(
        getMlpBiasVectorSample(layerIndex, kind) || [],
        measureCols
    );
    const safeKind = String(kind || '').trim().toLowerCase() === 'down' ? 'down' : 'up';
    const semantic = {
        componentKind: 'mlp',
        layerIndex,
        stage: `mlp.${safeKind}.bias`,
        role: `mlp-${safeKind}-bias-row`,
        rowIndex: 0
    };
    return [{
        id: buildSceneNodeId(semantic),
        index: 0,
        label: '',
        semantic,
        rawValues: values,
        gradientCss: buildMlpBiasGradientCss(),
        title: safeKind === 'down' ? 'b_down' : 'b_up'
    }];
}

function createMlpTextNode({
    text = '',
    tex = '',
    metadata = null,
    ...rest
} = {}) {
    return createTextNode({
        ...rest,
        text,
        tex,
        presentation: VIEW2D_TEXT_PRESENTATIONS.LABEL,
        metadata: {
            ...(metadata && typeof metadata === 'object' ? metadata : {}),
            renderMode: 'dom-katex'
        }
    });
}

function createHiddenSpacer({
    semantic = {},
    role = 'layout-spacer',
    width = 1,
    height = 1
} = {}) {
    return createMatrixNode({
        role,
        semantic,
        dimensions: {
            rows: 1,
            cols: 1
        },
        presentation: VIEW2D_MATRIX_PRESENTATIONS.CARD,
        shape: VIEW2D_MATRIX_SHAPES.MATRIX,
        visual: {
            styleKey: VIEW2D_STYLE_KEYS.RESIDUAL
        },
        metadata: {
            hidden: true,
            card: {
                width: Math.max(1, Math.floor(width)),
                height: Math.max(1, Math.floor(height)),
                cornerRadius: 0
            }
        }
    });
}

function createMlpOperatorNode({
    text = '',
    metadata = null,
    ...rest
} = {}) {
    return createOperatorNode({
        ...rest,
        text,
        metadata: {
            ...(metadata && typeof metadata === 'object' ? metadata : {}),
            renderMode: 'dom-katex'
        }
    });
}

export function buildMlpDetailSceneModel({
    activationSource = null,
    mlpDetailTarget = null,
    tokenRefs = [],
    visualTokens = null,
    isSmallScreen = false
} = {}) {
    const layerIndex = normalizeIndex(mlpDetailTarget?.layerIndex);
    if (!Number.isFinite(layerIndex) || !tokenRefs.length) {
        return null;
    }

    const rowCount = Math.max(1, tokenRefs.length);
    const resolvedLayoutMetrics = resolveMhsaTokenMatrixLayoutMetrics({
        rowCount,
        isSmallScreen
    });
    const baseSemantic = {
        componentKind: 'mlp',
        layerIndex,
        stage: 'mlp-detail'
    };
    const vectorMetrics = resolveVectorMetrics(rowCount, {
        isSmallScreen,
        columnCount: D_MODEL
    });
    const mlpUpVectorMetrics = resolveVectorMetrics(rowCount, {
        isSmallScreen,
        columnCount: MLP_INTERMEDIATE_SIZE
    });
    const residualRows = buildResidualRowItems(tokenRefs, {
        layerIndex,
        measureCols: vectorMetrics.measureCols,
        getVector: (tokenRef) => (
            typeof activationSource?.getLayerLn2 === 'function'
                ? activationSource.getLayerLn2(layerIndex, 'shift', tokenRef.tokenIndex, D_MODEL)
                : null
        )
    });
    const mlpUpRows = buildMlpUpRowItems(tokenRefs, {
        layerIndex,
        measureCols: mlpUpVectorMetrics.measureCols,
        getVector: (tokenRef) => (
            typeof activationSource?.getMlpUp === 'function'
                ? activationSource.getMlpUp(layerIndex, tokenRef.tokenIndex, MLP_INTERMEDIATE_SIZE)
                : null
        )
    });
    const mlpUpCopyRows = buildMlpUpRowItems(tokenRefs, {
        layerIndex,
        measureCols: mlpUpVectorMetrics.measureCols,
        rowRole: 'mlp-up-copy-row',
        getVector: (tokenRef) => (
            typeof activationSource?.getMlpUp === 'function'
                ? activationSource.getMlpUp(layerIndex, tokenRef.tokenIndex, MLP_INTERMEDIATE_SIZE)
                : null
        )
    });
    const mlpActivationRows = buildMlpUpRowItems(tokenRefs, {
        layerIndex,
        measureCols: mlpUpVectorMetrics.measureCols,
        stage: 'mlp.activation',
        rowRole: 'mlp-activation-row',
        getVector: (tokenRef) => (
            typeof activationSource?.getMlpActivation === 'function'
                ? activationSource.getMlpActivation(layerIndex, tokenRef.tokenIndex, MLP_INTERMEDIATE_SIZE)
                : null
        )
    });
    const mlpActivationCopyRows = buildMlpUpRowItems(tokenRefs, {
        layerIndex,
        measureCols: mlpUpVectorMetrics.measureCols,
        stage: 'mlp.activation',
        rowRole: 'mlp-activation-copy-row',
        getVector: (tokenRef) => (
            typeof activationSource?.getMlpActivation === 'function'
                ? activationSource.getMlpActivation(layerIndex, tokenRef.tokenIndex, MLP_INTERMEDIATE_SIZE)
                : null
        )
    });
    const mlpDownRows = buildMlpUpRowItems(tokenRefs, {
        layerIndex,
        measureCols: vectorMetrics.measureCols,
        stage: 'mlp.down',
        rowRole: 'mlp-down-row',
        getVector: (tokenRef) => (
            typeof activationSource?.getMlpDown === 'function'
                ? activationSource.getMlpDown(layerIndex, tokenRef.tokenIndex, D_MODEL)
                : null
        )
    });
    const mlpUpBiasRows = buildMlpBiasRowItems(layerIndex, {
        kind: 'up',
        measureCols: mlpUpVectorMetrics.measureCols
    });
    const mlpDownBiasRows = buildMlpBiasRowItems(layerIndex, {
        kind: 'down',
        measureCols: vectorMetrics.measureCols
    });

    if (!residualRows.length) {
        return null;
    }

    const inputNode = createVectorStripMatrixNode({
        role: 'projection-source-xln',
        semantic: buildSemantic(baseSemantic, {
            stage: 'mlp-input',
            role: 'projection-source-xln'
        }),
        labelTex: 'x_{\\ln}',
        labelText: 'x_ln',
        rowItems: residualRows,
        rowCount,
        columnCount: D_MODEL,
        measureCols: vectorMetrics.measureCols,
        compactWidth: vectorMetrics.compactWidth,
        rowHeight: vectorMetrics.rowHeight,
        captionPosition: 'bottom',
        captionLabelScale: INPUT_CAPTION_LABEL_SCALE,
        visualStyleKey: VIEW2D_STYLE_KEYS.RESIDUAL,
        stripMetadata: createView2dVectorStripMetadata({
            compactWidth: vectorMetrics.compactWidth,
            rowHeight: vectorMetrics.rowHeight,
            rowGap: vectorMetrics.rowGap,
            paddingX: vectorMetrics.paddingX,
            paddingY: vectorMetrics.paddingY,
            cornerRadius: 10,
            bandCount: Math.max(12, vectorMetrics.measureCols),
            hoverScaleY: 1.12,
            hoverGlowBlur: 10,
            hideSurface: true
        }),
        metadata: {
            kind: 'projection-source-xln',
            disableEdgeOrnament: true
        }
    });
    const mlpUpWeightSemantic = buildSemantic(baseSemantic, {
        stage: 'mlp-up'
    });
    const weightReferenceExtent = Math.round(
        Math.max(vectorMetrics.compactWidth, mlpUpVectorMetrics.compactWidth) * MLP_WEIGHT_REFERENCE_EXTENT_SCALE
    );
    const weightCardSize = resolveRelativeCardSize({
        rows: D_MODEL,
        cols: MLP_INTERMEDIATE_SIZE,
        referenceCount: D_MODEL,
        referenceExtent: weightReferenceExtent,
        minWidth: isSmallScreen ? MLP_WEIGHT_MIN_WIDTH_SMALL : MLP_WEIGHT_MIN_WIDTH,
        maxWidth: isSmallScreen ? MLP_WEIGHT_MAX_WIDTH_SMALL : MLP_WEIGHT_MAX_WIDTH,
        minHeight: isSmallScreen ? MLP_WEIGHT_MIN_HEIGHT_SMALL : MLP_WEIGHT_MIN_HEIGHT,
        maxHeight: isSmallScreen ? MLP_WEIGHT_MAX_HEIGHT_SMALL : MLP_WEIGHT_MAX_HEIGHT
    });
    weightCardSize.height = Math.max(
        isSmallScreen ? MLP_WEIGHT_MIN_HEIGHT_SMALL : MLP_WEIGHT_MIN_HEIGHT,
        Math.round(vectorMetrics.compactWidth)
    );
    weightCardSize.width = Math.max(
        isSmallScreen ? MLP_WEIGHT_MIN_WIDTH_SMALL : MLP_WEIGHT_MIN_WIDTH,
        Math.round(mlpUpVectorMetrics.compactWidth)
    );
    const weightNode = createCaptionedCardMatrixNode({
        role: 'mlp-up-weight',
        semantic: buildSemantic(mlpUpWeightSemantic, {
            role: 'mlp-up-weight'
        }),
        labelTex: 'W_{\\mathrm{up}}',
        labelText: 'W_up',
        rowCount: D_MODEL,
        columnCount: MLP_INTERMEDIATE_SIZE,
        cardWidth: weightCardSize.width,
        cardHeight: weightCardSize.height,
        cardCornerRadius: 10,
        captionPosition: 'bottom',
        captionLabelScale: 1.12,
        captionDimensionsScale: 0.94,
        visualStyleKey: VIEW2D_STYLE_KEYS.MLP,
        disableCardSurfaceEffects: true,
        metadata: {
            kind: 'mlp-up'
        }
    });
    const outputNode = createVectorStripMatrixNode({
        role: 'mlp-up-output',
        semantic: buildSemantic(mlpUpWeightSemantic, {
            role: 'mlp-up-output'
        }),
        labelTex: 'a',
        labelText: 'a',
        rowItems: mlpUpRows,
        rowCount,
        columnCount: MLP_INTERMEDIATE_SIZE,
        measureCols: mlpUpVectorMetrics.measureCols,
        compactWidth: mlpUpVectorMetrics.compactWidth,
        rowHeight: mlpUpVectorMetrics.rowHeight,
        captionPosition: 'bottom',
        captionLabelScale: INPUT_CAPTION_LABEL_SCALE,
        visualStyleKey: VIEW2D_STYLE_KEYS.RESIDUAL,
        stripMetadata: createView2dVectorStripMetadata({
            compactWidth: mlpUpVectorMetrics.compactWidth,
            rowHeight: mlpUpVectorMetrics.rowHeight,
            rowGap: mlpUpVectorMetrics.rowGap,
            paddingX: mlpUpVectorMetrics.paddingX,
            paddingY: mlpUpVectorMetrics.paddingY,
            cornerRadius: 10,
            bandCount: Math.max(12, mlpUpVectorMetrics.measureCols),
            hoverScaleY: 1.12,
            hoverGlowBlur: 10,
            hideSurface: true
        }),
        metadata: {
            kind: 'mlp-up'
        }
    });
    const biasNode = createVectorStripMatrixNode({
        role: 'mlp-up-bias',
        semantic: buildSemantic(mlpUpWeightSemantic, {
            stage: 'mlp.up.bias',
            role: 'mlp-up-bias'
        }),
        labelTex: 'b_{\\mathrm{up}}',
        labelText: 'b_up',
        rowItems: mlpUpBiasRows,
        rowCount: 1,
        columnCount: MLP_INTERMEDIATE_SIZE,
        measureCols: mlpUpVectorMetrics.measureCols,
        compactWidth: mlpUpVectorMetrics.compactWidth,
        rowHeight: isSmallScreen ? MLP_BIAS_ROW_HEIGHT_SMALL : MLP_BIAS_ROW_HEIGHT,
        captionPosition: 'bottom',
        captionLabelScale: INPUT_CAPTION_LABEL_SCALE,
        captionPreferStandardSizing: true,
        visualStyleKey: VIEW2D_STYLE_KEYS.MLP,
        stripMetadata: createView2dVectorStripMetadata({
            compactWidth: mlpUpVectorMetrics.compactWidth,
            rowHeight: isSmallScreen ? MLP_BIAS_ROW_HEIGHT_SMALL : MLP_BIAS_ROW_HEIGHT,
            rowGap: 0,
            paddingX: 0,
            paddingY: 0,
            cornerRadius: MLP_BIAS_CORNER_RADIUS,
            bandCount: Math.max(12, mlpUpVectorMetrics.measureCols),
            hoverScaleY: 1.12,
            hoverGlowBlur: 10,
            hideSurface: true
        }),
        metadata: {
            kind: 'mlp-up-bias'
        }
    });
    const geluCopyNode = createVectorStripMatrixNode({
        role: 'mlp-up-output-copy',
        semantic: buildSemantic(baseSemantic, {
            stage: 'mlp-activation',
            role: 'mlp-up-output-copy'
        }),
        labelTex: 'a',
        labelText: 'a',
        rowItems: mlpUpCopyRows,
        rowCount,
        columnCount: MLP_INTERMEDIATE_SIZE,
        measureCols: mlpUpVectorMetrics.measureCols,
        compactWidth: mlpUpVectorMetrics.compactWidth,
        rowHeight: mlpUpVectorMetrics.rowHeight,
        captionPosition: 'bottom',
        captionLabelScale: INPUT_CAPTION_LABEL_SCALE,
        visualStyleKey: VIEW2D_STYLE_KEYS.RESIDUAL,
        stripMetadata: createView2dVectorStripMetadata({
            compactWidth: mlpUpVectorMetrics.compactWidth,
            rowHeight: mlpUpVectorMetrics.rowHeight,
            rowGap: mlpUpVectorMetrics.rowGap,
            paddingX: mlpUpVectorMetrics.paddingX,
            paddingY: mlpUpVectorMetrics.paddingY,
            cornerRadius: 10,
            bandCount: Math.max(12, mlpUpVectorMetrics.measureCols),
            hoverScaleY: 1.12,
            hoverGlowBlur: 10,
            hideSurface: true
        }),
        metadata: {
            kind: 'mlp-up-copy'
        }
    });
    const activationNode = createVectorStripMatrixNode({
        role: 'mlp-activation-output',
        semantic: buildSemantic(baseSemantic, {
            stage: 'mlp.activation',
            role: 'mlp-activation-output'
        }),
        labelTex: 'z',
        labelText: 'z',
        rowItems: mlpActivationRows,
        rowCount,
        columnCount: MLP_INTERMEDIATE_SIZE,
        measureCols: mlpUpVectorMetrics.measureCols,
        compactWidth: mlpUpVectorMetrics.compactWidth,
        rowHeight: mlpUpVectorMetrics.rowHeight,
        captionPosition: 'bottom',
        captionLabelScale: INPUT_CAPTION_LABEL_SCALE,
        visualStyleKey: VIEW2D_STYLE_KEYS.RESIDUAL,
        stripMetadata: createView2dVectorStripMetadata({
            compactWidth: mlpUpVectorMetrics.compactWidth,
            rowHeight: mlpUpVectorMetrics.rowHeight,
            rowGap: mlpUpVectorMetrics.rowGap,
            paddingX: mlpUpVectorMetrics.paddingX,
            paddingY: mlpUpVectorMetrics.paddingY,
            cornerRadius: 10,
            bandCount: Math.max(12, mlpUpVectorMetrics.measureCols),
            hoverScaleY: 1.12,
            hoverGlowBlur: 10,
            hideSurface: true
        }),
        metadata: {
            kind: 'mlp-activation'
        }
    });
    const activationCopyNode = createVectorStripMatrixNode({
        role: 'mlp-activation-output-copy',
        semantic: buildSemantic(baseSemantic, {
            stage: 'mlp.activation',
            role: 'mlp-activation-output-copy'
        }),
        labelTex: 'z',
        labelText: 'z',
        rowItems: mlpActivationCopyRows,
        rowCount,
        columnCount: MLP_INTERMEDIATE_SIZE,
        measureCols: mlpUpVectorMetrics.measureCols,
        compactWidth: mlpUpVectorMetrics.compactWidth,
        rowHeight: mlpUpVectorMetrics.rowHeight,
        captionPosition: 'bottom',
        captionLabelScale: INPUT_CAPTION_LABEL_SCALE,
        visualStyleKey: VIEW2D_STYLE_KEYS.RESIDUAL,
        stripMetadata: createView2dVectorStripMetadata({
            compactWidth: mlpUpVectorMetrics.compactWidth,
            rowHeight: mlpUpVectorMetrics.rowHeight,
            rowGap: mlpUpVectorMetrics.rowGap,
            paddingX: mlpUpVectorMetrics.paddingX,
            paddingY: mlpUpVectorMetrics.paddingY,
            cornerRadius: 10,
            bandCount: Math.max(12, mlpUpVectorMetrics.measureCols),
            hoverScaleY: 1.12,
            hoverGlowBlur: 10,
            hideSurface: true
        }),
        metadata: {
            kind: 'mlp-activation-copy'
        }
    });
    const mlpDownWeightSemantic = buildSemantic(baseSemantic, {
        stage: 'mlp-down'
    });
    const downWeightCardSize = resolveRelativeCardSize({
        rows: MLP_INTERMEDIATE_SIZE,
        cols: D_MODEL,
        referenceCount: D_MODEL,
        referenceExtent: weightReferenceExtent,
        minWidth: isSmallScreen ? MLP_WEIGHT_MIN_WIDTH_SMALL : MLP_WEIGHT_MIN_WIDTH,
        maxWidth: isSmallScreen ? MLP_WEIGHT_MAX_WIDTH_SMALL : MLP_WEIGHT_MAX_WIDTH,
        minHeight: isSmallScreen ? MLP_WEIGHT_MIN_HEIGHT_SMALL : MLP_WEIGHT_MIN_HEIGHT,
        maxHeight: isSmallScreen ? MLP_WEIGHT_MAX_HEIGHT_SMALL : MLP_WEIGHT_MAX_HEIGHT
    });
    downWeightCardSize.height = Math.max(
        isSmallScreen ? MLP_WEIGHT_MIN_HEIGHT_SMALL : MLP_WEIGHT_MIN_HEIGHT,
        Math.round(mlpUpVectorMetrics.compactWidth)
    );
    downWeightCardSize.width = Math.max(
        isSmallScreen ? MLP_WEIGHT_MIN_WIDTH_SMALL : MLP_WEIGHT_MIN_WIDTH,
        Math.round(vectorMetrics.compactWidth)
    );
    const downWeightNode = createCaptionedCardMatrixNode({
        role: 'mlp-down-weight',
        semantic: buildSemantic(mlpDownWeightSemantic, {
            role: 'mlp-down-weight'
        }),
        labelTex: 'W_{\\mathrm{down}}',
        labelText: 'W_down',
        rowCount: MLP_INTERMEDIATE_SIZE,
        columnCount: D_MODEL,
        cardWidth: downWeightCardSize.width,
        cardHeight: downWeightCardSize.height,
        cardCornerRadius: 10,
        captionPosition: 'bottom',
        captionLabelScale: 1.12,
        captionDimensionsScale: 0.94,
        visualStyleKey: VIEW2D_STYLE_KEYS.MLP,
        disableCardSurfaceEffects: true,
        metadata: {
            kind: 'mlp-down'
        }
    });
    const downBiasNode = createVectorStripMatrixNode({
        role: 'mlp-down-bias',
        semantic: buildSemantic(mlpDownWeightSemantic, {
            stage: 'mlp.down.bias',
            role: 'mlp-down-bias'
        }),
        labelTex: 'b_{\\mathrm{down}}',
        labelText: 'b_down',
        rowItems: mlpDownBiasRows,
        rowCount: 1,
        columnCount: D_MODEL,
        measureCols: vectorMetrics.measureCols,
        compactWidth: vectorMetrics.compactWidth,
        rowHeight: isSmallScreen ? MLP_BIAS_ROW_HEIGHT_SMALL : MLP_BIAS_ROW_HEIGHT,
        captionPosition: 'bottom',
        captionLabelScale: INPUT_CAPTION_LABEL_SCALE,
        captionPreferStandardSizing: true,
        visualStyleKey: VIEW2D_STYLE_KEYS.MLP,
        stripMetadata: createView2dVectorStripMetadata({
            compactWidth: vectorMetrics.compactWidth,
            rowHeight: isSmallScreen ? MLP_BIAS_ROW_HEIGHT_SMALL : MLP_BIAS_ROW_HEIGHT,
            rowGap: 0,
            paddingX: 0,
            paddingY: 0,
            cornerRadius: MLP_BIAS_CORNER_RADIUS,
            bandCount: Math.max(12, vectorMetrics.measureCols),
            hoverScaleY: 1.12,
            hoverGlowBlur: 10,
            hideSurface: true
        }),
        metadata: {
            kind: 'mlp-down-bias'
        }
    });
    const downOutputNode = createVectorStripMatrixNode({
        role: 'mlp-down-output',
        semantic: buildSemantic(mlpDownWeightSemantic, {
            role: 'mlp-down-output'
        }),
        labelTex: `\\textcolor{${hexToKatexColor(FINAL_MLP_COLOR)}}{\\mathrm{MLP}}(x_{\\ln})`,
        labelText: 'MLP(x_ln)',
        rowItems: mlpDownRows,
        rowCount,
        columnCount: D_MODEL,
        measureCols: vectorMetrics.measureCols,
        compactWidth: vectorMetrics.compactWidth,
        rowHeight: vectorMetrics.rowHeight,
        captionPosition: 'bottom',
        captionLabelScale: INPUT_CAPTION_LABEL_SCALE,
        visualStyleKey: VIEW2D_STYLE_KEYS.RESIDUAL,
        stripMetadata: createView2dVectorStripMetadata({
            compactWidth: vectorMetrics.compactWidth,
            rowHeight: vectorMetrics.rowHeight,
            rowGap: vectorMetrics.rowGap,
            paddingX: vectorMetrics.paddingX,
            paddingY: vectorMetrics.paddingY,
            cornerRadius: 10,
            bandCount: Math.max(12, vectorMetrics.measureCols),
            hoverScaleY: 1.12,
            hoverGlowBlur: 10,
            hideSurface: true
        }),
        metadata: {
            kind: 'mlp-down',
            disableEdgeOrnament: true
        }
    });
    const outgoingArrowSpacerNode = createHiddenSpacer({
        semantic: buildSemantic(mlpDownWeightSemantic, {
            stage: 'mlp.down',
            role: 'outgoing-arrow-spacer'
        }),
        role: 'outgoing-arrow-spacer',
        width: isSmallScreen ? OUTGOING_ARROW_SPACER_WIDTH_SMALL : OUTGOING_ARROW_SPACER_WIDTH,
        height: 1
    });
    const geluCopyGroupNode = createGroupNode({
        role: 'mlp-gelu-copy-group',
        semantic: buildSemantic(baseSemantic, {
            stage: 'mlp-activation',
            role: 'mlp-gelu-copy-group'
        }),
        direction: VIEW2D_LAYOUT_DIRECTIONS.HORIZONTAL,
        align: 'center',
        gapKey: 'inline',
        children: [
            createMlpOperatorNode({
                role: 'mlp-gelu-open',
                semantic: buildSemantic(baseSemantic, {
                    stage: 'mlp-activation',
                    role: 'mlp-gelu-open',
                    operatorKey: 'open'
                }),
                text: '(',
                visual: {
                    styleKey: VIEW2D_STYLE_KEYS.OPERATOR
                },
                metadata: {
                    fontScale: MLP_GELU_GROUPING_OPERATOR_SCALE
                }
            }),
            geluCopyNode,
            createMlpOperatorNode({
                role: 'mlp-gelu-close',
                semantic: buildSemantic(baseSemantic, {
                    stage: 'mlp-activation',
                    role: 'mlp-gelu-close',
                    operatorKey: 'close'
                }),
                text: ')',
                visual: {
                    styleKey: VIEW2D_STYLE_KEYS.OPERATOR
                },
                metadata: {
                    fontScale: MLP_GELU_GROUPING_OPERATOR_SCALE
                }
            })
        ],
        metadata: {
            gapOverride: isSmallScreen ? MLP_GELU_COPY_GROUP_GAP_SMALL : MLP_GELU_COPY_GROUP_GAP
        }
    });
    const geluStageNode = createGroupNode({
        role: 'mlp-gelu-stage',
        semantic: buildSemantic(baseSemantic, {
            stage: 'mlp-activation',
            role: 'mlp-gelu-stage'
        }),
        direction: VIEW2D_LAYOUT_DIRECTIONS.HORIZONTAL,
        align: 'center',
        gapKey: 'inline',
        children: [
            createMlpTextNode({
                role: 'mlp-gelu-label',
                semantic: buildSemantic(baseSemantic, {
                    stage: 'mlp-activation',
                    role: 'mlp-gelu-label',
                    focusKey: 'gelu'
                }),
                tex: '\\mathrm{GELU}',
                text: 'GELU',
                visual: {
                    styleKey: VIEW2D_STYLE_KEYS.LABEL
                },
                metadata: {
                    minScreenHeightPx: 0,
                    fontScale: MLP_GELU_TEXT_FONT_SCALE
                }
            }),
            geluCopyGroupNode,
            createMlpOperatorNode({
                role: 'mlp-gelu-equals',
                semantic: buildSemantic(baseSemantic, {
                    stage: 'mlp-activation',
                    role: 'mlp-gelu-equals',
                    operatorKey: 'equals'
                }),
                text: '=',
                visual: {
                    styleKey: VIEW2D_STYLE_KEYS.OPERATOR
                }
            }),
            activationNode
        ],
        layout: {
            anchorAlign: {
                axis: 'x',
                selfNodeId: geluCopyNode.id,
                targetNodeId: outputNode.id,
                selfAnchor: VIEW2D_ANCHOR_SIDES.CENTER,
                targetAnchor: VIEW2D_ANCHOR_SIDES.CENTER
            }
        },
        metadata: {
            gapOverride: isSmallScreen ? MLP_GELU_LABEL_GAP_SMALL : MLP_GELU_LABEL_GAP
        }
    });
    const equationRowNode = createGroupNode({
        role: 'mlp-up-equation-row',
        semantic: buildSemantic(mlpUpWeightSemantic, {
            role: 'mlp-up-equation-row'
        }),
        direction: VIEW2D_LAYOUT_DIRECTIONS.HORIZONTAL,
        align: 'center',
        gapKey: 'projection',
        children: [
            weightNode,
            createMlpOperatorNode({
                role: 'mlp-up-plus',
                semantic: buildSemantic(mlpUpWeightSemantic, {
                    role: 'mlp-up-plus',
                    operatorKey: 'plus'
                }),
                text: '+',
                visual: {
                    styleKey: VIEW2D_STYLE_KEYS.OPERATOR
                }
            }),
            biasNode,
            createMlpOperatorNode({
                role: 'mlp-up-equals',
                semantic: buildSemantic(mlpUpWeightSemantic, {
                    role: 'mlp-up-equals',
                    operatorKey: 'equals'
                }),
                text: '=',
                visual: {
                    styleKey: VIEW2D_STYLE_KEYS.OPERATOR
                }
            }),
            outputNode
        ],
        metadata: {
            gapOverride: isSmallScreen ? MLP_UP_EQUATION_GAP_SMALL : MLP_UP_EQUATION_GAP
        }
    });
    const outputStageNode = createGroupNode({
        role: 'mlp-up-output-stage',
        semantic: buildSemantic(baseSemantic, {
            stage: 'mlp-activation',
            role: 'mlp-up-output-stage'
        }),
        direction: VIEW2D_LAYOUT_DIRECTIONS.VERTICAL,
        align: 'center',
        gapKey: 'default',
        children: [
            equationRowNode,
            geluStageNode
        ],
        layout: {
            anchorAlign: {
                axis: 'y',
                selfNodeId: equationRowNode.id,
                targetNodeId: inputNode.id,
                selfAnchor: VIEW2D_ANCHOR_SIDES.CENTER,
                targetAnchor: VIEW2D_ANCHOR_SIDES.CENTER
            }
        },
        metadata: {
            gapOverride: resolveMlpGeluStageGap(resolvedLayoutMetrics?.extraRows, isSmallScreen)
        }
    });
    const downEquationRowNode = createGroupNode({
        role: 'mlp-down-equation-row',
        semantic: buildSemantic(mlpDownWeightSemantic, {
            role: 'mlp-down-equation-row'
        }),
        direction: VIEW2D_LAYOUT_DIRECTIONS.HORIZONTAL,
        align: 'center',
        gapKey: 'projection',
        children: [
            downWeightNode,
            createMlpOperatorNode({
                role: 'mlp-down-plus',
                semantic: buildSemantic(mlpDownWeightSemantic, {
                    role: 'mlp-down-plus',
                    operatorKey: 'plus'
                }),
                text: '+',
                visual: {
                    styleKey: VIEW2D_STYLE_KEYS.OPERATOR
                }
            }),
            downBiasNode,
            createMlpOperatorNode({
                role: 'mlp-down-equals',
                semantic: buildSemantic(mlpDownWeightSemantic, {
                    role: 'mlp-down-equals',
                    operatorKey: 'equals'
                }),
                text: '=',
                visual: {
                    styleKey: VIEW2D_STYLE_KEYS.OPERATOR
                }
            }),
            downOutputNode,
            outgoingArrowSpacerNode
        ],
        metadata: {
            gapOverride: isSmallScreen ? MLP_UP_EQUATION_GAP_SMALL : MLP_UP_EQUATION_GAP
        }
    });
    const downFlowNode = createGroupNode({
        role: 'mlp-down-flow',
        semantic: buildSemantic(mlpDownWeightSemantic, {
            role: 'mlp-down-flow'
        }),
        direction: VIEW2D_LAYOUT_DIRECTIONS.HORIZONTAL,
        align: 'center',
        gapKey: 'projection',
        children: [
            activationCopyNode,
            createMlpOperatorNode({
                role: 'mlp-down-multiply',
                semantic: buildSemantic(mlpDownWeightSemantic, {
                    role: 'mlp-down-multiply',
                    operatorKey: 'multiply'
                }),
                text: '×',
                visual: {
                    styleKey: VIEW2D_STYLE_KEYS.OPERATOR
                }
            }),
            downEquationRowNode
        ],
        layout: {
            anchorAlign: {
                axis: 'x',
                selfNodeId: activationCopyNode.id,
                targetNodeId: activationNode.id,
                selfAnchor: VIEW2D_ANCHOR_SIDES.LEFT,
                targetAnchor: VIEW2D_ANCHOR_SIDES.RIGHT,
                offset: isSmallScreen ? MLP_ACTIVATION_COPY_OFFSET_SMALL : MLP_ACTIVATION_COPY_OFFSET
            }
        },
        metadata: {
            gapOverride: isSmallScreen ? MLP_UP_STAGE_INLINE_GAP_SMALL : MLP_UP_STAGE_INLINE_GAP
        }
    });
    const downStageNode = createGroupNode({
        role: 'mlp-down-stage',
        semantic: buildSemantic(mlpDownWeightSemantic, {
            role: 'mlp-down-stage'
        }),
        direction: VIEW2D_LAYOUT_DIRECTIONS.HORIZONTAL,
        align: 'center',
        gapKey: 'default',
        children: [downFlowNode],
        layout: {
            anchorAlign: {
                axis: 'y',
                selfNodeId: activationCopyNode.id,
                targetNodeId: activationNode.id,
                selfAnchor: VIEW2D_ANCHOR_SIDES.CENTER,
                targetAnchor: VIEW2D_ANCHOR_SIDES.CENTER
            }
        },
        metadata: {
            gapOverride: isSmallScreen ? MLP_UP_STAGE_INLINE_GAP_SMALL : MLP_UP_STAGE_INLINE_GAP
        }
    });

    const incomingArrowSpacerNode = createHiddenSpacer({
        semantic: buildSemantic(baseSemantic, {
            stage: 'mlp-input',
            role: 'incoming-arrow-spacer'
        }),
        role: 'incoming-arrow-spacer',
        width: isSmallScreen ? INCOMING_ARROW_SPACER_WIDTH_SMALL : INCOMING_ARROW_SPACER_WIDTH,
        height: Math.max(
            1,
            (rowCount * vectorMetrics.rowHeight)
            + (Math.max(0, rowCount - 1) * vectorMetrics.rowGap)
            + (vectorMetrics.paddingY * 2)
        )
    });

    const incomingConnectorNode = createConnectorNode({
        role: 'connector-mlp-input',
        semantic: buildSemantic(baseSemantic, {
            stage: 'connector-mlp-input',
            role: 'connector-mlp-input'
        }),
        source: createAnchorRef(incomingArrowSpacerNode.id, VIEW2D_ANCHOR_SIDES.LEFT),
        target: createAnchorRef(inputNode.id, VIEW2D_ANCHOR_SIDES.LEFT),
        route: VIEW2D_CONNECTOR_ROUTES.HORIZONTAL,
        gap: 0,
        targetGap: INPUT_CONNECTOR_TARGET_GAP,
        visual: {
            styleKey: VIEW2D_STYLE_KEYS.CONNECTOR_NEUTRAL
        },
        metadata: {
            preserveColor: true,
            strokeWidthScale: EDGE_CONNECTOR_STROKE_WIDTH_SCALE
        }
    });
    const geluConnectorNode = createConnectorNode({
        role: 'connector-mlp-gelu-input',
        semantic: buildSemantic(baseSemantic, {
            stage: 'connector-mlp-gelu-input',
            role: 'connector-mlp-gelu-input'
        }),
        source: createAnchorRef(outputNode.id, VIEW2D_ANCHOR_SIDES.BOTTOM),
        target: createAnchorRef(geluCopyNode.id, VIEW2D_ANCHOR_SIDES.TOP),
        route: VIEW2D_CONNECTOR_ROUTES.VERTICAL,
        gap: 0,
        sourceGap: GELU_CONNECTOR_SOURCE_GAP,
        targetGap: GELU_CONNECTOR_TARGET_GAP,
        visual: {
            styleKey: VIEW2D_STYLE_KEYS.CONNECTOR_NEUTRAL
        },
        metadata: {
            sourceAnchorMode: 'caption-bottom',
            preserveColor: true,
            strokeWidthScale: EDGE_CONNECTOR_STROKE_WIDTH_SCALE
        }
    });
    const mlpDownConnectorNode = createConnectorNode({
        role: 'connector-mlp-down-input',
        semantic: buildSemantic(baseSemantic, {
            stage: 'connector-mlp-down-input',
            role: 'connector-mlp-down-input'
        }),
        source: createAnchorRef(activationNode.id, VIEW2D_ANCHOR_SIDES.RIGHT),
        target: createAnchorRef(activationCopyNode.id, VIEW2D_ANCHOR_SIDES.LEFT),
        route: VIEW2D_CONNECTOR_ROUTES.HORIZONTAL,
        gap: 0,
        sourceGap: MLP_DOWN_CONNECTOR_SOURCE_GAP,
        targetGap: MLP_DOWN_CONNECTOR_TARGET_GAP,
        visual: {
            styleKey: VIEW2D_STYLE_KEYS.CONNECTOR_NEUTRAL
        },
        metadata: {
            preserveColor: true,
            strokeWidthScale: EDGE_CONNECTOR_STROKE_WIDTH_SCALE
        }
    });
    const mlpDownOutputOutgoingConnectorNode = createConnectorNode({
        role: 'connector-mlp-down-output-outgoing',
        semantic: buildSemantic(baseSemantic, {
            stage: 'connector-mlp-down-output-outgoing',
            role: 'connector-mlp-down-output-outgoing'
        }),
        source: createAnchorRef(downOutputNode.id, VIEW2D_ANCHOR_SIDES.RIGHT),
        target: createAnchorRef(outgoingArrowSpacerNode.id, VIEW2D_ANCHOR_SIDES.RIGHT),
        route: VIEW2D_CONNECTOR_ROUTES.HORIZONTAL,
        gap: 0,
        sourceGap: MLP_DOWN_OUTPUT_EDGE_SOURCE_GAP,
        visual: {
            styleKey: VIEW2D_STYLE_KEYS.CONNECTOR_NEUTRAL
        },
        metadata: {
            preserveColor: true,
            strokeWidthScale: EDGE_CONNECTOR_STROKE_WIDTH_SCALE,
            fixedScreenArrowHeadLengthPx: MLP_DOWN_OUTPUT_EDGE_ARROW_HEAD_LENGTH_SCREEN_PX,
            fixedScreenArrowHeadWingPx: MLP_DOWN_OUTPUT_EDGE_ARROW_HEAD_WING_SCREEN_PX,
            disableScreenSnap: true
        }
    });

    return createSceneModel({
        semantic: buildSemantic(baseSemantic, {
            role: 'scene'
        }),
        nodes: [
            createGroupNode({
                role: 'mlp-detail-stage',
                semantic: buildSemantic(baseSemantic, {
                    role: 'mlp-detail-stage'
                }),
                direction: VIEW2D_LAYOUT_DIRECTIONS.HORIZONTAL,
                align: 'start',
                gapKey: 'default',
                children: [
                    incomingArrowSpacerNode,
                    inputNode,
                    createMlpOperatorNode({
                        role: 'mlp-up-multiply',
                        semantic: buildSemantic(mlpUpWeightSemantic, {
                            role: 'mlp-up-multiply',
                            operatorKey: 'multiply'
                        }),
                        text: '×',
                        visual: {
                            styleKey: VIEW2D_STYLE_KEYS.OPERATOR
                        }
                    }),
                    outputStageNode,
                    downStageNode
                ],
                metadata: {
                    gapOverride: isSmallScreen ? MLP_UP_STAGE_INLINE_GAP_SMALL : MLP_UP_STAGE_INLINE_GAP
                }
            }),
            createGroupNode({
                role: 'connector-layer',
                semantic: buildSemantic(baseSemantic, {
                    stage: 'connector-layer',
                    role: 'connector-layer'
                }),
                direction: VIEW2D_LAYOUT_DIRECTIONS.OVERLAY,
                gapKey: 'default',
                children: [
                    incomingConnectorNode,
                    geluConnectorNode,
                    mlpDownConnectorNode,
                    mlpDownOutputOutgoingConnectorNode
                ]
            })
        ],
        metadata: {
            visualContract: 'selection-panel-mlp-v1',
            source: 'buildMlpDetailSceneModel',
            layerIndex,
            rowCount,
            isSmallScreen: !!isSmallScreen,
            layoutMetrics: {
                ...resolvedLayoutMetrics,
                cssVars: {
                    ...(resolvedLayoutMetrics?.cssVars || {}),
                    '--mhsa-token-matrix-canvas-pad-x-boost': isSmallScreen ? '-20px' : '-32px',
                    '--mhsa-token-matrix-canvas-pad-y-boost': isSmallScreen ? '-10px' : '-16px',
                    '--mhsa-token-matrix-stage-gap-boost': isSmallScreen ? '-6px' : '-8px'
                }
            },
            tokens: visualTokens || resolveView2dVisualTokens()
        }
    });
}
