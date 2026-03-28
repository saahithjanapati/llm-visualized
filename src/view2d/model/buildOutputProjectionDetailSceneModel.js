import {
    MHA_VALUE_SPECTRUM_COLOR,
    MHA_VALUE_HUE_SPREAD,
    MHA_VALUE_LIGHTNESS_MIN,
    MHA_VALUE_LIGHTNESS_MAX,
    MHA_VALUE_RANGE_MIN,
    MHA_VALUE_RANGE_MAX,
    MHA_VALUE_CLAMP_MAX,
    MHA_OUTPUT_PROJECTION_MATRIX_COLOR
} from '../../animations/LayerAnimationConstants.js';
import { getAttentionBiasVectorSample } from '../../data/biasParams.js';
import {
    D_HEAD,
    D_MODEL,
    RESIDUAL_COLOR_CLAMP
} from '../../ui/selectionPanelConstants.js';
import { NUM_HEAD_SETS_LAYER } from '../../utils/constants.js';
import {
    buildHueRangeOptions,
    mapValueToColor,
    mapValueToHueRange
} from '../../utils/colors.js';
import {
    buildSceneNodeId,
    createAnchorRef,
    createConnectorNode,
    createGroupNode,
    createMatrixNode,
    createOperatorNode,
    createSceneModel,
    createTextNode,
    VIEW2D_ANCHOR_SIDES,
    VIEW2D_CONNECTOR_ROUTES,
    VIEW2D_LAYOUT_DIRECTIONS,
    VIEW2D_MATRIX_PRESENTATIONS,
    VIEW2D_MATRIX_SHAPES
} from '../schema/sceneTypes.js';
import {
    resolveView2dVisualTokens,
    VIEW2D_STYLE_KEYS
} from '../theme/visualTokens.js';
import { createVectorStripMatrixNode } from './createResidualVectorMatrixNode.js';
import {
    createCaptionedCardMatrixNode,
    resolveRelativeCardSize
} from './createCaptionedCardMatrixNode.js';
import { createView2dVectorStripMetadata } from '../shared/vectorStrip.js';
import {
    resolveMhsaDimensionVisualExtent,
    resolveMhsaTokenVisualExtent
} from '../shared/mhsaDimensionSizing.js';

const HEAD_OUTPUT_MEASURE_COLS = 12;
const HEAD_OUTPUT_ROW_HEIGHT = 7;
const HEAD_OUTPUT_ROW_HEIGHT_SMALL = 6;
const HEAD_OUTPUT_STACK_GAP = 72;
const HEAD_OUTPUT_STACK_GAP_SMALL = 56;
const HEAD_OUTPUT_STACK_TOP_PADDING = 44;
const HEAD_OUTPUT_STACK_TOP_PADDING_SMALL = 32;
const OUTPUT_PROJECTION_STAGE_GAP = 96;
const OUTPUT_PROJECTION_STAGE_GAP_SMALL = 72;
const CONCAT_STAGE_GAP = 18;
const CONCAT_STAGE_GAP_SMALL = 14;
const CONCAT_LIST_ITEM_GAP = 12;
const CONCAT_LIST_ITEM_GAP_SMALL = 9;
const CONCAT_GROUP_GAP = 8;
const CONCAT_GROUP_GAP_SMALL = 6;
const CONCAT_LABEL_FONT_SCALE = 1.82;
const CONCAT_GROUPING_OPERATOR_SCALE = 1.38;
const CONCAT_ABOVE_HEAD_GAP = 20;
const CONCAT_ABOVE_HEAD_GAP_SMALL = 14;
const CONCAT_COPY_CONNECTOR_SOURCE_GAP = 8;
const CONCAT_COPY_CONNECTOR_TARGET_GAP = 10;
const CONCAT_OUTPUT_MEASURE_COLS = HEAD_OUTPUT_MEASURE_COLS * NUM_HEAD_SETS_LAYER;
const CONCAT_OUTPUT_BAND_SEPARATOR_OPACITY = 0.22;
const CONCAT_OUTPUT_COMPACT_WIDTH = 232;
const CONCAT_OUTPUT_COMPACT_WIDTH_SMALL = 196;
const INCOMING_ARROW_SPACER_WIDTH = 56;
const INCOMING_ARROW_SPACER_WIDTH_SMALL = 48;
const ARROW_HEAD_TARGET_GAP = 12;
const OUTPUT_HEAD_CAPTION_LABEL_SCALE = 0.9;
const OUTPUT_HEAD_SINGLE_ROW_LABEL_MIN_SCREEN_FONT_PX = 10.5;
const OUTPUT_HEAD_SINGLE_ROW_LABEL_MAX_SCREEN_FONT_PX = 13.5;
const OUTPUT_PROJECTION_VECTOR_ROW_HEIGHT = 7;
const OUTPUT_PROJECTION_VECTOR_ROW_HEIGHT_SMALL = 6;
const OUTPUT_PROJECTION_BIAS_ROW_HEIGHT = 14;
const OUTPUT_PROJECTION_BIAS_ROW_HEIGHT_SMALL = 12;
const OUTPUT_PROJECTION_BIAS_CORNER_RADIUS = 5;
const OUTPUT_PROJECTION_EQUATION_GAP = 12;
const OUTPUT_PROJECTION_EQUATION_GAP_SMALL = 10;
const OUTPUT_PROJECTION_WEIGHT_MIN_WIDTH = 84;
const OUTPUT_PROJECTION_WEIGHT_MAX_WIDTH = 148;
const OUTPUT_PROJECTION_WEIGHT_MIN_HEIGHT = 84;
const OUTPUT_PROJECTION_WEIGHT_MAX_HEIGHT = 148;
const OUTPUT_PROJECTION_OUTGOING_ARROW_SPACER_WIDTH = 60;
const OUTPUT_PROJECTION_OUTGOING_ARROW_SPACER_WIDTH_SMALL = 52;
const OUTPUT_PROJECTION_CONNECTOR_SOURCE_GAP = 8;
const OUTPUT_PROJECTION_CONNECTOR_TARGET_GAP = 12;
const OUTPUT_PROJECTION_BIAS_LABEL_SCALE = 1.58;
const OUTPUT_PROJECTION_MULTIPLY_OPERATOR_SCALE = 0.92;

const HEAD_OUTPUT_RANGE_OPTIONS = buildHueRangeOptions(MHA_VALUE_SPECTRUM_COLOR, {
    hueSpread: MHA_VALUE_HUE_SPREAD,
    minLightness: MHA_VALUE_LIGHTNESS_MIN,
    maxLightness: MHA_VALUE_LIGHTNESS_MAX,
    valueMin: MHA_VALUE_RANGE_MIN,
    valueMax: MHA_VALUE_RANGE_MAX,
    valueClampMax: MHA_VALUE_CLAMP_MAX
});
const OUTPUT_PROJECTION_RANGE_OPTIONS = buildHueRangeOptions(MHA_OUTPUT_PROJECTION_MATRIX_COLOR, {
    valueMin: -2,
    valueMax: 2,
    minLightness: 0.34,
    maxLightness: 0.72
});

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

function sampleVector(values = [], targetLength = HEAD_OUTPUT_MEASURE_COLS) {
    const safeValues = cleanNumberArray(values);
    const length = Math.max(1, Math.floor(targetLength || HEAD_OUTPUT_MEASURE_COLS));
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

function buildGradientCss(values = [], direction = '90deg', rangeOptions = HEAD_OUTPUT_RANGE_OPTIONS) {
    const safeValues = cleanNumberArray(values);
    if (!safeValues.length) return 'none';
    const stops = safeValues.map((value, index) => {
        const ratio = safeValues.length > 1 ? index / (safeValues.length - 1) : 0;
        return `${colorToCss(mapValueToHueRange(value, rangeOptions))} ${(ratio * 100).toFixed(4)}%`;
    });
    if (stops.length === 1) {
        return `linear-gradient(${direction}, ${stops[0].replace(' 0.0000%', ' 0%')}, ${stops[0].replace(' 0.0000%', ' 100%')})`;
    }
    return `linear-gradient(${direction}, ${stops.join(', ')})`;
}

function buildResidualGradientCss(values = [], direction = '90deg') {
    const safeValues = cleanNumberArray(values);
    if (!safeValues.length) return 'none';
    const stops = safeValues.map((value, index) => {
        const ratio = safeValues.length > 1 ? index / (safeValues.length - 1) : 0;
        return `${colorToCss(mapValueToColor(value, { clampMax: RESIDUAL_COLOR_CLAMP }))} ${(ratio * 100).toFixed(4)}%`;
    });
    if (stops.length === 1) {
        return `linear-gradient(${direction}, ${stops[0].replace(' 0.0000%', ' 0%')}, ${stops[0].replace(' 0.0000%', ' 100%')})`;
    }
    return `linear-gradient(${direction}, ${stops.join(', ')})`;
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

function resolveHeadOutputCompactWidth({
    isSmallScreen = false
} = {}) {
    return Math.max(
        isSmallScreen ? 92 : 104,
        Math.round(resolveMhsaDimensionVisualExtent(D_HEAD, {
            isSmallScreen,
            baseDimensionCount: D_HEAD,
            exponent: 0.18,
            baseExtentPx: isSmallScreen ? 98 : 112,
            minExtentPx: isSmallScreen ? 92 : 104,
            maxExtentPx: isSmallScreen ? 118 : 136
        }))
    );
}

function resolveOutputProjectionFeatureExtent({
    isSmallScreen = false
} = {}) {
    return Math.max(
        isSmallScreen ? 94 : 108,
        Math.round(resolveMhsaDimensionVisualExtent(D_MODEL, {
            isSmallScreen,
            baseDimensionCount: D_MODEL,
            exponent: 0.3,
            baseExtentPx: isSmallScreen ? 104 : 122,
            minExtentPx: isSmallScreen ? 94 : 108,
            maxExtentPx: isSmallScreen ? 124 : 144
        }))
    );
}

function resolveOutputProjectionVectorDimensions({
    rowCount = 1,
    baseCompactWidth = 72,
    baseRowHeight = OUTPUT_PROJECTION_VECTOR_ROW_HEIGHT,
    isSmallScreen = false
} = {}) {
    const safeRowCount = Number.isFinite(rowCount) ? Math.max(1, Math.floor(rowCount)) : 1;
    const safeBaseRowHeight = Math.max(1, Math.floor(Number(baseRowHeight) || OUTPUT_PROJECTION_VECTOR_ROW_HEIGHT));
    const rowTargetExtent = resolveMhsaTokenVisualExtent(safeRowCount, {
        isSmallScreen
    });
    const minRowHeight = safeRowCount >= 20 ? 2 : (safeRowCount >= 10 ? 3 : 4);
    return {
        compactWidth: Math.max(1, Math.floor(Number(baseCompactWidth) || 72)),
        rowHeight: Math.max(
            minRowHeight,
            Math.min(
                safeBaseRowHeight,
                Math.floor(rowTargetExtent / safeRowCount)
            )
        )
    };
}

function resolveOutputProjectionStageVisualMetrics(rowCount = 1, {
    isSmallScreen = false
} = {}) {
    const outputFeatureExtent = resolveOutputProjectionFeatureExtent({
        isSmallScreen
    });
    const outputDimensions = resolveOutputProjectionVectorDimensions({
        rowCount,
        baseCompactWidth: outputFeatureExtent,
        baseRowHeight: isSmallScreen ? OUTPUT_PROJECTION_VECTOR_ROW_HEIGHT_SMALL : OUTPUT_PROJECTION_VECTOR_ROW_HEIGHT,
        isSmallScreen
    });
    const weightCardSize = resolveRelativeCardSize({
        rows: D_MODEL,
        cols: D_MODEL,
        referenceCount: D_MODEL,
        referenceExtent: Math.round(outputDimensions.compactWidth * 1.08),
        minWidth: isSmallScreen ? OUTPUT_PROJECTION_WEIGHT_MIN_WIDTH - 8 : OUTPUT_PROJECTION_WEIGHT_MIN_WIDTH,
        maxWidth: isSmallScreen ? OUTPUT_PROJECTION_WEIGHT_MAX_WIDTH - 12 : OUTPUT_PROJECTION_WEIGHT_MAX_WIDTH,
        minHeight: isSmallScreen ? OUTPUT_PROJECTION_WEIGHT_MIN_HEIGHT - 8 : OUTPUT_PROJECTION_WEIGHT_MIN_HEIGHT,
        maxHeight: isSmallScreen ? OUTPUT_PROJECTION_WEIGHT_MAX_HEIGHT - 12 : OUTPUT_PROJECTION_WEIGHT_MAX_HEIGHT
    });
    weightCardSize.width = Math.min(
        isSmallScreen ? OUTPUT_PROJECTION_WEIGHT_MAX_WIDTH - 12 : OUTPUT_PROJECTION_WEIGHT_MAX_WIDTH,
        Math.max(
            isSmallScreen ? OUTPUT_PROJECTION_WEIGHT_MIN_WIDTH - 8 : OUTPUT_PROJECTION_WEIGHT_MIN_WIDTH,
            outputDimensions.compactWidth
        )
    );
    weightCardSize.height = weightCardSize.width;
    return {
        outputDimensions,
        biasCompactWidth: outputDimensions.compactWidth,
        biasRowHeight: isSmallScreen ? OUTPUT_PROJECTION_BIAS_ROW_HEIGHT_SMALL : OUTPUT_PROJECTION_BIAS_ROW_HEIGHT,
        weightCardSize
    };
}

function buildHeadOutputRowItems(activationSource = null, tokenRefs = [], {
    layerIndex = null,
    headIndex = null
} = {}) {
    return tokenRefs.map((tokenRef) => {
        const tokenIndex = normalizeIndex(tokenRef?.tokenIndex);
        const rawVector = typeof activationSource?.getAttentionWeightedSum === 'function'
            ? activationSource.getAttentionWeightedSum(layerIndex, headIndex, tokenIndex, D_HEAD)
            : null;
        const vectorValues = cleanNumberArray(rawVector, D_HEAD);
        const sampledValues = sampleVector(
            vectorValues,
            HEAD_OUTPUT_MEASURE_COLS
        );
        const label = typeof tokenRef?.tokenLabel === 'string' && tokenRef.tokenLabel.length
            ? tokenRef.tokenLabel
            : `Token ${(Number(tokenRef?.rowIndex) || 0) + 1}`;
        const semantic = {
            componentKind: 'output-projection',
            layerIndex,
            headIndex,
            stage: 'head-output',
            role: 'head-output-row',
            rowIndex: normalizeIndex(tokenRef?.rowIndex) ?? 0,
            ...(Number.isFinite(tokenIndex) ? { tokenIndex } : {})
        };
        return {
            id: buildSceneNodeId(semantic),
            index: normalizeIndex(tokenRef?.rowIndex) ?? 0,
            label,
            semantic,
            vectorValues,
            rawValues: sampledValues,
            gradientCss: buildGradientCss(sampledValues),
            title: `${label}: H_${Math.max(1, (headIndex ?? 0) + 1)}`
        };
    });
}

function sampleHeadOutputValues(activationSource = null, {
    layerIndex = null,
    headIndex = null,
    tokenIndex = null
} = {}) {
    const rawVector = typeof activationSource?.getAttentionWeightedSum === 'function'
        ? activationSource.getAttentionWeightedSum(layerIndex, headIndex, tokenIndex, D_HEAD)
        : null;
    return sampleVector(
        cleanNumberArray(rawVector, D_HEAD),
        HEAD_OUTPUT_MEASURE_COLS
    );
}

function buildConcatOutputRowItems(activationSource = null, tokenRefs = [], {
    layerIndex = null,
    stage = 'concatenate',
    rowRole = 'concat-output-row'
} = {}) {
    return tokenRefs.map((tokenRef) => {
        const tokenIndex = normalizeIndex(tokenRef?.tokenIndex);
        const label = typeof tokenRef?.tokenLabel === 'string' && tokenRef.tokenLabel.length
            ? tokenRef.tokenLabel
            : `Token ${(Number(tokenRef?.rowIndex) || 0) + 1}`;
        const sampledValues = Array.from({ length: NUM_HEAD_SETS_LAYER }, (_, headIndex) => (
            sampleHeadOutputValues(activationSource, {
                layerIndex,
                headIndex,
                tokenIndex
            })
        )).flat();
        const semantic = {
            componentKind: 'output-projection',
            layerIndex,
            stage,
            role: rowRole,
            rowIndex: normalizeIndex(tokenRef?.rowIndex) ?? 0,
            ...(Number.isFinite(tokenIndex) ? { tokenIndex } : {})
        };
        return {
            id: buildSceneNodeId(semantic),
            index: normalizeIndex(tokenRef?.rowIndex) ?? 0,
            label,
            semantic,
            rawValues: sampledValues,
            gradientCss: buildGradientCss(sampledValues),
            title: `${label}: concat(H)`
        };
    });
}

function buildOutputProjectionRowItems(activationSource = null, tokenRefs = [], {
    layerIndex = null
} = {}) {
    return tokenRefs.map((tokenRef) => {
        const tokenIndex = normalizeIndex(tokenRef?.tokenIndex);
        const rawVector = typeof activationSource?.getAttentionOutputProjection === 'function'
            ? activationSource.getAttentionOutputProjection(layerIndex, tokenIndex, D_MODEL)
            : null;
        const sampledValues = sampleVector(
            cleanNumberArray(rawVector, D_MODEL),
            HEAD_OUTPUT_MEASURE_COLS
        );
        const label = typeof tokenRef?.tokenLabel === 'string' && tokenRef.tokenLabel.length
            ? tokenRef.tokenLabel
            : `Token ${(Number(tokenRef?.rowIndex) || 0) + 1}`;
        const semantic = {
            componentKind: 'output-projection',
            layerIndex,
            stage: 'attn-out',
            role: 'projection-output-row',
            rowIndex: normalizeIndex(tokenRef?.rowIndex) ?? 0,
            ...(Number.isFinite(tokenIndex) ? { tokenIndex } : {})
        };
        return {
            id: buildSceneNodeId(semantic),
            index: normalizeIndex(tokenRef?.rowIndex) ?? 0,
            label,
            semantic,
            rawValues: sampledValues,
            gradientCss: buildResidualGradientCss(sampledValues),
            title: `${label}: O`
        };
    });
}

function buildOutputProjectionBiasRowItems(layerIndex = null) {
    const rawValues = sampleVector(
        cleanNumberArray(getAttentionBiasVectorSample(layerIndex, 'output'), HEAD_OUTPUT_MEASURE_COLS),
        HEAD_OUTPUT_MEASURE_COLS
    );
    const rowSemantic = {
        componentKind: 'output-projection',
        layerIndex,
        stage: 'attn-out',
        role: 'projection-bias-row',
        rowIndex: 0
    };
    return [{
        id: buildSceneNodeId(rowSemantic),
        index: 0,
        label: '',
        semantic: rowSemantic,
        rawValues,
        rawValue: rawValues[0] ?? 0,
        gradientCss: buildGradientCss(rawValues, '90deg', OUTPUT_PROJECTION_RANGE_OPTIONS),
        title: 'b_O'
    }];
}

function createHeadOutputMatrixNode(activationSource = null, tokenRefs = [], {
    layerIndex = null,
    headIndex = null,
    rowCount = 1,
    isSmallScreen = false,
    role = 'head-output-matrix',
    semanticRole = 'head-output-matrix',
    compactWidth = null,
    rowHeight = null,
    captionLabelScale = OUTPUT_HEAD_CAPTION_LABEL_SCALE
} = {}) {
    const baseSemantic = {
        componentKind: 'output-projection',
        layerIndex,
        headIndex,
        stage: 'head-output'
    };
    const resolvedCompactWidth = Number.isFinite(compactWidth) && compactWidth > 0
        ? Math.max(1, Math.floor(compactWidth))
        : resolveHeadOutputCompactWidth({
            isSmallScreen
        });
    const resolvedRowHeight = Number.isFinite(rowHeight) && rowHeight > 0
        ? Math.max(1, Math.floor(rowHeight))
        : (isSmallScreen ? HEAD_OUTPUT_ROW_HEIGHT_SMALL : HEAD_OUTPUT_ROW_HEIGHT);
    const useFixedSingleRowCaptionSizing = rowCount === 1;
    const matrixNode = createVectorStripMatrixNode({
        role,
        semantic: buildSemantic(baseSemantic, {
            role: semanticRole
        }),
        labelTex: `H_{${headIndex + 1}}`,
        labelText: `H_${headIndex + 1}`,
        rowItems: buildHeadOutputRowItems(activationSource, tokenRefs, {
            layerIndex,
            headIndex
        }),
        rowCount,
        columnCount: D_HEAD,
        measureCols: HEAD_OUTPUT_MEASURE_COLS,
        compactWidth: resolvedCompactWidth,
        rowHeight: resolvedRowHeight,
        captionPosition: 'bottom',
        captionLabelScale,
        captionUniformLabelScale: captionLabelScale,
        visualStyleKey: VIEW2D_STYLE_KEYS.MHSA_HEAD_OUTPUT,
        stripMetadata: createView2dVectorStripMetadata({
            compactWidth: resolvedCompactWidth,
            rowHeight: resolvedRowHeight,
            rowGap: 0,
            paddingX: 0,
            paddingY: 0,
            cornerRadius: 10,
            bandCount: HEAD_OUTPUT_MEASURE_COLS,
            hoverScaleY: 1.12,
            hoverGlowBlur: 10,
            hideSurface: true
        }),
        metadata: {
            kind: 'head-output',
            headIndex
        }
    });
    if (useFixedSingleRowCaptionSizing && matrixNode?.metadata?.caption) {
        matrixNode.metadata.caption.lines = [{
            tex: matrixNode?.label?.tex || `H_{${headIndex + 1}}`,
            text: matrixNode?.label?.text || `H_${headIndex + 1}`
        }];
        delete matrixNode.metadata.caption.dimensionsTex;
        delete matrixNode.metadata.caption.dimensionsText;
        matrixNode.metadata.caption.labelMinScreenFontPx = OUTPUT_HEAD_SINGLE_ROW_LABEL_MIN_SCREEN_FONT_PX;
        matrixNode.metadata.caption.labelMaxScreenFontPx = OUTPUT_HEAD_SINGLE_ROW_LABEL_MAX_SCREEN_FONT_PX;
        delete matrixNode.metadata.caption.labelFixedScreenFontPx;
        delete matrixNode.metadata.caption.dimensionsMaxScreenFontPx;
        delete matrixNode.metadata.caption.dimensionsFixedScreenFontPx;
    }
    return matrixNode;
}

function createConcatOutputMatrixNode(activationSource = null, tokenRefs = [], {
    layerIndex = null,
    rowCount = 1,
    isSmallScreen = false,
    role = 'concat-output-matrix',
    semanticRole = 'concat-output-matrix',
    semanticStage = 'concatenate',
    rowRole = 'concat-output-row'
} = {}) {
    const compactWidth = isSmallScreen ? CONCAT_OUTPUT_COMPACT_WIDTH_SMALL : CONCAT_OUTPUT_COMPACT_WIDTH;
    const rowHeight = isSmallScreen ? HEAD_OUTPUT_ROW_HEIGHT_SMALL : HEAD_OUTPUT_ROW_HEIGHT;
    return createVectorStripMatrixNode({
        role,
        semantic: {
            componentKind: 'output-projection',
            layerIndex,
            stage: semanticStage,
            role: semanticRole
        },
        labelTex: 'H_{\\mathrm{concat}}',
        labelText: 'H_concat',
        rowItems: buildConcatOutputRowItems(activationSource, tokenRefs, {
            layerIndex,
            stage: semanticStage,
            rowRole
        }),
        rowCount,
        columnCount: D_HEAD * NUM_HEAD_SETS_LAYER,
        measureCols: CONCAT_OUTPUT_MEASURE_COLS,
        compactWidth,
        rowHeight,
        captionPosition: 'bottom',
        captionLabelScale: OUTPUT_HEAD_CAPTION_LABEL_SCALE,
        visualStyleKey: VIEW2D_STYLE_KEYS.MHSA_HEAD_OUTPUT,
        stripMetadata: createView2dVectorStripMetadata({
            compactWidth,
            rowHeight,
            rowGap: 0,
            paddingX: 0,
            paddingY: 0,
            cornerRadius: 10,
            bandCount: NUM_HEAD_SETS_LAYER,
            bandSeparatorOpacity: CONCAT_OUTPUT_BAND_SEPARATOR_OPACITY,
            hoverScaleY: 1.12,
            hoverGlowBlur: 10,
            hideSurface: true
        }),
        metadata: {
            kind: 'concat-output',
            interactiveBandHit: true
        }
    });
}

function createHeadOutputEntry(activationSource = null, tokenRefs = [], {
    layerIndex = null,
    headIndex = null,
    rowCount = 1,
    isSmallScreen = false
} = {}) {
    const baseSemantic = {
        componentKind: 'output-projection',
        layerIndex,
        headIndex,
        stage: 'head-output'
    };
    const matrixNode = createHeadOutputMatrixNode(activationSource, tokenRefs, {
        layerIndex,
        headIndex,
        rowCount,
        isSmallScreen
    });
    const spacerNode = createHiddenSpacer({
        semantic: buildSemantic(baseSemantic, {
            role: 'incoming-arrow-spacer'
        }),
        role: 'incoming-arrow-spacer',
        width: isSmallScreen ? INCOMING_ARROW_SPACER_WIDTH_SMALL : INCOMING_ARROW_SPACER_WIDTH,
        height: 1
    });
    const rowNode = createGroupNode({
        role: 'head-output-row-group',
        semantic: buildSemantic(baseSemantic, {
            role: 'head-output-row-group'
        }),
        direction: VIEW2D_LAYOUT_DIRECTIONS.HORIZONTAL,
        gapKey: 'default',
        children: [
            spacerNode,
            matrixNode
        ],
        metadata: {
            gapOverride: 0
        }
    });
    const connectorNode = createConnectorNode({
        role: 'head-output-connector',
        semantic: buildSemantic(baseSemantic, {
            role: 'head-output-connector'
        }),
        source: createAnchorRef(spacerNode.id, VIEW2D_ANCHOR_SIDES.LEFT),
        target: createAnchorRef(matrixNode.id, VIEW2D_ANCHOR_SIDES.LEFT),
        route: VIEW2D_CONNECTOR_ROUTES.HORIZONTAL,
        sourceGap: 0,
        targetGap: ARROW_HEAD_TARGET_GAP,
        visual: {
            styleKey: VIEW2D_STYLE_KEYS.CONNECTOR_NEUTRAL,
            stroke: 'rgba(255, 255, 255, 0.84)'
        },
        metadata: {
            preserveColor: true,
            strokeWidthScale: 0.72
        }
    });

    return {
        rowNode,
        matrixNode,
        connectorNode
    };
}

function createOutputProjectionDetailTextNode({
    role = '',
    semantic = {},
    text = '',
    tex = '',
    fontScale = 1
} = {}) {
    return createTextNode({
        role,
        semantic,
        text,
        tex,
        visual: {
            styleKey: VIEW2D_STYLE_KEYS.LABEL
        },
        metadata: {
            renderMode: 'dom-katex',
            minScreenHeightPx: 0,
            fontScale
        }
    });
}

function createOutputProjectionDetailOperatorNode({
    role = '',
    semantic = {},
    text = '',
    fontScale = 1
} = {}) {
    return createOperatorNode({
        role,
        semantic,
        text,
        visual: {
            styleKey: VIEW2D_STYLE_KEYS.OPERATOR
        },
        metadata: {
            renderMode: 'dom-katex',
            fontScale
        }
    });
}

function buildOutputProjectionSubscriptLabel(labelTex = '') {
    const safeLabelTex = typeof labelTex === 'string' ? labelTex.trim() : '';
    const simpleSubscriptMatch = safeLabelTex.match(/^([A-Za-z]+)_([A-Za-z]+)$/);
    if (!simpleSubscriptMatch) {
        return {
            labelTex: safeLabelTex,
            labelText: safeLabelTex
        };
    }
    const [, base, subscript] = simpleSubscriptMatch;
    return {
        labelTex: `${base}_{\\mathrm{${subscript}}}`,
        labelText: safeLabelTex
    };
}

function createConcatStageNode({
    activationSource = null,
    tokenRefs = [],
    layerIndex = null,
    rowCount = 1,
    alignTargetNodeId = '',
    isSmallScreen = false
} = {}) {
    const concatSemantic = {
        componentKind: 'output-projection',
        layerIndex,
        stage: 'concatenate',
        role: 'concat'
    };

    const concatEntries = Array.from({ length: NUM_HEAD_SETS_LAYER }, (_, headIndex) => {
        const itemSemantic = buildSemantic(concatSemantic, {
            role: 'concat-head-copy',
            headIndex
        });
        const matrixNode = createHeadOutputMatrixNode(activationSource, tokenRefs, {
            layerIndex,
            headIndex,
            rowCount,
            isSmallScreen,
            role: 'concat-head-copy-matrix',
            semanticRole: 'concat-head-copy-matrix'
        });
        const groupNode = createGroupNode({
            role: 'concat-head-copy-group',
            semantic: buildSemantic(itemSemantic, {
                role: 'concat-head-copy-group'
            }),
            direction: VIEW2D_LAYOUT_DIRECTIONS.HORIZONTAL,
            gapKey: 'inline',
            align: 'center',
            children: [
                matrixNode,
                ...(headIndex < (NUM_HEAD_SETS_LAYER - 1)
                    ? [createOutputProjectionDetailOperatorNode({
                        role: 'concat-separator',
                        semantic: buildSemantic(concatSemantic, {
                            role: 'concat-separator',
                            headIndex,
                            operatorKey: 'comma'
                        }),
                        text: ','
                    })]
                    : [])
            ],
            metadata: {
                gapOverride: isSmallScreen ? 2 : 3
            }
        });
        return {
            groupNode,
            matrixNode
        };
    });

    const concatListNode = createGroupNode({
        role: 'concat-list',
        semantic: buildSemantic(concatSemantic, {
            role: 'concat-list'
        }),
        direction: VIEW2D_LAYOUT_DIRECTIONS.HORIZONTAL,
        gapKey: 'inline',
        align: 'center',
        children: concatEntries.map((entry) => entry.groupNode),
        metadata: {
            gapOverride: isSmallScreen ? CONCAT_LIST_ITEM_GAP_SMALL : CONCAT_LIST_ITEM_GAP
        }
    });

    const concatWrappedListNode = createGroupNode({
        role: 'concat-group',
        semantic: buildSemantic(concatSemantic, {
            role: 'concat-group'
        }),
        direction: VIEW2D_LAYOUT_DIRECTIONS.HORIZONTAL,
        gapKey: 'inline',
        align: 'center',
        children: [
            createOutputProjectionDetailOperatorNode({
                role: 'concat-open',
                semantic: buildSemantic(concatSemantic, {
                    role: 'concat-open',
                    operatorKey: 'open'
                }),
                text: '(',
                fontScale: CONCAT_GROUPING_OPERATOR_SCALE
            }),
            concatListNode,
            createOutputProjectionDetailOperatorNode({
                role: 'concat-close',
                semantic: buildSemantic(concatSemantic, {
                    role: 'concat-close',
                    operatorKey: 'close'
                }),
                text: ')',
                fontScale: CONCAT_GROUPING_OPERATOR_SCALE
            })
        ],
        metadata: {
            gapOverride: isSmallScreen ? CONCAT_GROUP_GAP_SMALL : CONCAT_GROUP_GAP
        }
    });
    const concatOutputNode = createConcatOutputMatrixNode(activationSource, tokenRefs, {
        layerIndex,
        rowCount,
        isSmallScreen
    });

    const node = createGroupNode({
        role: 'concat',
        semantic: concatSemantic,
        direction: VIEW2D_LAYOUT_DIRECTIONS.HORIZONTAL,
        gapKey: 'inline',
        align: 'center',
        layout: alignTargetNodeId
            ? {
                anchorAlign: {
                    axis: 'y',
                    targetNodeId: alignTargetNodeId,
                    selfAnchor: VIEW2D_ANCHOR_SIDES.BOTTOM,
                    targetAnchor: VIEW2D_ANCHOR_SIDES.TOP,
                    offset: -(isSmallScreen ? CONCAT_ABOVE_HEAD_GAP_SMALL : CONCAT_ABOVE_HEAD_GAP)
                }
            }
            : null,
        children: [
            createOutputProjectionDetailTextNode({
                role: 'concat-label',
                semantic: buildSemantic(concatSemantic, {
                    role: 'concat-label'
                }),
                text: 'concat',
                tex: '\\mathrm{concat}',
                fontScale: CONCAT_LABEL_FONT_SCALE
            }),
            concatWrappedListNode,
            createOutputProjectionDetailOperatorNode({
                role: 'concat-equals',
                semantic: buildSemantic(concatSemantic, {
                    role: 'concat-equals',
                    operatorKey: 'equals'
                }),
                text: '='
            }),
            concatOutputNode
        ],
        metadata: {
            gapOverride: isSmallScreen ? CONCAT_STAGE_GAP_SMALL : CONCAT_STAGE_GAP
        }
    });
    return {
        node,
        copyMatrixNodes: concatEntries.map((entry) => entry.matrixNode),
        concatOutputNode
    };
}

function createOutputProjectionStageNode({
    activationSource = null,
    tokenRefs = [],
    layerIndex = null,
    rowCount = 1,
    alignTargetNodeId = '',
    isSmallScreen = false
} = {}) {
    const projectionSemantic = {
        componentKind: 'output-projection',
        layerIndex,
        stage: 'attn-out',
        role: 'projection-stage'
    };
    const visualMetrics = resolveOutputProjectionStageVisualMetrics(rowCount, {
        isSmallScreen
    });
    const concatOutputCopyNode = createConcatOutputMatrixNode(activationSource, tokenRefs, {
        layerIndex,
        rowCount,
        isSmallScreen,
        role: 'concat-output-copy-matrix',
        semanticRole: 'concat-output-copy-matrix',
        semanticStage: 'attn-out',
        rowRole: 'concat-output-copy-row'
    });
    const weightNode = createCaptionedCardMatrixNode({
        role: 'projection-weight',
        semantic: buildSemantic(projectionSemantic, {
            role: 'projection-weight'
        }),
        ...buildOutputProjectionSubscriptLabel('W_O'),
        rowCount: D_MODEL,
        columnCount: D_MODEL,
        cardWidth: visualMetrics.weightCardSize.width,
        cardHeight: visualMetrics.weightCardSize.height,
        cardCornerRadius: 10,
        captionPosition: 'bottom',
        captionMinScreenHeightPx: 28,
        captionLabelScale: 1.12,
        captionDimensionsScale: 0.94,
        captionPreferStandardSizing: true,
        visualStyleKey: VIEW2D_STYLE_KEYS.OUTPUT_PROJECTION,
        disableCardSurfaceEffects: true,
        metadata: {
            kind: 'output-projection'
        }
    });
    const biasNode = createVectorStripMatrixNode({
        role: 'projection-bias',
        semantic: buildSemantic(projectionSemantic, {
            role: 'projection-bias'
        }),
        ...buildOutputProjectionSubscriptLabel('b_O'),
        rowItems: buildOutputProjectionBiasRowItems(layerIndex),
        rowCount: 1,
        columnCount: D_MODEL,
        measureCols: HEAD_OUTPUT_MEASURE_COLS,
        compactWidth: visualMetrics.biasCompactWidth,
        rowHeight: visualMetrics.biasRowHeight,
        captionPosition: 'bottom',
        captionMinScreenHeightPx: 12,
        captionPreferStandardSizing: true,
        captionLabelScale: OUTPUT_PROJECTION_BIAS_LABEL_SCALE,
        visualStyleKey: VIEW2D_STYLE_KEYS.OUTPUT_PROJECTION,
        stripMetadata: createView2dVectorStripMetadata({
            compactWidth: visualMetrics.biasCompactWidth,
            rowHeight: visualMetrics.biasRowHeight,
            cornerRadius: OUTPUT_PROJECTION_BIAS_CORNER_RADIUS,
            bandCount: HEAD_OUTPUT_MEASURE_COLS,
            hoverScaleY: 1.12,
            hoverGlowBlur: 10,
            hideSurface: true
        }),
        metadata: {
            kind: 'output-projection'
        }
    });
    const outputNode = createVectorStripMatrixNode({
        role: 'projection-output',
        semantic: buildSemantic(projectionSemantic, {
            role: 'projection-output'
        }),
        labelTex: 'O',
        labelText: 'O',
        rowItems: buildOutputProjectionRowItems(activationSource, tokenRefs, {
            layerIndex
        }),
        rowCount,
        columnCount: D_MODEL,
        measureCols: HEAD_OUTPUT_MEASURE_COLS,
        compactWidth: visualMetrics.outputDimensions.compactWidth,
        rowHeight: visualMetrics.outputDimensions.rowHeight,
        captionPosition: 'bottom',
        captionLabelScale: OUTPUT_HEAD_CAPTION_LABEL_SCALE,
        visualStyleKey: VIEW2D_STYLE_KEYS.RESIDUAL,
        stripMetadata: createView2dVectorStripMetadata({
            compactWidth: visualMetrics.outputDimensions.compactWidth,
            rowHeight: visualMetrics.outputDimensions.rowHeight,
            cornerRadius: 10,
            bandCount: HEAD_OUTPUT_MEASURE_COLS,
            hoverScaleY: 1.12,
            hoverGlowBlur: 10,
            hideSurface: true
        }),
        metadata: {
            kind: 'output-projection'
        }
    });
    const outgoingArrowSpacerNode = createHiddenSpacer({
        semantic: buildSemantic(projectionSemantic, {
            role: 'projection-output-arrow-spacer'
        }),
        role: 'projection-output-arrow-spacer',
        width: isSmallScreen ? OUTPUT_PROJECTION_OUTGOING_ARROW_SPACER_WIDTH_SMALL : OUTPUT_PROJECTION_OUTGOING_ARROW_SPACER_WIDTH,
        height: 1
    });
    const equationNode = createGroupNode({
        role: 'output-projection-equation',
        semantic: buildSemantic(projectionSemantic, {
            role: 'output-projection-equation'
        }),
        direction: VIEW2D_LAYOUT_DIRECTIONS.HORIZONTAL,
        gapKey: 'inline',
        align: 'center',
        children: [
            concatOutputCopyNode,
            createOutputProjectionDetailOperatorNode({
                role: 'projection-multiply',
                semantic: buildSemantic(projectionSemantic, {
                    role: 'projection-multiply',
                    operatorKey: 'multiply'
                }),
                text: '×',
                fontScale: OUTPUT_PROJECTION_MULTIPLY_OPERATOR_SCALE
            }),
            weightNode,
            createOutputProjectionDetailOperatorNode({
                role: 'projection-plus',
                semantic: buildSemantic(projectionSemantic, {
                    role: 'projection-plus',
                    operatorKey: 'plus'
                }),
                text: '+'
            }),
            biasNode,
            createOutputProjectionDetailOperatorNode({
                role: 'projection-equals',
                semantic: buildSemantic(projectionSemantic, {
                    role: 'projection-equals',
                    operatorKey: 'equals'
                }),
                text: '='
            }),
            outputNode,
            outgoingArrowSpacerNode
        ],
        metadata: {
            gapOverride: isSmallScreen ? OUTPUT_PROJECTION_EQUATION_GAP_SMALL : OUTPUT_PROJECTION_EQUATION_GAP
        }
    });

    return {
        node: createGroupNode({
            role: 'output-projection-stage',
            semantic: buildSemantic(projectionSemantic, {
                role: 'output-projection-stage'
            }),
            direction: VIEW2D_LAYOUT_DIRECTIONS.HORIZONTAL,
            gapKey: 'default',
            align: 'center',
            layout: alignTargetNodeId
                ? {
                    anchorAlign: {
                        axis: 'y',
                        targetNodeId: alignTargetNodeId,
                        selfAnchor: VIEW2D_ANCHOR_SIDES.CENTER,
                        targetAnchor: VIEW2D_ANCHOR_SIDES.CENTER,
                        offset: 0
                    }
                }
                : null,
            children: [
                equationNode
            ],
            metadata: {
                gapOverride: 0
            }
        }),
        concatOutputCopyNode,
        weightNode,
        biasNode,
        outputNode,
        outgoingArrowSpacerNode
    };
}

export function buildOutputProjectionDetailSceneModel({
    activationSource = null,
    outputProjectionDetailTarget = null,
    tokenRefs = [],
    visualTokens = null,
    isSmallScreen = false
} = {}) {
    const layerIndex = normalizeIndex(outputProjectionDetailTarget?.layerIndex);
    if (!Number.isFinite(layerIndex) || !tokenRefs.length) {
        return null;
    }

    const rowCount = Math.max(1, tokenRefs.length);
    const headEntries = Array.from({ length: NUM_HEAD_SETS_LAYER }, (_, headIndex) => createHeadOutputEntry(
        activationSource,
        tokenRefs,
        {
            layerIndex,
            headIndex,
            rowCount,
            isSmallScreen
        }
    ));
    const headStackTopSpacerNode = createHiddenSpacer({
        semantic: {
            componentKind: 'output-projection',
            layerIndex,
            stage: 'detail',
            role: 'output-projection-detail-head-stack-top-spacer'
        },
        role: 'output-projection-detail-head-stack-top-spacer',
        width: 1,
        height: isSmallScreen ? HEAD_OUTPUT_STACK_TOP_PADDING_SMALL : HEAD_OUTPUT_STACK_TOP_PADDING
    });
    const headRowsNode = createGroupNode({
        role: 'output-projection-detail-head-rows',
        semantic: {
            componentKind: 'output-projection',
            layerIndex,
            stage: 'detail',
            role: 'output-projection-detail-head-rows'
        },
        direction: VIEW2D_LAYOUT_DIRECTIONS.VERTICAL,
        gapKey: 'default',
        children: headEntries.map((entry) => entry.rowNode),
        metadata: {
            gapOverride: isSmallScreen ? HEAD_OUTPUT_STACK_GAP_SMALL : HEAD_OUTPUT_STACK_GAP
        }
    });
    const headStackNode = createGroupNode({
        role: 'output-projection-detail-head-stack',
        semantic: {
            componentKind: 'output-projection',
            layerIndex,
            stage: 'detail',
            role: 'output-projection-detail-head-stack'
        },
        direction: VIEW2D_LAYOUT_DIRECTIONS.VERTICAL,
        gapKey: 'default',
        children: [
            headStackTopSpacerNode,
            headRowsNode
        ],
        metadata: {
            gapOverride: 0
        }
    });
    const concatStage = createConcatStageNode({
        activationSource,
        tokenRefs,
        layerIndex,
        rowCount,
        alignTargetNodeId: headStackTopSpacerNode.id,
        isSmallScreen
    });
    const outputProjectionStage = createOutputProjectionStageNode({
        activationSource,
        tokenRefs,
        layerIndex,
        rowCount,
        alignTargetNodeId: concatStage.concatOutputNode.id,
        isSmallScreen
    });
    const concatCopyConnectorNodes = headEntries.map((entry, headIndex) => {
        const copyMatrixNode = concatStage.copyMatrixNodes[headIndex] || null;
        if (!entry?.matrixNode?.id || !copyMatrixNode?.id) return null;
        return createConnectorNode({
            role: 'concat-copy-connector',
            semantic: {
                componentKind: 'output-projection',
                layerIndex,
                headIndex,
                stage: 'concatenate',
                role: 'concat-copy-connector'
            },
            source: createAnchorRef(entry.matrixNode.id, VIEW2D_ANCHOR_SIDES.RIGHT),
            target: createAnchorRef(copyMatrixNode.id, VIEW2D_ANCHOR_SIDES.BOTTOM),
            route: VIEW2D_CONNECTOR_ROUTES.ELBOW,
            sourceGap: CONCAT_COPY_CONNECTOR_SOURCE_GAP,
            targetGap: CONCAT_COPY_CONNECTOR_TARGET_GAP,
            visual: {
                styleKey: VIEW2D_STYLE_KEYS.CONNECTOR_NEUTRAL,
                stroke: 'rgba(255, 255, 255, 0.84)'
            },
            metadata: {
                targetAnchorMode: 'caption-bottom',
                preserveColor: true,
                preserveFocusOpacity: true,
                strokeWidthScale: 0.72
            }
        });
    }).filter(Boolean);
    const concatToProjectionConnectorNode = createConnectorNode({
        role: 'concat-output-projection-connector',
        semantic: {
            componentKind: 'output-projection',
            layerIndex,
            stage: 'attn-out',
            role: 'concat-output-projection-connector'
        },
        source: createAnchorRef(concatStage.concatOutputNode.id, VIEW2D_ANCHOR_SIDES.RIGHT),
        target: createAnchorRef(outputProjectionStage.concatOutputCopyNode.id, VIEW2D_ANCHOR_SIDES.LEFT),
        route: VIEW2D_CONNECTOR_ROUTES.HORIZONTAL,
        sourceGap: OUTPUT_PROJECTION_CONNECTOR_SOURCE_GAP,
        targetGap: OUTPUT_PROJECTION_CONNECTOR_TARGET_GAP,
        visual: {
            styleKey: VIEW2D_STYLE_KEYS.CONNECTOR_NEUTRAL,
            stroke: 'rgba(255, 255, 255, 0.84)'
        },
        metadata: {
            preserveColor: true,
            strokeWidthScale: 0.72
        }
    });
    const projectionOutputConnectorNode = createConnectorNode({
        role: 'projection-output-connector',
        semantic: {
            componentKind: 'output-projection',
            layerIndex,
            stage: 'attn-out',
            role: 'projection-output-connector'
        },
        source: createAnchorRef(outputProjectionStage.outputNode.id, VIEW2D_ANCHOR_SIDES.RIGHT),
        target: createAnchorRef(outputProjectionStage.outgoingArrowSpacerNode.id, VIEW2D_ANCHOR_SIDES.RIGHT),
        route: VIEW2D_CONNECTOR_ROUTES.HORIZONTAL,
        sourceGap: OUTPUT_PROJECTION_CONNECTOR_SOURCE_GAP,
        targetGap: 0,
        visual: {
            styleKey: VIEW2D_STYLE_KEYS.CONNECTOR_NEUTRAL,
            stroke: 'rgba(255, 255, 255, 0.84)'
        },
        metadata: {
            preserveColor: true,
            strokeWidthScale: 0.72
        }
    });

    return createSceneModel({
        semantic: {
            componentKind: 'output-projection',
            layerIndex,
            stage: 'detail',
            role: 'scene'
        },
        nodes: [
            createGroupNode({
                role: 'output-projection-detail-stage',
                semantic: {
                    componentKind: 'output-projection',
                    layerIndex,
                    stage: 'detail',
                    role: 'output-projection-detail-stage'
                },
                direction: VIEW2D_LAYOUT_DIRECTIONS.HORIZONTAL,
                gapKey: 'stage',
                align: 'center',
                children: [
                    headStackNode,
                    concatStage.node,
                    outputProjectionStage.node
                ],
                metadata: {
                    gapOverride: isSmallScreen ? OUTPUT_PROJECTION_STAGE_GAP_SMALL : OUTPUT_PROJECTION_STAGE_GAP
                }
            }),
            createGroupNode({
                role: 'output-projection-detail-connectors',
                semantic: {
                    componentKind: 'output-projection',
                    layerIndex,
                    stage: 'detail',
                    role: 'output-projection-detail-connectors'
                },
                direction: VIEW2D_LAYOUT_DIRECTIONS.OVERLAY,
                gapKey: 'default',
                children: [
                    ...headEntries.map((entry) => entry.connectorNode),
                    ...concatCopyConnectorNodes,
                    concatToProjectionConnectorNode,
                    projectionOutputConnectorNode
                ]
            })
        ],
        metadata: {
            visualContract: 'selection-panel-output-projection-v1',
            source: 'buildOutputProjectionDetailSceneModel',
            layerIndex,
            headCount: NUM_HEAD_SETS_LAYER,
            rowCount,
            isSmallScreen: !!isSmallScreen,
            layoutMetrics: {
                cssVars: {
                    '--mhsa-token-matrix-canvas-pad-x-boost': isSmallScreen ? '-16px' : '-28px',
                    '--mhsa-token-matrix-canvas-pad-y-boost': isSmallScreen ? '-32px' : '-48px'
                }
            },
            tokens: visualTokens || resolveView2dVisualTokens()
        }
    });
}
