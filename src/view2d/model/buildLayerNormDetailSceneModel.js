import { MHA_FINAL_Q_COLOR } from '../../animations/LayerAnimationConstants.js';
import { getLayerNormParamData } from '../../data/layerNormParams.js';
import {
    D_MODEL,
    RESIDUAL_COLOR_CLAMP
} from '../../ui/selectionPanelConstants.js';
import { resolveMhsaTokenMatrixLayoutMetrics } from '../../ui/selectionPanelMhsaLayoutUtils.js';
import {
    buildHueRangeOptions,
    mapValueToColor,
    mapValueToHueRange
} from '../../utils/colors.js';
import {
    formatLayerNormLabel,
    formatLayerNormParamLabel
} from '../../utils/layerNormLabels.js';
import {
    createAnchorRef,
    createConnectorNode,
    createGroupNode,
    createMatrixNode,
    createOperatorNode,
    createSceneModel,
    createTextNode,
    buildSceneNodeId,
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
    createVectorStripMatrixNode,
    RESIDUAL_VECTOR_MEASURE_COLS
} from './createResidualVectorMatrixNode.js';
import { createView2dVectorStripMetadata } from '../shared/vectorStrip.js';
import {
    resolveMhsaDimensionVisualExtent,
    resolveMhsaTokenVisualExtent
} from '../shared/mhsaDimensionSizing.js';

const BASE_VECTOR_ROW_HEIGHT = 8;
const BASE_VECTOR_ROW_HEIGHT_SMALL = 7;
const BASE_VECTOR_ROW_GAP = 0;
const BASE_VECTOR_PADDING_Y = 0;
const BASE_VECTOR_PADDING_X = 0;
const INPUT_CAPTION_LABEL_SCALE = 1;
const EDGE_CONNECTOR_STROKE_WIDTH_SCALE = 0.88;
const INPUT_CONNECTOR_TARGET_GAP = 8;
const NORMALIZATION_CONNECTOR_SOURCE_GAP = 8;
const NORMALIZATION_CONNECTOR_TARGET_GAP = 10;
const COPY_CONNECTOR_SOURCE_GAP = 8;
const COPY_CONNECTOR_TARGET_GAP = 10;
const OUTPUT_CONNECTOR_SOURCE_GAP = 8;
const INCOMING_ARROW_SPACER_WIDTH = 60;
const INCOMING_ARROW_SPACER_WIDTH_SMALL = 52;
const NORMALIZATION_ARROW_SPACER_WIDTH = 108;
const NORMALIZATION_ARROW_SPACER_WIDTH_SMALL = 90;
const COPY_ARROW_SPACER_WIDTH = 44;
const COPY_ARROW_SPACER_WIDTH_SMALL = 36;
const OUTGOING_ARROW_SPACER_WIDTH = 56;
const OUTGOING_ARROW_SPACER_WIDTH_SMALL = 48;
const EQUATION_STAGE_GAP = 12;
const EQUATION_STAGE_GAP_SMALL = 10;
const PIPELINE_STAGE_GAP = 16;
const PIPELINE_STAGE_GAP_SMALL = 12;
const NORMALIZATION_BRIDGE_GAP = 14;
const NORMALIZATION_BRIDGE_GAP_SMALL = 12;
const NORMALIZATION_EQUATION_FONT_SCALE = 1.22;
const CONNECTOR_LAYER_GAP = 0;
const HADAMARD_OPERATOR_TEXT = '⊙';
const LAYER_NORM_EQUATION_TEX = '\\frac{x - \\mu}{\\sqrt{\\sigma^2 + \\epsilon}}';

const LAYER_NORM_PARAM_RANGE_OPTIONS = buildHueRangeOptions(MHA_FINAL_Q_COLOR, {
    hueSpread: 0.1,
    minLightness: 0.34,
    maxLightness: 0.74,
    valueMin: -1.8,
    valueMax: 1.8
});

function normalizeIndex(value) {
    return Number.isFinite(value) ? Math.max(0, Math.floor(value)) : null;
}

function normalizeLayerNormKind(kind = null) {
    const lower = String(kind || '').trim().toLowerCase();
    if (lower === 'ln1' || lower === 'ln2' || lower === 'final') return lower;
    return null;
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

function sampleVector(values = [], targetLength = RESIDUAL_VECTOR_MEASURE_COLS) {
    const safeValues = cleanNumberArray(values);
    const length = Math.max(1, Math.floor(targetLength || RESIDUAL_VECTOR_MEASURE_COLS));
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

function buildLayerNormParamGradientCss(values = [], direction = '90deg') {
    const safeValues = cleanNumberArray(values);
    if (!safeValues.length) return 'none';
    const stops = safeValues.map((value, index) => {
        const ratio = safeValues.length > 1 ? index / (safeValues.length - 1) : 0;
        return `${colorToCss(mapValueToHueRange(value, LAYER_NORM_PARAM_RANGE_OPTIONS))} ${(ratio * 100).toFixed(4)}%`;
    });
    if (stops.length === 1) {
        return `linear-gradient(${direction}, ${stops[0].replace(' 0.0000%', ' 0%')}, ${stops[0].replace(' 0.0000%', ' 100%')})`;
    }
    return `linear-gradient(${direction}, ${stops.join(', ')})`;
}

function multiplyVectors(left = [], right = []) {
    const lhs = cleanNumberArray(left);
    const rhs = cleanNumberArray(right);
    const length = Math.max(lhs.length, rhs.length);
    return Array.from({ length }, (_, index) => (
        (Number.isFinite(lhs[index]) ? lhs[index] : 0) * (Number.isFinite(rhs[index]) ? rhs[index] : 0)
    ));
}

function addVectors(left = [], right = []) {
    const lhs = cleanNumberArray(left);
    const rhs = cleanNumberArray(right);
    const length = Math.max(lhs.length, rhs.length);
    return Array.from({ length }, (_, index) => (
        (Number.isFinite(lhs[index]) ? lhs[index] : 0) + (Number.isFinite(rhs[index]) ? rhs[index] : 0)
    ));
}

function resolveMeasureCols(columnCount = D_MODEL) {
    const dimensionRatio = Math.max(1, Number(columnCount) / D_MODEL);
    return Math.max(
        RESIDUAL_VECTOR_MEASURE_COLS,
        Math.round(RESIDUAL_VECTOR_MEASURE_COLS * Math.pow(dimensionRatio, 0.44))
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

function resolveLayerNormStageConfig(layerNormKind = 'ln1', layerIndex = null, layerCount = null) {
    const safeKind = normalizeLayerNormKind(layerNormKind);
    if (!safeKind) return null;
    const fallbackLastLayer = Number.isFinite(layerCount) && layerCount > 0
        ? Math.max(0, Math.floor(layerCount) - 1)
        : 0;
    const resolvedLayerIndex = safeKind === 'final'
        ? (
            Number.isFinite(layerIndex)
                ? Math.max(0, Math.floor(layerIndex))
                : fallbackLastLayer
        )
        : normalizeIndex(layerIndex);
    if ((safeKind === 'ln1' || safeKind === 'ln2') && !Number.isFinite(resolvedLayerIndex)) {
        return null;
    }

    if (safeKind === 'ln1') {
        return {
            layerNormKind: safeKind,
            layerIndex: resolvedLayerIndex,
            title: formatLayerNormLabel('ln1'),
            inputStage: 'layer.incoming',
            normStage: 'ln1.norm',
            scaleStage: 'ln1.scale',
            shiftStage: 'ln1.shift',
            paramScaleStage: 'ln1.param.scale',
            paramShiftStage: 'ln1.param.shift',
            outputLabelTex: 'x_{\\ln}',
            outputLabelText: 'x_ln'
        };
    }
    if (safeKind === 'ln2') {
        return {
            layerNormKind: safeKind,
            layerIndex: resolvedLayerIndex,
            title: formatLayerNormLabel('ln2'),
            inputStage: 'residual.post_attention',
            normStage: 'ln2.norm',
            scaleStage: 'ln2.scale',
            shiftStage: 'ln2.shift',
            paramScaleStage: 'ln2.param.scale',
            paramShiftStage: 'ln2.param.shift',
            outputLabelTex: 'x_{\\ln}',
            outputLabelText: 'x_ln'
        };
    }
    return {
        layerNormKind: 'final',
        layerIndex: resolvedLayerIndex,
        title: 'Final LayerNorm',
        inputStage: 'residual.post_mlp',
        normStage: 'final_ln.norm',
        scaleStage: 'final_ln.scale',
        shiftStage: 'final_ln.shift',
        paramScaleStage: 'final_ln.param.scale',
        paramShiftStage: 'final_ln.param.shift',
        outputLabelTex: 'x_{\\mathrm{final}}',
        outputLabelText: 'x_final'
    };
}

function resolveLayerNormInputVector(activationSource = null, stageConfig = null, tokenIndex = null) {
    if (!stageConfig) return null;
    if (stageConfig.layerNormKind === 'ln1') {
        return typeof activationSource?.getLayerIncoming === 'function'
            ? activationSource.getLayerIncoming(stageConfig.layerIndex, tokenIndex, D_MODEL)
            : null;
    }
    if (stageConfig.layerNormKind === 'ln2') {
        return typeof activationSource?.getPostAttentionResidual === 'function'
            ? activationSource.getPostAttentionResidual(stageConfig.layerIndex, tokenIndex, D_MODEL)
            : null;
    }
    return typeof activationSource?.getPostMlpResidual === 'function'
        ? activationSource.getPostMlpResidual(stageConfig.layerIndex, tokenIndex, D_MODEL)
        : null;
}

function resolveLayerNormStageVector(activationSource = null, stageConfig = null, stageKey = 'norm', tokenIndex = null) {
    if (!stageConfig) return null;
    if (stageConfig.layerNormKind === 'ln1') {
        return typeof activationSource?.getLayerLn1 === 'function'
            ? activationSource.getLayerLn1(stageConfig.layerIndex, stageKey, tokenIndex, D_MODEL)
            : null;
    }
    if (stageConfig.layerNormKind === 'ln2') {
        return typeof activationSource?.getLayerLn2 === 'function'
            ? activationSource.getLayerLn2(stageConfig.layerIndex, stageKey, tokenIndex, D_MODEL)
            : null;
    }
    return typeof activationSource?.getFinalLayerNorm === 'function'
        ? activationSource.getFinalLayerNorm(stageKey, tokenIndex, D_MODEL)
        : null;
}

function buildLayerNormActivationRows(tokenRefs = [], {
    stageConfig = null,
    measureCols = RESIDUAL_VECTOR_MEASURE_COLS,
    componentKind = 'layer-norm',
    stage = '',
    role = '',
    labelBuilder = null,
    gradientBuilder = buildResidualGradientCss,
    getVector = () => null
} = {}) {
    return tokenRefs.map((tokenRef) => {
        const rawValues = cleanNumberArray(getVector(tokenRef) || []);
        const sampledValues = sampleVector(rawValues, measureCols);
        const semantic = {
            componentKind,
            ...(Number.isFinite(stageConfig?.layerIndex) ? { layerIndex: stageConfig.layerIndex } : {}),
            ...(stageConfig?.layerNormKind ? { layerNormKind: stageConfig.layerNormKind } : {}),
            stage,
            role,
            rowIndex: tokenRef.rowIndex,
            tokenIndex: tokenRef.tokenIndex
        };
        const label = typeof labelBuilder === 'function'
            ? labelBuilder(tokenRef)
            : (
                typeof tokenRef?.tokenLabel === 'string' && tokenRef.tokenLabel.length
                    ? tokenRef.tokenLabel
                    : `Token ${tokenRef.rowIndex + 1}`
            );
        return {
            id: buildSceneNodeId(semantic),
            index: tokenRef.rowIndex,
            label,
            semantic,
            rawValues: sampledValues,
            gradientCss: gradientBuilder(sampledValues),
            title: label
        };
    });
}

function buildLayerNormParamRows(stageConfig = null, {
    param = 'scale',
    measureCols = RESIDUAL_VECTOR_MEASURE_COLS
} = {}) {
    if (!stageConfig) return [];
    const rawParamValues = cleanNumberArray(getLayerNormParamData(
        stageConfig.layerIndex,
        stageConfig.layerNormKind,
        param,
        D_MODEL
    ) || []);
    const sampledValues = sampleVector(rawParamValues, measureCols);
    const safeParam = String(param || '').trim().toLowerCase() === 'shift' ? 'shift' : 'scale';
    const semantic = {
        componentKind: 'layer-norm',
        ...(Number.isFinite(stageConfig.layerIndex) ? { layerIndex: stageConfig.layerIndex } : {}),
        ...(stageConfig.layerNormKind ? { layerNormKind: stageConfig.layerNormKind } : {}),
        stage: safeParam === 'scale' ? stageConfig.paramScaleStage : stageConfig.paramShiftStage,
        role: `layer-norm-${safeParam}-row`,
        rowIndex: 0
    };
    return [{
        id: buildSceneNodeId(semantic),
        index: 0,
        label: '',
        semantic,
        rawValues: sampledValues,
        gradientCss: buildLayerNormParamGradientCss(sampledValues),
        title: formatLayerNormParamLabel(stageConfig.layerNormKind, safeParam)
    }];
}

function createLayerNormTextNode({
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

function createLayerNormOperatorNode({
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

function createLayerNormVectorNode({
    role = '',
    semantic = {},
    labelTex = '',
    labelText = '',
    rowItems = [],
    rowCount = 1,
    columnCount = D_MODEL,
    measureCols = RESIDUAL_VECTOR_MEASURE_COLS,
    compactWidth = 96,
    rowHeight = BASE_VECTOR_ROW_HEIGHT,
    visualStyleKey = VIEW2D_STYLE_KEYS.RESIDUAL,
    stripStyleKey = VIEW2D_STYLE_KEYS.RESIDUAL,
    disableEdgeOrnament = false
} = {}) {
    return createVectorStripMatrixNode({
        role,
        semantic,
        labelTex,
        labelText,
        rowItems,
        rowCount,
        columnCount,
        measureCols,
        compactWidth,
        rowHeight,
        captionPosition: 'bottom',
        captionMinScreenHeightPx: 1,
        captionLabelScale: INPUT_CAPTION_LABEL_SCALE,
        visualStyleKey,
        stripMetadata: createView2dVectorStripMetadata({
            compactWidth,
            rowHeight,
            rowGap: BASE_VECTOR_ROW_GAP,
            paddingX: BASE_VECTOR_PADDING_X,
            paddingY: BASE_VECTOR_PADDING_Y,
            cornerRadius: 10,
            bandCount: Math.max(12, measureCols),
            hoverScaleY: 1.12,
            hoverGlowBlur: 10,
            hideSurface: true
        }),
        metadata: {
            kind: stripStyleKey,
            ...(disableEdgeOrnament ? { disableEdgeOrnament: true } : {})
        }
    });
}

export function buildLayerNormDetailSceneModel({
    activationSource = null,
    layerNormDetailTarget = null,
    tokenRefs = [],
    layerCount = null,
    visualTokens = null,
    isSmallScreen = false
} = {}) {
    const stageConfig = resolveLayerNormStageConfig(
        layerNormDetailTarget?.layerNormKind,
        layerNormDetailTarget?.layerIndex,
        layerCount
    );
    if (!stageConfig || !tokenRefs.length) {
        return null;
    }

    const rowCount = Math.max(1, tokenRefs.length);
    const resolvedLayoutMetrics = resolveMhsaTokenMatrixLayoutMetrics({
        rowCount,
        isSmallScreen
    });
    const baseSemantic = {
        componentKind: 'layer-norm',
        ...(Number.isFinite(stageConfig.layerIndex) ? { layerIndex: stageConfig.layerIndex } : {}),
        layerNormKind: stageConfig.layerNormKind,
        stage: stageConfig.layerNormKind === 'final' ? 'final-ln' : stageConfig.layerNormKind
    };
    const vectorMetrics = resolveVectorMetrics(rowCount, {
        isSmallScreen,
        columnCount: D_MODEL
    });

    const inputRows = buildLayerNormActivationRows(tokenRefs, {
        stageConfig,
        componentKind: 'residual',
        stage: stageConfig.inputStage,
        role: 'layer-norm-input-row',
        measureCols: vectorMetrics.measureCols,
        getVector: (tokenRef) => resolveLayerNormInputVector(
            activationSource,
            stageConfig,
            tokenRef.tokenIndex
        )
    });
    if (!inputRows.length) {
        return null;
    }

    const normalizedRows = buildLayerNormActivationRows(tokenRefs, {
        stageConfig,
        stage: stageConfig.normStage,
        role: 'layer-norm-normalized-row',
        measureCols: vectorMetrics.measureCols,
        getVector: (tokenRef) => (
            resolveLayerNormStageVector(activationSource, stageConfig, 'norm', tokenRef.tokenIndex)
            || resolveLayerNormInputVector(activationSource, stageConfig, tokenRef.tokenIndex)
        )
    });
    const normalizedCopyRows = buildLayerNormActivationRows(tokenRefs, {
        stageConfig,
        stage: stageConfig.normStage,
        role: 'layer-norm-normalized-copy-row',
        measureCols: vectorMetrics.measureCols,
        getVector: (tokenRef) => (
            resolveLayerNormStageVector(activationSource, stageConfig, 'norm', tokenRef.tokenIndex)
            || resolveLayerNormInputVector(activationSource, stageConfig, tokenRef.tokenIndex)
        )
    });
    const scaleRows = buildLayerNormParamRows(stageConfig, {
        param: 'scale',
        measureCols: vectorMetrics.measureCols
    });
    const scaledRows = buildLayerNormActivationRows(tokenRefs, {
        stageConfig,
        stage: stageConfig.scaleStage,
        role: 'layer-norm-scaled-row',
        measureCols: vectorMetrics.measureCols,
        getVector: (tokenRef) => {
            const normalizedVector = resolveLayerNormStageVector(activationSource, stageConfig, 'norm', tokenRef.tokenIndex)
                || resolveLayerNormInputVector(activationSource, stageConfig, tokenRef.tokenIndex)
                || [];
            const scaleVector = getLayerNormParamData(stageConfig.layerIndex, stageConfig.layerNormKind, 'scale', D_MODEL)
                || [];
            return resolveLayerNormStageVector(activationSource, stageConfig, 'scale', tokenRef.tokenIndex)
                || multiplyVectors(normalizedVector, scaleVector);
        }
    });
    const scaledCopyRows = buildLayerNormActivationRows(tokenRefs, {
        stageConfig,
        stage: stageConfig.scaleStage,
        role: 'layer-norm-scaled-copy-row',
        measureCols: vectorMetrics.measureCols,
        getVector: (tokenRef) => {
            const normalizedVector = resolveLayerNormStageVector(activationSource, stageConfig, 'norm', tokenRef.tokenIndex)
                || resolveLayerNormInputVector(activationSource, stageConfig, tokenRef.tokenIndex)
                || [];
            const scaleVector = getLayerNormParamData(stageConfig.layerIndex, stageConfig.layerNormKind, 'scale', D_MODEL)
                || [];
            return resolveLayerNormStageVector(activationSource, stageConfig, 'scale', tokenRef.tokenIndex)
                || multiplyVectors(normalizedVector, scaleVector);
        }
    });
    const shiftRows = buildLayerNormParamRows(stageConfig, {
        param: 'shift',
        measureCols: vectorMetrics.measureCols
    });
    const outputRows = buildLayerNormActivationRows(tokenRefs, {
        stageConfig,
        componentKind: stageConfig.layerNormKind === 'final' ? 'layer-norm' : 'residual',
        stage: stageConfig.shiftStage,
        role: 'layer-norm-output-row',
        measureCols: vectorMetrics.measureCols,
        getVector: (tokenRef) => {
            const normalizedVector = resolveLayerNormStageVector(activationSource, stageConfig, 'norm', tokenRef.tokenIndex)
                || resolveLayerNormInputVector(activationSource, stageConfig, tokenRef.tokenIndex)
                || [];
            const scaleVector = getLayerNormParamData(stageConfig.layerIndex, stageConfig.layerNormKind, 'scale', D_MODEL)
                || [];
            const shiftVector = getLayerNormParamData(stageConfig.layerIndex, stageConfig.layerNormKind, 'shift', D_MODEL)
                || [];
            const scaledVector = resolveLayerNormStageVector(activationSource, stageConfig, 'scale', tokenRef.tokenIndex)
                || multiplyVectors(normalizedVector, scaleVector);
            return resolveLayerNormStageVector(activationSource, stageConfig, 'shift', tokenRef.tokenIndex)
                || addVectors(scaledVector, shiftVector);
        }
    });

    const inputNode = createLayerNormVectorNode({
        role: 'layer-norm-input',
        semantic: buildSemantic(baseSemantic, {
            stage: stageConfig.inputStage,
            role: 'layer-norm-input'
        }),
        labelTex: 'x',
        labelText: 'x',
        rowItems: inputRows,
        rowCount,
        measureCols: vectorMetrics.measureCols,
        compactWidth: vectorMetrics.compactWidth,
        rowHeight: vectorMetrics.rowHeight,
        visualStyleKey: VIEW2D_STYLE_KEYS.RESIDUAL,
        stripStyleKey: 'layer-norm-input',
        disableEdgeOrnament: true
    });
    const normalizedNode = createLayerNormVectorNode({
        role: 'layer-norm-normalized',
        semantic: buildSemantic(baseSemantic, {
            stage: stageConfig.normStage,
            role: 'layer-norm-normalized'
        }),
        labelTex: '\\hat{x}',
        labelText: 'x_hat',
        rowItems: normalizedRows,
        rowCount,
        measureCols: vectorMetrics.measureCols,
        compactWidth: vectorMetrics.compactWidth,
        rowHeight: vectorMetrics.rowHeight,
        visualStyleKey: VIEW2D_STYLE_KEYS.RESIDUAL,
        stripStyleKey: 'layer-norm-normalized'
    });
    const normalizedCopyNode = createLayerNormVectorNode({
        role: 'layer-norm-normalized-copy',
        semantic: buildSemantic(baseSemantic, {
            stage: stageConfig.normStage,
            role: 'layer-norm-normalized-copy'
        }),
        labelTex: '\\hat{x}',
        labelText: 'x_hat',
        rowItems: normalizedCopyRows,
        rowCount,
        measureCols: vectorMetrics.measureCols,
        compactWidth: vectorMetrics.compactWidth,
        rowHeight: vectorMetrics.rowHeight,
        visualStyleKey: VIEW2D_STYLE_KEYS.RESIDUAL,
        stripStyleKey: 'layer-norm-normalized-copy'
    });
    const scaleNode = createLayerNormVectorNode({
        role: 'layer-norm-scale',
        semantic: buildSemantic(baseSemantic, {
            stage: stageConfig.paramScaleStage,
            role: 'layer-norm-scale'
        }),
        labelTex: '\\gamma',
        labelText: 'gamma',
        rowItems: scaleRows,
        rowCount: 1,
        measureCols: vectorMetrics.measureCols,
        compactWidth: vectorMetrics.compactWidth,
        rowHeight: isSmallScreen ? BASE_VECTOR_ROW_HEIGHT_SMALL + 5 : BASE_VECTOR_ROW_HEIGHT + 6,
        visualStyleKey: VIEW2D_STYLE_KEYS.LAYER_NORM,
        stripStyleKey: 'layer-norm-scale'
    });
    const scaledNode = createLayerNormVectorNode({
        role: 'layer-norm-scaled',
        semantic: buildSemantic(baseSemantic, {
            stage: stageConfig.scaleStage,
            role: 'layer-norm-scaled'
        }),
        labelTex: '\\gamma \\odot \\hat{x}',
        labelText: 'gamma ⊙ x_hat',
        rowItems: scaledRows,
        rowCount,
        measureCols: vectorMetrics.measureCols,
        compactWidth: vectorMetrics.compactWidth,
        rowHeight: vectorMetrics.rowHeight,
        visualStyleKey: VIEW2D_STYLE_KEYS.RESIDUAL,
        stripStyleKey: 'layer-norm-scaled'
    });
    const scaledCopyNode = createLayerNormVectorNode({
        role: 'layer-norm-scaled-copy',
        semantic: buildSemantic(baseSemantic, {
            stage: stageConfig.scaleStage,
            role: 'layer-norm-scaled-copy'
        }),
        labelTex: '\\gamma \\odot \\hat{x}',
        labelText: 'gamma ⊙ x_hat',
        rowItems: scaledCopyRows,
        rowCount,
        measureCols: vectorMetrics.measureCols,
        compactWidth: vectorMetrics.compactWidth,
        rowHeight: vectorMetrics.rowHeight,
        visualStyleKey: VIEW2D_STYLE_KEYS.RESIDUAL,
        stripStyleKey: 'layer-norm-scaled-copy'
    });
    const shiftNode = createLayerNormVectorNode({
        role: 'layer-norm-shift',
        semantic: buildSemantic(baseSemantic, {
            stage: stageConfig.paramShiftStage,
            role: 'layer-norm-shift'
        }),
        labelTex: '\\beta',
        labelText: 'beta',
        rowItems: shiftRows,
        rowCount: 1,
        measureCols: vectorMetrics.measureCols,
        compactWidth: vectorMetrics.compactWidth,
        rowHeight: isSmallScreen ? BASE_VECTOR_ROW_HEIGHT_SMALL + 5 : BASE_VECTOR_ROW_HEIGHT + 6,
        visualStyleKey: VIEW2D_STYLE_KEYS.LAYER_NORM,
        stripStyleKey: 'layer-norm-shift'
    });
    const outputNode = createLayerNormVectorNode({
        role: 'layer-norm-output',
        semantic: buildSemantic(baseSemantic, {
            stage: stageConfig.shiftStage,
            role: 'layer-norm-output'
        }),
        labelTex: stageConfig.outputLabelTex,
        labelText: stageConfig.outputLabelText,
        rowItems: outputRows,
        rowCount,
        measureCols: vectorMetrics.measureCols,
        compactWidth: vectorMetrics.compactWidth,
        rowHeight: vectorMetrics.rowHeight,
        visualStyleKey: VIEW2D_STYLE_KEYS.RESIDUAL,
        stripStyleKey: 'layer-norm-output',
        disableEdgeOrnament: true
    });

    const incomingArrowSpacerNode = createHiddenSpacer({
        semantic: buildSemantic(baseSemantic, {
            stage: stageConfig.inputStage,
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
    const normalizationArrowSpacerNode = createHiddenSpacer({
        semantic: buildSemantic(baseSemantic, {
            stage: stageConfig.normStage,
            role: 'layer-norm-normalization-arrow-spacer'
        }),
        role: 'layer-norm-normalization-arrow-spacer',
        width: isSmallScreen ? NORMALIZATION_ARROW_SPACER_WIDTH_SMALL : NORMALIZATION_ARROW_SPACER_WIDTH,
        height: 1
    });
    const normalizedCopySpacerNode = createHiddenSpacer({
        semantic: buildSemantic(baseSemantic, {
            stage: stageConfig.normStage,
            role: 'layer-norm-copy-arrow-spacer'
        }),
        role: 'layer-norm-copy-arrow-spacer',
        width: isSmallScreen ? COPY_ARROW_SPACER_WIDTH_SMALL : COPY_ARROW_SPACER_WIDTH,
        height: 1
    });
    const scaledCopySpacerNode = createHiddenSpacer({
        semantic: buildSemantic(baseSemantic, {
            stage: stageConfig.scaleStage,
            role: 'layer-norm-scaled-copy-arrow-spacer'
        }),
        role: 'layer-norm-scaled-copy-arrow-spacer',
        width: isSmallScreen ? COPY_ARROW_SPACER_WIDTH_SMALL : COPY_ARROW_SPACER_WIDTH,
        height: 1
    });
    const outgoingArrowSpacerNode = createHiddenSpacer({
        semantic: buildSemantic(baseSemantic, {
            stage: stageConfig.shiftStage,
            role: 'outgoing-arrow-spacer'
        }),
        role: 'outgoing-arrow-spacer',
        width: isSmallScreen ? OUTGOING_ARROW_SPACER_WIDTH_SMALL : OUTGOING_ARROW_SPACER_WIDTH,
        height: 1
    });

    const normalizationEquationNode = createLayerNormTextNode({
        role: 'layer-norm-normalization-equation',
        semantic: buildSemantic(baseSemantic, {
            stage: stageConfig.normStage,
            role: 'layer-norm-normalization-equation'
        }),
        tex: LAYER_NORM_EQUATION_TEX,
        text: '(x - mu) / sqrt(sigma^2 + epsilon)',
        visual: {
            styleKey: VIEW2D_STYLE_KEYS.LABEL
        },
        metadata: {
            minScreenHeightPx: 0,
            fontScale: NORMALIZATION_EQUATION_FONT_SCALE
        }
    });
    const normalizationBridgeNode = createGroupNode({
        role: 'layer-norm-normalization-bridge',
        semantic: buildSemantic(baseSemantic, {
            stage: stageConfig.normStage,
            role: 'layer-norm-normalization-bridge'
        }),
        direction: VIEW2D_LAYOUT_DIRECTIONS.VERTICAL,
        align: 'center',
        gapKey: 'inline',
        children: [
            normalizationEquationNode,
            normalizationArrowSpacerNode
        ],
        metadata: {
            gapOverride: isSmallScreen ? NORMALIZATION_BRIDGE_GAP_SMALL : NORMALIZATION_BRIDGE_GAP
        }
    });

    const normalizedCopyBridgeNode = createGroupNode({
        role: 'layer-norm-normalized-copy-bridge',
        semantic: buildSemantic(baseSemantic, {
            stage: stageConfig.normStage,
            role: 'layer-norm-normalized-copy-bridge'
        }),
        direction: VIEW2D_LAYOUT_DIRECTIONS.VERTICAL,
        align: 'center',
        gapKey: 'inline',
        children: [normalizedCopySpacerNode]
    });
    const scaledCopyBridgeNode = createGroupNode({
        role: 'layer-norm-scaled-copy-bridge',
        semantic: buildSemantic(baseSemantic, {
            stage: stageConfig.scaleStage,
            role: 'layer-norm-scaled-copy-bridge'
        }),
        direction: VIEW2D_LAYOUT_DIRECTIONS.VERTICAL,
        align: 'center',
        gapKey: 'inline',
        children: [scaledCopySpacerNode]
    });

    const scalingStageNode = createGroupNode({
        role: 'layer-norm-scaling-stage',
        semantic: buildSemantic(baseSemantic, {
            stage: stageConfig.scaleStage,
            role: 'layer-norm-scaling-stage'
        }),
        direction: VIEW2D_LAYOUT_DIRECTIONS.HORIZONTAL,
        align: 'center',
        gapKey: 'projection',
        children: [
            normalizedCopyBridgeNode,
            normalizedCopyNode,
            createLayerNormOperatorNode({
                role: 'layer-norm-hadamard',
                semantic: buildSemantic(baseSemantic, {
                    stage: stageConfig.scaleStage,
                    role: 'layer-norm-hadamard',
                    operatorKey: 'hadamard'
                }),
                text: HADAMARD_OPERATOR_TEXT,
                visual: {
                    styleKey: VIEW2D_STYLE_KEYS.OPERATOR
                }
            }),
            scaleNode,
            createLayerNormOperatorNode({
                role: 'layer-norm-scale-equals',
                semantic: buildSemantic(baseSemantic, {
                    stage: stageConfig.scaleStage,
                    role: 'layer-norm-scale-equals',
                    operatorKey: 'equals'
                }),
                text: '=',
                visual: {
                    styleKey: VIEW2D_STYLE_KEYS.OPERATOR
                }
            }),
            scaledNode
        ],
        metadata: {
            gapOverride: isSmallScreen ? EQUATION_STAGE_GAP_SMALL : EQUATION_STAGE_GAP
        }
    });
    const shiftStageNode = createGroupNode({
        role: 'layer-norm-shift-stage',
        semantic: buildSemantic(baseSemantic, {
            stage: stageConfig.shiftStage,
            role: 'layer-norm-shift-stage'
        }),
        direction: VIEW2D_LAYOUT_DIRECTIONS.HORIZONTAL,
        align: 'center',
        gapKey: 'projection',
        children: [
            scaledCopyBridgeNode,
            scaledCopyNode,
            createLayerNormOperatorNode({
                role: 'layer-norm-shift-plus',
                semantic: buildSemantic(baseSemantic, {
                    stage: stageConfig.shiftStage,
                    role: 'layer-norm-shift-plus',
                    operatorKey: 'plus'
                }),
                text: '+',
                visual: {
                    styleKey: VIEW2D_STYLE_KEYS.OPERATOR
                }
            }),
            shiftNode,
            createLayerNormOperatorNode({
                role: 'layer-norm-shift-equals',
                semantic: buildSemantic(baseSemantic, {
                    stage: stageConfig.shiftStage,
                    role: 'layer-norm-shift-equals',
                    operatorKey: 'equals'
                }),
                text: '=',
                visual: {
                    styleKey: VIEW2D_STYLE_KEYS.OPERATOR
                }
            }),
            outputNode,
            outgoingArrowSpacerNode
        ],
        metadata: {
            gapOverride: isSmallScreen ? EQUATION_STAGE_GAP_SMALL : EQUATION_STAGE_GAP
        }
    });

    const incomingConnectorNode = createConnectorNode({
        role: 'connector-layer-norm-input',
        semantic: buildSemantic(baseSemantic, {
            stage: 'connector-layer-norm-input',
            role: 'connector-layer-norm-input'
        }),
        source: createAnchorRef(incomingArrowSpacerNode.id, VIEW2D_ANCHOR_SIDES.LEFT),
        target: createAnchorRef(inputNode.id, VIEW2D_ANCHOR_SIDES.LEFT),
        route: VIEW2D_CONNECTOR_ROUTES.HORIZONTAL,
        gap: CONNECTOR_LAYER_GAP,
        targetGap: INPUT_CONNECTOR_TARGET_GAP,
        visual: {
            styleKey: VIEW2D_STYLE_KEYS.CONNECTOR_NEUTRAL
        },
        metadata: {
            preserveColor: true,
            strokeWidthScale: EDGE_CONNECTOR_STROKE_WIDTH_SCALE
        }
    });
    const normalizationConnectorNode = createConnectorNode({
        role: 'connector-layer-norm-normalization',
        semantic: buildSemantic(baseSemantic, {
            stage: 'connector-layer-norm-normalization',
            role: 'connector-layer-norm-normalization'
        }),
        source: createAnchorRef(inputNode.id, VIEW2D_ANCHOR_SIDES.RIGHT),
        target: createAnchorRef(normalizedNode.id, VIEW2D_ANCHOR_SIDES.LEFT),
        route: VIEW2D_CONNECTOR_ROUTES.HORIZONTAL,
        gap: CONNECTOR_LAYER_GAP,
        sourceGap: NORMALIZATION_CONNECTOR_SOURCE_GAP,
        targetGap: NORMALIZATION_CONNECTOR_TARGET_GAP,
        visual: {
            styleKey: VIEW2D_STYLE_KEYS.CONNECTOR_NEUTRAL
        },
        metadata: {
            preserveColor: true,
            strokeWidthScale: EDGE_CONNECTOR_STROKE_WIDTH_SCALE
        }
    });
    const normalizedCopyConnectorNode = createConnectorNode({
        role: 'connector-layer-norm-copy-normalized',
        semantic: buildSemantic(baseSemantic, {
            stage: 'connector-layer-norm-copy-normalized',
            role: 'connector-layer-norm-copy-normalized'
        }),
        source: createAnchorRef(normalizedNode.id, VIEW2D_ANCHOR_SIDES.RIGHT),
        target: createAnchorRef(normalizedCopyNode.id, VIEW2D_ANCHOR_SIDES.LEFT),
        route: VIEW2D_CONNECTOR_ROUTES.HORIZONTAL,
        gap: CONNECTOR_LAYER_GAP,
        sourceGap: COPY_CONNECTOR_SOURCE_GAP,
        targetGap: COPY_CONNECTOR_TARGET_GAP,
        visual: {
            styleKey: VIEW2D_STYLE_KEYS.CONNECTOR_NEUTRAL
        },
        metadata: {
            preserveColor: true,
            strokeWidthScale: EDGE_CONNECTOR_STROKE_WIDTH_SCALE
        }
    });
    const scaledCopyConnectorNode = createConnectorNode({
        role: 'connector-layer-norm-copy-scaled',
        semantic: buildSemantic(baseSemantic, {
            stage: 'connector-layer-norm-copy-scaled',
            role: 'connector-layer-norm-copy-scaled'
        }),
        source: createAnchorRef(scaledNode.id, VIEW2D_ANCHOR_SIDES.RIGHT),
        target: createAnchorRef(scaledCopyNode.id, VIEW2D_ANCHOR_SIDES.LEFT),
        route: VIEW2D_CONNECTOR_ROUTES.HORIZONTAL,
        gap: CONNECTOR_LAYER_GAP,
        sourceGap: COPY_CONNECTOR_SOURCE_GAP,
        targetGap: COPY_CONNECTOR_TARGET_GAP,
        visual: {
            styleKey: VIEW2D_STYLE_KEYS.CONNECTOR_NEUTRAL
        },
        metadata: {
            preserveColor: true,
            strokeWidthScale: EDGE_CONNECTOR_STROKE_WIDTH_SCALE
        }
    });
    const outputConnectorNode = createConnectorNode({
        role: 'connector-layer-norm-output',
        semantic: buildSemantic(baseSemantic, {
            stage: 'connector-layer-norm-output',
            role: 'connector-layer-norm-output'
        }),
        source: createAnchorRef(outputNode.id, VIEW2D_ANCHOR_SIDES.RIGHT),
        target: createAnchorRef(outgoingArrowSpacerNode.id, VIEW2D_ANCHOR_SIDES.RIGHT),
        route: VIEW2D_CONNECTOR_ROUTES.HORIZONTAL,
        gap: CONNECTOR_LAYER_GAP,
        sourceGap: OUTPUT_CONNECTOR_SOURCE_GAP,
        visual: {
            styleKey: VIEW2D_STYLE_KEYS.CONNECTOR_NEUTRAL
        },
        metadata: {
            preserveColor: true,
            strokeWidthScale: EDGE_CONNECTOR_STROKE_WIDTH_SCALE
        }
    });

    return createSceneModel({
        semantic: buildSemantic(baseSemantic, {
            role: 'scene'
        }),
        nodes: [
            createGroupNode({
                role: 'layer-norm-detail-stage',
                semantic: buildSemantic(baseSemantic, {
                    role: 'layer-norm-detail-stage'
                }),
                direction: VIEW2D_LAYOUT_DIRECTIONS.HORIZONTAL,
                align: 'center',
                gapKey: 'default',
                children: [
                    incomingArrowSpacerNode,
                    inputNode,
                    normalizationBridgeNode,
                    normalizedNode,
                    scalingStageNode,
                    shiftStageNode
                ],
                metadata: {
                    gapOverride: isSmallScreen ? PIPELINE_STAGE_GAP_SMALL : PIPELINE_STAGE_GAP
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
                    normalizationConnectorNode,
                    normalizedCopyConnectorNode,
                    scaledCopyConnectorNode,
                    outputConnectorNode
                ]
            })
        ],
        metadata: {
            visualContract: 'selection-panel-layer-norm-v1',
            source: 'buildLayerNormDetailSceneModel',
            layerNormKind: stageConfig.layerNormKind,
            layerIndex: stageConfig.layerIndex,
            rowCount,
            isSmallScreen: !!isSmallScreen,
            layoutMetrics: {
                ...resolvedLayoutMetrics,
                cssVars: {
                    ...(resolvedLayoutMetrics?.cssVars || {}),
                    '--mhsa-token-matrix-canvas-pad-x-boost': isSmallScreen ? '-18px' : '-26px',
                    '--mhsa-token-matrix-canvas-pad-y-boost': isSmallScreen ? '-8px' : '-12px',
                    '--mhsa-token-matrix-stage-gap-boost': isSmallScreen ? '-4px' : '-6px'
                }
            },
            tokens: visualTokens || resolveView2dVisualTokens()
        }
    });
}
