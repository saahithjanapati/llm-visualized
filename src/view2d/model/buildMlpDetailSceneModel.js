import { buildHueRangeOptions, mapValueToColor, mapValueToHueRange } from '../../utils/colors.js';
import {
    D_MODEL,
    FINAL_MLP_COLOR,
    RESIDUAL_COLOR_CLAMP
} from '../../ui/selectionPanelConstants.js';
import {
    buildSceneNodeId,
    createAnchorRef,
    createConnectorNode,
    createGroupNode,
    createOperatorNode,
    createSceneModel,
    createTextNode,
    VIEW2D_ANCHOR_SIDES,
    VIEW2D_CONNECTOR_ROUTES,
    VIEW2D_LAYOUT_DIRECTIONS
} from '../schema/sceneTypes.js';
import {
    resolveView2dVisualTokens,
    VIEW2D_STYLE_KEYS
} from '../theme/visualTokens.js';
import { createCaptionedCardMatrixNode, resolveRelativeCardSize } from './createCaptionedCardMatrixNode.js';
import { createVectorStripMatrixNode } from './createResidualVectorMatrixNode.js';
import { createView2dVectorStripMetadata } from '../shared/vectorStrip.js';
import { resolveMhsaDimensionVisualExtent } from '../shared/mhsaDimensionSizing.js';

const MLP_HIDDEN_WIDTH = D_MODEL * 4;
const INPUT_MEASURE_COLS = 12;
const BASE_VECTOR_ROW_HEIGHT = 8;
const BASE_VECTOR_ROW_HEIGHT_SMALL = 7;
const BASE_VECTOR_ROW_GAP = 2;
const BASE_VECTOR_PADDING_Y = 1;
const BASE_VECTOR_PADDING_X = 1;
const GELU_LABEL_FONT_SCALE = 1.16;
const GELU_LABEL_FONT_SCALE_SMALL = 1.08;
const PAREN_MIN_FONT_SCALE = 1.85;
const PAREN_MIN_FONT_SCALE_SMALL = 1.68;
const PAREN_MAX_FONT_SCALE = 3.1;
const PAREN_HEIGHT_RATIO = 0.034;
const COPY_MATRIX_SCALE = 0.92;
const DOWNSTREAM_COPY_SCALE = 0.9;
const SCENE_GAP_Y = 68;
const SCENE_GAP_Y_SMALL = 52;
const FLOW_GAP = 32;
const FLOW_GAP_SMALL = 24;
const INLINE_GAP = 10;
const INLINE_GAP_SMALL = 8;
const STAGE_ALIGNMENT_OFFSET = 18;
const STAGE_ALIGNMENT_OFFSET_SMALL = 14;
const CONNECTOR_STROKE = 'rgba(255, 255, 255, 0.84)';

const MLP_RANGE_OPTIONS = buildHueRangeOptions(FINAL_MLP_COLOR, {
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

function buildValueColors(values = [], {
    clampMax = RESIDUAL_COLOR_CLAMP,
    rangeOptions = null
} = {}) {
    const safeValues = cleanNumberArray(values);
    if (!safeValues.length) return [];
    return safeValues.map((value) => colorToCss(
        rangeOptions
            ? mapValueToHueRange(value, rangeOptions)
            : mapValueToColor(value, { clampMax })
    ));
}

function buildGradientCss(values = [], options = {}) {
    const fillColors = buildValueColors(values, options);
    if (!fillColors.length) return 'none';
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
    isSmallScreen = false,
    scale = 1
} = {}) {
    const baseExtentPx = isSmallScreen ? 112 : 126;
    const minExtentPx = isSmallScreen ? 84 : 96;
    const maxExtentPx = isSmallScreen ? 196 : 232;
    return Math.max(
        68,
        Math.round(
            resolveMhsaDimensionVisualExtent(columnCount, {
                isSmallScreen,
                baseDimensionCount: D_MODEL,
                exponent: 0.34,
                baseExtentPx,
                minExtentPx,
                maxExtentPx
            }) * Math.max(0.5, Number(scale) || 1)
        )
    );
}

function resolveVectorMetrics(rowCount = 1, {
    isSmallScreen = false,
    columnCount = D_MODEL,
    scale = 1
} = {}) {
    const baseRowHeight = isSmallScreen ? BASE_VECTOR_ROW_HEIGHT_SMALL : BASE_VECTOR_ROW_HEIGHT;
    const rowHeight = Math.max(4, Math.round(baseRowHeight * Math.max(0.75, Number(scale) || 1)));
    const rowGap = Math.max(0, Math.round(BASE_VECTOR_ROW_GAP * Math.max(0.75, Number(scale) || 1)));
    const paddingX = BASE_VECTOR_PADDING_X;
    const paddingY = BASE_VECTOR_PADDING_Y;
    const compactWidth = resolveCompactWidth(columnCount, {
        isSmallScreen,
        scale
    });
    const measureCols = Math.max(
        8,
        Math.round(resolveMeasureCols(columnCount) * Math.max(0.78, Number(scale) || 1))
    );
    const estimatedHeight = (Math.max(1, rowCount) * rowHeight)
        + (Math.max(0, rowCount - 1) * rowGap)
        + (paddingY * 2);
    return {
        rowHeight,
        rowGap,
        paddingX,
        paddingY,
        compactWidth,
        measureCols,
        estimatedHeight
    };
}

function buildActivationRowItems(tokenRefs = [], {
    baseSemantic = {},
    role = 'row',
    columnCount = D_MODEL,
    measureCols = INPUT_MEASURE_COLS,
    clampMax = RESIDUAL_COLOR_CLAMP,
    rangeOptions = null,
    getVector = () => null
} = {}) {
    return tokenRefs.map((tokenRef) => {
        const values = sampleVector(getVector(tokenRef) || [], measureCols);
        const semantic = buildSemantic(baseSemantic, {
            role,
            rowIndex: tokenRef.rowIndex,
            tokenIndex: tokenRef.tokenIndex
        });
        return {
            id: buildSceneNodeId(semantic),
            index: tokenRef.rowIndex,
            label: typeof tokenRef?.tokenLabel === 'string' && tokenRef.tokenLabel.length
                ? tokenRef.tokenLabel
                : `Token ${tokenRef.rowIndex + 1}`,
            semantic,
            rawValues: values,
            gradientCss: buildGradientCss(values, {
                clampMax,
                rangeOptions
            }),
            title: typeof tokenRef?.tokenLabel === 'string' && tokenRef.tokenLabel.length
                ? tokenRef.tokenLabel
                : `Token ${tokenRef.rowIndex + 1}`
        };
    });
}

function createActivationMatrixNode({
    role = 'mlp-activation',
    semantic = {},
    labelTex = '',
    labelText = '',
    rowItems = [],
    rowCount = 1,
    columnCount = D_MODEL,
    visualStyleKey = VIEW2D_STYLE_KEYS.MLP,
    isSmallScreen = false,
    scale = 1,
    metadata = null
} = {}) {
    const vectorMetrics = resolveVectorMetrics(rowCount, {
        isSmallScreen,
        columnCount,
        scale
    });
    return createVectorStripMatrixNode({
        role,
        semantic,
        labelTex,
        labelText,
        rowItems,
        rowCount,
        columnCount,
        measureCols: vectorMetrics.measureCols,
        compactWidth: vectorMetrics.compactWidth,
        rowHeight: vectorMetrics.rowHeight,
        captionPosition: 'float-top',
        captionMinScreenHeightPx: 24,
        visualStyleKey,
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
            kind: role,
            ...metadata
        }
    });
}

function createFlowConnector({
    role = 'connector',
    semantic = {},
    fromNode = null,
    toNode = null,
    sourceAnchor = VIEW2D_ANCHOR_SIDES.RIGHT,
    targetAnchor = VIEW2D_ANCHOR_SIDES.LEFT,
    route = VIEW2D_CONNECTOR_ROUTES.HORIZONTAL,
    gap = 0
} = {}) {
    if (!fromNode?.id || !toNode?.id) return null;
    return createConnectorNode({
        role,
        semantic,
        source: createAnchorRef(fromNode.id, sourceAnchor),
        target: createAnchorRef(toNode.id, targetAnchor),
        route,
        gap,
        visual: {
            styleKey: VIEW2D_STYLE_KEYS.CONNECTOR_NEUTRAL,
            stroke: CONNECTOR_STROKE
        },
        metadata: {
            preserveColor: true,
            strokeWidthScale: 0.78,
            arrowTipTouchTarget: true
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
    const baseSemantic = {
        componentKind: 'mlp',
        layerIndex,
        stage: 'mlp-detail'
    };

    const inputMetrics = resolveVectorMetrics(rowCount, {
        isSmallScreen,
        columnCount: D_MODEL
    });
    const wideMetrics = resolveVectorMetrics(rowCount, {
        isSmallScreen,
        columnCount: MLP_HIDDEN_WIDTH
    });
    const wideCopyMetrics = resolveVectorMetrics(rowCount, {
        isSmallScreen,
        columnCount: MLP_HIDDEN_WIDTH,
        scale: COPY_MATRIX_SCALE
    });
    const downstreamCopyMetrics = resolveVectorMetrics(rowCount, {
        isSmallScreen,
        columnCount: MLP_HIDDEN_WIDTH,
        scale: DOWNSTREAM_COPY_SCALE
    });
    const parenFontScale = Math.max(
        isSmallScreen ? PAREN_MIN_FONT_SCALE_SMALL : PAREN_MIN_FONT_SCALE,
        Math.min(
            PAREN_MAX_FONT_SCALE,
            wideCopyMetrics.estimatedHeight * PAREN_HEIGHT_RATIO
        )
    );

    const residualRows = buildActivationRowItems(tokenRefs, {
        baseSemantic: {
            componentKind: 'residual',
            layerIndex,
            stage: 'ln2.shift'
        },
        role: 'mlp-input-row',
        columnCount: D_MODEL,
        measureCols: inputMetrics.measureCols,
        getVector: (tokenRef) => (
            typeof activationSource?.getLayerLn2 === 'function'
                ? activationSource.getLayerLn2(layerIndex, 'shift', tokenRef.tokenIndex, D_MODEL)
                : null
        )
    });
    const upRows = buildActivationRowItems(tokenRefs, {
        baseSemantic,
        role: 'mlp-up-output-row',
        columnCount: MLP_HIDDEN_WIDTH,
        measureCols: wideMetrics.measureCols,
        rangeOptions: MLP_RANGE_OPTIONS,
        getVector: (tokenRef) => (
            typeof activationSource?.getMlpUp === 'function'
                ? activationSource.getMlpUp(layerIndex, tokenRef.tokenIndex, MLP_HIDDEN_WIDTH)
                : null
        )
    });
    const activationRows = buildActivationRowItems(tokenRefs, {
        baseSemantic,
        role: 'mlp-activation-output-row',
        columnCount: MLP_HIDDEN_WIDTH,
        measureCols: wideMetrics.measureCols,
        rangeOptions: MLP_RANGE_OPTIONS,
        getVector: (tokenRef) => (
            typeof activationSource?.getMlpActivation === 'function'
                ? activationSource.getMlpActivation(layerIndex, tokenRef.tokenIndex, MLP_HIDDEN_WIDTH)
                : null
        )
    });
    const downRows = buildActivationRowItems(tokenRefs, {
        baseSemantic,
        role: 'mlp-down-output-row',
        columnCount: D_MODEL,
        measureCols: inputMetrics.measureCols,
        rangeOptions: MLP_RANGE_OPTIONS,
        getVector: (tokenRef) => (
            typeof activationSource?.getMlpDown === 'function'
                ? activationSource.getMlpDown(layerIndex, tokenRef.tokenIndex, D_MODEL)
                : null
        )
    });

    if (!residualRows.length || !upRows.length || !activationRows.length || !downRows.length) {
        return null;
    }

    const upWeightCardSize = resolveRelativeCardSize({
        rows: D_MODEL,
        cols: MLP_HIDDEN_WIDTH,
        referenceCount: D_MODEL,
        referenceExtent: isSmallScreen ? 88 : 96,
        exponent: 0.38,
        minWidth: isSmallScreen ? 80 : 90,
        maxWidth: isSmallScreen ? 210 : 248,
        minHeight: isSmallScreen ? 84 : 94,
        maxHeight: isSmallScreen ? 162 : 188
    });
    const downWeightCardSize = resolveRelativeCardSize({
        rows: MLP_HIDDEN_WIDTH,
        cols: D_MODEL,
        referenceCount: D_MODEL,
        referenceExtent: isSmallScreen ? 88 : 96,
        exponent: 0.38,
        minWidth: isSmallScreen ? 80 : 90,
        maxWidth: isSmallScreen ? 158 : 182,
        minHeight: isSmallScreen ? 112 : 126,
        maxHeight: isSmallScreen ? 236 : 276
    });

    const inputNode = createActivationMatrixNode({
        role: 'mlp-input',
        semantic: buildSemantic(baseSemantic, {
            stage: 'mlp-input',
            role: 'mlp-input'
        }),
        labelTex: 'X_{\\mathrm{LN2}}',
        labelText: 'X_LN2',
        rowItems: residualRows,
        rowCount,
        columnCount: D_MODEL,
        visualStyleKey: VIEW2D_STYLE_KEYS.RESIDUAL,
        isSmallScreen,
        metadata: {
            kind: 'mlp-input'
        }
    });
    const upMultiplyNode = createOperatorNode({
        role: 'mlp-up-multiply',
        semantic: buildSemantic(baseSemantic, {
            stage: 'mlp-up',
            role: 'mlp-up-multiply',
            operatorKey: 'times'
        }),
        text: '×',
        visual: {
            styleKey: VIEW2D_STYLE_KEYS.OPERATOR
        },
        metadata: {
            fontScale: isSmallScreen ? 0.96 : 1.04,
            kind: 'mlp-up-multiply'
        }
    });
    const upWeightNode = createCaptionedCardMatrixNode({
        role: 'mlp-up-weight',
        semantic: buildSemantic(baseSemantic, {
            stage: 'mlp-up',
            role: 'mlp-up-weight'
        }),
        labelTex: 'W_{\\mathrm{up}}',
        labelText: 'W_up',
        rowCount: D_MODEL,
        columnCount: MLP_HIDDEN_WIDTH,
        cardWidth: upWeightCardSize.width,
        cardHeight: upWeightCardSize.height,
        cardCornerRadius: 16,
        captionPosition: 'bottom',
        captionMinScreenHeightPx: 22,
        captionLabelScale: 0.96,
        visualStyleKey: VIEW2D_STYLE_KEYS.MLP,
        metadata: {
            kind: 'mlp-up-weight'
        }
    });
    const upOutputNode = createActivationMatrixNode({
        role: 'mlp-up-output',
        semantic: buildSemantic(baseSemantic, {
            stage: 'mlp-up',
            role: 'mlp-up-output'
        }),
        labelTex: 'H_{\\mathrm{up}}',
        labelText: 'H_up',
        rowItems: upRows,
        rowCount,
        columnCount: MLP_HIDDEN_WIDTH,
        visualStyleKey: VIEW2D_STYLE_KEYS.MLP,
        isSmallScreen,
        metadata: {
            kind: 'mlp-up-output'
        }
    });

    const geluLabelNode = createTextNode({
        role: 'gelu-label',
        semantic: buildSemantic(baseSemantic, {
            stage: 'mlp-activation',
            role: 'gelu-label'
        }),
        text: 'GELU',
        tex: '\\operatorname{GELU}',
        visual: {
            styleKey: VIEW2D_STYLE_KEYS.LABEL
        },
        metadata: {
            fontScale: isSmallScreen ? GELU_LABEL_FONT_SCALE_SMALL : GELU_LABEL_FONT_SCALE,
            kind: 'gelu-label'
        }
    });
    const geluOpenParenNode = createOperatorNode({
        role: 'gelu-open-paren',
        semantic: buildSemantic(baseSemantic, {
            stage: 'mlp-activation',
            role: 'gelu-open-paren'
        }),
        text: '(',
        visual: {
            styleKey: VIEW2D_STYLE_KEYS.OPERATOR
        },
        metadata: {
            fontScale: parenFontScale,
            kind: 'gelu-open-paren'
        }
    });
    const upOutputCopyNode = createActivationMatrixNode({
        role: 'mlp-up-output-copy',
        semantic: buildSemantic(baseSemantic, {
            stage: 'mlp-activation',
            role: 'mlp-up-output-copy'
        }),
        labelTex: 'H_{\\mathrm{up}}',
        labelText: 'H_up',
        rowItems: upRows,
        rowCount,
        columnCount: MLP_HIDDEN_WIDTH,
        visualStyleKey: VIEW2D_STYLE_KEYS.MLP,
        isSmallScreen,
        scale: COPY_MATRIX_SCALE,
        metadata: {
            kind: 'mlp-up-output-copy'
        }
    });
    const geluCloseParenNode = createOperatorNode({
        role: 'gelu-close-paren',
        semantic: buildSemantic(baseSemantic, {
            stage: 'mlp-activation',
            role: 'gelu-close-paren'
        }),
        text: ')',
        visual: {
            styleKey: VIEW2D_STYLE_KEYS.OPERATOR
        },
        metadata: {
            fontScale: parenFontScale,
            kind: 'gelu-close-paren'
        }
    });
    const geluCallNode = createGroupNode({
        role: 'gelu-call',
        semantic: buildSemantic(baseSemantic, {
            stage: 'mlp-activation',
            role: 'gelu-call'
        }),
        direction: VIEW2D_LAYOUT_DIRECTIONS.HORIZONTAL,
        gapKey: 'inline',
        metadata: {
            gapOverride: isSmallScreen ? INLINE_GAP_SMALL : INLINE_GAP
        },
        children: [
            geluLabelNode,
            geluOpenParenNode,
            upOutputCopyNode,
            geluCloseParenNode
        ]
    });
    const activationOutputNode = createActivationMatrixNode({
        role: 'mlp-activation-output',
        semantic: buildSemantic(baseSemantic, {
            stage: 'mlp-activation',
            role: 'mlp-activation-output'
        }),
        labelTex: 'H_{\\mathrm{GELU}}',
        labelText: 'H_GELU',
        rowItems: activationRows,
        rowCount,
        columnCount: MLP_HIDDEN_WIDTH,
        visualStyleKey: VIEW2D_STYLE_KEYS.MLP,
        isSmallScreen,
        metadata: {
            kind: 'mlp-activation-output'
        }
    });
    const activationCopyNode = createActivationMatrixNode({
        role: 'mlp-activation-copy',
        semantic: buildSemantic(baseSemantic, {
            stage: 'mlp-down',
            role: 'mlp-activation-copy'
        }),
        labelTex: 'H_{\\mathrm{GELU}}',
        labelText: 'H_GELU',
        rowItems: activationRows,
        rowCount,
        columnCount: MLP_HIDDEN_WIDTH,
        visualStyleKey: VIEW2D_STYLE_KEYS.MLP,
        isSmallScreen,
        scale: DOWNSTREAM_COPY_SCALE,
        metadata: {
            kind: 'mlp-activation-copy'
        }
    });
    const downMultiplyNode = createOperatorNode({
        role: 'mlp-down-multiply',
        semantic: buildSemantic(baseSemantic, {
            stage: 'mlp-down',
            role: 'mlp-down-multiply',
            operatorKey: 'times'
        }),
        text: '×',
        visual: {
            styleKey: VIEW2D_STYLE_KEYS.OPERATOR
        },
        metadata: {
            fontScale: isSmallScreen ? 0.96 : 1.04,
            kind: 'mlp-down-multiply'
        }
    });
    const downWeightNode = createCaptionedCardMatrixNode({
        role: 'mlp-down-weight',
        semantic: buildSemantic(baseSemantic, {
            stage: 'mlp-down',
            role: 'mlp-down-weight'
        }),
        labelTex: 'W_{\\mathrm{down}}',
        labelText: 'W_down',
        rowCount: MLP_HIDDEN_WIDTH,
        columnCount: D_MODEL,
        cardWidth: downWeightCardSize.width,
        cardHeight: downWeightCardSize.height,
        cardCornerRadius: 16,
        captionPosition: 'bottom',
        captionMinScreenHeightPx: 22,
        captionLabelScale: 0.96,
        visualStyleKey: VIEW2D_STYLE_KEYS.MLP,
        metadata: {
            kind: 'mlp-down-weight'
        }
    });
    const outputNode = createActivationMatrixNode({
        role: 'mlp-output',
        semantic: buildSemantic(baseSemantic, {
            stage: 'mlp-output',
            role: 'mlp-output'
        }),
        labelTex: 'Y_{\\mathrm{MLP}}',
        labelText: 'Y_MLP',
        rowItems: downRows,
        rowCount,
        columnCount: D_MODEL,
        visualStyleKey: VIEW2D_STYLE_KEYS.MLP,
        isSmallScreen,
        metadata: {
            kind: 'mlp-output'
        }
    });

    const topStageNode = createGroupNode({
        role: 'mlp-up-stage',
        semantic: buildSemantic(baseSemantic, {
            stage: 'mlp-up',
            role: 'mlp-up-stage'
        }),
        direction: VIEW2D_LAYOUT_DIRECTIONS.HORIZONTAL,
        gapKey: 'stage',
        metadata: {
            gapOverride: isSmallScreen ? FLOW_GAP_SMALL : FLOW_GAP
        },
        children: [
            inputNode,
            upMultiplyNode,
            upWeightNode,
            upOutputNode
        ]
    });
    const bottomStageNode = createGroupNode({
        role: 'mlp-down-stage',
        semantic: buildSemantic(baseSemantic, {
            stage: 'mlp-down',
            role: 'mlp-down-stage'
        }),
        direction: VIEW2D_LAYOUT_DIRECTIONS.HORIZONTAL,
        gapKey: 'stage',
        metadata: {
            gapOverride: isSmallScreen ? FLOW_GAP_SMALL : FLOW_GAP
        },
        layout: {
            anchorAlign: {
                axis: 'x',
                selfNodeId: geluCallNode.id,
                selfAnchor: VIEW2D_ANCHOR_SIDES.CENTER,
                targetNodeId: upOutputNode.id,
                targetAnchor: VIEW2D_ANCHOR_SIDES.CENTER,
                offset: isSmallScreen ? STAGE_ALIGNMENT_OFFSET_SMALL : STAGE_ALIGNMENT_OFFSET
            }
        },
        children: [
            geluCallNode,
            activationOutputNode,
            activationCopyNode,
            downMultiplyNode,
            downWeightNode,
            outputNode
        ]
    });

    const connectors = [
        createFlowConnector({
            role: 'mlp-up-output-flow',
            semantic: buildSemantic(baseSemantic, {
                stage: 'mlp-up',
                role: 'mlp-up-output-flow'
            }),
            fromNode: upWeightNode,
            toNode: upOutputNode
        }),
        createFlowConnector({
            role: 'mlp-up-to-gelu-flow',
            semantic: buildSemantic(baseSemantic, {
                stage: 'mlp-activation',
                role: 'mlp-up-to-gelu-flow'
            }),
            fromNode: upOutputNode,
            toNode: geluCallNode,
            sourceAnchor: VIEW2D_ANCHOR_SIDES.BOTTOM,
            targetAnchor: VIEW2D_ANCHOR_SIDES.TOP,
            route: VIEW2D_CONNECTOR_ROUTES.VERTICAL
        }),
        createFlowConnector({
            role: 'mlp-gelu-output-flow',
            semantic: buildSemantic(baseSemantic, {
                stage: 'mlp-activation',
                role: 'mlp-gelu-output-flow'
            }),
            fromNode: geluCallNode,
            toNode: activationOutputNode
        }),
        createFlowConnector({
            role: 'mlp-activation-copy-flow',
            semantic: buildSemantic(baseSemantic, {
                stage: 'mlp-down',
                role: 'mlp-activation-copy-flow'
            }),
            fromNode: activationOutputNode,
            toNode: activationCopyNode
        }),
        createFlowConnector({
            role: 'mlp-down-output-flow',
            semantic: buildSemantic(baseSemantic, {
                stage: 'mlp-down',
                role: 'mlp-down-output-flow'
            }),
            fromNode: downWeightNode,
            toNode: outputNode
        })
    ].filter(Boolean);

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
                direction: VIEW2D_LAYOUT_DIRECTIONS.VERTICAL,
                gapKey: 'default',
                metadata: {
                    gapOverride: isSmallScreen ? SCENE_GAP_Y_SMALL : SCENE_GAP_Y
                },
                children: [
                    topStageNode,
                    bottomStageNode
                ]
            }),
            createGroupNode({
                role: 'mlp-detail-connectors',
                semantic: buildSemantic(baseSemantic, {
                    role: 'mlp-detail-connectors'
                }),
                direction: VIEW2D_LAYOUT_DIRECTIONS.OVERLAY,
                gapKey: 'default',
                children: connectors
            })
        ],
        metadata: {
            visualContract: 'selection-panel-mlp-v1',
            source: 'buildMlpDetailSceneModel',
            layerIndex,
            rowCount,
            isSmallScreen: !!isSmallScreen,
            layoutMetrics: {
                cssVars: {
                    '--mhsa-token-matrix-canvas-pad-x-boost': isSmallScreen ? '-20px' : '-32px',
                    '--mhsa-token-matrix-canvas-pad-y-boost': isSmallScreen ? '-10px' : '-16px',
                    '--mhsa-token-matrix-stage-gap-boost': isSmallScreen ? '-6px' : '-8px'
                }
            },
            tokens: visualTokens || resolveView2dVisualTokens()
        }
    });
}
