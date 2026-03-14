import { mapValueToColor } from '../../utils/colors.js';
import {
    D_MODEL,
    RESIDUAL_COLOR_CLAMP
} from '../../ui/selectionPanelConstants.js';
import {
    buildSceneNodeId,
    createAnchorRef,
    createConnectorNode,
    createGroupNode,
    createMatrixNode,
    createSceneModel,
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
import { createView2dVectorStripMetadata } from '../shared/vectorStrip.js';
import { resolveMhsaDimensionVisualExtent } from '../shared/mhsaDimensionSizing.js';

const INPUT_MEASURE_COLS = 12;
const BASE_VECTOR_ROW_HEIGHT = 8;
const BASE_VECTOR_ROW_HEIGHT_SMALL = 7;
const BASE_VECTOR_ROW_GAP = 0;
const BASE_VECTOR_PADDING_Y = 0;
const BASE_VECTOR_PADDING_X = 0;
const INPUT_CAPTION_LABEL_SCALE = 0.9;
const INCOMING_ARROW_SPACER_WIDTH = 60;
const INCOMING_ARROW_SPACER_WIDTH_SMALL = 52;
const EDGE_CONNECTOR_STROKE_WIDTH_SCALE = 0.88;
const INPUT_CONNECTOR_TARGET_GAP = 8;

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
    const rowHeight = isSmallScreen ? BASE_VECTOR_ROW_HEIGHT_SMALL : BASE_VECTOR_ROW_HEIGHT;
    return {
        rowHeight,
        rowGap: BASE_VECTOR_ROW_GAP,
        paddingX: BASE_VECTOR_PADDING_X,
        paddingY: BASE_VECTOR_PADDING_Y,
        compactWidth: resolveCompactWidth(columnCount, {
            isSmallScreen
        }),
        measureCols: resolveMeasureCols(columnCount),
        rowCount: Math.max(1, Math.floor(rowCount || 1))
    };
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
    const vectorMetrics = resolveVectorMetrics(rowCount, {
        isSmallScreen,
        columnCount: D_MODEL
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

    const incomingArrowSpacerNode = createHiddenSpacer({
        semantic: buildSemantic(baseSemantic, {
            stage: 'mlp-input',
            role: 'incoming-arrow-spacer'
        }),
        role: 'incoming-arrow-spacer',
        width: isSmallScreen ? INCOMING_ARROW_SPACER_WIDTH_SMALL : INCOMING_ARROW_SPACER_WIDTH,
        height: 1
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
                gapKey: 'default',
                children: [
                    incomingArrowSpacerNode,
                    inputNode
                ]
            }),
            createGroupNode({
                role: 'connector-layer',
                semantic: buildSemantic(baseSemantic, {
                    stage: 'connector-layer',
                    role: 'connector-layer'
                }),
                direction: VIEW2D_LAYOUT_DIRECTIONS.OVERLAY,
                gapKey: 'default',
                children: [incomingConnectorNode]
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
