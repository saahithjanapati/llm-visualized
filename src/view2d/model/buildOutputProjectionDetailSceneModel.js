import { MHA_FINAL_V_COLOR } from '../../animations/LayerAnimationConstants.js';
import { D_HEAD } from '../../ui/selectionPanelConstants.js';
import { NUM_HEAD_SETS_LAYER } from '../../utils/constants.js';
import { buildHueRangeOptions, mapValueToHueRange } from '../../utils/colors.js';
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

const HEAD_OUTPUT_MEASURE_COLS = 12;
const HEAD_OUTPUT_ROW_HEIGHT = 7;
const HEAD_OUTPUT_ROW_HEIGHT_SMALL = 6;
const HEAD_OUTPUT_STACK_GAP = 40;
const HEAD_OUTPUT_STACK_GAP_SMALL = 32;
const INCOMING_ARROW_SPACER_WIDTH = 56;
const INCOMING_ARROW_SPACER_WIDTH_SMALL = 48;
const ARROW_HEAD_TARGET_GAP = 12;
const OUTPUT_HEAD_CAPTION_LABEL_SCALE = 0.9;

const HEAD_OUTPUT_RANGE_OPTIONS = buildHueRangeOptions(MHA_FINAL_V_COLOR, {
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

function buildGradientCss(values = [], direction = '90deg') {
    const safeValues = cleanNumberArray(values);
    if (!safeValues.length) return 'none';
    const stops = safeValues.map((value, index) => {
        const ratio = safeValues.length > 1 ? index / (safeValues.length - 1) : 0;
        return `${colorToCss(mapValueToHueRange(value, HEAD_OUTPUT_RANGE_OPTIONS))} ${(ratio * 100).toFixed(4)}%`;
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

function buildHeadOutputRowItems(activationSource = null, tokenRefs = [], {
    layerIndex = null,
    headIndex = null
} = {}) {
    return tokenRefs.map((tokenRef) => {
        const tokenIndex = normalizeIndex(tokenRef?.tokenIndex);
        const rawVector = typeof activationSource?.getAttentionWeightedSum === 'function'
            ? activationSource.getAttentionWeightedSum(layerIndex, headIndex, tokenIndex, D_HEAD)
            : null;
        const sampledValues = sampleVector(
            cleanNumberArray(rawVector, D_HEAD),
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
            rawValues: sampledValues,
            gradientCss: buildGradientCss(sampledValues),
            title: `${label}: H_${Math.max(1, (headIndex ?? 0) + 1)}`
        };
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
    const compactWidth = resolveHeadOutputCompactWidth({
        isSmallScreen
    });
    const rowHeight = isSmallScreen ? HEAD_OUTPUT_ROW_HEIGHT_SMALL : HEAD_OUTPUT_ROW_HEIGHT;
    const matrixNode = createVectorStripMatrixNode({
        role: 'head-output-matrix',
        semantic: buildSemantic(baseSemantic, {
            role: 'head-output-matrix'
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
                direction: VIEW2D_LAYOUT_DIRECTIONS.VERTICAL,
                gapKey: 'default',
                children: headEntries.map((entry) => entry.rowNode),
                metadata: {
                    gapOverride: isSmallScreen ? HEAD_OUTPUT_STACK_GAP_SMALL : HEAD_OUTPUT_STACK_GAP
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
                children: headEntries.map((entry) => entry.connectorNode)
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
