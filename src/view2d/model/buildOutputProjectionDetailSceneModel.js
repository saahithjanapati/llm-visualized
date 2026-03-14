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
import { createView2dVectorStripMetadata } from '../shared/vectorStrip.js';
import { resolveMhsaDimensionVisualExtent } from '../shared/mhsaDimensionSizing.js';

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
const CONCAT_LABEL_FONT_SCALE = 1.32;
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
    layerIndex = null
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
            stage: 'concatenate',
            role: 'concat-output-row',
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
    return createVectorStripMatrixNode({
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
}

function createConcatOutputMatrixNode(activationSource = null, tokenRefs = [], {
    layerIndex = null,
    rowCount = 1,
    isSmallScreen = false
} = {}) {
    const compactWidth = isSmallScreen ? CONCAT_OUTPUT_COMPACT_WIDTH_SMALL : CONCAT_OUTPUT_COMPACT_WIDTH;
    const rowHeight = isSmallScreen ? HEAD_OUTPUT_ROW_HEIGHT_SMALL : HEAD_OUTPUT_ROW_HEIGHT;
    return createVectorStripMatrixNode({
        role: 'concat-output-matrix',
        semantic: {
            componentKind: 'output-projection',
            layerIndex,
            stage: 'concatenate',
            role: 'concat-output-matrix'
        },
        labelTex: 'H_{\\mathrm{concat}}',
        labelText: 'H_concat',
        rowItems: buildConcatOutputRowItems(activationSource, tokenRefs, {
            layerIndex
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
                strokeWidthScale: 0.72
            }
        });
    }).filter(Boolean);

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
                    concatStage.node
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
                    ...concatCopyConnectorNodes
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
