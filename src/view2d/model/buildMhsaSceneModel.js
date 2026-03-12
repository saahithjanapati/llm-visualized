import { buildMhsaTokenMatrixPreviewData } from '../../ui/selectionPanelMhsaTokenMatrixUtils.js';
import { resolveMhsaTokenMatrixLayoutMetrics } from '../../ui/selectionPanelMhsaLayoutUtils.js';
import {
    createCaptionedCardMatrixNode,
    resolveRelativeCardSize
} from './createCaptionedCardMatrixNode.js';
import {
    createTransposeVectorStripMatrixNode,
    createVectorStripMatrixNode
} from './createResidualVectorMatrixNode.js';
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
    VIEW2D_MATRIX_SHAPES,
    VIEW2D_TEXT_PRESENTATIONS
} from '../schema/sceneTypes.js';
import {
    resolveView2dVisualTokens,
    VIEW2D_STYLE_KEYS
} from '../theme/visualTokens.js';
import { createView2dVectorStripMetadata } from '../shared/vectorStrip.js';
import { formatView2dMatrixDimensions } from '../shared/formatMatrixDimensions.js';

const MHSA_SOURCE_ANCHOR_WIDTH = 22;
const MHSA_SOURCE_ANCHOR_MIN_HEIGHT = 108;
const MHSA_SOURCE_ANCHOR_MAX_HEIGHT = 176;
const MHSA_SOURCE_ANCHOR_ROW_HEIGHT = 18;
const PROJECTION_STACK_GAP = 120;
const PROJECTION_STACK_GAP_SMALL = 92;
const PROJECTION_SIDECAR_GAP = 30;
const PROJECTION_SIDECAR_GAP_SMALL = 24;
const PROJECTION_STAGE_INLINE_GAP = 12;
const PROJECTION_STAGE_INLINE_GAP_SMALL = 10;
const PROJECTION_EQUATION_GAP = 12;
const PROJECTION_EQUATION_GAP_SMALL = 10;
const PROJECTION_MULTIPLY_OPERATOR_SCALE = 0.82;
const PROJECTION_XLN_COMPACT_WIDTH = 88;
const PROJECTION_XLN_COMPACT_WIDTH_SMALL = 76;
const PROJECTION_XLN_ROW_HEIGHT = 7;
const PROJECTION_XLN_ROW_HEIGHT_SMALL = 6;
const PROJECTION_WEIGHT_REFERENCE_EXTENT = 92;
const PROJECTION_WEIGHT_REFERENCE_EXTENT_SMALL = 84;
const PROJECTION_WEIGHT_MIN_WIDTH = 48;
const PROJECTION_WEIGHT_MAX_WIDTH = 138;
const PROJECTION_WEIGHT_MIN_HEIGHT = 56;
const PROJECTION_WEIGHT_MAX_HEIGHT = 144;
const PROJECTION_CAPTION_LABEL_SCALE = 0.42;
const PROJECTION_CAPTION_DIMENSIONS_SCALE = 0.5;
const PROJECTION_BIAS_ROW_HEIGHT = 14;
const PROJECTION_BIAS_CAPTION_MIN_SCREEN_HEIGHT = 12;
const PROJECTION_BIAS_COMPACT_WIDTH = 58;
const PROJECTION_BIAS_COMPACT_WIDTH_SMALL = 50;
const PROJECTION_BIAS_CORNER_RADIUS = 5;
const PROJECTION_OUTPUT_COMPACT_WIDTH = 72;
const PROJECTION_OUTPUT_COMPACT_WIDTH_SMALL = 62;
const PROJECTION_OUTPUT_ROW_HEIGHT = 7;
const PROJECTION_OUTPUT_ROW_HEIGHT_SMALL = 6;
const PROJECTION_XLN_TARGET_GAP = 8;
const ATTENTION_TRANSPOSE_MIN_WIDTH = 24;
const ATTENTION_TRANSPOSE_MIN_WIDTH_SMALL = 20;
const ATTENTION_TRANSPOSE_CORNER_RADIUS = 8;
const ATTENTION_STAGE_VERTICAL_LIFT = 104;
const ATTENTION_STAGE_VERTICAL_LIFT_SMALL = 80;
const ATTENTION_STAGE_VERTICAL_LIFT_PER_EXTRA_ROW = 6;
const ATTENTION_STAGE_VERTICAL_LIFT_PER_EXTRA_ROW_SMALL = 5;
const MHSA_CONNECTOR_STROKE = 'rgba(255, 255, 255, 0.84)';

function normalizeIndex(value) {
    return Number.isFinite(value) ? Math.floor(value) : null;
}

function buildSemantic(baseSemantic, extra = {}) {
    return {
        ...baseSemantic,
        ...extra
    };
}

function buildLabel(labelTex = '', fallbackText = '') {
    return {
        tex: typeof labelTex === 'string' ? labelTex : '',
        text: typeof fallbackText === 'string' && fallbackText.length
            ? fallbackText
            : (typeof labelTex === 'string' ? labelTex : '')
    };
}

function createCardMetadata(width = null, height = null, {
    hidden = false,
    cornerRadius = null
} = {}) {
    const card = {};
    if (Number.isFinite(width) && width > 0) card.width = Math.floor(width);
    if (Number.isFinite(height) && height > 0) card.height = Math.floor(height);
    if (Number.isFinite(cornerRadius) && cornerRadius >= 0) card.cornerRadius = Math.floor(cornerRadius);
    const metadata = Object.keys(card).length ? { card } : {};
    if (hidden) metadata.hidden = true;
    return Object.keys(metadata).length ? metadata : null;
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
        metadata: createCardMetadata(width, height, {
            hidden: true,
            cornerRadius: 0
        })
    });
}

function resolveAttentionStageLift(rowCount = 1, isSmallScreen = false) {
    const safeRowCount = Number.isFinite(rowCount) ? Math.max(1, Math.floor(rowCount)) : 1;
    const extraRows = Math.max(0, safeRowCount - 5);
    const baseLift = isSmallScreen ? ATTENTION_STAGE_VERTICAL_LIFT_SMALL : ATTENTION_STAGE_VERTICAL_LIFT;
    const perExtraRow = isSmallScreen
        ? ATTENTION_STAGE_VERTICAL_LIFT_PER_EXTRA_ROW_SMALL
        : ATTENTION_STAGE_VERTICAL_LIFT_PER_EXTRA_ROW;
    return Math.max(0, Math.round(baseLift + (extraRows * perExtraRow)));
}

function buildGradientRowItems(rows = [], baseSemantic = {}, role = 'row') {
    return rows.map((rowData) => {
        const semantic = buildSemantic(baseSemantic, {
            role,
            rowIndex: rowData.rowIndex,
            tokenIndex: rowData.tokenIndex
        });
        return {
            id: buildSceneNodeId(semantic),
            index: rowData.rowIndex,
            label: rowData.tokenLabel || `Token ${rowData.rowIndex + 1}`,
            semantic,
            rawValue: Number.isFinite(rowData.rawValue) ? rowData.rawValue : null,
            rawValues: Array.isArray(rowData.rawValues) ? [...rowData.rawValues] : null,
            gradientCss: rowData.gradientCss || 'none',
            title: typeof rowData.title === 'string' ? rowData.title : null
        };
    });
}

function buildProjectionInputRowItems(rows = [], baseSemantic = {}, branchKey = '') {
    return rows.map((rowData) => {
        const semantic = {
            componentKind: 'residual',
            layerIndex: baseSemantic?.layerIndex,
            headIndex: baseSemantic?.headIndex,
            stage: 'ln1.shift',
            role: 'x-ln-row',
            rowIndex: rowData.rowIndex,
            tokenIndex: rowData.tokenIndex,
            ...(branchKey ? { branchKey } : {})
        };
        return {
            id: buildSceneNodeId(semantic),
            index: rowData.rowIndex,
            label: rowData.tokenLabel || `Token ${rowData.rowIndex + 1}`,
            semantic,
            rawValue: Number.isFinite(rowData.rawValue) ? rowData.rawValue : null,
            rawValues: Array.isArray(rowData.rawValues) ? [...rowData.rawValues] : null,
            gradientCss: rowData.gradientCss || 'none',
            title: typeof rowData.title === 'string' ? rowData.title : null
        };
    });
}

function buildProjectionOutputRowItems(rows = [], baseSemantic = {}, branchKey = '') {
    return rows.map((rowData) => {
        const semantic = {
            componentKind: 'mhsa',
            layerIndex: baseSemantic?.layerIndex,
            headIndex: baseSemantic?.headIndex,
            stage: `qkv.${branchKey}`,
            role: 'projection-output-row',
            rowIndex: rowData.rowIndex,
            tokenIndex: rowData.tokenIndex,
            ...(branchKey ? { branchKey } : {})
        };
        return {
            id: buildSceneNodeId(semantic),
            index: rowData.rowIndex,
            label: rowData.tokenLabel || `Token ${rowData.rowIndex + 1}`,
            semantic,
            rawValue: Number.isFinite(rowData.rawValue) ? rowData.rawValue : null,
            rawValues: Array.isArray(rowData.rawValues) ? [...rowData.rawValues] : null,
            gradientCss: rowData.gradientCss || 'none',
            title: typeof rowData.title === 'string' ? rowData.title : null
        };
    });
}

function buildProjectionBiasRowItems(projectionData = null, projectionSemantic = {}) {
    const rowSemantic = buildSemantic(projectionSemantic, {
        role: 'projection-bias-row',
        rowIndex: 0
    });
    return [{
        id: buildSceneNodeId(rowSemantic),
        index: 0,
        label: '',
        semantic: rowSemantic,
        rawValue: Number.isFinite(projectionData?.biasValue) ? projectionData.biasValue : null,
        gradientCss: projectionData?.biasVectorGradientCss || projectionData?.biasGradientCss || 'none',
        title: typeof projectionData?.biasLabelTex === 'string' ? projectionData.biasLabelTex : null
    }];
}

function buildAttentionGridRowItems(rows = [], baseSemantic = {}) {
    return rows.map((rowData) => {
        const rowSemantic = buildSemantic(baseSemantic, {
            role: 'attention-row',
            rowIndex: rowData.rowIndex
        });
        return {
            id: buildSceneNodeId(rowSemantic),
            index: rowData.rowIndex,
            label: rowData.tokenLabel || `Token ${rowData.rowIndex + 1}`,
            semantic: rowSemantic,
            cells: Array.isArray(rowData.cells)
                ? rowData.cells.map((cellData) => {
                    const cellSemantic = buildSemantic(baseSemantic, {
                        role: 'attention-cell',
                        rowIndex: cellData.rowIndex,
                        colIndex: cellData.colIndex
                    });
                    return {
                        id: buildSceneNodeId(cellSemantic),
                        rowIndex: cellData.rowIndex,
                        colIndex: cellData.colIndex,
                        semantic: cellSemantic,
                        rowLabel: cellData.rowTokenLabel || rowData.tokenLabel || '',
                        colLabel: cellData.colTokenLabel || '',
                        rawValue: Number.isFinite(cellData.rawValue) ? cellData.rawValue : null,
                        fillCss: cellData.fillCss || 'transparent',
                        isMasked: !!cellData.isMasked,
                        isEmpty: !!cellData.isEmpty,
                        title: typeof cellData.title === 'string' ? cellData.title : null
                    };
                })
                : []
        };
    });
}

function buildTransposeColumnItems(columns = [], baseSemantic = {}) {
    return columns.map((columnData) => {
        const semantic = buildSemantic(baseSemantic, {
            stage: 'qkv.k',
            role: 'transpose-column',
            colIndex: columnData.colIndex,
            tokenIndex: columnData.colIndex,
            branchKey: 'k'
        });
        return {
            id: buildSceneNodeId(semantic),
            index: columnData.colIndex,
            label: columnData.tokenLabel || `Token ${columnData.colIndex + 1}`,
            semantic,
            rawValue: Number.isFinite(columnData.rawValue) ? columnData.rawValue : null,
            fillCss: columnData.fillCss || 'transparent',
            title: typeof columnData.tokenLabel === 'string' ? columnData.tokenLabel : null
        };
    });
}

function buildProjectionSourceAnchorNode({
    baseSemantic,
    rowCount = 1,
    isSmallScreen = false
}) {
    const safeRowCount = Number.isFinite(rowCount) ? Math.max(1, Math.floor(rowCount)) : 1;
    const cardHeight = Math.max(
        isSmallScreen ? MHSA_SOURCE_ANCHOR_MIN_HEIGHT - 16 : MHSA_SOURCE_ANCHOR_MIN_HEIGHT,
        Math.min(
            isSmallScreen ? MHSA_SOURCE_ANCHOR_MAX_HEIGHT - 20 : MHSA_SOURCE_ANCHOR_MAX_HEIGHT,
            (safeRowCount * MHSA_SOURCE_ANCHOR_ROW_HEIGHT) + (isSmallScreen ? 40 : 52)
        )
    );
    const cardWidth = isSmallScreen ? MHSA_SOURCE_ANCHOR_WIDTH - 4 : MHSA_SOURCE_ANCHOR_WIDTH;
    return createMatrixNode({
        role: 'projection-source-anchor',
        semantic: buildSemantic(baseSemantic, {
            stage: 'projection-source',
            role: 'projection-source-anchor'
        }),
        dimensions: {
            rows: 1,
            cols: 1
        },
        presentation: VIEW2D_MATRIX_PRESENTATIONS.CARD,
        shape: VIEW2D_MATRIX_SHAPES.MATRIX,
        visual: {
            styleKey: VIEW2D_STYLE_KEYS.MHSA_HEAD
        },
        metadata: createCardMetadata(cardWidth, cardHeight, {
            hidden: true,
            cornerRadius: 0
        })
    });
}

function buildProjectionStageNode({
    baseSemantic,
    previewData,
    projectionData,
    stageIndex,
    isSmallScreen = false
}) {
    const projectionKind = String(projectionData?.kind || '').toLowerCase();
    const projectionSemantic = buildSemantic(baseSemantic, {
        stage: `projection-${projectionKind}`,
        stageIndex
    });
    const inputDimensionCaption = formatView2dMatrixDimensions(previewData.rowCount, previewData.columnCount);
    const biasDimensionCaption = formatView2dMatrixDimensions(1, projectionData.outputColumnCount);
    const styleKey = projectionKind === 'q'
        ? VIEW2D_STYLE_KEYS.MHSA_Q
        : (projectionKind === 'k' ? VIEW2D_STYLE_KEYS.MHSA_K : VIEW2D_STYLE_KEYS.MHSA_V);

    const xInputNode = createVectorStripMatrixNode({
        role: 'x-ln-copy',
        semantic: buildSemantic(projectionSemantic, {
            role: 'x-ln-copy',
            branchKey: projectionKind
        }),
        labelTex: 'X_{\\ln}',
        labelText: 'X_ln',
        rowItems: buildProjectionInputRowItems(previewData.rows, baseSemantic, projectionKind),
        rowCount: previewData.rowCount,
        compactWidth: isSmallScreen ? PROJECTION_XLN_COMPACT_WIDTH_SMALL : PROJECTION_XLN_COMPACT_WIDTH,
        rowHeight: isSmallScreen ? PROJECTION_XLN_ROW_HEIGHT_SMALL : PROJECTION_XLN_ROW_HEIGHT,
        captionPosition: 'bottom',
        captionDimensionsTex: inputDimensionCaption.tex,
        captionDimensionsText: inputDimensionCaption.text,
        captionScaleWithNode: true,
        visualStyleKey: VIEW2D_STYLE_KEYS.RESIDUAL
    });

    const weightCardSize = resolveRelativeCardSize({
        rows: projectionData.weightRowCount,
        cols: projectionData.weightColumnCount,
        referenceCount: previewData.columnCount,
        referenceExtent: isSmallScreen ? PROJECTION_WEIGHT_REFERENCE_EXTENT_SMALL : PROJECTION_WEIGHT_REFERENCE_EXTENT,
        minWidth: PROJECTION_WEIGHT_MIN_WIDTH,
        maxWidth: PROJECTION_WEIGHT_MAX_WIDTH,
        minHeight: PROJECTION_WEIGHT_MIN_HEIGHT,
        maxHeight: PROJECTION_WEIGHT_MAX_HEIGHT
    });
    const weightNode = createCaptionedCardMatrixNode({
        role: 'projection-weight',
        semantic: buildSemantic(projectionSemantic, { role: 'projection-weight' }),
        labelTex: projectionData.weightLabelTex,
        labelText: projectionData.weightLabelTex,
        rowCount: projectionData.weightRowCount,
        columnCount: projectionData.weightColumnCount,
        cardWidth: weightCardSize.width,
        cardHeight: weightCardSize.height,
        cardCornerRadius: 10,
        captionMinScreenHeightPx: 28,
        visualStyleKey: styleKey,
        captionScaleWithNode: true,
        captionLabelScale: PROJECTION_CAPTION_LABEL_SCALE,
        captionDimensionsScale: PROJECTION_CAPTION_DIMENSIONS_SCALE,
        background: projectionData.weightGradientCss || 'none',
        metadata: {
            kind: projectionKind
        }
    });

    const biasNode = createVectorStripMatrixNode({
        role: 'projection-bias',
        semantic: buildSemantic(projectionSemantic, { role: 'projection-bias' }),
        labelTex: projectionData.biasLabelTex,
        labelText: projectionData.biasLabelTex,
        rowItems: buildProjectionBiasRowItems(projectionData, projectionSemantic),
        rowCount: 1,
        columnCount: projectionData.outputColumnCount,
        compactWidth: isSmallScreen ? PROJECTION_BIAS_COMPACT_WIDTH_SMALL : PROJECTION_BIAS_COMPACT_WIDTH,
        rowHeight: PROJECTION_BIAS_ROW_HEIGHT,
        captionPosition: 'bottom',
        captionMinScreenHeightPx: PROJECTION_BIAS_CAPTION_MIN_SCREEN_HEIGHT,
        captionDimensionsTex: biasDimensionCaption.tex,
        captionDimensionsText: biasDimensionCaption.text,
        captionScaleWithNode: true,
        captionLabelScale: PROJECTION_CAPTION_LABEL_SCALE,
        captionDimensionsScale: PROJECTION_CAPTION_DIMENSIONS_SCALE,
        visualStyleKey: styleKey,
        stripMetadata: createView2dVectorStripMetadata({
            compactWidth: isSmallScreen ? PROJECTION_BIAS_COMPACT_WIDTH_SMALL : PROJECTION_BIAS_COMPACT_WIDTH,
            rowHeight: PROJECTION_BIAS_ROW_HEIGHT,
            cornerRadius: PROJECTION_BIAS_CORNER_RADIUS,
            hideSurface: true
        }),
        metadata: {
            kind: projectionKind
        }
    });

    const outputDimensionCaption = formatView2dMatrixDimensions(
        projectionData.outputRowCount,
        projectionData.outputColumnCount
    );
    const outputNode = createVectorStripMatrixNode({
        role: 'projection-output',
        semantic: buildSemantic(projectionSemantic, { role: 'projection-output' }),
        labelTex: projectionData.outputLabelTex,
        labelText: projectionData.outputLabelTex,
        rowItems: buildProjectionOutputRowItems(projectionData.outputRows, baseSemantic, projectionKind),
        rowCount: projectionData.outputRowCount,
        columnCount: projectionData.outputColumnCount,
        compactWidth: isSmallScreen ? PROJECTION_OUTPUT_COMPACT_WIDTH_SMALL : PROJECTION_OUTPUT_COMPACT_WIDTH,
        rowHeight: isSmallScreen ? PROJECTION_OUTPUT_ROW_HEIGHT_SMALL : PROJECTION_OUTPUT_ROW_HEIGHT,
        captionPosition: 'bottom',
        captionDimensionsTex: outputDimensionCaption.tex,
        captionDimensionsText: outputDimensionCaption.text,
        captionScaleWithNode: true,
        visualStyleKey: styleKey,
        metadata: {
            kind: projectionKind,
            stageIndex
        }
    });

    const equationNode = createGroupNode({
        role: 'projection-equation',
        semantic: buildSemantic(projectionSemantic, { role: 'projection-equation' }),
        direction: VIEW2D_LAYOUT_DIRECTIONS.HORIZONTAL,
        gapKey: 'projection',
        children: [
            weightNode,
            createOperatorNode({
                role: 'projection-plus',
                semantic: buildSemantic(projectionSemantic, { role: 'projection-plus', operatorKey: 'plus' }),
                text: '+',
                visual: { styleKey: VIEW2D_STYLE_KEYS.OPERATOR }
            }),
            biasNode,
            createOperatorNode({
                role: 'projection-equals',
                semantic: buildSemantic(projectionSemantic, { role: 'projection-equals', operatorKey: 'equals' }),
                text: '=',
                visual: { styleKey: VIEW2D_STYLE_KEYS.OPERATOR }
            }),
            outputNode
        ],
        metadata: {
            gapOverride: isSmallScreen ? PROJECTION_EQUATION_GAP_SMALL : PROJECTION_EQUATION_GAP
        }
    });

    return createGroupNode({
        role: 'projection-stage',
        semantic: buildSemantic(projectionSemantic, { role: 'projection-stage' }),
        direction: VIEW2D_LAYOUT_DIRECTIONS.HORIZONTAL,
        gapKey: 'projection',
        children: [
            xInputNode,
            createOperatorNode({
                role: 'projection-multiply',
                semantic: buildSemantic(projectionSemantic, { role: 'projection-multiply', operatorKey: 'multiply' }),
                text: 'x',
                visual: { styleKey: VIEW2D_STYLE_KEYS.OPERATOR },
                metadata: {
                    fontScale: PROJECTION_MULTIPLY_OPERATOR_SCALE
                }
            }),
            equationNode
        ],
        metadata: {
            gapOverride: isSmallScreen ? PROJECTION_STAGE_INLINE_GAP_SMALL : PROJECTION_STAGE_INLINE_GAP,
            kind: projectionKind,
            stageIndex
        }
    });
}

function buildAttentionStageNode({
    baseSemantic,
    scoreStage,
    queryStageIndex,
    valueStageIndex,
    isSmallScreen = false
}) {
    const attentionSemantic = buildSemantic(baseSemantic, {
        stage: 'attention'
    });
    const attentionQueryCompactWidth = isSmallScreen
        ? PROJECTION_OUTPUT_COMPACT_WIDTH_SMALL
        : PROJECTION_OUTPUT_COMPACT_WIDTH;
    const attentionQueryRowHeight = isSmallScreen
        ? PROJECTION_OUTPUT_ROW_HEIGHT_SMALL
        : PROJECTION_OUTPUT_ROW_HEIGHT;
    const queryDimensionCaption = formatView2dMatrixDimensions(
        scoreStage.queryRowCount,
        scoreStage.queryColumnCount
    );
    const transposeDimensionCaption = formatView2dMatrixDimensions(
        scoreStage.transposeRowCount,
        scoreStage.transposeColumnCount
    );
    const transposeTargetWidth = Math.max(
        isSmallScreen ? ATTENTION_TRANSPOSE_MIN_WIDTH_SMALL : ATTENTION_TRANSPOSE_MIN_WIDTH,
        scoreStage.transposeColumnCount * attentionQueryRowHeight
    );
    const transposeColumnWidth = Math.max(
        1,
        Math.floor(transposeTargetWidth / Math.max(1, scoreStage.transposeColumnCount))
    );

    const querySourceNode = createVectorStripMatrixNode({
        role: 'attention-query-source',
        semantic: buildSemantic(attentionSemantic, { role: 'attention-query-source' }),
        labelTex: scoreStage.queryLabelTex,
        labelText: scoreStage.queryLabelTex,
        rowItems: buildGradientRowItems(scoreStage.queryRows, attentionSemantic, 'attention-query-row'),
        rowCount: scoreStage.queryRowCount,
        columnCount: scoreStage.queryColumnCount,
        compactWidth: attentionQueryCompactWidth,
        rowHeight: attentionQueryRowHeight,
        captionPosition: 'bottom',
        captionDimensionsTex: queryDimensionCaption.tex,
        captionDimensionsText: queryDimensionCaption.text,
        captionScaleWithNode: true,
        visualStyleKey: VIEW2D_STYLE_KEYS.MHSA_Q,
        metadata: {
            stageIndex: queryStageIndex
        }
    });

    const transposeNode = createTransposeVectorStripMatrixNode({
        role: 'attention-key-transpose',
        semantic: buildSemantic(attentionSemantic, { role: 'attention-key-transpose' }),
        labelTex: scoreStage.transposeLabelTex,
        labelText: scoreStage.transposeLabelTex,
        columnItems: buildTransposeColumnItems(scoreStage.transposeColumns, attentionSemantic),
        rowCount: scoreStage.transposeRowCount,
        columnCount: scoreStage.transposeColumnCount,
        compactHeight: attentionQueryCompactWidth,
        columnWidth: transposeColumnWidth,
        columnGap: 0,
        paddingX: 0,
        paddingY: 0,
        cornerRadius: ATTENTION_TRANSPOSE_CORNER_RADIUS,
        captionPosition: 'bottom',
        captionDimensionsTex: transposeDimensionCaption.tex,
        captionDimensionsText: transposeDimensionCaption.text,
        captionScaleWithNode: true,
        visualStyleKey: VIEW2D_STYLE_KEYS.MHSA_K
    });

    const preScoreNode = createMatrixNode({
        role: 'attention-pre-score',
        semantic: buildSemantic(attentionSemantic, { role: 'attention-pre-score' }),
        label: buildLabel(scoreStage.outputLabelTex, scoreStage.outputLabelTex),
        dimensions: {
            rows: scoreStage.outputRowCount,
            cols: scoreStage.outputColumnCount
        },
        presentation: VIEW2D_MATRIX_PRESENTATIONS.GRID,
        rowItems: buildAttentionGridRowItems(scoreStage.outputRows, buildSemantic(attentionSemantic, { stage: 'attention-pre-score' })),
        visual: {
            styleKey: VIEW2D_STYLE_KEYS.MHSA_SCORE
        }
    });

    const maskedInputNode = createMatrixNode({
        role: 'attention-masked-input',
        semantic: buildSemantic(attentionSemantic, { role: 'attention-masked-input' }),
        label: buildLabel(scoreStage.outputLabelTex, scoreStage.outputLabelTex),
        dimensions: {
            rows: scoreStage.outputRowCount,
            cols: scoreStage.outputColumnCount
        },
        presentation: VIEW2D_MATRIX_PRESENTATIONS.GRID,
        rowItems: buildAttentionGridRowItems(scoreStage.outputRows, buildSemantic(attentionSemantic, { stage: 'attention-masked-input' })),
        visual: {
            styleKey: VIEW2D_STYLE_KEYS.MHSA_SCORE
        }
    });

    const maskNode = createMatrixNode({
        role: 'attention-mask',
        semantic: buildSemantic(attentionSemantic, { role: 'attention-mask' }),
        label: buildLabel(scoreStage.maskLabelTex, scoreStage.maskLabelTex),
        dimensions: {
            rows: scoreStage.outputRowCount,
            cols: scoreStage.outputColumnCount
        },
        presentation: VIEW2D_MATRIX_PRESENTATIONS.GRID,
        rowItems: buildAttentionGridRowItems(scoreStage.maskRows, buildSemantic(attentionSemantic, { stage: 'attention-mask' })),
        visual: {
            styleKey: VIEW2D_STYLE_KEYS.MHSA_MASK
        }
    });

    const postNode = createMatrixNode({
        role: 'attention-post',
        semantic: buildSemantic(attentionSemantic, { role: 'attention-post' }),
        label: buildLabel(scoreStage.postLabelTex, scoreStage.postLabelTex),
        dimensions: {
            rows: scoreStage.postRowCount,
            cols: scoreStage.postColumnCount
        },
        presentation: VIEW2D_MATRIX_PRESENTATIONS.GRID,
        rowItems: buildAttentionGridRowItems(scoreStage.postRows, buildSemantic(attentionSemantic, { stage: 'attention-post' })),
        visual: {
            styleKey: VIEW2D_STYLE_KEYS.MHSA_POST
        }
    });

    const softmaxPrefixNode = createGroupNode({
        role: 'attention-softmax-prefix',
        semantic: buildSemantic(attentionSemantic, { role: 'attention-softmax-prefix' }),
        direction: VIEW2D_LAYOUT_DIRECTIONS.HORIZONTAL,
        gapKey: 'inline',
        children: [
            createTextNode({
                role: 'attention-softmax-label',
                semantic: buildSemantic(attentionSemantic, { role: 'attention-softmax-label', focusKey: 'softmax' }),
                tex: scoreStage.softmaxLabelTex,
                text: 'softmax',
                presentation: VIEW2D_TEXT_PRESENTATIONS.LABEL,
                visual: { styleKey: VIEW2D_STYLE_KEYS.LABEL }
            }),
            createOperatorNode({
                role: 'attention-softmax-open',
                semantic: buildSemantic(attentionSemantic, { role: 'attention-softmax-open', operatorKey: 'open' }),
                text: '(',
                visual: { styleKey: VIEW2D_STYLE_KEYS.OPERATOR }
            })
        ]
    });

    const headOutputStageNode = Array.isArray(scoreStage.valueRows) && Array.isArray(scoreStage.headOutputRows)
        ? createGroupNode({
            role: 'attention-head-output-stage',
            semantic: buildSemantic(attentionSemantic, { role: 'attention-head-output-stage', stage: 'head-output' }),
            direction: VIEW2D_LAYOUT_DIRECTIONS.HORIZONTAL,
            gapKey: 'head-output',
            children: [
                createMatrixNode({
                    role: 'attention-post-copy',
                    semantic: buildSemantic(attentionSemantic, { role: 'attention-post-copy', stage: 'head-output' }),
                    label: buildLabel(scoreStage.postLabelTex, scoreStage.postLabelTex),
                    dimensions: {
                        rows: scoreStage.postRowCount,
                        cols: scoreStage.postColumnCount
                    },
                    presentation: VIEW2D_MATRIX_PRESENTATIONS.GRID,
                    rowItems: buildAttentionGridRowItems(
                        scoreStage.postRows,
                        buildSemantic(attentionSemantic, { stage: 'attention-post-copy' })
                    ),
                    visual: {
                        styleKey: VIEW2D_STYLE_KEYS.MHSA_POST
                    }
                }),
                createOperatorNode({
                    role: 'attention-head-output-multiply',
                    semantic: buildSemantic(attentionSemantic, { role: 'attention-head-output-multiply', operatorKey: 'multiply' }),
                    text: 'x',
                    visual: { styleKey: VIEW2D_STYLE_KEYS.OPERATOR }
                }),
                createMatrixNode({
                    role: 'attention-value-post',
                    semantic: buildSemantic(attentionSemantic, { role: 'attention-value-post', stage: 'head-output' }),
                    label: buildLabel(scoreStage.valueLabelTex, scoreStage.valueLabelTex),
                    dimensions: {
                        rows: scoreStage.valueRowCount,
                        cols: scoreStage.valueColumnCount
                    },
                    presentation: VIEW2D_MATRIX_PRESENTATIONS.COMPACT_ROWS,
                    rowItems: buildGradientRowItems(scoreStage.valueRows, buildSemantic(attentionSemantic, {
                        stage: 'attention-value-post'
                    }), 'attention-value-post-row'),
                    visual: {
                        styleKey: VIEW2D_STYLE_KEYS.MHSA_V
                    },
                    metadata: {
                        stageIndex: valueStageIndex,
                        ...createView2dVectorStripMetadata()
                    }
                }),
                createOperatorNode({
                    role: 'attention-head-output-equals',
                    semantic: buildSemantic(attentionSemantic, { role: 'attention-head-output-equals', operatorKey: 'equals' }),
                    text: '=',
                    visual: { styleKey: VIEW2D_STYLE_KEYS.OPERATOR }
                }),
                createMatrixNode({
                    role: 'attention-head-output',
                    semantic: buildSemantic(attentionSemantic, { role: 'attention-head-output', stage: 'head-output' }),
                    label: buildLabel(scoreStage.headOutputLabelTex, scoreStage.headOutputLabelTex),
                    dimensions: {
                        rows: scoreStage.headOutputRowCount,
                        cols: scoreStage.headOutputColumnCount
                    },
                    presentation: VIEW2D_MATRIX_PRESENTATIONS.COMPACT_ROWS,
                    rowItems: buildGradientRowItems(
                        scoreStage.headOutputRows,
                        buildSemantic(attentionSemantic, { stage: 'attention-head-output' }),
                        'attention-head-output-row'
                    ),
                    visual: {
                        styleKey: VIEW2D_STYLE_KEYS.MHSA_HEAD_OUTPUT
                    },
                    metadata: createView2dVectorStripMetadata()
                })
            ]
        })
        : null;

    const softmaxFlowChildren = [
        softmaxPrefixNode,
        maskedInputNode,
        createOperatorNode({
            role: 'attention-softmax-plus',
            semantic: buildSemantic(attentionSemantic, { role: 'attention-softmax-plus', operatorKey: 'plus' }),
            text: '+',
            visual: { styleKey: VIEW2D_STYLE_KEYS.OPERATOR }
        }),
        maskNode,
        createOperatorNode({
            role: 'attention-softmax-close',
            semantic: buildSemantic(attentionSemantic, { role: 'attention-softmax-close', operatorKey: 'close' }),
            text: ')',
            visual: { styleKey: VIEW2D_STYLE_KEYS.OPERATOR }
        }),
        createOperatorNode({
            role: 'attention-softmax-equals',
            semantic: buildSemantic(attentionSemantic, { role: 'attention-softmax-equals', operatorKey: 'equals' }),
            text: '=',
            visual: { styleKey: VIEW2D_STYLE_KEYS.OPERATOR }
        }),
        postNode
    ];
    if (headOutputStageNode) {
        softmaxFlowChildren.push(headOutputStageNode);
    }

    const softmaxStageNode = createGroupNode({
        role: 'attention-softmax-stage',
        semantic: buildSemantic(attentionSemantic, { role: 'attention-softmax-stage' }),
        direction: VIEW2D_LAYOUT_DIRECTIONS.VERTICAL,
        gapKey: 'softmax',
        children: [
            preScoreNode,
            createGroupNode({
                role: 'attention-softmax-flow',
                semantic: buildSemantic(attentionSemantic, { role: 'attention-softmax-flow' }),
                direction: VIEW2D_LAYOUT_DIRECTIONS.HORIZONTAL,
                gapKey: 'softmax',
                children: softmaxFlowChildren
            })
        ]
    });

    return createGroupNode({
        role: 'attention-stage',
        semantic: buildSemantic(attentionSemantic, { role: 'attention-stage' }),
        direction: VIEW2D_LAYOUT_DIRECTIONS.HORIZONTAL,
        gapKey: 'attention',
        children: [
            createOperatorNode({
                role: 'attention-open',
                semantic: buildSemantic(attentionSemantic, { role: 'attention-open', operatorKey: 'open' }),
                text: '(',
                visual: { styleKey: VIEW2D_STYLE_KEYS.OPERATOR }
            }),
            querySourceNode,
            createOperatorNode({
                role: 'attention-multiply',
                semantic: buildSemantic(attentionSemantic, { role: 'attention-multiply', operatorKey: 'multiply' }),
                text: 'x',
                visual: { styleKey: VIEW2D_STYLE_KEYS.OPERATOR }
            }),
            transposeNode,
            createOperatorNode({
                role: 'attention-close',
                semantic: buildSemantic(attentionSemantic, { role: 'attention-close', operatorKey: 'close' }),
                text: ')',
                visual: { styleKey: VIEW2D_STYLE_KEYS.OPERATOR }
            }),
            createOperatorNode({
                role: 'attention-divide',
                semantic: buildSemantic(attentionSemantic, { role: 'attention-divide', operatorKey: 'divide' }),
                text: '/',
                visual: { styleKey: VIEW2D_STYLE_KEYS.OPERATOR }
            }),
            createTextNode({
                role: 'attention-scale',
                semantic: buildSemantic(attentionSemantic, { role: 'attention-scale', focusKey: 'scale' }),
                tex: scoreStage.scaleLabelTex,
                text: 'sqrt(d_h)',
                presentation: VIEW2D_TEXT_PRESENTATIONS.LABEL,
                visual: { styleKey: VIEW2D_STYLE_KEYS.LABEL }
            }),
            createOperatorNode({
                role: 'attention-equals',
                semantic: buildSemantic(attentionSemantic, { role: 'attention-equals', operatorKey: 'equals' }),
                text: '=',
                visual: { styleKey: VIEW2D_STYLE_KEYS.OPERATOR }
            }),
            softmaxStageNode
        ],
        metadata: {
            queryStageIndex,
            valueStageIndex
        }
    });
}

function buildProjectionIngressConnectorNodes({
    baseSemantic,
    layoutMetrics,
    projectionNodes,
    sourceAnchorNode
}) {
    const connectorGaps = layoutMetrics?.connectorGaps || {};
    const findProjectionInput = (kind) => projectionNodes.find((stageNode) => stageNode?.metadata?.kind === kind)
        ?.children?.[0] || null;

    return ['q', 'k', 'v'].map((kind) => {
        const targetNode = findProjectionInput(kind);
        if (!sourceAnchorNode || !targetNode) return null;
        return createConnectorNode({
            role: `connector-xln-${kind}`,
            semantic: buildSemantic(baseSemantic, {
                stage: `connector-xln-${kind}`,
                role: 'connector-xln',
                branchKey: kind
            }),
            source: createAnchorRef(sourceAnchorNode.id, VIEW2D_ANCHOR_SIDES.LEFT),
            target: createAnchorRef(targetNode.id, VIEW2D_ANCHOR_SIDES.LEFT),
            route: VIEW2D_CONNECTOR_ROUTES.HORIZONTAL,
            gap: 0,
            sourceGap: 0,
            targetGap: PROJECTION_XLN_TARGET_GAP,
            visual: {
                styleKey: VIEW2D_STYLE_KEYS.CONNECTOR_NEUTRAL,
                stroke: MHSA_CONNECTOR_STROKE
            },
            metadata: {
                preserveColor: true,
                strokeWidthScale: 0.88
            }
        });
    }).filter(Boolean);
}

function buildConnectorNodes({
    baseSemantic,
    layoutMetrics,
    projectionNodes,
    attentionNode
}) {
    const connectorGaps = layoutMetrics?.connectorGaps || {};
    const findProjectionOutput = (kind) => projectionNodes.find((stageNode) => stageNode?.metadata?.kind === kind)
        ?.children?.[2]?.children?.[4] || null;
    const findAttentionNode = (role) => {
        const queue = Array.isArray(attentionNode?.children) ? [...attentionNode.children] : [];
        while (queue.length) {
            const node = queue.shift();
            if (!node || typeof node !== 'object') continue;
            if (node.role === role) return node;
            if (Array.isArray(node.children) && node.children.length) {
                queue.unshift(...node.children);
            }
        }
        return null;
    };

    const qOutputNode = findProjectionOutput('q');
    const kOutputNode = findProjectionOutput('k');
    const vOutputNode = findProjectionOutput('v');
    const querySourceNode = findAttentionNode('attention-query-source');
    const transposeNode = findAttentionNode('attention-key-transpose');
    const preScoreNode = findAttentionNode('attention-pre-score');
    const maskedInputNode = findAttentionNode('attention-masked-input');
    const postNode = findAttentionNode('attention-post');
    const postCopyNode = findAttentionNode('attention-post-copy');
    const valuePostNode = findAttentionNode('attention-value-post');

    return [
        qOutputNode && querySourceNode
            ? createConnectorNode({
                role: 'connector-q',
                semantic: buildSemantic(baseSemantic, { stage: 'connector-q', role: 'connector-q' }),
                source: createAnchorRef(qOutputNode.id, VIEW2D_ANCHOR_SIDES.RIGHT),
                target: createAnchorRef(querySourceNode.id, VIEW2D_ANCHOR_SIDES.TOP),
                route: VIEW2D_CONNECTOR_ROUTES.HORIZONTAL,
                gap: connectorGaps.projection,
                gapKey: 'projection',
                visual: {
                    styleKey: VIEW2D_STYLE_KEYS.CONNECTOR_NEUTRAL,
                    stroke: MHSA_CONNECTOR_STROKE
                },
                metadata: {
                    preserveColor: true
                }
            })
            : null,
        kOutputNode && transposeNode
            ? createConnectorNode({
                role: 'connector-k',
                semantic: buildSemantic(baseSemantic, { stage: 'connector-k', role: 'connector-k' }),
                source: createAnchorRef(kOutputNode.id, VIEW2D_ANCHOR_SIDES.RIGHT),
                target: createAnchorRef(transposeNode.id, VIEW2D_ANCHOR_SIDES.BOTTOM),
                route: VIEW2D_CONNECTOR_ROUTES.HORIZONTAL,
                gap: connectorGaps.transpose,
                gapKey: 'transpose',
                visual: {
                    styleKey: VIEW2D_STYLE_KEYS.CONNECTOR_NEUTRAL,
                    stroke: MHSA_CONNECTOR_STROKE
                },
                metadata: {
                    preserveColor: true
                }
            })
            : null,
        preScoreNode && maskedInputNode
            ? createConnectorNode({
                role: 'connector-pre',
                semantic: buildSemantic(baseSemantic, { stage: 'connector-pre', role: 'connector-pre' }),
                source: createAnchorRef(preScoreNode.id, VIEW2D_ANCHOR_SIDES.BOTTOM),
                target: createAnchorRef(maskedInputNode.id, VIEW2D_ANCHOR_SIDES.TOP),
                route: VIEW2D_CONNECTOR_ROUTES.VERTICAL,
                gap: connectorGaps.pre,
                gapKey: 'pre',
                visual: {
                    styleKey: VIEW2D_STYLE_KEYS.CONNECTOR_NEUTRAL,
                    stroke: MHSA_CONNECTOR_STROKE
                },
                metadata: {
                    preserveColor: true
                }
            })
            : null,
        postNode && postCopyNode
            ? createConnectorNode({
                role: 'connector-post',
                semantic: buildSemantic(baseSemantic, { stage: 'connector-post', role: 'connector-post' }),
                source: createAnchorRef(postNode.id, VIEW2D_ANCHOR_SIDES.RIGHT),
                target: createAnchorRef(postCopyNode.id, VIEW2D_ANCHOR_SIDES.LEFT),
                route: VIEW2D_CONNECTOR_ROUTES.HORIZONTAL,
                gap: connectorGaps.post,
                gapKey: 'post',
                visual: {
                    styleKey: VIEW2D_STYLE_KEYS.CONNECTOR_NEUTRAL,
                    stroke: MHSA_CONNECTOR_STROKE
                },
                metadata: {
                    preserveColor: true
                }
            })
            : null,
        vOutputNode && valuePostNode
            ? createConnectorNode({
                role: 'connector-v',
                semantic: buildSemantic(baseSemantic, { stage: 'connector-v', role: 'connector-v' }),
                source: createAnchorRef(vOutputNode.id, VIEW2D_ANCHOR_SIDES.RIGHT),
                target: createAnchorRef(valuePostNode.id, VIEW2D_ANCHOR_SIDES.BOTTOM),
                route: VIEW2D_CONNECTOR_ROUTES.HORIZONTAL,
                gap: connectorGaps.value,
                gapKey: 'value',
                visual: {
                    styleKey: VIEW2D_STYLE_KEYS.CONNECTOR_NEUTRAL,
                    stroke: MHSA_CONNECTOR_STROKE
                },
                metadata: {
                    preserveColor: true
                }
            })
            : null
    ].filter(Boolean);
}

export function buildMhsaSceneModel({
    previewData = null,
    activationSource = null,
    layerIndex = null,
    headIndex = null,
    sampleStep = 64,
    tokenIndices = null,
    tokenLabels = null,
    isSmallScreen = false,
    layoutMetrics = null,
    visualTokens = null
} = {}) {
    const resolvedPreviewData = previewData || buildMhsaTokenMatrixPreviewData({
        activationSource,
        layerIndex: normalizeIndex(layerIndex) ?? undefined,
        headIndex: normalizeIndex(headIndex) ?? undefined,
        sampleStep,
        tokenIndices,
        tokenLabels
    });
    if (
        !resolvedPreviewData?.rowCount
        || !resolvedPreviewData?.columnCount
        || !Array.isArray(resolvedPreviewData.rows)
        || !Array.isArray(resolvedPreviewData.projections)
        || !resolvedPreviewData.projections.length
    ) {
        return null;
    }

    const resolvedLayerIndex = normalizeIndex(layerIndex);
    const resolvedHeadIndex = normalizeIndex(headIndex);
    const baseSemantic = {
        componentKind: 'mhsa',
        layerIndex: resolvedLayerIndex,
        headIndex: resolvedHeadIndex
    };
    const resolvedLayoutMetrics = layoutMetrics || resolveMhsaTokenMatrixLayoutMetrics({
        rowCount: resolvedPreviewData.rowCount,
        isSmallScreen
    });
    const resolvedTokens = visualTokens || resolveView2dVisualTokens();
    const queryStageIndex = Math.max(
        0,
        resolvedPreviewData.projections.findIndex((projectionData) => String(projectionData?.kind || '').toLowerCase() === 'q')
    );
    const valueStageIndex = Math.max(
        0,
        resolvedPreviewData.projections.findIndex((projectionData) => String(projectionData?.kind || '').toLowerCase() === 'v')
    );

    const projectionNodes = resolvedPreviewData.projections
        .map((projectionData, stageIndex) => {
            const validProjection = projectionData
                && projectionData.weightRowCount
                && projectionData.weightColumnCount
                && projectionData.outputRowCount
                && projectionData.outputColumnCount
                && Array.isArray(projectionData.outputRows);
            if (!validProjection) return null;
            return buildProjectionStageNode({
                baseSemantic,
                previewData: resolvedPreviewData,
                projectionData,
                stageIndex,
                isSmallScreen
            });
        })
        .filter(Boolean);
    const projectionSourceAnchorNode = buildProjectionSourceAnchorNode({
        baseSemantic,
        rowCount: resolvedPreviewData.rowCount,
        isSmallScreen
    });
    const projectionStackNode = createGroupNode({
        role: 'projection-stack',
        semantic: buildSemantic(baseSemantic, { stage: 'projection-stack', role: 'projection-stack' }),
        direction: VIEW2D_LAYOUT_DIRECTIONS.VERTICAL,
        gapKey: 'stack',
        children: projectionNodes,
        metadata: {
            gapOverride: isSmallScreen ? PROJECTION_STACK_GAP_SMALL : PROJECTION_STACK_GAP
        }
    });
    const attentionStageLift = resolveAttentionStageLift(resolvedPreviewData.rowCount, isSmallScreen);

    const attentionNode = resolvedPreviewData.attentionScoreStage
        && Array.isArray(resolvedPreviewData.attentionScoreStage.queryRows)
        && Array.isArray(resolvedPreviewData.attentionScoreStage.outputRows)
        ? buildAttentionStageNode({
            baseSemantic,
            scoreStage: resolvedPreviewData.attentionScoreStage,
            queryStageIndex,
            valueStageIndex,
            isSmallScreen
        })
        : null;
    const attentionRootNode = attentionNode
        ? createGroupNode({
            role: 'attention-stage-positioner',
            semantic: buildSemantic(baseSemantic, {
                stage: 'attention-stage-positioner',
                role: 'attention-stage-positioner'
            }),
            direction: VIEW2D_LAYOUT_DIRECTIONS.VERTICAL,
            gapKey: 'default',
            children: [
                attentionNode,
                createHiddenSpacer({
                    semantic: buildSemantic(baseSemantic, {
                        stage: 'attention-stage-positioner',
                        role: 'attention-stage-shift-spacer'
                    }),
                    role: 'attention-stage-shift-spacer',
                    width: 1,
                    height: attentionStageLift * 2
                })
            ],
            metadata: {
                gapOverride: 0
            }
        })
        : null;

    const rootNodes = [
        createGroupNode({
            role: 'projection-sidecar',
            semantic: buildSemantic(baseSemantic, { stage: 'projection-sidecar', role: 'projection-sidecar' }),
            direction: VIEW2D_LAYOUT_DIRECTIONS.HORIZONTAL,
            gapKey: 'projection',
            children: [
                projectionSourceAnchorNode,
                projectionStackNode
            ],
            metadata: {
                gapOverride: isSmallScreen ? PROJECTION_SIDECAR_GAP_SMALL : PROJECTION_SIDECAR_GAP
            }
        })
    ];

    const connectorNodes = buildProjectionIngressConnectorNodes({
        baseSemantic,
        layoutMetrics: resolvedLayoutMetrics,
        projectionNodes,
        sourceAnchorNode: projectionSourceAnchorNode
    });
    if (attentionRootNode) {
        rootNodes.push(attentionRootNode);
        connectorNodes.push(...buildConnectorNodes({
            baseSemantic,
            layoutMetrics: resolvedLayoutMetrics,
            projectionNodes,
            attentionNode
        }));
    }

    if (connectorNodes.length) {
        rootNodes.push(
            createGroupNode({
                role: 'connector-layer',
                semantic: buildSemantic(baseSemantic, { stage: 'connector-layer', role: 'connector-layer' }),
                direction: VIEW2D_LAYOUT_DIRECTIONS.OVERLAY,
                gapKey: 'default',
                children: connectorNodes
            })
        );
    }

    return createSceneModel({
        semantic: buildSemantic(baseSemantic, { role: 'scene' }),
        nodes: rootNodes,
        metadata: {
            visualContract: 'selection-panel-mhsa-v1',
            source: 'selectionPanelMhsaTokenMatrixUtils',
            rowCount: resolvedPreviewData.rowCount,
            columnCount: resolvedPreviewData.columnCount,
            bandCount: resolvedPreviewData.bandCount,
            sampleStep: resolvedPreviewData.sampleStep,
            layoutMetrics: resolvedLayoutMetrics,
            tokens: resolvedTokens
        }
    });
}
