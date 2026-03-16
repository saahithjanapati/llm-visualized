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
import {
    resolveMhsaDimensionVisualExtent,
    resolveMhsaTokenVisualExtent
} from '../shared/mhsaDimensionSizing.js';

const PROJECTION_STACK_GAP = 188;
const PROJECTION_STACK_GAP_SMALL = 148;
const PROJECTION_STACK_GAP_PER_EXTRA_ROW = 10;
const PROJECTION_STACK_GAP_PER_EXTRA_ROW_SMALL = 8;
const PROJECTION_STACK_GAP_MAX_EXTRA = 80;
const PROJECTION_STACK_GAP_MAX_EXTRA_SMALL = 64;
const PROJECTION_SIDECAR_GAP = 84;
const PROJECTION_SIDECAR_GAP_SMALL = 64;
const ATTENTION_STAGE_SIDECAR_GAP = -12;
const ATTENTION_STAGE_SIDECAR_GAP_SMALL = -8;
const PROJECTION_STAGE_INLINE_GAP = 12;
const PROJECTION_STAGE_INLINE_GAP_SMALL = 10;
const PROJECTION_EQUATION_GAP = 12;
const PROJECTION_EQUATION_GAP_SMALL = 10;
const PROJECTION_MULTIPLY_OPERATOR_SCALE = 0.82;
const ATTENTION_GROUPING_OPERATOR_SCALE = 1.28;
const PROJECTION_XLN_ROW_HEIGHT = 7;
const PROJECTION_XLN_ROW_HEIGHT_SMALL = 6;
const PROJECTION_WEIGHT_MIN_WIDTH = 52;
const PROJECTION_WEIGHT_MAX_WIDTH = 138;
const PROJECTION_WEIGHT_MIN_HEIGHT = 56;
const PROJECTION_WEIGHT_MAX_HEIGHT = 144;
const PROJECTION_WEIGHT_CAPTION_LABEL_SCALE_MIN = 1.12;
const PROJECTION_WEIGHT_CAPTION_LABEL_SCALE_MAX = 4.0;
const PROJECTION_WEIGHT_CAPTION_LABEL_SCALE_BOOST = 2.2;
const PROJECTION_STAGE_VISUAL_SCALE_MAX = 1.5;
const PROJECTION_STAGE_VISUAL_SCALE_MAX_SMALL = 1.34;
const PROJECTION_STAGE_WEIGHT_REFERENCE_SCALE = 1.08;
const PROJECTION_STAGE_BIAS_WIDTH_RATIO = 0.84;
const PROJECTION_STAGE_BIAS_HEIGHT_BLEND = 0.64;
const VECTOR_STRIP_ROW_HEIGHT_BLEND_PER_EXTRA_ROW = 0.08;
const VECTOR_STRIP_ROW_HEIGHT_BLEND_PER_EXTRA_ROW_SMALL = 0.07;
const VECTOR_STRIP_ROW_HEIGHT_MAX_BLEND = 0.56;
const VECTOR_STRIP_ROW_HEIGHT_MAX_BLEND_SMALL = 0.46;
const VECTOR_STRIP_MAX_ROW_HEIGHT_BOOST = 8;
const VECTOR_STRIP_MAX_ROW_HEIGHT_BOOST_SMALL = 6;
const VECTOR_STRIP_WIDTH_BOOST_PER_ROW_HEIGHT = 8;
const VECTOR_STRIP_WIDTH_BOOST_PER_ROW_HEIGHT_SMALL = 7;
const VECTOR_STRIP_MAX_WIDTH_BOOST = 64;
const VECTOR_STRIP_MAX_WIDTH_BOOST_SMALL = 48;
const PROJECTION_BIAS_ROW_HEIGHT = 14;
const PROJECTION_BIAS_CAPTION_MIN_SCREEN_HEIGHT = 12;
const PROJECTION_BIAS_LABEL_MIN_SCREEN_FONT_PX = 15;
const PROJECTION_KEY_BIAS_LABEL_MIN_SCREEN_FONT_PX = 18;
const PROJECTION_BIAS_CORNER_RADIUS = 5;
const PROJECTION_BIAS_CAPTION_LABEL_SCALE = 1.8;
const PROJECTION_QUERY_BIAS_CAPTION_LABEL_SCALE = 2.0;
const PROJECTION_KEY_BIAS_CAPTION_LABEL_SCALE = 2.75;
const PROJECTION_OUTPUT_COMPACT_WIDTH = 72;
const PROJECTION_OUTPUT_COMPACT_WIDTH_SMALL = 62;
const PROJECTION_OUTPUT_ROW_HEIGHT = 7;
const PROJECTION_OUTPUT_ROW_HEIGHT_SMALL = 6;
const PROJECTION_XLN_TARGET_GAP = 8;
const ATTENTION_TRANSPOSE_CORNER_RADIUS = 8;
const ATTENTION_QKT_GROUP_GAP = 12;
const ATTENTION_QKT_GROUP_GAP_SMALL = 10;
const ATTENTION_CLOSE_DIVISOR_GAP = 8;
const ATTENTION_CLOSE_DIVISOR_GAP_SMALL = 6;
const ATTENTION_DIVISOR_GAP = -16;
const ATTENTION_DIVISOR_GAP_SMALL = -13;
const ATTENTION_SCALE_TEXT_OFFSET_X = -8;
const ATTENTION_SCALE_TEXT_OFFSET_X_SMALL = -6;
const ATTENTION_RESULT_CLUSTER_GAP = -14;
const ATTENTION_RESULT_CLUSTER_GAP_SMALL = -11;
const ATTENTION_SOFTMAX_PERSISTENT_FONT_PX = 16;
const ATTENTION_SOFTMAX_PERSISTENT_FONT_PX_SMALL = 15;
const ATTENTION_SOFTMAX_ZOOMED_OUT_FONT_PX = 17;
const ATTENTION_SOFTMAX_ZOOMED_OUT_FONT_PX_SMALL = 16;
const ATTENTION_SOFTMAX_GROUPING_OPERATOR_SCALE = 1.34;
const ATTENTION_SOFTMAX_PREFIX_VISUAL_HEIGHT_ESTIMATE = 42;
const ATTENTION_SOFTMAX_PREFIX_VISUAL_HEIGHT_ESTIMATE_SMALL = 38;
const ATTENTION_RESULT_STAGE_GAP = 36;
const ATTENTION_RESULT_STAGE_GAP_SMALL = 28;
const ATTENTION_RESULT_GROUP_GAP = 8;
const ATTENTION_RESULT_GROUP_GAP_SMALL = 6;
const ATTENTION_SOFTMAX_PREFIX_GAP = 6;
const ATTENTION_SOFTMAX_PREFIX_GAP_SMALL = 4;
const ATTENTION_SOFTMAX_BODY_GAP = 4;
const ATTENTION_SOFTMAX_BODY_GAP_SMALL = 3;
const ATTENTION_SOFTMAX_CLOSE_EQUALS_GAP = 4;
const ATTENTION_SOFTMAX_CLOSE_EQUALS_GAP_SMALL = 3;
const ATTENTION_SOFTMAX_FLOW_GAP = 88;
const ATTENTION_SOFTMAX_FLOW_GAP_SMALL = 66;
const ATTENTION_SOFTMAX_ROW_OFFSET = 48;
const ATTENTION_SOFTMAX_ROW_OFFSET_SMALL = 38;
const ATTENTION_SOFTMAX_RESULT_ALIGNMENT_OFFSET = 36;
const ATTENTION_SOFTMAX_RESULT_ALIGNMENT_OFFSET_SMALL = 29;
const ATTENTION_POST_COPY_STAGE_SPACER = 48;
const ATTENTION_POST_COPY_STAGE_SPACER_SMALL = 36;
const ATTENTION_HEAD_OUTPUT_GAP = 10;
const ATTENTION_HEAD_OUTPUT_GAP_SMALL = 8;
const ATTENTION_HEAD_OUTPUT_PRODUCT_GAP = 6;
const ATTENTION_HEAD_OUTPUT_PRODUCT_GAP_SMALL = 4;
const ATTENTION_MATRIX_CAPTION_LABEL_MIN_SCREEN_FONT_PX = 14;
const ATTENTION_MATRIX_CAPTION_DIMENSIONS_MIN_SCREEN_FONT_PX = 11;
const ATTENTION_PRE_SCORE_CAPTION_LABEL_SCALE = 0.82;
const ATTENTION_MASK_CAPTION_LABEL_SCALE = 0.82;
const ATTENTION_GRID_PADDING = 4;
const ATTENTION_GRID_CARD_CORNER_RADIUS = 12;
const ATTENTION_GRID_CELL_CORNER_RADIUS_SCALE = 0.9;
const ATTENTION_SOFTMAX_TEXT_FONT_SCALE = 1.28;
const ATTENTION_SCALE_TEXT_FONT_SCALE = 1.4;
const ATTENTION_STAGE_VERTICAL_LIFT = 144;
const ATTENTION_STAGE_VERTICAL_LIFT_SMALL = 110;
const ATTENTION_STAGE_VERTICAL_LIFT_PER_EXTRA_ROW = 6;
const ATTENTION_STAGE_VERTICAL_LIFT_PER_EXTRA_ROW_SMALL = 5;
const ATTENTION_CONNECTOR_CAPTION_EXIT_GAP = 4;
const ATTENTION_PRE_CONNECTOR_SOURCE_OFFSET_Y = 16;
const ATTENTION_VALUE_CONNECTOR_SOURCE_OFFSET_Y = 0;
const ATTENTION_VALUE_CONNECTOR_SOURCE_GAP = 8;
const ATTENTION_VALUE_CONNECTOR_TARGET_GAP = 8;
const MHSA_CONNECTOR_STROKE = 'rgba(255, 255, 255, 0.84)';
const MHSA_SYMBOL_CAPTION_LABEL_SCALE = 0.9;
const MHSA_WEIGHT_CAPTION_LABEL_SCALE_FACTOR = 0.72;
const MHSA_WEIGHT_CAPTION_LABEL_SCALE_MIN = 1.02;
const ATTENTION_SOFTMAX_CORE_FLOW_GAP_BOOST_RATIO = 0.15;
const ATTENTION_SOFTMAX_CORE_FLOW_GAP_BOOST_MAX = 10;
const ATTENTION_LEFT_HAND_SIDE_GAP_BASE = 10;
const ATTENTION_LEFT_HAND_SIDE_GAP_BASE_SMALL = 8;
const ATTENTION_LEFT_HAND_SIDE_GAP_BOOST_RATIO = 0.1;
const ATTENTION_LEFT_HAND_SIDE_GAP_BOOST_MAX = 8;
const INCOMING_ARROW_SPACER_WIDTH = 60;
const INCOMING_ARROW_SPACER_WIDTH_SMALL = 52;
const OUTGOING_ARROW_SPACER_WIDTH = 44;
const OUTGOING_ARROW_SPACER_WIDTH_SMALL = 38;
const EDGE_CONNECTOR_STROKE_WIDTH_SCALE = 0.88;
const HEAD_OUTPUT_EDGE_SOURCE_GAP = 8;
const KV_CACHE_BRANCH_OPACITY = 0.4;
const KV_CACHE_BRANCH_STROKE = 'rgba(255, 255, 255, 0.34)';
const KV_CACHE_BRANCH_STROKE_WIDTH_SCALE = 0.74;
const KV_CACHE_BRANCH_LIFT = 68;
const KV_CACHE_BRANCH_LIFT_SMALL = 56;
const KV_CACHE_BRANCH_OFFSET_X = 74;
const KV_CACHE_BRANCH_OFFSET_X_SMALL = 56;

function normalizeIndex(value) {
    return Number.isFinite(value) ? Math.floor(value) : null;
}

function hasNumericValue(value) {
    return typeof value === 'number' && !Number.isNaN(value);
}

function buildSemantic(baseSemantic, extra = {}) {
    return {
        ...baseSemantic,
        ...extra
    };
}

function resolveAttentionSoftmaxCoreFlowGap(layoutMetrics = null, isSmallScreen = false) {
    const baseGap = isSmallScreen ? 14 : 18;
    const rawBoost = layoutMetrics?.cssVars?.['--mhsa-token-matrix-softmax-stage-gap-boost'];
    const boost = typeof rawBoost === 'string'
        ? Number.parseFloat(rawBoost)
        : (Number.isFinite(rawBoost) ? Number(rawBoost) : 0);
    const safeBoost = Number.isFinite(boost) ? Math.max(0, boost) : 0;
    const resolvedBoost = Math.min(
        ATTENTION_SOFTMAX_CORE_FLOW_GAP_BOOST_MAX,
        Math.round(safeBoost * ATTENTION_SOFTMAX_CORE_FLOW_GAP_BOOST_RATIO)
    );
    return baseGap + resolvedBoost;
}

function resolveAttentionLeftHandSideGap(layoutMetrics = null, isSmallScreen = false) {
    const baseGap = isSmallScreen ? ATTENTION_LEFT_HAND_SIDE_GAP_BASE_SMALL : ATTENTION_LEFT_HAND_SIDE_GAP_BASE;
    const rawBoost = layoutMetrics?.cssVars?.['--mhsa-token-matrix-attention-flow-gap-boost'];
    const boost = typeof rawBoost === 'string'
        ? Number.parseFloat(rawBoost)
        : (Number.isFinite(rawBoost) ? Number(rawBoost) : 0);
    const safeBoost = Number.isFinite(boost) ? Math.max(0, boost) : 0;
    const resolvedBoost = Math.min(
        ATTENTION_LEFT_HAND_SIDE_GAP_BOOST_MAX,
        Math.round(safeBoost * ATTENTION_LEFT_HAND_SIDE_GAP_BOOST_RATIO)
    );
    return baseGap + resolvedBoost;
}

function findNestedNodeByRole(rootNode = null, role = '') {
    const safeRole = String(role || '').trim();
    if (!rootNode || !safeRole.length) return null;
    const queue = Array.isArray(rootNode?.children) ? [...rootNode.children] : [];
    while (queue.length) {
        const node = queue.shift();
        if (!node || typeof node !== 'object') continue;
        if (node.role === safeRole) {
            return node;
        }
        if (Array.isArray(node.children) && node.children.length) {
            queue.unshift(...node.children);
        }
    }
    return null;
}

function buildLabel(labelTex = '', fallbackText = '') {
    return {
        tex: typeof labelTex === 'string' ? labelTex : '',
        text: typeof fallbackText === 'string' && fallbackText.length
            ? fallbackText
            : (typeof labelTex === 'string' ? labelTex : '')
    };
}

function createMhsaOperatorNode({
    metadata = null,
    ...rest
} = {}) {
    return createOperatorNode({
        ...rest,
        metadata: {
            ...(metadata && typeof metadata === 'object' ? metadata : {}),
            renderMode: 'dom-katex'
        }
    });
}

function buildMhsaProjectionSubscriptLabel(labelTex = '') {
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

function createPersistentAttentionGridMetadata({
    paddingX = ATTENTION_GRID_PADDING,
    paddingY = ATTENTION_GRID_PADDING,
    cellCornerRadiusScale = ATTENTION_GRID_CELL_CORNER_RADIUS_SCALE
} = {}) {
    return {
        grid: {
            preserveDetail: true,
            ...(Number.isFinite(paddingX) && paddingX >= 0
                ? { paddingX: Math.max(0, Math.floor(paddingX)) }
                : {}),
            ...(Number.isFinite(paddingY) && paddingY >= 0
                ? { paddingY: Math.max(0, Math.floor(paddingY)) }
                : {}),
            ...(Number.isFinite(cellCornerRadiusScale) && cellCornerRadiusScale >= 0
                ? { cellCornerRadiusScale: Math.max(0, Number(cellCornerRadiusScale)) }
                : {})
        },
        ...createCardMetadata(null, null, {
            cornerRadius: ATTENTION_GRID_CARD_CORNER_RADIUS
        })
    };
}

function createCaptionMetadata({
    position = 'bottom',
    dimensionsTex = '',
    dimensionsText = '',
    minScreenHeightPx = 28,
    renderMode = 'dom-katex',
    labelScale = null,
    dimensionsScale = null,
    preferStandardSizing = false,
    labelMinScreenFontPx = null,
    dimensionsMinScreenFontPx = null
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
    if (Number.isFinite(labelScale) && labelScale > 0) {
        caption.labelScale = Number(labelScale);
    }
    if (Number.isFinite(dimensionsScale) && dimensionsScale > 0) {
        caption.dimensionsScale = Number(dimensionsScale);
    }
    if (preferStandardSizing === true) {
        caption.preferStandardSizing = true;
    }
    if (Number.isFinite(labelMinScreenFontPx) && labelMinScreenFontPx > 0) {
        caption.labelMinScreenFontPx = Number(labelMinScreenFontPx);
    }
    if (Number.isFinite(dimensionsMinScreenFontPx) && dimensionsMinScreenFontPx > 0) {
        caption.dimensionsMinScreenFontPx = Number(dimensionsMinScreenFontPx);
    }
    return Object.keys(caption).length ? { caption } : null;
}

function createAttentionMatrixCaptionMetadata(rows = 1, cols = 1, {
    position = 'bottom',
    minScreenHeightPx = 28,
    labelScale = null,
    dimensionsScale = null,
    labelMinScreenFontPx = ATTENTION_MATRIX_CAPTION_LABEL_MIN_SCREEN_FONT_PX,
    dimensionsMinScreenFontPx = ATTENTION_MATRIX_CAPTION_DIMENSIONS_MIN_SCREEN_FONT_PX
} = {}) {
    const dimensionCaption = formatView2dMatrixDimensions(rows, cols);
    return createCaptionMetadata({
        position,
        dimensionsTex: dimensionCaption.tex,
        dimensionsText: dimensionCaption.text,
        minScreenHeightPx,
        renderMode: 'dom-katex',
        labelScale,
        dimensionsScale,
        preferStandardSizing: true,
        labelMinScreenFontPx,
        dimensionsMinScreenFontPx
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
        metadata: createCardMetadata(width, height, {
            hidden: true,
            cornerRadius: 0
        })
    });
}

function resolveAttentionStageLift({
    rowCount = 1,
    extraRows = null,
    isSmallScreen = false
} = {}) {
    const safeRowCount = Number.isFinite(rowCount) ? Math.max(1, Math.floor(rowCount)) : 1;
    const resolvedExtraRows = Number.isFinite(extraRows)
        ? Math.max(0, Math.floor(extraRows))
        : Math.max(0, safeRowCount - 5);
    const baseLift = isSmallScreen ? ATTENTION_STAGE_VERTICAL_LIFT_SMALL : ATTENTION_STAGE_VERTICAL_LIFT;
    const perExtraRow = isSmallScreen
        ? ATTENTION_STAGE_VERTICAL_LIFT_PER_EXTRA_ROW_SMALL
        : ATTENTION_STAGE_VERTICAL_LIFT_PER_EXTRA_ROW;
    return Math.max(0, Math.round(baseLift + (resolvedExtraRows * perExtraRow)));
}

function resolveProjectionStackGap({
    rowCount = 1,
    extraRows = null,
    isSmallScreen = false
} = {}) {
    const safeRowCount = Number.isFinite(rowCount) ? Math.max(1, Math.floor(rowCount)) : 1;
    const resolvedExtraRows = Number.isFinite(extraRows)
        ? Math.max(0, Math.floor(extraRows))
        : Math.max(0, safeRowCount - 5);
    const baseGap = isSmallScreen ? PROJECTION_STACK_GAP_SMALL : PROJECTION_STACK_GAP;
    const perExtraRow = isSmallScreen
        ? PROJECTION_STACK_GAP_PER_EXTRA_ROW_SMALL
        : PROJECTION_STACK_GAP_PER_EXTRA_ROW;
    const maxExtra = isSmallScreen
        ? PROJECTION_STACK_GAP_MAX_EXTRA_SMALL
        : PROJECTION_STACK_GAP_MAX_EXTRA;
    return Math.max(
        0,
        Math.round(
            baseGap
            + Math.min(maxExtra, resolvedExtraRows * perExtraRow)
        )
    );
}

function resolveVectorStripDimensions({
    rowCount = 1,
    extraRows = null,
    baseCompactWidth = 72,
    baseRowHeight = 7,
    isSmallScreen = false,
    layoutMetrics = null
} = {}) {
    const safeRowCount = Number.isFinite(rowCount) ? Math.max(1, Math.floor(rowCount)) : 1;
    const safeBaseCompactWidth = Math.max(1, Math.floor(Number(baseCompactWidth) || 72));
    const safeBaseRowHeight = Math.max(1, Math.floor(Number(baseRowHeight) || 7));
    const rowTargetExtent = resolveMhsaTokenVisualExtent(safeRowCount, {
        isSmallScreen
    });
    const minRowHeight = safeRowCount >= 20 ? 2 : (safeRowCount >= 10 ? 3 : 4);
    const resolvedRowHeight = Math.max(
        minRowHeight,
        Math.min(
            safeBaseRowHeight,
            Math.floor(rowTargetExtent / safeRowCount)
        )
    );
    return {
        compactWidth: safeBaseCompactWidth,
        rowHeight: resolvedRowHeight
    };
}

function resolveProjectionStageVisualMetrics({
    previewRowCount = 1,
    previewColumnCount = 1,
    projectionData = null,
    extraRows = null,
    isSmallScreen = false,
    layoutMetrics = null
} = {}) {
    const previewFeatureExtent = resolveMhsaDimensionVisualExtent(previewColumnCount, {
        isSmallScreen
    });
    const outputFeatureExtent = resolveMhsaDimensionVisualExtent(projectionData?.outputColumnCount, {
        isSmallScreen
    });
    const xLnDimensions = resolveVectorStripDimensions({
        rowCount: previewRowCount,
        extraRows,
        baseCompactWidth: previewFeatureExtent,
        baseRowHeight: isSmallScreen ? PROJECTION_XLN_ROW_HEIGHT_SMALL : PROJECTION_XLN_ROW_HEIGHT,
        isSmallScreen,
        layoutMetrics
    });
    const outputDimensions = resolveVectorStripDimensions({
        rowCount: projectionData?.outputRowCount,
        extraRows,
        baseCompactWidth: outputFeatureExtent,
        baseRowHeight: isSmallScreen ? PROJECTION_OUTPUT_ROW_HEIGHT_SMALL : PROJECTION_OUTPUT_ROW_HEIGHT,
        isSmallScreen,
        layoutMetrics
    });
    const baseOutputCompactWidth = outputFeatureExtent;
    const baseOutputRowHeight = isSmallScreen ? PROJECTION_OUTPUT_ROW_HEIGHT_SMALL : PROJECTION_OUTPUT_ROW_HEIGHT;
    const projectionScale = Math.min(
        isSmallScreen ? PROJECTION_STAGE_VISUAL_SCALE_MAX_SMALL : PROJECTION_STAGE_VISUAL_SCALE_MAX,
        Math.max(
            1,
            outputDimensions.compactWidth / Math.max(1, baseOutputCompactWidth),
            outputDimensions.rowHeight / Math.max(1, baseOutputRowHeight)
        )
    );
    const weightReferenceExtent = Math.round(
        previewFeatureExtent * Math.max(1, projectionScale * PROJECTION_STAGE_WEIGHT_REFERENCE_SCALE)
    );
    const weightMinWidth = Math.round(PROJECTION_WEIGHT_MIN_WIDTH * Math.min(1.18, projectionScale));
    const weightMaxWidth = Math.round(PROJECTION_WEIGHT_MAX_WIDTH * projectionScale);
    const weightMinHeight = Math.round(PROJECTION_WEIGHT_MIN_HEIGHT * Math.min(1.12, projectionScale));
    const weightMaxHeight = Math.round(PROJECTION_WEIGHT_MAX_HEIGHT * projectionScale);
    const weightCardSize = resolveRelativeCardSize({
        rows: projectionData?.weightRowCount,
        cols: projectionData?.weightColumnCount,
        referenceCount: previewColumnCount,
        referenceExtent: weightReferenceExtent,
        minWidth: weightMinWidth,
        maxWidth: weightMaxWidth,
        minHeight: weightMinHeight,
        maxHeight: weightMaxHeight
    });
    weightCardSize.height = Math.max(
        weightMinHeight,
        Math.min(weightMaxHeight, xLnDimensions.compactWidth)
    );
    weightCardSize.width = Math.max(
        weightMinWidth,
        Math.min(weightMaxWidth, outputDimensions.compactWidth)
    );
    const biasCompactWidth = outputDimensions.compactWidth;
    const biasRowHeight = Math.max(
        PROJECTION_BIAS_ROW_HEIGHT,
        Math.round(
            PROJECTION_BIAS_ROW_HEIGHT
            * Math.min(1.28, 1 + ((projectionScale - 1) * PROJECTION_STAGE_BIAS_HEIGHT_BLEND))
        )
    );

    return {
        xLnDimensions,
        outputDimensions,
        weightCardSize,
        biasCompactWidth,
        biasRowHeight,
        projectionScale
    };
}

function resolveAttentionGridLayoutMetrics(layoutMetrics = null, isSmallScreen = false) {
    const componentOverrides = layoutMetrics?.componentOverrides || {};
    const cellSize = Number.isFinite(componentOverrides.gridCellSize) && componentOverrides.gridCellSize > 0
        ? Math.max(1, Math.floor(componentOverrides.gridCellSize))
        : (isSmallScreen ? 9 : 10);
    const cellGap = Number.isFinite(componentOverrides.gridCellGap) && componentOverrides.gridCellGap >= 0
        ? Math.max(0, Math.floor(componentOverrides.gridCellGap))
        : 2;
    const gridPaddingX = Number.isFinite(componentOverrides.gridPaddingX) && componentOverrides.gridPaddingX >= 0
        ? Math.max(0, Math.floor(componentOverrides.gridPaddingX))
        : (isSmallScreen ? 8 : 10);
    const gridPaddingY = Number.isFinite(componentOverrides.gridPaddingY) && componentOverrides.gridPaddingY >= 0
        ? Math.max(0, Math.floor(componentOverrides.gridPaddingY))
        : (isSmallScreen ? 8 : 10);
    return {
        cellSize,
        cellGap,
        contentPaddingX: gridPaddingX,
        contentPaddingY: gridPaddingY
    };
}

function resolveAttentionFlowRowHeight(rowCount = 1, isSmallScreen = false, layoutMetrics = null) {
    const safeRowCount = Number.isFinite(rowCount) ? Math.max(1, Math.floor(rowCount)) : 1;
    const {
        cellSize,
        cellGap,
        contentPaddingX,
        contentPaddingY
    } = resolveAttentionGridLayoutMetrics(layoutMetrics, isSmallScreen);
    return Math.max(
        (safeRowCount * cellSize)
        + (Math.max(0, safeRowCount - 1) * cellGap)
        + (contentPaddingY * 2),
        (safeRowCount * cellSize)
        + (Math.max(0, safeRowCount - 1) * cellGap)
        + (contentPaddingX * 2)
    );
}

function resolveAttentionCaptionBlockHeight(isSmallScreen = false) {
    const captionGap = isSmallScreen ? 8 : 10;
    const captionLineHeight = isSmallScreen ? 12 : 14;
    return captionGap + (captionLineHeight * 2);
}

function resolveAttentionMatrixBlockHeight(rowCount = 1, isSmallScreen = false, layoutMetrics = null) {
    return resolveAttentionFlowRowHeight(rowCount, isSmallScreen, layoutMetrics)
        + resolveAttentionCaptionBlockHeight(isSmallScreen);
}

function resolveAttentionSoftmaxPrefixOffset(rowCount = 1, isSmallScreen = false, layoutMetrics = null) {
    const flowHeight = resolveAttentionFlowRowHeight(rowCount, isSmallScreen, layoutMetrics);
    const prefixVisualHeightEstimate = isSmallScreen
        ? ATTENTION_SOFTMAX_PREFIX_VISUAL_HEIGHT_ESTIMATE_SMALL
        : ATTENTION_SOFTMAX_PREFIX_VISUAL_HEIGHT_ESTIMATE;
    return Math.max(1, Math.round((flowHeight - prefixVisualHeightEstimate) * 0.5));
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
            stage: 'ln1.output',
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

function buildProjectionCacheRowItems(rows = [], baseSemantic = {}, branchKey = '') {
    return rows.map((rowData) => {
        const semantic = {
            componentKind: 'mhsa',
            layerIndex: baseSemantic?.layerIndex,
            headIndex: baseSemantic?.headIndex,
            stage: `kv-cache.${branchKey}`,
            role: 'projection-cache-row',
            rowIndex: rowData.rowIndex,
            tokenIndex: rowData.tokenIndex,
            ...(branchKey ? { branchKey } : {}),
            cacheKind: branchKey
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
            rowIndex: rowData.rowIndex,
            tokenIndex: rowData.tokenIndex
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
                        queryTokenIndex: normalizeIndex(cellData.queryTokenIndex),
                        keyTokenIndex: normalizeIndex(cellData.keyTokenIndex),
                        queryTokenLabel: typeof cellData.queryTokenLabel === 'string'
                            ? cellData.queryTokenLabel
                            : (cellData.rowTokenLabel || rowData.tokenLabel || ''),
                        keyTokenLabel: typeof cellData.keyTokenLabel === 'string'
                            ? cellData.keyTokenLabel
                            : (cellData.colTokenLabel || ''),
                        preScore: hasNumericValue(cellData.preScore) ? cellData.preScore : null,
                        postScore: hasNumericValue(cellData.postScore) ? cellData.postScore : null,
                        maskValue: hasNumericValue(cellData.maskValue) ? cellData.maskValue : null,
                        rawValue: hasNumericValue(cellData.rawValue) ? cellData.rawValue : null,
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
            tokenIndex: columnData.tokenIndex ?? columnData.colIndex,
            branchKey: 'k'
        });
        return {
            id: buildSceneNodeId(semantic),
            index: columnData.colIndex,
            label: columnData.tokenLabel || `Token ${columnData.colIndex + 1}`,
            semantic,
            rawValue: Number.isFinite(columnData.rawValue) ? columnData.rawValue : null,
            rawValues: Array.isArray(columnData.rawValues) ? [...columnData.rawValues] : null,
            fillCss: columnData.fillCss || 'transparent',
            title: typeof columnData.tokenLabel === 'string' ? columnData.tokenLabel : null
        };
    });
}

function resolveTransposeStripLayout({
    targetHeight = PROJECTION_OUTPUT_COMPACT_WIDTH,
    columnUnit = PROJECTION_OUTPUT_ROW_HEIGHT
} = {}) {
    const resolvedTargetHeight = Math.max(1, Math.floor(targetHeight || PROJECTION_OUTPUT_COMPACT_WIDTH));
    const resolvedColumnUnit = Math.max(1, Math.floor(columnUnit || PROJECTION_OUTPUT_ROW_HEIGHT));
    return {
        colWidth: resolvedColumnUnit,
        colGap: 0,
        colHeight: resolvedTargetHeight,
        paddingX: 0,
        paddingY: 0
    };
}

function buildProjectionSourceNode({
    baseSemantic,
    previewData,
    rowCount = 1,
    extraRows = null,
    isSmallScreen = false,
    layoutMetrics = null
}) {
    const dimensionCaption = formatView2dMatrixDimensions(
        rowCount,
        previewData?.columnCount
    );
    const previewFeatureExtent = resolveMhsaDimensionVisualExtent(previewData?.columnCount, {
        isSmallScreen
    });
    const xLnDimensions = resolveVectorStripDimensions({
        rowCount,
        extraRows,
        baseCompactWidth: previewFeatureExtent,
        baseRowHeight: isSmallScreen ? PROJECTION_XLN_ROW_HEIGHT_SMALL : PROJECTION_XLN_ROW_HEIGHT,
        isSmallScreen,
        layoutMetrics
    });
    return createVectorStripMatrixNode({
        role: 'projection-source-xln',
        semantic: buildSemantic(baseSemantic, {
            stage: 'projection-source',
            role: 'projection-source-xln'
        }),
        labelTex: 'x_{\\ln}',
        labelText: 'x_ln',
        rowItems: buildProjectionInputRowItems(previewData?.rows, baseSemantic),
        rowCount,
        columnCount: previewData?.columnCount,
        compactWidth: xLnDimensions.compactWidth,
        rowHeight: xLnDimensions.rowHeight,
        captionPosition: 'bottom',
        captionLabelScale: MHSA_SYMBOL_CAPTION_LABEL_SCALE,
        captionDimensionsTex: dimensionCaption.tex,
        captionDimensionsText: dimensionCaption.text,
        visualStyleKey: VIEW2D_STYLE_KEYS.RESIDUAL,
        metadata: {
            disableEdgeOrnament: true
        }
    });
}

function normalizeKvCacheState(kvCacheState = null) {
    const kvCacheModeEnabled = !!kvCacheState?.kvCacheModeEnabled;
    const kvCachePrefillActive = !!(kvCacheModeEnabled && kvCacheState?.kvCachePrefillActive);
    return {
        kvCacheModeEnabled,
        kvCachePrefillActive
    };
}

function shouldShowProjectionCacheBranch(projectionKind = '', kvCacheState = null) {
    const safeKind = String(projectionKind || '').trim().toLowerCase();
    if (safeKind !== 'k' && safeKind !== 'v') return false;
    return !!(kvCacheState?.kvCacheModeEnabled && kvCacheState?.kvCachePrefillActive);
}

function resolveProjectionCacheBranchMetrics({
    compactWidth = 0,
    rowCount = 1,
    rowHeight = 1,
    isSmallScreen = false
} = {}) {
    const safeCompactWidth = Math.max(1, Math.floor(Number(compactWidth) || 0));
    const safeRowCount = Math.max(1, Math.floor(Number(rowCount) || 1));
    const safeRowHeight = Math.max(1, Math.floor(Number(rowHeight) || 1));
    const contentHeight = safeRowCount * safeRowHeight;
    return {
        offsetX: Math.max(
            isSmallScreen ? KV_CACHE_BRANCH_OFFSET_X_SMALL : KV_CACHE_BRANCH_OFFSET_X,
            Math.round(safeCompactWidth * (isSmallScreen ? 0.03 : 0.04))
        ),
        lift: Math.max(
            isSmallScreen ? KV_CACHE_BRANCH_LIFT_SMALL : KV_CACHE_BRANCH_LIFT,
            Math.round(contentHeight * (isSmallScreen ? 1.08 : 1.18))
        )
    };
}

function buildProjectionCacheBranch({
    baseSemantic,
    projectionData,
    projectionKind = '',
    styleKey = VIEW2D_STYLE_KEYS.MHSA_K,
    outputNode = null,
    outputDimensions = null,
    isSmallScreen = false
} = {}) {
    if (!outputNode?.id) return null;
    const outputDimensionCaption = formatView2dMatrixDimensions(
        projectionData?.outputRowCount,
        projectionData?.outputColumnCount
    );
    const branchMetrics = resolveProjectionCacheBranchMetrics({
        compactWidth: outputDimensions?.compactWidth,
        rowCount: projectionData?.outputRowCount,
        rowHeight: outputDimensions?.rowHeight,
        isSmallScreen
    });
    const cacheSemantic = buildSemantic(baseSemantic, {
        stage: `kv-cache.${projectionKind}`,
        role: 'projection-cache',
        branchKey: projectionKind,
        cacheKind: projectionKind
    });
    const cacheLabel = String(projectionData?.outputLabelTex || projectionKind.toUpperCase() || '').trim()
        || projectionKind.toUpperCase();
    const cacheNode = createVectorStripMatrixNode({
        role: 'projection-cache',
        semantic: cacheSemantic,
        labelTex: `${cacheLabel}_{\\mathrm{cache}}`,
        labelText: `${cacheLabel}_cache`,
        rowItems: buildProjectionCacheRowItems(projectionData?.outputRows, baseSemantic, projectionKind),
        rowCount: projectionData?.outputRowCount,
        columnCount: projectionData?.outputColumnCount,
        compactWidth: outputDimensions?.compactWidth,
        rowHeight: outputDimensions?.rowHeight,
        captionPosition: 'bottom',
        captionLabelScale: MHSA_SYMBOL_CAPTION_LABEL_SCALE,
        captionDimensionsTex: outputDimensionCaption.tex,
        captionDimensionsText: outputDimensionCaption.text,
        visualStyleKey: styleKey,
        metadata: {
            kind: projectionKind,
            cacheNode: true
        }
    });
    cacheNode.visual = {
        ...(cacheNode.visual || {}),
        opacity: KV_CACHE_BRANCH_OPACITY
    };

    const cacheWrapperNode = createGroupNode({
        role: 'projection-cache-branch',
        semantic: buildSemantic(baseSemantic, {
            stage: `kv-cache.${projectionKind}`,
            role: 'projection-cache-branch',
            branchKey: projectionKind,
            cacheKind: projectionKind
        }),
        direction: VIEW2D_LAYOUT_DIRECTIONS.OVERLAY,
        gapKey: 'default',
        layout: {
            anchorAlign: {
                axis: 'x',
                selfNodeId: cacheNode.id,
                targetNodeId: outputNode.id,
                selfAnchor: VIEW2D_ANCHOR_SIDES.CENTER,
                targetAnchor: VIEW2D_ANCHOR_SIDES.CENTER,
                offset: -branchMetrics.offsetX
            }
        },
        children: [
            createGroupNode({
                role: 'projection-cache-branch-y-anchor',
                semantic: buildSemantic(baseSemantic, {
                    stage: `kv-cache.${projectionKind}`,
                    role: 'projection-cache-branch-y-anchor',
                    branchKey: projectionKind,
                    cacheKind: projectionKind
                }),
                direction: VIEW2D_LAYOUT_DIRECTIONS.OVERLAY,
                gapKey: 'default',
                layout: {
                    anchorAlign: {
                        axis: 'y',
                        selfNodeId: cacheNode.id,
                        targetNodeId: outputNode.id,
                        selfAnchor: VIEW2D_ANCHOR_SIDES.BOTTOM,
                        targetAnchor: VIEW2D_ANCHOR_SIDES.TOP,
                        offset: -branchMetrics.lift
                    }
                },
                children: [cacheNode],
                metadata: {
                    gapOverride: 0,
                    kind: projectionKind,
                    cacheNode: true
                }
            })
        ],
        metadata: {
            gapOverride: 0,
            kind: projectionKind,
            cacheNode: true
        }
    });

    const connectorNode = createConnectorNode({
        role: `connector-${projectionKind}-cache`,
        semantic: buildSemantic(baseSemantic, {
            stage: `connector-${projectionKind}-cache`,
            role: 'connector-kv-cache',
            branchKey: projectionKind,
            cacheKind: projectionKind
        }),
        source: createAnchorRef(outputNode.id, VIEW2D_ANCHOR_SIDES.TOP),
        target: createAnchorRef(cacheNode.id, VIEW2D_ANCHOR_SIDES.RIGHT),
        route: VIEW2D_CONNECTOR_ROUTES.ELBOW,
        gap: 0,
        sourceGap: 6,
        targetGap: 6,
        visual: {
            styleKey: VIEW2D_STYLE_KEYS.CONNECTOR_NEUTRAL,
            stroke: KV_CACHE_BRANCH_STROKE
        },
        metadata: {
            preserveColor: true,
            strokeWidthScale: KV_CACHE_BRANCH_STROKE_WIDTH_SCALE
        }
    });

    return {
        cacheNode,
        cacheOverlayNode: cacheWrapperNode,
        cacheConnectorNode: connectorNode
    };
}

function buildProjectionStageNode({
    baseSemantic,
    previewData,
    projectionData,
    stageIndex,
    extraRows = null,
    isSmallScreen = false,
    layoutMetrics = null,
    kvCacheState = null
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
    const projectionVisualMetrics = resolveProjectionStageVisualMetrics({
        previewRowCount: previewData.rowCount,
        previewColumnCount: previewData.columnCount,
        projectionData,
        extraRows,
        isSmallScreen,
        layoutMetrics
    });
    const projectionBiasCaptionLabelScale = projectionKind === 'q'
        ? PROJECTION_QUERY_BIAS_CAPTION_LABEL_SCALE
        : (projectionKind === 'k'
            ? PROJECTION_KEY_BIAS_CAPTION_LABEL_SCALE
            : PROJECTION_BIAS_CAPTION_LABEL_SCALE);
    const projectionBiasLabelMinScreenFontPx = projectionKind === 'k'
        ? PROJECTION_KEY_BIAS_LABEL_MIN_SCREEN_FONT_PX
        : PROJECTION_BIAS_LABEL_MIN_SCREEN_FONT_PX;
    const xLnDimensions = projectionVisualMetrics.xLnDimensions;
    const projectionOutputDimensions = projectionVisualMetrics.outputDimensions;

    const xInputNode = createVectorStripMatrixNode({
        role: 'x-ln-copy',
        semantic: buildSemantic(projectionSemantic, {
            role: 'x-ln-copy',
            branchKey: projectionKind
        }),
        labelTex: 'x_{\\ln}',
        labelText: 'x_ln',
        rowItems: buildProjectionInputRowItems(previewData.rows, baseSemantic, projectionKind),
        rowCount: previewData.rowCount,
        compactWidth: xLnDimensions.compactWidth,
        rowHeight: xLnDimensions.rowHeight,
        captionPosition: 'bottom',
        captionLabelScale: MHSA_SYMBOL_CAPTION_LABEL_SCALE,
        captionDimensionsTex: inputDimensionCaption.tex,
        captionDimensionsText: inputDimensionCaption.text,
        visualStyleKey: VIEW2D_STYLE_KEYS.RESIDUAL
    });

    const weightCardSize = projectionVisualMetrics.weightCardSize;
    const projectionXlnCaptionExtent = Math.min(
        xLnDimensions.compactWidth,
        previewData.rowCount * xLnDimensions.rowHeight
    );
    const projectionWeightCaptionExtent = Math.max(1, Math.min(weightCardSize.width, weightCardSize.height));
    const projectionWeightCaptionLabelScale = Math.max(
        PROJECTION_WEIGHT_CAPTION_LABEL_SCALE_MIN,
        Math.min(
            PROJECTION_WEIGHT_CAPTION_LABEL_SCALE_MAX,
            (projectionXlnCaptionExtent / projectionWeightCaptionExtent) * PROJECTION_WEIGHT_CAPTION_LABEL_SCALE_BOOST
        )
    );
    const weightNode = createCaptionedCardMatrixNode({
        role: 'projection-weight',
        semantic: buildSemantic(projectionSemantic, { role: 'projection-weight' }),
        ...buildMhsaProjectionSubscriptLabel(projectionData.weightLabelTex),
        rowCount: projectionData.weightRowCount,
        columnCount: projectionData.weightColumnCount,
        cardWidth: weightCardSize.width,
        cardHeight: weightCardSize.height,
        cardCornerRadius: 10,
        captionMinScreenHeightPx: 28,
        captionLabelScale: Math.max(
            MHSA_WEIGHT_CAPTION_LABEL_SCALE_MIN,
            projectionWeightCaptionLabelScale * MHSA_WEIGHT_CAPTION_LABEL_SCALE_FACTOR
        ),
        visualStyleKey: styleKey,
        background: projectionData.weightGradientCss || 'none',
        disableCardSurfaceEffects: true,
        metadata: {
            kind: projectionKind
        }
    });

    const biasNode = createVectorStripMatrixNode({
        role: 'projection-bias',
        semantic: buildSemantic(projectionSemantic, { role: 'projection-bias' }),
        ...buildMhsaProjectionSubscriptLabel(projectionData.biasLabelTex),
        rowItems: buildProjectionBiasRowItems(projectionData, projectionSemantic),
        rowCount: 1,
        columnCount: projectionData.outputColumnCount,
        compactWidth: projectionVisualMetrics.biasCompactWidth,
        rowHeight: projectionVisualMetrics.biasRowHeight,
        captionPosition: 'bottom',
        captionMinScreenHeightPx: PROJECTION_BIAS_CAPTION_MIN_SCREEN_HEIGHT,
        captionPreferStandardSizing: true,
        captionLabelScale: projectionBiasCaptionLabelScale,
        captionLabelMinScreenFontPx: projectionBiasLabelMinScreenFontPx,
        captionDimensionsTex: biasDimensionCaption.tex,
        captionDimensionsText: biasDimensionCaption.text,
        visualStyleKey: styleKey,
        stripMetadata: createView2dVectorStripMetadata({
            compactWidth: projectionVisualMetrics.biasCompactWidth,
            rowHeight: projectionVisualMetrics.biasRowHeight,
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
        compactWidth: projectionOutputDimensions.compactWidth,
        rowHeight: projectionOutputDimensions.rowHeight,
        captionPosition: 'bottom',
        captionLabelScale: MHSA_SYMBOL_CAPTION_LABEL_SCALE,
        captionDimensionsTex: outputDimensionCaption.tex,
        captionDimensionsText: outputDimensionCaption.text,
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
            createMhsaOperatorNode({
                role: 'projection-plus',
                semantic: buildSemantic(projectionSemantic, { role: 'projection-plus', operatorKey: 'plus' }),
                text: '+',
                visual: { styleKey: VIEW2D_STYLE_KEYS.OPERATOR }
            }),
            biasNode,
            createMhsaOperatorNode({
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

    const stageNode = createGroupNode({
        role: 'projection-stage',
        semantic: buildSemantic(projectionSemantic, { role: 'projection-stage' }),
        direction: VIEW2D_LAYOUT_DIRECTIONS.HORIZONTAL,
        gapKey: 'projection',
        children: [
            xInputNode,
            createMhsaOperatorNode({
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

    const cacheBranch = shouldShowProjectionCacheBranch(projectionKind, kvCacheState)
        ? buildProjectionCacheBranch({
            baseSemantic,
            projectionData,
            projectionKind,
            styleKey,
            outputNode,
            outputDimensions: projectionOutputDimensions,
            isSmallScreen
        })
        : null;

    return {
        stageNode,
        cacheNode: cacheBranch?.cacheNode || null,
        cacheOverlayNode: cacheBranch?.cacheOverlayNode || null,
        cacheConnectorNode: cacheBranch?.cacheConnectorNode || null
    };
}

function buildAttentionStageNode({
    baseSemantic,
    scoreStage,
    queryStageIndex,
    valueStageIndex,
    extraRows = null,
    isSmallScreen = false,
    layoutMetrics = null,
    visualTokens = null
}) {
    const resolvedTokens = visualTokens || resolveView2dVisualTokens();
    const attentionSemantic = buildSemantic(baseSemantic, {
        stage: 'attention'
    });
    const attentionFeatureExtent = resolveMhsaDimensionVisualExtent(scoreStage.queryColumnCount, {
        isSmallScreen
    });
    const attentionVectorDimensions = resolveVectorStripDimensions({
        rowCount: scoreStage.queryRowCount,
        extraRows,
        baseCompactWidth: attentionFeatureExtent,
        baseRowHeight: isSmallScreen ? PROJECTION_OUTPUT_ROW_HEIGHT_SMALL : PROJECTION_OUTPUT_ROW_HEIGHT,
        isSmallScreen,
        layoutMetrics
    });
    const attentionQueryCompactWidth = attentionVectorDimensions.compactWidth;
    const attentionQueryRowHeight = attentionVectorDimensions.rowHeight;
    const transposeStripLayout = resolveTransposeStripLayout({
        targetHeight: attentionQueryCompactWidth,
        columnUnit: attentionQueryRowHeight
    });
    const queryDimensionCaption = formatView2dMatrixDimensions(
        scoreStage.queryRowCount,
        scoreStage.queryColumnCount
    );
    const transposeDimensionCaption = formatView2dMatrixDimensions(
        scoreStage.transposeRowCount,
        scoreStage.transposeColumnCount
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
        captionLabelScale: MHSA_SYMBOL_CAPTION_LABEL_SCALE,
        captionDimensionsTex: queryDimensionCaption.tex,
        captionDimensionsText: queryDimensionCaption.text,
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
        displayRowCount: scoreStage.transposeRowCount,
        displayColumnCount: scoreStage.transposeColumnCount,
        measureRows: scoreStage.transposeRowCount,
        measureCols: scoreStage.transposeColumnCount,
        colWidth: transposeStripLayout.colWidth,
        colHeight: transposeStripLayout.colHeight,
        colGap: transposeStripLayout.colGap,
        paddingX: transposeStripLayout.paddingX,
        paddingY: transposeStripLayout.paddingY,
        cornerRadius: ATTENTION_TRANSPOSE_CORNER_RADIUS,
        captionPosition: 'bottom',
        captionLabelScale: MHSA_SYMBOL_CAPTION_LABEL_SCALE,
        captionDimensionsTex: transposeDimensionCaption.tex,
        captionDimensionsText: transposeDimensionCaption.text,
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
        },
        metadata: {
            ...createPersistentAttentionGridMetadata(),
            ...createAttentionMatrixCaptionMetadata(scoreStage.outputRowCount, scoreStage.outputColumnCount, {
                labelScale: ATTENTION_PRE_SCORE_CAPTION_LABEL_SCALE
            })
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
        },
        metadata: {
            ...createPersistentAttentionGridMetadata(),
            ...createAttentionMatrixCaptionMetadata(scoreStage.outputRowCount, scoreStage.outputColumnCount, {
                labelScale: ATTENTION_PRE_SCORE_CAPTION_LABEL_SCALE
            })
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
        },
        metadata: {
            ...createPersistentAttentionGridMetadata(),
            ...createAttentionMatrixCaptionMetadata(scoreStage.outputRowCount, scoreStage.outputColumnCount, {
                labelScale: ATTENTION_MASK_CAPTION_LABEL_SCALE
            })
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
        },
        metadata: {
            ...createPersistentAttentionGridMetadata(),
            ...createAttentionMatrixCaptionMetadata(scoreStage.postRowCount, scoreStage.postColumnCount)
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
                visual: { styleKey: VIEW2D_STYLE_KEYS.LABEL },
                metadata: {
                    renderMode: 'dom-katex',
                    minScreenHeightPx: 0,
                    fontScale: ATTENTION_SOFTMAX_TEXT_FONT_SCALE,
                    persistentMinScreenFontPx: isSmallScreen
                        ? ATTENTION_SOFTMAX_PERSISTENT_FONT_PX_SMALL
                        : ATTENTION_SOFTMAX_PERSISTENT_FONT_PX,
                    zoomedOutMinScreenFontPx: isSmallScreen
                        ? ATTENTION_SOFTMAX_ZOOMED_OUT_FONT_PX_SMALL
                        : ATTENTION_SOFTMAX_ZOOMED_OUT_FONT_PX
                }
            }),
            createMhsaOperatorNode({
                role: 'attention-softmax-open',
                semantic: buildSemantic(attentionSemantic, { role: 'attention-softmax-open', operatorKey: 'open' }),
                text: '(',
                visual: { styleKey: VIEW2D_STYLE_KEYS.OPERATOR },
                metadata: {
                    fontScale: ATTENTION_SOFTMAX_GROUPING_OPERATOR_SCALE
                }
            })
        ],
        metadata: {
            gapOverride: isSmallScreen ? ATTENTION_SOFTMAX_PREFIX_GAP_SMALL : ATTENTION_SOFTMAX_PREFIX_GAP
        }
    });

    const headOutputNode = Array.isArray(scoreStage.valueRows) && Array.isArray(scoreStage.headOutputRows)
        ? createVectorStripMatrixNode({
            role: 'attention-head-output',
            semantic: buildSemantic(attentionSemantic, { role: 'attention-head-output', stage: 'head-output' }),
            labelTex: scoreStage.headOutputLabelTex,
            labelText: scoreStage.headOutputLabelTex,
            rowItems: buildGradientRowItems(
                scoreStage.headOutputRows,
                buildSemantic(attentionSemantic, { stage: 'attention-head-output' }),
                'attention-head-output-row'
            ),
            rowCount: scoreStage.headOutputRowCount,
            columnCount: scoreStage.headOutputColumnCount,
            compactWidth: attentionVectorDimensions.compactWidth,
            rowHeight: attentionVectorDimensions.rowHeight,
            captionPosition: 'bottom',
            captionLabelScale: MHSA_SYMBOL_CAPTION_LABEL_SCALE,
            visualStyleKey: VIEW2D_STYLE_KEYS.MHSA_HEAD_OUTPUT,
            metadata: {
                disableEdgeOrnament: true
            }
        })
        : null;
    const headOutputOutgoingSpacerNode = headOutputNode
        ? createHiddenSpacer({
            semantic: buildSemantic(attentionSemantic, {
                stage: 'head-output',
                role: 'outgoing-arrow-spacer'
            }),
            role: 'outgoing-arrow-spacer',
            width: isSmallScreen ? OUTGOING_ARROW_SPACER_WIDTH_SMALL : OUTGOING_ARROW_SPACER_WIDTH,
            height: 1
        })
        : null;

    const headOutputStageNode = headOutputNode
        ? createGroupNode({
            role: 'attention-head-output-stage',
            semantic: buildSemantic(attentionSemantic, { role: 'attention-head-output-stage', stage: 'head-output' }),
            direction: VIEW2D_LAYOUT_DIRECTIONS.HORIZONTAL,
            gapKey: 'head-output',
            children: [
                createGroupNode({
                    role: 'attention-head-output-product',
                    semantic: buildSemantic(attentionSemantic, { role: 'attention-head-output-product', stage: 'head-output' }),
                    direction: VIEW2D_LAYOUT_DIRECTIONS.HORIZONTAL,
                    gapKey: 'inline',
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
                            },
                            metadata: {
                                ...createPersistentAttentionGridMetadata(),
                                ...createAttentionMatrixCaptionMetadata(scoreStage.postRowCount, scoreStage.postColumnCount)
                            }
                        }),
                        createMhsaOperatorNode({
                            role: 'attention-head-output-multiply',
                            semantic: buildSemantic(attentionSemantic, { role: 'attention-head-output-multiply', operatorKey: 'multiply' }),
                            text: 'x',
                            visual: { styleKey: VIEW2D_STYLE_KEYS.OPERATOR }
                        }),
                        createVectorStripMatrixNode({
                            role: 'attention-value-post',
                            semantic: buildSemantic(attentionSemantic, { role: 'attention-value-post', stage: 'head-output' }),
                            labelTex: scoreStage.valueLabelTex,
                            labelText: scoreStage.valueLabelTex,
                            rowItems: buildGradientRowItems(scoreStage.valueRows, buildSemantic(attentionSemantic, {
                                stage: 'attention-value-post'
                            }), 'attention-value-post-row'),
                            rowCount: scoreStage.valueRowCount,
                            columnCount: scoreStage.valueColumnCount,
                            compactWidth: attentionVectorDimensions.compactWidth,
                            rowHeight: attentionVectorDimensions.rowHeight,
                            captionPosition: 'bottom',
                            captionLabelScale: MHSA_SYMBOL_CAPTION_LABEL_SCALE,
                            visualStyleKey: VIEW2D_STYLE_KEYS.MHSA_V,
                            metadata: {
                                stageIndex: valueStageIndex
                            }
                        })
                    ],
                    metadata: {
                        gapOverride: isSmallScreen
                            ? ATTENTION_HEAD_OUTPUT_PRODUCT_GAP_SMALL
                            : ATTENTION_HEAD_OUTPUT_PRODUCT_GAP
                    }
                }),
                createMhsaOperatorNode({
                    role: 'attention-head-output-equals',
                    semantic: buildSemantic(attentionSemantic, { role: 'attention-head-output-equals', operatorKey: 'equals' }),
                    text: '=',
                    visual: { styleKey: VIEW2D_STYLE_KEYS.OPERATOR }
                }),
                headOutputNode,
                headOutputOutgoingSpacerNode
            ],
            metadata: {
                gapOverride: isSmallScreen
                    ? ATTENTION_HEAD_OUTPUT_GAP_SMALL
                    : ATTENTION_HEAD_OUTPUT_GAP
            }
        })
        : null;

    const softmaxCoreFlowChildren = [
        maskedInputNode,
        createMhsaOperatorNode({
            role: 'attention-softmax-plus',
            semantic: buildSemantic(attentionSemantic, { role: 'attention-softmax-plus', operatorKey: 'plus' }),
            text: '+',
            visual: { styleKey: VIEW2D_STYLE_KEYS.OPERATOR }
        }),
        maskNode,
        createGroupNode({
            role: 'attention-softmax-close-equals-group',
            semantic: buildSemantic(attentionSemantic, { role: 'attention-softmax-close-equals-group' }),
            direction: VIEW2D_LAYOUT_DIRECTIONS.HORIZONTAL,
            gapKey: 'inline',
            children: [
                createMhsaOperatorNode({
                    role: 'attention-softmax-close',
                    semantic: buildSemantic(attentionSemantic, { role: 'attention-softmax-close', operatorKey: 'close' }),
                    text: ')',
                    visual: { styleKey: VIEW2D_STYLE_KEYS.OPERATOR },
                    metadata: {
                        fontScale: ATTENTION_SOFTMAX_GROUPING_OPERATOR_SCALE
                    }
                }),
                createMhsaOperatorNode({
                    role: 'attention-softmax-equals',
                    semantic: buildSemantic(attentionSemantic, { role: 'attention-softmax-equals', operatorKey: 'equals' }),
                    text: '=',
                    visual: { styleKey: VIEW2D_STYLE_KEYS.OPERATOR }
                })
            ],
            metadata: {
                gapOverride: isSmallScreen
                    ? ATTENTION_SOFTMAX_CLOSE_EQUALS_GAP_SMALL
                    : ATTENTION_SOFTMAX_CLOSE_EQUALS_GAP
            }
        }),
        postNode
    ];
    if (headOutputStageNode) {
        softmaxCoreFlowChildren.push(
            createHiddenSpacer({
                semantic: buildSemantic(attentionSemantic, {
                    role: 'attention-post-copy-stage-spacer'
                }),
                role: 'attention-post-copy-stage-spacer',
                width: isSmallScreen
                    ? ATTENTION_POST_COPY_STAGE_SPACER_SMALL
                    : ATTENTION_POST_COPY_STAGE_SPACER,
                height: 1
            })
        );
        softmaxCoreFlowChildren.push(headOutputStageNode);
    }

    const softmaxCoreFlowNode = createGroupNode({
        role: 'attention-softmax-core-flow',
        semantic: buildSemantic(attentionSemantic, { role: 'attention-softmax-core-flow' }),
        direction: VIEW2D_LAYOUT_DIRECTIONS.HORIZONTAL,
        gapKey: 'softmax',
        children: softmaxCoreFlowChildren,
        metadata: {
            gapOverride: resolveAttentionSoftmaxCoreFlowGap(layoutMetrics, isSmallScreen)
        }
    });

    const softmaxCoreGap = isSmallScreen
        ? ATTENTION_SOFTMAX_FLOW_GAP_SMALL
        : ATTENTION_SOFTMAX_FLOW_GAP;
    const softmaxRowOffset = isSmallScreen
        ? ATTENTION_SOFTMAX_ROW_OFFSET_SMALL
        : ATTENTION_SOFTMAX_ROW_OFFSET;
    const softmaxPrefixOffset = resolveAttentionSoftmaxPrefixOffset(
        scoreStage.outputRowCount,
        isSmallScreen,
        layoutMetrics
    );

    const softmaxPrefixColumnNode = createGroupNode({
        role: 'attention-softmax-prefix-column',
        semantic: buildSemantic(attentionSemantic, { role: 'attention-softmax-prefix-column' }),
        direction: VIEW2D_LAYOUT_DIRECTIONS.VERTICAL,
        gapKey: 'default',
        align: 'end',
        children: [
            createHiddenSpacer({
                semantic: buildSemantic(attentionSemantic, {
                    role: 'attention-softmax-prefix-offset'
                }),
                role: 'attention-softmax-prefix-offset',
                width: 1,
                height: softmaxPrefixOffset
            }),
            softmaxPrefixNode
        ],
        metadata: {
            gapOverride: 0
        }
    });

    const softmaxBodyNode = createGroupNode({
        role: 'attention-softmax-body',
        semantic: buildSemantic(attentionSemantic, { role: 'attention-softmax-body' }),
        direction: VIEW2D_LAYOUT_DIRECTIONS.HORIZONTAL,
        gapKey: 'inline',
        align: 'start',
        children: [
            softmaxPrefixColumnNode,
            softmaxCoreFlowNode
        ],
        layout: {
            anchorAlign: {
                axis: 'x',
                selfNodeId: maskedInputNode.id,
                targetNodeId: preScoreNode.id,
                selfAnchor: VIEW2D_ANCHOR_SIDES.LEFT,
                targetAnchor: VIEW2D_ANCHOR_SIDES.LEFT
            }
        },
        metadata: {
            gapOverride: isSmallScreen ? ATTENTION_SOFTMAX_BODY_GAP_SMALL : ATTENTION_SOFTMAX_BODY_GAP
        }
    });
    const attentionResultEquationRowNode = createGroupNode({
        role: 'attention-result-equation-row',
        semantic: buildSemantic(attentionSemantic, { role: 'attention-result-equation-row' }),
        direction: VIEW2D_LAYOUT_DIRECTIONS.HORIZONTAL,
        gapKey: 'inline',
        align: 'center',
        children: [
            createMhsaOperatorNode({
                role: 'attention-equals',
                semantic: buildSemantic(attentionSemantic, { role: 'attention-equals', operatorKey: 'equals' }),
                text: '=',
                visual: { styleKey: VIEW2D_STYLE_KEYS.OPERATOR }
            }),
            preScoreNode
        ],
        metadata: {
            gapOverride: isSmallScreen ? ATTENTION_RESULT_GROUP_GAP_SMALL : ATTENTION_RESULT_GROUP_GAP
        }
    });

    const attentionResultContentNode = createGroupNode({
        role: 'attention-result-content',
        semantic: buildSemantic(attentionSemantic, { role: 'attention-result-content' }),
        direction: VIEW2D_LAYOUT_DIRECTIONS.VERTICAL,
        gapKey: 'softmax',
        align: 'start',
        children: [
            attentionResultEquationRowNode,
            softmaxBodyNode
        ],
        metadata: {
            gapOverride: softmaxCoreGap + softmaxRowOffset
        }
    });

    const attentionResultTopOffset = resolveAttentionFlowRowHeight(
        scoreStage.outputRowCount,
        isSmallScreen,
        layoutMetrics
    ) + softmaxCoreGap + (
        isSmallScreen
            ? ATTENTION_SOFTMAX_RESULT_ALIGNMENT_OFFSET_SMALL
            : ATTENTION_SOFTMAX_RESULT_ALIGNMENT_OFFSET
    );

    const divisorNode = createGroupNode({
        role: 'attention-divisor-group',
        semantic: buildSemantic(attentionSemantic, { role: 'attention-divisor-group' }),
        direction: VIEW2D_LAYOUT_DIRECTIONS.HORIZONTAL,
        gapKey: 'inline',
        children: [
            createMhsaOperatorNode({
                role: 'attention-divide',
                semantic: buildSemantic(attentionSemantic, { role: 'attention-divide', operatorKey: 'divide' }),
                text: '/',
                visual: { styleKey: VIEW2D_STYLE_KEYS.OPERATOR }
            }),
            createTextNode({
                role: 'attention-scale',
                semantic: buildSemantic(attentionSemantic, { role: 'attention-scale', focusKey: 'scale' }),
                tex: scoreStage.scaleLabelTex,
                text: 'sqrt(d_head)',
                presentation: VIEW2D_TEXT_PRESENTATIONS.LABEL,
                visual: { styleKey: VIEW2D_STYLE_KEYS.LABEL },
                layout: {
                    offsetX: isSmallScreen
                        ? ATTENTION_SCALE_TEXT_OFFSET_X_SMALL
                        : ATTENTION_SCALE_TEXT_OFFSET_X
                },
                metadata: {
                    renderMode: 'dom-katex',
                    minScreenHeightPx: 0,
                    fontScale: ATTENTION_SCALE_TEXT_FONT_SCALE
                }
            }),
        ],
        metadata: {
            gapOverride: isSmallScreen ? ATTENTION_DIVISOR_GAP_SMALL : ATTENTION_DIVISOR_GAP
        }
    });

    const divisorClusterNode = createGroupNode({
        role: 'attention-divisor-cluster',
        semantic: buildSemantic(attentionSemantic, { role: 'attention-divisor-cluster' }),
        direction: VIEW2D_LAYOUT_DIRECTIONS.HORIZONTAL,
        gapKey: 'inline',
        children: [
            createMhsaOperatorNode({
                role: 'attention-close',
                semantic: buildSemantic(attentionSemantic, {
                    role: 'attention-close',
                    operatorKey: 'close',
                    clusterKey: 'divisor'
                }),
                text: ')',
                visual: { styleKey: VIEW2D_STYLE_KEYS.OPERATOR },
                metadata: {
                    fontScale: ATTENTION_GROUPING_OPERATOR_SCALE
                }
            }),
            divisorNode
        ],
        metadata: {
            gapOverride: isSmallScreen ? ATTENTION_CLOSE_DIVISOR_GAP_SMALL : ATTENTION_CLOSE_DIVISOR_GAP
        }
    });

    const qktEquationNode = createGroupNode({
        role: 'attention-qkt-equation',
        semantic: buildSemantic(attentionSemantic, { role: 'attention-qkt-equation' }),
        direction: VIEW2D_LAYOUT_DIRECTIONS.HORIZONTAL,
        gapKey: 'inline',
        children: [
            createMhsaOperatorNode({
                role: 'attention-open',
                semantic: buildSemantic(attentionSemantic, { role: 'attention-open', operatorKey: 'open' }),
                text: '(',
                visual: { styleKey: VIEW2D_STYLE_KEYS.OPERATOR },
                metadata: {
                    fontScale: ATTENTION_GROUPING_OPERATOR_SCALE
                }
            }),
            querySourceNode,
            createMhsaOperatorNode({
                role: 'attention-multiply',
                semantic: buildSemantic(attentionSemantic, { role: 'attention-multiply', operatorKey: 'multiply' }),
                text: 'x',
                visual: { styleKey: VIEW2D_STYLE_KEYS.OPERATOR }
            }),
            transposeNode
        ],
        metadata: {
            gapOverride: isSmallScreen ? ATTENTION_QKT_GROUP_GAP_SMALL : ATTENTION_QKT_GROUP_GAP
        }
    });

    const attentionLeftHandSideNode = createGroupNode({
        role: 'attention-left-hand-side',
        semantic: buildSemantic(attentionSemantic, { role: 'attention-left-hand-side' }),
        direction: VIEW2D_LAYOUT_DIRECTIONS.HORIZONTAL,
        gapKey: 'attention',
        align: 'center',
        children: [
            qktEquationNode,
            divisorClusterNode
        ],
        metadata: {
            gapOverride: resolveAttentionLeftHandSideGap(layoutMetrics, isSmallScreen)
        }
    });

    const attentionResultNode = createGroupNode({
        role: 'attention-result-stage',
        semantic: buildSemantic(attentionSemantic, { role: 'attention-result-stage' }),
        direction: VIEW2D_LAYOUT_DIRECTIONS.VERTICAL,
        gapKey: 'default',
        align: 'start',
        children: [
            createHiddenSpacer({
                semantic: buildSemantic(attentionSemantic, {
                    role: 'attention-result-top-offset'
                }),
                role: 'attention-result-top-offset',
                width: 1,
                height: attentionResultTopOffset
            }),
            attentionResultContentNode
        ],
        metadata: {
            gapOverride: 0
        }
    });

    return createGroupNode({
        role: 'attention-stage',
        semantic: buildSemantic(attentionSemantic, { role: 'attention-stage' }),
        direction: VIEW2D_LAYOUT_DIRECTIONS.HORIZONTAL,
        gapKey: 'inline',
        children: [
            attentionLeftHandSideNode,
            attentionResultNode
        ],
        metadata: {
            queryStageIndex,
            valueStageIndex,
            gapOverride: isSmallScreen ? ATTENTION_RESULT_CLUSTER_GAP_SMALL : ATTENTION_RESULT_CLUSTER_GAP
        }
    });
}

function buildProjectionIngressConnectorNodes({
    baseSemantic,
    layoutMetrics,
    projectionNodes,
    sourceNode
}) {
    const connectorGaps = layoutMetrics?.connectorGaps || {};
    const findProjectionInput = (kind) => projectionNodes.find((stageNode) => stageNode?.metadata?.kind === kind)
        ?.children?.[0] || null;

    return ['q', 'k', 'v'].map((kind) => {
        const targetNode = findProjectionInput(kind);
        if (!sourceNode || !targetNode) return null;
        return createConnectorNode({
            role: `connector-xln-${kind}`,
            semantic: buildSemantic(baseSemantic, {
                stage: `connector-xln-${kind}`,
                role: 'connector-xln',
                branchKey: kind
            }),
            source: createAnchorRef(sourceNode.id, VIEW2D_ANCHOR_SIDES.RIGHT),
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

function buildEdgeConnectorNodes({
    baseSemantic,
    projectionSourceNode = null,
    incomingArrowSpacerNode = null,
    headOutputNode = null,
    outgoingArrowSpacerNode = null
} = {}) {
    const connectorNodes = [];

    if (incomingArrowSpacerNode && projectionSourceNode) {
        connectorNodes.push(
            createConnectorNode({
                role: 'connector-source-xln',
                semantic: buildSemantic(baseSemantic, {
                    stage: 'connector-source-xln',
                    role: 'connector-source-xln'
                }),
                source: createAnchorRef(incomingArrowSpacerNode.id, VIEW2D_ANCHOR_SIDES.LEFT),
                target: createAnchorRef(projectionSourceNode.id, VIEW2D_ANCHOR_SIDES.LEFT),
                route: VIEW2D_CONNECTOR_ROUTES.HORIZONTAL,
                gap: 0,
                targetGap: PROJECTION_XLN_TARGET_GAP,
                visual: {
                    styleKey: VIEW2D_STYLE_KEYS.CONNECTOR_NEUTRAL,
                    stroke: MHSA_CONNECTOR_STROKE
                },
                metadata: {
                    preserveColor: true,
                    strokeWidthScale: EDGE_CONNECTOR_STROKE_WIDTH_SCALE
                }
            })
        );
    }

    if (headOutputNode && outgoingArrowSpacerNode) {
        connectorNodes.push(
            createConnectorNode({
                role: 'connector-head-output-outgoing',
                semantic: buildSemantic(baseSemantic, {
                    stage: 'connector-head-output-outgoing',
                    role: 'connector-head-output-outgoing'
                }),
                source: createAnchorRef(headOutputNode.id, VIEW2D_ANCHOR_SIDES.RIGHT),
                target: createAnchorRef(outgoingArrowSpacerNode.id, VIEW2D_ANCHOR_SIDES.RIGHT),
                route: VIEW2D_CONNECTOR_ROUTES.HORIZONTAL,
                gap: 0,
                sourceGap: HEAD_OUTPUT_EDGE_SOURCE_GAP,
                visual: {
                    styleKey: VIEW2D_STYLE_KEYS.CONNECTOR_NEUTRAL,
                    stroke: MHSA_CONNECTOR_STROKE
                },
                metadata: {
                    preserveColor: true
                }
            })
        );
    }

    return connectorNodes;
}

function buildConnectorNodes({
    baseSemantic,
    layoutMetrics,
    projectionNodes,
    attentionNode,
    isSmallScreen = false
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
                route: VIEW2D_CONNECTOR_ROUTES.ELBOW,
                gap: connectorGaps.transpose,
                targetGap: ATTENTION_CONNECTOR_CAPTION_EXIT_GAP,
                gapKey: 'transpose',
                visual: {
                    styleKey: VIEW2D_STYLE_KEYS.CONNECTOR_NEUTRAL,
                    stroke: MHSA_CONNECTOR_STROKE
                },
                metadata: {
                    preserveColor: true,
                    targetAnchorMode: 'caption-bottom'
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
                sourceGap: ATTENTION_CONNECTOR_CAPTION_EXIT_GAP,
                gapKey: 'pre',
                visual: {
                    styleKey: VIEW2D_STYLE_KEYS.CONNECTOR_NEUTRAL,
                    stroke: MHSA_CONNECTOR_STROKE
                },
                metadata: {
                    preserveColor: true,
                    sourceAnchorMode: 'caption-bottom',
                    sourceAnchorOffsetY: ATTENTION_PRE_CONNECTOR_SOURCE_OFFSET_Y
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
                route: VIEW2D_CONNECTOR_ROUTES.ELBOW,
                gap: connectorGaps.value,
                sourceGap: ATTENTION_VALUE_CONNECTOR_SOURCE_GAP,
                targetGap: ATTENTION_VALUE_CONNECTOR_TARGET_GAP,
                gapKey: 'value',
                visual: {
                    styleKey: VIEW2D_STYLE_KEYS.CONNECTOR_NEUTRAL,
                    stroke: MHSA_CONNECTOR_STROKE
                },
                metadata: {
                    preserveColor: true,
                    sourceAnchorOffsetY: ATTENTION_VALUE_CONNECTOR_SOURCE_OFFSET_Y,
                    targetAnchorMode: 'caption-bottom'
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
    visualTokens = null,
    kvCacheState = null
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
    const resolvedKvCacheState = normalizeKvCacheState(kvCacheState);
    const queryStageIndex = Math.max(
        0,
        resolvedPreviewData.projections.findIndex((projectionData) => String(projectionData?.kind || '').toLowerCase() === 'q')
    );
    const valueStageIndex = Math.max(
        0,
        resolvedPreviewData.projections.findIndex((projectionData) => String(projectionData?.kind || '').toLowerCase() === 'v')
    );

    const projectionStageEntries = resolvedPreviewData.projections
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
                extraRows: resolvedLayoutMetrics?.extraRows,
                isSmallScreen,
                layoutMetrics: resolvedLayoutMetrics,
                kvCacheState: resolvedKvCacheState
            });
        })
        .filter(Boolean);
    const projectionNodes = projectionStageEntries
        .map((entry) => entry.stageNode)
        .filter(Boolean);
    const projectionCacheOverlayNodes = projectionStageEntries
        .map((entry) => entry.cacheOverlayNode)
        .filter(Boolean);
    const projectionCacheConnectorNodes = projectionStageEntries
        .map((entry) => entry.cacheConnectorNode)
        .filter(Boolean);
    const projectionSourceNode = buildProjectionSourceNode({
        baseSemantic,
        previewData: resolvedPreviewData,
        rowCount: resolvedPreviewData.rowCount,
        extraRows: resolvedLayoutMetrics?.extraRows,
        isSmallScreen,
        layoutMetrics: resolvedLayoutMetrics
    });
    const projectionStackNode = createGroupNode({
        role: 'projection-stack',
        semantic: buildSemantic(baseSemantic, { stage: 'projection-stack', role: 'projection-stack' }),
        direction: VIEW2D_LAYOUT_DIRECTIONS.VERTICAL,
        gapKey: 'stack',
        children: projectionNodes,
        metadata: {
            gapOverride: resolveProjectionStackGap({
                rowCount: resolvedPreviewData.rowCount,
                extraRows: resolvedLayoutMetrics?.extraRows,
                isSmallScreen
            })
        }
    });
    const attentionStageLift = resolveAttentionStageLift({
        rowCount: resolvedPreviewData.rowCount,
        extraRows: resolvedLayoutMetrics?.extraRows,
        isSmallScreen
    });

    const attentionNode = resolvedPreviewData.attentionScoreStage
        && Array.isArray(resolvedPreviewData.attentionScoreStage.queryRows)
        && Array.isArray(resolvedPreviewData.attentionScoreStage.outputRows)
        ? buildAttentionStageNode({
            baseSemantic,
            scoreStage: resolvedPreviewData.attentionScoreStage,
            queryStageIndex,
            valueStageIndex,
            extraRows: resolvedLayoutMetrics?.extraRows,
            isSmallScreen,
            layoutMetrics: resolvedLayoutMetrics,
            visualTokens: resolvedTokens
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

    const incomingArrowSpacerNode = createHiddenSpacer({
        semantic: buildSemantic(baseSemantic, {
            stage: 'projection-sidecar',
            role: 'incoming-arrow-spacer'
        }),
        role: 'incoming-arrow-spacer',
        width: isSmallScreen ? INCOMING_ARROW_SPACER_WIDTH_SMALL : INCOMING_ARROW_SPACER_WIDTH,
        height: 1
    });

    const projectionSidecarNode = createGroupNode({
        role: 'projection-sidecar',
        semantic: buildSemantic(baseSemantic, { stage: 'projection-sidecar', role: 'projection-sidecar' }),
        direction: VIEW2D_LAYOUT_DIRECTIONS.HORIZONTAL,
        gapKey: 'projection',
        children: [
            incomingArrowSpacerNode,
            projectionSourceNode,
            projectionStackNode
        ],
        metadata: {
            gapOverride: isSmallScreen ? PROJECTION_SIDECAR_GAP_SMALL : PROJECTION_SIDECAR_GAP
        }
    });

    const mainFlowChildren = attentionRootNode
        ? [projectionSidecarNode, attentionRootNode]
        : [projectionSidecarNode];
    const rootNodes = [
        createGroupNode({
            role: 'mhsa-main-flow',
            semantic: buildSemantic(baseSemantic, { stage: 'mhsa-main-flow', role: 'mhsa-main-flow' }),
            direction: VIEW2D_LAYOUT_DIRECTIONS.HORIZONTAL,
            gapKey: 'stage',
            align: 'center',
            children: mainFlowChildren,
            metadata: {
                gapOverride: attentionRootNode
                    ? (isSmallScreen ? ATTENTION_STAGE_SIDECAR_GAP_SMALL : ATTENTION_STAGE_SIDECAR_GAP)
                    : 0
            }
        })
    ];
    if (projectionCacheOverlayNodes.length) {
        rootNodes.push(
            createGroupNode({
                role: 'projection-cache-overlay',
                semantic: buildSemantic(baseSemantic, {
                    stage: 'kv-cache',
                    role: 'projection-cache-overlay'
                }),
                direction: VIEW2D_LAYOUT_DIRECTIONS.OVERLAY,
                gapKey: 'default',
                children: projectionCacheOverlayNodes,
                metadata: {
                    gapOverride: 0
                }
            })
        );
    }

    const edgeHeadOutputNode = findNestedNodeByRole(attentionNode, 'attention-head-output');
    const edgeHeadOutputOutgoingSpacerNode = findNestedNodeByRole(attentionNode, 'outgoing-arrow-spacer');
    const connectorNodes = buildProjectionIngressConnectorNodes({
        baseSemantic,
        layoutMetrics: resolvedLayoutMetrics,
        projectionNodes,
        sourceNode: projectionSourceNode
    });
    connectorNodes.push(...projectionCacheConnectorNodes);
    connectorNodes.unshift(...buildEdgeConnectorNodes({
        baseSemantic,
        projectionSourceNode,
        incomingArrowSpacerNode,
        headOutputNode: edgeHeadOutputNode,
        outgoingArrowSpacerNode: edgeHeadOutputOutgoingSpacerNode
    }));
    if (attentionRootNode) {
        connectorNodes.push(...buildConnectorNodes({
            baseSemantic,
            layoutMetrics: resolvedLayoutMetrics,
            projectionNodes,
            attentionNode,
            isSmallScreen
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
            kvCacheState: resolvedKvCacheState,
            layoutMetrics: resolvedLayoutMetrics,
            tokens: resolvedTokens
        }
    });
}
