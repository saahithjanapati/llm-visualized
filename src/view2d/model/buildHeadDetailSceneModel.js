import { D_MODEL } from '../../ui/selectionPanelConstants.js';
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
import { createView2dVectorStripMetadata } from '../shared/vectorStrip.js';

const HEAD_DETAIL_COPY_DEFS = Object.freeze([
    Object.freeze({
        key: 'q',
        tex: 'X_{\\ln}^{Q}',
        text: 'X_ln^Q'
    }),
    Object.freeze({
        key: 'k',
        tex: 'X_{\\ln}^{K}',
        text: 'X_ln^K'
    }),
    Object.freeze({
        key: 'v',
        tex: 'X_{\\ln}^{V}',
        text: 'X_ln^V'
    })
]);

const DETAIL_LAYOUT = Object.freeze({
    sourceWidth: 2,
    sourceHeight: 2,
    branchSpacerWidth: 44,
    branchSpacerHeight: 2,
    copyWidth: 176,
    copyWidthSmall: 156,
    rowHeight: 8,
    rowHeightSmall: 7,
    rowGap: 2,
    stackGap: 22,
    stackGapSmall: 18,
    captionMinScreenHeightPx: 18
});

function normalizeIndex(value) {
    return Number.isFinite(value) ? Math.max(0, Math.floor(value)) : null;
}

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

function createCaptionMetadata({
    position = 'bottom',
    styleKey = null,
    dimensionsTex = '',
    dimensionsText = '',
    minScreenHeightPx = null
} = {}) {
    const caption = {};
    const safePosition = String(position || '').trim().toLowerCase();
    if (safePosition === 'top' || safePosition === 'bottom' || safePosition === 'inside-top') {
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
    return Object.keys(caption).length ? { caption } : null;
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

function buildResidualCopyRowItems(rows = [], {
    layerIndex = null,
    headIndex = null,
    branchKey = ''
} = {}) {
    return rows.map((rowItem, rowIndex) => {
        const tokenIndex = normalizeIndex(rowItem?.tokenIndex);
        const semantic = {
            componentKind: 'residual',
            layerIndex,
            headIndex,
            stage: 'ln1.shift',
            role: 'x-ln-row',
            rowIndex,
            ...(Number.isFinite(tokenIndex) ? { tokenIndex } : {}),
            ...(branchKey ? { branchKey } : {})
        };
        return {
            id: buildSceneNodeId(semantic),
            index: rowIndex,
            label: typeof rowItem?.label === 'string' && rowItem.label.length
                ? rowItem.label
                : `Token ${rowIndex + 1}`,
            semantic,
            rawValues: Array.isArray(rowItem?.rawValues) ? [...rowItem.rawValues] : null,
            gradientCss: rowItem?.gradientCss || 'none',
            title: typeof rowItem?.title === 'string' ? rowItem.title : null
        };
    });
}

function buildResidualCopyNode(copyDef, rows = [], {
    layerIndex = null,
    headIndex = null,
    rowCount = 1,
    isSmallScreen = false
} = {}) {
    const copyWidth = isSmallScreen ? DETAIL_LAYOUT.copyWidthSmall : DETAIL_LAYOUT.copyWidth;
    const rowHeight = isSmallScreen ? DETAIL_LAYOUT.rowHeightSmall : DETAIL_LAYOUT.rowHeight;
    const branchKey = copyDef?.key || '';
    const semantic = {
        componentKind: 'mhsa',
        layerIndex,
        headIndex,
        stage: 'head-detail',
        role: 'x-ln-copy',
        ...(branchKey ? { branchKey } : {})
    };
    return createMatrixNode({
        role: 'x-ln-copy',
        semantic,
        label: buildLabel(copyDef?.tex || 'X_{\\ln}', copyDef?.text || 'X_ln'),
        dimensions: {
            rows: rowCount,
            cols: D_MODEL
        },
        presentation: VIEW2D_MATRIX_PRESENTATIONS.COMPACT_ROWS,
        shape: VIEW2D_MATRIX_SHAPES.MATRIX,
        rowItems: buildResidualCopyRowItems(rows, {
            layerIndex,
            headIndex,
            branchKey
        }),
        visual: {
            styleKey: VIEW2D_STYLE_KEYS.RESIDUAL
        },
        metadata: mergeMetadata(
            createCaptionMetadata({
                position: 'bottom',
                styleKey: VIEW2D_STYLE_KEYS.LABEL,
                dimensionsTex: `${rowCount} \\\\times ${D_MODEL}`,
                dimensionsText: `${rowCount} × ${D_MODEL}`,
                minScreenHeightPx: DETAIL_LAYOUT.captionMinScreenHeightPx
            }),
            createView2dVectorStripMetadata({
                compactWidth: copyWidth,
                rowHeight,
                rowGap: DETAIL_LAYOUT.rowGap,
                hideSurface: true
            })
        )
    });
}

export function buildHeadDetailSceneModel({
    headDetailPreview = null,
    headDetailTarget = null,
    visualTokens = null,
    isSmallScreen = false
} = {}) {
    const layerIndex = normalizeIndex(headDetailTarget?.layerIndex);
    const headIndex = normalizeIndex(headDetailTarget?.headIndex);
    const previewRows = Array.isArray(headDetailPreview?.rowItems) ? headDetailPreview.rowItems : [];
    const rowCount = Math.max(1, previewRows.length || 1);
    if (!Number.isFinite(layerIndex) || !Number.isFinite(headIndex) || !previewRows.length) {
        return null;
    }

    const sourceNode = createHiddenSpacer({
        semantic: {
            componentKind: 'mhsa',
            layerIndex,
            headIndex,
            stage: 'head-detail',
            role: 'source-anchor'
        },
        role: 'source-anchor',
        width: DETAIL_LAYOUT.sourceWidth,
        height: DETAIL_LAYOUT.sourceHeight
    });
    const branchSpacer = createHiddenSpacer({
        semantic: {
            componentKind: 'mhsa',
            layerIndex,
            headIndex,
            stage: 'head-detail',
            role: 'branch-spacer'
        },
        role: 'branch-spacer',
        width: DETAIL_LAYOUT.branchSpacerWidth,
        height: DETAIL_LAYOUT.branchSpacerHeight
    });
    const copyNodes = HEAD_DETAIL_COPY_DEFS.map((copyDef) => buildResidualCopyNode(copyDef, previewRows, {
        layerIndex,
        headIndex,
        rowCount,
        isSmallScreen
    }));

    const stackGroup = createGroupNode({
        role: 'x-ln-copy-stack',
        semantic: {
            componentKind: 'mhsa',
            layerIndex,
            headIndex,
            stage: 'head-detail',
            role: 'x-ln-copy-stack'
        },
        direction: VIEW2D_LAYOUT_DIRECTIONS.VERTICAL,
        gapKey: 'default',
        children: copyNodes,
        metadata: {
            gapOverride: isSmallScreen ? DETAIL_LAYOUT.stackGapSmall : DETAIL_LAYOUT.stackGap
        }
    });

    const connectorNodes = copyNodes.map((copyNode, copyIndex) => createConnectorNode({
        role: `x-ln-copy-connector-${copyIndex}`,
        semantic: {
            componentKind: 'mhsa',
            layerIndex,
            headIndex,
            stage: 'head-detail',
            role: 'x-ln-copy-connector',
            copyIndex
        },
        source: createAnchorRef(sourceNode.id, VIEW2D_ANCHOR_SIDES.RIGHT),
        target: createAnchorRef(copyNode.id, VIEW2D_ANCHOR_SIDES.LEFT),
        route: VIEW2D_CONNECTOR_ROUTES.HORIZONTAL,
        visual: {
            styleKey: VIEW2D_STYLE_KEYS.CONNECTOR_NEUTRAL,
            stroke: 'rgb(255, 255, 255)'
        },
        metadata: {
            preserveColor: true
        }
    }));

    return createSceneModel({
        semantic: {
            componentKind: 'mhsa',
            layerIndex,
            headIndex,
            stage: 'head-detail',
            role: 'scene'
        },
        nodes: [
            createGroupNode({
                role: 'head-detail-stage',
                semantic: {
                    componentKind: 'mhsa',
                    layerIndex,
                    headIndex,
                    stage: 'head-detail',
                    role: 'head-detail-stage'
                },
                direction: VIEW2D_LAYOUT_DIRECTIONS.HORIZONTAL,
                gapKey: 'default',
                children: [
                    sourceNode,
                    branchSpacer,
                    stackGroup
                ]
            }),
            createGroupNode({
                role: 'head-detail-connectors',
                semantic: {
                    componentKind: 'mhsa',
                    layerIndex,
                    headIndex,
                    stage: 'head-detail',
                    role: 'head-detail-connectors'
                },
                direction: VIEW2D_LAYOUT_DIRECTIONS.OVERLAY,
                gapKey: 'default',
                children: connectorNodes
            })
        ],
        metadata: {
            source: 'buildHeadDetailSceneModel',
            rowCount,
            isSmallScreen: !!isSmallScreen,
            tokens: visualTokens || resolveView2dVisualTokens()
        }
    });
}
