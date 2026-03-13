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
import { formatView2dMatrixDimensions } from '../shared/formatMatrixDimensions.js';
import { D_MODEL } from '../../ui/selectionPanelConstants.js';

const HEAD_DETAIL_COPY_DEFS = Object.freeze([
    Object.freeze({
        key: 'q'
    }),
    Object.freeze({
        key: 'k'
    }),
    Object.freeze({
        key: 'v'
    })
]);

const DETAIL_LAYOUT = Object.freeze({
    sourceWidth: 56,
    sourceHeight: 2,
    branchSpacerWidth: 28,
    branchSpacerHeight: 2,
    connectorGap: 0,
    stackGap: 96,
    stackGapSmall: 72,
    captionMinScreenHeightPx: 28
});

function normalizeIndex(value) {
    return Number.isFinite(value) ? Math.max(0, Math.floor(value)) : null;
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
    const branchKey = copyDef?.key || '';
    const branchLabel = branchKey ? branchKey.toUpperCase() : '';
    const dimensionCaption = formatView2dMatrixDimensions(rowCount, D_MODEL);
    const semantic = {
        componentKind: 'mhsa',
        layerIndex,
        headIndex,
        stage: 'head-detail',
        role: 'x-ln-copy',
        ...(branchKey ? { branchKey } : {})
    };
    return createVectorStripMatrixNode({
        role: 'x-ln-copy',
        semantic,
        labelTex: branchLabel ? `X_{\\ln}^{${branchLabel}}` : 'X_{\\ln}',
        labelText: branchLabel ? `X_ln^${branchLabel}` : 'X_ln',
        visualStyleKey: VIEW2D_STYLE_KEYS.RESIDUAL,
        captionDimensionsTex: dimensionCaption.tex,
        captionDimensionsText: dimensionCaption.text,
        rowItems: buildResidualCopyRowItems(rows, {
            layerIndex,
            headIndex,
            branchKey
        }),
        rowCount,
        captionPosition: 'float-top',
        captionMinScreenHeightPx: isSmallScreen
            ? Math.max(18, DETAIL_LAYOUT.captionMinScreenHeightPx - 4)
            : DETAIL_LAYOUT.captionMinScreenHeightPx
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
        source: createAnchorRef(sourceNode.id, VIEW2D_ANCHOR_SIDES.CENTER),
        target: createAnchorRef(copyNode.id, VIEW2D_ANCHOR_SIDES.LEFT),
        route: VIEW2D_CONNECTOR_ROUTES.HORIZONTAL,
        gap: DETAIL_LAYOUT.connectorGap,
        visual: {
            styleKey: VIEW2D_STYLE_KEYS.CONNECTOR_NEUTRAL,
            stroke: 'rgba(255, 255, 255, 0.84)'
        },
        metadata: {
            preserveColor: true,
            strokeWidthScale: 0.72,
            arrowTipTouchTarget: true
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
            visualContract: 'selection-panel-mhsa-v1',
            source: 'buildHeadDetailSceneModel',
            rowCount,
            isSmallScreen: !!isSmallScreen,
            layoutMetrics: {
                cssVars: {
                    '--mhsa-token-matrix-canvas-pad-x-boost': isSmallScreen ? '-32px' : '-48px',
                    '--mhsa-token-matrix-canvas-pad-y-boost': isSmallScreen ? '-24px' : '-32px'
                }
            },
            tokens: visualTokens || resolveView2dVisualTokens()
        }
    });
}
