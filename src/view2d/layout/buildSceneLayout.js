import {
    VIEW2D_ANCHOR_SIDES,
    VIEW2D_CONNECTOR_ROUTES,
    VIEW2D_LAYOUT_DIRECTIONS,
    VIEW2D_MATRIX_PRESENTATIONS,
    VIEW2D_NODE_KINDS
} from '../schema/sceneTypes.js';
import { resolveView2dVisualTokens } from '../theme/visualTokens.js';
import {
    fitView2dText,
    measureView2dText
} from '../textMeasurement.js';
import {
    resolveView2dCaptionLines,
    resolveView2dCaptionMeasurementText,
    resolveView2dCaptionPosition
} from '../captionUtils.js';
import {
    inflateBounds,
    LayoutRegistry,
    unionBounds
} from './LayoutRegistry.js';

const CSS_VAR_NAMES = Object.freeze({
    canvasPadX: '--mhsa-token-matrix-canvas-pad-x-boost',
    canvasPadY: '--mhsa-token-matrix-canvas-pad-y-boost',
    stackRowGap: '--mhsa-token-matrix-stack-row-gap-boost',
    stackColumnGap: '--mhsa-token-matrix-stack-column-gap-boost',
    stageGap: '--mhsa-token-matrix-stage-gap-boost',
    projectionGap: '--mhsa-token-matrix-projection-gap-boost',
    attentionGap: '--mhsa-token-matrix-attention-flow-gap-boost',
    inlineGap: '--mhsa-token-matrix-inline-gap-boost',
    headOutputGap: '--mhsa-token-matrix-head-output-gap-boost',
    softmaxGap: '--mhsa-token-matrix-softmax-stage-gap-boost',
    headCopyOffset: '--mhsa-token-matrix-head-copy-offset-boost'
});

function parsePx(value, fallback = 0) {
    if (typeof value === 'number' && Number.isFinite(value)) return value;
    if (typeof value !== 'string') return fallback;
    const numeric = Number.parseFloat(value);
    return Number.isFinite(numeric) ? numeric : fallback;
}

function resolveTextMetrics(text = '', baseFontSize = 12, textFit = null) {
    return fitView2dText(text, {
        baseFontSize,
        maxWidth: textFit?.maxWidth ?? null
    });
}

function measureTextWidth(text = '', fontSize = 12) {
    return measureView2dText(text, { fontSize }).inkWidth;
}

function resolveMeasuredValue(value, fallback = null) {
    return Number.isFinite(value) && value > 0
        ? Math.max(1, Math.floor(value))
        : fallback;
}

function createAnchors(contentBounds) {
    if (!contentBounds) {
        return {
            [VIEW2D_ANCHOR_SIDES.LEFT]: { x: 0, y: 0 },
            [VIEW2D_ANCHOR_SIDES.RIGHT]: { x: 0, y: 0 },
            [VIEW2D_ANCHOR_SIDES.TOP]: { x: 0, y: 0 },
            [VIEW2D_ANCHOR_SIDES.BOTTOM]: { x: 0, y: 0 },
            [VIEW2D_ANCHOR_SIDES.CENTER]: { x: 0, y: 0 }
        };
    }
    const centerX = contentBounds.x + (contentBounds.width / 2);
    const centerY = contentBounds.y + (contentBounds.height / 2);
    return {
        [VIEW2D_ANCHOR_SIDES.LEFT]: { x: contentBounds.x, y: centerY },
        [VIEW2D_ANCHOR_SIDES.RIGHT]: { x: contentBounds.x + contentBounds.width, y: centerY },
        [VIEW2D_ANCHOR_SIDES.TOP]: { x: centerX, y: contentBounds.y },
        [VIEW2D_ANCHOR_SIDES.BOTTOM]: { x: centerX, y: contentBounds.y + contentBounds.height },
        [VIEW2D_ANCHOR_SIDES.CENTER]: { x: centerX, y: centerY }
    };
}

function resolveLayoutConfig(scene, {
    isSmallScreen = false,
    layoutMetrics = null,
    visualTokens = null
} = {}) {
    const metrics = layoutMetrics || scene?.metadata?.layoutMetrics || {};
    const cssVars = metrics?.cssVars || {};
    const tokens = visualTokens || scene?.metadata?.tokens || resolveView2dVisualTokens();

    const padXBoost = parsePx(cssVars[CSS_VAR_NAMES.canvasPadX]);
    const padYBoost = parsePx(cssVars[CSS_VAR_NAMES.canvasPadY]);
    const stackRowGapBoost = parsePx(cssVars[CSS_VAR_NAMES.stackRowGap]);
    const stackColumnGapBoost = parsePx(cssVars[CSS_VAR_NAMES.stackColumnGap]);
    const stageGapBoost = parsePx(cssVars[CSS_VAR_NAMES.stageGap]);
    const projectionGapBoost = parsePx(cssVars[CSS_VAR_NAMES.projectionGap]);
    const attentionGapBoost = parsePx(cssVars[CSS_VAR_NAMES.attentionGap]);
    const inlineGapBoost = parsePx(cssVars[CSS_VAR_NAMES.inlineGap]);
    const headOutputGapBoost = parsePx(cssVars[CSS_VAR_NAMES.headOutputGap]);
    const softmaxGapBoost = parsePx(cssVars[CSS_VAR_NAMES.softmaxGap]);
    const headCopyOffsetBoost = parsePx(cssVars[CSS_VAR_NAMES.headCopyOffset]);

    const component = {
        contentPaddingX: isSmallScreen ? 10 : 12,
        contentPaddingY: isSmallScreen ? 8 : 10,
        captionGap: isSmallScreen ? 8 : 10,
        captionLineHeight: isSmallScreen ? 12 : 14,
        captionFontSize: Math.max(10, tokens?.typography?.captionFontSize || 11),
        labelFontSize: Math.max(10, tokens?.typography?.labelFontSize || 12),
        operatorFontSize: Math.max(16, tokens?.typography?.operatorFontSize || 20),
        bandedRowHeight: isSmallScreen ? 18 : 22,
        bandedRowGap: isSmallScreen ? 6 : 8,
        compactRowHeight: isSmallScreen ? 12 : 16,
        compactRowGap: isSmallScreen ? 4 : 6,
        gridCellSize: isSmallScreen ? 12 : 14,
        gridCellGap: isSmallScreen ? 3 : 4,
        transposeColWidth: isSmallScreen ? 10 : 12,
        transposeColGap: isSmallScreen ? 4 : 6,
        transposeCellHeight: isSmallScreen ? 12 : 14,
        weightCardWidth: isSmallScreen ? 96 : 112,
        weightCardHeight: isSmallScreen ? 84 : 96,
        biasBarHeight: isSmallScreen ? 14 : 18,
        rowLabelGutterWidth: isSmallScreen ? 72 : 92,
        compactMinWidth: isSmallScreen ? 74 : 92,
        operatorSidePadding: isSmallScreen ? 8 : 10
    };

    return {
        scenePaddingX: (isSmallScreen ? 32 : 48) + padXBoost,
        scenePaddingY: (isSmallScreen ? 24 : 32) + padYBoost,
        rootGap: (isSmallScreen ? 40 : 56) + stackColumnGapBoost,
        groupGaps: {
            default: isSmallScreen ? 14 : 18,
            projection: (isSmallScreen ? 14 : 18) + projectionGapBoost,
            stackHorizontal: (isSmallScreen ? 40 : 56) + stackColumnGapBoost,
            stackVertical: (isSmallScreen ? 18 : 24) + stackRowGapBoost,
            stage: (isSmallScreen ? 16 : 20) + stageGapBoost,
            attention: (isSmallScreen ? 16 : 22) + attentionGapBoost,
            inline: (isSmallScreen ? 10 : 14) + inlineGapBoost,
            softmax: (isSmallScreen ? 14 : 18) + softmaxGapBoost,
            'head-output': (isSmallScreen ? 16 : 20) + headOutputGapBoost + headCopyOffsetBoost
        },
        component,
        tokens
    };
}

function resolveGroupGap(node, config) {
    const explicitGap = Number(node?.metadata?.gapOverride);
    if (Number.isFinite(explicitGap) && explicitGap >= 0) {
        return explicitGap;
    }
    const direction = node?.layout?.direction || VIEW2D_LAYOUT_DIRECTIONS.HORIZONTAL;
    const gapKey = String(node?.layout?.gapKey || 'default');
    if (gapKey === 'stack') {
        return direction === VIEW2D_LAYOUT_DIRECTIONS.VERTICAL
            ? config.groupGaps.stackVertical
            : config.groupGaps.stackHorizontal;
    }
    return config.groupGaps[gapKey] ?? config.groupGaps.default;
}

function measureLeafNode(node, config) {
    const captionLines = resolveView2dCaptionLines(node);
    const captionPosition = resolveView2dCaptionPosition(node);
    const shouldMeasureCaptionWidth = captionLines.length && captionPosition !== 'inside-top';
    const reserveCaptionSpace = captionLines.length && captionPosition === 'top';
    const captionWidth = shouldMeasureCaptionWidth
        ? Math.max(
            ...captionLines.map((line) => measureTextWidth(
                resolveView2dCaptionMeasurementText(line),
                config.component.captionFontSize
            ))
        )
        : 0;
    const captionHeight = reserveCaptionSpace
        ? (config.component.captionGap + (captionLines.length * config.component.captionLineHeight))
        : 0;
    let contentWidth = 0;
    let contentHeight = 0;
    let layoutData = {};

    if (node.kind === VIEW2D_NODE_KINDS.MATRIX) {
        const measuredRows = resolveMeasuredValue(node.metadata?.measure?.rows, null);
        const measuredCols = resolveMeasuredValue(node.metadata?.measure?.cols, null);
        switch (node.presentation) {
        case VIEW2D_MATRIX_PRESENTATIONS.BANDED_ROWS: {
            const rowCount = Math.max(1, measuredRows || node.rowItems?.length || node.dimensions?.rows || 1);
            const maxBarWidth = Math.max(148, (measuredCols || node.dimensions?.cols || 1) * 1.15);
            contentWidth = config.component.rowLabelGutterWidth + maxBarWidth + (config.component.contentPaddingX * 2);
            contentHeight = (rowCount * config.component.bandedRowHeight)
                + (Math.max(0, rowCount - 1) * config.component.bandedRowGap)
                + (config.component.contentPaddingY * 2);
            layoutData = {
                rowHeight: config.component.bandedRowHeight,
                rowGap: config.component.bandedRowGap,
                labelGutterWidth: config.component.rowLabelGutterWidth,
                innerPaddingX: config.component.contentPaddingX,
                innerPaddingY: config.component.contentPaddingY,
                barWidth: maxBarWidth
            };
            break;
        }
        case VIEW2D_MATRIX_PRESENTATIONS.COMPACT_ROWS: {
            const rowCount = Math.max(1, measuredRows || node.rowItems?.length || node.dimensions?.rows || 1);
            const compactRows = node.metadata?.compactRows || {};
            const compactWidthOverride = resolveMeasuredValue(compactRows.compactWidth, null);
            const rowHeight = resolveMeasuredValue(compactRows.rowHeight, config.component.compactRowHeight);
            const rowGap = Number.isFinite(compactRows.rowGap) && compactRows.rowGap >= 0
                ? Math.max(0, Math.floor(compactRows.rowGap))
                : config.component.compactRowGap;
            const innerPaddingX = Number.isFinite(compactRows.paddingX) && compactRows.paddingX >= 0
                ? Math.max(0, Math.floor(compactRows.paddingX))
                : config.component.contentPaddingX;
            const innerPaddingY = Number.isFinite(compactRows.paddingY) && compactRows.paddingY >= 0
                ? Math.max(0, Math.floor(compactRows.paddingY))
                : config.component.contentPaddingY;
            const compactWidth = compactWidthOverride
                ?? Math.max(
                    config.component.compactMinWidth,
                    (measuredCols || node.dimensions?.cols || 1) * config.component.gridCellSize
                );
            contentWidth = compactWidth + (innerPaddingX * 2);
            contentHeight = (rowCount * rowHeight)
                + (Math.max(0, rowCount - 1) * rowGap)
                + (innerPaddingY * 2);
            layoutData = {
                rowHeight,
                rowGap,
                innerPaddingX,
                innerPaddingY,
                compactWidth
            };
            break;
        }
        case VIEW2D_MATRIX_PRESENTATIONS.GRID: {
            const rowCount = Math.max(1, measuredRows || node.dimensions?.rows || 1);
            const colCount = Math.max(1, measuredCols || node.dimensions?.cols || 1);
            contentWidth = (colCount * config.component.gridCellSize)
                + (Math.max(0, colCount - 1) * config.component.gridCellGap)
                + (config.component.contentPaddingX * 2);
            contentHeight = (rowCount * config.component.gridCellSize)
                + (Math.max(0, rowCount - 1) * config.component.gridCellGap)
                + (config.component.contentPaddingY * 2);
            layoutData = {
                cellSize: config.component.gridCellSize,
                cellGap: config.component.gridCellGap,
                innerPaddingX: config.component.contentPaddingX,
                innerPaddingY: config.component.contentPaddingY
            };
            break;
        }
        case VIEW2D_MATRIX_PRESENTATIONS.COLUMN_STRIP: {
            const rowCount = Math.max(1, measuredRows || node.dimensions?.rows || 1);
            const colCount = Math.max(1, measuredCols || node.columnItems?.length || node.dimensions?.cols || 1);
            const columnStrip = node.metadata?.columnStrip || {};
            const colWidth = resolveMeasuredValue(columnStrip.colWidth, config.component.transposeColWidth);
            const colGap = Number.isFinite(columnStrip.colGap) && columnStrip.colGap >= 0
                ? Math.max(0, Math.floor(columnStrip.colGap))
                : config.component.transposeColGap;
            const innerPaddingX = Number.isFinite(columnStrip.paddingX) && columnStrip.paddingX >= 0
                ? Math.max(0, Math.floor(columnStrip.paddingX))
                : config.component.contentPaddingX;
            const innerPaddingY = Number.isFinite(columnStrip.paddingY) && columnStrip.paddingY >= 0
                ? Math.max(0, Math.floor(columnStrip.paddingY))
                : config.component.contentPaddingY;
            const colHeight = resolveMeasuredValue(
                columnStrip.colHeight,
                rowCount * config.component.transposeCellHeight
            );
            contentWidth = (colCount * colWidth)
                + (Math.max(0, colCount - 1) * colGap)
                + (innerPaddingX * 2);
            contentHeight = colHeight + (innerPaddingY * 2);
            layoutData = {
                colWidth,
                colGap,
                colHeight,
                innerPaddingX,
                innerPaddingY
            };
            break;
        }
        case VIEW2D_MATRIX_PRESENTATIONS.ACCENT_BAR:
            contentWidth = Math.max(
                56,
                (measuredCols || node.dimensions?.cols || 1) * (config.component.gridCellSize * 0.85)
            ) + (config.component.contentPaddingX * 2);
            contentHeight = config.component.biasBarHeight + (config.component.contentPaddingY * 2);
            layoutData = {
                barHeight: config.component.biasBarHeight,
                innerPaddingX: config.component.contentPaddingX,
                innerPaddingY: config.component.contentPaddingY
            };
            break;
        case VIEW2D_MATRIX_PRESENTATIONS.CARD:
            contentWidth = resolveMeasuredValue(node.metadata?.card?.width, config.component.weightCardWidth);
            contentHeight = resolveMeasuredValue(node.metadata?.card?.height, config.component.weightCardHeight);
            layoutData = {
                cardWidth: contentWidth,
                cardHeight: contentHeight,
                cardRadius: resolveMeasuredValue(node.metadata?.card?.cornerRadius, null)
            };
            break;
        default:
            contentWidth = 64;
            contentHeight = 64;
            layoutData = {};
            break;
        }
        const cardRadius = resolveMeasuredValue(node.metadata?.card?.cornerRadius, null);
        if (cardRadius !== null) {
            layoutData = {
                ...layoutData,
                cardRadius
            };
        }
    } else if (node.kind === VIEW2D_NODE_KINDS.TEXT) {
        const textMetrics = resolveTextMetrics(
            node.text || node.tex,
            config.component.labelFontSize,
            node.metadata?.textFit || null
        );
        contentWidth = textMetrics.width;
        contentHeight = textMetrics.height;
        layoutData = {
            fontSize: textMetrics.fontSize,
            maxWidth: textMetrics.maxWidth,
            paddingX: textMetrics.paddingX
        };
    } else if (node.kind === VIEW2D_NODE_KINDS.OPERATOR) {
        const operatorScale = Number.isFinite(node.metadata?.fontScale) && node.metadata.fontScale > 0
            ? Number(node.metadata.fontScale)
            : 1;
        const operatorFontSize = config.component.operatorFontSize * operatorScale;
        const rawWidth = measureTextWidth(node.text || '', operatorFontSize);
        contentWidth = rawWidth + (config.component.operatorSidePadding * 2);
        contentHeight = operatorFontSize * 1.45;
        layoutData = {
            fontSize: operatorFontSize
        };
    }

    const totalWidth = Math.max(contentWidth, captionWidth);
    const totalHeight = contentHeight + captionHeight;

    return {
        width: totalWidth,
        height: totalHeight,
        contentWidth,
        contentHeight,
        captionLines,
        captionPosition,
        captionWidth,
        captionHeight,
        layoutData
    };
}

function measureNode(node, config, cache) {
    if (!node || typeof node !== 'object') {
        return {
            width: 0,
            height: 0,
            contentWidth: 0,
            contentHeight: 0,
            captionLines: [],
            captionPosition: 'bottom',
            captionWidth: 0,
            captionHeight: 0,
            layoutData: {},
            childMeasurements: []
        };
    }
    if (cache.has(node.id)) {
        return cache.get(node.id);
    }
    let measurement;
    if (node.kind === VIEW2D_NODE_KINDS.GROUP) {
        const childMeasurements = Array.isArray(node.children)
            ? node.children.map((child) => measureNode(child, config, cache))
            : [];
        const direction = node.layout?.direction || VIEW2D_LAYOUT_DIRECTIONS.HORIZONTAL;
        const gap = resolveGroupGap(node, config);
        if (direction === VIEW2D_LAYOUT_DIRECTIONS.VERTICAL) {
            measurement = {
                width: childMeasurements.length ? Math.max(...childMeasurements.map((child) => child.width)) : 0,
                height: childMeasurements.reduce((acc, child) => acc + child.height, 0)
                    + (Math.max(0, childMeasurements.length - 1) * gap),
                contentWidth: childMeasurements.length ? Math.max(...childMeasurements.map((child) => child.width)) : 0,
                contentHeight: childMeasurements.reduce((acc, child) => acc + child.height, 0)
                    + (Math.max(0, childMeasurements.length - 1) * gap),
                captionLines: [],
                captionPosition: 'bottom',
                captionWidth: 0,
                captionHeight: 0,
                layoutData: {
                    gap,
                    direction,
                    align: node.layout?.align || 'center'
                },
                childMeasurements
            };
        } else if (direction === VIEW2D_LAYOUT_DIRECTIONS.OVERLAY) {
            measurement = {
                width: childMeasurements.length ? Math.max(...childMeasurements.map((child) => child.width)) : 0,
                height: childMeasurements.length ? Math.max(...childMeasurements.map((child) => child.height)) : 0,
                contentWidth: childMeasurements.length ? Math.max(...childMeasurements.map((child) => child.width)) : 0,
                contentHeight: childMeasurements.length ? Math.max(...childMeasurements.map((child) => child.height)) : 0,
                captionLines: [],
                captionPosition: 'bottom',
                captionWidth: 0,
                captionHeight: 0,
                layoutData: {
                    gap,
                    direction,
                    align: node.layout?.align || 'center'
                },
                childMeasurements
            };
        } else {
            measurement = {
                width: childMeasurements.reduce((acc, child) => acc + child.width, 0)
                    + (Math.max(0, childMeasurements.length - 1) * gap),
                height: childMeasurements.length ? Math.max(...childMeasurements.map((child) => child.height)) : 0,
                contentWidth: childMeasurements.reduce((acc, child) => acc + child.width, 0)
                    + (Math.max(0, childMeasurements.length - 1) * gap),
                contentHeight: childMeasurements.length ? Math.max(...childMeasurements.map((child) => child.height)) : 0,
                captionLines: [],
                captionPosition: 'bottom',
                captionWidth: 0,
                captionHeight: 0,
                layoutData: {
                    gap,
                    direction,
                    align: node.layout?.align || 'center'
                },
                childMeasurements
            };
        }
    } else if (node.kind === VIEW2D_NODE_KINDS.CONNECTOR) {
        measurement = {
            width: 0,
            height: 0,
            contentWidth: 0,
            contentHeight: 0,
            captionLines: [],
            captionPosition: 'bottom',
            captionWidth: 0,
            captionHeight: 0,
            layoutData: {},
            childMeasurements: []
        };
    } else {
        measurement = {
            ...measureLeafNode(node, config),
            childMeasurements: []
        };
    }
    cache.set(node.id, measurement);
    return measurement;
}

function registerNodeEntry(registry, node, {
    bounds,
    contentBounds,
    labelBounds = null,
    dimensionBounds = null,
    depth = 0,
    parentId = null,
    layoutData = null
} = {}) {
    registry.setNodeEntry(node.id, {
        kind: node.kind,
        role: node.role,
        parentId,
        depth,
        bounds,
        contentBounds,
        labelBounds,
        dimensionBounds,
        anchors: createAnchors(contentBounds || bounds),
        semantic: node.semantic || null,
        layoutData,
        metadata: node.metadata || null
    });
}

function placeNode(node, x, y, measurement, registry, config, depth = 0, parentId = null) {
    if (node.kind === VIEW2D_NODE_KINDS.CONNECTOR) {
        return {
            bounds: { x, y, width: 0, height: 0 },
            contentBounds: { x, y, width: 0, height: 0 }
        };
    }

    if (node.kind === VIEW2D_NODE_KINDS.GROUP) {
        const bounds = { x, y, width: measurement.width, height: measurement.height };
        const contentBounds = { ...bounds };
        registerNodeEntry(registry, node, {
            bounds,
            contentBounds,
            depth,
            parentId,
            layoutData: measurement.layoutData
        });

        const direction = node.layout?.direction || VIEW2D_LAYOUT_DIRECTIONS.HORIZONTAL;
        const gap = measurement.layoutData?.gap || 0;
        const align = measurement.layoutData?.align || 'center';
        if (direction === VIEW2D_LAYOUT_DIRECTIONS.VERTICAL) {
            let cursorY = y;
            node.children.forEach((child, index) => {
                const childMeasurement = measurement.childMeasurements[index];
                const childX = align === 'start'
                    ? x
                    : (align === 'end'
                        ? x + (measurement.width - childMeasurement.width)
                        : x + ((measurement.width - childMeasurement.width) / 2));
                placeNode(child, childX, cursorY, childMeasurement, registry, config, depth + 1, node.id);
                cursorY += childMeasurement.height + gap;
            });
        } else if (direction === VIEW2D_LAYOUT_DIRECTIONS.OVERLAY) {
            node.children.forEach((child, index) => {
                const childMeasurement = measurement.childMeasurements[index];
                const childX = align === 'start'
                    ? x
                    : (align === 'end'
                        ? x + (measurement.width - childMeasurement.width)
                        : x + ((measurement.width - childMeasurement.width) / 2));
                const childY = align === 'top'
                    ? y
                    : (align === 'bottom'
                        ? y + (measurement.height - childMeasurement.height)
                        : y + ((measurement.height - childMeasurement.height) / 2));
                placeNode(child, childX, childY, childMeasurement, registry, config, depth + 1, node.id);
            });
        } else {
            let cursorX = x;
            node.children.forEach((child, index) => {
                const childMeasurement = measurement.childMeasurements[index];
                const childY = align === 'start'
                    ? y
                    : (align === 'end'
                        ? y + (measurement.height - childMeasurement.height)
                        : y + ((measurement.height - childMeasurement.height) / 2));
                placeNode(child, cursorX, childY, childMeasurement, registry, config, depth + 1, node.id);
                cursorX += childMeasurement.width + gap;
            });
        }
        return { bounds, contentBounds };
    }

    const contentX = x + ((measurement.width - measurement.contentWidth) / 2);
    const contentY = measurement.captionPosition === 'top'
        ? y + measurement.captionHeight
        : y;
    const contentBounds = {
        x: contentX,
        y: contentY,
        width: measurement.contentWidth,
        height: measurement.contentHeight
    };
    const bounds = {
        x,
        y,
        width: measurement.width,
        height: measurement.height
    };
    let labelBounds = null;
    let dimensionBounds = null;
    if (measurement.captionLines.length && (measurement.captionHeight > 0 || measurement.captionPosition === 'float-top')) {
        const lineHeight = config.component.captionLineHeight;
        const floatTopGap = Math.max(2, Math.round(config.component.captionGap * 0.4));
        const captionY = measurement.captionPosition === 'top'
            ? y
            : (measurement.captionPosition === 'float-top'
                ? contentBounds.y - floatTopGap - (measurement.captionLines.length * lineHeight)
                : y + measurement.contentHeight + config.component.captionGap);
        labelBounds = {
            x: x + ((measurement.width - measurement.captionWidth) / 2),
            y: captionY,
            width: measurement.captionWidth,
            height: lineHeight
        };
        if (measurement.captionLines.length > 1) {
            dimensionBounds = {
                x: labelBounds.x,
                y: labelBounds.y + lineHeight,
                width: labelBounds.width,
                height: lineHeight
            };
        }
    }

    registerNodeEntry(registry, node, {
        bounds,
        contentBounds,
        labelBounds,
        dimensionBounds,
        depth,
        parentId,
        layoutData: measurement.layoutData
    });

    return { bounds, contentBounds };
}

function pointBounds(points = []) {
    if (!Array.isArray(points) || !points.length) {
        return { x: 0, y: 0, width: 0, height: 0 };
    }
    const minX = Math.min(...points.map((point) => point.x));
    const minY = Math.min(...points.map((point) => point.y));
    const maxX = Math.max(...points.map((point) => point.x));
    const maxY = Math.max(...points.map((point) => point.y));
    return {
        x: minX,
        y: minY,
        width: Math.max(0, maxX - minX),
        height: Math.max(0, maxY - minY)
    };
}

function offsetPoint(point, anchor, gap = 0) {
    const safeGap = Number.isFinite(gap) ? Math.max(0, gap) : 0;
    switch (anchor) {
    case VIEW2D_ANCHOR_SIDES.LEFT:
        return { x: point.x - safeGap, y: point.y };
    case VIEW2D_ANCHOR_SIDES.RIGHT:
        return { x: point.x + safeGap, y: point.y };
    case VIEW2D_ANCHOR_SIDES.TOP:
        return { x: point.x, y: point.y - safeGap };
    case VIEW2D_ANCHOR_SIDES.BOTTOM:
        return { x: point.x, y: point.y + safeGap };
    default:
        return { ...point };
    }
}

function buildConnectorPath(
    sourcePoint,
    targetPoint,
    sourceAnchor,
    targetAnchor,
    sourceGap = 0,
    targetGap = 0,
    route = 'horizontal'
) {
    const leadSource = offsetPoint(sourcePoint, sourceAnchor, sourceGap);
    const leadTarget = offsetPoint(targetPoint, targetAnchor, targetGap);
    const dx = Math.abs(leadTarget.x - leadSource.x);
    const dy = Math.abs(leadTarget.y - leadSource.y);
    if (dx < 0.5 || dy < 0.5) {
        return [
            leadSource,
            leadTarget
        ];
    }
    if (route === VIEW2D_CONNECTOR_ROUTES.ELBOW) {
        const elbowPoint = (sourceAnchor === VIEW2D_ANCHOR_SIDES.TOP || sourceAnchor === VIEW2D_ANCHOR_SIDES.BOTTOM)
            ? { x: leadSource.x, y: leadTarget.y }
            : { x: leadTarget.x, y: leadSource.y };
        return [
            leadSource,
            elbowPoint,
            leadTarget
        ];
    }
    if (targetAnchor === VIEW2D_ANCHOR_SIDES.TOP || targetAnchor === VIEW2D_ANCHOR_SIDES.BOTTOM) {
        return [
            leadSource,
            { x: leadTarget.x, y: leadSource.y },
            leadTarget
        ];
    }
    if (route === 'vertical') {
        const midY = (leadSource.y + leadTarget.y) / 2;
        return [
            leadSource,
            { x: leadSource.x, y: midY },
            { x: leadTarget.x, y: midY },
            leadTarget
        ];
    }
    const midX = (leadSource.x + leadTarget.x) / 2;
    return [
        leadSource,
        { x: midX, y: leadSource.y },
        { x: midX, y: leadTarget.y },
        leadTarget
    ];
}

export function buildSceneLayout(scene, {
    isSmallScreen = false,
    layoutMetrics = null,
    visualTokens = null
} = {}) {
    if (!scene || !Array.isArray(scene.nodes)) {
        return null;
    }

    const registry = new LayoutRegistry();
    const config = resolveLayoutConfig(scene, {
        isSmallScreen,
        layoutMetrics,
        visualTokens
    });
    const cache = new Map();

    const rootNodes = scene.nodes.filter((node) => node?.kind !== VIEW2D_NODE_KINDS.CONNECTOR);
    const measuredRoots = rootNodes.map((node) => ({
        node,
        measurement: measureNode(node, config, cache)
    }));
    const regularRoots = measuredRoots.filter(({ node }) => node?.layout?.direction !== VIEW2D_LAYOUT_DIRECTIONS.OVERLAY);
    const overlayRoots = measuredRoots.filter(({ node }) => node?.layout?.direction === VIEW2D_LAYOUT_DIRECTIONS.OVERLAY);

    const rootWidth = regularRoots.reduce((acc, entry) => acc + entry.measurement.width, 0)
        + (Math.max(0, regularRoots.length - 1) * config.rootGap);
    const rootHeight = regularRoots.length
        ? Math.max(...regularRoots.map((entry) => entry.measurement.height))
        : 0;
    const sceneBounds = {
        x: 0,
        y: 0,
        width: rootWidth + (config.scenePaddingX * 2),
        height: Math.max(rootHeight, ...overlayRoots.map((entry) => entry.measurement.height), 0) + (config.scenePaddingY * 2)
    };
    registry.setSceneBounds(sceneBounds);

    let cursorX = config.scenePaddingX;
    regularRoots.forEach(({ node, measurement }) => {
        const nodeY = config.scenePaddingY + ((rootHeight - measurement.height) / 2);
        placeNode(node, cursorX, nodeY, measurement, registry, config, 0, null);
        cursorX += measurement.width + config.rootGap;
    });

    overlayRoots.forEach(({ node, measurement }) => {
        const overlayX = config.scenePaddingX;
        const overlayY = config.scenePaddingY + ((rootHeight - measurement.height) / 2);
        placeNode(node, overlayX, overlayY, measurement, registry, config, 0, null);
    });

    scene.nodes.forEach((node) => {
        if (node?.kind !== VIEW2D_NODE_KINDS.GROUP || !Array.isArray(node.children)) return;
        node.children.forEach((child) => {
            if (child?.kind !== VIEW2D_NODE_KINDS.CONNECTOR) return;
            const sourcePoint = registry.resolveAnchor(child.source);
            const targetPoint = registry.resolveAnchor(child.target);
            if (!sourcePoint || !targetPoint) return;
            const pathPoints = buildConnectorPath(
                sourcePoint,
                targetPoint,
                child.source?.anchor,
                child.target?.anchor,
                child.sourceGap ?? child.gap ?? 0,
                child.targetGap ?? child.gap ?? 0,
                child.route
            );
            registry.setConnectorEntry(child.id, {
                role: child.role,
                source: child.source,
                target: child.target,
                bounds: inflateBounds(pointBounds(pathPoints), 2),
                pathPoints,
                semantic: child.semantic || null,
                metadata: child.metadata || null
            });
        });
    });

    return {
        registry,
        sceneBounds: registry.getSceneBounds(),
        config,
        contentBounds: unionBounds(registry.getNodeEntries().map((entry) => entry.bounds))
    };
}
