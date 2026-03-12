import { resolveRenderPixelRatio } from '../../../utils/constants.js';
import {
    flattenSceneNodes,
    VIEW2D_MATRIX_PRESENTATIONS,
    VIEW2D_NODE_KINDS
} from '../../schema/sceneTypes.js';
import { buildSceneLayout } from '../../layout/buildSceneLayout.js';
import { resolveView2dStyle, resolveView2dVisualTokens } from '../../theme/visualTokens.js';

function clampPositive(value, fallback = 1) {
    return Number.isFinite(value) && value > 0 ? value : fallback;
}

function roundRectPath(ctx, x, y, width, height, radius = 8) {
    const r = Math.max(0, Math.min(radius, width / 2, height / 2));
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.arcTo(x + width, y, x + width, y + height, r);
    ctx.arcTo(x + width, y + height, x, y + height, r);
    ctx.arcTo(x, y + height, x, y, r);
    ctx.arcTo(x, y, x + width, y, r);
    ctx.closePath();
}

function splitTopLevel(input = '') {
    const parts = [];
    let depth = 0;
    let current = '';
    for (let index = 0; index < input.length; index += 1) {
        const char = input[index];
        if (char === '(') {
            depth += 1;
            current += char;
            continue;
        }
        if (char === ')') {
            depth = Math.max(0, depth - 1);
            current += char;
            continue;
        }
        if (char === ',' && depth === 0) {
            parts.push(current.trim());
            current = '';
            continue;
        }
        current += char;
    }
    if (current.trim().length) {
        parts.push(current.trim());
    }
    return parts;
}

function isCanvasColor(value = '') {
    return typeof value === 'string'
        && value.length > 0
        && !value.includes('gradient(');
}

const PARSED_LINEAR_GRADIENT_CACHE = new Map();
const CAPTION_MIN_SCREEN_HEIGHT_PX = 18;
const TEXT_MIN_SCREEN_HEIGHT_PX = 10;
const MATRIX_DETAIL_MIN_SCREEN_WIDTH_PX = 72;
const MATRIX_DETAIL_MIN_SCREEN_HEIGHT_PX = 24;

function parseLinearGradient(input = '') {
    const key = String(input || '');
    if (PARSED_LINEAR_GRADIENT_CACHE.has(key)) {
        return PARSED_LINEAR_GRADIENT_CACHE.get(key);
    }
    const match = key.match(/^linear-gradient\((.+)\)$/i);
    if (!match) return null;
    const tokens = splitTopLevel(match[1]);
    if (tokens.length < 2) return null;
    const angleToken = tokens[0];
    const angle = angleToken.endsWith('deg') ? Number.parseFloat(angleToken) : 90;
    const stops = tokens.slice(1).map((token) => {
        const stopMatch = token.match(/^(.*)\s+([0-9.]+)%$/);
        if (!stopMatch) {
            return {
                color: token.trim(),
                offset: null
            };
        }
        return {
            color: stopMatch[1].trim(),
            offset: Math.max(0, Math.min(1, Number.parseFloat(stopMatch[2]) / 100))
        };
    }).filter((stop) => stop.color.length);
    if (!stops.length) return null;
    const parsed = {
        angle: Number.isFinite(angle) ? angle : 90,
        stops
    };
    if (PARSED_LINEAR_GRADIENT_CACHE.size > 4096) {
        PARSED_LINEAR_GRADIENT_CACHE.clear();
    }
    PARSED_LINEAR_GRADIENT_CACHE.set(key, parsed);
    return parsed;
}

function createCanvasGradient(ctx, fillStyle, bounds, fallbackColor) {
    const gradient = parseLinearGradient(fillStyle);
    if (!gradient || !bounds) return null;
    const radians = (gradient.angle - 90) * (Math.PI / 180);
    const centerX = bounds.x + (bounds.width / 2);
    const centerY = bounds.y + (bounds.height / 2);
    const radius = Math.max(bounds.width, bounds.height) / 2;
    const dx = Math.cos(radians) * radius;
    const dy = Math.sin(radians) * radius;
    const canvasGradient = ctx.createLinearGradient(centerX - dx, centerY - dy, centerX + dx, centerY + dy);
    const denominator = Math.max(1, gradient.stops.length - 1);
    gradient.stops.forEach((stop, index) => {
        canvasGradient.addColorStop(
            stop.offset ?? (index / denominator),
            isCanvasColor(stop.color) ? stop.color : fallbackColor
        );
    });
    return canvasGradient;
}

function resolveFill(ctx, fillStyle, bounds, fallbackColor) {
    if (isCanvasColor(fillStyle)) return fillStyle;
    const gradient = createCanvasGradient(ctx, fillStyle, bounds, fallbackColor);
    return gradient || fallbackColor;
}

function cloneBounds(bounds = null) {
    if (!bounds || typeof bounds !== 'object') return null;
    return {
        x: Number.isFinite(bounds.x) ? bounds.x : 0,
        y: Number.isFinite(bounds.y) ? bounds.y : 0,
        width: Number.isFinite(bounds.width) ? bounds.width : 0,
        height: Number.isFinite(bounds.height) ? bounds.height : 0
    };
}

function intersectsBounds(a = null, b = null) {
    if (!a || !b) return false;
    const aRight = (Number(a.x) || 0) + Math.max(0, Number(a.width) || 0);
    const aBottom = (Number(a.y) || 0) + Math.max(0, Number(a.height) || 0);
    const bRight = (Number(b.x) || 0) + Math.max(0, Number(b.width) || 0);
    const bBottom = (Number(b.y) || 0) + Math.max(0, Number(b.height) || 0);
    return aRight >= (Number(b.x) || 0)
        && bRight >= (Number(a.x) || 0)
        && aBottom >= (Number(b.y) || 0)
        && bBottom >= (Number(a.y) || 0);
}

function resolveVisibleWorldBounds(resolution, {
    offsetX = 0,
    offsetY = 0,
    worldScale = 1,
    sceneBounds = null
} = {}) {
    const safeScale = Math.max(0.0001, Number.isFinite(worldScale) ? worldScale : 1);
    if (!resolution || !sceneBounds) return cloneBounds(sceneBounds);
    const overscan = 96 / safeScale;
    return {
        x: ((-offsetX) / safeScale) - overscan,
        y: ((-offsetY) / safeScale) - overscan,
        width: (resolution.width / safeScale) + (overscan * 2),
        height: (resolution.height / safeScale) + (overscan * 2)
    };
}

function drawDebugOverlay(ctx, resolution, debugState = {}) {
    if (!ctx || !resolution) return;
    const width = Math.max(1, Number(resolution.width) || 1);
    const height = Math.max(1, Number(resolution.height) || 1);
    const hasError = typeof debugState.error === 'string' && debugState.error.length > 0;
    const frameColor = hasError ? 'rgba(255, 96, 96, 0.98)' : 'rgba(54, 255, 214, 0.96)';
    const sceneRectColor = hasError ? 'rgba(255, 140, 140, 0.92)' : 'rgba(110, 176, 255, 0.92)';

    ctx.save();
    ctx.globalAlpha = 1;
    ctx.globalCompositeOperation = 'source-over';
    ctx.filter = 'none';
    ctx.shadowBlur = 0;
    ctx.shadowColor = 'transparent';

    ctx.strokeStyle = frameColor;
    ctx.lineWidth = 1.5;
    ctx.strokeRect(1.5, 1.5, Math.max(1, width - 3), Math.max(1, height - 3));

    ctx.beginPath();
    ctx.moveTo(10, 10);
    ctx.lineTo(34, 10);
    ctx.moveTo(10, 10);
    ctx.lineTo(10, 34);
    ctx.stroke();

    if (Number.isFinite(debugState.offsetX) && Number.isFinite(debugState.offsetY)
        && Number.isFinite(debugState.fittedWidth) && Number.isFinite(debugState.fittedHeight)
    ) {
        ctx.strokeStyle = sceneRectColor;
        ctx.lineWidth = 1;
        ctx.strokeRect(
            debugState.offsetX,
            debugState.offsetY,
            Math.max(1, debugState.fittedWidth),
            Math.max(1, debugState.fittedHeight)
        );
    }

    const lines = [
        `canvas ${Math.round(width)}x${Math.round(height)} dpr ${Number(debugState.dpr || 0).toFixed(2)}`,
        `scene ${Math.round(debugState.sceneBounds?.width || 0)}x${Math.round(debugState.sceneBounds?.height || 0)}`,
        `scale ${Number(debugState.worldScale || 0).toFixed(4)} offset ${Math.round(debugState.offsetX || 0)},${Math.round(debugState.offsetY || 0)}`,
        `nodes ${Math.round(debugState.nodeCount || 0)} connectors ${Math.round(debugState.connectorCount || 0)}`
    ];
    if (hasError) {
        lines.push(`error ${debugState.error}`);
    }

    const paddingX = 14;
    const lineHeight = 14;
    const textBlockHeight = (lines.length * lineHeight) + 12;
    ctx.fillStyle = 'rgba(0, 0, 0, 0.72)';
    ctx.fillRect(12, 12, Math.min(width - 24, 420), textBlockHeight);
    ctx.font = '11px ui-monospace, SFMono-Regular, Menlo, monospace';
    ctx.textAlign = 'left';
    ctx.textBaseline = 'top';
    ctx.fillStyle = 'rgba(238, 243, 251, 0.96)';
    lines.forEach((line, index) => {
        ctx.fillText(String(line), paddingX, 18 + (index * lineHeight));
    });
    ctx.restore();
}

export function syncCanvasResolution(canvas, {
    width = null,
    height = null,
    dpr = null,
    dprCap = null
} = {}) {
    if (!canvas) return null;
    const rect = typeof canvas.getBoundingClientRect === 'function'
        ? canvas.getBoundingClientRect()
        : null;
    const logicalWidth = clampPositive(width ?? rect?.width ?? canvas.clientWidth ?? canvas.width ?? 1, 1);
    const logicalHeight = clampPositive(height ?? rect?.height ?? canvas.clientHeight ?? canvas.height ?? 1, 1);
    const nextDpr = clampPositive(
        dpr ?? resolveRenderPixelRatio({
            viewportWidth: logicalWidth,
            viewportHeight: logicalHeight,
            dprCap
        }),
        1
    );

    const pixelWidth = Math.max(1, Math.round(logicalWidth * nextDpr));
    const pixelHeight = Math.max(1, Math.round(logicalHeight * nextDpr));
    if (canvas.width !== pixelWidth) {
        canvas.width = pixelWidth;
    }
    if (canvas.height !== pixelHeight) {
        canvas.height = pixelHeight;
    }
    return {
        width: logicalWidth,
        height: logicalHeight,
        dpr: nextDpr,
        pixelWidth,
        pixelHeight
    };
}

function drawCaption(ctx, entry, node, config, worldScale = 1) {
    const lines = [];
    if (entry.labelBounds) {
        lines.push({
            text: node.label?.text || node.label?.tex || '',
            bounds: entry.labelBounds
        });
    }
    if (entry.dimensionBounds && Number.isFinite(node.dimensions?.rows) && Number.isFinite(node.dimensions?.cols)) {
        lines.push({
            text: `(${node.dimensions.rows}, ${node.dimensions.cols})`,
            bounds: entry.dimensionBounds
        });
    }
    if (!lines.length) return;
    const maxHeight = Math.max(...lines.map(({ bounds }) => Number(bounds?.height) || 0));
    if ((maxHeight * Math.max(0.0001, worldScale)) < CAPTION_MIN_SCREEN_HEIGHT_PX) {
        return;
    }
    ctx.save();
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.font = `500 ${config.component.captionFontSize}px ui-monospace, SFMono-Regular, Menlo, monospace`;
    ctx.fillStyle = config.tokens.palette.mutedText;
    lines.forEach(({ text, bounds }) => {
        ctx.fillText(text, bounds.x + (bounds.width / 2), bounds.y + (bounds.height / 2));
    });
    ctx.restore();
}

function drawMatrixNode(ctx, node, entry, config, worldScale = 1) {
    const contentBounds = entry.contentBounds || entry.bounds;
    const style = resolveView2dStyle(node.visual?.styleKey) || {};
    const accent = style.accent || config.tokens.palette.neutral;
    const background = config.tokens.palette.panelBackground;
    const border = config.tokens.palette.border;
    const projectedWidth = contentBounds.width * Math.max(0.0001, worldScale);
    const projectedHeight = contentBounds.height * Math.max(0.0001, worldScale);
    const useSummaryInterior = projectedWidth < MATRIX_DETAIL_MIN_SCREEN_WIDTH_PX
        || projectedHeight < MATRIX_DETAIL_MIN_SCREEN_HEIGHT_PX;

    ctx.save();
    roundRectPath(ctx, contentBounds.x, contentBounds.y, contentBounds.width, contentBounds.height, config.tokens.matrix.cornerRadius);
    ctx.fillStyle = background;
    ctx.fill();
    ctx.lineWidth = config.tokens.matrix.borderWidth;
    ctx.strokeStyle = border;
    ctx.stroke();

    const layoutData = entry.layoutData || {};
    const rowItems = Array.isArray(node.rowItems) ? node.rowItems : [];
    const columnItems = Array.isArray(node.columnItems) ? node.columnItems : [];
    if (node.presentation === VIEW2D_MATRIX_PRESENTATIONS.BANDED_ROWS) {
        if (useSummaryInterior) {
            const barBounds = {
                x: contentBounds.x + layoutData.innerPaddingX,
                y: contentBounds.y + Math.max(layoutData.innerPaddingY, contentBounds.height * 0.28),
                width: Math.max(1, contentBounds.width - (layoutData.innerPaddingX * 2)),
                height: Math.max(3, Math.min(contentBounds.height * 0.38, contentBounds.height - (layoutData.innerPaddingY * 2)))
            };
            roundRectPath(ctx, barBounds.x, barBounds.y, barBounds.width, barBounds.height, 6);
            ctx.fillStyle = accent;
            ctx.fill();
        } else {
            const showRowLabels = (layoutData.rowHeight * Math.max(0.0001, worldScale)) >= CAPTION_MIN_SCREEN_HEIGHT_PX;
            rowItems.forEach((rowItem, index) => {
                const rowY = contentBounds.y + layoutData.innerPaddingY + index * (layoutData.rowHeight + layoutData.rowGap);
                const barX = contentBounds.x + layoutData.innerPaddingX + layoutData.labelGutterWidth;
                const barBounds = {
                    x: barX,
                    y: rowY,
                    width: layoutData.barWidth,
                    height: layoutData.rowHeight
                };
                roundRectPath(ctx, barBounds.x, barBounds.y, barBounds.width, barBounds.height, 6);
                ctx.fillStyle = resolveFill(ctx, rowItem.gradientCss, barBounds, accent);
                ctx.fill();

                if (showRowLabels) {
                    ctx.fillStyle = config.tokens.palette.mutedText;
                    ctx.font = `500 ${config.component.captionFontSize}px ui-monospace, SFMono-Regular, Menlo, monospace`;
                    ctx.textAlign = 'left';
                    ctx.textBaseline = 'middle';
                    ctx.fillText(
                        rowItem.label || '',
                        contentBounds.x + layoutData.innerPaddingX,
                        rowY + (layoutData.rowHeight / 2)
                    );
                }
            });
        }
    } else if (node.presentation === VIEW2D_MATRIX_PRESENTATIONS.COMPACT_ROWS) {
        if (useSummaryInterior) {
            const rowBounds = {
                x: contentBounds.x + layoutData.innerPaddingX,
                y: contentBounds.y + layoutData.innerPaddingY,
                width: Math.max(1, Math.min(layoutData.compactWidth, contentBounds.width - (layoutData.innerPaddingX * 2))),
                height: Math.max(3, Math.min(contentBounds.height - (layoutData.innerPaddingY * 2), contentBounds.height * 0.46))
            };
            roundRectPath(ctx, rowBounds.x, rowBounds.y, rowBounds.width, rowBounds.height, 5);
            ctx.fillStyle = accent;
            ctx.fill();
        } else {
            rowItems.forEach((rowItem, index) => {
                const rowY = contentBounds.y + layoutData.innerPaddingY + index * (layoutData.rowHeight + layoutData.rowGap);
                const rowBounds = {
                    x: contentBounds.x + layoutData.innerPaddingX,
                    y: rowY,
                    width: layoutData.compactWidth,
                    height: layoutData.rowHeight
                };
                roundRectPath(ctx, rowBounds.x, rowBounds.y, rowBounds.width, rowBounds.height, 5);
                ctx.fillStyle = resolveFill(ctx, rowItem.gradientCss, rowBounds, accent);
                ctx.fill();
            });
        }
    } else if (node.presentation === VIEW2D_MATRIX_PRESENTATIONS.GRID) {
        rowItems.forEach((rowItem, rowIndex) => {
            const cells = Array.isArray(rowItem.cells) ? rowItem.cells : [];
            cells.forEach((cellItem, colIndex) => {
                const cellBounds = {
                    x: contentBounds.x + layoutData.innerPaddingX + colIndex * (layoutData.cellSize + layoutData.cellGap),
                    y: contentBounds.y + layoutData.innerPaddingY + rowIndex * (layoutData.cellSize + layoutData.cellGap),
                    width: layoutData.cellSize,
                    height: layoutData.cellSize
                };
                ctx.fillStyle = resolveFill(ctx, cellItem.fillCss, cellBounds, cellItem.isMasked ? '#050608' : accent);
                ctx.fillRect(cellBounds.x, cellBounds.y, cellBounds.width, cellBounds.height);
            });
        });
    } else if (node.presentation === VIEW2D_MATRIX_PRESENTATIONS.COLUMN_STRIP) {
        columnItems.forEach((columnItem, index) => {
            const colBounds = {
                x: contentBounds.x + layoutData.innerPaddingX + index * (layoutData.colWidth + layoutData.colGap),
                y: contentBounds.y + layoutData.innerPaddingY,
                width: layoutData.colWidth,
                height: layoutData.colHeight
            };
            roundRectPath(ctx, colBounds.x, colBounds.y, colBounds.width, colBounds.height, 5);
            ctx.fillStyle = resolveFill(ctx, columnItem.fillCss, colBounds, accent);
            ctx.fill();
        });
    } else if (node.presentation === VIEW2D_MATRIX_PRESENTATIONS.ACCENT_BAR) {
        const barBounds = {
            x: contentBounds.x + layoutData.innerPaddingX,
            y: contentBounds.y + layoutData.innerPaddingY,
            width: Math.max(1, contentBounds.width - (layoutData.innerPaddingX * 2)),
            height: layoutData.barHeight
        };
        roundRectPath(ctx, barBounds.x, barBounds.y, barBounds.width, barBounds.height, 6);
        ctx.fillStyle = resolveFill(ctx, node.visual?.background, barBounds, accent);
        ctx.fill();
    } else if (node.presentation === VIEW2D_MATRIX_PRESENTATIONS.CARD) {
        const cardGradient = ctx.createLinearGradient(contentBounds.x, contentBounds.y, contentBounds.x + contentBounds.width, contentBounds.y + contentBounds.height);
        cardGradient.addColorStop(0, 'rgba(255,255,255,0.08)');
        cardGradient.addColorStop(0.5, accent);
        cardGradient.addColorStop(1, 'rgba(0,0,0,0.45)');
        ctx.fillStyle = cardGradient;
        ctx.fill();
    }
    ctx.restore();
    drawCaption(ctx, entry, node, config, worldScale);
}

function drawTextLikeNode(ctx, node, entry, config, worldScale = 1) {
    const bounds = entry.contentBounds || entry.bounds;
    if ((bounds.height * Math.max(0.0001, worldScale)) < TEXT_MIN_SCREEN_HEIGHT_PX) {
        return;
    }
    ctx.save();
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    if (node.kind === VIEW2D_NODE_KINDS.OPERATOR) {
        ctx.font = `600 ${entry.layoutData?.fontSize || config.component.operatorFontSize}px ui-monospace, SFMono-Regular, Menlo, monospace`;
        ctx.fillStyle = resolveView2dStyle(node.visual?.styleKey)?.color || config.tokens.palette.text;
    } else {
        ctx.font = `500 ${entry.layoutData?.fontSize || config.component.labelFontSize}px ui-monospace, SFMono-Regular, Menlo, monospace`;
        ctx.fillStyle = resolveView2dStyle(node.visual?.styleKey)?.color || config.tokens.palette.text;
    }
    ctx.fillText(node.text || node.tex || '', bounds.x + (bounds.width / 2), bounds.y + (bounds.height / 2));
    ctx.restore();
    drawCaption(ctx, entry, node, config, worldScale);
}

function drawConnectorArrowHead(ctx, startPoint, endPoint, strokeWidth, fillStyle) {
    if (!startPoint || !endPoint) return;
    const dx = endPoint.x - startPoint.x;
    const dy = endPoint.y - startPoint.y;
    const length = Math.hypot(dx, dy);
    if (!(length > 0.0001)) return;

    const ux = dx / length;
    const uy = dy / length;
    const size = Math.max(strokeWidth * 3.6, 6);
    const wing = Math.max(strokeWidth * 1.7, 3.2);
    const baseX = endPoint.x - (ux * size);
    const baseY = endPoint.y - (uy * size);
    const perpX = -uy;
    const perpY = ux;

    ctx.save();
    ctx.beginPath();
    ctx.moveTo(endPoint.x, endPoint.y);
    ctx.lineTo(baseX + (perpX * wing), baseY + (perpY * wing));
    ctx.lineTo(baseX - (perpX * wing), baseY - (perpY * wing));
    ctx.closePath();
    ctx.fillStyle = fillStyle;
    ctx.fill();
    ctx.restore();
}

function drawConnector(ctx, connectorEntry, config, accent = null, worldScale = 1) {
    const points = Array.isArray(connectorEntry.pathPoints) ? connectorEntry.pathPoints : [];
    if (points.length < 2) return;
    const accentStroke = accent || config.tokens.palette.neutral;
    const foregroundStroke = 'rgba(216, 222, 232, 0.96)';
    const safeWorldScale = Math.max(0.0001, Number.isFinite(worldScale) ? worldScale : 1);
    const glowWidth = Math.max(4, config.tokens.connector.glowWidth + 2) / safeWorldScale;
    const strokeWidth = Math.max(1.4, config.tokens.connector.strokeWidth * 1.15) / safeWorldScale;
    const tailPoint = points[Math.max(0, points.length - 2)];
    const headPoint = points[points.length - 1];

    ctx.save();
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.globalCompositeOperation = 'source-over';
    ctx.filter = 'none';
    ctx.globalAlpha = 1;
    ctx.beginPath();
    ctx.moveTo(points[0].x, points[0].y);
    for (let index = 1; index < points.length; index += 1) {
        ctx.lineTo(points[index].x, points[index].y);
    }
    ctx.lineWidth = glowWidth;
    ctx.strokeStyle = accentStroke;
    ctx.globalAlpha = 0.42;
    ctx.shadowColor = accentStroke;
    ctx.shadowBlur = (config.tokens.connector.glowWidth * 1.8) / safeWorldScale;
    ctx.stroke();

    ctx.beginPath();
    ctx.moveTo(points[0].x, points[0].y);
    for (let index = 1; index < points.length; index += 1) {
        ctx.lineTo(points[index].x, points[index].y);
    }
    ctx.lineWidth = strokeWidth;
    ctx.globalAlpha = 1;
    ctx.shadowBlur = 0;
    ctx.strokeStyle = foregroundStroke;
    ctx.stroke();
    drawConnectorArrowHead(
        ctx,
        tailPoint,
        headPoint,
        strokeWidth,
        foregroundStroke
    );
    ctx.restore();
}

function resolveRenderViewportTransform(sceneBounds, resolution, viewportTransform = null) {
    const fitPaddingPx = Math.max(12, Math.round(Math.min(resolution.width, resolution.height) * 0.04));
    const availableWidth = Math.max(1, resolution.width - (fitPaddingPx * 2));
    const availableHeight = Math.max(1, resolution.height - (fitPaddingPx * 2));
    const widthFitScale = availableWidth / Math.max(1, sceneBounds.width);
    const heightFitScale = availableHeight / Math.max(1, sceneBounds.height);
    const minReadableHeight = Math.min(
        availableHeight,
        Math.max(220, availableHeight * 0.4)
    );
    const readableScaleFloor = minReadableHeight / Math.max(1, sceneBounds.height);

    const fallbackTransform = {
        source: 'auto-fit',
        fitPaddingPx,
        availableWidth,
        availableHeight,
        widthFitScale,
        heightFitScale,
        minReadableHeight,
        readableScaleFloor,
        worldScale: Math.max(
            0.01,
            Math.min(2, Math.min(
                heightFitScale,
                Math.max(widthFitScale, readableScaleFloor)
            ))
        ),
        offsetX: 0,
        offsetY: 0
    };
    fallbackTransform.offsetX = (resolution.width - (sceneBounds.width * fallbackTransform.worldScale)) / 2;
    fallbackTransform.offsetY = (resolution.height - (sceneBounds.height * fallbackTransform.worldScale)) / 2;

    if (!viewportTransform || typeof viewportTransform !== 'object') {
        return fallbackTransform;
    }

    const externalScale = Number.isFinite(viewportTransform.scale) && viewportTransform.scale > 0
        ? viewportTransform.scale
        : null;
    if (!externalScale) {
        return fallbackTransform;
    }

    return {
        ...fallbackTransform,
        source: viewportTransform.source || 'external',
        worldScale: externalScale,
        offsetX: Number.isFinite(viewportTransform.offsetX) ? viewportTransform.offsetX : fallbackTransform.offsetX,
        offsetY: Number.isFinite(viewportTransform.offsetY) ? viewportTransform.offsetY : fallbackTransform.offsetY,
        viewportTransform: { ...viewportTransform }
    };
}

export class CanvasSceneRenderer {
    constructor({
        canvas = null,
        dprCap = null
    } = {}) {
        this.canvas = canvas;
        this.ctx = canvas?.getContext?.('2d') || null;
        this.dprCap = dprCap;
        this.scene = null;
        this.layout = null;
        this.metrics = null;
        this.drawableNodes = [];
        this.connectors = [];
        this.lastRenderState = null;
    }

    setCanvas(canvas = null) {
        this.canvas = canvas;
        this.ctx = canvas?.getContext?.('2d') || null;
        return this.ctx;
    }

    setScene(scene, layout = null, options = {}) {
        this.scene = scene || null;
        this.layout = layout || (scene ? buildSceneLayout(scene, options) : null);
        this.metrics = this.layout?.config || null;
        const registry = this.layout?.registry || null;
        const allNodes = this.scene ? flattenSceneNodes(this.scene) : [];
        this.drawableNodes = allNodes
            .filter((node) => ![VIEW2D_NODE_KINDS.CONNECTOR, VIEW2D_NODE_KINDS.GROUP].includes(node.kind))
            .map((node) => ({
                node,
                entry: registry?.getNodeEntry(node.id) || null
            }))
            .filter((item) => item.entry);
        this.connectors = allNodes
            .filter((node) => node.kind === VIEW2D_NODE_KINDS.CONNECTOR)
            .map((node) => ({
                node,
                entry: registry?.getConnectorEntry(node.id) || null,
                stroke: resolveView2dStyle(node.visual?.styleKey)?.stroke || null
            }))
            .filter((item) => item.entry);
        return this.layout;
    }

    resize(options = {}) {
        if (!this.canvas) return null;
        return syncCanvasResolution(this.canvas, {
            ...options,
            dprCap: options?.dprCap ?? this.dprCap
        });
    }

    getLastRenderState() {
        return this.lastRenderState
            ? {
                ...this.lastRenderState,
                sceneBounds: cloneBounds(this.lastRenderState.sceneBounds)
            }
            : null;
    }

    render({
        width = null,
        height = null,
        dpr = null,
        clear = true,
        debug = false,
        viewportTransform = null
    } = {}) {
        if (!this.canvas || !this.ctx || !this.scene || !this.layout?.registry) {
            this.lastRenderState = {
                ok: false,
                error: 'missing-canvas-scene-or-layout'
            };
            return false;
        }
        const resolution = this.resize({ width, height, dpr });
        if (!resolution) {
            this.lastRenderState = {
                ok: false,
                error: 'failed-to-resize-canvas'
            };
            return false;
        }

        const ctx = this.ctx;
        ctx.setTransform(resolution.dpr, 0, 0, resolution.dpr, 0, 0);
        ctx.globalAlpha = 1;
        ctx.globalCompositeOperation = 'source-over';
        ctx.filter = 'none';
        ctx.shadowBlur = 0;
        ctx.shadowColor = 'transparent';
        if (clear) {
            ctx.clearRect(0, 0, resolution.width, resolution.height);
        }

        const config = this.metrics || this.layout.config || {
            tokens: resolveView2dVisualTokens(),
            component: { captionFontSize: 11, labelFontSize: 12, operatorFontSize: 20 }
        };
        const sceneBounds = this.layout.sceneBounds || this.layout.registry.getSceneBounds();
        const resolvedViewport = resolveRenderViewportTransform(
            sceneBounds,
            resolution,
            viewportTransform
        );
        const worldScale = resolvedViewport.worldScale;
        const offsetX = resolvedViewport.offsetX;
        const offsetY = resolvedViewport.offsetY;
        const visibleWorldBounds = resolveVisibleWorldBounds(resolution, {
            offsetX,
            offsetY,
            worldScale,
            sceneBounds
        });
        const visibleDrawableNodes = this.drawableNodes.filter(({ entry }) => intersectsBounds(entry.bounds, visibleWorldBounds));
        const visibleConnectors = this.connectors.filter(({ entry }) => intersectsBounds(entry.bounds, visibleWorldBounds));

        const renderState = {
            ok: true,
            error: '',
            width: resolution.width,
            height: resolution.height,
            dpr: resolution.dpr,
            pixelWidth: resolution.pixelWidth,
            pixelHeight: resolution.pixelHeight,
            sceneBounds: cloneBounds(sceneBounds),
            fitPaddingPx: resolvedViewport.fitPaddingPx,
            availableWidth: resolvedViewport.availableWidth,
            availableHeight: resolvedViewport.availableHeight,
            widthFitScale: resolvedViewport.widthFitScale,
            heightFitScale: resolvedViewport.heightFitScale,
            minReadableHeight: resolvedViewport.minReadableHeight,
            readableScaleFloor: resolvedViewport.readableScaleFloor,
            worldScale,
            offsetX,
            offsetY,
            fittedWidth: sceneBounds.width * worldScale,
            fittedHeight: sceneBounds.height * worldScale,
            viewportSource: resolvedViewport.source || 'auto-fit',
            viewportTransform: resolvedViewport.viewportTransform || null,
            nodeCount: this.drawableNodes.length,
            connectorCount: this.connectors.length,
            visibleNodeCount: visibleDrawableNodes.length,
            visibleConnectorCount: visibleConnectors.length
        };
        this.lastRenderState = renderState;

        try {
            ctx.save();
            ctx.translate(offsetX, offsetY);
            ctx.scale(worldScale, worldScale);
            ctx.fillStyle = config.tokens.palette.sceneBackground;
            ctx.fillRect(0, 0, sceneBounds.width, sceneBounds.height);

            visibleDrawableNodes.forEach(({ node, entry }) => {
                if (node.kind === VIEW2D_NODE_KINDS.MATRIX) {
                    drawMatrixNode(ctx, node, entry, config, worldScale);
                } else if (node.kind === VIEW2D_NODE_KINDS.TEXT || node.kind === VIEW2D_NODE_KINDS.OPERATOR) {
                    drawTextLikeNode(ctx, node, entry, config, worldScale);
                }
            });

            visibleConnectors.forEach(({ entry, stroke }) => {
                drawConnector(ctx, entry, config, stroke, worldScale);
            });

            ctx.restore();
            if (debug) {
                drawDebugOverlay(ctx, resolution, renderState);
            }
            return true;
        } catch (error) {
            ctx.restore();
            renderState.ok = false;
            renderState.error = error instanceof Error ? error.message : String(error);
            this.lastRenderState = renderState;
            if (debug) {
                drawDebugOverlay(ctx, resolution, renderState);
            }
            return false;
        }
    }
}
