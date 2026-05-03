import { resolveRenderPixelRatio } from '../../../utils/constants.js';
import {
    flattenSceneNodes,
    VIEW2D_MATRIX_PRESENTATIONS,
    VIEW2D_NODE_KINDS
} from '../../schema/sceneTypes.js';
import { buildSceneLayout } from '../../layout/buildSceneLayout.js';
import {
    VIEW2D_VECTOR_STRIP_DEFAULTS,
    VIEW2D_VECTOR_STRIP_VARIANT
} from '../../shared/vectorStrip.js';
import { resolveMhsaDetailFixedTextSizing } from '../../shared/mhsaDetailFixedLabelSizing.js';
import { VIEW2D_TEXT_ZOOM_BEHAVIORS } from '../../shared/mhsaDetailFixedLabelSizing.js';
import {
    measureView2dText,
    resolveView2dTextFont
} from '../../textMeasurement.js';
import { resolveView2dStyle, resolveView2dVisualTokens, VIEW2D_STYLE_KEYS } from '../../theme/visualTokens.js';
import {
    resolveView2dCaptionLines,
    resolveView2dCaptionPosition,
    resolveView2dCaptionStyleKey
} from '../../captionUtils.js';
import {
    drawSimpleTex,
    hasSimpleTexMarkup
} from '../../simpleTex.js';
import {
    normalizeSceneFocusState,
    resolveSceneColumnSelectionAlpha,
    resolveSceneElementFocusAlpha,
    resolveSceneGridCellAlpha,
    resolveSceneNodeFocusAlpha,
    resolveSceneRowSelectionAlpha
} from '../../sceneFocusState.js';

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

const VIEW2D_CARD_SHAPES = Object.freeze({
    CURVED_TRAPEZOID: 'curved-trapezoid'
});

function curvedTrapezoidPath(ctx, bounds, shapeConfig = null) {
    const x = Number(bounds?.x) || 0;
    const y = Number(bounds?.y) || 0;
    const width = Math.max(1, Number(bounds?.width) || 0);
    const height = Math.max(1, Number(bounds?.height) || 0);
    const centerY = y + (height / 2);
    const clampHeightRatio = (value, fallback) => (
        Number.isFinite(value)
            ? Math.max(0.18, Math.min(1, Number(value)))
            : fallback
    );
    const leftInset = Number.isFinite(shapeConfig?.leftInset)
        ? Math.max(0, Math.min(width * 0.4, Number(shapeConfig.leftInset)))
        : 0;
    const rightInset = Number.isFinite(shapeConfig?.rightInset)
        ? Math.max(0, Math.min(width * 0.28, Number(shapeConfig.rightInset)))
        : Math.min(width * 0.06, 8);
    const leftHeightRatio = clampHeightRatio(shapeConfig?.leftHeightRatio, 1);
    const rightHeightRatio = clampHeightRatio(shapeConfig?.rightHeightRatio, 0.38);
    const leftHalfHeight = Math.max(1, (height * leftHeightRatio) / 2);
    const rightHalfHeight = Math.max(1, (height * rightHeightRatio) / 2);
    const cornerRadius = Number.isFinite(shapeConfig?.cornerRadius)
        ? Math.max(0, Number(shapeConfig.cornerRadius))
        : Math.min(18, height * 0.18);
    const topLeft = {
        x: x + leftInset,
        y: centerY - leftHalfHeight
    };
    const topRight = {
        x: x + width - rightInset,
        y: centerY - rightHalfHeight
    };
    const bottomRight = {
        x: x + width - rightInset,
        y: centerY + rightHalfHeight
    };
    const bottomLeft = {
        x: x + leftInset,
        y: centerY + leftHalfHeight
    };

    const points = [topLeft, topRight, bottomRight, bottomLeft];
    const corners = points.map((point, index) => {
        const previous = points[(index + points.length - 1) % points.length];
        const next = points[(index + 1) % points.length];
        const toPreviousX = previous.x - point.x;
        const toPreviousY = previous.y - point.y;
        const toNextX = next.x - point.x;
        const toNextY = next.y - point.y;
        const previousLength = Math.hypot(toPreviousX, toPreviousY);
        const nextLength = Math.hypot(toNextX, toNextY);
        const cutDistance = Math.min(
            cornerRadius,
            previousLength * 0.5,
            nextLength * 0.5
        );
        const previousUnitX = previousLength > 0 ? toPreviousX / previousLength : 0;
        const previousUnitY = previousLength > 0 ? toPreviousY / previousLength : 0;
        const nextUnitX = nextLength > 0 ? toNextX / nextLength : 0;
        const nextUnitY = nextLength > 0 ? toNextY / nextLength : 0;
        return {
            point,
            start: {
                x: point.x + (previousUnitX * cutDistance),
                y: point.y + (previousUnitY * cutDistance)
            },
            end: {
                x: point.x + (nextUnitX * cutDistance),
                y: point.y + (nextUnitY * cutDistance)
            }
        };
    });

    ctx.beginPath();
    ctx.moveTo(corners[0].start.x, corners[0].start.y);
    corners.forEach((corner, index) => {
        ctx.quadraticCurveTo(
            corner.point.x,
            corner.point.y,
            corner.end.x,
            corner.end.y
        );
        const nextCorner = corners[(index + 1) % corners.length];
        ctx.lineTo(nextCorner.start.x, nextCorner.start.y);
    });
    ctx.closePath();
}

function traceCardPath(ctx, bounds, cornerRadius = 8, cardMetadata = null) {
    const cardShape = String(cardMetadata?.shape || '').trim().toLowerCase();
    if (cardShape === VIEW2D_CARD_SHAPES.CURVED_TRAPEZOID) {
        curvedTrapezoidPath(ctx, bounds, cardMetadata?.shapeConfig || null);
        return;
    }
    roundRectPath(ctx, bounds.x, bounds.y, bounds.width, bounds.height, cornerRadius);
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

function parseCanvasColor(value = '') {
    const raw = String(value || '').trim();
    if (!raw.length) return null;
    const hexMatch = raw.match(/^#([0-9a-f]{3}|[0-9a-f]{6})$/i);
    if (hexMatch) {
        const hex = hexMatch[1];
        if (hex.length === 3) {
            return {
                r: Number.parseInt(`${hex[0]}${hex[0]}`, 16),
                g: Number.parseInt(`${hex[1]}${hex[1]}`, 16),
                b: Number.parseInt(`${hex[2]}${hex[2]}`, 16),
                a: 1
            };
        }
        return {
            r: Number.parseInt(hex.slice(0, 2), 16),
            g: Number.parseInt(hex.slice(2, 4), 16),
            b: Number.parseInt(hex.slice(4, 6), 16),
            a: 1
        };
    }
    const rgbMatch = raw.match(/^rgba?\(\s*([0-9.]+)\s*,\s*([0-9.]+)\s*,\s*([0-9.]+)(?:\s*,\s*([0-9.]+))?\s*\)$/i);
    if (rgbMatch) {
        return {
            r: Math.max(0, Math.min(255, Number.parseFloat(rgbMatch[1]))),
            g: Math.max(0, Math.min(255, Number.parseFloat(rgbMatch[2]))),
            b: Math.max(0, Math.min(255, Number.parseFloat(rgbMatch[3]))),
            a: Math.max(0, Math.min(1, rgbMatch[4] === undefined ? 1 : Number.parseFloat(rgbMatch[4])))
        };
    }
    return null;
}

function flattenColorAgainstBlack(value = '', opacity = 1) {
    const parsed = parseCanvasColor(value);
    if (!parsed) return null;
    const alpha = Math.max(0, Math.min(1, parsed.a * (Number.isFinite(opacity) ? opacity : 1)));
    const scale = (channel) => Math.round(Math.max(0, Math.min(255, channel * alpha)));
    return `rgb(${scale(parsed.r)}, ${scale(parsed.g)}, ${scale(parsed.b)})`;
}

function resolveNodeVisualOpacity(node = null) {
    const opacity = Number(node?.visual?.opacity);
    return Number.isFinite(opacity)
        ? Math.max(0, Math.min(1, opacity))
        : 1;
}

const PARSED_LINEAR_GRADIENT_CACHE = new Map();
const CAPTION_MIN_SCREEN_HEIGHT_PX = 18;
const TEXT_MIN_SCREEN_HEIGHT_PX = 10;
const PERSISTENT_OPERATOR_MIN_SCREEN_FONT_PX = 8.5;
const PERSISTENT_PLUS_OPERATOR_MIN_SCREEN_FONT_PX = 7.6;
const PLUS_OPERATOR_FONT_SCALE = 0.92;
const PLUS_OPERATOR_FONT_WEIGHT = 500;
const MATRIX_DETAIL_MIN_SCREEN_WIDTH_PX = 72;
const MATRIX_DETAIL_MIN_SCREEN_HEIGHT_PX = 24;
const INTERACTION_DETAIL_RELEASE_RATIO = 0.88;
const VECTOR_STRIP_HIGHLIGHT_STROKE_SCREEN_PX = 0.8;
const GRID_FOCUSED_CELL_STROKE_SCREEN_PX = 0.92;
const MATRIX_ORNAMENT_ARROW_GAP_SCREEN_PX = 9;
const MATRIX_ORNAMENT_ARROW_SHAFT_SCREEN_PX = 30;
const MATRIX_ORNAMENT_ARROW_SHAFT_SCREEN_PX_INCOMING = 42;
const MATRIX_ORNAMENT_ARROW_STROKE_SCREEN_PX = 0.88;
const MATRIX_ORNAMENT_ARROW_HEAD_LENGTH_SCREEN_PX = 6;
const MATRIX_ORNAMENT_ARROW_HEAD_WING_SCREEN_PX = 3.2;
const OVERVIEW_RENDER_CACHE_MIN_OVERSCAN_PX = 160;
const OVERVIEW_RENDER_CACHE_MAX_OVERSCAN_PX = 240;
const OVERVIEW_RENDER_CACHE_OVERSCAN_VIEWPORT_RATIO = 0.35;
const OVERVIEW_RENDER_CACHE_COVERAGE_EPSILON_PX = 0.5;

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

function drawVectorStripRows(
    ctx,
    rowItems,
    contentBounds,
    layoutData,
    accent,
    cornerRadius = 0,
    {
        hoveredRowIndex = null,
        focusedRowIndices = null,
        previousHoveredRowIndex = null,
        hoverRowBlend = 1,
        dimmingStrength = 0,
        dimmedRowOpacity = 0.18,
        baseAlpha = 1,
        focusedRowBaseAlpha = null,
        bandCount = 12,
        bandSeparatorOpacity = 0,
        focusedBandCells = null,
        hoverScaleY = 1,
        hoverGlowColor = null,
        hoverGlowBlur = 0,
        hoverStrokeColor = null
    } = {}
) {
    const compactWidth = Math.max(
        1,
        Math.min(layoutData.compactWidth, contentBounds.width - (layoutData.innerPaddingX * 2))
    );
    const resolvedBandCount = Number.isFinite(bandCount) ? Math.max(1, Math.floor(bandCount)) : 12;
    const resolvedBandSeparatorOpacity = Number.isFinite(bandSeparatorOpacity)
        ? Math.max(0, Math.min(1, Number(bandSeparatorOpacity)))
        : 0;
    const resolvedHoverScaleY = Number.isFinite(hoverScaleY) ? Math.max(1, hoverScaleY) : 1;
    const resolvedHoverGlowBlur = Number.isFinite(hoverGlowBlur) ? Math.max(0, hoverGlowBlur) : 0;
    const resolvedBaseAlpha = Math.max(0, Math.min(1, Number.isFinite(baseAlpha) ? baseAlpha : 1));
    const resolvedFocusedRowBaseAlpha = Math.max(
        0,
        Math.min(
            1,
            Number.isFinite(focusedRowBaseAlpha)
                ? focusedRowBaseAlpha
                : resolvedBaseAlpha
        )
    );
    const focusedBandMap = new Map();
    if (focusedBandCells instanceof Map) {
        focusedBandCells.forEach((bandSet, rowIndex) => {
            if (!Number.isFinite(rowIndex) || !(bandSet instanceof Set) || !bandSet.size) return;
            const normalizedRowIndex = Math.max(0, Math.floor(rowIndex));
            const normalizedBandSet = new Set(
                Array.from(bandSet)
                    .filter((value) => Number.isFinite(value))
                    .map((value) => Math.max(0, Math.floor(value)))
            );
            if (normalizedBandSet.size) {
                focusedBandMap.set(normalizedRowIndex, normalizedBandSet);
            }
        });
    } else if (Array.isArray(focusedBandCells)) {
        focusedBandCells.forEach((selection) => {
            const rowIndex = Number.isFinite(selection?.rowIndex) ? Math.max(0, Math.floor(selection.rowIndex)) : null;
            const bandIndex = Number.isFinite(selection?.bandIndex) ? Math.max(0, Math.floor(selection.bandIndex)) : null;
            if (!Number.isFinite(rowIndex) || !Number.isFinite(bandIndex)) return;
            let bandSet = focusedBandMap.get(rowIndex);
            if (!bandSet) {
                bandSet = new Set();
                focusedBandMap.set(rowIndex, bandSet);
            }
            bandSet.add(bandIndex);
        });
    }
    const hasFocusedBands = focusedBandMap.size > 0;
    const focusedRowIndexSet = focusedRowIndices instanceof Set
        ? focusedRowIndices
        : new Set(
            Array.isArray(focusedRowIndices)
                ? focusedRowIndices
                    .filter((value) => Number.isFinite(value))
                    .map((value) => Math.max(0, Math.floor(value)))
                : []
        );
    const hasFocusedRows = focusedRowIndexSet.size > 0;
    const resolveHighlightStrength = (index) => {
        if (hasFocusedRows) {
            return focusedRowIndexSet.has(index) ? 1 : 0;
        }
        let strength = 0;
        if (Number.isFinite(hoveredRowIndex) && index === hoveredRowIndex) {
            strength = Math.max(strength, Math.max(0, Math.min(1, hoverRowBlend)));
        }
        if (Number.isFinite(previousHoveredRowIndex) && index === previousHoveredRowIndex) {
            strength = Math.max(strength, 1 - Math.max(0, Math.min(1, hoverRowBlend)));
        }
        return strength;
    };
    const rowDrawEntries = rowItems.map((rowItem, index) => {
        const rowY = contentBounds.y + layoutData.innerPaddingY + index * (layoutData.rowHeight + layoutData.rowGap);
        const baseBounds = {
            x: contentBounds.x + layoutData.innerPaddingX,
            y: rowY,
            width: compactWidth,
            height: layoutData.rowHeight
        };
        const highlightStrength = resolveHighlightStrength(index);
        const scaleY = 1 + ((resolvedHoverScaleY - 1) * highlightStrength);
        const scaledHeight = baseBounds.height * scaleY;
        const scaledBounds = {
            ...baseBounds,
            y: baseBounds.y - ((scaledHeight - baseBounds.height) * 0.5),
            height: scaledHeight
        };
        const inactiveRowOpacity = Math.max(0, Math.min(1, Number.isFinite(dimmedRowOpacity) ? dimmedRowOpacity : 0.18));
        const alpha = hasFocusedRows
            ? (focusedRowIndexSet.has(index)
                ? resolvedFocusedRowBaseAlpha
                : (resolvedBaseAlpha * inactiveRowOpacity))
            : hasFocusedBands
                ? (focusedBandMap.has(index)
                    ? resolvedFocusedRowBaseAlpha
                    : (resolvedBaseAlpha * inactiveRowOpacity))
                : (
                    resolvedBaseAlpha * resolveRowDimmingAlpha(index, hoveredRowIndex, {
                        previousHoveredRowIndex,
                        hoverRowBlend,
                        dimStrength: dimmingStrength,
                        dimmedRowOpacity
                    })
                );
        return {
            rowItem,
            index,
            bounds: scaledBounds,
            alpha: Number(alpha.toFixed(3)),
            highlightStrength
        };
    });
    ctx.save();
    roundRectPath(ctx, contentBounds.x, contentBounds.y, contentBounds.width, contentBounds.height, cornerRadius);
    ctx.clip();
    rowDrawEntries
        .sort((left, right) => left.highlightStrength - right.highlightStrength)
        .forEach(({ rowItem, index, bounds, alpha, highlightStrength }) => {
            const focusedBands = focusedBandMap.get(index) || null;
            const hasFocusedBandsForRow = !!(focusedBands && focusedBands.size);
            const rowFill = resolveFill(ctx, rowItem.gradientCss, bounds, accent);
            const bandWidth = resolvedBandCount > 0 ? (bounds.width / resolvedBandCount) : bounds.width;
            if (highlightStrength > 0.001 && typeof hoverGlowColor === 'string' && hoverGlowColor.length && resolvedHoverGlowBlur > 0) {
                ctx.save();
                ctx.globalAlpha = alpha * highlightStrength;
                ctx.fillStyle = rowFill;
                ctx.shadowColor = hoverGlowColor;
                ctx.shadowBlur = resolvedHoverGlowBlur;
                ctx.fillRect(bounds.x, bounds.y, bounds.width, bounds.height);
                ctx.restore();
            }
            if (hasFocusedBandsForRow) {
                ctx.fillStyle = rowFill;
                ctx.globalAlpha = alpha * Math.max(0, Math.min(1, Number.isFinite(dimmedRowOpacity) ? dimmedRowOpacity : 0.18));
                ctx.fillRect(bounds.x, bounds.y, bounds.width, bounds.height);

                focusedBands.forEach((bandIndex) => {
                    if (bandIndex < 0 || bandIndex >= resolvedBandCount) return;
                    const bandBounds = {
                        x: bounds.x + (bandWidth * bandIndex),
                        y: bounds.y,
                        width: bandIndex === (resolvedBandCount - 1)
                            ? Math.max(0, bounds.x + bounds.width - (bounds.x + (bandWidth * bandIndex)))
                            : bandWidth,
                        height: bounds.height
                    };
                    ctx.save();
                    ctx.beginPath();
                    ctx.rect(bandBounds.x, bandBounds.y, bandBounds.width, bandBounds.height);
                    ctx.clip();
                    ctx.fillStyle = rowFill;
                    ctx.globalAlpha = alpha;
                    ctx.fillRect(bounds.x, bounds.y, bounds.width, bounds.height);
                    if (typeof hoverGlowColor === 'string' && hoverGlowColor.length && resolvedHoverGlowBlur > 0) {
                        ctx.shadowColor = hoverGlowColor;
                        ctx.shadowBlur = resolvedHoverGlowBlur;
                        ctx.fillRect(bounds.x, bounds.y, bounds.width, bounds.height);
                    }
                    ctx.restore();

                    if (typeof hoverStrokeColor === 'string' && hoverStrokeColor.length) {
                        ctx.save();
                        ctx.globalAlpha = alpha;
                        ctx.lineWidth = VECTOR_STRIP_HIGHLIGHT_STROKE_SCREEN_PX;
                        ctx.strokeStyle = hoverStrokeColor;
                        ctx.strokeRect(
                            bandBounds.x + 0.5,
                            bandBounds.y + 0.5,
                            Math.max(0, bandBounds.width - 1),
                            Math.max(0, bandBounds.height - 1)
                        );
                        ctx.restore();
                    }
                });
            } else {
                ctx.fillStyle = rowFill;
                ctx.globalAlpha = alpha;
                ctx.fillRect(bounds.x, bounds.y, bounds.width, bounds.height);
                if (highlightStrength > 0.001 && typeof hoverStrokeColor === 'string' && hoverStrokeColor.length) {
                    ctx.save();
                    ctx.globalAlpha = alpha * highlightStrength;
                    ctx.lineWidth = VECTOR_STRIP_HIGHLIGHT_STROKE_SCREEN_PX;
                    ctx.strokeStyle = hoverStrokeColor;
                    ctx.strokeRect(
                        bounds.x + 0.5,
                        bounds.y + 0.5,
                        Math.max(0, bounds.width - 1),
                        Math.max(0, bounds.height - 1)
                    );
                    ctx.restore();
                }
            }
            if (resolvedBandCount > 1 && resolvedBandSeparatorOpacity > 0 && bounds.width > 4) {
                const separatorWidth = Math.max(0.35, Math.min(1, bandWidth * 0.06));
                ctx.fillStyle = `rgba(255, 255, 255, ${resolvedBandSeparatorOpacity})`;
                for (let bandIndex = 1; bandIndex < resolvedBandCount; bandIndex += 1) {
                    const separatorX = bounds.x + (bandWidth * bandIndex) - (separatorWidth * 0.5);
                    ctx.fillRect(separatorX, bounds.y, separatorWidth, bounds.height);
                }
            }
        });
    ctx.restore();
}

function drawVectorStripColumns(
    ctx,
    columnItems,
    contentBounds,
    layoutData,
    accent,
    cornerRadius = 0,
    {
        focusedColumnIndex = null,
        dimmedColumnOpacity = 0.18,
        baseAlpha = 1,
        hoverScaleX = 1,
        hoverGlowColor = null,
        hoverGlowBlur = 0,
        hoverStrokeColor = null
    } = {}
) {
    const resolvedHoverScaleX = Number.isFinite(hoverScaleX) ? Math.max(1, hoverScaleX) : 1;
    const resolvedHoverGlowBlur = Number.isFinite(hoverGlowBlur) ? Math.max(0, hoverGlowBlur) : 0;
    const hasFocusedColumn = Number.isFinite(focusedColumnIndex);
    const inactiveOpacity = Math.max(0, Math.min(1, Number.isFinite(dimmedColumnOpacity) ? dimmedColumnOpacity : 0.18));
    const baseOpacity = Math.max(0, Math.min(1, Number.isFinite(baseAlpha) ? baseAlpha : 1));
    const columnDrawEntries = columnItems.map((columnItem, index) => {
        const baseBounds = {
            x: contentBounds.x + layoutData.innerPaddingX + index * (layoutData.colWidth + layoutData.colGap),
            y: contentBounds.y + layoutData.innerPaddingY,
            width: layoutData.colWidth,
            height: layoutData.colHeight
        };
        const highlightStrength = hasFocusedColumn && index === focusedColumnIndex ? 1 : 0;
        const scaleX = 1 + ((resolvedHoverScaleX - 1) * highlightStrength);
        const scaledWidth = baseBounds.width * scaleX;
        const scaledBounds = {
            ...baseBounds,
            x: baseBounds.x - ((scaledWidth - baseBounds.width) * 0.5),
            width: scaledWidth
        };
        const alpha = baseOpacity * (hasFocusedColumn
            ? (highlightStrength > 0 ? 1 : inactiveOpacity)
            : 1);
        return {
            columnItem,
            index,
            bounds: scaledBounds,
            alpha,
            highlightStrength
        };
    });

    ctx.save();
    roundRectPath(ctx, contentBounds.x, contentBounds.y, contentBounds.width, contentBounds.height, cornerRadius);
    ctx.clip();
    columnDrawEntries
        .sort((left, right) => left.highlightStrength - right.highlightStrength)
        .forEach(({ columnItem, bounds, alpha, highlightStrength }) => {
            if (highlightStrength > 0.001 && typeof hoverGlowColor === 'string' && hoverGlowColor.length && resolvedHoverGlowBlur > 0) {
                ctx.save();
                ctx.globalAlpha = alpha * highlightStrength;
                ctx.fillStyle = resolveFill(ctx, columnItem.fillCss, bounds, accent);
                ctx.shadowColor = hoverGlowColor;
                ctx.shadowBlur = resolvedHoverGlowBlur;
                ctx.fillRect(bounds.x, bounds.y, bounds.width, bounds.height);
                ctx.restore();
            }
            ctx.globalAlpha = alpha;
            ctx.fillStyle = resolveFill(ctx, columnItem.fillCss, bounds, accent);
            ctx.fillRect(bounds.x, bounds.y, bounds.width, bounds.height);
            if (highlightStrength > 0.001 && typeof hoverStrokeColor === 'string' && hoverStrokeColor.length) {
                ctx.save();
                ctx.globalAlpha = alpha * highlightStrength;
                ctx.lineWidth = VECTOR_STRIP_HIGHLIGHT_STROKE_SCREEN_PX;
                ctx.strokeStyle = hoverStrokeColor;
                ctx.strokeRect(
                    bounds.x + 0.5,
                    bounds.y + 0.5,
                    Math.max(0, bounds.width - 1),
                    Math.max(0, bounds.height - 1)
                );
                ctx.restore();
            }
        });
    ctx.restore();
}

function resolveGridCellCornerRadius(cellSize = 0, worldScale = 1, radiusScale = 1) {
    const safeCellSize = Number.isFinite(cellSize) ? Math.max(0, Number(cellSize)) : 0;
    const safeWorldScale = Math.max(0.0001, Number.isFinite(worldScale) ? worldScale : 1);
    const safeRadiusScale = Number.isFinite(radiusScale) ? Math.max(0, Number(radiusScale)) : 1;
    const screenCellSize = safeCellSize * safeWorldScale;
    const radiusPx = Math.max(1, Math.min(2, screenCellSize * 0.11));
    return (radiusPx * safeRadiusScale) / safeWorldScale;
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

function unionRenderableBounds(boundsList = []) {
    const validBounds = boundsList
        .map((bounds) => cloneBounds(bounds))
        .filter((bounds) => bounds && Number.isFinite(bounds.width) && Number.isFinite(bounds.height));
    if (!validBounds.length) return null;

    const minX = Math.min(...validBounds.map((bounds) => bounds.x));
    const minY = Math.min(...validBounds.map((bounds) => bounds.y));
    const maxX = Math.max(...validBounds.map((bounds) => bounds.x + bounds.width));
    const maxY = Math.max(...validBounds.map((bounds) => bounds.y + bounds.height));
    return {
        x: minX,
        y: minY,
        width: Math.max(0, maxX - minX),
        height: Math.max(0, maxY - minY)
    };
}

function containsPoint(bounds = null, x = 0, y = 0) {
    if (!bounds) return false;
    const minX = Number.isFinite(bounds.x) ? bounds.x : 0;
    const minY = Number.isFinite(bounds.y) ? bounds.y : 0;
    const maxX = minX + Math.max(0, Number(bounds.width) || 0);
    const maxY = minY + Math.max(0, Number(bounds.height) || 0);
    return x >= minX && x <= maxX && y >= minY && y <= maxY;
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

function resolveMatrixHoveredRowIndex(node = null, hoveredRow = null) {
    if (!node?.id || !hoveredRow || hoveredRow.nodeId !== node.id) return null;
    return Number.isFinite(hoveredRow.rowIndex) ? Math.max(0, Math.floor(hoveredRow.rowIndex)) : null;
}

function resolveRowDimmingAlpha(rowIndex = 0, hoveredRowIndex = null, {
    previousHoveredRowIndex = null,
    hoverRowBlend = 1,
    dimStrength = 0,
    dimmedRowOpacity = 0.18
} = {}) {
    const safeDimStrength = Math.max(0, Math.min(1, Number.isFinite(dimStrength) ? dimStrength : 0));
    if (safeDimStrength <= 0) return 1;
    const targetOpacity = Math.max(0, Math.min(1, Number.isFinite(dimmedRowOpacity) ? dimmedRowOpacity : 0.18));
    const baseAlpha = 1 - ((1 - targetOpacity) * safeDimStrength);
    const safeBlend = Math.max(0, Math.min(1, Number.isFinite(hoverRowBlend) ? hoverRowBlend : 1));
    const isCurrentHoveredRow = Number.isFinite(hoveredRowIndex) && Math.floor(rowIndex) === Math.floor(hoveredRowIndex);
    const isPreviousHoveredRow = Number.isFinite(previousHoveredRowIndex)
        && Math.floor(rowIndex) === Math.floor(previousHoveredRowIndex);
    if (isCurrentHoveredRow && isPreviousHoveredRow) {
        return 1;
    }
    if (isCurrentHoveredRow) {
        return baseAlpha + ((1 - baseAlpha) * safeBlend);
    }
    if (isPreviousHoveredRow) {
        return baseAlpha + ((1 - baseAlpha) * (1 - safeBlend));
    }
    return baseAlpha;
}

function hasLocalMatrixSceneSelection(node = null, sceneFocusState = null) {
    if (!node?.id || !sceneFocusState) return false;
    return !!(
        sceneFocusState.rowSelections?.get(node.id)?.size
        || sceneFocusState.columnSelections?.get(node.id)?.size
        || sceneFocusState.cellSelections?.get(node.id)?.size
    );
}

function resolveMatrixSurfaceFocusAlpha(node = null, sceneFocusState = null, focusAlpha = 1) {
    const nodeAlpha = Math.max(0, Math.min(1, Number.isFinite(focusAlpha) ? focusAlpha : 1));
    if (
        nodeAlpha < 0.995
        || !sceneFocusState
        || !hasLocalMatrixSceneSelection(node, sceneFocusState)
    ) {
        return nodeAlpha;
    }
    const inactiveOpacity = Number.isFinite(sceneFocusState?.inactiveOpacity)
        ? Math.max(0, Math.min(1, Number(sceneFocusState.inactiveOpacity)))
        : 0.18;
    return nodeAlpha * Math.max(0.24, inactiveOpacity);
}

function resolveSceneInactiveFilter(sceneFocusState = null, focusAlpha = 1, config = null) {
    const nodeAlpha = Math.max(0, Math.min(1, Number.isFinite(focusAlpha) ? focusAlpha : 1));
    if (!sceneFocusState || nodeAlpha >= 0.995) return 'none';
    const filter = typeof config?.tokens?.dimming?.inactiveFilter === 'string'
        ? config.tokens.dimming.inactiveFilter.trim()
        : '';
    return filter.length ? filter : 'none';
}

function resolveSceneNodeFilter(nodeId = '', sceneFocusState = null, focusAlpha = 1, config = null, {
    disableInactiveFilter = false
} = {}) {
    if (!sceneFocusState) return 'none';
    const isExplicitDimNode = typeof nodeId === 'string'
        && nodeId.length
        && sceneFocusState.dimNodeIds?.has(nodeId);
    if (disableInactiveFilter && !isExplicitDimNode) {
        return 'none';
    }
    return resolveSceneInactiveFilter(sceneFocusState, focusAlpha, config);
}

function resolveRowHitBounds(node = null, entry = null, rowIndex = null) {
    if (!node || node.kind !== VIEW2D_NODE_KINDS.MATRIX) return null;
    if (!Number.isFinite(rowIndex) || rowIndex < 0) return null;
    const rowItems = Array.isArray(node.rowItems) ? node.rowItems : [];
    if (rowIndex >= rowItems.length) return null;
    const contentBounds = entry?.contentBounds || entry?.bounds || null;
    const layoutData = entry?.layoutData || null;
    if (!contentBounds || !layoutData) return null;
    if (
        node.presentation !== VIEW2D_MATRIX_PRESENTATIONS.COMPACT_ROWS
        && node.presentation !== VIEW2D_MATRIX_PRESENTATIONS.BANDED_ROWS
    ) {
        return null;
    }

    const rowHeight = Math.max(0, Number(layoutData.rowHeight) || 0);
    const rowGap = Math.max(0, Number(layoutData.rowGap) || 0);
    const innerPaddingX = Math.max(0, Number(layoutData.innerPaddingX) || 0);
    const innerPaddingY = Math.max(0, Number(layoutData.innerPaddingY) || 0);
    const rowY = contentBounds.y + innerPaddingY + rowIndex * (rowHeight + rowGap);

    if (node.presentation === VIEW2D_MATRIX_PRESENTATIONS.BANDED_ROWS) {
        return {
            x: contentBounds.x + innerPaddingX,
            y: rowY,
            width: Math.max(1, contentBounds.width - (innerPaddingX * 2)),
            height: rowHeight
        };
    }

    const compactWidth = Math.max(
        1,
        Math.min(
            Number(layoutData.compactWidth) || 0,
            contentBounds.width - (innerPaddingX * 2)
        )
    );
    return {
        x: contentBounds.x + innerPaddingX,
        y: rowY,
        width: compactWidth,
        height: rowHeight
    };
}

function resolveTrackIndex(relativeOffset = 0, itemSize = 0, itemGap = 0, itemCount = 0) {
    const safeItemCount = Number.isFinite(itemCount) ? Math.max(0, Math.floor(itemCount)) : 0;
    const safeItemSize = Number(itemSize) || 0;
    if (!Number.isFinite(relativeOffset) || safeItemCount <= 0 || !(safeItemSize > 0)) {
        return null;
    }

    const safeItemGap = Math.max(0, Number(itemGap) || 0);
    const stride = safeItemSize + safeItemGap;
    if (!(stride > 0)) return null;

    const index = Math.floor(relativeOffset / stride);
    if (index < 0 || index >= safeItemCount) return null;

    const offsetWithinStride = relativeOffset - (index * stride);
    if (offsetWithinStride < 0 || offsetWithinStride > safeItemSize) {
        return null;
    }

    return {
        index,
        stride,
        offsetWithinStride
    };
}

const OVERVIEW_RESIDUAL_ROW_SCREEN_HOVER_PADDING_PX = 10;
const OVERVIEW_RESIDUAL_ROW_MIN_SCREEN_TARGET_WIDTH_PX = 42;
const OVERVIEW_RESIDUAL_ROW_MIN_SCREEN_TARGET_HEIGHT_PX = 28;

function isOverviewResidualVectorStripNode(node = null) {
    return !!(
        node?.kind === VIEW2D_NODE_KINDS.MATRIX
        && node?.role === 'module-card'
        && node?.semantic?.componentKind === 'residual'
        && node?.presentation === VIEW2D_MATRIX_PRESENTATIONS.COMPACT_ROWS
        && node?.metadata?.compactRows?.variant === VIEW2D_VECTOR_STRIP_VARIANT
    );
}

function expandScreenBounds(bounds = null, {
    minWidthPx = 0,
    minHeightPx = 0,
    paddingPx = 0
} = {}) {
    if (!bounds) return null;
    const safePaddingPx = Number.isFinite(paddingPx) ? Math.max(0, Number(paddingPx)) : 0;
    const width = Math.max(0, Number(bounds.width) || 0);
    const height = Math.max(0, Number(bounds.height) || 0);
    const extraWidth = Math.max(0, Number.isFinite(minWidthPx) ? Number(minWidthPx) - width : 0);
    const extraHeight = Math.max(0, Number.isFinite(minHeightPx) ? Number(minHeightPx) - height : 0);
    return {
        x: (Number(bounds.x) || 0) - safePaddingPx - (extraWidth * 0.5),
        y: (Number(bounds.y) || 0) - safePaddingPx - (extraHeight * 0.5),
        width: width + (safePaddingPx * 2) + extraWidth,
        height: height + (safePaddingPx * 2) + extraHeight
    };
}

function resolveMatrixRowHit(node = null, entry = null, x = 0, y = 0, detailScale = 1) {
    if (!node || node.kind !== VIEW2D_NODE_KINDS.MATRIX) return null;
    if (!Number.isFinite(x) || !Number.isFinite(y)) return null;
    const rowItems = Array.isArray(node.rowItems) ? node.rowItems : [];
    if (!rowItems.length) return null;
    const isOverviewResidualStrip = isOverviewResidualVectorStripNode(node);
    const hitBounds = isOverviewResidualStrip
        ? (entry?.bounds || entry?.contentBounds)
        : (entry?.contentBounds || entry?.bounds);
    if (!containsPoint(hitBounds, x, y)) return null;
    if (
        node.presentation !== VIEW2D_MATRIX_PRESENTATIONS.COMPACT_ROWS
        && node.presentation !== VIEW2D_MATRIX_PRESENTATIONS.BANDED_ROWS
    ) {
        return null;
    }

    const contentBounds = entry?.contentBounds || entry?.bounds || null;
    const layoutData = entry?.layoutData || null;
    if (!contentBounds || !layoutData) return null;
    const innerPaddingY = Math.max(0, Number(layoutData.innerPaddingY) || 0);
    const rowHeight = Math.max(1, Number(layoutData.rowHeight) || 0);
    const rowGap = Math.max(0, Number(layoutData.rowGap) || 0);
    const relativeY = y - (contentBounds.y + innerPaddingY);
    const resolvedTrack = resolveTrackIndex(relativeY, rowHeight, rowGap, rowItems.length);

    if (resolvedTrack) {
        const bounds = resolveRowHitBounds(node, entry, resolvedTrack.index);
        if (!bounds) return null;
        return {
            rowIndex: resolvedTrack.index,
            rowItem: rowItems[resolvedTrack.index],
            bounds
        };
    }

    const compactRowsMeta = node.metadata?.compactRows || {};
    const isVectorStripRows = node.presentation === VIEW2D_MATRIX_PRESENTATIONS.COMPACT_ROWS
        && compactRowsMeta.variant === VIEW2D_VECTOR_STRIP_VARIANT;
    if (!isVectorStripRows) return null;

    const stride = Math.max(1, rowHeight + rowGap);
    const approxRowIndex = Math.max(
        0,
        Math.min(
            rowItems.length - 1,
            Math.round(relativeY / stride)
        )
    );
    const approxBounds = resolveRowHitBounds(node, entry, approxRowIndex);
    if (!approxBounds) return null;
    return {
        rowIndex: approxRowIndex,
        rowItem: rowItems[approxRowIndex],
        bounds: approxBounds
    };
}

function resolveApproximateOverviewResidualRowHitAtScreenPoint(drawableNodes = [], renderState = null, x = 0, y = 0) {
    const worldScale = Number(renderState?.worldScale);
    if (!(worldScale > 0) || !Number.isFinite(x) || !Number.isFinite(y)) return null;

    const screenHitCache = renderState?.overviewScreenHitCache || null;
    const cachedNodes = Array.isArray(screenHitCache?.nodes) ? screenHitCache.nodes : null;
    const fallbackNodes = cachedNodes?.length
        ? cachedNodes
        : drawableNodes.map(({ node, entry }) => ({
            node,
            entry,
            screenEntryBounds: projectWorldBoundsToScreen(entry?.bounds || entry?.contentBounds, renderState),
            screenContentBounds: projectWorldBoundsToScreen(entry?.contentBounds || entry?.bounds, renderState)
        }));

    const resolveTopmostScreenNodeIdAtPoint = () => {
        for (let index = fallbackNodes.length - 1; index >= 0; index -= 1) {
            const drawable = fallbackNodes[index];
            const node = drawable?.node || null;
            const screenBounds = drawable?.screenEntryBounds || null;
            if (!node || !screenBounds || !containsPoint(screenBounds, x, y)) continue;
            return node.id;
        }
        return '';
    };
    const topmostScreenNodeId = (
        typeof screenHitCache?.resolveTopmostScreenNodeIdAtPoint === 'function'
            ? screenHitCache.resolveTopmostScreenNodeIdAtPoint(x, y)
            : resolveTopmostScreenNodeIdAtPoint()
    );

    let bestCandidate = null;
    let bestScore = Number.POSITIVE_INFINITY;
    fallbackNodes.forEach(({ node, entry, screenEntryBounds, screenContentBounds }) => {
        if (!isOverviewResidualVectorStripNode(node) || !entry?.contentBounds || !entry?.layoutData) {
            return;
        }
        const rowItems = Array.isArray(node.rowItems) ? node.rowItems : [];
        if (!rowItems.length) return;

        if (!screenEntryBounds || !screenContentBounds) return;

        const expandedEntryBounds = expandScreenBounds(screenEntryBounds, {
            minWidthPx: OVERVIEW_RESIDUAL_ROW_MIN_SCREEN_TARGET_WIDTH_PX,
            minHeightPx: OVERVIEW_RESIDUAL_ROW_MIN_SCREEN_TARGET_HEIGHT_PX,
            paddingPx: OVERVIEW_RESIDUAL_ROW_SCREEN_HOVER_PADDING_PX
        });
        if (!containsPoint(expandedEntryBounds, x, y)) return;

        const layoutData = entry.layoutData;
        const innerPaddingXWorld = Math.max(0, Number(layoutData.innerPaddingX) || 0);
        const innerPaddingYWorld = Math.max(0, Number(layoutData.innerPaddingY) || 0);
        const compactWidthWorld = Math.max(
            1,
            Math.min(
                Number(layoutData.compactWidth) || 0,
                Math.max(1, (Number(entry.contentBounds?.width) || 0) - (innerPaddingXWorld * 2))
            )
        );
        const rowTargetWidthWorld = Math.max(
            compactWidthWorld,
            Math.max(1, Number(entry.bounds?.width) || 0),
            Math.max(1, Number(screenEntryBounds.width) || 0) / worldScale
        );
        const rowAreaBounds = expandScreenBounds({
            x: screenEntryBounds.x,
            y: screenContentBounds.y + (innerPaddingYWorld * worldScale),
            width: rowTargetWidthWorld * worldScale,
            height: Math.max(
                1,
                (screenContentBounds.height - (innerPaddingYWorld * worldScale * 2))
            )
        }, {
            minWidthPx: OVERVIEW_RESIDUAL_ROW_MIN_SCREEN_TARGET_WIDTH_PX,
            minHeightPx: OVERVIEW_RESIDUAL_ROW_MIN_SCREEN_TARGET_HEIGHT_PX,
            paddingPx: OVERVIEW_RESIDUAL_ROW_SCREEN_HOVER_PADDING_PX
        });
        if (!containsPoint(rowAreaBounds, x, y)) return;
        if (topmostScreenNodeId && topmostScreenNodeId !== node.id) return;

        const rowHeightPx = Math.max(0.0001, (Number(layoutData.rowHeight) || 0) * worldScale);
        const rowGapPx = Math.max(0, (Number(layoutData.rowGap) || 0) * worldScale);
        const stridePx = Math.max(0.0001, rowHeightPx + rowGapPx);
        const rowCenterBaseY = screenContentBounds.y + (innerPaddingYWorld * worldScale) + (rowHeightPx * 0.5);
        const rowCenterX = screenEntryBounds.x + ((rowTargetWidthWorld * worldScale) * 0.5);

        let bestRowIndex = 0;
        let bestRowDistance = Number.POSITIVE_INFINITY;
        rowItems.forEach((_rowItem, rowIndex) => {
            const rowCenterY = rowCenterBaseY + (rowIndex * stridePx);
            const rowDistance = Math.abs(y - rowCenterY);
            if (rowDistance < bestRowDistance) {
                bestRowDistance = rowDistance;
                bestRowIndex = rowIndex;
            }
        });

        const score = bestRowDistance + (Math.abs(x - rowCenterX) * 0.05);
        if (score >= bestScore) return;
        const bounds = resolveRowHitBounds(node, entry, bestRowIndex);
        if (!bounds) return;

        bestScore = score;
        bestCandidate = {
            entry,
            node,
            rowHit: {
                rowIndex: bestRowIndex,
                rowItem: rowItems[bestRowIndex],
                bounds
            },
            cellHit: null,
            columnHit: null
        };
    });

    return bestCandidate;
}

function buildOverviewScreenHitCache(drawableNodes = [], renderState = null) {
    if (!Array.isArray(drawableNodes) || !drawableNodes.length) {
        return {
            nodes: [],
            resolveTopmostScreenNodeIdAtPoint: () => ''
        };
    }
    const nodes = drawableNodes.map(({ node, entry }) => ({
        node: node || null,
        entry: entry || null,
        screenEntryBounds: projectWorldBoundsToScreen(entry?.bounds || entry?.contentBounds, renderState),
        screenContentBounds: projectWorldBoundsToScreen(entry?.contentBounds || entry?.bounds, renderState)
    }));
    return {
        nodes,
        resolveTopmostScreenNodeIdAtPoint(x = 0, y = 0) {
            for (let index = nodes.length - 1; index >= 0; index -= 1) {
                const drawable = nodes[index];
                const node = drawable?.node || null;
                const screenBounds = drawable?.screenEntryBounds || null;
                if (!node || !screenBounds || !containsPoint(screenBounds, x, y)) continue;
                return node.id;
            }
            return '';
        }
    };
}

function resolveMatrixCellHit(node = null, entry = null, x = 0, y = 0) {
    if (!node || node.kind !== VIEW2D_NODE_KINDS.MATRIX) return null;
    if (!Number.isFinite(x) || !Number.isFinite(y)) return null;
    if (!containsPoint(entry?.contentBounds || entry?.bounds, x, y)) return null;
    if (node.presentation === VIEW2D_MATRIX_PRESENTATIONS.COMPACT_ROWS) {
        const compactRowsMeta = node.metadata?.compactRows || {};
        const interactiveBandHit = compactRowsMeta.interactiveBandHit === true || node.metadata?.interactiveBandHit === true;
        const isVectorStripRows = compactRowsMeta.variant === VIEW2D_VECTOR_STRIP_VARIANT;
        if (interactiveBandHit && isVectorStripRows) {
            const rowItems = Array.isArray(node.rowItems) ? node.rowItems : [];
            const layoutData = entry?.layoutData || null;
            const contentBounds = entry?.contentBounds || entry?.bounds || null;
            if (!rowItems.length || !layoutData || !contentBounds) return null;
            const innerPaddingX = Math.max(0, Number(layoutData.innerPaddingX) || 0);
            const innerPaddingY = Math.max(0, Number(layoutData.innerPaddingY) || 0);
            const rowHeight = Math.max(1, Number(layoutData.rowHeight) || 0);
            const rowGap = Math.max(0, Number(layoutData.rowGap) || 0);
            const compactWidth = Math.max(
                1,
                Math.min(
                    Number(layoutData.compactWidth) || 0,
                    contentBounds.width - (innerPaddingX * 2)
                )
            );
            const relativeY = y - (contentBounds.y + innerPaddingY);
            const resolvedRow = resolveTrackIndex(relativeY, rowHeight, rowGap, rowItems.length);
            if (!resolvedRow) return null;
            const bandCount = Number.isFinite(compactRowsMeta.bandCount)
                ? Math.max(1, Math.floor(compactRowsMeta.bandCount))
                : VIEW2D_VECTOR_STRIP_DEFAULTS.bandCount;
            const relativeX = x - (contentBounds.x + innerPaddingX);
            const resolvedBand = resolveTrackIndex(relativeX, compactWidth / bandCount, 0, bandCount);
            if (!resolvedBand) return null;
            const rowIndex = resolvedRow.index;
            const colIndex = resolvedBand.index;
            return {
                rowIndex,
                colIndex,
                rowItem: rowItems[rowIndex],
                cellItem: {
                    rowItem: rowItems[rowIndex],
                    bandIndex: colIndex,
                    semantic: {
                        ...(rowItems[rowIndex]?.semantic || node?.semantic || {}),
                        rowIndex,
                        colIndex
                    }
                },
                bounds: {
                    x: contentBounds.x + innerPaddingX + (resolvedBand.index * (compactWidth / bandCount)),
                    y: contentBounds.y + innerPaddingY + rowIndex * (rowHeight + rowGap),
                    width: compactWidth / bandCount,
                    height: rowHeight
                }
            };
        }
        return null;
    }
    if (node.presentation !== VIEW2D_MATRIX_PRESENTATIONS.GRID) return null;
    const rowItems = Array.isArray(node.rowItems) ? node.rowItems : [];
    const layoutData = entry?.layoutData || null;
    const contentBounds = entry?.contentBounds || entry?.bounds || null;
    if (!rowItems.length || !layoutData || !contentBounds) return null;
    const cellSize = Math.max(1, Number(layoutData.cellSize) || 0);
    const cellGap = Math.max(0, Number(layoutData.cellGap) || 0);
    const innerPaddingX = Math.max(0, Number(layoutData.innerPaddingX) || 0);
    const innerPaddingY = Math.max(0, Number(layoutData.innerPaddingY) || 0);
    const relativeY = y - (contentBounds.y + innerPaddingY);
    const resolvedRow = resolveTrackIndex(relativeY, cellSize, cellGap, rowItems.length);
    if (!resolvedRow) return null;

    const cells = Array.isArray(rowItems[resolvedRow.index]?.cells) ? rowItems[resolvedRow.index].cells : [];
    if (!cells.length) return null;

    const relativeX = x - (contentBounds.x + innerPaddingX);
    const resolvedCol = resolveTrackIndex(relativeX, cellSize, cellGap, cells.length);
    if (!resolvedCol) return null;

    return {
        rowIndex: resolvedRow.index,
        colIndex: resolvedCol.index,
        cellItem: cells[resolvedCol.index],
        bounds: {
            x: contentBounds.x + innerPaddingX + resolvedCol.index * (cellSize + cellGap),
            y: contentBounds.y + innerPaddingY + resolvedRow.index * (cellSize + cellGap),
            width: cellSize,
            height: cellSize
        }
    };
}

function resolveMatrixColumnHit(node = null, entry = null, x = 0, y = 0) {
    if (!node || node.kind !== VIEW2D_NODE_KINDS.MATRIX) return null;
    if (node.presentation !== VIEW2D_MATRIX_PRESENTATIONS.COLUMN_STRIP) return null;
    if (!Number.isFinite(x) || !Number.isFinite(y)) return null;
    if (!containsPoint(entry?.contentBounds || entry?.bounds, x, y)) return null;
    const columnItems = Array.isArray(node.columnItems) ? node.columnItems : [];
    const layoutData = entry?.layoutData || null;
    const contentBounds = entry?.contentBounds || entry?.bounds || null;
    if (!columnItems.length || !layoutData || !contentBounds) return null;
    const colWidth = Math.max(1, Number(layoutData.colWidth) || 0);
    const colGap = Math.max(0, Number(layoutData.colGap) || 0);
    const innerPaddingX = Math.max(0, Number(layoutData.innerPaddingX) || 0);
    const innerPaddingY = Math.max(0, Number(layoutData.innerPaddingY) || 0);
    const relativeX = x - (contentBounds.x + innerPaddingX);
    const resolvedCol = resolveTrackIndex(relativeX, colWidth, colGap, columnItems.length);
    if (!resolvedCol) return null;

    return {
        colIndex: resolvedCol.index,
        columnItem: columnItems[resolvedCol.index],
        bounds: {
            x: contentBounds.x + innerPaddingX + resolvedCol.index * (colWidth + colGap),
            y: contentBounds.y + innerPaddingY,
            width: colWidth,
            height: Math.max(1, Number(layoutData.colHeight) || 0)
        }
    };
}

function normalizeHeadDetailTarget(target = null) {
    if (!target || typeof target !== 'object') return null;
    const layerIndex = Number.isFinite(target.layerIndex) ? Math.max(0, Math.floor(target.layerIndex)) : null;
    const headIndex = Number.isFinite(target.headIndex) ? Math.max(0, Math.floor(target.headIndex)) : null;
    if (!Number.isFinite(layerIndex) || !Number.isFinite(headIndex)) return null;
    return {
        layerIndex,
        headIndex
    };
}

function buildHeadDetailSemanticTarget(target = null, role = 'head') {
    const resolvedTarget = normalizeHeadDetailTarget(target);
    if (!resolvedTarget) return null;
    return {
        componentKind: 'mhsa',
        layerIndex: resolvedTarget.layerIndex,
        headIndex: resolvedTarget.headIndex,
        stage: 'attention',
        role
    };
}

function normalizeConcatDetailTarget(target = null) {
    if (!target || typeof target !== 'object') return null;
    const layerIndex = Number.isFinite(target.layerIndex) ? Math.max(0, Math.floor(target.layerIndex)) : null;
    if (!Number.isFinite(layerIndex)) return null;
    return {
        layerIndex
    };
}

function buildConcatDetailSemanticTarget(target = null, role = 'concat') {
    const resolvedTarget = normalizeConcatDetailTarget(target);
    if (!resolvedTarget) return null;
    return {
        componentKind: 'mhsa',
        layerIndex: resolvedTarget.layerIndex,
        stage: 'concatenate',
        role
    };
}

function normalizeOutputProjectionDetailTarget(target = null) {
    if (!target || typeof target !== 'object') return null;
    const layerIndex = Number.isFinite(target.layerIndex) ? Math.max(0, Math.floor(target.layerIndex)) : null;
    if (!Number.isFinite(layerIndex)) return null;
    return {
        layerIndex
    };
}

function buildOutputProjectionDetailSemanticTarget(target = null, role = 'projection-weight') {
    const resolvedTarget = normalizeOutputProjectionDetailTarget(target);
    if (!resolvedTarget) return null;
    return {
        componentKind: 'output-projection',
        layerIndex: resolvedTarget.layerIndex,
        stage: 'attn-out',
        role
    };
}

function normalizeMlpDetailTarget(target = null) {
    if (!target || typeof target !== 'object') return null;
    const layerIndex = Number.isFinite(target.layerIndex) ? Math.max(0, Math.floor(target.layerIndex)) : null;
    if (!Number.isFinite(layerIndex)) return null;
    return {
        layerIndex
    };
}

function buildMlpDetailSemanticTarget(target = null, role = 'module') {
    const resolvedTarget = normalizeMlpDetailTarget(target);
    if (!resolvedTarget) return null;
    return {
        componentKind: 'mlp',
        layerIndex: resolvedTarget.layerIndex,
        stage: 'mlp',
        role
    };
}

function normalizeLayerNormDetailTarget(target = null) {
    if (!target || typeof target !== 'object') return null;
    const layerNormKind = String(target.layerNormKind || '').trim().toLowerCase();
    if (layerNormKind !== 'ln1' && layerNormKind !== 'ln2' && layerNormKind !== 'final') {
        return null;
    }
    const layerIndex = Number.isFinite(target.layerIndex) ? Math.max(0, Math.floor(target.layerIndex)) : null;
    if (layerNormKind === 'final') {
        return {
            layerNormKind,
            ...(Number.isFinite(layerIndex) ? { layerIndex } : {})
        };
    }
    if (!Number.isFinite(layerIndex)) return null;
    return {
        layerNormKind,
        layerIndex
    };
}

function buildLayerNormDetailSemanticTarget(target = null, role = 'module') {
    const resolvedTarget = normalizeLayerNormDetailTarget(target);
    if (!resolvedTarget) return null;
    return {
        componentKind: 'layer-norm',
        ...(Number.isFinite(resolvedTarget.layerIndex) ? { layerIndex: resolvedTarget.layerIndex } : {}),
        stage: resolvedTarget.layerNormKind === 'final' ? 'final-ln' : resolvedTarget.layerNormKind,
        role
    };
}

function drawHeadDetailStage(ctx, resolution, headDetailPreview = null, config = null) {
    if (!ctx || !resolution) return;
    const width = Math.max(1, Number(resolution.width) || 1);
    const height = Math.max(1, Number(resolution.height) || 1);
    const residualStyle = resolveView2dStyle(VIEW2D_STYLE_KEYS.RESIDUAL) || {};
    const previewRows = Array.isArray(headDetailPreview?.rowItems) ? headDetailPreview.rowItems : [];
    const copyCount = Math.max(1, Math.floor(headDetailPreview?.xLnCopies || 3));
    const rowCount = Math.max(1, previewRows.length || 4);
    const stackWidth = Math.max(120, Math.min(width * 0.24, 184));
    const rowGap = VIEW2D_VECTOR_STRIP_DEFAULTS.rowGap;
    const verticalInset = Math.max(28, Math.min(56, height * 0.14));
    const copyGap = Math.max(28, Math.min(52, height * 0.11));
    const innerPaddingY = VIEW2D_VECTOR_STRIP_DEFAULTS.paddingY;
    const stackHeightTarget = Math.max(132, height - (verticalInset * 2));
    const rowHeight = Math.max(
        5,
        Math.floor(
            (stackHeightTarget - (copyGap * (copyCount - 1)) - (innerPaddingY * 2 * copyCount) - (rowGap * ((rowCount - 1) * copyCount)))
            / Math.max(1, rowCount * copyCount)
        )
    );
    const copyHeight = (rowCount * rowHeight) + ((rowCount - 1) * rowGap) + (innerPaddingY * 2);
    const stackHeight = (copyCount * copyHeight) + ((copyCount - 1) * copyGap);
    const stackX = Math.max(88, Math.min(width * 0.5, width - stackWidth - 28));
    const stackTop = Math.max(verticalInset, (height - stackHeight) * 0.5);
    const stackBounds = {
        x: stackX,
        y: stackTop,
        width: stackWidth,
        height: stackHeight
    };
    const arrowY = height * 0.5;
    const arrowStartX = Math.max(0, width * 0.03);
    const branchOriginX = Math.max(28, Math.min(stackBounds.x - Math.max(42, width * 0.08), width * 0.18));
    const connectorStrokeWidth = Math.max(1, Math.min(width, height) * 0.0025);
    const arrowHeadInset = Math.max(7, Math.min(14, Math.min(width, height) * 0.02));
    const arrowTipOverlap = Math.max(1, connectorStrokeWidth * 0.75);
    const arrowEndX = stackBounds.x - arrowHeadInset;
    const rowItems = previewRows.length
        ? previewRows
        : Array.from({ length: rowCount }, (_, index) => ({
            gradientCss: residualStyle.fill || 'rgba(116, 240, 255, 0.84)',
            index
        }));

    ctx.save();
    ctx.fillStyle = '#000';
    ctx.fillRect(0, 0, width, height);

    ctx.strokeStyle = 'rgba(255, 255, 255, 0.54)';
    ctx.lineWidth = connectorStrokeWidth;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.beginPath();
    ctx.moveTo(arrowStartX, arrowY);
    ctx.lineTo(branchOriginX, arrowY);
    ctx.stroke();

    ctx.strokeStyle = 'rgba(255, 255, 255, 0.4)';
    for (let copyIndex = 0; copyIndex < copyCount; copyIndex += 1) {
        const targetY = stackBounds.y + copyIndex * (copyHeight + copyGap) + (copyHeight * 0.5);
        ctx.beginPath();
        ctx.moveTo(branchOriginX, arrowY);
        if (Math.abs(targetY - arrowY) > 0.5) {
            ctx.lineTo(branchOriginX, targetY);
        }
        ctx.lineTo(arrowEndX, targetY);
        ctx.stroke();
        drawConnectorArrowHead(
            ctx,
            { x: Math.max(branchOriginX, stackBounds.x - (arrowHeadInset * 2)), y: targetY },
            { x: stackBounds.x + arrowTipOverlap, y: targetY },
            ctx.lineWidth,
            'rgba(255, 255, 255, 0.72)'
        );
    }

    const accent = residualStyle.accent || config?.tokens?.palette?.neutral || 'rgba(116, 240, 255, 0.84)';
    const layoutData = {
        innerPaddingX: 0,
        innerPaddingY,
        compactWidth: stackBounds.width,
        rowHeight,
        rowGap
    };
    for (let copyIndex = 0; copyIndex < copyCount; copyIndex += 1) {
        drawVectorStripRows(
            ctx,
            rowItems,
            {
                x: stackBounds.x,
                y: stackBounds.y + copyIndex * (copyHeight + copyGap),
                width: stackBounds.width,
                height: copyHeight
            },
            layoutData,
            accent,
            8,
            {
                bandCount: 12
            }
        );
    }
    ctx.restore();
}

function drawConcatDetailStage(ctx, resolution, concatDetailPreview = null) {
    if (!ctx || !resolution) return;
    const width = Math.max(1, Number(resolution.width) || 1);
    const height = Math.max(1, Number(resolution.height) || 1);
    const arrowCount = Math.max(1, Math.floor(concatDetailPreview?.arrowCount || 12));
    const strokeWidth = Math.max(1.25, Math.min(width, height) * 0.0026);
    const connectorStroke = 'rgba(156, 162, 171, 0.78)';
    const arrowFill = 'rgba(188, 194, 204, 0.92)';
    const trackInsetY = Math.max(30, Math.min(42, height * 0.09));
    const trackTop = trackInsetY;
    const trackBottom = Math.max(trackTop, height - trackInsetY);
    const slotStep = arrowCount > 1 ? (trackBottom - trackTop) / (arrowCount - 1) : 0;
    const arrowEndX = Math.max(70, Math.min(width * 0.58, width - 64));
    const arrowLength = Math.max(42, Math.min(width * 0.08, 78));
    const arrowStartX = Math.max(24, arrowEndX - arrowLength);
    const arrowHeadInset = Math.max(6, strokeWidth * 3.8);
    const frameBounds = {
        x: Math.min(width - 56, arrowEndX + Math.max(18, width * 0.04)),
        y: Math.max(36, (height * 0.5) - Math.max(18, height * 0.06)),
        width: Math.max(88, Math.min(width * 0.16, 122)),
        height: Math.max(36, Math.min(height * 0.12, 52))
    };

    ctx.save();
    ctx.fillStyle = '#000';
    ctx.fillRect(0, 0, width, height);

    ctx.strokeStyle = connectorStroke;
    ctx.lineWidth = strokeWidth;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    for (let arrowIndex = 0; arrowIndex < arrowCount; arrowIndex += 1) {
        const targetY = trackTop + (slotStep * arrowIndex);
        ctx.beginPath();
        ctx.moveTo(arrowStartX, targetY);
        ctx.lineTo(arrowEndX - arrowHeadInset, targetY);
        ctx.stroke();
        drawConnectorArrowHead(
            ctx,
            { x: arrowEndX - arrowHeadInset, y: targetY },
            { x: arrowEndX, y: targetY },
            strokeWidth,
            arrowFill
        );
    }

    roundRectPath(ctx, frameBounds.x, frameBounds.y, frameBounds.width, frameBounds.height, 10);
    ctx.fillStyle = '#000';
    ctx.fill();
    ctx.strokeStyle = 'rgba(241, 244, 248, 0.96)';
    ctx.stroke();

    ctx.restore();
}

function drawOutputProjectionDetailStage(ctx, resolution, outputProjectionDetailPreview = null) {
    if (!ctx || !resolution) return;
    const width = Math.max(1, Number(resolution.width) || 1);
    const height = Math.max(1, Number(resolution.height) || 1);
    const detailStyle = resolveView2dStyle(VIEW2D_STYLE_KEYS.OUTPUT_PROJECTION) || {};
    const frameWidth = Math.max(132, Math.min(width * 0.15, 156));
    const frameHeight = Math.max(232, Math.min(height * 0.72, 372));
    const frameBounds = {
        x: Math.max(56, Math.min(width * 0.58, width - frameWidth - 44)),
        y: Math.max(28, (height - frameHeight) * 0.5),
        width: frameWidth,
        height: frameHeight
    };
    const arrowY = frameBounds.y + (frameBounds.height * 0.5);
    const arrowStartX = Math.max(34, width * 0.18);
    const arrowHeadInset = Math.max(6, Math.min(width, height) * 0.0065);
    const strokeWidth = Math.max(1.4, Math.min(width, height) * 0.0029);
    const frameStroke = detailStyle.stroke || detailStyle.accent || 'rgba(202, 92, 255, 0.96)';
    const connectorStroke = detailStyle.accent || 'rgba(224, 132, 255, 0.9)';
    const arrowCount = Math.max(1, Math.floor(outputProjectionDetailPreview?.arrowCount || 1));

    ctx.save();
    ctx.fillStyle = '#000';
    ctx.fillRect(0, 0, width, height);

    roundRectPath(ctx, frameBounds.x, frameBounds.y, frameBounds.width, frameBounds.height, 16);
    ctx.fillStyle = '#000';
    ctx.fill();
    ctx.lineWidth = strokeWidth;
    ctx.strokeStyle = frameStroke;
    ctx.stroke();

    ctx.strokeStyle = connectorStroke;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    for (let arrowIndex = 0; arrowIndex < arrowCount; arrowIndex += 1) {
        ctx.beginPath();
        ctx.moveTo(arrowStartX, arrowY);
        ctx.lineTo(frameBounds.x - arrowHeadInset, arrowY);
        ctx.stroke();
        drawConnectorArrowHead(
            ctx,
            { x: frameBounds.x - (arrowHeadInset * 2), y: arrowY },
            { x: frameBounds.x, y: arrowY },
            strokeWidth,
            frameStroke
        );
    }

    ctx.restore();
}

function isSelectedHeadCardNode(node = null, target = null) {
    const resolvedTarget = normalizeHeadDetailTarget(target);
    const semantic = node?.semantic || null;
    if (!resolvedTarget || !semantic) return false;
    return node?.role === 'head-card'
        && semantic.componentKind === 'mhsa'
        && semantic.stage === 'attention'
        && semantic.layerIndex === resolvedTarget.layerIndex
        && semantic.headIndex === resolvedTarget.headIndex;
}

function resolveHeadDetailFocusBounds(layout = null, target = null, role = 'head') {
    const registry = layout?.registry || null;
    if (!registry || typeof registry.resolveBoundsForSemanticTarget !== 'function') return null;
    return cloneBounds(registry.resolveBoundsForSemanticTarget(buildHeadDetailSemanticTarget(target, role)) || null);
}

function resolveConcatDetailFocusBounds(layout = null, target = null, role = 'concat') {
    const registry = layout?.registry || null;
    if (!registry || typeof registry.resolveBoundsForSemanticTarget !== 'function') return null;
    return cloneBounds(registry.resolveBoundsForSemanticTarget(buildConcatDetailSemanticTarget(target, role)) || null);
}

function resolveOutputProjectionDetailFocusBounds(layout = null, target = null, role = 'projection-weight') {
    const registry = layout?.registry || null;
    if (!registry || typeof registry.resolveBoundsForSemanticTarget !== 'function') return null;
    return cloneBounds(registry.resolveBoundsForSemanticTarget(buildOutputProjectionDetailSemanticTarget(target, role)) || null);
}

function resolveMlpDetailFocusBounds(layout = null, target = null, role = 'module') {
    const registry = layout?.registry || null;
    if (!registry || typeof registry.resolveBoundsForSemanticTarget !== 'function') return null;
    return cloneBounds(registry.resolveBoundsForSemanticTarget(buildMlpDetailSemanticTarget(target, role)) || null);
}

function resolveLayerNormDetailFocusBounds(layout = null, target = null, role = 'module') {
    const registry = layout?.registry || null;
    if (!registry || typeof registry.resolveBoundsForSemanticTarget !== 'function') return null;
    return cloneBounds(registry.resolveBoundsForSemanticTarget(buildLayerNormDetailSemanticTarget(target, role)) || null);
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

function resolveOverviewRenderCacheOverscanPx(resolution = null) {
    const width = Math.max(1, Number(resolution?.width) || 0);
    const height = Math.max(1, Number(resolution?.height) || 0);
    return Math.max(
        OVERVIEW_RENDER_CACHE_MIN_OVERSCAN_PX,
        Math.min(
            OVERVIEW_RENDER_CACHE_MAX_OVERSCAN_PX,
            Math.round(Math.min(width, height) * OVERVIEW_RENDER_CACHE_OVERSCAN_VIEWPORT_RATIO)
        )
    );
}

function resolveOverviewRenderCacheLogicalWidth(cache = null) {
    if (Number.isFinite(cache?.logicalWidth) && cache.logicalWidth > 0) {
        return Number(cache.logicalWidth);
    }
    const dpr = Number.isFinite(cache?.dpr) && cache.dpr > 0 ? Number(cache.dpr) : 1;
    return Math.max(1, (Number(cache?.pixelWidth) || 0) / dpr);
}

function resolveOverviewRenderCacheLogicalHeight(cache = null) {
    if (Number.isFinite(cache?.logicalHeight) && cache.logicalHeight > 0) {
        return Number(cache.logicalHeight);
    }
    const dpr = Number.isFinite(cache?.dpr) && cache.dpr > 0 ? Number(cache.dpr) : 1;
    return Math.max(1, (Number(cache?.pixelHeight) || 0) / dpr);
}

function resolveOverviewRenderCacheViewportPixelWidth(cache = null) {
    if (Number.isFinite(cache?.viewportPixelWidth) && cache.viewportPixelWidth > 0) {
        return Math.floor(Number(cache.viewportPixelWidth));
    }
    return Math.max(1, Math.floor(Number(cache?.pixelWidth) || 0));
}

function resolveOverviewRenderCacheViewportPixelHeight(cache = null) {
    if (Number.isFinite(cache?.viewportPixelHeight) && cache.viewportPixelHeight > 0) {
        return Math.floor(Number(cache.viewportPixelHeight));
    }
    return Math.max(1, Math.floor(Number(cache?.pixelHeight) || 0));
}

function resolveOverviewRenderCacheViewportOffsetX(cache = null) {
    if (Number.isFinite(cache?.viewportOffsetX)) {
        return Number(cache.viewportOffsetX);
    }
    return Number.isFinite(cache?.offsetX) ? Number(cache.offsetX) : 0;
}

function resolveOverviewRenderCacheViewportOffsetY(cache = null) {
    if (Number.isFinite(cache?.viewportOffsetY)) {
        return Number(cache.viewportOffsetY);
    }
    return Number.isFinite(cache?.offsetY) ? Number(cache.offsetY) : 0;
}

function resolveOverviewRenderCacheRenderOffsetX(cache = null) {
    if (Number.isFinite(cache?.renderOffsetX)) {
        return Number(cache.renderOffsetX);
    }
    return Number.isFinite(cache?.offsetX) ? Number(cache.offsetX) : 0;
}

function resolveOverviewRenderCacheRenderOffsetY(cache = null) {
    if (Number.isFinite(cache?.renderOffsetY)) {
        return Number(cache.renderOffsetY);
    }
    return Number.isFinite(cache?.offsetY) ? Number(cache.offsetY) : 0;
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

function drawCaption(ctx, entry, node, config, worldScale = 1, detailScale = worldScale, focusAlpha = 1, fixedTextSizing = null) {
    if (String(node?.metadata?.caption?.renderMode || '').trim().toLowerCase() === 'dom-katex') {
        return;
    }
    const safeWorldScale = Math.max(0.0001, Number.isFinite(worldScale) ? worldScale : 1);
    const safeDetailScale = Math.max(0.0001, Number.isFinite(detailScale) ? detailScale : safeWorldScale);
    const revealScreenFontPx = (
        Number.isFinite(fixedTextSizing?.captionLabelScreenFontPx) && fixedTextSizing.captionLabelScreenFontPx > 0
            ? Number(fixedTextSizing.captionLabelScreenFontPx)
            : ((Number.isFinite(config?.component?.labelFontSize) ? Number(config.component.labelFontSize) : 12) * safeDetailScale)
    );
    if (revealScreenFontPx < TEXT_MIN_SCREEN_HEIGHT_PX) {
        return;
    }
    const captionLines = resolveView2dCaptionLines(node);
    if (!captionLines.length) return;
    const captionPosition = resolveView2dCaptionPosition(node);
    const lines = [];

    if (captionPosition === 'inside-top') {
        const contentBounds = entry.contentBounds || entry.bounds;
        if (!contentBounds) return;
        const lineHeight = config.component.captionLineHeight;
        const insetX = Math.max(4, Math.min(10, contentBounds.width * 0.08));
        const insetY = Math.max(3, Math.min(6, contentBounds.height * 0.12));
        captionLines.forEach((line, index) => {
            lines.push({
                ...line,
                bounds: {
                    x: contentBounds.x + insetX,
                    y: contentBounds.y + insetY + (index * lineHeight),
                    width: Math.max(1, contentBounds.width - (insetX * 2)),
                    height: lineHeight
                }
            });
        });
    } else {
        if (entry.labelBounds && captionLines[0]) {
            lines.push({
                ...captionLines[0],
                bounds: entry.labelBounds
            });
        }
        if (entry.dimensionBounds && captionLines[1]) {
            lines.push({
                ...captionLines[1],
                bounds: entry.dimensionBounds
            });
        }
    }

    if (!lines.length) return;

    const captionStyle = resolveView2dStyle(
        resolveView2dCaptionStyleKey(node, VIEW2D_STYLE_KEYS.CAPTION)
    ) || {};
    const captionColor = captionStyle.color || config.tokens.palette.mutedText;
    const captionFontSize = fixedTextSizing?.captionLabelScreenFontPx
        ? (fixedTextSizing.captionLabelScreenFontPx / safeWorldScale)
        : config.component.captionFontSize;
    const captionDimensionsFontSize = fixedTextSizing?.captionDimensionsScreenFontPx
        ? (fixedTextSizing.captionDimensionsScreenFontPx / safeWorldScale)
        : captionFontSize;
    const visualOpacity = resolveNodeVisualOpacity(node);

    ctx.save();
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillStyle = captionColor;
    ctx.globalAlpha = Math.max(0, Math.min(1, Number.isFinite(focusAlpha) ? focusAlpha : 1))
        * visualOpacity;
    if (captionPosition === 'inside-top') {
        const clipBounds = entry.contentBounds || entry.bounds;
        if (clipBounds) {
            ctx.beginPath();
            ctx.rect(clipBounds.x, clipBounds.y, clipBounds.width, clipBounds.height);
            ctx.clip();
        }
    }
    lines.forEach(({ text, tex, bounds }, index) => {
        const lineFontSize = index === 0 ? captionFontSize : captionDimensionsFontSize;
        ctx.font = `500 ${lineFontSize}px ui-monospace, SFMono-Regular, Menlo, monospace`;
        const displayTex = typeof tex === 'string' ? tex : '';
        if (displayTex.length && hasSimpleTexMarkup(displayTex)) {
            drawSimpleTex(ctx, displayTex, {
                x: bounds.x + (bounds.width / 2),
                y: bounds.y + (bounds.height / 2),
                fontSize: lineFontSize,
                fontWeight: 500,
                color: captionColor
            });
            return;
        }
        ctx.fillText(text || displayTex, bounds.x + (bounds.width / 2), bounds.y + (bounds.height / 2));
    });
    ctx.restore();
}

function drawCardSurfaceEffects(
    ctx,
    bounds,
    cornerRadius,
    style,
    safeWorldScale,
    projectedWidth,
    projectedHeight,
    focusAlpha = 1,
    cardMetadata = null
) {
    if (!bounds) return;
    const baseAlpha = Math.max(0, Math.min(1, Number.isFinite(focusAlpha) ? focusAlpha : 1));
    if (baseAlpha <= 0) return;

    const glowColor = typeof style.cardGlowColor === 'string' ? style.cardGlowColor : null;
    const glowBlur = Number.isFinite(style.cardGlowBlur) ? style.cardGlowBlur : 16;
    const glowOpacity = Number.isFinite(style.cardGlowOpacity) ? style.cardGlowOpacity : 0.2;
    const hotspotColor = typeof style.cardHotspotColor === 'string' ? style.cardHotspotColor : 'rgba(255,255,255,0.12)';
    const innerGlowColor = typeof style.cardInnerGlowColor === 'string' ? style.cardInnerGlowColor : null;
    const sheenColor = typeof style.cardSheenColor === 'string' ? style.cardSheenColor : 'rgba(255,255,255,0.14)';
    const edgeHighlight = typeof style.cardEdgeHighlight === 'string' ? style.cardEdgeHighlight : 'rgba(255,255,255,0.18)';

    if (glowColor) {
        ctx.save();
        traceCardPath(ctx, bounds, cornerRadius, cardMetadata);
        ctx.fillStyle = glowColor;
        ctx.globalAlpha = glowOpacity * baseAlpha;
        ctx.shadowColor = glowColor;
        ctx.shadowBlur = glowBlur / safeWorldScale;
        ctx.fill();
        ctx.restore();

        ctx.save();
        traceCardPath(ctx, bounds, cornerRadius, cardMetadata);
        ctx.lineWidth = Math.max(0.7, 1.2) / safeWorldScale;
        ctx.strokeStyle = glowColor;
        ctx.shadowColor = glowColor;
        ctx.shadowBlur = (glowBlur * 0.8) / safeWorldScale;
        ctx.globalAlpha = Math.min(1, glowOpacity + 0.28) * baseAlpha;
        ctx.stroke();
        ctx.restore();
    }

    ctx.save();
    traceCardPath(ctx, bounds, cornerRadius, cardMetadata);
    ctx.clip();

    const sheen = ctx.createLinearGradient(bounds.x, bounds.y, bounds.x + bounds.width, bounds.y + bounds.height);
    sheen.addColorStop(0, sheenColor);
    sheen.addColorStop(0.24, 'rgba(255,255,255,0.06)');
    sheen.addColorStop(0.52, 'rgba(255,255,255,0.0)');
    sheen.addColorStop(1, 'rgba(255,255,255,0.04)');
    ctx.fillStyle = sheen;
    ctx.fillRect(bounds.x, bounds.y, bounds.width, bounds.height);

    if (innerGlowColor) {
        const innerGlow = ctx.createRadialGradient(
            bounds.x + (bounds.width * 0.48),
            bounds.y + (bounds.height * 0.5),
            0,
            bounds.x + (bounds.width * 0.48),
            bounds.y + (bounds.height * 0.5),
            Math.max(bounds.width, bounds.height) * 0.72
        );
        innerGlow.addColorStop(0, innerGlowColor);
        innerGlow.addColorStop(0.56, 'rgba(255,255,255,0.0)');
        innerGlow.addColorStop(1, 'rgba(255,255,255,0.0)');
        ctx.fillStyle = innerGlow;
        ctx.fillRect(bounds.x, bounds.y, bounds.width, bounds.height);
    }

    const hotspot = ctx.createRadialGradient(
        bounds.x + (bounds.width * 0.72),
        bounds.y + (bounds.height * 0.18),
        0,
        bounds.x + (bounds.width * 0.72),
        bounds.y + (bounds.height * 0.18),
        Math.max(bounds.width, bounds.height) * 0.7
    );
    hotspot.addColorStop(0, hotspotColor);
    hotspot.addColorStop(0.42, 'rgba(255,255,255,0.0)');
    hotspot.addColorStop(1, 'rgba(255,255,255,0.0)');
    ctx.fillStyle = hotspot;
    ctx.fillRect(bounds.x, bounds.y, bounds.width, bounds.height);
    ctx.restore();

    ctx.save();
    traceCardPath(ctx, bounds, cornerRadius, cardMetadata);
    ctx.lineWidth = Math.max(0.45, 0.8) / safeWorldScale;
    ctx.strokeStyle = edgeHighlight;
    ctx.globalAlpha = baseAlpha;
    ctx.stroke();
    ctx.restore();
}

function drawCardEdgeStrokes(ctx, bounds, cornerRadius, style, safeWorldScale, focusAlpha = 1) {
    if (!bounds) return;
    const baseAlpha = Math.max(0, Math.min(1, Number.isFinite(focusAlpha) ? focusAlpha : 1));
    if (baseAlpha <= 0) return;
    const lineWidth = Math.max(
        0.8,
        Number.isFinite(style.edgeStrokeWidth) ? style.edgeStrokeWidth : 2.8
    ) / Math.max(0.0001, safeWorldScale);
    const gradientStops = Array.isArray(style?.edgeStrokeGradientStops)
        ? style.edgeStrokeGradientStops
        : null;
    const strokeGradient = gradientStops?.length
        ? (() => {
            const gradient = ctx.createLinearGradient(bounds.x, bounds.y, bounds.x + bounds.width, bounds.y);
            gradientStops.forEach((stop) => {
                const offset = Math.max(0, Math.min(1, Number(stop?.offset) || 0));
                const color = typeof stop?.color === 'string' && stop.color.length
                    ? stop.color
                    : null;
                if (color) {
                    gradient.addColorStop(offset, color);
                }
            });
            return gradient;
        })()
        : null;
    const edgeStrokeColors = style?.edgeStrokeColors;
    const inset = lineWidth * 0.5;
    const arcInset = Math.max(cornerRadius * 0.75, inset + 2);

    if (strokeGradient) {
        ctx.save();
        roundRectPath(
            ctx,
            bounds.x + inset,
            bounds.y + inset,
            Math.max(1, bounds.width - lineWidth),
            Math.max(1, bounds.height - lineWidth),
            Math.max(0, cornerRadius - inset)
        );
        ctx.lineWidth = lineWidth;
        ctx.strokeStyle = strokeGradient;
        ctx.globalAlpha = baseAlpha;
        ctx.stroke();
        ctx.restore();
        return;
    }

    if (!edgeStrokeColors || typeof edgeStrokeColors !== 'object') return;
    const drawSegment = (color, startX, startY, endX, endY) => {
        if (typeof color !== 'string' || !color.length) return;
        ctx.save();
        ctx.beginPath();
        ctx.moveTo(startX, startY);
        ctx.lineTo(endX, endY);
        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';
        ctx.lineWidth = lineWidth;
        ctx.strokeStyle = color;
        ctx.globalAlpha = baseAlpha;
        ctx.stroke();
        ctx.restore();
    };

    drawSegment(
        edgeStrokeColors.left,
        bounds.x + inset,
        bounds.y + arcInset,
        bounds.x + inset,
        bounds.y + bounds.height - arcInset
    );
    drawSegment(
        edgeStrokeColors.horizontal,
        bounds.x + arcInset,
        bounds.y + inset,
        bounds.x + bounds.width - arcInset,
        bounds.y + inset
    );
    drawSegment(
        edgeStrokeColors.horizontal,
        bounds.x + arcInset,
        bounds.y + bounds.height - inset,
        bounds.x + bounds.width - arcInset,
        bounds.y + bounds.height - inset
    );
    drawSegment(
        edgeStrokeColors.right,
        bounds.x + bounds.width - inset,
        bounds.y + arcInset,
        bounds.x + bounds.width - inset,
        bounds.y + bounds.height - arcInset
    );
}

function resolveEmbeddedSceneRenderCache(embeddedScene = null) {
    if (!embeddedScene?.scene || !embeddedScene?.layout?.registry) return null;
    if (embeddedScene.renderCache) {
        return embeddedScene.renderCache;
    }

    const allNodes = flattenSceneNodes(embeddedScene.scene);
    embeddedScene.renderCache = {
        drawableNodes: allNodes
            .filter((node) => ![VIEW2D_NODE_KINDS.CONNECTOR, VIEW2D_NODE_KINDS.GROUP].includes(node.kind))
            .filter((node) => !node?.metadata?.hidden)
            .map((node) => ({
                node,
                entry: embeddedScene.layout.registry.getNodeEntry(node.id) || null
            }))
            .filter((item) => item.entry),
        connectors: allNodes
            .filter((node) => node.kind === VIEW2D_NODE_KINDS.CONNECTOR)
            .map((node) => ({
                entry: embeddedScene.layout.registry.getConnectorEntry(node.id) || null,
                stroke: (
                    (typeof node.visual?.stroke === 'string' && node.visual.stroke.length)
                        ? node.visual.stroke
                        : (resolveView2dStyle(node.visual?.styleKey)?.stroke || null)
                )
            }))
            .filter((item) => item.entry)
    };
    return embeddedScene.renderCache;
}

function drawEmbeddedScene(
    ctx,
    contentBounds,
    embeddedScene,
    config,
    worldScale,
    detailScale,
    cornerRadius,
    {
        fastPath = false,
        focusAlpha = 1
    } = {}
) {
    if (!contentBounds || !embeddedScene?.scene || !embeddedScene?.layout?.registry) return;
    const renderCache = resolveEmbeddedSceneRenderCache(embeddedScene);
    const sceneBounds = embeddedScene.layout.sceneBounds || embeddedScene.layout.registry.getSceneBounds();
    if (!renderCache || !sceneBounds?.width || !sceneBounds?.height) return;
    const embeddedFocusAlpha = Math.max(0, Math.min(1, Number.isFinite(focusAlpha) ? focusAlpha : 1));

    const paddingX = Math.max(0, Number(embeddedScene.paddingX) || 0);
    const paddingY = Math.max(0, Number(embeddedScene.paddingY) || 0);
    const viewportBounds = {
        x: contentBounds.x + paddingX,
        y: contentBounds.y + paddingY,
        width: Math.max(1, contentBounds.width - (paddingX * 2)),
        height: Math.max(1, contentBounds.height - (paddingY * 2))
    };
    const nestedScale = Math.min(
        viewportBounds.width / Math.max(1, sceneBounds.width),
        viewportBounds.height / Math.max(1, sceneBounds.height)
    );
    if (!(nestedScale > 0)) return;

    const offsetX = viewportBounds.x + ((viewportBounds.width - (sceneBounds.width * nestedScale)) / 2) - (sceneBounds.x * nestedScale);
    const offsetY = viewportBounds.y + ((viewportBounds.height - (sceneBounds.height * nestedScale)) / 2) - (sceneBounds.y * nestedScale);
    const nestedConfig = embeddedScene.layout?.config || config;
    const nestedWorldScale = Math.max(0.0001, worldScale * nestedScale);
    const nestedDetailScale = Math.max(0.0001, detailScale * nestedScale);
    const nestedFixedTextSizing = resolveMhsaDetailFixedTextSizing(renderCache.scene, viewportBounds.width * worldScale);

    ctx.save();
    roundRectPath(
        ctx,
        viewportBounds.x,
        viewportBounds.y,
        viewportBounds.width,
        viewportBounds.height,
        Math.max(0, cornerRadius - 6)
    );
    ctx.clip();
    ctx.translate(offsetX, offsetY);
    ctx.scale(nestedScale, nestedScale);

    renderCache.drawableNodes.forEach(({ node, entry }) => {
        if (node.kind === VIEW2D_NODE_KINDS.MATRIX) {
            drawMatrixNode(ctx, node, entry, nestedConfig, nestedWorldScale, nestedDetailScale, {
                skipSurfaceEffects: fastPath,
                fastPath,
                focusAlpha: embeddedFocusAlpha,
                fixedTextSizing: nestedFixedTextSizing
            });
        } else if (node.kind === VIEW2D_NODE_KINDS.TEXT || node.kind === VIEW2D_NODE_KINDS.OPERATOR) {
            drawTextLikeNode(
                ctx,
                node,
                entry,
                nestedConfig,
                nestedWorldScale,
                nestedDetailScale,
                embeddedFocusAlpha,
                nestedFixedTextSizing
            );
        }
    });

    renderCache.connectors.forEach(({ entry, stroke }) => {
        drawConnector(ctx, entry, nestedConfig, stroke, nestedWorldScale, {
            focusAlpha: embeddedFocusAlpha,
            emphasize: embeddedFocusAlpha >= 0.995
        });
    });

    ctx.restore();
}

function drawMatrixNode(
    ctx,
    node,
    entry,
    config,
    worldScale = 1,
    detailScale = worldScale,
    {
        skipSurfaceEffects = false,
        fastPath = false,
        headDetailTarget = null,
        headDetailDepthActive = false,
        interactionState = null,
        focusAlpha = 1,
        fixedTextSizing = null,
        disableInactiveFilter = false
    } = {}
) {
    const contentBounds = entry.contentBounds || entry.bounds;
    const resolvedStyle = (
        headDetailDepthActive && isSelectedHeadCardNode(node, headDetailTarget)
            ? resolveView2dStyle(VIEW2D_STYLE_KEYS.MHSA_HEAD_DETAIL_FRAME)
            : resolveView2dStyle(node.visual?.styleKey)
    ) || {};
    const style = {
        ...resolvedStyle,
        ...(typeof node.visual?.accent === 'string' && node.visual.accent.length
            ? { accent: node.visual.accent }
            : {}),
        ...(typeof node.visual?.background === 'string' && node.visual.background.length
            ? { fill: node.visual.background }
            : {}),
        ...(typeof node.visual?.stroke === 'string' && node.visual.stroke.length
            ? { stroke: node.visual.stroke }
            : {}),
        ...(Object.prototype.hasOwnProperty.call(node.visual || {}, 'disableCardSurfaceEffects')
            ? { disableCardSurfaceEffects: node.visual.disableCardSurfaceEffects }
            : {})
    };
    const accent = style.accent || config.tokens.palette.neutral;
    const background = style.fill || config.tokens.palette.panelBackground;
    const border = style.stroke || config.tokens.palette.border;
    const compactRowsMeta = node.metadata?.compactRows || {};
    const columnStripMeta = node.metadata?.columnStrip || {};
    const gridMeta = node.metadata?.grid || {};
    const isVectorStripRows = node.presentation === VIEW2D_MATRIX_PRESENTATIONS.COMPACT_ROWS
        && compactRowsMeta.variant === VIEW2D_VECTOR_STRIP_VARIANT;
    const isVectorStripColumns = node.presentation === VIEW2D_MATRIX_PRESENTATIONS.COLUMN_STRIP
        && columnStripMeta.variant === VIEW2D_VECTOR_STRIP_VARIANT;
    const preserveGridDetail = node.presentation === VIEW2D_MATRIX_PRESENTATIONS.GRID
        && gridMeta.preserveDetail === true;
    const hideSurface = compactRowsMeta.hideSurface === true || columnStripMeta.hideSurface === true;
    const vectorStripBandCount = Number.isFinite(compactRowsMeta.bandCount)
        ? Math.max(1, Math.floor(compactRowsMeta.bandCount))
        : 12;
    const vectorStripBandSeparatorOpacity = Number.isFinite(compactRowsMeta.bandSeparatorOpacity)
        ? Math.max(0, Math.min(1, Number(compactRowsMeta.bandSeparatorOpacity)))
        : 0;
    const vectorStripHoverScaleY = fastPath
        ? 1
        : (
            Number.isFinite(compactRowsMeta.hoverScaleY)
                ? Math.max(1, Number(compactRowsMeta.hoverScaleY))
                : 1
        );
    const vectorStripHoverGlowColor = fastPath
        ? null
        : (
            typeof compactRowsMeta.hoverGlowColor === 'string'
                ? compactRowsMeta.hoverGlowColor
                : null
        );
    const vectorStripHoverGlowBlur = fastPath
        ? 0
        : (
            Number.isFinite(compactRowsMeta.hoverGlowBlur)
                ? Math.max(0, Number(compactRowsMeta.hoverGlowBlur))
                : 0
        );
    const vectorStripHoverStrokeColor = fastPath
        ? null
        : (
            typeof compactRowsMeta.hoverStrokeColor === 'string'
                ? compactRowsMeta.hoverStrokeColor
                : null
        );
    const vectorStripDimmedRowOpacity = Number.isFinite(compactRowsMeta.dimmedRowOpacity)
        ? Math.max(0, Math.min(1, Number(compactRowsMeta.dimmedRowOpacity)))
        : 0.18;
    const vectorStripHoverScaleX = fastPath
        ? 1
        : (
            Number.isFinite(columnStripMeta.hoverScaleX)
                ? Math.max(1, Number(columnStripMeta.hoverScaleX))
                : 1
        );
    const vectorStripDimmedColumnOpacity = Number.isFinite(columnStripMeta.dimmedColumnOpacity)
        ? Math.max(0, Math.min(1, Number(columnStripMeta.dimmedColumnOpacity)))
        : 0.18;
    const safeWorldScale = Math.max(0.0001, Number.isFinite(worldScale) ? worldScale : 1);
    const safeDetailScale = Math.max(0.0001, Number.isFinite(detailScale) ? detailScale : safeWorldScale);
    const cornerRadius = Number.isFinite(entry.layoutData?.cardRadius)
        ? entry.layoutData.cardRadius
        : config.tokens.matrix.cornerRadius;
    const cardMetadata = node.metadata?.card || null;
    const projectedWidth = contentBounds.width * safeDetailScale;
    const projectedHeight = contentBounds.height * safeDetailScale;
    const summaryWidthThreshold = MATRIX_DETAIL_MIN_SCREEN_WIDTH_PX * (fastPath ? 1.45 : 1);
    const summaryHeightThreshold = MATRIX_DETAIL_MIN_SCREEN_HEIGHT_PX * (fastPath ? 1.35 : 1);
    const useSummaryInterior = !preserveGridDetail && (
        projectedWidth < summaryWidthThreshold
        || projectedHeight < summaryHeightThreshold
    );
    const embeddedScene = node.metadata?.embeddedScene || null;
    const visualOpacity = resolveNodeVisualOpacity(node);
    const nodeFocusAlpha = Math.max(0, Math.min(1, Number.isFinite(focusAlpha) ? focusAlpha : 1))
        * visualOpacity;
    const sceneFocusState = interactionState?.sceneFocusState || null;
    const surfaceFocusAlpha = resolveMatrixSurfaceFocusAlpha(node, sceneFocusState, nodeFocusAlpha);
    const inactiveNodeFilter = resolveSceneNodeFilter(node.id, sceneFocusState, nodeFocusAlpha, config, {
        disableInactiveFilter
    });

    ctx.save();
    ctx.globalAlpha = surfaceFocusAlpha;
    ctx.filter = inactiveNodeFilter;
    if (!hideSurface) {
        traceCardPath(ctx, contentBounds, cornerRadius, cardMetadata);
        ctx.fillStyle = resolveFill(ctx, background, contentBounds, accent);
        ctx.fill();
        ctx.lineWidth = config.tokens.matrix.borderWidth;
        ctx.strokeStyle = border;
        ctx.stroke();
        if (
            !skipSurfaceEffects
            && node.presentation === VIEW2D_MATRIX_PRESENTATIONS.CARD
            && style.disableCardSurfaceEffects !== true
        ) {
            drawCardSurfaceEffects(
                ctx,
                contentBounds,
                cornerRadius,
                style,
                safeWorldScale,
                projectedWidth,
                projectedHeight,
                surfaceFocusAlpha,
                cardMetadata
            );
        }
    }

    if (node.presentation === VIEW2D_MATRIX_PRESENTATIONS.CARD && embeddedScene) {
        drawEmbeddedScene(
            ctx,
            contentBounds,
            embeddedScene,
            config,
            safeWorldScale,
            safeDetailScale,
            cornerRadius,
            {
                fastPath,
                focusAlpha: nodeFocusAlpha
            }
        );
    }

    const layoutData = entry.layoutData || {};
    const rowItems = Array.isArray(node.rowItems) ? node.rowItems : [];
    const columnItems = Array.isArray(node.columnItems) ? node.columnItems : [];
    const hoveredRowIndex = resolveMatrixHoveredRowIndex(node, interactionState?.hoveredRow);
    const previousHoveredRowIndex = resolveMatrixHoveredRowIndex(node, interactionState?.previousHoveredRow);
    const hoverRowBlend = Math.max(0, Math.min(1, Number(interactionState?.hoverRowBlend) || 1));
    const explicitHoverDimStrength = Number(interactionState?.hoverDimStrength);
    const hoverDimStrength = rowItems.length > 1
        ? (
            Number.isFinite(explicitHoverDimStrength)
                ? Math.max(0, Math.min(1, explicitHoverDimStrength))
                : (Number.isFinite(hoveredRowIndex) ? 1 : 0)
        )
        : 0;
    const shouldDimRows = hoverDimStrength > 0 && rowItems.length > 1;
    const dimmedRowOpacity = vectorStripDimmedRowOpacity;
    ctx.globalAlpha = nodeFocusAlpha;
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
            const showRowLabels = !fastPath
                && (layoutData.rowHeight * safeDetailScale) >= CAPTION_MIN_SCREEN_HEIGHT_PX;
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
                const sceneRowAlpha = resolveSceneRowSelectionAlpha(node.id, index, sceneFocusState);
                ctx.globalAlpha = nodeFocusAlpha * (sceneRowAlpha ?? resolveRowDimmingAlpha(index, hoveredRowIndex, {
                    previousHoveredRowIndex,
                    hoverRowBlend,
                    dimStrength: hoverDimStrength,
                    dimmedRowOpacity
                }));
                ctx.fill();
                ctx.globalAlpha = nodeFocusAlpha;

                if (showRowLabels) {
                    ctx.fillStyle = config.tokens.palette.mutedText;
                    const rowLabelFontSize = fixedTextSizing?.rowLabelScreenFontPx
                        ? (fixedTextSizing.rowLabelScreenFontPx / safeWorldScale)
                        : config.component.captionFontSize;
                    ctx.font = `500 ${rowLabelFontSize}px ui-monospace, SFMono-Regular, Menlo, monospace`;
                    ctx.textAlign = 'left';
                    ctx.textBaseline = 'middle';
                    const sceneLabelAlpha = resolveSceneRowSelectionAlpha(node.id, index, sceneFocusState);
                    ctx.globalAlpha = nodeFocusAlpha * (sceneLabelAlpha ?? resolveRowDimmingAlpha(index, hoveredRowIndex, {
                        previousHoveredRowIndex,
                        hoverRowBlend,
                        dimStrength: hoverDimStrength,
                        dimmedRowOpacity
                    }));
                    ctx.fillText(
                        rowItem.label || '',
                        contentBounds.x + layoutData.innerPaddingX,
                        rowY + (layoutData.rowHeight / 2)
                    );
                    ctx.globalAlpha = nodeFocusAlpha;
                }
            });
        }
    } else if (node.presentation === VIEW2D_MATRIX_PRESENTATIONS.COMPACT_ROWS) {
        if (isVectorStripRows) {
            const sceneRowSelections = sceneFocusState?.rowSelections?.get(node.id) || null;
            const sceneFocusedRowIndices = sceneRowSelections?.size
                ? Array.from(sceneRowSelections)
                : null;
            const sceneCellSelections = sceneFocusState?.cellSelections?.get(node.id) || null;
            const sceneFocusedBandCells = sceneCellSelections?.size
                ? Array.from(sceneCellSelections).map((value) => {
                    const [rowIndex, bandIndex] = String(value || '').split(':').map((part) => Number.parseInt(part, 10));
                    return {
                        rowIndex,
                        bandIndex
                    };
                }).filter((selection) => (
                    Number.isFinite(selection.rowIndex)
                    && Number.isFinite(selection.bandIndex)
                ))
                : null;
            const focusedRowBaseAlpha = node?.metadata?.nextCachePreviewNode === true
                ? Math.max(0, Math.min(1, Number.isFinite(focusAlpha) ? focusAlpha : 1))
                : nodeFocusAlpha;
            drawVectorStripRows(
                ctx,
                rowItems,
                contentBounds,
                layoutData,
                accent,
                cornerRadius,
                sceneFocusedRowIndices?.length
                    ? {
                        focusedRowIndices: sceneFocusedRowIndices,
                        dimmedRowOpacity: sceneFocusState?.inactiveOpacity || dimmedRowOpacity,
                        baseAlpha: nodeFocusAlpha,
                        focusedRowBaseAlpha,
                        bandCount: vectorStripBandCount,
                        bandSeparatorOpacity: vectorStripBandSeparatorOpacity,
                        focusedBandCells: sceneFocusedBandCells,
                        hoverScaleY: vectorStripHoverScaleY,
                        hoverGlowColor: vectorStripHoverGlowColor,
                        hoverGlowBlur: vectorStripHoverGlowBlur,
                        hoverStrokeColor: vectorStripHoverStrokeColor
                    }
                    : (
                        shouldDimRows
                            ? {
                                hoveredRowIndex,
                                previousHoveredRowIndex,
                                hoverRowBlend,
                                dimmingStrength: hoverDimStrength,
                                dimmedRowOpacity,
                                baseAlpha: nodeFocusAlpha,
                                focusedRowBaseAlpha,
                                bandCount: vectorStripBandCount,
                                bandSeparatorOpacity: vectorStripBandSeparatorOpacity,
                                focusedBandCells: sceneFocusedBandCells,
                                hoverScaleY: vectorStripHoverScaleY,
                                hoverGlowColor: vectorStripHoverGlowColor,
                                hoverGlowBlur: vectorStripHoverGlowBlur,
                                hoverStrokeColor: vectorStripHoverStrokeColor
                            }
                            : {
                                baseAlpha: nodeFocusAlpha,
                                focusedRowBaseAlpha,
                                bandCount: vectorStripBandCount,
                                bandSeparatorOpacity: vectorStripBandSeparatorOpacity,
                                focusedBandCells: sceneFocusedBandCells,
                                hoverScaleY: vectorStripHoverScaleY,
                                hoverGlowColor: vectorStripHoverGlowColor,
                                hoverGlowBlur: vectorStripHoverGlowBlur,
                                hoverStrokeColor: vectorStripHoverStrokeColor
                            }
                    )
            );
        } else if (useSummaryInterior) {
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
                const sceneRowAlpha = resolveSceneRowSelectionAlpha(node.id, index, sceneFocusState);
                ctx.globalAlpha = nodeFocusAlpha * (sceneRowAlpha ?? resolveRowDimmingAlpha(index, hoveredRowIndex, {
                    previousHoveredRowIndex,
                    hoverRowBlend,
                    dimStrength: hoverDimStrength,
                    dimmedRowOpacity
                }));
                ctx.fill();
                ctx.globalAlpha = nodeFocusAlpha;
            });
        }
    } else if (node.presentation === VIEW2D_MATRIX_PRESENTATIONS.GRID) {
        if (!preserveGridDetail && (useSummaryInterior || fastPath)) {
            const gridBounds = {
                x: contentBounds.x + layoutData.innerPaddingX,
                y: contentBounds.y + layoutData.innerPaddingY,
                width: Math.max(1, contentBounds.width - (layoutData.innerPaddingX * 2)),
                height: Math.max(1, contentBounds.height - (layoutData.innerPaddingY * 2))
            };
            roundRectPath(ctx, gridBounds.x, gridBounds.y, gridBounds.width, gridBounds.height, 6);
            ctx.fillStyle = accent;
            ctx.fill();
        } else {
            const cellCornerRadius = resolveGridCellCornerRadius(
                layoutData.cellSize,
                safeWorldScale,
                gridMeta.cellCornerRadiusScale
            );
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
                    const cellAlpha = resolveSceneGridCellAlpha(node.id, rowIndex, colIndex, sceneFocusState);
                    ctx.globalAlpha = nodeFocusAlpha * (Number.isFinite(cellAlpha) ? cellAlpha : 1);
                    roundRectPath(
                        ctx,
                        cellBounds.x,
                        cellBounds.y,
                        cellBounds.width,
                        cellBounds.height,
                        cellCornerRadius
                    );
                    ctx.fill();
                    if (Number.isFinite(cellAlpha) && cellAlpha >= 0.995) {
                        ctx.lineWidth = Math.max(0.85, GRID_FOCUSED_CELL_STROKE_SCREEN_PX / safeWorldScale);
                        ctx.strokeStyle = 'rgba(255, 255, 255, 0.88)';
                        roundRectPath(
                            ctx,
                            cellBounds.x + (0.5 / safeWorldScale),
                            cellBounds.y + (0.5 / safeWorldScale),
                            Math.max(0, cellBounds.width - (1 / safeWorldScale)),
                            Math.max(0, cellBounds.height - (1 / safeWorldScale)),
                            Math.max(0, cellCornerRadius - (0.5 / safeWorldScale))
                        );
                        ctx.stroke();
                    }
                    ctx.globalAlpha = nodeFocusAlpha;
                });
            });
        }
    } else if (node.presentation === VIEW2D_MATRIX_PRESENTATIONS.COLUMN_STRIP) {
        const sceneColumnSelections = sceneFocusState?.columnSelections?.get(node.id) || null;
        const sceneFocusedColumnIndex = sceneColumnSelections?.size
            ? Array.from(sceneColumnSelections)[0]
            : null;
        if (isVectorStripColumns) {
            drawVectorStripColumns(
                ctx,
                columnItems,
                contentBounds,
                layoutData,
                accent,
                cornerRadius,
                {
                    focusedColumnIndex: Number.isFinite(sceneFocusedColumnIndex) ? sceneFocusedColumnIndex : null,
                    dimmedColumnOpacity: sceneFocusState?.inactiveOpacity || vectorStripDimmedColumnOpacity,
                    baseAlpha: nodeFocusAlpha,
                    hoverScaleX: vectorStripHoverScaleX,
                    hoverGlowColor: fastPath
                        ? null
                        : (
                            typeof columnStripMeta.hoverGlowColor === 'string'
                                ? columnStripMeta.hoverGlowColor
                                : null
                        ),
                    hoverGlowBlur: fastPath
                        ? 0
                        : (
                            Number.isFinite(columnStripMeta.hoverGlowBlur)
                                ? Math.max(0, Number(columnStripMeta.hoverGlowBlur))
                                : 0
                        ),
                    hoverStrokeColor: fastPath
                        ? null
                        : (
                            typeof columnStripMeta.hoverStrokeColor === 'string'
                                ? columnStripMeta.hoverStrokeColor
                                : null
                        )
                }
            );
        } else if (useSummaryInterior || fastPath) {
            const stripBounds = {
                x: contentBounds.x + layoutData.innerPaddingX,
                y: contentBounds.y + layoutData.innerPaddingY,
                width: Math.max(1, contentBounds.width - (layoutData.innerPaddingX * 2)),
                height: Math.max(1, Math.min(layoutData.colHeight, contentBounds.height - (layoutData.innerPaddingY * 2)))
            };
            roundRectPath(ctx, stripBounds.x, stripBounds.y, stripBounds.width, stripBounds.height, 5);
            ctx.fillStyle = accent;
            ctx.fill();
        } else {
            columnItems.forEach((columnItem, index) => {
                const colBounds = {
                    x: contentBounds.x + layoutData.innerPaddingX + index * (layoutData.colWidth + layoutData.colGap),
                    y: contentBounds.y + layoutData.innerPaddingY,
                    width: layoutData.colWidth,
                    height: layoutData.colHeight
                };
                roundRectPath(ctx, colBounds.x, colBounds.y, colBounds.width, colBounds.height, 5);
                ctx.fillStyle = resolveFill(ctx, columnItem.fillCss, colBounds, accent);
                const columnAlpha = resolveSceneColumnSelectionAlpha(node.id, index, sceneFocusState);
                ctx.globalAlpha = nodeFocusAlpha * (Number.isFinite(columnAlpha) ? columnAlpha : 1);
                ctx.fill();
                ctx.globalAlpha = nodeFocusAlpha;
            });
        }
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
    }
    if (!hideSurface) {
        ctx.globalAlpha = surfaceFocusAlpha;
        drawCardEdgeStrokes(ctx, contentBounds, cornerRadius, style, safeWorldScale, surfaceFocusAlpha);
    }
    ctx.restore();
    drawMatrixConnectorOrnaments(ctx, node, contentBounds, config, safeWorldScale, nodeFocusAlpha);
    drawCaption(ctx, entry, node, config, safeWorldScale, safeDetailScale, focusAlpha, fixedTextSizing);
}

function drawMatrixEdgeArrow(
    ctx,
    bounds,
    config,
    worldScale = 1,
    focusAlpha = 1,
    {
        direction = 'right'
    } = {}
) {
    if (!bounds) return;
    const safeWorldScale = Math.max(0.0001, Number.isFinite(worldScale) ? worldScale : 1);
    const gap = MATRIX_ORNAMENT_ARROW_GAP_SCREEN_PX / safeWorldScale;
    const isLeft = String(direction || '').trim().toLowerCase() === 'left';
    const shaftLength = (
        isLeft
            ? MATRIX_ORNAMENT_ARROW_SHAFT_SCREEN_PX_INCOMING
            : MATRIX_ORNAMENT_ARROW_SHAFT_SCREEN_PX
    ) / safeWorldScale;
    const anchorX = isLeft
        ? bounds.x - gap
        : bounds.x + bounds.width + gap;
    const startPoint = isLeft
        ? {
            x: anchorX - shaftLength,
            y: bounds.y + (bounds.height * 0.5)
        }
        : {
            x: anchorX,
            y: bounds.y + (bounds.height * 0.5)
        };
    const endPoint = isLeft
        ? {
            x: anchorX,
            y: startPoint.y
        }
        : {
            x: startPoint.x + shaftLength,
            y: startPoint.y
        };
    drawConnector(
        ctx,
        {
            pathPoints: [startPoint, endPoint],
            metadata: {
                strokeWidthScale: 1,
                preserveColor: true,
                fixedScreenStrokeWidthPx: MATRIX_ORNAMENT_ARROW_STROKE_SCREEN_PX,
                fixedScreenArrowHeadLengthPx: MATRIX_ORNAMENT_ARROW_HEAD_LENGTH_SCREEN_PX,
                fixedScreenArrowHeadWingPx: MATRIX_ORNAMENT_ARROW_HEAD_WING_SCREEN_PX,
                disableScreenSnap: true
            }
        },
        config,
        resolveView2dStyle(VIEW2D_STYLE_KEYS.CONNECTOR_NEUTRAL)?.stroke || config?.tokens?.palette?.neutral || null,
        safeWorldScale,
        {
            focusAlpha,
            emphasize: false
        }
    );
}

function drawMatrixConnectorOrnaments(
    ctx,
    node,
    bounds,
    config,
    worldScale = 1,
    focusAlpha = 1
) {
    if (!node || !bounds) return;
    if (node.metadata?.disableEdgeOrnament === true) {
        return;
    }
    if (node.role === 'attention-head-output') {
        drawMatrixEdgeArrow(ctx, bounds, config, worldScale, focusAlpha, {
            direction: 'right'
        });
        return;
    }
    if (node.role === 'projection-source-xln') {
        drawMatrixEdgeArrow(ctx, bounds, config, worldScale, focusAlpha, {
            direction: 'left'
        });
    }
}

function resolveTextLikeClipBounds(bounds, node, renderedFontSize, config, fontWeight = 500) {
    if (!bounds) return null;
    const baseClipBounds = {
        x: bounds.x - 1,
        y: bounds.y,
        width: bounds.width + 2,
        height: bounds.height
    };
    if (node?.kind !== VIEW2D_NODE_KINDS.OPERATOR) {
        return baseClipBounds;
    }

    const measurement = measureView2dText(node.text || node.tex || ' ', {
        fontSize: renderedFontSize,
        fontWeight
    });
    const operatorSidePadding = Number.isFinite(config?.component?.operatorSidePadding)
        ? Math.max(0, Number(config.component.operatorSidePadding))
        : 0;
    const extraSidePadding = Math.max(2, renderedFontSize * 0.06);
    const extraVerticalPadding = Math.max(2, renderedFontSize * 0.08);
    const requiredWidth = Math.max(
        baseClipBounds.width,
        measurement.inkWidth + (operatorSidePadding * 2) + (extraSidePadding * 2)
    );
    const requiredHeight = Math.max(
        baseClipBounds.height,
        measurement.height + (extraVerticalPadding * 2)
    );
    const extraWidth = Math.max(0, requiredWidth - baseClipBounds.width);
    const extraHeight = Math.max(0, requiredHeight - baseClipBounds.height);
    const topExpansion = extraHeight * 0.45;

    return {
        x: baseClipBounds.x - (extraWidth / 2),
        y: baseClipBounds.y - topExpansion,
        width: baseClipBounds.width + extraWidth,
        height: baseClipBounds.height + extraHeight
    };
}

function drawTextLikeNode(ctx, node, entry, config, worldScale = 1, detailScale = worldScale, focusAlpha = 1, fixedTextSizing = null) {
    const bounds = entry.contentBounds || entry.bounds;
    const renderMode = String(node?.metadata?.renderMode || '').trim().toLowerCase();
    if (renderMode === 'dom-katex') {
        return;
    }
    const safeWorldScale = Math.max(0.0001, Number.isFinite(worldScale) ? worldScale : 1);
    const safeDetailScale = Math.max(0.0001, Number.isFinite(detailScale) ? detailScale : safeWorldScale);
    const visualOpacity = resolveNodeVisualOpacity(node);
    const isPersistentOperator = node.kind === VIEW2D_NODE_KINDS.OPERATOR
        && (
            node.role === 'residual-add-operator'
            || node.visual?.styleKey === 'residual.add-symbol'
        );
    const detailOperatorMinScreenHeightPx = node.kind === VIEW2D_NODE_KINDS.OPERATOR
        && Number.isFinite(fixedTextSizing?.operatorMinScreenHeightPx)
        ? Math.max(0, Number(fixedTextSizing.operatorMinScreenHeightPx))
        : null;
    const revealScreenFontPx = node.kind === VIEW2D_NODE_KINDS.OPERATOR
        ? (
            Number.isFinite(fixedTextSizing?.textScreenFontPx) && fixedTextSizing.textScreenFontPx > 0
                ? Number(fixedTextSizing.textScreenFontPx)
                : ((Number.isFinite(config?.component?.labelFontSize) ? Number(config.component.labelFontSize) : 12) * safeDetailScale)
        )
        : (
            Number.isFinite(fixedTextSizing?.textScreenFontPx) && fixedTextSizing.textScreenFontPx > 0
                ? Number(fixedTextSizing.textScreenFontPx)
                : ((Number.isFinite(config?.component?.labelFontSize) ? Number(config.component.labelFontSize) : 12) * safeDetailScale)
        );
    const persistentMinScreenFontPx = node.kind !== VIEW2D_NODE_KINDS.OPERATOR
        && Number.isFinite(node.metadata?.persistentMinScreenFontPx)
        && node.metadata.persistentMinScreenFontPx > 0
        ? Number(node.metadata.persistentMinScreenFontPx)
        : null;
    const effectiveRevealScreenFontPx = persistentMinScreenFontPx
        ? Math.max(revealScreenFontPx, persistentMinScreenFontPx)
        : revealScreenFontPx;
    const effectiveRevealThresholdPx = detailOperatorMinScreenHeightPx ?? TEXT_MIN_SCREEN_HEIGHT_PX;
    if (!isPersistentOperator && effectiveRevealScreenFontPx < effectiveRevealThresholdPx) {
        return;
    }
    ctx.save();
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    const text = node.text || node.tex || '';
    const tex = node.tex || '';
    let renderedFontSize = entry.layoutData?.fontSize || config.component.labelFontSize;
    let renderedFontWeight = 500;
    let renderBounds = bounds;
    if (node.kind === VIEW2D_NODE_KINDS.OPERATOR) {
        const baseFontSize = entry.layoutData?.fontSize || config.component.operatorFontSize;
        const isPlusOperator = node.semantic?.operatorKey === 'plus' || node.text === '+';
        const operatorZoomBehavior = fixedTextSizing?.operatorBehavior || VIEW2D_TEXT_ZOOM_BEHAVIORS.SCREEN_ADAPTIVE;
        const useSceneRelativeOperatorSizing = operatorZoomBehavior === VIEW2D_TEXT_ZOOM_BEHAVIORS.SCENE_RELATIVE;
        const defaultPersistentMinScreenFontPx = isPersistentOperator
            ? (isPlusOperator
                ? PERSISTENT_PLUS_OPERATOR_MIN_SCREEN_FONT_PX
                : PERSISTENT_OPERATOR_MIN_SCREEN_FONT_PX)
            : 0;
        const detailOperatorMinScreenFontPx = Number.isFinite(fixedTextSizing?.operatorMinScreenFontPx)
            && fixedTextSizing.operatorMinScreenFontPx > 0
            ? Number(fixedTextSizing.operatorMinScreenFontPx)
            : 0;
        const persistentMinScreenFontPx = Math.max(
            defaultPersistentMinScreenFontPx,
            detailOperatorMinScreenFontPx
        );
        const useScreenFixedOperatorSizing = operatorZoomBehavior === VIEW2D_TEXT_ZOOM_BEHAVIORS.SCREEN_FIXED;
        const fixedOperatorFontSize = useScreenFixedOperatorSizing
            ? null
            : (!useSceneRelativeOperatorSizing && fixedTextSizing?.textScreenFontPx
                ? (fixedTextSizing.textScreenFontPx / safeWorldScale)
                : null);
        const adjustedFontSize = useSceneRelativeOperatorSizing
            ? baseFontSize
            : (
                fixedOperatorFontSize
                ?? (
                    persistentMinScreenFontPx > 0
                        ? Math.max(baseFontSize, persistentMinScreenFontPx / safeWorldScale)
                        : baseFontSize
                )
            );
        const screenFixedFontSize = persistentMinScreenFontPx > 0
            ? Math.max(baseFontSize, persistentMinScreenFontPx)
            : baseFontSize;
        renderedFontSize = isPlusOperator
            ? (useScreenFixedOperatorSizing ? screenFixedFontSize : adjustedFontSize) * PLUS_OPERATOR_FONT_SCALE
            : (useScreenFixedOperatorSizing ? screenFixedFontSize : adjustedFontSize);
        renderedFontWeight = isPlusOperator ? PLUS_OPERATOR_FONT_WEIGHT : 600;
        if (useScreenFixedOperatorSizing) {
            renderBounds = {
                x: bounds.x * safeWorldScale,
                y: bounds.y * safeWorldScale,
                width: bounds.width * safeWorldScale,
                height: bounds.height * safeWorldScale
            };
            ctx.scale(1 / safeWorldScale, 1 / safeWorldScale);
        }
        ctx.font = `${renderedFontWeight} ${renderedFontSize}px ui-monospace, SFMono-Regular, Menlo, monospace`;
        ctx.fillStyle = resolveView2dStyle(node.visual?.styleKey)?.color || config.tokens.palette.text;
    } else {
        const maxWidth = Number.isFinite(entry.layoutData?.maxWidth) && entry.layoutData.maxWidth > 0
            ? entry.layoutData.maxWidth
            : (Number.isFinite(node.metadata?.textFit?.maxWidth) && node.metadata.textFit.maxWidth > 0
                ? node.metadata.textFit.maxWidth
                : null);
        const horizontalPadding = Number.isFinite(entry.layoutData?.paddingX) && entry.layoutData.paddingX >= 0
            ? entry.layoutData.paddingX
            : 0;
        if (maxWidth && text.length) {
            const availableTextWidth = Math.max(1, maxWidth - (horizontalPadding * 2));
            const measuredWidth = measureView2dText(text, { fontSize: renderedFontSize }).inkWidth;
            if (measuredWidth > availableTextWidth) {
                renderedFontSize = Math.max(1, renderedFontSize * (availableTextWidth / Math.max(1, measuredWidth)));
            }
        }
        if (fixedTextSizing?.textScreenFontPx) {
            renderedFontSize = fixedTextSizing.textScreenFontPx / safeWorldScale;
        } else if (persistentMinScreenFontPx) {
            renderedFontSize = Math.max(renderedFontSize, persistentMinScreenFontPx / safeWorldScale);
        }
        ctx.font = resolveView2dTextFont({
            fontSize: renderedFontSize,
            fontWeight: renderedFontWeight
        });
        ctx.fillStyle = resolveView2dStyle(node.visual?.styleKey)?.color || config.tokens.palette.text;
    }
    const clipBounds = resolveTextLikeClipBounds(renderBounds, node, renderedFontSize, config, renderedFontWeight);
    ctx.beginPath();
    ctx.rect(clipBounds.x, clipBounds.y, clipBounds.width, clipBounds.height);
    ctx.clip();
    ctx.globalAlpha = Math.max(0, Math.min(1, Number.isFinite(focusAlpha) ? focusAlpha : 1))
        * visualOpacity;
    if (tex && hasSimpleTexMarkup(tex)) {
        drawSimpleTex(ctx, tex, {
            x: renderBounds.x + (renderBounds.width / 2),
            y: renderBounds.y + (renderBounds.height / 2),
            fontSize: renderedFontSize,
            fontWeight: renderedFontWeight,
            color: resolveView2dStyle(node.visual?.styleKey)?.color || config.tokens.palette.text
        });
    } else {
        ctx.fillText(
            text,
            renderBounds.x + (renderBounds.width / 2),
            renderBounds.y + (renderBounds.height / 2)
        );
    }
    ctx.restore();
    drawCaption(ctx, entry, node, config, safeWorldScale, safeDetailScale, focusAlpha, fixedTextSizing);
}

function drawConnectorArrowHead(ctx, startPoint, endPoint, strokeWidth, fillStyle, {
    worldScale = 1,
    fixedScreenLengthPx = null,
    fixedScreenWingPx = null
} = {}) {
    if (!startPoint || !endPoint) return;
    const dx = endPoint.x - startPoint.x;
    const dy = endPoint.y - startPoint.y;
    const length = Math.hypot(dx, dy);
    if (!(length > 0.0001)) return;

    const safeWorldScale = Math.max(0.0001, Number.isFinite(worldScale) ? worldScale : 1);
    const ux = dx / length;
    const uy = dy / length;
    const size = Number.isFinite(fixedScreenLengthPx) && fixedScreenLengthPx > 0
        ? Number(fixedScreenLengthPx) / safeWorldScale
        : Math.max(strokeWidth * 3.6, 6);
    const wing = Number.isFinite(fixedScreenWingPx) && fixedScreenWingPx > 0
        ? Number(fixedScreenWingPx) / safeWorldScale
        : Math.max(strokeWidth * 1.7, 3.2);
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

function clamp01(value = 0) {
    return Math.max(0, Math.min(1, Number.isFinite(value) ? value : 0));
}

function lerpValue(start = 0, end = 0, alpha = 0) {
    return start + ((end - start) * clamp01(alpha));
}

function smoothstep(edge0 = 0, edge1 = 1, value = 0) {
    if (edge0 === edge1) {
        return value >= edge1 ? 1 : 0;
    }
    const t = clamp01((value - edge0) / (edge1 - edge0));
    return t * t * (3 - (2 * t));
}

function resolveConnectorScreenStrokeWidthPx(worldScale = 1) {
    const safeWorldScale = Math.max(0.0001, Number.isFinite(worldScale) ? worldScale : 1);
    const smallRamp = smoothstep(0.18, 0.35, safeWorldScale);
    const mediumRamp = smoothstep(0.35, 0.7, safeWorldScale);
    const largeRamp = smoothstep(0.7, 1.25, safeWorldScale);
    return 0.58
        + (lerpValue(0, 0.72 - 0.58, smallRamp))
        + (lerpValue(0, 0.88 - 0.72, mediumRamp))
        + (lerpValue(0, 1.02 - 0.88, largeRamp));
}

function resolveConnectorStrokeOpacity(worldScale = 1) {
    const safeWorldScale = Math.max(0.0001, Number.isFinite(worldScale) ? worldScale : 1);
    const smallRamp = smoothstep(0.18, 0.35, safeWorldScale);
    const mediumRamp = smoothstep(0.35, 0.7, safeWorldScale);
    const largeRamp = smoothstep(0.7, 1.25, safeWorldScale);
    return 0.46
        + (lerpValue(0, 0.54 - 0.46, smallRamp))
        + (lerpValue(0, 0.66 - 0.54, mediumRamp))
        + (lerpValue(0, 0.78 - 0.66, largeRamp));
}

function snapWorldPointToConnectorGrid(point, worldScale = 1) {
    if (!point || typeof point !== 'object') return point;
    const safeWorldScale = Math.max(0.0001, Number.isFinite(worldScale) ? worldScale : 1);
    const snap = (value) => {
        const screenValue = Number(value) * safeWorldScale;
        return (Math.round(screenValue * 2) / 2) / safeWorldScale;
    };
    return {
        x: snap(point.x),
        y: snap(point.y)
    };
}

function drawConnector(
    ctx,
    connectorEntry,
    config,
    accent = null,
    worldScale = 1,
    {
        skipArrowHead = false,
        focusAlpha = 1,
        emphasize = false,
        disableScreenSnap = false
    } = {}
) {
    const points = Array.isArray(connectorEntry.pathPoints) ? connectorEntry.pathPoints : [];
    if (points.length < 2) return;
    const foregroundStroke = accent || config.tokens.palette.neutral;
    const safeWorldScale = Math.max(0.0001, Number.isFinite(worldScale) ? worldScale : 1);
    const strokeWidthScale = Number.isFinite(connectorEntry?.metadata?.strokeWidthScale)
        ? Math.max(0.2, connectorEntry.metadata.strokeWidthScale)
        : 1;
    const fixedScreenStrokeWidthPx = Number.isFinite(connectorEntry?.metadata?.fixedScreenStrokeWidthPx)
        && connectorEntry.metadata.fixedScreenStrokeWidthPx > 0
        ? Number(connectorEntry.metadata.fixedScreenStrokeWidthPx)
        : null;
    const targetScreenWidthPx = fixedScreenStrokeWidthPx ?? resolveConnectorScreenStrokeWidthPx(safeWorldScale);
    const strokeWidth = (Math.max(0.42, targetScreenWidthPx) * strokeWidthScale * (emphasize ? 1.18 : 1)) / safeWorldScale;
    const fixedScreenArrowHeadLengthPx = Number.isFinite(connectorEntry?.metadata?.fixedScreenArrowHeadLengthPx)
        && connectorEntry.metadata.fixedScreenArrowHeadLengthPx > 0
        ? Number(connectorEntry.metadata.fixedScreenArrowHeadLengthPx)
        : null;
    const fixedScreenArrowHeadWingPx = Number.isFinite(connectorEntry?.metadata?.fixedScreenArrowHeadWingPx)
        && connectorEntry.metadata.fixedScreenArrowHeadWingPx > 0
        ? Number(connectorEntry.metadata.fixedScreenArrowHeadWingPx)
        : null;
    // Keep pinch-zoom frames and settled frames on the same vector coordinates.
    const shouldSnapToScreen = connectorEntry?.metadata?.enableScreenSnap === true
        && !disableScreenSnap
        && connectorEntry?.metadata?.disableScreenSnap !== true;
    const strokeOpacity = resolveConnectorStrokeOpacity(safeWorldScale);
    const flattenedForegroundStroke = connectorEntry?.metadata?.preserveColor === true
        ? foregroundStroke
        : (flattenColorAgainstBlack(foregroundStroke, strokeOpacity) || foregroundStroke);
    const renderPoints = shouldSnapToScreen
        ? points.map((point) => snapWorldPointToConnectorGrid(point, safeWorldScale))
        : points;
    const tailPoint = renderPoints[Math.max(0, renderPoints.length - 2)];
    const headPoint = renderPoints[renderPoints.length - 1];
    const rawTailPoint = points[Math.max(0, points.length - 2)];
    const rawHeadPoint = points[points.length - 1];

    ctx.save();
    ctx.lineCap = 'butt';
    ctx.lineJoin = 'miter';
    ctx.miterLimit = 2;
    ctx.globalCompositeOperation = 'source-over';
    ctx.filter = 'none';
    ctx.globalAlpha = Math.max(0, Math.min(1, Number.isFinite(focusAlpha) ? focusAlpha : 1));
    ctx.beginPath();
    ctx.moveTo(renderPoints[0].x, renderPoints[0].y);
    for (let index = 1; index < renderPoints.length; index += 1) {
        ctx.lineTo(renderPoints[index].x, renderPoints[index].y);
    }
    ctx.lineWidth = strokeWidth;
    ctx.shadowBlur = 0;
    ctx.strokeStyle = flattenedForegroundStroke;
    ctx.stroke();
    if (!skipArrowHead) {
        let arrowHeadEndPoint = rawHeadPoint || headPoint;
        if (connectorEntry?.metadata?.arrowTipTouchTarget === true && rawTailPoint && rawHeadPoint) {
            const dx = rawHeadPoint.x - rawTailPoint.x;
            const dy = rawHeadPoint.y - rawTailPoint.y;
            const length = Math.hypot(dx, dy);
            if (length > 0.0001) {
                const overlap = Math.max(0.9 / safeWorldScale, strokeWidth * 0.82);
                arrowHeadEndPoint = {
                    x: rawHeadPoint.x + ((dx / length) * overlap),
                    y: rawHeadPoint.y + ((dy / length) * overlap)
                };
            }
        }
        drawConnectorArrowHead(
            ctx,
            tailPoint,
            arrowHeadEndPoint,
            strokeWidth,
            flattenedForegroundStroke,
            {
                worldScale: safeWorldScale,
                fixedScreenLengthPx: fixedScreenArrowHeadLengthPx,
                fixedScreenWingPx: fixedScreenArrowHeadWingPx
            }
        );
    }
    ctx.restore();
}

function resolveConnectorFocusAlpha(node = null, sceneFocusState = null) {
    if (!sceneFocusState) return 1;
    if (node?.metadata?.preserveFocusOpacity === true) {
        return 1;
    }
    return resolveSceneElementFocusAlpha(sceneFocusState.activeConnectorIds.has(node?.id), sceneFocusState);
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

function resolveInteractionDetailScale(worldScale = 1, latchedDetailScale = null, interacting = false) {
    const safeWorldScale = Math.max(0.0001, Number.isFinite(worldScale) ? worldScale : 1);
    if (!interacting) return safeWorldScale;
    const safeLatched = Math.max(0.0001, Number.isFinite(latchedDetailScale) ? latchedDetailScale : 0);
    if (!(safeLatched > 0)) return safeWorldScale;
    if (safeWorldScale >= safeLatched) {
        return safeLatched;
    }
    if (safeWorldScale <= (safeLatched * INTERACTION_DETAIL_RELEASE_RATIO)) {
        return safeWorldScale;
    }
    return safeLatched;
}

function collectVisibleEntries(entries = [], visibleWorldBounds = null, target = []) {
    target.length = 0;
    for (let index = 0; index < entries.length; index += 1) {
        const entry = entries[index];
        if (!entry?.entry || !intersectsBounds(entry.entry.bounds, visibleWorldBounds)) continue;
        target.push(entry);
    }
    return target;
}

function resolveSceneBackgroundFill(config = null) {
    const sceneStyleFill = resolveView2dStyle(VIEW2D_STYLE_KEYS.SCENE)?.fill || '';
    const paletteFill = typeof config?.tokens?.palette?.sceneBackground === 'string'
        ? config.tokens.palette.sceneBackground
        : '';
    const preferredFill = paletteFill.trim().length ? paletteFill.trim() : sceneStyleFill;
    if (!preferredFill || preferredFill === 'transparent' || preferredFill === 'rgba(0, 0, 0, 0)') {
        return 'rgb(0, 0, 0)';
    }
    return preferredFill;
}

function buildOverviewVisibleIndex(visibleDrawableNodes = [], visibleConnectors = []) {
    const nodesById = new Map();
    const connectorsById = new Map();
    const connectorsByNodeId = new Map();
    for (let index = 0; index < visibleDrawableNodes.length; index += 1) {
        const drawable = visibleDrawableNodes[index];
        const nodeId = typeof drawable?.node?.id === 'string' ? drawable.node.id : '';
        if (!nodeId) continue;
        nodesById.set(nodeId, drawable);
    }
    const registerConnectorForNode = (nodeId = '', connector = null) => {
        if (!nodeId || !connector) return;
        const existing = connectorsByNodeId.get(nodeId);
        if (existing) {
            existing.push(connector);
            return;
        }
        connectorsByNodeId.set(nodeId, [connector]);
    };
    for (let index = 0; index < visibleConnectors.length; index += 1) {
        const connector = visibleConnectors[index];
        const connectorId = typeof connector?.node?.id === 'string' ? connector.node.id : '';
        if (connectorId) {
            connectorsById.set(connectorId, connector);
        }
        const sourceNodeId = typeof connector?.entry?.source?.nodeId === 'string' ? connector.entry.source.nodeId : '';
        const targetNodeId = typeof connector?.entry?.target?.nodeId === 'string' ? connector.entry.target.nodeId : '';
        registerConnectorForNode(sourceNodeId, connector);
        if (targetNodeId && targetNodeId !== sourceNodeId) {
            registerConnectorForNode(targetNodeId, connector);
        }
    }
    return {
        nodesById,
        connectorsById,
        connectorsByNodeId
    };
}

function createPreparedSceneState(scene = null) {
    if (!scene?.nodes || !Array.isArray(scene.nodes)) return null;
    const layout = buildSceneLayout(scene, {
        isSmallScreen: !!scene?.metadata?.isSmallScreen,
        visualTokens: scene?.metadata?.tokens || null
    });
    const registry = layout?.registry || null;
    if (!registry) return null;
    const allNodes = flattenSceneNodes(scene);
    const drawableNodes = allNodes
        .filter((node) => ![VIEW2D_NODE_KINDS.CONNECTOR, VIEW2D_NODE_KINDS.GROUP].includes(node.kind))
        .filter((node) => !node?.metadata?.hidden)
        .map((node) => ({
            node,
            entry: registry.getNodeEntry(node.id) || null
        }))
        .filter((item) => item.entry);
    const connectors = allNodes
        .filter((node) => node.kind === VIEW2D_NODE_KINDS.CONNECTOR)
        .map((node) => ({
            node,
            entry: registry.getConnectorEntry(node.id) || null,
            stroke: (
                (typeof node.visual?.stroke === 'string' && node.visual.stroke.length)
                    ? node.visual.stroke
                    : (resolveView2dStyle(node.visual?.styleKey)?.stroke || null)
            )
        }))
        .filter((item) => item.entry);
    const visibleBounds = unionRenderableBounds([
        ...drawableNodes.map((item) => item.entry?.bounds || null),
        ...connectors.map((item) => item.entry?.bounds || null)
    ]) || cloneBounds(layout.sceneBounds || registry.getSceneBounds() || null);
    return {
        scene,
        layout,
        drawableNodes,
        drawableNodesById: new Map(drawableNodes.map((item) => [item.node.id, item])),
        connectors,
        visibleBounds
    };
}

function resolvePreparedSceneHitAtPoint(preparedSceneState = null, x = 0, y = 0, detailScale = 1) {
    const registry = preparedSceneState?.layout?.registry || null;
    if (!registry || !Number.isFinite(x) || !Number.isFinite(y)) return null;
    const entry = typeof registry.resolveRawNodeEntryAtPoint === 'function'
        ? registry.resolveRawNodeEntryAtPoint(x, y, {
            includeGroups: false
        })
        : registry.resolveNodeEntryAtPoint(x, y, {
            includeGroups: false
        });
    if (!entry) return null;
    const drawable = preparedSceneState.drawableNodesById.get(entry.nodeId) || null;
    const node = drawable?.node || null;
    return {
        entry,
        node,
        rowHit: resolveMatrixRowHit(node, entry, x, y, detailScale),
        cellHit: resolveMatrixCellHit(node, entry, x, y),
        columnHit: resolveMatrixColumnHit(node, entry, x, y)
    };
}

function resolveSceneHitAtPoint(registry = null, drawableNodesById = null, x = 0, y = 0, detailScale = 1) {
    if (!registry || !Number.isFinite(x) || !Number.isFinite(y)) return null;
    const entry = typeof registry.resolveRawNodeEntryAtPoint === 'function'
        ? registry.resolveRawNodeEntryAtPoint(x, y, {
            includeGroups: false
        })
        : registry.resolveNodeEntryAtPoint(x, y, {
            includeGroups: false
        });
    if (!entry) return null;
    const drawable = drawableNodesById?.get(entry.nodeId) || null;
    const node = drawable?.node || null;
    return {
        entry,
        node,
        rowHit: resolveMatrixRowHit(node, entry, x, y, detailScale),
        cellHit: resolveMatrixCellHit(node, entry, x, y),
        columnHit: resolveMatrixColumnHit(node, entry, x, y)
    };
}

function projectWorldBoundsToScreen(bounds = null, renderState = null) {
    if (!bounds || typeof bounds !== 'object') return null;
    const worldScale = Number(renderState?.worldScale);
    if (!(worldScale > 0)) return null;
    return {
        x: ((Number(bounds.x) || 0) * worldScale) + (Number(renderState?.offsetX) || 0),
        y: ((Number(bounds.y) || 0) * worldScale) + (Number(renderState?.offsetY) || 0),
        width: Math.max(0, Number(bounds.width) || 0) * worldScale,
        height: Math.max(0, Number(bounds.height) || 0) * worldScale
    };
}

function createCanvasRenderSurface(pixelWidth = 0, pixelHeight = 0, referenceCanvas = null) {
    const safePixelWidth = Math.max(1, Math.floor(Number(pixelWidth) || 0));
    const safePixelHeight = Math.max(1, Math.floor(Number(pixelHeight) || 0));
    if (typeof OffscreenCanvas !== 'undefined') {
        return new OffscreenCanvas(safePixelWidth, safePixelHeight);
    }
    const documentRef = referenceCanvas?.ownerDocument
        || (typeof document !== 'undefined' ? document : null);
    if (documentRef?.createElement) {
        const surface = documentRef.createElement('canvas');
        surface.width = safePixelWidth;
        surface.height = safePixelHeight;
        return surface;
    }
    return null;
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
        this.drawableNodesById = new Map();
        this.connectors = [];
        this.lastRenderState = null;
        this.latchedDetailScale = null;
        this.visibleDrawableNodes = [];
        this.visibleConnectors = [];
        this.overviewVisibleIndex = null;
        this.overviewScreenHitCache = null;
        this.headDetailSceneState = null;
        this.activeDetailSceneRenderState = null;
        this.overviewRenderCache = {
            surface: null,
            ctx: null,
            scene: null,
            dpr: null,
            pixelWidth: 0,
            pixelHeight: 0,
            viewportPixelWidth: 0,
            viewportPixelHeight: 0,
            logicalWidth: 0,
            logicalHeight: 0,
            worldScale: null,
            viewportOffsetX: null,
            viewportOffsetY: null,
            renderOffsetX: null,
            renderOffsetY: null,
            offsetX: null,
            offsetY: null,
            overscanPx: 0
        };
    }

    setCanvas(canvas = null) {
        this.canvas = canvas;
        this.ctx = canvas?.getContext?.('2d') || null;
        this.invalidateOverviewRenderCache();
        return this.ctx;
    }

    setScene(scene, layout = null, options = {}) {
        this.scene = scene || null;
        this.layout = layout || (scene ? buildSceneLayout(scene, options) : null);
        this.metrics = this.layout?.config || null;
        this.latchedDetailScale = null;
        this.activeDetailSceneRenderState = null;
        this.overviewVisibleIndex = null;
        this.overviewScreenHitCache = null;
        this.invalidateOverviewRenderCache();
        const registry = this.layout?.registry || null;
        const allNodes = this.scene ? flattenSceneNodes(this.scene) : [];
        this.drawableNodes = allNodes
            .filter((node) => ![VIEW2D_NODE_KINDS.CONNECTOR, VIEW2D_NODE_KINDS.GROUP].includes(node.kind))
            .filter((node) => !node?.metadata?.hidden)
            .map((node) => ({
                node,
                entry: registry?.getNodeEntry(node.id) || null
            }))
            .filter((item) => item.entry);
        this.drawableNodesById = new Map(
            this.drawableNodes.map((item) => [item.node.id, item])
        );
        this.connectors = allNodes
            .filter((node) => node.kind === VIEW2D_NODE_KINDS.CONNECTOR)
            .map((node) => ({
                node,
                entry: registry?.getConnectorEntry(node.id) || null,
                stroke: resolveView2dStyle(node.visual?.styleKey)?.stroke || null
            }))
            .filter((item) => item.entry);
        this.headDetailSceneState = createPreparedSceneState(
            this.scene?.metadata?.mhsaHeadDetailScene
            || this.scene?.metadata?.headDetailScene
            || this.scene?.metadata?.outputProjectionDetailScene
            || this.scene?.metadata?.mlpDetailScene
            || this.scene?.metadata?.layerNormDetailScene
            || null
        );
        return this.layout;
    }

    invalidateOverviewRenderCache() {
        this.overviewRenderCache.scene = null;
        this.overviewRenderCache.viewportPixelWidth = 0;
        this.overviewRenderCache.viewportPixelHeight = 0;
        this.overviewRenderCache.logicalWidth = 0;
        this.overviewRenderCache.logicalHeight = 0;
        this.overviewRenderCache.worldScale = null;
        this.overviewRenderCache.viewportOffsetX = null;
        this.overviewRenderCache.viewportOffsetY = null;
        this.overviewRenderCache.renderOffsetX = null;
        this.overviewRenderCache.renderOffsetY = null;
        this.overviewRenderCache.offsetX = null;
        this.overviewRenderCache.offsetY = null;
        this.overviewRenderCache.overscanPx = 0;
    }

    ensureOverviewRenderCacheSurface(pixelWidth = 0, pixelHeight = 0) {
        const safePixelWidth = Math.max(1, Math.floor(Number(pixelWidth) || 0));
        const safePixelHeight = Math.max(1, Math.floor(Number(pixelHeight) || 0));
        if (
            this.overviewRenderCache.surface
            && this.overviewRenderCache.ctx
            && this.overviewRenderCache.pixelWidth === safePixelWidth
            && this.overviewRenderCache.pixelHeight === safePixelHeight
        ) {
            return this.overviewRenderCache;
        }

        const surface = createCanvasRenderSurface(safePixelWidth, safePixelHeight, this.canvas);
        const ctx = surface?.getContext?.('2d') || null;
        if (!surface || !ctx) {
            this.overviewRenderCache.surface = null;
            this.overviewRenderCache.ctx = null;
            this.overviewRenderCache.pixelWidth = 0;
            this.overviewRenderCache.pixelHeight = 0;
            return this.overviewRenderCache;
        }
        surface.width = safePixelWidth;
        surface.height = safePixelHeight;
        this.overviewRenderCache.surface = surface;
        this.overviewRenderCache.ctx = ctx;
        this.overviewRenderCache.pixelWidth = safePixelWidth;
        this.overviewRenderCache.pixelHeight = safePixelHeight;
        return this.overviewRenderCache;
    }

    updateOverviewRenderCache({
        resolution = null,
        config = null,
        sceneBounds = null,
        worldScale = 1,
        offsetX = 0,
        offsetY = 0,
        headDetailTarget = null,
        headDetailDepthActive = false
    } = {}) {
        const viewportWidth = Math.max(1, Number(resolution?.width) || 0);
        const viewportHeight = Math.max(1, Number(resolution?.height) || 0);
        const safeDpr = Number.isFinite(resolution?.dpr) && resolution.dpr > 0
            ? Number(resolution.dpr)
            : 1;
        const overscanPx = resolveOverviewRenderCacheOverscanPx(resolution);
        const logicalWidth = viewportWidth + (overscanPx * 2);
        const logicalHeight = viewportHeight + (overscanPx * 2);
        const pixelWidth = logicalWidth * safeDpr;
        const pixelHeight = logicalHeight * safeDpr;
        const cache = this.ensureOverviewRenderCacheSurface(pixelWidth, pixelHeight);
        if (
            !cache?.surface
            || !cache?.ctx
            || !sceneBounds
        ) {
            return false;
        }
        const renderOffsetX = (Number.isFinite(offsetX) ? Number(offsetX) : 0) + overscanPx;
        const renderOffsetY = (Number.isFinite(offsetY) ? Number(offsetY) : 0) + overscanPx;
        const cacheResolution = {
            width: logicalWidth,
            height: logicalHeight,
            dpr: safeDpr,
            pixelWidth: Math.max(1, Math.floor(pixelWidth)),
            pixelHeight: Math.max(1, Math.floor(pixelHeight))
        };
        const visibleWorldBounds = resolveVisibleWorldBounds(cacheResolution, {
            offsetX: renderOffsetX,
            offsetY: renderOffsetY,
            worldScale,
            sceneBounds
        });
        const visibleDrawableNodes = collectVisibleEntries(
            this.drawableNodes,
            visibleWorldBounds,
            []
        );
        const visibleConnectors = collectVisibleEntries(
            this.connectors,
            visibleWorldBounds,
            []
        );

        cache.ctx.setTransform(1, 0, 0, 1, 0, 0);
        cache.ctx.clearRect(0, 0, cache.pixelWidth, cache.pixelHeight);
        cache.ctx.setTransform(safeDpr, 0, 0, safeDpr, 0, 0);
        cache.ctx.globalAlpha = 1;
        cache.ctx.globalCompositeOperation = 'source-over';
        cache.ctx.filter = 'none';
        cache.ctx.shadowBlur = 0;
        cache.ctx.shadowColor = 'transparent';
        this.drawOverviewScene({
            ctx: cache.ctx,
            sceneBounds,
            config,
            worldScale,
            detailScale: Math.max(0.0001, Number(worldScale) || 1),
            offsetX: renderOffsetX,
            offsetY: renderOffsetY,
            visibleDrawableNodes,
            visibleConnectors,
            interactionFastPath: false,
            interactionState: null,
            sceneFocusState: null,
            fixedTextSizing: resolveMhsaDetailFixedTextSizing(this.scene, viewportWidth),
            headDetailTarget,
            headDetailDepthActive,
            disableInactiveFilter: false
        });
        cache.scene = this.scene;
        cache.dpr = Number(safeDpr.toFixed(4));
        cache.viewportPixelWidth = Math.max(1, Math.floor(Number(resolution?.pixelWidth) || 0));
        cache.viewportPixelHeight = Math.max(1, Math.floor(Number(resolution?.pixelHeight) || 0));
        cache.logicalWidth = Number(logicalWidth.toFixed(3));
        cache.logicalHeight = Number(logicalHeight.toFixed(3));
        cache.worldScale = Number.isFinite(worldScale) ? Number(worldScale.toFixed(6)) : null;
        cache.viewportOffsetX = Number.isFinite(offsetX) ? Number(offsetX.toFixed(3)) : null;
        cache.viewportOffsetY = Number.isFinite(offsetY) ? Number(offsetY.toFixed(3)) : null;
        cache.renderOffsetX = Number(renderOffsetX.toFixed(3));
        cache.renderOffsetY = Number(renderOffsetY.toFixed(3));
        cache.offsetX = Number.isFinite(offsetX) ? Number(offsetX.toFixed(3)) : null;
        cache.offsetY = Number.isFinite(offsetY) ? Number(offsetY.toFixed(3)) : null;
        cache.overscanPx = overscanPx;
        return true;
    }

    hasReusableOverviewRenderCache({
        resolution = null
    } = {}) {
        const cache = this.overviewRenderCache;
        return !!(
            cache?.surface
            && cache?.ctx
            && cache.scene === this.scene
            && cache.dpr === (Number.isFinite(resolution?.dpr) ? Number(Number(resolution.dpr).toFixed(4)) : null)
            && resolveOverviewRenderCacheViewportPixelWidth(cache) === Math.max(1, Math.floor(Number(resolution?.pixelWidth) || 0))
            && resolveOverviewRenderCacheViewportPixelHeight(cache) === Math.max(1, Math.floor(Number(resolution?.pixelHeight) || 0))
        );
    }

    matchesOverviewRenderCache({
        resolution = null,
        worldScale = 1,
        offsetX = 0,
        offsetY = 0
    } = {}) {
        const cache = this.overviewRenderCache;
        return !!(
            this.hasReusableOverviewRenderCache({
                resolution
            })
            && cache.worldScale === (Number.isFinite(worldScale) ? Number(worldScale.toFixed(6)) : null)
            && resolveOverviewRenderCacheViewportOffsetX(cache) === (Number.isFinite(offsetX) ? Number(offsetX.toFixed(3)) : null)
            && resolveOverviewRenderCacheViewportOffsetY(cache) === (Number.isFinite(offsetY) ? Number(offsetY.toFixed(3)) : null)
        );
    }

    canCoverOverviewViewportFromCache({
        resolution = null,
        worldScale = 1,
        offsetX = 0,
        offsetY = 0
    } = {}) {
        if (!this.hasReusableOverviewRenderCache({ resolution })) {
            return false;
        }
        const cache = this.overviewRenderCache;
        const cachedWorldScale = Number(cache?.worldScale);
        const safeWorldScale = Number(worldScale);
        if (!(cachedWorldScale > 0) || !(safeWorldScale > 0)) {
            return false;
        }

        const scaleRatio = safeWorldScale / cachedWorldScale;
        const renderOffsetX = resolveOverviewRenderCacheRenderOffsetX(cache);
        const renderOffsetY = resolveOverviewRenderCacheRenderOffsetY(cache);
        const logicalWidth = resolveOverviewRenderCacheLogicalWidth(cache);
        const logicalHeight = resolveOverviewRenderCacheLogicalHeight(cache);
        const viewportX = (Number.isFinite(offsetX) ? Number(offsetX) : 0) - (renderOffsetX * scaleRatio);
        const viewportY = (Number.isFinite(offsetY) ? Number(offsetY) : 0) - (renderOffsetY * scaleRatio);
        const viewportRight = viewportX + (logicalWidth * scaleRatio);
        const viewportBottom = viewportY + (logicalHeight * scaleRatio);
        const requiredWidth = Math.max(1, Number(resolution?.width) || 0);
        const requiredHeight = Math.max(1, Number(resolution?.height) || 0);
        return viewportX <= OVERVIEW_RENDER_CACHE_COVERAGE_EPSILON_PX
            && viewportY <= OVERVIEW_RENDER_CACHE_COVERAGE_EPSILON_PX
            && viewportRight >= (requiredWidth - OVERVIEW_RENDER_CACHE_COVERAGE_EPSILON_PX)
            && viewportBottom >= (requiredHeight - OVERVIEW_RENDER_CACHE_COVERAGE_EPSILON_PX);
    }

    drawTransformedOverviewRenderCache({
        resolution = null,
        worldScale = 1,
        offsetX = 0,
        offsetY = 0,
        background = null
    } = {}) {
        if (
            !this.hasReusableOverviewRenderCache({
                resolution
            })
            || !this.canCoverOverviewViewportFromCache({
                resolution,
                worldScale,
                offsetX,
                offsetY
            })
            || !this.ctx
            || typeof this.ctx.drawImage !== 'function'
        ) {
            return false;
        }
        const cache = this.overviewRenderCache;
        const cachedWorldScale = Number(cache?.worldScale);
        const safeWorldScale = Number(worldScale);
        if (!(cachedWorldScale > 0) || !(safeWorldScale > 0)) {
            return false;
        }

        const scaleRatio = safeWorldScale / cachedWorldScale;
        const safeDpr = Number.isFinite(resolution?.dpr) ? Number(resolution.dpr) : 1;
        const renderOffsetX = resolveOverviewRenderCacheRenderOffsetX(cache);
        const renderOffsetY = resolveOverviewRenderCacheRenderOffsetY(cache);
        const translateX = (
            (Number.isFinite(offsetX) ? Number(offsetX) : 0)
            - (renderOffsetX * scaleRatio)
        ) * safeDpr;
        const translateY = (
            (Number.isFinite(offsetY) ? Number(offsetY) : 0)
            - (renderOffsetY * scaleRatio)
        ) * safeDpr;

        this.ctx.save();
        this.ctx.setTransform(1, 0, 0, 1, 0, 0);
        if (typeof background === 'string' && background.length) {
            this.ctx.fillStyle = background;
            this.ctx.fillRect(
                0,
                0,
                Math.max(1, Math.floor(Number(resolution?.pixelWidth) || 0)),
                Math.max(1, Math.floor(Number(resolution?.pixelHeight) || 0))
            );
        }
        this.ctx.setTransform(scaleRatio, 0, 0, scaleRatio, translateX, translateY);
        this.ctx.drawImage(cache.surface, 0, 0);
        this.ctx.restore();
        return true;
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

    getActiveCaptionSceneState() {
        const preparedSceneState = this.activeDetailSceneRenderState?.preparedSceneState || null;
        if (preparedSceneState?.scene && preparedSceneState?.layout) {
            return {
                scene: preparedSceneState.scene,
                layout: preparedSceneState.layout
            };
        }
        return {
            scene: this.scene,
            layout: this.layout
        };
    }

    getHeadDetailSceneBounds() {
        const bounds = this.headDetailSceneState?.visibleBounds
            || this.headDetailSceneState?.layout?.sceneBounds
            || this.headDetailSceneState?.layout?.registry?.getSceneBounds?.()
            || null;
        return cloneBounds(bounds);
    }

    resolveScreenBounds(bounds = null) {
        const renderState = (
            this.activeDetailSceneRenderState?.worldScale > 0
                ? this.activeDetailSceneRenderState
                : this.lastRenderState
        );
        return projectWorldBoundsToScreen(bounds, renderState);
    }

    resolveInteractiveHitAtPoint(x = 0, y = 0) {
        if (!this.layout?.registry || !Number.isFinite(x) || !Number.isFinite(y)) return null;
        const detailScale = this.lastRenderState?.detailScale || this.lastRenderState?.worldScale || 1;
        return resolveSceneHitAtPoint(
            this.layout.registry,
            this.drawableNodesById,
            x,
            y,
            detailScale
        );
    }

    resolveInteractiveHitAtScreenPoint(x = 0, y = 0) {
        if (!Number.isFinite(x) || !Number.isFinite(y)) return null;
        if (
            this.activeDetailSceneRenderState?.preparedSceneState
            && Number.isFinite(this.activeDetailSceneRenderState?.worldScale)
            && this.activeDetailSceneRenderState.worldScale > 0
        ) {
            const sceneX = (x - this.activeDetailSceneRenderState.offsetX) / this.activeDetailSceneRenderState.worldScale;
            const sceneY = (y - this.activeDetailSceneRenderState.offsetY) / this.activeDetailSceneRenderState.worldScale;
            return resolvePreparedSceneHitAtPoint(
                this.activeDetailSceneRenderState.preparedSceneState,
                sceneX,
                sceneY,
                this.activeDetailSceneRenderState.detailScale
            );
        }
        const renderState = this.lastRenderState || null;
        const worldScale = Number(renderState?.worldScale);
        if (!(worldScale > 0)) return null;
        const sceneX = (x - (Number(renderState?.offsetX) || 0)) / worldScale;
        const sceneY = (y - (Number(renderState?.offsetY) || 0)) / worldScale;
        const directHit = this.resolveInteractiveHitAtPoint(sceneX, sceneY);
        if (directHit?.rowHit || directHit?.cellHit || directHit?.columnHit) {
            return directHit;
        }
        const fallbackRowHit = resolveApproximateOverviewResidualRowHitAtScreenPoint(
            this.visibleDrawableNodes?.length ? this.visibleDrawableNodes : this.drawableNodes,
            renderState,
            x,
            y
        );
        if (!fallbackRowHit) {
            return directHit;
        }
        if (directHit && fallbackRowHit?.node?.id === directHit?.node?.id) {
            return {
                ...directHit,
                rowHit: fallbackRowHit.rowHit || directHit.rowHit || null
            };
        }
        if (directHit) {
            return fallbackRowHit;
        }
        return fallbackRowHit;
    }

    drawOverviewScene({
        ctx = null,
        sceneBounds = null,
        config = null,
        worldScale = 1,
        detailScale = 1,
        offsetX = 0,
        offsetY = 0,
        visibleDrawableNodes = [],
        visibleConnectors = [],
        interactionFastPath = false,
        interactionState = null,
        sceneFocusState = null,
        fixedTextSizing = null,
        headDetailTarget = null,
        headDetailDepthActive = false,
        disableInactiveFilter = false
    } = {}) {
        if (!ctx || !sceneBounds) return false;

        ctx.save();
        ctx.translate(offsetX, offsetY);
        ctx.scale(worldScale, worldScale);
        ctx.fillStyle = resolveSceneBackgroundFill(config);
        ctx.fillRect(0, 0, sceneBounds.width, sceneBounds.height);

        visibleDrawableNodes.forEach(({ node, entry }) => {
            const focusAlpha = resolveSceneNodeFocusAlpha(node.id, sceneFocusState);
            if (node.kind === VIEW2D_NODE_KINDS.MATRIX) {
                drawMatrixNode(ctx, node, entry, config, worldScale, detailScale, {
                    skipSurfaceEffects: interactionFastPath,
                    fastPath: interactionFastPath,
                    headDetailTarget,
                    headDetailDepthActive,
                    interactionState,
                    focusAlpha,
                    fixedTextSizing,
                    disableInactiveFilter
                });
            } else if (node.kind === VIEW2D_NODE_KINDS.TEXT || node.kind === VIEW2D_NODE_KINDS.OPERATOR) {
                drawTextLikeNode(ctx, node, entry, config, worldScale, detailScale, focusAlpha, fixedTextSizing);
            }
        });

        visibleConnectors.forEach(({ node, entry, stroke }) => {
            const focusAlpha = resolveConnectorFocusAlpha(node, sceneFocusState);
            drawConnector(ctx, entry, config, stroke, worldScale, {
                focusAlpha,
                emphasize: focusAlpha >= 0.995,
                disableScreenSnap: interactionFastPath
            });
        });

        ctx.restore();
        return true;
    }

    drawFocusedOverviewNodes({
        ctx = null,
        visibleDrawableNodes = [],
        visibleConnectors = [],
        overviewVisibleIndex = null,
        config = null,
        worldScale = 1,
        detailScale = 1,
        fixedTextSizing = null,
        focusState = null,
        focusAlpha = 1,
        headDetailTarget = null,
        headDetailDepthActive = false
    } = {}) {
        if (!ctx || !focusState || focusAlpha <= 0.001) return;

        const focusedNodes = focusState.activeNodeIds?.size
            ? Array.from(focusState.activeNodeIds, (nodeId) => overviewVisibleIndex?.nodesById?.get(nodeId) || null)
                .filter(Boolean)
            : visibleDrawableNodes.filter(({ node }) => focusState.activeNodeIds?.has(node.id));
        focusedNodes.forEach(({ node, entry }) => {
            if (node.kind === VIEW2D_NODE_KINDS.MATRIX) {
                drawMatrixNode(ctx, node, entry, config, worldScale, detailScale, {
                    skipSurfaceEffects: true,
                    fastPath: true,
                    headDetailTarget,
                    headDetailDepthActive,
                    interactionState: null,
                    focusAlpha,
                    fixedTextSizing,
                    disableInactiveFilter: true
                });
            } else if (node.kind === VIEW2D_NODE_KINDS.TEXT || node.kind === VIEW2D_NODE_KINDS.OPERATOR) {
                drawTextLikeNode(ctx, node, entry, config, worldScale, detailScale, focusAlpha, fixedTextSizing);
            }
        });

        const focusedConnectors = focusState.activeConnectorIds?.size
            ? Array.from(
                focusState.activeConnectorIds,
                (connectorId) => overviewVisibleIndex?.connectorsById?.get(connectorId) || null
            ).filter(Boolean)
            : visibleConnectors.filter(({ node }) => focusState.activeConnectorIds?.has(node.id));
        focusedConnectors.forEach(({ node, entry, stroke }) => {
            drawConnector(ctx, entry, config, stroke, worldScale, {
                focusAlpha,
                emphasize: focusAlpha >= 0.995,
                disableScreenSnap: true
            });
        });
    }

    drawCachedOverviewRowHoverState({
        ctx = null,
        resolution = null,
        config = null,
        worldScale = 1,
        detailScale = 1,
        offsetX = 0,
        offsetY = 0,
        visibleDrawableNodes = [],
        visibleConnectors = [],
        overviewVisibleIndex = null,
        interactionState = null,
        interactionFastPath = false,
        fixedTextSizing = null,
        headDetailTarget = null,
        headDetailDepthActive = false
    } = {}) {
        if (!ctx) return false;

        const hoverNodeIds = new Set();
        const registerHoverNodeId = (hoverState = null) => {
            const nodeId = typeof hoverState?.nodeId === 'string' ? hoverState.nodeId : '';
            if (nodeId.length) {
                hoverNodeIds.add(nodeId);
            }
        };
        registerHoverNodeId(interactionState?.hoveredRow);
        registerHoverNodeId(interactionState?.previousHoveredRow);
        if (!hoverNodeIds.size) return false;

        const hoveredNodes = Array.from(hoverNodeIds, (nodeId) => overviewVisibleIndex?.nodesById?.get(nodeId) || null)
            .filter((drawable) => (
                !!drawable?.entry
                && drawable?.node?.kind === VIEW2D_NODE_KINDS.MATRIX
            ));
        if (!hoveredNodes.length && visibleDrawableNodes.length) {
            hoveredNodes.push(...visibleDrawableNodes.filter(({ node, entry }) => (
                !!entry
                && node?.kind === VIEW2D_NODE_KINDS.MATRIX
                && hoverNodeIds.has(node.id)
            )));
        }
        if (!hoveredNodes.length) return false;

        const didDrawBaseCache = this.drawTransformedOverviewRenderCache({
            resolution,
            worldScale,
            offsetX,
            offsetY,
            background: resolveSceneBackgroundFill(config)
        });
        if (!didDrawBaseCache) return false;

        const hoveredConnectors = [];
        const hoveredConnectorIds = new Set();
        const appendConnector = (connector = null) => {
            const connectorId = typeof connector?.node?.id === 'string' ? connector.node.id : '';
            if (!connector || !connectorId || hoveredConnectorIds.has(connectorId)) return;
            hoveredConnectorIds.add(connectorId);
            hoveredConnectors.push(connector);
        };
        hoverNodeIds.forEach((nodeId) => {
            const connected = overviewVisibleIndex?.connectorsByNodeId?.get(nodeId) || null;
            if (connected?.length) {
                connected.forEach((connector) => appendConnector(connector));
            }
        });
        if (!hoveredConnectors.length && visibleConnectors.length) {
            visibleConnectors.forEach((connector) => {
                const sourceNodeId = typeof connector?.entry?.source?.nodeId === 'string'
                    ? connector.entry.source.nodeId
                    : '';
                const targetNodeId = typeof connector?.entry?.target?.nodeId === 'string'
                    ? connector.entry.target.nodeId
                    : '';
                if (hoverNodeIds.has(sourceNodeId) || hoverNodeIds.has(targetNodeId)) {
                    appendConnector(connector);
                }
            });
        }

        ctx.save();
        ctx.translate(offsetX, offsetY);
        ctx.scale(worldScale, worldScale);

        hoveredNodes.forEach(({ node, entry }) => {
            drawMatrixNode(ctx, node, entry, config, worldScale, detailScale, {
                skipSurfaceEffects: interactionFastPath,
                fastPath: interactionFastPath,
                headDetailTarget,
                headDetailDepthActive,
                interactionState,
                focusAlpha: 1,
                fixedTextSizing,
                disableInactiveFilter: interactionState?.disableInactiveFilter === true
            });
        });

        hoveredConnectors.forEach(({ entry, stroke }) => {
            drawConnector(ctx, entry, config, stroke, worldScale, {
                focusAlpha: 1,
                emphasize: false,
                disableScreenSnap: interactionFastPath
            });
        });

        ctx.restore();
        return true;
    }

    render({
        width = null,
        height = null,
        dpr = null,
        dprCap = null,
        clear = true,
        debug = false,
        viewportTransform = null,
        detailViewportTransform = null,
        interacting = false,
        headDetailDepthActive = false,
        interactionState = null
    } = {}) {
        if (!this.canvas || !this.ctx || !this.scene || !this.layout?.registry) {
            this.lastRenderState = {
                ok: false,
                error: 'missing-canvas-scene-or-layout'
            };
            return false;
        }
        const resolution = this.resize({ width, height, dpr, dprCap });
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
        const sceneFocusState = normalizeSceneFocusState(interactionState?.detailSceneFocus, {
            inactiveOpacity: Number(config?.tokens?.dimming?.inactiveOpacity) || 0.18
        });
        const overviewFocusTransition = interactionState?.overviewFocusTransition || null;
        const overviewCurrentFocusState = normalizeSceneFocusState(
            overviewFocusTransition?.currentFocus,
            {
                inactiveOpacity: Number(config?.tokens?.dimming?.inactiveOpacity) || 0.18
            }
        );
        const overviewPreviousFocusState = normalizeSceneFocusState(
            overviewFocusTransition?.previousFocus,
            {
                inactiveOpacity: Number(config?.tokens?.dimming?.inactiveOpacity) || 0.18
            }
        );
        const disableInactiveFilter = interactionState?.disableInactiveFilter === true;
        const overviewFallbackFocusState = !headDetailDepthActive
            ? (overviewCurrentFocusState || overviewPreviousFocusState || null)
            : null;
        const activeOverviewSceneFocusState = sceneFocusState || overviewFallbackFocusState;
        const normalizedInteractionState = activeOverviewSceneFocusState
            ? {
                ...(interactionState || {}),
                sceneFocusState: activeOverviewSceneFocusState
            }
            : interactionState;
        const sceneBounds = this.layout.sceneBounds || this.layout.registry.getSceneBounds();
        const resolvedViewport = resolveRenderViewportTransform(
            sceneBounds,
            resolution,
            viewportTransform
        );
        const worldScale = resolvedViewport.worldScale;
        const offsetX = resolvedViewport.offsetX;
        const offsetY = resolvedViewport.offsetY;
        const safeWorldScale = Math.max(0.0001, Number.isFinite(worldScale) ? worldScale : 1);
        this.latchedDetailScale = resolveInteractionDetailScale(
            safeWorldScale,
            this.latchedDetailScale,
            interacting
        );
        const interactionFastPath = !!interacting;
        const detailScale = interactionFastPath
            ? safeWorldScale
            : this.latchedDetailScale;
        const visibleWorldBounds = resolveVisibleWorldBounds(resolution, {
            offsetX,
            offsetY,
            worldScale,
            sceneBounds
        });
        const visibleDrawableNodes = collectVisibleEntries(
            this.drawableNodes,
            visibleWorldBounds,
            this.visibleDrawableNodes
        );
        const visibleConnectors = collectVisibleEntries(
            this.connectors,
            visibleWorldBounds,
            this.visibleConnectors
        );
        const overviewVisibleIndex = buildOverviewVisibleIndex(
            visibleDrawableNodes,
            visibleConnectors
        );
        this.overviewVisibleIndex = overviewVisibleIndex;
        const activeHeadDetailTarget = normalizeHeadDetailTarget(this.scene?.metadata?.headDetailTarget);
        const activeConcatDetailTarget = normalizeConcatDetailTarget(this.scene?.metadata?.concatDetailTarget);
        const activeOutputProjectionDetailTarget = normalizeOutputProjectionDetailTarget(
            this.scene?.metadata?.outputProjectionDetailTarget
        );
        const activeMlpDetailTarget = normalizeMlpDetailTarget(
            this.scene?.metadata?.mlpDetailTarget
        );
        const activeLayerNormDetailTarget = normalizeLayerNormDetailTarget(
            this.scene?.metadata?.layerNormDetailTarget
        );
        const activeHeadDetailBounds = activeHeadDetailTarget
            ? (
                resolveHeadDetailFocusBounds(this.layout, activeHeadDetailTarget, 'head')
                || resolveHeadDetailFocusBounds(this.layout, activeHeadDetailTarget, 'head-card')
            )
            : null;
        const activeConcatDetailBounds = activeConcatDetailTarget
            ? (
                resolveConcatDetailFocusBounds(this.layout, activeConcatDetailTarget, 'concat-card')
                || resolveConcatDetailFocusBounds(this.layout, activeConcatDetailTarget, 'concat')
            )
            : null;
        const activeOutputProjectionDetailBounds = activeOutputProjectionDetailTarget
            ? (
                resolveOutputProjectionDetailFocusBounds(this.layout, activeOutputProjectionDetailTarget, 'projection-weight')
                || resolveOutputProjectionDetailFocusBounds(this.layout, activeOutputProjectionDetailTarget, 'module')
            )
            : null;
        const activeMlpDetailBounds = activeMlpDetailTarget
            ? (
                resolveMlpDetailFocusBounds(this.layout, activeMlpDetailTarget, 'module-card')
                || resolveMlpDetailFocusBounds(this.layout, activeMlpDetailTarget, 'module-title')
                || resolveMlpDetailFocusBounds(this.layout, activeMlpDetailTarget, 'module')
            )
            : null;
        const activeLayerNormDetailBounds = activeLayerNormDetailTarget
            ? (
                resolveLayerNormDetailFocusBounds(this.layout, activeLayerNormDetailTarget, 'module-card')
                || resolveLayerNormDetailFocusBounds(this.layout, activeLayerNormDetailTarget, 'module-title')
                || resolveLayerNormDetailFocusBounds(this.layout, activeLayerNormDetailTarget, 'module')
            )
            : null;
        const activeDetailTargetKind = activeOutputProjectionDetailTarget
            ? 'output-projection'
            : (
                activeConcatDetailTarget
                    ? 'concatenate'
                    : (
                        activeMlpDetailTarget
                            ? 'mlp'
                            : (activeLayerNormDetailTarget ? 'layer-norm' : (activeHeadDetailTarget ? 'head' : ''))
                    )
            );
        this.activeDetailSceneRenderState = null;
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
            detailScale,
            offsetX,
            offsetY,
            fittedWidth: sceneBounds.width * worldScale,
            fittedHeight: sceneBounds.height * worldScale,
            viewportSource: resolvedViewport.source || 'auto-fit',
            viewportTransform: resolvedViewport.viewportTransform || null,
            nodeCount: this.drawableNodes.length,
            connectorCount: this.connectors.length,
            visibleNodeCount: visibleDrawableNodes.length,
            visibleConnectorCount: visibleConnectors.length,
            overviewVisibleIndex,
            interactionFastPath,
            headDetailTarget: activeHeadDetailTarget ? { ...activeHeadDetailTarget } : null,
            headDetailBounds: cloneBounds(activeHeadDetailBounds),
            concatDetailTarget: activeConcatDetailTarget ? { ...activeConcatDetailTarget } : null,
            concatDetailBounds: cloneBounds(activeConcatDetailBounds),
            outputProjectionDetailTarget: activeOutputProjectionDetailTarget
                ? { ...activeOutputProjectionDetailTarget }
                : null,
            outputProjectionDetailBounds: cloneBounds(activeOutputProjectionDetailBounds),
            mlpDetailTarget: activeMlpDetailTarget ? { ...activeMlpDetailTarget } : null,
            mlpDetailBounds: cloneBounds(activeMlpDetailBounds),
            layerNormDetailTarget: activeLayerNormDetailTarget ? { ...activeLayerNormDetailTarget } : null,
            layerNormDetailBounds: cloneBounds(activeLayerNormDetailBounds),
            detailTargetKind: activeDetailTargetKind,
            headDetailDepthActive: !!headDetailDepthActive
        };
        renderState.overviewScreenHitCache = buildOverviewScreenHitCache(
            visibleDrawableNodes,
            renderState
        );
        this.overviewScreenHitCache = renderState.overviewScreenHitCache;
        this.lastRenderState = renderState;

        try {
            if (
                headDetailDepthActive
                && ['head', 'mlp', 'output-projection', 'layer-norm'].includes(activeDetailTargetKind)
                && this.headDetailSceneState?.layout?.registry
            ) {
                const detailSceneBounds = this.headDetailSceneState.visibleBounds
                    || this.headDetailSceneState.layout.sceneBounds
                    || this.headDetailSceneState.layout.registry.getSceneBounds();
                const detailViewport = resolveRenderViewportTransform(
                    detailSceneBounds,
                    resolution,
                    detailViewportTransform
                );
                const detailWorldScale = Math.max(0.0001, Number(detailViewport.worldScale) || 1);
                const detailOffsetX = Number(detailViewport.offsetX) || 0;
                const detailOffsetY = Number(detailViewport.offsetY) || 0;
                const detailVisibleWorldBounds = resolveVisibleWorldBounds(resolution, {
                    offsetX: detailOffsetX,
                    offsetY: detailOffsetY,
                    worldScale: detailWorldScale,
                    sceneBounds: detailSceneBounds
                });
                const detailVisibleDrawableNodes = collectVisibleEntries(
                    this.headDetailSceneState.drawableNodes,
                    detailVisibleWorldBounds,
                    []
                );
                const detailVisibleConnectors = collectVisibleEntries(
                    this.headDetailSceneState.connectors,
                    detailVisibleWorldBounds,
                    []
                );
                this.activeDetailSceneRenderState = {
                    preparedSceneState: this.headDetailSceneState,
                    worldScale: detailWorldScale,
                    offsetX: detailOffsetX,
                    offsetY: detailOffsetY,
                    detailScale: detailWorldScale
                };
                const detailFixedTextSizing = resolveMhsaDetailFixedTextSizing(
                    this.headDetailSceneState?.scene,
                    resolution.width
                );
                ctx.save();
                ctx.fillStyle = resolveSceneBackgroundFill(config);
                ctx.fillRect(0, 0, resolution.width, resolution.height);
                ctx.translate(detailOffsetX, detailOffsetY);
                ctx.scale(detailWorldScale, detailWorldScale);
                detailVisibleDrawableNodes.forEach(({ node, entry }) => {
                    const focusAlpha = resolveSceneNodeFocusAlpha(node.id, sceneFocusState);
                    if (node.kind === VIEW2D_NODE_KINDS.MATRIX) {
                        drawMatrixNode(ctx, node, entry, config, detailWorldScale, detailWorldScale, {
                            skipSurfaceEffects: interactionFastPath,
                            fastPath: interactionFastPath,
                            interactionState: normalizedInteractionState,
                            focusAlpha,
                            fixedTextSizing: detailFixedTextSizing,
                            // Canvas filters are disproportionately expensive in the dense
                            // deep-detail MHSA scene and make passive row hovers feel laggy.
                            disableInactiveFilter: true
                        });
                    } else if (node.kind === VIEW2D_NODE_KINDS.TEXT || node.kind === VIEW2D_NODE_KINDS.OPERATOR) {
                        drawTextLikeNode(
                            ctx,
                            node,
                            entry,
                            config,
                            detailWorldScale,
                            detailWorldScale,
                            focusAlpha,
                            detailFixedTextSizing
                        );
                    }
                });
                detailVisibleConnectors.forEach(({ node, entry, stroke }) => {
                    const focusAlpha = resolveConnectorFocusAlpha(node, sceneFocusState);
                    drawConnector(ctx, entry, config, stroke, detailWorldScale, {
                        focusAlpha,
                        emphasize: focusAlpha >= 0.995,
                        disableScreenSnap: interactionFastPath
                    });
                });
                ctx.restore();
                if (debug) {
                    drawDebugOverlay(ctx, resolution, renderState);
                }
                return true;
            }
            if (headDetailDepthActive && activeDetailTargetKind === 'concatenate') {
                drawConcatDetailStage(ctx, resolution, this.scene?.metadata?.concatDetailPreview, config);
                if (debug) {
                    drawDebugOverlay(ctx, resolution, renderState);
                }
                return true;
            }
            if (headDetailDepthActive && activeDetailTargetKind === 'output-projection') {
                drawOutputProjectionDetailStage(ctx, resolution, this.scene?.metadata?.outputProjectionDetailPreview, config);
                if (debug) {
                    drawDebugOverlay(ctx, resolution, renderState);
                }
                return true;
            }

            const canUseOverviewFocusCache = !!(
                !headDetailDepthActive
                && !interactionState?.hoveredRow
                && (overviewCurrentFocusState || overviewPreviousFocusState)
                && this.matchesOverviewRenderCache({
                    resolution,
                    worldScale,
                    offsetX,
                    offsetY
                })
                && typeof ctx.drawImage === 'function'
            );
            const canUseOverviewInteractionCache = !!(
                interactionFastPath
                && !headDetailDepthActive
                && !interactionState?.hoveredRow
                && !interactionState?.previousHoveredRow
                && !(Number(interactionState?.hoverDimStrength) > 0.001)
                && interactionState?.interactionKind !== 'zoom'
                && interactionState?.viewportAnimationActive !== true
                && !overviewCurrentFocusState
                && !overviewPreviousFocusState
                && !sceneFocusState
                && this.hasReusableOverviewRenderCache({
                    resolution
                })
                && typeof ctx.drawImage === 'function'
            );
            const canUseOverviewRowHoverCache = !!(
                !headDetailDepthActive
                && !overviewCurrentFocusState
                && !overviewPreviousFocusState
                && !sceneFocusState
                && (interactionState?.hoveredRow || interactionState?.previousHoveredRow)
                && interactionState?.interactionKind !== 'zoom'
                && interactionState?.viewportAnimationActive !== true
                && this.hasReusableOverviewRenderCache({
                    resolution
                })
                && typeof ctx.drawImage === 'function'
            );
            if (canUseOverviewInteractionCache) {
                const didDrawInteractionCache = this.drawTransformedOverviewRenderCache({
                    resolution,
                    worldScale,
                    offsetX,
                    offsetY,
                    background: resolveSceneBackgroundFill(config)
                });
                if (didDrawInteractionCache) {
                    if (debug) {
                        drawDebugOverlay(ctx, resolution, renderState);
                    }
                    return true;
                }
            }
            if (canUseOverviewRowHoverCache) {
                const fixedTextSizing = resolveMhsaDetailFixedTextSizing(this.scene, resolution.width);
                const didDrawRowHoverCache = this.drawCachedOverviewRowHoverState({
                    ctx,
                    resolution,
                    config,
                    worldScale,
                    detailScale,
                    offsetX,
                    offsetY,
                    visibleDrawableNodes,
                    visibleConnectors,
                    overviewVisibleIndex,
                    interactionState: normalizedInteractionState,
                    interactionFastPath,
                    fixedTextSizing,
                    headDetailTarget: activeHeadDetailTarget,
                    headDetailDepthActive: !!headDetailDepthActive
                });
                if (didDrawRowHoverCache) {
                    if (debug) {
                        drawDebugOverlay(ctx, resolution, renderState);
                    }
                    return true;
                }
            }
            if (canUseOverviewFocusCache) {
                const baseInactiveOpacity = Number(config?.tokens?.dimming?.inactiveOpacity) || 0.18;
                const dimStrength = Math.max(
                    0,
                    Math.min(1, Number(overviewFocusTransition?.dimStrength) || 0)
                );
                const focusBlend = Math.max(
                    0,
                    Math.min(1, Number(overviewFocusTransition?.focusBlend) || 0)
                );
                const overlayAlpha = (1 - Math.max(0, Math.min(1, baseInactiveOpacity))) * dimStrength;
                const fixedTextSizing = resolveMhsaDetailFixedTextSizing(this.scene, resolution.width);
                const didDrawFocusCache = this.drawTransformedOverviewRenderCache({
                    resolution,
                    worldScale,
                    offsetX,
                    offsetY,
                    background: resolveSceneBackgroundFill(config)
                });
                if (didDrawFocusCache) {
                    if (overlayAlpha > 0.001) {
                        ctx.save();
                        ctx.globalAlpha = overlayAlpha;
                        ctx.fillStyle = 'rgb(2, 4, 8)';
                        ctx.fillRect(0, 0, resolution.width, resolution.height);
                        ctx.restore();
                    }

                    ctx.save();
                    ctx.translate(offsetX, offsetY);
                    ctx.scale(worldScale, worldScale);
                    if (overviewPreviousFocusState) {
                        const previousFocusAlpha = overviewCurrentFocusState
                            ? (dimStrength * Math.max(0, 1 - focusBlend))
                            : dimStrength;
                        this.drawFocusedOverviewNodes({
                            ctx,
                            visibleDrawableNodes,
                            visibleConnectors,
                            overviewVisibleIndex,
                            config,
                            worldScale,
                            detailScale,
                            fixedTextSizing,
                            focusState: overviewPreviousFocusState,
                            focusAlpha: previousFocusAlpha,
                            headDetailTarget: activeHeadDetailTarget,
                            headDetailDepthActive: !!headDetailDepthActive
                        });
                    }
                    if (overviewCurrentFocusState) {
                        this.drawFocusedOverviewNodes({
                            ctx,
                            visibleDrawableNodes,
                            visibleConnectors,
                            overviewVisibleIndex,
                            config,
                            worldScale,
                            detailScale,
                            fixedTextSizing,
                            focusState: overviewCurrentFocusState,
                            focusAlpha: dimStrength * (overviewPreviousFocusState ? focusBlend : 1),
                            headDetailTarget: activeHeadDetailTarget,
                            headDetailDepthActive: !!headDetailDepthActive
                        });
                    }
                    ctx.restore();
                    if (debug) {
                        drawDebugOverlay(ctx, resolution, renderState);
                    }
                    return true;
                }
            }

            const fixedTextSizing = resolveMhsaDetailFixedTextSizing(this.scene, resolution.width);
            this.drawOverviewScene({
                ctx,
                sceneBounds,
                config,
                worldScale,
                detailScale,
                offsetX,
                offsetY,
                visibleDrawableNodes,
                visibleConnectors,
                interactionFastPath,
                interactionState: normalizedInteractionState,
                sceneFocusState: activeOverviewSceneFocusState,
                fixedTextSizing,
                headDetailTarget: activeHeadDetailTarget,
                headDetailDepthActive: !!headDetailDepthActive,
                disableInactiveFilter
            });
            const shouldRefreshOverviewRenderCache = !!(
                !headDetailDepthActive
                && !interactionFastPath
                && !interactionState?.hoveredRow
                && !overviewCurrentFocusState
                && !overviewPreviousFocusState
                && !sceneFocusState
            );
            if (shouldRefreshOverviewRenderCache) {
                this.updateOverviewRenderCache({
                    resolution,
                    config,
                    sceneBounds,
                    worldScale,
                    offsetX,
                    offsetY,
                    headDetailTarget: activeHeadDetailTarget,
                    headDetailDepthActive: !!headDetailDepthActive
                });
            }
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
