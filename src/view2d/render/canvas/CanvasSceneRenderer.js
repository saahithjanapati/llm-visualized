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
        dimmedRowOpacity = 0.18
    } = {}
) {
    const compactWidth = Math.max(
        1,
        Math.min(layoutData.compactWidth, contentBounds.width - (layoutData.innerPaddingX * 2))
    );
    const shouldDimRows = Number.isFinite(hoveredRowIndex);
    ctx.save();
    roundRectPath(ctx, contentBounds.x, contentBounds.y, contentBounds.width, contentBounds.height, cornerRadius);
    ctx.clip();
    rowItems.forEach((rowItem, index) => {
        const rowY = contentBounds.y + layoutData.innerPaddingY + index * (layoutData.rowHeight + layoutData.rowGap);
        const rowBounds = {
            x: contentBounds.x + layoutData.innerPaddingX,
            y: rowY,
            width: compactWidth,
            height: layoutData.rowHeight
        };
        ctx.fillStyle = resolveFill(ctx, rowItem.gradientCss, rowBounds, accent);
        ctx.globalAlpha = shouldDimRows && index !== hoveredRowIndex ? dimmedRowOpacity : 1;
        ctx.fillRect(rowBounds.x, rowBounds.y, rowBounds.width, rowBounds.height);
    });
    ctx.restore();
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

function resolveMatrixHoveredRowIndex(node = null, interactionState = null) {
    const hoveredRow = interactionState?.hoveredRow || null;
    if (!node?.id || !hoveredRow || hoveredRow.nodeId !== node.id) return null;
    return Number.isFinite(hoveredRow.rowIndex) ? Math.max(0, Math.floor(hoveredRow.rowIndex)) : null;
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

function resolveMatrixRowHit(node = null, entry = null, x = 0, y = 0, detailScale = 1) {
    if (!node || node.kind !== VIEW2D_NODE_KINDS.MATRIX) return null;
    if (!Number.isFinite(x) || !Number.isFinite(y)) return null;
    const rowItems = Array.isArray(node.rowItems) ? node.rowItems : [];
    if (!rowItems.length) return null;
    if (!containsPoint(entry?.contentBounds || entry?.bounds, x, y)) return null;
    if (
        node.presentation !== VIEW2D_MATRIX_PRESENTATIONS.COMPACT_ROWS
        && node.presentation !== VIEW2D_MATRIX_PRESENTATIONS.BANDED_ROWS
    ) {
        return null;
    }

    for (let index = 0; index < rowItems.length; index += 1) {
        const bounds = resolveRowHitBounds(node, entry, index);
        if (!containsPoint(bounds, x, y)) continue;
        return {
            rowIndex: index,
            rowItem: rowItems[index],
            bounds
        };
    }
    return null;
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
    const arrowHeadInset = Math.max(7, Math.min(14, Math.min(width, height) * 0.02));
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
    ctx.lineWidth = Math.max(1, Math.min(width, height) * 0.0025);
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
            { x: stackBounds.x, y: targetY },
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
            VIEW2D_VECTOR_STRIP_DEFAULTS.cornerRadius
        );
    }
    ctx.restore();
}

function drawConcatDetailStage(ctx, resolution, concatDetailPreview = null) {
    if (!ctx || !resolution) return;
    const width = Math.max(1, Number(resolution.width) || 1);
    const height = Math.max(1, Number(resolution.height) || 1);
    const arrowCount = Math.max(1, Math.floor(concatDetailPreview?.arrowCount || 12));
    const frameWidth = Math.max(112, Math.min(width * 0.13, 140));
    const frameHeight = Math.max(220, Math.min(height * 0.68, 360));
    const frameBounds = {
        x: Math.max(52, Math.min(width * 0.58, width - frameWidth - 40)),
        y: Math.max(32, (height - frameHeight) * 0.5),
        width: frameWidth,
        height: frameHeight
    };
    const arrowStartX = Math.max(34, width * 0.16);
    const arrowEndX = frameBounds.x;
    const strokeWidth = Math.max(1.25, Math.min(width, height) * 0.0026);
    const frameStroke = 'rgba(176, 182, 191, 0.88)';
    const connectorStroke = 'rgba(156, 162, 171, 0.78)';
    const innerTopInset = Math.max(16, Math.min(22, frameBounds.height * 0.08));
    const innerBottomInset = innerTopInset;
    const slotStep = (frameBounds.height - innerTopInset - innerBottomInset) / Math.max(1, arrowCount);
    const arrowHeadInset = Math.max(5, strokeWidth * 2.2);

    ctx.save();
    ctx.fillStyle = '#000';
    ctx.fillRect(0, 0, width, height);

    ctx.strokeStyle = frameStroke;
    ctx.lineWidth = strokeWidth;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    roundRectPath(ctx, frameBounds.x, frameBounds.y, frameBounds.width, frameBounds.height, 14);
    ctx.stroke();

    ctx.strokeStyle = connectorStroke;
    for (let arrowIndex = 0; arrowIndex < arrowCount; arrowIndex += 1) {
        const targetY = frameBounds.y + innerTopInset + (slotStep * (arrowIndex + 0.5));
        const leadX = arrowStartX + ((arrowEndX - arrowStartX) * 0.7);
        ctx.beginPath();
        ctx.moveTo(arrowStartX, targetY);
        ctx.lineTo(leadX, targetY);
        ctx.lineTo(arrowEndX - arrowHeadInset, targetY);
        ctx.stroke();
        drawConnectorArrowHead(
            ctx,
            { x: arrowEndX - (arrowHeadInset * 2), y: targetY },
            { x: arrowEndX, y: targetY },
            strokeWidth,
            frameStroke
        );
    }

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

function drawCaption(ctx, entry, node, config, detailScale = 1) {
    const captionLines = resolveView2dCaptionLines(node);
    if (!captionLines.length) return;
    const captionPosition = resolveView2dCaptionPosition(node);
    const minScreenHeightPx = Number.isFinite(node?.metadata?.caption?.minScreenHeightPx)
        && node.metadata.caption.minScreenHeightPx > 0
        ? node.metadata.caption.minScreenHeightPx
        : CAPTION_MIN_SCREEN_HEIGHT_PX;
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
    const maxHeight = Math.max(...lines.map(({ bounds }) => Number(bounds?.height) || 0));
    if ((maxHeight * Math.max(0.0001, detailScale)) < minScreenHeightPx) {
        return;
    }

    const captionStyle = resolveView2dStyle(
        resolveView2dCaptionStyleKey(node, VIEW2D_STYLE_KEYS.CAPTION)
    ) || {};
    const captionColor = captionStyle.color || config.tokens.palette.mutedText;

    ctx.save();
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.font = `500 ${config.component.captionFontSize}px ui-monospace, SFMono-Regular, Menlo, monospace`;
    ctx.fillStyle = captionColor;
    if (captionPosition === 'inside-top') {
        const clipBounds = entry.contentBounds || entry.bounds;
        if (clipBounds) {
            ctx.beginPath();
            ctx.rect(clipBounds.x, clipBounds.y, clipBounds.width, clipBounds.height);
            ctx.clip();
        }
    }
    lines.forEach(({ text, tex, bounds }) => {
        const displayTex = typeof tex === 'string' ? tex : '';
        if (displayTex.length && hasSimpleTexMarkup(displayTex)) {
            drawSimpleTex(ctx, displayTex, {
                x: bounds.x + (bounds.width / 2),
                y: bounds.y + (bounds.height / 2),
                fontSize: config.component.captionFontSize,
                fontWeight: 500,
                color: captionColor
            });
            return;
        }
        ctx.fillText(text || displayTex, bounds.x + (bounds.width / 2), bounds.y + (bounds.height / 2));
    });
    ctx.restore();
}

function drawCardSurfaceEffects(ctx, bounds, cornerRadius, style, safeWorldScale, projectedWidth, projectedHeight) {
    if (!bounds) return;

    const glowColor = typeof style.cardGlowColor === 'string' ? style.cardGlowColor : null;
    const glowBlur = Number.isFinite(style.cardGlowBlur) ? style.cardGlowBlur : 16;
    const glowOpacity = Number.isFinite(style.cardGlowOpacity) ? style.cardGlowOpacity : 0.2;
    const hotspotColor = typeof style.cardHotspotColor === 'string' ? style.cardHotspotColor : 'rgba(255,255,255,0.12)';
    const innerGlowColor = typeof style.cardInnerGlowColor === 'string' ? style.cardInnerGlowColor : null;
    const sheenColor = typeof style.cardSheenColor === 'string' ? style.cardSheenColor : 'rgba(255,255,255,0.14)';
    const edgeHighlight = typeof style.cardEdgeHighlight === 'string' ? style.cardEdgeHighlight : 'rgba(255,255,255,0.18)';

    if (glowColor) {
        ctx.save();
        roundRectPath(ctx, bounds.x, bounds.y, bounds.width, bounds.height, cornerRadius);
        ctx.fillStyle = glowColor;
        ctx.globalAlpha = glowOpacity;
        ctx.shadowColor = glowColor;
        ctx.shadowBlur = glowBlur / safeWorldScale;
        ctx.fill();
        ctx.restore();

        ctx.save();
        roundRectPath(ctx, bounds.x, bounds.y, bounds.width, bounds.height, cornerRadius);
        ctx.lineWidth = Math.max(0.7, 1.2) / safeWorldScale;
        ctx.strokeStyle = glowColor;
        ctx.shadowColor = glowColor;
        ctx.shadowBlur = (glowBlur * 0.8) / safeWorldScale;
        ctx.globalAlpha = Math.min(1, glowOpacity + 0.28);
        ctx.stroke();
        ctx.restore();
    }

    ctx.save();
    roundRectPath(ctx, bounds.x, bounds.y, bounds.width, bounds.height, cornerRadius);
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
    roundRectPath(ctx, bounds.x, bounds.y, bounds.width, bounds.height, cornerRadius);
    ctx.lineWidth = Math.max(0.45, 0.8) / safeWorldScale;
    ctx.strokeStyle = edgeHighlight;
    ctx.globalAlpha = 1;
    ctx.stroke();
    ctx.restore();
}

function drawCardEdgeStrokes(ctx, bounds, cornerRadius, style, safeWorldScale) {
    if (!bounds) return;
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
    { fastPath = false } = {}
) {
    if (!contentBounds || !embeddedScene?.scene || !embeddedScene?.layout?.registry) return;
    const renderCache = resolveEmbeddedSceneRenderCache(embeddedScene);
    const sceneBounds = embeddedScene.layout.sceneBounds || embeddedScene.layout.registry.getSceneBounds();
    if (!renderCache || !sceneBounds?.width || !sceneBounds?.height) return;

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
                fastPath
            });
        } else if (node.kind === VIEW2D_NODE_KINDS.TEXT || node.kind === VIEW2D_NODE_KINDS.OPERATOR) {
            drawTextLikeNode(ctx, node, entry, nestedConfig, nestedWorldScale, nestedDetailScale);
        }
    });

    renderCache.connectors.forEach(({ entry, stroke }) => {
        drawConnector(ctx, entry, nestedConfig, stroke, nestedWorldScale, {
            skipArrowHead: fastPath
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
        interactionState = null
    } = {}
) {
    const contentBounds = entry.contentBounds || entry.bounds;
    const style = (
        headDetailDepthActive && isSelectedHeadCardNode(node, headDetailTarget)
            ? resolveView2dStyle(VIEW2D_STYLE_KEYS.MHSA_HEAD_DETAIL_FRAME)
            : resolveView2dStyle(node.visual?.styleKey)
    ) || {};
    const accent = style.accent || config.tokens.palette.neutral;
    const background = style.fill || config.tokens.palette.panelBackground;
    const border = style.stroke || config.tokens.palette.border;
    const compactRowsMeta = node.metadata?.compactRows || {};
    const isVectorStripRows = node.presentation === VIEW2D_MATRIX_PRESENTATIONS.COMPACT_ROWS
        && compactRowsMeta.variant === VIEW2D_VECTOR_STRIP_VARIANT;
    const hideSurface = compactRowsMeta.hideSurface === true;
    const safeWorldScale = Math.max(0.0001, Number.isFinite(worldScale) ? worldScale : 1);
    const safeDetailScale = Math.max(0.0001, Number.isFinite(detailScale) ? detailScale : safeWorldScale);
    const cornerRadius = Number.isFinite(entry.layoutData?.cardRadius)
        ? entry.layoutData.cardRadius
        : config.tokens.matrix.cornerRadius;
    const projectedWidth = contentBounds.width * safeDetailScale;
    const projectedHeight = contentBounds.height * safeDetailScale;
    const summaryWidthThreshold = MATRIX_DETAIL_MIN_SCREEN_WIDTH_PX * (fastPath ? 1.45 : 1);
    const summaryHeightThreshold = MATRIX_DETAIL_MIN_SCREEN_HEIGHT_PX * (fastPath ? 1.35 : 1);
    const useSummaryInterior = projectedWidth < summaryWidthThreshold
        || projectedHeight < summaryHeightThreshold;
    const embeddedScene = node.metadata?.embeddedScene || null;

    ctx.save();
    if (!hideSurface) {
        roundRectPath(ctx, contentBounds.x, contentBounds.y, contentBounds.width, contentBounds.height, cornerRadius);
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
            drawCardSurfaceEffects(ctx, contentBounds, cornerRadius, style, safeWorldScale, projectedWidth, projectedHeight);
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
            { fastPath }
        );
    }

    const layoutData = entry.layoutData || {};
    const rowItems = Array.isArray(node.rowItems) ? node.rowItems : [];
    const columnItems = Array.isArray(node.columnItems) ? node.columnItems : [];
    const hoveredRowIndex = resolveMatrixHoveredRowIndex(node, interactionState);
    const shouldDimRows = Number.isFinite(hoveredRowIndex) && rowItems.length > 1;
    const dimmedRowOpacity = 0.18;
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
                ctx.globalAlpha = shouldDimRows && index !== hoveredRowIndex ? dimmedRowOpacity : 1;
                ctx.fill();
                ctx.globalAlpha = 1;

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
        if (isVectorStripRows) {
            drawVectorStripRows(
                ctx,
                rowItems,
                contentBounds,
                layoutData,
                accent,
                cornerRadius,
                shouldDimRows
                    ? {
                        hoveredRowIndex,
                        dimmedRowOpacity
                    }
                    : {}
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
                ctx.globalAlpha = shouldDimRows && index !== hoveredRowIndex ? dimmedRowOpacity : 1;
                ctx.fill();
                ctx.globalAlpha = 1;
            });
        }
    } else if (node.presentation === VIEW2D_MATRIX_PRESENTATIONS.GRID) {
        if (useSummaryInterior || fastPath) {
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
        }
    } else if (node.presentation === VIEW2D_MATRIX_PRESENTATIONS.COLUMN_STRIP) {
        if (useSummaryInterior || fastPath) {
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
                ctx.fill();
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
        drawCardEdgeStrokes(ctx, contentBounds, cornerRadius, style, safeWorldScale);
    }
    ctx.restore();
    drawCaption(ctx, entry, node, config, safeDetailScale);
}

function drawTextLikeNode(ctx, node, entry, config, worldScale = 1, detailScale = worldScale) {
    const bounds = entry.contentBounds || entry.bounds;
    const safeWorldScale = Math.max(0.0001, Number.isFinite(worldScale) ? worldScale : 1);
    const safeDetailScale = Math.max(0.0001, Number.isFinite(detailScale) ? detailScale : safeWorldScale);
    const isPersistentOperator = node.kind === VIEW2D_NODE_KINDS.OPERATOR
        && (
            node.role === 'residual-add-operator'
            || node.visual?.styleKey === 'residual.add-symbol'
        );
    if (!isPersistentOperator && (bounds.height * safeDetailScale) < TEXT_MIN_SCREEN_HEIGHT_PX) {
        return;
    }
    ctx.save();
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    const text = node.text || node.tex || '';
    const tex = node.tex || '';
    if (node.kind === VIEW2D_NODE_KINDS.OPERATOR) {
        const baseFontSize = entry.layoutData?.fontSize || config.component.operatorFontSize;
        const isPlusOperator = node.semantic?.operatorKey === 'plus' || node.text === '+';
        const persistentMinScreenFontPx = isPlusOperator
            ? PERSISTENT_PLUS_OPERATOR_MIN_SCREEN_FONT_PX
            : PERSISTENT_OPERATOR_MIN_SCREEN_FONT_PX;
        const adjustedFontSize = isPersistentOperator
            ? Math.max(baseFontSize, persistentMinScreenFontPx / safeWorldScale)
            : baseFontSize;
        const renderedFontSize = isPlusOperator
            ? adjustedFontSize * PLUS_OPERATOR_FONT_SCALE
            : adjustedFontSize;
        const fontWeight = isPlusOperator ? PLUS_OPERATOR_FONT_WEIGHT : 600;
        ctx.font = `${fontWeight} ${renderedFontSize}px ui-monospace, SFMono-Regular, Menlo, monospace`;
        ctx.fillStyle = resolveView2dStyle(node.visual?.styleKey)?.color || config.tokens.palette.text;
    } else {
        let renderedFontSize = entry.layoutData?.fontSize || config.component.labelFontSize;
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
        ctx.font = resolveView2dTextFont({
            fontSize: renderedFontSize,
            fontWeight: 500
        });
        ctx.fillStyle = resolveView2dStyle(node.visual?.styleKey)?.color || config.tokens.palette.text;
    }
    ctx.beginPath();
    ctx.rect(bounds.x - 1, bounds.y, bounds.width + 2, bounds.height);
    ctx.clip();
    if (tex && hasSimpleTexMarkup(tex)) {
        drawSimpleTex(ctx, tex, {
            x: bounds.x + (bounds.width / 2),
            y: bounds.y + (bounds.height / 2),
            fontSize: entry.layoutData?.fontSize || config.component.labelFontSize,
            fontWeight: node.kind === VIEW2D_NODE_KINDS.OPERATOR ? 600 : 500,
            color: resolveView2dStyle(node.visual?.styleKey)?.color || config.tokens.palette.text
        });
    } else {
        ctx.fillText(text, bounds.x + (bounds.width / 2), bounds.y + (bounds.height / 2));
    }
    ctx.restore();
    drawCaption(ctx, entry, node, config, safeDetailScale);
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
    { skipArrowHead = false } = {}
) {
    const points = Array.isArray(connectorEntry.pathPoints) ? connectorEntry.pathPoints : [];
    if (points.length < 2) return;
    const foregroundStroke = accent || config.tokens.palette.neutral;
    const safeWorldScale = Math.max(0.0001, Number.isFinite(worldScale) ? worldScale : 1);
    const targetScreenWidthPx = safeWorldScale < 0.35
        ? 0.58
        : (safeWorldScale < 0.7 ? 0.72 : (safeWorldScale < 1.25 ? 0.88 : 1.02));
    const strokeWidth = Math.max(0.42, targetScreenWidthPx) / safeWorldScale;
    const strokeOpacity = safeWorldScale < 0.35
        ? 0.46
        : (safeWorldScale < 0.7 ? 0.54 : (safeWorldScale < 1.25 ? 0.66 : 0.78));
    const flattenedForegroundStroke = connectorEntry?.metadata?.preserveColor === true
        ? foregroundStroke
        : (flattenColorAgainstBlack(foregroundStroke, strokeOpacity) || foregroundStroke);
    const snappedPoints = points.map((point) => snapWorldPointToConnectorGrid(point, safeWorldScale));
    const tailPoint = snappedPoints[Math.max(0, snappedPoints.length - 2)];
    const headPoint = snappedPoints[snappedPoints.length - 1];

    ctx.save();
    ctx.lineCap = 'butt';
    ctx.lineJoin = 'miter';
    ctx.miterLimit = 2;
    ctx.globalCompositeOperation = 'source-over';
    ctx.filter = 'none';
    ctx.globalAlpha = 1;
    ctx.beginPath();
    ctx.moveTo(snappedPoints[0].x, snappedPoints[0].y);
    for (let index = 1; index < snappedPoints.length; index += 1) {
        ctx.lineTo(snappedPoints[index].x, snappedPoints[index].y);
    }
    ctx.lineWidth = strokeWidth;
    ctx.shadowBlur = 0;
    ctx.strokeStyle = flattenedForegroundStroke;
    ctx.stroke();
    if (!skipArrowHead) {
        drawConnectorArrowHead(
            ctx,
            tailPoint,
            headPoint,
            strokeWidth,
            flattenedForegroundStroke
        );
    }
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
    return {
        scene,
        layout,
        drawableNodes,
        drawableNodesById: new Map(drawableNodes.map((item) => [item.node.id, item])),
        connectors: allNodes
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
            .filter((item) => item.entry)
    };
}

function resolvePreparedSceneHitAtPoint(preparedSceneState = null, x = 0, y = 0, detailScale = 1) {
    const registry = preparedSceneState?.layout?.registry || null;
    if (!registry || !Number.isFinite(x) || !Number.isFinite(y)) return null;
    const entry = registry.resolveNodeEntryAtPoint(x, y, {
        includeGroups: false
    });
    if (!entry) return null;
    const drawable = preparedSceneState.drawableNodesById.get(entry.nodeId) || null;
    const node = drawable?.node || null;
    return {
        entry,
        node,
        rowHit: resolveMatrixRowHit(node, entry, x, y, detailScale)
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
        this.drawableNodesById = new Map();
        this.connectors = [];
        this.lastRenderState = null;
        this.latchedDetailScale = null;
        this.visibleDrawableNodes = [];
        this.visibleConnectors = [];
        this.headDetailSceneState = null;
        this.activeDetailSceneRenderState = null;
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
        this.latchedDetailScale = null;
        this.activeDetailSceneRenderState = null;
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
        this.headDetailSceneState = createPreparedSceneState(this.scene?.metadata?.headDetailScene || null);
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

    resolveInteractiveHitAtPoint(x = 0, y = 0) {
        if (!this.layout?.registry || !Number.isFinite(x) || !Number.isFinite(y)) return null;
        const entry = this.layout.registry.resolveNodeEntryAtPoint(x, y, {
            includeGroups: false
        });
        if (!entry) return null;
        const drawable = this.drawableNodesById.get(entry.nodeId) || null;
        const node = drawable?.node || null;
        const detailScale = this.lastRenderState?.detailScale || this.lastRenderState?.worldScale || 1;
        return {
            entry,
            node,
            rowHit: resolveMatrixRowHit(node, entry, x, y, detailScale)
        };
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
        return this.resolveInteractiveHitAtPoint(sceneX, sceneY);
    }

    render({
        width = null,
        height = null,
        dpr = null,
        dprCap = null,
        clear = true,
        debug = false,
        viewportTransform = null,
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
        const activeHeadDetailTarget = normalizeHeadDetailTarget(this.scene?.metadata?.headDetailTarget);
        const activeConcatDetailTarget = normalizeConcatDetailTarget(this.scene?.metadata?.concatDetailTarget);
        const activeOutputProjectionDetailTarget = normalizeOutputProjectionDetailTarget(
            this.scene?.metadata?.outputProjectionDetailTarget
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
        const activeDetailTargetKind = activeHeadDetailTarget
            ? 'head'
            : (activeConcatDetailTarget ? 'concatenate' : (activeOutputProjectionDetailTarget ? 'output-projection' : ''));
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
            interactionFastPath,
            headDetailTarget: activeHeadDetailTarget ? { ...activeHeadDetailTarget } : null,
            headDetailBounds: cloneBounds(activeHeadDetailBounds),
            concatDetailTarget: activeConcatDetailTarget ? { ...activeConcatDetailTarget } : null,
            concatDetailBounds: cloneBounds(activeConcatDetailBounds),
            outputProjectionDetailTarget: activeOutputProjectionDetailTarget
                ? { ...activeOutputProjectionDetailTarget }
                : null,
            outputProjectionDetailBounds: cloneBounds(activeOutputProjectionDetailBounds),
            detailTargetKind: activeDetailTargetKind,
            headDetailDepthActive: !!headDetailDepthActive
        };
        this.lastRenderState = renderState;

        try {
            if (headDetailDepthActive && activeDetailTargetKind === 'head' && this.headDetailSceneState?.layout?.registry) {
                const detailSceneBounds = this.headDetailSceneState.layout.sceneBounds
                    || this.headDetailSceneState.layout.registry.getSceneBounds();
                const detailViewport = resolveRenderViewportTransform(detailSceneBounds, resolution, null);
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
                ctx.save();
                ctx.fillStyle = '#000';
                ctx.fillRect(0, 0, resolution.width, resolution.height);
                ctx.translate(detailOffsetX, detailOffsetY);
                ctx.scale(detailWorldScale, detailWorldScale);
                detailVisibleDrawableNodes.forEach(({ node, entry }) => {
                    if (node.kind === VIEW2D_NODE_KINDS.MATRIX) {
                        drawMatrixNode(ctx, node, entry, config, detailWorldScale, detailWorldScale, {
                            skipSurfaceEffects: interactionFastPath,
                            fastPath: interactionFastPath,
                            interactionState
                        });
                    } else if (node.kind === VIEW2D_NODE_KINDS.TEXT || node.kind === VIEW2D_NODE_KINDS.OPERATOR) {
                        drawTextLikeNode(ctx, node, entry, config, detailWorldScale, detailWorldScale);
                    }
                });
                detailVisibleConnectors.forEach(({ entry, stroke }) => {
                    drawConnector(ctx, entry, config, stroke, detailWorldScale, {
                        skipArrowHead: interactionFastPath
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

            ctx.save();
            ctx.translate(offsetX, offsetY);
            ctx.scale(worldScale, worldScale);
            ctx.fillStyle = config.tokens.palette.sceneBackground;
            ctx.fillRect(0, 0, sceneBounds.width, sceneBounds.height);

            visibleDrawableNodes.forEach(({ node, entry }) => {
                if (node.kind === VIEW2D_NODE_KINDS.MATRIX) {
                    drawMatrixNode(ctx, node, entry, config, worldScale, detailScale, {
                        skipSurfaceEffects: interactionFastPath,
                        fastPath: interactionFastPath,
                        headDetailTarget: activeHeadDetailTarget,
                        headDetailDepthActive: !!headDetailDepthActive,
                        interactionState
                    });
                } else if (node.kind === VIEW2D_NODE_KINDS.TEXT || node.kind === VIEW2D_NODE_KINDS.OPERATOR) {
                    drawTextLikeNode(ctx, node, entry, config, worldScale, detailScale);
                }
            });

            visibleConnectors.forEach(({ entry, stroke }) => {
                drawConnector(ctx, entry, config, stroke, worldScale, {
                    skipArrowHead: interactionFastPath
                });
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
