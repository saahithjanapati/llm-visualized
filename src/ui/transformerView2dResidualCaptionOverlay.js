import { resolveView2dCaptionLines } from '../view2d/captionUtils.js';
import { resolveView2dCaptionPosition } from '../view2d/captionUtils.js';
import {
    flattenSceneNodes,
    VIEW2D_NODE_KINDS
} from '../view2d/schema/sceneTypes.js';

const RESIDUAL_CAPTION_TOP_GAP_PX = 4;
const RESIDUAL_CAPTION_BOTTOM_GAP_PX = 4;
const RESIDUAL_CAPTION_FONT_RAMP_PX = 44;
const RESIDUAL_CAPTION_LABEL_MIN_FONT_PX = 12;
const RESIDUAL_CAPTION_LABEL_MAX_FONT_PX = 14;
const RESIDUAL_CAPTION_DIMENSIONS_MIN_FONT_PX = 10;
const RESIDUAL_CAPTION_DIMENSIONS_MAX_FONT_PX = 11;
const RESIDUAL_TOP_CAPTION_LABEL_RATIO = 0.34;
const RESIDUAL_TOP_CAPTION_DIMENSIONS_RATIO = 0.24;

function renderKatex(targetEl, tex = '', fallbackText = '') {
    if (!targetEl) return;
    const safeTex = typeof tex === 'string' ? tex.trim() : '';
    const safeFallback = String(fallbackText ?? '').trim();
    const cacheKey = `${safeTex}::${safeFallback}`;
    if (targetEl.dataset.renderKey === cacheKey) return;

    const katex = (typeof window !== 'undefined') ? window.katex : null;
    if (safeTex && katex && typeof katex.renderToString === 'function') {
        try {
            targetEl.innerHTML = katex.renderToString(safeTex, {
                throwOnError: false,
                displayMode: false
            });
            targetEl.dataset.renderKey = cacheKey;
            return;
        } catch (_) {
            // Fall through to the plain-text fallback.
        }
    }

    targetEl.textContent = safeFallback || safeTex;
    targetEl.dataset.renderKey = cacheKey;
}

function clamp(value, min, max) {
    return Math.max(min, Math.min(max, value));
}

function resolveCaptionFontPx({
    useMatrixRelativeSizing = false,
    projectedContentHeight = 0,
    sizeProgress = 0,
    minFontPx = 12,
    maxFontPx = 14,
    heightRatio = 0.34,
    scale = 1
} = {}) {
    const safeScale = Number.isFinite(scale) && scale > 0 ? Number(scale) : 1;
    if (useMatrixRelativeSizing) {
        return Math.max(
            minFontPx,
            projectedContentHeight * heightRatio * safeScale
        );
    }
    return (
        minFontPx
        + ((maxFontPx - minFontPx) * sizeProgress)
    ) * safeScale;
}

function intersectsCanvas(bounds = null, width = 0, height = 0) {
    if (!bounds) return false;
    return !(
        (bounds.x + bounds.width) < 0
        || (bounds.y + bounds.height) < 0
        || bounds.x > width
        || bounds.y > height
    );
}

function isResidualCaptionNode(node = null) {
    return node?.kind === VIEW2D_NODE_KINDS.MATRIX
        && String(node?.metadata?.caption?.renderMode || '').trim().toLowerCase() === 'dom-katex';
}

function createCaptionItem(documentRef) {
    const itemEl = documentRef.createElement('div');
    itemEl.className = 'detail-transformer-view2d-residual-caption';
    const labelEl = documentRef.createElement('div');
    labelEl.className = 'detail-transformer-view2d-residual-caption-line detail-transformer-view2d-residual-caption-line--label';
    const dimensionsEl = documentRef.createElement('div');
    dimensionsEl.className = 'detail-transformer-view2d-residual-caption-line detail-transformer-view2d-residual-caption-line--dimensions';
    itemEl.append(labelEl, dimensionsEl);
    return {
        itemEl,
        labelEl,
        dimensionsEl
    };
}

export function createTransformerView2dResidualCaptionOverlay({
    documentRef = typeof document !== 'undefined' ? document : null,
    parent = null
} = {}) {
    if (!documentRef || !parent) {
        return {
            element: null,
            sync: () => false,
            hide: () => {},
            destroy: () => {}
        };
    }

    const root = documentRef.createElement('div');
    root.className = 'detail-transformer-view2d-caption-overlay';
    parent.appendChild(root);

    const itemMap = new Map();

    function ensureItem(nodeId = '') {
        const safeNodeId = String(nodeId || '');
        if (!safeNodeId.length) return null;
        if (itemMap.has(safeNodeId)) {
            return itemMap.get(safeNodeId);
        }
        const item = createCaptionItem(documentRef);
        root.appendChild(item.itemEl);
        itemMap.set(safeNodeId, item);
        return item;
    }

    function hide() {
        root.style.display = 'none';
        itemMap.forEach(({ itemEl }) => {
            itemEl.hidden = true;
        });
    }

    return {
        element: root,
        sync({
            scene = null,
            layout = null,
            canvas = null,
            projectBounds = null,
            visible = false,
            enabled = true
        } = {}) {
            if (!visible || !enabled || !scene || !layout?.registry || !canvas || typeof projectBounds !== 'function') {
                hide();
                return false;
            }

            const canvasWidth = Math.max(1, canvas.clientWidth || canvas.width || 1);
            const canvasHeight = Math.max(1, canvas.clientHeight || canvas.height || 1);
            root.style.display = 'block';
            root.style.left = `${Math.round(canvas.offsetLeft || 0)}px`;
            root.style.top = `${Math.round(canvas.offsetTop || 0)}px`;
            root.style.width = `${Math.round(canvasWidth)}px`;
            root.style.height = `${Math.round(canvasHeight)}px`;

            const seenIds = new Set();
            flattenSceneNodes(scene).forEach((node) => {
                if (!isResidualCaptionNode(node)) return;
                const entry = layout.registry.getNodeEntry(node.id);
                if (!entry?.contentBounds) return;

                const lines = resolveView2dCaptionLines(node);
                if (!lines.length) return;

                const contentBounds = projectBounds(entry.contentBounds);
                if (!contentBounds) return;
                const minScreenHeightPx = Number.isFinite(node?.metadata?.caption?.minScreenHeightPx)
                    && node.metadata.caption.minScreenHeightPx > 0
                    ? node.metadata.caption.minScreenHeightPx
                    : 18;
                const projectedContentHeight = Math.max(0, Number(contentBounds.height) || 0);
                if (projectedContentHeight < minScreenHeightPx) return;
                if (!intersectsCanvas(contentBounds, canvasWidth, canvasHeight)) return;

                const item = ensureItem(node.id);
                if (!item) return;
                seenIds.add(node.id);
                const sizeProgress = clamp(
                    (projectedContentHeight - minScreenHeightPx) / RESIDUAL_CAPTION_FONT_RAMP_PX,
                    0,
                    1
                );
                const captionPosition = resolveView2dCaptionPosition(node);
                const useMatrixRelativeSizing = captionPosition !== 'bottom'
                    || node?.metadata?.caption?.scaleWithNode === true;
                const labelScale = Number.isFinite(node?.metadata?.caption?.labelScale)
                    && node.metadata.caption.labelScale > 0
                    ? Number(node.metadata.caption.labelScale)
                    : 1;
                const dimensionsScale = Number.isFinite(node?.metadata?.caption?.dimensionsScale)
                    && node.metadata.caption.dimensionsScale > 0
                    ? Number(node.metadata.caption.dimensionsScale)
                    : 1;
                const labelFontPx = resolveCaptionFontPx({
                    useMatrixRelativeSizing,
                    projectedContentHeight,
                    sizeProgress,
                    minFontPx: RESIDUAL_CAPTION_LABEL_MIN_FONT_PX,
                    maxFontPx: RESIDUAL_CAPTION_LABEL_MAX_FONT_PX,
                    heightRatio: RESIDUAL_TOP_CAPTION_LABEL_RATIO,
                    scale: labelScale
                });
                const dimensionsFontPx = resolveCaptionFontPx({
                    useMatrixRelativeSizing,
                    projectedContentHeight,
                    sizeProgress,
                    minFontPx: RESIDUAL_CAPTION_DIMENSIONS_MIN_FONT_PX,
                    maxFontPx: RESIDUAL_CAPTION_DIMENSIONS_MAX_FONT_PX,
                    heightRatio: RESIDUAL_TOP_CAPTION_DIMENSIONS_RATIO,
                    scale: dimensionsScale
                });
                const anchorX = contentBounds.x + (contentBounds.width / 2);
                const anchorY = captionPosition === 'bottom'
                    ? contentBounds.y + contentBounds.height + RESIDUAL_CAPTION_BOTTOM_GAP_PX
                    : contentBounds.y - RESIDUAL_CAPTION_TOP_GAP_PX;
                item.itemEl.hidden = false;
                item.itemEl.dataset.captionPosition = captionPosition === 'bottom' ? 'bottom' : 'top';
                item.itemEl.style.left = `${Math.round(anchorX)}px`;
                item.itemEl.style.top = `${Math.round(anchorY)}px`;
                item.itemEl.style.removeProperty('width');
                item.itemEl.style.setProperty('--detail-transformer-view2d-caption-label-size', `${labelFontPx.toFixed(2)}px`);
                item.itemEl.style.setProperty('--detail-transformer-view2d-caption-dimensions-size', `${dimensionsFontPx.toFixed(2)}px`);

                renderKatex(item.labelEl, lines[0]?.tex, lines[0]?.text);
                const dimensionsLine = lines[1] || null;
                if (dimensionsLine) {
                    renderKatex(item.dimensionsEl, dimensionsLine.tex, dimensionsLine.text);
                    item.dimensionsEl.hidden = false;
                } else {
                    item.dimensionsEl.hidden = true;
                    item.dimensionsEl.textContent = '';
                    delete item.dimensionsEl.dataset.renderKey;
                }
            });

            itemMap.forEach((item, nodeId) => {
                if (!seenIds.has(nodeId)) {
                    item.itemEl.hidden = true;
                }
            });

            return true;
        },
        hide,
        destroy() {
            if (root.parentElement) {
                root.parentElement.removeChild(root);
            }
            itemMap.clear();
        }
    };
}
