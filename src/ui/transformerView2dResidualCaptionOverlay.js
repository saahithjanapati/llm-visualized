import { resolveView2dCaptionLines } from '../view2d/captionUtils.js';
import { resolveView2dCaptionPosition } from '../view2d/captionUtils.js';
import {
    flattenSceneNodes,
    VIEW2D_NODE_KINDS
} from '../view2d/schema/sceneTypes.js';
import {
    resolveMhsaDetailFixedTextSizing,
    resolveView2dSceneTextZoomPolicy,
    VIEW2D_TEXT_ZOOM_BEHAVIORS
} from '../view2d/shared/mhsaDetailFixedLabelSizing.js';
import {
    normalizeSceneFocusState,
    resolveSceneNodeFocusAlpha
} from '../view2d/sceneFocusState.js';

const RESIDUAL_CAPTION_TOP_GAP_PX = 4;
const RESIDUAL_CAPTION_BOTTOM_GAP_PX = 4;
const RESIDUAL_CAPTION_FONT_RAMP_PX = 44;
const RESIDUAL_CAPTION_LABEL_MIN_FONT_PX = 14;
const RESIDUAL_CAPTION_LABEL_MAX_FONT_PX = 16;
const RESIDUAL_CAPTION_DIMENSIONS_MIN_FONT_PX = 12;
const RESIDUAL_CAPTION_DIMENSIONS_MAX_FONT_PX = 14;
const RESIDUAL_TOP_CAPTION_LABEL_RATIO = 0.34;
const RESIDUAL_TOP_CAPTION_DIMENSIONS_RATIO = 0.24;
const ZOOMED_OUT_FONT_SOFTEN_MIN_SCALE = 0.82;
const ZOOMED_OUT_FONT_SOFTEN_RANGE_PX = 24;
const MHSA_UNIFORM_CAPTION_MIN_SCREEN_HEIGHT_PX = 28;
const MHSA_UNIFORM_CAPTION_LABEL_MIN_FONT_PX = 17;
const MHSA_UNIFORM_CAPTION_LABEL_MAX_FONT_PX = 17;
const MHSA_UNIFORM_CAPTION_DIMENSIONS_MIN_FONT_PX = 14;
const MHSA_UNIFORM_CAPTION_DIMENSIONS_MAX_FONT_PX = 14;
const MHSA_DETAIL_SCENE_RELATIVE_CAPTION_LABEL_BOOST = 1.14;
const MHSA_DETAIL_SCENE_RELATIVE_CAPTION_DIMENSIONS_BOOST = 1.1;
const MHSA_DETAIL_SCENE_RELATIVE_DOM_TEXT_BOOST = 1.08;
const DEFAULT_CAPTION_LABEL_ROLE_SCALE = 1;
const DEFAULT_CAPTION_DIMENSIONS_ROLE_SCALE = 1;
const DEFAULT_CAPTION_KATEX_SUBSCRIPT_SCALE_EM = 0.8;
const DEFAULT_CAPTION_INLINE_SUBSCRIPT_SCALE_EM = 0.84;
const DEFAULT_CAPTION_INLINE_SUBSCRIPT_OFFSET_EM = 0.28;
const PROJECTION_BIAS_CAPTION_LABEL_ROLE_SCALE = 4.1;
const LAYER_NORM_PARAM_CAPTION_LABEL_ROLE_SCALE = 2.75;
const PROJECTION_BIAS_CAPTION_DIMENSIONS_ROLE_SCALE = 2.8;
const PROJECTION_BIAS_CAPTION_KATEX_SUBSCRIPT_SCALE_EM = 0.62;
const PROJECTION_BIAS_CAPTION_INLINE_SUBSCRIPT_SCALE_EM = 0.6;
const PROJECTION_BIAS_CAPTION_INLINE_SUBSCRIPT_OFFSET_EM = 0.36;
const OVERLAY_FOCUS_INACTIVE_OPACITY = 0.18;

function escapeHtml(value = '') {
    return String(value)
        .replaceAll('&', '&amp;')
        .replaceAll('<', '&lt;')
        .replaceAll('>', '&gt;')
        .replaceAll('"', '&quot;')
        .replaceAll("'", '&#39;');
}

function resolveInlineSubscriptParts(tex = '', fallbackText = '') {
    const safeTex = typeof tex === 'string' ? tex.trim() : '';
    const safeFallback = typeof fallbackText === 'string' ? fallbackText.trim() : '';
    const rawValue = safeTex || safeFallback;
    if (!rawValue.length) return null;

    const patterns = [
        /^([A-Za-z]+)_\{\\mathrm\{([A-Za-z]+)\}\}$/,
        /^([A-Za-z]+)_\{\\([A-Za-z]+)\}$/,
        /^([A-Za-z]+)_\{([A-Za-z]+)\}$/,
        /^([A-Za-z]+)_([A-Za-z]+)$/
    ];
    for (const pattern of patterns) {
        const match = rawValue.match(pattern);
        if (!match) continue;
        const [, base, sub] = match;
        if (!base || !sub) continue;
        return { base, sub };
    }
    return null;
}

function renderInlineSubscript(targetEl, base = '', sub = '') {
    targetEl.innerHTML = `
        <span class="detail-transformer-view2d-inline-subscript">
            <span class="detail-transformer-view2d-inline-subscript__base">${escapeHtml(base)}</span>
            <span class="detail-transformer-view2d-inline-subscript__sub">${escapeHtml(sub)}</span>
        </span>
    `;
}

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

    const inlineSubscriptParts = resolveInlineSubscriptParts(safeTex, safeFallback);
    if (inlineSubscriptParts) {
        renderInlineSubscript(targetEl, inlineSubscriptParts.base, inlineSubscriptParts.sub);
        targetEl.dataset.renderKey = cacheKey;
        return;
    }

    targetEl.textContent = safeFallback || safeTex;
    targetEl.dataset.renderKey = cacheKey;
}

function clamp(value, min, max) {
    return Math.max(min, Math.min(max, value));
}

export function resolveCaptionScreenExtent({
    captionPosition = 'bottom',
    projectedContentWidth = 0,
    projectedContentHeight = 0,
    scaleWithNode = false
} = {}) {
    const safeWidth = Number.isFinite(projectedContentWidth)
        ? Math.max(0, Number(projectedContentWidth))
        : 0;
    const safeHeight = Number.isFinite(projectedContentHeight)
        ? Math.max(0, Number(projectedContentHeight))
        : 0;
    const safePosition = String(captionPosition || '').trim().toLowerCase();
    if (scaleWithNode === true || safePosition !== 'bottom') {
        return safeHeight;
    }
    return Math.min(safeWidth, safeHeight);
}

export function resolveCaptionFontPx({
    useMatrixRelativeSizing = false,
    projectedContentHeight = 0,
    sizeProgress = 0,
    minFontPx = 12,
    maxFontPx = 14,
    heightRatio = 0.34,
    scale = 1,
    enforceMinFontPx = true
} = {}) {
    const safeProjectedContentHeight = Number.isFinite(projectedContentHeight)
        ? Math.max(0, Number(projectedContentHeight))
        : 0;
    const safeScale = Number.isFinite(scale) && scale > 0 ? Number(scale) : 1;
    if (useMatrixRelativeSizing) {
        const relativeFontPx = safeProjectedContentHeight * heightRatio * safeScale;
        return enforceMinFontPx
            ? Math.max(minFontPx, relativeFontPx)
            : relativeFontPx;
    }
    return (
        minFontPx
        + ((maxFontPx - minFontPx) * sizeProgress)
    ) * safeScale;
}

function resolveZoomedOutFontScale(projectedExtent = 0, minScreenHeightPx = 0) {
    const safeExtent = Number.isFinite(projectedExtent) ? Math.max(0, Number(projectedExtent)) : 0;
    const safeMinScreenHeightPx = Number.isFinite(minScreenHeightPx)
        ? Math.max(0, Number(minScreenHeightPx))
        : 0;
    const softenStartPx = safeMinScreenHeightPx + ZOOMED_OUT_FONT_SOFTEN_RANGE_PX;
    if (safeExtent >= softenStartPx) return 1;
    if (softenStartPx <= safeMinScreenHeightPx) return ZOOMED_OUT_FONT_SOFTEN_MIN_SCALE;
    const progress = clamp(
        (safeExtent - safeMinScreenHeightPx) / Math.max(1, softenStartPx - safeMinScreenHeightPx),
        0,
        1
    );
    return ZOOMED_OUT_FONT_SOFTEN_MIN_SCALE
        + ((1 - ZOOMED_OUT_FONT_SOFTEN_MIN_SCALE) * progress);
}

function applyZoomedOutFontDamping(fontPx = 0, projectedExtent = 0, minScreenHeightPx = 0, {
    absoluteMinFontPx = 0
} = {}) {
    const safeFontPx = Number.isFinite(fontPx) ? Math.max(0, Number(fontPx)) : 0;
    const dampedFontPx = safeFontPx * resolveZoomedOutFontScale(projectedExtent, minScreenHeightPx);
    return Math.max(
        Number.isFinite(absoluteMinFontPx) ? Math.max(0, Number(absoluteMinFontPx)) : 0,
        dampedFontPx
    );
}

function isSceneRelativeTextZoomBehavior(zoomBehavior = '') {
    return zoomBehavior === VIEW2D_TEXT_ZOOM_BEHAVIORS.SCENE_RELATIVE;
}

function resolveOverlayFontPx({
    zoomBehavior = VIEW2D_TEXT_ZOOM_BEHAVIORS.SCREEN_ADAPTIVE,
    baseFontPx = 0,
    projectedExtent = 0,
    minScreenHeightPx = 0,
    fixedScreenFontPx = null,
    minScreenFontPx = 0,
    maxScreenFontPx = null
} = {}) {
    const safeBaseFontPx = Number.isFinite(baseFontPx) ? Math.max(0, Number(baseFontPx)) : 0;
    if (zoomBehavior === VIEW2D_TEXT_ZOOM_BEHAVIORS.SCENE_RELATIVE) {
        return safeBaseFontPx;
    }
    if (zoomBehavior === VIEW2D_TEXT_ZOOM_BEHAVIORS.SCREEN_FIXED) {
        return Number.isFinite(fixedScreenFontPx) && fixedScreenFontPx > 0
            ? Number(fixedScreenFontPx)
            : safeBaseFontPx;
    }
    const dampedFontPx = applyZoomedOutFontDamping(
        safeBaseFontPx,
        projectedExtent,
        minScreenHeightPx,
        {
            absoluteMinFontPx: minScreenFontPx
        }
    );
    return Number.isFinite(maxScreenFontPx) && maxScreenFontPx > 0
        ? Math.min(dampedFontPx, Number(maxScreenFontPx))
        : dampedFontPx;
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

function applyCaptionRoleStyling(itemEl, node = null) {
    if (!itemEl) return;
    const nodeRole = String(node?.role || '').trim().toLowerCase();
    const isProjectionBiasNode = nodeRole === 'projection-bias'
        || nodeRole === 'mlp-up-bias'
        || nodeRole === 'mlp-down-bias';
    const isLayerNormParamNode = nodeRole === 'layer-norm-scale'
        || nodeRole === 'layer-norm-shift';
    itemEl.style.setProperty(
        '--detail-transformer-view2d-caption-label-role-scale',
        String(
            isProjectionBiasNode
                ? PROJECTION_BIAS_CAPTION_LABEL_ROLE_SCALE
                : (isLayerNormParamNode
                    ? LAYER_NORM_PARAM_CAPTION_LABEL_ROLE_SCALE
                    : DEFAULT_CAPTION_LABEL_ROLE_SCALE)
        )
    );
    itemEl.style.setProperty(
        '--detail-transformer-view2d-caption-dimensions-role-scale',
        String(isProjectionBiasNode
            ? PROJECTION_BIAS_CAPTION_DIMENSIONS_ROLE_SCALE
            : DEFAULT_CAPTION_DIMENSIONS_ROLE_SCALE)
    );
    itemEl.style.setProperty(
        '--detail-transformer-view2d-caption-katex-subscript-scale',
        `${isProjectionBiasNode
            ? PROJECTION_BIAS_CAPTION_KATEX_SUBSCRIPT_SCALE_EM
            : DEFAULT_CAPTION_KATEX_SUBSCRIPT_SCALE_EM}em`
    );
    itemEl.style.setProperty(
        '--detail-transformer-view2d-caption-inline-subscript-scale',
        `${isProjectionBiasNode
            ? PROJECTION_BIAS_CAPTION_INLINE_SUBSCRIPT_SCALE_EM
            : DEFAULT_CAPTION_INLINE_SUBSCRIPT_SCALE_EM}em`
    );
    itemEl.style.setProperty(
        '--detail-transformer-view2d-caption-inline-subscript-offset',
        `${isProjectionBiasNode
            ? PROJECTION_BIAS_CAPTION_INLINE_SUBSCRIPT_OFFSET_EM
            : DEFAULT_CAPTION_INLINE_SUBSCRIPT_OFFSET_EM}em`
    );
}

function resolveNodeVisualOpacity(node = null) {
    const opacity = Number(node?.visual?.opacity);
    return Number.isFinite(opacity)
        ? Math.max(0, Math.min(1, opacity))
        : 1;
}

function buildCaptionCandidate({
    node = null,
    lines = [],
    contentBounds = null,
    captionPosition = 'bottom',
    scaleWithNode = false,
    minScreenHeightPx = 18
} = {}) {
    const projectedContentWidth = Math.max(0, Number(contentBounds?.width) || 0);
    const projectedContentHeight = Math.max(0, Number(contentBounds?.height) || 0);
    return {
        node,
        lines,
        contentBounds,
        captionPosition,
        scaleWithNode,
        minScreenHeightPx,
        projectedContentHeight,
        projectedCaptionExtent: resolveCaptionScreenExtent({
            captionPosition,
            projectedContentWidth,
            projectedContentHeight,
            scaleWithNode
        })
    };
}

function resolveMhsaUniformCaptionState(candidates = []) {
    if (!Array.isArray(candidates) || !candidates.length) return null;
    const projectedCaptionExtents = candidates
        .map((candidate) => Number(candidate?.projectedCaptionExtent) || 0)
        .filter((extent) => extent > 0)
        .sort((left, right) => left - right);
    const medianProjectedCaptionExtent = projectedCaptionExtents.length
        ? (
            projectedCaptionExtents.length % 2 === 1
                ? projectedCaptionExtents[(projectedCaptionExtents.length - 1) / 2]
                : (
                    projectedCaptionExtents[(projectedCaptionExtents.length / 2) - 1]
                    + projectedCaptionExtents[projectedCaptionExtents.length / 2]
                ) / 2
        )
        : 0;
    const minProjectedCaptionExtent = candidates.reduce((minExtent, candidate) => (
        Math.min(minExtent, Number(candidate?.projectedCaptionExtent) || 0)
    ), Number.POSITIVE_INFINITY);
    const minScreenHeightPx = Math.max(
        MHSA_UNIFORM_CAPTION_MIN_SCREEN_HEIGHT_PX,
        ...candidates.map((candidate) => Number(candidate?.minScreenHeightPx) || 0)
    );
    return {
        minScreenHeightPx,
        visible: true,
        referenceProjectedCaptionExtent: Math.max(
            minScreenHeightPx,
            medianProjectedCaptionExtent || minProjectedCaptionExtent
        ),
        sizeProgress: clamp(
            (minProjectedCaptionExtent - minScreenHeightPx) / RESIDUAL_CAPTION_FONT_RAMP_PX,
            0,
            1
        )
    };
}

function isDomKatexTextNode(node = null) {
    return (
        node?.kind === VIEW2D_NODE_KINDS.TEXT
        || node?.kind === VIEW2D_NODE_KINDS.OPERATOR
    )
        && String(node?.metadata?.renderMode || '').trim().toLowerCase() === 'dom-katex';
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

function createTextItem(documentRef) {
    const itemEl = documentRef.createElement('div');
    itemEl.className = 'detail-transformer-view2d-dom-text';
    const labelEl = documentRef.createElement('div');
    labelEl.className = 'detail-transformer-view2d-dom-text-line';
    itemEl.append(labelEl);
    return {
        itemEl,
        labelEl
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
        item.itemEl.dataset.nodeId = safeNodeId;
        root.appendChild(item.itemEl);
        itemMap.set(safeNodeId, item);
        return item;
    }

    function ensureTextItem(nodeId = '') {
        const safeNodeId = String(nodeId || '');
        if (!safeNodeId.length) return null;
        if (itemMap.has(safeNodeId)) {
            return itemMap.get(safeNodeId);
        }
        const item = createTextItem(documentRef);
        item.itemEl.dataset.nodeId = safeNodeId;
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
            enabled = true,
            focusState = null
        } = {}) {
            if (!visible || !enabled || !scene || !layout?.registry || !canvas || typeof projectBounds !== 'function') {
                hide();
                return false;
            }

            const canvasWidth = Math.max(1, canvas.clientWidth || canvas.width || 1);
            const canvasHeight = Math.max(1, canvas.clientHeight || canvas.height || 1);
            const fixedTextSizing = resolveMhsaDetailFixedTextSizing(scene, canvasWidth);
            const textZoomPolicy = resolveView2dSceneTextZoomPolicy(scene);
            const useMhsaDetailSceneRelativeCaptionBoost = textZoomPolicy.useUniformMatrixCaptions
                && isSceneRelativeTextZoomBehavior(textZoomPolicy.captionBehavior);
            const useMhsaDetailSceneRelativeDomTextBoost = textZoomPolicy.useUniformMatrixCaptions
                && isSceneRelativeTextZoomBehavior(textZoomPolicy.domTextBehavior);
            const overlayFocusState = normalizeSceneFocusState(focusState, {
                inactiveOpacity: OVERLAY_FOCUS_INACTIVE_OPACITY
            });
            root.style.display = 'block';
            root.style.left = `${Math.round(canvas.offsetLeft || 0)}px`;
            root.style.top = `${Math.round(canvas.offsetTop || 0)}px`;
            root.style.width = `${Math.round(canvasWidth)}px`;
            root.style.height = `${Math.round(canvasHeight)}px`;

            const seenIds = new Set();
            const captionCandidates = [];
            flattenSceneNodes(scene).forEach((node) => {
                if (isResidualCaptionNode(node)) {
                    const entry = layout.registry.getNodeEntry(node.id);
                    if (!entry?.contentBounds) return;

                    const lines = resolveView2dCaptionLines(node);
                    if (!lines.length) return;

                    const contentBounds = projectBounds(entry.contentBounds);
                    if (!contentBounds) return;
                    if (!intersectsCanvas(contentBounds, canvasWidth, canvasHeight)) return;
                    const captionPosition = resolveView2dCaptionPosition(node);
                    const scaleWithNode = node?.metadata?.caption?.scaleWithNode === true;
                    const minScreenHeightPx = Number.isFinite(node?.metadata?.caption?.minScreenHeightPx)
                        && node.metadata.caption.minScreenHeightPx > 0
                        ? node.metadata.caption.minScreenHeightPx
                        : 18;
                    captionCandidates.push(buildCaptionCandidate({
                        node,
                        lines,
                        contentBounds,
                        captionPosition,
                        scaleWithNode,
                        minScreenHeightPx
                    }));
                    return;
                }

                if (!isDomKatexTextNode(node)) return;
                const entry = layout.registry.getNodeEntry(node.id);
                const entryBounds = entry?.contentBounds || entry?.bounds;
                if (!entryBounds) return;
                const projectedBounds = projectBounds(entryBounds);
                if (!projectedBounds) return;

                const isOperatorNode = node.kind === VIEW2D_NODE_KINDS.OPERATOR;
                const operatorMinScreenHeightPx = isOperatorNode
                    && Number.isFinite(fixedTextSizing?.operatorMinScreenHeightPx)
                    ? Math.max(0, Number(fixedTextSizing.operatorMinScreenHeightPx))
                    : null;
                const minScreenHeightPx = operatorMinScreenHeightPx
                    ?? (
                        Number.isFinite(node?.metadata?.minScreenHeightPx)
                            ? Math.max(0, Number(node.metadata.minScreenHeightPx))
                            : 10
                    );
                const projectedHeight = Math.max(0, Number(projectedBounds.height) || 0);
                if (projectedHeight < minScreenHeightPx) return;
                if (!intersectsCanvas(projectedBounds, canvasWidth, canvasHeight)) return;

                const worldHeight = Math.max(1, Number(entryBounds.height) || 1);
                const screenScale = projectedHeight / worldHeight;
                const baseFontPx = Math.max(
                    1,
                    (Number(entry?.layoutData?.fontSize) || 12) * screenScale
                );
                const baseLayoutFontPx = Math.max(1, Number(entry?.layoutData?.fontSize) || 12);
                const fixedScreenFontPx = Number.isFinite(node?.metadata?.fixedScreenFontPx)
                    && node.metadata.fixedScreenFontPx > 0
                    ? Number(node.metadata.fixedScreenFontPx)
                    : null;
                const persistentMinScreenFontPx = Number.isFinite(node?.metadata?.persistentMinScreenFontPx)
                    && node.metadata.persistentMinScreenFontPx > 0
                    ? Number(node.metadata.persistentMinScreenFontPx)
                    : null;
                const operatorMinScreenFontPx = isOperatorNode
                    && Number.isFinite(fixedTextSizing?.operatorMinScreenFontPx)
                    && fixedTextSizing.operatorMinScreenFontPx > 0
                    ? Number(fixedTextSizing.operatorMinScreenFontPx)
                    : persistentMinScreenFontPx;
                const zoomedOutMinScreenFontPx = Number.isFinite(node?.metadata?.zoomedOutMinScreenFontPx)
                    && node.metadata.zoomedOutMinScreenFontPx > 0
                    ? Number(node.metadata.zoomedOutMinScreenFontPx)
                    : null;
                const domTextZoomBehavior = isOperatorNode
                    ? textZoomPolicy.operatorBehavior
                    : textZoomPolicy.domTextBehavior;
                const useSceneRelativeOperatorSizing = isOperatorNode
                    && isSceneRelativeTextZoomBehavior(domTextZoomBehavior);
                const domTextBaseFontPx = isOperatorNode
                    ? (
                        useSceneRelativeOperatorSizing
                            ? baseFontPx
                            : Math.max(baseLayoutFontPx, operatorMinScreenFontPx || 0)
                    )
                    : (
                        isSceneRelativeTextZoomBehavior(textZoomPolicy.domTextBehavior)
                            ? baseFontPx
                            : (
                                persistentMinScreenFontPx
                                    ? Math.max(baseFontPx, persistentMinScreenFontPx)
                                    : baseFontPx
                            )
                    );
                const resolvedDomTextFontPx = resolveOverlayFontPx({
                    zoomBehavior: domTextZoomBehavior,
                    baseFontPx: domTextBaseFontPx * (
                        !isOperatorNode && useMhsaDetailSceneRelativeDomTextBoost
                            ? MHSA_DETAIL_SCENE_RELATIVE_DOM_TEXT_BOOST
                            : 1
                    ),
                    projectedExtent: projectedHeight,
                    minScreenHeightPx,
                    fixedScreenFontPx: isOperatorNode
                        ? (
                            useSceneRelativeOperatorSizing
                                ? null
                                : domTextBaseFontPx
                        )
                        : (fixedTextSizing?.textScreenFontPx ?? fixedScreenFontPx),
                    minScreenFontPx: isOperatorNode
                        ? (operatorMinScreenFontPx ?? 8)
                        : (
                            zoomedOutMinScreenFontPx
                        ?? (persistentMinScreenFontPx
                            ? Math.max(8, persistentMinScreenFontPx * 0.84)
                            : 8)
                        )
                });
                const item = ensureTextItem(node.id);
                if (!item) return;
                seenIds.add(node.id);
                item.itemEl.hidden = false;
                item.itemEl.dataset.nodeKind = isOperatorNode ? 'operator' : 'text';
                if (isOperatorNode) {
                    item.itemEl.dataset.operatorKey = String(node?.semantic?.operatorKey || '').trim().toLowerCase();
                } else {
                    delete item.itemEl.dataset.operatorKey;
                }
                item.itemEl.style.left = `${Math.round(projectedBounds.x + (projectedBounds.width / 2))}px`;
                item.itemEl.style.top = `${Math.round(projectedBounds.y + (projectedBounds.height / 2))}px`;
                item.itemEl.style.setProperty('--detail-transformer-view2d-dom-text-size', `${resolvedDomTextFontPx.toFixed(2)}px`);
                item.itemEl.style.opacity = String(
                    resolveSceneNodeFocusAlpha(node.id, overlayFocusState) * resolveNodeVisualOpacity(node)
                );
                renderKatex(item.labelEl, node.tex, node.text);
            });

            const uniformMhsaCaptionState = textZoomPolicy.useUniformMatrixCaptions
                ? resolveMhsaUniformCaptionState(captionCandidates)
                : null;
            const useSceneRelativeCaptionSizing = isSceneRelativeTextZoomBehavior(textZoomPolicy.captionBehavior);
            captionCandidates.forEach((candidate) => {
                const {
                    node,
                    lines,
                    contentBounds,
                    captionPosition,
                    scaleWithNode,
                    minScreenHeightPx,
                    projectedContentHeight,
                    projectedCaptionExtent
                } = candidate;
                const activeUniformCaptionState = uniformMhsaCaptionState;
                const shouldShow = activeUniformCaptionState
                    ? activeUniformCaptionState.visible
                    : projectedCaptionExtent >= minScreenHeightPx;
                if (!shouldShow) return;

                const item = ensureItem(node.id);
                if (!item) return;
                seenIds.add(node.id);
                item.itemEl.dataset.nodeRole = String(node.role || '');
                const sizeProgress = activeUniformCaptionState
                    ? activeUniformCaptionState.sizeProgress
                    : clamp(
                        (projectedCaptionExtent - minScreenHeightPx) / RESIDUAL_CAPTION_FONT_RAMP_PX,
                        0,
                        1
                    );
                const preferStandardSizing = node?.metadata?.caption?.preferStandardSizing === true;
                const useMatrixRelativeSizing = useSceneRelativeCaptionSizing
                    ? true
                    : (
                        uniformMhsaCaptionState
                            ? !preferStandardSizing
                            : (captionPosition !== 'bottom' || scaleWithNode)
                    );
                const labelScale = Number.isFinite(node?.metadata?.caption?.labelScale)
                    && node.metadata.caption.labelScale > 0
                    ? Number(node.metadata.caption.labelScale)
                    : 1;
                const dimensionsScale = Number.isFinite(node?.metadata?.caption?.dimensionsScale)
                    && node.metadata.caption.dimensionsScale > 0
                    ? Number(node.metadata.caption.dimensionsScale)
                    : 1;
                const useUniformMatrixCaptionSizing = useMhsaDetailSceneRelativeCaptionBoost
                    && !!activeUniformCaptionState;
                const effectiveLabelScale = (useUniformMatrixCaptionSizing ? 1 : labelScale) * (
                    useMhsaDetailSceneRelativeCaptionBoost
                        ? MHSA_DETAIL_SCENE_RELATIVE_CAPTION_LABEL_BOOST
                        : 1
                );
                const effectiveDimensionsScale = (useUniformMatrixCaptionSizing ? 1 : dimensionsScale) * (
                    useMhsaDetailSceneRelativeCaptionBoost
                        ? MHSA_DETAIL_SCENE_RELATIVE_CAPTION_DIMENSIONS_BOOST
                        : 1
                );
                const captionSizingExtent = activeUniformCaptionState
                    ? Math.min(
                        activeUniformCaptionState.referenceProjectedCaptionExtent,
                        projectedCaptionExtent
                    )
                    : projectedContentHeight;
                const dimensionsCaptionSizingExtent = captionSizingExtent;
                const labelFontPx = fixedTextSizing?.captionLabelScreenFontPx
                    ?? resolveCaptionFontPx({
                        useMatrixRelativeSizing,
                        projectedContentHeight: captionSizingExtent,
                        sizeProgress,
                        minFontPx: activeUniformCaptionState
                            ? MHSA_UNIFORM_CAPTION_LABEL_MIN_FONT_PX
                            : RESIDUAL_CAPTION_LABEL_MIN_FONT_PX,
                        maxFontPx: activeUniformCaptionState
                            ? MHSA_UNIFORM_CAPTION_LABEL_MAX_FONT_PX
                            : RESIDUAL_CAPTION_LABEL_MAX_FONT_PX,
                        heightRatio: RESIDUAL_TOP_CAPTION_LABEL_RATIO,
                        scale: effectiveLabelScale,
                        enforceMinFontPx: !useSceneRelativeCaptionSizing
                    });
                const dimensionsFontPx = fixedTextSizing?.captionDimensionsScreenFontPx
                    ?? resolveCaptionFontPx({
                        useMatrixRelativeSizing,
                        projectedContentHeight: dimensionsCaptionSizingExtent,
                        sizeProgress,
                        minFontPx: activeUniformCaptionState
                            ? MHSA_UNIFORM_CAPTION_DIMENSIONS_MIN_FONT_PX
                            : RESIDUAL_CAPTION_DIMENSIONS_MIN_FONT_PX,
                        maxFontPx: activeUniformCaptionState
                            ? MHSA_UNIFORM_CAPTION_DIMENSIONS_MAX_FONT_PX
                            : RESIDUAL_CAPTION_DIMENSIONS_MAX_FONT_PX,
                        heightRatio: RESIDUAL_TOP_CAPTION_DIMENSIONS_RATIO,
                        scale: effectiveDimensionsScale,
                        enforceMinFontPx: !useSceneRelativeCaptionSizing
                    });
                const labelMinScreenFontPx = Number.isFinite(node?.metadata?.caption?.labelMinScreenFontPx)
                    && node.metadata.caption.labelMinScreenFontPx > 0
                    ? Number(node.metadata.caption.labelMinScreenFontPx)
                    : null;
                const labelMaxScreenFontPx = Number.isFinite(node?.metadata?.caption?.labelMaxScreenFontPx)
                    && node.metadata.caption.labelMaxScreenFontPx > 0
                    ? Number(node.metadata.caption.labelMaxScreenFontPx)
                    : null;
                const dimensionsMinScreenFontPx = Number.isFinite(node?.metadata?.caption?.dimensionsMinScreenFontPx)
                    && node.metadata.caption.dimensionsMinScreenFontPx > 0
                    ? Number(node.metadata.caption.dimensionsMinScreenFontPx)
                    : null;
                const captionFixedScreenFontPx = Number.isFinite(node?.metadata?.caption?.fixedScreenFontPx)
                    && node.metadata.caption.fixedScreenFontPx > 0
                    ? Number(node.metadata.caption.fixedScreenFontPx)
                    : null;
                const captionZoomBehavior = Number.isFinite(captionFixedScreenFontPx)
                    ? VIEW2D_TEXT_ZOOM_BEHAVIORS.SCREEN_FIXED
                    : textZoomPolicy.captionBehavior;
                const resolvedLabelFontPx = resolveOverlayFontPx({
                    zoomBehavior: captionZoomBehavior,
                    baseFontPx: labelFontPx,
                    projectedExtent: projectedCaptionExtent,
                    minScreenHeightPx,
                    fixedScreenFontPx: captionFixedScreenFontPx ?? fixedTextSizing?.captionLabelScreenFontPx,
                    minScreenFontPx: labelMinScreenFontPx
                        ?? (activeUniformCaptionState ? 11.5 : 10.5),
                    maxScreenFontPx: labelMaxScreenFontPx
                });
                const resolvedDimensionsFontPx = resolveOverlayFontPx({
                    zoomBehavior: captionZoomBehavior,
                    baseFontPx: dimensionsFontPx,
                    projectedExtent: projectedCaptionExtent,
                    minScreenHeightPx,
                    fixedScreenFontPx: captionFixedScreenFontPx ?? fixedTextSizing?.captionDimensionsScreenFontPx,
                    minScreenFontPx: dimensionsMinScreenFontPx
                        ?? (activeUniformCaptionState ? 10.5 : 9.5)
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
                applyCaptionRoleStyling(item.itemEl, node);
                item.itemEl.style.setProperty('--detail-transformer-view2d-caption-label-size', `${resolvedLabelFontPx.toFixed(2)}px`);
                item.itemEl.style.setProperty('--detail-transformer-view2d-caption-dimensions-size', `${resolvedDimensionsFontPx.toFixed(2)}px`);
                item.itemEl.style.opacity = String(
                    resolveSceneNodeFocusAlpha(node.id, overlayFocusState) * resolveNodeVisualOpacity(node)
                );

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
