import { buildSceneLayout } from '../view2d/layout/buildSceneLayout.js';
import { resolveSemanticTargetBounds } from '../view2d/layout/resolveSemanticTargetBounds.js';
import {
    createMhsaDetailSceneIndex,
    resolveMhsaDetailHoverState
} from '../view2d/mhsaDetailInteraction.js';
import { buildTransformerSceneModel } from '../view2d/model/buildTransformerSceneModel.js';
import { CanvasSceneRenderer } from '../view2d/render/canvas/CanvasSceneRenderer.js';
import {
    normalizeRaycastLabel,
    simplifyLayerNormParamHoverLabel
} from '../engine/coreRaycastLabels.js';
import {
    buildResidualRowHoverPayload,
    buildSemanticNodeHoverPayload,
    buildSemanticTarget,
    deriveBaseSemanticTarget,
    describeTransformerView2dTarget,
    hasActiveDetailTarget as hasActiveDetailTargetState,
    isConcatOverviewEntry,
    isMhsaHeadOverviewEntry,
    isOutputProjectionOverviewEntry,
    resolveActiveFocusLabel,
    resolveActiveSemanticTarget,
    resolveConcatDetailTarget,
    resolveDetailTargetsFromSemanticTarget,
    resolveFocusSemanticTargets,
    resolveHeadDetailTarget,
    resolveOutputProjectionDetailTarget,
    resolveTransformerView2dActionContext
} from '../view2d/transformerView2dTargets.js';
import {
    resolveTransformerView2dRoute,
    syncTransformerView2dRoute
} from '../view2d/transformerView2dRoute.js';
import { View2dViewportController, resolveViewportFitTransform } from '../view2d/runtime/View2dViewportController.js';
import { createHoverLabelOverlay } from './hoverLabelOverlay.js';
import { createTransformerView2dResidualCaptionOverlay } from './transformerView2dResidualCaptionOverlay.js';
import { buildSelectionPromptContext } from './selectionPanelPromptContextUtils.js';
import { applyTokenChipColors } from './tokenChipColorUtils.js';

export const TRANSFORMER_VIEW2D_PANEL_ACTION_OPEN = 'open-transformer-view2d';
export {
    describeTransformerView2dTarget,
    resolveTransformerView2dActionContext,
    resolveTransformerView2dRoute,
    syncTransformerView2dRoute
};

const VIEW2D_DETAIL_ACTION_FOCUS = 'focus-selection';
const VIEW2D_DETAIL_ACTION_FIT = 'fit-scene';
const VIEW2D_DETAIL_ACTION_EXIT_DEEP = 'exit-deep-detail';
const VIEW2D_INTERACTION_SETTLE_MS = 140;
const VIEW2D_PREVIEW_DPR_CAP_IDLE = 1.5;
const VIEW2D_PREVIEW_DPR_CAP_INTERACTING = 1;
const VIEW2D_CLICK_SLOP_PX = 6;
const VIEW2D_KEYBOARD_PAN_PX_PER_SEC = 620;
const VIEW2D_KEYBOARD_ZOOM_RATE = 0.00165;
const VIEW2D_KEYBOARD_INITIAL_STEP_MS = 16;
const VIEW2D_ROW_HOVER_FADE_DURATION_MS = 180;
const VIEW2D_DETAIL_VIEWPORT_PADDING = 28;
const VIEW2D_DETAIL_VIEWPORT_MIN_SCALE = 0.035;
const VIEW2D_HEAD_DETAIL_VIEWPORT_MIN_SCALE = 0.06;
const VIEW2D_DETAIL_VIEWPORT_MAX_SCALE = 10;
const VIEW2D_HEAD_DETAIL_FRAME_PAD_PX = 5;
const VIEW2D_HEAD_DETAIL_FOCUS_PADDING = Object.freeze({
    top: 6,
    right: 6,
    bottom: 6,
    left: 6
});
const VIEW2D_HEAD_DETAIL_DEPTH_ENTER_RATIO = 0.97;
const VIEW2D_HEAD_DETAIL_DEPTH_EXIT_RATIO = 0.93;

function normalizeView2dTokenChipIndex(value = null) {
    return Number.isFinite(value) ? Math.max(0, Math.floor(value)) : null;
}

function resolveView2dSelectedTokenContext({
    activationSource = null,
    tokenIndices = null,
    tokenLabels = null,
    semanticTarget = null
} = {}) {
    const selectedTokenIndex = normalizeView2dTokenChipIndex(semanticTarget?.tokenIndex);
    if (!Number.isFinite(selectedTokenIndex)) {
        return {
            tokenIndex: null,
            tokenId: null,
            tokenText: ''
        };
    }

    const indices = Array.isArray(tokenIndices) ? tokenIndices : [];
    const labels = Array.isArray(tokenLabels) ? tokenLabels : [];
    const matchingLaneIndex = indices.findIndex((candidate) => (
        Number.isFinite(candidate) && Math.floor(candidate) === selectedTokenIndex
    ));

    let tokenText = '';
    if (
        matchingLaneIndex >= 0
        && matchingLaneIndex < labels.length
        && typeof labels[matchingLaneIndex] === 'string'
    ) {
        tokenText = labels[matchingLaneIndex];
    }
    if (!tokenText && typeof activationSource?.getTokenRawString === 'function') {
        const rawTokenText = activationSource.getTokenRawString(selectedTokenIndex);
        if (typeof rawTokenText === 'string') tokenText = rawTokenText;
    }
    if (!tokenText && typeof activationSource?.getTokenString === 'function') {
        const resolvedTokenText = activationSource.getTokenString(selectedTokenIndex);
        if (typeof resolvedTokenText === 'string') tokenText = resolvedTokenText;
    }

    const selectedTokenId = typeof activationSource?.getTokenId === 'function'
        ? normalizeView2dTokenChipIndex(activationSource.getTokenId(selectedTokenIndex))
        : null;

    return {
        tokenIndex: selectedTokenIndex,
        tokenId: selectedTokenId,
        tokenText
    };
}

function configureView2dTokenNavChip(chip, {
    tokenText = '',
    tokenIndex = null,
    tokenId = null
} = {}) {
    if (!chip) return;
    const safeTokenText = String(tokenText ?? '');
    const safeTokenIndex = normalizeView2dTokenChipIndex(tokenIndex);
    const safeTokenId = normalizeView2dTokenChipIndex(tokenId);

    chip.classList.add('detail-token-nav-chip');
    chip.dataset.tokenText = safeTokenText;
    if (Number.isFinite(safeTokenIndex)) {
        chip.dataset.tokenIndex = String(safeTokenIndex);
    } else {
        delete chip.dataset.tokenIndex;
    }
    if (Number.isFinite(safeTokenId)) {
        chip.dataset.tokenId = String(safeTokenId);
    } else {
        delete chip.dataset.tokenId;
    }
    chip.dataset.tokenNav = safeTokenText.length > 0 ? 'true' : 'false';

    if (safeTokenText.length > 0) {
        chip.tabIndex = 0;
        chip.setAttribute('role', 'button');
        chip.setAttribute('aria-label', `Open token details for ${safeTokenText}`);
    } else {
        chip.removeAttribute('tabindex');
        chip.removeAttribute('role');
        chip.removeAttribute('aria-label');
    }
}

function isTextEntryTarget(target) {
    if (!(target instanceof Element)) return false;
    const tagName = String(target.tagName || '').toLowerCase();
    if (tagName === 'input' || tagName === 'textarea' || tagName === 'select') return true;
    return target.isContentEditable === true;
}

function normalizeView2dKeyboardControlKey(key = '') {
    const lower = String(key || '').toLowerCase();
    if (lower === 'arrowleft' || lower === 'a') return 'pan-left';
    if (lower === 'arrowright' || lower === 'd') return 'pan-right';
    if (lower === 'arrowup' || lower === 'w') return 'pan-up';
    if (lower === 'arrowdown' || lower === 's') return 'pan-down';
    if (key === '+' || key === '=' || lower === 'add' || lower === 'numpadadd') return 'zoom-in';
    if (key === '-' || key === '_' || lower === 'subtract' || lower === 'numpadsubtract') return 'zoom-out';
    return '';
}

export function setDescriptionTransformerView2dAction(descriptionEl, {
    enabled = false,
    label = 'Open 2D canvas'
} = {}) {
    if (!descriptionEl) return;

    descriptionEl
        .querySelectorAll(`[data-detail-action="${TRANSFORMER_VIEW2D_PANEL_ACTION_OPEN}"]`)
        .forEach((node) => {
            const row = node.closest('.detail-description-action-row');
            if (row) {
                row.remove();
                return;
            }
            node.remove();
        });

    if (!enabled) return;

    const actionRow = document.createElement('div');
    actionRow.className = 'detail-description-action-row';

    const actionBtn = document.createElement('button');
    actionBtn.type = 'button';
    actionBtn.className = 'detail-description-action-link';
    actionBtn.dataset.detailAction = TRANSFORMER_VIEW2D_PANEL_ACTION_OPEN;
    actionBtn.textContent = label;
    actionBtn.setAttribute('aria-label', 'Open the scalable 2D transformer canvas');

    actionRow.appendChild(actionBtn);
    descriptionEl.appendChild(actionRow);
}

export function createTransformerView2dDetailView(panelEl) {
    if (!panelEl || typeof document === 'undefined') return null;

    const root = document.createElement('section');
    root.className = 'detail-transformer-view2d';
    root.setAttribute('aria-hidden', 'true');
    root.innerHTML = `
        <div class="detail-transformer-view2d-copy">
            <p>Prototype canvas for the new scalable 2D transformer view.</p>
            <p>It uses the semantic scene model and canvas renderer instead of the current DOM-heavy matrix layout.</p>
        </div>
        <div class="detail-transformer-view2d-stage">
            <div class="detail-transformer-view2d-hud">
                <div class="detail-transformer-view2d-toolbar">
                    <div class="detail-transformer-view2d-toolbar-copy">
                        <div class="detail-transformer-view2d-toolbar-title">2D canvas prototype</div>
                        <div class="detail-transformer-view2d-hint">Drag or use one finger to pan. Scroll, pinch, or use +/- to zoom. Use arrows or WASD to move. Use Focus selection to return to the current component. Use Back to graph to leave the deepest head view.</div>
                    </div>
                    <div class="detail-transformer-view2d-toolbar-actions">
                        <button
                            type="button"
                            class="detail-transformer-view2d-action"
                            data-transformer-view2d-action="${VIEW2D_DETAIL_ACTION_FOCUS}"
                        >
                            Focus selection
                        </button>
                        <button
                            type="button"
                            class="detail-transformer-view2d-action"
                            data-transformer-view2d-action="${VIEW2D_DETAIL_ACTION_FIT}"
                        >
                            Fit scene
                        </button>
                    </div>
                </div>
                <div class="detail-transformer-view2d-readout" aria-live="polite">
                    <div class="detail-transformer-view2d-stat">
                        <span class="detail-transformer-view2d-stat-label">Focus</span>
                        <span class="detail-transformer-view2d-stat-value" data-transformer-view2d-readout="focus">Transformer overview</span>
                    </div>
                    <div class="detail-transformer-view2d-stat">
                        <span class="detail-transformer-view2d-stat-label">Zoom</span>
                        <span class="detail-transformer-view2d-stat-value" data-transformer-view2d-readout="zoom">100%</span>
                    </div>
                    <div class="detail-transformer-view2d-stat">
                        <span class="detail-transformer-view2d-stat-label">Scene</span>
                        <span class="detail-transformer-view2d-stat-value" data-transformer-view2d-readout="scene">0 layers / 0 tokens</span>
                    </div>
                </div>
            </div>
            <div
                class="detail-transformer-view2d-canvas-card"
                tabindex="0"
                role="group"
                aria-label="Scalable 2D transformer canvas. Drag or use one finger to pan, pinch or scroll to zoom, and use arrow keys or W A S D to move around."
            >
                <div class="detail-transformer-view2d-deep-controls" aria-hidden="true" hidden>
                    <button
                        type="button"
                        class="detail-transformer-view2d-action detail-transformer-view2d-action--deep-exit"
                        data-transformer-view2d-action="${VIEW2D_DETAIL_ACTION_EXIT_DEEP}"
                    >
                        Back to graph
                    </button>
                </div>
                <div class="detail-transformer-view2d-detail-frame" aria-hidden="true"></div>
                <canvas class="detail-transformer-view2d-canvas" aria-label="Scalable 2D transformer canvas"></canvas>
                <div
                    class="detail-transformer-view2d-token-strip"
                    data-visible="false"
                    aria-hidden="true"
                >
                    <div
                        class="detail-transformer-view2d-token-strip__tokens"
                        data-transformer-view2d-role="token-strip-tokens"
                    ></div>
                </div>
            </div>
        </div>
    `;

    const header = panelEl.querySelector('.detail-header');
    if (header && header.nextSibling) {
        panelEl.insertBefore(root, header.nextSibling);
    } else {
        panelEl.appendChild(root);
    }

    const canvas = root.querySelector('.detail-transformer-view2d-canvas');
    const canvasCard = root.querySelector('.detail-transformer-view2d-canvas-card');
    const deepControls = root.querySelector('.detail-transformer-view2d-deep-controls');
    const deepExitBtn = root.querySelector(`[data-transformer-view2d-action="${VIEW2D_DETAIL_ACTION_EXIT_DEEP}"]`);
    const detailFrame = root.querySelector('.detail-transformer-view2d-detail-frame');
    const tokenStrip = root.querySelector('.detail-transformer-view2d-token-strip');
    const tokenStripTokens = root.querySelector('[data-transformer-view2d-role="token-strip-tokens"]');
    const hud = root.querySelector('.detail-transformer-view2d-hud');
    const focusBtn = root.querySelector(`[data-transformer-view2d-action="${VIEW2D_DETAIL_ACTION_FOCUS}"]`);
    const fitBtn = root.querySelector(`[data-transformer-view2d-action="${VIEW2D_DETAIL_ACTION_FIT}"]`);
    const focusReadout = root.querySelector('[data-transformer-view2d-readout="focus"]');
    const zoomReadout = root.querySelector('[data-transformer-view2d-readout="zoom"]');
    const sceneReadout = root.querySelector('[data-transformer-view2d-readout="scene"]');
    const hoverLabelOverlay = createHoverLabelOverlay({
        zIndex: 12
    });
    const residualCaptionOverlay = createTransformerView2dResidualCaptionOverlay({
        parent: canvasCard
    });

    const renderer = new CanvasSceneRenderer({
        canvas,
        dprCap: VIEW2D_PREVIEW_DPR_CAP_IDLE
    });
    const viewportController = new View2dViewportController({
        minScale: VIEW2D_DETAIL_VIEWPORT_MIN_SCALE,
        maxScale: VIEW2D_DETAIL_VIEWPORT_MAX_SCALE,
        padding: VIEW2D_DETAIL_VIEWPORT_PADDING
    });
    const detailViewportController = new View2dViewportController({
        minScale: VIEW2D_HEAD_DETAIL_VIEWPORT_MIN_SCALE,
        maxScale: VIEW2D_DETAIL_VIEWPORT_MAX_SCALE,
        padding: VIEW2D_DETAIL_VIEWPORT_PADDING
    });

    const state = {
        visible: false,
        scene: null,
        layout: null,
        activationSource: null,
        tokenIndices: null,
        tokenLabels: null,
        isSmallScreen: false,
        baseSemanticTarget: null,
        baseFocusLabel: 'Transformer overview',
        headDetailTarget: null,
        concatDetailTarget: null,
        outputProjectionDetailTarget: null,
        headDetailFocusScale: null,
        headDetailSceneFitScale: null,
        headDetailDepthActive: false,
        headDetailDepthAutoReentryBlocked: false,
        semanticTarget: null,
        focusLabel: 'Transformer overview',
        sceneLabel: '0 layers / 0 tokens',
        detailSceneIndex: null,
        detailSceneFocus: null,
        detailSceneHoverSignature: '',
        hoveredResidualRow: null,
        hoverDimming: {
            value: 0,
            target: 0,
            rafId: null,
            lastTime: 0,
            previousHoveredRow: null,
            rowBlend: 1,
            rowBlendTarget: 1
        },
        pointer: null,
        touchGesture: {
            pointers: new Map(),
            pinchActive: false,
            startDistance: 0,
            startScale: 1,
            anchorLocalX: 0,
            anchorLocalY: 0
        },
        animationFrame: null,
        renderFrame: null,
        interactionTimer: null,
        isInteracting: false,
        keyboardMotion: {
            activeKeys: new Set(),
            rafId: null,
            lastTime: 0,
            engaged: false
        },
        viewportSize: {
            width: 1,
            height: 1
        },
        pendingInitialFocus: false,
        autoFrameOnResize: false,
        lastAutoFrameViewportSize: null,
        tokenStripSignature: ''
    };

    function resetAutoFrameState({ pendingInitialFocus = false } = {}) {
        state.pendingInitialFocus = !!pendingInitialFocus;
        state.autoFrameOnResize = true;
        state.lastAutoFrameViewportSize = null;
    }

    function disableAutoFrameState() {
        state.pendingInitialFocus = false;
        state.autoFrameOnResize = false;
        state.lastAutoFrameViewportSize = null;
    }

    function trackAutoFrameViewportSize(width = 0, height = 0) {
        state.pendingInitialFocus = false;
        state.autoFrameOnResize = true;
        state.lastAutoFrameViewportSize = {
            width: Math.max(1, Math.floor(Number(width) || 0)),
            height: Math.max(1, Math.floor(Number(height) || 0))
        };
    }

    function shouldAutoFrameViewport(width = 0, height = 0) {
        const safeWidth = Math.max(1, Math.floor(Number(width) || 0));
        const safeHeight = Math.max(1, Math.floor(Number(height) || 0));
        if (state.pendingInitialFocus) {
            state.pendingInitialFocus = false;
            state.lastAutoFrameViewportSize = {
                width: safeWidth,
                height: safeHeight
            };
            return true;
        }
        if (!state.autoFrameOnResize) {
            return false;
        }
        const previousSize = state.lastAutoFrameViewportSize;
        const didViewportSizeChange = !previousSize
            || previousSize.width !== safeWidth
            || previousSize.height !== safeHeight;
        if (!didViewportSizeChange) {
            return false;
        }
        state.lastAutoFrameViewportSize = {
            width: safeWidth,
            height: safeHeight
        };
        return true;
    }

    function hasActiveDetailTarget() {
        return hasActiveDetailTargetState(state);
    }

    function setDetailTargets({
        headDetailTarget = null,
        concatDetailTarget = null,
        outputProjectionDetailTarget = null
    } = {}) {
        const resolvedOutputProjectionDetailTarget = resolveOutputProjectionDetailTarget(outputProjectionDetailTarget);
        const resolvedConcatDetailTarget = resolvedOutputProjectionDetailTarget
            ? null
            : resolveConcatDetailTarget(concatDetailTarget);
        const resolvedHeadDetailTarget = (resolvedOutputProjectionDetailTarget || resolvedConcatDetailTarget)
            ? null
            : resolveHeadDetailTarget(headDetailTarget);
        state.headDetailTarget = resolvedHeadDetailTarget;
        state.concatDetailTarget = resolvedConcatDetailTarget;
        state.outputProjectionDetailTarget = resolvedOutputProjectionDetailTarget;
    }

    function resetHeadDetailState(nextDepthActive = false) {
        state.headDetailFocusScale = null;
        state.headDetailSceneFitScale = null;
        state.headDetailDepthActive = !!nextDepthActive;
        state.headDetailDepthAutoReentryBlocked = false;
    }

    function syncActiveSelectionState() {
        applySemanticTarget(
            resolveActiveSemanticTarget(state),
            resolveActiveFocusLabel(state)
        );
    }

    function commitSceneSelection({
        animate = true,
        nextDepthActive = false
    } = {}) {
        resetHeadDetailState(nextDepthActive);
        syncActiveSelectionState();
        rebuildSceneState();
        if (!state.visible) return true;
        const { width, height } = measureCanvasSize();
        updateReadouts();
        render();
        focusSelection({ animate });
        trackAutoFrameViewportSize(width, height);
        return true;
    }

    function stopAnimation() {
        if (state.animationFrame !== null) {
            cancelAnimationFrame(state.animationFrame);
            state.animationFrame = null;
        }
        viewportController.animation = null;
        detailViewportController.animation = null;
    }

    function stopRenderLoop() {
        if (state.renderFrame === null) return;
        cancelAnimationFrame(state.renderFrame);
        state.renderFrame = null;
    }

    function measureCanvasSize() {
        const rect = typeof canvas?.getBoundingClientRect === 'function'
            ? canvas.getBoundingClientRect()
            : null;
        state.viewportSize = {
            width: Math.max(1, Math.floor(rect?.width || canvas?.clientWidth || canvas?.width || 1)),
            height: Math.max(1, Math.floor(rect?.height || canvas?.clientHeight || canvas?.height || 1))
        };
        return state.viewportSize;
    }

    function getCanvasSize() {
        const width = Number(state.viewportSize?.width) || 0;
        const height = Number(state.viewportSize?.height) || 0;
        if (width > 0 && height > 0) {
            return {
                width,
                height
            };
        }
        return measureCanvasSize();
    }

    function containsPoint(bounds = null, x = 0, y = 0) {
        if (!bounds) return false;
        const minX = Number.isFinite(bounds.x) ? bounds.x : 0;
        const minY = Number.isFinite(bounds.y) ? bounds.y : 0;
        const maxX = minX + Math.max(0, Number(bounds.width) || 0);
        const maxY = minY + Math.max(0, Number(bounds.height) || 0);
        return x >= minX && x <= maxX && y >= minY && y <= maxY;
    }

    function getActiveViewportController() {
        return state.headDetailDepthActive && state.headDetailTarget
            ? detailViewportController
            : viewportController;
    }

    function resolveHeadDetailSceneBounds() {
        const bounds = renderer.getHeadDetailSceneBounds();
        if (!bounds || !(bounds.width > 0) || !(bounds.height > 0)) {
            return null;
        }
        return bounds;
    }

    function syncHeadDetailViewport({
        forceFit = false,
        animate = false
    } = {}) {
        if (!state.headDetailTarget) return false;
        const bounds = resolveHeadDetailSceneBounds();
        if (!bounds) return false;
        const { width, height } = getCanvasSize();
        detailViewportController.setViewportSize(width, height);
        detailViewportController.setSceneBounds(bounds);
        const fitTransform = resolveViewportFitTransform(bounds, { width, height }, {
            padding: VIEW2D_DETAIL_VIEWPORT_PADDING,
            minScale: VIEW2D_HEAD_DETAIL_VIEWPORT_MIN_SCALE,
            maxScale: VIEW2D_DETAIL_VIEWPORT_MAX_SCALE
        });
        state.headDetailSceneFitScale = Number.isFinite(fitTransform?.scale) ? fitTransform.scale : null;
        if (!forceFit) {
            return true;
        }
        if (animate) {
            detailViewportController.flyToBounds(bounds, {
                animate: true,
                durationMs: 420,
                now: performance.now(),
                padding: VIEW2D_DETAIL_VIEWPORT_PADDING
            });
            animateViewport();
            return true;
        }
        detailViewportController.fitToBounds(bounds, {
            padding: VIEW2D_DETAIL_VIEWPORT_PADDING
        });
        return true;
    }

    function hideDetailFrame() {
        if (!detailFrame) return;
        detailFrame.classList.remove('is-visible');
        detailFrame.style.left = '';
        detailFrame.style.top = '';
        detailFrame.style.width = '';
        detailFrame.style.height = '';
        detailFrame.style.borderRadius = '';
    }

    function syncDetailFrame() {
        if (
            !detailFrame
            || !canvas
            || !canvasCard
            || !state.visible
            || !state.headDetailDepthActive
            || !state.headDetailTarget
        ) {
            hideDetailFrame();
            return;
        }

        const sceneBounds = resolveHeadDetailSceneBounds();
        const projectedBounds = renderer.resolveScreenBounds(sceneBounds);
        if (!sceneBounds || !projectedBounds) {
            hideDetailFrame();
            return;
        }

        const canvasRect = canvas.getBoundingClientRect();
        const cardRect = canvasCard.getBoundingClientRect();
        const cardWidth = Math.max(0, cardRect.width);
        const cardHeight = Math.max(0, cardRect.height);
        if (!(cardWidth > 0) || !(cardHeight > 0)) {
            hideDetailFrame();
            return;
        }

        const framePad = state.isSmallScreen
            ? Math.max(4, VIEW2D_HEAD_DETAIL_FRAME_PAD_PX - 1)
            : VIEW2D_HEAD_DETAIL_FRAME_PAD_PX;
        const rawLeft = (canvasRect.left - cardRect.left) + projectedBounds.x - framePad;
        const rawTop = (canvasRect.top - cardRect.top) + projectedBounds.y - framePad;
        const rawRight = rawLeft + projectedBounds.width + (framePad * 2);
        const rawBottom = rawTop + projectedBounds.height + (framePad * 2);
        const left = Math.max(0, Math.min(cardWidth, rawLeft));
        const top = Math.max(0, Math.min(cardHeight, rawTop));
        const right = Math.max(left, Math.min(cardWidth, rawRight));
        const bottom = Math.max(top, Math.min(cardHeight, rawBottom));
        const width = Math.max(0, right - left);
        const height = Math.max(0, bottom - top);
        if (!(width > 1) || !(height > 1)) {
            hideDetailFrame();
            return;
        }

        detailFrame.style.left = `${Math.round(left)}px`;
        detailFrame.style.top = `${Math.round(top)}px`;
        detailFrame.style.width = `${Math.round(width)}px`;
        detailFrame.style.height = `${Math.round(height)}px`;
        detailFrame.style.borderRadius = `${state.isSmallScreen ? 22 : 28}px`;
        detailFrame.classList.add('is-visible');
    }

    function resolveViewportWorldBounds(controller = viewportController) {
        const { width, height } = getCanvasSize();
        if (!(width > 0) || !(height > 0)) return null;
        const topLeft = controller.screenToWorld(0, 0);
        const bottomRight = controller.screenToWorld(width, height);
        const minX = Math.min(topLeft.x, bottomRight.x);
        const minY = Math.min(topLeft.y, bottomRight.y);
        const maxX = Math.max(topLeft.x, bottomRight.x);
        const maxY = Math.max(topLeft.y, bottomRight.y);
        return {
            x: minX,
            y: minY,
            width: Math.max(0, maxX - minX),
            height: Math.max(0, maxY - minY)
        };
    }

    function isViewportNearFocusBounds(bounds = null, controller = viewportController) {
        const viewportBounds = resolveViewportWorldBounds(controller);
        if (!bounds || !viewportBounds) return false;
        const centerX = viewportBounds.x + (viewportBounds.width * 0.5);
        const centerY = viewportBounds.y + (viewportBounds.height * 0.5);
        const marginX = Math.max(bounds.width * 0.45, viewportBounds.width * 0.1);
        const marginY = Math.max(bounds.height * 0.45, viewportBounds.height * 0.1);
        return containsPoint({
            x: bounds.x - marginX,
            y: bounds.y - marginY,
            width: bounds.width + (marginX * 2),
            height: bounds.height + (marginY * 2)
        }, centerX, centerY);
    }

    function stopHoverDimmingAnimation() {
        const hoverDimming = state.hoverDimming;
        if (!hoverDimming || hoverDimming.rafId === null) return;
        cancelAnimationFrame(hoverDimming.rafId);
        hoverDimming.rafId = null;
        hoverDimming.lastTime = 0;
    }

    function resetHoverRowBlend() {
        const hoverDimming = state.hoverDimming;
        if (!hoverDimming) return;
        hoverDimming.previousHoveredRow = null;
        hoverDimming.rowBlend = 1;
        hoverDimming.rowBlendTarget = 1;
    }

    function ensureHoverDimmingAnimation() {
        const hoverDimming = state.hoverDimming;
        if (!hoverDimming || hoverDimming.rafId !== null) return;
        const dimSettled = Math.abs(hoverDimming.target - hoverDimming.value) <= 0.01;
        const rowBlendSettled = Math.abs(hoverDimming.rowBlendTarget - hoverDimming.rowBlend) <= 0.01;
        if (dimSettled && rowBlendSettled) {
            hoverDimming.value = hoverDimming.target;
            hoverDimming.rowBlend = hoverDimming.rowBlendTarget;
            if (hoverDimming.rowBlendTarget >= 1) {
                hoverDimming.previousHoveredRow = null;
            }
            return;
        }

        const tick = (now) => {
            hoverDimming.rafId = null;
            if (!state.visible) {
                hoverDimming.value = hoverDimming.target;
                hoverDimming.rowBlend = hoverDimming.rowBlendTarget;
                if (hoverDimming.rowBlendTarget >= 1) {
                    hoverDimming.previousHoveredRow = null;
                }
                hoverDimming.lastTime = 0;
                return;
            }
            const previousTime = Number.isFinite(hoverDimming.lastTime) && hoverDimming.lastTime > 0
                ? hoverDimming.lastTime
                : now;
            const dt = Math.max(8, Math.min(32, now - previousTime));
            hoverDimming.lastTime = now;
            const alpha = 1 - Math.exp(-dt / VIEW2D_ROW_HOVER_FADE_DURATION_MS);
            hoverDimming.value += (hoverDimming.target - hoverDimming.value) * alpha;
            hoverDimming.rowBlend += (hoverDimming.rowBlendTarget - hoverDimming.rowBlend) * alpha;

            if (Math.abs(hoverDimming.target - hoverDimming.value) <= 0.01) {
                hoverDimming.value = hoverDimming.target;
            }
            if (Math.abs(hoverDimming.rowBlendTarget - hoverDimming.rowBlend) <= 0.01) {
                hoverDimming.rowBlend = hoverDimming.rowBlendTarget;
                if (hoverDimming.rowBlendTarget >= 1) {
                    hoverDimming.previousHoveredRow = null;
                }
            }

            const nextDimSettled = Math.abs(hoverDimming.target - hoverDimming.value) <= 0.01;
            const nextRowBlendSettled = Math.abs(hoverDimming.rowBlendTarget - hoverDimming.rowBlend) <= 0.01;
            render();
            if (nextDimSettled && nextRowBlendSettled) {
                hoverDimming.lastTime = 0;
                return;
            }
            hoverDimming.rafId = requestAnimationFrame(tick);
        };

        hoverDimming.lastTime = 0;
        hoverDimming.rafId = requestAnimationFrame(tick);
    }

    function setHoverDimmingTarget(target = 0, {
        immediate = false,
        shouldRender = true
    } = {}) {
        const hoverDimming = state.hoverDimming;
        if (!hoverDimming) return;
        const nextTarget = Math.max(0, Math.min(1, Number.isFinite(target) ? target : 0));
        hoverDimming.target = nextTarget;

        if (immediate) {
            stopHoverDimmingAnimation();
            hoverDimming.value = nextTarget;
            if (nextTarget <= 0.001) {
                resetHoverRowBlend();
            }
            if (shouldRender) scheduleRender();
            return;
        }

        if (
            Math.abs(hoverDimming.value - nextTarget) < 0.001
            && Math.abs(hoverDimming.rowBlendTarget - hoverDimming.rowBlend) < 0.001
        ) {
            hoverDimming.value = nextTarget;
            if (shouldRender) scheduleRender();
            return;
        }

        ensureHoverDimmingAnimation();
    }

    function setNodeText(node, nextText) {
        if (!node) return;
        const safeText = String(nextText ?? '');
        if (node.textContent !== safeText) {
            node.textContent = safeText;
        }
    }

    function normalizeCanvasHoverLabel(label = '', info = null) {
        const normalizedLabel = normalizeRaycastLabel(label, info);
        return simplifyLayerNormParamHoverLabel(normalizedLabel, info);
    }

    function clearCanvasHover({ scheduleRender: shouldScheduleRender = true } = {}) {
        const hadResidualHover = !!state.hoveredResidualRow;
        const hadDetailHover = !!state.detailSceneFocus;
        state.hoveredResidualRow = null;
        state.detailSceneFocus = null;
        state.detailSceneHoverSignature = '';
        hoverLabelOverlay.hide();
        resetHoverRowBlend();
        setHoverDimmingTarget(0, {
            immediate: !state.visible,
            shouldRender: shouldScheduleRender || hadResidualHover || hadDetailHover
        });
        if ((hadResidualHover || hadDetailHover) && shouldScheduleRender) {
            scheduleRender();
        }
    }

    function updateCanvasHover(event = null) {
        const allowHeadDetailHover = !!(
            state.headDetailDepthActive
            && state.headDetailTarget
            && state.detailSceneIndex
        );
        const suppressOverviewHover = !!(
            state.headDetailDepthActive
            && hasActiveDetailTarget()
            && !state.headDetailTarget
        );
        if (
            !state.visible
            || !canvas
            || !Number.isFinite(event?.clientX)
            || !Number.isFinite(event?.clientY)
            || event?.pointerType === 'touch'
        ) {
            clearCanvasHover();
            return null;
        }

        const rect = canvas.getBoundingClientRect();
        const localX = event.clientX - rect.left;
        const localY = event.clientY - rect.top;
        const hit = renderer.resolveInteractiveHitAtScreenPoint(localX, localY);
        if (suppressOverviewHover) {
            clearCanvasHover();
            return hit?.entry || null;
        }
        if (allowHeadDetailHover) {
            const detailHoverState = resolveMhsaDetailHoverState(state.detailSceneIndex, hit);
            const residualHoverPayload = buildResidualRowHoverPayload(hit?.rowHit, state.activationSource);
            if (!detailHoverState?.focusState) {
                clearCanvasHover();
                return hit?.entry || null;
            }

            const didChange = detailHoverState.signature !== state.detailSceneHoverSignature;
            state.hoveredResidualRow = null;
            resetHoverRowBlend();
            setHoverDimmingTarget(0, {
                immediate: true,
                shouldRender: false
            });
            state.detailSceneFocus = detailHoverState.focusState;
            state.detailSceneHoverSignature = detailHoverState.signature;
            const hoverInfo = residualHoverPayload?.info || detailHoverState.info || null;
            const hoverLabel = normalizeCanvasHoverLabel(
                residualHoverPayload?.label || detailHoverState.label || '',
                hoverInfo
            );
            if (hoverLabel) {
                hoverLabelOverlay.show({
                    clientX: event.clientX,
                    clientY: event.clientY,
                    label: hoverLabel,
                    info: hoverInfo,
                    activationSource: state.activationSource
                });
            } else {
                hoverLabelOverlay.hide();
            }
            if (didChange) {
                scheduleRender();
            }
            return hit?.entry || null;
        }

        const worldPoint = viewportController.screenToWorld(localX, localY);
        const worldHit = renderer.resolveInteractiveHitAtPoint(worldPoint.x, worldPoint.y);
        const residualHoverPayload = buildResidualRowHoverPayload(worldHit?.rowHit, state.activationSource);
        if (residualHoverPayload && worldHit?.node?.id && Number.isFinite(worldHit?.rowHit?.rowIndex)) {
            const hoverLabel = normalizeCanvasHoverLabel(
                residualHoverPayload.label,
                residualHoverPayload.info
            );
            if (!hoverLabel) {
                clearCanvasHover();
                return worldHit?.entry || hit?.entry || null;
            }

            const nextHoveredRow = {
                nodeId: worldHit.node.id,
                rowIndex: Math.max(0, Math.floor(worldHit.rowHit.rowIndex))
            };
            const prevHoveredRow = state.hoveredResidualRow;
            const didChange = !prevHoveredRow
                || prevHoveredRow.nodeId !== nextHoveredRow.nodeId
                || prevHoveredRow.rowIndex !== nextHoveredRow.rowIndex;
            if (didChange && prevHoveredRow) {
                state.hoverDimming.previousHoveredRow = {
                    nodeId: prevHoveredRow.nodeId,
                    rowIndex: prevHoveredRow.rowIndex
                };
                state.hoverDimming.rowBlend = 0;
                state.hoverDimming.rowBlendTarget = 1;
            } else if (!didChange) {
                resetHoverRowBlend();
            }
            state.hoveredResidualRow = nextHoveredRow;
            setHoverDimmingTarget(1, {
                shouldRender: didChange
            });
            hoverLabelOverlay.show({
                clientX: event.clientX,
                clientY: event.clientY,
                label: hoverLabel,
                info: residualHoverPayload.info,
                activationSource: state.activationSource
            });
            if (didChange) {
                scheduleRender();
            }
            return worldHit?.entry || null;
        }

        const semanticHoverPayload = buildSemanticNodeHoverPayload(worldHit);
        if (!semanticHoverPayload?.label) {
            clearCanvasHover();
            return worldHit?.entry || hit?.entry || null;
        }

        const hoverLabel = normalizeCanvasHoverLabel(
            semanticHoverPayload.label,
            semanticHoverPayload.info
        );
        if (!hoverLabel) {
            clearCanvasHover();
            return worldHit?.entry || hit?.entry || null;
        }

        const hadResidualHover = !!state.hoveredResidualRow;
        state.hoveredResidualRow = null;
        state.detailSceneFocus = null;
        state.detailSceneHoverSignature = '';
        resetHoverRowBlend();
        setHoverDimmingTarget(0, {
            immediate: true,
            shouldRender: false
        });
        hoverLabelOverlay.show({
            clientX: event.clientX,
            clientY: event.clientY,
            label: hoverLabel,
            info: semanticHoverPayload.info,
            activationSource: state.activationSource
        });
        if (hadResidualHover) {
            scheduleRender();
        }
        return worldHit?.entry || null;
    }

    function setInteractionActive(active = false) {
        const next = !!active;
        if (state.isInteracting === next) return;
        state.isInteracting = next;
        scheduleRender();
    }

    function clearInteractionTimer() {
        if (state.interactionTimer === null) return;
        clearTimeout(state.interactionTimer);
        state.interactionTimer = null;
    }

    function scheduleInteractionSettle() {
        clearInteractionTimer();
        state.interactionTimer = setTimeout(() => {
            state.interactionTimer = null;
            setInteractionActive(false);
        }, VIEW2D_INTERACTION_SETTLE_MS);
    }

    function markInteraction(activePointer = false) {
        clearInteractionTimer();
        setInteractionActive(true);
        if (!activePointer) {
            scheduleInteractionSettle();
        }
    }

    function focusCanvasSurface() {
        if (state.keyboardMotion) {
            state.keyboardMotion.engaged = true;
        }
        if (!state.visible || !canvasCard || typeof canvasCard.focus !== 'function') return;
        if (document.activeElement === canvasCard) return;
        try {
            canvasCard.focus({ preventScroll: true });
        } catch (_) {
            canvasCard.focus();
        }
    }

    function syncHeadDetailChrome() {
        const isDetailDeepActive = !!state.headDetailDepthActive;
        const isHeadDetailActive = isDetailDeepActive && hasActiveDetailTarget();
        const isHeadDetailSceneActive = isDetailDeepActive && !!state.headDetailTarget;
        const isConcatDetailActive = isDetailDeepActive && !!state.concatDetailTarget;
        root.classList.toggle('is-head-detail-active', isHeadDetailActive);
        root.classList.toggle('is-head-detail-scene-active', isHeadDetailSceneActive);
        root.classList.toggle('is-concat-detail-active', isConcatDetailActive);
        canvasCard?.classList.toggle('is-head-detail-active', isHeadDetailActive);
        canvasCard?.classList.toggle('is-head-detail-scene-active', isHeadDetailSceneActive);
        canvasCard?.classList.toggle('is-concat-detail-active', isConcatDetailActive);
        canvas?.classList.toggle('is-head-detail-active', isHeadDetailActive);
        canvas?.classList.toggle('is-head-detail-scene-active', isHeadDetailSceneActive);
        canvas?.classList.toggle('is-concat-detail-active', isConcatDetailActive);
        if (deepControls) {
            deepControls.hidden = !isHeadDetailSceneActive;
            deepControls.setAttribute('aria-hidden', isHeadDetailSceneActive ? 'false' : 'true');
        }
        if (hud) {
            hud.hidden = isHeadDetailActive;
            hud.setAttribute('aria-hidden', isHeadDetailActive ? 'true' : 'false');
        }
    }

    function computeHeadDetailFocusScale() {
        if (!hasActiveDetailTarget() || !state.layout?.registry) return null;
        const bounds = resolveSelectionFocusBounds();
        if (!bounds) return null;
        const { width, height } = getCanvasSize();
        const transform = resolveViewportFitTransform(bounds, { width, height }, {
            padding: VIEW2D_HEAD_DETAIL_FOCUS_PADDING,
            minScale: VIEW2D_DETAIL_VIEWPORT_MIN_SCALE,
            maxScale: VIEW2D_DETAIL_VIEWPORT_MAX_SCALE
        });
        return Number.isFinite(transform?.scale) ? transform.scale : null;
    }

    function updateHeadDetailDepthState({ preserveDeepState = false } = {}) {
        if (!hasActiveDetailTarget()) {
            state.headDetailFocusScale = null;
            state.headDetailSceneFitScale = null;
            state.headDetailDepthActive = false;
            return false;
        }

        if (state.headDetailTarget) {
            if (state.headDetailDepthActive) {
                const didSyncDetailViewport = syncHeadDetailViewport({
                    forceFit: !Number.isFinite(state.headDetailSceneFitScale) || state.headDetailSceneFitScale <= 0
                });
                if (!didSyncDetailViewport) {
                    state.headDetailDepthActive = false;
                    state.headDetailSceneFitScale = null;
                    return false;
                }
                state.headDetailDepthAutoReentryBlocked = false;
                return state.headDetailDepthActive;
            }

            const focusBounds = resolveSelectionFocusBounds();
            const focusScale = computeHeadDetailFocusScale();
            if (Number.isFinite(focusScale)) {
                state.headDetailFocusScale = focusScale;
            }

            const currentScale = Number(viewportController.getState().scale) || 1;
            const trackedFocusScale = Number(state.headDetailFocusScale) || null;
            if (!Number.isFinite(trackedFocusScale) || trackedFocusScale <= 0) {
                return false;
            }

            if (state.headDetailDepthAutoReentryBlocked) {
                const reentryResetScale = trackedFocusScale * VIEW2D_HEAD_DETAIL_DEPTH_EXIT_RATIO;
                if (currentScale < reentryResetScale) {
                    state.headDetailDepthAutoReentryBlocked = false;
                } else {
                    return false;
                }
            }

            const activateScale = trackedFocusScale * VIEW2D_HEAD_DETAIL_DEPTH_ENTER_RATIO;
            const viewportNearFocus = isViewportNearFocusBounds(focusBounds, viewportController);
            if (currentScale >= activateScale && viewportNearFocus) {
                state.headDetailDepthActive = syncHeadDetailViewport({ forceFit: true });
                if (state.headDetailDepthActive) {
                    state.headDetailDepthAutoReentryBlocked = false;
                }
            }
            return state.headDetailDepthActive;
        }

        const focusBounds = resolveSelectionFocusBounds();
        const focusScale = computeHeadDetailFocusScale();
        if (Number.isFinite(focusScale)) {
            state.headDetailFocusScale = focusScale;
        }

        const currentScale = Number(viewportController.getState().scale) || 1;
        const trackedFocusScale = Number(state.headDetailFocusScale) || null;
        if (!Number.isFinite(trackedFocusScale) || trackedFocusScale <= 0) {
            return !!state.headDetailDepthActive;
        }

        const activateScale = trackedFocusScale * VIEW2D_HEAD_DETAIL_DEPTH_ENTER_RATIO;
        const deactivateScale = trackedFocusScale * VIEW2D_HEAD_DETAIL_DEPTH_EXIT_RATIO;
        const viewportNearFocus = isViewportNearFocusBounds(focusBounds, viewportController);
        let nextDepthState = state.headDetailDepthActive;
        if (nextDepthState) {
            if (!preserveDeepState && (currentScale < deactivateScale || !viewportNearFocus)) {
                nextDepthState = false;
            }
        } else if (currentScale >= activateScale && viewportNearFocus) {
            nextDepthState = true;
        }

        state.headDetailDepthActive = nextDepthState;
        return nextDepthState;
    }

    function applySemanticTarget(semanticTarget = null, focusLabel = '') {
        state.semanticTarget = buildSemanticTarget(semanticTarget);
        state.focusLabel = String(focusLabel || '').trim() || describeTransformerView2dTarget(state.semanticTarget);
        renderTokenStrip();
        if (state.visible) {
            syncTransformerView2dRoute({
                active: true,
                semanticTarget: state.semanticTarget
            });
        }
    }

    function rebuildSceneState() {
        state.scene = buildTransformerSceneModel({
            activationSource: state.activationSource,
            tokenIndices: state.tokenIndices,
            tokenLabels: state.tokenLabels,
            isSmallScreen: state.isSmallScreen,
            headDetailTarget: state.headDetailTarget,
            concatDetailTarget: state.concatDetailTarget,
            outputProjectionDetailTarget: state.outputProjectionDetailTarget
        });
        state.detailSceneIndex = createMhsaDetailSceneIndex(
            state.scene?.metadata?.mhsaHeadDetailScene
            || state.scene?.metadata?.headDetailScene
            || null
        );
        state.detailSceneFocus = null;
        state.detailSceneHoverSignature = '';
        state.layout = buildSceneLayout(state.scene, {
            isSmallScreen: state.isSmallScreen
        });
        renderer.setScene(state.scene, state.layout);
        viewportController.setSceneBounds(state.layout?.sceneBounds || null);
        state.sceneLabel = `${state.scene?.metadata?.layerCount || 0} layers / ${state.scene?.metadata?.tokenCount || 0} tokens`;
        return !!state.scene && !!state.layout;
    }

    function openHeadDetail(headDetailTarget = null, { animate = true } = {}) {
        const resolvedTarget = resolveHeadDetailTarget(headDetailTarget);
        if (!resolvedTarget) return false;
        stopAnimation();
        clearCanvasHover({ scheduleRender: false });
        setDetailTargets({ headDetailTarget: resolvedTarget });
        return commitSceneSelection({ animate, nextDepthActive: false });
    }

    function openConcatDetail(concatDetailTarget = null, { animate = true } = {}) {
        const resolvedTarget = resolveConcatDetailTarget(concatDetailTarget);
        if (!resolvedTarget) return false;
        stopAnimation();
        clearCanvasHover({ scheduleRender: false });
        setDetailTargets({ concatDetailTarget: resolvedTarget });
        return commitSceneSelection({ animate, nextDepthActive: false });
    }

    function openOutputProjectionDetail(outputProjectionDetailTarget = null, { animate = true } = {}) {
        const resolvedTarget = resolveOutputProjectionDetailTarget(outputProjectionDetailTarget);
        if (!resolvedTarget) return false;
        stopAnimation();
        clearCanvasHover({ scheduleRender: false });
        setDetailTargets({ outputProjectionDetailTarget: resolvedTarget });
        return commitSceneSelection({ animate, nextDepthActive: false });
    }

    function closeDetail({ animate = true } = {}) {
        if (!hasActiveDetailTarget()) return false;
        stopAnimation();
        clearCanvasHover({ scheduleRender: false });
        setDetailTargets();
        return commitSceneSelection({ animate, nextDepthActive: false });
    }

    function exitDeepDetail({ animate = true } = {}) {
        if (!state.headDetailTarget || !state.headDetailDepthActive) return false;
        stopAnimation();
        clearCanvasHover({ scheduleRender: false });
        disableAutoFrameState();
        state.headDetailDepthActive = false;
        state.headDetailDepthAutoReentryBlocked = true;
        state.detailSceneFocus = null;
        state.detailSceneHoverSignature = '';
        const bounds = resolveSelectionFocusBounds();
        const padding = VIEW2D_HEAD_DETAIL_FOCUS_PADDING;
        state.focusLabel = resolveActiveFocusLabel(state);
        updateReadouts();

        if (!bounds) {
            render();
            return true;
        }

        if (animate) {
            viewportController.flyToBounds(bounds, {
                animate: true,
                durationMs: 420,
                now: performance.now(),
                padding
            });
            animateViewport();
            return true;
        }

        viewportController.fitToBounds(bounds, { padding });
        render();
        return true;
    }

    function updateReadouts() {
        syncHeadDetailChrome();
        const activeViewportController = getActiveViewportController();
        if (focusReadout) {
            setNodeText(focusReadout, state.focusLabel);
        }
        if (zoomReadout) {
            setNodeText(zoomReadout, `${Math.round((activeViewportController.getState().scale || 1) * 100)}%`);
        }
        if (sceneReadout) {
            setNodeText(sceneReadout, state.sceneLabel);
        }
        if (focusBtn) {
            const isDisabled = !state.semanticTarget;
            if (focusBtn.disabled !== isDisabled) {
                focusBtn.disabled = isDisabled;
            }
            const ariaDisabled = isDisabled ? 'true' : 'false';
            if (focusBtn.getAttribute('aria-disabled') !== ariaDisabled) {
                focusBtn.setAttribute('aria-disabled', ariaDisabled);
            }
        }
    }

    function renderTokenStrip() {
        if (!tokenStrip || !tokenStripTokens) return;

        const { entries, activeIndex } = buildSelectionPromptContext({
            activationSource: state.activationSource,
            laneTokenIndices: state.tokenIndices,
            tokenLabels: state.tokenLabels,
            ...resolveView2dSelectedTokenContext({
                activationSource: state.activationSource,
                tokenIndices: state.tokenIndices,
                tokenLabels: state.tokenLabels,
                semanticTarget: resolveActiveSemanticTarget(state)
            })
        });

        if (!entries.length) {
            state.tokenStripSignature = '';
            tokenStripTokens.replaceChildren();
            tokenStrip.dataset.visible = 'false';
            tokenStrip.setAttribute('aria-hidden', 'true');
            return;
        }

        const nextSignature = entries
            .map((entry, index) => [
                entry.laneIndex,
                entry.tokenIndex,
                entry.tokenId,
                entry.tokenLabel,
                entry.seed,
                index === activeIndex ? 'active' : ''
            ].join('|'))
            .join('\u241f');

        if (nextSignature !== state.tokenStripSignature) {
            const fragment = document.createDocumentFragment();
            entries.forEach((entry, index) => {
                const chip = document.createElement('span');
                chip.className = 'detail-transformer-view2d-token-strip__token detail-transformer-view2d-token';
                if (index === activeIndex) {
                    chip.classList.add('detail-transformer-view2d-token--selected');
                }
                applyTokenChipColors(chip, entry, index);
                chip.textContent = entry.displayText;
                chip.title = entry.titleText || '';
                configureView2dTokenNavChip(chip, {
                    tokenText: entry.tokenLabel,
                    tokenIndex: entry.tokenIndex,
                    tokenId: entry.tokenId
                });
                fragment.appendChild(chip);
            });
            tokenStripTokens.replaceChildren(fragment);
            state.tokenStripSignature = nextSignature;
        }

        tokenStrip.dataset.visible = state.visible ? 'true' : 'false';
        tokenStrip.setAttribute('aria-hidden', state.visible ? 'false' : 'true');
    }

    function render() {
        if (!state.visible || !state.scene || !state.layout) return false;
        const { width, height } = getCanvasSize();
        viewportController.setViewportSize(width, height);
        viewportController.setSceneBounds(state.layout.sceneBounds || null);
        const headDetailSceneBounds = resolveHeadDetailSceneBounds();
        detailViewportController.setViewportSize(width, height);
        detailViewportController.setSceneBounds(headDetailSceneBounds);
        updateHeadDetailDepthState();
        const didRender = renderer.render({
            width,
            height,
            dprCap: state.isInteracting
                ? VIEW2D_PREVIEW_DPR_CAP_INTERACTING
                : VIEW2D_PREVIEW_DPR_CAP_IDLE,
            viewportTransform: viewportController.getViewportTransform('detail-transformer-view2d'),
            detailViewportTransform: (
                state.headDetailDepthActive && state.headDetailTarget
                    ? detailViewportController.getViewportTransform('detail-transformer-view2d-head-detail')
                    : null
            ),
            interacting: state.isInteracting,
            headDetailDepthActive: state.headDetailDepthActive,
            interactionState: {
                hoveredRow: state.hoveredResidualRow,
                previousHoveredRow: state.hoverDimming?.previousHoveredRow || null,
                hoverRowBlend: state.hoverDimming?.rowBlend ?? 1,
                hoverDimStrength: state.hoverDimming?.value || 0,
                detailSceneFocus: state.headDetailDepthActive ? state.detailSceneFocus : null
            }
        });
        const captionSceneState = renderer.getActiveCaptionSceneState();
        residualCaptionOverlay.sync({
            scene: captionSceneState?.scene || null,
            layout: captionSceneState?.layout || null,
            canvas,
            projectBounds: (bounds) => renderer.resolveScreenBounds(bounds),
            visible: state.visible,
            enabled: true
        });
        syncDetailFrame();
        updateReadouts();
        return didRender;
    }

    function scheduleRender() {
        if (state.renderFrame !== null) return;
        state.renderFrame = requestAnimationFrame(() => {
            state.renderFrame = null;
            render();
        });
    }

    function animateViewport() {
        if (state.animationFrame !== null) return;
        const tick = (now) => {
            state.animationFrame = null;
            viewportController.step(now);
            detailViewportController.step(now);
            render();
            if (viewportController.animation || detailViewportController.animation) {
                state.animationFrame = requestAnimationFrame(tick);
            }
        };
        state.animationFrame = requestAnimationFrame(tick);
    }

    function fitScene({ animate = true } = {}) {
        if (state.headDetailDepthActive && state.headDetailTarget) {
            const didFit = syncHeadDetailViewport({
                forceFit: true,
                animate
            });
            if (didFit && !animate) {
                render();
            }
            return didFit;
        }
        if (!state.layout?.sceneBounds) return false;
        if (animate) {
            viewportController.flyToBounds(state.layout.sceneBounds, {
                animate: true,
                durationMs: 420,
                now: performance.now(),
                padding: VIEW2D_DETAIL_VIEWPORT_PADDING
            });
            animateViewport();
            return true;
        }
        viewportController.fitScene({ padding: VIEW2D_DETAIL_VIEWPORT_PADDING });
        render();
        return true;
    }

    function resolveSelectionFocusBounds() {
        if (!state.layout?.registry) return null;
        const focusTargets = resolveFocusSemanticTargets({
            semanticTarget: state.semanticTarget,
            baseSemanticTarget: state.baseSemanticTarget,
            headDetailTarget: state.headDetailTarget,
            concatDetailTarget: state.concatDetailTarget,
            outputProjectionDetailTarget: state.outputProjectionDetailTarget
        });
        for (const focusTarget of focusTargets) {
            const bounds = resolveSemanticTargetBounds(state.layout.registry, focusTarget);
            if (bounds) return bounds;
        }
        return null;
    }

    function focusSelection({ animate = true } = {}) {
        if (state.headDetailDepthActive && state.headDetailTarget) {
            const didFit = syncHeadDetailViewport({
                forceFit: true,
                animate
            });
            if (!didFit) return false;
            state.focusLabel = resolveActiveFocusLabel(state);
            updateReadouts();
            if (!animate) {
                render();
            }
            return true;
        }

        if (!state.layout?.registry || !state.semanticTarget) {
            return fitScene({ animate });
        }

        const bounds = resolveSelectionFocusBounds();
        const activeFocusLabel = resolveActiveFocusLabel(state);
        if (!bounds) {
            state.focusLabel = `${activeFocusLabel} (focus fallback: full scene)`;
            updateReadouts();
            return fitScene({ animate });
        }

        state.focusLabel = activeFocusLabel;
        const padding = hasActiveDetailTarget() ? VIEW2D_HEAD_DETAIL_FOCUS_PADDING : 36;
        const durationMs = hasActiveDetailTarget() ? 520 : 420;
        if (hasActiveDetailTarget()) {
            const { width, height } = getCanvasSize();
            const transform = resolveViewportFitTransform(bounds, { width, height }, {
                padding,
                minScale: VIEW2D_DETAIL_VIEWPORT_MIN_SCALE,
                maxScale: VIEW2D_DETAIL_VIEWPORT_MAX_SCALE
            });
            state.headDetailFocusScale = Number.isFinite(transform?.scale) ? transform.scale : null;
            state.headDetailDepthActive = animate !== true;
        }
        if (animate) {
            viewportController.flyToBounds(bounds, {
                animate: true,
                durationMs,
                now: performance.now(),
                padding
            });
            animateViewport();
            updateReadouts();
            return true;
        }
        viewportController.fitToBounds(bounds, { padding });
        render();
        return true;
    }

    function resolveCanvasPointerHit(event = null) {
        if (!canvas || !Number.isFinite(event?.clientX) || !Number.isFinite(event?.clientY)) {
            return null;
        }
        const rect = canvas.getBoundingClientRect();
        return renderer.resolveInteractiveHitAtScreenPoint(
            event.clientX - rect.left,
            event.clientY - rect.top
        )?.entry || null;
    }

    function updateCanvasCursor(entry = null) {
        if (!canvas) return;
        canvas.style.cursor = (
            isMhsaHeadOverviewEntry(entry)
            || isConcatOverviewEntry(entry)
            || isOutputProjectionOverviewEntry(entry)
        ) ? 'pointer' : '';
    }

    function onSceneNodeClick(entry = null) {
        if (isMhsaHeadOverviewEntry(entry)) {
            return openHeadDetail({
                layerIndex: entry.semantic.layerIndex,
                headIndex: entry.semantic.headIndex
            }, {
                animate: true
            });
        }
        if (isConcatOverviewEntry(entry)) {
            return openConcatDetail({
                layerIndex: entry.semantic.layerIndex
            }, {
                animate: true
            });
        }
        if (isOutputProjectionOverviewEntry(entry)) {
            return openOutputProjectionDetail({
                layerIndex: entry.semantic.layerIndex
            }, {
                animate: true
            });
        }
        return false;
    }

    function clearKeyboardMotion() {
        const keyboardMotion = state.keyboardMotion;
        if (!keyboardMotion) return;
        keyboardMotion.activeKeys.clear();
        keyboardMotion.lastTime = 0;
        if (keyboardMotion.rafId !== null) {
            cancelAnimationFrame(keyboardMotion.rafId);
            keyboardMotion.rafId = null;
        }
    }

    function disengageKeyboardMotion({ clearActiveKeys = true } = {}) {
        if (state.keyboardMotion) {
            state.keyboardMotion.engaged = false;
        }
        if (clearActiveKeys) {
            clearKeyboardMotion();
        }
    }

    function hasFocusedView2dKeyboardTarget() {
        return !!(root && document?.activeElement instanceof Element && root.contains(document.activeElement));
    }

    function applyKeyboardMotion(dt = VIEW2D_KEYBOARD_INITIAL_STEP_MS) {
        const activeKeys = state.keyboardMotion?.activeKeys;
        if (!activeKeys?.size) return false;
        const activeViewportController = getActiveViewportController();

        const safeDt = Math.min(32, Math.max(8, Number.isFinite(dt) ? dt : VIEW2D_KEYBOARD_INITIAL_STEP_MS));
        const panXDir = (activeKeys.has('pan-left') ? 1 : 0) + (activeKeys.has('pan-right') ? -1 : 0);
        const panYDir = (activeKeys.has('pan-up') ? 1 : 0) + (activeKeys.has('pan-down') ? -1 : 0);
        const zoomDir = (activeKeys.has('zoom-in') ? 1 : 0) + (activeKeys.has('zoom-out') ? -1 : 0);

        let changed = false;
        if (panXDir !== 0 || panYDir !== 0) {
            const distance = (VIEW2D_KEYBOARD_PAN_PX_PER_SEC * safeDt) / 1000;
            activeViewportController.panBy(
                panXDir * distance,
                panYDir * distance
            );
            changed = true;
        }

        if (zoomDir !== 0) {
            const { width, height } = getCanvasSize();
            const zoomFactor = Math.exp(zoomDir * safeDt * VIEW2D_KEYBOARD_ZOOM_RATE);
            activeViewportController.zoomAt(zoomFactor, width * 0.5, height * 0.5);
            changed = true;
        }

        if (changed) {
            disableAutoFrameState();
            markInteraction(false);
            scheduleRender();
        }
        return changed;
    }

    function tickKeyboardMotion(time) {
        const keyboardMotion = state.keyboardMotion;
        if (!keyboardMotion) return;
        keyboardMotion.rafId = null;
        if (!state.visible || !keyboardMotion.activeKeys.size) {
            clearKeyboardMotion();
            return;
        }

        const lastTime = Number.isFinite(keyboardMotion.lastTime) ? keyboardMotion.lastTime : time;
        const dt = time - lastTime;
        keyboardMotion.lastTime = time;
        applyKeyboardMotion(dt);

        if (keyboardMotion.activeKeys.size) {
            keyboardMotion.rafId = requestAnimationFrame((nextTime) => {
                tickKeyboardMotion(nextTime);
            });
        }
    }

    function startKeyboardMotion() {
        const keyboardMotion = state.keyboardMotion;
        if (!keyboardMotion || keyboardMotion.rafId !== null) return;
        keyboardMotion.lastTime = performance.now();
        keyboardMotion.rafId = requestAnimationFrame((time) => {
            tickKeyboardMotion(time);
        });
    }

    function handleKeyboardMotionKeyDown(event, {
        isWindowFallback = false
    } = {}) {
        if (!state.visible) return;
        if (event.ctrlKey || event.metaKey || event.altKey) return;
        if (isTextEntryTarget(event.target)) return;
        if (isWindowFallback) {
            if (!state.keyboardMotion?.engaged) return;
            if (hasFocusedView2dKeyboardTarget()) return;
        } else if (state.keyboardMotion) {
            state.keyboardMotion.engaged = true;
        }
        const controlKey = normalizeView2dKeyboardControlKey(event.key);
        if (!controlKey) return;

        state.keyboardMotion.activeKeys.add(controlKey);
        if (!event.repeat) {
            applyKeyboardMotion(VIEW2D_KEYBOARD_INITIAL_STEP_MS);
        }
        startKeyboardMotion();
        event.preventDefault();
        event.stopPropagation();
    }

    function handleKeyboardMotionKeyUp(event, {
        isWindowFallback = false
    } = {}) {
        if (isWindowFallback) {
            if (!state.keyboardMotion?.engaged && !state.keyboardMotion?.activeKeys.size) return;
            if (hasFocusedView2dKeyboardTarget()) return;
        }
        const controlKey = normalizeView2dKeyboardControlKey(event.key);
        if (!controlKey || !state.keyboardMotion) return;
        state.keyboardMotion.activeKeys.delete(controlKey);
        if (!state.keyboardMotion.activeKeys.size) {
            clearKeyboardMotion();
        }
        event.preventDefault();
        event.stopPropagation();
    }

    function onCanvasCardKeyDown(event) {
        handleKeyboardMotionKeyDown(event);
    }

    function onCanvasCardKeyUp(event) {
        handleKeyboardMotionKeyUp(event);
    }

    function onWindowKeyDown(event) {
        handleKeyboardMotionKeyDown(event, {
            isWindowFallback: true
        });
    }

    function onWindowKeyUp(event) {
        handleKeyboardMotionKeyUp(event, {
            isWindowFallback: true
        });
    }

    function onDocumentPointerDown(event) {
        if (!state.visible || !(event?.target instanceof Node)) return;
        if (root.contains(event.target)) return;
        disengageKeyboardMotion();
    }

    function onDocumentFocusIn(event) {
        if (!state.visible || !(event?.target instanceof Node)) return;
        if (root.contains(event.target)) return;
        disengageKeyboardMotion();
    }

    function releaseCanvasPointerCapture(pointerId = null) {
        if (!Number.isFinite(pointerId) || typeof canvas?.releasePointerCapture !== 'function') return;
        try {
            canvas.releasePointerCapture(pointerId);
        } catch (_) { /* no-op */ }
    }

    function beginPointerPan({
        pointerId = null,
        clientX = 0,
        clientY = 0
    } = {}) {
        clearCanvasHover({ scheduleRender: false });
        state.pointer = {
            pointerId: Number.isFinite(pointerId) ? pointerId : null,
            clientX: Number.isFinite(clientX) ? clientX : 0,
            clientY: Number.isFinite(clientY) ? clientY : 0,
            startClientX: Number.isFinite(clientX) ? clientX : 0,
            startClientY: Number.isFinite(clientY) ? clientY : 0,
            moved: false
        };
        if (Number.isFinite(state.pointer.pointerId) && typeof canvas?.setPointerCapture === 'function') {
            try {
                canvas.setPointerCapture(state.pointer.pointerId);
            } catch (_) { /* no-op */ }
        }
        stopAnimation();
        markInteraction(true);
        canvas.classList.add('is-panning');
    }

    function clearPointer(event = null, { scheduleSettle = true } = {}) {
        const capturedPointerId = Number.isFinite(state.pointer?.pointerId)
            ? state.pointer.pointerId
            : (Number.isFinite(event?.pointerId) ? event.pointerId : null);
        releaseCanvasPointerCapture(capturedPointerId);
        state.pointer = null;
        if (!state.touchGesture?.pinchActive) {
            canvas.classList.remove('is-panning');
        }
        if (scheduleSettle) {
            scheduleInteractionSettle();
        }
    }

    function trackTouchPointer(event) {
        if (!state.touchGesture?.pointers || !Number.isFinite(event?.pointerId)) return;
        state.touchGesture.pointers.set(event.pointerId, {
            clientX: Number.isFinite(event?.clientX) ? event.clientX : 0,
            clientY: Number.isFinite(event?.clientY) ? event.clientY : 0
        });
    }

    function untrackTouchPointer(pointerId) {
        if (!state.touchGesture?.pointers || !Number.isFinite(pointerId)) return;
        state.touchGesture.pointers.delete(pointerId);
    }

    function resetTouchGesture() {
        if (!state.touchGesture) return;
        state.touchGesture.pointers.clear();
        state.touchGesture.pinchActive = false;
        state.touchGesture.startDistance = 0;
        state.touchGesture.startScale = 1;
        state.touchGesture.anchorLocalX = 0;
        state.touchGesture.anchorLocalY = 0;
        canvas.classList.remove('is-panning');
    }

    function getTouchMetrics() {
        if (!canvas || !state.touchGesture?.pointers) return null;
        const points = Array.from(state.touchGesture.pointers.values())
            .filter((point) => Number.isFinite(point?.clientX) && Number.isFinite(point?.clientY));
        if (points.length < 2) return null;
        const [first, second] = points;
        const dx = second.clientX - first.clientX;
        const dy = second.clientY - first.clientY;
        const rect = canvas.getBoundingClientRect();
        return {
            distance: Math.hypot(dx, dy),
            canvasX: ((first.clientX + second.clientX) * 0.5) - rect.left,
            canvasY: ((first.clientY + second.clientY) * 0.5) - rect.top
        };
    }

    function beginTouchPinch() {
        if (!state.touchGesture?.pointers || state.touchGesture.pointers.size < 2) return false;
        const metrics = getTouchMetrics();
        if (!metrics || !(metrics.distance > 0)) return false;
        const activeViewportController = getActiveViewportController();
        const viewport = activeViewportController.getState();
        const currentScale = Number.isFinite(viewport.scale) && viewport.scale > 0 ? viewport.scale : 1;
        clearPointer(null, { scheduleSettle: false });
        state.touchGesture.pinchActive = true;
        state.touchGesture.startDistance = metrics.distance;
        state.touchGesture.startScale = currentScale;
        state.touchGesture.anchorLocalX = (metrics.canvasX - viewport.panX) / currentScale;
        state.touchGesture.anchorLocalY = (metrics.canvasY - viewport.panY) / currentScale;
        stopAnimation();
        markInteraction(true);
        canvas.classList.add('is-panning');
        return true;
    }

    function updateTouchPinch(event) {
        if (!state.touchGesture?.pointers || !Number.isFinite(event?.pointerId)) return false;
        const point = state.touchGesture.pointers.get(event.pointerId);
        if (!point) return false;
        point.clientX = Number.isFinite(event?.clientX) ? event.clientX : point.clientX;
        point.clientY = Number.isFinite(event?.clientY) ? event.clientY : point.clientY;
        if (!state.touchGesture.pinchActive) return false;
        const metrics = getTouchMetrics();
        if (!metrics || !(metrics.distance > 0)) return false;
        const distanceRatio = metrics.distance / Math.max(1, state.touchGesture.startDistance);
        const nextScale = state.touchGesture.startScale * distanceRatio;
        disableAutoFrameState();
        getActiveViewportController().setState({
            scale: nextScale,
            panX: metrics.canvasX - (state.touchGesture.anchorLocalX * nextScale),
            panY: metrics.canvasY - (state.touchGesture.anchorLocalY * nextScale)
        });
        markInteraction(true);
        scheduleRender();
        return true;
    }

    function endTouchPinch() {
        if (!state.touchGesture) return;
        state.touchGesture.pinchActive = false;
        state.touchGesture.startDistance = 0;
        state.touchGesture.startScale = 1;
        state.touchGesture.anchorLocalX = 0;
        state.touchGesture.anchorLocalY = 0;
        if (state.touchGesture.pointers.size === 1) {
            const [pointerId, point] = state.touchGesture.pointers.entries().next().value || [];
            if (Number.isFinite(pointerId) && point) {
                beginPointerPan({
                    pointerId,
                    clientX: point.clientX,
                    clientY: point.clientY
                });
                return;
            }
        }
        canvas.classList.remove('is-panning');
        scheduleInteractionSettle();
    }

    function onPointerDown(event) {
        if (!state.visible || (Number.isFinite(event?.button) && event.button !== 0)) return;
        focusCanvasSurface();
        if (event?.pointerType === 'touch') {
            trackTouchPointer(event);
            if (beginTouchPinch()) {
                event.preventDefault();
                return;
            }
        }
        if (state.touchGesture?.pinchActive) {
            event.preventDefault();
            return;
        }
        beginPointerPan({
            pointerId: event?.pointerId,
            clientX: event?.clientX,
            clientY: event?.clientY
        });
        event.preventDefault();
    }

    function onPointerMove(event) {
        if (event?.pointerType === 'touch' && updateTouchPinch(event)) {
            event.preventDefault();
            return;
        }
        if (!state.pointer) {
            updateCanvasCursor(updateCanvasHover(event) || resolveCanvasPointerHit(event));
            return;
        }
        if (!state.pointer || state.pointer.pointerId !== event.pointerId) return;
        const deltaX = event.clientX - state.pointer.clientX;
        const deltaY = event.clientY - state.pointer.clientY;
        state.pointer.clientX = event.clientX;
        state.pointer.clientY = event.clientY;
        if (!state.pointer.moved) {
            state.pointer.moved = Math.hypot(
                event.clientX - state.pointer.startClientX,
                event.clientY - state.pointer.startClientY
            ) >= VIEW2D_CLICK_SLOP_PX;
        }
        getActiveViewportController().panBy(deltaX, deltaY);
        disableAutoFrameState();
        markInteraction(true);
        scheduleRender();
        event.preventDefault();
    }

    function onWheel(event) {
        if (!state.visible) return;
        focusCanvasSurface();
        clearCanvasHover({ scheduleRender: false });
        const rect = canvas.getBoundingClientRect();
        const anchorX = event.clientX - rect.left;
        const anchorY = event.clientY - rect.top;
        const zoomMultiplier = Math.exp(-event.deltaY * 0.0015);
        stopAnimation();
        disableAutoFrameState();
        getActiveViewportController().zoomAt(zoomMultiplier, anchorX, anchorY);
        markInteraction(false);
        scheduleRender();
        event.preventDefault();
    }

    function onPointerUp(event) {
        if (event?.pointerType === 'touch') {
            untrackTouchPointer(event?.pointerId);
            if (state.touchGesture?.pinchActive) {
                endTouchPinch();
                event.preventDefault();
                return;
            }
        }
        if (!state.pointer || state.pointer.pointerId !== event.pointerId) return;
        const shouldTreatAsClick = state.pointer.moved !== true;
        const clickedEntry = shouldTreatAsClick ? resolveCanvasPointerHit(event) : null;
        clearPointer(event);
        updateCanvasCursor(clickedEntry);
        if (shouldTreatAsClick) {
            onSceneNodeClick(clickedEntry);
        }
    }

    function onPointerLeave() {
        clearCanvasHover();
        updateCanvasCursor(null);
    }

    canvasCard?.addEventListener('keydown', onCanvasCardKeyDown);
    canvasCard?.addEventListener('keyup', onCanvasCardKeyUp);
    canvasCard?.addEventListener('pointerdown', focusCanvasSurface);
    canvasCard?.addEventListener('blur', clearKeyboardMotion);
    canvas?.addEventListener('pointerdown', onPointerDown);
    canvas?.addEventListener('pointermove', onPointerMove);
    canvas?.addEventListener('pointerup', onPointerUp);
    canvas?.addEventListener('pointercancel', onPointerUp);
    canvas?.addEventListener('pointerleave', onPointerLeave);
    canvas?.addEventListener('wheel', onWheel, { passive: false });
    canvas?.addEventListener('dblclick', () => {
        focusCanvasSurface();
        focusSelection({ animate: true });
    });
    focusBtn?.addEventListener('click', () => {
        focusSelection({ animate: true });
        focusCanvasSurface();
    });
    fitBtn?.addEventListener('click', () => {
        if (hasActiveDetailTarget()) {
            closeDetail({ animate: false });
        }
        fitScene({ animate: true });
        focusCanvasSurface();
    });
    deepExitBtn?.addEventListener('click', () => {
        exitDeepDetail({ animate: true });
        focusCanvasSurface();
    });
    window.addEventListener('keydown', onWindowKeyDown);
    window.addEventListener('keyup', onWindowKeyUp);
    window.addEventListener('blur', clearKeyboardMotion);
    document.addEventListener('pointerdown', onDocumentPointerDown);
    document.addEventListener('focusin', onDocumentFocusIn);

    return {
        setVisible(visible = false) {
            state.visible = !!visible;
            root.classList.toggle('is-visible', state.visible);
            root.setAttribute('aria-hidden', state.visible ? 'false' : 'true');
            renderTokenStrip();
            if (!state.visible) {
                clearCanvasHover({ scheduleRender: false });
                stopHoverDimmingAnimation();
                setHoverDimmingTarget(0, { immediate: true, shouldRender: false });
                hideDetailFrame();
                residualCaptionOverlay.hide();
                clearPointer(null, { scheduleSettle: false });
                resetTouchGesture();
                disengageKeyboardMotion();
                stopAnimation();
                stopRenderLoop();
                clearInteractionTimer();
                state.isInteracting = false;
                disableAutoFrameState();
                updateCanvasCursor(null);
                return;
            }
            this.resizeAndRender();
        },
        open({
            activationSource = null,
            tokenIndices = null,
            tokenLabels = null,
            semanticTarget = null,
            focusLabel = 'Transformer overview',
            isSmallScreen = false
        } = {}) {
            state.activationSource = activationSource;
            state.tokenIndices = Array.isArray(tokenIndices) ? [...tokenIndices] : tokenIndices;
            state.tokenLabels = Array.isArray(tokenLabels) ? [...tokenLabels] : tokenLabels;
            state.isSmallScreen = !!isSmallScreen;
            state.baseSemanticTarget = deriveBaseSemanticTarget(semanticTarget);
            state.baseFocusLabel = String(focusLabel || '').trim() || describeTransformerView2dTarget(state.baseSemanticTarget);
            setDetailTargets(resolveDetailTargetsFromSemanticTarget(semanticTarget));
            clearCanvasHover({ scheduleRender: false });
            setHoverDimmingTarget(0, { immediate: true, shouldRender: false });
            hideDetailFrame();
            residualCaptionOverlay.hide();
            state.tokenStripSignature = '';
            resetHeadDetailState(hasActiveDetailTarget());
            syncActiveSelectionState();
            renderTokenStrip();
            rebuildSceneState();
            clearPointer(null, { scheduleSettle: false });
            resetTouchGesture();
            resetAutoFrameState({ pendingInitialFocus: true });
            measureCanvasSize();
            updateReadouts();
            const didRender = this.resizeAndRender();
            focusCanvasSurface();
            return didRender;
        },
        resizeAndRender() {
            if (!state.visible || !state.scene || !state.layout) return false;
            const { width, height } = measureCanvasSize();
            viewportController.setViewportSize(width, height);
            viewportController.setSceneBounds(state.layout.sceneBounds || null);
            if (shouldAutoFrameViewport(width, height)) {
                if (!focusSelection({ animate: false })) {
                    fitScene({ animate: false });
                }
            }
            return render();
        },
        getViewportState() {
            return getActiveViewportController().getState();
        }
    };
}
