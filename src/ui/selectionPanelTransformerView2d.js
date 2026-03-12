import { buildSceneLayout } from '../view2d/layout/buildSceneLayout.js';
import { resolveSemanticTargetBounds } from '../view2d/layout/resolveSemanticTargetBounds.js';
import { buildTransformerSceneModel } from '../view2d/model/buildTransformerSceneModel.js';
import { CanvasSceneRenderer } from '../view2d/render/canvas/CanvasSceneRenderer.js';
import {
    buildResidualRowHoverPayload,
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

export const TRANSFORMER_VIEW2D_PANEL_ACTION_OPEN = 'open-transformer-view2d';
export {
    describeTransformerView2dTarget,
    resolveTransformerView2dActionContext,
    resolveTransformerView2dRoute,
    syncTransformerView2dRoute
};

const VIEW2D_DETAIL_ACTION_FOCUS = 'focus-selection';
const VIEW2D_DETAIL_ACTION_FIT = 'fit-scene';
const VIEW2D_INTERACTION_SETTLE_MS = 140;
const VIEW2D_PREVIEW_DPR_CAP_IDLE = 1.5;
const VIEW2D_PREVIEW_DPR_CAP_INTERACTING = 1;
const VIEW2D_CLICK_SLOP_PX = 6;
const VIEW2D_KEYBOARD_PAN_PX_PER_SEC = 620;
const VIEW2D_KEYBOARD_ZOOM_RATE = 0.00165;
const VIEW2D_KEYBOARD_INITIAL_STEP_MS = 16;
const VIEW2D_HEAD_DETAIL_FOCUS_PADDING = Object.freeze({
    top: 6,
    right: 6,
    bottom: 6,
    left: 6
});
const VIEW2D_HEAD_DETAIL_DEPTH_ENTER_RATIO = 0.97;
const VIEW2D_HEAD_DETAIL_DEPTH_EXIT_RATIO = 0.93;
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
                        <div class="detail-transformer-view2d-hint">Drag or use one finger to pan. Scroll, pinch, or use +/- to zoom. Use arrows or WASD to move. Use Focus selection to return to the current component.</div>
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
                <canvas class="detail-transformer-view2d-canvas" aria-label="Scalable 2D transformer canvas"></canvas>
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
    const hud = root.querySelector('.detail-transformer-view2d-hud');
    const focusBtn = root.querySelector(`[data-transformer-view2d-action="${VIEW2D_DETAIL_ACTION_FOCUS}"]`);
    const fitBtn = root.querySelector(`[data-transformer-view2d-action="${VIEW2D_DETAIL_ACTION_FIT}"]`);
    const focusReadout = root.querySelector('[data-transformer-view2d-readout="focus"]');
    const zoomReadout = root.querySelector('[data-transformer-view2d-readout="zoom"]');
    const sceneReadout = root.querySelector('[data-transformer-view2d-readout="scene"]');
    const hoverLabelOverlay = createHoverLabelOverlay({
        zIndex: 12
    });

    const renderer = new CanvasSceneRenderer({
        canvas,
        dprCap: VIEW2D_PREVIEW_DPR_CAP_IDLE
    });
    const viewportController = new View2dViewportController({
        minScale: 0.035,
        maxScale: 10,
        padding: 28
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
        headDetailDepthActive: false,
        semanticTarget: null,
        focusLabel: 'Transformer overview',
        sceneLabel: '0 layers / 0 tokens',
        hoveredResidualRow: null,
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
            lastTime: 0
        },
        viewportSize: {
            width: 1,
            height: 1
        },
        pendingInitialFocus: false
    };

    function hasActiveDetailTarget() {
        return hasActiveDetailTargetState(state);
    }

    function setDetailTargets({
        headDetailTarget = null,
        concatDetailTarget = null,
        outputProjectionDetailTarget = null
    } = {}) {
        state.headDetailTarget = resolveHeadDetailTarget(headDetailTarget);
        state.concatDetailTarget = resolveConcatDetailTarget(concatDetailTarget);
        state.outputProjectionDetailTarget = resolveOutputProjectionDetailTarget(outputProjectionDetailTarget);
    }

    function resetHeadDetailState(nextDepthActive = false) {
        state.headDetailFocusScale = null;
        state.headDetailDepthActive = !!nextDepthActive;
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
        state.pendingInitialFocus = false;
        if (!state.visible) return true;
        measureCanvasSize();
        updateReadouts();
        render();
        focusSelection({ animate });
        return true;
    }

    function stopAnimation() {
        if (state.animationFrame === null) return;
        cancelAnimationFrame(state.animationFrame);
        state.animationFrame = null;
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

    function setNodeText(node, nextText) {
        if (!node) return;
        const safeText = String(nextText ?? '');
        if (node.textContent !== safeText) {
            node.textContent = safeText;
        }
    }

    function clearResidualRowHover({ scheduleRender: shouldScheduleRender = true } = {}) {
        const hadHover = !!state.hoveredResidualRow;
        state.hoveredResidualRow = null;
        hoverLabelOverlay.hide();
        if (hadHover && shouldScheduleRender) {
            scheduleRender();
        }
    }

    function updateResidualRowHover(event = null) {
        const allowHeadDetailHover = !!(state.headDetailDepthActive && state.headDetailTarget);
        if (
            !state.visible
            || !canvas
            || !Number.isFinite(event?.clientX)
            || !Number.isFinite(event?.clientY)
            || event?.pointerType === 'touch'
            || (hasActiveDetailTarget() && !allowHeadDetailHover)
        ) {
            clearResidualRowHover();
            return null;
        }

        const rect = canvas.getBoundingClientRect();
        const localX = event.clientX - rect.left;
        const localY = event.clientY - rect.top;
        const hit = allowHeadDetailHover
            ? renderer.resolveInteractiveHitAtScreenPoint(localX, localY)
            : (() => {
                const worldPoint = viewportController.screenToWorld(localX, localY);
                return renderer.resolveInteractiveHitAtPoint(worldPoint.x, worldPoint.y);
            })();
        const hoverPayload = buildResidualRowHoverPayload(hit?.rowHit, state.activationSource);
        if (!hoverPayload || !hit?.node?.id || !Number.isFinite(hit?.rowHit?.rowIndex)) {
            clearResidualRowHover();
            return hit?.entry || null;
        }

        const nextHoveredRow = {
            nodeId: hit.node.id,
            rowIndex: Math.max(0, Math.floor(hit.rowHit.rowIndex))
        };
        const prevHoveredRow = state.hoveredResidualRow;
        const didChange = !prevHoveredRow
            || prevHoveredRow.nodeId !== nextHoveredRow.nodeId
            || prevHoveredRow.rowIndex !== nextHoveredRow.rowIndex;
        state.hoveredResidualRow = nextHoveredRow;
        hoverLabelOverlay.show({
            clientX: event.clientX,
            clientY: event.clientY,
            label: hoverPayload.label,
            info: hoverPayload.info,
            activationSource: state.activationSource
        });
        if (didChange) {
            scheduleRender();
        }
        return hit?.entry || null;
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
        if (!state.visible || !canvasCard || typeof canvasCard.focus !== 'function') return;
        if (document.activeElement === canvasCard) return;
        try {
            canvasCard.focus({ preventScroll: true });
        } catch (_) {
            canvasCard.focus();
        }
    }

    function syncHeadDetailChrome() {
        const isHeadDetailActive = !!state.headDetailDepthActive;
        const isConcatDetailActive = isHeadDetailActive && !!state.concatDetailTarget;
        root.classList.toggle('is-head-detail-active', isHeadDetailActive);
        root.classList.toggle('is-concat-detail-active', isConcatDetailActive);
        canvasCard?.classList.toggle('is-head-detail-active', isHeadDetailActive);
        canvasCard?.classList.toggle('is-concat-detail-active', isConcatDetailActive);
        canvas?.classList.toggle('is-head-detail-active', isHeadDetailActive);
        canvas?.classList.toggle('is-concat-detail-active', isConcatDetailActive);
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
            minScale: 0.035,
            maxScale: 10
        });
        return Number.isFinite(transform?.scale) ? transform.scale : null;
    }

    function updateHeadDetailDepthState({ preserveDeepState = false } = {}) {
        if (!hasActiveDetailTarget()) {
            state.headDetailFocusScale = null;
            state.headDetailDepthActive = false;
            return false;
        }

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
        let nextDepthState = state.headDetailDepthActive;
        if (nextDepthState) {
            if (!preserveDeepState && currentScale < deactivateScale) {
                nextDepthState = false;
            }
        } else if (currentScale >= activateScale) {
            nextDepthState = true;
        }

        state.headDetailDepthActive = nextDepthState;
        return nextDepthState;
    }

    function applySemanticTarget(semanticTarget = null, focusLabel = '') {
        state.semanticTarget = buildSemanticTarget(semanticTarget);
        state.focusLabel = String(focusLabel || '').trim() || describeTransformerView2dTarget(state.semanticTarget);
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
        clearResidualRowHover({ scheduleRender: false });
        setDetailTargets({ headDetailTarget: resolvedTarget });
        return commitSceneSelection({ animate, nextDepthActive: false });
    }

    function openConcatDetail(concatDetailTarget = null, { animate = true } = {}) {
        const resolvedTarget = resolveConcatDetailTarget(concatDetailTarget);
        if (!resolvedTarget) return false;
        stopAnimation();
        clearResidualRowHover({ scheduleRender: false });
        setDetailTargets({ concatDetailTarget: resolvedTarget });
        return commitSceneSelection({ animate, nextDepthActive: false });
    }

    function openOutputProjectionDetail(outputProjectionDetailTarget = null, { animate = true } = {}) {
        const resolvedTarget = resolveOutputProjectionDetailTarget(outputProjectionDetailTarget);
        if (!resolvedTarget) return false;
        stopAnimation();
        clearResidualRowHover({ scheduleRender: false });
        setDetailTargets({ outputProjectionDetailTarget: resolvedTarget });
        return commitSceneSelection({ animate, nextDepthActive: false });
    }

    function closeDetail({ animate = true } = {}) {
        if (!hasActiveDetailTarget()) return false;
        stopAnimation();
        clearResidualRowHover({ scheduleRender: false });
        setDetailTargets();
        return commitSceneSelection({ animate, nextDepthActive: false });
    }

    function updateReadouts() {
        syncHeadDetailChrome();
        if (focusReadout) {
            setNodeText(focusReadout, state.focusLabel);
        }
        if (zoomReadout) {
            setNodeText(zoomReadout, `${Math.round((viewportController.getState().scale || 1) * 100)}%`);
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

    function render() {
        if (!state.visible || !state.scene || !state.layout) return false;
        const { width, height } = getCanvasSize();
        viewportController.setViewportSize(width, height);
        viewportController.setSceneBounds(state.layout.sceneBounds || null);
        updateHeadDetailDepthState();
        const didRender = renderer.render({
            width,
            height,
            dprCap: state.isInteracting
                ? VIEW2D_PREVIEW_DPR_CAP_INTERACTING
                : VIEW2D_PREVIEW_DPR_CAP_IDLE,
            viewportTransform: viewportController.getViewportTransform('detail-transformer-view2d'),
            interacting: state.isInteracting,
            headDetailDepthActive: state.headDetailDepthActive,
            interactionState: {
                hoveredRow: state.hoveredResidualRow
            }
        });
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
            render();
            if (viewportController.animation) {
                state.animationFrame = requestAnimationFrame(tick);
            }
        };
        state.animationFrame = requestAnimationFrame(tick);
    }

    function fitScene({ animate = true } = {}) {
        if (!state.layout?.sceneBounds) return false;
        if (animate) {
            viewportController.flyToBounds(state.layout.sceneBounds, {
                animate: true,
                durationMs: 420,
                now: performance.now(),
                padding: 28
            });
            animateViewport();
            return true;
        }
        viewportController.fitScene({ padding: 28 });
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
                minScale: 0.035,
                maxScale: 10
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
        if (!canvas || !state.layout?.registry || !Number.isFinite(event?.clientX) || !Number.isFinite(event?.clientY)) {
            return null;
        }
        const rect = canvas.getBoundingClientRect();
        const worldPoint = viewportController.screenToWorld(
            event.clientX - rect.left,
            event.clientY - rect.top
        );
        return state.layout.registry.resolveNodeEntryAtPoint(worldPoint.x, worldPoint.y, {
            includeGroups: false
        });
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

    function applyKeyboardMotion(dt = VIEW2D_KEYBOARD_INITIAL_STEP_MS) {
        const activeKeys = state.keyboardMotion?.activeKeys;
        if (!activeKeys?.size) return false;

        const safeDt = Math.min(32, Math.max(8, Number.isFinite(dt) ? dt : VIEW2D_KEYBOARD_INITIAL_STEP_MS));
        const panXDir = (activeKeys.has('pan-left') ? 1 : 0) + (activeKeys.has('pan-right') ? -1 : 0);
        const panYDir = (activeKeys.has('pan-up') ? 1 : 0) + (activeKeys.has('pan-down') ? -1 : 0);
        const zoomDir = (activeKeys.has('zoom-in') ? 1 : 0) + (activeKeys.has('zoom-out') ? -1 : 0);

        let changed = false;
        if (panXDir !== 0 || panYDir !== 0) {
            const distance = (VIEW2D_KEYBOARD_PAN_PX_PER_SEC * safeDt) / 1000;
            viewportController.panBy(
                panXDir * distance,
                panYDir * distance
            );
            changed = true;
        }

        if (zoomDir !== 0) {
            const { width, height } = getCanvasSize();
            const zoomFactor = Math.exp(zoomDir * safeDt * VIEW2D_KEYBOARD_ZOOM_RATE);
            viewportController.zoomAt(zoomFactor, width * 0.5, height * 0.5);
            changed = true;
        }

        if (changed) {
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

    function onCanvasCardKeyDown(event) {
        if (!state.visible) return;
        if (event.ctrlKey || event.metaKey || event.altKey) return;
        if (isTextEntryTarget(event.target)) return;
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

    function onCanvasCardKeyUp(event) {
        const controlKey = normalizeView2dKeyboardControlKey(event.key);
        if (!controlKey || !state.keyboardMotion) return;
        state.keyboardMotion.activeKeys.delete(controlKey);
        if (!state.keyboardMotion.activeKeys.size) {
            clearKeyboardMotion();
        }
        event.preventDefault();
        event.stopPropagation();
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
        clearResidualRowHover({ scheduleRender: false });
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
        const viewport = viewportController.getState();
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
        viewportController.setState({
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
            updateCanvasCursor(updateResidualRowHover(event) || resolveCanvasPointerHit(event));
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
        viewportController.panBy(deltaX, deltaY);
        markInteraction(true);
        scheduleRender();
        event.preventDefault();
    }

    function onWheel(event) {
        if (!state.visible) return;
        focusCanvasSurface();
        clearResidualRowHover({ scheduleRender: false });
        const rect = canvas.getBoundingClientRect();
        const anchorX = event.clientX - rect.left;
        const anchorY = event.clientY - rect.top;
        const zoomMultiplier = Math.exp(-event.deltaY * 0.0015);
        stopAnimation();
        viewportController.zoomAt(zoomMultiplier, anchorX, anchorY);
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
        clearResidualRowHover();
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

    return {
        setVisible(visible = false) {
            state.visible = !!visible;
            root.classList.toggle('is-visible', state.visible);
            root.setAttribute('aria-hidden', state.visible ? 'false' : 'true');
            if (!state.visible) {
                clearResidualRowHover({ scheduleRender: false });
                clearPointer(null, { scheduleSettle: false });
                resetTouchGesture();
                clearKeyboardMotion();
                stopAnimation();
                stopRenderLoop();
                clearInteractionTimer();
                state.isInteracting = false;
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
            clearResidualRowHover({ scheduleRender: false });
            resetHeadDetailState(hasActiveDetailTarget());
            syncActiveSelectionState();
            rebuildSceneState();
            clearPointer(null, { scheduleSettle: false });
            resetTouchGesture();
            state.pendingInitialFocus = true;
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
            if (state.pendingInitialFocus) {
                state.pendingInitialFocus = false;
                if (!focusSelection({ animate: false })) {
                    fitScene({ animate: false });
                }
            }
            return render();
        },
        getViewportState() {
            return viewportController.getState();
        }
    };
}
