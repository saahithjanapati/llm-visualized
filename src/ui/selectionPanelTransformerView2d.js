import { buildSceneLayout } from '../view2d/layout/buildSceneLayout.js';
import { resolveSemanticTargetBounds } from '../view2d/layout/resolveSemanticTargetBounds.js';
import {
    createMhsaDetailSceneIndex,
    resolveMhsaDetailHoverState
} from '../view2d/mhsaDetailInteraction.js';
import { buildTransformerSceneModel } from '../view2d/model/buildTransformerSceneModel.js';
import { appState } from '../state/appState.js';
import { CanvasSceneRenderer } from '../view2d/render/canvas/CanvasSceneRenderer.js';
import {
    normalizeRaycastLabel,
    simplifyLayerNormParamHoverLabel
} from '../engine/coreRaycastLabels.js';
import {
    buildResidualRowHoverPayload,
    buildResidualRowSelectionFocusState,
    buildSemanticNodeHoverFocusState,
    buildSemanticNodeHoverPayload,
    buildSemanticTarget,
    deriveBaseSemanticTarget,
    describeTransformerView2dTarget,
    hasActiveDetailTarget as hasActiveDetailTargetState,
    isConcatOverviewEntry,
    isLayerNormOverviewEntry,
    isMlpOverviewEntry,
    isMhsaHeadOverviewEntry,
    isOutputProjectionOverviewEntry,
    TRANSFORMER_VIEW2D_OVERVIEW_LABEL,
    resolveActiveFocusLabel,
    resolveActiveSemanticTarget,
    resolveConcatDetailTarget,
    resolveDetailTargetsFromSemanticTarget,
    resolveFocusSemanticTargets,
    resolveHeadDetailTarget,
    resolveLayerNormDetailTarget,
    resolveMlpDetailTarget,
    resolveOutputProjectionDetailTarget,
    resolveTransformerView2dStageHeader,
    resolveTransformerView2dActionContext,
    resolveTransformerView2dOpenTransitionMode
} from '../view2d/transformerView2dTargets.js';
import {
    resolveTransformerView2dRoute,
    syncTransformerView2dRoute
} from '../view2d/transformerView2dRoute.js';
import { View2dViewportController, resolveViewportFitTransform } from '../view2d/runtime/View2dViewportController.js';
import { createHoverLabelOverlay } from './hoverLabelOverlay.js';
import { createSelectionPanelHistoryNavigation } from './selectionPanelHistoryNavigationUtils.js';
import { createTransformerView2dResidualCaptionOverlay } from './transformerView2dResidualCaptionOverlay.js';
import { formatTokenLabelForPreview } from './selectionPanelFormatUtils.js';
import { ATTENTION_VALUE_PLACEHOLDER } from './selectionPanelConstants.js';
import { buildSelectionPromptContext } from './selectionPanelPromptContextUtils.js';
import {
    buildTransformerView2dOverviewState,
    TRANSFORMER_VIEW2D_OVERVIEW_FOCUS_LABEL
} from './selectionPanelTransformerView2dStateUtils.js';
import {
    resolveView2dPointerMoveIntent,
    shouldSuppressView2dDoubleClickFocus,
    shouldTreatView2dPointerReleaseAsClick
} from './selectionPanelTransformerView2dInteractionUtils.js';
import {
    hasTransformerView2dLockedDetailSelection,
    isTransformerView2dDetailSelectionLockActive,
    resolveTransformerView2dDetailClickLockAction,
    shouldFreezeTransformerView2dDetailHover,
    TRANSFORMER_VIEW2D_DETAIL_CLICK_LOCK_ACTIONS
} from './selectionPanelTransformerView2dLockUtils.js';
import {
    createTransformerView2dTokenHoverSync,
    resolveTransformerView2dTokenEntriesFromHoverPayload
} from './selectionPanelTransformerView2dTokenHoverUtils.js';
import {
    isTransformerView2dViewportAtFitScene,
    resolveTransformerView2dOverviewMinScale,
    shouldShowTransformerView2dFitSceneAction,
    TRANSFORMER_VIEW2D_OVERVIEW_MIN_SCALE_DEFAULT
} from './selectionPanelTransformerView2dViewportUtils.js';
import {
    shouldKeepTransformerView2dHeadDetailFitView,
    TRANSFORMER_VIEW2D_STAGED_HEAD_DETAIL_OVERVIEW_TO_HEAD_DURATION_MS
} from './selectionPanelTransformerView2dTransitionUtils.js';
import { initTransformerView2dTouchActionFallback } from './selectionPanelTransformerView2dTouchFallback.js';
import { applyTokenChipColors } from './tokenChipColorUtils.js';
import {
    normalizeTransformerView2dDetailInteractionTargets,
    resolveTransformerView2dDetailInteractionHoverState
} from '../view2d/transformerView2dDetailInteractionTargets.js';

export const TRANSFORMER_VIEW2D_PANEL_ACTION_OPEN = 'open-transformer-view2d';
export {
    describeTransformerView2dTarget,
    resolveTransformerView2dActionContext,
    resolveTransformerView2dOpenTransitionMode,
    resolveTransformerView2dRoute,
    syncTransformerView2dRoute
};

const VIEW2D_DETAIL_ACTION_FIT = 'fit-scene';
const VIEW2D_DETAIL_ACTION_EXIT_DEEP = 'exit-deep-detail';
const VIEW2D_DETAIL_ACTION_EXIT_TO_3D = 'exit-to-3d';
const VIEW2D_DETAIL_ACTION_CLOSE_SELECTION = 'close-selection';
const VIEW2D_SELECTION_SIDEBAR_CLOSE_ANIMATION_MS = 220;
const VIEW2D_SELECTION_SIDEBAR_VIEWPORT_TRANSITION_MS = 240;
const VIEW2D_INTERACTION_SETTLE_MS = 140;
const VIEW2D_PREVIEW_DPR_CAP_IDLE = 1.5;
const VIEW2D_PREVIEW_DPR_CAP_INTERACTING = 1;
const VIEW2D_KEYBOARD_PAN_PX_PER_SEC = 620;
const VIEW2D_KEYBOARD_ZOOM_RATE = 0.00165;
const VIEW2D_KEYBOARD_INITIAL_STEP_MS = 16;
const VIEW2D_ROW_HOVER_FADE_DURATION_MS = 180;
const VIEW2D_DETAIL_VIEWPORT_PADDING = 28;
const VIEW2D_MIN_EFFECTIVE_VIEWPORT_WIDTH_PX = 160;
const VIEW2D_DETAIL_VIEWPORT_MIN_SCALE = TRANSFORMER_VIEW2D_OVERVIEW_MIN_SCALE_DEFAULT;
const VIEW2D_HEAD_DETAIL_VIEWPORT_MIN_SCALE = 0.06;
const VIEW2D_DETAIL_VIEWPORT_MAX_SCALE = 10;
const VIEW2D_HEAD_DETAIL_VIEWPORT_PADDING = Object.freeze({
    top: 44,
    right: 56,
    bottom: 44,
    left: 56
});
const VIEW2D_HEAD_DETAIL_FRAME_PAD_PX = 5;
const VIEW2D_HEAD_DETAIL_FOCUS_PADDING = Object.freeze({
    top: 6,
    right: 6,
    bottom: 6,
    left: 6
});
const VIEW2D_HEAD_DETAIL_COMPONENT_FOCUS_PADDING = Object.freeze({
    top: 20,
    right: 24,
    bottom: 20,
    left: 24
});
const VIEW2D_STAGED_FOCUS_OVERVIEW_HOLD_MS = 280;
const VIEW2D_STAGED_FOCUS_OVERVIEW_TO_TARGET_DURATION_MS = 1120;
const VIEW2D_STAGED_HEAD_DETAIL_OVERVIEW_HOLD_MS = 300;
const VIEW2D_STAGED_HEAD_DETAIL_OVERVIEW_TO_HEAD_DURATION_MS =
    TRANSFORMER_VIEW2D_STAGED_HEAD_DETAIL_OVERVIEW_TO_HEAD_DURATION_MS;
const VIEW2D_HEAD_DETAIL_DEPTH_ENTER_RATIO = 0.97;
const VIEW2D_HEAD_DETAIL_DEPTH_EXIT_RATIO = 0.95;

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

function resolveActiveDetailScene(scene = null, {
    headDetailTarget = null,
    outputProjectionDetailTarget = null,
    mlpDetailTarget = null,
    layerNormDetailTarget = null
} = {}) {
    if (!scene?.metadata) return null;
    if (headDetailTarget) {
        return scene.metadata.mhsaHeadDetailScene
            || scene.metadata.headDetailScene
            || null;
    }
    if (outputProjectionDetailTarget) {
        return scene.metadata.outputProjectionDetailScene || null;
    }
    if (mlpDetailTarget) {
        return scene.metadata.mlpDetailScene || null;
    }
    if (layerNormDetailTarget) {
        return scene.metadata.layerNormDetailScene || null;
    }
    return null;
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
    const isButton = String(chip.tagName || '').toLowerCase() === 'button';

    if (safeTokenText.length > 0) {
        if (!isButton) {
            chip.tabIndex = 0;
            chip.setAttribute('role', 'button');
        } else {
            chip.removeAttribute('tabindex');
            chip.removeAttribute('role');
        }
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

function normalizeView2dSemanticTargets(targets = null) {
    const values = Array.isArray(targets)
        ? targets
        : (targets ? [targets] : []);
    const seen = new Set();
    return values.reduce((acc, target) => {
        const safeTarget = buildSemanticTarget(target);
        if (!safeTarget) return acc;
        const key = JSON.stringify(safeTarget);
        if (seen.has(key)) return acc;
        seen.add(key);
        acc.push(safeTarget);
        return acc;
    }, []);
}

export function setDescriptionTransformerView2dAction(descriptionEl, {
    enabled = false,
    label = 'View in 2D / matrix form'
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

export function createTransformerView2dDetailView(panelEl, {
    onExitTo3d = null,
    onOpenSelection = null,
    onCloseSelection = null
} = {}) {
    if (!panelEl || typeof document === 'undefined') return null;

    const root = document.createElement('section');
    root.className = 'detail-transformer-view2d';
    root.setAttribute('aria-hidden', 'true');
    root.innerHTML = `
        <div class="detail-transformer-view2d-stage">
            <div class="detail-transformer-view2d-hud">
                <div class="detail-transformer-view2d-toolbar">
                    <div class="detail-transformer-view2d-toolbar-copy" aria-live="polite" aria-atomic="true">
                        <div
                            class="detail-transformer-view2d-toolbar-title"
                            data-transformer-view2d-readout="layer"
                            hidden
                        ></div>
                        <div
                            class="detail-transformer-view2d-stage-title"
                            data-transformer-view2d-readout="stage"
                        >
                            ${TRANSFORMER_VIEW2D_OVERVIEW_LABEL}
                        </div>
                    </div>
                    <div class="detail-transformer-view2d-toolbar-actions">
                        <button
                            type="button"
                            class="detail-transformer-view2d-action"
                            data-transformer-view2d-action="${VIEW2D_DETAIL_ACTION_FIT}"
                        >
                            Fit scene
                        </button>
                        <button
                            type="button"
                            class="detail-transformer-view2d-action detail-transformer-view2d-action--back"
                            data-transformer-view2d-action="${VIEW2D_DETAIL_ACTION_EXIT_DEEP}"
                            aria-hidden="true"
                            hidden
                        >
                            Back to Outer View
                        </button>
                        <button
                            type="button"
                            class="detail-transformer-view2d-action detail-transformer-view2d-action--exit"
                            data-transformer-view2d-action="${VIEW2D_DETAIL_ACTION_EXIT_TO_3D}"
                            aria-label="Exit the 2D canvas and return to the 3D view"
                        >
                            Go to 3D
                        </button>
                    </div>
                </div>
            </div>
            <div class="detail-transformer-view2d-workspace">
                <div
                    class="detail-transformer-view2d-canvas-card"
                    tabindex="0"
                    role="group"
                    aria-label="Scalable 2D transformer canvas. Drag or use one finger to pan, pinch or scroll to zoom, and use arrow keys or W A S D to move around."
                >
                    <div class="detail-transformer-view2d-detail-frame" aria-hidden="true"></div>
                    <canvas class="detail-transformer-view2d-canvas" aria-label="Scalable 2D transformer canvas"></canvas>
                    <div
                        class="detail-transformer-view2d-token-strip"
                        data-visible="false"
                        aria-hidden="true"
                    >
                        <div
                            class="detail-transformer-view2d-token-strip__tokens prompt-token-strip__tokens"
                            data-transformer-view2d-role="token-strip-tokens"
                        ></div>
                    </div>
                </div>
                <aside
                    class="detail-transformer-view2d-selection-sidebar"
                    aria-hidden="true"
                >
                    <div class="detail-transformer-view2d-selection-sidebar-header">
                        <div class="detail-transformer-view2d-selection-sidebar-copy">
                            <div
                                class="detail-transformer-view2d-selection-sidebar-title detail-title"
                                data-transformer-view2d-role="selection-sidebar-title"
                                hidden
                            ></div>
                            <div
                                class="detail-transformer-view2d-selection-sidebar-subtitle detail-subtitle"
                                data-transformer-view2d-role="selection-sidebar-subtitle"
                                hidden
                            ></div>
                            <div
                                class="detail-transformer-view2d-selection-sidebar-subtitle detail-subtitle"
                                data-transformer-view2d-role="selection-sidebar-subtitle-secondary"
                                hidden
                            ></div>
                            <div
                                class="detail-transformer-view2d-selection-sidebar-subtitle detail-subtitle"
                                data-transformer-view2d-role="selection-sidebar-subtitle-tertiary"
                                hidden
                            ></div>
                        </div>
                        <button
                            type="button"
                            class="detail-transformer-view2d-selection-sidebar-close"
                            data-transformer-view2d-action="${VIEW2D_DETAIL_ACTION_CLOSE_SELECTION}"
                            aria-label="Hide selection details"
                            title="Close details"
                        >
                            ×
                        </button>
                    </div>
                    <div
                        class="detail-transformer-view2d-selection-sidebar-body"
                        data-transformer-view2d-role="selection-sidebar-body"
                    ></div>
                </aside>
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
    const workspace = root.querySelector('.detail-transformer-view2d-workspace');
    const canvasCard = root.querySelector('.detail-transformer-view2d-canvas-card');
    const detailFrame = root.querySelector('.detail-transformer-view2d-detail-frame');
    const tokenStrip = root.querySelector('.detail-transformer-view2d-token-strip');
    const tokenStripTokens = root.querySelector('[data-transformer-view2d-role="token-strip-tokens"]');
    const selectionSidebar = root.querySelector('.detail-transformer-view2d-selection-sidebar');
    const selectionSidebarHeader = root.querySelector('.detail-transformer-view2d-selection-sidebar-header');
    const selectionSidebarBody = root.querySelector('[data-transformer-view2d-role="selection-sidebar-body"]');
    const selectionSidebarTitle = root.querySelector('[data-transformer-view2d-role="selection-sidebar-title"]');
    const selectionSidebarSubtitle = root.querySelector('[data-transformer-view2d-role="selection-sidebar-subtitle"]');
    const selectionSidebarSubtitleSecondary = root.querySelector('[data-transformer-view2d-role="selection-sidebar-subtitle-secondary"]');
    const selectionSidebarSubtitleTertiary = root.querySelector('[data-transformer-view2d-role="selection-sidebar-subtitle-tertiary"]');
    const selectionSidebarHistoryNavigation = createSelectionPanelHistoryNavigation(document);
    if (selectionSidebarHeader && selectionSidebarHistoryNavigation?.nav) {
        selectionSidebarHeader.insertBefore(
            selectionSidebarHistoryNavigation.nav,
            selectionSidebarHeader.firstChild || null
        );
    }
    const tokenHoverSync = createTransformerView2dTokenHoverSync({
        container: tokenStripTokens
    });
    const hud = root.querySelector('.detail-transformer-view2d-hud');
    const fitBtn = root.querySelector(`[data-transformer-view2d-action="${VIEW2D_DETAIL_ACTION_FIT}"]`);
    const backToGraphBtn = root.querySelector(`[data-transformer-view2d-action="${VIEW2D_DETAIL_ACTION_EXIT_DEEP}"]`);
    const exitTo3dBtn = root.querySelector(`[data-transformer-view2d-action="${VIEW2D_DETAIL_ACTION_EXIT_TO_3D}"]`);
    const closeSelectionBtn = root.querySelector(`[data-transformer-view2d-action="${VIEW2D_DETAIL_ACTION_CLOSE_SELECTION}"]`);
    const stageLayerReadout = root.querySelector('[data-transformer-view2d-readout="layer"]');
    const stageTitleReadout = root.querySelector('[data-transformer-view2d-readout="stage"]');
    const hoverLabelOverlay = createHoverLabelOverlay({
        zIndex: 12
    });
    const residualCaptionOverlay = createTransformerView2dResidualCaptionOverlay({
        parent: canvasCard
    });

    initTransformerView2dTouchActionFallback(root);

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
        padding: VIEW2D_HEAD_DETAIL_VIEWPORT_PADDING
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
        baseFocusLabel: TRANSFORMER_VIEW2D_OVERVIEW_FOCUS_LABEL,
        headDetailTarget: null,
        concatDetailTarget: null,
        outputProjectionDetailTarget: null,
        mlpDetailTarget: null,
        layerNormDetailTarget: null,
        detailSemanticTargets: [],
        detailFocusLabel: '',
        pendingDetailInteractionTargets: [],
        headDetailFocusScale: null,
        headDetailSceneFitScale: null,
        headDetailDepthActive: false,
        headDetailDepthAutoReentryBlocked: false,
        semanticTarget: null,
        focusLabel: TRANSFORMER_VIEW2D_OVERVIEW_FOCUS_LABEL,
        detailSceneIndex: null,
        detailSceneFocus: null,
        detailSceneHoverSignature: '',
        detailScenePinnedFocus: null,
        detailScenePinnedSignature: '',
        detailScenePinnedTokenEntries: null,
        detailScenePinnedTokenSticky: false,
        detailSceneLockActive: false,
        overviewScenePinnedFocus: null,
        overviewScenePinnedSignature: '',
        overviewScenePinnedTokenEntries: null,
        overviewSceneLockActive: false,
        overviewSceneFocus: null,
        overviewSceneHoverSignature: '',
        hoveredResidualRow: null,
        hoverDimming: {
            value: 0,
            target: 0,
            rafId: null,
            lastTime: 0,
            previousHoveredRow: null,
            rowBlend: 1,
            rowBlendTarget: 1,
            previousOverviewFocus: null,
            overviewBlend: 1,
            overviewBlendTarget: 1
        },
        hoverTargetKey: '',
        hoverFrame: null,
        pendingHoverEvent: null,
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
        tokenStripSignature: '',
        stagedFocusTransition: null,
        stagedFocusRafId: null,
        stagedHeadDetailTransition: null,
        stagedHeadDetailRafId: null,
        stagedDetailTransition: null,
        stagedDetailRafId: null,
        overviewSelectionArmSignature: ''
    };
    let selectionSidebarCloseTimerId = null;

    function clearSelectionSidebarCloseTimer() {
        if (selectionSidebarCloseTimerId === null) return;
        clearTimeout(selectionSidebarCloseTimerId);
        selectionSidebarCloseTimerId = null;
    }

    function resetOverviewSelectionArm() {
        state.overviewSelectionArmSignature = '';
    }

    function shouldRequireTouchSelectionConfirmation(event = null, armSignature = '') {
        return event?.pointerType === 'touch'
            && typeof armSignature === 'string'
            && armSignature.length > 0
            && state.overviewSelectionArmSignature !== armSignature;
    }

    function applyTouchSelectionHover(event = null, {
        armSignature = '',
        detailHoverState = null
    } = {}) {
        if (!armSignature) return false;
        if (detailHoverState?.focusState) {
            lockPinnedDetailSceneFocus(detailHoverState, {
                persistTokenChip: true,
                lockSelection: false
            });
            state.overviewSelectionArmSignature = armSignature;
            return true;
        }
        if (Number.isFinite(event?.clientX) && Number.isFinite(event?.clientY)) {
            updateCanvasHover({
                clientX: event.clientX,
                clientY: event.clientY,
                pointerType: 'mouse'
            });
        }
        state.overviewSelectionArmSignature = armSignature;
        return true;
    }

    function requestSelectionOpen(selection = null) {
        if (!selection?.label || typeof onOpenSelection !== 'function') return false;
        return onOpenSelection(selection) !== false;
    }

    function requestSelectionClose() {
        const didHandleClose = (typeof onCloseSelection === 'function')
            ? onCloseSelection()
            : false;
        if (didHandleClose !== false) {
            return true;
        }
        if (!selectionSidebar || !isSelectionSidebarVisible()) {
            return false;
        }
        setSelectionSidebarVisible(false);
        return true;
    }

    function resolveTokenStripChipSelection(chip = null) {
        if (!chip || chip.dataset.tokenNav !== 'true') return null;
        const tokenText = formatTokenLabelForPreview(chip.dataset.tokenText || chip.textContent || '');
        if (!tokenText || tokenText === ATTENTION_VALUE_PLACEHOLDER) return null;
        const tokenIndex = normalizeView2dTokenChipIndex(Number(chip.dataset.tokenIndex));
        const tokenId = normalizeView2dTokenChipIndex(Number(chip.dataset.tokenId));
        const info = {
            tokenLabel: tokenText
        };
        if (Number.isFinite(tokenIndex)) info.tokenIndex = tokenIndex;
        if (Number.isFinite(tokenId)) info.tokenId = tokenId;
        return {
            label: `Token: ${tokenText}`,
            kind: 'label',
            info
        };
    }

    function requestTokenStripChipSelection(chip = null, event = null) {
        const selection = resolveTokenStripChipSelection(chip);
        if (!selection) return false;
        if (event && typeof event.preventDefault === 'function') event.preventDefault();
        if (event && typeof event.stopPropagation === 'function') event.stopPropagation();
        return requestSelectionOpen(selection);
    }

    function resolveCanvasSemanticSelection(hit = null) {
        return buildSemanticNodeHoverPayload(hit);
    }

    function setSelectionSidebarLine(element, {
        html = '',
        className = ''
    } = {}) {
        if (!element) return;
        if (className) {
            element.className = className;
        }
        const safeHtml = typeof html === 'string' ? html : '';
        element.innerHTML = safeHtml;
        const hasContent = safeHtml.trim().length > 0;
        element.hidden = !hasContent;
    }

    function clearSelectionSidebarHeader() {
        setSelectionSidebarLine(selectionSidebarTitle, {
            html: '',
            className: 'detail-transformer-view2d-selection-sidebar-title detail-title'
        });
        setSelectionSidebarLine(selectionSidebarSubtitle, {
            html: '',
            className: 'detail-transformer-view2d-selection-sidebar-subtitle detail-subtitle'
        });
        setSelectionSidebarLine(selectionSidebarSubtitleSecondary, {
            html: '',
            className: 'detail-transformer-view2d-selection-sidebar-subtitle detail-subtitle'
        });
        setSelectionSidebarLine(selectionSidebarSubtitleTertiary, {
            html: '',
            className: 'detail-transformer-view2d-selection-sidebar-subtitle detail-subtitle'
        });
    }

    function setSelectionSidebarVisible(visible = false, {
        immediate = false
    } = {}) {
        if (!selectionSidebar || !workspace) return;
        const nextVisible = !!visible;
        const wasAtFitScene = isViewportAtFitScene();
        clearSelectionSidebarCloseTimer();
        workspace.dataset.selectionSidebarVisible = nextVisible ? 'true' : 'false';
        selectionSidebar.setAttribute('aria-hidden', nextVisible ? 'false' : 'true');

        if (nextVisible) {
            selectionSidebar.classList.remove('is-closing');
            selectionSidebar.classList.add('is-visible');
        } else if (immediate || !state.isSmallScreen) {
            selectionSidebar.classList.remove('is-visible', 'is-closing');
            clearSelectionSidebarHeader();
        } else {
            selectionSidebar.classList.remove('is-visible');
            selectionSidebar.classList.add('is-closing');
            selectionSidebarCloseTimerId = setTimeout(() => {
                selectionSidebar.classList.remove('is-closing');
                clearSelectionSidebarHeader();
                selectionSidebarCloseTimerId = null;
                if (state.visible) {
                    syncViewportForSidebarVisibilityChange({
                        animate: false,
                        wasAtFitScene: false
                    });
                }
            }, VIEW2D_SELECTION_SIDEBAR_CLOSE_ANIMATION_MS);
        }

        if (state.visible) {
            syncViewportForSidebarVisibilityChange({
                animate: !immediate,
                wasAtFitScene
            });
            // Sidebar toggles should preserve the user's current viewport rather than
            // triggering the next resize pass to auto-frame back out.
            disableAutoFrameState();
        }
    }

    function setSelectionSidebarHeaderContent({
        titleHtml = '',
        titleClassName = 'detail-transformer-view2d-selection-sidebar-title detail-title',
        subtitleHtml = '',
        subtitleClassName = 'detail-transformer-view2d-selection-sidebar-subtitle detail-subtitle',
        subtitleSecondaryHtml = '',
        subtitleSecondaryClassName = 'detail-transformer-view2d-selection-sidebar-subtitle detail-subtitle',
        subtitleTertiaryHtml = '',
        subtitleTertiaryClassName = 'detail-transformer-view2d-selection-sidebar-subtitle detail-subtitle'
    } = {}) {
        setSelectionSidebarLine(selectionSidebarTitle, {
            html: titleHtml,
            className: titleClassName
        });
        setSelectionSidebarLine(selectionSidebarSubtitle, {
            html: subtitleHtml,
            className: subtitleClassName
        });
        setSelectionSidebarLine(selectionSidebarSubtitleSecondary, {
            html: subtitleSecondaryHtml,
            className: subtitleSecondaryClassName
        });
        setSelectionSidebarLine(selectionSidebarSubtitleTertiary, {
            html: subtitleTertiaryHtml,
            className: subtitleTertiaryClassName
        });
    }

    function scrollSelectionSidebarToTop() {
        if (!selectionSidebarBody) return;
        if (typeof selectionSidebarBody.scrollTo === 'function') {
            selectionSidebarBody.scrollTo({ top: 0, left: 0, behavior: 'auto' });
            return;
        }
        selectionSidebarBody.scrollTop = 0;
        selectionSidebarBody.scrollLeft = 0;
    }

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

    function getActiveSceneFocusState() {
        return state.headDetailDepthActive
            ? (state.detailSceneFocus || null)
            : (state.overviewSceneFocus || state.overviewScenePinnedFocus || null);
    }

    function shouldUseOverviewHoverRenderFastPath() {
        if (state.headDetailDepthActive) return false;
        if (state.hoveredResidualRow || state.overviewSceneFocus || state.hoverDimming?.previousOverviewFocus) return true;
        const hoverDimmingValue = Number(state.hoverDimming?.value) || 0;
        const hoverDimmingTarget = Number(state.hoverDimming?.target) || 0;
        return hoverDimmingValue > 0.001 || hoverDimmingTarget > 0.001;
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

    function syncViewportControllerConstraints(viewportWidth = 0, viewportInsets = null) {
        const horizontalInset = Math.max(
            0,
            Math.floor(
                (Number(viewportInsets?.left) || 0)
                + (Number(viewportInsets?.right) || 0)
            )
        );
        const effectiveViewportWidth = Math.max(
            1,
            Math.floor(Number(viewportWidth) || 0) - horizontalInset
        );
        const nextOverviewMinScale = resolveTransformerView2dOverviewMinScale({
            isSmallScreen: state.isSmallScreen,
            viewportWidth: effectiveViewportWidth
        });
        viewportController.minScale = nextOverviewMinScale;
        viewportController.maxScale = Math.max(nextOverviewMinScale, VIEW2D_DETAIL_VIEWPORT_MAX_SCALE);
        return nextOverviewMinScale;
    }

    function hasActiveDetailTarget() {
        return hasActiveDetailTargetState(state);
    }

    function isTransformerView2dGraphOverview() {
        return !state.semanticTarget && !state.baseSemanticTarget;
    }

    function hasSceneBackedDetailTarget() {
        return !!(
            state.headDetailTarget
            || state.outputProjectionDetailTarget
            || state.mlpDetailTarget
            || state.layerNormDetailTarget
        );
    }

    function shouldKeepHeadDetailSceneFitView() {
        return hasSceneBackedDetailTarget()
            && shouldKeepTransformerView2dHeadDetailFitView(state.detailSemanticTargets);
    }

    function shouldKeepSceneBackedDetailFitView() {
        return hasSceneBackedDetailTarget()
            && shouldKeepTransformerView2dHeadDetailFitView(state.detailSemanticTargets);
    }

    function setDetailTargets({
        headDetailTarget = null,
        concatDetailTarget = null,
        outputProjectionDetailTarget = null,
        mlpDetailTarget = null,
        layerNormDetailTarget = null
    } = {}) {
        const redirectedConcatDetailTarget = resolveConcatDetailTarget(concatDetailTarget);
        const resolvedOutputProjectionDetailTarget = resolveOutputProjectionDetailTarget(outputProjectionDetailTarget)
            || (
                redirectedConcatDetailTarget
                    ? { layerIndex: redirectedConcatDetailTarget.layerIndex }
                    : null
            );
        const resolvedConcatDetailTarget = null;
        const resolvedMlpDetailTarget = resolvedOutputProjectionDetailTarget
            ? null
            : resolveMlpDetailTarget(mlpDetailTarget);
        const resolvedLayerNormDetailTarget = (
            resolvedOutputProjectionDetailTarget
            || resolvedMlpDetailTarget
        )
            ? null
            : resolveLayerNormDetailTarget(layerNormDetailTarget);
        const resolvedHeadDetailTarget = (
            resolvedOutputProjectionDetailTarget
            || resolvedMlpDetailTarget
            || resolvedLayerNormDetailTarget
        )
            ? null
            : resolveHeadDetailTarget(headDetailTarget);
        state.headDetailTarget = resolvedHeadDetailTarget;
        state.concatDetailTarget = resolvedConcatDetailTarget;
        state.outputProjectionDetailTarget = resolvedOutputProjectionDetailTarget;
        state.mlpDetailTarget = resolvedMlpDetailTarget;
        state.layerNormDetailTarget = resolvedLayerNormDetailTarget;
    }

    function setDetailFocusTarget({
        detailSemanticTargets = null,
        detailFocusLabel = ''
    } = {}) {
        state.detailSemanticTargets = normalizeView2dSemanticTargets(detailSemanticTargets);
        state.detailFocusLabel = String(detailFocusLabel || '').trim();
    }

    function clearStagedFocusTransition() {
        if (state.stagedFocusRafId !== null) {
            cancelAnimationFrame(state.stagedFocusRafId);
            state.stagedFocusRafId = null;
        }
        state.stagedFocusTransition = null;
    }

    function clearStagedHeadDetailTransition() {
        if (state.stagedHeadDetailRafId !== null) {
            cancelAnimationFrame(state.stagedHeadDetailRafId);
            state.stagedHeadDetailRafId = null;
        }
        state.stagedHeadDetailTransition = null;
    }

    function clearStagedDetailTransition() {
        if (state.stagedDetailRafId !== null) {
            cancelAnimationFrame(state.stagedDetailRafId);
            state.stagedDetailRafId = null;
        }
        state.stagedDetailTransition = null;
    }

    function resolveHeadDetailFocusLabel() {
        return state.detailFocusLabel || resolveActiveFocusLabel(state);
    }

    function scheduleStagedFocusPhase(delayMs = 0, onReady = null) {
        if (typeof onReady !== 'function' || !state.stagedFocusTransition || !state.visible) {
            return false;
        }
        if (state.stagedFocusRafId !== null) {
            cancelAnimationFrame(state.stagedFocusRafId);
            state.stagedFocusRafId = null;
        }
        const safeDelayMs = Math.max(0, Math.floor(Number(delayMs) || 0));
        const tick = (now) => {
            state.stagedFocusRafId = null;
            const activeTransition = state.stagedFocusTransition;
            if (!activeTransition || !state.visible) return;
            const phaseStartTime = Number.isFinite(activeTransition.phaseStartTime)
                ? activeTransition.phaseStartTime
                : now;
            if (!Number.isFinite(activeTransition.phaseStartTime)) {
                state.stagedFocusTransition = {
                    ...activeTransition,
                    phaseStartTime
                };
            }
            if ((now - phaseStartTime) < safeDelayMs) {
                state.stagedFocusRafId = requestAnimationFrame(tick);
                return;
            }
            onReady(now, activeTransition);
        };
        state.stagedFocusRafId = requestAnimationFrame(tick);
        return true;
    }

    function scheduleStagedHeadDetailPhase(delayMs = 0, onReady = null) {
        if (typeof onReady !== 'function' || !state.stagedHeadDetailTransition || !state.visible) {
            return false;
        }
        if (state.stagedHeadDetailRafId !== null) {
            cancelAnimationFrame(state.stagedHeadDetailRafId);
            state.stagedHeadDetailRafId = null;
        }
        const safeDelayMs = Math.max(0, Math.floor(Number(delayMs) || 0));
        const tick = (now) => {
            state.stagedHeadDetailRafId = null;
            const activeTransition = state.stagedHeadDetailTransition;
            if (!activeTransition || !state.visible) return;
            const phaseStartTime = Number.isFinite(activeTransition.phaseStartTime)
                ? activeTransition.phaseStartTime
                : now;
            if (!Number.isFinite(activeTransition.phaseStartTime)) {
                state.stagedHeadDetailTransition = {
                    ...activeTransition,
                    phaseStartTime
                };
            }
            if ((now - phaseStartTime) < safeDelayMs) {
                state.stagedHeadDetailRafId = requestAnimationFrame(tick);
                return;
            }
            onReady(now, activeTransition);
        };
        state.stagedHeadDetailRafId = requestAnimationFrame(tick);
        return true;
    }

    function scheduleStagedDetailPhase(delayMs = 0, onReady = null) {
        if (typeof onReady !== 'function' || !state.stagedDetailTransition || !state.visible) {
            return false;
        }
        if (state.stagedDetailRafId !== null) {
            cancelAnimationFrame(state.stagedDetailRafId);
            state.stagedDetailRafId = null;
        }
        const safeDelayMs = Math.max(0, Math.floor(Number(delayMs) || 0));
        const tick = (now) => {
            state.stagedDetailRafId = null;
            const activeTransition = state.stagedDetailTransition;
            if (!activeTransition || !state.visible) return;
            const phaseStartTime = Number.isFinite(activeTransition.phaseStartTime)
                ? activeTransition.phaseStartTime
                : now;
            if (!Number.isFinite(activeTransition.phaseStartTime)) {
                state.stagedDetailTransition = {
                    ...activeTransition,
                    phaseStartTime
                };
            }
            if ((now - phaseStartTime) < safeDelayMs) {
                state.stagedDetailRafId = requestAnimationFrame(tick);
                return;
            }
            onReady(now, activeTransition);
        };
        state.stagedDetailRafId = requestAnimationFrame(tick);
        return true;
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
        nextDepthActive = false,
        focusDurationMs = null
    } = {}) {
        resetHeadDetailState(nextDepthActive);
        syncActiveSelectionState();
        rebuildSceneState();
        if (!state.visible) return true;
        const { width, height } = measureCanvasSize();
        updateReadouts();
        render();
        focusSelection({
            animate,
            durationMs: focusDurationMs
        });
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

    function resolveSelectionSidebarViewportInsets() {
        const emptyInsets = {
            top: 0,
            right: 0,
            bottom: 0,
            left: 0
        };
        if (!canvas || !selectionSidebar) return emptyInsets;
        if (
            !selectionSidebar.classList.contains('is-visible')
            && !selectionSidebar.classList.contains('is-closing')
        ) {
            return emptyInsets;
        }

        const canvasRect = typeof canvas.getBoundingClientRect === 'function'
            ? canvas.getBoundingClientRect()
            : null;
        const sidebarRect = typeof selectionSidebar.getBoundingClientRect === 'function'
            ? selectionSidebar.getBoundingClientRect()
            : null;
        const canvasWidth = Math.max(0, Number(canvasRect?.width) || 0);
        const canvasHeight = Math.max(0, Number(canvasRect?.height) || 0);
        if (!(canvasWidth > 0) || !(canvasHeight > 0)) {
            return emptyInsets;
        }

        const overlapWidth = Math.max(
            0,
            Math.min(Number(canvasRect?.right) || 0, Number(sidebarRect?.right) || 0)
                - Math.max(Number(canvasRect?.left) || 0, Number(sidebarRect?.left) || 0)
        );
        const overlapHeight = Math.max(
            0,
            Math.min(Number(canvasRect?.bottom) || 0, Number(sidebarRect?.bottom) || 0)
                - Math.max(Number(canvasRect?.top) || 0, Number(sidebarRect?.top) || 0)
        );
        const remainingWidth = canvasWidth - overlapWidth;
        if (
            !(overlapWidth > 0)
            || !(overlapHeight > 0)
            || remainingWidth < VIEW2D_MIN_EFFECTIVE_VIEWPORT_WIDTH_PX
        ) {
            return emptyInsets;
        }

        const canvasCenterX = (Number(canvasRect?.left) || 0) + (canvasWidth * 0.5);
        const sidebarCenterX = (Number(sidebarRect?.left) || 0) + ((Number(sidebarRect?.width) || 0) * 0.5);
        const insetWidth = Math.round(overlapWidth);
        return sidebarCenterX >= canvasCenterX
            ? {
                ...emptyInsets,
                right: insetWidth
            }
            : {
                ...emptyInsets,
                left: insetWidth
            };
    }

    function syncViewportControllers({
        preserveVisibleCenter = false,
        animate = false,
        durationMs = VIEW2D_SELECTION_SIDEBAR_VIEWPORT_TRANSITION_MS
    } = {}) {
        const { width, height } = getCanvasSize();
        const viewportInsets = resolveSelectionSidebarViewportInsets();
        const transitionNow = performance.now();
        viewportController.setViewportSize(width, height);
        detailViewportController.setViewportSize(width, height);
        viewportController.setViewportInsets(viewportInsets, {
            preserveVisibleCenter,
            animate,
            durationMs,
            now: transitionNow,
            source: 'detail-transformer-view2d-sidebar-shift'
        });
        detailViewportController.setViewportInsets(viewportInsets, {
            preserveVisibleCenter,
            animate,
            durationMs,
            now: transitionNow,
            source: 'detail-transformer-view2d-sidebar-shift'
        });
        return {
            width,
            height,
            viewportInsets
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

    function getActiveViewportController() {
        return state.headDetailDepthActive && hasSceneBackedDetailTarget()
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

    function resolveHeadDetailFocusBounds() {
        const registry = renderer.headDetailSceneState?.layout?.registry || null;
        if (!registry) return null;
        const candidates = Array.isArray(state.detailSemanticTargets)
            ? state.detailSemanticTargets
            : [];
        for (const candidate of candidates) {
            const bounds = resolveSemanticTargetBounds(registry, candidate);
            if (bounds) return bounds;
        }
        return null;
    }

    function syncHeadDetailViewport({
        forceFit = false,
        animate = false,
        durationMs = 420
    } = {}) {
        if (!hasSceneBackedDetailTarget()) return false;
        const bounds = resolveHeadDetailSceneBounds();
        if (!bounds) return false;
        const { width, height, viewportInsets } = syncViewportControllers();
        detailViewportController.setSceneBounds(bounds);
        const viewportPadding = VIEW2D_HEAD_DETAIL_VIEWPORT_PADDING;
        const fitTransform = resolveViewportFitTransform(bounds, { width, height }, {
            padding: viewportPadding,
            minScale: VIEW2D_HEAD_DETAIL_VIEWPORT_MIN_SCALE,
            maxScale: VIEW2D_DETAIL_VIEWPORT_MAX_SCALE,
            viewportInsets
        });
        state.headDetailSceneFitScale = Number.isFinite(fitTransform?.scale) ? fitTransform.scale : null;
        if (!forceFit) {
            return true;
        }
        if (animate) {
            detailViewportController.flyToBounds(bounds, {
                animate: true,
                durationMs,
                now: performance.now(),
                padding: viewportPadding
            });
            animateViewport();
            return true;
        }
        detailViewportController.fitToBounds(bounds, {
            padding: viewportPadding
        });
        return true;
    }

    function focusHeadDetailTarget({
        animate = true,
        durationMs = 360
    } = {}) {
        if (!state.headDetailDepthActive || !hasSceneBackedDetailTarget()) return false;
        const bounds = resolveHeadDetailFocusBounds();
        if (!bounds) return false;
        const padding = VIEW2D_HEAD_DETAIL_COMPONENT_FOCUS_PADDING;
        state.focusLabel = resolveHeadDetailFocusLabel();
        updateReadouts();
        if (animate) {
            detailViewportController.flyToBounds(bounds, {
                animate: true,
                durationMs,
                now: performance.now(),
                padding
            });
            animateViewport();
            return true;
        }
        detailViewportController.fitToBounds(bounds, {
            padding
        });
        return true;
    }

    function maybeStartStagedFocusTransition() {
        const transition = state.stagedFocusTransition;
        if (
            !transition
            || transition.phase !== 'overview-hold'
            || !state.visible
        ) {
            return false;
        }
        return scheduleStagedFocusPhase(VIEW2D_STAGED_FOCUS_OVERVIEW_HOLD_MS, (now, activeTransition) => {
            const latestTransition = state.stagedFocusTransition;
            if (
                !latestTransition
                || latestTransition.phase !== 'overview-hold'
                || !state.visible
            ) {
                return;
            }
            state.stagedFocusTransition = {
                ...activeTransition,
                phase: 'overview-to-focus',
                phaseStartTime: now
            };
            const didFocus = focusSelection({
                animate: true,
                durationMs: VIEW2D_STAGED_FOCUS_OVERVIEW_TO_TARGET_DURATION_MS
            });
            clearStagedFocusTransition();
            if (!didFocus) {
                fitScene({ animate: true });
            }
        });
    }

    function activateStagedHeadDetailDepth() {
        if (!state.headDetailTarget) return false;
        const didSyncViewport = syncHeadDetailViewport({
            forceFit: true,
            animate: false
        });
        if (!didSyncViewport) return false;
        state.headDetailDepthActive = true;
        state.headDetailDepthAutoReentryBlocked = false;
        state.focusLabel = resolveActiveFocusLabel(state);
        updateReadouts();
        render();
        return true;
    }

    function maybeStartStagedHeadDetailTransition() {
        const transition = state.stagedHeadDetailTransition;
        if (
            !transition
            || transition.phase !== 'overview-hold'
            || !state.visible
        ) {
            return false;
        }
        return scheduleStagedHeadDetailPhase(VIEW2D_STAGED_HEAD_DETAIL_OVERVIEW_HOLD_MS, (now, activeTransition) => {
            const latestTransition = state.stagedHeadDetailTransition;
            if (
                !latestTransition
                || latestTransition.phase !== 'overview-hold'
                || !state.visible
            ) {
                return;
            }
            state.stagedHeadDetailTransition = {
                ...activeTransition,
                phase: 'overview-to-head',
                phaseStartTime: now
            };
            const didOpenHeadDetail = openHeadDetail(activeTransition.headDetailTarget, {
                animate: true,
                focusDurationMs: VIEW2D_STAGED_HEAD_DETAIL_OVERVIEW_TO_HEAD_DURATION_MS
            });
            if (!didOpenHeadDetail) {
                clearStagedHeadDetailTransition();
                return;
            }
            scheduleStagedHeadDetailPhase(VIEW2D_STAGED_HEAD_DETAIL_OVERVIEW_TO_HEAD_DURATION_MS, () => {
                const latestTransition = state.stagedHeadDetailTransition;
                if (
                    !latestTransition
                    || latestTransition.phase !== 'overview-to-head'
                    || !state.visible
                    || !state.headDetailTarget
                ) {
                    return;
                }
                const didActivateDepth = activateStagedHeadDetailDepth();
                if (!didActivateDepth) {
                    clearStagedHeadDetailTransition();
                    return;
                }
                if (!shouldKeepHeadDetailSceneFitView()) {
                    focusHeadDetailTarget({
                        animate: true,
                        durationMs: 420
                    });
                }
                clearStagedHeadDetailTransition();
            });
        });
    }

    function openStagedSceneDetailTarget(detailTargets = null) {
        const resolvedTargets = detailTargets && typeof detailTargets === 'object'
            ? detailTargets
            : {};
        if (resolvedTargets.outputProjectionDetailTarget) {
            return openOutputProjectionDetail(resolvedTargets.outputProjectionDetailTarget, {
                animate: true
            });
        }
        if (resolvedTargets.mlpDetailTarget) {
            return openMlpDetail(resolvedTargets.mlpDetailTarget, {
                animate: true
            });
        }
        if (resolvedTargets.layerNormDetailTarget) {
            return openLayerNormDetail(resolvedTargets.layerNormDetailTarget, {
                animate: true
            });
        }
        if (resolvedTargets.concatDetailTarget) {
            return openConcatDetail(resolvedTargets.concatDetailTarget, {
                animate: true
            });
        }
        return false;
    }

    function maybeStartStagedDetailTransition() {
        const transition = state.stagedDetailTransition;
        if (
            !transition
            || transition.phase !== 'overview-hold'
            || !state.visible
        ) {
            return false;
        }
        return scheduleStagedDetailPhase(VIEW2D_STAGED_FOCUS_OVERVIEW_HOLD_MS, (now, activeTransition) => {
            const latestTransition = state.stagedDetailTransition;
            if (
                !latestTransition
                || latestTransition.phase !== 'overview-hold'
                || !state.visible
            ) {
                return;
            }
            state.stagedDetailTransition = {
                ...activeTransition,
                phase: 'overview-to-focus',
                phaseStartTime: now
            };
            const didFocus = focusSelection({
                animate: true,
                durationMs: VIEW2D_STAGED_FOCUS_OVERVIEW_TO_TARGET_DURATION_MS
            });
            if (!didFocus) {
                clearStagedDetailTransition();
                return;
            }
            scheduleStagedDetailPhase(VIEW2D_STAGED_FOCUS_OVERVIEW_TO_TARGET_DURATION_MS, () => {
                const latestStage = state.stagedDetailTransition;
                if (
                    !latestStage
                    || latestStage.phase !== 'overview-to-focus'
                    || !state.visible
                ) {
                    return;
                }
                openStagedSceneDetailTarget(latestStage.detailTargets);
                clearStagedDetailTransition();
            });
        });
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
            || !hasSceneBackedDetailTarget()
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
        const effectiveViewportRect = controller?.getEffectiveViewportRect?.() || null;
        const width = Number(effectiveViewportRect?.width) || 0;
        const height = Number(effectiveViewportRect?.height) || 0;
        if (!(width > 0) || !(height > 0)) return null;
        const originX = Number(effectiveViewportRect?.x) || 0;
        const originY = Number(effectiveViewportRect?.y) || 0;
        const topLeft = controller.screenToWorld(originX, originY);
        const bottomRight = controller.screenToWorld(originX + width, originY + height);
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

    function resetHoverOverviewBlend() {
        const hoverDimming = state.hoverDimming;
        if (!hoverDimming) return;
        hoverDimming.previousOverviewFocus = null;
        hoverDimming.overviewBlend = 1;
        hoverDimming.overviewBlendTarget = 1;
    }

    function ensureHoverDimmingAnimation() {
        const hoverDimming = state.hoverDimming;
        if (!hoverDimming || hoverDimming.rafId !== null) return;
        const shouldReleasePreviousOverviewFocus = ({
            dimValue = hoverDimming.value,
            dimTarget = hoverDimming.target,
            overviewBlendValue = hoverDimming.overviewBlend,
            overviewBlendTarget = hoverDimming.overviewBlendTarget
        } = {}) => {
            if (overviewBlendTarget < 0.999 || overviewBlendValue < 0.999) {
                return false;
            }
            if (state.overviewSceneFocus) {
                return true;
            }
            return dimTarget <= 0.01 && dimValue <= 0.01;
        };
        const dimSettled = Math.abs(hoverDimming.target - hoverDimming.value) <= 0.01;
        const rowBlendSettled = Math.abs(hoverDimming.rowBlendTarget - hoverDimming.rowBlend) <= 0.01;
        const overviewBlendSettled = Math.abs(hoverDimming.overviewBlendTarget - hoverDimming.overviewBlend) <= 0.01;
        if (dimSettled && rowBlendSettled && overviewBlendSettled) {
            hoverDimming.value = hoverDimming.target;
            hoverDimming.rowBlend = hoverDimming.rowBlendTarget;
            if (hoverDimming.rowBlendTarget >= 1) {
                hoverDimming.previousHoveredRow = null;
            }
            hoverDimming.overviewBlend = hoverDimming.overviewBlendTarget;
            if (shouldReleasePreviousOverviewFocus({
                dimValue: hoverDimming.value,
                dimTarget: hoverDimming.target,
                overviewBlendValue: hoverDimming.overviewBlend,
                overviewBlendTarget: hoverDimming.overviewBlendTarget
            })) {
                hoverDimming.previousOverviewFocus = null;
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
                hoverDimming.overviewBlend = hoverDimming.overviewBlendTarget;
                if (shouldReleasePreviousOverviewFocus({
                    dimValue: hoverDimming.value,
                    dimTarget: hoverDimming.target,
                    overviewBlendValue: hoverDimming.overviewBlend,
                    overviewBlendTarget: hoverDimming.overviewBlendTarget
                })) {
                    hoverDimming.previousOverviewFocus = null;
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
            hoverDimming.overviewBlend += (hoverDimming.overviewBlendTarget - hoverDimming.overviewBlend) * alpha;

            if (Math.abs(hoverDimming.target - hoverDimming.value) <= 0.01) {
                hoverDimming.value = hoverDimming.target;
            }
            if (Math.abs(hoverDimming.rowBlendTarget - hoverDimming.rowBlend) <= 0.01) {
                hoverDimming.rowBlend = hoverDimming.rowBlendTarget;
                if (hoverDimming.rowBlendTarget >= 1) {
                    hoverDimming.previousHoveredRow = null;
                }
            }
            if (Math.abs(hoverDimming.overviewBlendTarget - hoverDimming.overviewBlend) <= 0.01) {
                hoverDimming.overviewBlend = hoverDimming.overviewBlendTarget;
                if (shouldReleasePreviousOverviewFocus({
                    dimValue: hoverDimming.value,
                    dimTarget: hoverDimming.target,
                    overviewBlendValue: hoverDimming.overviewBlend,
                    overviewBlendTarget: hoverDimming.overviewBlendTarget
                })) {
                    hoverDimming.previousOverviewFocus = null;
                }
            }

            const nextDimSettled = Math.abs(hoverDimming.target - hoverDimming.value) <= 0.01;
            const nextRowBlendSettled = Math.abs(hoverDimming.rowBlendTarget - hoverDimming.rowBlend) <= 0.01;
            const nextOverviewBlendSettled = Math.abs(hoverDimming.overviewBlendTarget - hoverDimming.overviewBlend) <= 0.01;
            render();
            if (nextDimSettled && nextRowBlendSettled && nextOverviewBlendSettled) {
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
                resetHoverOverviewBlend();
            }
            if (shouldRender) scheduleRender();
            return;
        }

        if (
            Math.abs(hoverDimming.value - nextTarget) < 0.001
            && Math.abs(hoverDimming.rowBlendTarget - hoverDimming.rowBlend) < 0.001
            && Math.abs(hoverDimming.overviewBlendTarget - hoverDimming.overviewBlend) < 0.001
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

    function snapshotHoverEvent(event = null) {
        if (!Number.isFinite(event?.clientX) || !Number.isFinite(event?.clientY)) {
            return null;
        }
        return {
            clientX: Number(event.clientX),
            clientY: Number(event.clientY),
            pointerId: Number.isFinite(event?.pointerId) ? Number(event.pointerId) : null,
            pointerType: String(event?.pointerType || '')
        };
    }

    function buildCanvasHoverTargetKey(hit = null, mode = '') {
        const nodeId = typeof hit?.node?.id === 'string' ? hit.node.id : '';
        if (!nodeId.length) {
            return mode ? `${mode}:none` : 'none';
        }

        const parts = [mode || 'hover', nodeId];
        if (Number.isFinite(hit?.cellHit?.rowIndex) && Number.isFinite(hit?.cellHit?.colIndex)) {
            parts.push(`cell:${Math.max(0, Math.floor(hit.cellHit.rowIndex))}:${Math.max(0, Math.floor(hit.cellHit.colIndex))}`);
        }
        if (Number.isFinite(hit?.rowHit?.rowIndex)) {
            parts.push(`row:${Math.max(0, Math.floor(hit.rowHit.rowIndex))}`);
        }
        if (Number.isFinite(hit?.columnHit?.colIndex)) {
            parts.push(`col:${Math.max(0, Math.floor(hit.columnHit.colIndex))}`);
        }
        return parts.join('|');
    }

    function resetCanvasHoverTargetKey() {
        state.hoverTargetKey = '';
    }

    function cancelScheduledHoverUpdate() {
        state.pendingHoverEvent = null;
        if (state.hoverFrame === null) return;
        cancelAnimationFrame(state.hoverFrame);
        state.hoverFrame = null;
    }

    function clearPinnedDetailSceneFocus({ scheduleRender: shouldScheduleRender = true } = {}) {
        const hadPinnedFocus = hasTransformerView2dLockedDetailSelection(state.detailScenePinnedFocus);
        state.detailScenePinnedFocus = null;
        state.detailScenePinnedSignature = '';
        state.detailScenePinnedTokenEntries = null;
        state.detailScenePinnedTokenSticky = false;
        state.detailSceneLockActive = false;
        state.detailSceneFocus = null;
        state.detailSceneHoverSignature = '';
        state.overviewSceneFocus = null;
        state.overviewSceneHoverSignature = '';
        resetOverviewSelectionArm();
        resetHoverOverviewBlend();
        resetCanvasHoverTargetKey();
        if (hadPinnedFocus) {
            tokenHoverSync.clearCanvasEntry({ emit: true });
        }
        hoverLabelOverlay.hide();
        if (hadPinnedFocus && shouldScheduleRender) {
            scheduleRender();
        }
        return hadPinnedFocus;
    }

    function clearPinnedOverviewSceneFocus({ scheduleRender: shouldScheduleRender = true } = {}) {
        const hadPinnedFocus = !!state.overviewScenePinnedFocus;
        const hadOverviewFocus = !!state.overviewSceneFocus;
        const hadResidualHover = !!state.hoveredResidualRow;
        state.overviewScenePinnedFocus = null;
        state.overviewScenePinnedSignature = '';
        state.overviewScenePinnedTokenEntries = null;
        state.overviewSceneLockActive = false;
        state.overviewSceneFocus = null;
        state.overviewSceneHoverSignature = '';
        state.hoveredResidualRow = null;
        resetOverviewSelectionArm();
        resetHoverOverviewBlend();
        resetHoverRowBlend();
        resetCanvasHoverTargetKey();
        tokenHoverSync.clearCanvasEntry({ emit: true });
        hoverLabelOverlay.hide();
        setHoverDimmingTarget(0, {
            immediate: !state.visible,
            shouldRender: false
        });
        if ((hadPinnedFocus || hadOverviewFocus || hadResidualHover) && shouldScheduleRender) {
            scheduleRender();
        }
        return hadPinnedFocus;
    }

    function clearPinnedSceneSelectionLocks({ scheduleRender: shouldScheduleRender = true } = {}) {
        const didClearDetailSelectionLock = clearPinnedDetailSceneFocus({ scheduleRender: false });
        const didClearOverviewSelectionLock = clearPinnedOverviewSceneFocus({ scheduleRender: false });
        if ((didClearDetailSelectionLock || didClearOverviewSelectionLock) && shouldScheduleRender) {
            scheduleRender();
        }
        return didClearDetailSelectionLock || didClearOverviewSelectionLock;
    }

    function lockPinnedDetailSceneFocus(detailHoverState = null, {
        scheduleRender: shouldScheduleRender = true,
        persistTokenChip = false,
        lockSelection = true
    } = {}) {
        if (!detailHoverState?.focusState) return false;
        const pinnedTokenEntries = resolveTransformerView2dTokenEntriesFromHoverPayload(detailHoverState);
        const nextSignature = typeof detailHoverState.signature === 'string'
            ? detailHoverState.signature
            : '';
        const didChange = (
            !state.detailScenePinnedFocus
            || state.detailScenePinnedSignature !== nextSignature
        );
        state.hoveredResidualRow = null;
        resetHoverRowBlend();
        setHoverDimmingTarget(0, {
            immediate: true,
            shouldRender: false
        });
        state.detailScenePinnedFocus = detailHoverState.focusState;
        state.detailScenePinnedSignature = nextSignature;
        state.detailScenePinnedTokenEntries = pinnedTokenEntries;
        state.detailScenePinnedTokenSticky = persistTokenChip === true && pinnedTokenEntries.length > 0;
        state.detailSceneLockActive = lockSelection === true;
        state.detailSceneFocus = detailHoverState.focusState;
        state.detailSceneHoverSignature = nextSignature;
        state.overviewSceneFocus = null;
        state.overviewSceneHoverSignature = '';
        resetOverviewSelectionArm();
        resetHoverOverviewBlend();
        resetCanvasHoverTargetKey();
        if (state.detailScenePinnedTokenSticky) {
            tokenHoverSync.setCanvasEntry(pinnedTokenEntries, { emit: true });
        } else {
            tokenHoverSync.clearCanvasEntry({ emit: true });
        }
        hoverLabelOverlay.hide();
        if (didChange && shouldScheduleRender) {
            scheduleRender();
        }
        return true;
    }

    function lockPinnedOverviewSceneFocus(overviewFocusState = null, {
        scheduleRender: shouldScheduleRender = true
    } = {}) {
        if (!overviewFocusState?.focusState) return false;
        const pinnedTokenEntries = resolveTransformerView2dTokenEntriesFromHoverPayload(overviewFocusState);
        const nextSignature = typeof overviewFocusState.signature === 'string'
            ? overviewFocusState.signature
            : '';
        const didChange = (
            !state.overviewScenePinnedFocus
            || state.overviewScenePinnedSignature !== nextSignature
        );
        state.hoveredResidualRow = null;
        resetHoverRowBlend();
        state.detailSceneFocus = state.detailScenePinnedFocus || null;
        state.detailSceneHoverSignature = state.detailScenePinnedSignature || '';
        state.overviewScenePinnedFocus = overviewFocusState.focusState;
        state.overviewScenePinnedSignature = nextSignature;
        state.overviewScenePinnedTokenEntries = pinnedTokenEntries;
        state.overviewSceneLockActive = true;
        state.overviewSceneFocus = overviewFocusState.focusState;
        state.overviewSceneHoverSignature = nextSignature;
        resetOverviewSelectionArm();
        resetHoverOverviewBlend();
        resetCanvasHoverTargetKey();
        setHoverDimmingTarget(1, {
            immediate: true,
            shouldRender: false
        });
        if (pinnedTokenEntries.length) {
            tokenHoverSync.setCanvasEntry(pinnedTokenEntries, { emit: true });
        } else {
            tokenHoverSync.clearCanvasEntry({ emit: true });
        }
        hoverLabelOverlay.hide();
        if (didChange && shouldScheduleRender) {
            scheduleRender();
        }
        return true;
    }

    function clearCanvasHover({
        scheduleRender: shouldScheduleRender = true,
        force = false,
        preserveSelectionArm = false
    } = {}) {
        if (!preserveSelectionArm) {
            resetOverviewSelectionArm();
        }
        if (state.detailScenePinnedFocus && force !== true) {
            const hadResidualHover = !!state.hoveredResidualRow;
            const hadOverviewFocus = !!state.overviewSceneFocus;
            if (state.detailScenePinnedTokenSticky && state.detailScenePinnedTokenEntries) {
                tokenHoverSync.setCanvasEntry(state.detailScenePinnedTokenEntries, { emit: true });
            } else {
                tokenHoverSync.clearCanvasEntry({ emit: true });
            }
            resetCanvasHoverTargetKey();
            state.hoveredResidualRow = null;
            state.overviewSceneFocus = null;
            state.overviewSceneHoverSignature = '';
            resetHoverOverviewBlend();
            state.detailSceneFocus = state.detailScenePinnedFocus;
            state.detailSceneHoverSignature = state.detailScenePinnedSignature;
            hoverLabelOverlay.hide();
            resetHoverRowBlend();
            setHoverDimmingTarget(0, {
                immediate: !state.visible,
                shouldRender: false
            });
            if ((hadResidualHover || hadOverviewFocus) && shouldScheduleRender) {
                scheduleRender();
            }
            return;
        }
        if (state.overviewScenePinnedFocus && force !== true) {
            const hadResidualHover = !!state.hoveredResidualRow;
            const hadOverviewFocus = !!state.overviewSceneFocus;
            if (state.overviewScenePinnedTokenEntries?.length) {
                tokenHoverSync.setCanvasEntry(state.overviewScenePinnedTokenEntries, { emit: true });
            } else {
                tokenHoverSync.clearCanvasEntry({ emit: true });
            }
            resetCanvasHoverTargetKey();
            state.hoveredResidualRow = null;
            state.detailSceneFocus = state.detailScenePinnedFocus || null;
            state.detailSceneHoverSignature = state.detailScenePinnedSignature || '';
            resetHoverRowBlend();
            resetHoverOverviewBlend();
            state.overviewSceneFocus = state.overviewScenePinnedFocus;
            state.overviewSceneHoverSignature = state.overviewScenePinnedSignature;
            hoverLabelOverlay.hide();
            setHoverDimmingTarget(1, {
                immediate: !state.visible,
                shouldRender: false
            });
            if ((hadResidualHover || hadOverviewFocus) && shouldScheduleRender) {
                scheduleRender();
            }
            return;
        }
        tokenHoverSync.clearCanvasEntry({ emit: true });
        resetCanvasHoverTargetKey();
        const hadResidualHover = !!state.hoveredResidualRow;
        const hadDetailHover = !!state.detailSceneFocus;
        const hadOverviewFocus = !!state.overviewSceneFocus || !!state.hoverDimming?.previousOverviewFocus;
        const shouldAnimateOverviewFade = hadOverviewFocus && !(!state.visible || force === true);
        state.hoveredResidualRow = null;
        if (shouldAnimateOverviewFade) {
            state.hoverDimming.previousOverviewFocus = state.overviewSceneFocus || state.hoverDimming.previousOverviewFocus || null;
            state.hoverDimming.overviewBlend = 1;
            state.hoverDimming.overviewBlendTarget = 1;
        } else {
            resetHoverOverviewBlend();
        }
        state.overviewSceneFocus = null;
        state.overviewSceneHoverSignature = '';
        state.detailSceneFocus = null;
        state.detailSceneHoverSignature = '';
        hoverLabelOverlay.hide();
        resetHoverRowBlend();
        setHoverDimmingTarget(0, {
            immediate: !state.visible,
            shouldRender: shouldScheduleRender || hadResidualHover || hadDetailHover || hadOverviewFocus
        });
        if ((hadResidualHover || hadDetailHover || hadOverviewFocus) && shouldScheduleRender) {
            scheduleRender();
        }
    }

    function updateCanvasHover(event = null) {
        const allowDetailSceneHover = !!(
            state.headDetailDepthActive
            && hasSceneBackedDetailTarget()
            && state.detailSceneIndex
        );
        const freezeDetailSceneHover = shouldFreezeTransformerView2dDetailHover({
            allowDetailSceneHover,
            detailSceneSelectionLocked: state.detailSceneLockActive
        });
        const freezeOverviewHover = !!(
            !state.headDetailDepthActive
            && state.overviewSceneLockActive
            && state.overviewScenePinnedFocus
        );
        const suppressOverviewHover = !!(
            state.headDetailDepthActive
            && hasActiveDetailTarget()
            && !state.detailSceneIndex
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
        if (freezeDetailSceneHover) {
            clearCanvasHover({ scheduleRender: false });
            return hit?.entry || null;
        }
        if (freezeOverviewHover) {
            clearCanvasHover({ scheduleRender: false });
            return hit?.entry || null;
        }
        if (allowDetailSceneHover) {
            const detailHoverKey = buildCanvasHoverTargetKey(hit, 'detail');
            if (detailHoverKey === state.hoverTargetKey) {
                hoverLabelOverlay.move({
                    clientX: event.clientX,
                    clientY: event.clientY
                });
                return hit?.entry || null;
            }

            const detailHoverState = resolveMhsaDetailHoverState(state.detailSceneIndex, hit, {
                interactionKind: 'hover'
            });
            if (!detailHoverState?.focusState) {
                clearCanvasHover();
                return hit?.entry || null;
            }

            const hoverInfo = detailHoverState?.info || null;
            const hoverLabel = normalizeCanvasHoverLabel(
                detailHoverState?.label || '',
                hoverInfo
            );
            resetOverviewSelectionArm();
            tokenHoverSync.setCanvasEntryFromHoverPayload(detailHoverState, { emit: true });
            const didChange = detailHoverState.signature !== state.detailSceneHoverSignature;
            state.hoveredResidualRow = null;
            state.overviewSceneFocus = null;
            state.overviewSceneHoverSignature = '';
            resetHoverOverviewBlend();
            resetHoverRowBlend();
            setHoverDimmingTarget(0, {
                immediate: true,
                shouldRender: false
            });
            state.detailSceneFocus = detailHoverState.focusState;
            state.detailSceneHoverSignature = detailHoverState.signature;
            state.hoverTargetKey = detailHoverKey;
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
            const residualHoverKey = buildCanvasHoverTargetKey(worldHit, 'overview-row');
            if (residualHoverKey === state.hoverTargetKey) {
                tokenHoverSync.setCanvasEntryFromResidualHoverPayload(residualHoverPayload, { emit: true });
                hoverLabelOverlay.move({
                    clientX: event.clientX,
                    clientY: event.clientY
                });
                return worldHit?.entry || null;
            }
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
            const hadOverviewFocus = !!state.overviewSceneFocus;
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
            state.overviewSceneFocus = null;
            state.overviewSceneHoverSignature = '';
            resetHoverOverviewBlend();
            state.hoverTargetKey = residualHoverKey;
            tokenHoverSync.setCanvasEntryFromResidualHoverPayload(residualHoverPayload, { emit: true });
            setHoverDimmingTarget(1, {
                shouldRender: didChange || hadOverviewFocus
            });
            hoverLabelOverlay.show({
                clientX: event.clientX,
                clientY: event.clientY,
                label: hoverLabel,
                info: residualHoverPayload.info,
                activationSource: state.activationSource
            });
            if (didChange || hadOverviewFocus) {
                scheduleRender();
            }
            return worldHit?.entry || null;
        }

        resetOverviewSelectionArm();
        const semanticHoverPayload = buildSemanticNodeHoverPayload(worldHit);
        if (!semanticHoverPayload?.label) {
            clearCanvasHover();
            return worldHit?.entry || hit?.entry || null;
        }

        const semanticHoverKey = buildCanvasHoverTargetKey(worldHit, 'overview-node');
        if (semanticHoverKey === state.hoverTargetKey) {
            tokenHoverSync.setCanvasEntryFromHoverPayload(semanticHoverPayload, { emit: true });
            hoverLabelOverlay.move({
                clientX: event.clientX,
                clientY: event.clientY
            });
            return worldHit?.entry || null;
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
        const previousOverviewFocusSignature = state.overviewSceneHoverSignature || '';
        const semanticHoverFocusState = buildSemanticNodeHoverFocusState(state.scene, worldHit);
        const previousOverviewFocusState = state.overviewSceneFocus || null;
        state.hoveredResidualRow = null;
        state.detailSceneFocus = state.detailScenePinnedFocus || null;
        state.detailSceneHoverSignature = state.detailScenePinnedSignature || '';
        state.overviewSceneFocus = semanticHoverFocusState?.focusState || null;
        state.overviewSceneHoverSignature = semanticHoverFocusState?.signature || '';
        if (
            previousOverviewFocusState
            && previousOverviewFocusSignature
            && previousOverviewFocusSignature !== state.overviewSceneHoverSignature
        ) {
            state.hoverDimming.previousOverviewFocus = previousOverviewFocusState;
            state.hoverDimming.overviewBlend = 0;
            state.hoverDimming.overviewBlendTarget = 1;
        } else if (!state.overviewSceneFocus) {
            resetHoverOverviewBlend();
        } else {
            state.hoverDimming.previousOverviewFocus = null;
            state.hoverDimming.overviewBlend = 1;
            state.hoverDimming.overviewBlendTarget = 1;
        }
        tokenHoverSync.setCanvasEntryFromHoverPayload(semanticHoverPayload, { emit: true });
        resetHoverRowBlend();
        setHoverDimmingTarget(1, {
            immediate: !state.visible,
            shouldRender: false
        });
        state.hoverTargetKey = semanticHoverKey;
        hoverLabelOverlay.show({
            clientX: event.clientX,
            clientY: event.clientY,
            label: hoverLabel,
            info: semanticHoverPayload.info,
            activationSource: state.activationSource
        });
        if (
            hadResidualHover
            || previousOverviewFocusSignature !== state.overviewSceneHoverSignature
            || Math.abs((state.hoverDimming?.value || 0) - 1) > 0.001
        ) {
            scheduleRender();
        }
        return worldHit?.entry || null;
    }

    function processCanvasHoverEvent(event = null) {
        const hoveredEntry = updateCanvasHover(event) || resolveCanvasPointerHit(event);
        updateCanvasCursor(hoveredEntry);
    }

    function scheduleCanvasHoverUpdate(event = null) {
        const hoverEvent = snapshotHoverEvent(event);
        if (!hoverEvent || hoverEvent.pointerType === 'touch') {
            cancelScheduledHoverUpdate();
            clearCanvasHover();
            updateCanvasCursor(null);
            return;
        }

        // Passive hover should keep the high-detail canvas path active.
        clearInteractionTimer();
        setInteractionActive(false);

        if (state.hoverFrame !== null) {
            state.pendingHoverEvent = hoverEvent;
            return;
        }

        processCanvasHoverEvent(hoverEvent);
        state.hoverFrame = requestAnimationFrame(() => {
            state.hoverFrame = null;
            const nextHoverEvent = state.pendingHoverEvent;
            state.pendingHoverEvent = null;
            if (!nextHoverEvent) return;
            scheduleCanvasHoverUpdate(nextHoverEvent);
        });
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
        const isHeadDetailSceneActive = isDetailDeepActive && hasSceneBackedDetailTarget();
        const isConcatDetailActive = isDetailDeepActive && !!state.concatDetailTarget;
        const canReturnToGraphOverview = !isTransformerView2dGraphOverview();
        root.classList.toggle('is-head-detail-active', isHeadDetailActive);
        root.classList.toggle('is-head-detail-scene-active', isHeadDetailSceneActive);
        root.classList.toggle('is-concat-detail-active', isConcatDetailActive);
        canvasCard?.classList.toggle('is-head-detail-active', isHeadDetailActive);
        canvasCard?.classList.toggle('is-head-detail-scene-active', isHeadDetailSceneActive);
        canvasCard?.classList.toggle('is-concat-detail-active', isConcatDetailActive);
        canvas?.classList.toggle('is-head-detail-active', isHeadDetailActive);
        canvas?.classList.toggle('is-head-detail-scene-active', isHeadDetailSceneActive);
        canvas?.classList.toggle('is-concat-detail-active', isConcatDetailActive);
        if (backToGraphBtn) {
            backToGraphBtn.hidden = !canReturnToGraphOverview;
            backToGraphBtn.setAttribute('aria-hidden', canReturnToGraphOverview ? 'false' : 'true');
        }
        if (hud) {
            hud.hidden = false;
            hud.setAttribute('aria-hidden', 'false');
        }
    }

    function resolveFitSceneActionState() {
        if (state.headDetailDepthActive && hasSceneBackedDetailTarget()) {
            return {
                controllerState: detailViewportController.getState(),
                fitBounds: resolveHeadDetailSceneBounds(),
                padding: VIEW2D_HEAD_DETAIL_VIEWPORT_PADDING,
                minScale: VIEW2D_HEAD_DETAIL_VIEWPORT_MIN_SCALE,
                maxScale: VIEW2D_DETAIL_VIEWPORT_MAX_SCALE
            };
        }

        return {
            controllerState: viewportController.getState(),
            fitBounds: state.layout?.sceneBounds || null,
            padding: VIEW2D_DETAIL_VIEWPORT_PADDING,
            minScale: VIEW2D_DETAIL_VIEWPORT_MIN_SCALE,
            maxScale: VIEW2D_DETAIL_VIEWPORT_MAX_SCALE
        };
    }

    function isViewportAtFitScene() {
        if (!state.visible || !state.scene || !state.layout) return false;
        return isTransformerView2dViewportAtFitScene(resolveFitSceneActionState());
    }

    function syncFitSceneActionVisibility() {
        if (!fitBtn) return;

        const shouldShowFitAction = !!(
            state.visible
            && state.scene
            && state.layout
            && shouldShowTransformerView2dFitSceneAction(resolveFitSceneActionState())
        );
        const nextVisibility = shouldShowFitAction ? 'true' : 'false';

        if (fitBtn.dataset.fitVisible !== nextVisibility) {
            fitBtn.dataset.fitVisible = nextVisibility;
        }

        fitBtn.disabled = !shouldShowFitAction;
        fitBtn.setAttribute('aria-hidden', shouldShowFitAction ? 'false' : 'true');
        if (shouldShowFitAction) {
            fitBtn.removeAttribute('tabindex');
        } else {
            fitBtn.tabIndex = -1;
        }
    }

    function syncViewportForSidebarVisibilityChange({
        animate = true,
        wasAtFitScene = false
    } = {}) {
        if (!state.visible) return false;
        if (animate) {
            stopAnimation();
        }
        measureCanvasSize();
        const { width, viewportInsets } = syncViewportControllers({
            preserveVisibleCenter: !wasAtFitScene,
            animate: animate && !wasAtFitScene,
            durationMs: VIEW2D_SELECTION_SIDEBAR_VIEWPORT_TRANSITION_MS
        });
        syncViewportControllerConstraints(width, viewportInsets);
        updateReadouts();
        if (wasAtFitScene) {
            return fitScene({
                animate,
                durationMs: VIEW2D_SELECTION_SIDEBAR_VIEWPORT_TRANSITION_MS
            });
        }
        if (animate && (viewportController.animation || detailViewportController.animation)) {
            animateViewport();
            return true;
        }
        scheduleRender();
        return true;
    }

    function computeHeadDetailFocusScale() {
        if (!hasActiveDetailTarget() || !state.layout?.registry) return null;
        const bounds = resolveSelectionFocusBounds();
        if (!bounds) return null;
        const { width, height, viewportInsets } = syncViewportControllers();
        const transform = resolveViewportFitTransform(bounds, { width, height }, {
            padding: VIEW2D_HEAD_DETAIL_FOCUS_PADDING,
            minScale: VIEW2D_DETAIL_VIEWPORT_MIN_SCALE,
            maxScale: VIEW2D_DETAIL_VIEWPORT_MAX_SCALE,
            viewportInsets
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

        const suppressAutoDepthActivation = (
            state.stagedHeadDetailTransition?.phase === 'overview-to-head'
            && !state.headDetailDepthActive
        );

        if (hasSceneBackedDetailTarget()) {
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
            if (suppressAutoDepthActivation) {
                return false;
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
            outputProjectionDetailTarget: state.outputProjectionDetailTarget,
            mlpDetailTarget: state.mlpDetailTarget,
            layerNormDetailTarget: state.layerNormDetailTarget,
            kvCacheState: {
                kvCacheModeEnabled: !!appState.kvCacheModeEnabled,
                kvCachePrefillActive: !!appState.kvCachePrefillActive,
                kvCacheDecodeActive: !!(appState.kvCacheModeEnabled && !appState.kvCachePrefillActive),
                kvCachePassIndex: Number.isFinite(appState.kvCachePassIndex)
                    ? Math.max(0, Math.floor(appState.kvCachePassIndex))
                    : 0
            }
        });
        const activeDetailScene = resolveActiveDetailScene(state.scene, {
            headDetailTarget: state.headDetailTarget,
            outputProjectionDetailTarget: state.outputProjectionDetailTarget,
            mlpDetailTarget: state.mlpDetailTarget,
            layerNormDetailTarget: state.layerNormDetailTarget
        });
        state.detailSceneIndex = activeDetailScene
            ? createMhsaDetailSceneIndex(activeDetailScene)
            : null;
        state.detailScenePinnedFocus = null;
        state.detailScenePinnedSignature = '';
        state.detailScenePinnedTokenEntries = null;
        state.detailScenePinnedTokenSticky = false;
        state.detailSceneLockActive = false;
        state.detailSceneFocus = null;
        state.detailSceneHoverSignature = '';
        state.overviewSceneFocus = null;
        state.overviewSceneHoverSignature = '';
        resetOverviewSelectionArm();
        resetHoverOverviewBlend();
        resetCanvasHoverTargetKey();
        tokenHoverSync.clearCanvasEntry({ emit: true });
        if (state.detailSceneIndex && state.pendingDetailInteractionTargets.length) {
            const initialDetailHoverState = resolveTransformerView2dDetailInteractionHoverState(
                state.detailSceneIndex,
                state.pendingDetailInteractionTargets
            );
            if (initialDetailHoverState?.focusState) {
                lockPinnedDetailSceneFocus(initialDetailHoverState, {
                    scheduleRender: false,
                    persistTokenChip: false,
                    lockSelection: false
                });
            }
            state.pendingDetailInteractionTargets = [];
        }
        state.layout = buildSceneLayout(state.scene, {
            isSmallScreen: state.isSmallScreen
        });
        renderer.setScene(state.scene, state.layout);
        viewportController.setSceneBounds(state.layout?.sceneBounds || null);
        return !!state.scene && !!state.layout;
    }

    function openHeadDetail(headDetailTarget = null, {
        animate = true,
        focusDurationMs = null,
        nextDepthActive = false
    } = {}) {
        const resolvedTarget = resolveHeadDetailTarget(headDetailTarget);
        if (!resolvedTarget) return false;
        stopAnimation();
        cancelScheduledHoverUpdate();
        clearCanvasHover({ scheduleRender: false });
        setDetailTargets({ headDetailTarget: resolvedTarget });
        return commitSceneSelection({
            animate,
            nextDepthActive,
            focusDurationMs
        });
    }

    function openConcatDetail(concatDetailTarget = null, { animate = true } = {}) {
        const resolvedTarget = resolveConcatDetailTarget(concatDetailTarget);
        if (!resolvedTarget) return false;
        stopAnimation();
        cancelScheduledHoverUpdate();
        clearCanvasHover({ scheduleRender: false });
        setDetailTargets({ concatDetailTarget: resolvedTarget });
        return commitSceneSelection({ animate, nextDepthActive: false });
    }

    function openOutputProjectionDetail(outputProjectionDetailTarget = null, { animate = true } = {}) {
        const resolvedTarget = resolveOutputProjectionDetailTarget(outputProjectionDetailTarget);
        if (!resolvedTarget) return false;
        stopAnimation();
        cancelScheduledHoverUpdate();
        clearCanvasHover({ scheduleRender: false });
        setDetailTargets({ outputProjectionDetailTarget: resolvedTarget });
        return commitSceneSelection({ animate, nextDepthActive: true });
    }

    function openMlpDetail(mlpDetailTarget = null, { animate = true } = {}) {
        const resolvedTarget = resolveMlpDetailTarget(mlpDetailTarget);
        if (!resolvedTarget) return false;
        stopAnimation();
        cancelScheduledHoverUpdate();
        clearCanvasHover({ scheduleRender: false });
        setDetailTargets({ mlpDetailTarget: resolvedTarget });
        return commitSceneSelection({ animate, nextDepthActive: true });
    }

    function openLayerNormDetail(layerNormDetailTarget = null, { animate = true } = {}) {
        const resolvedTarget = resolveLayerNormDetailTarget(layerNormDetailTarget);
        if (!resolvedTarget) return false;
        stopAnimation();
        cancelScheduledHoverUpdate();
        clearCanvasHover({ scheduleRender: false });
        setDetailTargets({ layerNormDetailTarget: resolvedTarget });
        return commitSceneSelection({ animate, nextDepthActive: true });
    }

    function returnToGraphOverview({ animate = true } = {}) {
        stopAnimation();
        clearStagedFocusTransition();
        clearStagedHeadDetailTransition();
        clearStagedDetailTransition();
        cancelScheduledHoverUpdate();
        clearCanvasHover({ scheduleRender: false });
        clearPinnedSceneSelectionLocks({ scheduleRender: false });
        const overviewState = buildTransformerView2dOverviewState();
        state.baseSemanticTarget = overviewState.baseSemanticTarget;
        state.baseFocusLabel = overviewState.baseFocusLabel;
        setDetailTargets(overviewState.detailTargets);
        setDetailFocusTarget({
            detailSemanticTargets: overviewState.detailSemanticTargets,
            detailFocusLabel: overviewState.detailFocusLabel
        });
        state.pendingDetailInteractionTargets = [...overviewState.pendingDetailInteractionTargets];
        return commitSceneSelection({
            animate,
            nextDepthActive: false
        });
    }

    function resolveDisplayedStageSemanticTarget() {
        return state.headDetailDepthActive
            ? state.semanticTarget
            : null;
    }

    function updateReadouts() {
        syncHeadDetailChrome();
        syncFitSceneActionVisibility();
        const stageHeader = resolveTransformerView2dStageHeader(resolveDisplayedStageSemanticTarget());
        if (stageLayerReadout) {
            setNodeText(stageLayerReadout, stageHeader.layerLabel);
            stageLayerReadout.hidden = !stageHeader.layerLabel;
        }
        if (stageTitleReadout) {
            setNodeText(stageTitleReadout, stageHeader.stageLabel);
            stageTitleReadout.setAttribute('aria-label', stageHeader.fullLabel);
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
            tokenHoverSync.clear({ emit: true });
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
                const chip = document.createElement('button');
                chip.type = 'button';
                chip.className = 'detail-transformer-view2d-token-strip__token prompt-token-strip__token detail-transformer-view2d-token';
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

        tokenHoverSync.applyState();
        tokenStrip.dataset.visible = state.visible ? 'true' : 'false';
        tokenStrip.setAttribute('aria-hidden', state.visible ? 'false' : 'true');
    }

    function render() {
        if (!state.visible || !state.scene || !state.layout) return false;
        const { width, height, viewportInsets } = syncViewportControllers();
        syncViewportControllerConstraints(width, viewportInsets);
        viewportController.setSceneBounds(state.layout.sceneBounds || null);
        const headDetailSceneBounds = resolveHeadDetailSceneBounds();
        detailViewportController.setSceneBounds(headDetailSceneBounds);
        updateHeadDetailDepthState();
        const hoverRenderFastPath = shouldUseOverviewHoverRenderFastPath();
        const viewportAnimationActive = !!(viewportController.animation || detailViewportController.animation);
        const useFastRenderPath = state.isInteracting || hoverRenderFastPath || viewportAnimationActive;
        const didRender = renderer.render({
            width,
            height,
            dprCap: useFastRenderPath
                ? VIEW2D_PREVIEW_DPR_CAP_INTERACTING
                : VIEW2D_PREVIEW_DPR_CAP_IDLE,
            viewportTransform: viewportController.getViewportTransform('detail-transformer-view2d'),
            detailViewportTransform: (
                state.headDetailDepthActive && hasSceneBackedDetailTarget()
                    ? detailViewportController.getViewportTransform('detail-transformer-view2d-head-detail')
                    : null
            ),
            interacting: useFastRenderPath,
            headDetailDepthActive: state.headDetailDepthActive,
            interactionState: {
                hoveredRow: state.hoveredResidualRow,
                previousHoveredRow: state.hoverDimming?.previousHoveredRow || null,
                hoverRowBlend: state.hoverDimming?.rowBlend ?? 1,
                hoverDimStrength: state.hoverDimming?.value || 0,
                detailSceneFocus: getActiveSceneFocusState(),
                disableInactiveFilter: hoverRenderFastPath,
                overviewFocusTransition: state.headDetailDepthActive
                    ? null
                    : {
                        currentFocus: state.overviewSceneFocus || null,
                        previousFocus: state.hoverDimming?.previousOverviewFocus || null,
                        focusBlend: state.hoverDimming?.overviewBlend ?? 1,
                        dimStrength: state.hoverDimming?.value || 0
                    }
            }
        });
        const activeSceneFocusState = getActiveSceneFocusState();
        const captionSceneState = renderer.getActiveCaptionSceneState();
        residualCaptionOverlay.sync({
            scene: captionSceneState?.scene || null,
            layout: captionSceneState?.layout || null,
            canvas,
            projectBounds: (bounds) => renderer.resolveScreenBounds(bounds),
            visible: state.visible,
            enabled: !viewportAnimationActive,
            focusState: activeSceneFocusState
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

    function fitScene({
        animate = true,
        durationMs = 420
    } = {}) {
        if (state.headDetailDepthActive && hasSceneBackedDetailTarget()) {
            const didFit = syncHeadDetailViewport({
                forceFit: true,
                animate,
                durationMs
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
                durationMs,
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
            outputProjectionDetailTarget: state.outputProjectionDetailTarget,
            mlpDetailTarget: state.mlpDetailTarget,
            layerNormDetailTarget: state.layerNormDetailTarget
        });
        for (const focusTarget of focusTargets) {
            const bounds = resolveSemanticTargetBounds(state.layout.registry, focusTarget);
            if (bounds) return bounds;
        }
        return null;
    }

    function focusSelection({
        animate = true,
        durationMs = null
    } = {}) {
        if (state.headDetailDepthActive && hasSceneBackedDetailTarget()) {
            if (shouldKeepHeadDetailSceneFitView()) {
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
            if (focusHeadDetailTarget({ animate })) {
                if (!animate) {
                    render();
                }
                return true;
            }
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

        if (shouldKeepSceneBackedDetailFitView()) {
            state.focusLabel = state.detailFocusLabel || resolveActiveFocusLabel(state);
            updateReadouts();
            return fitScene({ animate });
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
        const resolvedDurationMs = Number.isFinite(durationMs)
            ? Math.max(1, Math.floor(durationMs))
            : (hasActiveDetailTarget() ? 520 : 420);
        if (hasActiveDetailTarget()) {
            const { width, height, viewportInsets } = syncViewportControllers();
            const transform = resolveViewportFitTransform(bounds, { width, height }, {
                padding,
                minScale: VIEW2D_DETAIL_VIEWPORT_MIN_SCALE,
                maxScale: VIEW2D_DETAIL_VIEWPORT_MAX_SCALE,
                viewportInsets
            });
            state.headDetailFocusScale = Number.isFinite(transform?.scale) ? transform.scale : null;
            state.headDetailDepthActive = animate !== true;
        }
        if (animate) {
            viewportController.flyToBounds(bounds, {
                animate: true,
                durationMs: resolvedDurationMs,
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

    function resolveCanvasScreenHit(event = null) {
        if (!canvas || !Number.isFinite(event?.clientX) || !Number.isFinite(event?.clientY)) {
            return null;
        }
        const rect = canvas.getBoundingClientRect();
        return renderer.resolveInteractiveHitAtScreenPoint(
            event.clientX - rect.left,
            event.clientY - rect.top
        ) || null;
    }

    function updateCanvasCursor(entry = null) {
        if (!canvas) return;
        const semanticSelection = entry ? resolveCanvasSemanticSelection({ entry }) : null;
        canvas.style.cursor = (
            isMhsaHeadOverviewEntry(entry)
            || isConcatOverviewEntry(entry)
            || isOutputProjectionOverviewEntry(entry)
            || isMlpOverviewEntry(entry)
            || isLayerNormOverviewEntry(entry)
            || !!semanticSelection
        ) ? 'pointer' : '';
    }

    function shouldOpenCanvasSceneNode(entry = null) {
        return !!(
            isMhsaHeadOverviewEntry(entry)
            || isConcatOverviewEntry(entry)
            || isOutputProjectionOverviewEntry(entry)
            || isMlpOverviewEntry(entry)
            || isLayerNormOverviewEntry(entry)
        );
    }

    function onSceneNodeClick(entry = null) {
        resetOverviewSelectionArm();
        clearStagedHeadDetailTransition();
        clearStagedDetailTransition();
        const clearDetailFocusTarget = () => {
            setDetailFocusTarget({
                detailSemanticTargets: [],
                detailFocusLabel: ''
            });
            state.pendingDetailInteractionTargets = [];
        };
        if (isMhsaHeadOverviewEntry(entry)) {
            clearDetailFocusTarget();
            return openHeadDetail({
                layerIndex: entry.semantic.layerIndex,
                headIndex: entry.semantic.headIndex
            }, {
                animate: true,
                nextDepthActive: true
            });
        }
        if (isConcatOverviewEntry(entry)) {
            clearDetailFocusTarget();
            return openConcatDetail({
                layerIndex: entry.semantic.layerIndex
            }, {
                animate: true
            });
        }
        if (isOutputProjectionOverviewEntry(entry)) {
            clearDetailFocusTarget();
            return openOutputProjectionDetail({
                layerIndex: entry.semantic.layerIndex
            }, {
                animate: true
            });
        }
        if (isMlpOverviewEntry(entry)) {
            clearDetailFocusTarget();
            return openMlpDetail({
                layerIndex: entry.semantic.layerIndex
            }, {
                animate: true
            });
        }
        if (isLayerNormOverviewEntry(entry)) {
            clearDetailFocusTarget();
            return openLayerNormDetail({
                layerNormKind: entry.semantic.stage === 'final-ln' ? 'final' : entry.semantic.stage,
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
            const zoomFactor = Math.exp(zoomDir * safeDt * VIEW2D_KEYBOARD_ZOOM_RATE);
            // Use the controller's effective visible center so keyboard zoom stays
            // stable when the canvas viewport is inset by the docked sidebar.
            activeViewportController.zoomAt(zoomFactor);
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

        clearStagedHeadDetailTransition();
        clearStagedDetailTransition();
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
        clearPinnedSceneSelectionLocks({ scheduleRender: true });
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
        pointerType = '',
        clientX = 0,
        clientY = 0,
        suppressClick = false
    } = {}) {
        cancelScheduledHoverUpdate();
        const isTouchPointer = String(pointerType || '').toLowerCase() === 'touch';
        clearCanvasHover({
            scheduleRender: false,
            preserveSelectionArm: isTouchPointer && !!state.overviewSelectionArmSignature
        });
        state.pointer = {
            pointerId: Number.isFinite(pointerId) ? pointerId : null,
            pointerType: String(pointerType || ''),
            clientX: Number.isFinite(clientX) ? clientX : 0,
            clientY: Number.isFinite(clientY) ? clientY : 0,
            startClientX: Number.isFinite(clientX) ? clientX : 0,
            startClientY: Number.isFinite(clientY) ? clientY : 0,
            moved: false,
            suppressClick: suppressClick === true
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
        cancelScheduledHoverUpdate();
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
                    pointerType: 'touch',
                    clientX: point.clientX,
                    clientY: point.clientY,
                    suppressClick: true
                });
                return;
            }
        }
        canvas.classList.remove('is-panning');
        scheduleInteractionSettle();
    }

    function onPointerDown(event) {
        if (!state.visible || (Number.isFinite(event?.button) && event.button !== 0)) return;
        clearStagedHeadDetailTransition();
        clearStagedDetailTransition();
        focusCanvasSurface();
        if (event?.pointerType === 'touch') {
            trackTouchPointer(event);
            if (beginTouchPinch()) {
                resetOverviewSelectionArm();
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
            pointerType: event?.pointerType,
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
            scheduleCanvasHoverUpdate(event);
            return;
        }
        if (!state.pointer || state.pointer.pointerId !== event.pointerId) return;
        const pointerIntent = resolveView2dPointerMoveIntent({
            pointerType: state.pointer.pointerType,
            startClientX: state.pointer.startClientX,
            startClientY: state.pointer.startClientY,
            previousClientX: state.pointer.clientX,
            previousClientY: state.pointer.clientY,
            clientX: event.clientX,
            clientY: event.clientY,
            moved: state.pointer.moved,
            suppressClick: state.pointer.suppressClick
        });
        state.pointer.clientX = event.clientX;
        state.pointer.clientY = event.clientY;
        state.pointer.moved = pointerIntent.moved;
        state.pointer.suppressClick = pointerIntent.suppressClick;
        if (pointerIntent.shouldPan) {
            resetOverviewSelectionArm();
            getActiveViewportController().panBy(pointerIntent.deltaX, pointerIntent.deltaY);
            disableAutoFrameState();
            markInteraction(true);
            scheduleRender();
        }
        event.preventDefault();
    }

    function onWheel(event) {
        if (!state.visible) return;
        clearStagedHeadDetailTransition();
        clearStagedDetailTransition();
        focusCanvasSurface();
        cancelScheduledHoverUpdate();
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
        const pointerReleaseIntent = resolveView2dPointerMoveIntent({
            pointerType: state.pointer.pointerType,
            startClientX: state.pointer.startClientX,
            startClientY: state.pointer.startClientY,
            previousClientX: state.pointer.clientX,
            previousClientY: state.pointer.clientY,
            clientX: Number.isFinite(event?.clientX) ? event.clientX : state.pointer.clientX,
            clientY: Number.isFinite(event?.clientY) ? event.clientY : state.pointer.clientY,
            moved: state.pointer.moved,
            suppressClick: state.pointer.suppressClick
        });
        const shouldTreatAsClick = shouldTreatView2dPointerReleaseAsClick(pointerReleaseIntent);
        const clickedHit = shouldTreatAsClick ? resolveCanvasScreenHit(event) : null;
        const clickedEntry = clickedHit?.entry || null;
        clearPointer(event);
        updateCanvasCursor(clickedEntry);
        if (shouldTreatAsClick) {
            if (state.headDetailDepthActive && state.detailSceneIndex) {
                const detailHoverState = resolveMhsaDetailHoverState(state.detailSceneIndex, clickedHit, {
                    interactionKind: 'click'
                });
                const detailArmSignature = detailHoverState?.focusState && detailHoverState?.signature
                    ? `detail:${detailHoverState.signature}`
                    : '';
                if (shouldRequireTouchSelectionConfirmation(event, detailArmSignature)) {
                    applyTouchSelectionHover(event, {
                        armSignature: detailArmSignature,
                        detailHoverState
                    });
                    return;
                }
                if (detailArmSignature) {
                    resetOverviewSelectionArm();
                }
                const detailClickLockAction = resolveTransformerView2dDetailClickLockAction({
                    detailSceneSelectionLocked: state.detailSceneLockActive,
                    detailScenePinnedSignature: state.detailScenePinnedSignature,
                    detailHoverState
                });
                if (detailClickLockAction === TRANSFORMER_VIEW2D_DETAIL_CLICK_LOCK_ACTIONS.LOCK_TARGET) {
                    const didLockTarget = lockPinnedDetailSceneFocus(detailHoverState, {
                        scheduleRender: false,
                        persistTokenChip: true
                    });
                    requestSelectionOpen(detailHoverState);
                    if (didLockTarget) {
                        scheduleRender();
                    }
                    return;
                }
                if (detailClickLockAction === TRANSFORMER_VIEW2D_DETAIL_CLICK_LOCK_ACTIONS.CLEAR_LOCK) {
                    const didClearSelectionLock = clearPinnedDetailSceneFocus({ scheduleRender: false });
                    requestSelectionClose();
                    if (didClearSelectionLock) {
                        scheduleRender();
                    }
                    return;
                }
                if (detailClickLockAction === TRANSFORMER_VIEW2D_DETAIL_CLICK_LOCK_ACTIONS.IGNORE) {
                    if (requestSelectionOpen(detailHoverState)) {
                        return;
                    }
                    return;
                }
                if (clearPinnedDetailSceneFocus({ scheduleRender: true })) {
                    return;
                }
            }
            const residualSelection = buildResidualRowHoverPayload(clickedHit?.rowHit, state.activationSource);
            const residualSelectionFocusState = buildResidualRowSelectionFocusState(state.scene, clickedHit);
            const residualArmSignature = residualSelection
                ? buildCanvasHoverTargetKey(clickedHit, 'overview-row-selection')
                : '';
            if (residualSelection) {
                if (shouldRequireTouchSelectionConfirmation(event, residualArmSignature)) {
                    applyTouchSelectionHover(event, {
                        armSignature: residualArmSignature
                    });
                    return;
                }
                resetOverviewSelectionArm();
                if (requestSelectionOpen(residualSelection)) {
                    if (residualSelectionFocusState?.focusState) {
                        lockPinnedOverviewSceneFocus({
                            ...residualSelection,
                            ...residualSelectionFocusState
                        }, {
                            scheduleRender: true
                        });
                    } else {
                        clearPinnedOverviewSceneFocus({ scheduleRender: true });
                    }
                    return;
                }
            }
            const semanticSelection = resolveCanvasSemanticSelection(clickedHit);
            const semanticSelectionFocusState = semanticSelection
                ? buildSemanticNodeHoverFocusState(state.scene, clickedHit)
                : null;
            const semanticArmSignature = semanticSelection
                ? buildCanvasHoverTargetKey(clickedHit, 'overview-node-selection')
                : '';
            if (semanticSelection) {
                if (shouldRequireTouchSelectionConfirmation(event, semanticArmSignature)) {
                    applyTouchSelectionHover(event, {
                        armSignature: semanticArmSignature
                    });
                    return;
                }
                resetOverviewSelectionArm();
                if (requestSelectionOpen(semanticSelection)) {
                    if (semanticSelectionFocusState?.focusState) {
                        lockPinnedOverviewSceneFocus({
                            ...semanticSelection,
                            ...semanticSelectionFocusState
                        }, {
                            scheduleRender: true
                        });
                    } else {
                        clearPinnedOverviewSceneFocus({ scheduleRender: true });
                    }
                    return;
                }
            }
            const shouldOpenSceneNode = shouldOpenCanvasSceneNode(clickedEntry);
            if (shouldOpenSceneNode) {
                clearPinnedOverviewSceneFocus({ scheduleRender: false });
                onSceneNodeClick(clickedEntry);
                return;
            }
            clearPinnedOverviewSceneFocus({ scheduleRender: true });
        }
    }

    function onPointerLeave() {
        cancelScheduledHoverUpdate();
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
    canvas?.addEventListener('dblclick', (event) => {
        focusCanvasSurface();
        const clickedHit = resolveCanvasScreenHit(event);
        const clickedEntry = clickedHit?.entry || null;
        if (shouldOpenCanvasSceneNode(clickedEntry)) {
            clearPinnedOverviewSceneFocus({ scheduleRender: false });
            onSceneNodeClick(clickedEntry);
            event?.preventDefault?.();
            return;
        }
        if (shouldSuppressView2dDoubleClickFocus({
            headDetailDepthActive: state.headDetailDepthActive,
            hasActiveDetailTarget: hasActiveDetailTarget(),
            hasDetailSceneIndex: !!state.detailSceneIndex
        })) {
            event?.preventDefault?.();
            return;
        }
        focusSelection({ animate: true });
    });
    fitBtn?.addEventListener('click', () => {
        clearStagedHeadDetailTransition();
        clearStagedDetailTransition();
        fitScene({ animate: true });
        focusCanvasSurface();
    });
    backToGraphBtn?.addEventListener('click', () => {
        clearStagedHeadDetailTransition();
        clearStagedDetailTransition();
        returnToGraphOverview({ animate: true });
        focusCanvasSurface();
    });
    exitTo3dBtn?.addEventListener('click', () => {
        if (typeof onExitTo3d === 'function') {
            onExitTo3d();
        }
    });
    closeSelectionBtn?.addEventListener('click', () => {
        const didClearSelectionLock = clearPinnedSceneSelectionLocks({ scheduleRender: false });
        requestSelectionClose();
        if (didClearSelectionLock) {
            scheduleRender();
        }
        focusCanvasSurface();
    });
    tokenStripTokens?.addEventListener('click', (event) => {
        const chip = event?.target && typeof event.target.closest === 'function'
            ? event.target.closest('.detail-token-nav-chip[data-token-nav="true"]')
            : null;
        if (!chip || !tokenStripTokens.contains(chip)) return;
        requestTokenStripChipSelection(chip, event);
    });
    tokenStripTokens?.addEventListener('keydown', (event) => {
        const key = event?.key;
        if (key !== 'Enter' && key !== ' ' && key !== 'Spacebar') return;
        const chip = event?.target && typeof event.target.closest === 'function'
            ? event.target.closest('.detail-token-nav-chip[data-token-nav="true"]')
            : null;
        if (!chip || !tokenStripTokens.contains(chip)) return;
        requestTokenStripChipSelection(chip, event);
    });
    window.addEventListener('keydown', onWindowKeyDown);
    window.addEventListener('keyup', onWindowKeyUp);
    window.addEventListener('blur', clearKeyboardMotion);
    document.addEventListener('pointerdown', onDocumentPointerDown);
    document.addEventListener('focusin', onDocumentFocusIn);

    return {
        hasSelectionLock() {
            return isTransformerView2dDetailSelectionLockActive(state.detailSceneLockActive)
                || state.overviewSceneLockActive === true;
        },
        clearSelectionLock({
            scheduleRender: shouldScheduleRender = true
        } = {}) {
            return clearPinnedSceneSelectionLocks({
                scheduleRender: shouldScheduleRender
            });
        },
        setVisible(visible = false) {
            state.visible = !!visible;
            root.classList.toggle('is-visible', state.visible);
            root.setAttribute('aria-hidden', state.visible ? 'false' : 'true');
            renderTokenStrip();
            if (!state.visible) {
                setSelectionSidebarVisible(false, { immediate: true });
                tokenHoverSync.clear({ emit: true });
                clearStagedFocusTransition();
                clearStagedHeadDetailTransition();
                clearStagedDetailTransition();
                cancelScheduledHoverUpdate();
                clearPinnedSceneSelectionLocks({ scheduleRender: false });
                clearCanvasHover({ scheduleRender: false, force: true });
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
                resetOverviewSelectionArm();
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
            focusLabel = TRANSFORMER_VIEW2D_OVERVIEW_FOCUS_LABEL,
            detailSemanticTargets = null,
            detailFocusLabel = '',
            detailInteractionTargets = null,
            transitionMode = '',
            isSmallScreen = false
        } = {}) {
            clearStagedFocusTransition();
            clearStagedHeadDetailTransition();
            clearStagedDetailTransition();
            state.isSmallScreen = !!isSmallScreen;
            setSelectionSidebarVisible(false, { immediate: true });
            state.activationSource = activationSource;
            state.tokenIndices = Array.isArray(tokenIndices) ? [...tokenIndices] : tokenIndices;
            state.tokenLabels = Array.isArray(tokenLabels) ? [...tokenLabels] : tokenLabels;
            const normalizedDetailSemanticTargets = normalizeView2dSemanticTargets(detailSemanticTargets);
            const normalizedTransitionMode = String(
                transitionMode
                || resolveTransformerView2dOpenTransitionMode({
                    semanticTarget
                })
                || ''
            ).trim().toLowerCase();
            const resolvedDetailTargets = resolveDetailTargetsFromSemanticTarget(semanticTarget);
            const hasDeepDetailTarget = !!(
                resolvedDetailTargets.concatDetailTarget
                || resolvedDetailTargets.outputProjectionDetailTarget
                || resolvedDetailTargets.mlpDetailTarget
                || resolvedDetailTargets.layerNormDetailTarget
                || resolvedDetailTargets.headDetailTarget
            );
            const shouldStageFocusEntry = (
                normalizedTransitionMode === 'staged-focus'
                && !!buildSemanticTarget(semanticTarget)
                && !hasDeepDetailTarget
            );
            const shouldStageHeadDetailEntry = (
                normalizedTransitionMode === 'staged-head-detail'
                && !!resolvedDetailTargets.headDetailTarget
            );
            const shouldStageSceneDetailEntry = (
                normalizedTransitionMode === 'staged-detail'
                && !resolvedDetailTargets.headDetailTarget
                && !!(
                    resolvedDetailTargets.concatDetailTarget
                    || resolvedDetailTargets.outputProjectionDetailTarget
                    || resolvedDetailTargets.mlpDetailTarget
                    || resolvedDetailTargets.layerNormDetailTarget
                )
            );
            state.baseSemanticTarget = shouldStageHeadDetailEntry
                ? null
                : deriveBaseSemanticTarget(semanticTarget);
            state.baseFocusLabel = shouldStageHeadDetailEntry
                ? TRANSFORMER_VIEW2D_OVERVIEW_FOCUS_LABEL
                : (String(focusLabel || '').trim() || describeTransformerView2dTarget(state.baseSemanticTarget));
            setDetailTargets(
                shouldStageHeadDetailEntry || shouldStageSceneDetailEntry
                    ? {}
                    : resolvedDetailTargets
            );
            setDetailFocusTarget({
                detailSemanticTargets: normalizedDetailSemanticTargets,
                detailFocusLabel
            });
            state.pendingDetailInteractionTargets = normalizeTransformerView2dDetailInteractionTargets(
                detailInteractionTargets
            );
            if (shouldStageHeadDetailEntry) {
                state.stagedHeadDetailTransition = {
                    headDetailTarget: resolvedDetailTargets.headDetailTarget,
                    phase: 'overview-hold',
                    phaseStartTime: null
                };
            } else if (shouldStageSceneDetailEntry) {
                state.stagedDetailTransition = {
                    detailTargets: resolvedDetailTargets,
                    phase: 'overview-hold',
                    phaseStartTime: null
                };
            } else if (shouldStageFocusEntry) {
                state.stagedFocusTransition = {
                    phase: 'overview-hold',
                    phaseStartTime: null
                };
            }
            cancelScheduledHoverUpdate();
            clearPinnedSceneSelectionLocks({ scheduleRender: false });
            clearCanvasHover({ scheduleRender: false, force: true });
            tokenHoverSync.clear({ emit: true });
            setHoverDimmingTarget(0, { immediate: true, shouldRender: false });
            hideDetailFrame();
            residualCaptionOverlay.hide();
            state.tokenStripSignature = '';
            resetOverviewSelectionArm();
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
            if (shouldStageFocusEntry) {
                maybeStartStagedFocusTransition();
            }
            if (shouldStageHeadDetailEntry) {
                maybeStartStagedHeadDetailTransition();
            }
            if (shouldStageSceneDetailEntry) {
                maybeStartStagedDetailTransition();
            }
            focusCanvasSurface();
            return didRender;
        },
        getSelectionSidebarBody() {
            return selectionSidebarBody || null;
        },
        isSelectionSidebarVisible() {
            return !!selectionSidebar
                && (
                    selectionSidebar.classList.contains('is-visible')
                    || selectionSidebar.classList.contains('is-closing')
                );
        },
        setSelectionSidebarVisible(visible = false, options = null) {
            setSelectionSidebarVisible(visible, options || {});
        },
        setSelectionSidebarHeaderContent(content = null) {
            setSelectionSidebarHeaderContent(content || {});
        },
        scrollSelectionSidebarToTop() {
            scrollSelectionSidebarToTop();
        },
        resizeAndRender() {
            if (!state.visible || !state.scene || !state.layout) return false;
            measureCanvasSize();
            const { width, height, viewportInsets } = syncViewportControllers({
                preserveVisibleCenter: true
            });
            syncViewportControllerConstraints(width, viewportInsets);
            viewportController.setSceneBounds(state.layout.sceneBounds || null);
            if (shouldAutoFrameViewport(width, height)) {
                const shouldOpenFromOverview = (
                    state.stagedFocusTransition?.phase === 'overview-hold'
                    || state.stagedDetailTransition?.phase === 'overview-hold'
                    || state.stagedHeadDetailTransition?.phase === 'overview-hold'
                );
                if (shouldOpenFromOverview) {
                    fitScene({ animate: false });
                } else if (!focusSelection({ animate: false })) {
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
