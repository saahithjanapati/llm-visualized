import { buildSceneLayout } from '../view2d/layout/buildSceneLayout.js';
import { resolveSemanticTargetBounds } from '../view2d/layout/resolveSemanticTargetBounds.js';
import { buildTransformerSceneModel } from '../view2d/model/buildTransformerSceneModel.js';
import { CanvasSceneRenderer } from '../view2d/render/canvas/CanvasSceneRenderer.js';
import { View2dViewportController } from '../view2d/runtime/View2dViewportController.js';
import {
    findUserDataNumber,
    findUserDataString,
    getActivationDataFromSelection,
    isLogitBarSelection,
    isQkvMatrixLabel,
    isSelfAttentionSelection,
    isValueSelection,
    isWeightedSumSelection
} from './selectionPanelSelectionUtils.js';
import { resolveLayerNormKind } from '../utils/layerNormLabels.js';

export const TRANSFORMER_VIEW2D_PANEL_ACTION_OPEN = 'open-transformer-view2d';

const VIEW2D_DETAIL_ACTION_FOCUS = 'focus-selection';
const VIEW2D_DETAIL_ACTION_FIT = 'fit-scene';
const VIEW2D_INTERACTION_DPR = 1;
const VIEW2D_INTERACTION_SETTLE_MS = 140;
const VIEW2D_KEYBOARD_PAN_PX_PER_SEC = 620;
const VIEW2D_KEYBOARD_ZOOM_RATE = 0.00145;
const VIEW2D_KEYBOARD_INITIAL_STEP_MS = 16;

function normalizeOptionalIndex(value) {
    return Number.isFinite(value) ? Math.max(0, Math.floor(value)) : null;
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

function buildSemanticTarget(rawTarget = null) {
    if (!rawTarget || typeof rawTarget !== 'object') return null;
    const target = Object.entries(rawTarget).reduce((acc, [key, value]) => {
        if (!key) return acc;
        if (typeof value === 'number') {
            if (Number.isFinite(value)) acc[key] = Math.floor(value);
            return acc;
        }
        if (typeof value === 'string') {
            const safe = value.trim();
            if (safe.length) acc[key] = safe;
            return acc;
        }
        if (typeof value === 'boolean') {
            acc[key] = value;
        }
        return acc;
    }, {});
    return Object.keys(target).length ? target : null;
}

function isTopUnembeddingLabel(lower = '') {
    return lower.includes('vocab embedding (top)')
        || lower.includes('vocabulary embedding (top)')
        || lower.includes('vocab unembedding')
        || lower.includes('vocabulary unembedding')
        || lower.includes('unembedding');
}

function resolveLayerNormStage(kind = null) {
    if (kind === 'ln1') return 'ln1';
    if (kind === 'ln2') return 'ln2';
    if (kind === 'final') return 'final-ln';
    return null;
}

function resolveLayerLabel(layerIndex = null) {
    return Number.isFinite(layerIndex) ? `Layer ${Math.floor(layerIndex) + 1}` : '';
}

export function describeTransformerView2dTarget(target = null) {
    if (!target) return 'Transformer overview';
    const layerLabel = resolveLayerLabel(target.layerIndex);
    if (target.componentKind === 'embedding') {
        if (target.stage === 'token') return 'Token embeddings';
        if (target.stage === 'position') return 'Position embeddings';
        return 'Embeddings';
    }
    if (target.componentKind === 'layer-norm') {
        if (target.stage === 'ln1') return layerLabel ? `${layerLabel} LayerNorm 1` : 'LayerNorm 1';
        if (target.stage === 'ln2') return layerLabel ? `${layerLabel} LayerNorm 2` : 'LayerNorm 2';
        return 'Final LayerNorm';
    }
    if (target.componentKind === 'mhsa') {
        if (Number.isFinite(target.headIndex)) {
            const headLabel = `Head ${Math.floor(target.headIndex) + 1}`;
            return layerLabel ? `${layerLabel} MHSA ${headLabel}` : `MHSA ${headLabel}`;
        }
        return layerLabel ? `${layerLabel} MHSA` : 'MHSA';
    }
    if (target.componentKind === 'output-projection') {
        return layerLabel ? `${layerLabel} output projection` : 'Output projection';
    }
    if (target.componentKind === 'mlp') {
        if (target.stage === 'mlp-up') return layerLabel ? `${layerLabel} MLP up projection` : 'MLP up projection';
        if (target.stage === 'mlp-down') return layerLabel ? `${layerLabel} MLP down projection` : 'MLP down projection';
        if (target.stage === 'mlp-activation') return layerLabel ? `${layerLabel} GELU activation` : 'GELU activation';
        return layerLabel ? `${layerLabel} MLP` : 'MLP';
    }
    if (target.componentKind === 'residual') {
        if (target.stage === 'incoming') return layerLabel ? `${layerLabel} incoming residual` : 'Incoming residual';
        if (target.stage === 'post-attn-add') return layerLabel ? `${layerLabel} post-attention residual` : 'Post-attention residual';
        if (target.stage === 'post-mlp-add') return layerLabel ? `${layerLabel} post-MLP residual` : 'Post-MLP residual';
        if (target.stage === 'outgoing') return layerLabel ? `${layerLabel} outgoing residual` : 'Outgoing residual';
        return layerLabel ? `${layerLabel} residual stream` : 'Residual stream';
    }
    if (target.componentKind === 'logits') {
        if (target.role === 'unembedding') return 'Unembedding';
        if (target.role === 'logits-topk') return 'Logits';
        return 'Logits';
    }
    return 'Transformer overview';
}

export function resolveTransformerView2dActionContext(selectionInfo = null, normalizedLabel = '') {
    const label = String(normalizedLabel || selectionInfo?.label || '').trim();
    if (!label.length) return null;

    const lower = label.toLowerCase();
    const activationData = getActivationDataFromSelection(selectionInfo);
    const stageLower = String(activationData?.stage || '').toLowerCase();
    const layerIndex = normalizeOptionalIndex(findUserDataNumber(selectionInfo, 'layerIndex'));
    const headIndex = normalizeOptionalIndex(findUserDataNumber(selectionInfo, 'headIndex'));
    const explicitLayerNormKind = findUserDataString(selectionInfo, 'layerNormKind');
    const layerNormKind = resolveLayerNormKind({
        label,
        stage: stageLower,
        explicitKind: explicitLayerNormKind
    });
    const topUnembeddingLabel = isTopUnembeddingLabel(lower);

    let semanticTarget = null;

    if (layerNormKind) {
        const stage = resolveLayerNormStage(layerNormKind);
        if (stage && (layerNormKind === 'final' || Number.isFinite(layerIndex))) {
            semanticTarget = buildSemanticTarget({
                componentKind: 'layer-norm',
                layerIndex: layerNormKind === 'final' ? null : layerIndex,
                stage,
                role: 'module'
            });
        }
    } else if (
        isLogitBarSelection(label, selectionInfo)
        || lower.includes('top logit bars')
        || topUnembeddingLabel
        || lower === 'logit'
    ) {
        semanticTarget = buildSemanticTarget({
            componentKind: 'logits',
            stage: 'output',
            role: topUnembeddingLabel ? 'unembedding' : 'logits-topk'
        });
    } else if (
        stageLower.startsWith('embedding.token')
        || lower.includes('token embedding')
    ) {
        semanticTarget = buildSemanticTarget({
            componentKind: 'embedding',
            stage: 'token',
            role: 'token-embedding'
        });
    } else if (
        stageLower.startsWith('embedding.position')
        || lower.includes('position embedding')
        || lower.includes('positional embedding')
    ) {
        semanticTarget = buildSemanticTarget({
            componentKind: 'embedding',
            stage: 'position',
            role: 'position-embedding'
        });
    } else if (
        stageLower.startsWith('embedding.sum')
        || lower.includes('embedding sum')
        || (lower.includes('vocabulary embedding') && !topUnembeddingLabel)
        || (lower.includes('vocab embedding') && !topUnembeddingLabel)
    ) {
        semanticTarget = buildSemanticTarget({
            componentKind: 'embedding',
            stage: stageLower.startsWith('embedding.sum') || lower.includes('embedding sum') ? 'sum' : 'input',
            role: stageLower.startsWith('embedding.sum') || lower.includes('embedding sum') ? 'sum-output' : 'module'
        });
    } else if (
        lower.includes('output projection matrix')
        || stageLower === 'attention.output_projection'
        || lower.includes('attention output projection')
    ) {
        if (Number.isFinite(layerIndex)) {
            semanticTarget = buildSemanticTarget({
                componentKind: 'output-projection',
                layerIndex,
                stage: 'attn-out',
                role: lower.includes('output projection matrix') ? 'projection-weight' : 'projection-output'
            });
        }
    } else if (
        lower.includes('mlp up weight matrix')
        || lower.includes('mlp up projection')
    ) {
        if (Number.isFinite(layerIndex)) {
            semanticTarget = buildSemanticTarget({
                componentKind: 'mlp',
                layerIndex,
                stage: 'mlp-up',
                role: 'mlp-up'
            });
        }
    } else if (
        lower.includes('mlp down weight matrix')
        || lower.includes('mlp down projection')
    ) {
        if (Number.isFinite(layerIndex)) {
            semanticTarget = buildSemanticTarget({
                componentKind: 'mlp',
                layerIndex,
                stage: 'mlp-down',
                role: 'mlp-down'
            });
        }
    } else if (
        lower.includes('gelu')
        || lower.includes('mlp activation')
    ) {
        if (Number.isFinite(layerIndex)) {
            semanticTarget = buildSemanticTarget({
                componentKind: 'mlp',
                layerIndex,
                stage: 'mlp-activation',
                role: 'mlp-activation'
            });
        }
    } else if (
        lower.includes('mlp')
        && Number.isFinite(layerIndex)
    ) {
        semanticTarget = buildSemanticTarget({
            componentKind: 'mlp',
            layerIndex,
            stage: 'mlp',
            role: 'module'
        });
    } else if (
        stageLower.startsWith('qkv.')
        || stageLower.startsWith('attention.')
        || isSelfAttentionSelection(label, selectionInfo)
        || isWeightedSumSelection(label, selectionInfo)
        || isValueSelection(label, selectionInfo)
        || isQkvMatrixLabel(label)
        || lower.includes('self-attention')
    ) {
        if (Number.isFinite(layerIndex)) {
            semanticTarget = buildSemanticTarget({
                componentKind: 'mhsa',
                layerIndex,
                headIndex,
                stage: 'attention',
                role: Number.isFinite(headIndex) ? 'head' : 'module'
            });
        }
    } else if (
        stageLower.startsWith('layer.incoming')
        || lower.includes('incoming residual')
    ) {
        if (Number.isFinite(layerIndex)) {
            semanticTarget = buildSemanticTarget({
                componentKind: 'residual',
                layerIndex,
                stage: 'incoming',
                role: 'module'
            });
        }
    } else if (
        stageLower.includes('post_attention')
        || lower.includes('post-attention residual')
        || lower.includes('post attention residual')
    ) {
        if (Number.isFinite(layerIndex)) {
            semanticTarget = buildSemanticTarget({
                componentKind: 'residual',
                layerIndex,
                stage: 'post-attn-add',
                role: 'module'
            });
        }
    } else if (
        stageLower.includes('post_mlp')
        || lower.includes('post-mlp residual')
        || lower.includes('post mlp residual')
    ) {
        if (Number.isFinite(layerIndex)) {
            semanticTarget = buildSemanticTarget({
                componentKind: 'residual',
                layerIndex,
                stage: 'post-mlp-add',
                role: 'module'
            });
        }
    } else if (lower.includes('residual stream vector') && Number.isFinite(layerIndex)) {
        semanticTarget = buildSemanticTarget({
            componentKind: 'residual',
            layerIndex,
            stage: 'incoming',
            role: 'module'
        });
    }

    if (!semanticTarget) return null;
    return {
        semanticTarget,
        focusLabel: describeTransformerView2dTarget(semanticTarget),
        actionLabel: 'Open 2D canvas'
    };
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
                        <div class="detail-transformer-view2d-hint">Drag to pan. Scroll or +/- to zoom. Use arrows or WASD to move. Use Focus selection to return to the current component.</div>
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
                aria-label="Scalable 2D transformer canvas. Drag to pan, scroll or use plus and minus to zoom, and use arrow keys or W A S D to move around."
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
    const focusBtn = root.querySelector(`[data-transformer-view2d-action="${VIEW2D_DETAIL_ACTION_FOCUS}"]`);
    const fitBtn = root.querySelector(`[data-transformer-view2d-action="${VIEW2D_DETAIL_ACTION_FIT}"]`);
    const focusReadout = root.querySelector('[data-transformer-view2d-readout="focus"]');
    const zoomReadout = root.querySelector('[data-transformer-view2d-readout="zoom"]');
    const sceneReadout = root.querySelector('[data-transformer-view2d-readout="scene"]');

    const renderer = new CanvasSceneRenderer({ canvas });
    const viewportController = new View2dViewportController({
        minScale: 0.035,
        maxScale: 6,
        padding: 28
    });

    const state = {
        visible: false,
        scene: null,
        layout: null,
        semanticTarget: null,
        focusLabel: 'Transformer overview',
        sceneLabel: '0 layers / 0 tokens',
        pointer: null,
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

    function updateReadouts() {
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
        const didRender = renderer.render({
            width,
            height,
            dpr: state.isInteracting ? VIEW2D_INTERACTION_DPR : null,
            viewportTransform: viewportController.getViewportTransform('detail-transformer-view2d')
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

    function focusSelection({ animate = true } = {}) {
        if (!state.layout?.registry || !state.semanticTarget) {
            return fitScene({ animate });
        }

        const bounds = resolveSemanticTargetBounds(state.layout.registry, state.semanticTarget);
        if (!bounds) {
            state.focusLabel = `${describeTransformerView2dTarget(state.semanticTarget)} (focus fallback: full scene)`;
            updateReadouts();
            return fitScene({ animate });
        }

        state.focusLabel = describeTransformerView2dTarget(state.semanticTarget);
        if (animate) {
            viewportController.flyToBounds(bounds, {
                animate: true,
                durationMs: 420,
                now: performance.now(),
                padding: 36
            });
            animateViewport();
            updateReadouts();
            return true;
        }
        viewportController.fitToBounds(bounds, { padding: 36 });
        render();
        return true;
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

    function onPointerDown(event) {
        if (!state.visible || event.button !== 0) return;
        focusCanvasSurface();
        state.pointer = {
            pointerId: event.pointerId,
            clientX: event.clientX,
            clientY: event.clientY
        };
        canvas.setPointerCapture?.(event.pointerId);
        stopAnimation();
        markInteraction(true);
        canvas.classList.add('is-panning');
        event.preventDefault();
    }

    function onPointerMove(event) {
        if (!state.pointer || state.pointer.pointerId !== event.pointerId) return;
        const deltaX = event.clientX - state.pointer.clientX;
        const deltaY = event.clientY - state.pointer.clientY;
        state.pointer.clientX = event.clientX;
        state.pointer.clientY = event.clientY;
        viewportController.panBy(deltaX, deltaY);
        markInteraction(true);
        scheduleRender();
        event.preventDefault();
    }

    function clearPointer(event = null) {
        if (state.pointer && event && state.pointer.pointerId === event.pointerId) {
            canvas.releasePointerCapture?.(event.pointerId);
        }
        state.pointer = null;
        canvas.classList.remove('is-panning');
        scheduleInteractionSettle();
    }

    function onWheel(event) {
        if (!state.visible) return;
        focusCanvasSurface();
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

    canvasCard?.addEventListener('keydown', onCanvasCardKeyDown);
    canvasCard?.addEventListener('keyup', onCanvasCardKeyUp);
    canvasCard?.addEventListener('pointerdown', focusCanvasSurface);
    canvasCard?.addEventListener('blur', clearKeyboardMotion);
    canvas?.addEventListener('pointerdown', onPointerDown);
    canvas?.addEventListener('pointermove', onPointerMove);
    canvas?.addEventListener('pointerup', clearPointer);
    canvas?.addEventListener('pointercancel', clearPointer);
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
        fitScene({ animate: true });
        focusCanvasSurface();
    });

    return {
        setVisible(visible = false) {
            state.visible = !!visible;
            root.classList.toggle('is-visible', state.visible);
            root.setAttribute('aria-hidden', state.visible ? 'false' : 'true');
            if (!state.visible) {
                clearPointer();
                clearKeyboardMotion();
                stopAnimation();
                stopRenderLoop();
                clearInteractionTimer();
                state.isInteracting = false;
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
            state.semanticTarget = buildSemanticTarget(semanticTarget);
            state.focusLabel = focusLabel || describeTransformerView2dTarget(state.semanticTarget);
            state.scene = buildTransformerSceneModel({
                activationSource,
                tokenIndices,
                tokenLabels,
                isSmallScreen
            });
            state.layout = buildSceneLayout(state.scene, {
                isSmallScreen
            });
            renderer.setScene(state.scene, state.layout);
            viewportController.setSceneBounds(state.layout.sceneBounds || null);
            state.sceneLabel = `${state.scene?.metadata?.layerCount || 0} layers / ${state.scene?.metadata?.tokenCount || 0} tokens`;
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
