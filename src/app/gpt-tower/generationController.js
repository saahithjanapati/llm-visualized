import { setNumVectorLanes, USE_PHYSICAL_MATERIALS } from '../../utils/constants.js';
import { setAnimationLaneCount } from '../../animations/LayerAnimationConstants.js';
import { applyPhysicalMaterialsToScene } from '../../utils/materialUtils.js';
import { appState } from '../../state/appState.js';
import { addEmbeddingAndTokenChips } from './tokenChips.js';
import { formatTokenLabel } from './tokenLabels.js';

const DEFAULT_ADVANCE_SECONDS = 10;

export function buildPassState({
    activationSource,
    laneCount,
    fallbackTokenLabels = [],
    fallbackPositionLabels = []
} = {}) {
    const count = Math.max(1, Math.floor(laneCount || 1));
    const laneTokenIndices = activationSource && typeof activationSource.getLaneTokenIndices === 'function'
        ? activationSource.getLaneTokenIndices(count)
        : Array.from({ length: count }, (_, idx) => idx);

    const tokenLabels = laneTokenIndices.map((tokenIndex, laneIdx) => {
        const raw = activationSource && typeof activationSource.getTokenString === 'function'
            ? activationSource.getTokenString(tokenIndex)
            : null;
        if (raw !== null && raw !== undefined && raw !== '') return raw;
        return fallbackTokenLabels[laneIdx] ?? '';
    });

    const positionLabels = laneTokenIndices.map((tokenIndex, laneIdx) => {
        if (activationSource && Number.isFinite(tokenIndex)) {
            return String(tokenIndex + 1);
        }
        return fallbackPositionLabels[laneIdx] ?? String(laneIdx + 1);
    });

    return { laneTokenIndices, tokenLabels, positionLabels };
}

function resolveTokenCount(activationSource, fallbackCount) {
    const count = activationSource && typeof activationSource.getTokenCount === 'function'
        ? activationSource.getTokenCount()
        : fallbackCount;
    return Number.isFinite(count) && count > 0 ? Math.floor(count) : fallbackCount;
}

function createAdvanceOverlay() {
    let root = document.getElementById('generationOverlay');
    if (!root) {
        root = document.createElement('div');
        root.id = 'generationOverlay';
        root.dataset.visible = 'false';
        root.dataset.paused = 'false';
        root.innerHTML = `
            <div class="generation-header">
                <div class="generation-title">
                    <span data-role="title-prefix">Going to next token in</span>
                    <span data-role="countdown-wrap"><span data-role="countdown">10</span>s</span>
                </div>
                <div class="generation-meta" data-role="token"></div>
            </div>
            <div class="generation-bar" aria-hidden="true">
                <div class="generation-bar-fill" data-role="bar-fill"></div>
            </div>
            <div class="generation-actions">
                <button type="button" data-role="stay">Stay</button>
                <button type="button" data-role="advance" class="primary">Advance</button>
            </div>
        `;
        document.body.appendChild(root);
    }

    const titlePrefix = root.querySelector('[data-role="title-prefix"]');
    const countdownWrap = root.querySelector('[data-role="countdown-wrap"]');
    const countdownEl = root.querySelector('[data-role="countdown"]');
    const tokenEl = root.querySelector('[data-role="token"]');
    const barFill = root.querySelector('[data-role="bar-fill"]');
    const stayBtn = root.querySelector('[data-role="stay"]');
    const advanceBtn = root.querySelector('[data-role="advance"]');

    return { root, titlePrefix, countdownWrap, countdownEl, tokenEl, barFill, stayBtn, advanceBtn };
}

export function initGenerationController({
    pipeline,
    activationSource,
    initialLaneCount,
    initialPassState,
    fallbackTokenLabels = [],
    fallbackPositionLabels = [],
    numLayers,
    cameraReturnPosition,
    cameraReturnTarget,
    selectionPanel,
    autoAdvanceSeconds = DEFAULT_ADVANCE_SECONDS
} = {}) {
    if (!pipeline) return null;

    const totalTokenCount = resolveTokenCount(activationSource, initialLaneCount);
    const maxLaneCount = Math.max(1, totalTokenCount || initialLaneCount);
    const canLoop = !!activationSource && maxLaneCount > initialLaneCount;

    const overlay = createAdvanceOverlay();
    let currentLaneCount = Math.max(1, Math.floor(initialLaneCount || 1));
    let passComplete = false;
    let autoAdvancePaused = false;
    let countdownActive = false;
    const safeAdvanceSeconds = Number.isFinite(autoAdvanceSeconds) ? autoAdvanceSeconds : DEFAULT_ADVANCE_SECONDS;
    let countdownMs = Math.max(1, Math.floor(safeAdvanceSeconds * 1000));
    let remainingMs = countdownMs;
    let lastTick = null;
    let rafId = null;
    let chipCleanup = null;

    const clearOverlay = () => {
        overlay.root.dataset.visible = 'false';
    };

    const updateOverlay = () => {
        if (!passComplete) {
            clearOverlay();
            return;
        }

        overlay.root.dataset.visible = 'true';
        overlay.root.dataset.paused = autoAdvancePaused ? 'true' : 'false';

        const atEnd = currentLaneCount >= maxLaneCount;
        const nextTokenIndex = currentLaneCount;
        const nextTokenNumber = nextTokenIndex + 1;
        const remainingSeconds = Math.max(0, Math.ceil(remainingMs / 1000));

        if (overlay.countdownEl) overlay.countdownEl.textContent = String(remainingSeconds);

        if (atEnd) {
            if (overlay.titlePrefix) overlay.titlePrefix.textContent = 'End of capture';
            if (overlay.countdownWrap) overlay.countdownWrap.style.display = 'none';
        } else if (autoAdvancePaused) {
            if (overlay.titlePrefix) overlay.titlePrefix.textContent = 'Auto-advance paused';
            if (overlay.countdownWrap) overlay.countdownWrap.style.display = 'none';
        } else {
            if (overlay.titlePrefix) overlay.titlePrefix.textContent = 'Going to next token in';
            if (overlay.countdownWrap) overlay.countdownWrap.style.display = '';
        }

        const nextTokenLabel = (!atEnd && activationSource && typeof activationSource.getTokenString === 'function')
            ? activationSource.getTokenString(nextTokenIndex)
            : null;
        const formatted = nextTokenLabel ? formatTokenLabel(nextTokenLabel) : '';
        const tokenLine = atEnd
            ? `Token ${maxLaneCount} / ${maxLaneCount}`
            : (formatted
                ? `Next token ${nextTokenNumber} / ${maxLaneCount}: ${formatted}`
                : `Next token ${nextTokenNumber} / ${maxLaneCount}`);
        if (overlay.tokenEl) overlay.tokenEl.textContent = tokenLine;

        const progress = atEnd ? 1 : Math.max(0, Math.min(1, 1 - remainingMs / countdownMs));
        if (overlay.barFill) {
            overlay.barFill.style.width = `${(progress * 100).toFixed(1)}%`;
        }

        if (overlay.stayBtn) {
            overlay.stayBtn.disabled = atEnd;
            overlay.stayBtn.textContent = autoAdvancePaused ? 'Resume auto' : 'Stay';
        }
        if (overlay.advanceBtn) {
            overlay.advanceBtn.disabled = atEnd;
            overlay.advanceBtn.textContent = 'Advance';
        }
    };

    const syncSelectionPanel = (passState) => {
        if (!selectionPanel) return;
        selectionPanel.updateData?.({
            activationSource,
            laneTokenIndices: passState.laneTokenIndices,
            tokenLabels: passState.tokenLabels
        });
        selectionPanel.close?.();
    };

    const rebuildPass = ({ laneCount, passState, resetPipeline = false } = {}) => {
        const nextLaneCount = Math.max(1, Math.floor(laneCount || 1));
        const state = passState || buildPassState({
            activationSource,
            laneCount: nextLaneCount,
            fallbackTokenLabels,
            fallbackPositionLabels
        });

        if (resetPipeline) {
            setNumVectorLanes(nextLaneCount);
            setAnimationLaneCount(nextLaneCount);
            pipeline.resetForNewPass({ activationSource, laneCount: nextLaneCount });
            if (appState.autoCameraFollow) {
                pipeline.setAutoCameraFollow?.(true, { immediate: true, resetView: true });
            } else {
                pipeline.focusOverview?.({ immediate: true });
            }
        }

        appState.topEmbedActivated = false;
        appState.lastEqSignature = '';
        appState.lastEqKey = '';

        if (chipCleanup && typeof chipCleanup.dispose === 'function') {
            chipCleanup.dispose();
        }
        chipCleanup = addEmbeddingAndTokenChips({
            pipeline,
            laneCount: nextLaneCount,
            activationSource,
            laneTokenIndices: state.laneTokenIndices,
            tokenLabels: state.tokenLabels,
            positionLabels: state.positionLabels,
            cameraReturnPosition,
            cameraReturnTarget,
            numLayers
        });

        applyPhysicalMaterialsToScene(pipeline?.engine?.scene, USE_PHYSICAL_MATERIALS);
        syncSelectionPanel(state);

        currentLaneCount = nextLaneCount;
        passComplete = false;
        autoAdvancePaused = false;
        countdownActive = false;
        remainingMs = countdownMs;
        lastTick = null;
        clearOverlay();
    };

    rebuildPass({ laneCount: currentLaneCount, passState: initialPassState, resetPipeline: false });

    if (!canLoop) {
        clearOverlay();
        return {
            dispose: () => {
                if (chipCleanup?.dispose) chipCleanup.dispose();
                if (rafId && typeof cancelAnimationFrame === 'function') cancelAnimationFrame(rafId);
            }
        };
    }

    const advanceToNextPass = () => {
        if (currentLaneCount >= maxLaneCount) {
            passComplete = true;
            countdownActive = false;
            autoAdvancePaused = true;
            updateOverlay();
            return;
        }
        const nextLane = Math.min(maxLaneCount, currentLaneCount + 1);
        rebuildPass({ laneCount: nextLane, resetPipeline: true });
    };

    if (overlay.stayBtn) {
        overlay.stayBtn.onclick = (event) => {
            event.preventDefault();
            if (!passComplete || currentLaneCount >= maxLaneCount) return;
            autoAdvancePaused = !autoAdvancePaused;
            countdownActive = !autoAdvancePaused;
            lastTick = null;
            updateOverlay();
        };
    }

    if (overlay.advanceBtn) {
        overlay.advanceBtn.onclick = (event) => {
            event.preventDefault();
            if (!passComplete) return;
            advanceToNextPass();
        };
    }

    const tick = (now) => {
        if (!pipeline) return;
        if (!passComplete) {
            const isComplete = typeof pipeline.isForwardPassComplete === 'function'
                ? pipeline.isForwardPassComplete()
                : false;
            if (isComplete) {
                passComplete = true;
                remainingMs = countdownMs;
                countdownActive = !autoAdvancePaused;
                lastTick = now;
                updateOverlay();
            }
        } else if (countdownActive) {
            const paused = appState.userPaused || appState.modalPaused;
            if (!paused) {
                if (lastTick == null) lastTick = now;
                const delta = Math.max(0, now - lastTick);
                remainingMs = Math.max(0, remainingMs - delta);
                lastTick = now;
            } else {
                lastTick = now;
            }
            if (remainingMs <= 0) {
                advanceToNextPass();
            }
            updateOverlay();
        }

        rafId = requestAnimationFrame(tick);
    };

    rafId = requestAnimationFrame(tick);

    return {
        advance: advanceToNextPass,
        dispose: () => {
            if (chipCleanup?.dispose) chipCleanup.dispose();
            if (rafId && typeof cancelAnimationFrame === 'function') cancelAnimationFrame(rafId);
        }
    };
}

export default initGenerationController;
