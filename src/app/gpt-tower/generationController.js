import { setNumVectorLanes, USE_PHYSICAL_MATERIALS } from '../../utils/constants.js';
import { setAnimationLaneCount } from '../../animations/LayerAnimationConstants.js';
import { applyPhysicalMaterialsToScene } from '../../utils/materialUtils.js';
import {
    getLaneOpacityScale,
    setTrailOpacityRuntimeMultiplier,
    setTrailLineWidthRuntimeMultiplier
} from '../../utils/trailConstants.js';
import { refreshTrailDisplayScales } from '../../utils/trailUtils.js';
import { appState } from '../../state/appState.js';
import { addEmbeddingAndTokenChips } from './tokenChips.js';
import { formatTokenLabel } from './tokenLabels.js';
import { resolveLogitTokenSeed } from './logitColor.js';
import { initTouchClickFallback } from '../../ui/touchClickFallback.js';

const DEFAULT_ADVANCE_SECONDS = 10;
const KV_DECODE_SINGLE_LANE_TRAIL_WIDTH_MULTIPLIER = 8.0;
const KV_DECODE_SINGLE_LANE_TRAIL_OPACITY_BOOST = 2.2;
const NEXT_TOKEN_DESKTOP_MEDIA_QUERY = '(min-width: 881px) and (min-aspect-ratio: 1/1)';
const NEXT_TOKEN_PANEL_GAP_PX = 18;
const NEXT_TOKEN_VIEWPORT_GUTTER_PX = 12;

export function buildPassState({
    activationSource,
    laneCount,
    laneTokenIndices = null,
    laneLayoutIndices = null,
    totalLaneCount = null,
    fallbackTokenLabels = [],
    fallbackPositionLabels = []
} = {}) {
    const count = Math.max(1, Math.floor(laneCount || 1));
    const resolvedLaneTokenIndices = Array.isArray(laneTokenIndices) && laneTokenIndices.length
        ? laneTokenIndices.slice(0, count)
        : (activationSource && typeof activationSource.getLaneTokenIndices === 'function'
            ? activationSource.getLaneTokenIndices(count)
            : Array.from({ length: count }, (_, idx) => idx));
    while (resolvedLaneTokenIndices.length < count) {
        resolvedLaneTokenIndices.push(resolvedLaneTokenIndices.length);
    }

    const resolvedLaneLayoutIndices = Array.isArray(laneLayoutIndices) && laneLayoutIndices.length
        ? laneLayoutIndices.slice(0, count)
        : Array.from({ length: count }, (_, idx) => idx);
    while (resolvedLaneLayoutIndices.length < count) {
        resolvedLaneLayoutIndices.push(resolvedLaneLayoutIndices.length);
    }

    const resolvedTotalLaneCount = Number.isFinite(totalLaneCount)
        ? Math.max(1, Math.floor(totalLaneCount))
        : Math.max(
            count,
            resolvedLaneLayoutIndices.reduce(
                (max, laneIdx) => Math.max(max, Number.isFinite(laneIdx) ? Math.floor(laneIdx) + 1 : 0),
                0
            )
        );

    const tokenLabels = resolvedLaneTokenIndices.map((tokenIndex, laneIdx) => {
        const raw = activationSource && typeof activationSource.getTokenString === 'function'
            ? activationSource.getTokenString(tokenIndex)
            : null;
        if (raw !== null && raw !== undefined && raw !== '') return raw;
        return fallbackTokenLabels[laneIdx] ?? '';
    });

    const positionLabels = resolvedLaneTokenIndices.map((tokenIndex, laneIdx) => {
        if (activationSource && Number.isFinite(tokenIndex)) {
            return String(tokenIndex + 1);
        }
        return fallbackPositionLabels[laneIdx] ?? String(laneIdx + 1);
    });

    return {
        laneTokenIndices: resolvedLaneTokenIndices,
        laneLayoutIndices: resolvedLaneLayoutIndices,
        totalLaneCount: resolvedTotalLaneCount,
        tokenLabels,
        positionLabels
    };
}

function resolveTokenCount(activationSource, fallbackCount) {
    const count = activationSource && typeof activationSource.getTokenCount === 'function'
        ? activationSource.getTokenCount()
        : fallbackCount;
    return Number.isFinite(count) && count > 0 ? Math.floor(count) : fallbackCount;
}

function sanitizeLogitToken(token) {
    if (token === null || token === undefined) return '';
    const raw = String(token);
    if (!raw.length) return '';
    return raw.replace(/\n/g, '\\n').replace(/\t/g, '\\t');
}

function resolveGeneratedLogitToken(activationSource, laneTokenIndices = []) {
    if (!activationSource || typeof activationSource.getLogitsForToken !== 'function') return null;
    if (!Array.isArray(laneTokenIndices) || !laneTokenIndices.length) return null;
    const lastTokenIndexRaw = laneTokenIndices[laneTokenIndices.length - 1];
    if (!Number.isFinite(lastTokenIndexRaw)) return null;
    const lastTokenIndex = Math.floor(lastTokenIndexRaw);

    const logitTopK = typeof activationSource.getLogitTopK === 'function'
        ? activationSource.getLogitTopK()
        : null;
    const safeTopK = Number.isFinite(logitTopK) && logitTopK > 0 ? Math.floor(logitTopK) : null;
    const logitRow = activationSource.getLogitsForToken(lastTokenIndex, safeTopK);
    if (!Array.isArray(logitRow) || !logitRow.length) return null;

    const tokenCount = typeof activationSource.getTokenCount === 'function'
        ? activationSource.getTokenCount()
        : 0;
    const hasNextToken = Number.isFinite(tokenCount) && (lastTokenIndex + 1) < tokenCount;
    const nextTokenRaw = hasNextToken && typeof activationSource.getTokenString === 'function'
        ? activationSource.getTokenString(lastTokenIndex + 1)
        : null;

    let bestIdx = -1;
    let bestProb = -Infinity;
    let pickedIdx = -1;
    for (let i = 0; i < logitRow.length; i += 1) {
        const entry = logitRow[i];
        const prob = Number(entry?.prob);
        if (Number.isFinite(prob) && prob > bestProb) {
            bestProb = prob;
            bestIdx = i;
        }
        if (
            pickedIdx === -1
            && typeof nextTokenRaw === 'string'
            && typeof entry?.token === 'string'
            && entry.token === nextTokenRaw
        ) {
            pickedIdx = i;
        }
    }

    const chosenIdx = pickedIdx !== -1
        ? pickedIdx
        : (bestIdx !== -1 ? bestIdx : 0);
    const chosenEntry = logitRow[chosenIdx];
    if (!chosenEntry || typeof chosenEntry !== 'object') return null;

    const tokenId = Number.isFinite(chosenEntry.token_id) ? Math.floor(chosenEntry.token_id) : null;
    const tokenText = typeof chosenEntry.token === 'string'
        ? formatTokenLabel(sanitizeLogitToken(chosenEntry.token))
        : '';
    const fallbackText = Number.isFinite(tokenId) ? `#${tokenId}` : '';
    const tokenLabel = tokenText || fallbackText;
    if (!tokenLabel) return null;

    return {
        tokenLabel,
        tokenId,
        tokenIndex: hasNextToken ? Math.floor(lastTokenIndex + 1) : null,
        seed: resolveLogitTokenSeed(chosenEntry, chosenIdx),
        selectionLabel: `Chosen token: ${tokenLabel}`,
        logitEntry: chosenEntry
    };
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

function createNextTokenButton() {
    const existing = document.getElementById('nextTokenBtn');
    if (existing) {
        if (document.body && existing.parentElement !== document.body) {
            document.body.appendChild(existing);
        }
        return existing;
    }
    if (!document.body) return null;

    const btn = document.createElement('button');
    btn.id = 'nextTokenBtn';
    btn.type = 'button';
    btn.textContent = 'Next token';
    btn.title = 'Advance to next token';
    btn.setAttribute('aria-label', 'Advance to next token');
    btn.dataset.visible = 'false';
    btn.style.display = 'none';
    document.body.appendChild(btn);
    return btn;
}

function resolveVisibleRect(element) {
    if (!element || typeof element.getBoundingClientRect !== 'function' || typeof window === 'undefined') {
        return null;
    }
    const style = window.getComputedStyle(element);
    if (style.display === 'none' || style.visibility === 'hidden') return null;
    const rect = element.getBoundingClientRect();
    if (!(rect.width > 0 && rect.height > 0)) return null;
    return rect;
}

function isDesktopNextTokenDockLayout() {
    if (typeof window === 'undefined') return false;
    if (typeof window.matchMedia === 'function') {
        return window.matchMedia(NEXT_TOKEN_DESKTOP_MEDIA_QUERY).matches;
    }
    return window.innerWidth >= 881 && window.innerWidth >= window.innerHeight;
}

function syncNextTokenButtonLayout(
    button,
    {
        hudStack = null,
        detailPanel = null,
        resizeHandle = null
    } = {}
) {
    if (!button || typeof document === 'undefined') return;

    const hudStackEl = hudStack || document.getElementById('hudStack');
    const detailPanelEl = detailPanel || document.getElementById('detailPanel');
    const resizeHandleEl = resizeHandle || document.getElementById('detailPanelResizeHandle');
    const canDockLeftOfPanel = isDesktopNextTokenDockLayout()
        && !!hudStackEl
        && !!detailPanelEl
        && hudStackEl.classList.contains('detail-open')
        && detailPanelEl.classList.contains('is-open');

    if (!canDockLeftOfPanel) {
        button.dataset.layout = 'corner';
        button.style.removeProperty('left');
        button.style.removeProperty('right');
        return;
    }

    const hudRect = resolveVisibleRect(hudStackEl);
    const handleRect = resolveVisibleRect(resizeHandleEl);
    const buttonRect = resolveVisibleRect(button);
    const blockerLeft = Math.min(
        hudRect?.left ?? Number.POSITIVE_INFINITY,
        handleRect?.left ?? Number.POSITIVE_INFINITY
    );
    const buttonWidth = buttonRect?.width || button.offsetWidth || 0;

    if (!Number.isFinite(blockerLeft) || !(buttonWidth > 0)) {
        button.dataset.layout = 'corner';
        button.style.removeProperty('left');
        button.style.removeProperty('right');
        return;
    }

    const nextLeft = Math.max(
        NEXT_TOKEN_VIEWPORT_GUTTER_PX,
        Math.round(blockerLeft - buttonWidth - NEXT_TOKEN_PANEL_GAP_PX)
    );
    button.dataset.layout = 'panel-docked';
    button.style.left = `${nextLeft}px`;
    button.style.right = 'auto';
}

function initNextTokenButtonLayoutSync(button) {
    if (!button || typeof document === 'undefined' || typeof window === 'undefined') return null;

    const hudStack = document.getElementById('hudStack');
    const detailPanel = document.getElementById('detailPanel');
    const resizeHandle = document.getElementById('detailPanelResizeHandle');
    let syncRafId = null;
    let resizeObserver = null;
    let mutationObserver = null;

    const runSync = () => {
        syncRafId = null;
        syncNextTokenButtonLayout(button, { hudStack, detailPanel, resizeHandle });
    };

    const scheduleSync = () => {
        if (typeof window.requestAnimationFrame !== 'function') {
            runSync();
            return;
        }
        if (syncRafId !== null) return;
        syncRafId = window.requestAnimationFrame(runSync);
    };

    if (typeof ResizeObserver !== 'undefined') {
        resizeObserver = new ResizeObserver(() => {
            scheduleSync();
        });
        if (hudStack) resizeObserver.observe(hudStack);
        resizeObserver.observe(button);
    }

    if (typeof MutationObserver !== 'undefined') {
        mutationObserver = new MutationObserver(() => {
            scheduleSync();
        });
        if (hudStack) {
            mutationObserver.observe(hudStack, {
                attributes: true,
                attributeFilter: ['class']
            });
        }
        if (detailPanel) {
            mutationObserver.observe(detailPanel, {
                attributes: true,
                attributeFilter: ['class']
            });
        }
    }

    window.addEventListener('resize', scheduleSync);
    window.visualViewport?.addEventListener?.('resize', scheduleSync);
    scheduleSync();

    return () => {
        window.removeEventListener('resize', scheduleSync);
        window.visualViewport?.removeEventListener?.('resize', scheduleSync);
        if (resizeObserver) resizeObserver.disconnect();
        if (mutationObserver) mutationObserver.disconnect();
        if (syncRafId !== null && typeof window.cancelAnimationFrame === 'function') {
            window.cancelAnimationFrame(syncRafId);
            syncRafId = null;
        }
    };
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
    promptTokenStrip,
    autoAdvanceSeconds = DEFAULT_ADVANCE_SECONDS
} = {}) {
    if (!pipeline) return null;

    const totalTokenCount = resolveTokenCount(activationSource, initialLaneCount);
    const maxLaneCount = Math.max(1, totalTokenCount || initialLaneCount);
    const canLoop = !!activationSource && maxLaneCount > initialLaneCount;
    const kvPrefillBaseLaneCount = Math.max(1, Math.floor(initialLaneCount || 1));

    const overlay = createAdvanceOverlay();
    const overlayTouchCleanup = initTouchClickFallback(overlay.root, { selector: 'button' });
    const nextTokenBtn = createNextTokenButton();
    const nextTokenButtonLayoutCleanup = initNextTokenButtonLayoutSync(nextTokenBtn);
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
    let kvModeEnabled = !!appState.kvCacheModeEnabled;
    let kvSessionBaseLaneCount = kvModeEnabled
        ? kvPrefillBaseLaneCount
        : null;

    const resolveKvSessionBase = (laneCountValue) => {
        const initialBase = kvPrefillBaseLaneCount;
        if (!kvModeEnabled) return initialBase;
        if (Number.isFinite(kvSessionBaseLaneCount) && kvSessionBaseLaneCount > 0) {
            return Math.max(1, Math.floor(kvSessionBaseLaneCount));
        }
        // Defensive fallback when toggle event ordering misses transition state.
        // KV mode should treat only the initial pass as prefill, even when the
        // toggle is enabled mid-sequence.
        const fallback = initialBase;
        kvSessionBaseLaneCount = fallback;
        return fallback;
    };

    const syncKvCachePassState = (laneCountValue) => {
        const base = resolveKvSessionBase(laneCountValue);
        const next = Math.max(1, Math.floor(laneCountValue || 1));
        const passIndex = Math.max(0, next - base);
        appState.kvCachePassIndex = passIndex;
        appState.kvCachePrefillActive = kvModeEnabled && passIndex === 0;
    };

    const resolvePassPlan = (laneCountValue) => {
        const totalLaneCount = Math.max(1, Math.floor(laneCountValue || 1));
        const base = resolveKvSessionBase(totalLaneCount);
        const passIndex = Math.max(0, totalLaneCount - base);
        const kvCacheDecodeActive = kvModeEnabled && passIndex > 0;
        const activeLaneCount = kvCacheDecodeActive ? 1 : totalLaneCount;
        const laneLayoutIndices = kvCacheDecodeActive
            ? [Math.max(0, totalLaneCount - 1)]
            : Array.from({ length: totalLaneCount }, (_, idx) => idx);

        const fullLaneTokenIndices = activationSource && typeof activationSource.getLaneTokenIndices === 'function'
            ? activationSource.getLaneTokenIndices(totalLaneCount)
            : Array.from({ length: totalLaneCount }, (_, idx) => idx);
        const laneTokenIndices = kvCacheDecodeActive
            ? [fullLaneTokenIndices[Math.max(0, totalLaneCount - 1)] ?? Math.max(0, totalLaneCount - 1)]
            : fullLaneTokenIndices.slice(0, totalLaneCount);

        return {
            passIndex,
            totalLaneCount,
            activeLaneCount,
            kvCacheDecodeActive,
            laneLayoutIndices,
            laneTokenIndices,
            fullLaneTokenIndices
        };
    };

    const shouldClearKvCacheForPass = ({ passPlan, fromCompletedPass = false } = {}) => {
        if (!kvModeEnabled) return true;
        if (!passPlan || !passPlan.kvCacheDecodeActive) return passPlan?.passIndex === 0;
        if (passPlan.passIndex === 0) return true;
        // Enabling KV mid-sequence should not reuse stale visuals captured
        // under non-KV semantics. We clear once and then bootstrap decode cache.
        return !fromCompletedPass;
    };

    const resolveTrailRuntimeStyleForPass = (passPlan) => {
        const decodeSingleLaneActive = !!(passPlan?.kvCacheDecodeActive && passPlan?.activeLaneCount === 1);
        if (!decodeSingleLaneActive) {
            return {
                opacityMultiplier: 1.0,
                lineWidthMultiplier: 1.0
            };
        }

        const totalLaneCount = Math.max(1, Math.floor(passPlan?.totalLaneCount || passPlan?.activeLaneCount || 1));
        const activeLaneCount = Math.max(1, Math.floor(passPlan?.activeLaneCount || 1));
        const activeLaneScale = getLaneOpacityScale(activeLaneCount);
        const totalLaneScale = getLaneOpacityScale(totalLaneCount);
        // Decode renders one active lane. Compensate for total-lane dimming so
        // "skip to last pass -> enable KV cache" keeps the same single-lane
        // trail brightness as other KV decode entries.
        const laneScaleCompensation = totalLaneScale > 0
            ? (activeLaneScale / totalLaneScale)
            : 1.0;
        // Keep single-lane KV decode trail thickness fixed so it does not vary
        // with total token/layout count for the pass.
        const lineWidthCompensation = KV_DECODE_SINGLE_LANE_TRAIL_WIDTH_MULTIPLIER;

        // Decode-only single-lane trails get a modest opacity bump so they remain
        // legible without looking overdrawn where paths overlap.
        return {
            opacityMultiplier: KV_DECODE_SINGLE_LANE_TRAIL_OPACITY_BOOST * laneScaleCompensation,
            lineWidthMultiplier: lineWidthCompensation
        };
    };

    const applyTrailRuntimeStyleForPass = (passPlan) => {
        const style = resolveTrailRuntimeStyleForPass(passPlan);
        setTrailOpacityRuntimeMultiplier(style.opacityMultiplier);
        setTrailLineWidthRuntimeMultiplier(style.lineWidthMultiplier);
        refreshTrailDisplayScales(pipeline?.engine?.scene);
    };

    const clearOverlay = () => {
        overlay.root.dataset.visible = 'false';
    };

    const updateNextTokenButton = () => {
        if (!nextTokenBtn) return;
        const atEnd = currentLaneCount >= maxLaneCount;
        const shouldShow = passComplete && autoAdvancePaused && !atEnd;
        const next = shouldShow ? 'true' : 'false';
        if (nextTokenBtn.dataset.visible !== next) {
            nextTokenBtn.dataset.visible = next;
            nextTokenBtn.style.display = shouldShow ? '' : 'none';
        }
        nextTokenBtn.disabled = !shouldShow;
        syncNextTokenButtonLayout(nextTokenBtn);
    };

    const updateOverlay = () => {
        if (!passComplete) {
            clearOverlay();
            updateNextTokenButton();
            return;
        }

        const atEnd = currentLaneCount >= maxLaneCount;
        if (autoAdvancePaused && !atEnd) {
            clearOverlay();
            updateNextTokenButton();
            return;
        }

        overlay.root.dataset.visible = 'true';
        overlay.root.dataset.paused = autoAdvancePaused ? 'true' : 'false';

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

        updateNextTokenButton();
    };

    const syncSelectionPanel = (passState, attentionState = null) => {
        if (!selectionPanel) return;
        selectionPanel.updateData?.({
            activationSource,
            laneTokenIndices: passState.laneTokenIndices,
            tokenLabels: passState.tokenLabels,
            attentionTokenIndices: attentionState?.laneTokenIndices || passState.laneTokenIndices,
            attentionTokenLabels: attentionState?.tokenLabels || passState.tokenLabels
        });
        selectionPanel.close?.();
    };

    const syncPromptTokenStrip = (
        passState,
        attentionState = null,
        { showGeneratedLogitChip = false } = {}
    ) => {
        if (!promptTokenStrip || typeof promptTokenStrip.update !== 'function') return;
        const sourceState = attentionState || passState;
        const labels = Array.isArray(sourceState?.tokenLabels)
            ? sourceState.tokenLabels.map((token) => formatTokenLabel(token))
            : [];
        const tokenIndices = Array.isArray(sourceState?.laneTokenIndices)
            ? sourceState.laneTokenIndices.slice(0, labels.length)
            : null;
        const tokenIds = Array.isArray(tokenIndices) && activationSource && typeof activationSource.getTokenId === 'function'
            ? tokenIndices.map((tokenIndex) => (
                Number.isFinite(tokenIndex) ? activationSource.getTokenId(tokenIndex) : null
            ))
            : null;
        const generatedToken = showGeneratedLogitChip
            ? resolveGeneratedLogitToken(activationSource, tokenIndices || [])
            : null;
        promptTokenStrip.update({
            tokenLabels: labels,
            tokenIndices,
            tokenIds,
            generatedToken
        });
    };

    let latestPassState = null;
    let latestAttentionState = null;

    const rebuildPass = ({ laneCount, passState, resetPipeline = false, fromCompletedPass = false } = {}) => {
        const nextLaneCount = Math.max(1, Math.floor(laneCount || 1));
        const passPlan = resolvePassPlan(nextLaneCount);
        const state = passState || buildPassState({
            activationSource,
            laneCount: passPlan.activeLaneCount,
            laneTokenIndices: passPlan.laneTokenIndices,
            laneLayoutIndices: passPlan.laneLayoutIndices,
            totalLaneCount: passPlan.totalLaneCount,
            fallbackTokenLabels,
            fallbackPositionLabels
        });
        const attentionState = buildPassState({
            activationSource,
            laneCount: passPlan.totalLaneCount,
            laneTokenIndices: passPlan.fullLaneTokenIndices,
            laneLayoutIndices: Array.from({ length: passPlan.totalLaneCount }, (_, idx) => idx),
            totalLaneCount: passPlan.totalLaneCount,
            fallbackTokenLabels,
            fallbackPositionLabels
        });
        const laneLayoutIndices = Array.isArray(state.laneLayoutIndices) && state.laneLayoutIndices.length
            ? state.laneLayoutIndices
            : passPlan.laneLayoutIndices;
        const stateTotalLaneCount = Number.isFinite(state.totalLaneCount)
            ? Math.max(1, Math.floor(state.totalLaneCount))
            : passPlan.totalLaneCount;
        syncKvCachePassState(nextLaneCount);
        applyTrailRuntimeStyleForPass(passPlan);

        let preserveCameraPose = false;
        if (resetPipeline) {
            setNumVectorLanes(passPlan.totalLaneCount);
            setAnimationLaneCount(passPlan.totalLaneCount);
            // KV flags are split out so pass-rebuild intent stays readable.
            const clearKvForPass = shouldClearKvCacheForPass({ passPlan, fromCompletedPass });
            const shouldCaptureKvForCompletedPass = !!(kvModeEnabled && fromCompletedPass);
            const shouldReuseKvCache = !!passPlan.kvCacheDecodeActive;
            const shouldBootstrapKvFromActivation = !!(kvModeEnabled && passPlan.kvCacheDecodeActive);
            pipeline.resetForNewPass({
                activationSource,
                laneCount: passPlan.activeLaneCount,
                laneLayoutCount: passPlan.totalLaneCount,
                laneLayoutIndices: passPlan.laneLayoutIndices,
                laneTokenIndices: state.laneTokenIndices,
                kvCacheModeEnabled: kvModeEnabled,
                kvCacheDecodeActive: passPlan.kvCacheDecodeActive,
                preservePreviousTrails: false,
                captureKvCache: shouldCaptureKvForCompletedPass,
                reuseKvCache: shouldReuseKvCache,
                clearKvCache: clearKvForPass,
                bootstrapKvCacheFromActivation: shouldBootstrapKvFromActivation
            });
            const followEnabled = (typeof pipeline.isAutoCameraFollowEnabled === 'function')
                ? pipeline.isAutoCameraFollowEnabled()
                : appState.autoCameraFollow;
            if (followEnabled) {
                pipeline.setAutoCameraFollow?.(true, { immediate: true, resetView: true });
            } else {
                preserveCameraPose = true;
            }
        }

        appState.topEmbedActivated = false;
        appState.lastEqSignature = '';
        appState.lastEqKey = '';

        if (chipCleanup && typeof chipCleanup.dispose === 'function') {
            chipCleanup.dispose();
        }
        const chipPassState = attentionState;
        const chipAnimateLaneIndices = passPlan.kvCacheDecodeActive
            ? [Math.max(0, passPlan.totalLaneCount - 1)]
            : null;
        chipCleanup = addEmbeddingAndTokenChips({
            pipeline,
            laneCount: passPlan.activeLaneCount,
            laneLayoutIndices,
            laneLayoutCount: stateTotalLaneCount,
            activationSource,
            laneTokenIndices: state.laneTokenIndices,
            tokenLabels: state.tokenLabels,
            positionLabels: state.positionLabels,
            chipLaneCount: passPlan.totalLaneCount,
            chipLaneLayoutIndices: chipPassState.laneLayoutIndices,
            chipLaneTokenIndices: chipPassState.laneTokenIndices,
            chipTokenLabels: chipPassState.tokenLabels,
            chipPositionLabels: chipPassState.positionLabels,
            animateChipLaneIndices: chipAnimateLaneIndices,
            drawStaticChipConnectors: true,
            cameraReturnPosition,
            cameraReturnTarget,
            numLayers,
            preserveCameraPose
        });

        applyPhysicalMaterialsToScene(pipeline?.engine?.scene, USE_PHYSICAL_MATERIALS);
        syncSelectionPanel(state, attentionState);
        latestPassState = state;
        latestAttentionState = attentionState;
        syncPromptTokenStrip(state, attentionState, { showGeneratedLogitChip: false });

        currentLaneCount = nextLaneCount;
        passComplete = false;
        autoAdvancePaused = false;
        countdownActive = false;
        remainingMs = countdownMs;
        lastTick = null;
        clearOverlay();
        updateNextTokenButton();
    };

    const handleKvCacheModeChanged = (event) => {
        const detail = event && event.detail ? event.detail : null;
        const nextEnabled = !!(detail && detail.enabled);
        const prevEnabled = (detail && typeof detail.previousEnabled === 'boolean')
            ? detail.previousEnabled
            : kvModeEnabled;
        const isEnablingKv = nextEnabled && !prevEnabled;
        const isDisablingKv = !nextEnabled && prevEnabled;
        kvModeEnabled = nextEnabled;
        appState.kvCacheModeEnabled = nextEnabled;
        if (isEnablingKv) {
            // In KV mode, only the initial pass is treated as prefill. If KV is
            // enabled later, restart directly into decode semantics for the
            // current token count.
            kvSessionBaseLaneCount = kvPrefillBaseLaneCount;
        } else if (isDisablingKv) {
            kvSessionBaseLaneCount = null;
        } else if (nextEnabled && !(Number.isFinite(kvSessionBaseLaneCount) && kvSessionBaseLaneCount > 0)) {
            // Guard against event ordering that skips the transition branch.
            kvSessionBaseLaneCount = kvPrefillBaseLaneCount;
        }
        syncKvCachePassState(currentLaneCount);
        pipeline?.dispatchEvent?.(new Event('progress'));
        if (isEnablingKv) {
            // Always restart immediately when enabling so the active pass is
            // rebuilt using KV semantics for the current token count.
            rebuildPass({ laneCount: currentLaneCount, resetPipeline: true });
        } else if (!passComplete) {
            rebuildPass({ laneCount: currentLaneCount, resetPipeline: true });
        } else {
            updateOverlay();
        }
    };

    if (typeof window !== 'undefined' && typeof window.addEventListener === 'function') {
        window.addEventListener('kvCacheModeChanged', handleKvCacheModeChanged);
    }

    rebuildPass({
        laneCount: currentLaneCount,
        passState: initialPassState,
        resetPipeline: kvModeEnabled
    });

    const hasNextForwardPass = () => currentLaneCount < maxLaneCount;
    const hasLastForwardPass = hasNextForwardPass;
    const markNoFurtherPasses = () => {
        passComplete = true;
        countdownActive = false;
        autoAdvancePaused = true;
        updateOverlay();
    };
    const prepareUiForImmediatePassJump = () => {
        autoAdvancePaused = false;
        countdownActive = false;
        clearOverlay();
        updateNextTokenButton();
    };

    if (!canLoop) {
        clearOverlay();
        return {
            advance: () => false,
            requestNextForwardPass: () => false,
            requestLastForwardPass: () => false,
            hasNextForwardPass: () => false,
            hasLastForwardPass: () => false,
            isForwardPassJumpPending: () => false,
            isNextForwardPassPending: () => false,
            dispose: () => {
                if (typeof window !== 'undefined' && typeof window.removeEventListener === 'function') {
                    window.removeEventListener('kvCacheModeChanged', handleKvCacheModeChanged);
                }
                if (chipCleanup?.dispose) chipCleanup.dispose();
                if (rafId && typeof cancelAnimationFrame === 'function') cancelAnimationFrame(rafId);
                if (overlayTouchCleanup) overlayTouchCleanup();
                promptTokenStrip?.dispose?.();
            }
        };
    }

    const advanceToNextPass = ({ fromCompletedPass = true } = {}) => {
        if (!hasNextForwardPass()) {
            markNoFurtherPasses();
            return false;
        }
        const nextLane = Math.min(maxLaneCount, currentLaneCount + 1);
        rebuildPass({
            laneCount: nextLane,
            resetPipeline: true,
            fromCompletedPass: !!fromCompletedPass
        });
        return true;
    };

    const advanceToLastPass = ({ fromCompletedPass = true } = {}) => {
        if (!hasLastForwardPass()) {
            markNoFurtherPasses();
            return false;
        }
        rebuildPass({
            laneCount: maxLaneCount,
            resetPipeline: true,
            fromCompletedPass: !!fromCompletedPass
        });
        return true;
    };

    const requestNextForwardPass = () => {
        if (!hasNextForwardPass()) {
            return false;
        }
        // "Next pass" is an immediate jump. If the current pass is already
        // complete we preserve/cache from visuals; otherwise we rebuild the
        // next pass and hydrate decode cache from activation data.
        const currentPassComplete = passComplete || (
            typeof pipeline?.isForwardPassComplete === 'function' && pipeline.isForwardPassComplete()
        );
        prepareUiForImmediatePassJump();
        return advanceToNextPass({ fromCompletedPass: currentPassComplete });
    };

    const requestLastForwardPass = () => {
        if (!hasLastForwardPass()) {
            return false;
        }
        prepareUiForImmediatePassJump();
        const currentPassComplete = passComplete || (
            typeof pipeline?.isForwardPassComplete === 'function' && pipeline.isForwardPassComplete()
        );
        return advanceToLastPass({ fromCompletedPass: currentPassComplete });
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

    if (nextTokenBtn) {
        nextTokenBtn.onclick = (event) => {
            event.preventDefault();
            if (!passComplete || currentLaneCount >= maxLaneCount) return;
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
                syncPromptTokenStrip(latestPassState, latestAttentionState, { showGeneratedLogitChip: true });
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
        requestNextForwardPass,
        requestLastForwardPass,
        hasNextForwardPass,
        hasLastForwardPass,
        isForwardPassJumpPending: () => false,
        isNextForwardPassPending: () => false,
        dispose: () => {
            if (typeof window !== 'undefined' && typeof window.removeEventListener === 'function') {
                window.removeEventListener('kvCacheModeChanged', handleKvCacheModeChanged);
            }
            if (chipCleanup?.dispose) chipCleanup.dispose();
            if (rafId && typeof cancelAnimationFrame === 'function') cancelAnimationFrame(rafId);
            if (overlayTouchCleanup) overlayTouchCleanup();
            if (nextTokenButtonLayoutCleanup) nextTokenButtonLayoutCleanup();
            promptTokenStrip?.dispose?.();
        }
    };
}

export default initGenerationController;
