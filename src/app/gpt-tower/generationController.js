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
import {
    KV_CACHE_MODE_CHANGED_EVENT,
    dispatchKvCacheModeStateSync
} from '../../state/kvCacheModeEvents.js';
import { getIncompleteUtf8TokenDisplay } from '../../utils/tokenEncodingNotes.js';
import { resolveLogitEntryText } from '../../utils/logitTokenText.js';
import { addEmbeddingAndTokenChips } from './tokenChips.js';
import { formatTokenLabel } from './tokenLabels.js';
import { resolveLogitTokenSeed } from './logitColor.js';
import {
    resolveGenerationRoute,
    syncGenerationRoute
} from './generationRoute.js';
import {
    resolveKvCachePassMode,
    resolveKvPrefillBaseLaneCount
} from './kvCachePassMode.js';
import { initTouchClickFallback } from '../../ui/touchClickFallback.js';

const DEFAULT_ADVANCE_SECONDS = 10;
const KV_DECODE_SINGLE_LANE_TRAIL_WIDTH_MULTIPLIER = 8.0;
const KV_DECODE_SINGLE_LANE_TRAIL_OPACITY_BOOST = 2.2;
const NEXT_TOKEN_DESKTOP_MEDIA_QUERY = '(min-width: 881px) and (min-aspect-ratio: 1/1)';
const NEXT_TOKEN_MOBILE_MEDIA_QUERY = '(max-aspect-ratio: 1/1), (max-width: 880px)';
const NEXT_TOKEN_PANEL_GAP_PX = 18;
const NEXT_TOKEN_VIEWPORT_GUTTER_PX = 12;
const PASS_INTRO_ENGINE_PAUSE_REASON = 'generation-pass-intro';
const NEXT_TOKEN_BTN_LABEL = 'Next Token';
const RESTART_GENERATION_BTN_LABEL = 'Restart generation';

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

function resolveGeneratedLogitTokenId(entry) {
    const rawTokenId = Number(entry?.token_id ?? entry?.tokenId);
    return Number.isFinite(rawTokenId) ? Math.floor(rawTokenId) : null;
}

function findMatchingGeneratedLogitEntryIndex(logitRow, {
    nextTokenRaw = null,
    nextTokenId = null
} = {}) {
    if (!Array.isArray(logitRow) || !logitRow.length) return -1;

    if (Number.isFinite(nextTokenId)) {
        for (let i = 0; i < logitRow.length; i += 1) {
            if (resolveGeneratedLogitTokenId(logitRow[i]) === Math.floor(nextTokenId)) {
                return i;
            }
        }
    }

    if (typeof nextTokenRaw === 'string' && nextTokenRaw.length) {
        for (let i = 0; i < logitRow.length; i += 1) {
            if (typeof logitRow[i]?.token === 'string' && logitRow[i].token === nextTokenRaw) {
                return i;
            }
        }
    }

    return -1;
}

function buildResolvedGeneratedToken({
    tokenRaw = null,
    tokenId = null,
    tokenIndex = null,
    logitEntry = null,
    seedFallbackIndex = 0
} = {}) {
    const resolvedTokenId = Number.isFinite(tokenId) ? Math.floor(tokenId) : null;
    const incompleteTokenDisplay = getIncompleteUtf8TokenDisplay(resolvedTokenId);
    const resolvedTokenRaw = incompleteTokenDisplay
        || ((typeof tokenRaw === 'string' && tokenRaw.length)
            ? tokenRaw
            : resolveLogitEntryText(logitEntry));
    const tokenText = resolvedTokenRaw
        ? formatTokenLabel(sanitizeLogitToken(resolvedTokenRaw))
        : '';
    const fallbackText = Number.isFinite(resolvedTokenId) ? `#${resolvedTokenId}` : '';
    const tokenLabel = tokenText || fallbackText;
    if (!tokenLabel) return null;

    const seedSource = logitEntry && typeof logitEntry === 'object'
        ? logitEntry
        : {
            token_id: resolvedTokenId,
            token: resolvedTokenRaw || tokenLabel
        };

    return {
        tokenLabel,
        tokenId: resolvedTokenId,
        tokenIndex: Number.isFinite(tokenIndex) ? Math.floor(tokenIndex) : null,
        seed: resolveLogitTokenSeed(seedSource, seedFallbackIndex),
        selectionLabel: `Chosen token: ${tokenLabel}`,
        logitEntry: (logitEntry && typeof logitEntry === 'object') ? logitEntry : null
    };
}

export function resolveGeneratedLogitToken(activationSource, laneTokenIndices = []) {
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
    const nextTokenIndex = lastTokenIndex + 1;
    const hasNextToken = Number.isFinite(tokenCount) && nextTokenIndex < tokenCount;
    const nextTokenRaw = hasNextToken && typeof activationSource.getTokenString === 'function'
        ? activationSource.getTokenString(nextTokenIndex)
        : null;
    const nextTokenId = hasNextToken && typeof activationSource.getTokenId === 'function'
        ? activationSource.getTokenId(nextTokenIndex)
        : null;

    let bestIdx = -1;
    let bestProb = -Infinity;
    for (let i = 0; i < logitRow.length; i += 1) {
        const entry = logitRow[i];
        const prob = Number(entry?.prob);
        if (Number.isFinite(prob) && prob > bestProb) {
            bestProb = prob;
            bestIdx = i;
        }
    }

    const matchedNextIdx = hasNextToken
        ? findMatchingGeneratedLogitEntryIndex(logitRow, {
            nextTokenRaw,
            nextTokenId
        })
        : -1;
    if (hasNextToken) {
        const matchedEntry = matchedNextIdx !== -1 ? logitRow[matchedNextIdx] : null;
        return buildResolvedGeneratedToken({
            tokenRaw: nextTokenRaw,
            tokenId: nextTokenId,
            tokenIndex: nextTokenIndex,
            logitEntry: matchedEntry,
            seedFallbackIndex: matchedNextIdx !== -1 ? matchedNextIdx : nextTokenIndex
        });
    }

    const chosenIdx = bestIdx !== -1 ? bestIdx : 0;
    const chosenEntry = logitRow[chosenIdx];
    if (!chosenEntry || typeof chosenEntry !== 'object') return null;
    return buildResolvedGeneratedToken({
        tokenRaw: chosenEntry.token,
        tokenId: resolveGeneratedLogitTokenId(chosenEntry),
        tokenIndex: null,
        logitEntry: chosenEntry,
        seedFallbackIndex: chosenIdx
    });
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
    btn.textContent = 'Next Token';
    btn.title = 'Advance to next token';
    btn.setAttribute('aria-label', 'Advance to next token');
    btn.dataset.visible = 'false';
    btn.style.display = 'none';
    document.body.appendChild(btn);
    return btn;
}

function initNextTokenButtonTouchFallback(button) {
    if (!button || typeof document === 'undefined' || !document.body) return null;
    // On mobile the button is mounted directly under body, so reuse the shared
    // tap fallback path that synthesizes activation when the native click is missed.
    return initTouchClickFallback(document.body, {
        selector: 'body > #nextTokenBtn',
        tapSlopPx: 20
    });
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

function isMobileNextTokenInlineLayout() {
    if (typeof window === 'undefined') return false;
    if (typeof window.matchMedia === 'function') {
        return window.matchMedia(NEXT_TOKEN_MOBILE_MEDIA_QUERY).matches;
    }
    return window.innerWidth <= 880 || window.innerHeight >= window.innerWidth;
}

function parsePixelValue(value) {
    const parsed = Number.parseFloat(value);
    return Number.isFinite(parsed) ? parsed : null;
}

function resolveCssPixelLength(element, property, cssValue) {
    if (!element || typeof window === 'undefined' || !property || !cssValue) return null;

    const previousValue = element.style.getPropertyValue(property);
    const previousPriority = element.style.getPropertyPriority(property);
    element.style.setProperty(property, cssValue, 'important');
    const resolvedValue = parsePixelValue(window.getComputedStyle(element).getPropertyValue(property));

    if (previousValue) {
        element.style.setProperty(property, previousValue, previousPriority);
    } else {
        element.style.removeProperty(property);
    }

    return resolvedValue;
}

function resetNextTokenButtonMobileStripLayout(button) {
    if (!button) return;
    delete button.dataset.mobileStripLayout;
    button.style.removeProperty('bottom');
}

function mountNextTokenButtonInBody(button) {
    if (!button || typeof document === 'undefined' || !document.body) return;
    if (button.parentElement !== document.body) {
        document.body.appendChild(button);
    }
}

function syncNextTokenButtonTopControlsLayout(button) {
    if (!button || typeof document === 'undefined') return false;
    if (!isDesktopNextTokenDockLayout()) return false;

    const topControls = document.getElementById('topControls');
    const pauseBtn = document.getElementById('pauseBtn');
    if (!topControls || !pauseBtn) return false;

    if (button.parentElement !== topControls || button.nextElementSibling !== pauseBtn) {
        topControls.insertBefore(button, pauseBtn);
    }
    button.dataset.layout = 'top-controls';
    button.style.removeProperty('left');
    button.style.removeProperty('right');
    button.style.removeProperty('bottom');
    return true;
}

function syncNextTokenButtonMobileStripLayout(
    button,
    {
        promptTokenStrip = null
    } = {}
) {
    resetNextTokenButtonMobileStripLayout(button);
    if (!button || typeof document === 'undefined' || typeof window === 'undefined') return false;
    if (!isMobileNextTokenInlineLayout()) return false;

    const promptTokenStripEl = promptTokenStrip || document.getElementById('promptTokenStrip');
    if (!promptTokenStripEl || promptTokenStripEl.dataset.visible !== 'true') return false;

    const stripRect = resolveVisibleRect(promptTokenStripEl);
    const buttonRect = resolveVisibleRect(button);
    const buttonWidth = buttonRect?.width || button.offsetWidth || 0;
    if (!stripRect || !(buttonWidth > 0)) return false;

    const computedStyle = window.getComputedStyle(button);
    const baseRightPx = parsePixelValue(computedStyle.getPropertyValue('right'))
        ?? resolveCssPixelLength(button, 'right', 'var(--next-token-btn-right-base)');
    const baseBottomPx = resolveCssPixelLength(button, 'bottom', 'var(--next-token-btn-bottom-base)');
    const inlineGapPx = resolveCssPixelLength(button, 'margin-right', 'var(--next-token-btn-inline-gap, 0px)') ?? 0;
    const viewportWidth = window.innerWidth || document.documentElement?.clientWidth || 0;
    if (!Number.isFinite(baseRightPx) || !Number.isFinite(baseBottomPx) || !(viewportWidth > 0)) return false;

    const buttonLeft = viewportWidth - baseRightPx - buttonWidth;
    const fitsInline = (stripRect.right + inlineGapPx) <= buttonLeft;
    if (!fitsInline) return false;

    // Keep the button on the bottom row when the chip strip leaves enough
    // horizontal room to the right on narrow/mobile layouts.
    button.dataset.mobileStripLayout = 'inline';
    button.style.bottom = `${baseBottomPx}px`;
    return true;
}

function syncNextTokenButtonLayout(
    button,
    {
        hudStack = null,
        detailPanel = null,
        resizeHandle = null,
        promptTokenStrip = null
    } = {}
) {
    if (!button || typeof document === 'undefined') return;
    resetNextTokenButtonMobileStripLayout(button);
    if (syncNextTokenButtonTopControlsLayout(button)) return;
    mountNextTokenButtonInBody(button);

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
        syncNextTokenButtonMobileStripLayout(button, { promptTokenStrip });
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

function initNextTokenButtonLayoutSync(
    button,
    {
        promptTokenStrip = null
    } = {}
) {
    if (!button || typeof document === 'undefined' || typeof window === 'undefined') return null;

    const hudStack = document.getElementById('hudStack');
    const detailPanel = document.getElementById('detailPanel');
    const resizeHandle = document.getElementById('detailPanelResizeHandle');
    const promptTokenStripEl = promptTokenStrip || document.getElementById('promptTokenStrip');
    let syncRafId = null;
    let resizeObserver = null;
    let mutationObserver = null;

    const runSync = () => {
        syncRafId = null;
        syncNextTokenButtonLayout(button, {
            hudStack,
            detailPanel,
            resizeHandle,
            promptTokenStrip: promptTokenStripEl
        });
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
        if (promptTokenStripEl) resizeObserver.observe(promptTokenStripEl);
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
        if (promptTokenStripEl) {
            mutationObserver.observe(promptTokenStripEl, {
                attributes: true,
                attributeFilter: ['data-visible']
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

function waitForAnimationFrames(frameCount = 1) {
    const safeCount = Math.max(0, Math.floor(frameCount));
    if (safeCount <= 0 || typeof requestAnimationFrame !== 'function') {
        return Promise.resolve();
    }
    return new Promise((resolve) => {
        let remaining = safeCount;
        const step = () => {
            remaining -= 1;
            if (remaining <= 0) {
                resolve();
                return;
            }
            requestAnimationFrame(step);
        };
        requestAnimationFrame(step);
    });
}

export function initGenerationController({
    pipeline,
    activationSource,
    initialLaneCount,
    baseLaneCount = initialLaneCount,
    initialPassState,
    fallbackTokenLabels = [],
    fallbackPositionLabels = [],
    numLayers,
    cameraReturnPosition,
    cameraReturnTarget,
    selectionPanel,
    promptTokenStrip,
    passIntroOverlay = null,
    startupOverviewHoldMs = 1000,
    startupOverviewTransitionMs = 1400,
    autoAdvanceSeconds = DEFAULT_ADVANCE_SECONDS
} = {}) {
    if (!pipeline) return null;

    const totalTokenCount = resolveTokenCount(activationSource, initialLaneCount);
    const maxLaneCount = Math.max(1, totalTokenCount || initialLaneCount);
    const generationBaseLaneCount = Math.max(
        1,
        Math.min(maxLaneCount, Math.floor(baseLaneCount || initialLaneCount || 1))
    );
    const canLoop = !!activationSource && maxLaneCount > generationBaseLaneCount;
    const kvPrefillBaseLaneCount = resolveKvPrefillBaseLaneCount({
        initialLaneCount,
        baseLaneCount: generationBaseLaneCount
    });

    const overlay = createAdvanceOverlay();
    const overlayTouchCleanup = initTouchClickFallback(overlay.root, { selector: 'button' });
    const nextTokenBtn = createNextTokenButton();
    const nextTokenButtonTouchCleanup = initNextTokenButtonTouchFallback(nextTokenBtn);
    const promptTokenStripEl = promptTokenStrip?.getRootElement?.() || document.getElementById('promptTokenStrip');
    const nextTokenButtonLayoutCleanup = initNextTokenButtonLayoutSync(nextTokenBtn, {
        promptTokenStrip: promptTokenStripEl
    });
    let currentLaneCount = Math.max(1, Math.floor(initialLaneCount || 1));
    let passComplete = false;
    let forwardPassJumpPending = false;
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
    const syncCurrentRoute = ({
        laneCount = currentLaneCount,
        historyMode = 'replace'
    } = {}) => syncGenerationRoute({
        laneCount,
        baseLaneCount: generationBaseLaneCount,
        maxLaneCount,
        kvCacheModeEnabled: kvModeEnabled,
        historyMode
    });
    const resolveRouteStateFromCurrentUrl = () => resolveGenerationRoute(window.location, {
        defaultLaneCount: generationBaseLaneCount,
        baseLaneCount: generationBaseLaneCount,
        maxLaneCount
    });

    const resolveKvSessionBase = (laneCountValue) => {
        const initialBase = kvPrefillBaseLaneCount;
        if (!kvModeEnabled) return initialBase;
        if (Number.isFinite(kvSessionBaseLaneCount) && kvSessionBaseLaneCount > 0) {
            return Math.max(1, Math.floor(kvSessionBaseLaneCount));
        }
        // Defensive fallback when toggle event ordering misses transition state.
        // KV mode should compare against the prompt/base token window, even
        // when the toggle is enabled after the scene has already advanced.
        const fallback = initialBase;
        kvSessionBaseLaneCount = fallback;
        return fallback;
    };

    const syncKvCachePassState = (laneCountValue) => {
        const base = resolveKvSessionBase(laneCountValue);
        const passMode = resolveKvCachePassMode({
            laneCount: laneCountValue,
            kvModeEnabled,
            prefillBaseLaneCount: base
        });
        appState.kvCachePassIndex = passMode.passIndex;
        appState.kvCachePrefillActive = passMode.kvCachePrefillActive;
    };

    const applyKvModeEnabled = (nextEnabled, {
        syncUi = false
    } = {}) => {
        const prevEnabled = kvModeEnabled;
        const next = !!nextEnabled;
        const isEnablingKv = next && !prevEnabled;
        const isDisablingKv = !next && prevEnabled;

        kvModeEnabled = next;
        appState.kvCacheModeEnabled = next;
        if (isEnablingKv) {
            // In KV mode, the prompt/base token window is the prefill pass. If
            // KV is enabled later, rebuilding the current larger token window
            // still resolves correctly as decode against that base.
            kvSessionBaseLaneCount = kvPrefillBaseLaneCount;
        } else if (isDisablingKv) {
            kvSessionBaseLaneCount = null;
        } else if (next && !(Number.isFinite(kvSessionBaseLaneCount) && kvSessionBaseLaneCount > 0)) {
            // Guard against event ordering that skips the transition branch.
            kvSessionBaseLaneCount = kvPrefillBaseLaneCount;
        }

        syncKvCachePassState(currentLaneCount);
        if (syncUi) {
            dispatchKvCacheModeStateSync(next);
        }

        return {
            prevEnabled,
            nextEnabled: next,
            isEnablingKv,
            isDisablingKv
        };
    };

    const resolvePassPlan = (laneCountValue) => {
        const base = resolveKvSessionBase(laneCountValue);
        const passMode = resolveKvCachePassMode({
            laneCount: laneCountValue,
            kvModeEnabled,
            prefillBaseLaneCount: base
        });
        const totalLaneCount = passMode.totalLaneCount;
        const passIndex = passMode.passIndex;
        const kvCacheDecodeActive = passMode.kvCacheDecodeActive;
        const activeLaneCount = passMode.activeLaneCount;
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
        const shouldShow = passComplete && autoAdvancePaused;
        const isRestartAction = passComplete && autoAdvancePaused && atEnd;
        const buttonLabel = isRestartAction ? RESTART_GENERATION_BTN_LABEL : NEXT_TOKEN_BTN_LABEL;
        const buttonTitle = isRestartAction ? RESTART_GENERATION_BTN_LABEL : 'Advance to next token';
        const next = shouldShow ? 'true' : 'false';
        if (nextTokenBtn.dataset.visible !== next) {
            nextTokenBtn.dataset.visible = next;
            nextTokenBtn.style.display = shouldShow ? '' : 'none';
        }
        nextTokenBtn.disabled = !shouldShow;
        if (nextTokenBtn.textContent !== buttonLabel) {
            nextTokenBtn.textContent = buttonLabel;
        }
        if (nextTokenBtn.title !== buttonTitle) {
            nextTokenBtn.title = buttonTitle;
        }
        if (nextTokenBtn.getAttribute('aria-label') !== buttonTitle) {
            nextTokenBtn.setAttribute('aria-label', buttonTitle);
        }
        syncNextTokenButtonLayout(nextTokenBtn, { promptTokenStrip: promptTokenStripEl });
    };

    const updateOverlay = () => {
        if (!passComplete) {
            clearOverlay();
            updateNextTokenButton();
            return;
        }

        const atEnd = currentLaneCount >= maxLaneCount;
        if (atEnd) {
            clearOverlay();
            updateNextTokenButton();
            return;
        }

        if (autoAdvancePaused) {
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

        if (overlay.titlePrefix) overlay.titlePrefix.textContent = 'Going to next token in';
        if (overlay.countdownWrap) overlay.countdownWrap.style.display = '';

        const nextTokenLabel = (activationSource && typeof activationSource.getTokenString === 'function')
            ? activationSource.getTokenString(nextTokenIndex)
            : null;
        const formatted = nextTokenLabel ? formatTokenLabel(nextTokenLabel) : '';
        const tokenLine = formatted
            ? `Next token ${nextTokenNumber} / ${maxLaneCount}: ${formatted}`
            : `Next token ${nextTokenNumber} / ${maxLaneCount}`;
        if (overlay.tokenEl) overlay.tokenEl.textContent = tokenLine;

        const progress = Math.max(0, Math.min(1, 1 - remainingMs / countdownMs));
        if (overlay.barFill) {
            overlay.barFill.style.width = `${(progress * 100).toFixed(1)}%`;
        }

        if (overlay.stayBtn) {
            overlay.stayBtn.textContent = autoAdvancePaused ? 'Resume auto' : 'Stay';
        }
        if (overlay.advanceBtn) {
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

    const playPendingPassIntro = async ({
        passState,
        attentionState = null,
        previousTokenCount = null
    } = {}) => {
        if (!passIntroOverlay || typeof passIntroOverlay.play !== 'function') return;
        const sourceState = attentionState || passState;
        if (!sourceState) return;

        let startupCameraIntroPromise = null;
        await passIntroOverlay.play({
            laneCount: sourceState.totalLaneCount ?? sourceState.laneTokenIndices?.length ?? 0,
            laneTokenIndices: sourceState.laneTokenIndices,
            tokenLabels: sourceState.tokenLabels,
            presentation: 'jump-append',
            previousTokenCount,
            onBeforeHide: async () => {
                if (!startupCameraIntroPromise) {
                    startupCameraIntroPromise = Promise.resolve(
                        pipeline?.playStartupCameraIntro?.({
                            holdMs: startupOverviewHoldMs,
                            transitionMs: startupOverviewTransitionMs,
                            replay: true
                        }) ?? false
                    );
                }
                await waitForAnimationFrames(2);
            }
        });

        await (startupCameraIntroPromise ?? pipeline?.playStartupCameraIntro?.({
            holdMs: startupOverviewHoldMs,
            transitionMs: startupOverviewTransitionMs,
            replay: true
        }));
        await waitForAnimationFrames(1);
    };

    let latestPassState = null;
    let latestAttentionState = null;

    const rebuildPass = ({
        laneCount,
        passState,
        resetPipeline = false,
        fromCompletedPass = false,
        routeHistory = 'replace'
    } = {}) => {
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
        if (routeHistory !== 'ignore') {
            syncCurrentRoute({
                laneCount: nextLaneCount,
                historyMode: routeHistory
            });
        }
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
        const {
            isEnablingKv
        } = applyKvModeEnabled(detail && detail.enabled);
        pipeline?.dispatchEvent?.(new Event('progress'));
        if (isEnablingKv) {
            // Always restart immediately when enabling so the active pass is
            // rebuilt using KV semantics for the current token count.
            rebuildPass({ laneCount: currentLaneCount, resetPipeline: true });
        } else if (!passComplete) {
            rebuildPass({ laneCount: currentLaneCount, resetPipeline: true });
        } else {
            syncCurrentRoute({
                laneCount: currentLaneCount,
                historyMode: 'replace'
            });
            updateOverlay();
        }
    };

    if (typeof window !== 'undefined' && typeof window.addEventListener === 'function') {
        window.addEventListener(KV_CACHE_MODE_CHANGED_EVENT, handleKvCacheModeChanged);
    }

    const syncFromUrl = ({ historyMode = 'replace' } = {}) => {
        const routeState = resolveRouteStateFromCurrentUrl();
        const targetLaneCount = routeState.laneCount;
        const targetKvModeEnabled = !!routeState.kvCacheModeEnabled;
        const laneCountChanged = targetLaneCount !== currentLaneCount;
        const kvModeChanged = targetKvModeEnabled !== kvModeEnabled;

        if (!laneCountChanged && !kvModeChanged) {
            syncCurrentRoute({
                laneCount: currentLaneCount,
                historyMode
            });
            return false;
        }
        if (forwardPassJumpPending) return false;
        if (kvModeChanged) {
            applyKvModeEnabled(targetKvModeEnabled, { syncUi: true });
        }
        rebuildPass({
            laneCount: targetLaneCount,
            resetPipeline: true,
            routeHistory: historyMode
        });
        updateOverlay();
        updateNextTokenButton();
        return true;
    };

    const handleRoutePopState = () => {
        syncFromUrl({ historyMode: 'replace' });
    };

    if (typeof window !== 'undefined' && typeof window.addEventListener === 'function') {
        const previousRoutePopstateListener = window.__llmVisualizedGenerationRoutePopstateListener;
        if (typeof previousRoutePopstateListener === 'function') {
            window.removeEventListener('popstate', previousRoutePopstateListener);
        }
        window.__llmVisualizedGenerationRoutePopstateListener = handleRoutePopState;
        window.addEventListener('popstate', handleRoutePopState);
    }

    const initialPassPlan = resolvePassPlan(currentLaneCount);
    rebuildPass({
        laneCount: currentLaneCount,
        passState: initialPassPlan.kvCacheDecodeActive ? null : initialPassState,
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
            isForwardPassJumpPending: () => forwardPassJumpPending,
            isNextForwardPassPending: () => forwardPassJumpPending,
            syncFromUrl,
            dispose: () => {
                if (typeof window !== 'undefined' && typeof window.removeEventListener === 'function') {
                    window.removeEventListener(KV_CACHE_MODE_CHANGED_EVENT, handleKvCacheModeChanged);
                    if (window.__llmVisualizedGenerationRoutePopstateListener === handleRoutePopState) {
                        delete window.__llmVisualizedGenerationRoutePopstateListener;
                    }
                    window.removeEventListener('popstate', handleRoutePopState);
                }
                if (chipCleanup?.dispose) chipCleanup.dispose();
                if (rafId && typeof cancelAnimationFrame === 'function') cancelAnimationFrame(rafId);
                if (overlayTouchCleanup) overlayTouchCleanup();
                if (nextTokenButtonTouchCleanup) nextTokenButtonTouchCleanup();
                if (nextTokenButtonLayoutCleanup) nextTokenButtonLayoutCleanup();
                promptTokenStrip?.dispose?.();
            }
        };
    }

    const runForwardPassJump = ({
        targetLaneCount,
        fromCompletedPass = true,
        historyMode = 'push'
    } = {}) => {
        if (forwardPassJumpPending) return false;

        const nextLaneCount = Math.max(1, Math.floor(targetLaneCount || 1));
        if (nextLaneCount <= currentLaneCount) {
            if (!hasNextForwardPass()) {
                markNoFurtherPasses();
            }
            return false;
        }

        forwardPassJumpPending = true;
        passComplete = false;
        autoAdvancePaused = false;
        countdownActive = false;
        clearOverlay();
        updateNextTokenButton();

        Promise.resolve().then(async () => {
            const engine = pipeline?.engine;
            engine?.pause?.(PASS_INTRO_ENGINE_PAUSE_REASON);
            const priorTokenCount = currentLaneCount;
            try {
                rebuildPass({
                    laneCount: nextLaneCount,
                    resetPipeline: true,
                    fromCompletedPass: !!fromCompletedPass,
                    routeHistory: historyMode
                });
                await playPendingPassIntro({
                    passState: latestPassState,
                    attentionState: latestAttentionState,
                    previousTokenCount: priorTokenCount
                });
            } catch (err) {
                console.error('Forward-pass jump intro failed:', err);
            } finally {
                engine?.resume?.(PASS_INTRO_ENGINE_PAUSE_REASON);
                forwardPassJumpPending = false;
                updateOverlay();
                updateNextTokenButton();
            }
        });

        return true;
    };

    const advanceToNextPass = ({
        fromCompletedPass = true,
        historyMode = 'push'
    } = {}) => {
        if (!hasNextForwardPass()) {
            markNoFurtherPasses();
            return false;
        }
        return runForwardPassJump({
            targetLaneCount: Math.min(maxLaneCount, currentLaneCount + 1),
            fromCompletedPass,
            historyMode
        });
    };

    const advanceToLastPass = ({
        fromCompletedPass = true,
        historyMode = 'push'
    } = {}) => {
        if (!hasLastForwardPass()) {
            markNoFurtherPasses();
            return false;
        }
        return runForwardPassJump({
            targetLaneCount: maxLaneCount,
            fromCompletedPass,
            historyMode
        });
    };

    const restartGeneration = ({ historyMode = 'push' } = {}) => {
        if (forwardPassJumpPending) return false;

        forwardPassJumpPending = true;
        passComplete = false;
        autoAdvancePaused = false;
        countdownActive = false;
        clearOverlay();
        updateNextTokenButton();

        Promise.resolve().then(async () => {
            const engine = pipeline?.engine;
            engine?.pause?.(PASS_INTRO_ENGINE_PAUSE_REASON);
            try {
                rebuildPass({
                    laneCount: generationBaseLaneCount,
                    resetPipeline: true,
                    routeHistory: historyMode
                });
                syncPromptTokenStrip(latestPassState, latestAttentionState, { showGeneratedLogitChip: false });
            } catch (err) {
                console.error('Restart generation failed:', err);
            } finally {
                engine?.resume?.(PASS_INTRO_ENGINE_PAUSE_REASON);
                forwardPassJumpPending = false;
                updateOverlay();
                updateNextTokenButton();
            }
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
        return advanceToNextPass({
            fromCompletedPass: currentPassComplete,
            historyMode: 'push'
        });
    };

    const requestLastForwardPass = () => {
        if (!hasLastForwardPass()) {
            return false;
        }
        prepareUiForImmediatePassJump();
        const currentPassComplete = passComplete || (
            typeof pipeline?.isForwardPassComplete === 'function' && pipeline.isForwardPassComplete()
        );
        return advanceToLastPass({
            fromCompletedPass: currentPassComplete,
            historyMode: 'push'
        });
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
            advanceToNextPass({ historyMode: 'push' });
        };
    }

    if (nextTokenBtn) {
        nextTokenBtn.onclick = (event) => {
            event.preventDefault();
            if (!passComplete) return;
            if (currentLaneCount >= maxLaneCount) {
                restartGeneration({ historyMode: 'push' });
                return;
            }
            advanceToNextPass({ historyMode: 'push' });
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
                const atLastPass = currentLaneCount >= maxLaneCount;
                autoAdvancePaused = atLastPass ? true : autoAdvancePaused;
                countdownActive = atLastPass ? false : !autoAdvancePaused;
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
                advanceToNextPass({ historyMode: 'push' });
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
        isForwardPassJumpPending: () => forwardPassJumpPending,
        isNextForwardPassPending: () => forwardPassJumpPending,
        syncFromUrl,
        dispose: () => {
            if (typeof window !== 'undefined' && typeof window.removeEventListener === 'function') {
                window.removeEventListener(KV_CACHE_MODE_CHANGED_EVENT, handleKvCacheModeChanged);
                if (window.__llmVisualizedGenerationRoutePopstateListener === handleRoutePopState) {
                    delete window.__llmVisualizedGenerationRoutePopstateListener;
                }
                window.removeEventListener('popstate', handleRoutePopState);
            }
            if (chipCleanup?.dispose) chipCleanup.dispose();
            if (rafId && typeof cancelAnimationFrame === 'function') cancelAnimationFrame(rafId);
            if (overlayTouchCleanup) overlayTouchCleanup();
            if (nextTokenButtonTouchCleanup) nextTokenButtonTouchCleanup();
            if (nextTokenButtonLayoutCleanup) nextTokenButtonLayoutCleanup();
            promptTokenStrip?.dispose?.();
        }
    };
}

export default initGenerationController;
