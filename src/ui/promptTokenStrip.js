import { appState } from '../state/appState.js';
import {
    TOKEN_CHIP_HOVER_SYNC_EVENT,
    dispatchTokenChipHoverSync,
    normalizeTokenChipEntry,
    tokenChipEntriesMatch
} from './tokenChipHoverSync.js';
import { initTouchClickFallback } from './touchClickFallback.js';
import {
    applyTokenChipColors,
    buildPromptTokenChipEntries,
    setActivePromptTokenChipEntries
} from './tokenChipColorUtils.js';

const STRIP_ID = 'promptTokenStrip';
const PROMPT_TOKEN_STRIP_HOVER_SOURCE = 'prompt-token-strip';
const PROMPT_TOKEN_STRIP_COLLISION_GAP_PX = 12;
const PROMPT_TOKEN_STRIP_COLLISION_MAX_WIDTH_VAR = '--prompt-token-strip-collision-max-width';

function normalizeTokenText(token) {
    if (token === null || token === undefined) return '';
    const raw = String(token);
    if (!raw.length) return '';
    return raw
        .replace(/\r/g, '')
        .replace(/\n/g, '\u21b5')
        .replace(/\t/g, '\u21e5')
        .replace(/ /g, '\u00A0');
}

function buildDom() {
    let root = document.getElementById(STRIP_ID);
    if (!root) {
        root = document.createElement('div');
        root.id = STRIP_ID;
        root.dataset.visible = 'false';
        document.body.appendChild(root);
    }
    const legacyHeader = root.querySelector('.prompt-token-strip__header');
    if (legacyHeader && legacyHeader.parentElement === root) {
        root.removeChild(legacyHeader);
    }
    let tokensEl = root.querySelector('[data-role="tokens"]');
    if (!tokensEl) {
        tokensEl = document.createElement('div');
        tokensEl.className = 'prompt-token-strip__tokens';
        tokensEl.dataset.role = 'tokens';
        root.appendChild(tokensEl);
    }
    return {
        root,
        tokensEl
    };
}

function setBodyVisibilityFlag(visible) {
    if (typeof document === 'undefined' || !document.body) return;
    if (visible) {
        document.body.dataset.promptTokenStripVisible = 'true';
    } else {
        delete document.body.dataset.promptTokenStripVisible;
    }
}

function setPromptTokenStripHeightVar(heightPx = 0) {
    if (typeof document === 'undefined' || !document.body) return;
    const safeHeight = Number.isFinite(heightPx) ? Math.max(0, Math.ceil(heightPx)) : 0;
    document.body.style.setProperty('--prompt-token-strip-height', `${safeHeight}px`);
}

function isDetailPanelSuppressingStripOnMobile() {
    if (typeof window === 'undefined' || typeof document === 'undefined' || !document.body) return false;
    if (!document.body.classList.contains('detail-mobile-focus')) return false;
    if (typeof window.matchMedia === 'function') {
        return window.matchMedia('(max-aspect-ratio: 1/1), (max-width: 880px)').matches;
    }
    return window.innerWidth <= 880 || window.innerHeight <= window.innerWidth;
}

function rectsOverlapVertically(a, b, epsilon = 0.5) {
    if (!a || !b) return false;
    return a.bottom > (b.top + epsilon) && a.top < (b.bottom - epsilon);
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

export function initPromptTokenStrip({ onTokenClick = null } = {}) {
    if (typeof document === 'undefined') {
        return {
            update: () => {},
            dispose: () => {}
        };
    }

    const dom = buildDom();
    const detailPanel = document.getElementById('detailPanel');
    const resizeHandle = document.getElementById('detailPanelResizeHandle');
    let lastSignature = '';
    let clickableEntries = [];
    let stripEnabled = appState.showPromptTokenStrip !== false;
    let bodyClassObserver = null;
    let stripSizeObserver = null;
    let detailPanelObserver = null;
    let touchClickCleanup = null;
    let hoveredEntry = null;
    let mirroredEntry = null;
    let layoutSyncFrame = null;
    let lastCollisionMaxWidthPx = null;

    const clearCollisionMaxWidth = () => {
        if (lastCollisionMaxWidthPx === null) return;
        dom.root.style.removeProperty(PROMPT_TOKEN_STRIP_COLLISION_MAX_WIDTH_VAR);
        lastCollisionMaxWidthPx = null;
    };

    const setCollisionMaxWidth = (widthPx) => {
        if (!Number.isFinite(widthPx)) {
            clearCollisionMaxWidth();
            return;
        }
        const nextWidthPx = Math.max(0, Math.floor(widthPx));
        if (nextWidthPx === lastCollisionMaxWidthPx) return;
        dom.root.style.setProperty(PROMPT_TOKEN_STRIP_COLLISION_MAX_WIDTH_VAR, `${nextWidthPx}px`);
        lastCollisionMaxWidthPx = nextWidthPx;
    };

    const resolveCollisionMaxWidth = () => {
        if (dom.root.dataset.visible !== 'true') return null;
        const stripRect = resolveVisibleRect(dom.root);
        if (!stripRect) return null;

        const blockerRects = [];
        if (detailPanel?.classList.contains('is-open')) {
            const detailPanelRect = resolveVisibleRect(detailPanel);
            if (detailPanelRect) blockerRects.push(detailPanelRect);
            const resizeHandleRect = resolveVisibleRect(resizeHandle);
            if (resizeHandleRect) blockerRects.push(resizeHandleRect);
        }
        if (!blockerRects.length) return null;

        const hasVerticalOverlap = blockerRects.some((rect) => rectsOverlapVertically(stripRect, rect));
        if (!hasVerticalOverlap) return null;

        const blockerLeft = Math.min(...blockerRects.map((rect) => rect.left));
        return blockerLeft - stripRect.left - PROMPT_TOKEN_STRIP_COLLISION_GAP_PX;
    };

    const syncLayoutMetrics = () => {
        layoutSyncFrame = null;
        if (dom.root.dataset.visible !== 'true') {
            clearCollisionMaxWidth();
            setPromptTokenStripHeightVar(0);
            return;
        }

        setCollisionMaxWidth(resolveCollisionMaxWidth());
        setPromptTokenStripHeightVar(dom.root.getBoundingClientRect().height);
    };

    const scheduleLayoutSync = () => {
        if (layoutSyncFrame !== null) return;
        if (typeof window === 'undefined' || typeof window.requestAnimationFrame !== 'function') {
            syncLayoutMetrics();
            return;
        }
        layoutSyncFrame = window.requestAnimationFrame(syncLayoutMetrics);
    };

    const updateVisibility = () => {
        const shouldShow = stripEnabled
            && clickableEntries.length > 0
            && !isDetailPanelSuppressingStripOnMobile();
        dom.root.dataset.visible = shouldShow ? 'true' : 'false';
        setBodyVisibilityFlag(shouldShow);
        scheduleLayoutSync();
    };

    const setEnabled = (enabled) => {
        stripEnabled = !!enabled;
        appState.showPromptTokenStrip = stripEnabled;
        updateVisibility();
    };

    const entriesEquivalent = (a, b) => {
        if (!a && !b) return true;
        if (!a || !b) return false;
        return tokenChipEntriesMatch(a, b);
    };

    const resolveEntryFromTarget = (target) => {
        const chip = target?.closest?.('[data-token-entry-index]');
        if (!chip || !dom.tokensEl.contains(chip)) return null;
        const rawIndex = Number(chip.dataset.tokenEntryIndex);
        if (!Number.isFinite(rawIndex)) return null;
        return clickableEntries[Math.floor(rawIndex)] || null;
    };

    const applyActiveTokenState = () => {
        const chips = dom.tokensEl.querySelectorAll('[data-token-entry-index]');
        chips.forEach((chip) => {
            const rawIndex = Number(chip.dataset.tokenEntryIndex);
            const entry = Number.isFinite(rawIndex) ? clickableEntries[Math.floor(rawIndex)] : null;
            const isActive = !!entry && (
                tokenChipEntriesMatch(entry, hoveredEntry)
                || tokenChipEntriesMatch(entry, mirroredEntry)
            );
            chip.classList.toggle('is-token-chip-active', isActive);
            chip.dataset.tokenActive = isActive ? 'true' : 'false';
        });
    };

    const setHoveredEntry = (entry, { emit = true } = {}) => {
        const normalizedEntry = normalizeTokenChipEntry(entry);
        if (entriesEquivalent(hoveredEntry, normalizedEntry)) {
            applyActiveTokenState();
            return;
        }
        hoveredEntry = normalizedEntry;
        applyActiveTokenState();
        if (emit) {
            dispatchTokenChipHoverSync(normalizedEntry, {
                active: !!normalizedEntry,
                source: PROMPT_TOKEN_STRIP_HOVER_SOURCE
            });
        }
    };

    const handleTokenClick = (event) => {
        const entry = resolveEntryFromTarget(event?.target);
        if (!entry) return;
        if (typeof onTokenClick === 'function') {
            onTokenClick({ ...entry });
        }
    };

    const handleTokenPointerOver = (event) => {
        setHoveredEntry(resolveEntryFromTarget(event?.target), { emit: true });
    };

    const handleTokenPointerOut = (event) => {
        const fromEntry = resolveEntryFromTarget(event?.target);
        if (!fromEntry) return;
        const toEntry = resolveEntryFromTarget(event?.relatedTarget);
        setHoveredEntry(toEntry, { emit: true });
    };

    const handleTokenFocusIn = (event) => {
        setHoveredEntry(resolveEntryFromTarget(event?.target), { emit: true });
    };

    const handleTokenFocusOut = (event) => {
        const fromEntry = resolveEntryFromTarget(event?.target);
        if (!fromEntry) return;
        const toEntry = resolveEntryFromTarget(event?.relatedTarget);
        setHoveredEntry(toEntry, { emit: true });
    };

    const handleVisibilityChanged = (event) => {
        const detail = event?.detail || null;
        const enabled = (detail && typeof detail.enabled === 'boolean')
            ? detail.enabled
            : (appState.showPromptTokenStrip !== false);
        setEnabled(enabled);
    };

    const handleTokenChipHoverSync = (event) => {
        const detail = event?.detail || null;
        if (!detail || detail.source === PROMPT_TOKEN_STRIP_HOVER_SOURCE) return;
        mirroredEntry = detail.active ? normalizeTokenChipEntry(detail) : null;
        applyActiveTokenState();
    };

    dom.tokensEl.addEventListener('click', handleTokenClick);
    dom.tokensEl.addEventListener('pointerover', handleTokenPointerOver);
    dom.tokensEl.addEventListener('pointerout', handleTokenPointerOut);
    dom.tokensEl.addEventListener('focusin', handleTokenFocusIn);
    dom.tokensEl.addEventListener('focusout', handleTokenFocusOut);
    touchClickCleanup = initTouchClickFallback(dom.tokensEl, {
        selector: '.prompt-token-strip__token'
    });
    if (typeof window !== 'undefined' && typeof window.addEventListener === 'function') {
        window.addEventListener('promptTokenStripVisibilityChanged', handleVisibilityChanged);
        window.addEventListener(TOKEN_CHIP_HOVER_SYNC_EVENT, handleTokenChipHoverSync);
    }
    if (typeof MutationObserver !== 'undefined' && document.body) {
        bodyClassObserver = new MutationObserver((mutationList) => {
            const classChanged = mutationList.some((mutation) => (
                mutation?.type === 'attributes' && mutation.attributeName === 'class'
            ));
            if (classChanged) {
                updateVisibility();
            }
        });
        bodyClassObserver.observe(document.body, {
            attributes: true,
            attributeFilter: ['class']
        });
    }
    if (typeof ResizeObserver !== 'undefined') {
        stripSizeObserver = new ResizeObserver(() => {
            scheduleLayoutSync();
        });
        stripSizeObserver.observe(dom.root);
        if (detailPanel) {
            detailPanelObserver = new ResizeObserver(() => {
                scheduleLayoutSync();
            });
            detailPanelObserver.observe(detailPanel);
        }
    }
    if (typeof window !== 'undefined' && typeof window.addEventListener === 'function') {
        window.addEventListener('resize', scheduleLayoutSync);
        window.visualViewport?.addEventListener?.('resize', scheduleLayoutSync);
    }

    const render = ({
        tokenLabels = [],
        tokenIndices = null,
        tokenIds = null,
        generatedToken = null
    } = {}) => {
        const entries = buildPromptTokenChipEntries({
            tokenLabels,
            tokenIndices,
            tokenIds,
            generatedToken
        });
        const colorState = setActivePromptTokenChipEntries(entries);

        if (!entries.length) {
            dom.tokensEl.innerHTML = '';
            clickableEntries = [];
            lastSignature = '';
            if (hoveredEntry) {
                setHoveredEntry(null, { emit: true });
            } else {
                applyActiveTokenState();
            }
            updateVisibility();
            return;
        }

        const signature = entries
            .map((entry) => [
                entry.entryType || '',
                entry.laneIndex,
                entry.tokenIndex,
                entry.tokenId,
                entry.tokenLabel,
                entry.selectionLabel || '',
                entry.seed
            ].join('|'))
            .join('\u241f');
        clickableEntries = entries;
        if (signature === lastSignature) {
            applyActiveTokenState();
            updateVisibility();
            return;
        }
        lastSignature = signature;

        const fragment = document.createDocumentFragment();
        entries.forEach((entry, index) => {
            const tokenEl = document.createElement('button');
            tokenEl.type = 'button';
            tokenEl.className = 'prompt-token-strip__token';
            if (entry.entryType === 'generated') {
                tokenEl.classList.add('prompt-token-strip__token--generated');
            }
            applyTokenChipColors(tokenEl, entry, index, { lookup: colorState.lookup });
            tokenEl.textContent = normalizeTokenText(entry.tokenLabel);
            tokenEl.dataset.tokenEntryIndex = String(index);
            fragment.appendChild(tokenEl);
        });

        dom.tokensEl.replaceChildren(fragment);
        applyActiveTokenState();
        updateVisibility();
        scheduleLayoutSync();
    };

    return {
        update: ({
            tokenLabels = [],
            tokenIndices = null,
            tokenIds = null,
            generatedToken = null
        } = {}) => {
            render({ tokenLabels, tokenIndices, tokenIds, generatedToken });
        },
        setEnabled,
        getRootElement: () => dom.root,
        getTokenElement: (index) => {
            const safeIndex = Number(index);
            if (!Number.isFinite(safeIndex)) return null;
            return dom.tokensEl.querySelector(`[data-token-entry-index="${Math.floor(safeIndex)}"]`);
        },
        dispose: () => {
            setBodyVisibilityFlag(false);
            setPromptTokenStripHeightVar(0);
            clickableEntries = [];
            setHoveredEntry(null, { emit: true });
            mirroredEntry = null;
            clearCollisionMaxWidth();
            setActivePromptTokenChipEntries([]);
            dom.tokensEl.removeEventListener('click', handleTokenClick);
            dom.tokensEl.removeEventListener('pointerover', handleTokenPointerOver);
            dom.tokensEl.removeEventListener('pointerout', handleTokenPointerOut);
            dom.tokensEl.removeEventListener('focusin', handleTokenFocusIn);
            dom.tokensEl.removeEventListener('focusout', handleTokenFocusOut);
            touchClickCleanup?.();
            touchClickCleanup = null;
            if (typeof window !== 'undefined' && typeof window.removeEventListener === 'function') {
                window.removeEventListener('promptTokenStripVisibilityChanged', handleVisibilityChanged);
                window.removeEventListener(TOKEN_CHIP_HOVER_SYNC_EVENT, handleTokenChipHoverSync);
                window.removeEventListener('resize', scheduleLayoutSync);
                window.visualViewport?.removeEventListener?.('resize', scheduleLayoutSync);
            }
            if (bodyClassObserver) {
                bodyClassObserver.disconnect();
                bodyClassObserver = null;
            }
            if (stripSizeObserver) {
                stripSizeObserver.disconnect();
                stripSizeObserver = null;
            }
            if (detailPanelObserver) {
                detailPanelObserver.disconnect();
                detailPanelObserver = null;
            }
            if (layoutSyncFrame !== null && typeof window !== 'undefined' && typeof window.cancelAnimationFrame === 'function') {
                window.cancelAnimationFrame(layoutSyncFrame);
                layoutSyncFrame = null;
            }
            if (dom.root && dom.root.parentElement) {
                dom.root.parentElement.removeChild(dom.root);
            }
        }
    };
}

export default initPromptTokenStrip;
