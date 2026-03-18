import infoMarkdown from './infoModalContent.md?raw';
import {
    PROJECT_INFO_PAGE_PATH,
    buildProjectInfoPageUrl,
    resolveProjectInfoNavigationReturnHref,
    resolveProjectInfoBackHref
} from './projectInfoNavigation.js';
import {
    acquireModalUiLock,
    acquireSceneBackgroundInteractionLock
} from './overlayLockManager.js';
import { enhanceProjectInfoContent } from './projectInfoContent.js';
import { renderSimpleMarkdown } from './simpleMarkdown.js';

export const PROJECT_INFO_OVERLAY_PAUSE_REASON = 'project-info-overlay';

function scheduleFocus(target) {
    if (!target || typeof target.focus !== 'function') return;
    if (typeof window !== 'undefined' && typeof window.requestAnimationFrame === 'function') {
        window.requestAnimationFrame(() => target.focus());
        return;
    }
    setTimeout(() => target.focus(), 0);
}

function normalizePathname(pathname = '') {
    const rawPathname = String(pathname || '').trim();
    if (!rawPathname.length || rawPathname === '/') return '/';
    return rawPathname.replace(/\/+$/, '') || '/';
}

function isProjectInfoRoute(locationRef = null) {
    const candidate = locationRef || (typeof window !== 'undefined' ? window.location : null);
    return normalizePathname(candidate?.pathname || '/') === normalizePathname(PROJECT_INFO_PAGE_PATH);
}

export function initProjectInfoOverlay({
    pipeline = null
} = {}) {
    if (typeof document === 'undefined') {
        return {
            open: () => false,
            close: () => false,
            isOpen: () => false
        };
    }

    const overlay = document.getElementById('projectInfoOverlay');
    const modal = overlay?.querySelector('.project-info-modal') || null;
    const closeBtn = document.getElementById('projectInfoClose');
    const contentEl = document.getElementById('projectInfoOverlayContent');

    if (!overlay || !modal || !closeBtn || !contentEl) {
        return {
            open: () => false,
            close: () => false,
            isOpen: () => false
        };
    }

    if (contentEl.childElementCount === 0) {
        contentEl.innerHTML = renderSimpleMarkdown(infoMarkdown);
    }
    enhanceProjectInfoContent(contentEl);

    let isOverlayOpen = false;
    let restoreFocusEl = null;
    let releaseModalUiLock = null;
    let releaseSceneInteractionLock = null;
    let historyCloseRestoreFocus = true;
    let historyTraversalPending = false;

    const lockBackgroundInteraction = () => {
        if (releaseSceneInteractionLock) return;
        releaseSceneInteractionLock = acquireSceneBackgroundInteractionLock(pipeline?.engine || null);
    };

    const unlockBackgroundInteraction = () => {
        if (!releaseSceneInteractionLock) return;
        releaseSceneInteractionLock();
        releaseSceneInteractionLock = null;
    };

    const closeInternal = ({
        restoreFocus = true
    } = {}) => {
        if (!isOverlayOpen) return false;

        isOverlayOpen = false;
        overlay.style.display = 'none';
        overlay.setAttribute('aria-hidden', 'true');
        pipeline?.engine?.resume?.(PROJECT_INFO_OVERLAY_PAUSE_REASON);
        unlockBackgroundInteraction();
        if (releaseModalUiLock) {
            releaseModalUiLock();
            releaseModalUiLock = null;
        }

        const nextRestoreFocusEl = restoreFocusEl;
        restoreFocusEl = null;
        if (restoreFocus && nextRestoreFocusEl) {
            scheduleFocus(nextRestoreFocusEl);
        }
        return true;
    };

    const syncInfoRouteIntoHistory = () => {
        if (typeof window === 'undefined') return false;
        if (isProjectInfoRoute(window.location)) return false;
        if (typeof window.history?.pushState !== 'function') return false;
        const returnHref = resolveProjectInfoNavigationReturnHref(window.location);
        const currentHref = `${window.location.pathname}${window.location.search}${window.location.hash}`;
        if (
            returnHref !== currentHref
            && typeof window.history?.replaceState === 'function'
        ) {
            window.history.replaceState(window.history.state, '', returnHref);
        }
        window.history.pushState(window.history.state, '', buildProjectInfoPageUrl(window.location, { returnHref }));
        return true;
    };

    const restoreVisualizationRouteInPlace = () => {
        if (typeof window === 'undefined') return false;
        if (typeof window.history?.replaceState !== 'function') return false;
        window.history.replaceState(window.history.state, '', resolveProjectInfoBackHref(window.location));
        return true;
    };

    const close = ({
        restoreFocus = true,
        syncHistory = true
    } = {}) => {
        if (!isOverlayOpen) return false;
        if (syncHistory && isProjectInfoRoute(window.location)) {
            if (historyTraversalPending) return true;
            historyCloseRestoreFocus = restoreFocus;
            if (Number.isFinite(window.history.length) && window.history.length > 1 && typeof window.history.back === 'function') {
                historyTraversalPending = true;
                window.history.back();
                return true;
            }
            restoreVisualizationRouteInPlace();
        }
        historyTraversalPending = false;
        historyCloseRestoreFocus = true;
        return closeInternal({ restoreFocus });
    };

    const open = ({
        syncHistory = true,
        restoreFocusTarget = null
    } = {}) => {
        if (syncHistory) {
            syncInfoRouteIntoHistory();
        }
        if (isOverlayOpen) {
            scheduleFocus(closeBtn);
            return true;
        }

        restoreFocusEl = restoreFocusTarget
            || (document.activeElement && typeof document.activeElement.focus === 'function'
                ? document.activeElement
                : null);
        lockBackgroundInteraction();
        releaseModalUiLock = acquireModalUiLock();
        pipeline?.engine?.pause?.(PROJECT_INFO_OVERLAY_PAUSE_REASON);
        overlay.style.display = 'flex';
        overlay.setAttribute('aria-hidden', 'false');
        isOverlayOpen = true;
        scheduleFocus(closeBtn);
        return true;
    };

    const handlePopState = () => {
        if (isProjectInfoRoute(window.location)) {
            if (!isOverlayOpen) {
                open({ restoreFocusTarget: null, syncHistory: false });
            }
            return;
        }
        if (isOverlayOpen) {
            historyTraversalPending = false;
            const shouldRestoreFocus = historyCloseRestoreFocus;
            historyCloseRestoreFocus = true;
            closeInternal({ restoreFocus: shouldRestoreFocus });
        }
    };

    overlay.addEventListener('click', (event) => {
        if (event.target === overlay) {
            close();
        }
    });

    closeBtn.addEventListener('click', (event) => {
        event.preventDefault();
        close();
    });

    if (typeof window !== 'undefined' && typeof window.addEventListener === 'function') {
        const previousPopstateListener = window.__llmVisualizedProjectInfoOverlayPopstateListener;
        if (typeof previousPopstateListener === 'function') {
            window.removeEventListener('popstate', previousPopstateListener);
        }
        window.__llmVisualizedProjectInfoOverlayPopstateListener = handlePopState;
        window.addEventListener('popstate', handlePopState);
    }

    document.addEventListener('keydown', (event) => {
        if (!isOverlayOpen) return;
        if (event.key !== 'Escape') return;
        event.preventDefault();
        close();
    });

    return {
        open,
        close,
        isOpen: () => isOverlayOpen
    };
}
