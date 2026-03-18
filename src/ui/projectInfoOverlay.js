import infoMarkdown from './infoModalContent.md?raw';
import { appState } from '../state/appState.js';
import { buildProjectInfoPageUrl } from './projectInfoNavigation.js';
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
    const standaloneLink = document.getElementById('projectInfoStandaloneLink');
    const contentEl = document.getElementById('projectInfoOverlayContent');

    if (!overlay || !modal || !closeBtn || !standaloneLink || !contentEl) {
        return {
            open: () => false,
            close: () => false,
            isOpen: () => false
        };
    }

    if (contentEl.childElementCount === 0) {
        contentEl.innerHTML = renderSimpleMarkdown(infoMarkdown);
    }

    let isOverlayOpen = false;
    let restoreFocusEl = null;
    let previousModalPausedState = false;
    let previousBodyOverflow = '';
    let previousCanvasPointerEvents = null;
    let previousControlsState = null;

    const syncStandaloneLinkHref = () => {
        standaloneLink.setAttribute('href', buildProjectInfoPageUrl(window.location));
    };

    const lockBackgroundInteraction = () => {
        const engine = pipeline?.engine;
        if (!engine) return;

        const controls = engine.controls;
        if (controls && !previousControlsState) {
            previousControlsState = {
                enabled: controls.enabled,
                enableRotate: controls.enableRotate,
                enablePan: controls.enablePan,
                enableZoom: controls.enableZoom
            };
            controls.enabled = false;
            controls.enableRotate = false;
            controls.enablePan = false;
            controls.enableZoom = false;
        }

        const canvas = engine.renderer?.domElement || null;
        if (canvas && previousCanvasPointerEvents === null) {
            previousCanvasPointerEvents = canvas.style.pointerEvents || '';
            canvas.style.pointerEvents = 'none';
        }

        engine.resetInteractionState?.();
    };

    const unlockBackgroundInteraction = () => {
        const engine = pipeline?.engine;
        if (!engine) return;

        const controls = engine.controls;
        if (controls && previousControlsState) {
            controls.enabled = previousControlsState.enabled;
            controls.enableRotate = previousControlsState.enableRotate;
            controls.enablePan = previousControlsState.enablePan;
            controls.enableZoom = previousControlsState.enableZoom;
            previousControlsState = null;
        }

        const canvas = engine.renderer?.domElement || null;
        if (canvas && previousCanvasPointerEvents !== null) {
            canvas.style.pointerEvents = previousCanvasPointerEvents;
            previousCanvasPointerEvents = null;
        }

        engine.resetInteractionState?.();
    };

    const close = ({
        restoreFocus = true
    } = {}) => {
        if (!isOverlayOpen) return false;

        isOverlayOpen = false;
        overlay.style.display = 'none';
        overlay.setAttribute('aria-hidden', 'true');
        pipeline?.engine?.resume?.(PROJECT_INFO_OVERLAY_PAUSE_REASON);
        unlockBackgroundInteraction();
        appState.modalPaused = previousModalPausedState;
        document.body.style.overflow = previousBodyOverflow;

        const nextRestoreFocusEl = restoreFocusEl;
        restoreFocusEl = null;
        if (restoreFocus && nextRestoreFocusEl) {
            scheduleFocus(nextRestoreFocusEl);
        }
        return true;
    };

    const open = ({
        restoreFocusTarget = null
    } = {}) => {
        syncStandaloneLinkHref();
        if (isOverlayOpen) {
            scheduleFocus(closeBtn);
            return true;
        }

        restoreFocusEl = restoreFocusTarget
            || (document.activeElement && typeof document.activeElement.focus === 'function'
                ? document.activeElement
                : null);
        previousModalPausedState = !!appState.modalPaused;
        previousBodyOverflow = document.body.style.overflow || '';
        lockBackgroundInteraction();
        pipeline?.engine?.pause?.(PROJECT_INFO_OVERLAY_PAUSE_REASON);
        appState.modalPaused = true;
        document.body.style.overflow = 'hidden';
        overlay.style.display = 'flex';
        overlay.setAttribute('aria-hidden', 'false');
        isOverlayOpen = true;
        scheduleFocus(closeBtn);
        return true;
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
