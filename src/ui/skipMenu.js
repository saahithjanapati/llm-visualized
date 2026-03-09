import { appState } from '../state/appState.js';
import { initTouchClickFallback } from './touchClickFallback.js';

const PAUSE_REASON = 'skip-options-modal';
const SKIP_OPTION_ORDER = Object.freeze([
    'skipConveyorBtn',
    'skipLayerBtn',
    'skipToEndBtn',
    'skipNextPassBtn',
    'skipLastPassBtn'
]);

const SKIP_OPTION_LABELS = Object.freeze({
    skipLayerBtn: 'Skip layer',
    skipConveyorBtn: 'Skip attention',
    skipToEndBtn: 'Skip to end',
    skipNextPassBtn: 'Next pass',
    skipLastPassBtn: 'Last pass'
});

const SKIP_OPTION_DESCRIPTIONS = Object.freeze({
    skipLayerBtn: 'Completes only the current transformer layer, then resumes normal playback.',
    skipConveyorBtn: 'Skips the active attention conveyor and jumps to the concat/output stage.',
    skipToEndBtn: 'Fast-forwards through all remaining layers in the current forward pass.',
    skipNextPassBtn: 'Jumps to the next forward pass for the next token in generation.',
    skipLastPassBtn: 'Jumps directly to the final available forward pass.'
});

function getOptionButtons() {
    return SKIP_OPTION_ORDER
        .map((id) => document.getElementById(id))
        .filter(Boolean);
}

function isVisible(el) {
    return !!(el && el.dataset.visible === 'true' && el.style.display !== 'none');
}

export function initSkipMenu(pipeline) {
    const menu = document.getElementById('skipMenu');
    const toggle = document.getElementById('skipMenuToggle');
    const items = document.getElementById('skipMenuItems');
    const overlay = document.getElementById('skipOptionsOverlay');
    const modal = document.getElementById('skipOptionsModal');
    const closeBtn = document.getElementById('skipOptionsClose');
    const optionList = document.getElementById('skipOptionsList');
    const emptyMessage = document.getElementById('skipOptionsEmpty');

    if (!menu || !toggle || !items) return () => {};
    const modalTouchCleanup = initTouchClickFallback(modal, {
        selector: '.skip-option, #skipOptionsClose',
        tapSlopPx: 20,
        activateOnPointerDownSelector: '#skipOptionsClose'
    });

    let rafId = null;
    let modalOpen = false;
    let menuPausedAnimation = false;
    let previousModalPausedState = false;
    let previousBodyOverflow = '';
    let previousCanvasPointerEvents = null;
    let previousControlsState = null;
    let restoreFocusEl = null;

    const setOpen = (open) => {
        const next = open ? 'true' : 'false';
        if (menu.dataset.open !== next) {
            menu.dataset.open = next;
        }
        toggle.setAttribute('aria-expanded', next);
    };

    const getAvailableOptions = () => getOptionButtons().filter((btn) => isVisible(btn) && !btn.disabled);

    const setMenuVisible = (visible) => {
        menu.dataset.visible = visible ? 'true' : 'false';
        menu.style.display = visible ? '' : 'none';
    };

    const lockBackgroundInteraction = () => {
        const engine = pipeline?.engine;
        if (!engine) return;

        engine.resetInteractionState?.();

        const controls = engine.controls;
        if (controls && !previousControlsState) {
            previousControlsState = {
                enabled: controls.enabled,
                enableRotate: controls.enableRotate,
                enablePan: controls.enablePan,
                enableZoom: controls.enableZoom
            };
        }
        if (controls) {
            controls.enabled = false;
            controls.enableRotate = false;
            controls.enablePan = false;
            controls.enableZoom = false;
        }

        const canvas = engine?.renderer?.domElement;
        if (canvas && previousCanvasPointerEvents === null) {
            previousCanvasPointerEvents = canvas.style.pointerEvents || '';
            canvas.style.pointerEvents = 'none';
        }
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

        const canvas = engine?.renderer?.domElement;
        if (canvas && previousCanvasPointerEvents !== null) {
            canvas.style.pointerEvents = previousCanvasPointerEvents;
            previousCanvasPointerEvents = null;
        }

        engine.resetInteractionState?.();
    };

    const pauseForModal = () => {
        if (menuPausedAnimation) return;
        previousModalPausedState = !!appState.modalPaused;
        previousBodyOverflow = document.body.style.overflow || '';
        lockBackgroundInteraction();
        pipeline?.engine?.pause?.(PAUSE_REASON);
        appState.modalPaused = true;
        document.body.style.overflow = 'hidden';
        menuPausedAnimation = true;
    };

    const resumeFromModal = () => {
        if (!menuPausedAnimation) return;
        pipeline?.engine?.resume?.(PAUSE_REASON);
        unlockBackgroundInteraction();
        appState.modalPaused = previousModalPausedState;
        document.body.style.overflow = previousBodyOverflow;
        menuPausedAnimation = false;
    };

    const renderOptions = () => {
        if (!optionList || !emptyMessage) return 0;
        optionList.replaceChildren();

        const options = getAvailableOptions();
        options.forEach((sourceButton) => {
            const id = sourceButton.id;
            const optionButton = document.createElement('button');
            optionButton.type = 'button';
            optionButton.className = 'skip-option';
            optionButton.dataset.targetId = id;
            optionButton.setAttribute('role', 'listitem');

            const title = document.createElement('span');
            title.className = 'skip-option-title';
            title.textContent = SKIP_OPTION_LABELS[id] || sourceButton.textContent || 'Skip action';

            const description = document.createElement('span');
            description.className = 'skip-option-description';
            description.textContent = SKIP_OPTION_DESCRIPTIONS[id] || 'Fast-forward the current animation state.';

            optionButton.appendChild(title);
            optionButton.appendChild(description);
            optionList.appendChild(optionButton);
        });

        const hasOptions = options.length > 0;
        emptyMessage.hidden = hasOptions;
        return options.length;
    };

    const closeModal = ({ restoreFocus = true } = {}) => {
        if (!modalOpen) return;
        modalOpen = false;
        if (overlay) {
            overlay.style.display = 'none';
            overlay.setAttribute('aria-hidden', 'true');
        }
        setOpen(false);
        resumeFromModal();
        if (restoreFocus && restoreFocusEl && typeof restoreFocusEl.focus === 'function') {
            restoreFocusEl.focus();
        }
        restoreFocusEl = null;
    };

    const openModal = () => {
        if (!overlay || !optionList) return;
        if (menu.dataset.visible !== 'true') return;
        const optionCount = renderOptions();
        if (optionCount <= 0) return;

        restoreFocusEl = (document.activeElement && typeof document.activeElement.focus === 'function')
            ? document.activeElement
            : null;
        pauseForModal();
        modalOpen = true;
        overlay.style.display = 'flex';
        overlay.setAttribute('aria-hidden', 'false');
        setOpen(true);

        const firstAction = optionList.querySelector('.skip-option');
        const focusTarget = firstAction || closeBtn || modal;
        if (focusTarget && typeof focusTarget.focus === 'function') {
            window.requestAnimationFrame(() => focusTarget.focus());
        }
    };

    const onToggleClick = (event) => {
        event.preventDefault();
        event.stopPropagation();
        if (toggle.disabled) return;
        if (modalOpen) {
            closeModal();
            return;
        }
        openModal();
    };

    const onOverlayClick = (event) => {
        if (!overlay) return;
        if (event.target === overlay) closeModal();
    };

    const onCloseClick = (event) => {
        event.preventDefault();
        closeModal();
    };

    const onOptionClick = (event) => {
        const actionButton = event.target instanceof Element
            ? event.target.closest('.skip-option')
            : null;
        if (!actionButton) return;
        event.preventDefault();

        const targetId = actionButton.dataset.targetId;
        if (!targetId) return;
        const sourceButton = document.getElementById(targetId);
        if (!sourceButton || sourceButton.disabled) {
            closeModal();
            return;
        }

        closeModal({ restoreFocus: false });
        window.requestAnimationFrame(() => {
            sourceButton.click();
        });
    };

    const onWindowKeyDown = (event) => {
        if (event.key !== 'Escape' || !modalOpen) return;
        event.preventDefault();
        closeModal();
    };

    const blockBackgroundZoomGesture = (event) => {
        if (!modalOpen) return;
        event.preventDefault();
        event.stopPropagation();
    };

    const updateVisibility = () => {
        const anyVisible = getOptionButtons().some((btn) => isVisible(btn));
        setMenuVisible(anyVisible);
        if (!anyVisible && modalOpen) {
            closeModal({ restoreFocus: false });
        }
    };

    const updateSkipState = () => {
        const skippingLayer = typeof pipeline?.isSkipLayerActive === 'function' && pipeline.isSkipLayerActive();
        const skippingAll = typeof pipeline?.isSkipToEndActive === 'function' && pipeline.isSkipToEndActive();
        const isSkipping = skippingLayer || skippingAll;
        const menuVisible = menu.dataset.visible === 'true';

        toggle.disabled = isSkipping || !menuVisible;
        toggle.dataset.state = isSkipping ? 'skipping' : 'ready';
        toggle.setAttribute('aria-busy', isSkipping ? 'true' : 'false');

        if (isSkipping && modalOpen) {
            closeModal({ restoreFocus: false });
        }
    };

    const scheduleFrame = (typeof window !== 'undefined' && typeof window.requestAnimationFrame === 'function')
        ? window.requestAnimationFrame.bind(window)
        : (cb) => setTimeout(cb, 120);
    const cancelFrame = (typeof window !== 'undefined' && typeof window.cancelAnimationFrame === 'function')
        ? window.cancelAnimationFrame.bind(window)
        : (id) => clearTimeout(id);

    const tick = () => {
        updateVisibility();
        updateSkipState();
        rafId = scheduleFrame(tick);
    };

    toggle.addEventListener('click', onToggleClick);
    overlay?.addEventListener('click', onOverlayClick);
    overlay?.addEventListener('wheel', blockBackgroundZoomGesture, { capture: true, passive: false });
    overlay?.addEventListener('gesturestart', blockBackgroundZoomGesture, { capture: true, passive: false });
    overlay?.addEventListener('gesturechange', blockBackgroundZoomGesture, { capture: true, passive: false });
    overlay?.addEventListener('gestureend', blockBackgroundZoomGesture, { capture: true, passive: false });
    closeBtn?.addEventListener('click', onCloseClick);
    optionList?.addEventListener('click', onOptionClick);
    window.addEventListener('keydown', onWindowKeyDown);

    tick();

    return () => {
        toggle.removeEventListener('click', onToggleClick);
        overlay?.removeEventListener('click', onOverlayClick);
        overlay?.removeEventListener('wheel', blockBackgroundZoomGesture, true);
        overlay?.removeEventListener('gesturestart', blockBackgroundZoomGesture, true);
        overlay?.removeEventListener('gesturechange', blockBackgroundZoomGesture, true);
        overlay?.removeEventListener('gestureend', blockBackgroundZoomGesture, true);
        closeBtn?.removeEventListener('click', onCloseClick);
        optionList?.removeEventListener('click', onOptionClick);
        window.removeEventListener('keydown', onWindowKeyDown);
        if (rafId !== null) cancelFrame(rafId);
        modalTouchCleanup?.();
        closeModal({ restoreFocus: false });
        unlockBackgroundInteraction();
    };
}

export default initSkipMenu;
