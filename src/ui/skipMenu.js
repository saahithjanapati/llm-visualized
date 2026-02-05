export function initSkipMenu(pipeline) {
    const menu = document.getElementById('skipMenu');
    const toggle = document.getElementById('skipMenuToggle');
    const items = document.getElementById('skipMenuItems');
    const skipLayerBtn = document.getElementById('skipLayerBtn');
    const skipConveyorBtn = document.getElementById('skipConveyorBtn');
    const skipToEndBtn = document.getElementById('skipToEndBtn');

    if (!menu || !toggle || !items) return () => {};

    const isVisible = (el) => !!(el && el.dataset.visible === 'true' && el.style.display !== 'none');

    const setOpen = (open) => {
        const next = open ? 'true' : 'false';
        if (menu.dataset.open === next) return;
        menu.dataset.open = next;
        toggle.setAttribute('aria-expanded', next);
    };

    const defaultToggleLabel = toggle?.textContent || 'Skip';

    const updateVisibility = () => {
        const anyVisible = isVisible(skipLayerBtn) || isVisible(skipConveyorBtn) || isVisible(skipToEndBtn);
        menu.dataset.visible = anyVisible ? 'true' : 'false';
        menu.style.display = anyVisible ? '' : 'none';
        if (!anyVisible) setOpen(false);
    };

    const updateSkipState = () => {
        if (!toggle) return;
        const skippingLayer = typeof pipeline?.isSkipLayerActive === 'function' && pipeline.isSkipLayerActive();
        const skippingAll = typeof pipeline?.isSkipToEndActive === 'function' && pipeline.isSkipToEndActive();
        const isSkipping = skippingLayer || skippingAll;
        toggle.disabled = !!isSkipping;
        toggle.dataset.state = isSkipping ? 'skipping' : 'ready';
        toggle.textContent = isSkipping ? 'Skipping' : defaultToggleLabel;
        toggle.setAttribute('aria-busy', isSkipping ? 'true' : 'false');
        if (isSkipping) setOpen(false);
    };

    const onToggleClick = (event) => {
        event.preventDefault();
        event.stopPropagation();
        if (menu.dataset.visible !== 'true') return;
        setOpen(menu.dataset.open !== 'true');
    };

    const onDocumentClick = (event) => {
        if (!menu.contains(event.target)) {
            setOpen(false);
        }
    };

    const onItemClick = () => {
        setOpen(false);
    };

    toggle.addEventListener('click', onToggleClick);
    document.addEventListener('click', onDocumentClick);
    skipLayerBtn?.addEventListener('click', onItemClick);
    skipConveyorBtn?.addEventListener('click', onItemClick);
    skipToEndBtn?.addEventListener('click', onItemClick);

    const scheduleFrame = (typeof window !== 'undefined' && typeof window.requestAnimationFrame === 'function')
        ? window.requestAnimationFrame.bind(window)
        : (cb) => setTimeout(cb, 120);
    const cancelFrame = (typeof window !== 'undefined' && typeof window.cancelAnimationFrame === 'function')
        ? window.cancelAnimationFrame.bind(window)
        : (id) => clearTimeout(id);

    let rafId = null;
    const tick = () => {
        updateVisibility();
        updateSkipState();
        rafId = scheduleFrame(tick);
    };
    tick();

    return () => {
        toggle.removeEventListener('click', onToggleClick);
        document.removeEventListener('click', onDocumentClick);
        skipLayerBtn?.removeEventListener('click', onItemClick);
        skipConveyorBtn?.removeEventListener('click', onItemClick);
        skipToEndBtn?.removeEventListener('click', onItemClick);
        if (rafId !== null) cancelFrame(rafId);
    };
}

export default initSkipMenu;
