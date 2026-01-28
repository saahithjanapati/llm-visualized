export function initSkipLayerButton(pipeline) {
    if (!pipeline) return () => {};

    const existingButton = document.getElementById('skipLayerBtn');
    const menuItems = document.getElementById('skipMenuItems');
    const topControls = document.getElementById('topControls');
    const button = existingButton || (() => {
        const btn = document.createElement('button');
        btn.id = 'skipLayerBtn';
        btn.type = 'button';
        btn.textContent = 'Skip layer';
        btn.title = 'Skip current layer';
        btn.setAttribute('aria-label', 'Skip current layer');
        btn.dataset.visible = 'true';
        if (menuItems) {
            const insertBefore = menuItems.querySelector('#skipConveyorBtn')
                || menuItems.querySelector('#skipToEndBtn')
                || null;
            if (insertBefore) {
                menuItems.insertBefore(btn, insertBefore);
            } else {
                menuItems.appendChild(btn);
            }
        } else if (topControls) {
            topControls.insertBefore(btn, document.getElementById('settingsBtn'));
        } else {
            document.body.appendChild(btn);
        }
        return btn;
    })();

    const setVisible = (show) => {
        const next = show ? 'true' : 'false';
        if (button.dataset.visible === next) return;
        button.dataset.visible = next;
        button.style.display = show ? '' : 'none';
    };

    const onClick = (event) => {
        event.preventDefault();
        if (pipeline && typeof pipeline.skipCurrentLayer === 'function') {
            pipeline.skipCurrentLayer();
        }
    };

    button.addEventListener('click', onClick);

    const scheduleFrame = (typeof window !== 'undefined' && typeof window.requestAnimationFrame === 'function')
        ? window.requestAnimationFrame.bind(window)
        : (cb) => setTimeout(cb, 120);
    const cancelFrame = (typeof window !== 'undefined' && typeof window.cancelAnimationFrame === 'function')
        ? window.cancelAnimationFrame.bind(window)
        : (id) => clearTimeout(id);

    let rafId = null;
    const update = () => {
        const complete = typeof pipeline.isForwardPassComplete === 'function' && pipeline.isForwardPassComplete();
        if (complete) {
            setVisible(false);
            return;
        }
        const skippingLayer = typeof pipeline.isSkipLayerActive === 'function' && pipeline.isSkipLayerActive();
        const skippingAll = typeof pipeline.isSkipToEndActive === 'function' && pipeline.isSkipToEndActive();
        setVisible(true);
        button.disabled = skippingLayer || skippingAll;
        button.dataset.state = skippingLayer ? 'skipping' : 'ready';
        button.textContent = skippingLayer ? 'Skipping layer...' : 'Skip layer';
        button.setAttribute('aria-busy', skippingLayer ? 'true' : 'false');
        rafId = scheduleFrame(update);
    };
    update();

    return () => {
        button.removeEventListener('click', onClick);
        if (rafId !== null) cancelFrame(rafId);
    };
}

export default initSkipLayerButton;
