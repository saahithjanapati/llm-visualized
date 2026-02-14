export function initSkipNextPassButton({ pipeline, generationController } = {}) {
    const existingButton = document.getElementById('skipNextPassBtn');
    const menuItems = document.getElementById('skipMenuItems');
    const topControls = document.getElementById('topControls');
    const button = existingButton || (() => {
        const btn = document.createElement('button');
        btn.id = 'skipNextPassBtn';
        btn.type = 'button';
        btn.textContent = 'Next pass';
        btn.title = 'Skip to next forward pass';
        btn.setAttribute('aria-label', 'Skip to next forward pass');
        btn.dataset.visible = 'false';
        btn.style.display = 'none';
        if (menuItems) {
            menuItems.appendChild(btn);
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

    const hasNextPass = () => (
        typeof generationController?.hasNextForwardPass === 'function'
            ? generationController.hasNextForwardPass()
            : false
    );

    const onClick = (event) => {
        event.preventDefault();
        if (!hasNextPass()) return;
        if (typeof generationController?.requestNextForwardPass === 'function') {
            generationController.requestNextForwardPass();
        } else if (typeof generationController?.advance === 'function') {
            generationController.advance();
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
        const hasNext = hasNextPass();
        const skippingLayer = typeof pipeline?.isSkipLayerActive === 'function' && pipeline.isSkipLayerActive();
        const skippingAll = typeof pipeline?.isSkipToEndActive === 'function' && pipeline.isSkipToEndActive();
        const pendingJump = typeof generationController?.isForwardPassJumpPending === 'function'
            ? generationController.isForwardPassJumpPending()
            : (typeof generationController?.isNextForwardPassPending === 'function'
                && generationController.isNextForwardPassPending());
        const isBusy = skippingLayer || skippingAll || pendingJump;

        setVisible(hasNext);
        button.disabled = !hasNext || isBusy;
        button.dataset.state = isBusy ? 'skipping' : 'ready';
        button.textContent = isBusy ? 'Skipping' : 'Next pass';
        button.setAttribute('aria-busy', isBusy ? 'true' : 'false');

        rafId = scheduleFrame(update);
    };
    update();

    return () => {
        button.removeEventListener('click', onClick);
        if (rafId !== null) cancelFrame(rafId);
    };
}

export default initSkipNextPassButton;
