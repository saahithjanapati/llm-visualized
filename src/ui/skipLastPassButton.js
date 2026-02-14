export function initSkipLastPassButton({ pipeline, generationController } = {}) {
    const existingButton = document.getElementById('skipLastPassBtn');
    const menuItems = document.getElementById('skipMenuItems');
    const topControls = document.getElementById('topControls');
    const button = existingButton || (() => {
        const btn = document.createElement('button');
        btn.id = 'skipLastPassBtn';
        btn.type = 'button';
        btn.textContent = 'Last pass';
        btn.title = 'Skip to final forward pass';
        btn.setAttribute('aria-label', 'Skip to final forward pass');
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

    const hasLastPass = () => {
        if (typeof generationController?.hasLastForwardPass === 'function') {
            return generationController.hasLastForwardPass();
        }
        if (typeof generationController?.hasNextForwardPass === 'function') {
            return generationController.hasNextForwardPass();
        }
        return false;
    };

    const onClick = (event) => {
        event.preventDefault();
        if (!hasLastPass()) return;
        if (typeof generationController?.requestLastForwardPass === 'function') {
            generationController.requestLastForwardPass();
        } else if (typeof generationController?.requestNextForwardPass === 'function') {
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
        const hasLast = hasLastPass();
        const skippingLayer = typeof pipeline?.isSkipLayerActive === 'function' && pipeline.isSkipLayerActive();
        const skippingAll = typeof pipeline?.isSkipToEndActive === 'function' && pipeline.isSkipToEndActive();
        const pendingJump = typeof generationController?.isForwardPassJumpPending === 'function'
            ? generationController.isForwardPassJumpPending()
            : (typeof generationController?.isNextForwardPassPending === 'function'
                && generationController.isNextForwardPassPending());
        const isBusy = skippingLayer || skippingAll || pendingJump;

        setVisible(hasLast);
        button.disabled = !hasLast || isBusy;
        button.dataset.state = isBusy ? 'skipping' : 'ready';
        button.textContent = isBusy ? 'Skipping' : 'Last pass';
        button.setAttribute('aria-busy', isBusy ? 'true' : 'false');

        rafId = scheduleFrame(update);
    };
    update();

    return () => {
        button.removeEventListener('click', onClick);
        if (rafId !== null) cancelFrame(rafId);
    };
}

export default initSkipLastPassButton;
