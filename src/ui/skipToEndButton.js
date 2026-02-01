import { appState } from '../state/appState.js';

export function initSkipToEndButton(pipeline) {
    if (!pipeline) return () => {};

    const existingButton = document.getElementById('skipToEndBtn');
    const topControls = document.getElementById('topControls');
    const equationsPanel = document.getElementById('equationsPanel');
    const button = existingButton || (() => {
        const btn = document.createElement('button');
        btn.id = 'skipToEndBtn';
        btn.type = 'button';
        btn.textContent = 'Skip to end';
        btn.title = 'Skip to end of forward pass';
        btn.setAttribute('aria-label', 'Skip to end of forward pass');
        btn.dataset.visible = 'true';
        if (topControls) {
            topControls.insertBefore(btn, document.getElementById('settingsBtn'));
        } else {
            document.body.appendChild(btn);
        }
        return btn;
    })();

    const applyEquationsVisibility = () => {
        if (!equationsPanel) return;
        const shouldShow = appState.showEquations && !appState.equationsSuppressed;
        const nextDisplay = shouldShow ? 'block' : 'none';
        if (equationsPanel.style.display !== nextDisplay) {
            equationsPanel.style.display = nextDisplay;
        }
    };

    const setVisible = (show) => {
        const next = show ? 'true' : 'false';
        if (button.dataset.visible === next) return;
        button.dataset.visible = next;
        button.style.display = show ? '' : 'none';
    };

    const onClick = (event) => {
        event.preventDefault();
        if (pipeline?.setAutoCameraFollow) {
            pipeline.setAutoCameraFollow(false, { immediate: true });
        }
        if (pipeline?.focusOverview) {
            pipeline.focusOverview({ immediate: false, durationMs: 1400 });
        }
        try {
            if (typeof window !== 'undefined' && typeof window.dispatchEvent === 'function') {
                window.dispatchEvent(new CustomEvent('autoCameraFollowRequest', { detail: { enabled: false, reason: 'skipToEnd' } }));
            }
        } catch (_) { /* ignore */ }
        if (pipeline && typeof pipeline.skipToEndForwardPass === 'function') {
            pipeline.skipToEndForwardPass();
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

        const skipping = typeof pipeline.isSkipToEndActive === 'function' && pipeline.isSkipToEndActive();
        if (appState.equationsSuppressed !== skipping) {
            appState.equationsSuppressed = skipping;
            applyEquationsVisibility();
        }
        setVisible(true);
        button.disabled = !!skipping;
        button.dataset.state = skipping ? 'skipping' : 'ready';
        button.textContent = skipping ? 'Skipping...' : 'Skip to end';
        button.setAttribute('aria-busy', skipping ? 'true' : 'false');
        rafId = scheduleFrame(update);
    };
    update();

    return () => {
        button.removeEventListener('click', onClick);
        if (rafId !== null) cancelFrame(rafId);
    };
}

export default initSkipToEndButton;
