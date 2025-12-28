export function initConveyorSkipButton(pipeline) {
    if (!pipeline) return () => {};

    const existingButton = document.getElementById('skipConveyorBtn');
    const topControls = document.getElementById('topControls');
    const button = existingButton || (() => {
        const btn = document.createElement('button');
        btn.id = 'skipConveyorBtn';
        btn.type = 'button';
        btn.textContent = 'Skip attention';
        btn.title = 'Skip attention conveyor';
        btn.setAttribute('aria-label', 'Skip attention conveyor');
        btn.dataset.visible = 'false';
        btn.style.display = 'none';
        if (topControls) {
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

    const getActiveLayer = () => {
        const layers = pipeline?._layers;
        if (!layers || !layers.length) return null;
        const idx = pipeline?._currentLayerIdx ?? 0;
        return layers[idx] || null;
    };

    const isSkippable = () => {
        const layer = getActiveLayer();
        const mhsa = layer && layer.mhsaAnimation;
        if (!mhsa) return false;
        if (mhsa.mhaPassThroughPhase !== 'mha_pass_through_complete') return false;
        if (mhsa.rowMergePhase && mhsa.rowMergePhase !== 'not_started') return false;
        const sa = mhsa.selfAttentionAnimator;
        if (!sa) return false;
        if (typeof sa.isConveyorActive === 'function') return sa.isConveyorActive();
        return sa.phase !== 'complete';
    };

    const onClick = (event) => {
        event.preventDefault();
        const layer = getActiveLayer();
        const mhsa = layer && layer.mhsaAnimation;
        if (mhsa && typeof mhsa.skipSelfAttentionAndStartConcat === 'function') {
            mhsa.skipSelfAttentionAndStartConcat();
        } else if (mhsa && mhsa.selfAttentionAnimator && typeof mhsa.selfAttentionAnimator.forceComplete === 'function') {
            mhsa.selfAttentionAnimator.forceComplete();
        }
        setVisible(false);
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
        setVisible(isSkippable());
        rafId = scheduleFrame(update);
    };
    update();

    return () => {
        button.removeEventListener('click', onClick);
        if (rafId !== null) cancelFrame(rafId);
    };
}

export default initConveyorSkipButton;
