import { appState } from '../state/appState.js';

export function initPauseButton(pipeline) {
    const button = document.getElementById('pauseBtn');
    if (!button) return;

    const engine = pipeline?.engine;
    if (!engine) return;

    let isPaused = false;

    const updateButtonVisuals = () => {
        button.dataset.state = isPaused ? 'paused' : 'running';
        button.setAttribute('aria-pressed', String(isPaused));
        const label = isPaused ? 'Resume animation' : 'Pause animation';
        button.setAttribute('aria-label', label);
        button.setAttribute('title', label);
        button.textContent = isPaused ? 'Resume' : 'Pause';
    };

    const applyPauseState = (nextPaused) => {
        if (isPaused === nextPaused) return;
        isPaused = nextPaused;
        appState.userPaused = isPaused;
        if (isPaused) {
            engine.pause?.('manual');
        } else {
            engine.resume?.('manual');
        }
        updateButtonVisuals();
    };

    const toggle = () => {
        applyPauseState(!isPaused);
    };

    const onClick = (event) => {
        event.preventDefault();
        toggle();
    };

    const onKeyDown = (event) => {
        if (event.key === ' ' || event.key === 'Enter') {
            event.preventDefault();
            toggle();
        }
    };

    button.addEventListener('click', onClick);
    button.addEventListener('keydown', onKeyDown);

    updateButtonVisuals();

    return () => {
        button.removeEventListener('click', onClick);
        button.removeEventListener('keydown', onKeyDown);
    };
}

export default initPauseButton;
