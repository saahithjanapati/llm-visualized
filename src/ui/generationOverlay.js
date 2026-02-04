const DEFAULT_TEXT_HOLD_MS = 900;
const DEFAULT_TOKEN_HOLD_MS = 1200;

const COLOR_SEED = 47;
const tokenColorForIndex = (idx) => {
    const hue = (idx * COLOR_SEED) % 360;
    return `hsl(${hue}, 78%, 72%)`;
};

export function initGenerationOverlay() {
    const overlay = document.getElementById('generationOverlay');
    const titleEl = document.getElementById('generationTitle');
    const plainEl = document.getElementById('generationPlain');
    const tokenizedEl = document.getElementById('generationTokenized');
    const hintEl = document.getElementById('generationHint');

    if (!overlay || !plainEl || !tokenizedEl) {
        return null;
    }

    let timers = [];

    const clearTimers = () => {
        timers.forEach((timer) => clearTimeout(timer));
        timers = [];
    };

    const setVisible = (visible) => {
        overlay.dataset.visible = visible ? 'true' : 'false';
        overlay.setAttribute('aria-hidden', visible ? 'false' : 'true');
    };

    const setPhase = (phase) => {
        overlay.dataset.phase = phase;
    };

    const setTitle = (title) => {
        if (titleEl) titleEl.textContent = title || 'Prompt';
    };

    const setHint = (text) => {
        if (!hintEl) return;
        hintEl.textContent = text || '';
    };

    const setPlainText = (text) => {
        plainEl.textContent = text || '';
    };

    const setTokens = (tokens, highlightIndex = null) => {
        tokenizedEl.innerHTML = '';
        const list = Array.isArray(tokens) ? tokens : [];
        list.forEach((token, idx) => {
            const span = document.createElement('span');
            span.className = 'gen-token';
            if (Number.isFinite(highlightIndex) && idx === highlightIndex) {
                span.className += ' gen-token--new';
            }
            span.style.setProperty('--token-color', tokenColorForIndex(idx));
            span.textContent = token;
            tokenizedEl.appendChild(span);
        });
    };

    const showSequence = ({
        title = 'Prompt',
        plainText = '',
        tokens = [],
        highlightIndex = null,
        hint = '',
        textHoldMs = DEFAULT_TEXT_HOLD_MS,
        tokenHoldMs = DEFAULT_TOKEN_HOLD_MS,
        onComplete = null
    } = {}) => {
        clearTimers();
        setTitle(title);
        setHint(hint);
        setPlainText(plainText);
        setTokens(tokens, highlightIndex);
        setPhase('text');
        setVisible(true);

        const toTokens = setTimeout(() => {
            setPhase('tokens');
        }, Math.max(0, textHoldMs));
        timers.push(toTokens);

        const done = setTimeout(() => {
            if (typeof onComplete === 'function') {
                onComplete();
            }
        }, Math.max(0, textHoldMs + tokenHoldMs));
        timers.push(done);
    };

    const hide = () => {
        clearTimers();
        setVisible(false);
        setPhase('text');
    };

    return {
        showSequence,
        hide,
        setVisible,
        setPhase,
        clearTimers
    };
}

export default initGenerationOverlay;
