function renderEquationLineMarkup(tex) {
    const safeTex = typeof tex === 'string' ? tex.trim() : '';
    if (!safeTex) return '';
    const katex = (typeof window !== 'undefined') ? window.katex : null;
    if (katex && typeof katex.renderToString === 'function') {
        try {
            return katex.renderToString(safeTex, { throwOnError: false, displayMode: true });
        } catch (_) {
            // Fall back to raw text below.
        }
    }
    return safeTex;
}

export function renderSelectionPreviewEquations(containerEl, entries = []) {
    if (!containerEl) return;
    const safeEntries = Array.isArray(entries)
        ? entries.filter((entry) => typeof entry?.tex === 'string' && entry.tex.trim().length > 0)
        : [];

    containerEl.innerHTML = '';
    const hasEntries = safeEntries.length > 0;
    containerEl.classList.toggle('is-visible', hasEntries);
    containerEl.setAttribute('aria-hidden', hasEntries ? 'false' : 'true');
    if (!hasEntries) return;

    safeEntries.forEach((entry) => {
        const lineEl = document.createElement('div');
        lineEl.className = 'detail-preview-equation';
        lineEl.dataset.active = entry.active === false ? 'false' : 'true';

        const markup = renderEquationLineMarkup(entry.tex);
        if (markup === entry.tex) {
            lineEl.textContent = entry.tex;
        } else {
            lineEl.innerHTML = markup;
        }

        containerEl.appendChild(lineEl);
    });
}
