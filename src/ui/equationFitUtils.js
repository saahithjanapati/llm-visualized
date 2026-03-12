function measureKatexBaseBounds(katexRoot) {
    if (!katexRoot) return null;
    const bases = katexRoot.querySelectorAll('.katex-html .base');
    if (!bases || !bases.length) return null;

    let left = Infinity;
    let right = -Infinity;
    let top = Infinity;
    let bottom = -Infinity;

    bases.forEach((base) => {
        const rect = base.getBoundingClientRect();
        if (!(rect.width > 0 && rect.height > 0)) return;
        left = Math.min(left, rect.left);
        right = Math.max(right, rect.right);
        top = Math.min(top, rect.top);
        bottom = Math.max(bottom, rect.bottom);
    });

    if (!Number.isFinite(left) || !Number.isFinite(right) || !Number.isFinite(top) || !Number.isFinite(bottom)) {
        return null;
    }

    return {
        width: Math.max(0, right - left),
        height: Math.max(0, bottom - top)
    };
}

function isKatexRoot(el) {
    return !!(el?.classList?.contains('katex') && el.parentElement?.classList?.contains('katex-display'));
}

function isKatexDisplay(el) {
    return !!el?.classList?.contains('katex-display');
}

function measureEquationElement(el) {
    if (!el) return { width: 0, height: 0 };

    const katexRoot = isKatexRoot(el)
        ? el
        : el.querySelector('.katex-display > .katex');
    const katexDisplay = isKatexDisplay(el)
        ? el
        : el.querySelector('.katex-display');
    const baseBounds = measureKatexBaseBounds(katexRoot);

    if (baseBounds && katexRoot) {
        const rootRect = katexRoot.getBoundingClientRect();
        return {
            width: Math.max(0, baseBounds.width + 1),
            height: Math.max(0, baseBounds.height, katexRoot.scrollHeight, rootRect.height || 0)
        };
    }

    if (katexRoot) {
        const rect = katexRoot.getBoundingClientRect();
        return {
            width: Math.max(0, katexRoot.scrollWidth, rect.width || 0),
            height: Math.max(0, katexRoot.scrollHeight, rect.height || 0)
        };
    }

    if (katexDisplay) {
        const rect = katexDisplay.getBoundingClientRect();
        return {
            width: Math.max(0, katexDisplay.scrollWidth, rect.width || 0),
            height: Math.max(0, katexDisplay.scrollHeight, rect.height || 0)
        };
    }

    const rect = typeof el.getBoundingClientRect === 'function'
        ? el.getBoundingClientRect()
        : { width: 0, height: 0 };
    return {
        width: Math.max(0, el.scrollWidth || 0, rect.width || 0),
        height: Math.max(0, el.scrollHeight || 0, rect.height || 0)
    };
}

export function readEquationBaseFontPx(containerEl, fallbackPx = 12) {
    if (!containerEl || typeof window === 'undefined') return fallbackPx;

    const previous = containerEl.style.fontSize;
    containerEl.style.fontSize = '';
    const parsed = Number.parseFloat(window.getComputedStyle(containerEl).fontSize);
    containerEl.style.fontSize = previous;

    return Number.isFinite(parsed) ? parsed : fallbackPx;
}

export function readEquationContentSize(containerEl) {
    if (!containerEl) return { width: 0, height: 0 };

    const previewLines = Array.from(containerEl.children || []).filter((child) => (
        child?.classList?.contains('detail-preview-equation')
    ));
    if (previewLines.length) {
        return previewLines.reduce((acc, lineEl) => {
            const size = measureEquationElement(lineEl);
            acc.width = Math.max(acc.width, size.width);
            acc.height += size.height;
            return acc;
        }, { width: 0, height: 0 });
    }

    return measureEquationElement(containerEl);
}
