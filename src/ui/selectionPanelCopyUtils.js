function escapeHtml(value) {
    return String(value)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
}

function renderDescriptionHtml(text) {
    if (!text) return '';
    const raw = String(text);
    const katex = (typeof window !== 'undefined' && window.katex) ? window.katex : null;
    if (!katex || typeof katex.renderToString !== 'function' || !raw.includes('$')) {
        return escapeHtml(raw).replace(/\n/g, '<br />');
    }

    const parts = [];
    let cursor = 0;
    while (cursor < raw.length) {
        const nextDisplay = raw.indexOf('$$', cursor);
        const nextInline = raw.indexOf('$', cursor);
        if (nextDisplay === -1 && nextInline === -1) {
            parts.push({ type: 'text', value: raw.slice(cursor) });
            break;
        }
        let start = nextInline;
        let isDisplay = false;
        if (nextDisplay !== -1 && (nextInline === -1 || nextDisplay <= nextInline)) {
            start = nextDisplay;
            isDisplay = true;
        }
        if (start > cursor) {
            parts.push({ type: 'text', value: raw.slice(cursor, start) });
        }
        if (isDisplay) {
            const end = raw.indexOf('$$', start + 2);
            if (end === -1) {
                parts.push({ type: 'text', value: raw.slice(start) });
                break;
            }
            parts.push({ type: 'math', value: raw.slice(start + 2, end), display: true });
            cursor = end + 2;
        } else {
            const end = raw.indexOf('$', start + 1);
            if (end === -1) {
                parts.push({ type: 'text', value: raw.slice(start) });
                break;
            }
            parts.push({ type: 'math', value: raw.slice(start + 1, end), display: false });
            cursor = end + 1;
        }
    }

    return parts.map((part) => {
        if (part.type === 'text') {
            return escapeHtml(part.value).replace(/\n/g, '<br />');
        }
        try {
            return katex.renderToString(part.value, { throwOnError: false, displayMode: part.display });
        } catch (_) {
            const fallback = part.display ? `$$${part.value}$$` : `$${part.value}$`;
            return escapeHtml(fallback);
        }
    }).join('');
}

export function setDescriptionContent(element, text) {
    if (!element) return;
    element.innerHTML = renderDescriptionHtml(text || '');
}

function isVisibleForContextCopy(element) {
    if (!element || element.hidden) return false;
    if (typeof window !== 'undefined' && typeof window.getComputedStyle === 'function') {
        const style = window.getComputedStyle(element);
        if (style.display === 'none' || style.visibility === 'hidden') return false;
    }
    return true;
}

export function collectVisibleContextText(root, { excludeSelectors = '' } = {}) {
    if (!root || typeof document === 'undefined' || typeof NodeFilter === 'undefined') return [];
    const lines = [];
    const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
    let node = walker.nextNode();
    while (node) {
        const parent = node.parentElement;
        if (!parent) {
            node = walker.nextNode();
            continue;
        }
        if (excludeSelectors && parent.closest(excludeSelectors)) {
            node = walker.nextNode();
            continue;
        }
        if (!isVisibleForContextCopy(parent)) {
            node = walker.nextNode();
            continue;
        }
        const line = String(node.textContent || '').replace(/\s+/g, ' ').trim();
        if (line && lines[lines.length - 1] !== line) {
            lines.push(line);
        }
        node = walker.nextNode();
    }
    return lines;
}

function fallbackCopyText(text) {
    if (typeof document === 'undefined' || !document.body) return false;
    const area = document.createElement('textarea');
    area.value = text;
    area.setAttribute('readonly', '');
    area.style.position = 'fixed';
    area.style.top = '-10000px';
    area.style.left = '-10000px';
    area.style.opacity = '0';
    document.body.appendChild(area);
    try {
        area.focus({ preventScroll: true });
    } catch (_) {
        area.focus();
    }
    area.select();
    area.setSelectionRange(0, area.value.length);
    let copied = false;
    try {
        copied = !!document.execCommand('copy');
    } catch (_) {
        copied = false;
    }
    document.body.removeChild(area);
    return copied;
}

export async function copyTextToClipboard(text) {
    const value = String(text || '');
    if (!value.trim().length) return false;
    if (
        typeof navigator !== 'undefined'
        && navigator.clipboard
        && typeof navigator.clipboard.writeText === 'function'
        && (typeof window === 'undefined' || window.isSecureContext)
    ) {
        try {
            await navigator.clipboard.writeText(value);
            return true;
        } catch (_) {
            // Fall back to execCommand copy path below.
        }
    }
    return fallbackCopyText(value);
}
