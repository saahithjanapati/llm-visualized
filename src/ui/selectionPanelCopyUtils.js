import { resolveTokenChipColors } from './tokenChipColorUtils.js';

function escapeHtml(value) {
    return String(value)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
}

const INLINE_TOKEN_NAV_PATTERN = /\[\[token-nav\|([^|\]]*)\|([^|\]]*)\|([^|\]]*)\]\]/g;
const INLINE_DETAIL_ACTION_PATTERN = /\[\[detail-action\|([^|\]]*)\|([^|\]]*)\|([^|\]]*)\]\]/g;

function decodeInlineTokenNavText(value) {
    if (typeof value !== 'string' || !value.length) return '';
    try {
        return decodeURIComponent(value);
    } catch (_) {
        return value;
    }
}

function decodeInlineDetailActionPayload(value) {
    if (typeof value !== 'string' || !value.length) return {};
    try {
        const parsed = JSON.parse(decodeURIComponent(value));
        return parsed && typeof parsed === 'object' ? parsed : {};
    } catch (_) {
        return {};
    }
}

function renderInlineTokenNavHtml(tokenTextEncoded, tokenIndexRaw, tokenIdRaw) {
    const tokenText = decodeInlineTokenNavText(tokenTextEncoded).trim();
    if (!tokenText) return '';
    const tokenIndex = Number(tokenIndexRaw);
    const tokenId = Number(tokenIdRaw);
    const safeTokenIndex = Number.isFinite(tokenIndex) ? Math.floor(tokenIndex) : null;
    const safeTokenId = Number.isFinite(tokenId) ? Math.floor(tokenId) : null;
    const colors = resolveTokenChipColors({
        tokenLabel: tokenText,
        tokenIndex: safeTokenIndex,
        tokenId: safeTokenId
    }, Number.isFinite(safeTokenIndex) ? safeTokenIndex : 0);
    const chipStyle = [
        `--token-color-border: ${colors.border}`,
        `--token-color-fill: ${colors.fill}`,
        `--token-color-fill-hover: ${colors.fillHover}`
    ].join('; ');

    const attrs = [
        'class="detail-subtitle-token-chip detail-token-nav-chip detail-description-token-chip"',
        'data-token-nav="true"',
        `data-token-text="${escapeHtml(tokenText)}"`,
        `style="${escapeHtml(chipStyle)}"`,
        'tabindex="0"',
        'role="button"',
        `aria-label="${escapeHtml(`Open token details for ${tokenText}`)}"`,
        `title="${escapeHtml(tokenText)}"`
    ];
    if (Number.isFinite(safeTokenIndex)) {
        attrs.push(`data-token-index="${safeTokenIndex}"`);
    }
    if (Number.isFinite(safeTokenId)) {
        attrs.push(`data-token-id="${safeTokenId}"`);
    }

    return `<span ${attrs.join(' ')}>${escapeHtml(tokenText)}</span>`;
}

function splitMathParts(raw) {
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
    return parts;
}

function renderTextWithMath(text, { allowPlaceholders = true } = {}) {
    if (!text) return '';
    const raw = String(text);
    const katex = (typeof window !== 'undefined' && window.katex) ? window.katex : null;
    if (!katex || typeof katex.renderToString !== 'function' || !raw.includes('$')) {
        return allowPlaceholders
            ? renderDescriptionTextSegment(raw)
            : escapeHtml(raw).replace(/\n/g, '<br />');
    }

    return splitMathParts(raw).map((part) => {
        if (part.type === 'text') {
            return allowPlaceholders
                ? renderDescriptionTextSegment(part.value)
                : escapeHtml(part.value).replace(/\n/g, '<br />');
        }
        try {
            return katex.renderToString(part.value, { throwOnError: false, displayMode: part.display });
        } catch (_) {
            const fallback = part.display ? `$$${part.value}$$` : `$${part.value}$`;
            return escapeHtml(fallback);
        }
    }).join('');
}

function renderInlineDetailActionHtml(linkTextEncoded, actionRaw, payloadRaw) {
    const linkText = decodeInlineTokenNavText(linkTextEncoded).trim();
    const action = typeof actionRaw === 'string' ? actionRaw.trim() : '';
    if (!linkText || !action) return escapeHtml(linkText || '');
    const payload = decodeInlineDetailActionPayload(payloadRaw);
    const attrs = [
        'type="button"',
        'class="detail-description-action-link detail-description-inline-action-link"',
        `data-detail-action="${escapeHtml(action)}"`,
        `aria-label="${escapeHtml(linkText)}"`
    ];
    if (payload && Object.keys(payload).length > 0) {
        attrs.push(`data-detail-payload="${escapeHtml(encodeURIComponent(JSON.stringify(payload)))}"`);
    }
    return `<button ${attrs.join(' ')}>${renderTextWithMath(linkText, { allowPlaceholders: false })}</button>`;
}

function getNextInlinePlaceholder(raw, cursor) {
    INLINE_TOKEN_NAV_PATTERN.lastIndex = cursor;
    INLINE_DETAIL_ACTION_PATTERN.lastIndex = cursor;
    const tokenMatch = INLINE_TOKEN_NAV_PATTERN.exec(raw);
    const actionMatch = INLINE_DETAIL_ACTION_PATTERN.exec(raw);
    if (!tokenMatch && !actionMatch) return null;
    if (!actionMatch || (tokenMatch && tokenMatch.index <= actionMatch.index)) {
        return {
            type: 'token-nav',
            match: tokenMatch
        };
    }
    return {
        type: 'detail-action',
        match: actionMatch
    };
}

function renderDescriptionTextSegment(text) {
    if (!text) return '';
    const raw = String(text);
    let html = '';
    let cursor = 0;
    while (cursor < raw.length) {
        const next = getNextInlinePlaceholder(raw, cursor);
        if (!next) break;
        const { type, match } = next;
        if (match.index > cursor) {
            html += escapeHtml(raw.slice(cursor, match.index)).replace(/\n/g, '<br />');
        }
        if (type === 'token-nav') {
            html += renderInlineTokenNavHtml(match[1], match[2], match[3]);
        } else {
            html += renderInlineDetailActionHtml(match[1], match[2], match[3]);
        }
        cursor = match.index + match[0].length;
    }
    if (cursor < raw.length) {
        html += escapeHtml(raw.slice(cursor)).replace(/\n/g, '<br />');
    }
    return html;
}

export function getDescriptionPlainText(text) {
    if (!text) return '';
    return String(text)
        .replace(INLINE_TOKEN_NAV_PATTERN, (_, tokenTextEncoded) => decodeInlineTokenNavText(tokenTextEncoded))
        .replace(INLINE_DETAIL_ACTION_PATTERN, (_, linkTextEncoded) => decodeInlineTokenNavText(linkTextEncoded));
}

function renderDescriptionHtml(text) {
    if (!text) return '';
    return renderTextWithMath(text, { allowPlaceholders: true });
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
