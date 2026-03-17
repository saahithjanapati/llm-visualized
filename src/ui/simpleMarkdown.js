function escapeHtml(value = '') {
    return String(value)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
}

function renderInlineMarkdown(text = '') {
    let html = escapeHtml(text);

    html = html.replace(
        /\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g,
        '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>'
    );
    html = html.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
    html = html.replace(/\*([^*]+)\*/g, '<em>$1</em>');

    return html;
}

function flushParagraph(lines, blocks) {
    if (!lines.length) return;
    blocks.push(`<p>${renderInlineMarkdown(lines.join(' '))}</p>`);
    lines.length = 0;
}

function flushList(items, blocks) {
    if (!items.length) return;
    blocks.push(`<ul>${items.map((item) => `<li>${renderInlineMarkdown(item)}</li>`).join('')}</ul>`);
    items.length = 0;
}

function flushBlockquote(lines, blocks) {
    if (!lines.length) return;
    blocks.push(`<blockquote><p>${renderInlineMarkdown(lines.join(' '))}</p></blockquote>`);
    lines.length = 0;
}

export function renderSimpleMarkdown(markdown = '') {
    const lines = String(markdown || '').replace(/\r\n?/g, '\n').split('\n');
    const blocks = [];
    const paragraphLines = [];
    const listItems = [];
    const blockquoteLines = [];

    const flushAll = () => {
        flushParagraph(paragraphLines, blocks);
        flushList(listItems, blocks);
        flushBlockquote(blockquoteLines, blocks);
    };

    lines.forEach((line) => {
        const trimmed = line.trim();

        if (!trimmed) {
            flushAll();
            return;
        }

        const headingMatch = trimmed.match(/^(#{1,3})\s+(.+)$/);
        if (headingMatch) {
            flushAll();
            const level = Math.min(headingMatch[1].length, 3);
            blocks.push(`<h${level}>${renderInlineMarkdown(headingMatch[2].trim())}</h${level}>`);
            return;
        }

        const blockquoteMatch = trimmed.match(/^>\s?(.*)$/);
        if (blockquoteMatch) {
            flushParagraph(paragraphLines, blocks);
            flushList(listItems, blocks);
            blockquoteLines.push(blockquoteMatch[1].trim());
            return;
        }

        const listMatch = trimmed.match(/^-\s+(.+)$/);
        if (listMatch) {
            flushParagraph(paragraphLines, blocks);
            flushBlockquote(blockquoteLines, blocks);
            listItems.push(listMatch[1].trim());
            return;
        }

        flushList(listItems, blocks);
        flushBlockquote(blockquoteLines, blocks);
        paragraphLines.push(trimmed);
    });

    flushAll();
    return blocks.join('');
}
