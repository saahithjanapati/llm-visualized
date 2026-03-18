import { describe, expect, it } from 'vitest';
import { renderSimpleMarkdown } from './simpleMarkdown.js';

describe('renderSimpleMarkdown', () => {
    it('renders headings, blockquotes, lists, links, internal anchors, keycaps, and escaped content', () => {
        const html = renderSimpleMarkdown([
            '# LLM visualized',
            '',
            '> Share feedback in [this form](https://example.com/form).',
            '',
            '- Pan with `W` `A` `S` `D`',
            '- Inspect the 2D view',
            '',
            'Jump back to the [feedback form](#project-info-feedback).',
            '',
            'Paragraph with <unsafe> markup.'
        ].join('\n'));

        expect(html).toContain('<h1>LLM visualized</h1>');
        expect(html).toContain('<blockquote><p>Share feedback in <a href="https://example.com/form" target="_blank" rel="noopener noreferrer">this form</a>.</p></blockquote>');
        expect(html).toContain('<ul><li>Pan with <kbd>W</kbd> <kbd>A</kbd> <kbd>S</kbd> <kbd>D</kbd></li><li>Inspect the 2D view</li></ul>');
        expect(html).toContain('<p>Jump back to the <a href="#project-info-feedback">feedback form</a>.</p>');
        expect(html).toContain('<p>Paragraph with &lt;unsafe&gt; markup.</p>');
    });
});
