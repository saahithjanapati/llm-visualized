import { describe, expect, it } from 'vitest';
import { renderSimpleMarkdown } from './simpleMarkdown.js';

describe('renderSimpleMarkdown', () => {
    it('renders headings, blockquotes, lists, links, and escaped content', () => {
        const html = renderSimpleMarkdown([
            '# LLM visualized',
            '',
            '> Share feedback in [this form](https://example.com/form).',
            '',
            '- Follow the model state',
            '- Inspect the 2D view',
            '',
            'Paragraph with <unsafe> markup.'
        ].join('\n'));

        expect(html).toContain('<h1>LLM visualized</h1>');
        expect(html).toContain('<blockquote><p>Share feedback in <a href="https://example.com/form" target="_blank" rel="noopener noreferrer">this form</a>.</p></blockquote>');
        expect(html).toContain('<ul><li>Follow the model state</li><li>Inspect the 2D view</li></ul>');
        expect(html).toContain('<p>Paragraph with &lt;unsafe&gt; markup.</p>');
    });
});
