// @vitest-environment jsdom

import { describe, expect, it, vi } from 'vitest';
import {
    enhanceProjectInfoContent,
    PROJECT_INFO_FEEDBACK_TARGET_ID
} from './projectInfoContent.js';

describe('projectInfoContent', () => {
    it('assigns the feedback anchor and smooth-scrolls to it for internal links', () => {
        document.body.innerHTML = `
            <div id="content">
                <blockquote><p>Feedback lives here.</p></blockquote>
                <p><a href="#project-info-feedback">form above</a></p>
            </div>
        `;

        const contentEl = document.getElementById('content');
        const feedbackTarget = contentEl?.querySelector('blockquote');
        const feedbackLink = contentEl?.querySelector('a[href="#project-info-feedback"]');

        feedbackTarget.scrollIntoView = vi.fn();
        feedbackTarget.focus = vi.fn();

        enhanceProjectInfoContent(contentEl);
        feedbackLink?.dispatchEvent(new MouseEvent('click', { bubbles: true, cancelable: true }));

        expect(feedbackTarget?.id).toBe(PROJECT_INFO_FEEDBACK_TARGET_ID);
        expect(feedbackTarget?.getAttribute('tabindex')).toBe('-1');
        expect(feedbackTarget?.classList.contains('project-info-scroll-target')).toBe(true);
        expect(feedbackTarget.scrollIntoView).toHaveBeenCalledWith({ behavior: 'smooth', block: 'start' });
        expect(feedbackTarget.focus).toHaveBeenCalled();
    });
});
