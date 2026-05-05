// @vitest-environment jsdom

import { describe, expect, it, vi } from 'vitest';
import {
    enhanceProjectInfoContent,
    PROJECT_INFO_FEEDBACK_SPOTLIGHT_CLASS,
    PROJECT_INFO_FEEDBACK_TARGET_ID,
    PROJECT_INFO_INTRO_SECTION_CLASS,
    PROJECT_INFO_SECTION_CLASS
} from './projectInfoContent.js';

describe('projectInfoContent', () => {
    it('assigns the feedback anchor, smooth-scrolls to it, and briefly spotlights it for internal links', () => {
        vi.useFakeTimers();
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
        expect(feedbackTarget?.classList.contains(PROJECT_INFO_FEEDBACK_SPOTLIGHT_CLASS)).toBe(true);

        vi.advanceTimersByTime(1800);

        expect(feedbackTarget?.classList.contains(PROJECT_INFO_FEEDBACK_SPOTLIGHT_CLASS)).toBe(false);
        vi.useRealTimers();
    });

    it('groups rendered content into semantic sections for responsive layouts', () => {
        document.body.innerHTML = `
            <div id="content">
                <blockquote><p>Feedback lives here.</p></blockquote>
                <p>Intro copy.</p>
                <h2>Overview</h2>
                <p>Overview copy.</p>
                <h2>Keyboard Controls</h2>
                <p>Keyboard copy.</p>
            </div>
        `;

        const contentEl = document.getElementById('content');

        enhanceProjectInfoContent(contentEl);
        enhanceProjectInfoContent(contentEl);

        const sections = Array.from(contentEl?.querySelectorAll(`.${PROJECT_INFO_SECTION_CLASS}`) || []);

        expect(sections).toHaveLength(3);
        expect(sections[0]?.classList.contains(PROJECT_INFO_INTRO_SECTION_CLASS)).toBe(true);
        expect(sections[0]?.querySelector('blockquote')).not.toBeNull();
        expect(sections[1]?.getAttribute('aria-labelledby')).toBe('project-info-overview');
        expect(sections[2]?.getAttribute('aria-labelledby')).toBe('project-info-keyboard-controls');
    });
});
