export const PROJECT_INFO_FEEDBACK_TARGET_ID = 'project-info-feedback';

function focusScrollTarget(target) {
    if (!target || typeof target.focus !== 'function') return;
    try {
        target.focus({ preventScroll: true });
    } catch {
        target.focus();
    }
}

export function enhanceProjectInfoContent(contentEl) {
    if (!contentEl || typeof contentEl.querySelector !== 'function') return;

    const feedbackTarget = contentEl.querySelector('blockquote');
    if (feedbackTarget) {
        if (!feedbackTarget.id) {
            feedbackTarget.id = PROJECT_INFO_FEEDBACK_TARGET_ID;
        }
        if (!feedbackTarget.hasAttribute('tabindex')) {
            feedbackTarget.setAttribute('tabindex', '-1');
        }
        feedbackTarget.classList.add('project-info-scroll-target');
    }

    if (contentEl.dataset.projectInfoInternalAnchorsBound === 'true') {
        return;
    }
    contentEl.dataset.projectInfoInternalAnchorsBound = 'true';

    contentEl.addEventListener('click', (event) => {
        const anchor = typeof event.target?.closest === 'function'
            ? event.target.closest('a[href^="#"]')
            : null;
        if (!anchor || !contentEl.contains(anchor)) return;

        const href = anchor.getAttribute('href') || '';
        const targetId = href.slice(1).trim();
        if (!targetId) return;

        const target = document.getElementById(decodeURIComponent(targetId));
        if (!target || !contentEl.contains(target)) return;

        event.preventDefault();
        if (typeof target.scrollIntoView === 'function') {
            target.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
        focusScrollTarget(target);
    });
}
