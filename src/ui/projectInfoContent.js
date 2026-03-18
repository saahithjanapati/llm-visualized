export const PROJECT_INFO_FEEDBACK_TARGET_ID = 'project-info-feedback';
export const PROJECT_INFO_FEEDBACK_SPOTLIGHT_CLASS = 'is-feedback-spotlighted';

const FEEDBACK_SPOTLIGHT_DURATION_MS = 1800;
const feedbackSpotlightTimeouts = new WeakMap();

function focusScrollTarget(target) {
    if (!target || typeof target.focus !== 'function') return;
    try {
        target.focus({ preventScroll: true });
    } catch {
        target.focus();
    }
}

function spotlightScrollTarget(target) {
    if (!target?.classList) return;

    const previousTimeout = feedbackSpotlightTimeouts.get(target);
    if (previousTimeout) {
        clearTimeout(previousTimeout);
    }

    target.classList.remove(PROJECT_INFO_FEEDBACK_SPOTLIGHT_CLASS);
    void target.offsetWidth;
    target.classList.add(PROJECT_INFO_FEEDBACK_SPOTLIGHT_CLASS);

    const timeoutId = setTimeout(() => {
        target.classList.remove(PROJECT_INFO_FEEDBACK_SPOTLIGHT_CLASS);
        feedbackSpotlightTimeouts.delete(target);
    }, FEEDBACK_SPOTLIGHT_DURATION_MS);

    feedbackSpotlightTimeouts.set(target, timeoutId);
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
        spotlightScrollTarget(target);
    });
}
