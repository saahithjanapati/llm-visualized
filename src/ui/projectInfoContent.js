export const PROJECT_INFO_FEEDBACK_TARGET_ID = 'project-info-feedback';
export const PROJECT_INFO_FEEDBACK_SPOTLIGHT_CLASS = 'is-feedback-spotlighted';
export const PROJECT_INFO_SECTION_CLASS = 'project-info-section';
export const PROJECT_INFO_INTRO_SECTION_CLASS = 'project-info-section--intro';

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

function createProjectInfoSection(isIntro = false) {
    const section = document.createElement('section');
    section.classList.add(PROJECT_INFO_SECTION_CLASS);
    if (isIntro) {
        section.classList.add(PROJECT_INFO_INTRO_SECTION_CLASS);
    }
    return section;
}

function labelSectionFromHeading(section) {
    const heading = section?.querySelector?.('h2');
    if (!heading) return;

    if (!heading.id) {
        const text = heading.textContent || 'project-info-section';
        const slug = text
            .toLowerCase()
            .replace(/[^a-z0-9]+/g, '-')
            .replace(/^-+|-+$/g, '')
            || 'project-info-section';
        heading.id = `project-info-${slug}`;
    }
    section.setAttribute('aria-labelledby', heading.id);
}

function groupProjectInfoSections(contentEl) {
    if (contentEl.dataset.projectInfoSectionsBuilt === 'true') return;

    const children = Array.from(contentEl.children);
    if (!children.length) return;

    const sections = [];
    let currentSection = createProjectInfoSection(true);

    children.forEach((child) => {
        if (child.tagName === 'H2') {
            if (currentSection.children.length > 0) {
                sections.push(currentSection);
            }
            currentSection = createProjectInfoSection(false);
        }
        currentSection.appendChild(child);
    });

    if (currentSection.children.length > 0) {
        sections.push(currentSection);
    }

    sections.forEach((section) => {
        labelSectionFromHeading(section);
        contentEl.appendChild(section);
    });

    contentEl.dataset.projectInfoSectionsBuilt = 'true';
}

export function enhanceProjectInfoContent(contentEl) {
    if (!contentEl || typeof contentEl.querySelector !== 'function') return;

    groupProjectInfoSections(contentEl);

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
