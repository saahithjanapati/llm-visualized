import infoMarkdown from '../ui/infoModalContent.md?raw';
import { syncProjectInfoBackLink } from '../ui/projectInfoNavigation.js';
import { renderSimpleMarkdown } from '../ui/simpleMarkdown.js';
import './page.css';

const contentEl = document.getElementById('infoPageContent');
const backLinkEl = document.querySelector('.info-page-back-link');

if (backLinkEl) {
    syncProjectInfoBackLink(backLinkEl);
}

if (contentEl) {
    contentEl.innerHTML = renderSimpleMarkdown(infoMarkdown);
}
