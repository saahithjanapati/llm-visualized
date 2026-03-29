import infoMarkdown from '../ui/infoModalContent.md?raw';
import { trackGoogleAnalyticsPageView } from '../app/gpt-tower/googleAnalytics.js';
import {
    bindProjectInfoBackLink,
    syncProjectInfoBackLink
} from '../ui/projectInfoNavigation.js';
import { enhanceProjectInfoContent } from '../ui/projectInfoContent.js';
import { renderSimpleMarkdown } from '../ui/simpleMarkdown.js';
import './page.css';

const contentEl = document.getElementById('infoPageContent');
const backLinkEl = document.querySelector('.info-page-back-link');

trackGoogleAnalyticsPageView(window.location);

if (backLinkEl) {
    syncProjectInfoBackLink(backLinkEl);
    bindProjectInfoBackLink(backLinkEl);
}

if (contentEl) {
    contentEl.innerHTML = renderSimpleMarkdown(infoMarkdown);
    enhanceProjectInfoContent(contentEl);
}
