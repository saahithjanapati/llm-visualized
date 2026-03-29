import { trackGoogleAnalyticsPageView } from '../app/gpt-tower/googleAnalytics.js';
import {
    bindProjectInfoBackLink,
    syncProjectInfoBackLink
} from '../ui/projectInfoNavigation.js';
import './page.css';

const backLinkEl = document.querySelector('[data-essay-back-link]');

trackGoogleAnalyticsPageView(window.location);

if (backLinkEl) {
    syncProjectInfoBackLink(backLinkEl, { defaultHref: '/' });
    bindProjectInfoBackLink(backLinkEl, { defaultHref: '/' });
}
