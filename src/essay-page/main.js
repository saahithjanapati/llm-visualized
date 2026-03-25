import {
    bindProjectInfoBackLink,
    syncProjectInfoBackLink
} from '../ui/projectInfoNavigation.js';
import './page.css';

const backLinkEl = document.querySelector('[data-essay-back-link]');

if (backLinkEl) {
    syncProjectInfoBackLink(backLinkEl, { defaultHref: '/' });
    bindProjectInfoBackLink(backLinkEl, { defaultHref: '/' });
}
