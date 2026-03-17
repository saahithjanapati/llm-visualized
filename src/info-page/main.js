import infoMarkdown from '../ui/infoModalContent.md?raw';
import { renderSimpleMarkdown } from '../ui/simpleMarkdown.js';
import './page.css';

const contentEl = document.getElementById('infoPageContent');

if (contentEl) {
    contentEl.innerHTML = renderSimpleMarkdown(infoMarkdown);
}
