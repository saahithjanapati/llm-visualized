import {
    PANEL_ACTION_HISTORY_BACK,
    PANEL_ACTION_HISTORY_FORWARD
} from './selectionPanelConstants.js';

function createHistoryButton(documentRef, {
    action = '',
    className = '',
    text = '',
    ariaLabel = ''
} = {}) {
    const button = documentRef.createElement('button');
    button.type = 'button';
    button.className = className;
    button.dataset.detailAction = action;
    button.textContent = text;
    if (ariaLabel) {
        button.setAttribute('aria-label', ariaLabel);
    }
    return button;
}

export function createSelectionPanelHistoryNavigation(documentRef = null) {
    if (!documentRef || typeof documentRef.createElement !== 'function') return null;

    const nav = documentRef.createElement('div');
    nav.className = 'detail-history-nav';
    nav.setAttribute('aria-label', 'Detail navigation history');

    const backBtn = createHistoryButton(documentRef, {
        action: PANEL_ACTION_HISTORY_BACK,
        className: 'detail-history-btn detail-history-btn--back',
        text: '\u2039',
        ariaLabel: 'Previous detail page'
    });
    const forwardBtn = createHistoryButton(documentRef, {
        action: PANEL_ACTION_HISTORY_FORWARD,
        className: 'detail-history-btn detail-history-btn--forward',
        text: '\u203a',
        ariaLabel: 'Next detail page'
    });

    nav.append(backBtn, forwardBtn);

    return {
        nav,
        backBtn,
        forwardBtn
    };
}
