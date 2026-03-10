export const MHSA_INFO_PANEL_ACTION_OPEN = 'open-mhsa-info';

export function setDescriptionMhsaInfoAction(descriptionEl, enabled = false) {
    if (!descriptionEl) return;

    descriptionEl
        .querySelectorAll(`[data-detail-action="${MHSA_INFO_PANEL_ACTION_OPEN}"]`)
        .forEach((node) => {
            const row = node.closest('.detail-description-action-row');
            if (row) {
                row.remove();
                return;
            }
            node.remove();
        });

    if (!enabled) return;

    const actionRow = document.createElement('div');
    actionRow.className = 'detail-description-action-row';

    const actionBtn = document.createElement('button');
    actionBtn.type = 'button';
    actionBtn.className = 'detail-description-action-link';
    actionBtn.dataset.detailAction = MHSA_INFO_PANEL_ACTION_OPEN;
    actionBtn.textContent = 'View matrix form';
    actionBtn.setAttribute('aria-label', 'Open the multi-head self-attention matrix form view');

    actionRow.appendChild(actionBtn);
    descriptionEl.appendChild(actionRow);
}
