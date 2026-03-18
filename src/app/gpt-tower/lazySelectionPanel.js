const SELECTION_PANEL_LOADING_COPY = Object.freeze({
    details: Object.freeze({
        title: 'Loading details…',
        subtitle: 'Preparing the selection panel for first use.'
    }),
    view2d: Object.freeze({
        title: 'Loading 2D view…',
        subtitle: 'Preparing the matrix view for first use.'
    })
});

function setText(element, text = '') {
    if (!element) return;
    element.textContent = text;
}

function createLoadingStateElement(documentRef, subtitle = '') {
    const container = documentRef.createElement('div');
    container.className = 'detail-loading-state';

    const spinner = documentRef.createElement('div');
    spinner.className = 'detail-loading-spinner';
    spinner.setAttribute('aria-hidden', 'true');

    const body = documentRef.createElement('div');
    body.className = 'detail-loading-copy';
    body.textContent = subtitle;

    container.append(spinner, body);
    return container;
}

function resolveLoadingElements() {
    if (typeof document === 'undefined') return null;
    return {
        hudStack: document.getElementById('hudStack'),
        hudPanel: document.getElementById('hudPanel'),
        panel: document.getElementById('detailPanel'),
        title: document.getElementById('detailTitle'),
        subtitle: document.getElementById('detailSubtitle'),
        subtitleSecondary: document.getElementById('detailSubtitleSecondary'),
        subtitleTertiary: document.getElementById('detailSubtitleTertiary'),
        description: document.getElementById('detailDescription'),
        closeButton: document.getElementById('detailClose')
    };
}

function showSelectionPanelLoadingState(mode = 'details') {
    const elements = resolveLoadingElements();
    if (!elements?.panel || !elements.description || typeof document === 'undefined') return;

    const copy = SELECTION_PANEL_LOADING_COPY[mode] || SELECTION_PANEL_LOADING_COPY.details;
    elements.panel.classList.add('is-open', 'is-loading');
    elements.panel.setAttribute('aria-hidden', 'false');
    elements.hudStack?.classList.add('detail-open');
    elements.hudPanel?.classList.add('detail-open');
    elements.closeButton?.setAttribute('disabled', 'disabled');
    setText(elements.title, copy.title);
    setText(elements.subtitle, '');
    setText(elements.subtitleSecondary, '');
    setText(elements.subtitleTertiary, '');
    elements.description.replaceChildren(createLoadingStateElement(document, copy.subtitle));
}

function hideSelectionPanelLoadingState({ closePanel = false } = {}) {
    const elements = resolveLoadingElements();
    if (!elements?.panel || !elements.description) return;

    elements.panel.classList.remove('is-loading');
    elements.closeButton?.removeAttribute('disabled');

    if (closePanel) {
        elements.panel.classList.remove('is-open');
        elements.panel.setAttribute('aria-hidden', 'true');
        elements.hudStack?.classList.remove('detail-open');
        elements.hudPanel?.classList.remove('detail-open');
        setText(elements.title, 'Selection');
        setText(elements.subtitle, '');
        setText(elements.subtitleSecondary, '');
        setText(elements.subtitleTertiary, '');
        elements.description.replaceChildren();
    }
}

export function createLazySelectionPanel(options = {}) {
    const baseOptions = { ...options };
    let panel = null;
    let panelPromise = null;
    let latestData = {
        activationSource: baseOptions.activationSource,
        laneTokenIndices: baseOptions.laneTokenIndices,
        tokenLabels: baseOptions.tokenLabels
    };

    const updateLoadedPanelData = (data = {}) => {
        latestData = {
            ...latestData,
            ...data
        };
        panel?.updateData?.(data);
    };

    const loadPanel = async ({ mode = 'details' } = {}) => {
        if (panel) return panel;
        if (panelPromise) return panelPromise;

        showSelectionPanelLoadingState(mode);
        panelPromise = import('../../ui/selectionPanel.js')
            .then(({ initSelectionPanel }) => {
                const resolvedPanel = initSelectionPanel({
                    ...baseOptions,
                    activationSource: latestData.activationSource,
                    laneTokenIndices: latestData.laneTokenIndices,
                    tokenLabels: latestData.tokenLabels
                });
                resolvedPanel.updateData?.(latestData);
                panel = resolvedPanel;
                return resolvedPanel;
            })
            .finally(() => {
                panelPromise = null;
            });

        return panelPromise;
    };

    const withPanel = async (mode, action) => {
        try {
            const resolvedPanel = await loadPanel({ mode });
            const result = await action(resolvedPanel);
            hideSelectionPanelLoadingState();
            return result;
        } catch (error) {
            hideSelectionPanelLoadingState({ closePanel: true });
            throw error;
        }
    };

    return {
        handleSelection: (selection) => withPanel('details', (resolvedPanel) => resolvedPanel.handleSelection(selection)),
        close: () => {
            panel?.close?.();
        },
        updateData: (data) => {
            updateLoadedPanelData(data);
        },
        openTransformerView2d: (config) => withPanel('view2d', (resolvedPanel) => resolvedPanel.openTransformerView2d(config)),
        isTransformerView2dOpen: () => panel?.isTransformerView2dOpen?.() === true
    };
}
