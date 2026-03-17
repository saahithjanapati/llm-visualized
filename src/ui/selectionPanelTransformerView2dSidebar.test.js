import { beforeAll, describe, expect, it, vi } from 'vitest';

let SelectionPanel;

beforeAll(async () => {
    vi.stubGlobal('localStorage', {
        getItem: () => null,
        setItem: () => {},
        removeItem: () => {}
    });
    ({ SelectionPanel } = await import('./selectionPanel.js'));
});

function createClassListMock() {
    return {
        add: vi.fn(),
        remove: vi.fn(),
        toggle: vi.fn(),
        contains: vi.fn(() => false)
    };
}

function createPanelContext() {
    const panel = Object.create(SelectionPanel.prototype);
    panel.isOpen = true;
    panel.panel = {
        classList: createClassListMock()
    };
    panel.title = {
        innerHTML: 'MLP Up Weight Matrix',
        className: 'detail-title'
    };
    panel.subtitle = {
        classList: createClassListMock(),
        innerHTML: 'Layer 2 • Multilayer Perceptron',
        className: 'detail-subtitle',
        textContent: ''
    };
    panel.subtitleSecondary = {
        innerHTML: 'Token: hello',
        className: 'detail-subtitle'
    };
    panel.subtitleTertiary = {
        innerHTML: 'Why the model uses it in this layer.',
        className: 'detail-subtitle'
    };
    panel._lastSelection = null;
    panel._lastSelectionLabel = '';
    panel._transformerView2dDetailView = {
        setVisible: vi.fn(),
        open: vi.fn(),
        setSelectionSidebarHeaderContent: vi.fn(),
        isSelectionSidebarVisible: vi.fn(() => true)
    };
    panel._closeTransformerView2dSelectionSidebar = vi.fn();
    panel._showTransformerView2dSelectionSidebar = vi.fn();
    panel._syncMhsaViewRoute = vi.fn();
    panel._setInfoPreview = vi.fn();
    panel._setTitleText = vi.fn();
    panel._setSubtitleSecondaryText = vi.fn();
    panel._setSubtitleTertiaryText = vi.fn();
    panel._setHoverLabelSuppression = vi.fn();
    panel._scheduleResize = vi.fn();
    panel._scheduleSelectionEquationFit = vi.fn();
    panel._scheduleDimensionLabelFit = vi.fn();
    panel._onResize = vi.fn();
    panel._renderPreviewSnapshot = vi.fn();
    panel._startLoop = vi.fn();
    panel._stopLoop = vi.fn();
    panel._setAttentionVisibility = vi.fn();
    panel._setPanelTokenHoverEntry = vi.fn();
    panel._buildHistoryEntry = vi.fn(() => ({ key: 'history-entry' }));
    panel._pushHistoryEntry = vi.fn();
    panel._updateHistoryNavigationControls = vi.fn();
    panel._canToggleMhsaFullscreen = vi.fn(() => false);
    panel._setMhsaFullscreen = vi.fn();
    panel._isSmallScreen = vi.fn(() => false);
    panel.open = vi.fn();
    panel.engine = {
        pause: vi.fn()
    };
    panel.activationSource = { id: 'activation-source' };
    panel.currentPreview = { id: 'preview-root' };
    panel.attentionTokenIndices = [0, 1];
    panel.attentionTokenLabels = ['A', 'B'];
    panel.laneTokenIndices = [0, 1];
    panel.tokenLabels = ['A', 'B'];
    return panel;
}

describe('SelectionPanel transformer-view2d sidebar handoff', () => {
    it('opens the docked 2D selection sidebar when entering the canvas from a selection', () => {
        const panel = createPanelContext();
        const selection = {
            label: 'Post-Softmax Attention Score',
            kind: 'attentionSphere'
        };
        const view2dContext = {
            semanticTarget: {
                componentKind: 'mhsa',
                layerIndex: 1,
                headIndex: 2,
                stage: 'attention',
                role: 'head'
            },
            focusLabel: 'Layer 2 Attention Head 3',
            detailInteractionTargets: [],
            transitionMode: 'staged-head-detail'
        };

        const opened = panel._openTransformerView2dPreview({
            sourceSelection: selection,
            view2dContext,
            syncRoute: false,
            fromHistory: true
        });

        expect(opened).toBe(true);
        expect(panel._transformerView2dDetailView.setVisible).toHaveBeenCalledWith(true);
        expect(panel._transformerView2dDetailView.open).toHaveBeenCalled();
        expect(panel._showTransformerView2dSelectionSidebar).toHaveBeenCalledWith({
            scrollToTop: true
        });
        expect(panel._transformerView2dDetailView.setSelectionSidebarHeaderContent).toHaveBeenCalledWith(
            expect.objectContaining({
                titleHtml: 'MLP Up Weight Matrix',
                subtitleHtml: 'Layer 2 • Multilayer Perceptron',
                subtitleSecondaryHtml: 'Token: hello',
                subtitleTertiaryHtml: 'Why the model uses it in this layer.'
            })
        );
        expect(panel._startLoop).toHaveBeenCalled();
        expect(panel._stopLoop).not.toHaveBeenCalled();
    });

    it('does not auto-open the docked 2D selection sidebar on small screens when entering from a selection', () => {
        const panel = createPanelContext();
        panel._isSmallScreen = vi.fn(() => true);
        panel._transformerView2dDetailView.isSelectionSidebarVisible = vi.fn(() => false);
        const selection = {
            label: 'Post-Softmax Attention Score',
            kind: 'attentionSphere'
        };
        const view2dContext = {
            semanticTarget: {
                componentKind: 'mhsa',
                layerIndex: 1,
                headIndex: 2,
                stage: 'attention',
                role: 'head'
            },
            focusLabel: 'Layer 2 Attention Head 3',
            detailInteractionTargets: [],
            transitionMode: 'staged-head-detail'
        };

        const opened = panel._openTransformerView2dPreview({
            sourceSelection: selection,
            view2dContext,
            syncRoute: false,
            fromHistory: true
        });

        expect(opened).toBe(true);
        expect(panel._transformerView2dDetailView.setVisible).toHaveBeenCalledWith(true);
        expect(panel._transformerView2dDetailView.open).toHaveBeenCalledWith(
            expect.objectContaining({
                isSmallScreen: true
            })
        );
        expect(panel._showTransformerView2dSelectionSidebar).not.toHaveBeenCalled();
        expect(panel._transformerView2dDetailView.setSelectionSidebarHeaderContent).not.toHaveBeenCalled();
        expect(panel._stopLoop).toHaveBeenCalled();
    });

    it('does not force the 2D selection sidebar open for context-only entry without a source selection', () => {
        const panel = createPanelContext();
        const view2dContext = {
            semanticTarget: {
                componentKind: 'embedding',
                stage: 'embedding.token',
                role: 'module'
            },
            focusLabel: 'Token embeddings',
            detailInteractionTargets: [],
            transitionMode: 'staged-focus'
        };

        const opened = panel._openTransformerView2dPreview({
            sourceSelection: null,
            view2dContext,
            syncRoute: false,
            fromHistory: true
        });

        expect(opened).toBe(true);
        expect(panel._showTransformerView2dSelectionSidebar).not.toHaveBeenCalled();
        expect(panel._transformerView2dDetailView.setSelectionSidebarHeaderContent).not.toHaveBeenCalled();
        expect(panel._stopLoop).toHaveBeenCalled();
    });
});
