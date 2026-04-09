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

function createMockParentNode(name = 'parent') {
    return {
        name,
        children: [],
        appendChild(node) {
            if (!node) return node;
            if (node.nodeType === 11 && Array.isArray(node.childNodes)) {
                const nodes = node.childNodes.slice();
                node.childNodes.length = 0;
                nodes.forEach((child) => this.appendChild(child));
                return node;
            }
            if (node.parentNode && node.parentNode !== this && typeof node.parentNode.removeChild === 'function') {
                node.parentNode.removeChild(node);
            }
            if (!this.children.includes(node)) {
                this.children.push(node);
            }
            node.parentNode = this;
            return node;
        },
        insertBefore(node, referenceNode) {
            if (!referenceNode) {
                return this.appendChild(node);
            }
            if (node.parentNode && node.parentNode !== this && typeof node.parentNode.removeChild === 'function') {
                node.parentNode.removeChild(node);
            }
            const referenceIndex = this.children.indexOf(referenceNode);
            if (referenceIndex === -1) {
                this.children.push(node);
            } else {
                const existingIndex = this.children.indexOf(node);
                if (existingIndex !== -1) {
                    this.children.splice(existingIndex, 1);
                }
                this.children.splice(referenceIndex, 0, node);
            }
            node.parentNode = this;
            return node;
        },
        replaceChild(nextNode, prevNode) {
            const prevIndex = this.children.indexOf(prevNode);
            if (prevIndex === -1) {
                return this.appendChild(nextNode);
            }
            if (nextNode.parentNode && nextNode.parentNode !== this && typeof nextNode.parentNode.removeChild === 'function') {
                nextNode.parentNode.removeChild(nextNode);
            }
            this.children.splice(prevIndex, 1, nextNode);
            nextNode.parentNode = this;
            if (prevNode) {
                prevNode.parentNode = null;
            }
            return prevNode;
        },
        removeChild(node) {
            const index = this.children.indexOf(node);
            if (index !== -1) {
                this.children.splice(index, 1);
                node.parentNode = null;
            }
            return node;
        }
    };
}

function createMockElement(name) {
    return {
        name,
        parentNode: null
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
    panel.transformerView2dActionRow = {
        hidden: false,
        setAttribute: vi.fn()
    };
    panel.transformerView2dActionBtn = {
        hidden: false,
        dataset: {},
        setAttribute: vi.fn(),
        removeAttribute: vi.fn(),
        title: ''
    };
    panel.transformerView2dActionBtnLabel = {
        textContent: ''
    };
    panel._lastSelection = null;
    panel._lastSelectionLabel = '';
    panel._transformerView2dDetailView = {
        setVisible: vi.fn(),
        open: vi.fn(),
        setSelectionSidebarHeaderContent: vi.fn(),
        isSelectionSidebarVisible: vi.fn(() => true),
        clearSelectionLock: vi.fn(() => false)
    };
    panel.close = vi.fn();
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
    panel._setAttentionValue = vi.fn();
    panel._applyAttentionDecodeStyling = vi.fn();
    panel._applyCopyContextButtonLayout = vi.fn();
    panel._buildHistoryEntry = vi.fn(() => ({ key: 'history-entry' }));
    panel._pushHistoryEntry = vi.fn();
    panel._updateHistoryNavigationControls = vi.fn();
    panel._resetHistoryNavigation = vi.fn();
    panel._canToggleMhsaFullscreen = vi.fn(() => false);
    panel._setMhsaFullscreen = vi.fn();
    panel._isSmallScreen = vi.fn(() => false);
    panel.open = vi.fn();
    panel._attentionPostAnimQueue = { clear: vi.fn() };
    panel._attentionPostAnimatedRows = { clear: vi.fn() };
    panel._currentTransformerView2dContext = null;
    panel._transformerView2dDetailOpen = false;
    panel._transformerView2dSelectionSidebarDockRecords = [];
    panel._transformerView2dSelectionSidebarDocked = false;
    panel._transformerView2dSelectionSidebarRestoreTimer = null;
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
    it('keeps the dev-mode data section directly above the copy-context row in the docked sidebar order', () => {
        const panel = createPanelContext();
        panel.previewRoot = { id: 'previewRoot' };
        panel.vectorLegend = { id: 'vectorLegend' };
        panel.equationsSection = { id: 'equationsSection' };
        panel.promptContextRow = { id: 'promptContextRow' };
        panel.previewMetaSection = { id: 'previewMetaSection' };
        panel.description = { id: 'description' };
        panel.metaSection = { id: 'metaSection' };
        panel.attentionRoot = { id: 'attentionRoot' };
        panel.dataSection = { id: 'dataSection' };
        panel.copyContextRow = { id: 'copyContextRow' };

        const sections = panel._getTransformerView2dSelectionSidebarSections();

        expect(sections.at(-2)).toBe(panel.dataSection);
        expect(sections.at(-1)).toBe(panel.copyContextRow);
    });

    it('restores docked 2D sidebar sections back to their original DOM slots', () => {
        const panel = createPanelContext();
        const host = createMockParentNode('host');
        const before = createMockElement('before');
        const sectionA = createMockElement('section-a');
        const between = createMockElement('between');
        const sectionB = createMockElement('section-b');
        const after = createMockElement('after');
        host.appendChild(before);
        host.appendChild(sectionA);
        host.appendChild(between);
        host.appendChild(sectionB);
        host.appendChild(after);

        const sidebarBody = createMockParentNode('sidebar');
        panel.previewRoot = sectionA;
        panel.vectorLegend = sectionB;
        panel._transformerView2dDetailView = {
            ...panel._transformerView2dDetailView,
            getSelectionSidebarBody: vi.fn(() => sidebarBody)
        };
        vi.stubGlobal('document', {
            createComment: () => ({ nodeType: 8, parentNode: null }),
            createDocumentFragment: () => ({ nodeType: 11, childNodes: [], appendChild(node) { this.childNodes.push(node); return node; } })
        });

        try {
            const docked = panel._dockTransformerView2dSelectionSidebarSections();

            expect(docked).toBe(true);
            expect(sidebarBody.children).toEqual([sectionA, sectionB]);
            expect(host.children).toEqual([
                before,
                panel._transformerView2dSelectionSidebarDockRecords[0].placeholder,
                between,
                panel._transformerView2dSelectionSidebarDockRecords[1].placeholder,
                after
            ]);

            const restored = panel._restoreTransformerView2dSelectionSidebarSections();

            expect(restored).toBe(true);
            expect(host.children).toEqual([before, sectionA, between, sectionB, after]);
            expect(sidebarBody.children).toEqual([]);
        } finally {
            vi.unstubAllGlobals();
            vi.stubGlobal('localStorage', {
                getItem: () => null,
                setItem: () => {},
                removeItem: () => {}
            });
        }
    });

    it('schedules hidden 2D view prewarm when the action button becomes available', () => {
        const panel = createPanelContext();
        panel._scheduleTransformerView2dDetailViewPrewarm = vi.fn();

        panel._setTransformerView2dActionButtonState({
            label: 'View in 2D / matrix form',
            ariaLabel: 'View Token embeddings in 2D / matrix form',
            title: 'View Token embeddings in 2D / matrix form'
        });

        expect(panel._scheduleTransformerView2dDetailViewPrewarm).toHaveBeenCalledWith({
            immediate: false
        });
    });

    it('prewarms the hidden 2D detail view off the click path', async () => {
        vi.useFakeTimers();
        vi.stubGlobal('requestAnimationFrame', vi.fn((callback) => setTimeout(() => callback(16), 16)));
        vi.stubGlobal('cancelAnimationFrame', vi.fn((id) => clearTimeout(id)));

        try {
            const panel = createPanelContext();
            panel._currentTransformerView2dContext = {
                semanticTarget: {
                    componentKind: 'embedding',
                    stage: 'embedding.token',
                    role: 'module'
                }
            };
            panel._transformerView2dDetailView = null;
            panel._transformerView2dDetailViewPromise = null;
            panel._transformerView2dDetailViewPrewarmHandle = null;
            panel._ensureTransformerView2dDetailView = vi.fn(() => Promise.resolve({}));

            const scheduled = panel._scheduleTransformerView2dDetailViewPrewarm({
                immediate: false
            });

            expect(scheduled).toBe(true);
            expect(panel._ensureTransformerView2dDetailView).not.toHaveBeenCalled();

            await vi.advanceTimersByTimeAsync(20);

            expect(panel._ensureTransformerView2dDetailView).toHaveBeenCalledTimes(1);
        } finally {
            vi.useRealTimers();
            vi.unstubAllGlobals();
            vi.stubGlobal('localStorage', {
                getItem: () => null,
                setItem: () => {},
                removeItem: () => {}
            });
        }
    });

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
            transitionMode: 'direct'
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
                initialSelectionSidebarVisible: true
            })
        );
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

    it('does not redundantly reopen the docked 2D sidebar when it is already visible', () => {
        const panel = createPanelContext();
        panel._dockTransformerView2dSelectionSidebarSections = vi.fn(() => true);
        panel._syncTransformerView2dSelectionSidebarHeader = vi.fn();
        panel._transformerView2dDetailView = {
            isSelectionSidebarVisible: vi.fn(() => true),
            setSelectionSidebarVisible: vi.fn(),
            scrollSelectionSidebarToTop: vi.fn()
        };

        const shown = SelectionPanel.prototype._showTransformerView2dSelectionSidebar.call(panel, {
            scrollToTop: true
        });

        expect(shown).toBe(true);
        expect(panel._transformerView2dDetailView.setSelectionSidebarVisible).not.toHaveBeenCalled();
        expect(panel._transformerView2dDetailView.scrollSelectionSidebarToTop).toHaveBeenCalledTimes(1);
    });

    it('avoids scheduling an extra resize when opening the docked 2D sidebar while the panel is already open', () => {
        const panel = createPanelContext();
        panel._dockTransformerView2dSelectionSidebarSections = vi.fn(() => true);
        panel._syncTransformerView2dSelectionSidebarHeader = vi.fn();
        panel._transformerView2dDetailView = {
            isSelectionSidebarVisible: vi.fn(() => false),
            setSelectionSidebarVisible: vi.fn(),
            scrollSelectionSidebarToTop: vi.fn()
        };

        const shown = SelectionPanel.prototype._showTransformerView2dSelectionSidebar.call(panel, {
            scrollToTop: true
        });

        expect(shown).toBe(true);
        expect(panel._onResize).toHaveBeenCalledTimes(1);
        expect(panel._renderPreviewSnapshot).toHaveBeenCalledTimes(1);
        expect(panel._startLoop).toHaveBeenCalledTimes(1);
        expect(panel._scheduleResize).not.toHaveBeenCalled();
    });

    it('still schedules a resize when opening the docked 2D sidebar while the panel is not yet open', () => {
        const panel = createPanelContext();
        panel.isOpen = false;
        panel._dockTransformerView2dSelectionSidebarSections = vi.fn(() => true);
        panel._syncTransformerView2dSelectionSidebarHeader = vi.fn();
        panel._transformerView2dDetailView = {
            isSelectionSidebarVisible: vi.fn(() => false),
            setSelectionSidebarVisible: vi.fn(),
            scrollSelectionSidebarToTop: vi.fn()
        };

        const shown = SelectionPanel.prototype._showTransformerView2dSelectionSidebar.call(panel, {
            scrollToTop: true
        });

        expect(shown).toBe(true);
        expect(panel._onResize).not.toHaveBeenCalled();
        expect(panel._renderPreviewSnapshot).not.toHaveBeenCalled();
        expect(panel._startLoop).not.toHaveBeenCalled();
        expect(panel._scheduleResize).toHaveBeenCalledTimes(1);
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
            transitionMode: 'direct'
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
                initialSelectionSidebarVisible: false,
                isSmallScreen: true
            })
        );
        expect(panel._showTransformerView2dSelectionSidebar).not.toHaveBeenCalled();
        expect(panel._transformerView2dDetailView.setSelectionSidebarHeaderContent).not.toHaveBeenCalled();
        expect(panel._stopLoop).toHaveBeenCalled();
    });

    it('passes residual overview lock targets through to the 2D canvas open call', () => {
        const panel = createPanelContext();
        const selection = {
            label: 'Residual Stream Vector',
            kind: 'laneVector'
        };
        const view2dContext = {
            semanticTarget: {
                componentKind: 'residual',
                layerIndex: 1,
                stage: 'incoming',
                role: 'module'
            },
            focusLabel: 'Layer 2 residual stream',
            initialOverviewSelectionLockTarget: {
                semanticTarget: {
                    componentKind: 'residual',
                    layerIndex: 1,
                    stage: 'incoming',
                    role: 'module'
                },
                tokenIndex: 1,
                tokenLabel: 'B'
            },
            detailInteractionTargets: [],
            transitionMode: 'staged-focus'
        };

        const opened = panel._openTransformerView2dPreview({
            sourceSelection: selection,
            view2dContext,
            syncRoute: false,
            fromHistory: true
        });

        expect(opened).toBe(true);
        expect(panel._transformerView2dDetailView.open).toHaveBeenCalledWith(
            expect.objectContaining({
                initialOverviewSelectionLockTarget: view2dContext.initialOverviewSelectionLockTarget
            })
        );
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
        expect(panel._transformerView2dDetailView.open).toHaveBeenCalledWith(
            expect.objectContaining({
                initialSelectionSidebarVisible: false
            })
        );
        expect(panel._transformerView2dDetailView.setSelectionSidebarHeaderContent).not.toHaveBeenCalled();
        expect(panel._stopLoop).toHaveBeenCalled();
    });

    it('does not inherit the active 3D selection when the top-level 2D overview is opened explicitly', () => {
        const panel = createPanelContext();
        panel._lastSelection = {
            label: 'MLP Up Weight Matrix',
            kind: 'mesh'
        };
        panel._lastSelectionLabel = 'MLP Up Weight Matrix';

        const opened = panel.openTransformerView2d({
            semanticTarget: null,
            focusLabel: 'GPT-2 (124M)',
            syncRoute: false
        });

        expect(opened).toBe(true);
        expect(panel._showTransformerView2dSelectionSidebar).not.toHaveBeenCalled();
        expect(panel._transformerView2dDetailView.open).toHaveBeenCalledWith(
            expect.objectContaining({
                semanticTarget: null,
                focusLabel: 'GPT-2 (124M)',
                initialSelectionSidebarVisible: false
            })
        );
        expect(panel._transformerView2dDetailView.setSelectionSidebarHeaderContent).not.toHaveBeenCalled();
        expect(panel._transformerView2dSourceSelection).toBeNull();
    });

    it('refreshes the open 2D canvas in place when panel data changes', () => {
        const panel = createPanelContext();
        panel._transformerView2dDetailOpen = true;
        panel._currentTransformerView2dContext = {
            semanticTarget: {
                componentKind: 'mhsa',
                layerIndex: 3,
                stage: 'attention',
                role: 'module'
            },
            focusLabel: 'Layer 4 attention',
            detailSemanticTargets: [],
            detailFocusLabel: '',
            detailInteractionTargets: [],
            transitionMode: 'staged-focus'
        };

        panel.updateData({
            activationSource: { id: 'next-activation-source' },
            laneTokenIndices: [0, 1, 2],
            tokenLabels: ['A', 'B', 'C'],
            attentionTokenIndices: [0, 1, 2],
            attentionTokenLabels: ['A', 'B', 'C']
        });

        expect(panel._resetHistoryNavigation).toHaveBeenCalled();
        expect(panel._transformerView2dDetailView.open).toHaveBeenCalledWith(
            expect.objectContaining({
                activationSource: { id: 'next-activation-source' },
                tokenIndices: [0, 1, 2],
                tokenLabels: ['A', 'B', 'C'],
                semanticTarget: panel._currentTransformerView2dContext.semanticTarget,
                focusLabel: 'Layer 4 attention',
                initialSelectionSidebarVisible: true
            })
        );
        expect(panel._showTransformerView2dSelectionSidebar).toHaveBeenCalledWith({
            scrollToTop: false
        });
    });

    it('uses the header close action to dismiss the 2D selection sidebar before closing the panel', () => {
        const panel = createPanelContext();
        panel._transformerView2dDetailOpen = true;
        panel._closeTransformerView2dSelectionSidebar = vi.fn(() => true);

        const handled = panel._handleCloseButtonAction({
            preventDefault: vi.fn(),
            stopPropagation: vi.fn(),
            cancelable: true
        });

        expect(handled).toBe(true);
        expect(panel._closeTransformerView2dSelectionSidebar).toHaveBeenCalledWith({
            restoreSections: false,
            restartLoop: false
        });
        expect(panel._transformerView2dDetailView.clearSelectionLock).not.toHaveBeenCalled();
        expect(panel.close).not.toHaveBeenCalled();
    });

    it('uses the header close action to clear the 2D selection lock before closing the panel', () => {
        const panel = createPanelContext();
        panel._transformerView2dDetailOpen = true;
        panel._closeTransformerView2dSelectionSidebar = vi.fn(() => false);
        panel._transformerView2dDetailView.clearSelectionLock = vi.fn(() => true);

        const handled = panel._handleCloseButtonAction({
            preventDefault: vi.fn(),
            stopPropagation: vi.fn(),
            cancelable: true
        });

        expect(handled).toBe(true);
        expect(panel._closeTransformerView2dSelectionSidebar).toHaveBeenCalled();
        expect(panel._transformerView2dDetailView.clearSelectionLock).toHaveBeenCalledWith({
            scheduleRender: true
        });
        expect(panel.close).not.toHaveBeenCalled();
    });
});
