import { beforeAll, describe, expect, it, vi } from 'vitest';
import { SELECTION_PANEL_SURFACES } from './selectionPanelNavigationSurfaceUtils.js';

let SelectionPanel;

beforeAll(async () => {
    vi.stubGlobal('localStorage', {
        getItem: () => null,
        setItem: () => {},
        removeItem: () => {}
    });
    ({ SelectionPanel } = await import('./selectionPanel.js'));
});

function createPanelContext() {
    const panel = Object.create(SelectionPanel.prototype);
    panel.panel = {
        querySelectorAll: () => []
    };
    panel._transformerView2dDetailOpen = false;
    panel._historyEntries = [];
    panel._historyIndex = -1;
    panel._updateHistoryNavigationControls = vi.fn();
    return panel;
}

describe('SelectionPanel history navigation surfaces', () => {
    it('treats the same selection as distinct history entries across 3D and 2D surfaces', () => {
        const panel = createPanelContext();
        const selection = {
            label: 'Token: hello',
            kind: 'label'
        };

        const panelEntry = panel._buildHistoryEntry('selection', selection, {
            selectionSurface: SELECTION_PANEL_SURFACES.PANEL
        });
        const view2dEntry = panel._buildHistoryEntry('selection', selection, {
            selectionSurface: SELECTION_PANEL_SURFACES.TRANSFORMER_VIEW2D
        });

        expect(panelEntry?.key).not.toBe(view2dEntry?.key);
        expect(panel._historyEntriesEqual(panelEntry, view2dEntry)).toBe(false);
    });

    it('replays selection history on the stored 2D surface', () => {
        const panel = createPanelContext();
        panel.showSelection = vi.fn();

        const selection = {
            label: 'Token: hello',
            kind: 'label'
        };

        const applied = panel._applyHistoryEntry({
            type: 'selection',
            selection,
            selectionSurface: SELECTION_PANEL_SURFACES.TRANSFORMER_VIEW2D
        });

        expect(applied).toBe(true);
        expect(panel.showSelection).toHaveBeenCalledWith(selection, expect.objectContaining({
            fromHistory: true,
            selectionSurface: SELECTION_PANEL_SURFACES.TRANSFORMER_VIEW2D,
            preserveTransformerView2d: true
        }));
    });
});
