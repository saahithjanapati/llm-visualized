import { describe, expect, it } from 'vitest';
import {
    isTransformerView2dSelectionSurface,
    normalizeSelectionPanelSurface,
    resolveSelectionPanelSurface,
    SELECTION_PANEL_SURFACES
} from './selectionPanelNavigationSurfaceUtils.js';

describe('selectionPanelNavigationSurfaceUtils', () => {
    it('normalizes recognized panel surfaces', () => {
        expect(normalizeSelectionPanelSurface('panel')).toBe(SELECTION_PANEL_SURFACES.PANEL);
        expect(normalizeSelectionPanelSurface('transformer-view2d')).toBe(
            SELECTION_PANEL_SURFACES.TRANSFORMER_VIEW2D
        );
    });

    it('defaults direct selection hops to the active 2D surface', () => {
        expect(resolveSelectionPanelSurface({
            transformerView2dDetailOpen: true
        })).toBe(SELECTION_PANEL_SURFACES.TRANSFORMER_VIEW2D);
    });

    it('lets explicit history surface override the active 2D surface', () => {
        expect(resolveSelectionPanelSurface({
            selectionSurface: SELECTION_PANEL_SURFACES.PANEL,
            transformerView2dDetailOpen: true
        })).toBe(SELECTION_PANEL_SURFACES.PANEL);
    });

    it('reports whether the resolved surface should preserve the 2D canvas', () => {
        expect(isTransformerView2dSelectionSurface({
            preserveTransformerView2d: true
        })).toBe(true);
        expect(isTransformerView2dSelectionSurface({
            selectionSurface: SELECTION_PANEL_SURFACES.PANEL
        })).toBe(false);
    });
});
