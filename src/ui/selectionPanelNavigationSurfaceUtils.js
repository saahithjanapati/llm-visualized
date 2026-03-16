export const SELECTION_PANEL_SURFACES = Object.freeze({
    PANEL: 'panel',
    TRANSFORMER_VIEW2D: 'transformer-view2d'
});

export function normalizeSelectionPanelSurface(
    value = null,
    fallback = SELECTION_PANEL_SURFACES.PANEL
) {
    const normalized = String(value || '').trim().toLowerCase();
    if (normalized === SELECTION_PANEL_SURFACES.TRANSFORMER_VIEW2D) {
        return SELECTION_PANEL_SURFACES.TRANSFORMER_VIEW2D;
    }
    if (normalized === SELECTION_PANEL_SURFACES.PANEL) {
        return SELECTION_PANEL_SURFACES.PANEL;
    }
    return fallback;
}

export function resolveSelectionPanelSurface({
    selectionSurface = null,
    preserveTransformerView2d = false,
    transformerView2dDetailOpen = false
} = {}) {
    const normalizedSelectionSurface = String(selectionSurface || '').trim();
    if (normalizedSelectionSurface.length) {
        return normalizeSelectionPanelSurface(normalizedSelectionSurface);
    }
    return (preserveTransformerView2d || transformerView2dDetailOpen)
        ? SELECTION_PANEL_SURFACES.TRANSFORMER_VIEW2D
        : SELECTION_PANEL_SURFACES.PANEL;
}

export function isTransformerView2dSelectionSurface(options = {}) {
    return resolveSelectionPanelSurface(options) === SELECTION_PANEL_SURFACES.TRANSFORMER_VIEW2D;
}
