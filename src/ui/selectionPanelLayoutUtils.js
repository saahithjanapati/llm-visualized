export const DESKTOP_SELECTION_PANEL_MIN_WIDTH_PX = 320;
export const DESKTOP_SELECTION_PANEL_MAX_WIDTH_PX = 760;
export const DESKTOP_SELECTION_PANEL_MIN_LEFT_GUTTER_PX = 120;

export function resolveDesktopSelectionPanelWidthBounds({
    viewportWidth = 0,
    minWidthPx = DESKTOP_SELECTION_PANEL_MIN_WIDTH_PX,
    maxWidthPx = DESKTOP_SELECTION_PANEL_MAX_WIDTH_PX,
    minLeftGutterPx = DESKTOP_SELECTION_PANEL_MIN_LEFT_GUTTER_PX
} = {}) {
    const safeViewportWidth = Number.isFinite(viewportWidth) ? viewportWidth : 0;
    const safeMinWidth = Math.max(240, Number.isFinite(minWidthPx) ? minWidthPx : DESKTOP_SELECTION_PANEL_MIN_WIDTH_PX);
    const safeMaxWidth = Math.max(safeMinWidth, Number.isFinite(maxWidthPx) ? maxWidthPx : DESKTOP_SELECTION_PANEL_MAX_WIDTH_PX);
    const safeLeftGutter = Math.max(24, Number.isFinite(minLeftGutterPx) ? minLeftGutterPx : DESKTOP_SELECTION_PANEL_MIN_LEFT_GUTTER_PX);

    if (!(safeViewportWidth > 0)) {
        return {
            minWidthPx: safeMinWidth,
            maxWidthPx: safeMaxWidth
        };
    }

    const viewportMaxWidth = Math.max(240, safeViewportWidth - safeLeftGutter);
    const clampedMaxWidth = Math.min(safeMaxWidth, viewportMaxWidth);
    const clampedMinWidth = Math.min(safeMinWidth, clampedMaxWidth);

    return {
        minWidthPx: clampedMinWidth,
        maxWidthPx: clampedMaxWidth
    };
}

export function clampDesktopSelectionPanelWidth(widthPx, options = {}) {
    const bounds = resolveDesktopSelectionPanelWidthBounds(options);
    const fallback = Number.isFinite(widthPx) ? widthPx : bounds.minWidthPx;
    return Math.min(bounds.maxWidthPx, Math.max(bounds.minWidthPx, fallback));
}
