export const DESKTOP_SELECTION_PANEL_MIN_WIDTH_PX = 320;
export const DESKTOP_SELECTION_PANEL_MAX_WIDTH_PX = 760;
export const DESKTOP_SELECTION_PANEL_MIN_LEFT_GUTTER_PX = 120;
export const COPY_CONTEXT_BUTTON_MIN_WIDTH_PX = 220;
export const COPY_CONTEXT_BUTTON_MAX_WIDTH_PX = 520;

function clamp(value, min, max) {
    return Math.min(max, Math.max(min, value));
}

function lerp(min, max, t) {
    return min + (max - min) * t;
}

function roundToPrecision(value, precision = 2) {
    const safePrecision = Number.isFinite(precision) ? Math.max(0, Math.floor(precision)) : 2;
    const factor = 10 ** safePrecision;
    return Math.round(value * factor) / factor;
}

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

export function resolveCopyContextButtonLayout(widthPx) {
    const safeWidth = Number.isFinite(widthPx) ? widthPx : COPY_CONTEXT_BUTTON_MAX_WIDTH_PX;
    const range = COPY_CONTEXT_BUTTON_MAX_WIDTH_PX - COPY_CONTEXT_BUTTON_MIN_WIDTH_PX;
    const ratio = range > 0
        ? clamp((safeWidth - COPY_CONTEXT_BUTTON_MIN_WIDTH_PX) / range, 0, 1)
        : 1;

    return {
        fontSizePx: roundToPrecision(lerp(10.25, 12.8, ratio)),
        iconSizePx: roundToPrecision(lerp(11.25, 14, ratio)),
        assistantSizePx: roundToPrecision(lerp(12.5, 15.25, ratio)),
        gapPx: roundToPrecision(lerp(5.5, 8, ratio)),
        paddingInlinePx: roundToPrecision(lerp(9, 12, ratio)),
        paddingBlockPx: roundToPrecision(lerp(7.5, 9, ratio)),
        borderRadiusPx: roundToPrecision(lerp(9, 10, ratio))
    };
}
