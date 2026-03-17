import { TRANSFORMER_VIEW2D_PANEL_ACTION_OPEN } from './selectionPanelTransformerView2d.js';

const DEFAULT_TRANSFORMER_VIEW2D_ACTION_LABEL = 'View in 2D / matrix form';

export function resolveSelectionPrimaryActionConfig({
    view2dContext = null,
    isSmallScreen = false
} = {}) {
    if (!view2dContext) return null;
    const focusLabel = String(view2dContext?.focusLabel || '').trim();
    const actionLabel = focusLabel
        ? `View ${focusLabel} in 2D / matrix form`
        : DEFAULT_TRANSFORMER_VIEW2D_ACTION_LABEL;
    return {
        action: TRANSFORMER_VIEW2D_PANEL_ACTION_OPEN,
        label: DEFAULT_TRANSFORMER_VIEW2D_ACTION_LABEL,
        ariaLabel: actionLabel,
        title: actionLabel
    };
}
