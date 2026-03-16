import { initTouchClickFallback } from './touchClickFallback.js';

export const TRANSFORMER_VIEW2D_TOUCH_ACTION_SELECTOR = [
    '.detail-transformer-view2d-action',
    '.detail-transformer-view2d-selection-sidebar-close'
].join(', ');

export function initTransformerView2dTouchActionFallback(container) {
    return initTouchClickFallback(container, {
        selector: TRANSFORMER_VIEW2D_TOUCH_ACTION_SELECTOR,
        activateOnPointerDownSelector: TRANSFORMER_VIEW2D_TOUCH_ACTION_SELECTOR
    });
}
