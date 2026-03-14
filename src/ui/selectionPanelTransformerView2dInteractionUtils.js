const VIEW2D_DEFAULT_CLICK_SLOP_PX = 6;
const VIEW2D_TOUCH_CLICK_SLOP_PX = 2;

export function resolveView2dClickSlopPx(pointerType = '') {
    return String(pointerType || '').toLowerCase() === 'touch'
        ? VIEW2D_TOUCH_CLICK_SLOP_PX
        : VIEW2D_DEFAULT_CLICK_SLOP_PX;
}

export function hasView2dPointerExceededClickSlop({
    pointerType = '',
    startClientX = 0,
    startClientY = 0,
    clientX = startClientX,
    clientY = startClientY
} = {}) {
    if (
        !Number.isFinite(startClientX)
        || !Number.isFinite(startClientY)
        || !Number.isFinite(clientX)
        || !Number.isFinite(clientY)
    ) {
        return false;
    }

    return Math.hypot(
        clientX - startClientX,
        clientY - startClientY
    ) >= resolveView2dClickSlopPx(pointerType);
}
