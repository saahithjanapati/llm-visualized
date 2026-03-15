const VIEW2D_DEFAULT_CLICK_SLOP_PX = 6;
const VIEW2D_TOUCH_CLICK_SLOP_PX = 2;

export function isView2dTouchPointerType(pointerType = '') {
    return String(pointerType || '').toLowerCase() === 'touch';
}

export function resolveView2dClickSlopPx(pointerType = '') {
    return isView2dTouchPointerType(pointerType)
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

export function resolveView2dPointerMoveIntent({
    pointerType = '',
    startClientX = 0,
    startClientY = 0,
    previousClientX = startClientX,
    previousClientY = startClientY,
    clientX = previousClientX,
    clientY = previousClientY,
    moved = false,
    suppressClick = false
} = {}) {
    const deltaX = (
        Number.isFinite(clientX) && Number.isFinite(previousClientX)
            ? clientX - previousClientX
            : 0
    );
    const deltaY = (
        Number.isFinite(clientY) && Number.isFinite(previousClientY)
            ? clientY - previousClientY
            : 0
    );
    const nextMoved = moved === true || hasView2dPointerExceededClickSlop({
        pointerType,
        startClientX,
        startClientY,
        clientX,
        clientY
    });
    const hasDelta = Math.abs(deltaX) > 0 || Math.abs(deltaY) > 0;
    const shouldPan = !isView2dTouchPointerType(pointerType) || nextMoved;

    return {
        deltaX,
        deltaY,
        moved: nextMoved,
        suppressClick: suppressClick === true || (shouldPan && hasDelta),
        shouldPan
    };
}

export function shouldTreatView2dPointerReleaseAsClick({
    moved = false,
    suppressClick = false
} = {}) {
    return moved !== true && suppressClick !== true;
}

export function shouldSuppressView2dDoubleClickFocus({
    headDetailDepthActive = false,
    hasActiveDetailTarget = false,
    hasDetailSceneIndex = false
} = {}) {
    return headDetailDepthActive === true
        && hasActiveDetailTarget === true
        && hasDetailSceneIndex === true;
}
