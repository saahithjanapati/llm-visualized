export function hasTransformerView2dLockedDetailSelection(detailScenePinnedFocus = null) {
    return !!(detailScenePinnedFocus && typeof detailScenePinnedFocus === 'object');
}

export const TRANSFORMER_VIEW2D_DETAIL_CLICK_LOCK_ACTIONS = Object.freeze({
    IGNORE: 'ignore',
    CLEAR_LOCK: 'clear-lock',
    LOCK_TARGET: 'lock-target',
    NONE: 'none'
});

export function isTransformerView2dDetailSelectionLockActive(detailSceneSelectionLocked = false) {
    return detailSceneSelectionLocked === true;
}

export function shouldFreezeTransformerView2dDetailHover({
    allowDetailSceneHover = false,
    detailSceneSelectionLocked = false
} = {}) {
    return !!allowDetailSceneHover
        && isTransformerView2dDetailSelectionLockActive(detailSceneSelectionLocked);
}

export function resolveTransformerView2dDetailClickLockAction({
    detailSceneSelectionLocked = false,
    detailScenePinnedSignature = '',
    detailHoverState = null
} = {}) {
    const hasLockedDetailSelection = isTransformerView2dDetailSelectionLockActive(detailSceneSelectionLocked);
    const clickedFocusableDetailTarget = !!detailHoverState?.focusState;
    const clickedSignature = typeof detailHoverState?.signature === 'string'
        ? detailHoverState.signature
        : '';
    const pinnedSignature = typeof detailScenePinnedSignature === 'string'
        ? detailScenePinnedSignature
        : '';
    const clickedLockedDetailTarget = !!(
        clickedFocusableDetailTarget
        && pinnedSignature.length
        && clickedSignature === pinnedSignature
    );

    if (hasLockedDetailSelection) {
        if (clickedFocusableDetailTarget) {
            return clickedLockedDetailTarget
                ? TRANSFORMER_VIEW2D_DETAIL_CLICK_LOCK_ACTIONS.IGNORE
                : TRANSFORMER_VIEW2D_DETAIL_CLICK_LOCK_ACTIONS.LOCK_TARGET;
        }
        return TRANSFORMER_VIEW2D_DETAIL_CLICK_LOCK_ACTIONS.CLEAR_LOCK;
    }

    return clickedFocusableDetailTarget
        ? TRANSFORMER_VIEW2D_DETAIL_CLICK_LOCK_ACTIONS.LOCK_TARGET
        : TRANSFORMER_VIEW2D_DETAIL_CLICK_LOCK_ACTIONS.NONE;
}
