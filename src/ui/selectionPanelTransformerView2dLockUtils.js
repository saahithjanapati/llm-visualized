export function hasTransformerView2dLockedDetailSelection(detailScenePinnedFocus = null) {
    return !!(detailScenePinnedFocus && typeof detailScenePinnedFocus === 'object');
}

export function shouldFreezeTransformerView2dDetailHover({
    allowDetailSceneHover = false,
    detailScenePinnedFocus = null
} = {}) {
    return !!allowDetailSceneHover
        && hasTransformerView2dLockedDetailSelection(detailScenePinnedFocus);
}
