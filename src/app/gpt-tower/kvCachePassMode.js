export function resolveKvPrefillBaseLaneCount({
    initialLaneCount = 1,
    baseLaneCount = initialLaneCount
} = {}) {
    // Prefill should cover the prompt/base token window for this generation
    // session. If the scene opens on a later token window, comparing against
    // that base still resolves the current pass as decode.
    return Math.max(1, Math.floor(baseLaneCount || initialLaneCount || 1));
}

export function resolveKvCachePassMode({
    laneCount,
    kvModeEnabled = false,
    prefillBaseLaneCount = 1
} = {}) {
    const totalLaneCount = Math.max(1, Math.floor(laneCount || 1));
    const safePrefillBaseLaneCount = Math.max(1, Math.floor(prefillBaseLaneCount || 1));
    const passIndex = Math.max(0, totalLaneCount - safePrefillBaseLaneCount);
    const kvCachePrefillActive = !!(kvModeEnabled && passIndex === 0);
    const kvCacheDecodeActive = !!(kvModeEnabled && passIndex > 0);

    return {
        totalLaneCount,
        passIndex,
        kvCachePrefillActive,
        kvCacheDecodeActive,
        activeLaneCount: kvCacheDecodeActive ? 1 : totalLaneCount
    };
}
