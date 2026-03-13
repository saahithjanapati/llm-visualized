export function resolveKvPrefillBaseLaneCount({ initialLaneCount = 1 } = {}) {
    void initialLaneCount;
    // Prefill only represents the first-token cache build. If the scene starts
    // on a later token window, KV mode should immediately use decode semantics.
    return 1;
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
