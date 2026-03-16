export const KV_CACHE_MODE_CHANGED_EVENT = 'kvCacheModeChanged';
export const KV_CACHE_MODE_STATE_SYNC_EVENT = 'kvCacheModeStateSync';

export function dispatchKvCacheModeChanged({
    enabled = false,
    previousEnabled = false
} = {}) {
    if (typeof window === 'undefined' || typeof window.dispatchEvent !== 'function') return false;
    window.dispatchEvent(new CustomEvent(KV_CACHE_MODE_CHANGED_EVENT, {
        detail: {
            enabled: !!enabled,
            previousEnabled: !!previousEnabled
        }
    }));
    return true;
}

export function dispatchKvCacheModeStateSync(enabled = false) {
    if (typeof window === 'undefined' || typeof window.dispatchEvent !== 'function') return false;
    window.dispatchEvent(new CustomEvent(KV_CACHE_MODE_STATE_SYNC_EVENT, {
        detail: {
            enabled: !!enabled
        }
    }));
    return true;
}
