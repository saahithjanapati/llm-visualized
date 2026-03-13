export const DEFAULT_MODAL_REOPEN_GUARD_MS = 450;

function defaultNow() {
    if (typeof performance !== 'undefined' && typeof performance.now === 'function') {
        return performance.now();
    }
    return Date.now();
}

export function createModalReopenGuard({
    cooldownMs = DEFAULT_MODAL_REOPEN_GUARD_MS,
    now = defaultNow
} = {}) {
    let suppressNextOpenUntil = 0;
    let suppressNextOpen = false;

    return {
        markClosed() {
            suppressNextOpenUntil = now() + cooldownMs;
            suppressNextOpen = true;
        },

        shouldAllowOpen() {
            if (!suppressNextOpen) return true;
            if (now() >= suppressNextOpenUntil) {
                suppressNextOpen = false;
                return true;
            }
            suppressNextOpen = false;
            return false;
        }
    };
}
