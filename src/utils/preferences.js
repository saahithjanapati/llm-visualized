export function getPreference(key, defaultValue) {
    try {
        const raw = localStorage.getItem(key);
        return raw === null ? defaultValue : JSON.parse(raw);
    } catch (_) {
        return defaultValue;
    }
}

export function setPreference(key, value) {
    try {
        localStorage.setItem(key, JSON.stringify(value));
    } catch (_) {
        // no-op
    }
}
