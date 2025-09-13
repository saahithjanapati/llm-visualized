export function getPreference(key, defaultValue) {
    try {
        const raw = localStorage.getItem(key);
        return raw === null ? defaultValue : JSON.parse(raw);
    } catch (err) {
        console.warn('getPreference failed:', err);
        return defaultValue;
    }
}

export function setPreference(key, value) {
    try {
        localStorage.setItem(key, JSON.stringify(value));
    } catch (err) {
        console.warn('setPreference failed:', err);
    }
}
