import { getPreference, setPreference } from './preferences.js';

export const CONSOLE_LOGGING_PREF_KEY = 'consoleLoggingEnabled';

const ENABLED_QUERY_VALUES = new Set(['', '1', 'true', 'yes', 'on']);
const DISABLED_QUERY_VALUES = new Set(['0', 'false', 'no', 'off']);

let consoleLoggingEnabled = false;
let consoleLoggingInitialized = false;

function resolveDefaultConsoleLoggingEnabled() {
    return !!import.meta.env?.DEV;
}

function resolveConsoleLoggingUrlOverride() {
    if (typeof window === 'undefined') return null;
    const params = new URLSearchParams(window.location.search);
    const queryKeys = ['consoleLogs', 'logs'];
    for (let i = 0; i < queryKeys.length; i++) {
        const rawValue = params.get(queryKeys[i]);
        if (rawValue === null) continue;
        const normalized = String(rawValue).trim().toLowerCase();
        if (ENABLED_QUERY_VALUES.has(normalized)) return true;
        if (DISABLED_QUERY_VALUES.has(normalized)) return false;
    }
    return null;
}

function syncConsoleLoggingFlag() {
    const root = typeof window !== 'undefined' ? window : globalThis;
    if (!root) return;
    root.__CONSOLE_LOGGING_ENABLED = consoleLoggingEnabled;
    root.__setConsoleLoggingEnabled = (enabled, options = {}) => (
        setConsoleLoggingEnabled(enabled, options)
    );
}

export function resolveInitialConsoleLoggingEnabled() {
    const urlOverride = resolveConsoleLoggingUrlOverride();
    if (typeof urlOverride === 'boolean') {
        return urlOverride;
    }
    return !!getPreference(CONSOLE_LOGGING_PREF_KEY, resolveDefaultConsoleLoggingEnabled());
}

export function initializeConsoleLogging() {
    if (consoleLoggingInitialized) return consoleLoggingEnabled;
    consoleLoggingEnabled = resolveInitialConsoleLoggingEnabled();
    consoleLoggingInitialized = true;
    syncConsoleLoggingFlag();
    return consoleLoggingEnabled;
}

export function isConsoleLoggingEnabled() {
    if (!consoleLoggingInitialized) {
        initializeConsoleLogging();
    }
    return consoleLoggingEnabled;
}

export function setConsoleLoggingEnabled(enabled, { persist = true } = {}) {
    consoleLoggingEnabled = !!enabled;
    consoleLoggingInitialized = true;
    syncConsoleLoggingFlag();
    if (persist) {
        setPreference(CONSOLE_LOGGING_PREF_KEY, consoleLoggingEnabled);
    }
    return consoleLoggingEnabled;
}

export function consoleLog(...args) {
    if (!isConsoleLoggingEnabled()) return;
    console.log(...args);
}

export function consoleInfo(...args) {
    if (!isConsoleLoggingEnabled()) return;
    console.info(...args);
}

export function consoleDebug(...args) {
    if (!isConsoleLoggingEnabled()) return;
    console.debug(...args);
}

export function consoleGroupCollapsed(...args) {
    if (!isConsoleLoggingEnabled()) return false;
    if (typeof console.groupCollapsed === 'function') {
        console.groupCollapsed(...args);
        return true;
    }
    console.log(...args);
    return false;
}

export function consoleGroupEnd() {
    if (!isConsoleLoggingEnabled()) return;
    if (typeof console.groupEnd === 'function') {
        console.groupEnd();
    }
}

initializeConsoleLogging();
