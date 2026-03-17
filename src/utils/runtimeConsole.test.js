// @vitest-environment jsdom

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

function createMockLocalStorage() {
    const store = new Map();
    return {
        getItem(key) {
            return store.has(key) ? store.get(key) : null;
        },
        setItem(key, value) {
            store.set(String(key), String(value));
        },
        removeItem(key) {
            store.delete(String(key));
        },
        clear() {
            store.clear();
        }
    };
}

describe('runtimeConsole', () => {
    let originalConsoleLog;
    let originalConsoleInfo;
    let originalConsoleDebug;

    beforeEach(() => {
        vi.resetModules();
        window.history.replaceState({}, '', '/');
        vi.stubGlobal('localStorage', createMockLocalStorage());
        localStorage.clear();
        delete window.__CONSOLE_LOGGING_ENABLED;
        delete window.__setConsoleLoggingEnabled;

        originalConsoleLog = console.log;
        originalConsoleInfo = console.info;
        originalConsoleDebug = console.debug;

        console.log = vi.fn();
        console.info = vi.fn();
        console.debug = vi.fn();
    });

    afterEach(() => {
        console.log = originalConsoleLog;
        console.info = originalConsoleInfo;
        console.debug = originalConsoleDebug;
        vi.unstubAllGlobals();
    });

    it('suppresses log, info, and debug output when disabled', async () => {
        localStorage.setItem('consoleLoggingEnabled', JSON.stringify(false));

        const runtimeConsole = await import('./runtimeConsole.js');
        runtimeConsole.consoleLog('hidden-log');
        runtimeConsole.consoleInfo('hidden-info');
        runtimeConsole.consoleDebug('hidden-debug');

        expect(runtimeConsole.isConsoleLoggingEnabled()).toBe(false);
        expect(window.__CONSOLE_LOGGING_ENABLED).toBe(false);
        expect(console.log).not.toHaveBeenCalled();
        expect(console.info).not.toHaveBeenCalled();
        expect(console.debug).not.toHaveBeenCalled();
    });

    it('emits console output and persists toggle changes when enabled', async () => {
        localStorage.setItem('consoleLoggingEnabled', JSON.stringify(true));

        const runtimeConsole = await import('./runtimeConsole.js');
        runtimeConsole.consoleLog('visible-log');

        expect(runtimeConsole.isConsoleLoggingEnabled()).toBe(true);
        expect(window.__CONSOLE_LOGGING_ENABLED).toBe(true);
        expect(console.log).toHaveBeenCalledWith('visible-log');
        expect(typeof window.__setConsoleLoggingEnabled).toBe('function');

        runtimeConsole.setConsoleLoggingEnabled(false);
        runtimeConsole.consoleInfo('hidden-after-toggle');

        expect(JSON.parse(localStorage.getItem('consoleLoggingEnabled'))).toBe(false);
        expect(console.info).not.toHaveBeenCalled();
    });

    it('lets the URL override console logging for the current session', async () => {
        localStorage.setItem('consoleLoggingEnabled', JSON.stringify(false));
        window.history.replaceState({}, '', '/?logs=1');

        const runtimeConsole = await import('./runtimeConsole.js');
        runtimeConsole.consoleInfo('url-enabled');

        expect(runtimeConsole.isConsoleLoggingEnabled()).toBe(true);
        expect(window.__CONSOLE_LOGGING_ENABLED).toBe(true);
        expect(console.info).toHaveBeenCalledWith('url-enabled');
        expect(JSON.parse(localStorage.getItem('consoleLoggingEnabled'))).toBe(false);
    });

    it('supports toggling logs at runtime from the browser console helper', async () => {
        localStorage.setItem('consoleLoggingEnabled', JSON.stringify(false));

        const runtimeConsole = await import('./runtimeConsole.js');
        expect(runtimeConsole.isConsoleLoggingEnabled()).toBe(false);

        window.__setConsoleLoggingEnabled(true, { persist: false });
        runtimeConsole.consoleInfo('runtime-enabled');

        expect(runtimeConsole.isConsoleLoggingEnabled()).toBe(true);
        expect(window.__CONSOLE_LOGGING_ENABLED).toBe(true);
        expect(console.info).toHaveBeenCalledWith('runtime-enabled');
        expect(localStorage.getItem('consoleLoggingEnabled')).toBe(JSON.stringify(false));
    });
});
