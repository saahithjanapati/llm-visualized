import { consoleInfo } from './runtimeConsole.js';

function isBrowserWindowAvailable() {
    return typeof window !== 'undefined';
}

const _loggedDebugKeys = new Set();

export function isRandomColorDebugEnabled() {
    return isBrowserWindowAvailable() && window.__RANDOM_COLOR_DEBUG === true;
}

export function isRandomColorDebugVerbose() {
    return isBrowserWindowAvailable() && window.__RANDOM_COLOR_DEBUG_VERBOSE === true;
}

export function shouldAttachRandomColorDebugStack() {
    return isBrowserWindowAvailable() && window.__RANDOM_COLOR_DEBUG_STACK === true;
}

function buildDebugDedupKey(event, details) {
    if (!details || typeof details !== 'object' || Array.isArray(details)) return event;
    const reason = details.reason || '';
    const label = details.label || '';
    const numSubsections = Number.isFinite(details.numSubsections) ? details.numSubsections : '';
    const instanceCount = Number.isFinite(details.instanceCount) ? details.instanceCount : '';
    return `${event}|${reason}|${label}|${numSubsections}|${instanceCount}`;
}

export function logRandomColorDebug(event, details = null) {
    if (!isRandomColorDebugEnabled()) return;

    const dedupKey = buildDebugDedupKey(event, details);
    if (!isRandomColorDebugVerbose() && _loggedDebugKeys.has(dedupKey)) {
        return;
    }
    _loggedDebugKeys.add(dedupKey);

    let payload = details;
    if (details && typeof details === 'object' && !Array.isArray(details)) {
        payload = { ...details };
        if (shouldAttachRandomColorDebugStack()) {
            payload.stack = new Error().stack;
        }
    }

    if (payload !== null && payload !== undefined) {
        consoleInfo(`[RandomColorDebug] ${event}`, payload);
    } else {
        consoleInfo(`[RandomColorDebug] ${event}`);
    }
}
