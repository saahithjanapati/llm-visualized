import precomputedComponentsSliceUrl from '../../../precomputed_components_slice.glb?url';
import { LOCAL_CAPTURE_ASSET_VERSION_BY_FILE } from './captureAssetManifest.generated.js';

export const DEFAULT_CAPTURE_FILE = 'capture.json';
export const PRECOMPUTED_COMPONENTS_SLICE_URL = precomputedComponentsSliceUrl;
const DEFAULT_CAPTURE_PATH = '/capture.json';
const CAPTURE_VERSION_QUERY_PARAM = 'v';

const LOCAL_CAPTURE_URL_BASE = 'https://llm-visualized.local';
const CAPTURE_URL_BY_FILE = new Map(
    Object.keys(LOCAL_CAPTURE_ASSET_VERSION_BY_FILE)
        .sort((left, right) => left.localeCompare(right))
        .map((fileName) => [fileName, `/${fileName}`])
);

function appendCaptureVersion(localUrl, version) {
    if (!version) return localUrl;
    if (localUrl.searchParams.has(CAPTURE_VERSION_QUERY_PARAM)) return localUrl;
    localUrl.searchParams.set(CAPTURE_VERSION_QUERY_PARAM, version);
    return localUrl;
}

export function resolveCaptureAssetUrl(captureFile = DEFAULT_CAPTURE_FILE) {
    const rawValue = String(captureFile || DEFAULT_CAPTURE_FILE).trim();
    if (!rawValue) {
        return DEFAULT_CAPTURE_PATH;
    }

    try {
        const externalUrl = new URL(rawValue);
        if (externalUrl.protocol === 'http:' || externalUrl.protocol === 'https:') {
            return externalUrl.toString();
        }
    } catch (_) {
        // Fall through to local-path handling below.
    }

    const localUrl = new URL(rawValue.startsWith('/') ? rawValue : `/${rawValue}`, LOCAL_CAPTURE_URL_BASE);
    const captureFileKey = localUrl.pathname.replace(/^\/+/, '');
    const mappedUrl = CAPTURE_URL_BY_FILE.get(captureFileKey);
    if (!mappedUrl) {
        return `${localUrl.pathname}${localUrl.search}${localUrl.hash}`;
    }

    const captureUrl = new URL(mappedUrl, LOCAL_CAPTURE_URL_BASE);
    captureUrl.search = localUrl.search;
    captureUrl.hash = localUrl.hash;
    appendCaptureVersion(captureUrl, LOCAL_CAPTURE_ASSET_VERSION_BY_FILE[captureFileKey] || '');
    return `${captureUrl.pathname}${captureUrl.search}${captureUrl.hash}`;
}
