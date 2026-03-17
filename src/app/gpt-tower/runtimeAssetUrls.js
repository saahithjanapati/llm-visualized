import captureUrl from '../../assets/captures/capture.json?url';
import capture2Url from '../../assets/captures/capture_2.json?url';
import precomputedComponentsQkvUrl from '../../../precomputed_components_qkv.glb?url';
import precomputedComponentsSliceUrl from '../../../precomputed_components_slice.glb?url';

export const DEFAULT_CAPTURE_FILE = 'capture.json';
export const PRECOMPUTED_COMPONENTS_QKV_URL = precomputedComponentsQkvUrl;
export const PRECOMPUTED_COMPONENTS_SLICE_URL = precomputedComponentsSliceUrl;

const LOCAL_CAPTURE_URL_BASE = 'https://llm-visualized.local';
const CAPTURE_URL_BY_FILE = new Map([
    ['capture.json', captureUrl],
    ['capture_2.json', capture2Url]
]);

export function resolveCaptureAssetUrl(captureFile = DEFAULT_CAPTURE_FILE) {
    const rawValue = String(captureFile || DEFAULT_CAPTURE_FILE).trim();
    if (!rawValue) {
        return captureUrl;
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

    const hashedUrl = new URL(mappedUrl, LOCAL_CAPTURE_URL_BASE);
    hashedUrl.search = localUrl.search;
    hashedUrl.hash = localUrl.hash;
    return `${hashedUrl.pathname}${hashedUrl.search}${hashedUrl.hash}`;
}
