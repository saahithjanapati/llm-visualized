const PAGE_VIEW_EVENT_NAME = 'page_view';
const PAGE_VIEW_TRACKER_KEY = '__llmVisualizedGoogleAnalyticsPageTracker';
const LAST_TRACKED_LOCATION_KEY = '__llmVisualizedGoogleAnalyticsLastTrackedLocation';

function resolveUrlLike(urlLike = null) {
    let candidate = urlLike;
    if (candidate == null) {
        if (typeof window === 'undefined') return null;
        candidate = window.location?.href || '';
    } else if (candidate instanceof URL) {
        return new URL(candidate.toString());
    } else if (typeof candidate === 'object') {
        if (typeof candidate.href === 'string' && candidate.href.length) {
            candidate = candidate.href;
        } else {
            candidate = `${candidate.pathname || ''}${candidate.search || ''}${candidate.hash || ''}`;
        }
    }

    const rawUrl = String(candidate || '').trim();
    if (!rawUrl.length) return null;

    const baseUrl = (typeof window !== 'undefined' && typeof window.location?.origin === 'string')
        ? window.location.origin
        : 'https://llm-visualized.local';

    try {
        return new URL(rawUrl, baseUrl);
    } catch (_) {
        return null;
    }
}

function readWindowValue(key) {
    if (typeof window === 'undefined') return null;
    const value = window[key];
    return typeof value === 'string' && value.length ? value : null;
}

function writeWindowValue(key, value) {
    if (typeof window === 'undefined') return;
    if (typeof value === 'string' && value.length) {
        window[key] = value;
        return;
    }
    delete window[key];
}

function resolveDocumentReferrer() {
    if (typeof document === 'undefined') return null;
    const referrer = String(document.referrer || '').trim();
    return referrer.length ? referrer : null;
}

function buildPageViewPayload(url, referrer = null) {
    const queryString = url.search.startsWith('?') ? url.search.slice(1) : '';
    const queryKeys = Array.from(new Set(Array.from(url.searchParams.keys())));
    const payload = {
        page_location: url.toString(),
        page_title: typeof document === 'undefined' ? '' : document.title
    };

    if (referrer) {
        payload.page_referrer = referrer;
    }
    if (queryString) {
        payload.page_query_string = queryString;
    }
    if (queryKeys.length) {
        payload.page_query_keys = queryKeys.join(',');
    }
    if (url.searchParams.has('token')) {
        payload.route_token = url.searchParams.get('token');
    }
    if (url.searchParams.has('generation')) {
        payload.route_generation = url.searchParams.get('generation');
    }
    if (url.searchParams.has('kvCache')) {
        payload.route_kv_cache = url.searchParams.get('kvCache');
    }
    if (url.searchParams.has('view')) {
        payload.route_view = url.searchParams.get('view');
    }

    return payload;
}

function resolvePageReferrer(explicitReferrer = null) {
    if (typeof explicitReferrer === 'string' && explicitReferrer.trim().length) {
        return explicitReferrer.trim();
    }
    return readWindowValue(LAST_TRACKED_LOCATION_KEY) || resolveDocumentReferrer();
}

export function trackGoogleAnalyticsPageView(urlLike = null, {
    referrer = null,
    skipDuplicate = true
} = {}) {
    if (typeof window === 'undefined' || typeof window.gtag !== 'function') return false;

    const url = resolveUrlLike(urlLike);
    if (!url) return false;

    const pageLocation = url.toString();
    const lastTrackedLocation = readWindowValue(LAST_TRACKED_LOCATION_KEY);
    if (skipDuplicate && lastTrackedLocation === pageLocation) {
        return false;
    }

    window.gtag(
        'event',
        PAGE_VIEW_EVENT_NAME,
        buildPageViewPayload(url, resolvePageReferrer(referrer))
    );
    writeWindowValue(LAST_TRACKED_LOCATION_KEY, pageLocation);
    return true;
}

function createTrackedHistoryMethod(originalMethod) {
    return function trackedHistoryMethod(...args) {
        const previousLocation = typeof window === 'undefined' ? null : window.location?.href || null;
        const result = originalMethod(...args);
        trackGoogleAnalyticsPageView(window.location, { referrer: previousLocation });
        return result;
    };
}

export function initGoogleAnalyticsPageTracking({
    trackInitialPage = true
} = {}) {
    if (typeof window === 'undefined' || typeof window.history !== 'object') return null;

    const existingTracker = window[PAGE_VIEW_TRACKER_KEY];
    if (existingTracker && typeof existingTracker.trackCurrentPageView === 'function') {
        if (trackInitialPage) {
            existingTracker.trackCurrentPageView();
        }
        return existingTracker;
    }

    const originalPushState = window.history.pushState.bind(window.history);
    const originalReplaceState = window.history.replaceState.bind(window.history);
    const handlePopState = () => {
        trackGoogleAnalyticsPageView(window.location);
    };

    window.history.pushState = createTrackedHistoryMethod(originalPushState);
    window.history.replaceState = createTrackedHistoryMethod(originalReplaceState);
    window.addEventListener('popstate', handlePopState);

    const tracker = {
        trackCurrentPageView: (options = {}) => trackGoogleAnalyticsPageView(window.location, options),
        dispose: () => {
            window.history.pushState = originalPushState;
            window.history.replaceState = originalReplaceState;
            window.removeEventListener('popstate', handlePopState);
            if (window[PAGE_VIEW_TRACKER_KEY] === tracker) {
                delete window[PAGE_VIEW_TRACKER_KEY];
            }
        }
    };

    window[PAGE_VIEW_TRACKER_KEY] = tracker;

    if (trackInitialPage) {
        tracker.trackCurrentPageView();
    }

    return tracker;
}

export default initGoogleAnalyticsPageTracking;
