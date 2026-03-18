export const PROJECT_INFO_PAGE_PATH = '/info/';
export const PROJECT_INFO_RETURN_TO_PARAM = 'returnTo';
export const PROJECT_INFO_DEFAULT_RETURN_PATH = '/';

function resolveLocationSnapshot(locationRef = null) {
    const candidate = locationRef || (typeof window !== 'undefined' ? window.location : null);
    if (!candidate) return null;

    const pathname = String(candidate.pathname || '').trim() || '/';
    const search = String(candidate.search || '');
    const hash = String(candidate.hash || '');
    const href = String(candidate.href || `${pathname}${search}${hash}`).trim();

    return {
        pathname,
        search,
        hash,
        href
    };
}

function resolveUrlLike(value = '') {
    const rawValue = String(value || '').trim();
    if (!rawValue.length) return null;
    try {
        return new URL(rawValue, 'https://llm-visualized.local');
    } catch (_) {
        return null;
    }
}

function normalizeProjectInfoReturnPath(value = '') {
    const url = resolveUrlLike(value);
    if (!url) return PROJECT_INFO_DEFAULT_RETURN_PATH;
    if (url.origin !== 'https://llm-visualized.local') {
        return PROJECT_INFO_DEFAULT_RETURN_PATH;
    }

    const normalizedPath = `${url.pathname}${url.search}${url.hash}` || PROJECT_INFO_DEFAULT_RETURN_PATH;
    if (normalizedPath === PROJECT_INFO_PAGE_PATH || normalizedPath.startsWith(`${PROJECT_INFO_PAGE_PATH}?`)) {
        return PROJECT_INFO_DEFAULT_RETURN_PATH;
    }
    if (!normalizedPath.startsWith('/') || normalizedPath.startsWith('//')) {
        return PROJECT_INFO_DEFAULT_RETURN_PATH;
    }
    return normalizedPath;
}

export function buildProjectInfoPageUrl(locationRef = null) {
    const currentLocation = resolveLocationSnapshot(locationRef);
    const infoUrl = new URL(PROJECT_INFO_PAGE_PATH, 'https://llm-visualized.local');
    if (currentLocation) {
        const returnPath = normalizeProjectInfoReturnPath(
            `${currentLocation.pathname}${currentLocation.search}${currentLocation.hash}`
        );
        if (returnPath !== PROJECT_INFO_DEFAULT_RETURN_PATH) {
            infoUrl.searchParams.set(PROJECT_INFO_RETURN_TO_PARAM, returnPath);
        }
    }
    return `${infoUrl.pathname}${infoUrl.search}${infoUrl.hash}`;
}

export function resolveProjectInfoBackHref(locationRef = null, {
    defaultHref = PROJECT_INFO_DEFAULT_RETURN_PATH
} = {}) {
    const currentLocation = resolveLocationSnapshot(locationRef);
    const normalizedFallbackHref = normalizeProjectInfoReturnPath(defaultHref);
    if (!currentLocation) return normalizedFallbackHref;

    const currentUrl = resolveUrlLike(currentLocation.href);
    const returnTo = String(currentUrl?.searchParams?.get(PROJECT_INFO_RETURN_TO_PARAM) || '').trim();
    if (!returnTo.length) return normalizedFallbackHref;

    return normalizeProjectInfoReturnPath(returnTo);
}

export function syncProjectInfoBackLink(anchorEl, {
    locationRef = null,
    defaultHref = PROJECT_INFO_DEFAULT_RETURN_PATH
} = {}) {
    if (!anchorEl || typeof anchorEl.setAttribute !== 'function') return false;
    const href = resolveProjectInfoBackHref(locationRef, { defaultHref });
    anchorEl.setAttribute('href', href);
    return true;
}

export function openProjectInfoPage(locationRef = null) {
    const resolvedLocation = locationRef || (typeof window !== 'undefined' ? window.location : null);
    if (!resolvedLocation || typeof resolvedLocation.assign !== 'function') {
        return false;
    }
    resolvedLocation.assign(buildProjectInfoPageUrl(resolvedLocation));
    return true;
}
