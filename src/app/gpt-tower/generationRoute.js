const GENERATION_ROUTE_PARAM_TOKEN = 'token';
const GENERATION_ROUTE_PARAM_GENERATION = 'generation';
const GENERATION_ROUTE_PARAM_KV_CACHE = 'kvCache';
const GENERATION_ROUTE_PARAM_KV_CACHE_LEGACY = 'kv';
const MAIN_ENTRY_ROUTE_PATHS = new Set(['/', '/index.html']);

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

    try {
        return new URL(rawUrl, 'https://llm-visualized.local');
    } catch (_) {
        return null;
    }
}

function clampLaneCount(value, {
    minLaneCount = 1,
    maxLaneCount = null
} = {}) {
    const safeMin = Math.max(1, Math.floor(minLaneCount || 1));
    const parsed = Number(value);
    if (!Number.isFinite(parsed)) return safeMin;
    const floored = Math.floor(parsed);
    const safeMax = Number.isFinite(maxLaneCount)
        ? Math.max(safeMin, Math.floor(maxLaneCount))
        : null;
    return safeMax == null
        ? Math.max(safeMin, floored)
        : Math.min(safeMax, Math.max(safeMin, floored));
}

function normalizeRoutePathname(pathname = '') {
    const raw = String(pathname || '').trim();
    if (!raw.length || raw === '/') return '/';
    const trimmed = raw.replace(/\/+$/, '');
    return trimmed.length ? trimmed : '/';
}

function isMainEntryRoutePath(pathname = '') {
    return MAIN_ENTRY_ROUTE_PATHS.has(normalizeRoutePathname(pathname));
}

function hasExplicitGenerationPositionRouteParams(searchParams) {
    if (!searchParams) return false;
    return searchParams.has(GENERATION_ROUTE_PARAM_TOKEN)
        || searchParams.has(GENERATION_ROUTE_PARAM_GENERATION);
}

function hasExplicitGenerationRouteParams(searchParams) {
    if (!searchParams) return false;
    return hasExplicitGenerationPositionRouteParams(searchParams)
        || searchParams.has(GENERATION_ROUTE_PARAM_KV_CACHE)
        || searchParams.has(GENERATION_ROUTE_PARAM_KV_CACHE_LEGACY);
}

function readRouteIndex(searchParams, key, {
    min = 0,
    max = null
} = {}) {
    if (!searchParams?.has(key)) return null;
    const parsed = Number(searchParams.get(key));
    if (!Number.isFinite(parsed)) return null;
    const floored = Math.floor(parsed);
    if (floored < min) return null;
    if (Number.isFinite(max) && floored > Math.floor(max)) {
        return Math.floor(max);
    }
    return floored;
}

function readRouteBoolean(searchParams, ...keys) {
    if (!searchParams) return null;
    for (const key of keys) {
        if (!searchParams.has(key)) continue;
        const normalized = String(searchParams.get(key) || '').trim().toLowerCase();
        if (!normalized.length) return true;
        if (
            normalized === '1'
            || normalized === 'true'
            || normalized === 'yes'
            || normalized === 'on'
            || normalized === 'enabled'
        ) {
            return true;
        }
        if (
            normalized === '0'
            || normalized === 'false'
            || normalized === 'no'
            || normalized === 'off'
            || normalized === 'disabled'
        ) {
            return false;
        }
        return true;
    }
    return null;
}

export function resolveGenerationRoute(urlLike = null, {
    defaultLaneCount = 1,
    baseLaneCount = defaultLaneCount,
    maxLaneCount = null
} = {}) {
    const safeBaseLaneCount = clampLaneCount(baseLaneCount, { maxLaneCount });
    const safeDefaultLaneCount = clampLaneCount(defaultLaneCount, {
        minLaneCount: safeBaseLaneCount,
        maxLaneCount
    });
    const url = resolveUrlLike(urlLike);
    if (!url) {
        return {
            laneCount: safeDefaultLaneCount,
            token: safeDefaultLaneCount,
            generation: Math.max(0, safeDefaultLaneCount - safeBaseLaneCount),
            kvCacheModeEnabled: false,
            hasExplicitRoute: false
        };
    }

    const token = readRouteIndex(url.searchParams, GENERATION_ROUTE_PARAM_TOKEN, {
        min: safeBaseLaneCount,
        max: maxLaneCount
    });
    const generation = readRouteIndex(url.searchParams, GENERATION_ROUTE_PARAM_GENERATION, { min: 0 });
    const kvCacheModeEnabled = readRouteBoolean(
        url.searchParams,
        GENERATION_ROUTE_PARAM_KV_CACHE,
        GENERATION_ROUTE_PARAM_KV_CACHE_LEGACY
    ) === true;

    let laneCount = safeDefaultLaneCount;
    let hasExplicitRoute = false;
    if (Number.isFinite(token)) {
        laneCount = token;
        hasExplicitRoute = true;
    } else if (Number.isFinite(generation)) {
        laneCount = safeBaseLaneCount + generation;
        hasExplicitRoute = true;
    }

    const safeLaneCount = clampLaneCount(laneCount, {
        minLaneCount: safeBaseLaneCount,
        maxLaneCount
    });

    return {
        laneCount: safeLaneCount,
        token: safeLaneCount,
        generation: Math.max(0, safeLaneCount - safeBaseLaneCount),
        kvCacheModeEnabled,
        hasExplicitRoute
    };
}

export function syncGenerationRoute({
    laneCount = 1,
    baseLaneCount = 1,
    maxLaneCount = null,
    kvCacheModeEnabled = false,
    historyMode = 'replace',
    forceExplicitRouteState = false
} = {}) {
    if (typeof window === 'undefined' || typeof window.history?.replaceState !== 'function') return false;

    const nextUrl = resolveUrlLike(window.location);
    if (!nextUrl) return false;

    const safeBaseLaneCount = clampLaneCount(baseLaneCount, { maxLaneCount });
    const safeLaneCount = clampLaneCount(laneCount, {
        minLaneCount: safeBaseLaneCount,
        maxLaneCount
    });
    const generation = Math.max(0, safeLaneCount - safeBaseLaneCount);
    const hadExplicitGenerationPositionRoute = hasExplicitGenerationPositionRouteParams(nextUrl.searchParams);
    const hadExplicitGenerationRoute = hasExplicitGenerationRouteParams(nextUrl.searchParams);
    const shouldPersistExplicitGenerationPosition = forceExplicitRouteState || hadExplicitGenerationPositionRoute;
    const shouldPersistExplicitKvRoute = forceExplicitRouteState || hadExplicitGenerationRoute;

    if (safeLaneCount <= safeBaseLaneCount && !shouldPersistExplicitGenerationPosition) {
        nextUrl.searchParams.delete(GENERATION_ROUTE_PARAM_TOKEN);
        nextUrl.searchParams.delete(GENERATION_ROUTE_PARAM_GENERATION);
    } else {
        nextUrl.searchParams.set(GENERATION_ROUTE_PARAM_TOKEN, String(safeLaneCount));
        nextUrl.searchParams.set(GENERATION_ROUTE_PARAM_GENERATION, String(generation));
    }
    nextUrl.searchParams.delete(GENERATION_ROUTE_PARAM_KV_CACHE_LEGACY);
    if (kvCacheModeEnabled) {
        nextUrl.searchParams.set(GENERATION_ROUTE_PARAM_KV_CACHE, '1');
    } else if (shouldPersistExplicitKvRoute) {
        nextUrl.searchParams.set(GENERATION_ROUTE_PARAM_KV_CACHE, '0');
    } else {
        nextUrl.searchParams.delete(GENERATION_ROUTE_PARAM_KV_CACHE);
    }

    const nextHref = `${nextUrl.pathname}${nextUrl.search}${nextUrl.hash}`;
    const currentHref = `${window.location.pathname}${window.location.search}${window.location.hash}`;
    if (nextHref === currentHref) return false;

    if (historyMode === 'push' && typeof window.history.pushState === 'function') {
        window.history.pushState(window.history.state, '', nextHref);
        return true;
    }

    window.history.replaceState(window.history.state, '', nextHref);
    return true;
}

export function syncMainEntryToFirstGenerationRoute({
    baseLaneCount = 1,
    maxLaneCount = null,
    kvCacheModeEnabled = false,
    historyMode = 'replace'
} = {}) {
    if (typeof window === 'undefined' || typeof window.history?.replaceState !== 'function') return false;

    const currentUrl = resolveUrlLike(window.location);
    if (!currentUrl || !isMainEntryRoutePath(currentUrl.pathname)) return false;
    if (hasExplicitGenerationRouteParams(currentUrl.searchParams)) return false;

    const safeBaseLaneCount = clampLaneCount(baseLaneCount, { maxLaneCount });
    const safeMaxLaneCount = Number.isFinite(maxLaneCount)
        ? Math.max(safeBaseLaneCount, Math.floor(maxLaneCount))
        : null;

    return syncGenerationRoute({
        laneCount: safeBaseLaneCount,
        baseLaneCount: safeBaseLaneCount,
        maxLaneCount: safeMaxLaneCount,
        kvCacheModeEnabled,
        historyMode,
        forceExplicitRouteState: true
    });
}

export default resolveGenerationRoute;
