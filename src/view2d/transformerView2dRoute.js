import {
    buildSemanticTarget,
    normalizeOptionalIndex
} from './transformerView2dTargets.js';

const TRANSFORMER_VIEW2D_ROUTE_VIEW = '2d';
const TRANSFORMER_VIEW2D_ROUTE_VALUES = new Set([
    TRANSFORMER_VIEW2D_ROUTE_VIEW,
    'transformer-2d',
    'transformer-view2d',
    'view2d'
]);
const TRANSFORMER_VIEW2D_ROUTE_PARAM_COMPONENT = 'component';
const TRANSFORMER_VIEW2D_ROUTE_PARAM_COMPONENT_LEGACY = 'componentKind';
const TRANSFORMER_VIEW2D_ROUTE_PARAM_LAYER = 'layer';
const TRANSFORMER_VIEW2D_ROUTE_PARAM_LAYER_LEGACY = 'layerIndex';
const TRANSFORMER_VIEW2D_ROUTE_PARAM_HEAD = 'head';
const TRANSFORMER_VIEW2D_ROUTE_PARAM_HEAD_LEGACY = 'headIndex';
const TRANSFORMER_VIEW2D_ROUTE_PARAM_STAGE = 'stage';
const TRANSFORMER_VIEW2D_ROUTE_PARAM_ROLE = 'role';

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

function readRouteString(searchParams, ...keys) {
    if (!searchParams) return '';
    for (const key of keys) {
        const value = String(searchParams.get(key) || '').trim();
        if (value.length) return value;
    }
    return '';
}

function readRouteIndex(searchParams, ...keys) {
    if (!searchParams) return null;
    for (const key of keys) {
        if (!searchParams.has(key)) continue;
        const value = Number(searchParams.get(key));
        if (Number.isFinite(value)) {
            return normalizeOptionalIndex(value);
        }
    }
    return null;
}

function clearTransformerView2dRouteParams(searchParams) {
    if (!searchParams) return;
    [
        TRANSFORMER_VIEW2D_ROUTE_PARAM_COMPONENT,
        TRANSFORMER_VIEW2D_ROUTE_PARAM_COMPONENT_LEGACY,
        TRANSFORMER_VIEW2D_ROUTE_PARAM_LAYER,
        TRANSFORMER_VIEW2D_ROUTE_PARAM_LAYER_LEGACY,
        TRANSFORMER_VIEW2D_ROUTE_PARAM_HEAD,
        TRANSFORMER_VIEW2D_ROUTE_PARAM_HEAD_LEGACY,
        TRANSFORMER_VIEW2D_ROUTE_PARAM_STAGE,
        TRANSFORMER_VIEW2D_ROUTE_PARAM_ROLE
    ].forEach((key) => {
        searchParams.delete(key);
    });
}

export function isTransformerView2dRouteValue(value = '') {
    return TRANSFORMER_VIEW2D_ROUTE_VALUES.has(String(value || '').trim().toLowerCase());
}

export function resolveTransformerView2dRoute(urlLike = null) {
    const url = resolveUrlLike(urlLike);
    if (!url) return null;

    const viewParam = String(url.searchParams.get('view') || '').trim().toLowerCase();
    const hashView = String(url.hash || '').replace(/^#/, '').trim().toLowerCase();
    if (!isTransformerView2dRouteValue(viewParam) && !isTransformerView2dRouteValue(hashView)) {
        return null;
    }

    return {
        semanticTarget: buildSemanticTarget({
            componentKind: readRouteString(
                url.searchParams,
                TRANSFORMER_VIEW2D_ROUTE_PARAM_COMPONENT,
                TRANSFORMER_VIEW2D_ROUTE_PARAM_COMPONENT_LEGACY
            ),
            layerIndex: readRouteIndex(
                url.searchParams,
                TRANSFORMER_VIEW2D_ROUTE_PARAM_LAYER,
                TRANSFORMER_VIEW2D_ROUTE_PARAM_LAYER_LEGACY
            ),
            headIndex: readRouteIndex(
                url.searchParams,
                TRANSFORMER_VIEW2D_ROUTE_PARAM_HEAD,
                TRANSFORMER_VIEW2D_ROUTE_PARAM_HEAD_LEGACY
            ),
            stage: readRouteString(url.searchParams, TRANSFORMER_VIEW2D_ROUTE_PARAM_STAGE),
            role: readRouteString(url.searchParams, TRANSFORMER_VIEW2D_ROUTE_PARAM_ROLE)
        })
    };
}

export function syncTransformerView2dRoute({
    active = false,
    semanticTarget = null
} = {}) {
    if (typeof window === 'undefined' || typeof window.history?.replaceState !== 'function') return false;

    const nextUrl = resolveUrlLike(window.location);
    if (!nextUrl) return false;

    const currentView = String(nextUrl.searchParams.get('view') || '').trim().toLowerCase();
    const currentHash = String(nextUrl.hash || '').replace(/^#/, '').trim().toLowerCase();

    if (active) {
        nextUrl.searchParams.set('view', TRANSFORMER_VIEW2D_ROUTE_VIEW);
        clearTransformerView2dRouteParams(nextUrl.searchParams);

        const target = buildSemanticTarget(semanticTarget);
        if (target?.componentKind) {
            nextUrl.searchParams.set(TRANSFORMER_VIEW2D_ROUTE_PARAM_COMPONENT, target.componentKind);
        }
        if (Number.isFinite(target?.layerIndex)) {
            nextUrl.searchParams.set(TRANSFORMER_VIEW2D_ROUTE_PARAM_LAYER, String(Math.floor(target.layerIndex)));
        }
        if (Number.isFinite(target?.headIndex)) {
            nextUrl.searchParams.set(TRANSFORMER_VIEW2D_ROUTE_PARAM_HEAD, String(Math.floor(target.headIndex)));
        }
        if (target?.stage) {
            nextUrl.searchParams.set(TRANSFORMER_VIEW2D_ROUTE_PARAM_STAGE, target.stage);
        }
        if (target?.role) {
            nextUrl.searchParams.set(TRANSFORMER_VIEW2D_ROUTE_PARAM_ROLE, target.role);
        }
        if (isTransformerView2dRouteValue(currentHash)) {
            nextUrl.hash = '';
        }
    } else {
        if (isTransformerView2dRouteValue(currentView)) {
            nextUrl.searchParams.delete('view');
        }
        clearTransformerView2dRouteParams(nextUrl.searchParams);
        if (isTransformerView2dRouteValue(currentHash)) {
            nextUrl.hash = '';
        }
    }

    const nextHref = `${nextUrl.pathname}${nextUrl.search}${nextUrl.hash}`;
    const currentHref = `${window.location.pathname}${window.location.search}${window.location.hash}`;
    if (nextHref !== currentHref) {
        window.history.replaceState(window.history.state, '', nextHref);
    }
    return true;
}
