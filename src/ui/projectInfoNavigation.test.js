// @vitest-environment jsdom

import { afterEach, describe, expect, it, vi } from 'vitest';

import {
    bindProjectInfoBackLink,
    buildProjectInfoPageUrl,
    openProjectInfoPage,
    PROJECT_INFO_ACTIVE_VISUALIZATION_MODES,
    resolveProjectInfoBackHref,
    resolveProjectInfoNavigationReturnHref,
    setProjectInfoActiveVisualizationMode,
    shouldUseProjectInfoHistoryBack,
    syncProjectInfoBackLink
} from './projectInfoNavigation.js';

function createLocationRef({
    href = 'https://llm-visualized.local/',
    pathname = '/',
    search = '',
    hash = ''
} = {}) {
    return {
        href,
        pathname,
        search,
        hash
    };
}

describe('projectInfoNavigation', () => {
    afterEach(() => {
        setProjectInfoActiveVisualizationMode('');
    });

    it('preserves the active 2D route when building the info page URL', () => {
        const infoHref = buildProjectInfoPageUrl(createLocationRef({
            href: 'https://llm-visualized.local/?view=2d&component=mhsa&layer=3',
            pathname: '/',
            search: '?view=2d&component=mhsa&layer=3'
        }));

        expect(infoHref).toBe('/info/?returnTo=%2F%3Fview%3D2d%26component%3Dmhsa%26layer%3D3');
    });

    it('drops stale 2D route params when the active visualization is 3D', () => {
        setProjectInfoActiveVisualizationMode(PROJECT_INFO_ACTIVE_VISUALIZATION_MODES.SCENE_3D);

        const returnHref = resolveProjectInfoNavigationReturnHref(createLocationRef({
            href: 'https://llm-visualized.local/?token=8&generation=4&view=2d&component=mhsa&layer=3',
            pathname: '/',
            search: '?token=8&generation=4&view=2d&component=mhsa&layer=3'
        }));
        const infoHref = buildProjectInfoPageUrl(createLocationRef({
            href: 'https://llm-visualized.local/?token=8&generation=4&view=2d&component=mhsa&layer=3',
            pathname: '/',
            search: '?token=8&generation=4&view=2d&component=mhsa&layer=3'
        }));

        expect(returnHref).toBe('/?token=8&generation=4');
        expect(infoHref).toBe('/info/?returnTo=%2F%3Ftoken%3D8%26generation%3D4');
    });

    it('routes back to the encoded visualization URL from the info page', () => {
        const backHref = resolveProjectInfoBackHref(createLocationRef({
            href: 'https://llm-visualized.local/info/?returnTo=%2F%3Fview%3D2d%26component%3Dmhsa%26layer%3D3',
            pathname: '/info/',
            search: '?returnTo=%2F%3Fview%3D2d%26component%3Dmhsa%26layer%3D3'
        }));

        expect(backHref).toBe('/?view=2d&component=mhsa&layer=3');
    });

    it('falls back to the main visualization route for invalid return targets', () => {
        const backHref = resolveProjectInfoBackHref(createLocationRef({
            href: 'https://llm-visualized.local/info/?returnTo=https%3A%2F%2Fevil.example%2Fsteal',
            pathname: '/info/',
            search: '?returnTo=https%3A%2F%2Fevil.example%2Fsteal'
        }));

        expect(backHref).toBe('/');
    });

    it('updates the info page back link href from the encoded return target', () => {
        document.body.innerHTML = '<a class="info-page-back-link" href="/">Back to visualization</a>';
        const anchorEl = document.querySelector('.info-page-back-link');

        const synced = syncProjectInfoBackLink(anchorEl, {
            locationRef: createLocationRef({
                href: 'https://llm-visualized.local/info/?returnTo=%2F%3Fview%3D2d',
                pathname: '/info/',
                search: '?returnTo=%2F%3Fview%3D2d'
            })
        });

        expect(synced).toBe(true);
        expect(anchorEl?.getAttribute('href')).toBe('/?view=2d');
    });

    it('navigates to the info page with the current route encoded as the return target', () => {
        const assign = vi.fn();
        const locationRef = {
            assign,
            href: 'https://llm-visualized.local/?view=2d&component=mhsa',
            pathname: '/',
            search: '?view=2d&component=mhsa',
            hash: ''
        };

        const opened = openProjectInfoPage(locationRef);

        expect(opened).toBe(true);
        expect(assign).toHaveBeenCalledWith('/info/?returnTo=%2F%3Fview%3D2d%26component%3Dmhsa');
    });

    it('uses the active 3D state when opening the info page from a stale 2D URL', () => {
        setProjectInfoActiveVisualizationMode(PROJECT_INFO_ACTIVE_VISUALIZATION_MODES.SCENE_3D);
        const assign = vi.fn();
        const locationRef = {
            assign,
            href: 'https://llm-visualized.local/?token=8&generation=4&view=2d&component=mhsa',
            pathname: '/',
            search: '?token=8&generation=4&view=2d&component=mhsa',
            hash: ''
        };

        const opened = openProjectInfoPage(locationRef);

        expect(opened).toBe(true);
        expect(assign).toHaveBeenCalledWith('/info/?returnTo=%2F%3Ftoken%3D8%26generation%3D4');
    });

    it('prefers history back when the referrer matches the encoded visualization route', () => {
        const shouldUseHistoryBack = shouldUseProjectInfoHistoryBack({
            locationRef: createLocationRef({
                href: 'https://llm-visualized.local/info/?returnTo=%2F%3Fview%3D2d%26component%3Dmhsa%26layer%3D3',
                pathname: '/info/',
                search: '?returnTo=%2F%3Fview%3D2d%26component%3Dmhsa%26layer%3D3'
            }),
            documentRef: {
                referrer: 'https://llm-visualized.local/?view=2d&component=mhsa&layer=3'
            },
            historyRef: {
                back: vi.fn(),
                length: 2
            }
        });

        expect(shouldUseHistoryBack).toBe(true);
    });

    it('does not use history back for mismatched or direct-entry info pages', () => {
        const shouldUseHistoryBack = shouldUseProjectInfoHistoryBack({
            locationRef: createLocationRef({
                href: 'https://llm-visualized.local/info/?returnTo=%2F%3Fview%3D2d',
                pathname: '/info/',
                search: '?returnTo=%2F%3Fview%3D2d'
            }),
            documentRef: {
                referrer: 'https://llm-visualized.local/'
            },
            historyRef: {
                back: vi.fn(),
                length: 1
            }
        });

        expect(shouldUseHistoryBack).toBe(false);
    });

    it('uses history traversal for the info-page back link when returning to the same app state', () => {
        document.body.innerHTML = '<a class="info-page-back-link" href="/">Back to visualization</a>';
        const anchorEl = document.querySelector('.info-page-back-link');
        const back = vi.fn();

        syncProjectInfoBackLink(anchorEl, {
            locationRef: createLocationRef({
                href: 'https://llm-visualized.local/info/?returnTo=%2F%3Fview%3D2d%26component%3Dmhsa',
                pathname: '/info/',
                search: '?returnTo=%2F%3Fview%3D2d%26component%3Dmhsa'
            })
        });
        bindProjectInfoBackLink(anchorEl, {
            locationRef: createLocationRef({
                href: 'https://llm-visualized.local/info/?returnTo=%2F%3Fview%3D2d%26component%3Dmhsa',
                pathname: '/info/',
                search: '?returnTo=%2F%3Fview%3D2d%26component%3Dmhsa'
            }),
            documentRef: {
                referrer: 'https://llm-visualized.local/?view=2d&component=mhsa'
            },
            historyRef: {
                back,
                length: 2
            }
        });

        const event = new MouseEvent('click', { bubbles: true, cancelable: true, button: 0 });
        anchorEl?.dispatchEvent(event);

        expect(back).toHaveBeenCalledTimes(1);
        expect(event.defaultPrevented).toBe(true);
    });
});
