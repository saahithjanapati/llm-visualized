// @vitest-environment jsdom

import { describe, expect, it, vi } from 'vitest';

import {
    buildProjectInfoPageUrl,
    openProjectInfoPage,
    resolveProjectInfoBackHref,
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
    it('preserves the active 2D route when building the info page URL', () => {
        const infoHref = buildProjectInfoPageUrl(createLocationRef({
            href: 'https://llm-visualized.local/?view=2d&component=mhsa&layer=3',
            pathname: '/',
            search: '?view=2d&component=mhsa&layer=3'
        }));

        expect(infoHref).toBe('/info/?returnTo=%2F%3Fview%3D2d%26component%3Dmhsa%26layer%3D3');
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
});
