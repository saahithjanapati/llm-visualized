// @vitest-environment jsdom

import { beforeEach, describe, expect, it } from 'vitest';

import {
    resolveGenerationRoute,
    syncMainEntryToFirstGenerationRoute,
    syncGenerationRoute
} from './generationRoute.js';

describe('generationRoute', () => {
    beforeEach(() => {
        window.history.replaceState({}, '', '/');
    });

    it('restores KV cache mode from explicit route state', () => {
        const route = resolveGenerationRoute('/?token=8&generation=2&kvCache=1', {
            defaultLaneCount: 6,
            baseLaneCount: 6,
            maxLaneCount: 12
        });

        expect(route).toMatchObject({
            laneCount: 8,
            token: 8,
            generation: 2,
            kvCacheModeEnabled: true,
            hasExplicitRoute: true
        });
    });

    it('supports the legacy KV alias and explicit disabled values', () => {
        const route = resolveGenerationRoute('/?generation=2&kv=0', {
            defaultLaneCount: 6,
            baseLaneCount: 6,
            maxLaneCount: 12
        });

        expect(route).toMatchObject({
            laneCount: 8,
            generation: 2,
            kvCacheModeEnabled: false,
            hasExplicitRoute: true
        });
    });

    it('syncs KV cache mode without disturbing the active 2D route params', () => {
        window.history.replaceState({}, '', '/?view=2d&component=mhsa');

        syncGenerationRoute({
            laneCount: 8,
            baseLaneCount: 6,
            maxLaneCount: 12,
            kvCacheModeEnabled: true
        });

        const enabledUrl = new URL(window.location.href);
        expect(enabledUrl.searchParams.get('view')).toBe('2d');
        expect(enabledUrl.searchParams.get('component')).toBe('mhsa');
        expect(enabledUrl.searchParams.get('token')).toBe('8');
        expect(enabledUrl.searchParams.get('generation')).toBe('2');
        expect(enabledUrl.searchParams.get('kvCache')).toBe('1');

        syncGenerationRoute({
            laneCount: 6,
            baseLaneCount: 6,
            maxLaneCount: 12,
            kvCacheModeEnabled: false
        });

        const resetUrl = new URL(window.location.href);
        expect(resetUrl.searchParams.get('view')).toBe('2d');
        expect(resetUrl.searchParams.get('component')).toBe('mhsa');
        expect(resetUrl.searchParams.get('token')).toBe('6');
        expect(resetUrl.searchParams.get('generation')).toBe('0');
        expect(resetUrl.searchParams.get('kvCache')).toBe('0');
    });

    it('does not add explicit base-pass params unless the route was already explicit', () => {
        window.history.replaceState({}, '', '/?view=2d&component=mhsa');

        syncGenerationRoute({
            laneCount: 6,
            baseLaneCount: 6,
            maxLaneCount: 12,
            kvCacheModeEnabled: false
        });

        const url = new URL(window.location.href);
        expect(url.searchParams.get('view')).toBe('2d');
        expect(url.searchParams.get('component')).toBe('mhsa');
        expect(url.searchParams.has('token')).toBe(false);
        expect(url.searchParams.has('generation')).toBe(false);
        expect(url.searchParams.has('kvCache')).toBe(false);
    });

    it('canonicalizes the bare main entry URL to the prompt pass that generates the first completion token', () => {
        const changed = syncMainEntryToFirstGenerationRoute({
            baseLaneCount: 4,
            maxLaneCount: 12
        });

        const url = new URL(window.location.href);
        expect(changed).toBe(true);
        expect(url.pathname).toBe('/');
        expect(url.searchParams.get('token')).toBe('4');
        expect(url.searchParams.get('generation')).toBe('0');
        expect(url.searchParams.get('kvCache')).toBe('0');
    });

    it('preserves unrelated query params when canonicalizing the main entry URL', () => {
        window.history.replaceState({}, '', '/index.html?capture=capture_2.json&fresh=1');

        const changed = syncMainEntryToFirstGenerationRoute({
            baseLaneCount: 4,
            maxLaneCount: 12
        });

        const url = new URL(window.location.href);
        expect(changed).toBe(true);
        expect(url.pathname).toBe('/index.html');
        expect(url.searchParams.get('capture')).toBe('capture_2.json');
        expect(url.searchParams.get('fresh')).toBe('1');
        expect(url.searchParams.get('token')).toBe('4');
        expect(url.searchParams.get('generation')).toBe('0');
        expect(url.searchParams.get('kvCache')).toBe('0');
    });

    it('does not override an existing explicit generation route', () => {
        window.history.replaceState({}, '', '/?token=8&generation=4&kvCache=0');

        const changed = syncMainEntryToFirstGenerationRoute({
            baseLaneCount: 4,
            maxLaneCount: 12
        });

        const url = new URL(window.location.href);
        expect(changed).toBe(false);
        expect(url.searchParams.get('token')).toBe('8');
        expect(url.searchParams.get('generation')).toBe('4');
        expect(url.searchParams.get('kvCache')).toBe('0');
    });

    it('still canonicalizes the base prompt pass when the capture ends at that pass', () => {
        const changed = syncMainEntryToFirstGenerationRoute({
            baseLaneCount: 4,
            maxLaneCount: 4
        });

        const url = new URL(window.location.href);
        expect(changed).toBe(true);
        expect(url.searchParams.get('token')).toBe('4');
        expect(url.searchParams.get('generation')).toBe('0');
        expect(url.searchParams.get('kvCache')).toBe('0');
    });
});
