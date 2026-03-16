// @vitest-environment jsdom

import { beforeEach, describe, expect, it } from 'vitest';

import {
    resolveGenerationRoute,
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
        expect(resetUrl.searchParams.has('token')).toBe(false);
        expect(resetUrl.searchParams.has('generation')).toBe(false);
        expect(resetUrl.searchParams.has('kvCache')).toBe(false);
    });
});
