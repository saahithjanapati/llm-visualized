// @vitest-environment jsdom
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import {
    resolveGenerationRoute,
    syncGenerationRoute
} from '../src/app/gpt-tower/generationRoute.js';

describe('generationRoute', () => {
    beforeEach(() => {
        window.history.replaceState({}, '', '/?capture=capture.json');
    });

    afterEach(() => {
        vi.restoreAllMocks();
    });

    it('prefers the token route when both token and generation are present', () => {
        const route = resolveGenerationRoute('/?token=7&generation=1', {
            defaultLaneCount: 4,
            baseLaneCount: 4,
            maxLaneCount: 12
        });

        expect(route).toMatchObject({
            laneCount: 7,
            token: 7,
            generation: 3,
            hasExplicitRoute: true
        });
    });

    it('resolves generation-only routes relative to the prompt base', () => {
        const route = resolveGenerationRoute('/?generation=2', {
            defaultLaneCount: 4,
            baseLaneCount: 4,
            maxLaneCount: 12
        });

        expect(route).toMatchObject({
            laneCount: 6,
            token: 6,
            generation: 2,
            hasExplicitRoute: true
        });
    });

    it('clamps token routes so they do not go earlier than the base prompt window', () => {
        const route = resolveGenerationRoute('/?token=2', {
            defaultLaneCount: 4,
            baseLaneCount: 4,
            maxLaneCount: 12
        });

        expect(route).toMatchObject({
            laneCount: 4,
            token: 4,
            generation: 0,
            hasExplicitRoute: false
        });
    });

    it('removes generation params at the base state while preserving unrelated route params', () => {
        window.history.replaceState({}, '', '/?view=2d&component=mhsa&token=6&generation=2');
        const replaceSpy = vi.spyOn(window.history, 'replaceState');

        const changed = syncGenerationRoute({
            laneCount: 4,
            baseLaneCount: 4,
            historyMode: 'replace'
        });

        const params = new URLSearchParams(window.location.search);
        expect(changed).toBe(true);
        expect(replaceSpy).toHaveBeenCalledOnce();
        expect(params.get('view')).toBe('2d');
        expect(params.get('component')).toBe('mhsa');
        expect(params.has('token')).toBe(false);
        expect(params.has('generation')).toBe(false);
    });

    it('pushes token and generation params for later passes', () => {
        const pushSpy = vi.spyOn(window.history, 'pushState');

        const changed = syncGenerationRoute({
            laneCount: 6,
            baseLaneCount: 4,
            historyMode: 'push'
        });

        const params = new URLSearchParams(window.location.search);
        expect(changed).toBe(true);
        expect(pushSpy).toHaveBeenCalledOnce();
        expect(params.get('token')).toBe('6');
        expect(params.get('generation')).toBe('2');
        expect(params.get('capture')).toBe('capture.json');
    });
});
