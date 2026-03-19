import { afterAll, afterEach, beforeAll, describe, expect, it, vi } from 'vitest';

let CoreEngine;

function createEngine() {
    const engine = Object.create(CoreEngine.prototype);
    engine._animationFrame = null;
    engine._disposed = false;
    engine._paused = false;
    engine._pauseReasons = new Set();
    engine._renderInvalidated = false;
    engine._needsFreshFrameAfterResume = false;
    engine._keyState = new Set();
    engine._layers = [];
    return engine;
}

describe('CoreEngine invalidation loop', () => {
    beforeAll(async () => {
        vi.stubGlobal('localStorage', {
            getItem: vi.fn(() => null),
            setItem: vi.fn(),
            removeItem: vi.fn()
        });
        ({ CoreEngine } = await import('./CoreEngine.js'));
    });

    afterEach(() => {
        vi.unstubAllGlobals();
    });

    afterAll(() => {
        vi.unstubAllGlobals();
    });

    it('schedules at most one animation frame per invalidation burst', () => {
        const engine = createEngine();
        const raf = vi.fn(() => 17);
        vi.stubGlobal('requestAnimationFrame', raf);

        expect(engine.requestRender('test')).toBe(true);
        expect(engine.requestRender('test-again')).toBe(false);
        expect(engine._renderInvalidated).toBe(true);
        expect(engine._animationFrame).toBe(17);
        expect(raf).toHaveBeenCalledTimes(1);
    });

    it('does not schedule frames while paused only for document visibility', () => {
        const engine = createEngine();
        engine._paused = true;
        engine._pauseReasons = new Set(['visibility']);
        const raf = vi.fn(() => 23);
        vi.stubGlobal('requestAnimationFrame', raf);

        expect(engine.requestRender('visibility')).toBe(false);
        expect(engine._renderInvalidated).toBe(true);
        expect(engine._animationFrame).toBe(null);
        expect(raf).not.toHaveBeenCalled();
    });

    it('keeps the loop running only while there is active frame work', () => {
        const engine = createEngine();

        expect(engine._shouldKeepAnimationLoopRunning()).toBe(false);
        expect(engine._shouldKeepAnimationLoopRunning({ activeTweenCount: 1 })).toBe(true);

        engine._layers = [{ update: () => {}, needsFrameUpdate: () => true }];
        expect(engine._shouldKeepAnimationLoopRunning()).toBe(true);
    });

    it('uses layer-specific frame demand when deciding whether to update paused layers', () => {
        const engine = createEngine();
        const layer = {
            update: () => {},
            updateWhenPaused: false,
            needsFrameUpdate: vi.fn(({ paused }) => paused)
        };

        expect(engine._shouldUpdateLayer(layer, { paused: false })).toBe(false);
        expect(engine._shouldUpdateLayer(layer, { paused: true })).toBe(true);
        expect(layer.needsFrameUpdate).toHaveBeenCalledTimes(2);
    });
});
