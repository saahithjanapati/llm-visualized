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

    it('prefers raycast roots from active and nearby layers while keeping always-on roots', () => {
        const engine = createEngine();
        const rootLayer0 = { userData: { layerIndex: 0 } };
        const rootLayer1 = { userData: { layerIndex: 1 } };
        const rootLayer2 = { userData: { layerIndex: 2 } };
        const rootLayer5 = { userData: { layerIndex: 5 } };
        const persistentRoot = { userData: {} };

        engine._layers = [
            { index: 0, isActive: false, _transitionPhase: 'complete' },
            { index: 1, isActive: false, _transitionPhase: 'complete' },
            { index: 2, isActive: true, _transitionPhase: 'complete' },
            { index: 5, isActive: false, _transitionPhase: 'complete' }
        ];
        engine._raycastRoots = [rootLayer0, rootLayer1, rootLayer2, rootLayer5, persistentRoot];

        expect(engine._resolvePreferredRaycastRoots()).toEqual([
            rootLayer1,
            rootLayer2,
            persistentRoot
        ]);
    });

    it('falls back to the full raycast root list when the preferred roots miss', () => {
        const engine = createEngine();
        const preferredRoot = { userData: { layerIndex: 1 } };
        const fullOnlyRoot = { userData: { layerIndex: 5 } };

        engine._layers = [
            { index: 1, isActive: true, _transitionPhase: 'complete' },
            { index: 5, isActive: false, _transitionPhase: 'complete' }
        ];
        engine._raycastRoots = [preferredRoot, fullOnlyRoot];
        const intersectSpy = vi.fn((roots) => {
            if (roots.length === 1 && roots[0] === preferredRoot) {
                return [];
            }
            return [{ object: fullOnlyRoot }];
        });
        engine._intersectRaycastRoots = intersectSpy;

        expect(engine._intersectPreferredRaycastRoots()).toEqual({
            intersects: [{ object: fullOnlyRoot }],
            passCount: 2
        });
        expect(intersectSpy).toHaveBeenCalledTimes(2);
    });

    it('avoids the full-root fallback when the preferred roots already hit', () => {
        const engine = createEngine();
        const preferredRoot = { userData: { layerIndex: 1 } };
        const fullOnlyRoot = { userData: { layerIndex: 5 } };

        engine._layers = [
            { index: 1, isActive: true, _transitionPhase: 'complete' },
            { index: 5, isActive: false, _transitionPhase: 'complete' }
        ];
        engine._raycastRoots = [preferredRoot, fullOnlyRoot];

        const preferredRoots = engine._resolvePreferredRaycastRoots();
        const intersectSpy = vi.fn(() => [{ object: preferredRoot }]);
        engine._intersectRaycastRoots = intersectSpy;

        expect(engine._intersectPreferredRaycastRoots()).toEqual({
            intersects: [{ object: preferredRoot }],
            passCount: 1
        });
        expect(intersectSpy).toHaveBeenCalledWith(preferredRoots);
        expect(intersectSpy).toHaveBeenCalledTimes(1);
    });

    it('reuses the raycast intersection target array', () => {
        const engine = createEngine();
        const oldHit = { object: { name: 'old' } };
        const newHit = { object: { name: 'new' } };
        engine._raycastRoots = [{ name: 'root' }];
        engine._raycastIntersections = [oldHit];
        engine._raycaster = {
            intersectObjects: vi.fn((roots, recursive, target) => {
                target.push(newHit);
                return target;
            })
        };

        const result = engine._getRaycastIntersections();

        expect(result).toBe(engine._raycastIntersections);
        expect(result).toEqual([newHit]);
        expect(engine._raycaster.intersectObjects).toHaveBeenCalledWith(
            engine._raycastRoots,
            true,
            result
        );
    });

    it('emits hover raycast selections only when the target changes', () => {
        const engine = createEngine();
        engine._raycastHoverSelectionKey = null;
        engine._raycastHoverHandler = vi.fn();
        const object = { uuid: 'same-object' };
        const selection = {
            label: 'Value Vector',
            kind: 'mergedKV',
            object,
            hit: { object, instanceId: 4 },
            info: { category: 'V', layerIndex: 1, headIndex: 2, tokenIndex: 3 }
        };

        engine._emitRaycastHoverSelection(selection);
        engine._emitRaycastHoverSelection({ ...selection });
        engine._emitRaycastHoverSelection({ ...selection, info: { ...selection.info, tokenIndex: 5 } });
        engine._emitRaycastHoverSelection(null);
        engine._emitRaycastHoverSelection(null);

        expect(engine._raycastHoverHandler).toHaveBeenCalledTimes(3);
        expect(engine._raycastHoverHandler.mock.calls[0][0]).toEqual(selection);
        expect(engine._raycastHoverHandler.mock.calls[1][0].info.tokenIndex).toBe(5);
        expect(engine._raycastHoverHandler.mock.calls[2][0]).toBeNull();
    });
});
