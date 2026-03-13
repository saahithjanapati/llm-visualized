import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import Gpt2Layer from './Gpt2Layer.js';

class MockTween {
    constructor(state) {
        this.state = state;
        this._onComplete = null;
    }

    to() {
        return this;
    }

    onComplete(fn) {
        this._onComplete = fn;
        return this;
    }

    start() {
        return this;
    }

    stop() {
        return this;
    }
}

let previousTWEEN;

beforeEach(() => {
    previousTWEEN = globalThis.TWEEN;
});

afterEach(() => {
    vi.restoreAllMocks();
    if (previousTWEEN === undefined) {
        delete globalThis.TWEEN;
    } else {
        globalThis.TWEEN = previousTWEEN;
    }
});

describe('Gpt2Layer pending additions', () => {
    it('tracks pending addition deadlines using playback time', () => {
        globalThis.TWEEN = { Tween: MockTween };

        const layer = Object.create(Gpt2Layer.prototype);
        layer._getBaseVectorLength = () => 8;
        layer._getNowMs = () => 99999;
        layer._getPlaybackNowMs = () => 1234;

        const lane = {
            originalVec: { instanceCount: 8 }
        };

        layer._scheduleAdditionCompletion(lane);

        expect(lane.__additionStartedAtMs).toBe(1234);
        expect(lane.__additionExpectedCompleteAtMs).toBeGreaterThan(1234);
        expect(lane._additionCompletionTween).toBeInstanceOf(MockTween);
    });

    it('forces pending additions through their registered finalizer', () => {
        const layer = Object.create(Gpt2Layer.prototype);
        layer.index = 3;
        layer._pendingAdditions = 1;
        layer._emitProgress = vi.fn();

        const stop = vi.fn();
        const lane = {
            additionComplete: false,
            laneIndex: 2,
            __additionPending: true,
            _additionCompletionTween: { stop }
        };

        lane.__pendingAdditionFinalize = vi.fn(() => {
            lane.promoted = true;
            layer._completePendingAddition(lane);
        });

        const completed = layer._forceCompletePendingAddition(lane, 'test');

        expect(completed).toBe(true);
        expect(lane.__pendingAdditionFinalize).toBeUndefined();
        expect(lane.promoted).toBe(true);
        expect(lane.additionComplete).toBe(true);
        expect(layer._pendingAdditions).toBe(0);
        expect(stop).toHaveBeenCalledTimes(1);
        expect(layer._emitProgress).toHaveBeenCalled();
    });
});
