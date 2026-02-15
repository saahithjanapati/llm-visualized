import { afterEach, describe, expect, it, vi } from 'vitest';
import {
    clearScheduledDelays,
    resolveSkipDelay,
    resolveSkipDuration,
    scheduleAfterDelay
} from '../src/animations/mhsa/mhsaTimingUtils.js';

describe('mhsaTimingUtils', () => {
    afterEach(() => {
        vi.useRealTimers();
    });

    it('scales skip delays and durations only in skip mode', () => {
        expect(resolveSkipDelay(false, 120)).toBe(120);
        expect(resolveSkipDelay(true, 120)).toBeCloseTo(3.6, 6);
        expect(resolveSkipDelay(true, 10_000)).toBe(5);

        expect(resolveSkipDuration(false, 80)).toBe(80);
        expect(resolveSkipDuration(true, 80)).toBe(4);
        expect(resolveSkipDuration(true, 10_000)).toBe(14);
    });

    it('schedules callbacks via timeout path when tween lib is unavailable', async () => {
        vi.useFakeTimers();

        let fired = false;
        const scheduledDelayTweens = new Set();
        const scheduledTimeoutIds = new Set();

        scheduleAfterDelay({
            callback: () => { fired = true; },
            delayMs: 100,
            skipToEndActive: true,
            scheduledDelayTweens,
            scheduledTimeoutIds,
            tweenLib: null
        });

        expect(scheduledDelayTweens.size).toBe(0);
        expect(scheduledTimeoutIds.size).toBe(1);

        await vi.advanceTimersByTimeAsync(4);
        expect(fired).toBe(true);
        expect(scheduledTimeoutIds.size).toBe(0);
    });

    it('returns cancel handle for timeout scheduling', () => {
        vi.useFakeTimers();

        let fired = false;
        const scheduledTimeoutIds = new Set();
        const cancel = scheduleAfterDelay({
            callback: () => { fired = true; },
            delayMs: 50,
            skipToEndActive: false,
            scheduledTimeoutIds,
            tweenLib: null
        });

        expect(typeof cancel).toBe('function');
        expect(scheduledTimeoutIds.size).toBe(1);

        cancel();
        expect(scheduledTimeoutIds.size).toBe(0);

        vi.advanceTimersByTime(60);
        expect(fired).toBe(false);
    });

    it('clears scheduled tween and timeout handles', () => {
        vi.useFakeTimers();

        const stopped = vi.fn();
        const tween = { stop: stopped };
        const scheduledDelayTweens = new Set([tween]);
        const timeoutId = setTimeout(() => {}, 1000);
        const scheduledTimeoutIds = new Set([timeoutId]);

        clearScheduledDelays(scheduledDelayTweens, scheduledTimeoutIds);

        expect(stopped).toHaveBeenCalledTimes(1);
        expect(scheduledDelayTweens.size).toBe(0);
        expect(scheduledTimeoutIds.size).toBe(0);
    });
});
