import { describe, expect, it, vi } from 'vitest';

Object.defineProperty(globalThis, 'localStorage', {
    configurable: true,
    value: {
        getItem: vi.fn(() => null),
        setItem: vi.fn(),
        removeItem: vi.fn(),
    },
});

const { interpolateCounterValue } = await import('./parameterCounter.js');

describe('interpolateCounterValue', () => {
    it('clamps negative elapsed time to the start value', () => {
        expect(interpolateCounterValue(0, 38_597_376, -4, 900)).toBe(0);
    });

    it('still eases forward across the animation duration', () => {
        const midpointValue = interpolateCounterValue(0, 38_597_376, 450, 900);

        expect(midpointValue).toBeGreaterThan(0);
        expect(midpointValue).toBeLessThan(38_597_376);
        expect(interpolateCounterValue(0, 38_597_376, 900, 900)).toBe(38_597_376);
    });
});
