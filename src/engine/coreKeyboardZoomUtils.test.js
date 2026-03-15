import { describe, expect, it } from 'vitest';

import {
    resolveKeyboardZoomStep,
    resolveKeyboardZoomTargetDistance
} from './coreKeyboardZoomUtils.js';

const BASE_OPTIONS = {
    direction: 1,
    deltaSeconds: 1 / 60,
    zoomSpeed: 0.68,
    minUnitsPerSecond: 1350,
    maxUnitsPerSecond: 5800,
    minDistance: 0.1,
    maxDistance: 80000
};

describe('coreKeyboardZoomUtils', () => {
    it('keeps close-range keyboard zoom near the existing base speed cap', () => {
        const zoomStep = resolveKeyboardZoomStep({
            ...BASE_OPTIONS,
            currentDistance: 16000
        });

        expect(zoomStep).toBeCloseTo((5800 / 60), 6);
    });

    it('boosts zoom-in speed when the camera is far from the target', () => {
        const nearStep = resolveKeyboardZoomStep({
            ...BASE_OPTIONS,
            currentDistance: 16000
        });
        const farStep = resolveKeyboardZoomStep({
            ...BASE_OPTIONS,
            currentDistance: 72000
        });

        expect(farStep).toBeGreaterThan(nearStep * 4);
    });

    it('still respects the configured minimum distance when zooming in', () => {
        const desiredDistance = resolveKeyboardZoomTargetDistance({
            ...BASE_OPTIONS,
            currentDistance: 1400,
            minDistance: 1390
        });

        expect(desiredDistance).toBe(1390);
    });

    it('does not apply the far-distance boost while zooming out', () => {
        const desiredDistance = resolveKeyboardZoomTargetDistance({
            ...BASE_OPTIONS,
            direction: -1,
            currentDistance: 72000
        });

        expect(desiredDistance).toBeCloseTo(72000 + (5800 / 60), 6);
    });
});
