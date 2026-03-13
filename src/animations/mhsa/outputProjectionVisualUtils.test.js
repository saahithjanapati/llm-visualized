import * as THREE from 'three';
import { describe, expect, it } from 'vitest';

import { applyOutputProjectionPassVisual } from './outputProjectionVisualUtils.js';

describe('applyOutputProjectionPassVisual', () => {
    const startColor = new THREE.Color(0x202020);
    const activeColor = new THREE.Color(0xffaa33);

    function run(progress) {
        return applyOutputProjectionPassVisual({
            progress,
            startColor,
            activeColor,
            targetColor: new THREE.Color(),
            startEmissiveIntensity: 0.05,
            peakEmissiveIntensity: 0.8,
            endEmissiveIntensity: 0.3,
        });
    }

    it('starts from the resting state and ends at the settled active state', () => {
        const start = run(0);
        const end = run(1);

        expect(start.emissiveIntensity).toBeCloseTo(0.05, 6);
        expect(start.color.r).toBeCloseTo(startColor.r, 6);
        expect(start.color.g).toBeCloseTo(startColor.g, 6);
        expect(start.color.b).toBeCloseTo(startColor.b, 6);

        expect(end.emissiveIntensity).toBeCloseTo(0.3, 6);
        expect(end.color.r).toBeCloseTo(activeColor.r, 6);
        expect(end.color.g).toBeCloseTo(activeColor.g, 6);
        expect(end.color.b).toBeCloseTo(activeColor.b, 6);
    });

    it('ramps emissive intensity up and then back down over the pass-through', () => {
        const early = run(0.2);
        const peak = run(0.58);
        const late = run(0.85);

        expect(early.emissiveIntensity).toBeGreaterThan(0.05);
        expect(peak.emissiveIntensity).toBeCloseTo(0.8, 6);
        expect(late.emissiveIntensity).toBeLessThan(peak.emissiveIntensity);
        expect(late.emissiveIntensity).toBeGreaterThan(0.3);
    });

    it('finishes the color ramp early so the brightness falloff happens on the lit matrix', () => {
        const mid = run(0.4);
        const late = run(0.9);

        expect(mid.color.r).toBeCloseTo(activeColor.r, 6);
        expect(mid.color.g).toBeCloseTo(activeColor.g, 6);
        expect(mid.color.b).toBeCloseTo(activeColor.b, 6);
        expect(late.color.r).toBeCloseTo(activeColor.r, 6);
        expect(late.color.g).toBeCloseTo(activeColor.g, 6);
        expect(late.color.b).toBeCloseTo(activeColor.b, 6);
    });
});
