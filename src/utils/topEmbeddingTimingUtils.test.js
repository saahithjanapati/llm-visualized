import { describe, expect, it } from 'vitest';

import {
    easeTopEmbeddingActivationProgress,
    getTopEmbeddingActivationEasedProgress,
    getTopEmbeddingActivationProgress,
    resolveTopEmbeddingActivationWindow,
    resolveTopLogitRevealY
} from './topEmbeddingTimingUtils.js';

describe('topEmbeddingTimingUtils', () => {
    it('keeps the activation start slightly inside the unembedding span', () => {
        const window = resolveTopEmbeddingActivationWindow(100, 200);

        expect(window).not.toBeNull();
        expect(window.activationStartY).toBeGreaterThan(100);
        expect(window.activationStartY).toBeLessThan(200);
    });

    it('reports zero progress before the activation window and one at the exit', () => {
        expect(getTopEmbeddingActivationProgress(100, 200, 100)).toBe(0);
        expect(getTopEmbeddingActivationProgress(100, 200, 200)).toBe(1);
    });

    it('uses the same smoothstep easing as the top unembedding color tween', () => {
        expect(easeTopEmbeddingActivationProgress(0)).toBe(0);
        expect(easeTopEmbeddingActivationProgress(0.5)).toBeCloseTo(0.5, 6);
        expect(easeTopEmbeddingActivationProgress(1)).toBe(1);
    });

    it('starts the logit reveal after activation has visibly begun', () => {
        const window = resolveTopEmbeddingActivationWindow(100, 200);
        const revealY = resolveTopLogitRevealY(100, 200);

        expect(window).not.toBeNull();
        expect(revealY).toBeGreaterThan(window.activationStartY);
        expect(getTopEmbeddingActivationEasedProgress(100, 200, revealY)).toBeGreaterThan(0.1);
    });

    it('handles zero-length spans without producing NaN progress', () => {
        expect(resolveTopLogitRevealY(120, 120)).toBe(120);
        expect(getTopEmbeddingActivationProgress(120, 120, 120)).toBe(1);
        expect(getTopEmbeddingActivationEasedProgress(120, 120, 120)).toBe(1);
    });
});

