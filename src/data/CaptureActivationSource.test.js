import { describe, expect, it } from 'vitest';

import { CaptureActivationSource } from './CaptureActivationSource.js';

describe('CaptureActivationSource.getAttentionScore', () => {
    it('does not clamp masked upper-triangular packed post-softmax lookups to the diagonal', () => {
        const source = new CaptureActivationSource({
            activations: {
                layers: [{
                    attention_scores: {
                        post: [{
                            v: [0.6, 0.25, 0.75, 1],
                            n: 3
                        }]
                    }
                }]
            }
        });

        expect(source.getAttentionScore(0, 'post', 0, 0, 0)).toBe(0.6);
        expect(source.getAttentionScore(0, 'post', 0, 1, 1)).toBe(0.75);
        expect(source.getAttentionScore(0, 'post', 0, 0, 2)).toBeNull();
    });
});
