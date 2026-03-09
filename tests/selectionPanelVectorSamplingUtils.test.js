import { describe, expect, it } from 'vitest';
import { resolveSelectionVectorSamplingData } from '../src/ui/selectionPanelVectorSamplingUtils.js';

describe('selectionPanelVectorSamplingUtils', () => {
    it('maps residual stream samples back to their source dimensions', () => {
        const values = Array.from({ length: 12 }, (_, index) => index / 10);
        const result = resolveSelectionVectorSamplingData({
            label: 'Residual Stream Vector',
            selectionInfo: {
                info: {
                    activationData: {
                        stage: 'layer.incoming',
                        values
                    }
                }
            },
            activationSource: {
                meta: {
                    config: {
                        residual_stride: 64
                    }
                }
            }
        });

        expect(result?.title).toBe('Vector sampling');
        expect(result?.text).toContain('Original vector length: 768');
        expect(result?.text).toContain('Samples used for color: 12');
        expect(result?.text).toContain('Sampling stride: 64');
        expect(result?.text).toContain('dim 0: 0.0000');
        expect(result?.text).toContain('dim 704: 1.1000');
    });

    it('applies MLP segment offsets when formatting expanded vectors', () => {
        const values = Array.from({ length: 12 }, (_, index) => (index + 1) / 100);
        const result = resolveSelectionVectorSamplingData({
            label: 'MLP Activation - token',
            selectionInfo: {
                info: {
                    activationData: {
                        stage: 'mlp.activation',
                        segmentIndex: 2,
                        values
                    }
                }
            },
            activationSource: {
                meta: {
                    config: {
                        mlp_stride: 64
                    }
                }
            }
        });

        expect(result?.title).toBe('MLP segment sampling');
        expect(result?.text).toContain('Original vector length: 3,072');
        expect(result?.text).toContain('MLP segment: 3/4 (dims 1536-2303)');
        expect(result?.text).toContain('Sampling stride: 64');
        expect(result?.text).toContain('dim 1536: 0.0100');
        expect(result?.text).toContain('dim 2240: 0.1200');
    });

    it('falls back to full-length indexing when a vector is not stride-sampled', () => {
        const result = resolveSelectionVectorSamplingData({
            label: 'Attention Weighted Sum',
            fallbackValues: Array.from({ length: 64 }, (_, index) => index),
            activationSource: {
                meta: {
                    config: {
                        attention_stride: 64
                    }
                }
            }
        });

        expect(result?.title).toBe('Attention head sampling');
        expect(result?.text).toContain('Original vector length: 64');
        expect(result?.text).toContain('Samples used for color: 64');
        expect(result?.text).toContain('Sampling stride: 1');
        expect(result?.text).toContain('dim 63: 63.0000');
    });
});
