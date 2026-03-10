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
        expect(result?.summaryRows).toEqual([
            { label: 'Original vector length', value: '768' },
            { label: 'Samples used for color', value: '12' },
            { label: 'Sampling stride', value: '64' }
        ]);
        expect(result?.description).toContain('sampled source dimensions');
        expect(result?.tableRows?.[0]).toEqual({
            sourceIndex: 0,
            dimension: '0',
            value: '0'
        });
        expect(result?.text).toContain('Original vector length: 768');
        expect(result?.text).toContain('Samples used for color: 12');
        expect(result?.text).toContain('Sampling stride: 64');
        expect(result?.text).toContain('dim 0: 0');
        expect(result?.text).toContain('dim 704: 1.1');
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
        expect(result?.description).toContain('MLP segment');
        expect(result?.text).toContain('Original vector length: 3,072');
        expect(result?.text).toContain('MLP segment: 3/4 (dims 1,536-2,303)');
        expect(result?.text).toContain('Sampling stride: 64');
        expect(result?.text).toContain('dim 1,536: 0.01');
        expect(result?.text).toContain('dim 2,240: 0.12');
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
        expect(result?.text).toContain('dim 63: 63');
    });

    it('rounds displayed values and collapses negative zero to zero', () => {
        const result = resolveSelectionVectorSamplingData({
            label: 'Attention Weighted Sum',
            fallbackValues: Array.from({ length: 64 }, (_, index) => (
                [0.0004, 0.0009, -0.0004, -0.0014][index] ?? 0
            ))
        });

        expect(result?.tableRows?.slice(0, 4)).toEqual([
            { sourceIndex: 0, dimension: '0', value: '0' },
            { sourceIndex: 1, dimension: '1', value: '0.001' },
            { sourceIndex: 2, dimension: '2', value: '0' },
            { sourceIndex: 3, dimension: '3', value: '-0.001' }
        ]);
    });
});
