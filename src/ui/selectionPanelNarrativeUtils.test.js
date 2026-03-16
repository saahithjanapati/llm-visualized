import { describe, expect, it } from 'vitest';

import {
    resolveDescription,
    resolveSelectionPreviewEquations
} from './selectionPanelNarrativeUtils.js';
import {
    MLP_DOWN_BIAS_TOOLTIP_LABEL,
    MLP_UP_BIAS_TOOLTIP_LABEL
} from '../utils/mlpLabels.js';

describe('selectionPanelNarrativeUtils bias narratives', () => {
    it('returns no equation preview for incoming residual stream vectors', () => {
        const selection = {
            label: 'Residual Stream Vector',
            info: {
                activationData: {
                    stage: 'layer.incoming',
                    layerIndex: 4,
                    tokenIndex: 1,
                    tokenLabel: 'the'
                }
            }
        };

        const entries = resolveSelectionPreviewEquations('Residual Stream Vector', selection);
        expect(entries).toEqual([]);
    });

    it('returns no equation preview for post-layernorm residual vectors', () => {
        const selection = {
            label: 'Post LayerNorm 1 Residual Vector',
            info: {
                activationData: {
                    stage: 'ln1.output',
                    layerIndex: 4,
                    tokenIndex: 1,
                    tokenLabel: 'the'
                }
            }
        };

        const entries = resolveSelectionPreviewEquations('Post LayerNorm 1 Residual Vector', selection);
        expect(entries).toEqual([]);
    });

    it('describes normalized residual stream vectors as post-normalization states', () => {
        const selection = {
            label: 'Normalized Residual Stream Vector',
            info: {
                activationData: {
                    stage: 'ln1.norm',
                    layerIndex: 4,
                    tokenIndex: 1,
                    tokenLabel: 'the'
                }
            }
        };

        const description = resolveDescription('Normalized Residual Stream Vector', 'vector', selection);
        expect(description).toContain('normalization step');
        expect(description).toContain('already completed');
        expect(description).toContain('scale and shift');
    });

    it('describes LayerNorm product-vector stages as intermediate affine states', () => {
        const selection = {
            label: 'LayerNorm 1 Product Vector',
            info: {
                activationData: {
                    stage: 'ln1.product',
                    layerIndex: 4,
                    tokenIndex: 1,
                    tokenLabel: 'the'
                }
            }
        };

        const description = resolveDescription('LayerNorm 1 Product Vector', 'vector', selection);
        expect(description).toContain('after normalization');
        expect(description).toContain('multiplication by the learned scale vector');
        expect(description).toContain('before the learned shift');
    });

    it('treats final_ln.scale as the final LayerNorm product-vector stage, not the gamma parameter', () => {
        const selection = {
            label: 'LayerNorm (Top) Product Vector',
            info: {
                activationData: {
                    stage: 'final_ln.scale',
                    tokenIndex: 1,
                    tokenLabel: 'the'
                }
            }
        };

        const description = resolveDescription('LayerNorm (Top) Product Vector', 'vector', selection);
        expect(description).toContain('after normalization');
        expect(description).toContain('before the learned shift');
        expect(description).not.toContain('learned scale vector for the final LayerNorm');
    });

    it('returns bias-specific copy for the MLP up bias term', () => {
        const selection = {
            label: MLP_UP_BIAS_TOOLTIP_LABEL,
            info: {
                activationData: {
                    stage: 'mlp.up.bias',
                    layerIndex: 4
                }
            }
        };

        const description = resolveDescription(MLP_UP_BIAS_TOOLTIP_LABEL, 'vector', selection);
        expect(description).toContain('b_{\\text{up}}');
        expect(description).toContain('3,072');
        expect(description).toContain('shared across every token position');
    });

    it('returns bias-specific copy for the MLP down bias term', () => {
        const selection = {
            label: MLP_DOWN_BIAS_TOOLTIP_LABEL,
            info: {
                activationData: {
                    stage: 'mlp.down.bias',
                    layerIndex: 4
                }
            }
        };

        const description = resolveDescription(MLP_DOWN_BIAS_TOOLTIP_LABEL, 'vector', selection);
        expect(description).toContain('b_{\\text{down}}');
        expect(description).toContain('768');
        expect(description).toContain('shared across all token positions');
    });

    it('highlights the up-projection equation for b_up selections', () => {
        const selection = {
            label: MLP_UP_BIAS_TOOLTIP_LABEL,
            info: {
                activationData: {
                    stage: 'mlp.up.bias'
                }
            }
        };

        const entries = resolveSelectionPreviewEquations(MLP_UP_BIAS_TOOLTIP_LABEL, selection);
        expect(entries).toHaveLength(3);
        expect(entries[0]?.active).toBe(true);
        expect(entries[0]?.tex).toContain('b_{\\text{up}}');
    });

    it('highlights the down-projection equation for b_down selections', () => {
        const selection = {
            label: MLP_DOWN_BIAS_TOOLTIP_LABEL,
            info: {
                activationData: {
                    stage: 'mlp.down.bias'
                }
            }
        };

        const entries = resolveSelectionPreviewEquations(MLP_DOWN_BIAS_TOOLTIP_LABEL, selection);
        expect(entries).toHaveLength(3);
        expect(entries[1]?.active).toBe(true);
        expect(entries[1]?.tex).toContain('b_{\\text{down}}');
    });

    it('returns query-bias-specific copy and highlights the Q projection equation', () => {
        const selection = {
            label: 'Query Bias Vector',
            info: {
                activationData: {
                    stage: 'qkv.q.bias',
                    layerIndex: 2,
                    headIndex: 1
                }
            }
        };

        const description = resolveDescription('Query Bias Vector', 'vector', selection);
        expect(description).toContain('query bias vector');
        expect(description).toContain('64');
        expect(description).toContain('same bias is reused at every token position');

        const entries = resolveSelectionPreviewEquations('Query Bias Vector', selection);
        expect(entries).toHaveLength(2);
        expect(entries[0]?.active).toBe(true);
        expect(entries[0]?.tex).toContain('b_{Q_i}');
    });

    it('returns value-bias-specific copy and highlights the V projection equation', () => {
        const selection = {
            label: 'Value Bias Vector',
            info: {
                activationData: {
                    stage: 'qkv.v.bias',
                    layerIndex: 2,
                    headIndex: 1
                }
            }
        };

        const description = resolveDescription('Value Bias Vector', 'vector', selection);
        expect(description).toContain('value bias vector');
        expect(description).toContain('64');
        expect(description).toContain('same bias is shared across tokens');

        const entries = resolveSelectionPreviewEquations('Value Bias Vector', selection);
        expect(entries).toHaveLength(2);
        expect(entries[0]?.active).toBe(true);
        expect(entries[0]?.tex).toContain('b_{V_i}');
    });

    it('returns output-projection-bias-specific copy and highlights the output projection equation', () => {
        const selection = {
            label: 'Output Projection Bias Vector',
            info: {
                activationData: {
                    stage: 'attention.output_projection.bias',
                    layerIndex: 2
                }
            }
        };

        const description = resolveDescription('Output Projection Bias Vector', 'vector', selection);
        expect(description).toContain('b_O');
        expect(description).toContain('768');
        expect(description).toContain('shared across every token position');

        const entries = resolveSelectionPreviewEquations('Output Projection Bias Vector', selection);
        expect(entries).toHaveLength(3);
        expect(entries[1]?.active).toBe(true);
        expect(entries[1]?.tex).toContain('b_O');
    });

    it('returns H_i-specific copy for attention weighted sums', () => {
        const selection = {
            label: 'Attention Weighted Sum',
            info: {
                activationData: {
                    stage: 'attention.weighted_sum',
                    layerIndex: 2,
                    headIndex: 1,
                    tokenIndex: 3,
                    tokenLabel: 'the'
                }
            }
        };

        const description = resolveDescription('Attention Weighted Sum', 'vector', selection);
        expect(description).toContain('64-dimensional');
        expect(description).toContain('H_i');
        expect(description).toContain('post-softmax attention weights');
        expect(description).toContain('output projection');
    });

    it('returns blocked causal-mask-specific copy and highlights the attention equation', () => {
        const selection = {
            label: 'Causal Mask',
            info: {
                activationData: {
                    stage: 'attention.mask',
                    layerIndex: 2,
                    headIndex: 1,
                    queryTokenIndex: 2,
                    queryTokenLabel: 'cat',
                    keyTokenIndex: 4,
                    keyTokenLabel: 'sat',
                    maskValue: Number.NEGATIVE_INFINITY,
                    isMasked: true
                }
            }
        };

        const description = resolveDescription('Causal Mask', 'attentionSphere', selection);
        expect(description).toContain('M_{\\mathrm{causal}}');
        expect(description).toContain('-\\infty');
        expect(description).toContain('future position');
        expect(description).toContain('next-token prediction');

        const entries = resolveSelectionPreviewEquations('Causal Mask', selection);
        expect(entries).toHaveLength(2);
        expect(entries[0]?.active).toBe(true);
        expect(entries[0]?.tex).toContain('+ M');
    });

    it('returns allowed causal-mask-specific copy when the mask value is zero', () => {
        const selection = {
            label: 'Causal Mask',
            info: {
                activationData: {
                    stage: 'attention.mask',
                    layerIndex: 2,
                    headIndex: 1,
                    queryTokenIndex: 4,
                    queryTokenLabel: 'sat',
                    keyTokenIndex: 2,
                    keyTokenLabel: 'cat',
                    maskValue: 0
                }
            }
        };

        const description = resolveDescription('Causal Mask', 'attentionSphere', selection);
        expect(description).toContain('$0$');
        expect(description).toContain('allowed');
        expect(description).toContain('leaves the raw query-key score unchanged');
    });
});
