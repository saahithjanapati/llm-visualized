import { describe, expect, it } from 'vitest';
import {
    describeTransformerView2dTarget,
    resolveTransformerView2dActionContext
} from '../src/ui/selectionPanelTransformerView2d.js';

describe('selectionPanelTransformerView2d', () => {
    it('maps embedding selections into embedding-focused 2D targets', () => {
        const context = resolveTransformerView2dActionContext({
            label: 'Token embedding',
            info: {
                activationData: {
                    stage: 'embedding.token'
                }
            }
        });

        expect(context?.semanticTarget).toEqual({
            componentKind: 'embedding',
            stage: 'token',
            role: 'token-embedding'
        });
        expect(context?.focusLabel).toBe('Token embeddings');
    });

    it('maps numbered layer norm selections by activation stage', () => {
        const context = resolveTransformerView2dActionContext({
            label: 'LayerNorm shift',
            info: {
                layerIndex: 5,
                activationData: {
                    layerIndex: 5,
                    stage: 'ln2.shift'
                }
            }
        });

        expect(context?.semanticTarget).toEqual({
            componentKind: 'layer-norm',
            layerIndex: 5,
            stage: 'ln2',
            role: 'module'
        });
        expect(context?.focusLabel).toBe('Layer 6 LayerNorm 2');
    });

    it('focuses MHSA heads when attention selections carry a head index', () => {
        const context = resolveTransformerView2dActionContext({
            label: 'Query vector',
            info: {
                layerIndex: 3,
                headIndex: 8,
                activationData: {
                    layerIndex: 3,
                    headIndex: 8,
                    stage: 'qkv.q'
                }
            }
        });

        expect(context?.semanticTarget).toEqual({
            componentKind: 'mhsa',
            layerIndex: 3,
            headIndex: 8,
            stage: 'attention',
            role: 'head'
        });
        expect(context?.focusLabel).toBe('Layer 4 MHSA Head 9');
    });

    it('maps output projection matrix selections to the projection weight target', () => {
        const context = resolveTransformerView2dActionContext({
            label: 'Output Projection Matrix',
            info: {
                layerIndex: 2,
                activationData: {
                    layerIndex: 2
                }
            }
        });

        expect(context?.semanticTarget).toEqual({
            componentKind: 'output-projection',
            layerIndex: 2,
            stage: 'attn-out',
            role: 'projection-weight'
        });
        expect(context?.focusLabel).toBe('Layer 3 output projection');
    });

    it('maps MLP projection selections to the correct stage', () => {
        const context = resolveTransformerView2dActionContext({
            label: 'MLP Up Weight Matrix',
            info: {
                layerIndex: 7,
                activationData: {
                    layerIndex: 7
                }
            }
        });

        expect(context?.semanticTarget).toEqual({
            componentKind: 'mlp',
            layerIndex: 7,
            stage: 'mlp-up',
            role: 'mlp-up'
        });
        expect(context?.focusLabel).toBe('Layer 8 MLP up projection');
    });

    it('maps logits selections and top unembedding labels into output-space targets', () => {
        const logitContext = resolveTransformerView2dActionContext({
            label: 'Logit',
            kind: 'logitBar'
        });
        const unembeddingContext = resolveTransformerView2dActionContext({
            label: 'Vocabulary Embedding (Top)'
        });

        expect(logitContext?.semanticTarget).toEqual({
            componentKind: 'logits',
            stage: 'output',
            role: 'logits-topk'
        });
        expect(unembeddingContext?.semanticTarget).toEqual({
            componentKind: 'logits',
            stage: 'output',
            role: 'unembedding'
        });
    });

    it('describes residual and final layer norm targets with human-readable labels', () => {
        expect(describeTransformerView2dTarget({
            componentKind: 'residual',
            layerIndex: 4,
            stage: 'post-attn-add'
        })).toBe('Layer 5 post-attention residual');

        expect(describeTransformerView2dTarget({
            componentKind: 'layer-norm',
            stage: 'final-ln',
            role: 'module'
        })).toBe('Final LayerNorm');
    });
});
