import { describe, expect, it } from 'vitest';
import {
    resolveDescription,
    resolveSelectionEquations,
    resolveSelectionPreviewEquations
} from '../src/ui/selectionPanelNarrativeUtils.js';

describe('selectionPanelNarrativeUtils', () => {
    it('builds selection-focused equations for known matrix labels', () => {
        const eq = resolveSelectionEquations('Query Weight Matrix', null);
        expect(eq).toContain('W_Q');
        expect(eq).toContain('H_i =');
        expect(eq).not.toContain('s_{t,j}');
    });

    it('uses fixed 12-head concat indexing for output projection equations', () => {
        const eq = resolveSelectionEquations('Output Projection Matrix', null);
        expect(eq).toContain('H = \\mathrm{Concat}(H_i)_{i=1}^{12}');
        expect(eq).not.toContain('_{i=1}^{h}');
    });

    it('returns layernorm equation context for normalized vectors', () => {
        const eq = resolveSelectionEquations('LayerNorm Normed Output', null);
        expect(eq).toContain('\\frac');
        expect(eq).toContain('\\gamma');
    });

    it('renders layernorm preview equations as a single combined line', () => {
        const previewEq = resolveSelectionPreviewEquations('LayerNorm Normed Output', null);
        expect(previewEq).toHaveLength(1);
        expect(previewEq[0].tex).toContain('\\frac');
        expect(previewEq[0].tex).toContain('\\gamma');
    });

    it('shows selection-specific embedding equations instead of the full embedding bundle everywhere', () => {
        const vocabEq = resolveSelectionEquations('Vocabulary Embedding', null);
        const posEq = resolveSelectionEquations('Positional Embedding', null);
        const tokenEq = resolveSelectionEquations('Token: hello', null);
        const positionEq = resolveSelectionEquations('Position: 3', null);
        const embeddingSumEq = resolveSelectionEquations('Embedding Sum', null);

        expect(vocabEq).toContain('\\textcolor');
        expect(vocabEq).toContain('x_t^{\\text{tok}}');
        expect(vocabEq).toContain('E}[\\mathrm{token}_t]');
        expect(vocabEq).not.toContain('x_t^{\\text{pos}}');
        expect(posEq).toContain('x_t^{\\text{pos}}');
        expect(posEq).toContain('P}[t]');
        expect(tokenEq).toContain('x_t =');
        expect(tokenEq).not.toContain('P}[t]');
        expect(positionEq).toContain('x_t =');
        expect(positionEq).not.toContain('E}[\\mathrm{token}_t]');
        expect(embeddingSumEq).toContain('x_t^{\\text{tok}}');
        expect(embeddingSumEq).toContain('x_t^{\\text{pos}}');
        expect(embeddingSumEq).toContain('x_t =');
    });

    it('keeps top vocabulary embedding focused on logits equations', () => {
        const eq = resolveSelectionEquations('Vocabulary Embedding (Top)', null);
        expect(eq).toContain('softmax');
        expect(eq).not.toContain('x_t^{\\text{tok}}');
    });

    it('describes vocabulary embedding with GPT-2 vocabulary and model width', () => {
        const desc = resolveDescription('Vocabulary Embedding', null, null);
        expect(desc).toContain('50,257');
        expect(desc).toContain('768-dimensional vector');
        expect(desc).toContain('50,257 rows and 768 columns');
    });

    it('describes and equations weighted value vectors explicitly', () => {
        const eq = resolveSelectionEquations('Weighted Value Vector', null);
        expect(eq).toContain('H_i =');
        expect(eq).toContain('\\mathrm{softmax}');
        expect(eq).toContain('V}_i');
        expect(eq).not.toContain('\\tilde{V}_{t,j}');

        const desc = resolveDescription('Weighted Value Vector', null, {
            info: {
                activationData: {
                    stage: 'attention.weighted_value'
                }
            }
        });
        expect(desc).toContain('Multiplying that scalar by the value vector');
    });

    it('resolves residual descriptions from activation stage context', () => {
        const desc = resolveDescription('anything', null, {
            info: {
                activationData: {
                    stage: 'residual.post_attention'
                }
            }
        });
        expect(desc).toContain('residual-stream');
    });

    it('references numbered layer norms without calling them blocks', () => {
        const desc = resolveDescription('Residual Stream Vector', null, {
            info: {
                layerIndex: 5,
                activationData: {
                    stage: 'layer.incoming'
                }
            }
        });
        expect(desc).toContain('LayerNorm%201%20in%20layer%206');
        expect(desc).not.toContain('block in layer 6');
    });

    it('resolves attention-score-specific copy from stage', () => {
        const desc = resolveDescription('Attention Score', null, {
            info: {
                activationData: {
                    stage: 'attention.post'
                }
            }
        });
        expect(desc).toContain('post-softmax attention weight');
    });

    it('uses contextual fallback copy for unknown labeled components', () => {
        const desc = resolveDescription('Custom Debug Anchor', 'instanced', null);
        expect(desc).toContain('"Custom Debug Anchor"');
        expect(desc).toContain('instanced scene element');
        expect(desc).not.toContain('This is a GPT-2 component that transforms token representations');
    });

    it('uses activation-stage fallback copy for unknown stage-mapped selections', () => {
        const desc = resolveDescription('Unknown Selection', null, {
            info: {
                activationData: {
                    stage: 'debug.custom_stage'
                }
            }
        });
        expect(desc).toContain('debug.custom_stage');
        expect(desc).not.toContain('This is a GPT-2 component that transforms token representations');
    });

    it('provides explicit copy for connector and cached KV labels', () => {
        const trailDesc = resolveDescription('Embedding Connector Trail', null, null);
        expect(trailDesc).toContain('handoff legible');
        expect(trailDesc).not.toContain('interactive scene component');

        const cachedKeyDesc = resolveDescription('Cached Key Vector', null, null);
        expect(cachedKeyDesc).toContain('computed on an earlier decoding step and stored');
        expect(cachedKeyDesc).not.toContain('interactive scene component');

        const cachedValueDesc = resolveDescription('Cached Value Vector', null, null);
        expect(cachedValueDesc).toContain('future queries can still read from that token');
        expect(cachedValueDesc).not.toContain('interactive scene component');
    });

    it('marks the preview overlay equations that should be emphasized for a selection', () => {
        const queryPreview = resolveSelectionPreviewEquations('Query Weight Matrix', null);
        expect(queryPreview).toHaveLength(2);
        expect(queryPreview[0]).toMatchObject({ active: true });
        expect(queryPreview[1]).toMatchObject({ active: false });
        expect(queryPreview[1].tex).toContain('H_i =');
        expect(queryPreview[1].tex).not.toContain('s_{t,j}');

        const weightedValuePreview = resolveSelectionPreviewEquations('Weighted Value Vector', null);
        expect(weightedValuePreview.map((entry) => entry.active)).toEqual([true, false]);
        expect(weightedValuePreview[0].tex).toContain('H_i =');
        expect(weightedValuePreview[0].tex).not.toContain('\\tilde{V}_{t,j}');

        const attentionScorePreview = resolveSelectionPreviewEquations('Attention Score', null);
        expect(attentionScorePreview.map((entry) => entry.active)).toEqual([true, false]);
        expect(attentionScorePreview[0].tex).toContain('H_i =');
        expect(attentionScorePreview[0].tex).not.toContain('s_{t,j}');

        const embeddingPreview = resolveSelectionPreviewEquations('Token: hello', null);
        expect(embeddingPreview.map((entry) => entry.tex)).toHaveLength(2);
        expect(embeddingPreview[0]).toMatchObject({ active: true });
        expect(embeddingPreview[1]).toMatchObject({ active: false });
    });
});
