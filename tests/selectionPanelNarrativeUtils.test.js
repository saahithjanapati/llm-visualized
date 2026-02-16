import { describe, expect, it } from 'vitest';
import { resolveDescription, resolveSelectionEquations } from '../src/ui/selectionPanelNarrativeUtils.js';

describe('selectionPanelNarrativeUtils', () => {
    it('builds equations for known matrix labels', () => {
        const eq = resolveSelectionEquations('Query Weight Matrix', null);
        expect(eq).toContain('W_Q');
        expect(eq).toContain('softmax');
    });

    it('returns layernorm equation block for normalized vectors', () => {
        const eq = resolveSelectionEquations('LayerNorm Normed Output', null);
        expect(eq).toContain('\\frac');
        expect(eq).not.toContain('\\gamma');
    });

    it('resolves residual descriptions from activation stage context', () => {
        const desc = resolveDescription('anything', null, {
            info: {
                activationData: {
                    stage: 'residual.post_attention'
                }
            }
        });
        expect(desc).toContain('residual stream');
    });

    it('resolves attention-score-specific copy from stage', () => {
        const desc = resolveDescription('Attention Score', null, {
            info: {
                activationData: {
                    stage: 'attention.post'
                }
            }
        });
        expect(desc).toContain('normalized attention weight');
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
        expect(trailDesc).toContain('handoff path');
        expect(trailDesc).not.toContain('interactive scene component');

        const cachedKeyDesc = resolveDescription('Cached Key Vector', null, null);
        expect(cachedKeyDesc).toContain('KV cache');
        expect(cachedKeyDesc).not.toContain('interactive scene component');

        const cachedValueDesc = resolveDescription('Cached Value Vector', null, null);
        expect(cachedValueDesc).toContain('KV cache');
        expect(cachedValueDesc).not.toContain('interactive scene component');
    });
});
