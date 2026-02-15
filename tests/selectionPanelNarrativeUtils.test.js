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
        expect(desc).toContain('residual stream vector');
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
});
