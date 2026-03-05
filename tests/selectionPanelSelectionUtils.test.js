import { describe, expect, it } from 'vitest';
import {
    getActivationDataFromSelection,
    isKvCacheVectorSelection,
    isLogitBarSelection,
    isSelfAttentionSelection,
    normalizeSelectionLabel,
    simplifyLayerNormParamDisplayLabel
} from '../src/ui/selectionPanelSelectionUtils.js';

describe('selectionPanelSelectionUtils', () => {
    it('resolves activation data from selection info object first', () => {
        const selection = {
            info: { activationData: { stage: 'attention.pre', tokenIndex: 1 } },
            object: { userData: { activationData: { stage: 'layer.incoming' } } }
        };
        const resolved = getActivationDataFromSelection(selection);
        expect(resolved?.stage).toBe('attention.pre');
        expect(resolved?.tokenIndex).toBe(1);
    });

    it('normalizes residual-related labels consistently', () => {
        expect(normalizeSelectionLabel('Post-layernorm residual')).toBe('Post LayerNorm Residual Vector');
        expect(normalizeSelectionLabel('Incoming residual stream')).toBe('Residual Stream Vector');
    });

    it('simplifies mlp projection labels to omit inline token text', () => {
        expect(simplifyLayerNormParamDisplayLabel('MLP Up Projection - hello')).toBe('MLP Up Projection');
        expect(simplifyLayerNormParamDisplayLabel('MLP Down Projection - world')).toBe('MLP Down Projection');
    });

    it('detects KV cache vectors from vectorRef metadata', () => {
        const selection = {
            info: {
                vectorRef: {
                    userData: { kvCachePersistent: true }
                }
            }
        };
        expect(isKvCacheVectorSelection(selection)).toBe(true);
    });

    it('classifies self-attention selections from activation stage', () => {
        const selection = {
            info: { activationData: { stage: 'attention.pre' } }
        };
        expect(isSelfAttentionSelection('anything', selection)).toBe(true);
        expect(isSelfAttentionSelection('post-layernorm residual', selection)).toBe(true);
        expect(isSelfAttentionSelection('post-layernorm residual', { info: { activationData: { stage: 'ln1.shift' } } })).toBe(false);
    });

    it('classifies logit bar selections from instance kind', () => {
        const selection = {
            object: { userData: { instanceKind: 'logitBar' } }
        };
        expect(isLogitBarSelection('Top logits', selection)).toBe(true);
        expect(isLogitBarSelection('unrelated label', { object: { userData: { instanceKind: 'other' } } })).toBe(false);
    });
});
