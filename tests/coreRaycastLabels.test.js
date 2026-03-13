import { describe, expect, it } from 'vitest';
import {
    isCachedKvSelection,
    normalizeRaycastLabel,
    simplifyLayerNormParamHoverLabel
} from '../src/engine/coreRaycastLabels.js';

describe('coreRaycastLabels', () => {
    it('normalizes residual labels', () => {
        expect(normalizeRaycastLabel('post-attention residual')).toBe('Residual Stream Vector');
        expect(normalizeRaycastLabel('post-layernorm residual')).toBe('Post LayerNorm Residual Vector');
        expect(normalizeRaycastLabel('anything else')).toBe('anything else');
    });

    it('simplifies layernorm param labels', () => {
        expect(simplifyLayerNormParamHoverLabel('LayerNorm scale matrix')).toBe('LayerNorm Scale');
        expect(simplifyLayerNormParamHoverLabel('LayerNorm shift matrix')).toBe('LayerNorm Shift');
        expect(simplifyLayerNormParamHoverLabel('Post LayerNorm Residual Vector', {
            activationData: { stage: 'ln1.shift' }
        })).toBe('Post LayerNorm Residual Vector');
        expect(simplifyLayerNormParamHoverLabel('Query Weight Matrix')).toBe('Query Weight Matrix');
    });

    it('uses numbered layernorm names in hover labels when context is available', () => {
        expect(simplifyLayerNormParamHoverLabel('LayerNorm', null, {
            userData: { layerNormKind: 'ln1' },
            parent: null
        })).toBe('LayerNorm 1');
        expect(simplifyLayerNormParamHoverLabel('LayerNorm shift matrix', {
            activationData: { stage: 'ln2.param.shift' }
        })).toBe('LayerNorm 2 Shift');
        expect(simplifyLayerNormParamHoverLabel('LN2 Normed', {
            activationData: { stage: 'ln2.norm' }
        })).toBe('LayerNorm 2 Normed');
    });

    it('detects cached kv markers from vectorRef and object chain', () => {
        const infoHit = isCachedKvSelection({
            vectorRef: { userData: { cachedKv: true } }
        });
        expect(infoHit).toBe(true);

        const parent = { userData: { kvCachePersistent: true }, parent: null };
        const child = { userData: {}, parent };
        expect(isCachedKvSelection(null, child)).toBe(true);
        expect(isCachedKvSelection(null, { userData: {}, parent: null })).toBe(false);
    });
});
