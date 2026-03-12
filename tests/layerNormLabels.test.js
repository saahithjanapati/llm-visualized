import { describe, expect, it } from 'vitest';
import { resolveLayerNormParamSpec } from '../src/utils/layerNormLabels.js';

describe('layerNormLabels', () => {
    it('resolves layernorm parameter vectors from param stages', () => {
        expect(resolveLayerNormParamSpec({
            label: 'LayerNorm 2 Scale',
            stage: 'ln2.param.scale'
        })).toEqual({
            layerNormKind: 'ln2',
            param: 'scale'
        });

        expect(resolveLayerNormParamSpec({
            label: 'Final LN Shift',
            stage: 'final_ln.param.shift'
        })).toEqual({
            layerNormKind: 'final',
            param: 'shift'
        });
    });

    it('does not treat token-state layernorm outputs as shared parameter vectors', () => {
        expect(resolveLayerNormParamSpec({
            label: 'Post LayerNorm Residual Vector',
            stage: 'ln1.shift'
        })).toBeNull();
    });
});
