import { describe, expect, it } from 'vitest';

import {
    normalizeRaycastLabel,
    simplifyLayerNormParamHoverLabel
} from './coreRaycastLabels.js';

describe('normalizeRaycastLabel', () => {
    it('uses normalized residual stream labels for LayerNorm normalized stages', () => {
        expect(normalizeRaycastLabel('LN1 Normed', {
            activationData: {
                stage: 'ln1.norm'
            }
        })).toBe('Normalized Residual Stream Vector');

        expect(normalizeRaycastLabel('LN2 Normed', {
            activationData: {
                stage: 'ln2.norm'
            }
        })).toBe('Normalized Residual Stream Vector');
    });

    it('uses product-vector labels for LayerNorm product stages instead of parameter labels', () => {
        const info = {
            activationData: {
                stage: 'ln1.scale',
                layerNormKind: 'ln1'
            }
        };

        expect(normalizeRaycastLabel('LayerNorm 1 Scale', info)).toBe('LayerNorm 1 Product Vector');
        expect(simplifyLayerNormParamHoverLabel('LayerNorm 1 Scale', info)).toBe('LayerNorm 1 Product Vector');
    });
});
