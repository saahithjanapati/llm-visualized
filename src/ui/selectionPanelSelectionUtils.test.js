import { describe, expect, it } from 'vitest';

import {
    normalizeSelectionLabel,
    simplifyLayerNormParamDisplayLabel
} from './selectionPanelSelectionUtils.js';

describe('normalizeSelectionLabel', () => {
    it('renames LayerNorm normalized-stage selections to normalized residual stream vectors', () => {
        expect(normalizeSelectionLabel('LN1 Normed', {
            info: {
                activationData: {
                    stage: 'ln1.norm'
                }
            }
        })).toBe('Normalized Residual Stream Vector');

        expect(normalizeSelectionLabel('LN2 Normed', {
            info: {
                activationData: {
                    stage: 'ln2.norm'
                }
            }
        })).toBe('Normalized Residual Stream Vector');
    });

    it('renames LayerNorm product-stage selections to product vectors instead of parameter labels', () => {
        const selection = {
            info: {
                activationData: {
                    stage: 'ln1.scale',
                    layerNormKind: 'ln1'
                }
            }
        };

        expect(normalizeSelectionLabel('LayerNorm 1 Scale', selection)).toBe('LayerNorm 1 Product Vector');
        expect(simplifyLayerNormParamDisplayLabel('LayerNorm 1 Scale', selection)).toBe('LayerNorm 1 Product Vector');
    });
});
