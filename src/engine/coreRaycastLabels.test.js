import { describe, expect, it } from 'vitest';

import { normalizeRaycastLabel } from './coreRaycastLabels.js';

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
});
