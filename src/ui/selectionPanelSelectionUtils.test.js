import { describe, expect, it } from 'vitest';

import { normalizeSelectionLabel } from './selectionPanelSelectionUtils.js';

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
});
