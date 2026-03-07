import { describe, expect, it } from 'vitest';
import { resolveLayerNormParamPreviewInstanceCount } from '../src/ui/selectionPanel.js';
import { PRISM_DIMENSIONS_PER_UNIT } from '../src/utils/constants.js';

describe('selectionPanel preview sizing', () => {
    it('uses the grouped vector ref count for layernorm parameter previews when available', () => {
        const selection = {
            info: {
                vectorRef: {
                    instanceCount: 12
                }
            }
        };

        expect(resolveLayerNormParamPreviewInstanceCount(selection, new Array(768).fill(0))).toBe(12);
    });

    it('groups raw layernorm parameter data into preview prisms when no vector ref is available', () => {
        const rawData = new Array(PRISM_DIMENSIONS_PER_UNIT * 2 + 1).fill(0);

        expect(resolveLayerNormParamPreviewInstanceCount(null, rawData)).toBe(3);
    });
});
