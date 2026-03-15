import { describe, expect, it } from 'vitest';

import { resolveLayerNormParameterSummary } from './selectionPanelLayerNormPreviewUtils.js';

function createLayer(index) {
    return {
        index,
        _getBaseVectorLength() {
            return 12;
        }
    };
}

describe('resolveLayerNormParameterSummary', () => {
    it.each([
        ['LayerNorm 1', 3],
        ['LayerNorm 2', 7]
    ])('uses GPT-2 hidden size for %s instead of prism display length', (label, layerIndex) => {
        const summary = resolveLayerNormParameterSummary(
            {
                label,
                info: { layerIndex }
            },
            {
                _layers: Array.from({ length: 12 }, (_, index) => createLayer(index))
            }
        );

        expect(summary.perParameterCount).toBe(768);
        expect(summary.totalParameterCount).toBe(1536);
        expect(summary.baseVectorLength).toBe(12);
    });
});
