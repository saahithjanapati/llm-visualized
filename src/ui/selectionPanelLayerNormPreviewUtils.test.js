import { describe, expect, it } from 'vitest';

import {
    resolveLayerNormParameterSummary,
    resolveLiveLayerNormNormalizedPreviewSelection
} from './selectionPanelLayerNormPreviewUtils.js';

function createLayer(index, lanes = []) {
    return {
        index,
        lanes,
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

    it('hydrates LN1 normalized selections with the live in-scene vector', () => {
        const dupVec = {
            group: { visible: true },
            mesh: { isInstancedMesh: true },
            instanceCount: 12
        };
        const selection = {
            label: 'Normalized Residual Stream Vector',
            info: {
                layerIndex: 3,
                tokenIndex: 5,
                activationData: {
                    stage: 'ln1.norm',
                    layerIndex: 3,
                    tokenIndex: 5
                }
            }
        };
        const hydrated = resolveLiveLayerNormNormalizedPreviewSelection(selection, {
            _layers: Array.from({ length: 12 }, (_, index) => (
                index === 3
                    ? createLayer(index, [{
                        tokenIndex: 5,
                        dupVec,
                        normStarted: true,
                        multStarted: false
                    }])
                    : createLayer(index)
            ))
        });

        expect(hydrated?.info?.vectorRef).toBe(dupVec);
        expect(hydrated?.object).toBe(dupVec.mesh);
        expect(hydrated?.hit?.instanceId).toBe(0);
    });

    it('hydrates LN2 normalized selections with the live in-scene vector', () => {
        const movingVecLN2 = {
            group: { visible: true },
            mesh: { isInstancedMesh: true },
            instanceCount: 12
        };
        const selection = {
            label: 'Normalized Residual Stream Vector',
            info: {
                layerIndex: 7,
                tokenIndex: 2,
                activationData: {
                    stage: 'ln2.norm',
                    layerIndex: 7,
                    tokenIndex: 2
                }
            }
        };
        const hydrated = resolveLiveLayerNormNormalizedPreviewSelection(selection, {
            _layers: Array.from({ length: 12 }, (_, index) => (
                index === 7
                    ? createLayer(index, [{
                        tokenIndex: 2,
                        movingVecLN2,
                        normStartedLN2: true,
                        multDoneLN2: false
                    }])
                    : createLayer(index)
            ))
        });

        expect(hydrated?.info?.vectorRef).toBe(movingVecLN2);
        expect(hydrated?.object).toBe(movingVecLN2.mesh);
        expect(hydrated?.hit?.instanceId).toBe(0);
    });
});
