import { describe, expect, it } from 'vitest';

import { resolveSelectionVectorSamplingData } from './selectionPanelVectorSamplingUtils.js';

function createSegmentValues(seed = 0, length = 12) {
    return Array.from({ length }, (_, index) => seed + index + 1);
}

function createBatchedMlpSelection(stage = 'mlp.activation') {
    const refs = [0, 1, 2, 3].map((segmentIndex) => ({
        _index: segmentIndex,
        rawData: createSegmentValues(segmentIndex * 100),
        userData: {
            activationData: {
                stage,
                segmentIndex,
                values: createSegmentValues(segmentIndex * 100)
            }
        },
        group: { userData: {} },
        mesh: { userData: {} }
    }));
    const batch = { _vectorRefs: [refs[2], refs[0], refs[3], refs[1]] };
    refs.forEach((ref) => {
        ref.isBatchedVectorRef = true;
        ref._batch = batch;
    });
    return {
        label: 'MLP Activation (post GELU)',
        info: {
            vectorRef: refs[2],
            activationData: refs[2].userData.activationData
        }
    };
}

describe('resolveSelectionVectorSamplingData', () => {
    it('aggregates all MLP segment values for batched expansion selections', () => {
        const data = resolveSelectionVectorSamplingData({
            label: 'MLP Activation (post GELU)',
            selectionInfo: createBatchedMlpSelection(),
            activationSource: {
                meta: {
                    config: {
                        mlp_stride: 64
                    }
                }
            }
        });

        expect(data?.title).toBe('MLP expansion sampling');
        expect(data?.summaryRows).toContainEqual({
            label: 'Original vector length',
            value: '3,072'
        });
        expect(data?.summaryRows).toContainEqual({
            label: 'Samples used for color',
            value: '48'
        });
        expect(data?.summaryRows).toContainEqual({
            label: 'Sampling stride',
            value: '64'
        });
        expect(data?.tableRows).toHaveLength(48);
        expect(data?.tableRows?.[0]).toMatchObject({
            dimension: '0',
            value: '1'
        });
        expect(data?.tableRows?.[12]).toMatchObject({
            dimension: '768',
            value: '101'
        });
        expect(data?.tableRows?.[47]).toMatchObject({
            dimension: '3,008',
            value: '312'
        });
    });

    it('keeps single-segment MLP sampling when no batched expansion is available', () => {
        const values = createSegmentValues(200);
        const data = resolveSelectionVectorSamplingData({
            label: 'MLP Activation (post GELU)',
            selectionInfo: {
                info: {
                    activationData: {
                        stage: 'mlp.activation',
                        segmentIndex: 2,
                        values
                    }
                }
            },
            activationSource: {
                meta: {
                    config: {
                        mlp_stride: 64
                    }
                }
            }
        });

        expect(data?.title).toBe('MLP segment sampling');
        expect(data?.summaryRows).toContainEqual({
            label: 'MLP segment',
            value: '3/4 (dims 1,536-2,303)'
        });
        expect(data?.summaryRows).toContainEqual({
            label: 'Samples used for color',
            value: '12'
        });
        expect(data?.tableRows).toHaveLength(12);
        expect(data?.tableRows?.[0]).toMatchObject({
            dimension: '1,536',
            value: '201'
        });
    });
});
