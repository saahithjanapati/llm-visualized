import { describe, expect, it } from 'vitest';

import { resolveLiveSceneState } from './selectionPanelLiveSceneStateUtils.js';

describe('resolveLiveSceneState', () => {
    it('reports the active live layer and attention stage during an in-flight layer pass', () => {
        const pipeline = {
            _numLayers: 12,
            _currentLayerIdx: 4,
            _layers: Array.from({ length: 12 }, (_, index) => ({
                index,
                lanes: index === 4 ? [{ horizPhase: 'travelMHSA' }] : [],
                _mhsaStart: index === 4,
                mhsaAnimation: index === 4
                    ? { mhaPassThroughPhase: 'parallel_pass_through_active' }
                    : null,
                isActive: index === 4
            })),
            isForwardPassComplete() {
                return false;
            }
        };

        const state = resolveLiveSceneState({ pipeline });

        expect(state).toMatchObject({
            currentLayerIndex: 4,
            displayStage: 'Multi-Head Self-Attention',
            detailStageKey: 'qkv_projection',
            detailStageLabel: 'Q/K/V projections',
            isFinalStage: false,
            forwardPassComplete: false
        });
    });

    it('reports the output head once the pipeline has advanced past the last layer', () => {
        const lastLayer = {
            index: 11,
            lanes: [{
                originalVec: {
                    group: {
                        position: { y: 10 }
                    }
                }
            }],
            __topEmbedEntryYLocal: 5,
            __topEmbedExitYLocal: 9
        };
        const pipeline = {
            _numLayers: 12,
            _currentLayerIdx: 12,
            _layers: Array.from({ length: 12 }, (_, index) => (index === 11 ? lastLayer : { index, lanes: [] })),
            isForwardPassComplete() {
                return true;
            }
        };

        const state = resolveLiveSceneState({ pipeline });

        expect(state).toMatchObject({
            currentLayerIndex: 11,
            displayStage: 'Output Logits',
            detailStageLabel: 'Output logits / unembedding',
            isFinalStage: true,
            forwardPassComplete: true
        });
    });
});
