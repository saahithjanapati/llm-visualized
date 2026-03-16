import { describe, expect, it } from 'vitest';

import {
    D_HEAD,
    D_MODEL
} from '../../ui/selectionPanelConstants.js';
import {
    flattenSceneNodes,
    VIEW2D_NODE_KINDS
} from '../schema/sceneTypes.js';
import { buildTransformerSceneModel } from './buildTransformerSceneModel.js';

function createVector(seed = 0, length = D_MODEL) {
    return Array.from({ length }, (_, index) => Number((seed + (index * 0.01)).toFixed(4)));
}

function resolveStageOffset(stage = '') {
    const lower = String(stage || '').trim().toLowerCase();
    if (lower === 'norm') return 0.1;
    if (lower === 'scale') return 0.2;
    if (lower === 'shift') return 0.3;
    return 0;
}

function createMockActivationSource({
    promptTokenCount = 6,
    completionTokenCount = 1
} = {}) {
    const safePromptTokenCount = Math.max(0, Math.floor(promptTokenCount));
    const safeCompletionTokenCount = Math.max(0, Math.floor(completionTokenCount));
    const tokenCount = safePromptTokenCount + safeCompletionTokenCount;
    return {
        meta: {
            prompt_tokens: Array.from({ length: safePromptTokenCount }, (_, index) => 101 + index),
            completion_tokens: Array.from({ length: safeCompletionTokenCount }, (_, index) => 401 + index)
        },
        getTokenCount() {
            return tokenCount;
        },
        getTokenId(tokenIndex = 0) {
            return 100 + tokenIndex;
        },
        getLayerIncoming(_layerIndex = 0, tokenIndex = 0, targetLength = D_MODEL) {
            return createVector(0.1 + tokenIndex, targetLength);
        },
        getLayerLn1(_layerIndex = 0, stage = 'shift', tokenIndex = 0, targetLength = D_MODEL) {
            return createVector(0.4 + resolveStageOffset(stage) + tokenIndex, targetLength);
        },
        getPostAttentionResidual(_layerIndex = 0, tokenIndex = 0, targetLength = D_MODEL) {
            return createVector(0.8 + tokenIndex, targetLength);
        },
        getLayerLn2(_layerIndex = 0, stage = 'shift', tokenIndex = 0, targetLength = D_MODEL) {
            return createVector(1.1 + resolveStageOffset(stage) + tokenIndex, targetLength);
        },
        getAttentionWeightedSum(_layerIndex = 0, headIndex = 0, tokenIndex = 0, targetLength = D_HEAD) {
            return createVector(1.5 + (headIndex * 0.1) + tokenIndex, targetLength);
        },
        getAttentionOutputProjection(_layerIndex = 0, tokenIndex = 0, targetLength = D_MODEL) {
            return createVector(1.9 + tokenIndex, targetLength);
        },
        getMlpUp(_layerIndex = 0, tokenIndex = 0, targetLength = D_MODEL * 4) {
            return createVector(2.3 + tokenIndex, targetLength);
        },
        getMlpActivation(_layerIndex = 0, tokenIndex = 0, targetLength = D_MODEL * 4) {
            return createVector(2.7 + tokenIndex, targetLength);
        },
        getMlpDown(_layerIndex = 0, tokenIndex = 0, targetLength = D_MODEL) {
            return createVector(3.1 + tokenIndex, targetLength);
        },
        getPostMlpResidual(_layerIndex = 0, tokenIndex = 0, targetLength = D_MODEL) {
            return createVector(3.5 + tokenIndex, targetLength);
        },
        getFinalLayerNorm(stage = 'shift', tokenIndex = 0, targetLength = D_MODEL) {
            return createVector(3.9 + resolveStageOffset(stage) + tokenIndex, targetLength);
        }
    };
}

function createDecodeKvState(passIndex = 1) {
    return {
        kvCacheModeEnabled: true,
        kvCachePrefillActive: false,
        kvCacheDecodeActive: true,
        kvCachePassIndex: passIndex
    };
}

function createPrefillKvState() {
    return {
        kvCacheModeEnabled: true,
        kvCachePrefillActive: true,
        kvCacheDecodeActive: false,
        kvCachePassIndex: 0
    };
}

function findOverviewResidualNodes(scene = null) {
    return flattenSceneNodes(scene).filter((node) => (
        node?.kind === VIEW2D_NODE_KINDS.MATRIX
        && node?.role === 'module-card'
        && node?.semantic?.componentKind === 'residual'
    ));
}

describe('buildTransformerSceneModel KV-cache decode window', () => {
    it('collapses overview embeddings and residual stream cards to the live decode token', () => {
        const scene = buildTransformerSceneModel({
            activationSource: createMockActivationSource(),
            tokenIndices: [3, 4, 5, 6],
            tokenLabels: ['Token D', 'Token E', 'Token F', 'Token G'],
            layerCount: 1,
            kvCacheState: createDecodeKvState()
        });

        const nodes = flattenSceneNodes(scene);
        const tokenChipNodes = nodes.filter((node) => (
            node?.kind === VIEW2D_NODE_KINDS.MATRIX
            && node?.role === 'input-token-chip'
        ));
        const positionChipNodes = nodes.filter((node) => (
            node?.kind === VIEW2D_NODE_KINDS.MATRIX
            && node?.role === 'input-position-chip'
        ));
        const residualNodes = findOverviewResidualNodes(scene);

        expect(scene?.metadata?.tokenCount).toBe(1);
        expect(scene?.metadata?.tokenIndices).toEqual([6]);
        expect(scene?.metadata?.kvCacheState).toMatchObject({
            kvCacheModeEnabled: true,
            kvCachePrefillActive: false,
            kvCacheDecodeActive: true,
            kvCachePassIndex: 1
        });
        expect(tokenChipNodes).toHaveLength(1);
        expect(positionChipNodes).toHaveLength(1);
        expect(tokenChipNodes[0]?.semantic?.tokenIndex).toBe(6);
        expect(positionChipNodes[0]?.semantic?.tokenIndex).toBe(6);
        expect(residualNodes.length).toBeGreaterThan(0);
        residualNodes.forEach((node) => {
            expect(node?.dimensions?.rows).toBe(1);
            expect(node?.rowItems).toHaveLength(1);
            expect(node?.rowItems?.[0]?.semantic?.tokenIndex).toBe(6);
        });
    });

    it('keeps the full token window during KV-cache prefill', () => {
        const scene = buildTransformerSceneModel({
            activationSource: createMockActivationSource(),
            tokenIndices: [3, 4, 5, 6],
            tokenLabels: ['Token D', 'Token E', 'Token F', 'Token G'],
            layerCount: 1,
            kvCacheState: createPrefillKvState()
        });

        const nodes = flattenSceneNodes(scene);
        const tokenChipNodes = nodes.filter((node) => (
            node?.kind === VIEW2D_NODE_KINDS.MATRIX
            && node?.role === 'input-token-chip'
        ));
        const residualNodes = findOverviewResidualNodes(scene);

        expect(scene?.metadata?.tokenCount).toBe(4);
        expect(scene?.metadata?.tokenIndices).toEqual([3, 4, 5, 6]);
        expect(scene?.metadata?.kvCacheState).toMatchObject({
            kvCacheModeEnabled: true,
            kvCachePrefillActive: true,
            kvCacheDecodeActive: false,
            kvCachePassIndex: 0
        });
        expect(tokenChipNodes).toHaveLength(4);
        residualNodes.forEach((node) => {
            expect(node?.dimensions?.rows).toBe(4);
            expect(node?.rowItems).toHaveLength(4);
        });
    });

    it('collapses non-MHSA detail scenes to single-row matrices during decode', () => {
        const activationSource = createMockActivationSource();
        const baseConfig = {
            activationSource,
            tokenIndices: [3, 4, 5, 6],
            tokenLabels: ['Token D', 'Token E', 'Token F', 'Token G'],
            layerCount: 2,
            kvCacheState: createDecodeKvState()
        };

        const outputProjectionScene = buildTransformerSceneModel({
            ...baseConfig,
            outputProjectionDetailTarget: {
                layerIndex: 0
            }
        })?.metadata?.outputProjectionDetailScene;
        const mlpScene = buildTransformerSceneModel({
            ...baseConfig,
            mlpDetailTarget: {
                layerIndex: 0
            }
        })?.metadata?.mlpDetailScene;
        const layerNormScene = buildTransformerSceneModel({
            ...baseConfig,
            layerNormDetailTarget: {
                layerNormKind: 'ln1',
                layerIndex: 0
            }
        })?.metadata?.layerNormDetailScene;

        const outputProjectionNodes = flattenSceneNodes(outputProjectionScene);
        const mlpNodes = flattenSceneNodes(mlpScene);
        const layerNormNodes = flattenSceneNodes(layerNormScene);

        const projectionOutputNode = outputProjectionNodes.find((node) => node?.role === 'projection-output');
        const mlpUpOutputNode = mlpNodes.find((node) => node?.role === 'mlp-up-output');
        const mlpDownOutputNode = mlpNodes.find((node) => node?.role === 'mlp-down-output');
        const layerNormInputNode = layerNormNodes.find((node) => node?.role === 'layer-norm-input');
        const layerNormNormalizedNode = layerNormNodes.find((node) => node?.role === 'layer-norm-normalized');
        const layerNormOutputNode = layerNormNodes.find((node) => node?.role === 'layer-norm-output');

        expect(outputProjectionScene).toBeTruthy();
        expect(projectionOutputNode?.dimensions?.rows).toBe(1);
        expect(projectionOutputNode?.rowItems).toHaveLength(1);
        expect(projectionOutputNode?.rowItems?.[0]?.semantic?.tokenIndex).toBe(6);

        expect(mlpScene).toBeTruthy();
        expect(mlpUpOutputNode?.dimensions?.rows).toBe(1);
        expect(mlpUpOutputNode?.rowItems).toHaveLength(1);
        expect(mlpDownOutputNode?.dimensions?.rows).toBe(1);
        expect(mlpDownOutputNode?.rowItems).toHaveLength(1);
        expect(mlpDownOutputNode?.rowItems?.[0]?.semantic?.tokenIndex).toBe(6);

        expect(layerNormScene).toBeTruthy();
        expect(layerNormInputNode?.dimensions?.rows).toBe(1);
        expect(layerNormNormalizedNode?.dimensions?.rows).toBe(1);
        expect(layerNormOutputNode?.dimensions?.rows).toBe(1);
        expect(layerNormOutputNode?.rowItems).toHaveLength(1);
        expect(layerNormOutputNode?.rowItems?.[0]?.semantic?.tokenIndex).toBe(6);
    });

    it('keeps the full token window for the MHSA head-detail scene during decode so cache rows can render', () => {
        const scene = buildTransformerSceneModel({
            activationSource: createMockActivationSource(),
            tokenIndices: [3, 4, 5, 6],
            tokenLabels: ['Token D', 'Token E', 'Token F', 'Token G'],
            layerCount: 1,
            headDetailTarget: {
                layerIndex: 0,
                headIndex: 0
            },
            kvCacheState: createDecodeKvState()
        });

        const detailScene = scene?.metadata?.mhsaHeadDetailScene;
        const detailNodes = flattenSceneNodes(detailScene);
        const keyOutputNode = detailNodes.find((node) => (
            node?.role === 'projection-output'
            && String(node?.metadata?.kind || '').toLowerCase() === 'k'
        )) || null;
        const valueOutputNode = detailNodes.find((node) => (
            node?.role === 'projection-output'
            && String(node?.metadata?.kind || '').toLowerCase() === 'v'
        )) || null;
        const keyCacheNode = detailNodes.find((node) => (
            node?.role === 'projection-cache'
            && String(node?.semantic?.branchKey || '').toLowerCase() === 'k'
        )) || null;
        const keyCacheSourceNode = detailNodes.find((node) => (
            node?.role === 'projection-cache-source'
            && String(node?.semantic?.branchKey || node?.metadata?.kind || '').toLowerCase() === 'k'
        )) || null;
        const valueCacheNode = detailNodes.find((node) => (
            node?.role === 'projection-cache'
            && String(node?.semantic?.branchKey || '').toLowerCase() === 'v'
        )) || null;
        const valueCacheSourceNode = detailNodes.find((node) => (
            node?.role === 'projection-cache-source'
            && String(node?.semantic?.branchKey || node?.metadata?.kind || '').toLowerCase() === 'v'
        )) || null;
        const keyOutputCopyNode = detailNodes.find((node) => (
            node?.role === 'projection-output-copy'
            && String(node?.metadata?.kind || '').toLowerCase() === 'k'
        )) || null;
        const valueOutputCopyNode = detailNodes.find((node) => (
            node?.role === 'projection-output-copy'
            && String(node?.metadata?.kind || '').toLowerCase() === 'v'
        )) || null;
        const keyConcatResultNode = detailNodes.find((node) => (
            node?.role === 'projection-cache-concat-result'
            && String(node?.metadata?.kind || node?.semantic?.branchKey || '').toLowerCase() === 'k'
        )) || null;
        const valueConcatResultNode = detailNodes.find((node) => (
            node?.role === 'projection-cache-concat-result'
            && String(node?.metadata?.kind || node?.semantic?.branchKey || '').toLowerCase() === 'v'
        )) || null;

        expect(detailScene).toBeTruthy();
        expect(detailScene?.metadata?.kvCacheState).toMatchObject({
            kvCacheModeEnabled: true,
            kvCachePrefillActive: false,
            kvCacheDecodeActive: true,
            kvCachePassIndex: 1
        });

        expect(keyOutputNode?.dimensions?.rows).toBe(1);
        expect(valueOutputNode?.dimensions?.rows).toBe(1);
        expect(keyOutputNode?.rowItems).toHaveLength(1);
        expect(valueOutputNode?.rowItems).toHaveLength(1);
        expect(keyOutputNode?.rowItems?.[0]?.semantic?.tokenIndex).toBe(6);
        expect(valueOutputNode?.rowItems?.[0]?.semantic?.tokenIndex).toBe(6);
        expect(keyOutputCopyNode?.dimensions?.rows).toBe(1);
        expect(valueOutputCopyNode?.dimensions?.rows).toBe(1);
        expect(keyOutputCopyNode?.rowItems?.[0]?.semantic?.tokenIndex).toBe(6);
        expect(valueOutputCopyNode?.rowItems?.[0]?.semantic?.tokenIndex).toBe(6);

        expect(keyCacheNode?.dimensions?.rows).toBe(3);
        expect(valueCacheNode?.dimensions?.rows).toBe(3);
        expect(keyCacheNode?.rowItems).toHaveLength(3);
        expect(valueCacheNode?.rowItems).toHaveLength(3);
        expect(keyCacheSourceNode?.dimensions?.rows).toBe(3);
        expect(valueCacheSourceNode?.dimensions?.rows).toBe(3);
        expect(keyCacheSourceNode?.rowItems).toHaveLength(3);
        expect(valueCacheSourceNode?.rowItems).toHaveLength(3);
        expect(keyCacheNode?.rowItems?.[0]?.semantic?.tokenIndex).toBe(3);
        expect(keyCacheNode?.rowItems?.[2]?.semantic?.tokenIndex).toBe(5);
        expect(valueCacheNode?.rowItems?.[0]?.semantic?.tokenIndex).toBe(3);
        expect(valueCacheNode?.rowItems?.[2]?.semantic?.tokenIndex).toBe(5);
        expect(keyCacheSourceNode?.rowItems?.[0]?.semantic?.tokenIndex).toBe(3);
        expect(keyCacheSourceNode?.rowItems?.[2]?.semantic?.tokenIndex).toBe(5);
        expect(valueCacheSourceNode?.rowItems?.[0]?.semantic?.tokenIndex).toBe(3);
        expect(valueCacheSourceNode?.rowItems?.[2]?.semantic?.tokenIndex).toBe(5);
        expect(keyConcatResultNode?.dimensions?.rows).toBe(4);
        expect(valueConcatResultNode?.dimensions?.rows).toBe(4);
        expect(keyConcatResultNode?.rowItems).toHaveLength(4);
        expect(valueConcatResultNode?.rowItems).toHaveLength(4);
        expect(keyConcatResultNode?.rowItems?.[0]?.semantic?.tokenIndex).toBe(3);
        expect(keyConcatResultNode?.rowItems?.[3]?.semantic?.tokenIndex).toBe(6);
        expect(valueConcatResultNode?.rowItems?.[0]?.semantic?.tokenIndex).toBe(3);
        expect(valueConcatResultNode?.rowItems?.[3]?.semantic?.tokenIndex).toBe(6);
        expect(keyConcatResultNode?.rowItems?.[2]?.semantic?.concatResultPart).toBe('cache');
        expect(keyConcatResultNode?.rowItems?.[3]?.semantic?.concatResultPart).toBe('live');
        expect(valueConcatResultNode?.rowItems?.[2]?.semantic?.concatResultPart).toBe('cache');
        expect(valueConcatResultNode?.rowItems?.[3]?.semantic?.concatResultPart).toBe('live');
    });
});
