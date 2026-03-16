import { describe, expect, it } from 'vitest';

import { CaptureActivationSource } from '../../data/CaptureActivationSource.js';
import { D_MODEL } from '../../ui/selectionPanelConstants.js';
import {
    buildPromptTokenChipEntries,
    resolvePromptTokenChipColorState,
    resolveTokenChipColors
} from '../../ui/tokenChipColorUtils.js';
import { NUM_HEAD_SETS_LAYER } from '../../utils/constants.js';
import { buildSceneLayout } from '../layout/buildSceneLayout.js';
import { flattenSceneNodes, VIEW2D_ANCHOR_SIDES, VIEW2D_NODE_KINDS } from '../schema/sceneTypes.js';
import { buildResidualRowHoverPayload } from '../transformerView2dTargets.js';
import { buildTransformerSceneModel } from './buildTransformerSceneModel.js';

function createVector(seed = 0, length = D_MODEL) {
    return Array.from({ length }, (_, index) => Number((seed + (index * 0.01)).toFixed(4)));
}

function createMockActivationSource({
    promptTokenCount = 2,
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
        getPostAttentionResidual(_layerIndex = 0, tokenIndex = 0, targetLength = D_MODEL) {
            return createVector(0.2 + tokenIndex, targetLength);
        },
        getAttentionWeightedSum(_layerIndex = 0, headIndex = 0, tokenIndex = 0, targetLength = D_MODEL) {
            return createVector(0.15 + (headIndex * 0.02) + tokenIndex, targetLength);
        },
        getPostMlpResidual(_layerIndex = 0, tokenIndex = 0, targetLength = D_MODEL) {
            return createVector(0.3 + tokenIndex, targetLength);
        }
    };
}

describe('buildTransformerSceneModel', () => {
    it('places vocabulary and position embeddings before the first incoming residual stream', () => {
        const scene = buildTransformerSceneModel({
            activationSource: createMockActivationSource(),
            tokenIndices: [0, 1],
            tokenLabels: ['Token A', 'Token B'],
            layerCount: 2
        });
        const nodes = flattenSceneNodes(scene);
        const vocabularyEmbeddingNode = nodes.find((node) => (
            node?.kind === VIEW2D_NODE_KINDS.MATRIX
            && node?.role === 'vocabulary-embedding-card'
            && node?.semantic?.componentKind === 'embedding'
            && node?.semantic?.stage === 'embedding.token'
        ));
        const positionEmbeddingNode = nodes.find((node) => (
            node?.kind === VIEW2D_NODE_KINDS.MATRIX
            && node?.role === 'position-embedding-card'
            && node?.semantic?.componentKind === 'embedding'
            && node?.semantic?.stage === 'embedding.position'
        ));
        const embeddingAddNode = nodes.find((node) => (
            node?.kind === VIEW2D_NODE_KINDS.MATRIX
            && node?.role === 'add-circle'
            && node?.semantic?.componentKind === 'embedding'
            && node?.semantic?.stage === 'embedding.sum'
        ));
        const firstIncomingResidualNode = nodes.find((node) => (
            node?.kind === VIEW2D_NODE_KINDS.MATRIX
            && node?.role === 'module-card'
            && node?.semantic?.componentKind === 'residual'
            && node?.semantic?.stage === 'incoming'
            && node?.semantic?.layerIndex === 0
        ));

        expect(vocabularyEmbeddingNode).toBeTruthy();
        expect(positionEmbeddingNode).toBeTruthy();
        expect(embeddingAddNode).toBeTruthy();
        expect(firstIncomingResidualNode).toBeTruthy();

        const connectors = nodes.filter((node) => node?.kind === VIEW2D_NODE_KINDS.CONNECTOR);
        expect(connectors.some((connector) => (
            connector?.source?.nodeId === vocabularyEmbeddingNode.id
            && connector?.target?.nodeId === embeddingAddNode.id
        ))).toBe(true);
        expect(connectors.some((connector) => (
            connector?.source?.nodeId === positionEmbeddingNode.id
            && connector?.target?.nodeId === embeddingAddNode.id
        ))).toBe(true);
        expect(connectors.some((connector) => (
            connector?.source?.nodeId === embeddingAddNode.id
            && connector?.target?.nodeId === firstIncomingResidualNode.id
        ))).toBe(true);

        const layout = buildSceneLayout(scene);
        const vocabularyEmbeddingEntry = layout?.registry?.getNodeEntry(vocabularyEmbeddingNode.id);
        const positionEmbeddingEntry = layout?.registry?.getNodeEntry(positionEmbeddingNode.id);
        const embeddingAddEntry = layout?.registry?.getNodeEntry(embeddingAddNode.id);
        const incomingResidualEntry = layout?.registry?.getNodeEntry(firstIncomingResidualNode.id);
        const firstLayerNormNode = nodes.find((node) => (
            node?.kind === VIEW2D_NODE_KINDS.MATRIX
            && node?.role === 'module-card'
            && node?.semantic?.componentKind === 'layer-norm'
            && node?.semantic?.stage === 'ln1'
            && node?.semantic?.layerIndex === 0
        ));
        const firstLayerNormEntry = layout?.registry?.getNodeEntry(firstLayerNormNode?.id);

        expect(vocabularyEmbeddingEntry).toBeTruthy();
        expect(positionEmbeddingEntry).toBeTruthy();
        expect(embeddingAddEntry).toBeTruthy();
        expect(incomingResidualEntry).toBeTruthy();
        expect(firstLayerNormEntry).toBeTruthy();
        expect(
            embeddingAddEntry.anchors[VIEW2D_ANCHOR_SIDES.RIGHT].x
        ).toBeLessThan(
            incomingResidualEntry.anchors[VIEW2D_ANCHOR_SIDES.LEFT].x
        );
        expect(
            incomingResidualEntry.anchors[VIEW2D_ANCHOR_SIDES.LEFT].x
            - embeddingAddEntry.anchors[VIEW2D_ANCHOR_SIDES.RIGHT].x
        ).toBeGreaterThan(80);
        expect(
            Math.abs(
                vocabularyEmbeddingEntry.anchors[VIEW2D_ANCHOR_SIDES.CENTER].y
                - embeddingAddEntry.anchors[VIEW2D_ANCHOR_SIDES.CENTER].y
            )
        ).toBeLessThan(0.5);
        expect(
            positionEmbeddingEntry.anchors[VIEW2D_ANCHOR_SIDES.CENTER].y
        ).toBeGreaterThan(
            vocabularyEmbeddingEntry.anchors[VIEW2D_ANCHOR_SIDES.CENTER].y
        );
        expect(
            Math.abs(
                embeddingAddEntry.anchors[VIEW2D_ANCHOR_SIDES.CENTER].y
                - incomingResidualEntry.anchors[VIEW2D_ANCHOR_SIDES.CENTER].y
            )
        ).toBeLessThan(0.5);
        expect(
            Math.abs(
                incomingResidualEntry.anchors[VIEW2D_ANCHOR_SIDES.CENTER].x
                - firstLayerNormEntry.anchors[VIEW2D_ANCHOR_SIDES.CENTER].x
            )
        ).toBeLessThan(0.5);
    });

    it('adds the final layer norm and unembedding to the right of the top residual stream', () => {
        const scene = buildTransformerSceneModel({
            activationSource: createMockActivationSource(),
            tokenIndices: [0, 1],
            tokenLabels: ['Token A', 'Token B'],
            layerCount: 1
        });
        const nodes = flattenSceneNodes(scene);
        const outgoingResidualNode = nodes.find((node) => (
            node?.kind === VIEW2D_NODE_KINDS.MATRIX
            && node?.role === 'module-card'
            && node?.semantic?.componentKind === 'residual'
            && node?.semantic?.stage === 'outgoing'
        ));
        const finalLayerNormNode = nodes.find((node) => (
            node?.kind === VIEW2D_NODE_KINDS.MATRIX
            && node?.role === 'module-card'
            && node?.semantic?.componentKind === 'layer-norm'
            && node?.semantic?.stage === 'final-ln'
        ));
        const unembeddingNode = nodes.find((node) => (
            node?.kind === VIEW2D_NODE_KINDS.MATRIX
            && node?.role === 'unembedding'
            && node?.semantic?.componentKind === 'logits'
            && node?.semantic?.stage === 'unembedding'
        ));
        const unembeddingTitleNode = nodes.find((node) => (
            node?.kind === VIEW2D_NODE_KINDS.TEXT
            && node?.role === 'module-title'
            && node?.semantic?.componentKind === 'logits'
            && node?.semantic?.stage === 'unembedding'
        ));
        const lastResidualAddNode = nodes.find((node) => (
            node?.kind === VIEW2D_NODE_KINDS.MATRIX
            && node?.role === 'add-circle'
            && node?.semantic?.componentKind === 'residual'
            && node?.semantic?.stage === 'post-mlp-add'
        ));

        expect(outgoingResidualNode).toBeTruthy();
        expect(finalLayerNormNode).toBeTruthy();
        expect(unembeddingNode).toBeTruthy();
        expect(unembeddingTitleNode?.text).toBe('Vocabulary Unembedding Matrix');
        expect(lastResidualAddNode).toBeTruthy();

        const connectors = nodes.filter((node) => node?.kind === VIEW2D_NODE_KINDS.CONNECTOR);
        expect(connectors.some((connector) => (
            connector?.source?.nodeId === lastResidualAddNode.id
            && connector?.target?.nodeId === outgoingResidualNode.id
        ))).toBe(true);
        expect(connectors.some((connector) => (
            connector?.source?.nodeId === outgoingResidualNode.id
            && connector?.target?.nodeId === finalLayerNormNode.id
        ))).toBe(true);
        expect(connectors.some((connector) => (
            connector?.source?.nodeId === finalLayerNormNode.id
            && connector?.target?.nodeId === unembeddingNode.id
        ))).toBe(true);

        const layout = buildSceneLayout(scene);
        const outgoingResidualEntry = layout?.registry?.getNodeEntry(outgoingResidualNode.id);
        const finalLayerNormEntry = layout?.registry?.getNodeEntry(finalLayerNormNode.id);
        const unembeddingEntry = layout?.registry?.getNodeEntry(unembeddingNode.id);
        const lastResidualAddEntry = layout?.registry?.getNodeEntry(lastResidualAddNode.id);

        expect(outgoingResidualEntry).toBeTruthy();
        expect(finalLayerNormEntry).toBeTruthy();
        expect(unembeddingEntry).toBeTruthy();
        expect(lastResidualAddEntry).toBeTruthy();
        expect(
            Math.abs(
                outgoingResidualEntry.anchors[VIEW2D_ANCHOR_SIDES.CENTER].y
                - lastResidualAddEntry.anchors[VIEW2D_ANCHOR_SIDES.CENTER].y
            )
        ).toBeLessThan(0.5);
        expect(
            finalLayerNormEntry.anchors[VIEW2D_ANCHOR_SIDES.LEFT].x
        ).toBeGreaterThan(
            outgoingResidualEntry.anchors[VIEW2D_ANCHOR_SIDES.RIGHT].x
        );
        expect(
            unembeddingEntry.anchors[VIEW2D_ANCHOR_SIDES.LEFT].x
        ).toBeGreaterThan(
            finalLayerNormEntry.anchors[VIEW2D_ANCHOR_SIDES.RIGHT].x
        );
        expect(
            unembeddingNode?.metadata?.card?.shapeConfig?.leftHeightRatio
        ).toBeLessThan(
            unembeddingNode?.metadata?.card?.shapeConfig?.rightHeightRatio
        );
    });

    it('stacks the active prompt tokens to the left of the vocabulary embedding card', () => {
        const scene = buildTransformerSceneModel({
            activationSource: createMockActivationSource(),
            tokenIndices: [0, 1, 2],
            tokenLabels: ['Token A', 'Token B', 'Token C'],
            layerCount: 1
        });

        const nodes = flattenSceneNodes(scene);
        const tokenChipNodes = nodes.filter((node) => (
            node?.kind === VIEW2D_NODE_KINDS.MATRIX
            && node?.role === 'input-token-chip'
            && node?.semantic?.componentKind === 'embedding'
            && node?.semantic?.stage === 'embedding.token'
        ));
        const tokenChipLabelNodes = nodes.filter((node) => (
            node?.kind === VIEW2D_NODE_KINDS.TEXT
            && node?.role === 'input-token-chip-label'
            && node?.semantic?.componentKind === 'embedding'
            && node?.semantic?.stage === 'embedding.token'
        ));
        const tokenChipStackNode = nodes.find((node) => (
            node?.role === 'input-token-chip-stack'
            && node?.semantic?.componentKind === 'embedding'
            && node?.semantic?.stage === 'embedding.token'
        ));
        const vocabularyEmbeddingNode = nodes.find((node) => (
            node?.kind === VIEW2D_NODE_KINDS.MATRIX
            && node?.role === 'vocabulary-embedding-card'
            && node?.semantic?.componentKind === 'embedding'
            && node?.semantic?.stage === 'embedding.token'
        ));
        const connectors = nodes.filter((node) => node?.kind === VIEW2D_NODE_KINDS.CONNECTOR);

        expect(tokenChipNodes).toHaveLength(3);
        expect(
            tokenChipLabelNodes.map((node) => String(node?.text || '').replace(/\u00A0/g, ' '))
        ).toEqual(['Token A', 'Token B', 'Token C']);
        expect(tokenChipStackNode).toBeTruthy();
        expect(vocabularyEmbeddingNode).toBeTruthy();
        expect(connectors.some((connector) => (
            connector?.source?.nodeId === tokenChipStackNode?.id
            && connector?.target?.nodeId === vocabularyEmbeddingNode?.id
        ))).toBe(true);

        const layout = buildSceneLayout(scene);
        const vocabularyEmbeddingEntry = layout?.registry?.getNodeEntry(vocabularyEmbeddingNode?.id);
        const chipEntries = tokenChipNodes.map((node) => layout?.registry?.getNodeEntry(node.id));

        chipEntries.forEach((chipEntry) => {
            expect(chipEntry).toBeTruthy();
            expect(
                chipEntry.anchors[VIEW2D_ANCHOR_SIDES.RIGHT].x
            ).toBeLessThan(
                vocabularyEmbeddingEntry.anchors[VIEW2D_ANCHOR_SIDES.LEFT].x
            );
        });
        expect(
            chipEntries[1].anchors[VIEW2D_ANCHOR_SIDES.CENTER].y
        ).toBeGreaterThan(
            chipEntries[0].anchors[VIEW2D_ANCHOR_SIDES.CENTER].y
        );
        expect(
            chipEntries[2].anchors[VIEW2D_ANCHOR_SIDES.CENTER].y
        ).toBeGreaterThan(
            chipEntries[1].anchors[VIEW2D_ANCHOR_SIDES.CENTER].y
        );
    });

    it('stacks visible position chips to the left of the position embedding card', () => {
        const scene = buildTransformerSceneModel({
            activationSource: createMockActivationSource(),
            tokenIndices: [0, 1, 2],
            tokenLabels: ['Token A', 'Token B', 'Token C'],
            layerCount: 1
        });

        const nodes = flattenSceneNodes(scene);
        const positionChipNodes = nodes.filter((node) => (
            node?.kind === VIEW2D_NODE_KINDS.MATRIX
            && node?.role === 'input-position-chip'
            && node?.semantic?.componentKind === 'embedding'
            && node?.semantic?.stage === 'embedding.position'
        ));
        const positionChipLabelNodes = nodes.filter((node) => (
            node?.kind === VIEW2D_NODE_KINDS.TEXT
            && node?.role === 'input-position-chip-label'
            && node?.semantic?.componentKind === 'embedding'
            && node?.semantic?.stage === 'embedding.position'
        ));
        const positionChipStackNode = nodes.find((node) => (
            node?.role === 'input-position-chip-stack'
            && node?.semantic?.componentKind === 'embedding'
            && node?.semantic?.stage === 'embedding.position'
        ));
        const positionEmbeddingNode = nodes.find((node) => (
            node?.kind === VIEW2D_NODE_KINDS.MATRIX
            && node?.role === 'position-embedding-card'
            && node?.semantic?.componentKind === 'embedding'
            && node?.semantic?.stage === 'embedding.position'
        ));
        const connectors = nodes.filter((node) => node?.kind === VIEW2D_NODE_KINDS.CONNECTOR);

        expect(positionChipNodes).toHaveLength(3);
        expect(
            positionChipLabelNodes.map((node) => String(node?.text || '').replace(/\u00A0/g, ' '))
        ).toEqual(['1', '2', '3']);
        expect(positionChipStackNode).toBeTruthy();
        expect(positionEmbeddingNode).toBeTruthy();
        expect(connectors.some((connector) => (
            connector?.source?.nodeId === positionChipStackNode?.id
            && connector?.target?.nodeId === positionEmbeddingNode?.id
        ))).toBe(true);

        const layout = buildSceneLayout(scene);
        const positionEmbeddingEntry = layout?.registry?.getNodeEntry(positionEmbeddingNode?.id);
        const chipEntries = positionChipNodes.map((node) => layout?.registry?.getNodeEntry(node.id));

        chipEntries.forEach((chipEntry) => {
            expect(chipEntry).toBeTruthy();
            expect(
                chipEntry.anchors[VIEW2D_ANCHOR_SIDES.RIGHT].x
            ).toBeLessThan(
                positionEmbeddingEntry.anchors[VIEW2D_ANCHOR_SIDES.LEFT].x
            );
        });
        expect(
            chipEntries[1].anchors[VIEW2D_ANCHOR_SIDES.CENTER].y
        ).toBeGreaterThan(
            chipEntries[0].anchors[VIEW2D_ANCHOR_SIDES.CENTER].y
        );
        expect(
            chipEntries[2].anchors[VIEW2D_ANCHOR_SIDES.CENTER].y
        ).toBeGreaterThan(
            chipEntries[1].anchors[VIEW2D_ANCHOR_SIDES.CENTER].y
        );
        expect(positionChipNodes[0]?.visual?.accent).toBe(positionChipNodes[1]?.visual?.accent);
        expect(positionChipNodes[1]?.visual?.accent).toBe(positionChipNodes[2]?.visual?.accent);
        expect(positionChipNodes[0]?.visual?.accent).not.toBeUndefined();
        const inputTokenChipNode = nodes.find((node) => (
            node?.kind === VIEW2D_NODE_KINDS.MATRIX
            && node?.role === 'input-token-chip'
            && node?.semantic?.componentKind === 'embedding'
            && node?.semantic?.stage === 'embedding.token'
        ));
        expect(positionChipNodes[0]?.visual?.accent).not.toBe(inputTokenChipNode?.visual?.accent);
    });

    it('renders NA rows for prompt continuations and chips for generated continuations beside the unembedding card', () => {
        const scene = buildTransformerSceneModel({
            activationSource: createMockActivationSource(),
            tokenIndices: [0, 1, 2],
            tokenLabels: ['Token A', 'Token B', 'Token C'],
            layerCount: 1
        });

        const nodes = flattenSceneNodes(scene);
        const unembeddingNode = nodes.find((node) => (
            node?.kind === VIEW2D_NODE_KINDS.MATRIX
            && node?.role === 'unembedding'
            && node?.semantic?.componentKind === 'logits'
            && node?.semantic?.stage === 'unembedding'
        ));
        const chosenTokenChipStackNode = nodes.find((node) => (
            node?.role === 'chosen-token-chip-stack'
            && node?.semantic?.componentKind === 'logits'
            && node?.semantic?.stage === 'output'
        ));
        const chosenTokenChipNodes = nodes.filter((node) => (
            node?.kind === VIEW2D_NODE_KINDS.MATRIX
            && node?.role === 'chosen-token-chip'
            && node?.semantic?.componentKind === 'logits'
            && node?.semantic?.stage === 'output'
        ));
        const chosenTokenChipLabelNodes = nodes.filter((node) => (
            node?.kind === VIEW2D_NODE_KINDS.TEXT
            && typeof node?.role === 'string'
            && node.role.startsWith('chosen-token-chip-label')
            && node?.semantic?.componentKind === 'logits'
            && node?.semantic?.stage === 'output'
        ));
        const connectors = nodes.filter((node) => node?.kind === VIEW2D_NODE_KINDS.CONNECTOR);

        expect(unembeddingNode).toBeTruthy();
        expect(chosenTokenChipStackNode).toBeTruthy();
        expect(chosenTokenChipNodes).toHaveLength(1);
        expect(
            chosenTokenChipLabelNodes.map((node) => String(node?.text || '').replace(/\u00A0/g, ' '))
        ).toEqual(['NA', 'Token C', 'NA']);
        expect(connectors.some((connector) => (
            connector?.source?.nodeId === unembeddingNode?.id
            && connector?.target?.nodeId === chosenTokenChipStackNode?.id
        ))).toBe(true);

        const layout = buildSceneLayout(scene);
        const unembeddingEntry = layout?.registry?.getNodeEntry(unembeddingNode?.id);
        const chosenTokenChipEntry = layout?.registry?.getNodeEntry(chosenTokenChipNodes[0]?.id);

        expect(chosenTokenChipEntry).toBeTruthy();
        expect(
            chosenTokenChipEntry.anchors[VIEW2D_ANCHOR_SIDES.LEFT].x
        ).toBeGreaterThan(
            unembeddingEntry.anchors[VIEW2D_ANCHOR_SIDES.RIGHT].x
        );
    });

    it('matches chosen-token chip colors to the visible token-strip color context for generated tokens', () => {
        const activationSource = createMockActivationSource({
            promptTokenCount: 6,
            completionTokenCount: 1
        });
        const tokenIndices = [4, 5, 6];
        const tokenLabels = ['Token E', 'Token F', 'Token G'];
        const scene = buildTransformerSceneModel({
            activationSource,
            tokenIndices,
            tokenLabels,
            layerCount: 1
        });

        const nodes = flattenSceneNodes(scene);
        const chosenTokenChipNode = nodes.find((node) => (
            node?.kind === VIEW2D_NODE_KINDS.MATRIX
            && node?.role === 'chosen-token-chip'
            && node?.semantic?.componentKind === 'logits'
            && node?.semantic?.stage === 'output'
            && node?.semantic?.tokenIndex === 6
        ));

        const promptEntries = buildPromptTokenChipEntries({
            tokenLabels,
            tokenIndices,
            tokenIds: tokenIndices.map((tokenIndex) => activationSource.getTokenId(tokenIndex))
        });
        const promptColorState = resolvePromptTokenChipColorState(promptEntries);
        const expectedAccent = resolveTokenChipColors({
            tokenIndex: 6,
            tokenId: activationSource.getTokenId(6),
            tokenLabel: 'Token G'
        }, 2, { lookup: promptColorState.lookup }).border;

        expect(chosenTokenChipNode?.visual?.accent).toBe(expectedAccent);
    });

    it('scales embedding and unembedding stream heights plus the vocab-position gap for denser token windows', () => {
        const baseTokenIndices = Array.from({ length: 5 }, (_, index) => index);
        const denseTokenIndices = Array.from({ length: 12 }, (_, index) => index);
        const baseScene = buildTransformerSceneModel({
            activationSource: createMockActivationSource({
                promptTokenCount: 4,
                completionTokenCount: 1
            }),
            tokenIndices: baseTokenIndices,
            tokenLabels: baseTokenIndices.map((index) => `Token ${index + 1}`),
            layerCount: 1
        });
        const denseScene = buildTransformerSceneModel({
            activationSource: createMockActivationSource({
                promptTokenCount: 9,
                completionTokenCount: 3
            }),
            tokenIndices: denseTokenIndices,
            tokenLabels: denseTokenIndices.map((index) => `Token ${index + 1}`),
            layerCount: 1
        });

        const baseNodes = flattenSceneNodes(baseScene);
        const denseNodes = flattenSceneNodes(denseScene);
        const baseVocabularyNode = baseNodes.find((node) => node?.role === 'vocabulary-embedding-card') || null;
        const denseVocabularyNode = denseNodes.find((node) => node?.role === 'vocabulary-embedding-card') || null;
        const basePositionNode = baseNodes.find((node) => node?.role === 'position-embedding-card') || null;
        const densePositionNode = denseNodes.find((node) => node?.role === 'position-embedding-card') || null;
        const baseUnembeddingNode = baseNodes.find((node) => node?.role === 'unembedding') || null;
        const denseUnembeddingNode = denseNodes.find((node) => node?.role === 'unembedding') || null;

        expect((denseVocabularyNode?.metadata?.card?.height || 0)).toBeGreaterThan(
            (baseVocabularyNode?.metadata?.card?.height || 0)
        );
        expect((densePositionNode?.metadata?.card?.height || 0)).toBeGreaterThan(
            (basePositionNode?.metadata?.card?.height || 0)
        );
        expect((denseUnembeddingNode?.metadata?.card?.height || 0)).toBeGreaterThan(
            (baseUnembeddingNode?.metadata?.card?.height || 0)
        );

        const baseLayout = buildSceneLayout(baseScene);
        const denseLayout = buildSceneLayout(denseScene);
        const baseVocabularyEntry = baseLayout?.registry?.getNodeEntry(baseVocabularyNode?.id || '');
        const basePositionEntry = baseLayout?.registry?.getNodeEntry(basePositionNode?.id || '');
        const denseVocabularyEntry = denseLayout?.registry?.getNodeEntry(denseVocabularyNode?.id || '');
        const densePositionEntry = denseLayout?.registry?.getNodeEntry(densePositionNode?.id || '');
        const baseGap = (basePositionEntry?.contentBounds?.y || 0)
            - ((baseVocabularyEntry?.contentBounds?.y || 0) + (baseVocabularyEntry?.contentBounds?.height || 0));
        const denseGap = (densePositionEntry?.contentBounds?.y || 0)
            - ((denseVocabularyEntry?.contentBounds?.y || 0) + (denseVocabularyEntry?.contentBounds?.height || 0));

        expect(denseGap).toBeGreaterThan(baseGap);
    });

    it('preserves prism-length residual values in the top X summary cards', () => {
        const incoming = [0, 1, 0, -1, 2, -2, 0.5, -0.5, 1.5, -1.5, 0.25, -0.25];
        const postAttention = incoming.map((value, index) => Number((value + ((index % 3) * 0.1)).toFixed(4)));
        const postMlp = incoming.map((value, index) => Number((value - ((index % 4) * 0.15)).toFixed(4)));
        const activationSource = new CaptureActivationSource({
            activations: {
                embeddings: {
                    token: [incoming],
                    position: [incoming]
                },
                layers: [{
                    incoming: [incoming],
                    post_attn_residual: [postAttention],
                    post_mlp_residual: [postMlp],
                    ln1: {
                        norm: [incoming]
                    },
                    ln2: {
                        norm: [postAttention]
                    }
                }]
            },
            logits: []
        });

        const scene = buildTransformerSceneModel({
            activationSource,
            tokenIndices: [0],
            tokenLabels: ['Token A'],
            layerCount: 1
        });
        const nodes = flattenSceneNodes(scene);
        const incomingNode = nodes.find((node) => (
            node?.kind === VIEW2D_NODE_KINDS.MATRIX
            && node?.role === 'module-card'
            && node?.semantic?.componentKind === 'residual'
            && node?.semantic?.stage === 'incoming'
            && node?.semantic?.layerIndex === 0
        ));
        const postAttentionNode = nodes.find((node) => (
            node?.kind === VIEW2D_NODE_KINDS.MATRIX
            && node?.role === 'module-card'
            && node?.semantic?.componentKind === 'residual'
            && node?.semantic?.stage === 'post-attn-residual'
            && node?.semantic?.layerIndex === 0
        ));
        const outgoingNode = nodes.find((node) => (
            node?.kind === VIEW2D_NODE_KINDS.MATRIX
            && node?.role === 'module-card'
            && node?.semantic?.componentKind === 'residual'
            && node?.semantic?.stage === 'outgoing'
        ));

        expect(incomingNode?.rowItems?.[0]?.rawValues).toEqual(incoming);
        expect(postAttentionNode?.rowItems?.[0]?.rawValues).toEqual(postAttention);
        expect(outgoingNode?.rowItems?.[0]?.rawValues).toEqual(postMlp);
    });

    it('maps outgoing residual hover rows to the post-MLP residual activation', () => {
        const payload = buildResidualRowHoverPayload({
            rowItem: {
                label: 'Token B',
                semantic: {
                    componentKind: 'residual',
                    layerIndex: 11,
                    stage: 'outgoing',
                    tokenIndex: 1
                }
            }
        }, createMockActivationSource());

        expect(payload?.label).toBe('Residual Stream Vector');
        expect(payload?.info?.tokenId).toBe(101);
        expect(payload?.info?.activationData?.stage).toBe('residual.post_mlp');
    });

    it('maps post-layernorm residual hover rows to layernorm-specific labels', () => {
        const payload = buildResidualRowHoverPayload({
            rowItem: {
                label: 'Token A',
                semantic: {
                    componentKind: 'residual',
                    layerIndex: 3,
                    stage: 'ln1.output',
                    tokenIndex: 0
                }
            }
        }, createMockActivationSource());

        expect(payload?.label).toBe('Post LayerNorm 1 Residual Vector');
        expect(payload?.info?.activationData?.label).toBe('Post LayerNorm 1 Residual Vector');
        expect(payload?.info?.activationData?.stage).toBe('ln1.output');
        expect(payload?.info?.activationData?.sourceStage).toBe('ln1.shift');
    });

    it('builds a stacked output-projection detail scene when that detail target is active', () => {
        const scene = buildTransformerSceneModel({
            activationSource: createMockActivationSource(),
            tokenIndices: [0, 1],
            tokenLabels: ['Token A', 'Token B'],
            layerCount: 1,
            outputProjectionDetailTarget: {
                layerIndex: 0
            }
        });

        const detailScene = scene?.metadata?.outputProjectionDetailScene || null;
        const detailNodes = flattenSceneNodes(detailScene);
        const matrixNodes = detailNodes.filter((node) => (
            node?.kind === VIEW2D_NODE_KINDS.MATRIX
            && node?.role === 'head-output-matrix'
        ));

        expect(detailScene).toBeTruthy();
        expect(scene?.metadata?.outputProjectionDetailPreview?.arrowCount).toBe(12);
        expect(matrixNodes).toHaveLength(12);
    });

    it('builds a scene-backed layer norm detail scene when that detail target is active', () => {
        const scene = buildTransformerSceneModel({
            activationSource: createMockActivationSource(),
            tokenIndices: [0, 1],
            tokenLabels: ['Token A', 'Token B'],
            layerCount: 2,
            layerNormDetailTarget: {
                layerNormKind: 'ln1',
                layerIndex: 0
            }
        });

        const detailScene = scene?.metadata?.layerNormDetailScene || null;
        const detailNodes = flattenSceneNodes(detailScene);
        const normalizedNode = detailNodes.find((node) => (
            node?.kind === VIEW2D_NODE_KINDS.MATRIX
            && node?.role === 'layer-norm-normalized'
        ));
        const outputNode = detailNodes.find((node) => (
            node?.kind === VIEW2D_NODE_KINDS.MATRIX
            && node?.role === 'layer-norm-output'
        ));

        expect(scene?.metadata?.layerNormDetailTarget).toEqual({
            layerNormKind: 'ln1',
            layerIndex: 0
        });
        expect(detailScene).toBeTruthy();
        expect(normalizedNode?.label?.tex).toBe('\\hat{x}');
        expect(outputNode?.label?.tex).toBe('x_{\\ln}');
    });

    it('routes head outputs directly into output projection without a concat stage', () => {
        const scene = buildTransformerSceneModel({
            activationSource: createMockActivationSource(),
            tokenIndices: [0, 1],
            tokenLabels: ['Token A', 'Token B'],
            layerCount: 1
        });

        const nodes = flattenSceneNodes(scene);
        const concatNode = nodes.find((node) => node?.semantic?.stage === 'concatenate');
        const outputProjectionNode = nodes.find((node) => (
            node?.kind === VIEW2D_NODE_KINDS.MATRIX
            && node?.role === 'projection-weight'
            && node?.semantic?.componentKind === 'output-projection'
        ));
        const headCardNodes = nodes.filter((node) => (
            node?.kind === VIEW2D_NODE_KINDS.MATRIX
            && node?.role === 'head-card'
            && node?.semantic?.componentKind === 'mhsa'
            && node?.semantic?.stage === 'attention'
        ));
        const directHeadConnectors = nodes.filter((node) => (
            node?.kind === VIEW2D_NODE_KINDS.CONNECTOR
            && node?.target?.nodeId === outputProjectionNode?.id
            && headCardNodes.some((headCardNode) => headCardNode.id === node?.source?.nodeId)
        ));

        expect(concatNode).toBeUndefined();
        expect(outputProjectionNode).toBeTruthy();
        expect(headCardNodes).toHaveLength(NUM_HEAD_SETS_LAYER);
        expect(directHeadConnectors).toHaveLength(NUM_HEAD_SETS_LAYER);
    });

    it('redirects concat detail targets to the output-projection detail scene', () => {
        const scene = buildTransformerSceneModel({
            activationSource: createMockActivationSource(),
            tokenIndices: [0, 1],
            tokenLabels: ['Token A', 'Token B'],
            layerCount: 1,
            concatDetailTarget: {
                layerIndex: 0
            }
        });

        expect(scene?.metadata?.concatDetailTarget).toBeNull();
        expect(scene?.metadata?.concatDetailPreview).toBeNull();
        expect(scene?.metadata?.outputProjectionDetailTarget).toEqual({ layerIndex: 0 });
        expect(scene?.metadata?.outputProjectionDetailScene).toBeTruthy();
    });

    it('renders the output-projection overview title on two separate text lines', () => {
        const scene = buildTransformerSceneModel({
            activationSource: createMockActivationSource(),
            tokenIndices: [0, 1],
            tokenLabels: ['Token A', 'Token B'],
            layerCount: 1
        });

        const nodes = flattenSceneNodes(scene);
        const topTitleNode = nodes.find((node) => (
            node?.kind === VIEW2D_NODE_KINDS.TEXT
            && node?.semantic?.componentKind === 'output-projection'
            && node?.role === 'module-title-top'
        ));
        const bottomTitleNode = nodes.find((node) => (
            node?.kind === VIEW2D_NODE_KINDS.TEXT
            && node?.semantic?.componentKind === 'output-projection'
            && node?.role === 'module-title-bottom'
        ));

        expect(topTitleNode?.text).toBe('Output');
        expect(bottomTitleNode?.text).toBe('Projection');
    });

    it('renders the MLP overview title on two separate text lines', () => {
        const scene = buildTransformerSceneModel({
            activationSource: createMockActivationSource(),
            tokenIndices: [0, 1],
            tokenLabels: ['Token A', 'Token B'],
            layerCount: 1
        });

        const nodes = flattenSceneNodes(scene);
        const topTitleNode = nodes.find((node) => (
            node?.kind === VIEW2D_NODE_KINDS.TEXT
            && node?.semantic?.componentKind === 'mlp'
            && node?.role === 'module-title-top'
        ));
        const bottomTitleNode = nodes.find((node) => (
            node?.kind === VIEW2D_NODE_KINDS.TEXT
            && node?.semantic?.componentKind === 'mlp'
            && node?.role === 'module-title-bottom'
        ));

        expect(topTitleNode?.text).toBe('Multilayer');
        expect(bottomTitleNode?.text).toBe('Perceptron');
    });

    it('uses full embedding-matrix labels for the left-side input cards', () => {
        const scene = buildTransformerSceneModel({
            activationSource: createMockActivationSource(),
            tokenIndices: [0, 1],
            tokenLabels: ['Token A', 'Token B'],
            layerCount: 1
        });

        const nodes = flattenSceneNodes(scene);
        const vocabularyTitleNode = nodes.find((node) => (
            node?.kind === VIEW2D_NODE_KINDS.TEXT
            && node?.semantic?.componentKind === 'embedding'
            && node?.semantic?.stage === 'embedding.token'
            && node?.role === 'module-title'
        ));
        const positionTitleNode = nodes.find((node) => (
            node?.kind === VIEW2D_NODE_KINDS.TEXT
            && node?.semantic?.componentKind === 'embedding'
            && node?.semantic?.stage === 'embedding.position'
            && node?.role === 'module-title'
        ));

        expect(vocabularyTitleNode?.text).toBe('Vocabulary Embedding Matrix');
        expect(positionTitleNode?.text).toBe('Position Embedding Matrix');
    });

    it('uses the shared default zoom reveal behavior for overview component labels', () => {
        const scene = buildTransformerSceneModel({
            activationSource: createMockActivationSource(),
            tokenIndices: [0, 1],
            tokenLabels: ['Token A', 'Token B'],
            layerCount: 1
        });

        const nodes = flattenSceneNodes(scene);
        const expectDefaultRevealLabel = (predicate) => {
            const node = nodes.find(predicate);
            expect(node).toBeTruthy();
            expect(Number.isFinite(node?.metadata?.persistentMinScreenFontPx)).toBe(false);
        };

        expectDefaultRevealLabel((node) => (
            node?.kind === VIEW2D_NODE_KINDS.TEXT
            && node?.semantic?.componentKind === 'layer-norm'
            && node?.semantic?.stage === 'ln1'
            && node?.role === 'module-title'
        ));
        expectDefaultRevealLabel((node) => (
            node?.kind === VIEW2D_NODE_KINDS.TEXT
            && node?.semantic?.componentKind === 'mhsa'
            && node?.role === 'head-label'
        ));
        expectDefaultRevealLabel((node) => (
            node?.kind === VIEW2D_NODE_KINDS.TEXT
            && node?.semantic?.componentKind === 'output-projection'
            && node?.role === 'module-title-top'
        ));
        expectDefaultRevealLabel((node) => (
            node?.kind === VIEW2D_NODE_KINDS.TEXT
            && node?.semantic?.componentKind === 'output-projection'
            && node?.role === 'module-title-bottom'
        ));
        expectDefaultRevealLabel((node) => (
            node?.kind === VIEW2D_NODE_KINDS.TEXT
            && node?.semantic?.componentKind === 'mlp'
            && node?.role === 'module-title-top'
        ));
        expectDefaultRevealLabel((node) => (
            node?.kind === VIEW2D_NODE_KINDS.TEXT
            && node?.semantic?.componentKind === 'mlp'
            && node?.role === 'module-title-bottom'
        ));
        expectDefaultRevealLabel((node) => (
            node?.kind === VIEW2D_NODE_KINDS.TEXT
            && node?.semantic?.componentKind === 'embedding'
            && node?.semantic?.stage === 'embedding.token'
            && node?.role === 'module-title'
        ));
        expectDefaultRevealLabel((node) => (
            node?.kind === VIEW2D_NODE_KINDS.TEXT
            && node?.semantic?.componentKind === 'embedding'
            && node?.semantic?.stage === 'embedding.position'
            && node?.role === 'module-title'
        ));
    });
});
