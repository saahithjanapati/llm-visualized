import { describe, expect, it } from 'vitest';

import { D_MODEL } from '../../ui/selectionPanelConstants.js';
import { NUM_HEAD_SETS_LAYER } from '../../utils/constants.js';
import { buildSceneLayout } from '../layout/buildSceneLayout.js';
import { flattenSceneNodes, VIEW2D_ANCHOR_SIDES, VIEW2D_NODE_KINDS } from '../schema/sceneTypes.js';
import { buildResidualRowHoverPayload } from '../transformerView2dTargets.js';
import { buildTransformerSceneModel } from './buildTransformerSceneModel.js';

function createVector(seed = 0, length = D_MODEL) {
    return Array.from({ length }, (_, index) => Number((seed + (index * 0.01)).toFixed(4)));
}

function createMockActivationSource() {
    return {
        meta: {
            prompt_tokens: [101, 102],
            completion_tokens: [103]
        },
        getTokenCount() {
            return 3;
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
        const lastResidualAddNode = nodes.find((node) => (
            node?.kind === VIEW2D_NODE_KINDS.MATRIX
            && node?.role === 'add-circle'
            && node?.semantic?.componentKind === 'residual'
            && node?.semantic?.stage === 'post-mlp-add'
        ));

        expect(outgoingResidualNode).toBeTruthy();
        expect(finalLayerNormNode).toBeTruthy();
        expect(unembeddingNode).toBeTruthy();
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
    });

    it('places chosen generated-token chips to the right of the unembedding card', () => {
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
            && node?.role === 'chosen-token-chip-label'
            && node?.semantic?.componentKind === 'logits'
            && node?.semantic?.stage === 'output'
        ));
        const connectors = nodes.filter((node) => node?.kind === VIEW2D_NODE_KINDS.CONNECTOR);

        expect(unembeddingNode).toBeTruthy();
        expect(chosenTokenChipStackNode).toBeTruthy();
        expect(chosenTokenChipNodes).toHaveLength(1);
        expect(
            chosenTokenChipLabelNodes.map((node) => String(node?.text || '').replace(/\u00A0/g, ' '))
        ).toEqual(['Token C']);
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
                    stage: 'ln1.shift',
                    tokenIndex: 0
                }
            }
        }, createMockActivationSource());

        expect(payload?.label).toBe('Post LayerNorm 1 Residual Vector');
        expect(payload?.info?.activationData?.label).toBe('Post LayerNorm 1 Residual Vector');
        expect(payload?.info?.activationData?.stage).toBe('ln1.shift');
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
});
