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
        getTokenCount() {
            return 2;
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
    it('adds the final layer norm to the right of the top residual stream', () => {
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
        const lastResidualAddNode = nodes.find((node) => (
            node?.kind === VIEW2D_NODE_KINDS.MATRIX
            && node?.role === 'add-circle'
            && node?.semantic?.componentKind === 'residual'
            && node?.semantic?.stage === 'post-mlp-add'
        ));

        expect(outgoingResidualNode).toBeTruthy();
        expect(finalLayerNormNode).toBeTruthy();
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

        const layout = buildSceneLayout(scene);
        const outgoingResidualEntry = layout?.registry?.getNodeEntry(outgoingResidualNode.id);
        const finalLayerNormEntry = layout?.registry?.getNodeEntry(finalLayerNormNode.id);
        const lastResidualAddEntry = layout?.registry?.getNodeEntry(lastResidualAddNode.id);

        expect(outgoingResidualEntry).toBeTruthy();
        expect(finalLayerNormEntry).toBeTruthy();
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
