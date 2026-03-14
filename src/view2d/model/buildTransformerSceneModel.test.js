import { describe, expect, it } from 'vitest';

import { D_MODEL } from '../../ui/selectionPanelConstants.js';
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
});
