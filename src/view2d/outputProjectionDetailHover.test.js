import { describe, expect, it } from 'vitest';

import { D_HEAD } from '../ui/selectionPanelConstants.js';
import {
    createMhsaDetailSceneIndex,
    resolveMhsaDetailHoverState
} from './mhsaDetailInteraction.js';
import { buildOutputProjectionDetailSceneModel } from './model/buildOutputProjectionDetailSceneModel.js';
import { flattenSceneNodes } from './schema/sceneTypes.js';

function createVector(seed = 0, length = D_HEAD) {
    return Array.from({ length }, (_, index) => Number((seed + (index * 0.01)).toFixed(4)));
}

function createMockActivationSource() {
    return {
        getAttentionWeightedSum(_layerIndex = 0, headIndex = 0, tokenIndex = 0, targetLength = D_HEAD) {
            return createVector((headIndex * 0.2) + (tokenIndex * 0.05), targetLength);
        }
    };
}

describe('output projection detail hover', () => {
    it('maps head-output matrix rows to the weighted-sum tooltip payload', () => {
        const scene = buildOutputProjectionDetailSceneModel({
            activationSource: createMockActivationSource(),
            outputProjectionDetailTarget: {
                layerIndex: 3
            },
            tokenRefs: [
                { rowIndex: 0, tokenIndex: 0, tokenLabel: 'Token A' },
                { rowIndex: 1, tokenIndex: 1, tokenLabel: 'Token B' }
            ]
        });
        const index = createMhsaDetailSceneIndex(scene);
        const nodes = flattenSceneNodes(scene);
        const matrixNode = nodes.find((node) => (
            node?.role === 'head-output-matrix'
            && node?.semantic?.headIndex === 4
        )) || null;
        const otherMatrixNode = nodes.find((node) => (
            node?.role === 'head-output-matrix'
            && node?.semantic?.headIndex === 5
        )) || null;
        const connectorNode = nodes.find((node) => (
            node?.role === 'head-output-connector'
            && node?.semantic?.headIndex === 4
        )) || null;

        const hoverState = resolveMhsaDetailHoverState(index, {
            node: matrixNode,
            rowHit: {
                rowIndex: 1,
                rowItem: matrixNode?.rowItems?.[1]
            }
        });

        expect(hoverState?.label).toBe('Attention Weighted Sum');
        expect(hoverState?.info?.isWeightedSum).toBe(true);
        expect(hoverState?.info?.activationData?.label).toBe('Attention Weighted Sum');
        expect(hoverState?.info?.activationData?.stage).toBe('attention.weighted_sum');
        expect(hoverState?.info?.activationData?.layerIndex).toBe(3);
        expect(hoverState?.info?.activationData?.headIndex).toBe(4);
        expect(hoverState?.info?.activationData?.tokenIndex).toBe(1);
        expect(hoverState?.info?.activationData?.tokenLabel).toBe('Token B');
        expect(hoverState?.focusState?.activeNodeIds).toContain(matrixNode?.id);
        expect(hoverState?.focusState?.activeNodeIds).not.toContain(otherMatrixNode?.id);
        expect(hoverState?.focusState?.activeConnectorIds).toContain(connectorNode?.id);
        expect(hoverState?.focusState?.rowSelections).toContainEqual({
            nodeId: matrixNode?.id,
            rowIndex: 1
        });
    });
});
