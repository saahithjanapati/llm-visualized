import { describe, expect, it } from 'vitest';

import { D_HEAD } from '../ui/selectionPanelConstants.js';
import {
    createMhsaDetailSceneIndex,
    resolveMhsaDetailHoverState
} from './mhsaDetailInteraction.js';
import { buildOutputProjectionDetailSceneModel } from './model/buildOutputProjectionDetailSceneModel.js';
import { flattenSceneNodes } from './schema/sceneTypes.js';
import { resolveTransformerView2dTokenEntryFromHoverPayload } from '../ui/selectionPanelTransformerView2dTokenHoverUtils.js';

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
    it('maps output-projection rows to the weighted-sum tooltip payload and mirrors row focus across the copy', () => {
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
        const copyMatrixNode = nodes.find((node) => (
            node?.role === 'concat-head-copy-matrix'
            && node?.semantic?.headIndex === 4
        )) || null;
        const otherMatrixNode = nodes.find((node) => (
            node?.role === 'head-output-matrix'
            && node?.semantic?.headIndex === 5
        )) || null;
        const otherCopyMatrixNode = nodes.find((node) => (
            node?.role === 'concat-head-copy-matrix'
            && node?.semantic?.headIndex === 1
        )) || null;
        const connectorNode = nodes.find((node) => (
            node?.role === 'head-output-connector'
            && node?.semantic?.headIndex === 4
        )) || null;
        const otherConnectorNode = nodes.find((node) => (
            node?.role === 'head-output-connector'
            && node?.semantic?.headIndex === 1
        )) || null;
        const copyConnectorNode = nodes.find((node) => (
            node?.role === 'concat-copy-connector'
            && node?.semantic?.headIndex === 4
        )) || null;
        const otherCopyConnectorNode = nodes.find((node) => (
            node?.role === 'concat-copy-connector'
            && node?.semantic?.headIndex === 1
        )) || null;
        const concatOutputNode = nodes.find((node) => (
            node?.role === 'concat-output-matrix'
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
        expect(hoverState?.focusState?.activeNodeIds).toContain(copyMatrixNode?.id);
        expect(hoverState?.focusState?.activeNodeIds).not.toContain(otherMatrixNode?.id);
        expect(hoverState?.focusState?.activeConnectorIds).toContain(connectorNode?.id);
        expect(hoverState?.focusState?.activeConnectorIds).toContain(copyConnectorNode?.id);
        expect(hoverState?.focusState?.rowSelections).toContainEqual({
            nodeId: matrixNode?.id,
            rowIndex: 1
        });
        expect(hoverState?.focusState?.rowSelections).toContainEqual({
            nodeId: copyMatrixNode?.id,
            rowIndex: 1
        });
        expect(hoverState?.focusState?.cellSelections).toContainEqual({
            nodeId: concatOutputNode?.id,
            rowIndex: 1,
            colIndex: 4
        });
        expect(resolveTransformerView2dTokenEntryFromHoverPayload(hoverState)).toEqual({
            tokenIndex: 1,
            tokenId: null,
            tokenLabel: 'Token B'
        });

        const mirroredHoverState = resolveMhsaDetailHoverState(index, {
            node: copyMatrixNode,
            rowHit: {
                rowIndex: 1,
                rowItem: copyMatrixNode?.rowItems?.[1]
            }
        });

        expect(mirroredHoverState?.focusState?.activeNodeIds).toContain(matrixNode?.id);
        expect(mirroredHoverState?.focusState?.activeNodeIds).toContain(copyMatrixNode?.id);
        expect(mirroredHoverState?.focusState?.activeConnectorIds).toContain(connectorNode?.id);
        expect(mirroredHoverState?.focusState?.activeConnectorIds).toContain(copyConnectorNode?.id);
        expect(mirroredHoverState?.focusState?.rowSelections).toContainEqual({
            nodeId: matrixNode?.id,
            rowIndex: 1
        });
        expect(mirroredHoverState?.focusState?.rowSelections).toContainEqual({
            nodeId: copyMatrixNode?.id,
            rowIndex: 1
        });
        expect(mirroredHoverState?.focusState?.cellSelections).toContainEqual({
            nodeId: concatOutputNode?.id,
            rowIndex: 1,
            colIndex: 4
        });
        expect(resolveTransformerView2dTokenEntryFromHoverPayload(mirroredHoverState)).toEqual({
            tokenIndex: 1,
            tokenId: null,
            tokenLabel: 'Token B'
        });

        const concatHoverState = resolveMhsaDetailHoverState(index, {
            node: concatOutputNode,
            cellHit: {
                rowIndex: 1,
                colIndex: 4,
                rowItem: concatOutputNode?.rowItems?.[1],
                cellItem: {
                    rowItem: concatOutputNode?.rowItems?.[1],
                    bandIndex: 4,
                    semantic: {
                        ...(concatOutputNode?.rowItems?.[1]?.semantic || {}),
                        rowIndex: 1,
                        colIndex: 4
                    }
                }
            }
        });

        expect(concatHoverState?.label).toBe('Attention Weighted Sum');
        expect(concatHoverState?.info?.activationData?.label).toBe('Attention Weighted Sum');
        expect(concatHoverState?.info?.activationData?.stage).toBe('attention.weighted_sum');
        expect(concatHoverState?.info?.activationData?.headIndex).toBe(4);
        expect(concatHoverState?.focusState?.activeNodeIds).toContain(matrixNode?.id);
        expect(concatHoverState?.focusState?.activeNodeIds).toContain(copyMatrixNode?.id);
        expect(concatHoverState?.focusState?.activeNodeIds).not.toContain(otherMatrixNode?.id);
        expect(concatHoverState?.focusState?.activeNodeIds).not.toContain(otherCopyMatrixNode?.id);
        expect(concatHoverState?.focusState?.activeNodeIds).toContain(concatOutputNode?.id);
        expect(concatHoverState?.focusState?.activeConnectorIds).toContain(connectorNode?.id);
        expect(concatHoverState?.focusState?.activeConnectorIds).not.toContain(otherConnectorNode?.id);
        expect(concatHoverState?.focusState?.activeConnectorIds).toContain(copyConnectorNode?.id);
        expect(concatHoverState?.focusState?.activeConnectorIds).not.toContain(otherCopyConnectorNode?.id);
        expect(concatHoverState?.focusState?.rowSelections).toContainEqual({
            nodeId: matrixNode?.id,
            rowIndex: 1
        });
        expect(concatHoverState?.focusState?.rowSelections).toContainEqual({
            nodeId: copyMatrixNode?.id,
            rowIndex: 1
        });
        expect(concatHoverState?.focusState?.cellSelections).toContainEqual({
            nodeId: concatOutputNode?.id,
            rowIndex: 1,
            colIndex: 4
        });
        expect(resolveTransformerView2dTokenEntryFromHoverPayload(concatHoverState)).toEqual({
            tokenIndex: 1,
            tokenId: null,
            tokenLabel: 'Token B'
        });
    });
});
