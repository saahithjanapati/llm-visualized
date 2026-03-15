import { describe, expect, it } from 'vitest';

import {
    D_HEAD,
    D_MODEL
} from '../ui/selectionPanelConstants.js';
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
        },
        getAttentionOutputProjection(_layerIndex = 0, tokenIndex = 0, targetLength = D_MODEL) {
            return createVector(0.4 + (tokenIndex * 0.07), targetLength);
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
        const concatOutputCopyNode = nodes.find((node) => (
            node?.role === 'concat-output-copy-matrix'
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
        expect(hoverState?.info?.activationData?.values).toHaveLength(D_HEAD);
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
        expect(hoverState?.focusState?.rowSelections).toContainEqual({
            nodeId: concatOutputNode?.id,
            rowIndex: 1
        });
        expect(hoverState?.focusState?.rowSelections).toContainEqual({
            nodeId: concatOutputCopyNode?.id,
            rowIndex: 1
        });
        expect(hoverState?.focusState?.cellSelections).toContainEqual({
            nodeId: concatOutputNode?.id,
            rowIndex: 1,
            colIndex: 4
        });
        expect(hoverState?.focusState?.cellSelections).toContainEqual({
            nodeId: concatOutputCopyNode?.id,
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
        expect(mirroredHoverState?.focusState?.rowSelections).toContainEqual({
            nodeId: concatOutputNode?.id,
            rowIndex: 1
        });
        expect(mirroredHoverState?.focusState?.rowSelections).toContainEqual({
            nodeId: concatOutputCopyNode?.id,
            rowIndex: 1
        });
        expect(mirroredHoverState?.focusState?.cellSelections).toContainEqual({
            nodeId: concatOutputNode?.id,
            rowIndex: 1,
            colIndex: 4
        });
        expect(mirroredHoverState?.focusState?.cellSelections).toContainEqual({
            nodeId: concatOutputCopyNode?.id,
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

    it('focuses the post-concat output-projection equation nodes on the right-hand stage', () => {
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
        const concatOutputNode = nodes.find((node) => node?.role === 'concat-output-matrix') || null;
        const concatOutputCopyNode = nodes.find((node) => node?.role === 'concat-output-copy-matrix') || null;
        const projectionWeightNode = nodes.find((node) => node?.role === 'projection-weight') || null;
        const projectionBiasNode = nodes.find((node) => node?.role === 'projection-bias') || null;
        const projectionOutputNode = nodes.find((node) => node?.role === 'projection-output') || null;
        const concatToProjectionConnectorNode = nodes.find((node) => (
            node?.role === 'concat-output-projection-connector'
        )) || null;
        const projectionOutputConnectorNode = nodes.find((node) => (
            node?.role === 'projection-output-connector'
        )) || null;
        const headOutputMatrixNodes = nodes.filter((node) => node?.role === 'head-output-matrix');
        const concatHeadCopyMatrixNodes = nodes.filter((node) => node?.role === 'concat-head-copy-matrix');
        const headOutputRowCount = headOutputMatrixNodes.length;
        const copyHeadOutputRowCount = concatHeadCopyMatrixNodes.length;

        const outputHoverState = resolveMhsaDetailHoverState(index, {
            node: projectionOutputNode,
            rowHit: {
                rowIndex: 1,
                rowItem: projectionOutputNode?.rowItems?.[1]
            }
        });

        expect(outputHoverState?.label).toBe('Attention Output Vector');
        expect(outputHoverState?.info?.activationData?.stage).toBe('attention.output_projection');
        expect(outputHoverState?.info?.activationData?.tokenIndex).toBe(1);
        expect(outputHoverState?.info?.activationData?.values).toEqual(
            projectionOutputNode?.rowItems?.[1]?.rawValues
        );
        expect(outputHoverState?.focusState?.activeNodeIds).toContain(concatOutputNode?.id);
        expect(outputHoverState?.focusState?.activeNodeIds).toContain(concatOutputCopyNode?.id);
        expect(outputHoverState?.focusState?.activeNodeIds).toContain(projectionWeightNode?.id);
        expect(outputHoverState?.focusState?.activeNodeIds).toContain(projectionBiasNode?.id);
        expect(outputHoverState?.focusState?.activeNodeIds).toContain(projectionOutputNode?.id);
        expect(outputHoverState?.focusState?.activeConnectorIds).toContain(concatToProjectionConnectorNode?.id);
        expect(outputHoverState?.focusState?.activeConnectorIds).toContain(projectionOutputConnectorNode?.id);
        expect(outputHoverState?.focusState?.rowSelections).toContainEqual({
            nodeId: concatOutputNode?.id,
            rowIndex: 1
        });
        expect(outputHoverState?.focusState?.rowSelections).toContainEqual({
            nodeId: concatOutputCopyNode?.id,
            rowIndex: 1
        });
        expect(outputHoverState?.focusState?.rowSelections).toContainEqual({
            nodeId: projectionOutputNode?.id,
            rowIndex: 1
        });
        expect(headOutputRowCount).toBe(copyHeadOutputRowCount);
        expect(headOutputRowCount).toBeGreaterThan(0);
        headOutputMatrixNodes.forEach((matrixNode) => {
            expect(outputHoverState?.focusState?.rowSelections).toContainEqual({
                nodeId: matrixNode?.id,
                rowIndex: 1
            });
        });
        concatHeadCopyMatrixNodes.forEach((matrixNode) => {
            expect(outputHoverState?.focusState?.rowSelections).toContainEqual({
                nodeId: matrixNode?.id,
                rowIndex: 1
            });
        });
        expect(resolveTransformerView2dTokenEntryFromHoverPayload(outputHoverState)).toEqual({
            tokenIndex: 1,
            tokenId: null,
            tokenLabel: 'Token B'
        });

        const weightHoverState = resolveMhsaDetailHoverState(index, {
            node: projectionWeightNode
        });

        expect(weightHoverState?.label).toBe('Output Projection Matrix');
        expect(weightHoverState?.info?.activationData?.stage).toBe('attention.output_projection');
        expect(weightHoverState?.focusState?.activeNodeIds).toContain(concatOutputNode?.id);
        expect(weightHoverState?.focusState?.activeNodeIds).toContain(concatOutputCopyNode?.id);
        expect(weightHoverState?.focusState?.activeNodeIds).toContain(projectionWeightNode?.id);
        expect(weightHoverState?.focusState?.activeNodeIds).toContain(projectionBiasNode?.id);
        expect(weightHoverState?.focusState?.activeNodeIds).toContain(projectionOutputNode?.id);
        expect(weightHoverState?.focusState?.activeConnectorIds).toContain(concatToProjectionConnectorNode?.id);
        expect(weightHoverState?.focusState?.activeConnectorIds).toContain(projectionOutputConnectorNode?.id);
    });

    it('maps the output-projection bias row to the bias tooltip payload and keeps the projection path focused', () => {
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
        const concatOutputNode = nodes.find((node) => node?.role === 'concat-output-matrix') || null;
        const concatOutputCopyNode = nodes.find((node) => node?.role === 'concat-output-copy-matrix') || null;
        const projectionWeightNode = nodes.find((node) => node?.role === 'projection-weight') || null;
        const projectionBiasNode = nodes.find((node) => node?.role === 'projection-bias') || null;
        const projectionOutputNode = nodes.find((node) => node?.role === 'projection-output') || null;
        const concatToProjectionConnectorNode = nodes.find((node) => (
            node?.role === 'concat-output-projection-connector'
        )) || null;
        const projectionOutputConnectorNode = nodes.find((node) => (
            node?.role === 'projection-output-connector'
        )) || null;

        const hoverState = resolveMhsaDetailHoverState(index, {
            node: projectionBiasNode,
            rowHit: {
                rowIndex: 0,
                rowItem: projectionBiasNode?.rowItems?.[0]
            }
        });

        expect(hoverState?.label).toBe('Output Projection Bias Vector');
        expect(hoverState?.info?.activationData?.stage).toBe('attention.output_projection.bias');
        expect(hoverState?.info?.activationData?.parameterType).toBe('bias');
        expect(hoverState?.info?.activationData?.values).toHaveLength(12);
        expect(hoverState?.focusState?.activeNodeIds).toContain(concatOutputNode?.id);
        expect(hoverState?.focusState?.activeNodeIds).toContain(concatOutputCopyNode?.id);
        expect(hoverState?.focusState?.activeNodeIds).toContain(projectionWeightNode?.id);
        expect(hoverState?.focusState?.activeNodeIds).toContain(projectionBiasNode?.id);
        expect(hoverState?.focusState?.activeNodeIds).toContain(projectionOutputNode?.id);
        expect(hoverState?.focusState?.activeConnectorIds).toContain(concatToProjectionConnectorNode?.id);
        expect(hoverState?.focusState?.activeConnectorIds).toContain(projectionOutputConnectorNode?.id);
    });
});
