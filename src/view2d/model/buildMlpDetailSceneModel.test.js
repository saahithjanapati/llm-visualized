import { describe, expect, it } from 'vitest';

import { D_MODEL } from '../../ui/selectionPanelConstants.js';
import { buildSceneLayout } from '../layout/buildSceneLayout.js';
import { flattenSceneNodes, VIEW2D_NODE_KINDS } from '../schema/sceneTypes.js';
import { createMhsaDetailSceneIndex, resolveMhsaDetailHoverState } from '../mhsaDetailInteraction.js';
import { buildMlpDetailSceneModel } from './buildMlpDetailSceneModel.js';

function createVector(seed = 0, length = D_MODEL) {
    return Array.from({ length }, (_, index) => Number((seed + (index * 0.01)).toFixed(4)));
}

function createMockActivationSource() {
    return {
        getLayerLn2(_layerIndex = 0, _stage = 'shift', tokenIndex = 0, targetLength = D_MODEL) {
            return createVector(0.25 + tokenIndex, targetLength);
        }
    };
}

describe('buildMlpDetailSceneModel', () => {
    function buildScene() {
        return buildMlpDetailSceneModel({
            activationSource: createMockActivationSource(),
            mlpDetailTarget: {
                layerIndex: 4
            },
            tokenRefs: [
                {
                    rowIndex: 0,
                    tokenIndex: 0,
                    tokenLabel: 'Token A'
                },
                {
                    rowIndex: 1,
                    tokenIndex: 1,
                    tokenLabel: 'Token B'
                }
            ]
        });
    }

    it('reduces the detail scene to the incoming lower-case x_ln residual matrix', () => {
        const scene = buildScene();

        const nodes = flattenSceneNodes(scene);
        const matrixNodes = nodes.filter((node) => node?.kind === VIEW2D_NODE_KINDS.MATRIX);
        const connectorNodes = nodes.filter((node) => node?.kind === VIEW2D_NODE_KINDS.CONNECTOR);
        const inputNode = nodes.find((node) => node?.role === 'projection-source-xln') || null;
        const incomingSpacerNode = nodes.find((node) => node?.role === 'incoming-arrow-spacer') || null;
        const inputConnectorNode = nodes.find((node) => node?.role === 'connector-mlp-input') || null;

        expect(matrixNodes).toHaveLength(2);
        expect(connectorNodes).toHaveLength(1);
        expect(inputNode?.role).toBe('projection-source-xln');
        expect(inputNode?.label?.tex).toBe('x_{\\ln}');
        expect(inputNode?.label?.text).toBe('x_ln');
        expect(inputNode?.metadata?.disableEdgeOrnament).toBe(true);
        expect(inputNode?.semantic).toMatchObject({
            componentKind: 'mlp',
            layerIndex: 4,
            stage: 'mlp-input',
            role: 'projection-source-xln'
        });
        expect(inputNode?.metadata?.caption?.position).toBe('bottom');
        expect(inputNode?.metadata?.compactRows?.rowGap).toBe(0);
        expect(inputNode?.metadata?.compactRows?.paddingY).toBe(0);
        expect(inputNode?.rowItems).toHaveLength(2);
        expect(inputNode?.rowItems?.every((rowItem) => (
            rowItem?.semantic?.componentKind === 'residual'
            && rowItem?.semantic?.stage === 'ln2.shift'
            && rowItem?.semantic?.role === 'x-ln-row'
        ))).toBe(true);
        expect(nodes.some((node) => (
            node?.role === 'mlp-up-weight'
            || node?.role === 'mlp-down-weight'
            || node?.role === 'gelu-label'
            || node?.role === 'mlp-output'
        ))).toBe(false);
        expect(inputConnectorNode?.source).toMatchObject({
            nodeId: incomingSpacerNode?.id,
            anchor: 'left'
        });
        expect(inputConnectorNode?.target).toMatchObject({
            nodeId: inputNode?.id,
            anchor: 'left'
        });
        expect(inputConnectorNode?.targetGap).toBeGreaterThan(0);

        const layout = buildSceneLayout(scene);
        const inputEntry = layout?.registry?.getNodeEntry(inputNode?.id);
        expect(inputEntry?.contentBounds).toBeTruthy();
    });

    it('reuses the detail-scene row hover behavior for ln2 x_ln rows', () => {
        const scene = buildScene();
        const nodes = flattenSceneNodes(scene);
        const inputNode = nodes.find((node) => node?.role === 'projection-source-xln') || null;
        const index = createMhsaDetailSceneIndex(scene);
        const hoverState = resolveMhsaDetailHoverState(index, {
            node: inputNode,
            rowHit: {
                rowIndex: 1,
                rowItem: inputNode?.rowItems?.[1]
            }
        });

        expect(hoverState?.label).toBe('Post LayerNorm 2 Residual Vector');
        expect(hoverState?.info?.activationData?.stage).toBe('ln2.shift');
        expect(hoverState?.info?.activationData?.tokenIndex).toBe(1);
        expect(hoverState?.info?.activationData?.tokenLabel).toBe('Token B');
        expect(hoverState?.focusState?.activeNodeIds).toContain(inputNode?.id);
        expect(hoverState?.focusState?.rowSelections).toContainEqual({
            nodeId: inputNode?.id,
            rowIndex: 1
        });
    });
});
