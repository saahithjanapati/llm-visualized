import { describe, expect, it } from 'vitest';

import {
    D_HEAD,
    D_MODEL
} from '../../ui/selectionPanelConstants.js';
import { buildSceneLayout } from '../layout/buildSceneLayout.js';
import {
    flattenSceneNodes,
    VIEW2D_ANCHOR_SIDES,
    VIEW2D_NODE_KINDS
} from '../schema/sceneTypes.js';
import { VIEW2D_STYLE_KEYS } from '../theme/visualTokens.js';
import { buildOutputProjectionDetailSceneModel } from './buildOutputProjectionDetailSceneModel.js';

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

describe('buildOutputProjectionDetailSceneModel', () => {
    it('builds 12 vertically stacked H_i matrices with left-side arrow connectors', () => {
        const scene = buildOutputProjectionDetailSceneModel({
            activationSource: createMockActivationSource(),
            outputProjectionDetailTarget: {
                layerIndex: 3
            },
            tokenRefs: [
                { rowIndex: 0, tokenIndex: 0, tokenLabel: 'Token A' },
                { rowIndex: 1, tokenIndex: 1, tokenLabel: 'Token B' },
                { rowIndex: 2, tokenIndex: 2, tokenLabel: 'Token C' }
            ]
        });

        const nodes = flattenSceneNodes(scene);
        const matrixNodes = nodes.filter((node) => (
            node?.kind === VIEW2D_NODE_KINDS.MATRIX
            && node?.role === 'head-output-matrix'
        ));
        const connectorNodes = nodes.filter((node) => (
            node?.kind === VIEW2D_NODE_KINDS.CONNECTOR
            && node?.role === 'head-output-connector'
        ));
        const concatCopyConnectorNodes = nodes.filter((node) => (
            node?.kind === VIEW2D_NODE_KINDS.CONNECTOR
            && node?.role === 'concat-copy-connector'
        ));
        const concatLabelNode = nodes.find((node) => (
            node?.kind === VIEW2D_NODE_KINDS.TEXT
            && node?.role === 'concat-label'
        ));
        const concatOpenNode = nodes.find((node) => (
            node?.kind === VIEW2D_NODE_KINDS.OPERATOR
            && node?.role === 'concat-open'
        ));
        const concatCloseNode = nodes.find((node) => (
            node?.kind === VIEW2D_NODE_KINDS.OPERATOR
            && node?.role === 'concat-close'
        ));
        const concatEqualsNode = nodes.find((node) => (
            node?.kind === VIEW2D_NODE_KINDS.OPERATOR
            && node?.role === 'concat-equals'
        ));
        const concatHeadNodes = nodes.filter((node) => (
            node?.kind === VIEW2D_NODE_KINDS.MATRIX
            && node?.role === 'concat-head-copy-matrix'
        ));
        const headStackTopSpacerNode = nodes.find((node) => (
            node?.kind === VIEW2D_NODE_KINDS.MATRIX
            && node?.role === 'output-projection-detail-head-stack-top-spacer'
        ));
        const concatOutputNode = nodes.find((node) => (
            node?.kind === VIEW2D_NODE_KINDS.MATRIX
            && node?.role === 'concat-output-matrix'
        ));
        const concatOutputCopyNode = nodes.find((node) => (
            node?.kind === VIEW2D_NODE_KINDS.MATRIX
            && node?.role === 'concat-output-copy-matrix'
        ));
        const projectionWeightNode = nodes.find((node) => (
            node?.kind === VIEW2D_NODE_KINDS.MATRIX
            && node?.role === 'projection-weight'
        ));
        const projectionBiasNode = nodes.find((node) => (
            node?.kind === VIEW2D_NODE_KINDS.MATRIX
            && node?.role === 'projection-bias'
        ));
        const projectionOutputNode = nodes.find((node) => (
            node?.kind === VIEW2D_NODE_KINDS.MATRIX
            && node?.role === 'projection-output'
        ));
        const concatToProjectionConnectorNode = nodes.find((node) => (
            node?.kind === VIEW2D_NODE_KINDS.CONNECTOR
            && node?.role === 'concat-output-projection-connector'
        ));
        const projectionOutputConnectorNode = nodes.find((node) => (
            node?.kind === VIEW2D_NODE_KINDS.CONNECTOR
            && node?.role === 'projection-output-connector'
        ));
        const projectionMultiplyNode = nodes.find((node) => (
            node?.kind === VIEW2D_NODE_KINDS.OPERATOR
            && node?.role === 'projection-multiply'
        ));
        const projectionPlusNode = nodes.find((node) => (
            node?.kind === VIEW2D_NODE_KINDS.OPERATOR
            && node?.role === 'projection-plus'
        ));
        const projectionEqualsNode = nodes.find((node) => (
            node?.kind === VIEW2D_NODE_KINDS.OPERATOR
            && node?.role === 'projection-equals'
        ));
        const concatSeparatorNodes = nodes.filter((node) => (
            node?.kind === VIEW2D_NODE_KINDS.OPERATOR
            && node?.role === 'concat-separator'
        ));

        expect(scene?.metadata?.visualContract).toBe('selection-panel-output-projection-v1');
        expect(matrixNodes).toHaveLength(12);
        expect(connectorNodes).toHaveLength(12);
        expect(concatCopyConnectorNodes).toHaveLength(12);
        expect(concatLabelNode?.tex).toBe('\\mathrm{concat}');
        expect(concatOpenNode?.text).toBe('(');
        expect(concatCloseNode?.text).toBe(')');
        expect(concatEqualsNode?.text).toBe('=');
        expect(concatCopyConnectorNodes[0]?.metadata?.targetAnchorMode).toBe('caption-bottom');
        expect(concatHeadNodes).toHaveLength(12);
        expect(headStackTopSpacerNode).toBeTruthy();
        expect(concatSeparatorNodes).toHaveLength(11);
        expect(concatOutputNode?.label?.tex).toBe('H_{\\mathrm{concat}}');
        expect(concatOutputCopyNode?.label?.tex).toBe('H_{\\mathrm{concat}}');
        expect(concatOutputNode?.dimensions).toEqual({
            rows: 3,
            cols: D_HEAD * 12
        });
        expect(concatOutputCopyNode?.semantic).toMatchObject({
            componentKind: 'output-projection',
            layerIndex: 3,
            stage: 'attn-out',
            role: 'concat-output-copy-matrix'
        });
        expect(concatOutputNode?.rowItems).toHaveLength(3);
        expect(concatOutputNode?.metadata?.compactRows?.bandCount).toBe(12);
        expect(concatOutputNode?.metadata?.compactRows?.bandSeparatorOpacity).toBeGreaterThan(0);
        expect(projectionWeightNode?.label?.tex).toBe('W_{\\mathrm{O}}');
        expect(projectionWeightNode?.visual?.styleKey).toBe(VIEW2D_STYLE_KEYS.OUTPUT_PROJECTION);
        expect(projectionWeightNode?.dimensions).toEqual({
            rows: D_MODEL,
            cols: D_MODEL
        });
        expect(projectionWeightNode?.metadata?.caption?.sceneRelativeExtentExponent).toBeCloseTo(0.72, 5);
        expect(projectionBiasNode?.label?.tex).toBe('b_{\\mathrm{O}}');
        expect(projectionBiasNode?.dimensions).toEqual({
            rows: 1,
            cols: D_MODEL
        });
        expect(projectionOutputNode?.label?.tex).toBe('O');
        expect(projectionOutputNode?.visual?.styleKey).toBe(VIEW2D_STYLE_KEYS.RESIDUAL);
        expect(projectionOutputNode?.dimensions).toEqual({
            rows: 3,
            cols: D_MODEL
        });
        expect(projectionOutputNode?.rowItems?.[1]?.gradientCss).not.toBe('none');
        expect(projectionOutputNode?.rowItems?.[1]?.semantic).toMatchObject({
            componentKind: 'output-projection',
            layerIndex: 3,
            stage: 'attn-out',
            role: 'projection-output-row',
            rowIndex: 1,
            tokenIndex: 1
        });
        expect(concatToProjectionConnectorNode).toBeTruthy();
        expect(projectionOutputConnectorNode).toBeTruthy();
        expect(projectionMultiplyNode?.text).toBe('×');
        expect(projectionPlusNode?.text).toBe('+');
        expect(projectionEqualsNode?.text).toBe('=');

        expect(matrixNodes[0]?.label?.tex).toBe('H_{1}');
        expect(matrixNodes[11]?.label?.tex).toBe('H_{12}');
        expect(concatHeadNodes[0]?.label?.tex).toBe('H_{1}');
        expect(concatHeadNodes[11]?.label?.tex).toBe('H_{12}');
        expect(concatHeadNodes[0]?.rowItems).toHaveLength(3);
        expect(concatHeadNodes[0]?.metadata?.compactRows?.compactWidth).toBe(
            matrixNodes[0]?.metadata?.compactRows?.compactWidth
        );
        expect(concatHeadNodes[0]?.metadata?.compactRows?.rowHeight).toBe(
            matrixNodes[0]?.metadata?.compactRows?.rowHeight
        );
        expect(matrixNodes[0]?.semantic).toMatchObject({
            componentKind: 'output-projection',
            layerIndex: 3,
            headIndex: 0,
            stage: 'head-output',
            role: 'head-output-matrix'
        });
        expect(matrixNodes[7]?.semantic?.headIndex).toBe(7);
        expect(matrixNodes[0]?.rowItems).toHaveLength(3);
        expect(matrixNodes[0]?.rowItems?.[1]?.gradientCss).not.toBe('none');
        expect(matrixNodes[0]?.rowItems?.[1]?.semantic).toMatchObject({
            componentKind: 'output-projection',
            layerIndex: 3,
            headIndex: 0,
            stage: 'head-output',
            role: 'head-output-row',
            rowIndex: 1,
            tokenIndex: 1
        });

        const layout = buildSceneLayout(scene);
        const firstMatrixEntry = layout?.registry?.getNodeEntry(matrixNodes[0]?.id);
        const firstConnectorEntry = layout?.registry?.getConnectorEntry(connectorNodes[0]?.id);
        const firstConcatCopyConnectorEntry = layout?.registry?.getConnectorEntry(concatCopyConnectorNodes[0]?.id);
        const headStackTopSpacerEntry = layout?.registry?.getNodeEntry(headStackTopSpacerNode?.id);
        const concatLabelEntry = layout?.registry?.getNodeEntry(concatLabelNode?.id);
        const concatOpenEntry = layout?.registry?.getNodeEntry(concatOpenNode?.id);
        const firstConcatHeadEntry = layout?.registry?.getNodeEntry(concatHeadNodes[0]?.id);
        const concatCloseEntry = layout?.registry?.getNodeEntry(concatCloseNode?.id);
        const concatEqualsEntry = layout?.registry?.getNodeEntry(concatEqualsNode?.id);
        const concatOutputEntry = layout?.registry?.getNodeEntry(concatOutputNode?.id);
        const concatOutputCopyEntry = layout?.registry?.getNodeEntry(concatOutputCopyNode?.id);
        const projectionWeightEntry = layout?.registry?.getNodeEntry(projectionWeightNode?.id);
        const projectionBiasEntry = layout?.registry?.getNodeEntry(projectionBiasNode?.id);
        const projectionOutputEntry = layout?.registry?.getNodeEntry(projectionOutputNode?.id);
        const concatToProjectionConnectorEntry = layout?.registry?.getConnectorEntry(concatToProjectionConnectorNode?.id);
        const projectionOutputConnectorEntry = layout?.registry?.getConnectorEntry(projectionOutputConnectorNode?.id);
        const projectionMultiplyEntry = layout?.registry?.getNodeEntry(projectionMultiplyNode?.id);
        const projectionPlusEntry = layout?.registry?.getNodeEntry(projectionPlusNode?.id);
        const projectionEqualsEntry = layout?.registry?.getNodeEntry(projectionEqualsNode?.id);

        expect(firstMatrixEntry?.contentBounds).toBeTruthy();
        expect(firstMatrixEntry?.contentBounds?.y).toBeGreaterThanOrEqual(
            (headStackTopSpacerEntry?.contentBounds?.y ?? 0)
            + (headStackTopSpacerEntry?.contentBounds?.height ?? 0)
        );
        expect(firstConnectorEntry?.pathPoints?.length).toBeGreaterThan(1);
        expect(firstConcatCopyConnectorEntry?.pathPoints?.length).toBeGreaterThan(1);
        expect(concatLabelEntry?.contentBounds?.x).toBeGreaterThan(
            (firstMatrixEntry?.contentBounds?.x ?? 0) + (firstMatrixEntry?.contentBounds?.width ?? 0)
        );
        expect(concatOpenEntry?.contentBounds?.x).toBeGreaterThan(
            (concatLabelEntry?.contentBounds?.x ?? 0) + (concatLabelEntry?.contentBounds?.width ?? 0)
        );
        expect(firstConcatHeadEntry?.contentBounds?.x).toBeGreaterThan(
            (concatOpenEntry?.contentBounds?.x ?? 0) + (concatOpenEntry?.contentBounds?.width ?? 0)
        );
        expect(concatEqualsEntry?.contentBounds?.x).toBeGreaterThan(
            (concatCloseEntry?.contentBounds?.x ?? 0) + (concatCloseEntry?.contentBounds?.width ?? 0)
        );
        expect(concatOutputEntry?.contentBounds?.x).toBeGreaterThan(
            (concatEqualsEntry?.contentBounds?.x ?? 0) + (concatEqualsEntry?.contentBounds?.width ?? 0)
        );
        expect(concatOutputCopyEntry?.contentBounds?.x).toBeGreaterThan(
            (concatOutputEntry?.contentBounds?.x ?? 0) + (concatOutputEntry?.contentBounds?.width ?? 0)
        );
        expect(projectionWeightEntry?.contentBounds?.x).toBeGreaterThan(
            (projectionMultiplyEntry?.contentBounds?.x ?? 0) + (projectionMultiplyEntry?.contentBounds?.width ?? 0)
        );
        expect(projectionBiasEntry?.contentBounds?.x).toBeGreaterThan(
            (projectionPlusEntry?.contentBounds?.x ?? 0) + (projectionPlusEntry?.contentBounds?.width ?? 0)
        );
        expect(projectionOutputEntry?.contentBounds?.x).toBeGreaterThan(
            (projectionEqualsEntry?.contentBounds?.x ?? 0) + (projectionEqualsEntry?.contentBounds?.width ?? 0)
        );
        expect(
            (firstConcatHeadEntry?.contentBounds?.y ?? Number.POSITIVE_INFINITY)
            + (firstConcatHeadEntry?.contentBounds?.height ?? 0)
        ).toBeLessThan(
            firstMatrixEntry?.contentBounds?.y ?? Number.NEGATIVE_INFINITY
        );
        expect(concatToProjectionConnectorEntry?.pathPoints?.length).toBeGreaterThan(1);
        expect(projectionOutputConnectorEntry?.pathPoints?.length).toBeGreaterThan(1);
        expect(firstConnectorEntry?.pathPoints?.at(-1)?.x).toBeLessThan(
            firstMatrixEntry?.anchors?.[VIEW2D_ANCHOR_SIDES.LEFT]?.x ?? Number.POSITIVE_INFINITY
        );
        expect(concatOutputCopyEntry?.contentBounds?.y).toBeLessThan(
            firstMatrixEntry?.contentBounds?.y ?? Number.POSITIVE_INFINITY
        );
        expect(Math.abs(
            (concatOutputCopyEntry?.contentBounds?.y ?? 0)
            - (concatOutputEntry?.contentBounds?.y ?? 0)
        )).toBeLessThan(8);
    });
});
