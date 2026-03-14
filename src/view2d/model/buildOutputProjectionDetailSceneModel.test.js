import { describe, expect, it } from 'vitest';

import { D_HEAD } from '../../ui/selectionPanelConstants.js';
import { buildSceneLayout } from '../layout/buildSceneLayout.js';
import {
    flattenSceneNodes,
    VIEW2D_ANCHOR_SIDES,
    VIEW2D_NODE_KINDS
} from '../schema/sceneTypes.js';
import { buildOutputProjectionDetailSceneModel } from './buildOutputProjectionDetailSceneModel.js';

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

        expect(scene?.metadata?.visualContract).toBe('selection-panel-output-projection-v1');
        expect(matrixNodes).toHaveLength(12);
        expect(connectorNodes).toHaveLength(12);

        expect(matrixNodes[0]?.label?.tex).toBe('H_{1}');
        expect(matrixNodes[11]?.label?.tex).toBe('H_{12}');
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

        expect(firstMatrixEntry?.contentBounds).toBeTruthy();
        expect(firstConnectorEntry?.pathPoints?.length).toBeGreaterThan(1);
        expect(firstConnectorEntry?.pathPoints?.at(-1)?.x).toBeLessThan(
            firstMatrixEntry?.anchors?.[VIEW2D_ANCHOR_SIDES.LEFT]?.x ?? Number.POSITIVE_INFINITY
        );
    });
});
