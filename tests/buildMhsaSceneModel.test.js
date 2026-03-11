import { describe, expect, it } from 'vitest';
import { D_HEAD, D_MODEL } from '../src/ui/selectionPanelConstants.js';
import { buildMhsaSceneModel } from '../src/view2d/model/buildMhsaSceneModel.js';
import { flattenSceneNodes } from '../src/view2d/schema/sceneTypes.js';

function createActivationSource(tokenCount = 3) {
    const promptTokens = Array.from({ length: tokenCount }, (_, index) => index);
    const tokenDisplayStrings = promptTokens.map((index) => `tok_${index}`);

    return {
        meta: {
            prompt_tokens: promptTokens,
            token_display_strings: tokenDisplayStrings
        },
        getLayerLn1(layerIndex, mode, tokenIndex) {
            if (mode !== 'shift') return null;
            return Array.from({ length: D_MODEL }, (_, index) => {
                const centered = (index % 9) - 4;
                return (tokenIndex + 1) * centered * 0.0625;
            });
        },
        getLayerQKVVector(layerIndex, kind, headIndex, tokenIndex, width) {
            const scale = kind === 'q' ? 0.125 : (kind === 'k' ? 0.175 : 0.225);
            return Array.from({ length: width }, (_, index) => {
                const centered = (index % 11) - 5;
                return (tokenIndex + 1) * centered * scale;
            });
        },
        getLayerQKVScalar(layerIndex, kind, headIndex, tokenIndex) {
            const scale = kind === 'q' ? 0.2 : (kind === 'k' ? 0.28 : 0.34);
            return (tokenIndex + 1) * scale;
        },
        getAttentionScoresRow(layerIndex, stage, headIndex, tokenIndex) {
            return Array.from({ length: tokenCount }, (_, columnIndex) => {
                if (columnIndex > tokenIndex) {
                    return stage === 'pre' ? -1000 : 0;
                }
                if (stage === 'pre') {
                    return ((tokenIndex + 1) * (columnIndex + 1)) / 5;
                }
                return 1 / (tokenIndex + 1);
            });
        },
        getAttentionWeightedSum(layerIndex, headIndex, tokenIndex, width) {
            return Array.from({ length: width }, (_, index) => {
                const centered = (index % 13) - 6;
                return (tokenIndex + 1) * centered * 0.08;
            });
        }
    };
}

describe('buildMhsaSceneModel', () => {
    it('adapts MHSA preview data into generic matrix, operator, and connector nodes', () => {
        const scene = buildMhsaSceneModel({
            activationSource: createActivationSource(3),
            layerIndex: 2,
            headIndex: 4,
            sampleStep: 256
        });

        expect(scene).not.toBeNull();
        expect(scene.semantic.componentKind).toBe('mhsa');
        expect(scene.metadata.rowCount).toBe(3);

        const nodes = flattenSceneNodes(scene);
        const qOutputNode = nodes.find((node) => node.role === 'projection-output' && node.metadata?.kind === 'q');
        const postNode = nodes.find((node) => node.role === 'attention-post');
        const headOutputNode = nodes.find((node) => node.role === 'attention-head-output');
        const connectorRoles = nodes
            .filter((node) => node.kind === 'connector')
            .map((node) => node.role);

        expect(qOutputNode?.dimensions).toEqual({ rows: 3, cols: D_HEAD });
        expect(postNode?.dimensions).toEqual({ rows: 3, cols: 3 });
        expect(postNode?.rowItems).toHaveLength(3);
        expect(postNode?.rowItems[2]?.cells).toHaveLength(3);
        expect(headOutputNode?.dimensions).toEqual({ rows: 3, cols: D_HEAD });
        expect(connectorRoles).toEqual(expect.arrayContaining([
            'connector-q',
            'connector-k',
            'connector-pre',
            'connector-post',
            'connector-v'
        ]));
    });

    it('scales matrix dimensions and layout metadata as token counts grow', () => {
        const smallScene = buildMhsaSceneModel({
            activationSource: createActivationSource(3),
            layerIndex: 1,
            headIndex: 1,
            sampleStep: 256
        });
        const largeScene = buildMhsaSceneModel({
            activationSource: createActivationSource(7),
            layerIndex: 1,
            headIndex: 1,
            sampleStep: 256
        });

        expect(largeScene.metadata.rowCount).toBe(7);
        expect(largeScene.metadata.layoutMetrics.connectorGaps.default)
            .toBeGreaterThan(smallScene.metadata.layoutMetrics.connectorGaps.default);

        const largeNodes = flattenSceneNodes(largeScene);
        const largeQOutputNode = largeNodes.find((node) => node.role === 'projection-output' && node.metadata?.kind === 'q');
        const largePostNode = largeNodes.find((node) => node.role === 'attention-post');

        expect(largeQOutputNode?.dimensions).toEqual({ rows: 7, cols: D_HEAD });
        expect(largePostNode?.dimensions).toEqual({ rows: 7, cols: 7 });
        expect(largePostNode?.rowItems).toHaveLength(7);
        expect(largePostNode?.rowItems[6]?.cells).toHaveLength(7);
    });
});
