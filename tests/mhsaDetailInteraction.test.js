import { describe, expect, it } from 'vitest';
import { D_HEAD, D_MODEL } from '../src/ui/selectionPanelConstants.js';
import { buildMhsaSceneModel } from '../src/view2d/model/buildMhsaSceneModel.js';
import {
    createMhsaDetailSceneIndex,
    resolveMhsaDetailHoverState
} from '../src/view2d/mhsaDetailInteraction.js';
import { flattenSceneNodes } from '../src/view2d/schema/sceneTypes.js';

function createActivationSource(tokenCount = 4) {
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

describe('mhsaDetailInteraction', () => {
    it('uses the same branch focus for W_q hover as the corresponding query path', () => {
        const scene = buildMhsaSceneModel({
            activationSource: createActivationSource(4),
            layerIndex: 2,
            headIndex: 3,
            sampleStep: 256
        });
        const index = createMhsaDetailSceneIndex(scene);
        const nodes = flattenSceneNodes(scene);
        const qWeightNode = nodes.find((node) => node.role === 'projection-weight' && node.metadata?.kind === 'q');
        const qQueryNode = nodes.find((node) => node.role === 'attention-query-source');
        const qIngressConnector = nodes.find((node) => node.role === 'connector-xln-q');

        const weightHover = resolveMhsaDetailHoverState(index, {
            node: qWeightNode
        });
        const queryHover = resolveMhsaDetailHoverState(index, {
            node: qQueryNode
        });

        expect(weightHover?.label).toBe('Query Weight Matrix');
        expect(weightHover?.info).toEqual({
            layerIndex: 2,
            headIndex: 3,
            activationData: {
                label: 'Query Weight Matrix',
                layerIndex: 2,
                headIndex: 3
            }
        });
        expect(weightHover?.focusState?.activeConnectorIds).toContain(qIngressConnector?.id);
        expect(weightHover?.focusState).toEqual(queryHover?.focusState);
    });

    it('uses the same branch focus for W_k hover as the corresponding key path', () => {
        const scene = buildMhsaSceneModel({
            activationSource: createActivationSource(4),
            layerIndex: 2,
            headIndex: 3,
            sampleStep: 256
        });
        const index = createMhsaDetailSceneIndex(scene);
        const nodes = flattenSceneNodes(scene);
        const kWeightNode = nodes.find((node) => node.role === 'projection-weight' && node.metadata?.kind === 'k');
        const kTransposeNode = nodes.find((node) => node.role === 'attention-key-transpose');
        const kIngressConnector = nodes.find((node) => node.role === 'connector-xln-k');

        const weightHover = resolveMhsaDetailHoverState(index, {
            node: kWeightNode
        });
        const keyHover = resolveMhsaDetailHoverState(index, {
            node: kTransposeNode
        });

        expect(weightHover?.focusState?.activeConnectorIds).toContain(kIngressConnector?.id);
        expect(weightHover?.focusState).toEqual(keyHover?.focusState);
    });

    it('uses the same branch focus for W_v hover as the corresponding value path', () => {
        const scene = buildMhsaSceneModel({
            activationSource: createActivationSource(4),
            layerIndex: 2,
            headIndex: 3,
            sampleStep: 256
        });
        const index = createMhsaDetailSceneIndex(scene);
        const nodes = flattenSceneNodes(scene);
        const vWeightNode = nodes.find((node) => node.role === 'projection-weight' && node.metadata?.kind === 'v');
        const vValueNode = nodes.find((node) => node.role === 'attention-value-post');
        const vIngressConnector = nodes.find((node) => node.role === 'connector-xln-v');

        const weightHover = resolveMhsaDetailHoverState(index, {
            node: vWeightNode
        });
        const valueHover = resolveMhsaDetailHoverState(index, {
            node: vValueNode
        });

        expect(weightHover?.focusState?.activeConnectorIds).toContain(vIngressConnector?.id);
        expect(weightHover?.focusState).toEqual(valueHover?.focusState);
    });

    it('uses the same branch focus and head/layer tooltip metadata for bias-vector hover', () => {
        const scene = buildMhsaSceneModel({
            activationSource: createActivationSource(4),
            layerIndex: 0,
            headIndex: 4,
            sampleStep: 256
        });
        const index = createMhsaDetailSceneIndex(scene);
        const nodes = flattenSceneNodes(scene);
        const vBiasNode = nodes.find((node) => node.role === 'projection-bias' && node.metadata?.kind === 'v');
        const vWeightNode = nodes.find((node) => node.role === 'projection-weight' && node.metadata?.kind === 'v');

        const biasHover = resolveMhsaDetailHoverState(index, {
            node: vBiasNode,
            rowHit: {
                rowIndex: 0,
                rowItem: vBiasNode?.rowItems?.[0] || null
            }
        });
        const weightHover = resolveMhsaDetailHoverState(index, {
            node: vWeightNode
        });

        expect(biasHover?.label).toBe('Value Bias Vector');
        expect(biasHover?.info).toEqual({
            layerIndex: 0,
            headIndex: 4,
            activationData: {
                label: 'Value Bias Vector',
                layerIndex: 0,
                headIndex: 4
            }
        });
        expect(biasHover?.focusState).toEqual(weightHover?.focusState);
    });

    it('routes projection-output row hover to the corresponding Q/K/V vector tooltip metadata', () => {
        const scene = buildMhsaSceneModel({
            activationSource: createActivationSource(5),
            layerIndex: 1,
            headIndex: 4,
            sampleStep: 256
        });
        const index = createMhsaDetailSceneIndex(scene);
        const nodes = flattenSceneNodes(scene);
        const qOutputNode = nodes.find((node) => node.role === 'projection-output' && node.metadata?.kind === 'q');
        const qOutputRow = qOutputNode?.rowItems?.[2] || null;
        const qQueryNode = nodes.find((node) => node.role === 'attention-query-source');
        const qQueryRow = qQueryNode?.rowItems?.[2] || null;

        const outputHover = resolveMhsaDetailHoverState(index, {
            node: qOutputNode,
            rowHit: {
                rowIndex: 2,
                rowItem: qOutputRow
            }
        });
        const queryHover = resolveMhsaDetailHoverState(index, {
            node: qQueryNode,
            rowHit: {
                rowIndex: 2,
                rowItem: qQueryRow
            }
        });

        expect(outputHover?.label).toBe('Query Vector');
        expect(outputHover?.info).toEqual({
            tokenLabel: 'tok_2',
            tokenIndex: 2,
            layerIndex: 1,
            headIndex: 4,
            activationData: {
                label: 'Query Vector',
                stage: 'qkv.q',
                tokenIndex: 2,
                tokenLabel: 'tok_2',
                layerIndex: 1,
                headIndex: 4
            }
        });
        expect(outputHover?.focusState).toEqual(queryHover?.focusState);
        expect(queryHover?.label).toBe('Query Vector');
        expect(queryHover?.info).toEqual({
            tokenLabel: 'tok_2',
            tokenIndex: 2,
            layerIndex: 1,
            headIndex: 4,
            activationData: {
                label: 'Query Vector',
                stage: 'qkv.q',
                tokenIndex: 2,
                tokenLabel: 'tok_2',
                layerIndex: 1,
                headIndex: 4
            }
        });
    });

    it('routes K^T column hover to the corresponding key-vector tooltip metadata', () => {
        const scene = buildMhsaSceneModel({
            activationSource: createActivationSource(5),
            layerIndex: 1,
            headIndex: 4,
            sampleStep: 256
        });
        const index = createMhsaDetailSceneIndex(scene);
        const nodes = flattenSceneNodes(scene);
        const transposeNode = nodes.find((node) => node.role === 'attention-key-transpose');
        const transposeColumn = transposeNode?.columnItems?.[3] || null;
        const kWeightNode = nodes.find((node) => node.role === 'projection-weight' && node.metadata?.kind === 'k');

        const transposeHover = resolveMhsaDetailHoverState(index, {
            node: transposeNode,
            columnHit: {
                colIndex: 3,
                columnItem: transposeColumn
            }
        });
        const weightHover = resolveMhsaDetailHoverState(index, {
            node: kWeightNode
        });

        expect(transposeHover?.label).toBe('Key Vector');
        expect(transposeHover?.info).toEqual({
            tokenLabel: 'tok_3',
            tokenIndex: 3,
            layerIndex: 1,
            headIndex: 4,
            activationData: {
                label: 'Key Vector',
                stage: 'qkv.k',
                tokenIndex: 3,
                tokenLabel: 'tok_3',
                layerIndex: 1,
                headIndex: 4
            }
        });
        expect(weightHover?.focusState?.activeNodeIds.every((nodeId) => (
            transposeHover?.focusState?.activeNodeIds.includes(nodeId)
        ))).toBe(true);
        expect(transposeHover?.focusState?.activeConnectorIds).toEqual(weightHover?.focusState?.activeConnectorIds);
        expect(transposeHover?.focusState?.columnSelections).toContainEqual({
            nodeId: transposeNode?.id,
            colIndex: 3
        });
    });
});
