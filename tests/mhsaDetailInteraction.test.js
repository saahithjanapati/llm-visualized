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
    it('narrows W_q hover to the local X_ln, Q, and query-source branch', () => {
        const scene = buildMhsaSceneModel({
            activationSource: createActivationSource(4),
            layerIndex: 2,
            headIndex: 3,
            sampleStep: 256
        });
        const index = createMhsaDetailSceneIndex(scene);
        const nodes = flattenSceneNodes(scene);
        const sourceNode = nodes.find((node) => node.role === 'projection-source-xln');
        const qInputNode = nodes.find((node) => node.role === 'x-ln-copy' && node.semantic?.branchKey === 'q');
        const qWeightNode = nodes.find((node) => node.role === 'projection-weight' && node.metadata?.kind === 'q');
        const qValueNode = nodes.find((node) => node.role === 'projection-output' && node.metadata?.kind === 'q');
        const qQueryNode = nodes.find((node) => node.role === 'attention-query-source');
        const kValueNode = nodes.find((node) => node.role === 'projection-output' && node.metadata?.kind === 'k');
        const vValueNode = nodes.find((node) => node.role === 'projection-output' && node.metadata?.kind === 'v');
        const qIngressConnector = nodes.find((node) => node.role === 'connector-xln-q');
        const qConnectorNode = nodes.find((node) => node.role === 'connector-q');

        const weightHover = resolveMhsaDetailHoverState(index, { node: qWeightNode });

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
        expect(weightHover?.focusState?.activeNodeIds).toContain(sourceNode?.id);
        expect(weightHover?.focusState?.activeNodeIds).toContain(qInputNode?.id);
        expect(weightHover?.focusState?.activeNodeIds).toContain(qWeightNode?.id);
        expect(weightHover?.focusState?.activeNodeIds).toContain(qValueNode?.id);
        expect(weightHover?.focusState?.activeNodeIds).toContain(qQueryNode?.id);
        expect(weightHover?.focusState?.activeNodeIds).not.toContain(kValueNode?.id);
        expect(weightHover?.focusState?.activeNodeIds).not.toContain(vValueNode?.id);
        expect(weightHover?.focusState?.activeConnectorIds).toContain(qIngressConnector?.id);
        expect(weightHover?.focusState?.activeConnectorIds).toContain(qConnectorNode?.id);
    });

    it('narrows W_k hover to the local X_ln, K, and K^T branch', () => {
        const scene = buildMhsaSceneModel({
            activationSource: createActivationSource(4),
            layerIndex: 2,
            headIndex: 3,
            sampleStep: 256
        });
        const index = createMhsaDetailSceneIndex(scene);
        const nodes = flattenSceneNodes(scene);
        const sourceNode = nodes.find((node) => node.role === 'projection-source-xln');
        const kInputNode = nodes.find((node) => node.role === 'x-ln-copy' && node.semantic?.branchKey === 'k');
        const kWeightNode = nodes.find((node) => node.role === 'projection-weight' && node.metadata?.kind === 'k');
        const kValueNode = nodes.find((node) => node.role === 'projection-output' && node.metadata?.kind === 'k');
        const kTransposeNode = nodes.find((node) => node.role === 'attention-key-transpose');
        const qQueryNode = nodes.find((node) => node.role === 'attention-query-source');
        const kIngressConnector = nodes.find((node) => node.role === 'connector-xln-k');
        const kConnectorNode = nodes.find((node) => node.role === 'connector-k');

        const weightHover = resolveMhsaDetailHoverState(index, { node: kWeightNode });

        expect(weightHover?.label).toBe('Key Weight Matrix');
        expect(weightHover?.focusState?.activeNodeIds).toContain(sourceNode?.id);
        expect(weightHover?.focusState?.activeNodeIds).toContain(kInputNode?.id);
        expect(weightHover?.focusState?.activeNodeIds).toContain(kWeightNode?.id);
        expect(weightHover?.focusState?.activeNodeIds).toContain(kValueNode?.id);
        expect(weightHover?.focusState?.activeNodeIds).toContain(kTransposeNode?.id);
        expect(weightHover?.focusState?.activeNodeIds).not.toContain(qQueryNode?.id);
        expect(weightHover?.focusState?.activeConnectorIds).toContain(kIngressConnector?.id);
        expect(weightHover?.focusState?.activeConnectorIds).toContain(kConnectorNode?.id);
    });

    it('narrows W_v hover to the local X_ln, V, and weighted-value branch', () => {
        const scene = buildMhsaSceneModel({
            activationSource: createActivationSource(4),
            layerIndex: 2,
            headIndex: 3,
            sampleStep: 256
        });
        const index = createMhsaDetailSceneIndex(scene);
        const nodes = flattenSceneNodes(scene);
        const sourceNode = nodes.find((node) => node.role === 'projection-source-xln');
        const vInputNode = nodes.find((node) => node.role === 'x-ln-copy' && node.semantic?.branchKey === 'v');
        const vWeightNode = nodes.find((node) => node.role === 'projection-weight' && node.metadata?.kind === 'v');
        const vValueNode = nodes.find((node) => node.role === 'attention-value-post');
        const headOutputNode = nodes.find((node) => node.role === 'attention-head-output');
        const vIngressConnector = nodes.find((node) => node.role === 'connector-xln-v');
        const vConnectorNode = nodes.find((node) => node.role === 'connector-v');

        const weightHover = resolveMhsaDetailHoverState(index, { node: vWeightNode });

        expect(weightHover?.label).toBe('Value Weight Matrix');
        expect(weightHover?.focusState?.activeNodeIds).toContain(sourceNode?.id);
        expect(weightHover?.focusState?.activeNodeIds).toContain(vInputNode?.id);
        expect(weightHover?.focusState?.activeNodeIds).toContain(vWeightNode?.id);
        expect(weightHover?.focusState?.activeNodeIds).toContain(vValueNode?.id);
        expect(weightHover?.focusState?.activeNodeIds).not.toContain(headOutputNode?.id);
        expect(weightHover?.focusState?.activeConnectorIds).toContain(vIngressConnector?.id);
        expect(weightHover?.focusState?.activeConnectorIds).toContain(vConnectorNode?.id);
    });

    it('keeps bias-vector hover on the same local branch with head/layer tooltip metadata', () => {
        const scene = buildMhsaSceneModel({
            activationSource: createActivationSource(4),
            layerIndex: 0,
            headIndex: 4,
            sampleStep: 256
        });
        const index = createMhsaDetailSceneIndex(scene);
        const nodes = flattenSceneNodes(scene);
        const sourceNode = nodes.find((node) => node.role === 'projection-source-xln');
        const vInputNode = nodes.find((node) => node.role === 'x-ln-copy' && node.semantic?.branchKey === 'v');
        const vBiasNode = nodes.find((node) => node.role === 'projection-bias' && node.metadata?.kind === 'v');
        const vOutputNode = nodes.find((node) => node.role === 'projection-output' && node.metadata?.kind === 'v');
        const vValueNode = nodes.find((node) => node.role === 'attention-value-post');
        const connectorVNode = nodes.find((node) => node.role === 'connector-v');
        const connectorXlnVNode = nodes.find((node) => node.role === 'connector-xln-v');

        const biasHover = resolveMhsaDetailHoverState(index, {
            node: vBiasNode,
            rowHit: {
                rowIndex: 0,
                rowItem: vBiasNode?.rowItems?.[0] || null
            }
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
        expect(biasHover?.focusState?.activeNodeIds).toContain(sourceNode?.id);
        expect(biasHover?.focusState?.activeNodeIds).toContain(vInputNode?.id);
        expect(biasHover?.focusState?.activeNodeIds).toContain(vBiasNode?.id);
        expect(biasHover?.focusState?.activeNodeIds).toContain(vOutputNode?.id);
        expect(biasHover?.focusState?.activeNodeIds).toContain(vValueNode?.id);
        expect(biasHover?.focusState?.activeConnectorIds).toContain(connectorVNode?.id);
        expect(biasHover?.focusState?.activeConnectorIds).toContain(connectorXlnVNode?.id);
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

    it('routes K^T column hover to the corresponding key-vector tooltip metadata and score-axis focus', () => {
        const scene = buildMhsaSceneModel({
            activationSource: createActivationSource(5),
            layerIndex: 1,
            headIndex: 4,
            sampleStep: 256
        });
        const index = createMhsaDetailSceneIndex(scene);
        const nodes = flattenSceneNodes(scene);
        const transposeNode = nodes.find((node) => node.role === 'attention-key-transpose');
        const postNode = nodes.find((node) => node.role === 'attention-post');
        const postCopyNode = nodes.find((node) => node.role === 'attention-post-copy');
        const transposeColumn = transposeNode?.columnItems?.[3] || null;
        const connectorKNode = nodes.find((node) => node.role === 'connector-k');
        const connectorPreNode = nodes.find((node) => node.role === 'connector-pre');
        const connectorPostNode = nodes.find((node) => node.role === 'connector-post');

        const transposeHover = resolveMhsaDetailHoverState(index, {
            node: transposeNode,
            columnHit: {
                colIndex: 3,
                columnItem: transposeColumn
            }
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
        expect(transposeHover?.focusState?.activeConnectorIds).toContain(connectorKNode?.id);
        expect(transposeHover?.focusState?.activeConnectorIds).toContain(connectorPreNode?.id);
        expect(transposeHover?.focusState?.activeConnectorIds).toContain(connectorPostNode?.id);
        expect(transposeHover?.focusState?.columnSelections).toContainEqual({
            nodeId: transposeNode?.id,
            colIndex: 3
        });
        expect(transposeHover?.focusState?.columnSelections).toContainEqual({
            nodeId: postNode?.id,
            colIndex: 3
        });
        expect(transposeHover?.focusState?.columnSelections).toContainEqual({
            nodeId: postCopyNode?.id,
            colIndex: 3
        });
    });
});
