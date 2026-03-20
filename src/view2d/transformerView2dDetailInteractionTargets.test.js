import { describe, expect, it } from 'vitest';

import { D_HEAD, D_MODEL } from '../ui/selectionPanelConstants.js';
import { createMhsaDetailSceneIndex } from './mhsaDetailInteraction.js';
import { buildMhsaSceneModel } from './model/buildMhsaSceneModel.js';
import { flattenSceneNodes } from './schema/sceneTypes.js';
import {
    resolveTransformerView2dDetailInteractionHoverState,
    resolveTransformerView2dDetailInteractionTargets
} from './transformerView2dDetailInteractionTargets.js';

const TOKEN_LABELS = ['Token A', 'Token B'];

function createVectorValues(seed = 0) {
    return Array.from({ length: D_HEAD }, (_, index) => Number((seed + (index * 0.01)).toFixed(4)));
}

function createBaseRows(tokenLabels = TOKEN_LABELS) {
    return tokenLabels.map((tokenLabel, rowIndex) => ({
        rowIndex,
        tokenIndex: rowIndex,
        tokenLabel,
        rawValues: createVectorValues(rowIndex),
        gradientCss: `rgba(${120 + (rowIndex * 16)}, 220, 255, 0.9)`
    }));
}

function createProjectionOutputRows(label = 'Q', tokenLabels = TOKEN_LABELS) {
    return tokenLabels.map((tokenLabel, rowIndex) => ({
        rowIndex,
        tokenIndex: rowIndex,
        tokenLabel,
        rawValue: Number((rowIndex + 0.25).toFixed(3)),
        rawValues: createVectorValues(rowIndex + 1),
        gradientCss: `rgba(${180 - (rowIndex * 20)}, ${140 + (rowIndex * 24)}, 255, 0.88)`,
        title: `${tokenLabel}: ${label} vector`
    }));
}

function createGridRows(fillCss = 'rgba(255, 255, 255, 0.28)', tokenLabels = TOKEN_LABELS, {
    stageKey = 'pre'
} = {}) {
    return tokenLabels.map((tokenLabel, rowIndex) => ({
        rowIndex,
        tokenIndex: rowIndex,
        tokenLabel,
        cells: tokenLabels.map((colLabel, colIndex) => {
            const isMasked = colIndex > rowIndex;
            const preScore = Number(((rowIndex + 1) * (colIndex + 1) * 0.125).toFixed(3));
            const postScore = Number((((rowIndex + 1) + (colIndex + 1)) * 0.05).toFixed(3));
            const maskValue = isMasked ? Number.NEGATIVE_INFINITY : 0;
            return {
                rowIndex,
                colIndex,
                rowTokenLabel: tokenLabel,
                colTokenLabel: colLabel,
                queryTokenIndex: rowIndex,
                keyTokenIndex: colIndex,
                queryTokenLabel: tokenLabel,
                keyTokenLabel: colLabel,
                preScore,
                postScore: isMasked ? 0 : postScore,
                maskValue,
                rawValue: stageKey === 'mask'
                    ? maskValue
                    : (stageKey === 'post' ? (isMasked ? null : postScore) : preScore),
                fillCss,
                isMasked,
                isEmpty: stageKey === 'post' ? isMasked : false,
                title: `${tokenLabel} -> ${colLabel}`
            };
        }),
        hasAnyValue: true
    }));
}

function createPreviewData(tokenLabels = TOKEN_LABELS) {
    const rows = createBaseRows(tokenLabels);
    const queryOutputRows = createProjectionOutputRows('Q', tokenLabels);
    const keyOutputRows = createProjectionOutputRows('K', tokenLabels);
    const valueOutputRows = createProjectionOutputRows('V', tokenLabels);
    const attentionGridRows = createGridRows('rgba(255, 255, 255, 0.28)', tokenLabels, {
        stageKey: 'pre'
    });
    const maskGridRows = createGridRows('rgba(0, 0, 0, 0.94)', tokenLabels, {
        stageKey: 'mask'
    });
    const postGridRows = createGridRows('rgba(160, 220, 255, 0.34)', tokenLabels, {
        stageKey: 'post'
    });

    const createProjection = (kind, outputLabelTex, outputRows) => ({
        kind,
        weightLabelTex: `W_${kind.toLowerCase()}`,
        biasLabelTex: `b_${kind.toLowerCase()}`,
        outputLabelTex,
        weightRowCount: D_MODEL,
        weightColumnCount: D_HEAD,
        biasValue: 0.15,
        biasVectorGradientCss: 'rgba(255, 255, 255, 0.2)',
        outputRowCount: outputRows.length,
        outputColumnCount: D_HEAD,
        outputRows
    });

    return {
        rowCount: rows.length,
        columnCount: D_MODEL,
        bandCount: 12,
        sampleStep: 64,
        rows,
        projections: [
            createProjection('Q', 'Q', queryOutputRows),
            createProjection('K', 'K', keyOutputRows),
            createProjection('V', 'V', valueOutputRows)
        ],
        attentionScoreStage: {
            queryLabelTex: 'Q',
            queryRowCount: queryOutputRows.length,
            queryColumnCount: D_HEAD,
            queryRows: queryOutputRows,
            transposeLabelTex: 'K^{\\mathsf{T}}',
            transposeRowCount: D_HEAD,
            transposeColumnCount: keyOutputRows.length,
            transposeColumns: keyOutputRows.map((rowData) => ({
                colIndex: rowData.rowIndex,
                tokenIndex: rowData.tokenIndex,
                rawValue: rowData.rawValue,
                rawValues: rowData.rawValues,
                fillCss: rowData.gradientCss,
                tokenLabel: rowData.tokenLabel
            })),
            scaleLabelTex: '\\sqrt{d_{\\mathrm{head}}}',
            outputLabelTex: 'A_{\\mathrm{pre}}',
            outputRowCount: attentionGridRows.length,
            outputColumnCount: attentionGridRows.length,
            outputRows: attentionGridRows,
            maskLabelTex: 'M_{\\mathrm{causal}}',
            maskRows: maskGridRows,
            softmaxLabelTex: '\\mathrm{softmax}',
            postLabelTex: 'A_{\\mathrm{post}}',
            postRowCount: postGridRows.length,
            postColumnCount: postGridRows.length,
            postRows: postGridRows,
            valueLabelTex: 'V',
            valueRowCount: valueOutputRows.length,
            valueColumnCount: D_HEAD,
            valueRows: valueOutputRows,
            headOutputLabelTex: 'H_i',
            headOutputRowCount: valueOutputRows.length,
            headOutputColumnCount: D_HEAD,
            headOutputRows: valueOutputRows
        }
    };
}

function buildMhsaDetailFixtures({
    tokenLabels = TOKEN_LABELS,
    kvCacheState = null
} = {}) {
    const scene = buildMhsaSceneModel({
        previewData: createPreviewData(tokenLabels),
        layerIndex: 2,
        headIndex: 1,
        ...(kvCacheState ? { kvCacheState } : {})
    });
    const index = createMhsaDetailSceneIndex(scene);
    const nodes = flattenSceneNodes(scene);
    const findProjectionNode = (role, kind) => nodes.find((node) => (
        node.role === role
        && String(node.metadata?.kind || '').toLowerCase() === String(kind || '').toLowerCase()
    )) || null;
    const findProjectionInputNode = (kind) => nodes.find((node) => (
        node.role === 'x-ln-copy'
        && String(node.semantic?.branchKey || '').toLowerCase() === String(kind || '').toLowerCase()
    )) || null;
    return {
        index,
        projectionSourceNode: nodes.find((node) => node.role === 'projection-source-xln') || null,
        valueInputNode: findProjectionInputNode('v'),
        queryProjectionOutputNode: findProjectionNode('projection-output', 'q'),
        valueProjectionOutputNode: findProjectionNode('projection-output', 'v'),
        valueProjectionOutputCopyNode: findProjectionNode('projection-output-copy', 'v'),
        valueCacheNode: findProjectionNode('projection-cache', 'v'),
        valueCacheSourceNode: findProjectionNode('projection-cache-source', 'v'),
        valueCacheConcatResultNode: findProjectionNode('projection-cache-concat-result', 'v'),
        valueCacheNextNode: findProjectionNode('projection-cache-next', 'v'),
        valuePostNode: nodes.find((node) => node.role === 'attention-value-post') || null,
        preScoreNode: nodes.find((node) => node.role === 'attention-pre-score') || null
    };
}

describe('transformerView2dDetailInteractionTargets', () => {
    it('maps query-vector selections to row-level detail interaction targets', () => {
        const targets = resolveTransformerView2dDetailInteractionTargets({
            label: 'Query Vector',
            info: {
                layerIndex: 2,
                headIndex: 1,
                tokenIndex: 1,
                activationData: {
                    stage: 'qkv.q',
                    layerIndex: 2,
                    headIndex: 1,
                    tokenIndex: 1
                }
            }
        }, [
            {
                componentKind: 'mhsa',
                layerIndex: 2,
                headIndex: 1,
                stage: 'projection-q',
                role: 'projection-output'
            },
            {
                componentKind: 'mhsa',
                layerIndex: 2,
                headIndex: 1,
                stage: 'attention',
                role: 'attention-query-source'
            }
        ]);

        expect(targets).toEqual([
            {
                kind: 'row',
                semanticTarget: {
                    componentKind: 'mhsa',
                    layerIndex: 2,
                    headIndex: 1,
                    stage: 'projection-q',
                    role: 'projection-output'
                },
                tokenIndex: 1,
                queryTokenIndex: 1
            },
            {
                kind: 'row',
                semanticTarget: {
                    componentKind: 'mhsa',
                    layerIndex: 2,
                    headIndex: 1,
                    stage: 'attention',
                    role: 'attention-query-source'
                },
                tokenIndex: 1,
                queryTokenIndex: 1
            }
        ]);
    });

    it('maps attention-score selections to cell-level detail interaction targets', () => {
        const targets = resolveTransformerView2dDetailInteractionTargets({
            label: 'Pre-Softmax Attention Score',
            info: {
                layerIndex: 2,
                headIndex: 1,
                tokenIndex: 0,
                keyTokenIndex: 1,
                activationData: {
                    stage: 'attention.pre',
                    layerIndex: 2,
                    headIndex: 1,
                    tokenIndex: 0,
                    keyTokenIndex: 1
                }
            }
        }, [{
            componentKind: 'mhsa',
            layerIndex: 2,
            headIndex: 1,
            stage: 'attention',
            role: 'attention-pre-score'
        }]);

        expect(targets).toEqual([{
            kind: 'cell',
            semanticTarget: {
                componentKind: 'mhsa',
                layerIndex: 2,
                headIndex: 1,
                stage: 'attention',
                role: 'attention-pre-score'
            },
            tokenIndex: 0,
            queryTokenIndex: 0,
            keyTokenIndex: 1
        }]);
    });

    it('maps MLP down-projection selections to row-level detail interaction targets', () => {
        const targets = resolveTransformerView2dDetailInteractionTargets({
            label: 'MLP Down Projection',
            info: {
                layerIndex: 2,
                tokenIndex: 1,
                activationData: {
                    stage: 'mlp.down',
                    layerIndex: 2,
                    tokenIndex: 1
                }
            }
        }, [{
            componentKind: 'mlp',
            layerIndex: 2,
            stage: 'mlp-down',
            role: 'mlp-down-output'
        }]);

        expect(targets).toEqual([{
            kind: 'row',
            semanticTarget: {
                componentKind: 'mlp',
                layerIndex: 2,
                stage: 'mlp-down',
                role: 'mlp-down-output'
            },
            tokenIndex: 1,
            queryTokenIndex: 1
        }]);
    });

    it('maps layer-norm product selections to row-level detail interaction targets', () => {
        const targets = resolveTransformerView2dDetailInteractionTargets({
            label: 'LayerNorm 1 Product Vector',
            info: {
                layerIndex: 2,
                tokenIndex: 0,
                activationData: {
                    stage: 'ln1.product',
                    layerIndex: 2,
                    tokenIndex: 0,
                    layerNormKind: 'ln1'
                }
            }
        }, [{
            componentKind: 'layer-norm',
            layerIndex: 2,
            stage: 'ln1.scale',
            role: 'layer-norm-scaled'
        }]);

        expect(targets).toEqual([{
            kind: 'row',
            semanticTarget: {
                componentKind: 'layer-norm',
                layerIndex: 2,
                stage: 'ln1.scale',
                role: 'layer-norm-scaled'
            },
            tokenIndex: 0,
            queryTokenIndex: 0
        }]);
    });

    it('resolves a query-vector target into the same row-focused hover state as a detail-scene click', () => {
        const { index, queryProjectionOutputNode } = buildMhsaDetailFixtures();
        const hoverState = resolveTransformerView2dDetailInteractionHoverState(index, [{
            kind: 'row',
            semanticTarget: {
                componentKind: 'mhsa',
                layerIndex: 2,
                headIndex: 1,
                stage: 'projection-q',
                role: 'projection-output'
            },
            tokenIndex: 1
        }]);

        expect(hoverState?.label).toBe('Query Vector');
        expect(hoverState?.focusState?.rowSelections).toContainEqual({
            nodeId: queryProjectionOutputNode?.id,
            rowIndex: 1
        });
    });

    it('keeps decode Value Vector selection-open focus off the cache and concat-copy mirrors', () => {
        const {
            index,
            projectionSourceNode,
            valueInputNode,
            valueProjectionOutputNode,
            valueProjectionOutputCopyNode,
            valueCacheNode,
            valueCacheSourceNode,
            valueCacheConcatResultNode,
            valueCacheNextNode,
            valuePostNode
        } = buildMhsaDetailFixtures({
            tokenLabels: ['Token A', 'Token B', 'Token C', 'Token D'],
            kvCacheState: {
                kvCacheModeEnabled: true,
                kvCachePrefillActive: false,
                kvCacheDecodeActive: true,
                kvCachePassIndex: 1
            }
        });

        const hoverState = resolveTransformerView2dDetailInteractionHoverState(index, [
            {
                kind: 'row',
                semanticTarget: {
                    componentKind: 'mhsa',
                    layerIndex: 2,
                    headIndex: 1,
                    stage: 'projection-v',
                    role: 'projection-output'
                },
                tokenIndex: 3
            },
            {
                kind: 'row',
                semanticTarget: {
                    componentKind: 'mhsa',
                    layerIndex: 2,
                    headIndex: 1,
                    stage: 'head-output',
                    role: 'attention-value-post'
                },
                tokenIndex: 3
            }
        ], {
            interactionKind: 'selection-open'
        });

        expect(hoverState?.label).toBe('Value Vector');
        expect(hoverState?.focusState?.rowSelections).toContainEqual({
            nodeId: projectionSourceNode?.id,
            rowIndex: 0
        });
        expect(hoverState?.focusState?.rowSelections).toContainEqual({
            nodeId: valueInputNode?.id,
            rowIndex: 0
        });
        expect(hoverState?.focusState?.rowSelections).toContainEqual({
            nodeId: valueProjectionOutputNode?.id,
            rowIndex: 0
        });
        expect(hoverState?.focusState?.rowSelections).toContainEqual({
            nodeId: valueProjectionOutputCopyNode?.id,
            rowIndex: 0
        });
        expect(hoverState?.focusState?.rowSelections).toContainEqual({
            nodeId: valueCacheConcatResultNode?.id,
            rowIndex: 3
        });
        expect(hoverState?.focusState?.rowSelections).toContainEqual({
            nodeId: valueCacheNextNode?.id,
            rowIndex: 3
        });
        expect(hoverState?.focusState?.rowSelections).toContainEqual({
            nodeId: valuePostNode?.id,
            rowIndex: 3
        });
        expect(hoverState?.focusState?.activeNodeIds).toContain(valueProjectionOutputNode?.id);
        expect(hoverState?.focusState?.activeNodeIds).toContain(valueProjectionOutputCopyNode?.id);
        expect(hoverState?.focusState?.activeNodeIds).toContain(valueCacheConcatResultNode?.id);
        expect(hoverState?.focusState?.activeNodeIds).toContain(valueCacheNextNode?.id);
        expect(hoverState?.focusState?.activeNodeIds).toContain(valuePostNode?.id);
        expect(hoverState?.focusState?.activeNodeIds).not.toContain(valueCacheNode?.id);
        expect(hoverState?.focusState?.activeNodeIds).not.toContain(valueCacheSourceNode?.id);
        expect(
            hoverState?.focusState?.rowSelections?.some((selection) => selection.nodeId === valueCacheNode?.id)
        ).toBe(false);
        expect(
            hoverState?.focusState?.rowSelections?.some((selection) => selection.nodeId === valueCacheSourceNode?.id)
        ).toBe(false);
        expect(hoverState?.focusState?.activeConnectorIds).toContainEqual(expect.any(String));
    });

    it('resolves a score-cell target into the same cell-focused hover state as a detail-scene click', () => {
        const { index, preScoreNode } = buildMhsaDetailFixtures();
        const hoverState = resolveTransformerView2dDetailInteractionHoverState(index, [{
            kind: 'cell',
            semanticTarget: {
                componentKind: 'mhsa',
                layerIndex: 2,
                headIndex: 1,
                stage: 'attention',
                role: 'attention-pre-score'
            },
            queryTokenIndex: 0,
            keyTokenIndex: 1
        }]);

        expect(hoverState?.label).toBe('Pre-Softmax Attention Score');
        expect(hoverState?.focusState?.cellSelections).toContainEqual({
            nodeId: preScoreNode?.id,
            rowIndex: 0,
            colIndex: 1
        });
    });
});
