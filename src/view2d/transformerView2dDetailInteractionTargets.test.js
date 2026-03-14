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

function createPreviewData() {
    const rows = createBaseRows();
    const queryOutputRows = createProjectionOutputRows('Q');
    const keyOutputRows = createProjectionOutputRows('K');
    const valueOutputRows = createProjectionOutputRows('V');
    const attentionGridRows = createGridRows('rgba(255, 255, 255, 0.28)', TOKEN_LABELS, {
        stageKey: 'pre'
    });
    const maskGridRows = createGridRows('rgba(0, 0, 0, 0.94)', TOKEN_LABELS, {
        stageKey: 'mask'
    });
    const postGridRows = createGridRows('rgba(160, 220, 255, 0.34)', TOKEN_LABELS, {
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

function buildMhsaDetailFixtures() {
    const scene = buildMhsaSceneModel({
        previewData: createPreviewData(),
        layerIndex: 2,
        headIndex: 1
    });
    const index = createMhsaDetailSceneIndex(scene);
    const nodes = flattenSceneNodes(scene);
    return {
        index,
        queryProjectionOutputNode: nodes.find((node) => (
            node.role === 'projection-output'
            && String(node.metadata?.kind || '').toLowerCase() === 'q'
        )) || null,
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
