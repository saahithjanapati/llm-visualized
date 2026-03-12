import { buildMhsaTokenMatrixPreviewData } from '../../ui/selectionPanelMhsaTokenMatrixUtils.js';
import { resolveMhsaTokenMatrixLayoutMetrics } from '../../ui/selectionPanelMhsaLayoutUtils.js';
import {
    buildSceneNodeId,
    createAnchorRef,
    createConnectorNode,
    createGroupNode,
    createMatrixNode,
    createOperatorNode,
    createSceneModel,
    createTextNode,
    VIEW2D_ANCHOR_SIDES,
    VIEW2D_CONNECTOR_ROUTES,
    VIEW2D_LAYOUT_DIRECTIONS,
    VIEW2D_MATRIX_PRESENTATIONS,
    VIEW2D_MATRIX_SHAPES,
    VIEW2D_TEXT_PRESENTATIONS
} from '../schema/sceneTypes.js';
import {
    resolveView2dVisualTokens,
    VIEW2D_STYLE_KEYS
} from '../theme/visualTokens.js';
import { createView2dVectorStripMetadata } from '../shared/vectorStrip.js';

function normalizeIndex(value) {
    return Number.isFinite(value) ? Math.floor(value) : null;
}

function buildSemantic(baseSemantic, extra = {}) {
    return {
        ...baseSemantic,
        ...extra
    };
}

function buildLabel(labelTex = '', fallbackText = '') {
    return {
        tex: typeof labelTex === 'string' ? labelTex : '',
        text: typeof fallbackText === 'string' && fallbackText.length
            ? fallbackText
            : (typeof labelTex === 'string' ? labelTex : '')
    };
}

function buildGradientRowItems(rows = [], baseSemantic = {}, role = 'row') {
    return rows.map((rowData) => {
        const semantic = buildSemantic(baseSemantic, {
            role,
            rowIndex: rowData.rowIndex,
            tokenIndex: rowData.tokenIndex
        });
        return {
            id: buildSceneNodeId(semantic),
            index: rowData.rowIndex,
            label: rowData.tokenLabel || `Token ${rowData.rowIndex + 1}`,
            semantic,
            rawValue: Number.isFinite(rowData.rawValue) ? rowData.rawValue : null,
            rawValues: Array.isArray(rowData.rawValues) ? [...rowData.rawValues] : null,
            gradientCss: rowData.gradientCss || 'none',
            title: typeof rowData.title === 'string' ? rowData.title : null
        };
    });
}

function buildAttentionGridRowItems(rows = [], baseSemantic = {}) {
    return rows.map((rowData) => {
        const rowSemantic = buildSemantic(baseSemantic, {
            role: 'attention-row',
            rowIndex: rowData.rowIndex
        });
        return {
            id: buildSceneNodeId(rowSemantic),
            index: rowData.rowIndex,
            label: rowData.tokenLabel || `Token ${rowData.rowIndex + 1}`,
            semantic: rowSemantic,
            cells: Array.isArray(rowData.cells)
                ? rowData.cells.map((cellData) => {
                    const cellSemantic = buildSemantic(baseSemantic, {
                        role: 'attention-cell',
                        rowIndex: cellData.rowIndex,
                        colIndex: cellData.colIndex
                    });
                    return {
                        id: buildSceneNodeId(cellSemantic),
                        rowIndex: cellData.rowIndex,
                        colIndex: cellData.colIndex,
                        semantic: cellSemantic,
                        rowLabel: cellData.rowTokenLabel || rowData.tokenLabel || '',
                        colLabel: cellData.colTokenLabel || '',
                        rawValue: Number.isFinite(cellData.rawValue) ? cellData.rawValue : null,
                        fillCss: cellData.fillCss || 'transparent',
                        isMasked: !!cellData.isMasked,
                        isEmpty: !!cellData.isEmpty,
                        title: typeof cellData.title === 'string' ? cellData.title : null
                    };
                })
                : []
        };
    });
}

function buildTransposeColumnItems(columns = [], baseSemantic = {}) {
    return columns.map((columnData) => {
        const semantic = buildSemantic(baseSemantic, {
            role: 'transpose-column',
            colIndex: columnData.colIndex
        });
        return {
            id: buildSceneNodeId(semantic),
            index: columnData.colIndex,
            label: columnData.tokenLabel || `Token ${columnData.colIndex + 1}`,
            semantic,
            rawValue: Number.isFinite(columnData.rawValue) ? columnData.rawValue : null,
            fillCss: columnData.fillCss || 'transparent',
            title: typeof columnData.tokenLabel === 'string' ? columnData.tokenLabel : null
        };
    });
}

function buildProjectionStageNode({
    baseSemantic,
    previewData,
    projectionData,
    stageIndex
}) {
    const projectionKind = String(projectionData?.kind || '').toLowerCase();
    const projectionSemantic = buildSemantic(baseSemantic, {
        stage: `projection-${projectionKind}`,
        stageIndex
    });
    const styleKey = projectionKind === 'q'
        ? VIEW2D_STYLE_KEYS.MHSA_Q
        : (projectionKind === 'k' ? VIEW2D_STYLE_KEYS.MHSA_K : VIEW2D_STYLE_KEYS.MHSA_V);

    const xInputNode = createMatrixNode({
        role: 'projection-input',
        semantic: buildSemantic(projectionSemantic, { role: 'projection-input' }),
        label: buildLabel('X_{\\ln}', 'X_ln'),
        dimensions: {
            rows: previewData.rowCount,
            cols: previewData.columnCount
        },
        presentation: VIEW2D_MATRIX_PRESENTATIONS.BANDED_ROWS,
        shape: VIEW2D_MATRIX_SHAPES.MATRIX,
        rowItems: buildGradientRowItems(previewData.rows, projectionSemantic, 'projection-input-row'),
        visual: {
            styleKey: VIEW2D_STYLE_KEYS.MATRIX_INPUT
        }
    });

    const weightNode = createMatrixNode({
        role: 'projection-weight',
        semantic: buildSemantic(projectionSemantic, { role: 'projection-weight' }),
        label: buildLabel(projectionData.weightLabelTex, projectionData.weightLabelTex),
        dimensions: {
            rows: projectionData.weightRowCount,
            cols: projectionData.weightColumnCount
        },
        presentation: VIEW2D_MATRIX_PRESENTATIONS.CARD,
        shape: VIEW2D_MATRIX_SHAPES.MATRIX,
        visual: {
            styleKey: VIEW2D_STYLE_KEYS.MATRIX_WEIGHT,
            background: projectionData.weightGradientCss || 'none'
        },
        metadata: {
            kind: projectionKind
        }
    });

    const biasNode = createMatrixNode({
        role: 'projection-bias',
        semantic: buildSemantic(projectionSemantic, { role: 'projection-bias' }),
        label: buildLabel(projectionData.biasLabelTex, projectionData.biasLabelTex),
        dimensions: {
            rows: 1,
            cols: projectionData.outputColumnCount
        },
        presentation: VIEW2D_MATRIX_PRESENTATIONS.ACCENT_BAR,
        shape: VIEW2D_MATRIX_SHAPES.VECTOR,
        visual: {
            styleKey: VIEW2D_STYLE_KEYS.MATRIX_BIAS,
            background: projectionData.biasGradientCss || 'none'
        },
        metadata: {
            kind: projectionKind
        }
    });

    const outputNode = createMatrixNode({
        role: 'projection-output',
        semantic: buildSemantic(projectionSemantic, { role: 'projection-output' }),
        label: buildLabel(projectionData.outputLabelTex, projectionData.outputLabelTex),
        dimensions: {
            rows: projectionData.outputRowCount,
            cols: projectionData.outputColumnCount
        },
        presentation: VIEW2D_MATRIX_PRESENTATIONS.COMPACT_ROWS,
        shape: VIEW2D_MATRIX_SHAPES.MATRIX,
        rowItems: buildGradientRowItems(projectionData.outputRows, projectionSemantic, 'projection-output-row'),
        visual: {
            styleKey
        },
        metadata: {
            kind: projectionKind,
            stageIndex,
            ...createView2dVectorStripMetadata()
        }
    });

    const equationNode = createGroupNode({
        role: 'projection-equation',
        semantic: buildSemantic(projectionSemantic, { role: 'projection-equation' }),
        direction: VIEW2D_LAYOUT_DIRECTIONS.HORIZONTAL,
        gapKey: 'projection',
        children: [
            weightNode,
            createOperatorNode({
                role: 'projection-plus',
                semantic: buildSemantic(projectionSemantic, { role: 'projection-plus', operatorKey: 'plus' }),
                text: '+',
                visual: { styleKey: VIEW2D_STYLE_KEYS.OPERATOR }
            }),
            biasNode,
            createOperatorNode({
                role: 'projection-equals',
                semantic: buildSemantic(projectionSemantic, { role: 'projection-equals', operatorKey: 'equals' }),
                text: '=',
                visual: { styleKey: VIEW2D_STYLE_KEYS.OPERATOR }
            }),
            outputNode
        ]
    });

    return createGroupNode({
        role: 'projection-stage',
        semantic: buildSemantic(projectionSemantic, { role: 'projection-stage' }),
        direction: VIEW2D_LAYOUT_DIRECTIONS.HORIZONTAL,
        gapKey: 'projection',
        children: [
            xInputNode,
            createOperatorNode({
                role: 'projection-multiply',
                semantic: buildSemantic(projectionSemantic, { role: 'projection-multiply', operatorKey: 'multiply' }),
                text: 'x',
                visual: { styleKey: VIEW2D_STYLE_KEYS.OPERATOR }
            }),
            equationNode
        ],
        metadata: {
            kind: projectionKind,
            stageIndex
        }
    });
}

function buildAttentionStageNode({
    baseSemantic,
    scoreStage,
    queryStageIndex,
    valueStageIndex
}) {
    const attentionSemantic = buildSemantic(baseSemantic, {
        stage: 'attention'
    });

    const querySourceNode = createMatrixNode({
        role: 'attention-query-source',
        semantic: buildSemantic(attentionSemantic, { role: 'attention-query-source' }),
        label: buildLabel(scoreStage.queryLabelTex, scoreStage.queryLabelTex),
        dimensions: {
            rows: scoreStage.queryRowCount,
            cols: scoreStage.queryColumnCount
        },
        presentation: VIEW2D_MATRIX_PRESENTATIONS.COMPACT_ROWS,
        shape: VIEW2D_MATRIX_SHAPES.MATRIX,
        rowItems: buildGradientRowItems(scoreStage.queryRows, attentionSemantic, 'attention-query-row'),
        visual: {
            styleKey: VIEW2D_STYLE_KEYS.MHSA_Q
        },
        metadata: {
            stageIndex: queryStageIndex,
            ...createView2dVectorStripMetadata()
        }
    });

    const transposeNode = createMatrixNode({
        role: 'attention-key-transpose',
        semantic: buildSemantic(attentionSemantic, { role: 'attention-key-transpose' }),
        label: buildLabel(scoreStage.transposeLabelTex, scoreStage.transposeLabelTex),
        dimensions: {
            rows: scoreStage.transposeRowCount,
            cols: scoreStage.transposeColumnCount
        },
        presentation: VIEW2D_MATRIX_PRESENTATIONS.COLUMN_STRIP,
        shape: VIEW2D_MATRIX_SHAPES.MATRIX,
        columnItems: buildTransposeColumnItems(scoreStage.transposeColumns, attentionSemantic),
        visual: {
            styleKey: VIEW2D_STYLE_KEYS.MHSA_K
        }
    });

    const preScoreNode = createMatrixNode({
        role: 'attention-pre-score',
        semantic: buildSemantic(attentionSemantic, { role: 'attention-pre-score' }),
        label: buildLabel(scoreStage.outputLabelTex, scoreStage.outputLabelTex),
        dimensions: {
            rows: scoreStage.outputRowCount,
            cols: scoreStage.outputColumnCount
        },
        presentation: VIEW2D_MATRIX_PRESENTATIONS.GRID,
        rowItems: buildAttentionGridRowItems(scoreStage.outputRows, buildSemantic(attentionSemantic, { stage: 'attention-pre-score' })),
        visual: {
            styleKey: VIEW2D_STYLE_KEYS.MHSA_SCORE
        }
    });

    const maskedInputNode = createMatrixNode({
        role: 'attention-masked-input',
        semantic: buildSemantic(attentionSemantic, { role: 'attention-masked-input' }),
        label: buildLabel(scoreStage.outputLabelTex, scoreStage.outputLabelTex),
        dimensions: {
            rows: scoreStage.outputRowCount,
            cols: scoreStage.outputColumnCount
        },
        presentation: VIEW2D_MATRIX_PRESENTATIONS.GRID,
        rowItems: buildAttentionGridRowItems(scoreStage.outputRows, buildSemantic(attentionSemantic, { stage: 'attention-masked-input' })),
        visual: {
            styleKey: VIEW2D_STYLE_KEYS.MHSA_SCORE
        }
    });

    const maskNode = createMatrixNode({
        role: 'attention-mask',
        semantic: buildSemantic(attentionSemantic, { role: 'attention-mask' }),
        label: buildLabel(scoreStage.maskLabelTex, scoreStage.maskLabelTex),
        dimensions: {
            rows: scoreStage.outputRowCount,
            cols: scoreStage.outputColumnCount
        },
        presentation: VIEW2D_MATRIX_PRESENTATIONS.GRID,
        rowItems: buildAttentionGridRowItems(scoreStage.maskRows, buildSemantic(attentionSemantic, { stage: 'attention-mask' })),
        visual: {
            styleKey: VIEW2D_STYLE_KEYS.MHSA_MASK
        }
    });

    const postNode = createMatrixNode({
        role: 'attention-post',
        semantic: buildSemantic(attentionSemantic, { role: 'attention-post' }),
        label: buildLabel(scoreStage.postLabelTex, scoreStage.postLabelTex),
        dimensions: {
            rows: scoreStage.postRowCount,
            cols: scoreStage.postColumnCount
        },
        presentation: VIEW2D_MATRIX_PRESENTATIONS.GRID,
        rowItems: buildAttentionGridRowItems(scoreStage.postRows, buildSemantic(attentionSemantic, { stage: 'attention-post' })),
        visual: {
            styleKey: VIEW2D_STYLE_KEYS.MHSA_POST
        }
    });

    const softmaxPrefixNode = createGroupNode({
        role: 'attention-softmax-prefix',
        semantic: buildSemantic(attentionSemantic, { role: 'attention-softmax-prefix' }),
        direction: VIEW2D_LAYOUT_DIRECTIONS.HORIZONTAL,
        gapKey: 'inline',
        children: [
            createTextNode({
                role: 'attention-softmax-label',
                semantic: buildSemantic(attentionSemantic, { role: 'attention-softmax-label', focusKey: 'softmax' }),
                tex: scoreStage.softmaxLabelTex,
                text: 'softmax',
                presentation: VIEW2D_TEXT_PRESENTATIONS.LABEL,
                visual: { styleKey: VIEW2D_STYLE_KEYS.LABEL }
            }),
            createOperatorNode({
                role: 'attention-softmax-open',
                semantic: buildSemantic(attentionSemantic, { role: 'attention-softmax-open', operatorKey: 'open' }),
                text: '(',
                visual: { styleKey: VIEW2D_STYLE_KEYS.OPERATOR }
            })
        ]
    });

    const headOutputStageNode = Array.isArray(scoreStage.valueRows) && Array.isArray(scoreStage.headOutputRows)
        ? createGroupNode({
            role: 'attention-head-output-stage',
            semantic: buildSemantic(attentionSemantic, { role: 'attention-head-output-stage', stage: 'head-output' }),
            direction: VIEW2D_LAYOUT_DIRECTIONS.HORIZONTAL,
            gapKey: 'head-output',
            children: [
                createMatrixNode({
                    role: 'attention-post-copy',
                    semantic: buildSemantic(attentionSemantic, { role: 'attention-post-copy', stage: 'head-output' }),
                    label: buildLabel(scoreStage.postLabelTex, scoreStage.postLabelTex),
                    dimensions: {
                        rows: scoreStage.postRowCount,
                        cols: scoreStage.postColumnCount
                    },
                    presentation: VIEW2D_MATRIX_PRESENTATIONS.GRID,
                    rowItems: buildAttentionGridRowItems(
                        scoreStage.postRows,
                        buildSemantic(attentionSemantic, { stage: 'attention-post-copy' })
                    ),
                    visual: {
                        styleKey: VIEW2D_STYLE_KEYS.MHSA_POST
                    }
                }),
                createOperatorNode({
                    role: 'attention-head-output-multiply',
                    semantic: buildSemantic(attentionSemantic, { role: 'attention-head-output-multiply', operatorKey: 'multiply' }),
                    text: 'x',
                    visual: { styleKey: VIEW2D_STYLE_KEYS.OPERATOR }
                }),
                createMatrixNode({
                    role: 'attention-value-post',
                    semantic: buildSemantic(attentionSemantic, { role: 'attention-value-post', stage: 'head-output' }),
                    label: buildLabel(scoreStage.valueLabelTex, scoreStage.valueLabelTex),
                    dimensions: {
                        rows: scoreStage.valueRowCount,
                        cols: scoreStage.valueColumnCount
                    },
                    presentation: VIEW2D_MATRIX_PRESENTATIONS.COMPACT_ROWS,
                    rowItems: buildGradientRowItems(scoreStage.valueRows, buildSemantic(attentionSemantic, {
                        stage: 'attention-value-post'
                    }), 'attention-value-post-row'),
                    visual: {
                        styleKey: VIEW2D_STYLE_KEYS.MHSA_V
                    },
                    metadata: {
                        stageIndex: valueStageIndex,
                        ...createView2dVectorStripMetadata()
                    }
                }),
                createOperatorNode({
                    role: 'attention-head-output-equals',
                    semantic: buildSemantic(attentionSemantic, { role: 'attention-head-output-equals', operatorKey: 'equals' }),
                    text: '=',
                    visual: { styleKey: VIEW2D_STYLE_KEYS.OPERATOR }
                }),
                createMatrixNode({
                    role: 'attention-head-output',
                    semantic: buildSemantic(attentionSemantic, { role: 'attention-head-output', stage: 'head-output' }),
                    label: buildLabel(scoreStage.headOutputLabelTex, scoreStage.headOutputLabelTex),
                    dimensions: {
                        rows: scoreStage.headOutputRowCount,
                        cols: scoreStage.headOutputColumnCount
                    },
                    presentation: VIEW2D_MATRIX_PRESENTATIONS.COMPACT_ROWS,
                    rowItems: buildGradientRowItems(
                        scoreStage.headOutputRows,
                        buildSemantic(attentionSemantic, { stage: 'attention-head-output' }),
                        'attention-head-output-row'
                    ),
                    visual: {
                        styleKey: VIEW2D_STYLE_KEYS.MHSA_HEAD_OUTPUT
                    },
                    metadata: createView2dVectorStripMetadata()
                })
            ]
        })
        : null;

    const softmaxFlowChildren = [
        softmaxPrefixNode,
        maskedInputNode,
        createOperatorNode({
            role: 'attention-softmax-plus',
            semantic: buildSemantic(attentionSemantic, { role: 'attention-softmax-plus', operatorKey: 'plus' }),
            text: '+',
            visual: { styleKey: VIEW2D_STYLE_KEYS.OPERATOR }
        }),
        maskNode,
        createOperatorNode({
            role: 'attention-softmax-close',
            semantic: buildSemantic(attentionSemantic, { role: 'attention-softmax-close', operatorKey: 'close' }),
            text: ')',
            visual: { styleKey: VIEW2D_STYLE_KEYS.OPERATOR }
        }),
        createOperatorNode({
            role: 'attention-softmax-equals',
            semantic: buildSemantic(attentionSemantic, { role: 'attention-softmax-equals', operatorKey: 'equals' }),
            text: '=',
            visual: { styleKey: VIEW2D_STYLE_KEYS.OPERATOR }
        }),
        postNode
    ];
    if (headOutputStageNode) {
        softmaxFlowChildren.push(headOutputStageNode);
    }

    const softmaxStageNode = createGroupNode({
        role: 'attention-softmax-stage',
        semantic: buildSemantic(attentionSemantic, { role: 'attention-softmax-stage' }),
        direction: VIEW2D_LAYOUT_DIRECTIONS.VERTICAL,
        gapKey: 'softmax',
        children: [
            preScoreNode,
            createGroupNode({
                role: 'attention-softmax-flow',
                semantic: buildSemantic(attentionSemantic, { role: 'attention-softmax-flow' }),
                direction: VIEW2D_LAYOUT_DIRECTIONS.HORIZONTAL,
                gapKey: 'softmax',
                children: softmaxFlowChildren
            })
        ]
    });

    return createGroupNode({
        role: 'attention-stage',
        semantic: buildSemantic(attentionSemantic, { role: 'attention-stage' }),
        direction: VIEW2D_LAYOUT_DIRECTIONS.HORIZONTAL,
        gapKey: 'attention',
        children: [
            createOperatorNode({
                role: 'attention-open',
                semantic: buildSemantic(attentionSemantic, { role: 'attention-open', operatorKey: 'open' }),
                text: '(',
                visual: { styleKey: VIEW2D_STYLE_KEYS.OPERATOR }
            }),
            querySourceNode,
            createOperatorNode({
                role: 'attention-multiply',
                semantic: buildSemantic(attentionSemantic, { role: 'attention-multiply', operatorKey: 'multiply' }),
                text: 'x',
                visual: { styleKey: VIEW2D_STYLE_KEYS.OPERATOR }
            }),
            transposeNode,
            createOperatorNode({
                role: 'attention-close',
                semantic: buildSemantic(attentionSemantic, { role: 'attention-close', operatorKey: 'close' }),
                text: ')',
                visual: { styleKey: VIEW2D_STYLE_KEYS.OPERATOR }
            }),
            createOperatorNode({
                role: 'attention-divide',
                semantic: buildSemantic(attentionSemantic, { role: 'attention-divide', operatorKey: 'divide' }),
                text: '/',
                visual: { styleKey: VIEW2D_STYLE_KEYS.OPERATOR }
            }),
            createTextNode({
                role: 'attention-scale',
                semantic: buildSemantic(attentionSemantic, { role: 'attention-scale', focusKey: 'scale' }),
                tex: scoreStage.scaleLabelTex,
                text: 'sqrt(d_h)',
                presentation: VIEW2D_TEXT_PRESENTATIONS.LABEL,
                visual: { styleKey: VIEW2D_STYLE_KEYS.LABEL }
            }),
            createOperatorNode({
                role: 'attention-equals',
                semantic: buildSemantic(attentionSemantic, { role: 'attention-equals', operatorKey: 'equals' }),
                text: '=',
                visual: { styleKey: VIEW2D_STYLE_KEYS.OPERATOR }
            }),
            softmaxStageNode
        ],
        metadata: {
            queryStageIndex,
            valueStageIndex
        }
    });
}

function buildConnectorNodes({
    baseSemantic,
    layoutMetrics,
    projectionNodes,
    attentionNode
}) {
    const connectorGaps = layoutMetrics?.connectorGaps || {};
    const findProjectionOutput = (kind) => projectionNodes.find((stageNode) => stageNode?.metadata?.kind === kind)
        ?.children?.[2]?.children?.[4] || null;
    const findAttentionNode = (role) => {
        const queue = Array.isArray(attentionNode?.children) ? [...attentionNode.children] : [];
        while (queue.length) {
            const node = queue.shift();
            if (!node || typeof node !== 'object') continue;
            if (node.role === role) return node;
            if (Array.isArray(node.children) && node.children.length) {
                queue.unshift(...node.children);
            }
        }
        return null;
    };

    const qOutputNode = findProjectionOutput('q');
    const kOutputNode = findProjectionOutput('k');
    const vOutputNode = findProjectionOutput('v');
    const querySourceNode = findAttentionNode('attention-query-source');
    const transposeNode = findAttentionNode('attention-key-transpose');
    const preScoreNode = findAttentionNode('attention-pre-score');
    const maskedInputNode = findAttentionNode('attention-masked-input');
    const postNode = findAttentionNode('attention-post');
    const postCopyNode = findAttentionNode('attention-post-copy');
    const valuePostNode = findAttentionNode('attention-value-post');

    return [
        qOutputNode && querySourceNode
            ? createConnectorNode({
                role: 'connector-q',
                semantic: buildSemantic(baseSemantic, { stage: 'connector-q', role: 'connector-q' }),
                source: createAnchorRef(qOutputNode.id, VIEW2D_ANCHOR_SIDES.RIGHT),
                target: createAnchorRef(querySourceNode.id, VIEW2D_ANCHOR_SIDES.TOP),
                route: VIEW2D_CONNECTOR_ROUTES.HORIZONTAL,
                gap: connectorGaps.projection,
                gapKey: 'projection',
                visual: { styleKey: VIEW2D_STYLE_KEYS.CONNECTOR_Q }
            })
            : null,
        kOutputNode && transposeNode
            ? createConnectorNode({
                role: 'connector-k',
                semantic: buildSemantic(baseSemantic, { stage: 'connector-k', role: 'connector-k' }),
                source: createAnchorRef(kOutputNode.id, VIEW2D_ANCHOR_SIDES.RIGHT),
                target: createAnchorRef(transposeNode.id, VIEW2D_ANCHOR_SIDES.BOTTOM),
                route: VIEW2D_CONNECTOR_ROUTES.HORIZONTAL,
                gap: connectorGaps.transpose,
                gapKey: 'transpose',
                visual: { styleKey: VIEW2D_STYLE_KEYS.CONNECTOR_K }
            })
            : null,
        preScoreNode && maskedInputNode
            ? createConnectorNode({
                role: 'connector-pre',
                semantic: buildSemantic(baseSemantic, { stage: 'connector-pre', role: 'connector-pre' }),
                source: createAnchorRef(preScoreNode.id, VIEW2D_ANCHOR_SIDES.BOTTOM),
                target: createAnchorRef(maskedInputNode.id, VIEW2D_ANCHOR_SIDES.TOP),
                route: VIEW2D_CONNECTOR_ROUTES.VERTICAL,
                gap: connectorGaps.pre,
                gapKey: 'pre',
                visual: { styleKey: VIEW2D_STYLE_KEYS.CONNECTOR_NEUTRAL }
            })
            : null,
        postNode && postCopyNode
            ? createConnectorNode({
                role: 'connector-post',
                semantic: buildSemantic(baseSemantic, { stage: 'connector-post', role: 'connector-post' }),
                source: createAnchorRef(postNode.id, VIEW2D_ANCHOR_SIDES.RIGHT),
                target: createAnchorRef(postCopyNode.id, VIEW2D_ANCHOR_SIDES.LEFT),
                route: VIEW2D_CONNECTOR_ROUTES.HORIZONTAL,
                gap: connectorGaps.post,
                gapKey: 'post',
                visual: { styleKey: VIEW2D_STYLE_KEYS.CONNECTOR_POST }
            })
            : null,
        vOutputNode && valuePostNode
            ? createConnectorNode({
                role: 'connector-v',
                semantic: buildSemantic(baseSemantic, { stage: 'connector-v', role: 'connector-v' }),
                source: createAnchorRef(vOutputNode.id, VIEW2D_ANCHOR_SIDES.RIGHT),
                target: createAnchorRef(valuePostNode.id, VIEW2D_ANCHOR_SIDES.BOTTOM),
                route: VIEW2D_CONNECTOR_ROUTES.HORIZONTAL,
                gap: connectorGaps.value,
                gapKey: 'value',
                visual: { styleKey: VIEW2D_STYLE_KEYS.CONNECTOR_V }
            })
            : null
    ].filter(Boolean);
}

export function buildMhsaSceneModel({
    previewData = null,
    activationSource = null,
    layerIndex = null,
    headIndex = null,
    sampleStep = 64,
    tokenIndices = null,
    tokenLabels = null,
    isSmallScreen = false,
    layoutMetrics = null,
    visualTokens = null
} = {}) {
    const resolvedPreviewData = previewData || buildMhsaTokenMatrixPreviewData({
        activationSource,
        layerIndex: normalizeIndex(layerIndex) ?? undefined,
        headIndex: normalizeIndex(headIndex) ?? undefined,
        sampleStep,
        tokenIndices,
        tokenLabels
    });
    if (
        !resolvedPreviewData?.rowCount
        || !resolvedPreviewData?.columnCount
        || !Array.isArray(resolvedPreviewData.rows)
        || !Array.isArray(resolvedPreviewData.projections)
        || !resolvedPreviewData.projections.length
    ) {
        return null;
    }

    const resolvedLayerIndex = normalizeIndex(layerIndex);
    const resolvedHeadIndex = normalizeIndex(headIndex);
    const baseSemantic = {
        componentKind: 'mhsa',
        layerIndex: resolvedLayerIndex,
        headIndex: resolvedHeadIndex
    };
    const resolvedLayoutMetrics = layoutMetrics || resolveMhsaTokenMatrixLayoutMetrics({
        rowCount: resolvedPreviewData.rowCount,
        isSmallScreen
    });
    const resolvedTokens = visualTokens || resolveView2dVisualTokens();
    const queryStageIndex = Math.max(
        0,
        resolvedPreviewData.projections.findIndex((projectionData) => String(projectionData?.kind || '').toLowerCase() === 'q')
    );
    const valueStageIndex = Math.max(
        0,
        resolvedPreviewData.projections.findIndex((projectionData) => String(projectionData?.kind || '').toLowerCase() === 'v')
    );

    const projectionNodes = resolvedPreviewData.projections
        .map((projectionData, stageIndex) => {
            const validProjection = projectionData
                && projectionData.weightRowCount
                && projectionData.weightColumnCount
                && projectionData.outputRowCount
                && projectionData.outputColumnCount
                && Array.isArray(projectionData.outputRows);
            if (!validProjection) return null;
            return buildProjectionStageNode({
                baseSemantic,
                previewData: resolvedPreviewData,
                projectionData,
                stageIndex
            });
        })
        .filter(Boolean);

    const attentionNode = resolvedPreviewData.attentionScoreStage
        && Array.isArray(resolvedPreviewData.attentionScoreStage.queryRows)
        && Array.isArray(resolvedPreviewData.attentionScoreStage.outputRows)
        ? buildAttentionStageNode({
            baseSemantic,
            scoreStage: resolvedPreviewData.attentionScoreStage,
            queryStageIndex,
            valueStageIndex
        })
        : null;

    const rootNodes = [
        createGroupNode({
            role: 'projection-stack',
            semantic: buildSemantic(baseSemantic, { stage: 'projection-stack', role: 'projection-stack' }),
            direction: VIEW2D_LAYOUT_DIRECTIONS.VERTICAL,
            gapKey: 'stack',
            children: projectionNodes
        })
    ];

    if (attentionNode) {
        rootNodes.push(attentionNode);
        rootNodes.push(
            createGroupNode({
                role: 'connector-layer',
                semantic: buildSemantic(baseSemantic, { stage: 'connector-layer', role: 'connector-layer' }),
                direction: VIEW2D_LAYOUT_DIRECTIONS.OVERLAY,
                gapKey: 'default',
                children: buildConnectorNodes({
                    baseSemantic,
                    layoutMetrics: resolvedLayoutMetrics,
                    projectionNodes,
                    attentionNode
                })
            })
        );
    }

    return createSceneModel({
        semantic: buildSemantic(baseSemantic, { role: 'scene' }),
        nodes: rootNodes,
        metadata: {
            visualContract: 'selection-panel-mhsa-v1',
            source: 'selectionPanelMhsaTokenMatrixUtils',
            rowCount: resolvedPreviewData.rowCount,
            columnCount: resolvedPreviewData.columnCount,
            bandCount: resolvedPreviewData.bandCount,
            sampleStep: resolvedPreviewData.sampleStep,
            layoutMetrics: resolvedLayoutMetrics,
            tokens: resolvedTokens
        }
    });
}
