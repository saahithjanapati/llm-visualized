import { NUM_LAYERS } from '../../app/gpt-tower/config.js';
import {
    MHA_FINAL_Q_COLOR,
    MHA_FINAL_V_COLOR
} from '../../animations/LayerAnimationConstants.js';
import {
    D_HEAD,
    D_MODEL,
    FINAL_MLP_COLOR,
    RESIDUAL_COLOR_CLAMP,
    VOCAB_SIZE,
    CONTEXT_LEN
} from '../../ui/selectionPanelConstants.js';
import { mapValueToColor, buildHueRangeOptions, mapValueToHueRange } from '../../utils/colors.js';
import { NUM_HEAD_SETS_LAYER, TOP_LOGIT_BAR_COLOR } from '../../utils/constants.js';
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

const DEFAULT_VISIBLE_TOKEN_COUNT = 5;
const SUMMARY_MODEL_COLS = 18;
const SUMMARY_HEAD_COLS = 12;
const SUMMARY_MLP_COLS = 24;
const SUMMARY_LOGIT_COLS = 12;

const HEAD_RANGE_OPTIONS = buildHueRangeOptions(MHA_FINAL_Q_COLOR, {
    valueMin: -2,
    valueMax: 2,
    minLightness: 0.36,
    maxLightness: 0.74
});

const VALUE_RANGE_OPTIONS = buildHueRangeOptions(MHA_FINAL_V_COLOR, {
    valueMin: -2,
    valueMax: 2,
    minLightness: 0.34,
    maxLightness: 0.72
});

const MLP_RANGE_OPTIONS = buildHueRangeOptions(FINAL_MLP_COLOR, {
    valueMin: -2,
    valueMax: 2,
    minLightness: 0.34,
    maxLightness: 0.72
});

const LOGIT_RANGE_OPTIONS = buildHueRangeOptions(TOP_LOGIT_BAR_COLOR, {
    valueMin: 0,
    valueMax: 1,
    minLightness: 0.42,
    maxLightness: 0.74
});

function normalizeIndex(value) {
    return Number.isFinite(value) ? Math.max(0, Math.floor(value)) : null;
}

function cleanNumberArray(values = [], fallbackLength = 0) {
    if (!Array.isArray(values) || !values.length) {
        return fallbackLength > 0 ? new Array(fallbackLength).fill(0) : [];
    }
    return values.map((value) => (Number.isFinite(value) ? value : 0));
}

function sampleVector(values = [], targetLength = SUMMARY_MODEL_COLS) {
    const safeValues = cleanNumberArray(values);
    const length = Math.max(1, Math.floor(targetLength || SUMMARY_MODEL_COLS));
    if (!safeValues.length) return new Array(length).fill(0);
    if (safeValues.length <= length) {
        return [...safeValues];
    }
    return Array.from({ length }, (_, index) => {
        const start = Math.floor((index * safeValues.length) / length);
        const end = Math.max(start + 1, Math.floor(((index + 1) * safeValues.length) / length));
        let sum = 0;
        for (let cursor = start; cursor < end; cursor += 1) {
            sum += safeValues[cursor];
        }
        return sum / Math.max(1, end - start);
    });
}

function colorToCss(color) {
    return color?.isColor ? `#${color.getHexString()}` : 'transparent';
}

function buildGradientCss(values = [], {
    clampMax = RESIDUAL_COLOR_CLAMP,
    direction = '90deg',
    rangeOptions = null
} = {}) {
    const safeValues = cleanNumberArray(values);
    if (!safeValues.length) return 'none';
    const stops = safeValues.map((value, index) => {
        const ratio = safeValues.length > 1 ? index / (safeValues.length - 1) : 0;
        const color = rangeOptions
            ? mapValueToHueRange(value, rangeOptions)
            : mapValueToColor(value, { clampMax });
        return `${colorToCss(color)} ${(ratio * 100).toFixed(4)}%`;
    });
    if (stops.length === 1) {
        return `linear-gradient(${direction}, ${stops[0].replace(' 0.0000%', ' 0%')}, ${stops[0].replace(' 0.0000%', ' 100%')})`;
    }
    return `linear-gradient(${direction}, ${stops.join(', ')})`;
}

function buildLabel(labelTex = '', fallbackText = '') {
    return {
        tex: typeof labelTex === 'string' ? labelTex : '',
        text: typeof fallbackText === 'string' && fallbackText.length
            ? fallbackText
            : (typeof labelTex === 'string' ? labelTex : '')
    };
}

function buildSemantic(baseSemantic, extra = {}) {
    return {
        ...baseSemantic,
        ...extra
    };
}

function resolveVisibleTokenIndices(activationSource = null, tokenIndices = null, maxTokens = DEFAULT_VISIBLE_TOKEN_COUNT) {
    const safeMaxTokens = Math.max(1, Math.floor(maxTokens || DEFAULT_VISIBLE_TOKEN_COUNT));
    if (Array.isArray(tokenIndices) && tokenIndices.length) {
        return tokenIndices
            .map((value) => Number(value))
            .filter(Number.isFinite)
            .map((value) => Math.max(0, Math.floor(value)))
            .slice(0, safeMaxTokens);
    }
    const tokenCount = typeof activationSource?.getTokenCount === 'function'
        ? activationSource.getTokenCount()
        : (Array.isArray(activationSource?.meta?.prompt_tokens) ? activationSource.meta.prompt_tokens.length : 0);
    const count = Number.isFinite(tokenCount) && tokenCount > 0
        ? Math.min(safeMaxTokens, Math.floor(tokenCount))
        : safeMaxTokens;
    return Array.from({ length: count }, (_, index) => index);
}

function resolveTokenLabel(activationSource = null, tokenIndex = 0, fallbackLabel = null, fallbackIndex = 0) {
    if (typeof fallbackLabel === 'string' && fallbackLabel.trim().length) {
        return fallbackLabel.trim();
    }
    const label = typeof activationSource?.getTokenString === 'function'
        ? activationSource.getTokenString(tokenIndex)
        : null;
    if (typeof label === 'string' && label.trim().length) {
        return label.trim();
    }
    return `Token ${fallbackIndex + 1}`;
}

function resolveVisibleTokenRefs(activationSource = null, tokenIndices = null, tokenLabels = null) {
    return resolveVisibleTokenIndices(activationSource, tokenIndices).map((tokenIndex, rowIndex) => ({
        rowIndex,
        tokenIndex,
        tokenLabel: resolveTokenLabel(
            activationSource,
            tokenIndex,
            Array.isArray(tokenLabels) ? tokenLabels[rowIndex] : null,
            rowIndex
        )
    }));
}

function buildVectorRowItems(tokenRefs = [], {
    baseSemantic = {},
    role = 'row',
    measureCols = SUMMARY_MODEL_COLS,
    clampMax = RESIDUAL_COLOR_CLAMP,
    rangeOptions = null,
    getVector = () => null
} = {}) {
    return tokenRefs.map((tokenRef) => {
        const values = sampleVector(getVector(tokenRef) || [], measureCols);
        const semantic = buildSemantic(baseSemantic, {
            role,
            rowIndex: tokenRef.rowIndex,
            tokenIndex: tokenRef.tokenIndex
        });
        return {
            id: buildSceneNodeId(semantic),
            index: tokenRef.rowIndex,
            label: tokenRef.tokenLabel,
            semantic,
            rawValues: values,
            gradientCss: buildGradientCss(values, {
                clampMax,
                rangeOptions
            }),
            title: `${tokenRef.tokenLabel}`
        };
    });
}

function buildLogitRowItems(logitEntries = [], baseSemantic = {}) {
    return logitEntries.map((entry, index) => {
        const semantic = buildSemantic(baseSemantic, {
            role: 'logit-row',
            rowIndex: index
        });
        const score = Number(entry?.probability);
        const color = mapValueToHueRange(Number.isFinite(score) ? score : 0, LOGIT_RANGE_OPTIONS);
        const fillCss = `linear-gradient(90deg, ${colorToCss(color)} 0%, ${colorToCss(color)} 100%)`;
        return {
            id: buildSceneNodeId(semantic),
            index,
            label: entry?.tokenLabel || `Candidate ${index + 1}`,
            semantic,
            rawValue: Number.isFinite(score) ? score : 0,
            rawValues: [Number.isFinite(score) ? score : 0],
            gradientCss: fillCss,
            title: entry?.tokenLabel || `Candidate ${index + 1}`
        };
    });
}

function createMeasureMetadata(cols, rows = null) {
    const measure = {};
    if (Number.isFinite(cols) && cols > 0) measure.cols = Math.floor(cols);
    if (Number.isFinite(rows) && rows > 0) measure.rows = Math.floor(rows);
    return Object.keys(measure).length ? { measure } : null;
}

function createModuleTitleNode(baseSemantic = {}, text = '') {
    return createTextNode({
        role: 'module-title',
        semantic: buildSemantic(baseSemantic, { role: 'module-title' }),
        text,
        presentation: VIEW2D_TEXT_PRESENTATIONS.LABEL,
        visual: { styleKey: VIEW2D_STYLE_KEYS.LABEL }
    });
}

function createModuleGroup(baseSemantic = {}, title = '', children = []) {
    return createGroupNode({
        role: 'module',
        semantic: buildSemantic(baseSemantic, { role: 'module' }),
        direction: VIEW2D_LAYOUT_DIRECTIONS.VERTICAL,
        gapKey: 'default',
        children: [
            createModuleTitleNode(baseSemantic, title),
            createGroupNode({
                role: 'module-body',
                semantic: buildSemantic(baseSemantic, { role: 'module-body' }),
                direction: VIEW2D_LAYOUT_DIRECTIONS.HORIZONTAL,
                gapKey: 'inline',
                children
            })
        ]
    });
}

function buildResidualNode({
    semantic = {},
    labelTex = '',
    fallbackText = '',
    tokenRefs = [],
    getVector = () => null
} = {}) {
    return createMatrixNode({
        role: semantic.role || 'module',
        semantic,
        label: buildLabel(labelTex, fallbackText),
        dimensions: {
            rows: tokenRefs.length,
            cols: D_MODEL
        },
        presentation: VIEW2D_MATRIX_PRESENTATIONS.BANDED_ROWS,
        shape: VIEW2D_MATRIX_SHAPES.MATRIX,
        rowItems: buildVectorRowItems(tokenRefs, {
            baseSemantic: semantic,
            role: 'vector-row',
            measureCols: SUMMARY_MODEL_COLS,
            getVector
        }),
        visual: {
            styleKey: VIEW2D_STYLE_KEYS.RESIDUAL
        },
        metadata: createMeasureMetadata(SUMMARY_MODEL_COLS, tokenRefs.length)
    });
}

function buildSimpleModuleNode({
    componentKind = '',
    layerIndex = null,
    stage = '',
    title = '',
    labelTex = '',
    fallbackText = '',
    styleKey = VIEW2D_STYLE_KEYS.RESIDUAL,
    tokenRefs = [],
    measureCols = SUMMARY_MODEL_COLS,
    getVector = () => null
} = {}) {
    return createModuleGroup({
        componentKind,
        layerIndex,
        stage
    }, title, [
        createMatrixNode({
            role: 'module-summary',
            semantic: {
                componentKind,
                layerIndex,
                stage,
                role: 'module-summary'
            },
            label: buildLabel(labelTex, fallbackText),
            dimensions: {
                rows: tokenRefs.length,
                cols: D_MODEL
            },
            presentation: VIEW2D_MATRIX_PRESENTATIONS.BANDED_ROWS,
            shape: VIEW2D_MATRIX_SHAPES.MATRIX,
            rowItems: buildVectorRowItems(tokenRefs, {
                baseSemantic: {
                    componentKind,
                    layerIndex,
                    stage
                },
                role: 'vector-row',
                measureCols,
                rangeOptions: styleKey === VIEW2D_STYLE_KEYS.MLP ? MLP_RANGE_OPTIONS : null,
                getVector
            }),
            visual: {
                styleKey
            },
            metadata: createMeasureMetadata(measureCols, tokenRefs.length)
        })
    ]);
}

function buildEmbeddingModule({
    tokenRefs = [],
    activationSource = null
} = {}) {
    const baseSemantic = {
        componentKind: 'embedding',
        stage: 'input'
    };
    const tokenNode = createMatrixNode({
        role: 'token-embedding',
        semantic: buildSemantic(baseSemantic, { stage: 'token', role: 'token-embedding' }),
        label: buildLabel('E_{tok}', 'E_tok'),
        dimensions: {
            rows: tokenRefs.length,
            cols: D_MODEL
        },
        presentation: VIEW2D_MATRIX_PRESENTATIONS.BANDED_ROWS,
        shape: VIEW2D_MATRIX_SHAPES.MATRIX,
        rowItems: buildVectorRowItems(tokenRefs, {
            baseSemantic,
            role: 'token-row',
            measureCols: SUMMARY_MODEL_COLS,
            getVector: (tokenRef) => activationSource?.getEmbedding?.('token', tokenRef.tokenIndex, D_MODEL)
        }),
        visual: {
            styleKey: VIEW2D_STYLE_KEYS.EMBEDDING_TOKEN
        },
        metadata: createMeasureMetadata(SUMMARY_MODEL_COLS, tokenRefs.length)
    });

    const positionNode = createMatrixNode({
        role: 'position-embedding',
        semantic: buildSemantic(baseSemantic, { stage: 'position', role: 'position-embedding' }),
        label: buildLabel('E_{pos}', 'E_pos'),
        dimensions: {
            rows: tokenRefs.length,
            cols: D_MODEL
        },
        presentation: VIEW2D_MATRIX_PRESENTATIONS.BANDED_ROWS,
        shape: VIEW2D_MATRIX_SHAPES.MATRIX,
        rowItems: buildVectorRowItems(tokenRefs, {
            baseSemantic,
            role: 'position-row',
            measureCols: SUMMARY_MODEL_COLS,
            rangeOptions: VALUE_RANGE_OPTIONS,
            getVector: (tokenRef) => activationSource?.getEmbedding?.('position', tokenRef.tokenIndex, D_MODEL)
        }),
        visual: {
            styleKey: VIEW2D_STYLE_KEYS.EMBEDDING_POSITION
        },
        metadata: createMeasureMetadata(SUMMARY_MODEL_COLS, tokenRefs.length)
    });

    const sumNode = createMatrixNode({
        role: 'sum-output',
        semantic: buildSemantic(baseSemantic, { stage: 'sum', role: 'sum-output' }),
        label: buildLabel('X_0', 'X_0'),
        dimensions: {
            rows: tokenRefs.length,
            cols: D_MODEL
        },
        presentation: VIEW2D_MATRIX_PRESENTATIONS.BANDED_ROWS,
        shape: VIEW2D_MATRIX_SHAPES.MATRIX,
        rowItems: buildVectorRowItems(tokenRefs, {
            baseSemantic,
            role: 'sum-row',
            measureCols: SUMMARY_MODEL_COLS,
            getVector: (tokenRef) => activationSource?.getEmbedding?.('sum', tokenRef.tokenIndex, D_MODEL)
        }),
        visual: {
            styleKey: VIEW2D_STYLE_KEYS.RESIDUAL
        },
        metadata: createMeasureMetadata(SUMMARY_MODEL_COLS, tokenRefs.length)
    });

    const node = createModuleGroup(baseSemantic, 'Embeddings', [
        tokenNode,
        createOperatorNode({
            role: 'embedding-plus',
            semantic: buildSemantic(baseSemantic, { stage: 'plus', role: 'embedding-plus', operatorKey: 'plus' }),
            text: '+',
            visual: { styleKey: VIEW2D_STYLE_KEYS.OPERATOR }
        }),
        positionNode,
        createOperatorNode({
            role: 'embedding-equals',
            semantic: buildSemantic(baseSemantic, { stage: 'equals', role: 'embedding-equals', operatorKey: 'equals' }),
            text: '=',
            visual: { styleKey: VIEW2D_STYLE_KEYS.OPERATOR }
        }),
        sumNode
    ]);

    return {
        node,
        entryNode: sumNode,
        exitNode: sumNode
    };
}

function buildMhsaModule({
    layerIndex = 0,
    tokenRefs = [],
    activationSource = null
} = {}) {
    const baseSemantic = {
        componentKind: 'mhsa',
        layerIndex,
        stage: 'attention'
    };

    const headNodes = Array.from({ length: NUM_HEAD_SETS_LAYER }, (_, headIndex) => {
        const headSemantic = buildSemantic(baseSemantic, {
            headIndex,
            stage: 'head-stack'
        });
        const headVector = activationSource?.getAttentionWeightedSum?.(
            layerIndex,
            headIndex,
            tokenRefs[tokenRefs.length - 1]?.tokenIndex ?? 0,
            D_HEAD
        ) || activationSource?.getLayerQKVVector?.(
            layerIndex,
            'q',
            headIndex,
            tokenRefs[tokenRefs.length - 1]?.tokenIndex ?? 0,
            D_HEAD
        );
        const summaryValues = sampleVector(headVector || [], SUMMARY_HEAD_COLS);

        return createGroupNode({
            role: 'head',
            semantic: buildSemantic(headSemantic, { role: 'head' }),
            direction: VIEW2D_LAYOUT_DIRECTIONS.HORIZONTAL,
            gapKey: 'inline',
            children: [
                createTextNode({
                    role: 'head-badge',
                    semantic: buildSemantic(headSemantic, { role: 'head-badge' }),
                    text: `H${headIndex + 1}`,
                    presentation: VIEW2D_TEXT_PRESENTATIONS.LABEL,
                    visual: { styleKey: VIEW2D_STYLE_KEYS.CAPTION }
                }),
                createMatrixNode({
                    role: 'head-summary',
                    semantic: buildSemantic(headSemantic, { role: 'head-summary' }),
                    dimensions: {
                        rows: 1,
                        cols: D_HEAD
                    },
                    presentation: VIEW2D_MATRIX_PRESENTATIONS.COMPACT_ROWS,
                    shape: VIEW2D_MATRIX_SHAPES.MATRIX,
                    rowItems: [{
                        id: buildSceneNodeId(buildSemantic(headSemantic, { role: 'head-summary-row', rowIndex: 0 })),
                        index: 0,
                        label: `Head ${headIndex + 1}`,
                        semantic: buildSemantic(headSemantic, { role: 'head-summary-row', rowIndex: 0 }),
                        rawValues: summaryValues,
                        gradientCss: buildGradientCss(summaryValues, { rangeOptions: HEAD_RANGE_OPTIONS })
                    }],
                    visual: {
                        styleKey: VIEW2D_STYLE_KEYS.MHSA_HEAD
                    },
                    metadata: createMeasureMetadata(SUMMARY_HEAD_COLS, 1)
                })
            ]
        });
    });

    const node = createModuleGroup(baseSemantic, 'MHSA', [
        createGroupNode({
            role: 'head-stack',
            semantic: buildSemantic(baseSemantic, { role: 'head-stack' }),
            direction: VIEW2D_LAYOUT_DIRECTIONS.VERTICAL,
            gapKey: 'default',
            children: headNodes
        })
    ]);

    return {
        node,
        entryNode: node,
        exitNode: node
    };
}

function buildOutputProjectionModule({
    layerIndex = 0,
    tokenRefs = [],
    activationSource = null
} = {}) {
    const baseSemantic = {
        componentKind: 'output-projection',
        layerIndex,
        stage: 'attn-out'
    };
    const outputNode = createMatrixNode({
        role: 'projection-output',
        semantic: buildSemantic(baseSemantic, { role: 'projection-output' }),
        label: buildLabel('A_{out}', 'attn_out'),
        dimensions: {
            rows: tokenRefs.length,
            cols: D_MODEL
        },
        presentation: VIEW2D_MATRIX_PRESENTATIONS.BANDED_ROWS,
        shape: VIEW2D_MATRIX_SHAPES.MATRIX,
        rowItems: buildVectorRowItems(tokenRefs, {
            baseSemantic,
            role: 'output-row',
            measureCols: SUMMARY_MODEL_COLS,
            getVector: (tokenRef) => activationSource?.getAttentionOutputProjection?.(layerIndex, tokenRef.tokenIndex, D_MODEL)
        }),
        visual: {
            styleKey: VIEW2D_STYLE_KEYS.OUTPUT_PROJECTION
        },
        metadata: createMeasureMetadata(SUMMARY_MODEL_COLS, tokenRefs.length)
    });

    const weightNode = createMatrixNode({
        role: 'projection-weight',
        semantic: buildSemantic(baseSemantic, { role: 'projection-weight' }),
        label: buildLabel('W_O', 'W_O'),
        dimensions: {
            rows: D_MODEL,
            cols: D_MODEL
        },
        presentation: VIEW2D_MATRIX_PRESENTATIONS.CARD,
        shape: VIEW2D_MATRIX_SHAPES.MATRIX,
        visual: {
            styleKey: VIEW2D_STYLE_KEYS.MATRIX_WEIGHT
        }
    });

    const node = createModuleGroup(baseSemantic, 'Output proj', [
        weightNode,
        outputNode
    ]);

    return {
        node,
        entryNode: node,
        exitNode: outputNode
    };
}

function buildResidualAddModule({
    layerIndex = 0,
    stage = '',
    title = '',
    tokenRefs = [],
    getVector = () => null
} = {}) {
    const baseSemantic = {
        componentKind: 'residual',
        layerIndex,
        stage
    };
    const outputNode = createMatrixNode({
        role: 'residual-add-output',
        semantic: buildSemantic(baseSemantic, { role: 'residual-add-output' }),
        label: buildLabel('x_{res}', 'x_res'),
        dimensions: {
            rows: tokenRefs.length,
            cols: D_MODEL
        },
        presentation: VIEW2D_MATRIX_PRESENTATIONS.BANDED_ROWS,
        shape: VIEW2D_MATRIX_SHAPES.MATRIX,
        rowItems: buildVectorRowItems(tokenRefs, {
            baseSemantic,
            role: 'residual-row',
            measureCols: SUMMARY_MODEL_COLS,
            getVector
        }),
        visual: {
            styleKey: VIEW2D_STYLE_KEYS.RESIDUAL
        },
        metadata: createMeasureMetadata(SUMMARY_MODEL_COLS, tokenRefs.length)
    });

    const node = createModuleGroup(baseSemantic, title, [
        createOperatorNode({
            role: 'residual-add-operator',
            semantic: buildSemantic(baseSemantic, { role: 'residual-add-operator', operatorKey: 'plus' }),
            text: '+',
            visual: { styleKey: VIEW2D_STYLE_KEYS.OPERATOR }
        }),
        outputNode
    ]);

    return {
        node,
        entryNode: node,
        exitNode: outputNode
    };
}

function buildMlpModule({
    layerIndex = 0,
    tokenRefs = [],
    activationSource = null
} = {}) {
    const baseSemantic = {
        componentKind: 'mlp',
        layerIndex,
        stage: 'mlp'
    };
    const upNode = createMatrixNode({
        role: 'mlp-up',
        semantic: buildSemantic(baseSemantic, { stage: 'mlp-up', role: 'mlp-up' }),
        label: buildLabel('MLP_{up}', 'MLP_up'),
        dimensions: {
            rows: tokenRefs.length,
            cols: D_MODEL * 4
        },
        presentation: VIEW2D_MATRIX_PRESENTATIONS.BANDED_ROWS,
        shape: VIEW2D_MATRIX_SHAPES.MATRIX,
        rowItems: buildVectorRowItems(tokenRefs, {
            baseSemantic,
            role: 'mlp-up-row',
            measureCols: SUMMARY_MLP_COLS,
            rangeOptions: MLP_RANGE_OPTIONS,
            getVector: (tokenRef) => activationSource?.getMlpUp?.(layerIndex, tokenRef.tokenIndex, D_MODEL * 4)
        }),
        visual: {
            styleKey: VIEW2D_STYLE_KEYS.MLP
        },
        metadata: createMeasureMetadata(SUMMARY_MLP_COLS, tokenRefs.length)
    });

    const activationNode = createMatrixNode({
        role: 'mlp-activation',
        semantic: buildSemantic(baseSemantic, { stage: 'mlp-activation', role: 'mlp-activation' }),
        label: buildLabel('GELU', 'GELU'),
        dimensions: {
            rows: tokenRefs.length,
            cols: D_MODEL * 4
        },
        presentation: VIEW2D_MATRIX_PRESENTATIONS.BANDED_ROWS,
        shape: VIEW2D_MATRIX_SHAPES.MATRIX,
        rowItems: buildVectorRowItems(tokenRefs, {
            baseSemantic,
            role: 'mlp-activation-row',
            measureCols: SUMMARY_MLP_COLS,
            rangeOptions: MLP_RANGE_OPTIONS,
            getVector: (tokenRef) => activationSource?.getMlpActivation?.(layerIndex, tokenRef.tokenIndex, D_MODEL * 4)
        }),
        visual: {
            styleKey: VIEW2D_STYLE_KEYS.MLP
        },
        metadata: createMeasureMetadata(SUMMARY_MLP_COLS, tokenRefs.length)
    });

    const downNode = createMatrixNode({
        role: 'mlp-down',
        semantic: buildSemantic(baseSemantic, { stage: 'mlp-down', role: 'mlp-down' }),
        label: buildLabel('MLP_{down}', 'MLP_down'),
        dimensions: {
            rows: tokenRefs.length,
            cols: D_MODEL
        },
        presentation: VIEW2D_MATRIX_PRESENTATIONS.BANDED_ROWS,
        shape: VIEW2D_MATRIX_SHAPES.MATRIX,
        rowItems: buildVectorRowItems(tokenRefs, {
            baseSemantic,
            role: 'mlp-down-row',
            measureCols: SUMMARY_MODEL_COLS,
            rangeOptions: MLP_RANGE_OPTIONS,
            getVector: (tokenRef) => activationSource?.getMlpDown?.(layerIndex, tokenRef.tokenIndex, D_MODEL)
        }),
        visual: {
            styleKey: VIEW2D_STYLE_KEYS.MLP
        },
        metadata: createMeasureMetadata(SUMMARY_MODEL_COLS, tokenRefs.length)
    });

    const node = createModuleGroup(baseSemantic, 'MLP', [
        upNode,
        activationNode,
        downNode
    ]);

    return {
        node,
        entryNode: node,
        exitNode: downNode
    };
}

function buildLayerGroup({
    layerIndex = 0,
    tokenRefs = [],
    activationSource = null
} = {}) {
    const layerSemantic = {
        componentKind: 'layer',
        layerIndex
    };

    const residualInNode = buildResidualNode({
        semantic: {
            componentKind: 'residual',
            layerIndex,
            stage: 'incoming',
            role: 'module'
        },
        labelTex: 'x_{in}',
        fallbackText: 'x_in',
        tokenRefs,
        getVector: (tokenRef) => activationSource?.getLayerIncoming?.(layerIndex, tokenRef.tokenIndex, D_MODEL)
    });

    const ln1Node = buildSimpleModuleNode({
        componentKind: 'layer-norm',
        layerIndex,
        stage: 'ln1',
        title: 'LN1',
        labelTex: 'X_{\\ln 1}',
        fallbackText: 'X_ln1',
        styleKey: VIEW2D_STYLE_KEYS.LAYER_NORM,
        tokenRefs,
        measureCols: SUMMARY_MODEL_COLS,
        getVector: (tokenRef) => activationSource?.getLayerLn1?.(layerIndex, 'shift', tokenRef.tokenIndex, D_MODEL)
    });

    const mhsaModule = buildMhsaModule({
        layerIndex,
        tokenRefs,
        activationSource
    });

    const outputProjectionModule = buildOutputProjectionModule({
        layerIndex,
        tokenRefs,
        activationSource
    });

    const postAttentionAdd = buildResidualAddModule({
        layerIndex,
        stage: 'post-attn-add',
        title: 'Add',
        tokenRefs,
        getVector: (tokenRef) => activationSource?.getPostAttentionResidual?.(layerIndex, tokenRef.tokenIndex, D_MODEL)
    });

    const ln2Node = buildSimpleModuleNode({
        componentKind: 'layer-norm',
        layerIndex,
        stage: 'ln2',
        title: 'LN2',
        labelTex: 'X_{\\ln 2}',
        fallbackText: 'X_ln2',
        styleKey: VIEW2D_STYLE_KEYS.LAYER_NORM,
        tokenRefs,
        measureCols: SUMMARY_MODEL_COLS,
        getVector: (tokenRef) => activationSource?.getLayerLn2?.(layerIndex, 'shift', tokenRef.tokenIndex, D_MODEL)
    });

    const mlpModule = buildMlpModule({
        layerIndex,
        tokenRefs,
        activationSource
    });

    const postMlpAdd = buildResidualAddModule({
        layerIndex,
        stage: 'post-mlp-add',
        title: 'Add',
        tokenRefs,
        getVector: (tokenRef) => activationSource?.getPostMlpResidual?.(layerIndex, tokenRef.tokenIndex, D_MODEL)
    });

    const residualOutNode = buildResidualNode({
        semantic: {
            componentKind: 'residual',
            layerIndex,
            stage: 'outgoing',
            role: 'module'
        },
        labelTex: 'x_{out}',
        fallbackText: 'x_out',
        tokenRefs,
        getVector: (tokenRef) => activationSource?.getPostMlpResidual?.(layerIndex, tokenRef.tokenIndex, D_MODEL)
    });

    const contentNode = createGroupNode({
        role: 'layer-body',
        semantic: buildSemantic(layerSemantic, { role: 'layer-body' }),
        direction: VIEW2D_LAYOUT_DIRECTIONS.HORIZONTAL,
        gapKey: 'stage',
        children: [
            residualInNode,
            ln1Node,
            mhsaModule.node,
            outputProjectionModule.node,
            postAttentionAdd.node,
            ln2Node,
            mlpModule.node,
            postMlpAdd.node,
            residualOutNode
        ]
    });

    const node = createGroupNode({
        role: 'layer',
        semantic: buildSemantic(layerSemantic, { role: 'layer' }),
        direction: VIEW2D_LAYOUT_DIRECTIONS.VERTICAL,
        gapKey: 'default',
        children: [
            createTextNode({
                role: 'layer-title',
                semantic: buildSemantic(layerSemantic, { role: 'layer-title' }),
                text: `Layer ${layerIndex + 1}`,
                presentation: VIEW2D_TEXT_PRESENTATIONS.LABEL,
                visual: { styleKey: VIEW2D_STYLE_KEYS.LABEL }
            }),
            contentNode
        ]
    });

    const flow = [
        { from: residualInNode, to: ln1Node, key: `layer-${layerIndex}-incoming-ln1` },
        { from: ln1Node, to: mhsaModule.node, key: `layer-${layerIndex}-ln1-mhsa` },
        { from: mhsaModule.node, to: outputProjectionModule.node, key: `layer-${layerIndex}-mhsa-outproj` },
        { from: outputProjectionModule.node, to: postAttentionAdd.node, key: `layer-${layerIndex}-outproj-add1` },
        { from: postAttentionAdd.node, to: ln2Node, key: `layer-${layerIndex}-add1-ln2` },
        { from: ln2Node, to: mlpModule.node, key: `layer-${layerIndex}-ln2-mlp` },
        { from: mlpModule.node, to: postMlpAdd.node, key: `layer-${layerIndex}-mlp-add2` },
        { from: postMlpAdd.node, to: residualOutNode, key: `layer-${layerIndex}-add2-outgoing` }
    ];

    return {
        node,
        entryNode: residualInNode,
        exitNode: residualOutNode,
        flow
    };
}

function buildFinalLayerNormModule({
    tokenRefs = [],
    activationSource = null
} = {}) {
    return buildSimpleModuleNode({
        componentKind: 'layer-norm',
        stage: 'final-ln',
        title: 'Final LN',
        labelTex: 'X_{\\ln,f}',
        fallbackText: 'X_lnf',
        styleKey: VIEW2D_STYLE_KEYS.LAYER_NORM,
        tokenRefs,
        measureCols: SUMMARY_MODEL_COLS,
        getVector: (tokenRef) => activationSource?.getFinalLayerNorm?.('shift', tokenRef.tokenIndex, D_MODEL)
    });
}

function buildLogitsModule({
    tokenRefs = [],
    activationSource = null
} = {}) {
    const baseSemantic = {
        componentKind: 'logits',
        stage: 'output'
    };
    const finalTokenIndex = tokenRefs[tokenRefs.length - 1]?.tokenIndex ?? 0;
    const rawEntries = typeof activationSource?.getLogitsForToken === 'function'
        ? activationSource.getLogitsForToken(finalTokenIndex, 8)
        : null;
    const logitEntries = Array.isArray(rawEntries)
        ? rawEntries.slice(0, 8).map((entry, index) => ({
            tokenLabel: typeof entry?.token === 'string' && entry.token.length
                ? entry.token
                : `Candidate ${index + 1}`,
            probability: Number.isFinite(entry?.prob) ? entry.prob : Number(entry?.probability ?? entry?.value ?? 0)
        }))
        : [];

    const logitsNode = createMatrixNode({
        role: 'logits-topk',
        semantic: buildSemantic(baseSemantic, { role: 'logits-topk' }),
        label: buildLabel('logits', 'logits'),
        dimensions: {
            rows: logitEntries.length || 1,
            cols: VOCAB_SIZE
        },
        presentation: VIEW2D_MATRIX_PRESENTATIONS.BANDED_ROWS,
        shape: VIEW2D_MATRIX_SHAPES.MATRIX,
        rowItems: buildLogitRowItems(logitEntries, baseSemantic),
        visual: {
            styleKey: VIEW2D_STYLE_KEYS.LOGITS
        },
        metadata: createMeasureMetadata(SUMMARY_LOGIT_COLS, Math.max(1, logitEntries.length))
    });

    const unembeddingNode = createMatrixNode({
        role: 'unembedding',
        semantic: buildSemantic(baseSemantic, { stage: 'unembedding', role: 'unembedding' }),
        label: buildLabel('W_U', 'W_U'),
        dimensions: {
            rows: D_MODEL,
            cols: VOCAB_SIZE
        },
        presentation: VIEW2D_MATRIX_PRESENTATIONS.CARD,
        shape: VIEW2D_MATRIX_SHAPES.MATRIX,
        visual: {
            styleKey: VIEW2D_STYLE_KEYS.MATRIX_WEIGHT
        }
    });

    const node = createModuleGroup(baseSemantic, 'Logits', [
        unembeddingNode,
        logitsNode
    ]);

    return {
        node,
        entryNode: node,
        exitNode: logitsNode
    };
}

function createFlowConnector(fromNode, toNode, connectorKey, styleKey = VIEW2D_STYLE_KEYS.CONNECTOR_POST) {
    if (!fromNode?.id || !toNode?.id) return null;
    return createConnectorNode({
        role: `connector-${connectorKey}`,
        semantic: {
            componentKind: 'transformer',
            stage: `connector-${connectorKey}`,
            role: `connector-${connectorKey}`
        },
        source: createAnchorRef(fromNode.id, VIEW2D_ANCHOR_SIDES.RIGHT),
        target: createAnchorRef(toNode.id, VIEW2D_ANCHOR_SIDES.LEFT),
        route: VIEW2D_CONNECTOR_ROUTES.HORIZONTAL,
        gap: 10,
        gapKey: 'default',
        visual: { styleKey }
    });
}

export function buildTransformerSceneModel({
    activationSource = null,
    tokenIndices = null,
    tokenLabels = null,
    layerCount = NUM_LAYERS,
    isSmallScreen = false,
    visualTokens = null
} = {}) {
    const resolvedLayerCount = Number.isFinite(layerCount) ? Math.max(1, Math.floor(layerCount)) : NUM_LAYERS;
    const tokenRefs = resolveVisibleTokenRefs(activationSource, tokenIndices, tokenLabels);
    const resolvedTokens = visualTokens || resolveView2dVisualTokens();

    const embeddingModule = buildEmbeddingModule({
        tokenRefs,
        activationSource
    });

    const layerModules = Array.from({ length: resolvedLayerCount }, (_, layerIndex) => buildLayerGroup({
        layerIndex,
        tokenRefs,
        activationSource
    }));

    const finalLayerNormModule = buildFinalLayerNormModule({
        tokenRefs,
        activationSource
    });

    const logitsModule = buildLogitsModule({
        tokenRefs,
        activationSource
    });

    const connectors = [];
    if (layerModules.length) {
        connectors.push(createFlowConnector(
            embeddingModule.exitNode,
            layerModules[0].entryNode,
            'embedding-layer-0'
        ));
    }
    layerModules.forEach((layerModule, layerIndex) => {
        layerModule.flow.forEach(({ from, to, key }) => {
            connectors.push(createFlowConnector(from, to, key));
        });
        const nextLayerModule = layerModules[layerIndex + 1];
        if (nextLayerModule) {
            connectors.push(createFlowConnector(
                layerModule.exitNode,
                nextLayerModule.entryNode,
                `layer-${layerIndex}-to-layer-${layerIndex + 1}`
            ));
        } else {
            connectors.push(createFlowConnector(
                layerModule.exitNode,
                finalLayerNormModule,
                `layer-${layerIndex}-to-final-ln`
            ));
        }
    });
    connectors.push(createFlowConnector(
        finalLayerNormModule,
        logitsModule.node,
        'final-ln-to-logits'
    ));

    const rootNodes = [
        embeddingModule.node,
        ...layerModules.map((module) => module.node),
        finalLayerNormModule,
        logitsModule.node,
        createGroupNode({
            role: 'connector-layer',
            semantic: {
                componentKind: 'transformer',
                stage: 'connector-layer',
                role: 'connector-layer'
            },
            direction: VIEW2D_LAYOUT_DIRECTIONS.OVERLAY,
            gapKey: 'default',
            children: connectors.filter(Boolean)
        })
    ];

    return createSceneModel({
        semantic: {
            componentKind: 'transformer',
            role: 'scene'
        },
        nodes: rootNodes,
        metadata: {
            visualContract: 'transformer-overview-v1',
            source: 'buildTransformerSceneModel',
            layerCount: resolvedLayerCount,
            tokenCount: tokenRefs.length,
            tokenIndices: tokenRefs.map((tokenRef) => tokenRef.tokenIndex),
            isSmallScreen: !!isSmallScreen,
            tokens: resolvedTokens,
            focusBuilder: 'buildMhsaSceneModel'
        }
    });
}
