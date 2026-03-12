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
const SUMMARY_RESIDUAL_COLS = 12;
const RESIDUAL_STRIP_UNIT = 6;
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

function buildValueColors(values = [], {
    clampMax = RESIDUAL_COLOR_CLAMP,
    rangeOptions = null
} = {}) {
    const safeValues = cleanNumberArray(values);
    if (!safeValues.length) return [];
    return safeValues.map((value) => colorToCss(
        rangeOptions
            ? mapValueToHueRange(value, rangeOptions)
            : mapValueToColor(value, { clampMax })
    ));
}

function buildGradientCss(values = [], {
    clampMax = RESIDUAL_COLOR_CLAMP,
    direction = '90deg',
    rangeOptions = null
} = {}) {
    const fillColors = buildValueColors(values, {
        clampMax,
        rangeOptions
    });
    if (!fillColors.length) return 'none';
    const stops = fillColors.map((fillColor, index) => {
        const ratio = fillColors.length > 1 ? index / (fillColors.length - 1) : 0;
        return `${fillColor} ${(ratio * 100).toFixed(4)}%`;
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
        const fillColors = buildValueColors(values, {
            clampMax,
            rangeOptions
        });
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
            fillColors,
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

function mergeMetadata(...parts) {
    const merged = parts.reduce((acc, part) => {
        if (!part || typeof part !== 'object') return acc;
        return {
            ...acc,
            ...part
        };
    }, {});
    return Object.keys(merged).length ? merged : null;
}

function createCardMetadata(width = null, height = null, {
    hidden = false,
    cornerRadius = null
} = {}) {
    const card = {};
    if (Number.isFinite(width) && width > 0) card.width = Math.floor(width);
    if (Number.isFinite(height) && height > 0) card.height = Math.floor(height);
    if (Number.isFinite(cornerRadius) && cornerRadius >= 0) card.cornerRadius = Math.floor(cornerRadius);
    const metadata = Object.keys(card).length ? { card } : {};
    if (hidden) metadata.hidden = true;
    return Object.keys(metadata).length ? metadata : null;
}

function createCompactRowsMetadata({
    compactWidth = null,
    rowHeight = null,
    rowGap = null,
    paddingX = null,
    paddingY = null,
    variant = '',
    hideSurface = false,
    collapsedBinCount = null
} = {}) {
    const compactRows = {};
    if (Number.isFinite(compactWidth) && compactWidth > 0) compactRows.compactWidth = Math.floor(compactWidth);
    if (Number.isFinite(rowHeight) && rowHeight > 0) compactRows.rowHeight = Math.floor(rowHeight);
    if (Number.isFinite(rowGap) && rowGap >= 0) compactRows.rowGap = Math.floor(rowGap);
    if (Number.isFinite(paddingX) && paddingX >= 0) compactRows.paddingX = Math.floor(paddingX);
    if (Number.isFinite(paddingY) && paddingY >= 0) compactRows.paddingY = Math.floor(paddingY);
    if (typeof variant === 'string' && variant.trim().length) compactRows.variant = variant.trim();
    if (hideSurface) compactRows.hideSurface = true;
    if (Number.isFinite(collapsedBinCount) && collapsedBinCount > 0) {
        compactRows.collapsedBinCount = Math.floor(collapsedBinCount);
    }
    return Object.keys(compactRows).length ? { compactRows } : null;
}

function createModuleCardGroup({
    semantic = {},
    title = '',
    styleKey = VIEW2D_STYLE_KEYS.RESIDUAL,
    cardRole = 'module-card',
    cardWidth = 112,
    cardHeight = 96,
    dimensions = { rows: 1, cols: 1 },
    cardCornerRadius = null,
    textStyleKey = VIEW2D_STYLE_KEYS.LABEL
} = {}) {
    const cardNode = createMatrixNode({
        role: cardRole,
        semantic: buildSemantic(semantic, { role: cardRole }),
        dimensions,
        presentation: VIEW2D_MATRIX_PRESENTATIONS.CARD,
        shape: VIEW2D_MATRIX_SHAPES.MATRIX,
        visual: { styleKey },
        metadata: mergeMetadata(
            createCardMetadata(cardWidth, cardHeight, {
                cornerRadius: cardCornerRadius
            })
        )
    });

    return {
        node: createGroupNode({
            role: semantic.role || 'module',
            semantic,
            direction: VIEW2D_LAYOUT_DIRECTIONS.OVERLAY,
            gapKey: 'default',
            children: [
                cardNode,
                createTextNode({
                    role: 'module-title',
                    semantic: buildSemantic(semantic, { role: 'module-title' }),
                    text: title,
                    presentation: VIEW2D_TEXT_PRESENTATIONS.LABEL,
                    visual: { styleKey: textStyleKey }
                })
            ]
        }),
        cardNode
    };
}

function createHiddenSpacer(width = 1, height = 1, semantic = {}) {
    return createMatrixNode({
        role: 'layout-spacer',
        semantic: buildSemantic(semantic, { role: 'layout-spacer' }),
        dimensions: {
            rows: 1,
            cols: 1
        },
        presentation: VIEW2D_MATRIX_PRESENTATIONS.CARD,
        shape: VIEW2D_MATRIX_SHAPES.MATRIX,
        visual: {
            styleKey: VIEW2D_STYLE_KEYS.RESIDUAL
        },
        metadata: createCardMetadata(width, height, {
            hidden: true,
            cornerRadius: 0
        })
    });
}

function createFixedWidthColumn({
    semantic = {},
    role = 'layout-column',
    width = 100,
    height = 24,
    align = 'center',
    child = null
} = {}) {
    const spacer = createHiddenSpacer(width, height, semantic);
    return createGroupNode({
        role,
        semantic,
        direction: VIEW2D_LAYOUT_DIRECTIONS.OVERLAY,
        gapKey: 'default',
        align,
        children: child ? [spacer, child] : [spacer]
    });
}

function createResidualStateModule({
    semantic = {},
    title = '',
    tokenRefs = [],
    getVector = () => null
} = {}) {
    const rowItems = buildVectorRowItems(tokenRefs, {
        baseSemantic: semantic,
        role: 'residual-row',
        measureCols: SUMMARY_RESIDUAL_COLS,
        getVector
    });
    const rowCount = Math.max(1, rowItems.length);
    const cardNode = createMatrixNode({
        role: 'module-card',
        semantic: buildSemantic(semantic, { role: 'module-card' }),
        dimensions: {
            rows: rowCount,
            cols: D_MODEL
        },
        presentation: VIEW2D_MATRIX_PRESENTATIONS.COMPACT_ROWS,
        shape: VIEW2D_MATRIX_SHAPES.MATRIX,
        rowItems,
        visual: { styleKey: VIEW2D_STYLE_KEYS.RESIDUAL },
        metadata: mergeMetadata(
            createMeasureMetadata(SUMMARY_RESIDUAL_COLS, rowCount),
            createCompactRowsMetadata({
                compactWidth: SUMMARY_RESIDUAL_COLS * RESIDUAL_STRIP_UNIT,
                rowHeight: RESIDUAL_STRIP_UNIT,
                rowGap: 2,
                paddingX: 0,
                paddingY: 4,
                variant: 'vector-strip',
                hideSurface: true
            }),
            createCardMetadata(null, null, {
                cornerRadius: 6
            })
        )
    });

    return {
        node: createGroupNode({
            role: semantic.role || 'module',
            semantic,
            direction: VIEW2D_LAYOUT_DIRECTIONS.OVERLAY,
            gapKey: 'default',
            children: [
                cardNode,
                createTextNode({
                    role: 'module-title',
                    semantic: buildSemantic(semantic, { role: 'module-title' }),
                    text: title,
                    presentation: VIEW2D_TEXT_PRESENTATIONS.LABEL,
                    visual: { styleKey: VIEW2D_STYLE_KEYS.LABEL }
                })
            ]
        }),
        cardNode
    };
}

function createTopAddNode({
    semantic = {}
} = {}) {
    const cardNode = createMatrixNode({
        role: 'add-circle',
        semantic: buildSemantic(semantic, { role: 'add-circle' }),
        dimensions: {
            rows: 1,
            cols: 1
        },
        presentation: VIEW2D_MATRIX_PRESENTATIONS.CARD,
        shape: VIEW2D_MATRIX_SHAPES.MATRIX,
        visual: { styleKey: VIEW2D_STYLE_KEYS.RESIDUAL_ADD },
        metadata: createCardMetadata(28, 28, {
            cornerRadius: 999
        })
    });
    return {
        node: createGroupNode({
            role: semantic.role || 'module',
            semantic,
            direction: VIEW2D_LAYOUT_DIRECTIONS.OVERLAY,
            gapKey: 'default',
            children: [
                cardNode,
                createOperatorNode({
                    role: 'residual-add-operator',
                    semantic: buildSemantic(semantic, { role: 'residual-add-operator', operatorKey: 'plus' }),
                    text: '+',
                    visual: { styleKey: VIEW2D_STYLE_KEYS.RESIDUAL_ADD_SYMBOL }
                })
            ]
        }),
        cardNode
    };
}

function buildLayerNormModule({
    layerIndex = null,
    stage = '',
    title = ''
} = {}) {
    return createModuleCardGroup({
        semantic: {
            componentKind: 'layer-norm',
            layerIndex,
            stage,
            role: 'module'
        },
        title,
        styleKey: VIEW2D_STYLE_KEYS.LAYER_NORM,
        textStyleKey: VIEW2D_STYLE_KEYS.LABEL_DARK,
        cardWidth: 92,
        cardHeight: 42,
        cardCornerRadius: 999,
        dimensions: {
            rows: 1,
            cols: D_MODEL
        }
    });
}

function buildEmbeddingCardGroup({
    role = '',
    stage = '',
    title = '',
    styleKey = VIEW2D_STYLE_KEYS.EMBEDDING_TOKEN
} = {}) {
    return createModuleCardGroup({
        semantic: {
            componentKind: 'embedding',
            stage,
            role
        },
        title,
        styleKey,
        cardWidth: 92,
        cardHeight: 72,
        dimensions: {
            rows: CONTEXT_LEN,
            cols: D_MODEL
        }
    });
}

function buildMhsaModule({
    layerIndex = 0
} = {}) {
    const HEAD_CARD_WIDTH = 136;
    const HEAD_CARD_HEIGHT = 52;
    const HEAD_STACK_ALIGNMENT_OFFSET = 4;
    const baseSemantic = {
        componentKind: 'mhsa',
        layerIndex,
        stage: 'attention',
        role: 'module'
    };

    const headEntries = Array.from({ length: NUM_HEAD_SETS_LAYER }, (_, headIndex) => {
        const headSemantic = {
            componentKind: 'mhsa',
            layerIndex,
            headIndex,
            stage: 'attention',
            role: 'head'
        };
        const headCard = createMatrixNode({
            role: 'head-card',
            semantic: buildSemantic(headSemantic, { role: 'head-card' }),
            dimensions: {
                rows: 1,
                cols: D_HEAD
            },
            presentation: VIEW2D_MATRIX_PRESENTATIONS.CARD,
            shape: VIEW2D_MATRIX_SHAPES.MATRIX,
            visual: {
                styleKey: VIEW2D_STYLE_KEYS.MHSA_HEAD
            },
            metadata: createCardMetadata(HEAD_CARD_WIDTH, HEAD_CARD_HEIGHT, {
                cornerRadius: 12
            })
        });
        const labelNode = createTextNode({
            role: 'head-label',
            semantic: buildSemantic(headSemantic, { role: 'head-label' }),
            text: `Head ${headIndex + 1}`,
            presentation: VIEW2D_TEXT_PRESENTATIONS.LABEL,
            visual: { styleKey: VIEW2D_STYLE_KEYS.LABEL }
        });
        const headNode = createGroupNode({
            role: 'head',
            semantic: headSemantic,
            direction: VIEW2D_LAYOUT_DIRECTIONS.OVERLAY,
            gapKey: 'default',
            children: [
                headCard,
                labelNode
            ]
        });
        return {
            node: headNode,
            cardNode: headCard
        };
    });

    const node = createGroupNode({
        role: 'module',
        semantic: baseSemantic,
        direction: VIEW2D_LAYOUT_DIRECTIONS.VERTICAL,
        gapKey: 'default',
        children: [
            createGroupNode({
                role: 'head-stack',
                semantic: buildSemantic(baseSemantic, { role: 'head-stack' }),
                direction: VIEW2D_LAYOUT_DIRECTIONS.VERTICAL,
                gapKey: 'default',
                children: [
                    createHiddenSpacer(HEAD_CARD_WIDTH, HEAD_STACK_ALIGNMENT_OFFSET, buildSemantic(baseSemantic, {
                        stage: 'head-stack-offset',
                        role: 'head-stack-offset'
                    })),
                    ...headEntries.map((entry) => entry.node)
                ]
            })
        ]
    });

    return {
        node,
        headCardNodes: headEntries.map((entry) => entry.cardNode)
    };
}

function buildOutputProjectionModule({
    layerIndex = 0
} = {}) {
    return createModuleCardGroup({
        semantic: {
            componentKind: 'output-projection',
            layerIndex,
            stage: 'attn-out',
            role: 'module'
        },
        title: 'Out proj',
        styleKey: VIEW2D_STYLE_KEYS.OUTPUT_PROJECTION,
        cardRole: 'projection-weight',
        cardWidth: 110,
        cardHeight: 86,
        cardCornerRadius: 14,
        dimensions: {
            rows: D_MODEL,
            cols: D_MODEL
        }
    });
}

function buildResidualAddModule({
    layerIndex = 0,
    stage = ''
} = {}) {
    return createTopAddNode({
        semantic: {
            componentKind: 'residual',
            layerIndex,
            stage,
            role: 'module'
        }
    });
}

function buildMlpModule({
    layerIndex = 0
} = {}) {
    return createModuleCardGroup({
        semantic: {
            componentKind: 'mlp',
            layerIndex,
            stage: 'mlp',
            role: 'module'
        },
        title: 'MLP',
        styleKey: VIEW2D_STYLE_KEYS.MLP,
        cardWidth: 148,
        cardHeight: 92,
        cardCornerRadius: 14,
        dimensions: {
            rows: D_MODEL,
            cols: D_MODEL * 4
        }
    });
}

function buildLayerGroup({
    layerIndex = 0,
    activationSource = null,
    tokenRefs = []
} = {}) {
    const layerSemantic = {
        componentKind: 'layer',
        layerIndex
    };

    const COLUMN_WIDTHS = {
        residual: 136,
        heads: 200,
        outProj: 152,
        addIn: 80,
        ln2: 148,
        mlp: 196,
        addOut: 80
    };
    const TOP_ROW_HEIGHT = 58;

    const incomingResidual = createResidualStateModule({
        semantic: {
            componentKind: 'residual',
            layerIndex,
            stage: 'incoming',
            role: 'module'
        },
        title: 'x_in',
        tokenRefs,
        getVector: (tokenRef) => (
            typeof activationSource?.getLayerIncoming === 'function'
                ? activationSource.getLayerIncoming(layerIndex, tokenRef.tokenIndex, D_MODEL)
                : null
        )
    });
    const ln1Module = buildLayerNormModule({
        layerIndex,
        stage: 'ln1',
        title: 'LN1'
    });
    const mhsaModule = buildMhsaModule({
        layerIndex
    });
    const outputProjectionModule = buildOutputProjectionModule({
        layerIndex
    });
    const postAttentionAdd = buildResidualAddModule({
        componentKind: 'residual',
        layerIndex,
        stage: 'post-attn-add'
    });
    const postAttentionResidual = createResidualStateModule({
        semantic: {
            componentKind: 'residual',
            layerIndex,
            stage: 'post-attn-residual',
            role: 'module'
        },
        title: 'x_attn',
        tokenRefs,
        getVector: (tokenRef) => (
            typeof activationSource?.getPostAttentionResidual === 'function'
                ? activationSource.getPostAttentionResidual(layerIndex, tokenRef.tokenIndex, D_MODEL)
                : null
        )
    });
    const ln2Module = buildLayerNormModule({
        layerIndex,
        stage: 'ln2',
        title: 'LN2'
    });
    const mlpModule = buildMlpModule({
        layerIndex
    });
    const postMlpAdd = buildResidualAddModule({
        layerIndex,
        stage: 'post-mlp-add'
    });

    const topRow = createGroupNode({
        role: 'layer-top-row',
        semantic: buildSemantic(layerSemantic, { role: 'layer-top-row' }),
        direction: VIEW2D_LAYOUT_DIRECTIONS.HORIZONTAL,
        gapKey: 'stage',
        children: [
            createFixedWidthColumn({
                semantic: buildSemantic(layerSemantic, { stage: 'top-x-in', role: 'top-x-in' }),
                width: COLUMN_WIDTHS.residual,
                height: TOP_ROW_HEIGHT,
                child: incomingResidual.node
            }),
            createFixedWidthColumn({
                semantic: buildSemantic(layerSemantic, { stage: 'top-head-space', role: 'top-head-space' }),
                width: COLUMN_WIDTHS.heads,
                height: TOP_ROW_HEIGHT
            }),
            createFixedWidthColumn({
                semantic: buildSemantic(layerSemantic, { stage: 'top-outproj-space', role: 'top-outproj-space' }),
                width: COLUMN_WIDTHS.outProj,
                height: TOP_ROW_HEIGHT
            }),
            createFixedWidthColumn({
                semantic: buildSemantic(layerSemantic, { stage: 'top-post-attn-add', role: 'top-post-attn-add' }),
                width: COLUMN_WIDTHS.addIn,
                height: TOP_ROW_HEIGHT,
                child: postAttentionAdd.node
            }),
            createFixedWidthColumn({
                semantic: buildSemantic(layerSemantic, { stage: 'top-post-attn-residual', role: 'top-post-attn-residual' }),
                width: COLUMN_WIDTHS.ln2,
                height: TOP_ROW_HEIGHT,
                child: postAttentionResidual.node
            }),
            createFixedWidthColumn({
                semantic: buildSemantic(layerSemantic, { stage: 'top-mlp-space', role: 'top-mlp-space' }),
                width: COLUMN_WIDTHS.mlp,
                height: TOP_ROW_HEIGHT
            }),
            createFixedWidthColumn({
                semantic: buildSemantic(layerSemantic, { stage: 'top-post-mlp-add', role: 'top-post-mlp-add' }),
                width: COLUMN_WIDTHS.addOut,
                height: TOP_ROW_HEIGHT,
                child: postMlpAdd.node
            })
        ]
    });

    const bottomRow = createGroupNode({
        role: 'layer-bottom-row',
        semantic: buildSemantic(layerSemantic, { role: 'layer-bottom-row' }),
        direction: VIEW2D_LAYOUT_DIRECTIONS.HORIZONTAL,
        gapKey: 'stage',
        align: 'start',
        children: [
            createFixedWidthColumn({
                semantic: buildSemantic(layerSemantic, { stage: 'bottom-ln1', role: 'bottom-ln1' }),
                width: COLUMN_WIDTHS.residual,
                height: 96,
                align: 'center',
                child: ln1Module.node
            }),
            createFixedWidthColumn({
                semantic: buildSemantic(layerSemantic, { stage: 'bottom-heads', role: 'bottom-heads' }),
                width: COLUMN_WIDTHS.heads,
                height: 360,
                align: 'top',
                child: mhsaModule.node
            }),
            createFixedWidthColumn({
                semantic: buildSemantic(layerSemantic, { stage: 'bottom-outproj', role: 'bottom-outproj' }),
                width: COLUMN_WIDTHS.outProj,
                height: 96,
                align: 'center',
                child: outputProjectionModule.node
            }),
            createFixedWidthColumn({
                semantic: buildSemantic(layerSemantic, { stage: 'bottom-post-attn-add-space', role: 'bottom-post-attn-add-space' }),
                width: COLUMN_WIDTHS.addIn,
                height: 84
            }),
            createFixedWidthColumn({
                semantic: buildSemantic(layerSemantic, { stage: 'bottom-ln2', role: 'bottom-ln2' }),
                width: COLUMN_WIDTHS.ln2,
                height: 96,
                align: 'center',
                child: ln2Module.node
            }),
            createFixedWidthColumn({
                semantic: buildSemantic(layerSemantic, { stage: 'bottom-mlp', role: 'bottom-mlp' }),
                width: COLUMN_WIDTHS.mlp,
                height: 96,
                align: 'center',
                child: mlpModule.node
            }),
            createFixedWidthColumn({
                semantic: buildSemantic(layerSemantic, { stage: 'bottom-add-space', role: 'bottom-add-space' }),
                width: COLUMN_WIDTHS.addOut,
                height: 84
            })
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
            topRow,
            bottomRow
        ]
    });

    const flow = [
        {
            from: incomingResidual.cardNode,
            to: postAttentionAdd.cardNode,
            key: `layer-${layerIndex}-residual-top-post-attn-add`
        },
        {
            from: postAttentionAdd.cardNode,
            to: postAttentionResidual.cardNode,
            key: `layer-${layerIndex}-residual-top-post-attn-residual`
        },
        {
            from: postAttentionResidual.cardNode,
            to: postMlpAdd.cardNode,
            key: `layer-${layerIndex}-residual-top-post-mlp-add`
        },
        {
            from: incomingResidual.cardNode,
            to: ln1Module.cardNode,
            key: `layer-${layerIndex}-incoming-ln1`,
            sourceAnchor: VIEW2D_ANCHOR_SIDES.BOTTOM,
            targetAnchor: VIEW2D_ANCHOR_SIDES.TOP,
            route: VIEW2D_CONNECTOR_ROUTES.VERTICAL
        },
        ...mhsaModule.headCardNodes.map((headCardNode, headIndex) => ({
            from: ln1Module.cardNode,
            to: headCardNode,
            key: `layer-${layerIndex}-ln1-head-${headIndex}`
        })),
        ...mhsaModule.headCardNodes.map((headCardNode, headIndex) => ({
            from: headCardNode,
            to: outputProjectionModule.cardNode,
            key: `layer-${layerIndex}-head-${headIndex}-outproj`
        })),
        {
            from: outputProjectionModule.cardNode,
            to: postAttentionAdd.cardNode,
            key: `layer-${layerIndex}-outproj-add1`,
            sourceAnchor: VIEW2D_ANCHOR_SIDES.RIGHT,
            targetAnchor: VIEW2D_ANCHOR_SIDES.BOTTOM,
            route: VIEW2D_CONNECTOR_ROUTES.ELBOW
        },
        {
            from: postAttentionResidual.cardNode,
            to: ln2Module.cardNode,
            key: `layer-${layerIndex}-add1-ln2`,
            sourceAnchor: VIEW2D_ANCHOR_SIDES.BOTTOM,
            targetAnchor: VIEW2D_ANCHOR_SIDES.TOP,
            route: VIEW2D_CONNECTOR_ROUTES.VERTICAL,
            gap: 12
        },
        {
            from: ln2Module.cardNode,
            to: mlpModule.cardNode,
            key: `layer-${layerIndex}-ln2-mlp`
        },
        {
            from: mlpModule.cardNode,
            to: postMlpAdd.cardNode,
            key: `layer-${layerIndex}-mlp-add2`,
            sourceAnchor: VIEW2D_ANCHOR_SIDES.RIGHT,
            targetAnchor: VIEW2D_ANCHOR_SIDES.BOTTOM,
            route: VIEW2D_CONNECTOR_ROUTES.ELBOW
        }
    ];

    return {
        node,
        entryNode: incomingResidual.cardNode,
        exitNode: postMlpAdd.cardNode,
        flow
    };
}

function buildFinalLayerNormModule() {
    return buildLayerNormModule({
        stage: 'final-ln',
        title: 'Final LN'
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

function createFlowConnector({
    fromNode = null,
    toNode = null,
    connectorKey = '',
    styleKey = VIEW2D_STYLE_KEYS.CONNECTOR_POST,
    sourceAnchor = VIEW2D_ANCHOR_SIDES.RIGHT,
    targetAnchor = VIEW2D_ANCHOR_SIDES.LEFT,
    route = VIEW2D_CONNECTOR_ROUTES.HORIZONTAL,
    gap = 10
} = {}) {
    if (!fromNode?.id || !toNode?.id) return null;
    return createConnectorNode({
        role: `connector-${connectorKey}`,
        semantic: {
            componentKind: 'transformer',
            stage: `connector-${connectorKey}`,
            role: `connector-${connectorKey}`
        },
        source: createAnchorRef(fromNode.id, sourceAnchor),
        target: createAnchorRef(toNode.id, targetAnchor),
        route,
        gap,
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

    const layerModules = Array.from({ length: resolvedLayerCount }, (_, layerIndex) => buildLayerGroup({
        layerIndex,
        activationSource,
        tokenRefs
    }));

    const connectors = [];
    layerModules.forEach((layerModule, layerIndex) => {
        layerModule.flow.forEach(({ from, to, key, sourceAnchor, targetAnchor, route, styleKey, gap }) => {
            connectors.push(createFlowConnector({
                fromNode: from,
                toNode: to,
                connectorKey: key,
                sourceAnchor,
                targetAnchor,
                route,
                styleKey,
                gap
            }));
        });
        const nextLayerModule = layerModules[layerIndex + 1];
        if (nextLayerModule) {
            connectors.push(createFlowConnector({
                fromNode: layerModule.exitNode,
                toNode: nextLayerModule.entryNode,
                connectorKey: `layer-${layerIndex}-to-layer-${layerIndex + 1}`
            }));
        }
    });

    const rootNodes = [
        ...layerModules.map((module) => module.node),
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
