import { getAttentionBiasHeadSample } from '../data/biasParams.js';
import {
    MHA_FINAL_K_COLOR,
    MHA_FINAL_Q_COLOR,
    MHA_FINAL_V_COLOR,
    MHA_VALUE_SPECTRUM_COLOR,
    MHA_VALUE_HUE_SPREAD,
    MHA_VALUE_LIGHTNESS_MIN,
    MHA_VALUE_LIGHTNESS_MAX,
    MHA_VALUE_RANGE_MIN,
    MHA_VALUE_RANGE_MAX,
    MHA_VALUE_CLAMP_MAX
} from '../animations/LayerAnimationConstants.js';
import {
    buildHueRangeOptions,
    mapAttentionPostScoreToColor,
    mapValueToColor,
    mapValueToHueRange
} from '../utils/colors.js';
import {
    ATTENTION_PRE_COLOR_CLAMP,
    ATTENTION_PREVIEW_COLOR_DARKEN_FACTOR,
    D_HEAD,
    D_MODEL,
    RESIDUAL_COLOR_CLAMP
} from './selectionPanelConstants.js';
import { formatTokenLabelForPreview } from './selectionPanelFormatUtils.js';

const DEFAULT_SAMPLE_STEP = 64;
const DEFAULT_LAYER_INDEX = 5;
const DEFAULT_HEAD_INDEX = 5;
const DEFAULT_PROMPT_ROW_COUNT = 5;
const PROJECTION_VECTOR_PREVIEW_DARKEN_FACTOR = 0.98;
const QUERY_VECTOR_GRADIENT_OPTIONS = buildHueRangeOptions(MHA_FINAL_Q_COLOR, {
    hueSpread: MHA_VALUE_HUE_SPREAD,
    minLightness: MHA_VALUE_LIGHTNESS_MIN,
    maxLightness: MHA_VALUE_LIGHTNESS_MAX,
    valueMin: MHA_VALUE_RANGE_MIN,
    valueMax: MHA_VALUE_RANGE_MAX,
    valueClampMax: MHA_VALUE_CLAMP_MAX
});
const KEY_VECTOR_GRADIENT_OPTIONS = buildHueRangeOptions(MHA_FINAL_K_COLOR, {
    hueSpread: MHA_VALUE_HUE_SPREAD,
    minLightness: MHA_VALUE_LIGHTNESS_MIN,
    maxLightness: MHA_VALUE_LIGHTNESS_MAX,
    valueMin: MHA_VALUE_RANGE_MIN,
    valueMax: MHA_VALUE_RANGE_MAX,
    valueClampMax: MHA_VALUE_CLAMP_MAX
});
const VALUE_VECTOR_GRADIENT_OPTIONS = buildHueRangeOptions(MHA_VALUE_SPECTRUM_COLOR, {
    hueSpread: MHA_VALUE_HUE_SPREAD,
    minLightness: MHA_VALUE_LIGHTNESS_MIN,
    maxLightness: MHA_VALUE_LIGHTNESS_MAX,
    valueMin: MHA_VALUE_RANGE_MIN,
    valueMax: MHA_VALUE_RANGE_MAX,
    valueClampMax: MHA_VALUE_CLAMP_MAX
});
const PROJECTION_CONFIGS = [
    {
        kind: 'q',
        biasKind: 'query',
        weightLabelTex: 'W_q',
        biasLabelTex: 'b_q',
        outputLabelTex: 'Q',
        colorHex: MHA_FINAL_Q_COLOR,
        gradientOptions: QUERY_VECTOR_GRADIENT_OPTIONS
    },
    {
        kind: 'k',
        biasKind: 'key',
        weightLabelTex: 'W_k',
        biasLabelTex: 'b_k',
        outputLabelTex: 'K',
        colorHex: MHA_FINAL_K_COLOR,
        gradientOptions: KEY_VECTOR_GRADIENT_OPTIONS
    },
    {
        kind: 'v',
        biasKind: 'value',
        weightLabelTex: 'W_v',
        biasLabelTex: 'b_v',
        outputLabelTex: 'V',
        colorHex: MHA_FINAL_V_COLOR,
        gradientOptions: VALUE_VECTOR_GRADIENT_OPTIONS
    }
];

function colorToCss(color) {
    return color?.isColor ? `#${color.getHexString()}` : 'transparent';
}

function darkenColor(color, factor = 1) {
    if (!color?.isColor) return color;
    const safeFactor = Number.isFinite(factor) ? Math.max(0, Math.min(1, factor)) : 1;
    return color.clone().multiplyScalar(safeFactor);
}

function rgbToCss(rgb, alpha = 1) {
    const [r, g, b] = Array.isArray(rgb) && rgb.length === 3
        ? rgb
        : [255, 255, 255];
    return `rgba(${r}, ${g}, ${b}, ${alpha.toFixed(3)})`;
}

function hexToRgb(hex) {
    const value = Number(hex);
    const safe = Number.isFinite(value) ? Math.max(0, Math.min(0xFFFFFF, Math.floor(value))) : 0xFFFFFF;
    return [
        (safe >> 16) & 0xFF,
        (safe >> 8) & 0xFF,
        safe & 0xFF
    ];
}

function mixRgb(rgb, targetRgb, amount = 0.5) {
    const safeAmount = Number.isFinite(amount) ? Math.min(1, Math.max(0, amount)) : 0.5;
    return rgb.map((channel, index) => {
        const target = Array.isArray(targetRgb) ? Number(targetRgb[index]) : 255;
        return Math.round(channel + ((target - channel) * safeAmount));
    });
}

function normalizeTokenLabel(label, index) {
    if (typeof label !== 'string' || label.length <= 0) {
        return `Token ${index + 1}`;
    }
    return formatTokenLabelForPreview(label);
}

function cleanNumberArray(values, fallbackLength = 0) {
    if (!Array.isArray(values) || !values.length) {
        return fallbackLength > 0 ? new Array(fallbackLength).fill(0) : [];
    }
    return values.map((value) => (Number.isFinite(value) ? value : 0));
}

function buildGradientCssFromSamples(values, direction = '90deg') {
    const safeValues = cleanNumberArray(values);
    if (!safeValues.length) return 'none';
    if (safeValues.length === 1) {
        const color = colorToCss(mapValueToColor(safeValues[0], { clampMax: RESIDUAL_COLOR_CLAMP }));
        return `linear-gradient(${direction}, ${color} 0%, ${color} 100%)`;
    }
    const lastIndex = Math.max(1, safeValues.length - 1);
    const stops = safeValues.map((value, index) => {
        const percent = (index / lastIndex) * 100;
        return `${colorToCss(mapValueToColor(value, { clampMax: RESIDUAL_COLOR_CLAMP }))} ${percent.toFixed(4)}%`;
    });
    return `linear-gradient(${direction}, ${stops.join(', ')})`;
}

function buildHueGradientCssFromSamples(values, rangeOptions, direction = '90deg', darkenFactor = 1) {
    const safeValues = cleanNumberArray(values);
    if (!safeValues.length) return 'none';
    if (safeValues.length === 1) {
        return colorToCss(darkenColor(mapValueToHueRange(safeValues[0], rangeOptions), darkenFactor));
    }
    const lastIndex = Math.max(1, safeValues.length - 1);
    const stops = safeValues.map((value, index) => {
        const percent = (index / lastIndex) * 100;
        return `${colorToCss(darkenColor(mapValueToHueRange(value, rangeOptions), darkenFactor))} ${percent.toFixed(4)}%`;
    });
    return `linear-gradient(${direction}, ${stops.join(', ')})`;
}

function buildValueVectorGradientCss(values, direction = '90deg') {
    return buildHueGradientCssFromSamples(values, VALUE_VECTOR_GRADIENT_OPTIONS, direction);
}

function buildProjectionVectorGradientCss(values, scalarValue, rangeOptions, direction = '90deg', darkenFactor = 1) {
    if (Array.isArray(values) && values.length) {
        return buildHueGradientCssFromSamples(values, rangeOptions, direction, darkenFactor);
    }
    return colorToCss(
        darkenColor(
            mapValueToHueRange(Number.isFinite(scalarValue) ? scalarValue : 0, rangeOptions),
            darkenFactor
        )
    );
}

function buildAccentCss(rgb, value = 0) {
    const intensity = Math.min(1, Math.abs(Number(value) || 0) / RESIDUAL_COLOR_CLAMP);
    const base = Array.isArray(rgb) ? rgb : [255, 255, 255];
    const highlight = mixRgb(base, [255, 255, 255], 0.22);
    const core = mixRgb(base, [255, 255, 255], 0.06);
    const depth = mixRgb(base, [10, 14, 20], 0.16);
    const alphaStart = 0.72 + (intensity * 0.12);
    const alphaMid = 0.88 + (intensity * 0.08);
    const alphaEnd = 0.82 + (intensity * 0.12);
    return `linear-gradient(92deg, ${rgbToCss(highlight, alphaStart)} 0%, ${rgbToCss(core, alphaMid)} 52%, ${rgbToCss(depth, alphaEnd)} 100%)`;
}

function buildWeightCardCss(rgb) {
    const base = Array.isArray(rgb) ? rgb : [255, 255, 255];
    const top = mixRgb(base, [255, 255, 255], 0.28);
    const upperMid = mixRgb(base, [255, 255, 255], 0.14);
    const lowerMid = mixRgb(base, [255, 255, 255], 0.04);
    const bottom = mixRgb(base, [8, 12, 18], 0.18);
    return `linear-gradient(162deg, ${rgbToCss(top, 0.88)} 0%, ${rgbToCss(upperMid, 0.96)} 34%, ${rgbToCss(lowerMid, 0.98)} 64%, ${rgbToCss(bottom, 0.92)} 100%)`;
}

function buildAttentionScoreCellCss(value) {
    const color = mapValueToColor(value, { clampMax: ATTENTION_PRE_COLOR_CLAMP });
    if (color?.isColor && typeof color.clone === 'function') {
        return colorToCss(color.clone().multiplyScalar(ATTENTION_PREVIEW_COLOR_DARKEN_FACTOR));
    }
    return colorToCss(color);
}

function buildAttentionPostCellCss(value) {
    return colorToCss(mapAttentionPostScoreToColor(value));
}

function buildMaskCellCss(isMasked = false) {
    if (isMasked) return 'rgba(0, 0, 0, 0.94)';
    return buildAttentionScoreCellCss(0);
}

function resolvePreviewTokenIndices(activationSource, tokenIndices = null) {
    if (Array.isArray(tokenIndices) && tokenIndices.length) {
        return tokenIndices
            .map((value) => Number(value))
            .filter(Number.isFinite)
            .map((value) => Math.max(0, Math.floor(value)));
    }
    const promptTokens = Array.isArray(activationSource?.meta?.prompt_tokens)
        ? activationSource.meta.prompt_tokens
        : [];
    if (promptTokens.length) {
        return Array.from({ length: promptTokens.length }, (_, idx) => idx);
    }
    const tokenCount = typeof activationSource?.getTokenCount === 'function'
        ? activationSource.getTokenCount()
        : 0;
    if (Number.isFinite(tokenCount) && tokenCount > 0) {
        const rowCount = Math.floor(tokenCount);
        return Array.from({ length: rowCount }, (_, idx) => idx);
    }
    return Array.from({ length: DEFAULT_PROMPT_ROW_COUNT }, (_, idx) => idx);
}

function resolveTokenLabel(activationSource, tokenIndex, fallbackLabel = null, fallbackIndex = null) {
    if (typeof fallbackLabel === 'string' && fallbackLabel.trim().length > 0) {
        return normalizeTokenLabel(fallbackLabel, Number.isFinite(fallbackIndex) ? fallbackIndex : tokenIndex);
    }
    const tokenDisplayStrings = Array.isArray(activationSource?.meta?.token_display_strings)
        ? activationSource.meta.token_display_strings
        : [];
    if (typeof tokenDisplayStrings[tokenIndex] === 'string') {
        return normalizeTokenLabel(
            tokenDisplayStrings[tokenIndex],
            Number.isFinite(fallbackIndex) ? fallbackIndex : tokenIndex
        );
    }
    const label = typeof activationSource?.getTokenString === 'function'
        ? activationSource.getTokenString(tokenIndex)
        : null;
    return normalizeTokenLabel(label, Number.isFinite(fallbackIndex) ? fallbackIndex : tokenIndex);
}

function buildAttentionScoreStage({
    activationSource,
    layerIndex,
    headIndex,
    rows,
    queryProjection,
    keyProjection,
    valueProjection
} = {}) {
    if (
        !activationSource
        || !Array.isArray(rows)
        || !rows.length
        || !queryProjection
        || !keyProjection
    ) {
        return null;
    }

    const valueRows = rows.map((rowData) => {
        const projectionRow = Array.isArray(valueProjection?.outputRows)
            ? valueProjection.outputRows[rowData.rowIndex]
            : null;
        const rawValues = Array.isArray(projectionRow?.rawValues) && projectionRow.rawValues.length
            ? cleanNumberArray(projectionRow.rawValues, D_HEAD)
            : (
                typeof activationSource.getLayerQKVVector === 'function'
                    ? activationSource.getLayerQKVVector(layerIndex, 'v', headIndex, rowData.tokenIndex, D_HEAD)
                    : null
            );
        const safeValues = Array.isArray(rawValues) && rawValues.length
            ? cleanNumberArray(rawValues, D_HEAD)
            : null;
        return {
            rowIndex: rowData.rowIndex,
            tokenLabel: rowData.tokenLabel,
            rawValues: safeValues,
            gradientCss: projectionRow?.gradientCss || (
                Array.isArray(safeValues) && safeValues.length
                    ? buildProjectionVectorGradientCss(
                        safeValues,
                        null,
                        VALUE_VECTOR_GRADIENT_OPTIONS,
                        '90deg',
                        PROJECTION_VECTOR_PREVIEW_DARKEN_FACTOR
                    )
                    : 'none'
            ),
            title: `${rowData.tokenLabel}: value vector`
        };
    });

    const outputRows = rows.map((rowData) => {
        const scoreRow = typeof activationSource.getAttentionScoresRow === 'function'
            ? activationSource.getAttentionScoresRow(layerIndex, 'pre', headIndex, rowData.tokenIndex)
            : null;
        const cells = rows.map((colData) => {
            const rawValue = Array.isArray(scoreRow) ? scoreRow[colData.tokenIndex] : null;
            const safeValue = Number.isFinite(rawValue) ? rawValue : null;
            return {
                rowIndex: rowData.rowIndex,
                colIndex: colData.rowIndex,
                rowTokenLabel: rowData.tokenLabel,
                colTokenLabel: colData.tokenLabel,
                rawValue: safeValue,
                fillCss: Number.isFinite(safeValue) ? buildAttentionScoreCellCss(safeValue) : 'transparent'
            };
        });
        return {
            rowIndex: rowData.rowIndex,
            tokenLabel: rowData.tokenLabel,
            cells,
            hasAnyValue: cells.some((cell) => Number.isFinite(cell.rawValue))
        };
    });

    if (!outputRows.some((rowData) => rowData.hasAnyValue)) {
        return null;
    }

    const maskRows = rows.map((rowData) => ({
        rowIndex: rowData.rowIndex,
        tokenLabel: rowData.tokenLabel,
        cells: rows.map((colData) => {
            const isMasked = colData.rowIndex > rowData.rowIndex;
            return {
                rowIndex: rowData.rowIndex,
                colIndex: colData.rowIndex,
                rowTokenLabel: rowData.tokenLabel,
                colTokenLabel: colData.tokenLabel,
                rawValue: isMasked ? Number.NEGATIVE_INFINITY : 0,
                fillCss: buildMaskCellCss(isMasked),
                isMasked,
                isEmpty: false,
                title: `${rowData.tokenLabel} → ${colData.tokenLabel}: ${isMasked ? '-∞' : '0'}`
            };
        })
    }));

    const postRows = rows.map((rowData) => {
        const scoreRow = typeof activationSource.getAttentionScoresRow === 'function'
            ? activationSource.getAttentionScoresRow(layerIndex, 'post', headIndex, rowData.tokenIndex)
            : null;
        const cells = rows.map((colData) => {
            const isMasked = colData.rowIndex > rowData.rowIndex;
            const rawValue = Array.isArray(scoreRow) ? scoreRow[colData.tokenIndex] : null;
            const safeValue = Number.isFinite(rawValue) ? rawValue : null;
            return {
                rowIndex: rowData.rowIndex,
                colIndex: colData.rowIndex,
                rowTokenLabel: rowData.tokenLabel,
                colTokenLabel: colData.tokenLabel,
                rawValue: isMasked ? null : safeValue,
                fillCss: isMasked
                    ? buildMaskCellCss(true)
                    : (Number.isFinite(safeValue) ? buildAttentionPostCellCss(safeValue) : 'transparent'),
                isMasked,
                isEmpty: !isMasked && !Number.isFinite(safeValue),
                title: isMasked
                    ? `${rowData.tokenLabel} → ${colData.tokenLabel}: masked (softmax → 0)`
                    : undefined
            };
        });
        return {
            rowIndex: rowData.rowIndex,
            tokenLabel: rowData.tokenLabel,
            cells,
            hasAnyValue: cells.some((cell) => !cell.isMasked && Number.isFinite(cell.rawValue))
        };
    });

    const headOutputRows = rows.map((rowData) => {
        const rawValues = typeof activationSource.getAttentionWeightedSum === 'function'
            ? activationSource.getAttentionWeightedSum(layerIndex, headIndex, rowData.tokenIndex, D_HEAD)
            : null;
        const safeValues = Array.isArray(rawValues) && rawValues.length
            ? cleanNumberArray(rawValues, D_HEAD)
            : null;
        return {
            rowIndex: rowData.rowIndex,
            tokenLabel: rowData.tokenLabel,
            rawValues: safeValues,
            gradientCss: Array.isArray(safeValues) && safeValues.length
                ? buildValueVectorGradientCss(safeValues)
                : 'none',
            title: `${rowData.tokenLabel}: attention head output`
        };
    });

    return {
        queryLabelTex: queryProjection.outputLabelTex,
        queryRowCount: queryProjection.outputRowCount,
        queryColumnCount: queryProjection.outputColumnCount,
        queryRows: queryProjection.outputRows,
        transposeLabelTex: 'K^{\\mathsf{T}}',
        transposeRowCount: keyProjection.outputColumnCount,
        transposeColumnCount: keyProjection.outputRowCount,
        transposeColumns: keyProjection.outputRows.map((rowData) => ({
            colIndex: rowData.rowIndex,
            rawValue: rowData.rawValue,
            fillCss: rowData.gradientCss,
            tokenLabel: rows[rowData.rowIndex]?.tokenLabel || `Token ${rowData.rowIndex + 1}`
        })),
        scaleLabelTex: '\\sqrt{d_h}',
        outputLabelTex: 'A_{\\mathrm{pre}}',
        outputRowCount: outputRows.length,
        outputColumnCount: rows.length,
        outputRows,
        maskLabelTex: 'M_{\\mathrm{causal}}',
        maskRows,
        softmaxLabelTex: '\\mathrm{softmax}',
        postLabelTex: 'A_{\\mathrm{post}}',
        postRowCount: postRows.length,
        postColumnCount: rows.length,
        postRows,
        valueLabelTex: valueProjection?.outputLabelTex || 'V',
        valueRowCount: valueRows.length,
        valueColumnCount: D_HEAD,
        valueRows,
        headOutputLabelTex: 'H_i',
        headOutputRowCount: headOutputRows.length,
        headOutputColumnCount: D_HEAD,
        headOutputRows
    };
}

export function buildMhsaTokenMatrixPreviewData({
    activationSource,
    layerIndex = DEFAULT_LAYER_INDEX,
    headIndex = DEFAULT_HEAD_INDEX,
    sampleStep = DEFAULT_SAMPLE_STEP,
    tokenIndices = null,
    tokenLabels = null
} = {}) {
    if (!activationSource) return null;

    const resolvedTokenIndices = resolvePreviewTokenIndices(activationSource, tokenIndices);
    const rowCount = resolvedTokenIndices.length;
    const bandCount = Math.max(1, Math.ceil(D_MODEL / Math.max(1, Math.floor(sampleStep || DEFAULT_SAMPLE_STEP))));
    const rows = [];

    for (let rowIndex = 0; rowIndex < rowCount; rowIndex += 1) {
        const tokenIndex = resolvedTokenIndices[rowIndex];
        const lnShiftVector = typeof activationSource.getLayerLn1 === 'function'
            ? activationSource.getLayerLn1(layerIndex, 'shift', tokenIndex)
            : null;
        const rowSamples = cleanNumberArray(lnShiftVector, bandCount);
        rows.push({
            rowIndex,
            tokenIndex,
            tokenLabel: resolveTokenLabel(
                activationSource,
                tokenIndex,
                Array.isArray(tokenLabels) ? tokenLabels[rowIndex] : null,
                rowIndex
            ),
            rawValues: rowSamples,
            gradientCss: buildGradientCssFromSamples(rowSamples)
        });
    }

    const projections = PROJECTION_CONFIGS.map((config) => {
        const rgb = hexToRgb(config.colorHex);
        const outputRows = rows.map((rowData) => {
            const rawValues = typeof activationSource.getLayerQKVVector === 'function'
                ? activationSource.getLayerQKVVector(layerIndex, config.kind, headIndex, rowData.tokenIndex, D_HEAD)
                : null;
            const safeValues = Array.isArray(rawValues) && rawValues.length
                ? cleanNumberArray(rawValues, D_HEAD)
                : null;
            const scalar = typeof activationSource.getLayerQKVScalar === 'function'
                ? activationSource.getLayerQKVScalar(layerIndex, config.kind, headIndex, rowData.tokenIndex)
                : null;
            const safeScalar = Number.isFinite(scalar) ? scalar : 0;
            return {
                rowIndex: rowData.rowIndex,
                tokenIndex: rowData.tokenIndex,
                tokenLabel: rowData.tokenLabel,
                rawValue: safeScalar,
                rawValues: safeValues,
                gradientCss: buildProjectionVectorGradientCss(
                    safeValues,
                    safeScalar,
                    config.gradientOptions,
                    '90deg',
                    PROJECTION_VECTOR_PREVIEW_DARKEN_FACTOR
                )
            };
        });
        const biasSample = getAttentionBiasHeadSample(layerIndex, config.biasKind, headIndex);
        const safeBiasSample = Number.isFinite(biasSample) ? biasSample : 0;
        return {
            kind: config.kind.toUpperCase(),
            weightLabelTex: config.weightLabelTex,
            biasLabelTex: config.biasLabelTex,
            outputLabelTex: config.outputLabelTex,
            colorRgb: rgb,
            weightRowCount: D_MODEL,
            weightColumnCount: D_HEAD,
            weightGradientCss: buildWeightCardCss(rgb),
            biasGradientCss: buildAccentCss(rgb, safeBiasSample),
            outputRowCount: outputRows.length,
            outputColumnCount: D_HEAD,
            outputRows
        };
    });

    const attentionScoreStage = buildAttentionScoreStage({
        activationSource,
        layerIndex,
        headIndex,
        rows,
        queryProjection: projections[0] || null,
        keyProjection: projections[1] || null,
        valueProjection: projections[2] || null
    });

    return {
        rowCount: rows.length,
        columnCount: D_MODEL,
        sampleStep: Math.max(1, Math.floor(sampleStep || DEFAULT_SAMPLE_STEP)),
        bandCount,
        rows,
        projections,
        attentionScoreStage
    };
}
