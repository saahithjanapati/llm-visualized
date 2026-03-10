import { getAttentionBiasHeadSample } from '../data/biasParams.js';
import {
    MHA_FINAL_K_COLOR,
    MHA_FINAL_Q_COLOR,
    MHA_FINAL_V_COLOR
} from '../animations/LayerAnimationConstants.js';
import { mapValueToColor } from '../utils/colors.js';
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
const PROJECTION_CONFIGS = [
    {
        kind: 'q',
        biasKind: 'query',
        weightLabelTex: 'W_q',
        biasLabelTex: 'b_q',
        outputLabelTex: 'Q',
        colorHex: MHA_FINAL_Q_COLOR
    },
    {
        kind: 'k',
        biasKind: 'key',
        weightLabelTex: 'W_k',
        biasLabelTex: 'b_k',
        outputLabelTex: 'K',
        colorHex: MHA_FINAL_K_COLOR
    },
    {
        kind: 'v',
        biasKind: 'value',
        weightLabelTex: 'W_v',
        biasLabelTex: 'b_v',
        outputLabelTex: 'V',
        colorHex: MHA_FINAL_V_COLOR
    }
];

function colorToCss(color) {
    return color?.isColor ? `#${color.getHexString()}` : 'transparent';
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

function buildAccentCss(rgb, value = 0) {
    const intensity = Math.min(1, Math.abs(Number(value) || 0) / RESIDUAL_COLOR_CLAMP);
    const alphaStart = 0.52 + (intensity * 0.2);
    const alphaEnd = 0.78 + (intensity * 0.18);
    const [r, g, b] = Array.isArray(rgb) ? rgb : [255, 255, 255];
    return `linear-gradient(90deg, rgba(${r}, ${g}, ${b}, ${alphaStart.toFixed(3)}), rgba(${r}, ${g}, ${b}, ${alphaEnd.toFixed(3)}))`;
}

function buildSolidCss(rgb, value = 0) {
    const intensity = Math.min(1, Math.abs(Number(value) || 0) / RESIDUAL_COLOR_CLAMP);
    const alpha = 0.66 + (intensity * 0.24);
    const [r, g, b] = Array.isArray(rgb) ? rgb : [255, 255, 255];
    return `rgba(${r}, ${g}, ${b}, ${alpha.toFixed(3)})`;
}

function buildWeightCardCss(rgb) {
    const base = Array.isArray(rgb) ? rgb : [255, 255, 255];
    const top = mixRgb(base, [255, 255, 255], 0.18);
    const mid = mixRgb(base, [255, 255, 255], 0.04);
    const bottom = mixRgb(base, [8, 12, 18], 0.32);
    return [
        `radial-gradient(138% 112% at 92% 6%, ${rgbToCss(top, 0.52)} 0%, ${rgbToCss(top, 0.16)} 34%, ${rgbToCss(top, 0.04)} 56%, ${rgbToCss(top, 0.0)} 78%)`,
        `radial-gradient(124% 118% at 10% 94%, ${rgbToCss(bottom, 0.88)} 0%, ${rgbToCss(bottom, 0.42)} 36%, ${rgbToCss(bottom, 0.12)} 58%, ${rgbToCss(bottom, 0.0)} 82%)`,
        `radial-gradient(116% 92% at 46% 52%, ${rgbToCss(mid, 0.42)} 0%, ${rgbToCss(mid, 0.2)} 42%, ${rgbToCss(mid, 0.0)} 74%)`,
        `linear-gradient(160deg, ${rgbToCss(top, 0.82)} 0%, ${rgbToCss(mid, 0.92)} 44%, ${rgbToCss(bottom, 0.96)} 100%)`
    ].join(', ');
}

function buildAttentionScoreCellCss(value) {
    const color = mapValueToColor(value, { clampMax: ATTENTION_PRE_COLOR_CLAMP });
    if (color?.isColor && typeof color.clone === 'function') {
        return colorToCss(color.clone().multiplyScalar(ATTENTION_PREVIEW_COLOR_DARKEN_FACTOR));
    }
    return colorToCss(color);
}

function resolvePromptRowCount(activationSource) {
    const promptTokens = Array.isArray(activationSource?.meta?.prompt_tokens)
        ? activationSource.meta.prompt_tokens
        : [];
    if (promptTokens.length) return promptTokens.length;
    const tokenCount = typeof activationSource?.getTokenCount === 'function'
        ? activationSource.getTokenCount()
        : 0;
    if (Number.isFinite(tokenCount) && tokenCount > 0) {
        return Math.min(DEFAULT_PROMPT_ROW_COUNT, Math.floor(tokenCount));
    }
    return DEFAULT_PROMPT_ROW_COUNT;
}

function resolveTokenLabel(activationSource, tokenIndex) {
    const tokenDisplayStrings = Array.isArray(activationSource?.meta?.token_display_strings)
        ? activationSource.meta.token_display_strings
        : [];
    if (typeof tokenDisplayStrings[tokenIndex] === 'string') {
        return normalizeTokenLabel(tokenDisplayStrings[tokenIndex], tokenIndex);
    }
    const label = typeof activationSource?.getTokenString === 'function'
        ? activationSource.getTokenString(tokenIndex)
        : null;
    return normalizeTokenLabel(label, tokenIndex);
}

function buildAttentionScoreStage({
    activationSource,
    layerIndex,
    headIndex,
    rows,
    queryProjection,
    keyProjection
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

    const outputRows = rows.map((rowData) => {
        const scoreRow = typeof activationSource.getAttentionScoresRow === 'function'
            ? activationSource.getAttentionScoresRow(layerIndex, 'pre', headIndex, rowData.rowIndex)
            : null;
        const cells = rows.map((colData) => {
            const rawValue = Array.isArray(scoreRow) ? scoreRow[colData.rowIndex] : null;
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
        outputRows
    };
}

export function buildMhsaTokenMatrixPreviewData({
    activationSource,
    layerIndex = DEFAULT_LAYER_INDEX,
    headIndex = DEFAULT_HEAD_INDEX,
    sampleStep = DEFAULT_SAMPLE_STEP
} = {}) {
    if (!activationSource) return null;

    const rowCount = resolvePromptRowCount(activationSource);
    const bandCount = Math.max(1, Math.ceil(D_MODEL / Math.max(1, Math.floor(sampleStep || DEFAULT_SAMPLE_STEP))));
    const rows = [];

    for (let rowIndex = 0; rowIndex < rowCount; rowIndex += 1) {
        const lnShiftVector = typeof activationSource.getLayerLn1 === 'function'
            ? activationSource.getLayerLn1(layerIndex, 'shift', rowIndex)
            : null;
        const rowSamples = cleanNumberArray(lnShiftVector, bandCount);
        rows.push({
            rowIndex,
            tokenLabel: resolveTokenLabel(activationSource, rowIndex),
            rawValues: rowSamples,
            gradientCss: buildGradientCssFromSamples(rowSamples)
        });
    }

    const projections = PROJECTION_CONFIGS.map((config) => {
        const rgb = hexToRgb(config.colorHex);
        const outputRows = rows.map((rowData) => {
            const scalar = typeof activationSource.getLayerQKVScalar === 'function'
                ? activationSource.getLayerQKVScalar(layerIndex, config.kind, headIndex, rowData.rowIndex)
                : null;
            const safeScalar = Number.isFinite(scalar) ? scalar : 0;
            return {
                rowIndex: rowData.rowIndex,
                rawValue: safeScalar,
                gradientCss: buildSolidCss(rgb, safeScalar)
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
        keyProjection: projections[1] || null
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
