import { getAttentionBiasHeadSample } from '../data/biasParams.js';
import { mapValueToColor } from '../utils/colors.js';
import {
    D_HEAD,
    D_MODEL,
    RESIDUAL_COLOR_CLAMP
} from './selectionPanelConstants.js';
import { formatTokenLabelForPreview } from './selectionPanelFormatUtils.js';

const DEFAULT_SAMPLE_STEP = 64;
const DEFAULT_LAYER_INDEX = 5;
const DEFAULT_HEAD_INDEX = 5;
const DEFAULT_PROMPT_ROW_COUNT = 5;
const MHSA_QUERY_BLUE_RGB = [39, 110, 187];
const WEIGHT_CARD_TOP_RGB = [142, 182, 244];
const WEIGHT_CARD_MID_RGB = [110, 156, 246];
const WEIGHT_CARD_BOTTOM_RGB = [58, 108, 255];

function colorToCss(color) {
    return color?.isColor ? `#${color.getHexString()}` : 'transparent';
}

function rgbToCss(rgb, alpha = 1) {
    const [r, g, b] = Array.isArray(rgb) && rgb.length === 3
        ? rgb
        : [255, 255, 255];
    return `rgba(${r}, ${g}, ${b}, ${alpha.toFixed(3)})`;
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

function buildBlueAccentCss(value = 0) {
    const intensity = Math.min(1, Math.abs(Number(value) || 0) / RESIDUAL_COLOR_CLAMP);
    const alphaStart = 0.52 + (intensity * 0.2);
    const alphaEnd = 0.78 + (intensity * 0.18);
    const [r, g, b] = MHSA_QUERY_BLUE_RGB;
    return `linear-gradient(90deg, rgba(${r}, ${g}, ${b}, ${alphaStart.toFixed(3)}), rgba(${r}, ${g}, ${b}, ${alphaEnd.toFixed(3)}))`;
}

function buildBlueSolidCss(value = 0) {
    const intensity = Math.min(1, Math.abs(Number(value) || 0) / RESIDUAL_COLOR_CLAMP);
    const alpha = 0.66 + (intensity * 0.24);
    const [r, g, b] = MHSA_QUERY_BLUE_RGB;
    return `rgba(${r}, ${g}, ${b}, ${alpha.toFixed(3)})`;
}

function buildBlueWeightCardCss() {
    return [
        `radial-gradient(138% 112% at 92% 6%, ${rgbToCss(WEIGHT_CARD_TOP_RGB, 0.78)} 0%, ${rgbToCss(WEIGHT_CARD_TOP_RGB, 0.3)} 34%, ${rgbToCss(WEIGHT_CARD_TOP_RGB, 0.08)} 56%, ${rgbToCss(WEIGHT_CARD_TOP_RGB, 0.0)} 78%)`,
        `radial-gradient(124% 118% at 10% 94%, ${rgbToCss(WEIGHT_CARD_BOTTOM_RGB, 0.84)} 0%, ${rgbToCss(WEIGHT_CARD_BOTTOM_RGB, 0.34)} 36%, ${rgbToCss(WEIGHT_CARD_BOTTOM_RGB, 0.08)} 58%, ${rgbToCss(WEIGHT_CARD_BOTTOM_RGB, 0.0)} 82%)`,
        `radial-gradient(116% 92% at 46% 52%, ${rgbToCss(WEIGHT_CARD_MID_RGB, 0.38)} 0%, ${rgbToCss(WEIGHT_CARD_MID_RGB, 0.16)} 42%, ${rgbToCss(WEIGHT_CARD_MID_RGB, 0.0)} 74%)`,
        `linear-gradient(160deg, ${rgbToCss(WEIGHT_CARD_TOP_RGB, 0.74)} 0%, ${rgbToCss(WEIGHT_CARD_MID_RGB, 0.8)} 46%, ${rgbToCss(WEIGHT_CARD_BOTTOM_RGB, 0.86)} 100%)`
    ].join(', ');
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
    const queryRows = [];

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

        const qScalar = typeof activationSource.getLayerQKVScalar === 'function'
            ? activationSource.getLayerQKVScalar(layerIndex, 'q', headIndex, rowIndex)
            : null;
        const safeQScalar = Number.isFinite(qScalar) ? qScalar : 0;
        queryRows.push({
            rowIndex,
            rawValue: safeQScalar,
            gradientCss: buildBlueSolidCss(safeQScalar)
        });
    }

    const qBiasSample = getAttentionBiasHeadSample(layerIndex, 'query', headIndex);
    const safeQBiasSample = Number.isFinite(qBiasSample) ? qBiasSample : 0;

    return {
        rowCount: rows.length,
        columnCount: D_MODEL,
        sampleStep: Math.max(1, Math.floor(sampleStep || DEFAULT_SAMPLE_STEP)),
        bandCount,
        rows,
        qWeightRowCount: D_MODEL,
        qWeightColumnCount: D_HEAD,
        qWeightGradientCss: buildBlueWeightCardCss(),
        qBiasSampleCount: 1,
        qBiasGradientCss: buildBlueAccentCss(safeQBiasSample),
        qRowCount: queryRows.length,
        qColumnCount: D_HEAD,
        queryRows
    };
}
