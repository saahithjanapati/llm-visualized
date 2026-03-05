import * as THREE from 'three';
import { FontLoader } from 'three/examples/jsm/loaders/FontLoader.js';
import { WeightMatrixVisualization } from '../components/WeightMatrixVisualization.js';
import { VectorVisualizationInstancedPrism } from '../components/VectorVisualizationInstancedPrism.js';
import { LayerNormalizationVisualization } from '../components/LayerNormalizationVisualization.js';
import { appState } from '../state/appState.js';
import { createSciFiMaterial, updateSciFiMaterialColor } from '../utils/sciFiMaterial.js';
import { mapValueToColor, mapValueToGrayscale } from '../utils/colors.js';
import { initTouchClickFallback } from './touchClickFallback.js';
import { fitSelectionDimensionLabels } from './selectionPanelDimensionFitUtils.js';
import {
    findUserDataNumber,
    findUserDataString,
    getActivationDataFromSelection,
    inferQkvType,
    isAttentionScoreSelection,
    isKvCacheVectorSelection,
    isLayerNormLabel,
    isLayerNormSolidSelection,
    isLogitBarSelection,
    isParameterSelection,
    isQkvMatrixLabel,
    isResidualVectorSelection,
    isSelfAttentionSelection,
    isValueSelection,
    isWeightMatrixLabel,
    isWeightedSumSelection,
    normalizeSelectionLabel,
    resolveAttentionModeFromSelection,
    simplifyLayerNormParamDisplayLabel
} from './selectionPanelSelectionUtils.js';
import { resolveDescription, resolveSelectionEquations } from './selectionPanelNarrativeUtils.js';
import {
    MHA_MATRIX_PARAMS,
    MLP_MATRIX_PARAMS_UP,
    MLP_MATRIX_PARAMS_DOWN,
    EMBEDDING_MATRIX_PARAMS_VOCAB,
    EMBEDDING_MATRIX_PARAMS_POSITION,
    LAYER_NORM_FINAL_COLOR,
    VECTOR_LENGTH_PRISM,
    PRISM_DIMENSIONS_PER_UNIT,
    NUM_HEAD_SETS_LAYER,
    HIDE_INSTANCE_Y_OFFSET,
    ATTENTION_POST_SOFTMAX_GRAYSCALE_MIN,
    resolveRenderPixelRatio
} from '../utils/constants.js';
import { getIncompleteUtf8TokenNote } from '../utils/tokenEncodingNotes.js';
import {
    MHA_FINAL_Q_COLOR,
    MHA_FINAL_K_COLOR,
    MHA_FINAL_V_COLOR,
    MHA_OUTPUT_PROJECTION_MATRIX_COLOR,
    MHA_OUTPUT_PROJECTION_MATRIX_PARAMS,
    POSITION_EMBED_COLOR,
    LN_PARAMS
} from '../animations/LayerAnimationConstants.js';

const PREVIEW_LANES = 3;
const PREVIEW_SOLID_LANES = 5;
const PREVIEW_TOKEN_LANES = 1;
const PREVIEW_MATRIX_DEPTH = 320;
const PREVIEW_LANE_SPACING = 80;
const PREVIEW_TARGET_SIZE = 140;
// Base framing for most objects; vector previews can request additional padding.
const PREVIEW_FRAME_PADDING = 1.25;
const PREVIEW_BASE_DISTANCE_MULT = 1.15;
const PREVIEW_ROTATION_ENVELOPE_MARGIN = 1.06;
const PREVIEW_VECTOR_PADDING_MULT = 2.4;
const PREVIEW_VECTOR_DISTANCE_MULT = 1.9;
const PREVIEW_MOBILE_MATRIX_PADDING_MULT = 1.2;
const PREVIEW_MOBILE_MATRIX_DISTANCE_MULT = 1.22;
const PREVIEW_ROTATION_SPEED = 0.0035;
const PREVIEW_BASE_TILT_X = -0.12;
const PREVIEW_BASE_ROTATION_Y = 0.38;
const PREVIEW_TILT_AMPLITUDE = 0.02;
const PREVIEW_TILT_OSC_SPEED = 0.32;
const PREVIEW_FIT_LOCK_MS = 500;
const PREVIEW_FIT_LOCK_PX = 120;
const PREVIEW_FIT_LOCK_RATIO = 0.18;
const FINAL_MLP_COLOR = 0xc07a12;
const FINAL_VOCAB_TOP_COLOR = MHA_FINAL_Q_COLOR;
const PREVIEW_QKV_LANES = 3;
const PREVIEW_QKV_LANE_SPACING = 360;
const PREVIEW_VECTOR_LARGE_SCALE = 1.0;
const PREVIEW_VECTOR_SMALL_SCALE = 0.38;
const PREVIEW_TRAIL_COLOR = 0x6ea0ff;
const PREVIEW_QKV_X_SPREAD = 72;
const PREVIEW_QKV_START_Y = -150;
const PREVIEW_QKV_MATRIX_Y = -15;
const PREVIEW_QKV_OUTPUT_Y = 95;
const PREVIEW_QKV_EXIT_Y = PREVIEW_QKV_OUTPUT_Y + 60;
const PREVIEW_QKV_RISE_DURATION = 900;
const PREVIEW_QKV_CONVERT_DURATION = 420;
const PREVIEW_QKV_HOLD_DURATION = 520;
const PREVIEW_QKV_EXIT_DURATION = 420;
const PREVIEW_QKV_IDLE_DURATION = 260;
const PREVIEW_QKV_LANE_STAGGER = 0;
const PREVIEW_VECTOR_HEAD_INSTANCES = 1;
const PREVIEW_VECTOR_BODY_INSTANCES = VECTOR_LENGTH_PRISM;
const ATTENTION_PREVIEW_MAX_TOKENS = 16;
const ATTENTION_PREVIEW_TARGET_PX = 320;
const ATTENTION_PREVIEW_MIN_CELL = 4;
const ATTENTION_PREVIEW_MAX_CELL = 24;
const ATTENTION_PREVIEW_GAP = 4;
const ATTENTION_PREVIEW_TRIANGLE = 'lower';
const ATTENTION_PREVIEW_GRID_GAP = 8; // matches .detail-attention-grid column gap in CSS
const ATTENTION_PRE_COLOR_CLAMP = 5;
const ATTENTION_PREVIEW_COLOR_DARKEN_FACTOR = 0.84;
const ATTENTION_POP_OUT_MS = 120;
const ATTENTION_POST_REVEAL_DURATION_MS = 260;
const ATTENTION_POST_REVEAL_SWEEP_MS = 380;
const ATTENTION_POST_REVEAL_STAGGER_MIN_MS = 18;
const ATTENTION_POST_REVEAL_STAGGER_MAX_MS = 52;
const ATTENTION_PRE_REVEAL_DURATION_MS = 210;
const ATTENTION_PRE_REVEAL_SWEEP_MS = 120;
const ATTENTION_PRE_REVEAL_STAGGER_MIN_MS = 4;
const ATTENTION_PRE_REVEAL_STAGGER_MAX_MS = 26;
const ATTENTION_SCORE_DECIMALS = 4;
const ATTENTION_VALUE_PLACEHOLDER = '--';
const ATTENTION_DECODE_ROW_OFFSET_MULT = 0.68;
const ATTENTION_DECODE_ROW_OFFSET_MIN_PX = 8;
const ATTENTION_DECODE_ROW_OFFSET_MAX_PX = 20;
const RESIDUAL_COLOR_CLAMP = 2;
const SPACE_TOKEN_DISPLAY = '" "';
const PANEL_SHIFT_DURATION_MS = 520;
const DETAIL_EQUATION_FONT_MIN_PX = 9;
const DETAIL_EQUATION_FONT_MAX_PX = 20;
const DETAIL_EQUATION_FONT_MAX_SCALE = 1.5;
const DETAIL_EQUATION_FIT_BUFFER_PX = 1.25;
const COPY_CONTEXT_BUTTON_DEFAULT_LABEL = 'Have a question? Copy context';
const COPY_CONTEXT_SUCCESS_LABEL = 'Context copied to clipboard.';
const COPY_CONTEXT_ERROR_LABEL = 'Unable to copy context.';
const COPY_CONTEXT_EMPTY_LABEL = 'Nothing visible to copy yet.';
const COPY_CONTEXT_FEEDBACK_MS = 1000;
const COPY_CONTEXT_FADE_MS = 220;

const TOKEN_CHIP_STYLE = {
    padding: 80,
    minWidth: 220,
    minHeight: 100,
    height: 120,
    cornerRadius: 18,
    depth: 12,
    textDepth: 16,
    textSize: 52,
    textOffset: 1.2
};

const LOGIT_PREVIEW_TEXT_STYLE = {
    size: 360,
    depth: 44,
    fitWidth: 960,
    fitHeight: 300,
    minScale: 0.28,
    maxScale: 1.6,
    faceOffset: 0.03
};

const TOKEN_CHIP_FONT_URL = 'https://threejs.org/examples/fonts/helvetiker_regular.typeface.json';
let tokenChipFont = null;
let tokenChipFontPromise = null;

function requestTokenChipFont() {
    if (tokenChipFont) return Promise.resolve(tokenChipFont);
    if (tokenChipFontPromise) return tokenChipFontPromise;
    const loader = new FontLoader();
    tokenChipFontPromise = new Promise((resolve) => {
        loader.load(
            TOKEN_CHIP_FONT_URL,
            (font) => {
                tokenChipFont = font;
                resolve(font);
            },
            undefined,
            (err) => {
                console.warn('Selection token font failed to load, falling back to canvas text.', err);
                resolve(null);
            }
        );
    });
    return tokenChipFontPromise;
}

const D_MODEL = 768;
const D_HEAD = Math.floor(D_MODEL / NUM_HEAD_SETS_LAYER);
const VOCAB_SIZE = 50257;
const CONTEXT_LEN = 1024;

function formatNumber(value) {
    if (!Number.isFinite(value)) return 'TBD';
    return Math.round(value).toLocaleString('en-US');
}

function formatDims(inputDim, outputDim) {
    if (!Number.isFinite(inputDim) || !Number.isFinite(outputDim)) return 'TBD';
    return `input dimension: ${formatNumber(inputDim)} | output dimension: ${formatNumber(outputDim)}`;
}

function buildMetadata(params = 'TBD', inputDim = null, outputDim = null, length = null, biasDim = null) {
    const hasDims = Number.isFinite(inputDim) && Number.isFinite(outputDim);
    const hasLength = Number.isFinite(length);
    const hasBiasDim = Number.isFinite(biasDim);
    const paramCount = hasDims ? formatNumber(inputDim * outputDim) : params;
    return {
        params: paramCount,
        dims: hasDims ? formatDims(inputDim, outputDim) : 'TBD',
        inputDim: hasDims ? formatNumber(inputDim) : 'TBD',
        outputDim: hasDims ? formatNumber(outputDim) : 'TBD',
        length: hasLength ? formatNumber(length) : 'TBD',
        biasDim: hasBiasDim ? formatNumber(biasDim) : '',
        hasBiasDim,
        hasDims
    };
}

function normalizePreviewKeyValue(value) {
    if (typeof value === 'number') {
        return Number.isFinite(value) ? String(value) : '';
    }
    if (typeof value === 'boolean') {
        return value ? '1' : '0';
    }
    return (typeof value === 'string') ? value : '';
}

function buildSelectionPreviewKey(label, selection) {
    const info = (selection && typeof selection.info === 'object') ? selection.info : null;
    const hit = (selection && typeof selection.hit === 'object') ? selection.hit : null;
    const object = selection?.object || hit?.object || null;
    const objectUuid = (object && typeof object.uuid === 'string') ? object.uuid : '';
    const logitEntry = (info && typeof info.logitEntry === 'object') ? info.logitEntry : null;

    const parts = [
        label || '',
        selection?.kind || '',
        objectUuid,
        normalizePreviewKeyValue(hit?.instanceId),
        normalizePreviewKeyValue(findUserDataNumber(selection, 'layerIndex')),
        normalizePreviewKeyValue(findUserDataNumber(selection, 'headIndex')),
        normalizePreviewKeyValue(findUserDataNumber(selection, 'laneIndex')),
        normalizePreviewKeyValue(findUserDataNumber(selection, 'laneLayoutIndex')),
        normalizePreviewKeyValue(findUserDataNumber(selection, 'tokenIndex')),
        normalizePreviewKeyValue(findUserDataNumber(selection, 'tokenId')),
        normalizePreviewKeyValue(findUserDataNumber(selection, 'vectorIndex')),
        normalizePreviewKeyValue(findUserDataNumber(selection, 'prismIndex')),
        normalizePreviewKeyValue(findUserDataNumber(selection, 'row')),
        normalizePreviewKeyValue(findUserDataNumber(selection, 'col')),
        normalizePreviewKeyValue(findUserDataString(selection, 'category')),
        normalizePreviewKeyValue(findUserDataString(selection, 'stage')),
        normalizePreviewKeyValue(info?.mode),
        normalizePreviewKeyValue(info?.label),
        normalizePreviewKeyValue(info?.token_id),
        normalizePreviewKeyValue(info?.tokenId),
        normalizePreviewKeyValue(info?.logitIndex),
        normalizePreviewKeyValue(info?.barIndex),
        normalizePreviewKeyValue(logitEntry?.token_id),
        normalizePreviewKeyValue(logitEntry?.tokenId),
        normalizePreviewKeyValue(logitEntry?.index)
    ];
    return parts.join('|');
}

function isAttentionHeadVectorSelection(label, selectionInfo) {
    const lower = (label || '').toLowerCase();
    const category = String(selectionInfo?.info?.category || '').toUpperCase();
    const stage = String(getActivationDataFromSelection(selectionInfo)?.stage || '').toLowerCase();
    if (category === 'Q' || category === 'K' || category === 'V') return true;
    if (selectionInfo?.kind === 'mergedKV') return true;
    if (lower.includes('query vector') || lower.includes('key vector') || lower.includes('value vector')) return true;
    if (lower.includes('merged key vectors') || lower.includes('merged value vectors')) return true;
    if (lower.includes('attention weighted sum')) return true;
    if (stage.startsWith('qkv.')) return true;
    return false;
}

function isQkvHeadVectorSelection(label, selectionInfo) {
    const lower = (label || '').toLowerCase();
    const stage = String(getActivationDataFromSelection(selectionInfo)?.stage || '').toLowerCase();
    if (stage.startsWith('qkv.')) return true;
    if (lower.includes('query vector') || lower.includes('key vector') || lower.includes('value vector')) return true;
    return false;
}

function isMlpMiddleVectorSelection(label, selectionInfo) {
    const lower = (label || '').toLowerCase();
    const stage = String(getActivationDataFromSelection(selectionInfo)?.stage || '').toLowerCase();
    if (lower.includes('mlp up projection') || lower.includes('mlp activation')) return true;
    if (lower.includes('mlp expanded segments')) return true;
    if (stage.startsWith('mlp.up') || stage.startsWith('mlp.activation')) return true;
    return false;
}

function resolveVectorLength(label, selectionInfo) {
    const lower = (label || '').toLowerCase();
    const qkvOutputLength = selectionInfo?.info?.vectorRef?.userData?.qkvOutputLength;
    if (Number.isFinite(qkvOutputLength) && qkvOutputLength > 0) {
        return Math.max(1, Math.floor(qkvOutputLength));
    }
    if (lower.includes('post-layernorm residual') || lower.includes('post layernorm residual')) {
        return D_MODEL;
    }
    if (isMlpMiddleVectorSelection(label, selectionInfo)) {
        return D_MODEL * 4;
    }
    if (isAttentionHeadVectorSelection(label, selectionInfo)) {
        return D_HEAD;
    }
    return D_MODEL;
}

function formatValues(values, perLine = 8) {
    if (!values || typeof values.length !== 'number' || values.length === 0) return '(empty)';
    let result = '';
    for (let idx = 0; idx < values.length; idx += 1) {
        const num = Number(values[idx]);
        const formatted = Number.isFinite(num) ? num.toFixed(4) : '0.0000';
        const sep = idx === 0 ? '' : idx % perLine === 0 ? '\n' : ', ';
        result += sep + formatted;
    }
    return result;
}

function formatTokenWithIndex(index, label, fallback = 'Token') {
    const tokenText = formatTokenLabelForPreview(label);
    if (Number.isFinite(index) && tokenText) return `${index + 1} (${tokenText})`;
    if (Number.isFinite(index)) return String(index + 1);
    return tokenText || fallback;
}

function normalizeAttentionValuePart(value, fallback = ATTENTION_VALUE_PLACEHOLDER) {
    const text = typeof value === 'string' ? value.trim() : '';
    return text || fallback;
}

function formatAttentionSubtitleTokenPart(label, tokenIndex, roleLabel) {
    const tokenText = normalizeAttentionValuePart(formatTokenLabelForPreview(label));
    const positionText = Number.isFinite(tokenIndex)
        ? `Position ${Math.floor(tokenIndex) + 1}`
        : 'Position n/a';
    return `${roleLabel} ${tokenText} (${positionText})`;
}

function formatActivationData(data) {
    if (!data || typeof data !== 'object') return 'No activation data.';
    const lines = [];
    const stage = data.stage ? String(data.stage) : '';
    const stageLower = stage.toLowerCase();
    const isAttentionScore = stageLower.startsWith('attention.');
    if (stage) lines.push(`Stage: ${stage}`);
    if (Number.isFinite(data.layerIndex)) lines.push(`Layer: ${data.layerIndex + 1}`);
    if (isAttentionScore) {
        if (Number.isFinite(data.tokenIndex) || data.tokenLabel) {
            lines.push(`Source token: ${formatTokenWithIndex(data.tokenIndex, data.tokenLabel, 'Source')}`);
        }
        if (Number.isFinite(data.keyTokenIndex) || data.keyTokenLabel) {
            lines.push(`Target token: ${formatTokenWithIndex(data.keyTokenIndex, data.keyTokenLabel, 'Target')}`);
        }
    } else {
        if (Number.isFinite(data.tokenIndex)) {
            const tokenText = data.tokenLabel ? ` (${formatTokenLabelForPreview(data.tokenLabel)})` : '';
            lines.push(`Token: ${data.tokenIndex + 1}${tokenText}`);
        }
        if (Number.isFinite(data.keyTokenIndex)) {
            const keyText = data.keyTokenLabel ? ` (${formatTokenLabelForPreview(data.keyTokenLabel)})` : '';
            lines.push(`Key: ${data.keyTokenIndex + 1}${keyText}`);
        }
    }
    if (Number.isFinite(data.headIndex)) lines.push(`Head: ${data.headIndex + 1}`);
    if (Number.isFinite(data.segmentIndex)) lines.push(`Segment: ${data.segmentIndex + 1}`);
    if (Number.isFinite(data.preScore) || Number.isFinite(data.postScore)) {
        if (isAttentionScore) {
            const selectedMode = stageLower.includes('post') ? 'post' : 'pre';
            const selectedScore = selectedMode === 'post' ? data.postScore : data.preScore;
            if (Number.isFinite(selectedScore)) {
                lines.push(`Attention score (${selectedMode}-softmax): ${selectedScore.toFixed(ATTENTION_SCORE_DECIMALS)}`);
            }
        }
        if (Number.isFinite(data.preScore)) lines.push(`Pre-softmax: ${data.preScore.toFixed(ATTENTION_SCORE_DECIMALS)}`);
        if (Number.isFinite(data.postScore)) lines.push(`Post-softmax: ${data.postScore.toFixed(ATTENTION_SCORE_DECIMALS)}`);
    }
    if (data.values && typeof data.values.length === 'number') {
        lines.push(`Values (${data.values.length}):`);
        lines.push(formatValues(data.values));
    }
    if (data.notes) lines.push(String(data.notes));
    return lines.join('\n');
}

function colorToCss(color) {
    if (!color) return 'transparent';
    const target = color.isColor ? color : new THREE.Color(color);
    return `#${target.getHexString()}`;
}

function darkenColor(color, factor = 1) {
    const target = color?.isColor ? color.clone() : new THREE.Color(color);
    const safeFactor = Number.isFinite(factor) ? THREE.MathUtils.clamp(factor, 0, 1) : 1;
    return target.multiplyScalar(safeFactor);
}

function resolveAttentionPreviewCellColor(value, mode) {
    const safeMode = mode === 'post' ? 'post' : 'pre';
    if (safeMode === 'post') {
        return mapValueToGrayscale(value, { minValue: ATTENTION_POST_SOFTMAX_GRAYSCALE_MIN });
    }
    const baseColor = mapValueToColor(value, { clampMax: ATTENTION_PRE_COLOR_CLAMP });
    return darkenColor(baseColor, ATTENTION_PREVIEW_COLOR_DARKEN_FACTOR);
}

function buildSpectrumLegendGradient({ clampMax, steps = 13, darkenFactor = 1 } = {}) {
    const safeClamp = Number.isFinite(clampMax) ? Math.max(1e-6, Math.abs(clampMax)) : 1;
    const safeSteps = Math.max(3, Math.min(41, Math.floor(steps)));
    const stops = [];
    for (let i = 0; i < safeSteps; i += 1) {
        const t = safeSteps === 1 ? 0 : i / (safeSteps - 1);
        const value = THREE.MathUtils.lerp(-safeClamp, safeClamp, t);
        const color = colorToCss(darkenColor(
            mapValueToColor(value, { clampMax: safeClamp }),
            darkenFactor
        ));
        const pct = (t * 100).toFixed(1);
        stops.push(`${color} ${pct}%`);
    }
    return `linear-gradient(90deg, ${stops.join(', ')})`;
}

function formatTokenLabelForPreview(label) {
    if (typeof label !== 'string') return '';
    const normalized = label.replace(/^\u0120+/, (match) => ' '.repeat(match.length));
    if (!normalized.length) return SPACE_TOKEN_DISPLAY;
    const collapsed = normalized.replace(/\s+/g, ' ');
    const trimmed = collapsed.trim();
    return trimmed.length ? trimmed : SPACE_TOKEN_DISPLAY;
}

function resolveAttentionScoreSelectionSummary(selectionInfo, context = null) {
    const activation = getActivationDataFromSelection(selectionInfo);
    const stageLower = String(activation?.stage || '').toLowerCase();
    if (!activation) return null;

    const normalizedLabel = normalizeSelectionLabel(selectionInfo?.label, selectionInfo);
    const hasAttentionScoreData = Number.isFinite(activation.preScore) || Number.isFinite(activation.postScore);
    const isAttentionScore = isAttentionScoreSelection(normalizedLabel, selectionInfo)
        || stageLower.startsWith('attention.')
        || hasAttentionScoreData;
    if (!isAttentionScore) return null;

    const mode = resolveAttentionModeFromSelection(selectionInfo)
        || (stageLower.includes('post') ? 'post' : 'pre');
    const score = mode === 'post' ? activation.postScore : activation.preScore;
    const tokenIndices = Array.isArray(context?.tokenIndices) ? context.tokenIndices : [];
    const tokenLabels = Array.isArray(context?.tokenLabels) ? context.tokenLabels : [];

    const sourceTokenIndex = Number.isFinite(activation.tokenIndex) ? activation.tokenIndex : null;
    const targetTokenIndex = Number.isFinite(activation.keyTokenIndex) ? activation.keyTokenIndex : null;
    const row = Number.isFinite(sourceTokenIndex) ? tokenIndices.indexOf(sourceTokenIndex) : -1;
    const col = Number.isFinite(targetTokenIndex) ? tokenIndices.indexOf(targetTokenIndex) : -1;

    const sourceLabel = formatTokenLabelForPreview(
        activation.tokenLabel || (row >= 0 ? tokenLabels[row] : null)
    );
    const targetLabel = formatTokenLabelForPreview(
        activation.keyTokenLabel || (col >= 0 ? tokenLabels[col] : null)
    );
    const sourceText = normalizeAttentionValuePart(sourceLabel, 'Source');
    const targetText = normalizeAttentionValuePart(targetLabel, 'Target');
    const scoreText = Number.isFinite(score) ? score.toFixed(ATTENTION_SCORE_DECIMALS) : 'n/a';
    const hasSourceContext = Number.isFinite(sourceTokenIndex) || sourceLabel.length > 0;
    const hasTargetContext = Number.isFinite(targetTokenIndex) || targetLabel.length > 0;
    const tokenContextLine = (hasSourceContext || hasTargetContext)
        ? `${formatAttentionSubtitleTokenPart(sourceLabel, sourceTokenIndex, 'Source')} • ${formatAttentionSubtitleTokenPart(targetLabel, targetTokenIndex, 'Target')}`
        : '';

    return {
        mode,
        row: row >= 0 ? row : null,
        col: col >= 0 ? col : null,
        tokenContextLine,
        defaultValue: {
            source: sourceText,
            target: targetText,
            score: scoreText,
            empty: false
        }
    };
}

function computeAttentionCellSize(count) {
    const safeCount = Math.max(1, Math.floor(count || 1));
    const size = ATTENTION_PREVIEW_TARGET_PX / safeCount;
    return Math.max(ATTENTION_PREVIEW_MIN_CELL, Math.min(ATTENTION_PREVIEW_MAX_CELL, size));
}

function shouldRevealAttentionCell(progress, row, col, mode = 'pre', decodeProfile = null) {
    const decodeHighlightRow = Number.isFinite(decodeProfile?.highlightRow)
        ? Math.max(0, Math.floor(decodeProfile.highlightRow))
        : null;
    const decodeAnimating = !!(decodeProfile?.enabled && decodeProfile?.animating && decodeHighlightRow !== null);
    if (decodeAnimating) {
        // During KV-cache decode animation, preserve previously computed rows
        // and only allow the currently computed row to animate/reveal.
        if (row < decodeHighlightRow) return true;
        if (!progress) return false;
    }
    if (!progress) return true;
    if (mode === 'post') {
        const postCompletedRows = Number.isFinite(progress.postCompletedRows) ? progress.postCompletedRows : 0;
        return row < postCompletedRows;
    }
    const completedRows = Number.isFinite(progress.completedRows) ? progress.completedRows : 0;
    const activeRow = Number.isFinite(progress.activeRow) ? progress.activeRow : null;
    const activeCol = Number.isFinite(progress.activeCol) ? progress.activeCol : null;
    if (row < completedRows) return true;
    if (activeRow !== null && activeCol !== null && row === activeRow) {
        return col <= activeCol;
    }
    return false;
}

function countVisibleAttentionCellsInRow(row, count) {
    const safeCount = Math.max(1, Math.floor(count || 1));
    const safeRow = THREE.MathUtils.clamp(Math.floor(row || 0), 0, safeCount - 1);
    if (ATTENTION_PREVIEW_TRIANGLE === 'upper') {
        return Math.max(1, safeCount - safeRow);
    }
    return Math.max(1, safeRow + 1);
}

function getAttentionRevealOrder(row, col, count) {
    const safeCount = Math.max(1, Math.floor(count || 1));
    const safeRow = THREE.MathUtils.clamp(Math.floor(row || 0), 0, safeCount - 1);
    const safeCol = THREE.MathUtils.clamp(Math.floor(col || 0), 0, safeCount - 1);
    if (ATTENTION_PREVIEW_TRIANGLE === 'upper') {
        return Math.max(0, safeCol - safeRow);
    }
    return safeCol;
}

function getContentWidth(el) {
    if (!el || typeof window === 'undefined') return 0;
    const style = window.getComputedStyle(el);
    const paddingLeft = parseFloat(style.paddingLeft) || 0;
    const paddingRight = parseFloat(style.paddingRight) || 0;
    const width = (el.clientWidth || 0) - paddingLeft - paddingRight;
    return Math.max(0, width);
}

function measureMaxTokenLabelWidth(labels, referenceEl) {
    if (!Array.isArray(labels) || labels.length === 0 || !referenceEl || typeof window === 'undefined') return 0;
    const style = window.getComputedStyle(referenceEl);
    const fontStyle = style.fontStyle || 'normal';
    const fontWeight = style.fontWeight || '400';
    const fontSize = style.fontSize || '10px';
    const fontFamily = style.fontFamily || 'monospace';
    const font = `${fontStyle} ${fontWeight} ${fontSize} ${fontFamily}`;
    const canvas = measureMaxTokenLabelWidth._canvas || (measureMaxTokenLabelWidth._canvas = document.createElement('canvas'));
    const ctx = canvas.getContext('2d');
    if (!ctx) return 0;
    ctx.font = font;
    let maxWidth = 0;
    for (let i = 0; i < labels.length; i += 1) {
        const text = typeof labels[i] === 'string' ? labels[i] : String(labels[i] ?? '');
        const width = ctx.measureText(text).width;
        if (width > maxWidth) maxWidth = width;
    }
    const padding = 4; // subtle breathing room; padding is handled separately
    return maxWidth + padding;
}

function buildAttentionMatrixValues({ activationSource, layerIndex, headIndex, tokenIndices, mode }) {
    if (!activationSource || !Array.isArray(tokenIndices) || !tokenIndices.length) return null;
    const values = [];
    for (let i = 0; i < tokenIndices.length; i += 1) {
        const queryTokenIndex = tokenIndices[i];
        const row = activationSource.getAttentionScoresRow
            ? activationSource.getAttentionScoresRow(layerIndex, mode, headIndex, queryTokenIndex)
            : null;
        const rowValues = [];
        for (let j = 0; j < tokenIndices.length; j += 1) {
            const keyTokenIndex = tokenIndices[j];
            const value = Array.isArray(row) ? row[keyTokenIndex] : null;
            rowValues.push(Number.isFinite(value) ? value : null);
        }
        values.push(rowValues);
    }
    return values;
}

function clamp01(value) {
    if (!Number.isFinite(value)) return 0;
    return Math.min(1, Math.max(0, value));
}

function easeInOutCubic(t) {
    const clamped = clamp01(t);
    return clamped < 0.5
        ? 4 * clamped * clamped * clamped
        : 1 - Math.pow(-2 * clamped + 2, 3) / 2;
}

const TMP_BOX = new THREE.Box3();
const TMP_MATRIX = new THREE.Matrix4();
const TMP_POS = new THREE.Vector3();
const TMP_QUAT = new THREE.Quaternion();
const TMP_SCALE = new THREE.Vector3();
const TMP_CENTER = new THREE.Vector3();
const TMP_COLOR = new THREE.Color();

function resolveFinalPreviewColor(label) {
    const lower = (label || '').toLowerCase();
    if (lower.includes('query weight matrix')) return MHA_FINAL_Q_COLOR;
    if (lower.includes('key weight matrix')) return MHA_FINAL_K_COLOR;
    if (lower.includes('value weight matrix')) return MHA_FINAL_V_COLOR;
    if (lower.includes('output projection matrix')) return MHA_OUTPUT_PROJECTION_MATRIX_COLOR;
    if (lower.includes('mlp up weight matrix')) return FINAL_MLP_COLOR;
    if (lower.includes('mlp down weight matrix')) return FINAL_MLP_COLOR;
    if (lower.includes('vocab embedding') || lower.includes('unembedding')) {
        return lower.includes('top') ? FINAL_VOCAB_TOP_COLOR : MHA_FINAL_Q_COLOR;
    }
    if (lower.includes('positional embedding')) return POSITION_EMBED_COLOR;
    if (lower.includes('layernorm') || lower.includes('layer norm')) return LAYER_NORM_FINAL_COLOR;
    return null;
}

function clonePreviewMaterial(material) {
    if (!material) return material;
    const uniforms = material.userData?.sciFiUniforms;
    if (uniforms) {
        const clone = createSciFiMaterial({
            baseColor: material.color,
            accentColor: uniforms.uAccentColor?.value,
            secondaryColor: uniforms.uSecondaryColor?.value,
            edgeColor: uniforms.uEdgeColor?.value,
            emissiveColor: material.emissive,
            emissiveIntensity: material.emissiveIntensity,
            metalness: material.metalness,
            roughness: material.roughness,
            clearcoat: material.clearcoat,
            clearcoatRoughness: material.clearcoatRoughness,
            transmission: material.transmission,
            thickness: material.thickness,
            iridescence: material.iridescence,
            iridescenceIOR: material.iridescenceIOR,
            sheen: material.sheen,
            sheenColor: material.sheenColor,
            sheenRoughness: material.sheenRoughness,
            envMapIntensity: material.envMapIntensity,
            transparent: material.transparent,
            opacity: material.opacity,
            side: material.side,
            dimensions: uniforms.uDimensions?.value,
            stripeFrequency: uniforms.uStripeFrequency?.value,
            stripeStrength: uniforms.uStripeStrength?.value,
            rimIntensity: uniforms.uRimIntensity?.value,
            gradientSharpness: uniforms.uGradientSharpness?.value,
            gradientBias: uniforms.uGradientBias?.value,
            fresnelBoost: uniforms.uFresnelBoost?.value,
            accentMix: uniforms.uAccentMix?.value,
            glowFalloff: uniforms.uGlowFalloff?.value,
            depthAccentStrength: uniforms.uDepthAccentStrength?.value,
            scanlineFrequency: uniforms.uScanlineFrequency?.value,
            scanlineStrength: uniforms.uScanlineStrength?.value,
            glintStrength: uniforms.uGlintStrength?.value,
            noiseStrength: uniforms.uNoiseStrength?.value
        });
        clone.polygonOffset = material.polygonOffset;
        clone.polygonOffsetFactor = material.polygonOffsetFactor;
        clone.polygonOffsetUnits = material.polygonOffsetUnits;
        clone.depthWrite = material.depthWrite;
        clone.depthTest = material.depthTest;
        clone.alphaTest = material.alphaTest;
        clone.colorWrite = material.colorWrite;
        clone.toneMapped = material.toneMapped;
        clone.visible = material.visible;
        clone.envMap = material.envMap;
        clone.map = material.map;
        clone.normalMap = material.normalMap;
        clone.roughnessMap = material.roughnessMap;
        clone.metalnessMap = material.metalnessMap;
        clone.emissiveMap = material.emissiveMap;
        clone.alphaMap = material.alphaMap;
        return clone;
    }
    const clone = material.clone();
    if (material.onBeforeCompile) clone.onBeforeCompile = material.onBeforeCompile;
    if (material.customProgramCacheKey) clone.customProgramCacheKey = material.customProgramCacheKey;
    return clone;
}

function cloneMaterialsForPreview(object) {
    const materials = [];
    object.traverse((child) => {
        if (!child.material) return;
        if (Array.isArray(child.material)) {
            const cloned = child.material.map((mat) => clonePreviewMaterial(mat));
            child.material = cloned;
            materials.push(...cloned.filter(Boolean));
            return;
        }
        const clone = clonePreviewMaterial(child.material);
        child.material = clone;
        if (clone) materials.push(clone);
    });
    return materials;
}

function cloneGeometriesForPreview(object) {
    const geometries = [];
    object.traverse((child) => {
        if (!child.geometry || typeof child.geometry.clone !== 'function') return;
        const clonedGeo = child.geometry.clone();
        child.geometry = clonedGeo;
        geometries.push(clonedGeo);

        if (child.isInstancedMesh) {
            if (child.instanceMatrix && typeof child.instanceMatrix.clone === 'function') {
                child.instanceMatrix = child.instanceMatrix.clone();
                child.instanceMatrix.needsUpdate = true;
            }
            if (child.instanceColor && typeof child.instanceColor.clone === 'function') {
                child.instanceColor = child.instanceColor.clone();
                child.instanceColor.needsUpdate = true;
            }
        }
    });
    return geometries;
}

function applyLaneOverrideToInstancedMeshes(object, laneCount, laneSpacing) {
    if (!object || !Number.isFinite(laneCount) || laneCount < 1) return;
    const spacing = Number.isFinite(laneSpacing) ? laneSpacing : PREVIEW_QKV_LANE_SPACING;
    const mtx = new THREE.Matrix4();
    object.traverse((child) => {
        if (!child?.isInstancedMesh) return;
        // Only override lane spacing for vector prisms (they carry colorStart/End).
        const hasColorStart = child.geometry?.getAttribute?.('colorStart');
        const hasColorEnd = child.geometry?.getAttribute?.('colorEnd');
        if (!hasColorStart && !hasColorEnd) return;
        child.count = laneCount;
        for (let i = 0; i < laneCount; i++) {
            const z = (i - (laneCount - 1) / 2) * spacing;
            mtx.makeTranslation(0, 0, z);
            child.setMatrixAt(i, mtx);
        }
        child.instanceMatrix.needsUpdate = true;
        // Prevent frustum culling from clipping widely spaced instances.
        child.frustumCulled = false;
        if (child.geometry) {
            if (!child.geometry.boundingBox) child.geometry.computeBoundingBox();
            child.geometry.computeBoundingSphere();
        }
    });
}

function applyFinalColorToObject(object, color) {
    if (!object || color === null || color === undefined) return;
    const targetColor = color?.isColor ? color : new THREE.Color(color);
    object.traverse((child) => {
        if (!child.material) return;
        const materials = Array.isArray(child.material) ? child.material : [child.material];
        materials.forEach((mat) => {
            if (!mat) return;
            if (mat.userData?.sciFiUniforms) {
                updateSciFiMaterialColor(mat, targetColor);
            } else {
                if (mat.color?.copy) mat.color.copy(targetColor);
                if (mat.emissive?.copy) mat.emissive.copy(targetColor);
                if (typeof mat.emissiveIntensity === 'number') {
                    mat.emissiveIntensity = Math.max(mat.emissiveIntensity, 0.32);
                }
            }
            mat.needsUpdate = true;
        });
    });
}

function stripPreviewTrails(object) {
    if (!object || typeof object.traverse !== 'function') return;
    const toRemove = [];
    object.traverse((child) => {
        if (!child || !child.parent) return;
        const ud = child.userData || {};
        if (ud.isTrail || ud.trailRef) {
            toRemove.push(child);
        }
    });
    toRemove.forEach((child) => {
        try { if (child.parent) child.parent.remove(child); } catch (_) { /* optional cleanup */ }
    });
}

function buildSelectionClonePreview(selectionInfo, label) {
    const source = selectionInfo?.object || selectionInfo?.hit?.object;
    if (!source || !label) return null;
    let match = null;
    let current = source;
    while (current) {
        if (current.userData?.label === label) {
            match = current;
        }
        current = current.parent;
    }
    const root = match || source;
    if (!root || root.isScene) return null;
    const clone = root.clone(true);
    stripPreviewTrails(clone);
    clone.traverse((child) => {
        child.matrixAutoUpdate = true;
    });
    const previewGeometries = cloneGeometriesForPreview(clone);
    const previewMaterials = cloneMaterialsForPreview(clone);
    if (isQkvMatrixLabel(label)) {
        applyLaneOverrideToInstancedMeshes(clone, PREVIEW_QKV_LANES, PREVIEW_QKV_LANE_SPACING);
    }
    const finalColor = resolveFinalPreviewColor(label);
    if (!isLayerNormLabel(label)) {
        applyFinalColorToObject(clone, finalColor);
    }
    return {
        object: clone,
        dispose: () => {
            previewGeometries.forEach((geo) => geo && geo.dispose && geo.dispose());
            previewMaterials.forEach((mat) => mat && mat.dispose && mat.dispose());
        }
    };
}

function buildSharedClonePreview(selectionInfo, label) {
    const source = selectionInfo?.object || selectionInfo?.hit?.object;
    if (!source || !label) return null;
    let match = null;
    let current = source;
    while (current) {
        if (current.userData?.label === label) {
            match = current;
        }
        current = current.parent;
    }
    const root = match || source;
    if (!root || root.isScene) return null;
    const clone = root.clone(true);
    clone.traverse((child) => {
        child.matrixAutoUpdate = true;
        if (child.isInstancedMesh) {
            if (child.instanceMatrix) child.instanceMatrix.needsUpdate = true;
            if (child.instanceColor) child.instanceColor.needsUpdate = true;
        }
    });
    return {
        object: clone,
        dispose: () => {}
    };
}

function buildDirectClonePreview(selectionInfo) {
    const source = selectionInfo?.object || selectionInfo?.hit?.object;
    if (!source || source.isScene) return null;
    const clone = source.clone(true);
    clone.traverse((child) => {
        child.matrixAutoUpdate = true;
    });
    const previewGeometries = cloneGeometriesForPreview(clone);
    const previewMaterials = cloneMaterialsForPreview(clone);
    return {
        object: clone,
        dispose: () => {
            previewGeometries.forEach((geo) => geo && geo.dispose && geo.dispose());
            previewMaterials.forEach((mat) => mat && mat.dispose && mat.dispose());
        }
    };
}

// Compute bounds with instanced meshes included (Box3.setFromObject skips instances).
function getObjectBounds(object) {
    const bounds = new THREE.Box3();
    if (!object) return bounds;
    object.updateWorldMatrix(true, true);
    object.traverse((child) => {
        if (!child.geometry) return;
        if (child.isInstancedMesh) {
            if (!child.geometry.boundingBox) child.geometry.computeBoundingBox();
            if (!child.geometry.boundingBox) return;
            const instanceCount = Number.isFinite(child.count)
                ? child.count
                : (child.instanceMatrix?.count ?? 0);
            for (let i = 0; i < instanceCount; i++) {
                child.getMatrixAt(i, TMP_MATRIX);
                TMP_MATRIX.decompose(TMP_POS, TMP_QUAT, TMP_SCALE);
                // Skip hidden instances (moved to sentinel Y or shrunk)
                if (TMP_POS.y <= HIDE_INSTANCE_Y_OFFSET * 0.5 || TMP_SCALE.y < 0.01) continue;
                TMP_MATRIX.compose(TMP_POS, TMP_QUAT, TMP_SCALE);
                TMP_MATRIX.multiplyMatrices(child.matrixWorld, TMP_MATRIX);
                TMP_BOX.copy(child.geometry.boundingBox).applyMatrix4(TMP_MATRIX);
                bounds.union(TMP_BOX);
            }
            return;
        }
        if (!child.geometry.boundingBox) child.geometry.computeBoundingBox();
        if (!child.geometry.boundingBox) return;
        TMP_BOX.copy(child.geometry.boundingBox).applyMatrix4(child.matrixWorld);
        bounds.union(TMP_BOX);
    });
    return bounds;
}

function centerPreviewPivot(object) {
    if (!object) return;
    if (object.userData && object.userData.__previewPivotCentered) return;
    const bounds = getObjectBounds(object);
    if (bounds.isEmpty()) return;
    bounds.getCenter(TMP_CENTER);
    if (object.parent) {
        object.parent.updateWorldMatrix(true, true);
        object.parent.worldToLocal(TMP_CENTER);
    }
    object.position.sub(TMP_CENTER);
    if (!object.userData) object.userData = {};
    object.userData.__previewPivotCentered = true;
}

function getPreviewLaneCount(object) {
    if (!object) return 1;
    let laneCount = 1;
    object.traverse((child) => {
        if (!child || !child.isInstancedMesh) return;
        const count = Number.isFinite(child.count)
            ? child.count
            : (child.instanceMatrix?.count ?? 0);
        if (count > laneCount) laneCount = count;
    });
    return laneCount;
}

function getLaneZoomMultiplier(object) {
    const laneCount = getPreviewLaneCount(object);
    if (!Number.isFinite(laneCount) || laneCount <= 1) return 1;
    const extra = Math.min(0.35, (laneCount - 1) * 0.04);
    return 1 + extra;
}

function resolveMetadata(label, kind = null, selectionInfo = null) {
    const lower = (label || '').toLowerCase();
    if (lower.startsWith('token:') || lower.startsWith('position:')) {
        const oneHotLength = lower.startsWith('position:') ? CONTEXT_LEN : VOCAB_SIZE;
        return buildMetadata('TBD', null, null, oneHotLength);
    }
    if (lower.includes('query weight matrix')) {
        return buildMetadata(formatNumber(D_MODEL * D_HEAD), D_MODEL, D_HEAD, null, D_HEAD);
    }
    if (lower.includes('key weight matrix')) {
        return buildMetadata(formatNumber(D_MODEL * D_HEAD), D_MODEL, D_HEAD, null, D_HEAD);
    }
    if (lower.includes('value weight matrix')) {
        return buildMetadata(formatNumber(D_MODEL * D_HEAD), D_MODEL, D_HEAD, null, D_HEAD);
    }
    if (lower.includes('output projection matrix')) {
        return buildMetadata(formatNumber(D_MODEL * D_MODEL), D_MODEL, D_MODEL, null, D_MODEL);
    }
    if (lower.includes('mlp up weight matrix')) {
        return buildMetadata(formatNumber(D_MODEL * D_MODEL * 4), D_MODEL, D_MODEL * 4, null, D_MODEL * 4);
    }
    if (lower.includes('mlp down weight matrix')) {
        return buildMetadata(formatNumber(D_MODEL * D_MODEL * 4), D_MODEL * 4, D_MODEL, null, D_MODEL);
    }
    if (lower.includes('vocab embedding (top)') || lower.includes('unembedding')) {
        return buildMetadata(formatNumber(VOCAB_SIZE * D_MODEL), D_MODEL, VOCAB_SIZE);
    }
    if (lower.includes('vocab embedding')) {
        return buildMetadata(formatNumber(VOCAB_SIZE * D_MODEL), VOCAB_SIZE, D_MODEL);
    }
    if (lower.includes('positional embedding')) {
        return buildMetadata(formatNumber(CONTEXT_LEN * D_MODEL), CONTEXT_LEN, D_MODEL);
    }
    if (isLikelyVectorSelection(label, selectionInfo)) {
        return buildMetadata('TBD', null, null, resolveVectorLength(label, selectionInfo));
    }
    if (lower.includes('attention') || (kind === 'mergedKV')) {
        return buildMetadata();
    }
    return buildMetadata();
}

function escapeHtml(value) {
    return String(value)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
}

function renderDescriptionHtml(text) {
    if (!text) return '';
    const raw = String(text);
    const katex = (typeof window !== 'undefined' && window.katex) ? window.katex : null;
    if (!katex || typeof katex.renderToString !== 'function' || !raw.includes('$')) {
        return escapeHtml(raw).replace(/\n/g, '<br />');
    }

    const parts = [];
    let cursor = 0;
    while (cursor < raw.length) {
        const nextDisplay = raw.indexOf('$$', cursor);
        const nextInline = raw.indexOf('$', cursor);
        if (nextDisplay === -1 && nextInline === -1) {
            parts.push({ type: 'text', value: raw.slice(cursor) });
            break;
        }
        let start = nextInline;
        let isDisplay = false;
        if (nextDisplay !== -1 && (nextInline === -1 || nextDisplay <= nextInline)) {
            start = nextDisplay;
            isDisplay = true;
        }
        if (start > cursor) {
            parts.push({ type: 'text', value: raw.slice(cursor, start) });
        }
        if (isDisplay) {
            const end = raw.indexOf('$$', start + 2);
            if (end === -1) {
                parts.push({ type: 'text', value: raw.slice(start) });
                break;
            }
            parts.push({ type: 'math', value: raw.slice(start + 2, end), display: true });
            cursor = end + 2;
        } else {
            const end = raw.indexOf('$', start + 1);
            if (end === -1) {
                parts.push({ type: 'text', value: raw.slice(start) });
                break;
            }
            parts.push({ type: 'math', value: raw.slice(start + 1, end), display: false });
            cursor = end + 1;
        }
    }

    return parts.map((part) => {
        if (part.type === 'text') {
            return escapeHtml(part.value).replace(/\n/g, '<br />');
        }
        try {
            return katex.renderToString(part.value, { throwOnError: false, displayMode: part.display });
        } catch (_) {
            const fallback = part.display ? `$$${part.value}$$` : `$${part.value}$`;
            return escapeHtml(fallback);
        }
    }).join('');
}

function setDescriptionContent(element, text) {
    if (!element) return;
    element.innerHTML = renderDescriptionHtml(text || '');
}

function isVisibleForContextCopy(element) {
    if (!element || element.hidden) return false;
    if (typeof window !== 'undefined' && typeof window.getComputedStyle === 'function') {
        const style = window.getComputedStyle(element);
        if (style.display === 'none' || style.visibility === 'hidden') return false;
    }
    return true;
}

function collectVisibleContextText(root, { excludeSelectors = '' } = {}) {
    if (!root || typeof document === 'undefined' || typeof NodeFilter === 'undefined') return [];
    const lines = [];
    const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
    let node = walker.nextNode();
    while (node) {
        const parent = node.parentElement;
        if (!parent) {
            node = walker.nextNode();
            continue;
        }
        if (excludeSelectors && parent.closest(excludeSelectors)) {
            node = walker.nextNode();
            continue;
        }
        if (!isVisibleForContextCopy(parent)) {
            node = walker.nextNode();
            continue;
        }
        const line = String(node.textContent || '').replace(/\s+/g, ' ').trim();
        if (line && lines[lines.length - 1] !== line) {
            lines.push(line);
        }
        node = walker.nextNode();
    }
    return lines;
}

function fallbackCopyText(text) {
    if (typeof document === 'undefined' || !document.body) return false;
    const area = document.createElement('textarea');
    area.value = text;
    area.setAttribute('readonly', '');
    area.style.position = 'fixed';
    area.style.top = '-10000px';
    area.style.left = '-10000px';
    area.style.opacity = '0';
    document.body.appendChild(area);
    try {
        area.focus({ preventScroll: true });
    } catch (_) {
        area.focus();
    }
    area.select();
    area.setSelectionRange(0, area.value.length);
    let copied = false;
    try {
        copied = !!document.execCommand('copy');
    } catch (_) {
        copied = false;
    }
    document.body.removeChild(area);
    return copied;
}

async function copyTextToClipboard(text) {
    const value = String(text || '');
    if (!value.trim().length) return false;
    if (
        typeof navigator !== 'undefined'
        && navigator.clipboard
        && typeof navigator.clipboard.writeText === 'function'
        && (typeof window === 'undefined' || window.isSecureContext)
    ) {
        try {
            await navigator.clipboard.writeText(value);
            return true;
        } catch (_) {
            // Fall back to execCommand copy path below.
        }
    }
    return fallbackCopyText(value);
}

function extractTokenText(label) {
    if (!label) return '';
    const match = label.match(/^(token|position)\s*:\s*(.*)$/i);
    if (!match) {
        const trimmed = label.trim();
        return trimmed.length ? trimmed : SPACE_TOKEN_DISPLAY;
    }
    const extracted = match[2] || '';
    const trimmed = extracted.trim();
    return trimmed.length ? trimmed : SPACE_TOKEN_DISPLAY;
}

function sanitizeLogitTokenForPreview(token) {
    if (token === null || token === undefined) return '';
    const raw = String(token);
    if (!raw.length) return '';
    return raw.replace(/\n/g, '\\n').replace(/\t/g, '\\t');
}

function resolveLogitSelectionEntry(selectionInfo) {
    const directInfo = selectionInfo?.info;
    if (directInfo && typeof directInfo === 'object') {
        if (
            typeof directInfo.token === 'string'
            || Number.isFinite(directInfo.token_id)
            || Number.isFinite(directInfo.prob)
            || Number.isFinite(directInfo.logit)
        ) {
            return directInfo;
        }
        if (directInfo.logitEntry && typeof directInfo.logitEntry === 'object') {
            return directInfo.logitEntry;
        }
    }
    const source = selectionInfo?.object || selectionInfo?.hit?.object;
    const instanceId = selectionInfo?.hit?.instanceId;
    const entries = source?.userData?.instanceEntries;
    if (Array.isArray(entries) && Number.isFinite(instanceId) && instanceId >= 0 && instanceId < entries.length) {
        const entry = entries[instanceId];
        return entry && typeof entry === 'object' ? entry : null;
    }
    return null;
}

function resolveLogitPreviewTokenText(label, selectionInfo) {
    const entry = resolveLogitSelectionEntry(selectionInfo);
    if (typeof entry?.token === 'string') {
        const formatted = formatTokenLabelForPreview(sanitizeLogitTokenForPreview(entry.token));
        if (formatted) return formatted;
    }
    if (Number.isFinite(entry?.token_id)) return `#${Math.floor(entry.token_id)}`;
    const labelTokenMatch = String(label || '').match(/token\s+"([^"]+)"/i);
    if (labelTokenMatch && labelTokenMatch[1]) {
        const formatted = formatTokenLabelForPreview(labelTokenMatch[1]);
        if (formatted) return formatted;
    }
    const labelIdMatch = String(label || '').match(/\bid\s+(-?\d+)/i);
    if (labelIdMatch) {
        const parsed = Number(labelIdMatch[1]);
        if (Number.isFinite(parsed)) return `#${Math.floor(parsed)}`;
    }
    return 'Logit token';
}

function resolveLogitPreviewColor(selectionInfo) {
    const source = selectionInfo?.object || selectionInfo?.hit?.object;
    const instanceId = selectionInfo?.hit?.instanceId;
    const baseColor = source?.material?.color;
    const color = (baseColor && baseColor.isColor)
        ? TMP_COLOR.copy(baseColor)
        : TMP_COLOR.set(baseColor ?? 0xffffff);
    if (
        source
        && source.isInstancedMesh
        && typeof source.getColorAt === 'function'
        && Number.isFinite(instanceId)
        && instanceId >= 0
    ) {
        try { source.getColorAt(instanceId, color); } catch (_) { /* fallback to material color */ }
    }
    return color.clone();
}

function resolveLogitSelectionTokenId(label, entry) {
    if (Number.isFinite(entry?.token_id)) return Math.floor(entry.token_id);
    const labelIdMatch = String(label || '').match(/\bid\s+(-?\d+)/i);
    if (!labelIdMatch) return null;
    const parsed = Number(labelIdMatch[1]);
    return Number.isFinite(parsed) ? Math.floor(parsed) : null;
}

function resolveLogitSelectionProbability(label, entry) {
    if (Number.isFinite(entry?.prob)) return Number(entry.prob);
    const labelProbMatch = String(label || '').match(/\bp\s+(-?\d*\.?\d+(?:e[-+]?\d+)?)\b/i);
    if (!labelProbMatch) return null;
    const parsed = Number(labelProbMatch[1]);
    return Number.isFinite(parsed) ? parsed : null;
}

function formatLogitProbability(value) {
    if (!Number.isFinite(value)) return '';
    const abs = Math.abs(value);
    if (abs > 0 && abs < 0.001) return value.toExponential(2);
    return value.toFixed(4).replace(/\.?0+$/, '');
}

function resolveLogitSelectionHeader(label, selectionInfo) {
    if (!isLogitBarSelection(label, selectionInfo)) return null;
    const lower = String(label || '').toLowerCase();
    const entry = resolveLogitSelectionEntry(selectionInfo);
    const hasEntryData = !!(
        entry
        && typeof entry === 'object'
        && (
            typeof entry.token === 'string'
            || Number.isFinite(entry.token_id)
            || Number.isFinite(entry.prob)
            || Number.isFinite(entry.logit)
        )
    );
    const isSingleLogitLabel = lower === 'logit' || lower.startsWith('logit ');
    if (!hasEntryData && !isSingleLogitLabel) return null;

    const tokenText = resolveLogitPreviewTokenText(label, selectionInfo);
    const tokenId = resolveLogitSelectionTokenId(label, entry);
    const probability = resolveLogitSelectionProbability(label, entry);
    const subtitleParts = [];
    if (Number.isFinite(tokenId)) subtitleParts.push(`ID ${tokenId}`);
    if (Number.isFinite(probability)) subtitleParts.push(`Probability ${formatLogitProbability(probability)}`);

    return {
        title: tokenText ? `Logit Token: ${tokenText}` : 'Logit Token',
        subtitle: subtitleParts.join(' • ')
    };
}

function createLogitTextPreviewShared(labelText, options = {}) {
    const rawText = (typeof labelText === 'string') ? labelText : '';
    const text = rawText.trim().length ? rawText : 'Logit token';
    const color = options.color instanceof THREE.Color
        ? options.color.clone()
        : new THREE.Color(options.color ?? 0xffffff);
    const opacity = Number.isFinite(options.opacity) ? THREE.MathUtils.clamp(options.opacity, 0.1, 1) : 1;
    const font = tokenChipFont;

    const group = new THREE.Group();
    let textGeo = null;
    let textFaceGeo = null;
    let textPlaneGeo = null;
    let textMat = null;
    let textTexture = null;
    let usedGeometry = false;

    if (font && text.trim().length) {
        const shapes = font.generateShapes(text, LOGIT_PREVIEW_TEXT_STYLE.size, 2);
        if (Array.isArray(shapes) && shapes.length) {
            textGeo = new THREE.ExtrudeGeometry(shapes, {
                depth: LOGIT_PREVIEW_TEXT_STYLE.depth,
                curveSegments: 4,
                bevelEnabled: false
            });
            textGeo.computeBoundingBox();
            textGeo.computeVertexNormals();
            textGeo.translate(0, 0, -LOGIT_PREVIEW_TEXT_STYLE.depth / 2);
            textGeo.computeBoundingBox();

            const bounds = textGeo.boundingBox;
            const width = bounds ? Math.max(1, bounds.max.x - bounds.min.x) : 1;
            const height = bounds ? Math.max(1, bounds.max.y - bounds.min.y) : 1;
            const centerX = bounds ? (bounds.min.x + bounds.max.x) / 2 : 0;
            const centerY = bounds ? (bounds.min.y + bounds.max.y) / 2 : 0;
            if (bounds) {
                textGeo.translate(-centerX, -centerY, 0);
            }

            textMat = new THREE.MeshStandardMaterial({
                color: color.clone(),
                roughness: 0.32,
                metalness: 0.12,
                emissive: color.clone().multiplyScalar(0.15),
                emissiveIntensity: 0.9,
                transparent: opacity < 1,
                opacity,
                side: THREE.DoubleSide
            });
            const textMesh = new THREE.Mesh(textGeo, textMat);
            group.add(textMesh);

            textFaceGeo = new THREE.ShapeGeometry(shapes);
            textFaceGeo.computeVertexNormals();
            if (bounds) {
                textFaceGeo.translate(-centerX, -centerY, 0);
            }
            const frontFace = new THREE.Mesh(textFaceGeo, textMat);
            frontFace.position.z = LOGIT_PREVIEW_TEXT_STYLE.depth / 2 + LOGIT_PREVIEW_TEXT_STYLE.faceOffset;
            const backFace = new THREE.Mesh(textFaceGeo, textMat);
            backFace.position.z = -LOGIT_PREVIEW_TEXT_STYLE.depth / 2 - LOGIT_PREVIEW_TEXT_STYLE.faceOffset;
            group.add(frontFace, backFace);

            const fitScaleRaw = Math.min(
                LOGIT_PREVIEW_TEXT_STYLE.fitWidth / width,
                LOGIT_PREVIEW_TEXT_STYLE.fitHeight / height
            );
            const fitScale = THREE.MathUtils.clamp(
                Number.isFinite(fitScaleRaw) ? fitScaleRaw : 1,
                LOGIT_PREVIEW_TEXT_STYLE.minScale,
                LOGIT_PREVIEW_TEXT_STYLE.maxScale
            );
            group.scale.setScalar(fitScale);
            usedGeometry = true;
        }
    }

    if (!usedGeometry) {
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        const fontSize = 160;
        ctx.font = `700 ${fontSize}px ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial`;
        const measuredWidth = Math.ceil(ctx.measureText(text).width);
        const measuredHeight = Math.ceil(fontSize * 1.15);
        canvas.width = Math.max(512, measuredWidth + 160);
        canvas.height = Math.max(256, measuredHeight + 120);
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillStyle = '#ffffff';
        ctx.font = `700 ${fontSize}px ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial`;
        ctx.fillText(text, canvas.width / 2, canvas.height / 2);

        textTexture = new THREE.CanvasTexture(canvas);
        textTexture.minFilter = THREE.LinearFilter;
        textTexture.magFilter = THREE.LinearFilter;
        textTexture.needsUpdate = true;

        const aspect = canvas.width / canvas.height;
        let planeHeight = LOGIT_PREVIEW_TEXT_STYLE.fitHeight * 0.7;
        let planeWidth = planeHeight * aspect;
        const maxWidth = LOGIT_PREVIEW_TEXT_STYLE.fitWidth * 0.85;
        if (planeWidth > maxWidth) {
            planeWidth = maxWidth;
            planeHeight = planeWidth / aspect;
        }
        textPlaneGeo = new THREE.PlaneGeometry(planeWidth, planeHeight);
        textMat = new THREE.MeshBasicMaterial({
            map: textTexture,
            color,
            transparent: true,
            opacity,
            side: THREE.DoubleSide
        });
        group.add(new THREE.Mesh(textPlaneGeo, textMat));
    }

    return {
        group,
        dispose: () => {
            if (textGeo) textGeo.dispose();
            if (textFaceGeo) textFaceGeo.dispose();
            if (textPlaneGeo) textPlaneGeo.dispose();
            if (textMat) textMat.dispose();
            if (textTexture) textTexture.dispose();
        }
    };
}

function buildLogitBarPreview(label, selectionInfo) {
    const tokenText = resolveLogitPreviewTokenText(label, selectionInfo);
    return buildTokenChipPreview(tokenText);
}

function buildRoundedRectShape(width, height, radius) {
    const clampedRadius = Math.max(0, Math.min(radius, Math.min(width, height) / 2 - 1));
    const halfW = width / 2;
    const halfH = height / 2;
    const shape = new THREE.Shape();
    shape.moveTo(-halfW + clampedRadius, -halfH);
    shape.lineTo(halfW - clampedRadius, -halfH);
    shape.quadraticCurveTo(halfW, -halfH, halfW, -halfH + clampedRadius);
    shape.lineTo(halfW, halfH - clampedRadius);
    shape.quadraticCurveTo(halfW, halfH, halfW - clampedRadius, halfH);
    shape.lineTo(-halfW + clampedRadius, halfH);
    shape.quadraticCurveTo(-halfW, halfH, -halfW, halfH - clampedRadius);
    shape.lineTo(-halfW, -halfH + clampedRadius);
    shape.quadraticCurveTo(-halfW, -halfH, -halfW + clampedRadius, -halfH);
    shape.closePath();
    return shape;
}

function createTokenChipShared(labelText) {
    const rawText = (typeof labelText === 'string') ? labelText : '';
    const text = rawText.trim().length ? rawText : SPACE_TOKEN_DISPLAY;
    const font = tokenChipFont;
    let textGeo = null;
    let textMat = null;
    let textCullMat = null;
    let textMesh = null;
    let textGroup = null;
    let textTexture = null;
    let textPlaneAspect = 1;
    let textShapes = null;
    let textDepth = 0;
    let textFaceGeo = null;
    let bounds = null;
    let textWidth = 0;
    let textHeight = 0;
    let useGeometryText = false;
    const capOffset = 0.05;

    if (font && text.trim().length) {
        const desiredDepth = Number.isFinite(TOKEN_CHIP_STYLE.textDepth) ? TOKEN_CHIP_STYLE.textDepth : 0;
        const chipDepth = Number.isFinite(TOKEN_CHIP_STYLE.depth) ? TOKEN_CHIP_STYLE.depth : desiredDepth;
        textDepth = Number.isFinite(chipDepth) ? chipDepth + capOffset * 2 : desiredDepth;
        textShapes = font.generateShapes(text, TOKEN_CHIP_STYLE.textSize, 2);
        textGeo = new THREE.ExtrudeGeometry(textShapes, {
            depth: textDepth,
            curveSegments: 4,
            bevelEnabled: false
        });
        textGeo.computeBoundingBox();
        textGeo.computeVertexNormals();
        const textBounds = textGeo.boundingBox;
        if (textBounds && Number.isFinite(textBounds.max.x) && Number.isFinite(textBounds.min.x)) {
            textWidth = Math.max(0, textBounds.max.x - textBounds.min.x);
            textHeight = Math.max(0, textBounds.max.y - textBounds.min.y);
        }
        textGeo.translate(0, 0, -textDepth / 2);
        textGeo.computeBoundingBox();
        bounds = textGeo.boundingBox;
        useGeometryText = true;
    } else {
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        const fontSize = TOKEN_CHIP_STYLE.textSize;
        ctx.font = `600 ${fontSize}px ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial`;
        const textMetrics = ctx.measureText(text);
        textWidth = Math.ceil(textMetrics.width);
        textHeight = Math.ceil(fontSize * 1.15);
        canvas.width = Math.max(256, textWidth + 80);
        canvas.height = Math.max(128, textHeight + 60);
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.fillStyle = '#ffffff';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.font = `600 ${fontSize}px ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial`;
        ctx.fillText(text, canvas.width / 2, canvas.height / 2);

        textTexture = new THREE.CanvasTexture(canvas);
        textTexture.minFilter = THREE.LinearFilter;
        textTexture.magFilter = THREE.LinearFilter;
        textTexture.needsUpdate = true;
        textPlaneAspect = canvas.width / canvas.height;
    }

    const chipWidth = Math.max(TOKEN_CHIP_STYLE.minWidth, textWidth + TOKEN_CHIP_STYLE.padding);
    const chipHeight = typeof TOKEN_CHIP_STYLE.height === 'number'
        ? TOKEN_CHIP_STYLE.height
        : Math.max(TOKEN_CHIP_STYLE.minHeight, textHeight + TOKEN_CHIP_STYLE.padding);
    const chipRadius = Math.min(TOKEN_CHIP_STYLE.cornerRadius, Math.min(chipWidth, chipHeight) / 2 - 1);
    const chipShape = buildRoundedRectShape(chipWidth, chipHeight, chipRadius);
    const chipGeo = new THREE.ExtrudeGeometry(chipShape, { depth: TOKEN_CHIP_STYLE.depth, bevelEnabled: false });
    chipGeo.translate(0, 0, -TOKEN_CHIP_STYLE.depth / 2);
    chipGeo.computeVertexNormals();

    const chipMat = new THREE.MeshStandardMaterial({
        color: 0xf2e8d5,
        roughness: 0.84,
        metalness: 0.01,
        emissive: 0x000000,
        emissiveIntensity: 0,
        side: THREE.DoubleSide
    });
    const chipMesh = new THREE.Mesh(chipGeo, chipMat);

    const capMat = chipMat.clone();
    capMat.polygonOffset = false;
    capMat.polygonOffsetFactor = 0;
    capMat.polygonOffsetUnits = 0;
    const capGeo = new THREE.ShapeGeometry(chipShape);
    capGeo.computeVertexNormals();
    const frontCap = new THREE.Mesh(capGeo, capMat);
    frontCap.position.z = TOKEN_CHIP_STYLE.depth / 2 + capOffset;
    const backCap = new THREE.Mesh(capGeo, capMat);
    backCap.position.z = -TOKEN_CHIP_STYLE.depth / 2 - capOffset;
    backCap.rotation.y = Math.PI;

    if (useGeometryText) {
        if (textGeo && textWidth > 0 && textHeight > 0) {
            textMat = new THREE.MeshBasicMaterial({
                color: 0xffffff,
                side: THREE.DoubleSide,
                depthWrite: true,
                depthTest: true,
                polygonOffset: true,
                polygonOffsetFactor: -0.5,
                polygonOffsetUnits: -0.5
            });
            textCullMat = textMat.clone();
            textCullMat.colorWrite = false;
            textCullMat.depthWrite = false;
            textCullMat.transparent = true;
            textCullMat.opacity = 0;
            textGroup = new THREE.Group();
            textMesh = new THREE.Mesh(textGeo, [textCullMat, textMat]);
            textGroup.add(textMesh);
            if (textShapes) {
                const faceGeo = new THREE.ShapeGeometry(textShapes);
                faceGeo.computeVertexNormals();
                textFaceGeo = faceGeo;
                const faceOffset = 0.02;
                const frontFace = new THREE.Mesh(faceGeo, textMat);
                frontFace.position.z = textDepth / 2 + faceOffset;
                const backFace = new THREE.Mesh(faceGeo, textMat);
                backFace.position.z = -textDepth / 2 - faceOffset;
                textGroup.add(frontFace, backFace);
            }
            if (bounds) {
                const centerX = (bounds.min.x + bounds.max.x) / 2;
                const centerY = (bounds.min.y + bounds.max.y) / 2;
                textGroup.position.set(-centerX, -centerY, 0);
            }
        }
    } else if (textTexture) {
        let textPlaneHeight = chipHeight * 0.38;
        let textPlaneWidth = textPlaneHeight * textPlaneAspect;
        const maxTextWidth = chipWidth * 0.8;
        if (textPlaneWidth > maxTextWidth) {
            textPlaneWidth = maxTextWidth;
            textPlaneHeight = textPlaneWidth / textPlaneAspect;
        }
        textGeo = new THREE.PlaneGeometry(textPlaneWidth, textPlaneHeight);
        textMat = new THREE.MeshBasicMaterial({
            map: textTexture,
            transparent: true,
            depthWrite: true,
            depthTest: true,
            polygonOffset: true,
            polygonOffsetFactor: -0.5,
            polygonOffsetUnits: -0.5,
            side: THREE.DoubleSide
        });
        textMesh = new THREE.Mesh(textGeo, textMat);
        textMesh.position.z = TOKEN_CHIP_STYLE.depth / 2 + TOKEN_CHIP_STYLE.textOffset;
    }

    const group = new THREE.Group();
    group.add(chipMesh, frontCap, backCap);
    if (textGroup) {
        group.add(textGroup);
    } else if (textMesh) {
        group.add(textMesh);
    }

    return {
        group,
        dispose: () => {
            chipGeo.dispose();
            chipMat.dispose();
            capGeo.dispose();
            capMat.dispose();
            if (textGeo) textGeo.dispose();
            if (textFaceGeo) textFaceGeo.dispose();
            if (textMat) textMat.dispose();
            if (textCullMat) textCullMat.dispose();
            if (textTexture) textTexture.dispose();
        }
    };
}

function buildTokenChipPreview(labelText) {
    const shared = createTokenChipShared(labelText);
    const group = new THREE.Group();
    const laneCount = Math.max(1, Math.floor(PREVIEW_TOKEN_LANES));
    for (let i = 0; i < laneCount; i++) {
        const chip = (i === 0) ? shared.group : shared.group.clone(true);
        chip.position.z = (i - (laneCount - 1) / 2) * PREVIEW_LANE_SPACING * 0.6;
        group.add(chip);
    }
    return { object: group, dispose: shared.dispose };
}

function buildWeightMatrixPreview(params, colorHex, options = {}) {
    const depth = PREVIEW_MATRIX_DEPTH;
    const slitCount = PREVIEW_SOLID_LANES;
    const useInstancedSlices = options.useInstancedSlices !== undefined
        ? options.useInstancedSlices
        : true;
    const matrix = new WeightMatrixVisualization(
        null,
        new THREE.Vector3(0, 0, 0),
        params.width,
        params.height,
        depth,
        params.topWidthFactor,
        params.cornerRadius,
        slitCount,
        params.slitWidth,
        params.slitDepthFactor,
        params.slitBottomWidthFactor,
        params.slitTopWidthFactor,
        useInstancedSlices
    );
    if (colorHex !== null && colorHex !== undefined) {
        matrix.setColor(new THREE.Color(colorHex));
    }
    matrix.setMaterialProperties({ opacity: 0.98, transparent: false, emissiveIntensity: 0.42 });
    return {
        object: matrix.group,
        dispose: () => {
            const meshes = [matrix.mesh, matrix.frontCapMesh, matrix.backCapMesh];
            meshes.forEach(mesh => {
                if (!mesh || !mesh.material) return;
                if (Array.isArray(mesh.material)) {
                    mesh.material.forEach(mat => mat && mat.dispose && mat.dispose());
                } else {
                    mesh.material.dispose();
                }
            });
        }
    };
}

function extractPreviewVectorData(selectionInfo) {
    const vectorRef = selectionInfo?.info?.vectorRef;
    const activationStage = getActivationDataFromSelection(selectionInfo)?.stage;
    const preferActivationValues = typeof activationStage === 'string' && activationStage.toLowerCase().startsWith('qkv.');
    const baseCandidates = [
        vectorRef?.rawData,
        vectorRef?.userData?.activationData?.values,
        selectionInfo?.info?.activationData?.values,
        selectionInfo?.object?.userData?.activationData?.values,
        selectionInfo?.hit?.object?.userData?.activationData?.values,
        selectionInfo?.info?.vectorData,
        selectionInfo?.info?.values
    ];
    const candidates = preferActivationValues
        ? [
            vectorRef?.userData?.activationData?.values,
            selectionInfo?.info?.activationData?.values,
            selectionInfo?.object?.userData?.activationData?.values,
            selectionInfo?.hit?.object?.userData?.activationData?.values,
            vectorRef?.rawData,
            selectionInfo?.info?.vectorData,
            selectionInfo?.info?.values
        ]
        : baseCandidates;
    for (const arr of candidates) {
        if ((Array.isArray(arr) || ArrayBuffer.isView(arr)) && arr.length > 0) {
            return Array.from(arr).map((v) => Number.isFinite(v) ? v : 0);
        }
    }
    return null;
}

function applyDataToPreviewVector(vec, data) {
    if (!vec || !Array.isArray(data) || data.length === 0) return;
    const raw = data.slice();
    vec.updateDataAndSnapVisuals(raw);
    const numKeyColors = Math.min(30, Math.max(2, raw.length));
    vec.updateKeyColorsFromData(raw, numKeyColors, null, raw);
}

function createPreviewVector(options = {}) {
    const { colorHex, data = null, instanceCount = PREVIEW_VECTOR_BODY_INSTANCES } = options;
    const seedData = (Array.isArray(data) && data.length > 0)
        ? data
        : new Array(Math.max(1, Math.floor(instanceCount || PREVIEW_VECTOR_BODY_INSTANCES))).fill(0);
    const vec = new VectorVisualizationInstancedPrism(seedData, new THREE.Vector3(0, 0, 0), 1, instanceCount);
    vec.numSubsections = 1;
    if (Array.isArray(data) && data.length > 0) {
        applyDataToPreviewVector(vec, data);
    } else {
        const color = new THREE.Color(colorHex || 0xffffff);
        vec.currentKeyColors = [color.clone(), color.clone()];
        vec.updateInstanceGeometryAndColors();
    }
    return vec;
}

function buildVectorPreview(colorHex, selectionInfo = null) {
    const group = new THREE.Group();
    const vectors = [];
    const data = extractPreviewVectorData(selectionInfo);
    for (let i = 0; i < PREVIEW_LANES; i++) {
        const vec = createPreviewVector({ colorHex, data });
        vec.group.position.z = (i - (PREVIEW_LANES - 1) / 2) * PREVIEW_LANE_SPACING;
        group.add(vec.group);
        vectors.push(vec);
    }
    return {
        object: group,
        dispose: () => {
            vectors.forEach(vec => {
                if (vec.mesh?.geometry) vec.mesh.geometry.dispose();
                if (vec.mesh?.material) vec.mesh.material.dispose();
            });
        }
    };
}

function createTrailLine(colorHex) {
    const points = [new THREE.Vector3(), new THREE.Vector3()];
    const geometry = new THREE.BufferGeometry().setFromPoints(points);
    const material = new THREE.LineBasicMaterial({
        color: colorHex || PREVIEW_TRAIL_COLOR,
        transparent: true,
        opacity: 0.85,
        depthWrite: false,
        depthTest: true
    });
    const line = new THREE.Line(geometry, material);
    return {
        line,
        update: (start, end, opacity = 1.0) => {
            const pos = geometry.attributes.position.array;
            pos[0] = start.x; pos[1] = start.y; pos[2] = start.z;
            pos[3] = end.x; pos[4] = end.y; pos[5] = end.z;
            geometry.attributes.position.needsUpdate = true;
            if (material) {
                material.opacity = THREE.MathUtils.clamp(opacity, 0, 1);
            }
        },
        dispose: () => {
            geometry.dispose();
            material.dispose();
        }
    };
}

function buildQkvFlowPreview(highlightType, selectionInfo = null) {
    // Animated depiction of post-LN vectors rising into the Q/K/V projection,
    // leaving a trail, and emitting a smaller head vector above.
    const group = new THREE.Group();
    const lanes = [];
    const laneSpacing = PREVIEW_QKV_LANE_SPACING;
    const totalDuration = PREVIEW_QKV_RISE_DURATION + PREVIEW_QKV_CONVERT_DURATION
        + PREVIEW_QKV_HOLD_DURATION + PREVIEW_QKV_EXIT_DURATION + PREVIEW_QKV_IDLE_DURATION;
    const startTime = performance.now();
    const neutralColor = 0xa7b3c2;
    const highlightKey = typeof highlightType === 'string' ? highlightType.toUpperCase() : '';
    const highlightScale = (type) => (highlightKey === type ? 1.1 : 1.0);
    const highlightColor = (type) => {
        if (type === 'Q') return MHA_FINAL_Q_COLOR;
        if (type === 'K') return MHA_FINAL_K_COLOR;
        return MHA_FINAL_V_COLOR;
    };
    const baseData = extractPreviewVectorData(selectionInfo);
    const headData = Array.isArray(baseData) && baseData.length > 0
        ? [baseData.reduce((sum, v) => sum + v, 0) / baseData.length]
        : baseData;

    for (let i = 0; i < PREVIEW_QKV_LANES; i++) {
        const z = (i - (PREVIEW_QKV_LANES - 1) / 2) * laneSpacing;
        const x = (highlightKey === 'Q') ? -PREVIEW_QKV_X_SPREAD : (highlightKey === 'V' ? PREVIEW_QKV_X_SPREAD : 0);
        const incoming = createPreviewVector({
            colorHex: neutralColor,
            data: baseData,
            instanceCount: PREVIEW_VECTOR_BODY_INSTANCES
        });
        const outgoing = createPreviewVector({
            colorHex: highlightColor(highlightKey || 'K'),
            data: headData,
            instanceCount: PREVIEW_VECTOR_HEAD_INSTANCES
        });
        const trail = createTrailLine(PREVIEW_TRAIL_COLOR);
        const trailStart = new THREE.Vector3(x, PREVIEW_QKV_START_Y, z);
        const trailEnd = new THREE.Vector3(x, PREVIEW_QKV_START_Y, z);

        incoming.group.position.set(x, PREVIEW_QKV_START_Y, z);
        outgoing.group.position.set(x, PREVIEW_QKV_START_Y, z);
        incoming.group.scale.setScalar(PREVIEW_VECTOR_LARGE_SCALE);
        outgoing.group.scale.setScalar(PREVIEW_VECTOR_SMALL_SCALE * highlightScale(highlightKey || 'K'));

        group.add(incoming.group, outgoing.group, trail.line);
        lanes.push({ incoming, outgoing, trail, trailStart, trailEnd, x, z });
        incoming.group.visible = false;
        outgoing.group.visible = false;
        trail.line.visible = false;
    }

    const disposeVector = (vec) => {
        if (vec?.dispose) vec.dispose();
    };

    const dispose = () => {
        lanes.forEach((lane) => {
            disposeVector(lane.incoming);
            disposeVector(lane.outgoing);
            if (lane.trail?.dispose) lane.trail.dispose();
        });
    };

    const updateLane = (lane, localTime) => {
        const { incoming, outgoing, trail, trailStart, trailEnd, x, z } = lane;
        const endRise = PREVIEW_QKV_RISE_DURATION;
        const endConvert = endRise + PREVIEW_QKV_CONVERT_DURATION;
        const endHold = endConvert + PREVIEW_QKV_HOLD_DURATION;
        const endExit = endHold + PREVIEW_QKV_EXIT_DURATION;

        if (localTime < 0) {
            incoming.group.visible = false;
            outgoing.group.visible = false;
            trail.line.visible = false;
            incoming.group.position.set(x, PREVIEW_QKV_START_Y, z);
            outgoing.group.position.set(x, PREVIEW_QKV_START_Y, z);
            return;
        }

        if (localTime <= endRise) {
            const t = easeInOutCubic(localTime / PREVIEW_QKV_RISE_DURATION);
            incoming.group.visible = true;
            outgoing.group.visible = false;
            trail.line.visible = true;
            const y = THREE.MathUtils.lerp(PREVIEW_QKV_START_Y, PREVIEW_QKV_MATRIX_Y, t);
            incoming.group.position.set(x, y, z);
            trailStart.set(x, PREVIEW_QKV_START_Y, z);
            trailEnd.set(x, y, z);
            trail.update(trailStart, trailEnd, 0.85);
            return;
        }

        if (localTime <= endConvert) {
            const t = easeInOutCubic((localTime - endRise) / PREVIEW_QKV_CONVERT_DURATION);
            incoming.group.visible = true;
            outgoing.group.visible = true;
            trail.line.visible = true;

            const incomingScale = THREE.MathUtils.lerp(PREVIEW_VECTOR_LARGE_SCALE, PREVIEW_VECTOR_LARGE_SCALE * 0.45, t);
            incoming.group.scale.setScalar(incomingScale);
            incoming.group.position.set(x, PREVIEW_QKV_MATRIX_Y, z);

            const y = THREE.MathUtils.lerp(PREVIEW_QKV_MATRIX_Y, PREVIEW_QKV_OUTPUT_Y, t);
            outgoing.group.position.set(x, y, z);
            trailStart.set(x, PREVIEW_QKV_START_Y, z);
            trailEnd.set(x, y, z);
            trail.update(trailStart, trailEnd, 1.0 - t * 0.4);
            return;
        }

        if (localTime <= endHold) {
            incoming.group.visible = false;
            outgoing.group.visible = true;
            trail.line.visible = true;
            outgoing.group.position.set(x, PREVIEW_QKV_OUTPUT_Y, z);
            trailStart.set(x, PREVIEW_QKV_START_Y, z);
            trailEnd.set(x, PREVIEW_QKV_OUTPUT_Y, z);
            trail.update(trailStart, trailEnd, 0.55);
            return;
        }

        if (localTime <= endExit) {
            incoming.group.visible = false;
            const t = easeInOutCubic((localTime - endHold) / PREVIEW_QKV_EXIT_DURATION);
            const y = THREE.MathUtils.lerp(PREVIEW_QKV_OUTPUT_Y, PREVIEW_QKV_EXIT_Y, t);
            outgoing.group.position.set(x, y, z);
            const visible = t < 0.96;
            outgoing.group.visible = visible;
            trailStart.set(x, PREVIEW_QKV_START_Y, z);
            trailEnd.set(x, y, z);
            trail.update(trailStart, trailEnd, visible ? 0.35 : 0.0);
            return;
        }

        incoming.group.visible = false;
        outgoing.group.visible = false;
        trail.line.visible = false;
        incoming.group.position.set(x, PREVIEW_QKV_START_Y, z);
        outgoing.group.position.set(x, PREVIEW_QKV_START_Y, z);
    };

    const animate = (_, nowMs) => {
        const elapsed = ((nowMs - startTime) % totalDuration + totalDuration) % totalDuration;
        lanes.forEach((lane, idx) => {
            const laneTime = elapsed - idx * PREVIEW_QKV_LANE_STAGGER;
            updateLane(lane, laneTime);
        });
    };

    return { object: group, dispose, animate };
}

function buildQkvMatrixFlowPreview(label, selectionInfo) {
    const type = inferQkvType(label, selectionInfo);
    const matrixColor = type === 'Q' ? MHA_FINAL_Q_COLOR : (type === 'V' ? MHA_FINAL_V_COLOR : MHA_FINAL_K_COLOR);
    const matrixPreview = buildWeightMatrixPreview(MHA_MATRIX_PARAMS, matrixColor);
    const x = (type === 'Q') ? -PREVIEW_QKV_X_SPREAD : (type === 'V' ? PREVIEW_QKV_X_SPREAD : 0);

    if (matrixPreview?.object) {
        matrixPreview.object.position.x = x;
        matrixPreview.object.position.y = PREVIEW_QKV_MATRIX_Y;
        return { ...matrixPreview, animate: null };
    }

    return buildStackedBoxPreview(matrixColor);
}

function buildStackedBoxPreview(colorHex) {
    const group = new THREE.Group();
    const geometry = new THREE.BoxGeometry(140, 140, 8);
    const meshes = [];
    for (let i = 0; i < PREVIEW_SOLID_LANES; i++) {
        const material = new THREE.MeshStandardMaterial({
            color: colorHex || 0x1f1f1f,
            metalness: 0.25,
            roughness: 0.65,
            emissive: new THREE.Color(0x060606),
            emissiveIntensity: 0.3
        });
        const mesh = new THREE.Mesh(geometry, material);
        mesh.position.z = (i - (PREVIEW_SOLID_LANES - 1) / 2) * 18;
        group.add(mesh);
        meshes.push(mesh);
    }
    return {
        object: group,
        dispose: () => {
            geometry.dispose();
            meshes.forEach(mesh => mesh.material && mesh.material.dispose());
        }
    };
}

function isVectorMeshCandidate(obj) {
    const geo = obj?.geometry;
    if (!obj || !(obj.isMesh || obj.isInstancedMesh) || !geo || typeof geo.getAttribute !== 'function') return false;
    if (obj.userData?.isVector) return true;
    return !!(
        geo.getAttribute('colorStart')
        || geo.getAttribute('colorEnd')
        || geo.getAttribute('instanceColor')
        || obj.instanceColor
    );
}

function findVectorLikeObject(selectionInfo) {
    const directRef = selectionInfo?.info?.vectorRef;
    if (directRef?.group?.isObject3D) return directRef.group;
    if (directRef?.isObject3D) return directRef;
    const sources = [];
    if (selectionInfo?.hit?.object) sources.push(selectionInfo.hit.object);
    if (selectionInfo?.object) sources.push(selectionInfo.object);
    if (!sources.length) return null;

    const findInHierarchy = (root) => {
        let found = null;
        root.traverse((child) => {
            if (found) return;
            if (isVectorMeshCandidate(child)) {
                found = child;
            }
        });
        return found;
    };

    let vectorMesh = null;
    for (const src of sources) {
        vectorMesh = findInHierarchy(src);
        if (vectorMesh) break;
    }
    if (!vectorMesh) return null;

    // Prefer the nearest ancestor labeled as a vector if present.
    let candidate = vectorMesh;
    let walker = vectorMesh.parent;
    while (walker && !walker.isScene) {
        const lbl = walker.userData?.label;
        if (lbl && String(lbl).toLowerCase().includes('vector')) {
            candidate = walker;
        }
        walker = walker.parent;
    }
    return candidate;
}

function findVectorSourceMesh(selectionInfo) {
    const vectorRef = selectionInfo?.info?.vectorRef;
    if (vectorRef?.mesh && isVectorMeshCandidate(vectorRef.mesh)) return vectorRef.mesh;
    const vectorObject = findVectorLikeObject(selectionInfo);
    if (!vectorObject) return null;
    if (isVectorMeshCandidate(vectorObject)) return vectorObject;
    let mesh = null;
    vectorObject.traverse((child) => {
        if (mesh) return;
        if (isVectorMeshCandidate(child)) mesh = child;
    });
    return mesh;
}

function resolveVectorPreviewColor(label, selectionInfo) {
    const category = selectionInfo?.info?.category;
    if (typeof category === 'string') {
        const cat = category.toUpperCase();
        if (cat === 'Q') return MHA_FINAL_Q_COLOR;
        if (cat === 'K') return MHA_FINAL_K_COLOR;
        if (cat === 'V') return MHA_FINAL_V_COLOR;
    }
    const type = inferQkvType(label || '', selectionInfo);
    if (type === 'Q') return MHA_FINAL_Q_COLOR;
    if (type === 'K') return MHA_FINAL_K_COLOR;
    if (type === 'V') return MHA_FINAL_V_COLOR;
    const snapshot = extractMaterialSnapshot(selectionInfo);
    if (snapshot?.color) return snapshot.color;
    return resolveFinalPreviewColor(label || '');
}

function resolveVectorPreviewInstanceCount(selectionInfo, label = '') {
    const vectorRef = selectionInfo?.info?.vectorRef;
    // In KV-cache decode we can receive Q/K/V selections without a vectorRef.
    // In that case, avoid using the whole source mesh count (often full-width)
    // and match the non-KV behavior: preview only the per-head vector width.
    if (!vectorRef && isAttentionHeadVectorSelection(label, selectionInfo)) {
        const headLength = Math.max(1, Math.floor(resolveVectorLength(label, selectionInfo) || D_HEAD));
        return Math.max(1, Math.ceil(headLength / PRISM_DIMENSIONS_PER_UNIT));
    }
    const candidates = [
        vectorRef?.instanceCount,
        vectorRef?._batch?.prismCount,
        vectorRef?.mesh?.count,
        vectorRef?.mesh?.instanceMatrix?.count
    ];
    for (const count of candidates) {
        if (Number.isFinite(count) && count > 0) return Math.max(1, Math.floor(count));
    }
    const vectorMesh = findVectorSourceMesh(selectionInfo);
    if (vectorMesh?.isInstancedMesh) {
        const meshCount = Number.isFinite(vectorMesh.count)
            ? vectorMesh.count
            : vectorMesh.instanceMatrix?.count;
        if (Number.isFinite(meshCount) && meshCount > 0) {
            return Math.max(1, Math.floor(meshCount));
        }
    }
    return PREVIEW_VECTOR_BODY_INSTANCES;
}

function extractMaterialSnapshot(selectionInfo) {
    const root = selectionInfo?.object || selectionInfo?.hit?.object;
    if (!root) return null;
    let material = null;
    let current = root;
    while (current && !material) {
        if (current.material) {
            material = Array.isArray(current.material) ? current.material.find(Boolean) : current.material;
        }
        current = current.parent;
    }
    if (!material) return null;
    return {
        color: material.color ? material.color.clone() : null,
        emissive: material.emissive && material.emissive.clone ? material.emissive.clone() : null,
        emissiveIntensity: material.emissiveIntensity,
        opacity: material.opacity,
        transparent: material.transparent,
        metalness: material.metalness,
        roughness: material.roughness,
        clearcoat: material.clearcoat,
        clearcoatRoughness: material.clearcoatRoughness,
        transmission: material.transmission,
        thickness: material.thickness,
        iridescence: material.iridescence,
        sheen: material.sheen,
        sheenColor: material.sheenColor && material.sheenColor.clone ? material.sheenColor.clone() : material.sheenColor,
        envMapIntensity: material.envMapIntensity
    };
}

function applyMaterialSnapshot(object, snapshot) {
    if (!object || !snapshot) return;
    object.traverse((child) => {
        if (!child.material) return;
        const materials = Array.isArray(child.material) ? child.material : [child.material];
        materials.forEach((mat) => {
            if (!mat) return;
            if (snapshot.color) {
                if (mat.userData?.sciFiUniforms) {
                    updateSciFiMaterialColor(mat, snapshot.color);
                } else if (mat.color?.copy) {
                    mat.color.copy(snapshot.color);
                }
            }
            if (snapshot.emissive && mat.emissive?.copy) {
                mat.emissive.copy(snapshot.emissive);
            }
            if (Number.isFinite(snapshot.emissiveIntensity)) mat.emissiveIntensity = snapshot.emissiveIntensity;
            if (Number.isFinite(snapshot.opacity)) mat.opacity = snapshot.opacity;
            if (typeof snapshot.transparent === 'boolean') mat.transparent = snapshot.transparent;
            if (Number.isFinite(snapshot.metalness)) mat.metalness = snapshot.metalness;
            if (Number.isFinite(snapshot.roughness)) mat.roughness = snapshot.roughness;
            if (Number.isFinite(snapshot.clearcoat)) mat.clearcoat = snapshot.clearcoat;
            if (Number.isFinite(snapshot.clearcoatRoughness)) mat.clearcoatRoughness = snapshot.clearcoatRoughness;
            if (Number.isFinite(snapshot.transmission)) mat.transmission = snapshot.transmission;
            if (Number.isFinite(snapshot.thickness)) mat.thickness = snapshot.thickness;
            if (Number.isFinite(snapshot.iridescence)) mat.iridescence = snapshot.iridescence;
            if (Number.isFinite(snapshot.sheen)) mat.sheen = snapshot.sheen;
            if (snapshot.sheenColor && mat.sheenColor && mat.sheenColor.copy) {
                mat.sheenColor.copy(snapshot.sheenColor);
            }
            if (Number.isFinite(snapshot.envMapIntensity)) mat.envMapIntensity = snapshot.envMapIntensity;
        });
    });
}

function copyInstancedVectorSliceToPreview(previewVec, sourceMesh, sourceOffset = 0, sourceCount = null) {
    if (!previewVec?.mesh || !sourceMesh?.isInstancedMesh || typeof sourceMesh.getMatrixAt !== 'function') {
        return false;
    }
    const dstMesh = previewVec.mesh;
    const dstCount = Number.isFinite(previewVec.instanceCount)
        ? Math.max(1, Math.floor(previewVec.instanceCount))
        : 0;
    const srcTotal = Number.isFinite(sourceMesh.count)
        ? Math.max(0, Math.floor(sourceMesh.count))
        : Math.max(0, Math.floor(sourceMesh.instanceMatrix?.count || 0));
    const start = Math.max(0, Math.floor(sourceOffset || 0));
    const available = Math.max(0, srcTotal - start);
    const requested = Number.isFinite(sourceCount)
        ? Math.max(0, Math.floor(sourceCount))
        : dstCount;
    const copyCount = Math.min(dstCount, available, requested);
    if (copyCount <= 0) return false;

    for (let i = 0; i < copyCount; i += 1) {
        sourceMesh.getMatrixAt(start + i, TMP_MATRIX);
        dstMesh.setMatrixAt(i, TMP_MATRIX);
    }
    for (let i = copyCount; i < dstCount; i += 1) {
        TMP_MATRIX.makeScale(0.001, 0.001, 0.001);
        TMP_MATRIX.setPosition(0, HIDE_INSTANCE_Y_OFFSET, 0);
        dstMesh.setMatrixAt(i, TMP_MATRIX);
    }
    dstMesh.instanceMatrix.needsUpdate = true;

    if (sourceMesh.instanceColor?.array && dstMesh.instanceColor?.array) {
        const srcColors = sourceMesh.instanceColor.array;
        const dstColors = dstMesh.instanceColor.array;
        const srcStart = start * 3;
        const maxCopy = Math.min(copyCount * 3, srcColors.length - srcStart, dstColors.length);
        if (maxCopy > 0) {
            dstColors.set(srcColors.subarray(srcStart, srcStart + maxCopy), 0);
            dstMesh.instanceColor.needsUpdate = true;
        }
    }

    const copyAttr = (name) => {
        const srcAttr = sourceMesh.geometry?.getAttribute?.(name);
        const dstAttr = dstMesh.geometry?.getAttribute?.(name);
        if (!srcAttr?.array || !dstAttr?.array) return;
        const srcStart = start * 3;
        const maxCopy = Math.min(copyCount * 3, srcAttr.array.length - srcStart, dstAttr.array.length);
        if (maxCopy <= 0) return;
        dstAttr.array.set(srcAttr.array.subarray(srcStart, srcStart + maxCopy), 0);
        dstAttr.needsUpdate = true;
    };
    copyAttr('colorStart');
    copyAttr('colorEnd');
    return true;
}

function copyInstancedVectorColorsToPreview(previewVec, sourceMesh, sourceOffset = 0, sourceCount = null) {
    if (!previewVec?.mesh || !sourceMesh?.isInstancedMesh) return false;
    const dstMesh = previewVec.mesh;
    const dstCount = Number.isFinite(previewVec.instanceCount)
        ? Math.max(1, Math.floor(previewVec.instanceCount))
        : 0;
    const srcTotal = Number.isFinite(sourceMesh.count)
        ? Math.max(0, Math.floor(sourceMesh.count))
        : Math.max(0, Math.floor(sourceMesh.instanceMatrix?.count || 0));
    const start = Math.max(0, Math.floor(sourceOffset || 0));
    const available = Math.max(0, srcTotal - start);
    const requested = Number.isFinite(sourceCount)
        ? Math.max(0, Math.floor(sourceCount))
        : dstCount;
    const copyCount = Math.min(dstCount, available, requested);
    if (copyCount <= 0) return false;

    let copied = false;
    if (sourceMesh.instanceColor?.array && dstMesh.instanceColor?.array) {
        const srcColors = sourceMesh.instanceColor.array;
        const dstColors = dstMesh.instanceColor.array;
        const srcStart = start * 3;
        const maxCopy = Math.min(copyCount * 3, srcColors.length - srcStart, dstColors.length);
        if (maxCopy > 0) {
            dstColors.set(srcColors.subarray(srcStart, srcStart + maxCopy), 0);
            dstMesh.instanceColor.needsUpdate = true;
            copied = true;
        }
    }

    const copyAttr = (name) => {
        const srcAttr = sourceMesh.geometry?.getAttribute?.(name);
        const dstAttr = dstMesh.geometry?.getAttribute?.(name);
        if (!srcAttr?.array || !dstAttr?.array) return;
        const srcStart = start * 3;
        const maxCopy = Math.min(copyCount * 3, srcAttr.array.length - srcStart, dstAttr.array.length);
        if (maxCopy <= 0) return;
        dstAttr.array.set(srcAttr.array.subarray(srcStart, srcStart + maxCopy), 0);
        dstAttr.needsUpdate = true;
        copied = true;
    };
    copyAttr('colorStart');
    copyAttr('colorEnd');
    return copied;
}

function isInstancedVectorSliceInMotion(sourceMesh, sourceOffset = 0, sourceCount = null) {
    if (!sourceMesh?.isInstancedMesh || typeof sourceMesh.getMatrixAt !== 'function') {
        return false;
    }
    const srcTotal = Number.isFinite(sourceMesh.count)
        ? Math.max(0, Math.floor(sourceMesh.count))
        : Math.max(0, Math.floor(sourceMesh.instanceMatrix?.count || 0));
    const start = Math.max(0, Math.floor(sourceOffset || 0));
    const available = Math.max(0, srcTotal - start);
    const requested = Number.isFinite(sourceCount)
        ? Math.max(0, Math.floor(sourceCount))
        : available;
    const inspectCount = Math.min(available, requested);
    if (inspectCount <= 1) return false;

    let baselineY = null;
    let visibleCount = 0;
    let hiddenCount = 0;
    for (let i = 0; i < inspectCount; i += 1) {
        sourceMesh.getMatrixAt(start + i, TMP_MATRIX);
        TMP_MATRIX.decompose(TMP_POS, TMP_QUAT, TMP_SCALE);
        const hidden = TMP_POS.y <= HIDE_INSTANCE_Y_OFFSET * 0.5
            || TMP_SCALE.x < 0.01
            || TMP_SCALE.y < 0.01
            || TMP_SCALE.z < 0.01;
        if (hidden) {
            hiddenCount += 1;
            continue;
        }
        if (!Number.isFinite(TMP_POS.y)) continue;

        visibleCount += 1;
        if (baselineY === null) {
            baselineY = TMP_POS.y;
            continue;
        }

        // In stable vectors, all visible prisms share the same local Y.
        // Mid-addition vectors have per-prism Y offsets and should not be copied.
        if (Math.abs(TMP_POS.y - baselineY) > 0.25) {
            return true;
        }
    }

    if (hiddenCount > 0 && visibleCount > 0) {
        return true;
    }
    return false;
}

function shouldSkipLiveVectorTransformCopy(vectorRef, vectorMesh, fallbackCount = null, options = {}) {
    if (options?.forceLiveCopy === true) return false;
    if (vectorRef?.userData?.qkvProcessed === true) return false;

    if (vectorRef?.isBatchedVectorRef && vectorRef._batch?.mesh) {
        const batch = vectorRef._batch;
        const batchPrismCount = Number.isFinite(batch.prismCount)
            ? Math.max(1, Math.floor(batch.prismCount))
            : Number.isFinite(fallbackCount)
                ? Math.max(1, Math.floor(fallbackCount))
                : PREVIEW_VECTOR_BODY_INSTANCES;
        const index = Number.isFinite(vectorRef._index) ? Math.max(0, Math.floor(vectorRef._index)) : 0;
        if (isInstancedVectorSliceInMotion(batch.mesh, index * batchPrismCount, batchPrismCount)) {
            return true;
        }
    }

    if (vectorRef?.mesh?.isInstancedMesh) {
        const srcCount = Number.isFinite(vectorRef.instanceCount)
            ? Math.max(1, Math.floor(vectorRef.instanceCount))
            : Number.isFinite(fallbackCount)
                ? Math.max(1, Math.floor(fallbackCount))
                : PREVIEW_VECTOR_BODY_INSTANCES;
        if (isInstancedVectorSliceInMotion(vectorRef.mesh, 0, srcCount)) {
            return true;
        }
    }

    if (vectorMesh?.isInstancedMesh) {
        const inspectCount = Number.isFinite(fallbackCount)
            ? Math.max(1, Math.floor(fallbackCount))
            : null;
        if (isInstancedVectorSliceInMotion(vectorMesh, 0, inspectCount)) {
            return true;
        }
    }

    return false;
}

function tryCopyVectorAppearanceToPreview(vec, selectionInfo, vectorRef, vectorMesh, options = {}) {
    if (!vec || !vec.mesh) return false;
    if (shouldSkipLiveVectorTransformCopy(vectorRef, vectorMesh, vec.instanceCount, options)) {
        return false;
    }
    let copied = false;

    if (!copied && vectorRef?.isBatchedVectorRef && vectorRef._batch?.mesh) {
        const batch = vectorRef._batch;
        const batchPrismCount = Number.isFinite(batch.prismCount)
            ? Math.max(1, Math.floor(batch.prismCount))
            : vec.instanceCount;
        const index = Number.isFinite(vectorRef._index) ? Math.max(0, Math.floor(vectorRef._index)) : 0;
        copied = copyInstancedVectorSliceToPreview(vec, batch.mesh, index * batchPrismCount, batchPrismCount);
    }

    if (!copied && vectorRef?.mesh?.isInstancedMesh) {
        const srcCount = Number.isFinite(vectorRef.instanceCount)
            ? Math.max(1, Math.floor(vectorRef.instanceCount))
            : undefined;
        copied = copyInstancedVectorSliceToPreview(vec, vectorRef.mesh, 0, srcCount);
    }

    if (!copied && vectorMesh?.isInstancedMesh) {
        copied = copyInstancedVectorSliceToPreview(vec, vectorMesh, 0, vec.instanceCount);
    }

    if (!copied) return false;

    if (Array.isArray(vectorRef?.currentKeyColors) && vectorRef.currentKeyColors.length >= 2) {
        vec.currentKeyColors = vectorRef.currentKeyColors
            .map((color) => (color?.isColor ? color.clone() : new THREE.Color(color)))
            .filter((color) => color?.isColor);
        vec.numSubsections = Math.max(1, vec.currentKeyColors.length - 1);
    }

    const snapshot = extractMaterialSnapshot(selectionInfo);
    if (snapshot) {
        applyMaterialSnapshot(vec.group, snapshot);
    }
    return true;
}

function buildVectorClonePreview(selectionInfo, label = '') {
    const weightedSumSelection = isWeightedSumSelection(label, selectionInfo);
    const kvCacheVectorSelection = isKvCacheVectorSelection(selectionInfo);
    if (weightedSumSelection) {
        // Use the exact runtime vector geometry for weighted-sum selections.
        const directClone = buildDirectClonePreview(selectionInfo)
            || buildSelectionClonePreview(selectionInfo, label);
        if (directClone) return directClone;
    }

    const vectorRef = selectionInfo?.info?.vectorRef || null;
    const vectorMesh = findVectorSourceMesh(selectionInfo);
    if (!vectorRef && !vectorMesh) return null;

    const prismCount = resolveVectorPreviewInstanceCount(selectionInfo, label);
    const vec = createPreviewVector({
        colorHex: resolveVectorPreviewColor(label, selectionInfo),
        data: null,
        instanceCount: prismCount
    });

    const forceHeadDataFallback = !vectorRef && isAttentionHeadVectorSelection(label, selectionInfo);
    const copiedAppearance = forceHeadDataFallback
        ? false
        : tryCopyVectorAppearanceToPreview(vec, selectionInfo, vectorRef, vectorMesh, {
            forceLiveCopy: weightedSumSelection || kvCacheVectorSelection
        });
    if (!copiedAppearance) {
        const data = extractPreviewVectorData(selectionInfo);
        if (isQkvHeadVectorSelection(label, selectionInfo)) {
            const outputLength = Math.max(1, Math.floor(resolveVectorLength(label, selectionInfo) || D_HEAD));
            const processedData = (Array.isArray(data) && data.length > 0)
                ? data.slice(0, outputLength)
                : [0];
            const numKeyColors = Math.min(30, Math.max(2, processedData.length));
            vec.applyProcessedVisuals(
                processedData,
                outputLength,
                { numKeyColors, generationOptions: null },
                { setHiddenToBlack: false, hideByScaleOnly: true },
                processedData
            );
            const colorSourceMesh = (vectorRef?.mesh?.isInstancedMesh ? vectorRef.mesh : null)
                || (vectorMesh?.isInstancedMesh ? vectorMesh : null);
            if (colorSourceMesh) {
                const sourceOffset = (!vectorRef
                    && Number.isFinite(selectionInfo?.hit?.instanceId)
                    && Number.isFinite(colorSourceMesh.count)
                    && colorSourceMesh.count > vec.instanceCount)
                    ? Math.max(
                        0,
                        Math.floor(selectionInfo.hit.instanceId / Math.max(1, vec.instanceCount)) * Math.max(1, vec.instanceCount)
                    )
                    : 0;
                copyInstancedVectorColorsToPreview(vec, colorSourceMesh, sourceOffset, vec.instanceCount);
            }
        } else if (Array.isArray(data) && data.length > 0) {
            applyDataToPreviewVector(vec, data);
        } else if (Array.isArray(vectorRef?.currentKeyColors)) {
            const keyColors = vectorRef.currentKeyColors
                .map((color) => (color?.isColor ? color.clone() : new THREE.Color(color)))
                .filter((color) => color?.isColor);
            if (keyColors.length >= 2) {
                vec.currentKeyColors = keyColors;
                vec.numSubsections = keyColors.length - 1;
                vec.updateInstanceGeometryAndColors();
            }
        }
    }

    return {
        object: vec.group,
        dispose: () => {
            vec.dispose();
        }
    };
}

function buildAttentionSpherePreview(selectionInfo) {
    const hit = selectionInfo?.hit || null;
    const source = selectionInfo?.object || hit?.object || null;
    if (!source || !source.isInstancedMesh) return null;
    if (!source.userData?._attentionSphereInstanced && selectionInfo?.kind !== 'attentionSphere') return null;
    const instanceId = hit && typeof hit.instanceId === 'number' ? hit.instanceId : null;
    if (!Number.isFinite(instanceId)) return null;
    if (!source.geometry || typeof source.geometry.clone !== 'function') return null;

    const geometry = source.geometry.clone();
    const color = TMP_COLOR.copy(source.material?.color || 0xffffff);
    if (typeof source.getColorAt === 'function') {
        try { source.getColorAt(instanceId, color); } catch (_) { /* fallback to material color */ }
    }
    let instanceScale = 1;
    if (typeof source.getMatrixAt === 'function') {
        try {
            source.getMatrixAt(instanceId, TMP_MATRIX);
            TMP_MATRIX.decompose(TMP_POS, TMP_QUAT, TMP_SCALE);
            if (Number.isFinite(TMP_SCALE.x)) {
                instanceScale = Math.max(TMP_SCALE.x, TMP_SCALE.y, TMP_SCALE.z);
            }
        } catch (_) { /* ignore */ }
    }
    if (!Number.isFinite(instanceScale) || instanceScale < 0.1) instanceScale = 0.6;

    const material = new THREE.MeshStandardMaterial({
        color: color.clone(),
        roughness: 0.35,
        metalness: 0.1,
        emissive: color.clone().multiplyScalar(0.35),
        emissiveIntensity: 0.35
    });
    const mesh = new THREE.Mesh(geometry, material);
    mesh.scale.setScalar(instanceScale);
    return {
        object: mesh,
        dispose: () => {
            geometry.dispose();
            material.dispose();
        }
    };
}

function buildLayerNormPreview(label, selectionInfo) {
    const clonePreview = buildSelectionClonePreview(selectionInfo, label)
        || buildDirectClonePreview(selectionInfo);
    if (clonePreview) {
        applyFinalColorToObject(clonePreview.object, 0xffffff);
        return clonePreview;
    }

    const baseHoles = Number.isFinite(LN_PARAMS.numberOfHoles) ? LN_PARAMS.numberOfHoles : PREVIEW_SOLID_LANES;
    const depthScale = PREVIEW_SOLID_LANES / Math.max(1, baseHoles);
    const previewDepth = Math.max(
        180,
        Math.min(PREVIEW_MATRIX_DEPTH, LN_PARAMS.depth * depthScale)
    );
    const params = { ...LN_PARAMS, numberOfHoles: PREVIEW_SOLID_LANES, depth: previewDepth };
    const ln = new LayerNormalizationVisualization(
        new THREE.Vector3(0, 0, 0),
        params.width,
        params.height,
        params.depth,
        params.wallThickness,
        params.numberOfHoles,
        params.holeWidth,
        params.holeWidthFactor,
        undefined,
        true
    );
    ln.setColor(new THREE.Color(0xffffff));
    return {
        object: ln.group,
        dispose: () => ln.dispose()
    };
}

function isLikelyVectorSelection(label, selectionInfo) {
    const lower = (label || '').toLowerCase();
    if (lower.includes('vector') || lower.includes('residual')) return true;
    const cat = selectionInfo?.info?.category;
    if (cat && ['q', 'k', 'v', 'vector', 'residual'].includes(String(cat).toLowerCase())) return true;
    const kind = selectionInfo?.kind;
    if (kind && ['vector', 'residual', 'mergedkv'].includes(String(kind).toLowerCase())) return true;
    if (findVectorLikeObject(selectionInfo)) return true;
    return false;
}

function resolvePreviewObject(label, selectionInfo) {
    const lower = (label || '').toLowerCase();
    if (isLogitBarSelection(label, selectionInfo)) {
        return buildLogitBarPreview(label, selectionInfo);
    }
    const attentionSpherePreview = buildAttentionSpherePreview(selectionInfo);
    if (attentionSpherePreview) return attentionSpherePreview;
    const isVectorSelection = isLikelyVectorSelection(label, selectionInfo);
    if (isVectorSelection) {
        const vectorClone = buildVectorClonePreview(selectionInfo, label);
        if (vectorClone) return vectorClone;
    }
    if (isQkvMatrixLabel(lower)) {
        const type = inferQkvType(label, selectionInfo);
        const matrixColor = type === 'Q' ? MHA_FINAL_Q_COLOR : (type === 'V' ? MHA_FINAL_V_COLOR : MHA_FINAL_K_COLOR);
        const clonePreview = buildSelectionClonePreview(selectionInfo, label);
        if (clonePreview) return clonePreview;
        const preview = buildWeightMatrixPreview(MHA_MATRIX_PARAMS, matrixColor);
        const snapshot = extractMaterialSnapshot(selectionInfo);
        if (snapshot) {
            applyMaterialSnapshot(preview.object, snapshot);
            applyFinalColorToObject(preview.object, matrixColor);
        }
        return preview;
    }
    const clonePreview = buildSelectionClonePreview(selectionInfo, label)
        || buildDirectClonePreview(selectionInfo);
    if (clonePreview) {
        if (isLayerNormLabel(label)) {
            applyFinalColorToObject(clonePreview.object, 0xffffff);
        }
        return clonePreview;
    }
    if (lower.startsWith('token:') || lower.startsWith('position:')) {
        const clonePreview = buildSelectionClonePreview(selectionInfo, label);
        if (clonePreview) return clonePreview;
        return buildTokenChipPreview(extractTokenText(label));
    }
    if (lower.includes('output projection matrix')) {
        const height = MHA_MATRIX_PARAMS.height * MHA_OUTPUT_PROJECTION_MATRIX_PARAMS.heightFactor;
        const params = {
            ...MHA_MATRIX_PARAMS,
            width: MHA_OUTPUT_PROJECTION_MATRIX_PARAMS.width,
            height,
            topWidthFactor: MHA_OUTPUT_PROJECTION_MATRIX_PARAMS.topWidthFactor,
            cornerRadius: MHA_OUTPUT_PROJECTION_MATRIX_PARAMS.cornerRadius,
            slitWidth: MHA_OUTPUT_PROJECTION_MATRIX_PARAMS.slitWidth,
            slitDepthFactor: MHA_OUTPUT_PROJECTION_MATRIX_PARAMS.slitDepthFactor,
            slitBottomWidthFactor: MHA_OUTPUT_PROJECTION_MATRIX_PARAMS.slitBottomWidthFactor,
            slitTopWidthFactor: MHA_OUTPUT_PROJECTION_MATRIX_PARAMS.slitTopWidthFactor
        };
        return buildWeightMatrixPreview(params, MHA_OUTPUT_PROJECTION_MATRIX_COLOR);
    }
    if (lower.includes('mlp up weight matrix')) {
        return buildWeightMatrixPreview(MLP_MATRIX_PARAMS_UP, FINAL_MLP_COLOR);
    }
    if (lower.includes('mlp down weight matrix')) {
        return buildWeightMatrixPreview(MLP_MATRIX_PARAMS_DOWN, FINAL_MLP_COLOR);
    }
    if (lower.includes('vocab embedding') || lower.includes('unembedding')) {
        const clonePreview = buildSelectionClonePreview(selectionInfo, label)
            || buildDirectClonePreview(selectionInfo);
        if (clonePreview) {
            const color = lower.includes('top') ? FINAL_VOCAB_TOP_COLOR : MHA_FINAL_Q_COLOR;
            applyFinalColorToObject(clonePreview.object, color);
            return clonePreview;
        }
        const color = lower.includes('top') ? FINAL_VOCAB_TOP_COLOR : MHA_FINAL_Q_COLOR;
        return buildWeightMatrixPreview(EMBEDDING_MATRIX_PARAMS_VOCAB, color);
    }
    if (lower.includes('positional embedding')) {
        const clonePreview = buildSelectionClonePreview(selectionInfo, label)
            || buildDirectClonePreview(selectionInfo);
        if (clonePreview) return clonePreview;
        return buildWeightMatrixPreview(EMBEDDING_MATRIX_PARAMS_POSITION, POSITION_EMBED_COLOR);
    }
    if (isWeightMatrixLabel(lower)) {
        const color = resolveFinalPreviewColor(label);
        return buildWeightMatrixPreview(MHA_MATRIX_PARAMS, color);
    }

    if (lower.includes('query vector')) {
        return buildDirectClonePreview(selectionInfo)
            || buildSelectionClonePreview(selectionInfo, label)
            || buildVectorPreview(MHA_FINAL_Q_COLOR, selectionInfo);
    }
    if (lower.includes('key vector')) {
        return buildDirectClonePreview(selectionInfo)
            || buildSelectionClonePreview(selectionInfo, label)
            || buildVectorPreview(MHA_FINAL_K_COLOR, selectionInfo);
    }
    if (lower.includes('value vector')) {
        return buildDirectClonePreview(selectionInfo)
            || buildSelectionClonePreview(selectionInfo, label)
            || buildVectorPreview(MHA_FINAL_V_COLOR, selectionInfo);
    }
    if (selectionInfo?.kind === 'mergedKV') {
        const category = (selectionInfo.info?.category === 'V') ? 'V' : 'K';
        return buildDirectClonePreview(selectionInfo)
            || buildSelectionClonePreview(selectionInfo, label)
            || buildVectorPreview(category === 'V' ? MHA_FINAL_V_COLOR : MHA_FINAL_K_COLOR, selectionInfo);
    }
    if (isLikelyVectorSelection(label, selectionInfo)) {
        return buildDirectClonePreview(selectionInfo)
            || buildSelectionClonePreview(selectionInfo, label)
            || buildVectorPreview(null, selectionInfo);
    }

    if (lower.includes('layernorm') || lower.includes('layer norm')) {
        return buildLayerNormPreview(label, selectionInfo);
    }

    if (isAttentionScoreSelection(label, selectionInfo)) {
        return buildDirectClonePreview(selectionInfo)
            || buildSelectionClonePreview(selectionInfo, label)
            || buildStackedBoxPreview(0x1b1b1b);
    }

    if (lower.includes('attention')) {
        return buildStackedBoxPreview(0x1b1b1b);
    }

    return buildStackedBoxPreview(0x202020);
}

function fitObjectToView(object, camera, options = {}) {
    if (!object) return;
    const { baseScale, basePosition } = (() => {
        if (!object.userData) object.userData = {};
        if (object.userData.__previewBaseScale instanceof THREE.Vector3) {
            return {
                baseScale: object.userData.__previewBaseScale.clone(),
                basePosition: object.userData.__previewBasePosition instanceof THREE.Vector3
                    ? object.userData.__previewBasePosition.clone()
                    : new THREE.Vector3(object.position.x, object.position.y, object.position.z)
            };
        }
        const stored = new THREE.Vector3(object.scale.x, object.scale.y, object.scale.z);
        object.userData.__previewBaseScale = stored.clone();
        const storedPos = new THREE.Vector3(object.position.x, object.position.y, object.position.z);
        object.userData.__previewBasePosition = storedPos.clone();
        return { baseScale: stored, basePosition: storedPos };
    })();
    object.scale.copy(baseScale);
    object.position.copy(basePosition);
    const box = getObjectBounds(object);
    if (box.isEmpty()) return;
    const size = new THREE.Vector3();
    const center = new THREE.Vector3();
    box.getSize(size);
    box.getCenter(center);
    object.position.sub(center);
    const paddingMult = Number.isFinite(options.paddingMultiplier) ? options.paddingMultiplier : 1;
    const distanceMult = Number.isFinite(options.distanceMultiplier) ? options.distanceMultiplier : 1;
    const maxDim = Math.max(size.x, size.y, size.z);
    const scale = maxDim > 0 ? PREVIEW_TARGET_SIZE / (maxDim * PREVIEW_FRAME_PADDING * paddingMult) : 1;
    object.scale.set(baseScale.x * scale, baseScale.y * scale, baseScale.z * scale);

    const scaledBox = getObjectBounds(object);
    const scaledSize = new THREE.Vector3();
    scaledBox.getSize(scaledSize);
    const scaledMax = Math.max(scaledSize.x, scaledSize.y, scaledSize.z);
    const vFov = THREE.MathUtils.degToRad(camera.fov);
    const hFov = 2 * Math.atan(Math.tan(vFov / 2) * camera.aspect);
    const halfX = scaledSize.x * 0.5;
    const halfY = scaledSize.y * 0.5;
    const halfZ = scaledSize.z * 0.5;
    const tilt = Math.abs(PREVIEW_BASE_TILT_X);
    // Preview objects spin around Y, so use the radial XZ envelope to keep the
    // full rotating silhouette inside frame on narrow preview canvases.
    const radialHalf = Math.hypot(halfX, halfZ);
    const verticalHalf = Math.abs(halfY * Math.cos(tilt)) + Math.abs(radialHalf * Math.sin(tilt));
    const horizontalHalf = radialHalf;
    const distY = verticalHalf / Math.tan(vFov / 2);
    const distX = horizontalHalf / Math.tan(hFov / 2);
    const distance = Math.max(distX, distY)
        * PREVIEW_ROTATION_ENVELOPE_MARGIN
        * PREVIEW_BASE_DISTANCE_MULT
        * distanceMult;

    camera.near = Math.max(0.1, distance / 50);
    camera.far = Math.max(distance * 20, distance + scaledMax * 4);
    camera.position.set(0, 0, distance * 1.1);
    camera.lookAt(0, 0, 0);
    camera.updateProjectionMatrix();
}

class SelectionPanel {
    constructor(options = {}) {
        this.panel = document.getElementById('detailPanel');
        this.hudStack = document.getElementById('hudStack');
        this.hudPanel = document.getElementById('hudPanel');
        this.title = document.getElementById('detailTitle');
        this.subtitle = document.getElementById('detailSubtitle');
        this.subtitleSecondary = document.getElementById('detailSubtitleSecondary');
        this.params = document.getElementById('detailParams');
        this.paramsRow = document.getElementById('detailParamsRow');
        this.inputDim = document.getElementById('detailInputDim');
        this.inputDimLabel = document.getElementById('detailInputDimLabel');
        this.inputDimHalf = document.getElementById('detailInputDimHalf');
        this.outputDim = document.getElementById('detailOutputDim');
        this.outputDimLabel = document.getElementById('detailOutputDimLabel');
        this.outputDimHalf = document.getElementById('detailOutputDimHalf');
        this.biasDimRow = document.getElementById('detailBiasDimRow');
        this.biasDim = document.getElementById('detailBiasDim');
        this.metaSection = document.getElementById('detailMeta');
        this.tokenInfoRow = document.getElementById('detailTokenInfoRow');
        this.tokenInfoHeadPrimary = document.getElementById('detailTokenInfoHeadPrimary');
        this.tokenInfoHeadSecondary = document.getElementById('detailTokenInfoHeadSecondary');
        this.tokenInfoHeadTertiary = document.getElementById('detailTokenInfoHeadTertiary');
        this.tokenInfoText = document.getElementById('detailTokenInfoText');
        this.tokenInfoId = document.getElementById('detailTokenInfoId');
        this.tokenInfoPosition = document.getElementById('detailTokenInfoPosition');
        this.tokenEncodingRow = document.getElementById('detailTokenEncodingRow');
        this.tokenEncodingValue = document.getElementById('detailTokenEncodingValue');
        this.copyContextBtn = document.getElementById('detailCopyContextBtn');
        this.copyContextBtnLabel = document.getElementById('detailCopyContextBtnLabel');
        this.closeBtn = document.getElementById('detailClose');
        this.canvas = document.getElementById('detailCanvas');
        this.description = document.getElementById('detailDescription');
        this.equationsSection = document.getElementById('detailEquations');
        this.equationsBody = document.getElementById('detailEquationsBody');
        this.dataEl = document.getElementById('detailData');
        this.dataSection = document.getElementById('detailDataSection');
        this.attentionRoot = document.getElementById('detailAttention');
        this.attentionToggle = document.getElementById('detailAttentionToggle');
        this.attentionToggleLabel = document.getElementById('detailAttentionToggleLabel');
        this.attentionTokensTop = document.getElementById('detailAttentionTokensTop');
        this.attentionTokensLeft = document.getElementById('detailAttentionTokensLeft');
        this.attentionMatrix = document.getElementById('detailAttentionMatrix');
        this.attentionGrid = this.attentionRoot?.querySelector('.detail-attention-grid') || null;
        this.attentionAxisLeft = this.attentionRoot?.querySelector('.attention-axis-label--left') || null;
        this.attentionEmpty = document.getElementById('detailAttentionEmpty');
        this.attentionNote = document.getElementById('detailAttentionNote');
        this.attentionValue = document.getElementById('detailAttentionValue');
        this.attentionValueSource = document.getElementById('detailAttentionValueSource');
        this.attentionValueTarget = document.getElementById('detailAttentionValueTarget');
        this.attentionValueScore = document.getElementById('detailAttentionValueScore');
        this.attentionLegend = document.getElementById('detailAttentionLegend');
        this.attentionLegendTicks = this.attentionLegend
            ? Array.from(this.attentionLegend.querySelectorAll('.attention-legend-tick'))
            : [];
        this.attentionLegendLow = document.getElementById('detailAttentionLegendLow');
        this.attentionLegendHigh = document.getElementById('detailAttentionLegendHigh');
        this.vectorLegend = document.getElementById('detailVectorLegend');
        this.vectorLegendBar = document.getElementById('detailVectorLegendBar');
        this.vectorLegendTicks = this.vectorLegendBar
            ? Array.from(this.vectorLegendBar.querySelectorAll('.vector-legend-tick'))
            : [];
        this.vectorLegendLow = document.getElementById('detailVectorLegendLow');
        this.vectorLegendMid = document.getElementById('detailVectorLegendMid');
        this.vectorLegendHigh = document.getElementById('detailVectorLegendHigh');
        this.engine = options.engine || null;
        this.pipeline = options.pipeline || null;
        this._panelShiftPx = 0;
        this._panelShiftDurationMs = Number.isFinite(options.panelShiftDurationMs)
            ? Math.max(120, options.panelShiftDurationMs)
            : PANEL_SHIFT_DURATION_MS;

        if (!this.panel || !this.canvas || !this.title) {
            this.isReady = false;
            return;
        }

        this.isReady = true;
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0x000000);

        this.camera = new THREE.PerspectiveCamera(35, 1, 0.1, 1000);
        this.camera.position.set(0, 0, 220);

        this.renderer = new THREE.WebGLRenderer({ canvas: this.canvas, antialias: true, alpha: false });
        this.renderer.setPixelRatio(resolveRenderPixelRatio({
            viewportWidth: window.innerWidth,
            viewportHeight: window.innerHeight
        }));
        this.renderer.setClearColor(0x000000, 1);

        this._ambientBaseIntensity = 0.7;
        this._keyLightBaseIntensity = 0.9;
        this._keyLightBasePosition = new THREE.Vector3(25, 40, 40);
        this.ambientLight = new THREE.AmbientLight(0xffffff, this._ambientBaseIntensity);
        this.keyLight = new THREE.DirectionalLight(0xffffff, this._keyLightBaseIntensity);
        this.keyLight.position.copy(this._keyLightBasePosition);
        this.scene.add(this.ambientLight, this.keyLight);
        this._environmentTexture = null;
        this._sourceLightScene = null;
        this._sourceAmbientLight = null;
        this._sourceDirectionalLight = null;
        this._syncEnvironment();

        this.currentPreview = null;
        this.currentDispose = null;
        this.currentAnimator = null;
        this.isOpen = false;
        this._lastFrameTime = performance.now();
        this._rotationSpeedMult = 1;
        this._currentPreviewSelectionKey = null;
        this._lastFitOptions = null;
        this._mobilePauseActive = false;
        this._mobileFocusActive = false;
        this._pauseMainFlowOnMobileFocus = options.pauseMainFlowOnMobileFocus === true;
        this._pendingResizeRaf = null;
        this._previewRafId = null;
        this._pendingResizeTimeout = null;
        this._panelResizeObserver = null;
        this._pendingReveal = false;
        this._pendingRevealSize = null;
        this._pendingRevealTimer = null;
        this._fitLockUntil = 0;
        this._lastFitSize = null;
        this._equationFitState = {
            baseFontPx: null,
            lastFontPx: null,
            scheduled: false,
            pending: false
        };
        this._dimLabelFitState = {
            scheduled: false,
            pending: false
        };
        this._currentSelectionDescription = '';
        this._currentSelectionEquations = '';
        this._copyContextFeedbackTimer = null;
        this._copyContextFadeTimer = null;
        this._copyContextDefaultLabel = this.copyContextBtnLabel?.textContent?.trim()
            || this.copyContextBtn?.textContent?.trim()
            || COPY_CONTEXT_BUTTON_DEFAULT_LABEL;
        this._setCopyContextButtonLabel(this._copyContextDefaultLabel);

        this._animate = this._animate.bind(this);
        this._onResize = this._onResize.bind(this);
        this._onKeydown = this._onKeydown.bind(this);
        this._onCopyContextClick = this._onCopyContextClick.bind(this);
        this._onClosePointerDown = this._onClosePointerDown.bind(this);
        this._onDocumentPointerDown = this._onDocumentPointerDown.bind(this);
        this._blockPreviewGesture = this._blockPreviewGesture.bind(this);
        this._onAttentionPointerMove = this._onAttentionPointerMove.bind(this);
        this._onAttentionPointerDown = this._onAttentionPointerDown.bind(this);
        this._clearAttentionHover = this._clearAttentionHover.bind(this);
        this._onPanelPointerDown = this._onPanelPointerDown.bind(this);
        this._onPanelPointerEnter = this._onPanelPointerEnter.bind(this);
        this._onPanelPointerLeave = this._onPanelPointerLeave.bind(this);
        this._scheduleResize = this._scheduleResize.bind(this);
        this._scheduleSelectionEquationFit = this._scheduleSelectionEquationFit.bind(this);
        this._applySelectionEquationFit = this._applySelectionEquationFit.bind(this);
        this._scheduleDimensionLabelFit = this._scheduleDimensionLabelFit.bind(this);
        this._applyDimensionLabelFit = this._applyDimensionLabelFit.bind(this);

        this.activationSource = options.activationSource || null;
        this.laneTokenIndices = Array.isArray(options.laneTokenIndices) ? options.laneTokenIndices.slice() : null;
        this.tokenLabels = Array.isArray(options.tokenLabels) ? options.tokenLabels.slice() : null;
        this.attentionTokenIndices = Array.isArray(options.attentionTokenIndices)
            ? options.attentionTokenIndices.slice()
            : (Array.isArray(this.laneTokenIndices) ? this.laneTokenIndices.slice() : null);
        this.attentionTokenLabels = Array.isArray(options.attentionTokenLabels)
            ? options.attentionTokenLabels.slice()
            : (Array.isArray(this.tokenLabels) ? this.tokenLabels.slice() : null);
        this.maxAttentionTokens = Number.isFinite(options.maxAttentionTokens)
            ? Math.max(1, Math.floor(options.maxAttentionTokens))
            : ATTENTION_PREVIEW_MAX_TOKENS;
        this.attentionMode = this.attentionToggle?.checked ? 'post' : 'pre';
        this._updateAttentionToggleLabel(this.attentionMode);
        this._attentionContext = null;
        this._attentionTokenElsTop = [];
        this._attentionTokenElsLeft = [];
        this._attentionHoverCell = null;
        this._attentionHoverRow = null;
        this._attentionHoverCol = null;
        this._attentionValueDefault = {
            source: ATTENTION_VALUE_PLACEHOLDER,
            target: ATTENTION_VALUE_PLACEHOLDER,
            score: ATTENTION_VALUE_PLACEHOLDER,
            empty: true
        };
        this._attentionSelectionSummary = null;
        this._attentionPinned = false;
        this._attentionPinnedRow = null;
        this._attentionPinnedCol = null;
        this._attentionCells = null;
        this._attentionValues = null;
        this._attentionDynamic = false;
        this._attentionDynamicKey = '';
        this._attentionDecodeProfile = null;
        this._attentionRowSeparationPx = ATTENTION_DECODE_ROW_OFFSET_MIN_PX;
        this._attentionPostAnimQueue = new Set();
        this._attentionPostAnimatedRows = new Set();
        this._attentionLastPostCompleted = 0;

        this.closeBtn?.addEventListener('click', () => this.close());
        this.copyContextBtn?.addEventListener('click', this._onCopyContextClick);
        this.closeBtn?.addEventListener('pointerdown', this._onClosePointerDown);
        this.canvas.addEventListener('pointerdown', this._blockPreviewGesture, { passive: false });
        this.canvas.addEventListener('pointermove', this._blockPreviewGesture, { passive: false });
        this.canvas.addEventListener('pointerup', this._blockPreviewGesture, { passive: false });
        this.canvas.addEventListener('wheel', this._blockPreviewGesture, { passive: false });
        this.canvas.addEventListener('touchstart', this._blockPreviewGesture, { passive: false });
        this.canvas.addEventListener('touchmove', this._blockPreviewGesture, { passive: false });
        this.canvas.addEventListener('touchend', this._blockPreviewGesture, { passive: false });
        if (this.attentionToggle) {
            this.attentionToggle.addEventListener('change', () => {
                this.attentionMode = this.attentionToggle.checked ? 'post' : 'pre';
                this._updateAttentionToggleLabel(this.attentionMode);
                this._renderAttentionPreview();
            });
        }
        if (this.attentionMatrix) {
            this.attentionMatrix.addEventListener('pointermove', this._onAttentionPointerMove);
            this.attentionMatrix.addEventListener('pointerdown', this._onAttentionPointerDown);
            this.attentionMatrix.addEventListener('pointerleave', this._clearAttentionHover);
        }
        this._setAttentionValue(this._attentionValueDefault);
        this.panel.addEventListener('pointerdown', this._onPanelPointerDown, { capture: true });
        this.panel.addEventListener('pointerenter', this._onPanelPointerEnter);
        this.panel.addEventListener('pointerleave', this._onPanelPointerLeave);
        window.addEventListener('resize', this._onResize);
        if (window.visualViewport && typeof window.visualViewport.addEventListener === 'function') {
            window.visualViewport.addEventListener('resize', this._onResize);
        }
        document.addEventListener('keydown', this._onKeydown);
        document.addEventListener('pointerdown', this._onDocumentPointerDown, { capture: true });
        this._touchClickCleanup = initTouchClickFallback(this.panel, { selector: '.toggle-row' });
        this._observeResize();
        this._onResize();
        if (typeof document !== 'undefined' && document.fonts?.ready) {
            document.fonts.ready.then(() => {
                this._scheduleSelectionEquationFit();
                this._scheduleDimensionLabelFit();
            });
        }
    }

    _observeResize() {
        if (!('ResizeObserver' in window) || !this.canvas?.parentElement) return;
        this._resizeObserver = new ResizeObserver(() => this._onResize());
        this._resizeObserver.observe(this.canvas.parentElement);
        if (this.panel) {
            this._panelResizeObserver = new ResizeObserver(() => {
                this._scheduleSelectionEquationFit();
                this._scheduleDimensionLabelFit();
            });
            this._panelResizeObserver.observe(this.panel);
        }
        if (this.equationsSection) {
            this._equationsResizeObserver = new ResizeObserver(() => this._scheduleSelectionEquationFit());
            this._equationsResizeObserver.observe(this.equationsSection);
        }
    }

    _applyDimensionLabelFit() {
        fitSelectionDimensionLabels({
            inputDimLabel: this.inputDimLabel,
            outputDimLabel: this.outputDimLabel
        });
    }

    _scheduleDimensionLabelFit() {
        if (this._dimLabelFitState.scheduled) {
            this._dimLabelFitState.pending = true;
            return;
        }
        this._dimLabelFitState.scheduled = true;
        this._dimLabelFitState.pending = false;
        const schedule = (typeof window !== 'undefined' && typeof window.requestAnimationFrame === 'function')
            ? window.requestAnimationFrame.bind(window)
            : (cb) => setTimeout(cb, 16);
        schedule(() => {
            this._dimLabelFitState.scheduled = false;
            if (this._dimLabelFitState.pending) {
                this._dimLabelFitState.pending = false;
                this._scheduleDimensionLabelFit();
                return;
            }
            this._applyDimensionLabelFit();
        });
    }

    _setCopyContextButtonLabel(text) {
        const label = String(text || '').trim() || COPY_CONTEXT_BUTTON_DEFAULT_LABEL;
        if (this.copyContextBtnLabel) {
            this.copyContextBtnLabel.textContent = label;
            return;
        }
        if (this.copyContextBtn) {
            this.copyContextBtn.textContent = label;
        }
    }

    _resetCopyContextFeedback({ clearTimers = true } = {}) {
        if (clearTimers) {
            if (this._copyContextFeedbackTimer) {
                clearTimeout(this._copyContextFeedbackTimer);
                this._copyContextFeedbackTimer = null;
            }
            if (this._copyContextFadeTimer) {
                clearTimeout(this._copyContextFadeTimer);
                this._copyContextFadeTimer = null;
            }
        }
        if (!this.copyContextBtn) return;
        this._setCopyContextButtonLabel(this._copyContextDefaultLabel || COPY_CONTEXT_BUTTON_DEFAULT_LABEL);
        this.copyContextBtn.classList.remove('is-feedback-success', 'is-feedback-error', 'is-feedback-fade');
    }

    _showCopyContextFeedback(message, { error = false } = {}) {
        if (!this.copyContextBtn) return;
        this._resetCopyContextFeedback({ clearTimers: true });
        this._setCopyContextButtonLabel(message);
        this.copyContextBtn.classList.toggle('is-feedback-success', !error);
        this.copyContextBtn.classList.toggle('is-feedback-error', error);
        this._copyContextFeedbackTimer = setTimeout(() => {
            this._copyContextFeedbackTimer = null;
            if (!this.copyContextBtn) return;
            this.copyContextBtn.classList.add('is-feedback-fade');
            this._copyContextFadeTimer = setTimeout(() => {
                this._copyContextFadeTimer = null;
                if (!this.copyContextBtn) return;
                this._setCopyContextButtonLabel(this._copyContextDefaultLabel || COPY_CONTEXT_BUTTON_DEFAULT_LABEL);
                this.copyContextBtn.classList.remove('is-feedback-success', 'is-feedback-error', 'is-feedback-fade');
            }, COPY_CONTEXT_FADE_MS);
        }, COPY_CONTEXT_FEEDBACK_MS);
    }

    _buildSelectionContextPayload() {
        const sections = [];
        const title = (this.title?.textContent || '').trim();
        const subtitle = (this.subtitle?.textContent || '').trim();
        const subtitleSecondary = (this.subtitleSecondary?.textContent || '').trim();
        if (title) sections.push(`Selection: ${title}`);
        if (subtitle && subtitleSecondary) {
            sections.push(`Context: ${subtitle}\n${subtitleSecondary}`);
        } else if (subtitle) {
            sections.push(`Context: ${subtitle}`);
        } else if (subtitleSecondary) {
            sections.push(`Context: ${subtitleSecondary}`);
        }

        const descriptionText = String(this._currentSelectionDescription || '').trim();
        if (descriptionText) {
            sections.push(`Description:\n${descriptionText}`);
        }

        const equationText = String(this._currentSelectionEquations || '').trim();
        if (equationText) {
            sections.push(`Equations:\n${equationText}`);
        }

        const metaLines = collectVisibleContextText(this.metaSection, {
            excludeSelectors: '#detailCopyContextBtn, #detailClose'
        });
        if (metaLines.length) {
            sections.push(`Details:\n${metaLines.join('\n')}`);
        }

        if (this.vectorLegend?.classList.contains('is-visible')) {
            const legendLines = collectVisibleContextText(this.vectorLegend);
            if (legendLines.length) {
                sections.push(`Legend:\n${legendLines.join('\n')}`);
            }
        }

        if (this.attentionRoot?.classList.contains('is-visible')) {
            const attentionLines = collectVisibleContextText(this.attentionRoot);
            if (attentionLines.length) {
                sections.push(`Attention:\n${attentionLines.join('\n')}`);
            }
        }

        if (this.dataSection && this.dataSection.style.display !== 'none') {
            const dataLines = collectVisibleContextText(this.dataSection);
            if (dataLines.length) {
                sections.push(`Data:\n${dataLines.join('\n')}`);
            }
        }

        return sections.join('\n\n').replace(/\n{3,}/g, '\n\n').trim();
    }

    async _onCopyContextClick(event) {
        if (event) {
            event.preventDefault();
            event.stopPropagation();
        }
        const payload = this._buildSelectionContextPayload();
        if (!payload) {
            this._showCopyContextFeedback(COPY_CONTEXT_EMPTY_LABEL, { error: true });
            return;
        }
        const copied = await copyTextToClipboard(payload);
        this._showCopyContextFeedback(
            copied ? COPY_CONTEXT_SUCCESS_LABEL : COPY_CONTEXT_ERROR_LABEL,
            { error: !copied }
        );
    }

    _readSelectionEquationBaseFontPx() {
        if (!this.equationsBody || typeof window === 'undefined') return 12;
        const previous = this.equationsBody.style.fontSize;
        this.equationsBody.style.fontSize = '';
        const parsed = Number.parseFloat(window.getComputedStyle(this.equationsBody).fontSize);
        this.equationsBody.style.fontSize = previous;
        return Number.isFinite(parsed) ? parsed : 12;
    }

    _readSelectionEquationContentSize() {
        if (!this.equationsBody) return { width: 0, height: 0 };
        const katexDisplay = this.equationsBody.querySelector('.katex-display');
        const katexRoot = this.equationsBody.querySelector('.katex-display > .katex');
        const measureKatexBaseBounds = () => {
            if (!katexRoot) return null;
            const bases = katexRoot.querySelectorAll('.katex-html .base');
            if (!bases || !bases.length) return null;
            let left = Infinity;
            let right = -Infinity;
            let top = Infinity;
            let bottom = -Infinity;
            bases.forEach((base) => {
                const rect = base.getBoundingClientRect();
                if (!(rect.width > 0 && rect.height > 0)) return;
                left = Math.min(left, rect.left);
                right = Math.max(right, rect.right);
                top = Math.min(top, rect.top);
                bottom = Math.max(bottom, rect.bottom);
            });
            if (!Number.isFinite(left) || !Number.isFinite(right) || !Number.isFinite(top) || !Number.isFinite(bottom)) {
                return null;
            }
            return { width: Math.max(0, right - left), height: Math.max(0, bottom - top) };
        };
        const baseBounds = measureKatexBaseBounds();
        if (baseBounds && katexRoot) {
            const rootRect = katexRoot.getBoundingClientRect();
            return {
                width: Math.max(0, baseBounds.width + 1),
                height: Math.max(0, baseBounds.height, katexRoot.scrollHeight, rootRect.height || 0)
            };
        }
        if (katexRoot) {
            const rect = katexRoot.getBoundingClientRect();
            return {
                width: Math.max(0, katexRoot.scrollWidth, rect.width || 0),
                height: Math.max(0, katexRoot.scrollHeight, rect.height || 0)
            };
        }
        if (katexDisplay) {
            const rect = katexDisplay.getBoundingClientRect();
            return {
                width: Math.max(0, katexDisplay.scrollWidth, rect.width || 0),
                height: Math.max(0, katexDisplay.scrollHeight, rect.height || 0)
            };
        }
        return {
            width: this.equationsBody.scrollWidth,
            height: this.equationsBody.scrollHeight
        };
    }

    _applySelectionEquationFit() {
        if (!this.isReady || !this.isOpen || !this.equationsSection || !this.equationsBody) return;
        if (!this.equationsSection.classList.contains('is-visible')) return;

        const bodyRect = this.equationsBody.getBoundingClientRect();
        const availableWidth = Math.max(0, bodyRect.width);
        if (!(availableWidth > 0)) return;

        const sectionRect = this.equationsSection.getBoundingClientRect();
        const panelRect = this.panel.getBoundingClientRect();
        const getPx = (value) => {
            const parsed = Number.parseFloat(value);
            return Number.isFinite(parsed) ? parsed : 0;
        };
        const panelStyle = (typeof window !== 'undefined')
            ? window.getComputedStyle(this.panel)
            : null;
        const sectionStyle = (typeof window !== 'undefined')
            ? window.getComputedStyle(this.equationsSection)
            : null;
        const panelPaddingBottom = panelStyle ? getPx(panelStyle.paddingBottom) : 0;
        const sectionPaddingBottom = sectionStyle ? getPx(sectionStyle.paddingBottom) : 0;
        const lowerGuard = Math.max(4, panelPaddingBottom + sectionPaddingBottom + 2);
        const sectionTopInset = Math.max(0, bodyRect.top - sectionRect.top);
        const availableHeight = Math.max(
            0,
            Math.min(
                panelRect.bottom - bodyRect.top - lowerGuard,
                panelRect.height - sectionTopInset - lowerGuard
            )
        );
        if (!(availableHeight > 0)) return;
        const fitWidth = Math.max(0, availableWidth - DETAIL_EQUATION_FIT_BUFFER_PX);
        const fitHeight = Math.max(0, availableHeight - DETAIL_EQUATION_FIT_BUFFER_PX);
        if (!(fitWidth > 0 && fitHeight > 0)) return;

        const baseFontPx = this._readSelectionEquationBaseFontPx();
        if (this._equationFitState.baseFontPx === null || Math.abs(baseFontPx - this._equationFitState.baseFontPx) > 0.5) {
            this._equationFitState.baseFontPx = baseFontPx;
            this._equationFitState.lastFontPx = null;
        }

        const applyFontPx = (fontPx) => {
            this.equationsBody.style.fontSize = `${fontPx.toFixed(2)}px`;
        };
        const fitsAt = (fontPx) => {
            applyFontPx(fontPx);
            const fitted = this._readSelectionEquationContentSize();
            return {
                fits: fitted.width <= fitWidth + 0.5
                    && fitted.height <= fitHeight + 0.5,
                size: fitted
            };
        };

        const maxFontPx = Math.max(
            DETAIL_EQUATION_FONT_MIN_PX,
            Math.min(
                availableHeight,
                DETAIL_EQUATION_FONT_MAX_PX,
                this._equationFitState.baseFontPx * DETAIL_EQUATION_FONT_MAX_SCALE
            )
        );
        let low = DETAIL_EQUATION_FONT_MIN_PX;
        const lowProbe = fitsAt(low);
        if (!lowProbe.fits) {
            if (this._equationFitState.lastFontPx === null || Math.abs(this._equationFitState.lastFontPx - low) >= 0.1) {
                applyFontPx(low);
                this._equationFitState.lastFontPx = low;
            }
            return;
        }

        let high = maxFontPx;
        const highProbe = fitsAt(high);
        if (!highProbe.fits) {
            for (let pass = 0; pass < 9; pass += 1) {
                const mid = (low + high) * 0.5;
                const probe = fitsAt(mid);
                if (probe.fits) {
                    low = mid;
                } else {
                    high = mid;
                }
            }
        } else {
            low = high;
        }

        const targetFontPx = Math.max(DETAIL_EQUATION_FONT_MIN_PX, Math.min(maxFontPx, low));
        if (this._equationFitState.lastFontPx !== null && Math.abs(targetFontPx - this._equationFitState.lastFontPx) < 0.1) {
            applyFontPx(this._equationFitState.lastFontPx);
            return;
        }
        applyFontPx(targetFontPx);
        this._equationFitState.lastFontPx = targetFontPx;
    }

    _scheduleSelectionEquationFit() {
        if (this._equationFitState.scheduled) {
            this._equationFitState.pending = true;
            return;
        }
        this._equationFitState.scheduled = true;
        this._equationFitState.pending = false;
        const schedule = (typeof window !== 'undefined' && typeof window.requestAnimationFrame === 'function')
            ? window.requestAnimationFrame.bind(window)
            : (cb) => setTimeout(cb, 16);
        schedule(() => {
            this._equationFitState.scheduled = false;
            if (this._equationFitState.pending) {
                this._equationFitState.pending = false;
                this._scheduleSelectionEquationFit();
                return;
            }
            this._applySelectionEquationFit();
        });
    }

    _onResize() {
        if (!this.isReady) return;
        const rect = this.canvas.getBoundingClientRect();
        const width = Math.max(1, Math.floor(rect.width));
        const height = Math.max(1, Math.floor(rect.height));
        this.renderer.setPixelRatio(resolveRenderPixelRatio({
            viewportWidth: window.innerWidth,
            viewportHeight: window.innerHeight
        }));
        this.renderer.setSize(width, height, false);
        this.camera.aspect = width / height;
        this.camera.updateProjectionMatrix();
        if (this._pendingReveal) {
            const prev = this._pendingRevealSize;
            const changed = !prev || prev.width !== width || prev.height !== height;
            this._pendingRevealSize = { width, height };
            if (changed) {
                if (this._pendingRevealTimer) {
                    clearTimeout(this._pendingRevealTimer);
                }
                this._pendingRevealTimer = setTimeout(() => {
                    this._pendingRevealTimer = null;
                    this._finalizePendingReveal();
                }, 140);
            }
            this._updateMobileState();
            return;
        }
        if (this.currentPreview) {
            let allowFit = true;
            const now = performance.now();
            if (this._lastFitSize && now < this._fitLockUntil) {
                const dw = Math.abs(width - this._lastFitSize.width);
                const dh = Math.abs(height - this._lastFitSize.height);
                const ratioW = this._lastFitSize.width ? dw / this._lastFitSize.width : 1;
                const ratioH = this._lastFitSize.height ? dh / this._lastFitSize.height : 1;
                const bigChange = dw > PREVIEW_FIT_LOCK_PX || dh > PREVIEW_FIT_LOCK_PX
                    || ratioW > PREVIEW_FIT_LOCK_RATIO || ratioH > PREVIEW_FIT_LOCK_RATIO;
                if (!bigChange) {
                    allowFit = false;
                }
            }
            if (allowFit) {
                fitObjectToView(this.currentPreview, this.camera, this._lastFitOptions || {});
                this._noteFit(width, height);
            }
        }
        this._updateMobileState();
        this._syncSceneShift();
        this._scheduleSelectionEquationFit();
        this._scheduleDimensionLabelFit();
    }

    _finalizePendingReveal() {
        if (!this.isReady || !this._pendingReveal) return;
        this._pendingReveal = false;
        if (this.currentPreview) {
            fitObjectToView(this.currentPreview, this.camera, this._lastFitOptions || {});
            this._noteFit(this._pendingRevealSize?.width, this._pendingRevealSize?.height);
        }
        if (this.canvas) {
            this.canvas.style.opacity = '1';
        }
    }

    _noteFit(width, height) {
        let nextWidth = width;
        let nextHeight = height;
        if (!Number.isFinite(nextWidth) || !Number.isFinite(nextHeight)) {
            const rect = this.canvas?.getBoundingClientRect?.();
            if (rect) {
                nextWidth = Math.max(1, Math.floor(rect.width));
                nextHeight = Math.max(1, Math.floor(rect.height));
            }
        }
        if (Number.isFinite(nextWidth) && Number.isFinite(nextHeight)) {
            this._lastFitSize = { width: nextWidth, height: nextHeight };
        }
        this._fitLockUntil = performance.now() + PREVIEW_FIT_LOCK_MS;
    }

    _scheduleResize() {
        if (!this.isReady) return;
        if (this._pendingResizeRaf) {
            cancelAnimationFrame(this._pendingResizeRaf);
        }
        this._pendingResizeRaf = requestAnimationFrame(() => {
            this._pendingResizeRaf = null;
            this._onResize();
        });
        if (this._pendingResizeTimeout) {
            clearTimeout(this._pendingResizeTimeout);
        }
        this._pendingResizeTimeout = setTimeout(() => {
            this._pendingResizeTimeout = null;
            this._onResize();
        }, 280);
    }

    _onKeydown(event) {
        if (event.key === 'Escape' && this.isOpen) {
            this.close();
        }
    }

    _onClosePointerDown(event) {
        if (!this.isOpen) return;
        event.preventDefault();
        event.stopPropagation();
        this.close();
    }

    _blockPreviewGesture(event) {
        const isTouch = event.pointerType === 'touch' || event.type.startsWith('touch');
        if (!isTouch) return;
        if (event.cancelable) event.preventDefault();
        event.stopPropagation();
    }

    _setHoverLabelSuppression(suppressed) {
        if (this.engine && typeof this.engine.setHoverLabelsSuppressed === 'function') {
            this.engine.setHoverLabelsSuppressed(!!suppressed);
        }
    }

    _isSmallScreen() {
        if (typeof window === 'undefined') return false;
        if (typeof window.matchMedia === 'function') {
            return window.matchMedia('(max-aspect-ratio: 1/1), (max-width: 880px)').matches;
        }
        return window.innerWidth <= 880 || window.innerHeight <= window.innerWidth;
    }

    _updateMobileState() {
        const shouldFocus = this.isOpen && this._isSmallScreen();
        const shouldPauseMainFlow = shouldFocus && this._pauseMainFlowOnMobileFocus;
        if (shouldPauseMainFlow !== this._mobilePauseActive) {
            this._mobilePauseActive = shouldPauseMainFlow;
            if (this.engine) {
                if (shouldPauseMainFlow) {
                    this.engine.pause?.('detail-mobile');
                } else {
                    this.engine.resume?.('detail-mobile');
                }
            }
        }
        if (shouldFocus === this._mobileFocusActive) return;
        this._mobileFocusActive = shouldFocus;
        if (typeof document === 'undefined' || !document.body) return;
        if (shouldFocus) {
            document.body.classList.add('detail-mobile-focus');
        } else {
            document.body.classList.remove('detail-mobile-focus');
        }
    }

    _syncSceneShift({ immediate = false } = {}) {
        if (!this.pipeline || typeof this.pipeline.setScreenShiftPixels !== 'function') return;
        if (typeof window === 'undefined') return;

        let nextShift = 0;
        const shouldShift = this.isOpen && !this._isSmallScreen();
        if (shouldShift) {
            const anchor = this.hudStack || this.panel;
            const rect = anchor?.getBoundingClientRect?.();
            if (rect && Number.isFinite(rect.left)) {
                const viewportWidth = window.innerWidth
                    || document.documentElement?.clientWidth
                    || rect.right
                    || 0;
                if (Number.isFinite(viewportWidth) && viewportWidth > 0) {
                    nextShift = Math.max(0, (viewportWidth - rect.left) / 2);
                }
            }
        }

        if (!Number.isFinite(nextShift)) {
            nextShift = 0;
        }

        const delta = Math.abs(nextShift - this._panelShiftPx);
        if (delta < 0.5 && !immediate) {
            return;
        }

        this._panelShiftPx = nextShift;
        this.pipeline.setScreenShiftPixels(nextShift, {
            immediate,
            durationMs: this._panelShiftDurationMs
        });
    }

    _onPanelPointerDown(event) {
        const isTouch = event?.pointerType === 'touch'
            || event?.pointerType === 'pen'
            || (typeof event?.type === 'string' && event.type.startsWith('touch'));
        if (!isTouch) return;
        if (this.engine && typeof this.engine.resetInteractionState === 'function') {
            this.engine.resetInteractionState();
        }
        if (typeof document !== 'undefined' && document.body) {
            document.body.classList.add('touch-ui');
        }
        if (typeof window === 'undefined' || typeof window.getSelection !== 'function') return;
        const selection = window.getSelection();
        if (!selection || selection.isCollapsed) return;
        try {
            selection.removeAllRanges();
        } catch (_) { /* no-op */ }
    }

    _onPanelPointerEnter() {
        this._setHoverLabelSuppression(true);
    }

    _onPanelPointerLeave() {
        this._setHoverLabelSuppression(false);
    }

    _setAttentionVisibility(visible) {
        if (!this.attentionRoot) return;
        if (visible) {
            this.attentionRoot.classList.add('is-visible');
            this.attentionRoot.setAttribute('aria-hidden', 'false');
        } else {
            this.attentionRoot.classList.remove('is-visible');
            this.attentionRoot.setAttribute('aria-hidden', 'true');
        }
    }

    _setAttentionValue(value = null) {
        if (!this.attentionValue) return;
        const safeValue = value && typeof value === 'object' ? value : null;
        const source = normalizeAttentionValuePart(safeValue?.source);
        const target = normalizeAttentionValuePart(safeValue?.target);
        const score = normalizeAttentionValuePart(safeValue?.score);
        const isEmpty = safeValue ? safeValue.empty === true : true;
        if (this.attentionValueSource) {
            this.attentionValueSource.textContent = source;
            this.attentionValueSource.title = source === ATTENTION_VALUE_PLACEHOLDER ? '' : source;
        }
        if (this.attentionValueTarget) {
            this.attentionValueTarget.textContent = target;
            this.attentionValueTarget.title = target === ATTENTION_VALUE_PLACEHOLDER ? '' : target;
        }
        if (this.attentionValueScore) {
            this.attentionValueScore.textContent = score;
            this.attentionValueScore.title = score === ATTENTION_VALUE_PLACEHOLDER ? '' : score;
        }
        this.attentionValue.dataset.empty = isEmpty ? 'true' : 'false';
    }

    _resolveAttentionContext(selection) {
        const label = selection?.label || '';
        if (!isSelfAttentionSelection(label, selection)) return null;
        const headIndex = findUserDataNumber(selection, 'headIndex');
        const layerIndex = findUserDataNumber(selection, 'layerIndex');
        if (!Number.isFinite(headIndex) || !Number.isFinite(layerIndex)) return null;

        let tokenIndices = Array.isArray(this.attentionTokenIndices) ? this.attentionTokenIndices.slice() : null;
        if (!tokenIndices || !tokenIndices.length) {
            const tokenCount = this.activationSource && typeof this.activationSource.getTokenCount === 'function'
                ? this.activationSource.getTokenCount()
                : 0;
            const labelCount = Array.isArray(this.attentionTokenLabels) ? this.attentionTokenLabels.length : 0;
            const fallbackCount = Math.max(
                0,
                Math.min(this.maxAttentionTokens, tokenCount || labelCount || this.maxAttentionTokens)
            );
            tokenIndices = Array.from({ length: fallbackCount }, (_, idx) => idx);
        }
        if (!tokenIndices.length) return null;

        const totalCount = tokenIndices.length;
        tokenIndices = tokenIndices.slice(0, this.maxAttentionTokens);
        const trimmed = totalCount > tokenIndices.length;

        const tokenLabels = tokenIndices.map((tokenIndex, idx) => {
            let labelText = Array.isArray(this.attentionTokenLabels) ? this.attentionTokenLabels[idx] : null;
            if (!labelText && this.activationSource && typeof this.activationSource.getTokenString === 'function') {
                labelText = this.activationSource.getTokenString(tokenIndex);
            }
            const formatted = formatTokenLabelForPreview(labelText);
            if (formatted) return formatted;
            // If capture data exists, keep blanks as a single space instead of "Token N".
            if (this.activationSource) return SPACE_TOKEN_DISPLAY;
            return `Token ${tokenIndex + 1}`;
        });

        return {
            headIndex,
            layerIndex,
            tokenIndices,
            tokenLabels,
            trimmed,
            totalCount,
            hasSource: !!this.activationSource
        };
    }

    _resolveAttentionRuntime(context) {
        if (!context || !this.engine) return null;
        const layers = Array.isArray(this.engine._layers) ? this.engine._layers : null;
        if (!layers || !Number.isFinite(context.layerIndex)) return null;
        const layer = layers[context.layerIndex] || null;
        const mhsa = layer?.mhsaAnimation || null;
        const animator = mhsa?.selfAttentionAnimator || null;
        return { layer, mhsa, animator };
    }

    _resolveAttentionProgress(context) {
        const runtime = this._resolveAttentionRuntime(context);
        const animator = runtime?.animator;
        if (!animator || typeof animator.getAttentionProgress !== 'function') return null;
        if (animator.skipRequested) {
            const count = Array.isArray(context.tokenIndices) ? context.tokenIndices.length : 0;
            const filled = Math.max(0, count);
            return { completedRows: filled, postCompletedRows: filled, activeRow: null, activeCol: null };
        }
        const direct = animator.getAttentionProgress(context.headIndex);
        if (direct) return direct;
        if (animator.phase !== 'running') return null;
        const headIdx = context.headIndex;
        const completedRows = (animator.attentionCompletedRows && Number.isFinite(animator.attentionCompletedRows[headIdx]))
            ? animator.attentionCompletedRows[headIdx]
            : 0;
        const postCompletedRows = (animator.attentionPostCompletedRows && Number.isFinite(animator.attentionPostCompletedRows[headIdx]))
            ? animator.attentionPostCompletedRows[headIdx]
            : 0;
        const progress = animator.attentionProgress ? animator.attentionProgress[headIdx] : null;
        const activeRow = Number.isFinite(progress?.activeRow) ? progress.activeRow : null;
        const activeCol = Number.isFinite(progress?.activeCol) ? progress.activeCol : null;
        return { completedRows, postCompletedRows, activeRow, activeCol };
    }

    _resolveAttentionDecodeProfile(context, progress = null) {
        const runtime = this._resolveAttentionRuntime(context);
        const layer = runtime?.layer;
        const mhsa = runtime?.mhsa;
        if (!layer || !mhsa || !context) return null;

        const liveLanes = Array.isArray(mhsa.currentLanes) ? mhsa.currentLanes : [];
        const conveyorLanes = (typeof mhsa.getAttentionConveyorLanes === 'function')
            ? mhsa.getAttentionConveyorLanes()
            : null;
        const liveLaneCount = liveLanes.length;
        const conveyorLaneCount = Array.isArray(conveyorLanes) ? conveyorLanes.length : 0;
        const inferredDecode = liveLaneCount === 1 && conveyorLaneCount > 1;
        const kvModeEnabled = !!(
            layer?._kvCacheModeEnabled
            || (typeof mhsa._isKvCacheModeEnabled === 'function' && mhsa._isKvCacheModeEnabled())
            || inferredDecode
        );
        const kvDecodeActive = !!(layer?._kvCacheDecodeActive || mhsa?._kvCacheDecodeActive || inferredDecode);
        const tokenIndices = Array.isArray(context.tokenIndices) ? context.tokenIndices : [];
        if (!kvModeEnabled || !kvDecodeActive || tokenIndices.length <= 1) return null;

        let highlightRow = -1;
        const liveLane = liveLanes.find((lane) => lane && Number.isFinite(lane.tokenIndex)) || null;
        const queryTokenIndex = Number.isFinite(liveLane?.tokenIndex) ? liveLane.tokenIndex : null;
        if (Number.isFinite(queryTokenIndex)) {
            highlightRow = tokenIndices.indexOf(queryTokenIndex);
        }
        if (!Number.isFinite(highlightRow) || highlightRow < 0) {
            highlightRow = tokenIndices.length - 1;
        }
        highlightRow = Math.max(0, Math.min(tokenIndices.length - 1, highlightRow));

        return {
            enabled: true,
            highlightRow,
            // Keep historical rows fully visible during decode; only use
            // decode profile for row tracking/highlight behavior.
            dimRowsThrough: -1,
            separateRow: false,
            animating: runtime?.animator?.phase === 'running'
        };
    }

    _buildAttentionProgressKey(progress, decodeProfile = null) {
        const progressPart = progress
            ? `${progress.completedRows || 0}|${progress.postCompletedRows || 0}|${progress.activeRow ?? 'n'}|${progress.activeCol ?? 'n'}`
            : 'none';
        const decodePart = decodeProfile
            ? `${decodeProfile.highlightRow ?? 'n'}|${decodeProfile.separateRow ? 1 : 0}|${decodeProfile.animating ? 1 : 0}`
            : 'off';
        return `${progressPart}|decode:${decodePart}`;
    }

    _hasExplicitAttentionScoreSelection() {
        if (
            this._attentionPinned
            && Number.isFinite(this._attentionPinnedRow)
            && Number.isFinite(this._attentionPinnedCol)
        ) {
            return true;
        }
        return !!this._attentionSelectionSummary;
    }

    _hasAttentionFocusCell() {
        if (this._attentionPinned || !!this._attentionHoverCell) return true;
        return !!(this.attentionMatrix && this.attentionMatrix.classList.contains('has-focus-cell'));
    }

    _shouldSuppressAttentionEntryHighlight() {
        return this._hasExplicitAttentionScoreSelection() || this._hasAttentionFocusCell();
    }

    _applyAttentionDecodeStyling() {
        const profile = this._attentionDecodeProfile;
        const hasDecodeProfile = !!(profile && profile.enabled && Number.isFinite(profile.highlightRow));
        const highlightRow = hasDecodeProfile ? profile.highlightRow : -1;
        const dimRowsThrough = hasDecodeProfile && Number.isFinite(profile.dimRowsThrough)
            ? profile.dimRowsThrough
            : -1;
        const separateRow = !!(hasDecodeProfile && profile.separateRow);
        const suppressActiveDecodeRowHighlight = this._shouldSuppressAttentionEntryHighlight();

        if (this.attentionRoot) {
            this.attentionRoot.dataset.decodeKv = hasDecodeProfile ? 'true' : 'false';
            if (hasDecodeProfile && Number.isFinite(this._attentionRowSeparationPx)) {
                this.attentionRoot.style.setProperty('--attention-decode-row-offset', `${Math.round(this._attentionRowSeparationPx)}px`);
            } else {
                this.attentionRoot.style.removeProperty('--attention-decode-row-offset');
            }
        }

        if (!Array.isArray(this._attentionCells) || !this._attentionCells.length) return;
        for (let row = 0; row < this._attentionCells.length; row += 1) {
            const rowCells = this._attentionCells[row];
            if (!Array.isArray(rowCells)) continue;
            const rowIsActive = hasDecodeProfile
                && !suppressActiveDecodeRowHighlight
                && row === highlightRow;
            const rowIsDimmed = hasDecodeProfile && row <= dimRowsThrough;
            const rowIsSeparated = rowIsActive && separateRow;

            const leftToken = this._attentionTokenElsLeft?.[row];
            if (leftToken) {
                leftToken.classList.toggle('is-decode-dimmed', rowIsDimmed);
                leftToken.classList.toggle('is-decode-active', rowIsActive);
                leftToken.classList.toggle('is-decode-separated-row', rowIsSeparated);
            }

            for (let col = 0; col < rowCells.length; col += 1) {
                const cell = rowCells[col];
                if (!cell || cell.classList.contains('is-hidden')) continue;
                cell.classList.toggle('is-decode-dimmed', rowIsDimmed);
                cell.classList.toggle('is-decode-active-row', rowIsActive);
                cell.classList.toggle('is-decode-separated-row', rowIsSeparated);
            }
        }
    }

    _updateDynamicAttentionProgress() {
        if (!this._attentionContext) return;
        if (this._isSmallScreen && this._isSmallScreen()) {
            if (this._attentionDynamic) {
                this._attentionDynamic = false;
                this._attentionDynamicKey = '';
                this._attentionPostAnimQueue.clear();
                this._attentionPostAnimatedRows.clear();
                this._attentionLastPostCompleted = 0;
                this._applyAttentionReveal(null);
            }
            return;
        }
        const progress = this._resolveAttentionProgress(this._attentionContext);
        const decodeProfile = this._resolveAttentionDecodeProfile(this._attentionContext, progress);
        if (progress) {
            const nextPost = progress.postCompletedRows || 0;
            if (Number.isFinite(nextPost) && nextPost > this._attentionLastPostCompleted) {
                for (let row = this._attentionLastPostCompleted; row < nextPost; row += 1) {
                    this._attentionPostAnimQueue.add(row);
                }
                this._attentionLastPostCompleted = nextPost;
            }
        } else {
            this._attentionPostAnimQueue.clear();
            this._attentionPostAnimatedRows.clear();
            this._attentionLastPostCompleted = 0;
        }

        const nextDynamic = !!progress;
        const nextKey = this._buildAttentionProgressKey(progress, decodeProfile);
        if (this._attentionDynamic !== nextDynamic || this._attentionDynamicKey !== nextKey) {
            this._attentionDynamic = nextDynamic;
            this._attentionDynamicKey = nextKey;
            this._attentionDecodeProfile = decodeProfile;
            this._applyAttentionReveal(progress);
        }
    }

    _cancelAttentionPopOut(cell) {
        if (!cell) return;
        if (cell._attentionPopTimeout) {
            clearTimeout(cell._attentionPopTimeout);
            cell._attentionPopTimeout = null;
        }
        if (cell.dataset) {
            delete cell.dataset.popOut;
        }
        cell.classList.remove('attention-pop-out');
        if (cell._pendingEmpty) {
            delete cell._pendingEmpty;
        }
    }

    _applyAttentionCellRevealAnimation(cell, className, delayMs, durationMs) {
        if (!cell) return;
        const hadClass = cell.classList.contains(className);
        cell.classList.remove('post-softmax-reveal', 'pre-softmax-reveal');
        if (hadClass) {
            // Force a reflow when replaying the same keyframe class.
            void cell.offsetWidth;
        }
        const safeDelay = Number.isFinite(delayMs) ? Math.max(0, Math.round(delayMs)) : 0;
        const safeDuration = Number.isFinite(durationMs) ? Math.max(80, Math.round(durationMs)) : ATTENTION_POST_REVEAL_DURATION_MS;
        cell.style.animationDelay = `${safeDelay}ms`;
        cell.style.animationDuration = `${safeDuration}ms`;
        cell.classList.add(className);
    }

    _startAttentionPopOut(cell, durationMs = null) {
        if (!cell || !cell.dataset) return;
        if (cell.dataset.popOut === 'true') return;
        this._cancelAttentionPopOut(cell);
        cell.dataset.popOut = 'true';
        cell.classList.add('attention-pop-out');
        cell._pendingEmpty = true;
        const duration = Number.isFinite(durationMs) ? durationMs : ATTENTION_POP_OUT_MS;
        if (Number.isFinite(duration)) {
            cell.style.animationDuration = `${Math.max(60, Math.round(duration))}ms`;
        }
        cell._attentionPopTimeout = setTimeout(() => {
            cell._attentionPopTimeout = null;
            if (cell.dataset && cell.dataset.popOut === 'true') {
                delete cell.dataset.popOut;
            }
            cell.classList.remove('attention-pop-out');
            if (cell._pendingEmpty) {
                delete cell._pendingEmpty;
                cell.classList.add('is-empty');
                cell.style.backgroundColor = '';
                cell.removeAttribute('data-value');
                cell.title = '';
            }
        }, Math.max(60, Math.round(duration)));
    }

    _applyAttentionReveal(progress) {
        if (!this.attentionMatrix || !this._attentionCells || !this._attentionValues || !this._attentionContext) return;
        const mode = this.attentionMode === 'post' ? 'post' : 'pre';
        const tokenLabels = this._attentionContext.tokenLabels || [];
        const count = this._attentionCells.length;
        const suppressRevealPulse = this._shouldSuppressAttentionEntryHighlight();
        const postAnimDuration = ATTENTION_POST_REVEAL_DURATION_MS;
        const preAnimDuration = ATTENTION_PRE_REVEAL_DURATION_MS;
        let hasAnyValue = false;
        for (let row = 0; row < count; row += 1) {
            const rowCells = this._attentionCells[row];
            if (!rowCells) continue;
            const shouldAnimateRow = mode === 'post' && this._attentionPostAnimQueue.has(row);
            const visibleCellsInRow = countVisibleAttentionCellsInRow(row, count);
            const postAnimStagger = mode === 'post' && visibleCellsInRow > 1
                ? THREE.MathUtils.clamp(
                    Math.round(ATTENTION_POST_REVEAL_SWEEP_MS / (visibleCellsInRow - 1)),
                    ATTENTION_POST_REVEAL_STAGGER_MIN_MS,
                    ATTENTION_POST_REVEAL_STAGGER_MAX_MS
                )
                : 0;
            const preAnimStagger = mode === 'pre' && visibleCellsInRow > 1
                ? THREE.MathUtils.clamp(
                    Math.round(ATTENTION_PRE_REVEAL_SWEEP_MS / (visibleCellsInRow - 1)),
                    ATTENTION_PRE_REVEAL_STAGGER_MIN_MS,
                    ATTENTION_PRE_REVEAL_STAGGER_MAX_MS
                )
                : 0;
            for (let col = 0; col < count; col += 1) {
                const cell = rowCells[col];
                if (!cell || cell.classList.contains('is-hidden')) continue;
                const value = this._attentionValues[row]?.[col];
                if (!Number.isFinite(value)) {
                    this._cancelAttentionPopOut(cell);
                    cell.classList.add('is-empty');
                    cell.style.backgroundColor = '';
                    cell.classList.remove('post-softmax-reveal', 'pre-softmax-reveal');
                    cell.style.animationDelay = '';
                    cell.style.animationDuration = '';
                    continue;
                }
                const reveal = shouldRevealAttentionCell(progress, row, col, mode, this._attentionDecodeProfile);
                if (reveal) {
                    this._cancelAttentionPopOut(cell);
                    const wasEmpty = cell.classList.contains('is-empty');
                    const color = resolveAttentionPreviewCellColor(value, mode);
                    cell.style.backgroundColor = colorToCss(color);
                    const rowLabel = tokenLabels[row] || '';
                    const colLabel = tokenLabels[col] || '';
                    cell.title = `${rowLabel} → ${colLabel} (${mode === 'post' ? 'post' : 'pre'}): ${value.toFixed(ATTENTION_SCORE_DECIMALS)}`;
                    cell.dataset.value = String(value);
                    cell.classList.remove('is-empty');
                    hasAnyValue = true;
                    if (suppressRevealPulse) {
                        cell.classList.remove('post-softmax-reveal', 'pre-softmax-reveal');
                        cell.style.animationDelay = '';
                        cell.style.animationDuration = '';
                    } else if (mode === 'post') {
                        if (shouldAnimateRow && !this._attentionPostAnimatedRows.has(row)) {
                            const revealOrder = getAttentionRevealOrder(row, col, count);
                            this._applyAttentionCellRevealAnimation(
                                cell,
                                'post-softmax-reveal',
                                revealOrder * postAnimStagger,
                                postAnimDuration
                            );
                        } else {
                            cell.classList.remove('post-softmax-reveal');
                            cell.style.animationDelay = '';
                            cell.style.animationDuration = '';
                        }
                    } else if (mode === 'pre') {
                        if (wasEmpty) {
                            const revealOrder = getAttentionRevealOrder(row, col, count);
                            this._applyAttentionCellRevealAnimation(
                                cell,
                                'pre-softmax-reveal',
                                revealOrder * preAnimStagger,
                                preAnimDuration
                            );
                        } else {
                            cell.classList.remove('pre-softmax-reveal');
                            cell.style.animationDelay = '';
                            cell.style.animationDuration = '';
                        }
                    }
                } else {
                    const wasFilled = !cell.classList.contains('is-empty') || !!cell.dataset.value;
                    if (wasFilled || cell.dataset.popOut === 'true' || !!cell._pendingEmpty) {
                        this._cancelAttentionPopOut(cell);
                        cell.classList.add('is-empty');
                        cell.style.backgroundColor = '';
                        cell.removeAttribute('data-value');
                        cell.title = '';
                    }
                    if (mode === 'post') {
                        cell.classList.remove('post-softmax-reveal');
                        cell.style.animationDelay = '';
                        cell.style.animationDuration = '';
                    } else if (mode === 'pre') {
                        cell.classList.remove('pre-softmax-reveal');
                        cell.style.animationDelay = '';
                        cell.style.animationDuration = '';
                    }
                }
            }
            if (mode === 'post' && shouldAnimateRow) {
                this._attentionPostAnimQueue.delete(row);
                this._attentionPostAnimatedRows.add(row);
            }
        }
        if (this.attentionEmpty) {
            const hasValues = !!this._attentionValues;
            if (!hasValues) {
                this.attentionEmpty.style.display = 'block';
            } else {
                this.attentionEmpty.style.display = (hasAnyValue || this._attentionDynamic) ? 'none' : 'block';
            }
        }
        this._applyAttentionDecodeStyling();
        if (this._attentionPinned) {
            if (!this._restorePinnedAttentionCell()) {
                this._clearPinnedAttention();
            }
            return;
        }
        if (this._attentionHoverCell) {
            const cell = this._attentionHoverCell;
            const valid = this.attentionMatrix && this.attentionMatrix.contains(cell)
                && !cell.classList.contains('is-hidden')
                && !cell.classList.contains('is-empty');
            if (!valid) {
                this._clearAttentionHover(true);
            } else {
                this._setAttentionHoverFromCell(cell, { force: true });
            }
        }
    }

    _updateAttentionPreview(selection) {
        if (!this.attentionRoot) return;
        const context = this._resolveAttentionContext(selection);
        this._attentionContext = context;
        this._attentionSelectionSummary = resolveAttentionScoreSelectionSummary(selection, context);
        const preferredMode = resolveAttentionModeFromSelection(selection);
        if (preferredMode && preferredMode !== this.attentionMode) {
            this.attentionMode = preferredMode;
        }
        this._renderAttentionPreview();
    }

    _updateAttentionToggleLabel(mode) {
        const safeMode = mode === 'post' ? 'post' : 'pre';
        if (this.attentionToggle) {
            this.attentionToggle.checked = safeMode === 'post';
            this.attentionToggle.setAttribute(
                'aria-label',
                safeMode === 'post' ? 'Post-softmax' : 'Pre-softmax'
            );
        }
        if (this.attentionToggleLabel) {
            this.attentionToggleLabel.textContent = safeMode === 'post'
                ? 'Post-softmax'
                : 'Pre-softmax';
        }
    }

    _renderAttentionPreview() {
        if (!this.attentionRoot || !this.attentionMatrix || !this.attentionTokensTop || !this.attentionTokensLeft) return;
        const context = this._attentionContext;
        if (!context || !this.activationSource) {
            this._updateAttentionToggleLabel(this.attentionMode);
            this._setAttentionVisibility(false);
            this._clearPinnedAttention();
            this._attentionDynamic = false;
            this._attentionDynamicKey = '';
            this._attentionValues = null;
            this._attentionCells = null;
            this._attentionPostAnimQueue.clear();
            this._attentionPostAnimatedRows.clear();
            this._attentionLastPostCompleted = 0;
            this._attentionDecodeProfile = null;
            this._attentionSelectionSummary = null;
            this._applyAttentionDecodeStyling();
            return;
        }

        const { tokenIndices, tokenLabels, headIndex, layerIndex, trimmed, totalCount, hasSource } = context;
        const mode = this.attentionMode === 'post' ? 'post' : 'pre';
        this._updateAttentionToggleLabel(mode);
        this._updateAttentionLegend(mode);
        const values = hasSource
            ? buildAttentionMatrixValues({
                activationSource: this.activationSource,
                layerIndex,
                headIndex,
                tokenIndices,
                mode
            })
            : null;
        const allowDynamic = !(this._isSmallScreen && this._isSmallScreen());
        const progress = allowDynamic ? this._resolveAttentionProgress(context) : null;
        const decodeProfile = this._resolveAttentionDecodeProfile(context, progress);
        this._attentionDynamic = !!progress;
        this._attentionDynamicKey = this._buildAttentionProgressKey(progress, decodeProfile);
        this._attentionDecodeProfile = decodeProfile;
        this._attentionPostAnimQueue.clear();
        this._attentionPostAnimatedRows.clear();
        this._attentionLastPostCompleted = progress?.postCompletedRows || 0;
        this._attentionValues = values;

        this._setAttentionVisibility(true);
        if (this.attentionEmpty) this.attentionEmpty.style.display = 'none';
        if (this.attentionNote) {
            if (!hasSource) {
                this.attentionNote.textContent = 'Attention scores unavailable (no capture loaded).';
            } else {
                this.attentionNote.textContent = trimmed
                    ? `Showing first ${tokenIndices.length} of ${totalCount} tokens`
                    : '';
            }
        }
        this._attentionValueDefault = hasSource
            ? (this._attentionSelectionSummary?.defaultValue || {
                source: ATTENTION_VALUE_PLACEHOLDER,
                target: ATTENTION_VALUE_PLACEHOLDER,
                score: ATTENTION_VALUE_PLACEHOLDER,
                empty: true
            })
            : {
                source: ATTENTION_VALUE_PLACEHOLDER,
                target: ATTENTION_VALUE_PLACEHOLDER,
                score: ATTENTION_VALUE_PLACEHOLDER,
                empty: true
            };
        this._setAttentionValue(this._attentionValueDefault);

        const count = tokenIndices.length;
        this._attentionCells = Array.from({ length: count }, () => Array(count).fill(null));
        let cellSize = computeAttentionCellSize(count);
        const densityScale = Math.min(1, Math.max(0.35, 8 / Math.max(1, count)));
        const gap = Math.max(1, Math.round(ATTENTION_PREVIEW_GAP * densityScale));
        const gridGap = Math.max(2, Math.round(ATTENTION_PREVIEW_GRID_GAP * densityScale));
        const gridInset = Math.max(1, Math.round(gap * 0.5));
        const leftPad = Math.max(2, Math.round(4 * densityScale));
        const leftTokenPad = Math.max(2, Math.round(6 * densityScale));
        const labelWidth = measureMaxTokenLabelWidth(tokenLabels, this.attentionTokensLeft);
        const availableWidth = getContentWidth(this.attentionRoot);
        const axisLeftWidth = this.attentionAxisLeft
            ? Math.max(0, this.attentionAxisLeft.getBoundingClientRect().width || 0)
            : 0;
        if (labelWidth > 0 && availableWidth > 0) {
            const usable = availableWidth - (gridInset * 2) - axisLeftWidth - (gridGap * 2)
                - labelWidth - leftPad - leftTokenPad;
            if (usable > 0) {
                const maxCellByWidth = (usable - (count - 1) * gap) / count;
                if (Number.isFinite(maxCellByWidth) && maxCellByWidth > 0) {
                    cellSize = Math.min(cellSize, maxCellByWidth);
                }
            }
        }
        cellSize = Math.max(ATTENTION_PREVIEW_MIN_CELL, Math.min(ATTENTION_PREVIEW_MAX_CELL, Math.floor(cellSize)));
        this._attentionRowSeparationPx = THREE.MathUtils.clamp(
            Math.round(cellSize * ATTENTION_DECODE_ROW_OFFSET_MULT),
            ATTENTION_DECODE_ROW_OFFSET_MIN_PX,
            ATTENTION_DECODE_ROW_OFFSET_MAX_PX
        );
        this.attentionRoot.style.setProperty('--cell-size', `${cellSize}px`);
        this.attentionRoot.style.setProperty('--cell-gap', `${gap}px`);
        this.attentionRoot.style.setProperty('--attention-grid-gap', `${gridGap}px`);
        this.attentionRoot.style.setProperty('--attention-grid-inset', `${gridInset}px`);
        this.attentionRoot.style.setProperty('--attention-left-padding', `${leftPad}px`);
        this.attentionRoot.style.setProperty('--attention-left-token-padding', `${leftTokenPad}px`);
        this.attentionRoot.style.setProperty('--attention-matrix-justify', 'center');
        this.attentionTokensTop.style.gridTemplateColumns = `repeat(${count}, ${cellSize}px)`;
        this.attentionTokensTop.style.gap = `${gap}px`;
        this.attentionTokensLeft.style.gridTemplateRows = `repeat(${count}, ${cellSize}px)`;
        this.attentionTokensLeft.style.gap = `${gap}px`;
        this.attentionMatrix.style.gridTemplateColumns = `repeat(${count}, ${cellSize}px)`;
        this.attentionMatrix.style.gridTemplateRows = `repeat(${count}, ${cellSize}px)`;
        this.attentionMatrix.style.gap = `${gap}px`;

        this._attentionTokenElsTop = [];
        this._attentionTokenElsLeft = [];

        this.attentionTokensTop.innerHTML = '';
        this.attentionTokensLeft.innerHTML = '';
        this.attentionMatrix.innerHTML = '';

        const topFrag = document.createDocumentFragment();
        const leftFrag = document.createDocumentFragment();
        const matrixFrag = document.createDocumentFragment();

        for (let i = 0; i < count; i += 1) {
            const topToken = document.createElement('div');
            topToken.className = 'attention-token attention-token-top';
            const topLabel = document.createElement('span');
            topLabel.className = 'attention-token-top-label';
            topLabel.textContent = tokenLabels[i];
            topToken.appendChild(topLabel);
            topToken.title = tokenLabels[i];
            topFrag.appendChild(topToken);
            this._attentionTokenElsTop.push(topToken);

            const leftToken = document.createElement('div');
            leftToken.className = 'attention-token attention-token-left';
            leftToken.textContent = tokenLabels[i];
            leftToken.title = tokenLabels[i];
            leftFrag.appendChild(leftToken);
            this._attentionTokenElsLeft.push(leftToken);
        }

        let hasAnyValue = false;
        for (let row = 0; row < count; row += 1) {
            for (let col = 0; col < count; col += 1) {
                const cell = document.createElement('div');
                cell.className = 'attention-cell';
                cell.dataset.row = String(row);
                cell.dataset.col = String(col);
                const isVisible = ATTENTION_PREVIEW_TRIANGLE === 'upper'
                    ? col >= row
                    : col <= row;
                const value = values ? values[row]?.[col] : null;
                if (!isVisible) {
                    cell.classList.add('is-hidden');
                } else if (!Number.isFinite(value)) {
                    cell.classList.add('is-empty');
                } else {
                    const reveal = shouldRevealAttentionCell(progress, row, col, mode, decodeProfile);
                    if (reveal) {
                        const color = resolveAttentionPreviewCellColor(value, mode);
                        cell.style.backgroundColor = colorToCss(color);
                        const rowLabel = tokenLabels[row] || '';
                        const colLabel = tokenLabels[col] || '';
                        cell.title = `${rowLabel} → ${colLabel} (${mode === 'post' ? 'post' : 'pre'}): ${value.toFixed(ATTENTION_SCORE_DECIMALS)}`;
                        cell.dataset.value = String(value);
                        hasAnyValue = true;
                    } else {
                        cell.classList.add('is-empty');
                    }
                    cell.dataset.rowLabel = tokenLabels[row] || '';
                    cell.dataset.colLabel = tokenLabels[col] || '';
                }
                this._attentionCells[row][col] = cell;
                matrixFrag.appendChild(cell);
            }
        }

        this.attentionTokensTop.appendChild(topFrag);
        this.attentionTokensLeft.appendChild(leftFrag);
        this.attentionMatrix.appendChild(matrixFrag);
        this._applyAttentionDecodeStyling();
        if (this.attentionEmpty) {
            this.attentionEmpty.style.display = (hasAnyValue || this._attentionDynamic) ? 'none' : 'block';
        }
        let didRestoreSelection = this._restorePinnedAttentionCell();
        if (!didRestoreSelection && this._attentionSelectionSummary) {
            const selectedRow = this._attentionSelectionSummary.row;
            const selectedCol = this._attentionSelectionSummary.col;
            const selectedCell = Number.isFinite(selectedRow) && Number.isFinite(selectedCol)
                ? this._attentionCells?.[selectedRow]?.[selectedCol]
                : null;
            const isUsableCell = !!selectedCell
                && !selectedCell.classList.contains('is-hidden')
                && !selectedCell.classList.contains('is-empty');
            if (isUsableCell) {
                this._setAttentionHoverFromCell(selectedCell, { force: true });
                didRestoreSelection = true;
            }
        }
        if (!didRestoreSelection) {
            this._clearAttentionHover(true);
        }
    }

    _formatAttentionLegendTickLabel(value, { signed = false } = {}) {
        if (!Number.isFinite(value)) return '';
        const safeValue = Math.abs(value) < 1e-6 ? 0 : value;
        const isInteger = Math.abs(safeValue - Math.round(safeValue)) < 1e-6;
        let text = isInteger
            ? String(Math.round(safeValue))
            : safeValue.toFixed(2).replace(/\.?0+$/, '');
        if (signed && safeValue > 0) text = `+${text}`;
        return text;
    }

    _updateAttentionLegendTickLabels(mode) {
        if (!Array.isArray(this.attentionLegendTicks) || this.attentionLegendTicks.length === 0) return;
        const safeMode = mode === 'post' ? 'post' : 'pre';
        const edgeEpsilon = 1e-6;
        for (let i = 0; i < this.attentionLegendTicks.length; i += 1) {
            const tick = this.attentionLegendTicks[i];
            if (!tick) continue;
            const ratio = Number(tick.dataset.ratio);
            if (!Number.isFinite(ratio)) {
                tick.dataset.label = '';
                continue;
            }
            if (safeMode === 'pre' && ratio <= edgeEpsilon) {
                tick.dataset.label = `≤ -${ATTENTION_PRE_COLOR_CLAMP}`;
                continue;
            }
            if (safeMode === 'pre' && ratio >= (1 - edgeEpsilon)) {
                tick.dataset.label = `≥ +${ATTENTION_PRE_COLOR_CLAMP}`;
                continue;
            }
            const value = safeMode === 'post'
                ? ratio
                : THREE.MathUtils.lerp(-ATTENTION_PRE_COLOR_CLAMP, ATTENTION_PRE_COLOR_CLAMP, ratio);
            tick.dataset.label = this._formatAttentionLegendTickLabel(value, { signed: safeMode === 'pre' });
        }
    }

    _updateAttentionLegend(mode) {
        if (!this.attentionLegend || !this.attentionLegendLow || !this.attentionLegendHigh) return;
        const safeMode = mode === 'post' ? 'post' : 'pre';
        if (this.attentionRoot) {
            this.attentionRoot.dataset.attnMode = safeMode;
        }

        if (safeMode === 'post') {
            const low = colorToCss(resolveAttentionPreviewCellColor(0, 'post'));
            const high = colorToCss(resolveAttentionPreviewCellColor(1, 'post'));
            this.attentionLegend.style.setProperty('--attention-legend-gradient', `linear-gradient(90deg, ${low}, ${high})`);
            this.attentionLegend.dataset.mid = '';
            this.attentionLegendLow.textContent = '';
            this.attentionLegendHigh.textContent = '';
            this._updateAttentionLegendTickLabels('post');
            return;
        }

        const gradient = buildSpectrumLegendGradient({
            clampMax: ATTENTION_PRE_COLOR_CLAMP,
            steps: 15,
            darkenFactor: ATTENTION_PREVIEW_COLOR_DARKEN_FACTOR
        });
        this.attentionLegend.style.setProperty('--attention-legend-gradient', gradient);
        this.attentionLegend.dataset.mid = '';
        this.attentionLegendLow.textContent = '';
        this.attentionLegendHigh.textContent = '';
        this._updateAttentionLegendTickLabels('pre');
    }

    _updateVectorLegendTickLabels() {
        if (!Array.isArray(this.vectorLegendTicks) || this.vectorLegendTicks.length === 0) return;
        const edgeEpsilon = 1e-6;
        for (let i = 0; i < this.vectorLegendTicks.length; i += 1) {
            const tick = this.vectorLegendTicks[i];
            if (!tick) continue;
            const ratio = Number(tick.dataset.ratio);
            if (!Number.isFinite(ratio)) {
                tick.dataset.label = '';
                continue;
            }
            if (ratio <= edgeEpsilon) {
                tick.dataset.label = `≤ -${RESIDUAL_COLOR_CLAMP}`;
                continue;
            }
            if (ratio >= (1 - edgeEpsilon)) {
                tick.dataset.label = `≥ +${RESIDUAL_COLOR_CLAMP}`;
                continue;
            }
            const value = THREE.MathUtils.lerp(-RESIDUAL_COLOR_CLAMP, RESIDUAL_COLOR_CLAMP, ratio);
            tick.dataset.label = this._formatAttentionLegendTickLabel(value, { signed: true });
        }
    }

    _updateVectorLegend(selection) {
        if (!this.vectorLegend) return;
        const show = isResidualVectorSelection(selection?.label, selection)
            && isLikelyVectorSelection(selection?.label, selection);
        if (!show) {
            this.vectorLegend.classList.remove('is-visible');
            this.vectorLegend.setAttribute('aria-hidden', 'true');
            return;
        }

        const gradient = buildSpectrumLegendGradient({
            clampMax: RESIDUAL_COLOR_CLAMP,
            steps: 15
        });
        if (this.vectorLegendBar) {
            this.vectorLegendBar.style.setProperty('--vector-legend-gradient', gradient);
        }
        this._updateVectorLegendTickLabels();
        if (this.vectorLegendLow) this.vectorLegendLow.textContent = '';
        if (this.vectorLegendMid) this.vectorLegendMid.textContent = '';
        if (this.vectorLegendHigh) this.vectorLegendHigh.textContent = '';
        this.vectorLegend.classList.add('is-visible');
        this.vectorLegend.setAttribute('aria-hidden', 'false');
    }

    _setAttentionHoverFromCell(cell, { force = false } = {}) {
        if (!cell || cell.classList.contains('is-empty') || cell.classList.contains('is-hidden')) {
            this._clearAttentionHover(force);
            return;
        }
        const row = Number(cell.dataset.row);
        const col = Number(cell.dataset.col);
        const isPinnedSelection = this._attentionPinned
            && row === this._attentionPinnedRow
            && col === this._attentionPinnedCol;
        const isSameCell = row === this._attentionHoverRow
            && col === this._attentionHoverCol
            && cell === this._attentionHoverCell;
        const emphasizeTokenLabels = this.attentionMode === 'post';
        const leftToken = this._attentionTokenElsLeft[row];
        const topToken = this._attentionTokenElsTop[col];
        if (isSameCell) {
            cell.classList.toggle('is-pinned', isPinnedSelection);
            if (this.attentionMatrix) this.attentionMatrix.classList.add('has-focus-cell');
            if (leftToken) leftToken.classList.toggle('is-highlighted', emphasizeTokenLabels);
            if (topToken) topToken.classList.toggle('is-highlighted', emphasizeTokenLabels);
            return;
        }

        this._clearAttentionHover(force);
        this._attentionHoverCell = cell;
        this._attentionHoverRow = row;
        this._attentionHoverCol = col;
        if (this.attentionMatrix) this.attentionMatrix.classList.add('has-focus-cell');
        cell.classList.add('is-hovered');
        cell.classList.toggle('is-pinned', isPinnedSelection);
        if (leftToken) leftToken.classList.toggle('is-highlighted', emphasizeTokenLabels);
        if (topToken) topToken.classList.toggle('is-highlighted', emphasizeTokenLabels);
        const rawValue = cell.dataset.value;
        const valueNum = Number(rawValue);
        const rowLabel = cell.dataset.rowLabel || '';
        const colLabel = cell.dataset.colLabel || '';
        const sourceText = normalizeAttentionValuePart(rowLabel, 'Source');
        const targetText = normalizeAttentionValuePart(colLabel, 'Target');
        const scoreText = Number.isFinite(valueNum)
            ? valueNum.toFixed(ATTENTION_SCORE_DECIMALS)
            : ATTENTION_VALUE_PLACEHOLDER;
        this._setAttentionValue({
            source: sourceText,
            target: targetText,
            score: scoreText,
            empty: false
        });
    }

    _onAttentionPointerMove(event) {
        if (this._attentionPinned) return;
        const target = event.target;
        const cell = target && typeof target.closest === 'function'
            ? target.closest('.attention-cell')
            : null;
        if (!cell || !this.attentionMatrix || !this.attentionMatrix.contains(cell)) {
            this._clearAttentionHover();
            return;
        }
        this._setAttentionHoverFromCell(cell);
    }

    _onAttentionPointerDown(event) {
        if (event?.pointerType === 'mouse' && Number.isFinite(event.button) && event.button !== 0) return;
        const target = event.target;
        const cell = target && typeof target.closest === 'function'
            ? target.closest('.attention-cell')
            : null;
        if (!cell || !this.attentionMatrix || !this.attentionMatrix.contains(cell)) {
            this._clearPinnedAttention({ clearSelectionSummary: true });
            return;
        }
        if (cell.classList.contains('is-empty') || cell.classList.contains('is-hidden')) {
            this._clearPinnedAttention({ clearSelectionSummary: true });
            return;
        }
        const row = Number(cell.dataset.row);
        const col = Number(cell.dataset.col);
        if (!Number.isFinite(row) || !Number.isFinite(col)) {
            this._clearPinnedAttention({ clearSelectionSummary: true });
            return;
        }
        this._attentionPinned = true;
        this._attentionPinnedRow = row;
        this._attentionPinnedCol = col;
        this._setAttentionHoverFromCell(cell, { force: true });
    }

    _clearAttentionHover(force = false) {
        const forceFlag = force === true;
        if (this._attentionPinned && !forceFlag) return;
        if (this.attentionMatrix) this.attentionMatrix.classList.remove('has-focus-cell');
        if (this._attentionHoverCell) {
            this._attentionHoverCell.classList.remove('is-hovered', 'is-pinned');
        }
        if (Number.isFinite(this._attentionHoverRow)) {
            const leftToken = this._attentionTokenElsLeft[this._attentionHoverRow];
            if (leftToken) leftToken.classList.remove('is-highlighted');
        }
        if (Number.isFinite(this._attentionHoverCol)) {
            const topToken = this._attentionTokenElsTop[this._attentionHoverCol];
            if (topToken) topToken.classList.remove('is-highlighted');
        }
        this._attentionHoverCell = null;
        this._attentionHoverRow = null;
        this._attentionHoverCol = null;
        this._setAttentionValue(this._attentionValueDefault);
    }

    _clearPinnedAttention({ clearSelectionSummary = false } = {}) {
        this._attentionPinned = false;
        this._attentionPinnedRow = null;
        this._attentionPinnedCol = null;
        if (clearSelectionSummary) {
            this._attentionSelectionSummary = null;
            this._attentionValueDefault = {
                source: ATTENTION_VALUE_PLACEHOLDER,
                target: ATTENTION_VALUE_PLACEHOLDER,
                score: ATTENTION_VALUE_PLACEHOLDER,
                empty: true
            };
        }
        this._clearAttentionHover(true);
    }

    _restorePinnedAttentionCell() {
        if (!this._attentionPinned || !this.attentionMatrix) return false;
        if (!Number.isFinite(this._attentionPinnedRow) || !Number.isFinite(this._attentionPinnedCol)) return false;
        const selector = `.attention-cell[data-row="${this._attentionPinnedRow}"][data-col="${this._attentionPinnedCol}"]`;
        const cell = this.attentionMatrix.querySelector(selector);
        if (!cell || cell.classList.contains('is-empty') || cell.classList.contains('is-hidden')) {
            this._clearPinnedAttention();
            return false;
        }
        this._setAttentionHoverFromCell(cell, { force: true });
        return true;
    }

    _onDocumentPointerDown(event) {
        if (!Number.isFinite(event.clientX) || !Number.isFinite(event.clientY)) return;
        const hit = document.elementFromPoint(event.clientX, event.clientY);
        const insideAttentionMatrix = !!(
            this.attentionMatrix
            && hit
            && typeof hit.closest === 'function'
            && hit.closest('#detailAttentionMatrix') === this.attentionMatrix
        );
        const matrixCell = insideAttentionMatrix && typeof hit?.closest === 'function'
            ? hit.closest('.attention-cell')
            : null;
        const validMatrixCell = !!(
            matrixCell
            && !matrixCell.classList.contains('is-empty')
            && !matrixCell.classList.contains('is-hidden')
        );
        const shouldClearPinnedAttention = this.isOpen
            && this._attentionPinned
            && (!insideAttentionMatrix || !validMatrixCell);
        if (shouldClearPinnedAttention) {
            this._clearPinnedAttention({ clearSelectionSummary: true });
        }
        if (this.isOpen && hit && this.panel && this.panel.contains(hit)) {
            if (this.engine && typeof this.engine.resetInteractionState === 'function') {
                this.engine.resetInteractionState();
            }
        }
        if (!this.isOpen || !this.closeBtn) return;
        if (event.target === this.closeBtn) return;
        if (!hit || typeof hit.closest !== 'function') return;
        if (hit.closest('#detailClose') !== this.closeBtn) return;
        // Close even if the canvas captured the pointer event.
        event.preventDefault();
        event.stopPropagation();
        this.close();
    }

    _refreshSourceLightRefs() {
        const sourceScene = this.engine?.scene || null;
        if (sourceScene === this._sourceLightScene) return;
        this._sourceLightScene = sourceScene;
        this._sourceAmbientLight = null;
        this._sourceDirectionalLight = null;
        if (!sourceScene || typeof sourceScene.traverse !== 'function') return;
        sourceScene.traverse((node) => {
            if (!this._sourceAmbientLight && node?.isAmbientLight) {
                this._sourceAmbientLight = node;
            }
            if (!this._sourceDirectionalLight && node?.isDirectionalLight) {
                this._sourceDirectionalLight = node;
            }
        });
    }

    _syncPreviewLightsFromSource() {
        this._refreshSourceLightRefs();
        if (this.ambientLight) {
            const sourceAmbient = this._sourceAmbientLight;
            if (sourceAmbient?.color?.isColor) {
                this.ambientLight.color.copy(sourceAmbient.color);
            } else {
                this.ambientLight.color.set(0xffffff);
            }
            this.ambientLight.intensity = Number.isFinite(sourceAmbient?.intensity)
                ? sourceAmbient.intensity
                : this._ambientBaseIntensity;
        }

        if (this.keyLight) {
            const sourceDirectional = this._sourceDirectionalLight;
            if (sourceDirectional?.color?.isColor) {
                this.keyLight.color.copy(sourceDirectional.color);
            } else {
                this.keyLight.color.set(0xffffff);
            }
            this.keyLight.intensity = Number.isFinite(sourceDirectional?.intensity)
                ? sourceDirectional.intensity
                : this._keyLightBaseIntensity;
            if (sourceDirectional?.position?.isVector3) {
                this.keyLight.position.copy(sourceDirectional.position);
            } else {
                this.keyLight.position.copy(this._keyLightBasePosition);
            }
        }
    }

    _syncEnvironment() {
        const engineEnv = this.engine?.scene?.environment || null;
        const env = engineEnv || appState.environmentTexture;
        if (env && this._environmentTexture !== env) {
            this.scene.environment = env;
            this._environmentTexture = env;
        } else if (!env && this._environmentTexture) {
            this.scene.environment = null;
            this._environmentTexture = null;
        }
        this._syncPreviewLightsFromSource();

        const sourceRenderer = this.engine?.renderer || null;
        if (sourceRenderer) {
            this.renderer.toneMapping = sourceRenderer.toneMapping;
            this.renderer.toneMappingExposure = Number.isFinite(sourceRenderer.toneMappingExposure)
                ? sourceRenderer.toneMappingExposure
                : 1.0;
            if ('outputColorSpace' in this.renderer && 'outputColorSpace' in sourceRenderer) {
                this.renderer.outputColorSpace = sourceRenderer.outputColorSpace;
            } else if ('outputEncoding' in this.renderer && 'outputEncoding' in sourceRenderer) {
                this.renderer.outputEncoding = sourceRenderer.outputEncoding;
            }
        } else if (env) {
            this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
            this.renderer.toneMappingExposure = 1.0;
        } else {
            this.renderer.toneMapping = THREE.NoToneMapping;
            this.renderer.toneMappingExposure = 1.0;
        }
    }

    _isPreviewLoopActive() {
        return !!(this.isReady && this.isOpen && this.currentPreview);
    }

    _startLoop() {
        if (!this._isPreviewLoopActive()) return;
        if (this._previewRafId !== null) return;
        this._lastFrameTime = performance.now();
        this._previewRafId = requestAnimationFrame(this._animate);
    }

    _stopLoop() {
        if (this._previewRafId === null) return;
        cancelAnimationFrame(this._previewRafId);
        this._previewRafId = null;
    }

    _animate(time) {
        this._previewRafId = null;
        if (!this._isPreviewLoopActive()) return;
        const now = (typeof time === 'number') ? time : performance.now();
        const deltaMs = this._lastFrameTime ? (now - this._lastFrameTime) : 16.6667;
        this._lastFrameTime = now;

        this._syncEnvironment();

        if (typeof this.currentAnimator === 'function') {
            try {
                this.currentAnimator(deltaMs, now);
            } catch (err) {
                // Keep selection preview resilient to animation errors.
                console.warn('Selection preview animation error:', err);
            }
        }

        const rotationSpeedMult = Number.isFinite(this._rotationSpeedMult) ? this._rotationSpeedMult : 1;
        const clampedDelta = Math.min(Math.max(deltaMs, 0), 33.3334);
        const rotationStep = PREVIEW_ROTATION_SPEED * rotationSpeedMult * (clampedDelta / 16.6667);
        this.currentPreview.rotation.y += rotationStep;
        this.currentPreview.rotation.x = PREVIEW_BASE_TILT_X;
        this.currentPreview.rotation.z = 0;
        this._updateDynamicAttentionProgress();
        this.renderer.render(this.scene, this.camera);
        if (this._isPreviewLoopActive() && this._previewRafId === null) {
            this._previewRafId = requestAnimationFrame(this._animate);
        }
    }

    open() {
        if (!this.isReady) return;
        const wasOpen = this.isOpen;
        this.isOpen = true;
        this.panel.classList.add('is-open');
        this.hudStack?.classList.add('detail-open');
        this.hudPanel?.classList.add('detail-open');
        this.panel.setAttribute('aria-hidden', 'false');
        this._updateMobileState();
        this._syncSceneShift();
        if (!wasOpen) {
            if (this._pendingReveal) {
                this._pendingRevealSize = null;
            }
            this._scheduleResize();
        }
        this._startLoop();
        this._scheduleSelectionEquationFit();
        this._scheduleDimensionLabelFit();
    }

    close() {
        if (!this.isReady) return;
        this.isOpen = false;
        this._stopLoop();
        this._currentSelectionDescription = '';
        this._currentSelectionEquations = '';
        this._resetCopyContextFeedback();
        this.panel.classList.remove('is-open');
        this.panel.classList.remove('is-preview-hidden');
        this.hudStack?.classList.remove('detail-open');
        this.hudPanel?.classList.remove('detail-open');
        this.panel.setAttribute('aria-hidden', 'true');
        this._setHoverLabelSuppression(false);
        this._updateMobileState();
        this._syncSceneShift();
        if (this.description) setDescriptionContent(this.description, '');
        if (this.equationsBody) setDescriptionContent(this.equationsBody, '');
        if (this.equationsBody) this.equationsBody.style.fontSize = '';
        if (this.equationsSection) {
            this.equationsSection.classList.remove('is-visible');
            this.equationsSection.setAttribute('aria-hidden', 'true');
        }
        this._equationFitState.baseFontPx = null;
        this._equationFitState.lastFontPx = null;
        this._equationFitState.scheduled = false;
        this._equationFitState.pending = false;
        this._dimLabelFitState.scheduled = false;
        this._dimLabelFitState.pending = false;
        if (this.inputDimLabel) {
            this.inputDimLabel.style.fontSize = '';
            this.inputDimLabel.style.letterSpacing = '';
        }
        if (this.outputDimLabel) {
            this.outputDimLabel.style.fontSize = '';
            this.outputDimLabel.style.letterSpacing = '';
        }
        if (this._pendingResizeRaf) {
            cancelAnimationFrame(this._pendingResizeRaf);
            this._pendingResizeRaf = null;
        }
        if (this._pendingResizeTimeout) {
            clearTimeout(this._pendingResizeTimeout);
            this._pendingResizeTimeout = null;
        }
        if (this._pendingRevealTimer) {
            clearTimeout(this._pendingRevealTimer);
            this._pendingRevealTimer = null;
        }
        this._pendingRevealSize = null;
    }

    updateData({
        activationSource = null,
        laneTokenIndices = null,
        tokenLabels = null,
        attentionTokenIndices = null,
        attentionTokenLabels = null
    } = {}) {
        this.activationSource = activationSource;
        this.laneTokenIndices = Array.isArray(laneTokenIndices) ? laneTokenIndices.slice() : null;
        this.tokenLabels = Array.isArray(tokenLabels) ? tokenLabels.slice() : null;
        this.attentionTokenIndices = Array.isArray(attentionTokenIndices)
            ? attentionTokenIndices.slice()
            : (Array.isArray(this.laneTokenIndices) ? this.laneTokenIndices.slice() : null);
        this.attentionTokenLabels = Array.isArray(attentionTokenLabels)
            ? attentionTokenLabels.slice()
            : (Array.isArray(this.tokenLabels) ? this.tokenLabels.slice() : null);
        this._attentionContext = null;
        this._attentionCells = null;
        this._attentionValues = null;
        this._attentionDynamic = false;
        this._attentionDynamicKey = '';
        this._attentionDecodeProfile = null;
        this._attentionPostAnimQueue?.clear?.();
        this._attentionPostAnimatedRows?.clear?.();
        this._attentionLastPostCompleted = 0;
        this._attentionSelectionSummary = null;
        this._attentionValueDefault = {
            source: ATTENTION_VALUE_PLACEHOLDER,
            target: ATTENTION_VALUE_PLACEHOLDER,
            score: ATTENTION_VALUE_PLACEHOLDER,
            empty: true
        };
        this._setAttentionValue(this._attentionValueDefault);
        this._applyAttentionDecodeStyling();
    }

    _resolveVectorTokenPosition(selection, label) {
        const lower = (label || '').toLowerCase();
        const isVectorSelection = isLikelyVectorSelection(label, selection);
        const isTokenChipSelection = lower.startsWith('token:') || lower.startsWith('position:');
        if (!isVectorSelection && !isTokenChipSelection) return null;

        let laneIndex = findUserDataNumber(selection, 'laneIndex');
        let tokenIndex = findUserDataNumber(selection, 'tokenIndex');
        if (!Number.isFinite(tokenIndex) && Number.isFinite(laneIndex) && Array.isArray(this.laneTokenIndices)) {
            const mappedTokenIndex = this.laneTokenIndices[laneIndex];
            if (Number.isFinite(mappedTokenIndex)) tokenIndex = mappedTokenIndex;
        }
        if (!Number.isFinite(tokenIndex) && lower.startsWith('position:')) {
            const parsedPosition = Number(extractTokenText(label));
            if (Number.isFinite(parsedPosition) && parsedPosition > 0) {
                tokenIndex = Math.floor(parsedPosition - 1);
            }
        }

        let tokenLabel = findUserDataString(selection, 'tokenLabel');
        if (tokenLabel == null && lower.startsWith('token:')) {
            tokenLabel = extractTokenText(label);
        }
        if (tokenLabel == null && Number.isFinite(laneIndex) && Array.isArray(this.tokenLabels)) {
            const laneTokenLabel = this.tokenLabels[laneIndex];
            if (typeof laneTokenLabel === 'string') tokenLabel = laneTokenLabel;
        }
        if (tokenLabel != null && !Number.isFinite(laneIndex) && Array.isArray(this.tokenLabels)) {
            const formattedNeedle = formatTokenLabelForPreview(tokenLabel);
            const mappedLaneIndex = this.tokenLabels.findIndex((candidate) => (
                formatTokenLabelForPreview(candidate) === formattedNeedle
            ));
            if (mappedLaneIndex >= 0) {
                laneIndex = mappedLaneIndex;
                if (!Number.isFinite(tokenIndex) && Array.isArray(this.laneTokenIndices)) {
                    const mappedTokenIndex = this.laneTokenIndices[mappedLaneIndex];
                    if (Number.isFinite(mappedTokenIndex)) tokenIndex = mappedTokenIndex;
                }
            }
        }
        if (tokenLabel == null && Number.isFinite(tokenIndex) && this.activationSource && typeof this.activationSource.getTokenString === 'function') {
            const sourceToken = this.activationSource.getTokenString(tokenIndex);
            if (typeof sourceToken === 'string') tokenLabel = sourceToken;
        }
        if (tokenLabel == null && Number.isFinite(tokenIndex) && Array.isArray(this.laneTokenIndices) && Array.isArray(this.tokenLabels)) {
            const mappedLaneIndex = this.laneTokenIndices.findIndex((idx) => Number.isFinite(idx) && idx === tokenIndex);
            if (mappedLaneIndex >= 0 && mappedLaneIndex < this.tokenLabels.length && typeof this.tokenLabels[mappedLaneIndex] === 'string') {
                tokenLabel = this.tokenLabels[mappedLaneIndex];
            }
        }

        let tokenId = findUserDataNumber(selection, 'tokenId');
        if (!Number.isFinite(tokenId) && Number.isFinite(tokenIndex) && this.activationSource && typeof this.activationSource.getTokenId === 'function') {
            tokenId = this.activationSource.getTokenId(tokenIndex);
        }
        const tokenEncodingNote = getIncompleteUtf8TokenNote(tokenId);

        if (Number.isFinite(tokenIndex) && this.activationSource) {
            if (typeof this.activationSource.getTokenRawString === 'function') {
                const sourceRawToken = this.activationSource.getTokenRawString(tokenIndex);
                if (typeof sourceRawToken === 'string') tokenLabel = sourceRawToken;
            }
            if ((!tokenLabel || typeof tokenLabel !== 'string') && typeof this.activationSource.getTokenString === 'function') {
                const sourceToken = this.activationSource.getTokenString(tokenIndex);
                if (typeof sourceToken === 'string') tokenLabel = sourceToken;
            }
        }

        let tokenText = (typeof tokenLabel === 'string') ? tokenLabel : '';
        if (!tokenText && Number.isFinite(tokenIndex)) tokenText = `Token ${tokenIndex + 1}`;

        let tokenDisplayText = '';
        if (Number.isFinite(tokenIndex) && this.activationSource && typeof this.activationSource.getTokenString === 'function') {
            const sourceDisplayToken = this.activationSource.getTokenString(tokenIndex);
            if (typeof sourceDisplayToken === 'string') {
                tokenDisplayText = formatTokenLabelForPreview(sourceDisplayToken);
            }
        }
        if (!tokenDisplayText && tokenText) {
            tokenDisplayText = formatTokenLabelForPreview(tokenText);
        }
        if (!tokenDisplayText && tokenText) tokenDisplayText = tokenText;
        const tokenIdText = Number.isFinite(tokenId) ? String(Math.floor(tokenId)) : '';

        let positionText = '';
        if (Number.isFinite(tokenIndex)) {
            positionText = String(tokenIndex + 1);
        } else if (Number.isFinite(laneIndex)) {
            positionText = String(laneIndex + 1);
        }

        if (!tokenText && !tokenIdText && !positionText) return null;
        return {
            tokenText,
            tokenDisplayText,
            tokenIdText,
            positionText,
            tokenEncodingNote
        };
    }

    _updateVectorTokenPositionRows(selection, label) {
        const tokenInfoRow = this.tokenInfoRow;
        const tokenInfoHeadPrimary = this.tokenInfoHeadPrimary;
        const tokenInfoHeadSecondary = this.tokenInfoHeadSecondary;
        const tokenInfoHeadTertiary = this.tokenInfoHeadTertiary;
        const tokenInfoText = this.tokenInfoText;
        const tokenInfoId = this.tokenInfoId;
        const tokenInfoPosition = this.tokenInfoPosition;
        const lower = String(label || '').toLowerCase();
        const activationStage = String(getActivationDataFromSelection(selection)?.stage || '').toLowerCase();
        const isLogitSelection = !!resolveLogitSelectionHeader(label, selection);
        const isPositionEmbeddingSelection = lower.startsWith('position:')
            || lower.includes('position embedding')
            || lower.includes('positional embedding')
            || activationStage.startsWith('embedding.position');

        const hideRows = () => {
            if (tokenInfoRow) {
                tokenInfoRow.style.display = 'none';
                tokenInfoRow.dataset.empty = 'true';
                tokenInfoRow.dataset.layout = 'token';
            }
            if (tokenInfoHeadPrimary) tokenInfoHeadPrimary.textContent = 'Raw token';
            if (tokenInfoHeadSecondary) tokenInfoHeadSecondary.textContent = 'Token ID';
            if (tokenInfoHeadTertiary) tokenInfoHeadTertiary.textContent = 'Position';
            if (tokenInfoText) {
                tokenInfoText.textContent = ATTENTION_VALUE_PLACEHOLDER;
                tokenInfoText.title = '';
            }
            if (tokenInfoId) {
                tokenInfoId.textContent = ATTENTION_VALUE_PLACEHOLDER;
                tokenInfoId.title = '';
            }
            if (tokenInfoPosition) {
                tokenInfoPosition.textContent = ATTENTION_VALUE_PLACEHOLDER;
                tokenInfoPosition.title = '';
            }
        };

        if (isLogitSelection) {
            const entry = resolveLogitSelectionEntry(selection);
            const tokenText = resolveLogitPreviewTokenText(label, selection) || ATTENTION_VALUE_PLACEHOLDER;
            const tokenId = resolveLogitSelectionTokenId(label, entry);
            const probability = resolveLogitSelectionProbability(label, entry);
            const tokenIdText = Number.isFinite(tokenId) ? String(Math.floor(tokenId)) : ATTENTION_VALUE_PLACEHOLDER;
            const probabilityText = Number.isFinite(probability)
                ? `${(probability * 100).toFixed(2).replace(/\.?0+$/, '')}%`
                : ATTENTION_VALUE_PLACEHOLDER;

            if (tokenInfoHeadPrimary) tokenInfoHeadPrimary.textContent = 'Token text';
            if (tokenInfoHeadSecondary) tokenInfoHeadSecondary.textContent = 'Token ID';
            if (tokenInfoHeadTertiary) tokenInfoHeadTertiary.textContent = 'Selected probability';
            if (tokenInfoText) {
                tokenInfoText.textContent = tokenText;
                tokenInfoText.title = tokenText === ATTENTION_VALUE_PLACEHOLDER ? '' : tokenText;
            }
            if (tokenInfoId) {
                tokenInfoId.textContent = tokenIdText;
                tokenInfoId.title = tokenIdText === ATTENTION_VALUE_PLACEHOLDER ? '' : tokenIdText;
            }
            if (tokenInfoPosition) {
                tokenInfoPosition.textContent = probabilityText;
                tokenInfoPosition.title = probabilityText === ATTENTION_VALUE_PLACEHOLDER ? '' : probabilityText;
            }
            if (tokenInfoRow) {
                tokenInfoRow.style.display = '';
                tokenInfoRow.dataset.layout = 'token';
                const isEmpty = tokenText === ATTENTION_VALUE_PLACEHOLDER
                    && tokenIdText === ATTENTION_VALUE_PLACEHOLDER
                    && probabilityText === ATTENTION_VALUE_PLACEHOLDER;
                tokenInfoRow.dataset.empty = isEmpty ? 'true' : 'false';
            }
            return {
                tokenText,
                tokenDisplayText: tokenText,
                tokenIdText: Number.isFinite(tokenId) ? tokenIdText : '',
                positionText: '',
                tokenEncodingNote: getIncompleteUtf8TokenNote(tokenId)
            };
        }

        const metadata = this._resolveVectorTokenPosition(selection, label);
        if (!metadata) {
            hideRows();
            return null;
        }

        const tokenText = metadata.tokenText || ATTENTION_VALUE_PLACEHOLDER;
        const tokenDisplayText = metadata.tokenDisplayText || tokenText;
        const tokenIdText = metadata.tokenIdText || ATTENTION_VALUE_PLACEHOLDER;
        const positionText = metadata.positionText || ATTENTION_VALUE_PLACEHOLDER;
        const primaryText = isPositionEmbeddingSelection ? positionText : tokenText;
        const secondaryText = isPositionEmbeddingSelection ? tokenDisplayText : tokenIdText;
        const tertiaryText = isPositionEmbeddingSelection ? ATTENTION_VALUE_PLACEHOLDER : positionText;

        if (tokenInfoHeadPrimary) tokenInfoHeadPrimary.textContent = isPositionEmbeddingSelection ? 'Position' : 'Raw token';
        if (tokenInfoHeadSecondary) tokenInfoHeadSecondary.textContent = isPositionEmbeddingSelection ? 'Associated token' : 'Token ID';
        if (tokenInfoHeadTertiary) tokenInfoHeadTertiary.textContent = 'Position';
        if (tokenInfoText) {
            tokenInfoText.textContent = primaryText;
            tokenInfoText.title = primaryText === ATTENTION_VALUE_PLACEHOLDER ? '' : primaryText;
        }
        if (tokenInfoId) {
            tokenInfoId.textContent = secondaryText;
            tokenInfoId.title = secondaryText === ATTENTION_VALUE_PLACEHOLDER ? '' : secondaryText;
        }
        if (tokenInfoPosition) {
            tokenInfoPosition.textContent = tertiaryText;
            tokenInfoPosition.title = tertiaryText === ATTENTION_VALUE_PLACEHOLDER ? '' : tertiaryText;
        }
        if (tokenInfoRow) {
            tokenInfoRow.style.display = '';
            tokenInfoRow.dataset.layout = isPositionEmbeddingSelection ? 'position' : 'token';
            const isEmpty = primaryText === ATTENTION_VALUE_PLACEHOLDER
                && secondaryText === ATTENTION_VALUE_PLACEHOLDER
                && tertiaryText === ATTENTION_VALUE_PLACEHOLDER;
            tokenInfoRow.dataset.empty = isEmpty ? 'true' : 'false';
        }
        return metadata;
    }

    _setTokenEncodingNote(note) {
        if (!this.tokenEncodingRow || !this.tokenEncodingValue) return;
        const text = (typeof note === 'string') ? note.trim() : '';
        if (text) {
            this.tokenEncodingRow.style.display = '';
            this.tokenEncodingValue.textContent = text;
        } else {
            this.tokenEncodingRow.style.display = 'none';
            this.tokenEncodingValue.textContent = '';
        }
    }

    _resolveSelectionTokenEncodingNote(selection, label, vectorMetadata = null) {
        if (vectorMetadata && typeof vectorMetadata.tokenEncodingNote === 'string' && vectorMetadata.tokenEncodingNote.trim().length) {
            return vectorMetadata.tokenEncodingNote;
        }

        const entry = resolveLogitSelectionEntry(selection);
        if (Number.isFinite(entry?.token_id)) {
            return getIncompleteUtf8TokenNote(entry.token_id);
        }

        const lower = (label || '').toLowerCase();
        if (!lower.includes('logit')) return '';
        const labelIdMatch = String(label || '').match(/\bid\s+(-?\d+)/i);
        if (!labelIdMatch) return '';
        const parsedTokenId = Number(labelIdMatch[1]);
        return getIncompleteUtf8TokenNote(parsedTokenId);
    }

    showSelection(selection) {
        if (!this.isReady || !selection || !selection.label) return;

        this._resetCopyContextFeedback();
        const label = normalizeSelectionLabel(selection.label, selection);
        const displayLabel = simplifyLayerNormParamDisplayLabel(label, selection);
        const lower = label.toLowerCase();
        const hidePreviewForSelection = false;
        const previewSelectionKey = hidePreviewForSelection ? null : buildSelectionPreviewKey(label, selection);
        const metadata = resolveMetadata(label, selection.kind, selection);
        const logitHeader = resolveLogitSelectionHeader(label, selection);
        const vectorSubtitleMetadata = this._resolveVectorTokenPosition(selection, label);
        const isAttentionScore = isAttentionScoreSelection(label, selection);
        const attentionContextForSubtitle = isAttentionScore ? this._resolveAttentionContext(selection) : null;
        const attentionScoreSummary = isAttentionScore
            ? resolveAttentionScoreSelectionSummary(selection, attentionContextForSubtitle)
            : null;
        const attentionSubtitleLine = attentionScoreSummary?.tokenContextLine || '';
        this.panel.classList.toggle('is-preview-hidden', hidePreviewForSelection);
        this.title.textContent = logitHeader?.title || displayLabel;
        if (this.subtitle) {
            if (logitHeader) {
                this.subtitle.textContent = logitHeader.subtitle;
            } else {
                const layerIndex = findUserDataNumber(selection, 'layerIndex');
                const headIndex = findUserDataNumber(selection, 'headIndex');
                const isQkvOrCachedVectorSelection = isLikelyVectorSelection(label, selection)
                    && (isQkvHeadVectorSelection(label, selection) || isKvCacheVectorSelection(selection));
                const showHead = isQkvMatrixLabel(label)
                    || isAttentionScore
                    || isQkvOrCachedVectorSelection;
                const subtitleParts = [];
                if (Number.isFinite(layerIndex)) {
                    subtitleParts.push(`Layer ${layerIndex + 1}`);
                }
                if (showHead && Number.isFinite(headIndex)) {
                    subtitleParts.push(`Head ${headIndex + 1}`);
                }
                if (isQkvOrCachedVectorSelection) {
                    let positionText = '';
                    if (vectorSubtitleMetadata && typeof vectorSubtitleMetadata.positionText === 'string') {
                        positionText = vectorSubtitleMetadata.positionText.trim();
                    }
                    if (!positionText) {
                        const tokenIndex = findUserDataNumber(selection, 'tokenIndex');
                        if (Number.isFinite(tokenIndex)) {
                            positionText = String(Math.floor(tokenIndex) + 1);
                        }
                    }
                    if (positionText) {
                        subtitleParts.push(`Position ${positionText}`);
                    }
                }
                this.subtitle.textContent = subtitleParts.join(' • ');
            }
        }
        if (this.subtitleSecondary) {
            this.subtitleSecondary.textContent = logitHeader ? '' : attentionSubtitleLine;
        }
        const hideLayerNormFields = isLayerNormSolidSelection(label);
        const isLogitTokenSelection = !!logitHeader;
        const hideTensorDimsField = hideLayerNormFields
            || isAttentionScore
            || isLogitTokenSelection;
        const isTokenChipSelection = lower.startsWith('token:') || lower.startsWith('position:');
        const isVectorMetadata = isLikelyVectorSelection(label, selection) || isTokenChipSelection;
        const dimsRow = this.inputDim?.closest('.detail-row')
            || this.outputDim?.closest('.detail-row')
            || null;
        if (this.inputDimLabel) {
            this.inputDimLabel.textContent = isTokenChipSelection
                ? 'Length (one-hot encoded)'
                : (isVectorMetadata ? 'Length' : 'Input dimension');
        }
        if (this.outputDimLabel) this.outputDimLabel.textContent = 'Output dimension';
        if (this.inputDim) this.inputDim.textContent = hideTensorDimsField
            ? ''
            : (isVectorMetadata ? metadata.length : metadata.inputDim);
        if (this.outputDim) this.outputDim.textContent = hideTensorDimsField || isVectorMetadata ? '' : metadata.outputDim;
        if (this.outputDimHalf) this.outputDimHalf.style.display = (!hideTensorDimsField && isVectorMetadata) ? 'none' : '';
        if (this.inputDimHalf) this.inputDimHalf.style.flexBasis = isVectorMetadata ? '100%' : '';
        if (dimsRow) dimsRow.style.display = hideTensorDimsField ? 'none' : '';
        const showParamCount = !hideTensorDimsField && !isVectorMetadata && metadata.hasDims;
        if (this.paramsRow) this.paramsRow.style.display = showParamCount ? '' : 'none';
        if (this.params) this.params.textContent = showParamCount ? metadata.params : '';
        const showBiasDim = !hideTensorDimsField && !isVectorMetadata && metadata.hasBiasDim;
        if (this.biasDimRow) this.biasDimRow.style.display = showBiasDim ? '' : 'none';
        if (this.biasDim) this.biasDim.textContent = showBiasDim ? metadata.biasDim : '';
        const vectorTokenMetadata = this._updateVectorTokenPositionRows(selection, label);
        this._setTokenEncodingNote(this._resolveSelectionTokenEncodingNote(selection, label, vectorTokenMetadata));
        if (this.description) {
            const desc = resolveDescription(label, selection.kind, selection);
            this._currentSelectionDescription = desc || '';
            setDescriptionContent(this.description, desc || '');
        } else {
            this._currentSelectionDescription = '';
        }
        if (this.equationsSection && this.equationsBody) {
            const equations = resolveSelectionEquations(label, selection);
            this._currentSelectionEquations = equations || '';
            setDescriptionContent(this.equationsBody, equations || '');
            const hasEquations = !!equations;
            this.equationsSection.classList.toggle('is-visible', hasEquations);
            this.equationsSection.setAttribute('aria-hidden', hasEquations ? 'false' : 'true');
            if (!hasEquations) {
                this.equationsBody.style.fontSize = '';
            } else {
                this._scheduleSelectionEquationFit();
            }
        } else {
            this._currentSelectionEquations = '';
        }
        const isParam = isParameterSelection(label);
        if (this.dataSection) {
            this.dataSection.style.display = (isParam || hideLayerNormFields) ? 'none' : '';
        }
        if (this.dataEl && (isParam || hideLayerNormFields)) this.dataEl.textContent = '';
        if (this.metaSection) {
            const rows = Array.from(this.metaSection.querySelectorAll('.detail-row, .detail-token-info'));
            const hasVisibleRow = rows.some(row => row.style.display !== 'none');
            this.metaSection.style.display = hasVisibleRow ? '' : 'none';
        }
        if (this.metaSection && this.attentionRoot && this.panel) {
            if (isQkvMatrixLabel(label)) {
                if (this.attentionRoot.parentElement === this.panel) {
                    this.panel.insertBefore(this.metaSection, this.attentionRoot);
                }
            } else if (this.dataSection && this.dataSection.parentElement === this.panel) {
                this.panel.insertBefore(this.metaSection, this.dataSection);
            }
        }
        if (this.dataEl) {
            const activationData = (selection.object && selection.object.userData && selection.object.userData.activationData)
                || (selection.info && selection.info.activationData)
                || (selection.hit && selection.hit.object && selection.hit.object.userData && selection.hit.object.userData.activationData)
                || null;
            if (!isParam && !hideLayerNormFields) {
                this.dataEl.textContent = formatActivationData(activationData);
            }
        }
        this._updateVectorLegend(selection);

        const shouldReusePreview = !!previewSelectionKey
            && !!this.currentPreview
            && this._currentPreviewSelectionKey === previewSelectionKey;

        if (!shouldReusePreview && this.currentPreview) {
            this.scene.remove(this.currentPreview);
            if (this.currentDispose) {
                try { this.currentDispose(); } catch (_) { /* no-op */ }
            }
            this.currentPreview = null;
            this.currentDispose = null;
            this.currentAnimator = null;
            this._currentPreviewSelectionKey = null;
        }

        const preview = shouldReusePreview
            ? null
            : (hidePreviewForSelection ? null : resolvePreviewObject(label, selection));
        if (!shouldReusePreview && preview?.object) {
            const previewRoot = new THREE.Group();
            previewRoot.add(preview.object);
            centerPreviewPivot(preview.object);
            this.currentPreview = previewRoot;
            this.currentDispose = (typeof preview.dispose === 'function') ? preview.dispose : null;
            this.currentAnimator = (typeof preview.animate === 'function') ? preview.animate : null;
            this._currentPreviewSelectionKey = previewSelectionKey;
            const desiredRotation = new THREE.Euler(PREVIEW_BASE_TILT_X, PREVIEW_BASE_ROTATION_Y, 0);
            if (this.currentPreview?.rotation) {
                this.currentPreview.rotation.set(0, 0, 0);
            }
            this._lastFrameTime = performance.now();
            const isVectorPreview = isLikelyVectorSelection(label, selection);
            const isQkvMatrixPreview = isQkvMatrixLabel(label);
            const isOutputProjPreview = label.toLowerCase().includes('output projection matrix');
            const paddingMultiplier = isVectorPreview
                ? PREVIEW_VECTOR_PADDING_MULT
                : (isQkvMatrixPreview ? 0.75 : (isOutputProjPreview ? 0.85 : 1));
            const distanceMultiplier = isVectorPreview
                ? PREVIEW_VECTOR_DISTANCE_MULT
                : (isQkvMatrixPreview ? 0.85 : (isOutputProjPreview ? 0.8 : 1));
            const laneZoom = getLaneZoomMultiplier(this.currentPreview);
            const isSmallScreen = this._isSmallScreen();
            const isMatrixPreview = isWeightMatrixLabel(label) || isOutputProjPreview;
            const matrixPaddingBoost = (isSmallScreen && isMatrixPreview) ? PREVIEW_MOBILE_MATRIX_PADDING_MULT : 1;
            const matrixDistanceBoost = (isSmallScreen && isMatrixPreview) ? PREVIEW_MOBILE_MATRIX_DISTANCE_MULT : 1;
            const finalPadding = paddingMultiplier * laneZoom * matrixPaddingBoost;
            const finalDistance = distanceMultiplier * laneZoom * matrixDistanceBoost;
            this._rotationSpeedMult = 1;
            this._lastFitOptions = { paddingMultiplier: finalPadding, distanceMultiplier: finalDistance };
            if (!this.isOpen) {
                this._pendingReveal = true;
                if (this.canvas) this.canvas.style.opacity = '0';
            } else {
                this._pendingReveal = false;
                if (this.canvas) this.canvas.style.opacity = '1';
                fitObjectToView(this.currentPreview, this.camera, { paddingMultiplier: finalPadding, distanceMultiplier: finalDistance });
                this._noteFit();
            }
            if (this.currentPreview?.rotation) {
                this.currentPreview.rotation.copy(desiredRotation);
            }
            this.scene.add(this.currentPreview);
        } else if (!shouldReusePreview) {
            this.currentPreview = null;
            this.currentDispose = (typeof preview?.dispose === 'function') ? preview.dispose : null;
            this.currentAnimator = (typeof preview?.animate === 'function') ? preview.animate : null;
            this._currentPreviewSelectionKey = null;
            this._rotationSpeedMult = 1;
            this._lastFitOptions = null;
            this._pendingReveal = false;
            this._pendingRevealSize = null;
            this._stopLoop();
            if (this.canvas) this.canvas.style.opacity = '1';
        }

        this._updateAttentionPreview(selection);
        this.open();
        this._scheduleSelectionEquationFit();
        this._scheduleDimensionLabelFit();
    }
}

export function initSelectionPanel(options = {}) {
    requestTokenChipFont();
    const panel = new SelectionPanel(options);
    if (!panel.isReady) {
        return { handleSelection: () => {}, close: () => {} };
    }
    return {
        handleSelection: (selection) => panel.showSelection(selection),
        close: () => panel.close(),
        updateData: (data) => panel.updateData(data)
    };
}
