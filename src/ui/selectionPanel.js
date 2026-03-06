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
    buildAttentionScoreLabel,
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
    matchesAttentionScoreSelection,
    normalizeSelectionLabel,
    resolveAttentionModeFromSelection,
    simplifyLayerNormParamDisplayLabel
} from './selectionPanelSelectionUtils.js';
import { resolveDescription, resolveSelectionEquations } from './selectionPanelNarrativeUtils.js';
import {
    GELU_PANEL_ACTION_OPEN,
    createGeluDetailView,
    isMlpMatrixSelectionLabel,
    setDescriptionGeluAction
} from './selectionPanelGeluPreview.js';
import {
    computeAttentionCellSize,
    countVisibleAttentionCellsInRow,
    getAttentionRevealOrder,
    shouldRevealAttentionCell
} from './selectionPanelAttentionRevealUtils.js';
import {
    collectVisibleContextText,
    copyTextToClipboard,
    setDescriptionContent
} from './selectionPanelCopyUtils.js';
import {
    formatActivationData,
    formatAttentionSubtitleTokenPart,
    formatTokenLabelForPreview,
    formatTokenWithIndex,
    formatValues,
    normalizeAttentionValuePart
} from './selectionPanelFormatUtils.js';
import { buildSelectionPromptContext } from './selectionPanelPromptContextUtils.js';
import {
    applyMaterialSnapshot,
    copyInstancedVectorColorsToPreview,
    extractMaterialSnapshot,
    tryCopyVectorAppearanceToPreview
} from './selectionPanelVectorCloneUtils.js';
import {
    ATTENTION_DECODE_ROW_OFFSET_MAX_PX,
    ATTENTION_DECODE_ROW_OFFSET_MIN_PX,
    ATTENTION_DECODE_ROW_OFFSET_MULT,
    ATTENTION_POP_OUT_MS,
    ATTENTION_POST_REVEAL_DURATION_MS,
    ATTENTION_POST_REVEAL_STAGGER_MAX_MS,
    ATTENTION_POST_REVEAL_STAGGER_MIN_MS,
    ATTENTION_POST_REVEAL_SWEEP_MS,
    ATTENTION_PRE_COLOR_CLAMP,
    ATTENTION_PRE_REVEAL_DURATION_MS,
    ATTENTION_PRE_REVEAL_STAGGER_MAX_MS,
    ATTENTION_PRE_REVEAL_STAGGER_MIN_MS,
    ATTENTION_PRE_REVEAL_SWEEP_MS,
    ATTENTION_PREVIEW_COLOR_DARKEN_FACTOR,
    ATTENTION_PREVIEW_GAP,
    ATTENTION_PREVIEW_GRID_GAP,
    ATTENTION_PREVIEW_MAX_TOKENS,
    ATTENTION_PREVIEW_MAX_CELL,
    ATTENTION_PREVIEW_MIN_CELL,
    ATTENTION_PREVIEW_SIZE_OPTIONS,
    ATTENTION_PREVIEW_TRIANGLE,
    ATTENTION_SCORE_DECIMALS,
    ATTENTION_VALUE_PLACEHOLDER,
    CONTEXT_LEN,
    COPY_CONTEXT_BUTTON_DEFAULT_LABEL,
    COPY_CONTEXT_EMPTY_LABEL,
    COPY_CONTEXT_ERROR_LABEL,
    COPY_CONTEXT_FADE_MS,
    COPY_CONTEXT_FEEDBACK_MS,
    COPY_CONTEXT_SUCCESS_LABEL,
    DETAIL_EQUATION_FIT_BUFFER_PX,
    DETAIL_EQUATION_FONT_MAX_PX,
    DETAIL_EQUATION_FONT_MAX_SCALE,
    DETAIL_EQUATION_FONT_MIN_PX,
    D_HEAD,
    D_MODEL,
    FINAL_MLP_COLOR,
    FINAL_VOCAB_TOP_COLOR,
    LOGIT_PREVIEW_TEXT_STYLE,
    PANEL_ACTION_HISTORY_BACK,
    PANEL_ACTION_HISTORY_FORWARD,
    PANEL_SHIFT_DURATION_MS,
    PREVIEW_BASE_DISTANCE_MULT,
    PREVIEW_BASE_ROTATION_Y,
    PREVIEW_BASE_TILT_X,
    PREVIEW_FIT_LOCK_MS,
    PREVIEW_FIT_LOCK_PX,
    PREVIEW_FIT_LOCK_RATIO,
    PREVIEW_FRAME_PADDING,
    PREVIEW_LANES,
    PREVIEW_LANE_SPACING,
    PREVIEW_MATRIX_DEPTH,
    PREVIEW_MOBILE_MATRIX_DISTANCE_MULT,
    PREVIEW_MOBILE_MATRIX_PADDING_MULT,
    PREVIEW_QKV_CONVERT_DURATION,
    PREVIEW_QKV_EXIT_DURATION,
    PREVIEW_QKV_EXIT_Y,
    PREVIEW_QKV_HOLD_DURATION,
    PREVIEW_QKV_IDLE_DURATION,
    PREVIEW_QKV_LANES,
    PREVIEW_QKV_LANE_SPACING,
    PREVIEW_QKV_LANE_STAGGER,
    PREVIEW_QKV_MATRIX_Y,
    PREVIEW_QKV_OUTPUT_Y,
    PREVIEW_QKV_RISE_DURATION,
    PREVIEW_QKV_START_Y,
    PREVIEW_QKV_X_SPREAD,
    PREVIEW_ROTATION_ENVELOPE_MARGIN,
    PREVIEW_ROTATION_SPEED,
    PREVIEW_SOLID_LANES,
    PREVIEW_TARGET_SIZE,
    PREVIEW_TILT_AMPLITUDE,
    PREVIEW_TILT_OSC_SPEED,
    PREVIEW_TOKEN_LANES,
    PREVIEW_TRAIL_COLOR,
    PREVIEW_VECTOR_BODY_INSTANCES,
    PREVIEW_VECTOR_DISTANCE_MULT,
    PREVIEW_VECTOR_HEAD_INSTANCES,
    PREVIEW_VECTOR_LARGE_SCALE,
    PREVIEW_VECTOR_PADDING_MULT,
    PREVIEW_VECTOR_SMALL_SCALE,
    RESIDUAL_COLOR_CLAMP,
    SELECTION_PANEL_TOKEN_HOVER_SOURCE,
    SPACE_TOKEN_DISPLAY,
    TOKEN_CHIP_FONT_URL,
    TOKEN_CHIP_STYLE,
    VOCAB_SIZE
} from './selectionPanelConstants.js';
import {
    MHA_MATRIX_PARAMS,
    MLP_MATRIX_PARAMS_UP,
    MLP_MATRIX_PARAMS_DOWN,
    EMBEDDING_MATRIX_PARAMS_VOCAB,
    EMBEDDING_MATRIX_PARAMS_POSITION,
    LAYER_NORM_FINAL_COLOR,
    PRISM_DIMENSIONS_PER_UNIT,
    HIDE_INSTANCE_Y_OFFSET,
    ATTENTION_POST_SOFTMAX_GRAYSCALE_MIN,
    resolveRenderPixelRatio
} from '../utils/constants.js';
import { getIncompleteUtf8TokenNote } from '../utils/tokenEncodingNotes.js';
import { createTokenChipMesh } from '../utils/tokenChipMeshFactory.js';
import { getLogitTokenColorCss, resolveLogitTokenSeed } from '../app/gpt-tower/logitColor.js';
import {
    TOKEN_CHIP_HOVER_SYNC_EVENT,
    dispatchTokenChipHoverSync,
    normalizeTokenChipEntry,
    tokenChipEntriesMatch
} from './tokenChipHoverSync.js';
import {
    MHA_FINAL_Q_COLOR,
    MHA_FINAL_K_COLOR,
    MHA_FINAL_V_COLOR,
    MHA_OUTPUT_PROJECTION_MATRIX_COLOR,
    MHA_OUTPUT_PROJECTION_MATRIX_PARAMS,
    POSITION_EMBED_COLOR,
    LN_PARAMS
} from '../animations/LayerAnimationConstants.js';
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

function resolveLegendSampleT(ratio, edgeClampRatio = 0) {
    const safeRatio = clamp01(ratio);
    const safeEdgeClampRatio = Number.isFinite(edgeClampRatio)
        ? THREE.MathUtils.clamp(edgeClampRatio, 0, 0.49)
        : 0;
    if (safeEdgeClampRatio <= 1e-6) return safeRatio;
    if (safeRatio <= safeEdgeClampRatio) return 0;
    if (safeRatio >= (1 - safeEdgeClampRatio)) return 1;
    return (safeRatio - safeEdgeClampRatio) / (1 - (safeEdgeClampRatio * 2));
}

function buildLegendGradient({
    minValue = 0,
    maxValue = 1,
    steps = 13,
    edgeClampRatio = 0,
    resolveColor = null
} = {}) {
    const safeSteps = Math.max(3, Math.min(41, Math.floor(steps)));
    const safeMin = Number.isFinite(minValue) ? minValue : 0;
    const safeMax = Number.isFinite(maxValue) ? maxValue : 1;
    const safeEdgeClampRatio = Number.isFinite(edgeClampRatio)
        ? THREE.MathUtils.clamp(edgeClampRatio, 0, 0.49)
        : 0;
    const colorResolver = typeof resolveColor === 'function'
        ? resolveColor
        : ((value) => value);
    const startPct = safeEdgeClampRatio * 100;
    const endPct = (1 - safeEdgeClampRatio) * 100;
    const stops = [];
    const startColor = colorToCss(colorResolver(safeMin));
    const endColor = colorToCss(colorResolver(safeMax));

    if (safeEdgeClampRatio > 0) {
        stops.push(`${startColor} 0%`, `${startColor} ${startPct.toFixed(2)}%`);
    }

    for (let i = 0; i < safeSteps; i += 1) {
        const t = safeSteps === 1 ? 0 : i / (safeSteps - 1);
        const value = THREE.MathUtils.lerp(safeMin, safeMax, t);
        const color = colorToCss(colorResolver(value));
        const pct = THREE.MathUtils.lerp(startPct, endPct, t).toFixed(2);
        stops.push(`${color} ${pct}%`);
    }

    if (safeEdgeClampRatio > 0) {
        stops.push(`${endColor} ${endPct.toFixed(2)}%`, `${endColor} 100%`);
    }

    return `linear-gradient(90deg, ${stops.join(', ')})`;
}

function buildSpectrumLegendGradient({ clampMax, steps = 13, darkenFactor = 1, edgeClampRatio = 0 } = {}) {
    const safeClamp = Number.isFinite(clampMax) ? Math.max(1e-6, Math.abs(clampMax)) : 1;
    return buildLegendGradient({
        minValue: -safeClamp,
        maxValue: safeClamp,
        steps,
        edgeClampRatio,
        resolveColor: (value) => darkenColor(
            mapValueToColor(value, { clampMax: safeClamp }),
            darkenFactor
        )
    });
}

function resolveAttentionScoreSelectionSummary(selectionInfo, context = null) {
    const activation = getActivationDataFromSelection(selectionInfo);
    const stageLower = String(activation?.stage || '').toLowerCase();
    if (!activation) return null;

    const normalizedLabel = normalizeSelectionLabel(selectionInfo?.label, selectionInfo);
    const isScoreStage = stageLower === 'attention.pre' || stageLower === 'attention.post';
    const isAttentionScore = isAttentionScoreSelection(normalizedLabel, selectionInfo)
        || isScoreStage;
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
    const sourceTokenText = normalizeAttentionValuePart(sourceLabel);
    const targetTokenText = normalizeAttentionValuePart(targetLabel);
    const sourceText = sourceTokenText === ATTENTION_VALUE_PLACEHOLDER ? 'Source' : sourceTokenText;
    const targetText = targetTokenText === ATTENTION_VALUE_PLACEHOLDER ? 'Target' : targetTokenText;
    const scoreText = Number.isFinite(score) ? score.toFixed(ATTENTION_SCORE_DECIMALS) : 'n/a';
    const hasSourceContext = Number.isFinite(sourceTokenIndex) || sourceLabel.length > 0;
    const hasTargetContext = Number.isFinite(targetTokenIndex) || targetLabel.length > 0;
    const tokenContext = (hasSourceContext || hasTargetContext)
        ? {
            source: {
                role: 'Source',
                tokenText: sourceTokenText,
                tokenIndex: sourceTokenIndex,
                positionText: Number.isFinite(sourceTokenIndex)
                    ? `Position ${Math.floor(sourceTokenIndex) + 1}`
                    : 'Position n/a'
            },
            target: {
                role: 'Target',
                tokenText: targetTokenText,
                tokenIndex: targetTokenIndex,
                positionText: Number.isFinite(targetTokenIndex)
                    ? `Position ${Math.floor(targetTokenIndex) + 1}`
                    : 'Position n/a'
            },
            score: {
                value: scoreText
            }
        }
        : null;
    const tokenContextLine = (hasSourceContext || hasTargetContext)
        ? `${formatAttentionSubtitleTokenPart(sourceLabel, sourceTokenIndex, 'Source')} • ${formatAttentionSubtitleTokenPart(targetLabel, targetTokenIndex, 'Target')}`
        : '';

    return {
        mode,
        row: row >= 0 ? row : null,
        col: col >= 0 ? col : null,
        tokenContextLine,
        tokenContext,
        defaultValue: {
            source: sourceText,
            target: targetText,
            score: scoreText,
            sourceTokenIndex: Number.isFinite(sourceTokenIndex) ? Math.floor(sourceTokenIndex) : null,
            targetTokenIndex: Number.isFinite(targetTokenIndex) ? Math.floor(targetTokenIndex) : null,
            empty: false
        }
    };
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

function isTouchLikePointerEvent(event) {
    return event?.pointerType === 'touch'
        || event?.pointerType === 'pen'
        || (typeof event?.type === 'string' && event.type.startsWith('touch'));
}

function resolvePointerId(event) {
    return Number.isFinite(event?.pointerId) ? event.pointerId : null;
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

function hasVocabEmbeddingLabel(lower) {
    return lower.includes('vocab embedding') || lower.includes('vocabulary embedding');
}

function hasTopVocabEmbeddingLabel(lower) {
    return lower.includes('vocab embedding (top)')
        || lower.includes('vocabulary embedding (top)')
        || lower.includes('vocab unembedding')
        || lower.includes('vocabulary unembedding');
}

function resolveFinalPreviewColor(label) {
    const lower = (label || '').toLowerCase();
    if (lower.includes('query weight matrix')) return MHA_FINAL_Q_COLOR;
    if (lower.includes('key weight matrix')) return MHA_FINAL_K_COLOR;
    if (lower.includes('value weight matrix')) return MHA_FINAL_V_COLOR;
    if (lower.includes('output projection matrix')) return MHA_OUTPUT_PROJECTION_MATRIX_COLOR;
    if (lower.includes('mlp up weight matrix')) return FINAL_MLP_COLOR;
    if (lower.includes('mlp down weight matrix')) return FINAL_MLP_COLOR;
    if (hasVocabEmbeddingLabel(lower) || lower.includes('unembedding')) {
        return hasTopVocabEmbeddingLabel(lower) || lower.includes('unembedding')
            ? FINAL_VOCAB_TOP_COLOR
            : MHA_FINAL_Q_COLOR;
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
    if (hasTopVocabEmbeddingLabel(lower) || lower.includes('unembedding')) {
        return buildMetadata(formatNumber(VOCAB_SIZE * D_MODEL), D_MODEL, VOCAB_SIZE);
    }
    if (hasVocabEmbeddingLabel(lower)) {
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

function resolvePreviewTokenId(label, selectionInfo) {
    const source = selectionInfo?.object || selectionInfo?.hit?.object;
    const directInfo = selectionInfo?.info;
    const candidates = [
        findUserDataNumber(source, 'tokenId'),
        directInfo?.tokenId,
        directInfo?.token_id,
        directInfo?.logitEntry?.tokenId,
        directInfo?.logitEntry?.token_id,
        selectionInfo?.logitEntry?.tokenId,
        selectionInfo?.logitEntry?.token_id
    ];
    for (const candidate of candidates) {
        if (Number.isFinite(candidate)) return Math.floor(candidate);
    }
    const labelIdMatch = String(label || '').match(/\bid\s+(-?\d+)/i);
    if (!labelIdMatch) return null;
    const parsed = Number(labelIdMatch[1]);
    return Number.isFinite(parsed) ? Math.floor(parsed) : null;
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
        subtitle: subtitleParts.join(' • '),
        tokenText,
        tokenId
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
    const tokenId = resolveLogitSelectionTokenId(label, resolveLogitSelectionEntry(selectionInfo));
    return buildTokenChipPreview(tokenText, { tokenId });
}

function createTokenChipShared(labelText, tokenId = null) {
    const rawText = (typeof labelText === 'string') ? labelText : '';
    const text = rawText.trim().length ? rawText : SPACE_TOKEN_DISPLAY;
    return createTokenChipMesh({
        labelText: text,
        secondaryText: Number.isFinite(tokenId) ? String(Math.floor(tokenId)) : '',
        style: TOKEN_CHIP_STYLE,
        font: tokenChipFont
    });
}

function buildTokenChipPreview(labelText, { tokenId = null } = {}) {
    const shared = createTokenChipShared(labelText, tokenId);
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
    const activation = getActivationDataFromSelection(selectionInfo);
    const stageLower = String(activation?.stage || '').toLowerCase();
    const isAttentionScore = isAttentionScoreSelection(selectionInfo?.label, selectionInfo)
        || selectionInfo?.kind === 'attentionSphere'
        || stageLower === 'attention.pre'
        || stageLower === 'attention.post';
    if (!isAttentionScore) return null;

    const mode = resolveAttentionModeFromSelection(selectionInfo)
        || (stageLower.includes('post') ? 'post' : 'pre');
    const score = mode === 'post' ? activation?.postScore : activation?.preScore;
    const color = TMP_COLOR;
    let usedLiveColor = false;
    if (source?.isInstancedMesh) {
        if (!source.userData?._attentionSphereInstanced && selectionInfo?.kind !== 'attentionSphere') return null;
        const instanceId = hit && typeof hit.instanceId === 'number' ? hit.instanceId : null;
        if (Number.isFinite(instanceId)) {
            color.copy(source.material?.color || 0xffffff);
            if (typeof source.getColorAt === 'function') {
                try {
                    source.getColorAt(instanceId, color);
                    usedLiveColor = true;
                } catch (_) {
                    usedLiveColor = false;
                }
            }
        }
    }
    if (!usedLiveColor) {
        if (mode === 'post') {
            color.copy(mapValueToGrayscale(
                Number.isFinite(score) ? score : 0.5,
                { minValue: ATTENTION_POST_SOFTMAX_GRAYSCALE_MIN }
            ));
        } else {
            color.copy(mapValueToColor(
                Number.isFinite(score) ? score : 0,
                { clampMax: ATTENTION_PRE_COLOR_CLAMP }
            ));
        }
    }
    const geometry = new THREE.SphereGeometry(10, 12, 12);
    const material = new THREE.MeshStandardMaterial({
        color: color.clone(),
        roughness: 0.28,
        metalness: 0.1,
        emissive: color.clone().multiplyScalar(mode === 'post' ? 0.22 : 0.32),
        emissiveIntensity: mode === 'post' ? 0.28 : 0.36
    });
    const mesh = new THREE.Mesh(geometry, material);
    mesh.scale.setScalar(0.8);
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
    if (isAttentionScoreSelection(label, selectionInfo)) return false;
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
        return buildTokenChipPreview(extractTokenText(label), {
            tokenId: lower.startsWith('token:') ? resolvePreviewTokenId(label, selectionInfo) : null
        });
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
    if (hasVocabEmbeddingLabel(lower) || lower.includes('unembedding')) {
        const clonePreview = buildSelectionClonePreview(selectionInfo, label)
            || buildDirectClonePreview(selectionInfo);
        if (clonePreview) {
            const color = hasTopVocabEmbeddingLabel(lower) || lower.includes('unembedding')
                ? FINAL_VOCAB_TOP_COLOR
                : MHA_FINAL_Q_COLOR;
            applyFinalColorToObject(clonePreview.object, color);
            return clonePreview;
        }
        const color = hasTopVocabEmbeddingLabel(lower) || lower.includes('unembedding')
            ? FINAL_VOCAB_TOP_COLOR
            : MHA_FINAL_Q_COLOR;
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
        this.promptContextRow = document.getElementById('detailPromptContextRow');
        this.promptContextTokens = document.getElementById('detailPromptContextTokens');
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
        this.attentionValueScoreInner = this.attentionValueScore?.querySelector('.detail-attention-score-pill') || null;
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
        this._panelTokenHoverEntry = null;
        this._mirroredTokenHoverEntry = null;
        this._tokenHoverSyncSource = SELECTION_PANEL_TOKEN_HOVER_SOURCE;
        this._copyContextFeedbackTimer = null;
        this._copyContextFadeTimer = null;
        this._copyContextDefaultLabel = this.copyContextBtnLabel?.textContent?.trim()
            || this.copyContextBtn?.textContent?.trim()
            || COPY_CONTEXT_BUTTON_DEFAULT_LABEL;
        this._setCopyContextButtonLabel(this._copyContextDefaultLabel);
        this._geluDetailView = createGeluDetailView(this.panel);
        this._geluDetailOpen = false;
        this._geluSourceSelection = null;
        this._lastSelection = null;
        this._lastSelectionLabel = '';
        this._historyEntries = [];
        this._historyIndex = -1;
        this._historyBackBtn = null;
        this._historyForwardBtn = null;
        this._createHistoryNavigationControls();

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
        this._onSubtitleSecondaryClick = this._onSubtitleSecondaryClick.bind(this);
        this._onAttentionScoreValueClick = this._onAttentionScoreValueClick.bind(this);
        this._onAttentionScoreValuePointerUp = this._onAttentionScoreValuePointerUp.bind(this);
        this._onAttentionScoreValueKeydown = this._onAttentionScoreValueKeydown.bind(this);
        this._onPanelTokenChipClick = this._onPanelTokenChipClick.bind(this);
        this._onPanelTokenChipKeydown = this._onPanelTokenChipKeydown.bind(this);
        this._onPanelTokenChipPointerOver = this._onPanelTokenChipPointerOver.bind(this);
        this._onPanelTokenChipPointerOut = this._onPanelTokenChipPointerOut.bind(this);
        this._onPanelTokenChipFocusIn = this._onPanelTokenChipFocusIn.bind(this);
        this._onPanelTokenChipFocusOut = this._onPanelTokenChipFocusOut.bind(this);
        this._onTokenChipHoverSync = this._onTokenChipHoverSync.bind(this);
        this._onLegendPointerDown = this._onLegendPointerDown.bind(this);
        this._onLegendPointerMove = this._onLegendPointerMove.bind(this);
        this._onLegendPointerLeave = this._onLegendPointerLeave.bind(this);
        this._onLegendPointerUp = this._onLegendPointerUp.bind(this);
        this._onLegendPointerCancel = this._onLegendPointerCancel.bind(this);
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
        this._attentionScoreLink = null;
        this._legendHoverState = {
            kind: null,
            ratio: null
        };
        this._legendTouchState = {
            kind: null,
            pointerId: null
        };
        this._legendEdgeClampRatios = {
            vector: 0,
            attention: 0
        };
        this._legendHoverUi = {
            vector: this._createLegendHoverUi(this.vectorLegendBar, 'vector'),
            attention: this._createLegendHoverUi(this.attentionLegend, 'attention')
        };

        this.closeBtn?.addEventListener('click', () => this.close({ clearHistory: false }));
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
        [this.vectorLegendBar, this.attentionLegend].filter(Boolean).forEach((bar) => {
            bar.addEventListener('pointerdown', this._onLegendPointerDown);
            bar.addEventListener('pointerenter', this._onLegendPointerMove);
            bar.addEventListener('pointermove', this._onLegendPointerMove);
            bar.addEventListener('pointerleave', this._onLegendPointerLeave);
            bar.addEventListener('pointerup', this._onLegendPointerUp);
            bar.addEventListener('pointercancel', this._onLegendPointerCancel);
        });
        this.attentionValueScore?.addEventListener('click', this._onAttentionScoreValueClick);
        this.attentionValueScore?.addEventListener('pointerup', this._onAttentionScoreValuePointerUp);
        this.attentionValueScore?.addEventListener('keydown', this._onAttentionScoreValueKeydown);
        this._setAttentionValue(this._attentionValueDefault);
        this.panel.addEventListener('pointerdown', this._onPanelPointerDown, { capture: true });
        this.panel.addEventListener('pointerenter', this._onPanelPointerEnter);
        this.panel.addEventListener('pointerleave', this._onPanelPointerLeave);
        this.subtitleSecondary?.addEventListener('click', this._onSubtitleSecondaryClick);
        this.panel.addEventListener('click', this._onPanelTokenChipClick);
        this.panel.addEventListener('keydown', this._onPanelTokenChipKeydown);
        this.panel.addEventListener('pointerover', this._onPanelTokenChipPointerOver);
        this.panel.addEventListener('pointerout', this._onPanelTokenChipPointerOut);
        this.panel.addEventListener('focusin', this._onPanelTokenChipFocusIn);
        this.panel.addEventListener('focusout', this._onPanelTokenChipFocusOut);
        window.addEventListener('resize', this._onResize);
        window.addEventListener(TOKEN_CHIP_HOVER_SYNC_EVENT, this._onTokenChipHoverSync);
        if (window.visualViewport && typeof window.visualViewport.addEventListener === 'function') {
            window.visualViewport.addEventListener('resize', this._onResize);
        }
        document.addEventListener('keydown', this._onKeydown);
        document.addEventListener('pointerdown', this._onDocumentPointerDown, { capture: true });
        this._touchClickCleanup = initTouchClickFallback(this.panel, {
            selector: '.toggle-row, .detail-token-nav-chip[data-token-nav="true"], .detail-history-btn, .detail-description-action-link, .detail-copy-context-btn, .detail-attention-score-link[data-attention-score-link="true"]'
        });
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

    _createLegendHoverUi(bar, kind) {
        if (!bar || typeof document === 'undefined') return null;
        bar.dataset.legendKind = kind;

        const marker = document.createElement('div');
        marker.className = 'legend-hover-marker';
        marker.setAttribute('aria-hidden', 'true');

        const tooltip = document.createElement('div');
        tooltip.className = 'legend-hover-tooltip';
        tooltip.setAttribute('aria-hidden', 'true');

        const swatch = document.createElement('span');
        swatch.className = 'legend-hover-swatch';

        const value = document.createElement('span');
        value.className = 'legend-hover-value';

        tooltip.append(swatch, value);
        bar.append(marker, tooltip);

        return {
            bar,
            marker,
            tooltip,
            swatch,
            value
        };
    }

    _formatLegendHoverValue(value, { signed = false, decimals = 2 } = {}) {
        if (!Number.isFinite(value)) return ATTENTION_VALUE_PLACEHOLDER;
        const safeDecimals = Number.isFinite(decimals)
            ? THREE.MathUtils.clamp(Math.floor(decimals), 0, 6)
            : 2;
        const safeValue = Math.abs(value) < 1e-8 ? 0 : value;
        let text = safeValue.toFixed(safeDecimals).replace(/\.?0+$/, '');
        if (signed && safeValue > 0) text = `+${text}`;
        return text;
    }

    _resolveLegendBar(kind) {
        if (kind === 'vector') return this.vectorLegendBar;
        if (kind === 'attention') return this.attentionLegend;
        return null;
    }

    _resolveLegendEdgeClampRatio(kind) {
        const bar = this._resolveLegendBar(kind);
        if (!bar || typeof window === 'undefined' || typeof window.getComputedStyle !== 'function') {
            return this._legendEdgeClampRatios?.[kind] || 0;
        }
        const styles = window.getComputedStyle(bar);
        const edgeInsetPx = Number.parseFloat(styles.getPropertyValue('--legend-edge-tick-inset'));
        const width = bar.getBoundingClientRect().width
            || Number.parseFloat(styles.width)
            || bar.clientWidth
            || 0;
        if (Number.isFinite(edgeInsetPx) && edgeInsetPx >= 0 && width > 0) {
            const ratio = THREE.MathUtils.clamp(edgeInsetPx / width, 0, 0.49);
            this._legendEdgeClampRatios[kind] = ratio;
            return ratio;
        }
        return this._legendEdgeClampRatios?.[kind] || 0;
    }

    _buildAttentionLegendGradient(mode) {
        const safeMode = mode === 'post' ? 'post' : 'pre';
        const edgeClampRatio = this._resolveLegendEdgeClampRatio('attention');
        if (safeMode === 'post') {
            return buildLegendGradient({
                minValue: 0,
                maxValue: 1,
                steps: 13,
                edgeClampRatio,
                resolveColor: (value) => resolveAttentionPreviewCellColor(value, 'post')
            });
        }
        return buildSpectrumLegendGradient({
            clampMax: ATTENTION_PRE_COLOR_CLAMP,
            steps: 15,
            darkenFactor: ATTENTION_PREVIEW_COLOR_DARKEN_FACTOR,
            edgeClampRatio
        });
    }

    _buildVectorLegendGradient() {
        return buildSpectrumLegendGradient({
            clampMax: RESIDUAL_COLOR_CLAMP,
            steps: 15,
            edgeClampRatio: this._resolveLegendEdgeClampRatio('vector')
        });
    }

    _refreshVisibleLegendGradients() {
        if (this.vectorLegend?.classList.contains('is-visible') && this.vectorLegendBar) {
            this.vectorLegendBar.style.setProperty('--vector-legend-gradient', this._buildVectorLegendGradient());
        }
        if (this.attentionRoot?.classList.contains('is-visible') && this.attentionLegend) {
            this.attentionLegend.style.setProperty(
                '--attention-legend-gradient',
                this._buildAttentionLegendGradient(this.attentionMode)
            );
        }
    }

    _resolveLegendHoverSample(kind, ratio) {
        const safeRatio = clamp01(ratio);
        const sampleT = resolveLegendSampleT(safeRatio, this._resolveLegendEdgeClampRatio(kind));
        if (kind === 'vector') {
            const value = THREE.MathUtils.lerp(-RESIDUAL_COLOR_CLAMP, RESIDUAL_COLOR_CLAMP, sampleT);
            return {
                ratio: safeRatio,
                value,
                valueLabel: this._formatLegendHoverValue(value, { signed: true, decimals: 2 }),
                colorCss: colorToCss(mapValueToColor(value, { clampMax: RESIDUAL_COLOR_CLAMP }))
            };
        }
        if (kind !== 'attention') return null;

        const mode = this.attentionMode === 'post' ? 'post' : 'pre';
        const value = mode === 'post'
            ? sampleT
            : THREE.MathUtils.lerp(-ATTENTION_PRE_COLOR_CLAMP, ATTENTION_PRE_COLOR_CLAMP, sampleT);
        return {
            ratio: safeRatio,
            value,
            valueLabel: this._formatLegendHoverValue(value, {
                signed: mode === 'pre',
                decimals: mode === 'post' ? ATTENTION_SCORE_DECIMALS : 2
            }),
            colorCss: colorToCss(resolveAttentionPreviewCellColor(value, mode))
        };
    }

    _resolveLegendKindFromBar(bar) {
        const kind = bar?.dataset?.legendKind;
        return kind === 'attention' || kind === 'vector' ? kind : null;
    }

    _isLegendKindVisible(kind) {
        if (kind === 'vector') {
            return !!(this.vectorLegendBar && this.vectorLegend?.classList.contains('is-visible'));
        }
        if (kind === 'attention') {
            return !!(this.attentionLegend && this.attentionRoot?.classList.contains('is-visible'));
        }
        return false;
    }

    _hideLegendHover(kind = null) {
        const kinds = kind ? [kind] : ['vector', 'attention'];
        for (let i = 0; i < kinds.length; i += 1) {
            const ui = this._legendHoverUi?.[kinds[i]];
            if (!ui) continue;
            ui.marker.classList.remove('is-visible');
            ui.tooltip.classList.remove('is-visible');
        }
        if (!kind || this._legendTouchState.kind === kind) {
            this._clearLegendTouchState(kind);
        }
        if (!kind || this._legendHoverState.kind === kind) {
            this._legendHoverState.kind = null;
            this._legendHoverState.ratio = null;
        }
    }

    _renderLegendHover(kind, ratio) {
        if (!kind) return;
        const ui = this._legendHoverUi?.[kind];
        if (!ui || !this._isLegendKindVisible(kind)) {
            this._hideLegendHover(kind);
            return;
        }

        const sample = this._resolveLegendHoverSample(kind, ratio);
        const rect = ui.bar.getBoundingClientRect();
        if (!sample || !(rect.width > 0)) {
            this._hideLegendHover(kind);
            return;
        }

        this._hideLegendHover(kind === 'vector' ? 'attention' : 'vector');

        ui.swatch.style.backgroundColor = sample.colorCss;
        ui.value.textContent = sample.valueLabel;

        const markerX = sample.ratio * rect.width;
        ui.marker.style.left = `${markerX}px`;

        const tooltipWidth = ui.tooltip.offsetWidth || 0;
        const tooltipX = (tooltipWidth + 8 >= rect.width)
            ? rect.width / 2
            : THREE.MathUtils.clamp(
                markerX,
                tooltipWidth / 2 + 4,
                rect.width - tooltipWidth / 2 - 4
            );
        ui.tooltip.style.left = `${tooltipX}px`;

        ui.marker.classList.add('is-visible');
        ui.tooltip.classList.add('is-visible');
        this._legendHoverState.kind = kind;
        this._legendHoverState.ratio = sample.ratio;
    }

    _refreshLegendHover(kind = null) {
        const targetKind = kind || this._legendHoverState.kind;
        if (!targetKind || this._legendHoverState.kind !== targetKind) return;
        if (!Number.isFinite(this._legendHoverState.ratio)) {
            this._hideLegendHover(targetKind);
            return;
        }
        this._renderLegendHover(targetKind, this._legendHoverState.ratio);
    }

    _resolveLegendPointerRatio(bar, event) {
        if (!(bar instanceof Element) || !Number.isFinite(event?.clientX)) return null;
        const rect = bar.getBoundingClientRect();
        if (!(rect.width > 0)) return null;
        return clamp01((event.clientX - rect.left) / rect.width);
    }

    _clearLegendTouchState(kind = null) {
        if (kind && this._legendTouchState.kind !== kind) return;
        this._legendTouchState.kind = null;
        this._legendTouchState.pointerId = null;
    }

    _hasActiveLegendTouchPointer(event, kind) {
        if (!isTouchLikePointerEvent(event)) return false;
        if (!this._legendTouchState.kind || this._legendTouchState.kind !== kind) return false;
        const pointerId = resolvePointerId(event);
        return this._legendTouchState.pointerId === null
            || pointerId === null
            || this._legendTouchState.pointerId === pointerId;
    }

    _onLegendPointerDown(event) {
        const bar = event?.currentTarget instanceof Element ? event.currentTarget : null;
        const kind = this._resolveLegendKindFromBar(bar);
        const ratio = this._resolveLegendPointerRatio(bar, event);
        if (!kind || !Number.isFinite(ratio)) {
            this._hideLegendHover(kind);
            return;
        }

        if (isTouchLikePointerEvent(event)) {
            this._legendTouchState.kind = kind;
            this._legendTouchState.pointerId = resolvePointerId(event);
            if (Number.isFinite(event?.pointerId) && typeof bar?.setPointerCapture === 'function') {
                try {
                    bar.setPointerCapture(event.pointerId);
                } catch (_) { /* no-op */ }
            }
        }

        this._renderLegendHover(kind, ratio);
    }

    _onLegendPointerMove(event) {
        const bar = event?.currentTarget instanceof Element ? event.currentTarget : null;
        const kind = this._resolveLegendKindFromBar(bar);
        const ratio = this._resolveLegendPointerRatio(bar, event);
        if (!bar || !kind || !Number.isFinite(ratio)) {
            this._hideLegendHover(kind);
            return;
        }

        if (isTouchLikePointerEvent(event) && !this._hasActiveLegendTouchPointer(event, kind)) {
            return;
        }
        this._renderLegendHover(kind, ratio);
    }

    _onLegendPointerLeave(event) {
        if (isTouchLikePointerEvent(event)) {
            return;
        }
        const bar = event?.currentTarget instanceof Element ? event.currentTarget : null;
        const kind = this._resolveLegendKindFromBar(bar);
        this._hideLegendHover(kind);
    }

    _onLegendPointerUp(event) {
        const bar = event?.currentTarget instanceof Element ? event.currentTarget : null;
        const kind = this._resolveLegendKindFromBar(bar);
        if (!this._hasActiveLegendTouchPointer(event, kind)) return;
        if (Number.isFinite(event?.pointerId) && typeof bar?.releasePointerCapture === 'function') {
            try {
                bar.releasePointerCapture(event.pointerId);
            } catch (_) { /* no-op */ }
        }
        this._clearLegendTouchState(kind);
    }

    _onLegendPointerCancel(event) {
        const bar = event?.currentTarget instanceof Element ? event.currentTarget : null;
        const kind = this._resolveLegendKindFromBar(bar);
        if (!this._hasActiveLegendTouchPointer(event, kind)) return;
        if (Number.isFinite(event?.pointerId) && typeof bar?.releasePointerCapture === 'function') {
            try {
                bar.releasePointerCapture(event.pointerId);
            } catch (_) { /* no-op */ }
        }
        this._clearLegendTouchState(kind);
        this._hideLegendHover(kind);
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
            const legendLines = collectVisibleContextText(this.vectorLegend, {
                excludeSelectors: '.legend-hover-tooltip, .legend-hover-marker'
            });
            if (legendLines.length) {
                sections.push(`Legend:\n${legendLines.join('\n')}`);
            }
        }

        if (this.attentionRoot?.classList.contains('is-visible')) {
            const attentionLines = collectVisibleContextText(this.attentionRoot, {
                excludeSelectors: '.legend-hover-tooltip, .legend-hover-marker'
            });
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
            if (this._geluDetailOpen) {
                this._geluDetailView?.resizeAndRender();
            }
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
        this._refreshVisibleLegendGradients();
        this._refreshLegendHover();
        if (this._geluDetailOpen) {
            this._geluDetailView?.resizeAndRender();
        }
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

    _refreshReusedPreview() {
        if (!this.currentPreview) return;
        this._pendingReveal = false;
        this._pendingRevealSize = null;
        if (this._pendingRevealTimer) {
            clearTimeout(this._pendingRevealTimer);
            this._pendingRevealTimer = null;
        }
        if (this.canvas) {
            this.canvas.style.opacity = '1';
        }
        if (this.isOpen) {
            this._scheduleResize();
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
            this.close({ clearHistory: false });
        }
    }

    _onClosePointerDown(event) {
        if (!this.isOpen) return;
        event.preventDefault();
        event.stopPropagation();
        this.close({ clearHistory: false });
    }

    _blockPreviewGesture(event) {
        const isTouch = isTouchLikePointerEvent(event);
        if (!isTouch) return;
        if (event.cancelable) event.preventDefault();
        event.stopPropagation();
    }

    _setHoverLabelSuppression(suppressed) {
        if (this.engine && typeof this.engine.setHoverLabelsSuppressed === 'function') {
            this.engine.setHoverLabelsSuppressed(!!suppressed);
        }
    }

    _syncHoverLabelSuppressionFromHoverState() {
        if (!this.panel || typeof this.panel.matches !== 'function') return;
        const shouldSuppress = this.isOpen && this.panel.matches(':hover');
        this._setHoverLabelSuppression(shouldSuppress);
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
        const isTouch = isTouchLikePointerEvent(event);
        if (!isTouch) return;
        if (this.engine && typeof this.engine.resetInteractionState === 'function') {
            this.engine.resetInteractionState();
        }
        if (typeof document !== 'undefined' && document.body) {
            document.body.classList.add('touch-ui');
        }
        const legendBar = event?.target instanceof Element
            ? event.target.closest('.vector-legend-bar, .attention-legend-bar')
            : null;
        if (!legendBar || !this.panel?.contains(legendBar)) {
            this._hideLegendHover();
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
        this._setPanelTokenHoverEntry(null, { emit: true });
        this._hideLegendHover();
    }

    _createHistoryNavigationControls() {
        if (!this.panel || typeof document === 'undefined') return;
        const header = this.panel.querySelector('.detail-header');
        const titleGroup = header?.querySelector('.detail-title-group');
        if (!header || !titleGroup) return;

        const nav = document.createElement('div');
        nav.className = 'detail-history-nav';
        nav.setAttribute('aria-label', 'Detail navigation history');

        const backBtn = document.createElement('button');
        backBtn.type = 'button';
        backBtn.className = 'detail-history-btn detail-history-btn--back';
        backBtn.dataset.detailAction = PANEL_ACTION_HISTORY_BACK;
        backBtn.textContent = '\u2039';
        backBtn.setAttribute('aria-label', 'Previous detail page');

        const forwardBtn = document.createElement('button');
        forwardBtn.type = 'button';
        forwardBtn.className = 'detail-history-btn detail-history-btn--forward';
        forwardBtn.dataset.detailAction = PANEL_ACTION_HISTORY_FORWARD;
        forwardBtn.textContent = '\u203a';
        forwardBtn.setAttribute('aria-label', 'Next detail page');

        nav.append(backBtn, forwardBtn);
        header.insertBefore(nav, titleGroup);
        this._historyBackBtn = backBtn;
        this._historyForwardBtn = forwardBtn;
        this._updateHistoryNavigationControls();
    }

    _buildHistoryEntry(type, selection = null) {
        if (!selection || !selection.label) return null;
        const label = normalizeSelectionLabel(selection.label, selection);
        const keyBase = buildSelectionPreviewKey(label, selection);
        return {
            type,
            selection,
            label,
            key: `${type}:${keyBase}`
        };
    }

    _historyEntriesEqual(a, b) {
        return !!a && !!b && a.type === b.type && a.key === b.key;
    }

    _pushHistoryEntry(entry) {
        if (!entry) return;
        const current = this._historyEntries[this._historyIndex] || null;
        if (this._historyEntriesEqual(current, entry)) {
            this._updateHistoryNavigationControls();
            return;
        }
        if (this._historyIndex < this._historyEntries.length - 1) {
            this._historyEntries = this._historyEntries.slice(0, this._historyIndex + 1);
        }
        const existingIndex = this._historyEntries.findIndex((candidate) => this._historyEntriesEqual(candidate, entry));
        if (existingIndex >= 0) {
            this._historyEntries.splice(existingIndex, 1);
            if (existingIndex <= this._historyIndex) {
                this._historyIndex = Math.max(-1, this._historyIndex - 1);
            }
        }
        this._historyEntries.push(entry);
        this._historyIndex = this._historyEntries.length - 1;
        this._updateHistoryNavigationControls();
    }

    _resetHistoryNavigation() {
        this._historyEntries = [];
        this._historyIndex = -1;
        this._updateHistoryNavigationControls();
    }

    _updateHistoryNavigationControls() {
        const hasBack = this._historyIndex > 0;
        const hasForward = this._historyIndex >= 0 && this._historyIndex < this._historyEntries.length - 1;
        if (this._historyBackBtn) {
            this._historyBackBtn.disabled = !hasBack;
            this._historyBackBtn.dataset.disabled = hasBack ? 'false' : 'true';
        }
        if (this._historyForwardBtn) {
            this._historyForwardBtn.disabled = !hasForward;
            this._historyForwardBtn.dataset.disabled = hasForward ? 'false' : 'true';
        }
    }

    _applyHistoryEntry(entry) {
        if (!entry || !entry.selection) return false;
        if (entry.type === 'gelu') {
            this.showSelection(entry.selection, { fromHistory: true });
            this._openGeluDetailPreview({
                fromHistory: true,
                sourceSelection: entry.selection
            });
            return true;
        }
        if (entry.type === 'selection') {
            this.showSelection(entry.selection, { fromHistory: true });
            return true;
        }
        return false;
    }

    _navigateHistoryBack() {
        if (this._historyIndex <= 0) return false;
        this._historyIndex -= 1;
        this._updateHistoryNavigationControls();
        return this._applyHistoryEntry(this._historyEntries[this._historyIndex]);
    }

    _navigateHistoryForward() {
        if (this._historyIndex < 0 || this._historyIndex >= this._historyEntries.length - 1) return false;
        this._historyIndex += 1;
        this._updateHistoryNavigationControls();
        return this._applyHistoryEntry(this._historyEntries[this._historyIndex]);
    }

    _setTitleText(titleText = '') {
        if (!this.title) return;
        this.title.classList.remove('detail-title--token-context');
        this.title.textContent = String(titleText || '');
    }

    _configureTokenNavChip(chip, {
        tokenText = ATTENTION_VALUE_PLACEHOLDER,
        tokenIndex = null,
        tokenId = null,
        allowNavigation = true
    } = {}) {
        if (!chip) return;
        const safeTokenText = normalizeAttentionValuePart(tokenText);
        const safeTokenIndex = Number.isFinite(tokenIndex) ? Math.floor(tokenIndex) : null;
        const safeTokenId = Number.isFinite(tokenId) ? Math.floor(tokenId) : null;
        const canNavigate = !!allowNavigation && safeTokenText !== ATTENTION_VALUE_PLACEHOLDER;

        chip.classList.add('detail-token-nav-chip');
        chip.dataset.tokenText = safeTokenText;
        if (Number.isFinite(safeTokenIndex)) {
            chip.dataset.tokenIndex = String(safeTokenIndex);
        } else {
            delete chip.dataset.tokenIndex;
        }
        if (Number.isFinite(safeTokenId)) {
            chip.dataset.tokenId = String(safeTokenId);
        } else {
            delete chip.dataset.tokenId;
        }
        chip.dataset.tokenNav = canNavigate ? 'true' : 'false';
        if (canNavigate) {
            chip.tabIndex = 0;
            chip.setAttribute('role', 'button');
            chip.setAttribute('aria-label', `Open token details for ${safeTokenText}`);
        } else {
            chip.removeAttribute('tabindex');
            chip.removeAttribute('role');
            chip.removeAttribute('aria-label');
        }
    }

    _setTokenChipTitleContext({
        tokenText = ATTENTION_VALUE_PLACEHOLDER,
        tokenIndex = null,
        tokenId = null,
        prefixText = 'Token:',
        allowNavigation = true
    } = {}) {
        if (!this.title) return;
        const safeTokenText = normalizeAttentionValuePart(tokenText);
        const safeTokenIndex = Number.isFinite(tokenIndex) ? Math.floor(tokenIndex) : null;
        let safeTokenId = Number.isFinite(tokenId) ? Math.floor(tokenId) : null;
        const safePrefixText = (typeof prefixText === 'string' && prefixText.trim().length)
            ? prefixText.trim()
            : 'Token:';
        if (!Number.isFinite(safeTokenId)
            && Number.isFinite(safeTokenIndex)
            && this.activationSource
            && typeof this.activationSource.getTokenId === 'function') {
            safeTokenId = this.activationSource.getTokenId(safeTokenIndex);
            safeTokenId = Number.isFinite(safeTokenId) ? Math.floor(safeTokenId) : null;
        }
        const seed = resolveLogitTokenSeed(
            { token_id: safeTokenId, token: safeTokenText },
            Number.isFinite(safeTokenIndex) ? safeTokenIndex : 0
        );

        this.title.classList.add('detail-title--token-context');
        const fragment = document.createDocumentFragment();

        const prefix = document.createElement('span');
        prefix.className = 'detail-title-text-part';
        prefix.textContent = safePrefixText;

        const chip = document.createElement('span');
        chip.className = 'detail-subtitle-token-chip detail-title-token-chip';
        chip.style.setProperty('--token-color-border', getLogitTokenColorCss(seed, 0.92));
        chip.style.setProperty('--token-color-fill', getLogitTokenColorCss(seed, 0.2));
        chip.style.setProperty('--token-color-fill-hover', getLogitTokenColorCss(seed, 0.28));
        chip.textContent = safeTokenText;
        chip.title = safeTokenText === ATTENTION_VALUE_PLACEHOLDER ? '' : safeTokenText;
        this._configureTokenNavChip(chip, {
            tokenText: safeTokenText,
            tokenIndex: safeTokenIndex,
            tokenId: safeTokenId,
            allowNavigation
        });

        fragment.append(prefix, chip);
        this.title.replaceChildren(fragment);
    }

    _setSubtitleSecondaryText(text = '') {
        if (!this.subtitleSecondary) return;
        this.subtitleSecondary.classList.remove(
            'detail-subtitle--attention-context',
            'detail-subtitle--token-context'
        );
        this.subtitleSecondary.textContent = String(text || '');
    }

    _setSubtitleSecondaryTokenContext({
        tokenText = '',
        tokenIndex = null,
        tokenId = null,
        prefixText = 'Token:',
        allowNavigation = true
    } = {}) {
        if (!this.subtitleSecondary) return;

        const safeTokenIndex = Number.isFinite(tokenIndex) ? Math.floor(tokenIndex) : null;
        let safeTokenId = Number.isFinite(tokenId) ? Math.floor(tokenId) : null;
        let safeTokenText = normalizeAttentionValuePart(tokenText, '');
        if (!safeTokenText && Number.isFinite(safeTokenIndex)) {
            safeTokenText = `Token ${safeTokenIndex + 1}`;
        }
        if (!safeTokenText) {
            this._setSubtitleSecondaryText('');
            return;
        }
        if (!Number.isFinite(safeTokenId)
            && Number.isFinite(safeTokenIndex)
            && this.activationSource
            && typeof this.activationSource.getTokenId === 'function') {
            safeTokenId = this.activationSource.getTokenId(safeTokenIndex);
            safeTokenId = Number.isFinite(safeTokenId) ? Math.floor(safeTokenId) : null;
        }

        const seed = resolveLogitTokenSeed(
            { token_id: safeTokenId, token: safeTokenText },
            Number.isFinite(safeTokenIndex) ? safeTokenIndex : 0
        );

        this._setSubtitleSecondaryText('');
        this.subtitleSecondary.classList.add('detail-subtitle--token-context');

        const fragment = document.createDocumentFragment();
        const prefix = document.createElement('span');
        prefix.className = 'detail-subtitle-context-label';
        prefix.textContent = `${prefixText} `;

        const chip = document.createElement('span');
        chip.className = 'detail-subtitle-token-chip detail-subtitle-secondary-token-chip';
        chip.style.setProperty('--token-color-border', getLogitTokenColorCss(seed, 0.92));
        chip.style.setProperty('--token-color-fill', getLogitTokenColorCss(seed, 0.2));
        chip.style.setProperty('--token-color-fill-hover', getLogitTokenColorCss(seed, 0.28));
        chip.textContent = safeTokenText;
        chip.title = safeTokenText;
        this._configureTokenNavChip(chip, {
            tokenText: safeTokenText,
            tokenIndex: safeTokenIndex,
            tokenId: safeTokenId,
            allowNavigation
        });

        fragment.append(prefix, chip);
        this.subtitleSecondary.replaceChildren(fragment);
        this._applyTokenChipHoverState();
    }

    _openSelectionForTokenChip(chip, event = null) {
        if (!chip || chip.dataset.tokenNav !== 'true') return false;
        const tokenText = normalizeAttentionValuePart(chip.dataset.tokenText || chip.textContent);
        if (!tokenText || tokenText === ATTENTION_VALUE_PLACEHOLDER) return false;

        const rawTokenIndex = Number(chip.dataset.tokenIndex);
        const tokenIndex = Number.isFinite(rawTokenIndex) ? Math.floor(rawTokenIndex) : null;
        const rawTokenId = Number(chip.dataset.tokenId);
        const tokenId = Number.isFinite(rawTokenId) ? Math.floor(rawTokenId) : null;
        const tokenObj = this._findTokenChipSceneObject({ tokenIndex, tokenId, tokenText });

        const info = {
            tokenLabel: tokenText
        };
        if (Number.isFinite(tokenIndex)) info.tokenIndex = tokenIndex;
        if (Number.isFinite(tokenId)) info.tokenId = tokenId;

        const objectLabel = typeof tokenObj?.userData?.label === 'string' ? tokenObj.userData.label : '';
        const label = objectLabel.toLowerCase().startsWith('token:')
            ? objectLabel
            : `Token: ${tokenText}`;

        if (event && typeof event.preventDefault === 'function') event.preventDefault();
        if (event && typeof event.stopPropagation === 'function') event.stopPropagation();

        const selection = {
            label,
            info,
            kind: 'label'
        };
        if (tokenObj) selection.object = tokenObj;
        this.showSelection(selection);
        return true;
    }

    _onPanelTokenChipClick(event) {
        if (this._handlePanelActionTrigger(event)) return;
        const target = event?.target;
        const chip = target && typeof target.closest === 'function'
            ? target.closest('.detail-token-nav-chip')
            : null;
        if (!chip || !this.panel || !this.panel.contains(chip)) return;
        this._openSelectionForTokenChip(chip, event);
    }

    _onPanelTokenChipKeydown(event) {
        const key = event?.key;
        if (key !== 'Enter' && key !== ' ' && key !== 'Spacebar') return;
        const target = event?.target;
        const chip = target && typeof target.closest === 'function'
            ? target.closest('.detail-token-nav-chip')
            : null;
        if (!chip || !this.panel || !this.panel.contains(chip)) return;
        this._openSelectionForTokenChip(chip, event);
    }

    _resolvePanelTokenChipTarget(target) {
        if (!this.panel || !target || typeof target.closest !== 'function') return null;
        const chip = target.closest('.detail-token-nav-chip[data-token-nav="true"]');
        if (!chip || !this.panel.contains(chip)) return null;
        return chip;
    }

    _extractPanelTokenChipEntry(chip) {
        if (!chip) return null;
        return normalizeTokenChipEntry({
            tokenLabel: normalizeAttentionValuePart(chip.dataset.tokenText || chip.textContent),
            tokenIndex: chip.dataset.tokenIndex,
            tokenId: chip.dataset.tokenId
        });
    }

    _setPanelTokenHoverEntry(entry = null, { emit = true } = {}) {
        const normalizedEntry = normalizeTokenChipEntry(entry);
        const wasEmpty = !this._panelTokenHoverEntry;
        const nextEmpty = !normalizedEntry;
        const unchanged = (wasEmpty && nextEmpty)
            || (!wasEmpty && !nextEmpty && tokenChipEntriesMatch(this._panelTokenHoverEntry, normalizedEntry));
        if (!unchanged) {
            this._panelTokenHoverEntry = normalizedEntry;
            if (emit) {
                dispatchTokenChipHoverSync(normalizedEntry, {
                    active: !!normalizedEntry,
                    source: this._tokenHoverSyncSource
                });
            }
        }
        this._applyTokenChipHoverState();
    }

    _applyTokenChipHoverState() {
        if (!this.panel) return;
        const chips = this.panel.querySelectorAll('.detail-token-nav-chip');
        chips.forEach((chip) => {
            const canNavigate = chip.dataset.tokenNav === 'true';
            const chipEntry = canNavigate ? this._extractPanelTokenChipEntry(chip) : null;
            const isActive = !!chipEntry && (
                tokenChipEntriesMatch(chipEntry, this._panelTokenHoverEntry)
                || tokenChipEntriesMatch(chipEntry, this._mirroredTokenHoverEntry)
            );
            chip.classList.toggle('is-token-chip-active', isActive);
            chip.dataset.tokenActive = isActive ? 'true' : 'false';
        });
    }

    _onPanelTokenChipPointerOver(event) {
        const chip = this._resolvePanelTokenChipTarget(event?.target);
        if (!chip) return;
        this._setPanelTokenHoverEntry(this._extractPanelTokenChipEntry(chip), { emit: true });
    }

    _onPanelTokenChipPointerOut(event) {
        const fromChip = this._resolvePanelTokenChipTarget(event?.target);
        if (!fromChip) return;
        const toChip = this._resolvePanelTokenChipTarget(event?.relatedTarget);
        this._setPanelTokenHoverEntry(
            toChip ? this._extractPanelTokenChipEntry(toChip) : null,
            { emit: true }
        );
    }

    _onPanelTokenChipFocusIn(event) {
        const chip = this._resolvePanelTokenChipTarget(event?.target);
        if (!chip) return;
        this._setPanelTokenHoverEntry(this._extractPanelTokenChipEntry(chip), { emit: true });
    }

    _onPanelTokenChipFocusOut(event) {
        const fromChip = this._resolvePanelTokenChipTarget(event?.target);
        if (!fromChip) return;
        const toChip = this._resolvePanelTokenChipTarget(event?.relatedTarget);
        this._setPanelTokenHoverEntry(
            toChip ? this._extractPanelTokenChipEntry(toChip) : null,
            { emit: true }
        );
    }

    _onTokenChipHoverSync(event) {
        const detail = event?.detail || null;
        if (!detail || detail.source === this._tokenHoverSyncSource) return;
        this._mirroredTokenHoverEntry = detail.active ? normalizeTokenChipEntry(detail) : null;
        this._applyTokenChipHoverState();
    }

    _setSubtitlePrimaryQkvTokenContext({
        prefixParts = [],
        tokenText = ATTENTION_VALUE_PLACEHOLDER,
        positionText = ATTENTION_VALUE_PLACEHOLDER,
        tokenIndex = null,
        tokenId = null,
        tokenFirst = false
    } = {}) {
        if (!this.subtitle) return;
        const safePrefixParts = Array.isArray(prefixParts)
            ? prefixParts.filter((part) => typeof part === 'string' && part.trim().length > 0)
            : [];
        const safeTokenText = normalizeAttentionValuePart(tokenText);
        const safePositionText = normalizeAttentionValuePart(positionText).toLowerCase();
        const safeTokenFirst = tokenFirst === true;

        this.subtitle.classList.add('detail-subtitle--qkv-token-context');
        const fragment = document.createDocumentFragment();
        const appendSeparator = () => {
            const separator = document.createElement('span');
            separator.className = 'detail-subtitle-separator';
            separator.textContent = ' • ';
            fragment.appendChild(separator);
        };

        const appendTokenContext = () => {
            if (fragment.childNodes.length > 0) appendSeparator();
            const tokenPart = document.createElement('span');
            tokenPart.className = 'detail-subtitle-qkv-token-part';

            const chip = document.createElement('span');
            chip.className = 'detail-subtitle-token-chip';
            const seed = resolveLogitTokenSeed(
                {
                    token_id: Number.isFinite(tokenId) ? Math.floor(tokenId) : null,
                    token: safeTokenText
                },
                Number.isFinite(tokenIndex) ? Math.floor(tokenIndex) : 0
            );
            chip.style.setProperty('--token-color-border', getLogitTokenColorCss(seed, 0.92));
            chip.style.setProperty('--token-color-fill', getLogitTokenColorCss(seed, 0.2));
            chip.style.setProperty('--token-color-fill-hover', getLogitTokenColorCss(seed, 0.28));
            chip.textContent = safeTokenText;
            chip.title = safeTokenText === ATTENTION_VALUE_PLACEHOLDER ? '' : safeTokenText;
            this._configureTokenNavChip(chip, {
                tokenText: safeTokenText,
                tokenIndex,
                tokenId
            });

            const position = document.createElement('span');
            position.className = 'detail-subtitle-qkv-position';
            position.textContent = `(position ${safePositionText})`;

            tokenPart.append(chip, position);
            fragment.appendChild(tokenPart);
        };

        if (safeTokenFirst) appendTokenContext();

        safePrefixParts.forEach((part) => {
            if (fragment.childNodes.length > 0) appendSeparator();
            const textPart = document.createElement('span');
            textPart.className = 'detail-subtitle-text-part';
            textPart.textContent = part;
            fragment.appendChild(textPart);
        });

        if (!safeTokenFirst) appendTokenContext();
        this.subtitle.replaceChildren(fragment);
    }

    _findTokenChipSceneObject({
        tokenIndex = null,
        tokenId = null,
        tokenText = ''
    } = {}) {
        const scene = this.engine?.scene;
        if (!scene || typeof scene.traverse !== 'function') return null;

        const normalizedTokenText = formatTokenLabelForPreview(tokenText);
        let fallbackMatch = null;
        let visibleMatch = null;

        scene.traverse((node) => {
            if (!node || !node.userData || node.isScene) return;
            const label = typeof node.userData.label === 'string'
                ? node.userData.label
                : '';
            if (!label.toLowerCase().startsWith('token:')) return;

            const nodeTokenIndex = Number.isFinite(node.userData.tokenIndex)
                ? Math.floor(node.userData.tokenIndex)
                : null;
            const nodeTokenId = Number.isFinite(node.userData.tokenId)
                ? Math.floor(node.userData.tokenId)
                : null;
            const nodeTokenText = formatTokenLabelForPreview(node.userData.tokenLabel);

            if (Number.isFinite(tokenIndex) && Number.isFinite(nodeTokenIndex) && nodeTokenIndex !== tokenIndex) return;
            if (Number.isFinite(tokenId) && Number.isFinite(nodeTokenId) && nodeTokenId !== tokenId) return;
            if (
                normalizedTokenText
                && normalizedTokenText !== ATTENTION_VALUE_PLACEHOLDER
                && nodeTokenText
                && nodeTokenText !== ATTENTION_VALUE_PLACEHOLDER
                && nodeTokenText !== normalizedTokenText
            ) {
                return;
            }

            if (!fallbackMatch) fallbackMatch = node;
            if (!visibleMatch && node.visible !== false) visibleMatch = node;
        });

        return visibleMatch || fallbackMatch;
    }

    _onSubtitleSecondaryClick(event) {
        const target = event?.target;
        const chip = target && typeof target.closest === 'function'
            ? target.closest('.detail-attention-context-chip')
            : null;
        if (!chip || !this.subtitleSecondary || !this.subtitleSecondary.contains(chip)) return;
        this._openSelectionForTokenChip(chip, event);
    }

    _handlePanelActionTrigger(event) {
        const target = event?.target;
        if (!target || typeof target.closest !== 'function' || !this.panel) return false;
        const actionEl = target.closest('[data-detail-action]');
        if (!actionEl || !this.panel.contains(actionEl)) return false;
        const action = String(actionEl.dataset.detailAction || '').trim();
        if (!action) return false;

        if (typeof event.preventDefault === 'function') event.preventDefault();
        if (typeof event.stopPropagation === 'function') event.stopPropagation();

        if (action === PANEL_ACTION_HISTORY_BACK) {
            this._navigateHistoryBack();
            return true;
        }
        if (action === PANEL_ACTION_HISTORY_FORWARD) {
            this._navigateHistoryForward();
            return true;
        }
        if (action === GELU_PANEL_ACTION_OPEN) {
            this._openGeluDetailPreview();
            return true;
        }
        return false;
    }

    _openGeluDetailPreview({ fromHistory = false, sourceSelection = null } = {}) {
        const resolvedSelection = sourceSelection || this._lastSelection;
        const sourceLabel = this._lastSelectionLabel;
        const resolvedLabel = resolvedSelection?.label
            ? normalizeSelectionLabel(resolvedSelection.label, resolvedSelection)
            : sourceLabel;
        if (!resolvedSelection || !isMlpMatrixSelectionLabel(resolvedLabel)) return;
        this._geluSourceSelection = resolvedSelection;
        this._geluDetailOpen = true;
        this.panel.classList.add('is-gelu-view-open');
        this._setTitleText('GELU Activation');
        if (this.subtitle) {
            this.subtitle.classList.remove('detail-subtitle--qkv-token-context');
            this.subtitle.textContent = 'Gaussian Error Linear Unit used in GPT-2 MLP blocks';
        }
        this._setSubtitleSecondaryText('');
        this._geluDetailView?.setVisible(true);
        this._geluDetailView?.resizeAndRender();
        this._stopLoop();
        this._setAttentionVisibility(false);
        this._setPanelTokenHoverEntry(null, { emit: true });
        if (!fromHistory) {
            const entry = this._buildHistoryEntry('gelu', resolvedSelection);
            this._pushHistoryEntry(entry);
        } else {
            this._updateHistoryNavigationControls();
        }
    }

    _closeGeluDetailPreview({ restoreSelection = false, restartLoop = true } = {}) {
        if (!this._geluDetailOpen) return false;
        this._geluDetailOpen = false;
        this.panel.classList.remove('is-gelu-view-open');
        this._geluDetailView?.setVisible(false);
        const sourceSelection = this._geluSourceSelection;
        this._geluSourceSelection = null;
        if (restoreSelection && sourceSelection) {
            this.showSelection(sourceSelection);
            return true;
        }
        if (restartLoop && this.isOpen) {
            this._startLoop();
            this._scheduleResize();
        }
        return true;
    }

    _setAttentionVisibility(visible) {
        if (!this.attentionRoot) return;
        if (visible) {
            this.attentionRoot.classList.add('is-visible');
            this.attentionRoot.setAttribute('aria-hidden', 'false');
        } else {
            this.attentionRoot.classList.remove('is-visible');
            this.attentionRoot.setAttribute('aria-hidden', 'true');
            this._hideLegendHover('attention');
        }
    }

    _setAttentionValueTokenChip(cell, tokenText, {
        tokenIndex = null,
        tokenId = null,
        fallbackSeed = 0
    } = {}) {
        if (!cell) return;
        const safeText = normalizeAttentionValuePart(tokenText);
        cell.textContent = '';
        if (safeText === ATTENTION_VALUE_PLACEHOLDER) {
            cell.textContent = ATTENTION_VALUE_PLACEHOLDER;
            cell.title = '';
            return;
        }

        let resolvedTokenId = Number.isFinite(tokenId) ? Math.floor(tokenId) : null;
        const resolvedTokenIndex = Number.isFinite(tokenIndex) ? Math.floor(tokenIndex) : null;
        if (!Number.isFinite(resolvedTokenId)
            && Number.isFinite(resolvedTokenIndex)
            && this.activationSource
            && typeof this.activationSource.getTokenId === 'function') {
            resolvedTokenId = this.activationSource.getTokenId(resolvedTokenIndex);
        }
        const seed = resolveLogitTokenSeed(
            { token_id: resolvedTokenId, token: safeText },
            Number.isFinite(resolvedTokenIndex) ? resolvedTokenIndex : fallbackSeed
        );

        const chip = document.createElement('span');
        chip.className = 'attention-value-token-chip';
        chip.style.setProperty('--token-color-border', getLogitTokenColorCss(seed, 0.92));
        chip.style.setProperty('--token-color-fill', getLogitTokenColorCss(seed, 0.2));
        chip.style.setProperty('--token-color-fill-hover', getLogitTokenColorCss(seed, 0.28));
        chip.textContent = safeText;
        chip.title = safeText;
        this._configureTokenNavChip(chip, {
            tokenText: safeText,
            tokenIndex: resolvedTokenIndex,
            tokenId: resolvedTokenId
        });
        cell.appendChild(chip);
        cell.title = safeText;
    }

    _setAttentionValue(value = null) {
        if (!this.attentionValue) return;
        const safeValue = value && typeof value === 'object' ? value : null;
        const source = normalizeAttentionValuePart(safeValue?.source);
        const target = normalizeAttentionValuePart(safeValue?.target);
        const score = normalizeAttentionValuePart(safeValue?.score);
        const attentionScoreLink = safeValue?.attentionScoreLink || null;
        const sourceTokenIndex = Number.isFinite(safeValue?.sourceTokenIndex) ? Math.floor(safeValue.sourceTokenIndex) : null;
        const targetTokenIndex = Number.isFinite(safeValue?.targetTokenIndex) ? Math.floor(safeValue.targetTokenIndex) : null;
        const sourceTokenId = Number.isFinite(safeValue?.sourceTokenId) ? Math.floor(safeValue.sourceTokenId) : null;
        const targetTokenId = Number.isFinite(safeValue?.targetTokenId) ? Math.floor(safeValue.targetTokenId) : null;
        const isEmpty = safeValue ? safeValue.empty === true : true;
        if (this.attentionValueSource) {
            this._setAttentionValueTokenChip(this.attentionValueSource, source, {
                tokenIndex: sourceTokenIndex,
                tokenId: sourceTokenId,
                fallbackSeed: 0
            });
        }
        if (this.attentionValueTarget) {
            this._setAttentionValueTokenChip(this.attentionValueTarget, target, {
                tokenIndex: targetTokenIndex,
                tokenId: targetTokenId,
                fallbackSeed: 1
            });
        }
        if (this.attentionValueScore) {
            const scoreEl = this.attentionValueScoreInner || this.attentionValueScore;
            scoreEl.textContent = score;
            scoreEl.title = score === ATTENTION_VALUE_PLACEHOLDER ? '' : score;
            this.attentionValueScore.title = score === ATTENTION_VALUE_PLACEHOLDER ? '' : score;
        }
        this.attentionValue.dataset.empty = isEmpty ? 'true' : 'false';
        this._setAttentionScoreValueLink(
            !isEmpty ? attentionScoreLink : null,
            { sourceText: source, targetText: target, scoreText: score }
        );
        this._applyTokenChipHoverState();
    }

    _buildAttentionScoreValueLink({
        mode = this.attentionMode,
        row = null,
        col = null,
        sourceTokenIndex = null,
        targetTokenIndex = null
    } = {}) {
        const context = this._attentionContext;
        if (!context) return null;

        const safeMode = mode === 'post' ? 'post' : 'pre';
        const layerIndex = Number.isFinite(context.layerIndex) ? Math.floor(context.layerIndex) : null;
        const headIndex = Number.isFinite(context.headIndex) ? Math.floor(context.headIndex) : null;
        const tokenIndices = Array.isArray(context.tokenIndices) ? context.tokenIndices : [];
        const resolvedRow = Number.isFinite(row) ? Math.floor(row) : null;
        const resolvedCol = Number.isFinite(col) ? Math.floor(col) : null;
        const resolvedSourceTokenIndex = Number.isFinite(sourceTokenIndex)
            ? Math.floor(sourceTokenIndex)
            : (Number.isFinite(resolvedRow) && Number.isFinite(tokenIndices[resolvedRow])
                ? Math.floor(tokenIndices[resolvedRow])
                : null);
        const resolvedTargetTokenIndex = Number.isFinite(targetTokenIndex)
            ? Math.floor(targetTokenIndex)
            : (Number.isFinite(resolvedCol) && Number.isFinite(tokenIndices[resolvedCol])
                ? Math.floor(tokenIndices[resolvedCol])
                : null);

        if (
            !Number.isFinite(layerIndex)
            || !Number.isFinite(headIndex)
            || !Number.isFinite(resolvedSourceTokenIndex)
            || !Number.isFinite(resolvedTargetTokenIndex)
        ) {
            return null;
        }

        return {
            mode: safeMode,
            layerIndex,
            headIndex,
            row: resolvedRow,
            col: resolvedCol,
            sourceTokenIndex: resolvedSourceTokenIndex,
            targetTokenIndex: resolvedTargetTokenIndex
        };
    }

    _setAttentionScoreValueLink(link = null, {
        sourceText = '',
        targetText = '',
        scoreText = ''
    } = {}) {
        if (!this.attentionValueScore) return;
        const hasScore = normalizeAttentionValuePart(scoreText) !== ATTENTION_VALUE_PLACEHOLDER;
        const nextLink = link && typeof link === 'object'
            ? this._buildAttentionScoreValueLink(link)
            : null;

        this._attentionScoreLink = nextLink;
        this.attentionValueScore.classList.toggle('detail-attention-score-link', hasScore);
        this.attentionValueScore.dataset.attentionScoreLink = nextLink ? 'true' : 'false';

        if (!nextLink) {
            this.attentionValueScore.removeAttribute('tabindex');
            this.attentionValueScore.removeAttribute('role');
            this.attentionValueScore.removeAttribute('aria-label');
            delete this.attentionValueScore.dataset.attentionScoreMode;
            return;
        }

        const modeText = nextLink.mode === 'post' ? 'post-softmax' : 'pre-softmax';
        const safeSourceText = normalizeAttentionValuePart(sourceText, 'source token');
        const safeTargetText = normalizeAttentionValuePart(targetText, 'target token');
        this.attentionValueScore.dataset.attentionScoreMode = nextLink.mode;
        this.attentionValueScore.tabIndex = 0;
        this.attentionValueScore.setAttribute('role', 'button');
        this.attentionValueScore.setAttribute(
            'aria-label',
            `Open ${modeText} attention score for ${safeSourceText} to ${safeTargetText}`
        );
    }

    _findAttentionScoreSceneSelection(link) {
        const scene = this.engine?.scene || null;
        if (!scene || typeof scene.traverse !== 'function') return null;

        let match = null;
        scene.traverse((node) => {
            if (match || !node?.isInstancedMesh || !node.userData?._attentionSphereInstanced) return;
            const entries = Array.isArray(node.userData.instanceEntries) ? node.userData.instanceEntries : null;
            const labels = Array.isArray(node.userData.instanceLabels) ? node.userData.instanceLabels : null;
            if (!entries || entries.length === 0) return;

            for (let instanceId = 0; instanceId < entries.length; instanceId += 1) {
                const entry = entries[instanceId];
                if (!entry) continue;
                const label = typeof labels?.[instanceId] === 'string'
                    ? labels[instanceId]
                    : buildAttentionScoreLabel(link.mode);
                const selection = {
                    label,
                    kind: 'attentionSphere',
                    info: entry,
                    object: node,
                    hit: {
                        object: node,
                        instanceId
                    }
                };
                if (!matchesAttentionScoreSelection(selection, {
                    mode: link.mode,
                    layerIndex: link.layerIndex,
                    headIndex: link.headIndex,
                    tokenIndex: link.sourceTokenIndex,
                    keyTokenIndex: link.targetTokenIndex
                })) {
                    continue;
                }
                match = selection;
                break;
            }
        });

        return match;
    }

    _buildFallbackAttentionScoreSelection(link) {
        const label = buildAttentionScoreLabel(link.mode);
        const activationData = {
            label,
            stage: `attention.${link.mode}`,
            layerIndex: link.layerIndex,
            headIndex: link.headIndex,
            tokenIndex: link.sourceTokenIndex,
            keyTokenIndex: link.targetTokenIndex
        };

        if (this.activationSource && typeof this.activationSource.getTokenString === 'function') {
            const sourceTokenLabel = this.activationSource.getTokenString(link.sourceTokenIndex);
            const targetTokenLabel = this.activationSource.getTokenString(link.targetTokenIndex);
            if (typeof sourceTokenLabel === 'string') activationData.tokenLabel = sourceTokenLabel;
            if (typeof targetTokenLabel === 'string') activationData.keyTokenLabel = targetTokenLabel;
        }

        if (this.activationSource && typeof this.activationSource.getAttentionScore === 'function') {
            const score = this.activationSource.getAttentionScore(
                link.layerIndex,
                link.mode,
                link.headIndex,
                link.sourceTokenIndex,
                link.targetTokenIndex
            );
            if (Number.isFinite(score)) {
                if (link.mode === 'post') activationData.postScore = score;
                else activationData.preScore = score;
            }
        }

        return {
            label,
            kind: 'attentionSphere',
            info: { activationData }
        };
    }

    _resolveLinkedAttentionScoreSelection(link = null) {
        if (!link) return null;
        if (matchesAttentionScoreSelection(this._lastSelection, {
            mode: link.mode,
            layerIndex: link.layerIndex,
            headIndex: link.headIndex,
            tokenIndex: link.sourceTokenIndex,
            keyTokenIndex: link.targetTokenIndex
        })) {
            return this._lastSelection;
        }

        return this._findAttentionScoreSceneSelection(link)
            || this._buildFallbackAttentionScoreSelection(link);
    }

    _openLinkedAttentionScoreSelection(event = null) {
        const selection = this._resolveLinkedAttentionScoreSelection(this._attentionScoreLink);
        if (!selection?.label) return false;

        if (typeof event?.preventDefault === 'function') event.preventDefault();
        if (typeof event?.stopPropagation === 'function') event.stopPropagation();

        const normalizedLabel = normalizeSelectionLabel(selection.label, selection);
        const previewSelectionKey = buildSelectionPreviewKey(normalizedLabel, selection);
        if (
            previewSelectionKey
            && this.currentPreview
            && this._currentPreviewSelectionKey === previewSelectionKey
        ) {
            return true;
        }

        this.showSelection(selection, { scrollPanelToTop: true });
        return true;
    }

    _onAttentionScoreValueClick(event) {
        this._openLinkedAttentionScoreSelection(event);
    }

    _onAttentionScoreValuePointerUp(event) {
        const pointerType = String(event?.pointerType || '').toLowerCase();
        if (pointerType !== 'touch' && pointerType !== 'pen') return;
        this._openLinkedAttentionScoreSelection(event);
    }

    _onAttentionScoreValueKeydown(event) {
        const key = event?.key;
        if (key !== 'Enter' && key !== ' ' && key !== 'Spacebar') return;
        this._openLinkedAttentionScoreSelection(event);
    }

    _setSubtitleSecondaryAttentionContext(context = null) {
        if (!this.subtitleSecondary) return;
        this._setSubtitleSecondaryText('');
        if (!context || !context.source || !context.target) return;

        const resolveTokenId = (part) => {
            const directTokenId = Number.isFinite(part?.tokenId) ? Math.floor(part.tokenId) : null;
            if (Number.isFinite(directTokenId)) return directTokenId;
            const tokenIndex = Number.isFinite(part?.tokenIndex) ? Math.floor(part.tokenIndex) : null;
            if (!Number.isFinite(tokenIndex)) return null;
            if (!this.activationSource || typeof this.activationSource.getTokenId !== 'function') return null;
            return this.activationSource.getTokenId(tokenIndex);
        };

        const buildContextPart = (part, fallbackRole, fallbackIndex = 0) => {
            const roleRaw = normalizeAttentionValuePart(part?.role, fallbackRole);
            const roleText = roleRaw
                ? `${roleRaw.charAt(0).toUpperCase()}${roleRaw.slice(1).toLowerCase()}`
                : fallbackRole;
            const tokenText = normalizeAttentionValuePart(part?.tokenText);
            const positionRaw = normalizeAttentionValuePart(part?.positionText, 'position n/a');
            const positionText = positionRaw
                ? `${positionRaw.charAt(0).toLowerCase()}${positionRaw.slice(1)}`
                : 'position n/a';
            const tokenIndex = Number.isFinite(part?.tokenIndex) ? Math.floor(part.tokenIndex) : null;
            const tokenId = resolveTokenId(part);
            const seed = resolveLogitTokenSeed(
                { token_id: tokenId, token: tokenText },
                Number.isFinite(tokenIndex) ? tokenIndex : fallbackIndex
            );

            const partEl = document.createElement('span');
            partEl.className = 'detail-attention-context-part';

            const roleEl = document.createElement('span');
            roleEl.className = 'detail-attention-context-role';
            roleEl.textContent = `${roleText}:`;

            const chipEl = document.createElement('span');
            chipEl.className = 'detail-attention-context-chip';
            chipEl.style.setProperty('--token-color-border', getLogitTokenColorCss(seed, 0.92));
            chipEl.style.setProperty('--token-color-fill', getLogitTokenColorCss(seed, 0.2));
            chipEl.style.setProperty('--token-color-fill-hover', getLogitTokenColorCss(seed, 0.28));
            chipEl.textContent = tokenText;
            chipEl.title = tokenText === ATTENTION_VALUE_PLACEHOLDER ? '' : tokenText;
            this._configureTokenNavChip(chipEl, {
                tokenText,
                tokenIndex,
                tokenId
            });

            const positionEl = document.createElement('span');
            positionEl.className = 'detail-attention-context-position';
            positionEl.textContent = `(${positionText})`;

            partEl.append(roleEl, chipEl, positionEl);
            return partEl;
        };

        this.subtitleSecondary.classList.add('detail-subtitle--attention-context');
        const sourcePart = buildContextPart(context.source, 'Source', 0);
        const targetPart = buildContextPart(context.target, 'Target', 1);
        const mainContext = document.createElement('span');
        mainContext.className = 'detail-attention-context-main';
        mainContext.append(sourcePart, targetPart);

        const scoreValue = normalizeAttentionValuePart(context?.score?.value, ATTENTION_VALUE_PLACEHOLDER);
        const scoreWrap = document.createElement('span');
        scoreWrap.className = 'detail-attention-context-score';
        const scoreValueEl = document.createElement('span');
        scoreValueEl.className = 'detail-attention-context-score-value';
        scoreValueEl.textContent = scoreValue;
        scoreValueEl.title = scoreValue === ATTENTION_VALUE_PLACEHOLDER ? '' : scoreValue;
        scoreWrap.append(scoreValueEl);

        this.subtitleSecondary.replaceChildren(mainContext, scoreWrap);
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
        cell.classList.remove(
            'post-softmax-reveal',
            'post-softmax-reveal-focus',
            'pre-softmax-reveal',
            'pre-softmax-reveal-focus'
        );
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
        const useFocusedPostReveal = mode === 'post' && this._hasAttentionFocusCell();
        const useFocusedPreReveal = mode === 'pre' && this._shouldSuppressAttentionEntryHighlight();
        const postAnimDuration = ATTENTION_POST_REVEAL_DURATION_MS;
        const preAnimDuration = ATTENTION_PRE_REVEAL_DURATION_MS;
        let hasAnyValue = false;
        for (let row = 0; row < count; row += 1) {
            const rowCells = this._attentionCells[row];
            if (!rowCells) continue;
            const shouldAnimateRow = mode === 'post' && this._attentionPostAnimQueue.has(row);
            const visibleCellsInRow = countVisibleAttentionCellsInRow(row, count, ATTENTION_PREVIEW_TRIANGLE);
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
                    cell.classList.remove(
                        'post-softmax-reveal',
                        'post-softmax-reveal-focus',
                        'pre-softmax-reveal',
                        'pre-softmax-reveal-focus'
                    );
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
                    if (mode === 'post') {
                        if (shouldAnimateRow && !this._attentionPostAnimatedRows.has(row)) {
                            const revealOrder = getAttentionRevealOrder(row, col, count, ATTENTION_PREVIEW_TRIANGLE);
                            const revealClass = useFocusedPostReveal
                                && !cell.classList.contains('is-hovered')
                                && !cell.classList.contains('is-pinned')
                                ? 'post-softmax-reveal-focus'
                                : 'post-softmax-reveal';
                            this._applyAttentionCellRevealAnimation(
                                cell,
                                revealClass,
                                revealOrder * postAnimStagger,
                                postAnimDuration
                            );
                        } else {
                            cell.classList.remove('post-softmax-reveal', 'post-softmax-reveal-focus');
                            cell.style.animationDelay = '';
                            cell.style.animationDuration = '';
                        }
                    } else if (mode === 'pre') {
                        if (wasEmpty) {
                            const revealOrder = getAttentionRevealOrder(row, col, count, ATTENTION_PREVIEW_TRIANGLE);
                            this._applyAttentionCellRevealAnimation(
                                cell,
                                useFocusedPreReveal ? 'pre-softmax-reveal-focus' : 'pre-softmax-reveal',
                                revealOrder * preAnimStagger,
                                preAnimDuration
                            );
                        } else {
                            cell.classList.remove('pre-softmax-reveal', 'pre-softmax-reveal-focus');
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
                        cell.classList.remove('post-softmax-reveal', 'post-softmax-reveal-focus');
                        cell.style.animationDelay = '';
                        cell.style.animationDuration = '';
                    } else if (mode === 'pre') {
                        cell.classList.remove('pre-softmax-reveal', 'pre-softmax-reveal-focus');
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
            ? (() => {
                const defaultValue = this._attentionSelectionSummary?.defaultValue;
                if (!defaultValue) {
                    return {
                        source: ATTENTION_VALUE_PLACEHOLDER,
                        target: ATTENTION_VALUE_PLACEHOLDER,
                        score: ATTENTION_VALUE_PLACEHOLDER,
                        empty: true
                    };
                }
                return {
                    ...defaultValue,
                    attentionScoreLink: this._buildAttentionScoreValueLink({
                        mode,
                        row: this._attentionSelectionSummary?.row,
                        col: this._attentionSelectionSummary?.col,
                        sourceTokenIndex: defaultValue.sourceTokenIndex,
                        targetTokenIndex: defaultValue.targetTokenIndex
                    })
                };
            })()
            : {
                source: ATTENTION_VALUE_PLACEHOLDER,
                target: ATTENTION_VALUE_PLACEHOLDER,
                score: ATTENTION_VALUE_PLACEHOLDER,
                empty: true
            };
        this._setAttentionValue(this._attentionValueDefault);

        const count = tokenIndices.length;
        this._attentionCells = Array.from({ length: count }, () => Array(count).fill(null));
        let cellSize = computeAttentionCellSize(count, ATTENTION_PREVIEW_SIZE_OPTIONS);
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
                    const rowTokenIndex = tokenIndices[row];
                    const colTokenIndex = tokenIndices[col];
                    if (Number.isFinite(rowTokenIndex)) {
                        cell.dataset.rowTokenIndex = String(Math.floor(rowTokenIndex));
                    } else {
                        delete cell.dataset.rowTokenIndex;
                    }
                    if (Number.isFinite(colTokenIndex)) {
                        cell.dataset.colTokenIndex = String(Math.floor(colTokenIndex));
                    } else {
                        delete cell.dataset.colTokenIndex;
                    }
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
            this.attentionLegend.style.setProperty('--attention-legend-gradient', this._buildAttentionLegendGradient('post'));
            this.attentionLegend.dataset.mid = '';
            this.attentionLegendLow.textContent = '';
            this.attentionLegendHigh.textContent = '';
            this._updateAttentionLegendTickLabels('post');
            this._refreshLegendHover('attention');
            return;
        }

        const gradient = this._buildAttentionLegendGradient('pre');
        this.attentionLegend.style.setProperty('--attention-legend-gradient', gradient);
        this.attentionLegend.dataset.mid = '';
        this.attentionLegendLow.textContent = '';
        this.attentionLegendHigh.textContent = '';
        this._updateAttentionLegendTickLabels('pre');
        this._refreshLegendHover('attention');
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
                tick.dataset.label = `≤-${RESIDUAL_COLOR_CLAMP}`;
                continue;
            }
            if (ratio >= (1 - edgeEpsilon)) {
                tick.dataset.label = `≥+${RESIDUAL_COLOR_CLAMP}`;
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
            this._hideLegendHover('vector');
            return;
        }

        const gradient = this._buildVectorLegendGradient();
        if (this.vectorLegendBar) {
            this.vectorLegendBar.style.setProperty('--vector-legend-gradient', gradient);
        }
        this._updateVectorLegendTickLabels();
        if (this.vectorLegendLow) this.vectorLegendLow.textContent = '';
        if (this.vectorLegendMid) this.vectorLegendMid.textContent = '';
        if (this.vectorLegendHigh) this.vectorLegendHigh.textContent = '';
        this.vectorLegend.classList.add('is-visible');
        this.vectorLegend.setAttribute('aria-hidden', 'false');
        this._refreshLegendHover('vector');
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
        const rawRowTokenIndex = Number(cell.dataset.rowTokenIndex);
        const rawColTokenIndex = Number(cell.dataset.colTokenIndex);
        const sourceTokenIndex = Number.isFinite(rawRowTokenIndex) ? Math.floor(rawRowTokenIndex) : null;
        const targetTokenIndex = Number.isFinite(rawColTokenIndex) ? Math.floor(rawColTokenIndex) : null;
        const sourceText = normalizeAttentionValuePart(rowLabel, 'Source');
        const targetText = normalizeAttentionValuePart(colLabel, 'Target');
        const scoreText = Number.isFinite(valueNum)
            ? valueNum.toFixed(ATTENTION_SCORE_DECIMALS)
            : ATTENTION_VALUE_PLACEHOLDER;
        this._setAttentionValue({
            source: sourceText,
            target: targetText,
            score: scoreText,
            sourceTokenIndex,
            targetTokenIndex,
            attentionScoreLink: this._buildAttentionScoreValueLink({
                mode: this.attentionMode,
                row,
                col,
                sourceTokenIndex,
                targetTokenIndex
            }),
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
        const eventTarget = event.target instanceof Element ? event.target : null;
        const hit = document.elementFromPoint(event.clientX, event.clientY);
        const resolveClosest = (selector) => {
            const fromTarget = eventTarget && typeof eventTarget.closest === 'function'
                ? eventTarget.closest(selector)
                : null;
            if (fromTarget) return fromTarget;
            return hit && typeof hit.closest === 'function'
                ? hit.closest(selector)
                : null;
        };
        const legendBarHit = resolveClosest('.vector-legend-bar, .attention-legend-bar');
        if (isTouchLikePointerEvent(event) && !legendBarHit) {
            this._hideLegendHover();
        }
        const tokenNavChip = resolveClosest('.detail-token-nav-chip[data-token-nav="true"]');
        const attentionScoreLink = resolveClosest('.detail-attention-score-link[data-attention-score-link="true"]');
        const hitPanelTokenNavChip = !!(
            tokenNavChip
            && this.panel
            && this.panel.contains(tokenNavChip)
        );
        const hitPanelAttentionScoreLink = !!(
            attentionScoreLink
            && this.panel
            && this.panel.contains(attentionScoreLink)
        );
        const attentionMatrixRoot = resolveClosest('#detailAttentionMatrix');
        const insideAttentionMatrix = !!(
            this.attentionMatrix
            && attentionMatrixRoot === this.attentionMatrix
        );
        const matrixCell = insideAttentionMatrix
            ? resolveClosest('.attention-cell')
            : null;
        const validMatrixCell = !!(
            matrixCell
            && !matrixCell.classList.contains('is-empty')
            && !matrixCell.classList.contains('is-hidden')
        );
        const shouldClearPinnedAttention = this.isOpen
            && this._attentionPinned
            && !hitPanelTokenNavChip
            && !hitPanelAttentionScoreLink
            && (!insideAttentionMatrix || !validMatrixCell);
        if (shouldClearPinnedAttention) {
            this._clearPinnedAttention({ clearSelectionSummary: true });
        }
        const panelHit = (hit && this.panel && this.panel.contains(hit))
            ? hit
            : (eventTarget && this.panel && this.panel.contains(eventTarget) ? eventTarget : null);
        if (this.isOpen && panelHit) {
            if (this.engine && typeof this.engine.resetInteractionState === 'function') {
                this.engine.resetInteractionState();
            }
        }
        if (!this.isOpen || !this.closeBtn) return;
        if (event.target === this.closeBtn) return;
        const closeHit = resolveClosest('#detailClose');
        if (!closeHit || closeHit !== this.closeBtn) return;
        // Close even if the canvas captured the pointer event.
        event.preventDefault();
        event.stopPropagation();
        this.close({ clearHistory: false });
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
        return !!(this.isReady && this.isOpen && this.currentPreview && !this._geluDetailOpen);
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

    _scrollPanelToTop() {
        if (!this.panel) return;
        if (typeof this.panel.scrollTo === 'function') {
            this.panel.scrollTo({ top: 0, left: 0, behavior: 'auto' });
            return;
        }
        this.panel.scrollTop = 0;
        this.panel.scrollLeft = 0;
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
            if (typeof requestAnimationFrame === 'function') {
                requestAnimationFrame(() => {
                    if (!this.isOpen) return;
                    this._syncHoverLabelSuppressionFromHoverState();
                });
            }
        }
        this._syncHoverLabelSuppressionFromHoverState();
        this._startLoop();
        this._scheduleSelectionEquationFit();
        this._scheduleDimensionLabelFit();
    }

    close({ clearHistory = true } = {}) {
        if (!this.isReady) return;
        this.isOpen = false;
        this._stopLoop();
        this._closeGeluDetailPreview({ restoreSelection: false, restartLoop: false });
        this._currentSelectionDescription = '';
        this._currentSelectionEquations = '';
        this._lastSelection = null;
        this._lastSelectionLabel = '';
        if (clearHistory) {
            this._resetHistoryNavigation();
        } else {
            this._updateHistoryNavigationControls();
        }
        this._setPanelTokenHoverEntry(null, { emit: true });
        this._mirroredTokenHoverEntry = null;
        this._applyTokenChipHoverState();
        this._hideLegendHover();
        this._resetCopyContextFeedback();
        this.panel.classList.remove('is-open');
        this.panel.classList.remove('is-preview-hidden');
        this.panel.classList.remove('is-gelu-view-open');
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
        this._resetHistoryNavigation();
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
            tokenIndex: Number.isFinite(tokenIndex) ? Math.floor(tokenIndex) : null,
            tokenId: Number.isFinite(tokenId) ? Math.floor(tokenId) : null,
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

        if (
            isAttentionScoreSelection(label, selection)
            || isWeightedSumSelection(label, selection)
            || isQkvHeadVectorSelection(label, selection)
        ) {
            hideRows();
            return null;
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

    _updatePromptContextRow(tokenMetadata = null, { visible = false } = {}) {
        if (!this.promptContextRow || !this.promptContextTokens) return;
        if (!visible) {
            this.promptContextRow.style.display = 'none';
            this.promptContextTokens.replaceChildren();
            return;
        }

        const tokenIndex = Number.isFinite(tokenMetadata?.tokenIndex) ? Math.floor(tokenMetadata.tokenIndex) : null;
        const tokenId = Number.isFinite(tokenMetadata?.tokenId) ? Math.floor(tokenMetadata.tokenId) : null;
        const tokenText = typeof tokenMetadata?.tokenText === 'string'
            ? tokenMetadata.tokenText
            : '';
        const promptTokenIndices = Array.isArray(this.attentionTokenIndices) && this.attentionTokenIndices.length
            ? this.attentionTokenIndices
            : this.laneTokenIndices;
        const promptTokenLabels = Array.isArray(this.attentionTokenLabels) && this.attentionTokenLabels.length
            ? this.attentionTokenLabels
            : this.tokenLabels;
        const { entries, activeIndex } = buildSelectionPromptContext({
            activationSource: this.activationSource,
            laneTokenIndices: promptTokenIndices,
            tokenLabels: promptTokenLabels,
            selectedTokenIndex: tokenIndex,
            selectedTokenId: tokenId,
            selectedTokenText: tokenText
        });

        if (!entries.length || activeIndex < 0) {
            this.promptContextRow.style.display = 'none';
            this.promptContextTokens.replaceChildren();
            return;
        }

        const fragment = document.createDocumentFragment();
        const mutedBorder = 'rgba(105, 114, 126, 0.6)';
        const mutedFill = 'rgba(73, 80, 90, 0.1)';
        const mutedFillHover = 'rgba(108, 117, 131, 0.18)';

        entries.forEach((entry, index) => {
            const chip = document.createElement('span');
            const isSelected = index === activeIndex;
            chip.className = 'detail-subtitle-token-chip detail-token-nav-chip detail-prompt-context-token';
            if (isSelected) {
                chip.classList.add('detail-prompt-context-token--selected');
                chip.style.setProperty('--token-color-border', getLogitTokenColorCss(entry.seed, 0.92));
                chip.style.setProperty('--token-color-fill', getLogitTokenColorCss(entry.seed, 0.2));
                chip.style.setProperty('--token-color-fill-hover', getLogitTokenColorCss(entry.seed, 0.28));
            } else {
                chip.style.setProperty('--token-color-border', mutedBorder);
                chip.style.setProperty('--token-color-fill', mutedFill);
                chip.style.setProperty('--token-color-fill-hover', mutedFillHover);
            }
            chip.textContent = entry.displayText;
            chip.title = entry.titleText || '';
            this._configureTokenNavChip(chip, {
                tokenText: entry.tokenLabel,
                tokenIndex: entry.tokenIndex,
                tokenId: entry.tokenId
            });
            fragment.appendChild(chip);
        });

        this.promptContextTokens.replaceChildren(fragment);
        this.promptContextRow.style.display = '';
        this._applyTokenChipHoverState();
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

    showSelection(selection, options = {}) {
        if (!this.isReady || !selection || !selection.label) return;
        const fromHistory = options?.fromHistory === true;
        const scrollPanelToTop = options?.scrollPanelToTop === true;

        this._closeGeluDetailPreview({ restoreSelection: false, restartLoop: false });
        this._resetCopyContextFeedback();
        const label = normalizeSelectionLabel(selection.label, selection);
        this._lastSelection = selection;
        this._lastSelectionLabel = label;
        const displayLabel = simplifyLayerNormParamDisplayLabel(label, selection);
        const lower = label.toLowerCase();
        const hidePreviewForSelection = false;
        const previewSelectionKey = hidePreviewForSelection ? null : buildSelectionPreviewKey(label, selection);
        const metadata = resolveMetadata(label, selection.kind, selection);
        const logitHeader = resolveLogitSelectionHeader(label, selection);
        const vectorSubtitleMetadata = this._resolveVectorTokenPosition(selection, label);
        const activationStage = String(getActivationDataFromSelection(selection)?.stage || '').toLowerCase();
        const isAttentionScore = isAttentionScoreSelection(label, selection);
        const attentionContextForSubtitle = isAttentionScore ? this._resolveAttentionContext(selection) : null;
        const attentionScoreSummary = isAttentionScore
            ? resolveAttentionScoreSelectionSummary(selection, attentionContextForSubtitle)
            : null;
        const attentionSubtitleLine = attentionScoreSummary?.tokenContextLine || '';
        let subtitleSecondaryOverride = null;
        const isTokenChipSelection = lower.startsWith('token:');
        const isTokenOrPositionChipSelection = isTokenChipSelection || lower.startsWith('position:');
        const isChosenTokenSelection = lower.startsWith('chosen token:');
        const isPositionEmbeddingSelection = lower.startsWith('position:')
            || lower.includes('position embedding')
            || lower.includes('positional embedding')
            || activationStage.startsWith('embedding.position');
        this.panel.classList.toggle('is-preview-hidden', hidePreviewForSelection);
        if (logitHeader) {
            this._setTokenChipTitleContext({
                tokenText: logitHeader.tokenText || ATTENTION_VALUE_PLACEHOLDER,
                tokenId: Number.isFinite(logitHeader.tokenId) ? Math.floor(logitHeader.tokenId) : null,
                prefixText: 'Logit token:',
                allowNavigation: false
            });
        } else if (isTokenChipSelection) {
            const tokenText = (typeof vectorSubtitleMetadata?.tokenDisplayText === 'string')
                ? vectorSubtitleMetadata.tokenDisplayText.trim()
                : formatTokenLabelForPreview(extractTokenText(label));
            this._setTokenChipTitleContext({
                tokenText: tokenText || ATTENTION_VALUE_PLACEHOLDER,
                tokenIndex: Number.isFinite(vectorSubtitleMetadata?.tokenIndex)
                    ? Math.floor(vectorSubtitleMetadata.tokenIndex)
                    : null,
                tokenId: Number.isFinite(vectorSubtitleMetadata?.tokenId)
                    ? Math.floor(vectorSubtitleMetadata.tokenId)
                    : null
            });
        } else {
            this._setTitleText(displayLabel);
        }
        if (this.subtitle) {
            if (logitHeader) {
                this.subtitle.classList.remove('detail-subtitle--qkv-token-context');
                this.subtitle.textContent = logitHeader.subtitle;
            } else if (isTokenChipSelection) {
                const subtitleParts = [];
                const tokenId = Number.isFinite(vectorSubtitleMetadata?.tokenId)
                    ? Math.floor(vectorSubtitleMetadata.tokenId)
                    : null;
                const positionText = (typeof vectorSubtitleMetadata?.positionText === 'string')
                    ? vectorSubtitleMetadata.positionText.trim()
                    : '';
                this.subtitle.classList.remove('detail-subtitle--qkv-token-context');
                if (Number.isFinite(tokenId)) {
                    subtitleParts.push(`ID ${tokenId}`);
                }
                if (positionText) {
                    subtitleParts.push(`Position ${positionText}`);
                }
                this.subtitle.textContent = subtitleParts.join(' • ');
            } else {
                const layerIndex = findUserDataNumber(selection, 'layerIndex');
                const headIndex = findUserDataNumber(selection, 'headIndex');
                const isVectorSelection = isLikelyVectorSelection(label, selection);
                const isAttentionWeightedSumSelection = lower.includes('attention weighted sum');
                const isQkvOrCachedVectorSelection = isVectorSelection
                    && (isQkvHeadVectorSelection(label, selection) || isKvCacheVectorSelection(selection));
                const showHead = isQkvMatrixLabel(label)
                    || isAttentionScore
                    || isAttentionWeightedSumSelection
                    || isQkvOrCachedVectorSelection;
                const subtitleParts = [];
                let qkvTokenContext = null;
                if (Number.isFinite(layerIndex)) {
                    subtitleParts.push(`Layer ${layerIndex + 1}`);
                }
                if (showHead && Number.isFinite(headIndex)) {
                    subtitleParts.push(`Head ${headIndex + 1}`);
                }
                if (isVectorSelection) {
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
                    const tokenText = (typeof vectorSubtitleMetadata?.tokenDisplayText === 'string')
                        ? vectorSubtitleMetadata.tokenDisplayText.trim()
                        : '';
                    if (positionText && tokenText && !isAttentionWeightedSumSelection && !isAttentionScore) {
                        qkvTokenContext = {
                            tokenText,
                            positionText,
                            tokenIndex: Number.isFinite(vectorSubtitleMetadata?.tokenIndex)
                                ? Math.floor(vectorSubtitleMetadata.tokenIndex)
                                : null,
                            tokenId: Number.isFinite(vectorSubtitleMetadata?.tokenId)
                                ? Math.floor(vectorSubtitleMetadata.tokenId)
                                : null
                        };
                    } else if (positionText && !isAttentionScore) {
                        subtitleParts.push(`Position ${positionText}`);
                    }
                }
                if (qkvTokenContext) {
                    if (isQkvOrCachedVectorSelection) {
                        this._setSubtitlePrimaryQkvTokenContext({
                            prefixParts: [],
                            tokenText: qkvTokenContext.tokenText,
                            positionText: qkvTokenContext.positionText,
                            tokenIndex: qkvTokenContext.tokenIndex,
                            tokenId: qkvTokenContext.tokenId,
                            tokenFirst: true
                        });
                        subtitleSecondaryOverride = subtitleParts.join(' • ');
                    } else {
                        this._setSubtitlePrimaryQkvTokenContext({
                            prefixParts: subtitleParts,
                            tokenText: qkvTokenContext.tokenText,
                            positionText: qkvTokenContext.positionText,
                            tokenIndex: qkvTokenContext.tokenIndex,
                            tokenId: qkvTokenContext.tokenId,
                            tokenFirst: true
                        });
                    }
                } else {
                    this.subtitle.classList.remove('detail-subtitle--qkv-token-context');
                    this.subtitle.textContent = subtitleParts.join(' • ');
                }
            }
        }
        if (this.subtitleSecondary) {
            if (subtitleSecondaryOverride !== null) {
                this._setSubtitleSecondaryText(subtitleSecondaryOverride);
            } else if (logitHeader) {
                this._setSubtitleSecondaryText('');
            } else if (isPositionEmbeddingSelection) {
                this._setSubtitleSecondaryTokenContext({
                    tokenText: normalizeAttentionValuePart(
                        vectorSubtitleMetadata?.tokenDisplayText || vectorSubtitleMetadata?.tokenText,
                        ''
                    ),
                    tokenIndex: Number.isFinite(vectorSubtitleMetadata?.tokenIndex)
                        ? Math.floor(vectorSubtitleMetadata.tokenIndex)
                        : null,
                    tokenId: Number.isFinite(vectorSubtitleMetadata?.tokenId)
                        ? Math.floor(vectorSubtitleMetadata.tokenId)
                        : null,
                    prefixText: 'Token:'
                });
            } else if (attentionScoreSummary?.tokenContext) {
                this._setSubtitleSecondaryAttentionContext(attentionScoreSummary.tokenContext);
            } else {
                this._setSubtitleSecondaryText(attentionSubtitleLine);
            }
        }
        const hideLayerNormFields = isLayerNormSolidSelection(label);
        const isLogitTokenSelection = !!logitHeader;
        const hideTensorDimsField = hideLayerNormFields
            || isAttentionScore
            || isLogitTokenSelection
            || isTokenOrPositionChipSelection
            || isChosenTokenSelection;
        const isVectorMetadata = isLikelyVectorSelection(label, selection) || isTokenOrPositionChipSelection;
        const dimsRow = this.inputDim?.closest('.detail-row')
            || this.outputDim?.closest('.detail-row')
            || null;
        if (this.inputDimLabel) {
            this.inputDimLabel.textContent = isTokenOrPositionChipSelection
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
        this._updatePromptContextRow(vectorTokenMetadata, { visible: isTokenChipSelection });
        if (this.description) {
            const desc = resolveDescription(label, selection.kind, selection);
            this._currentSelectionDescription = desc || '';
            setDescriptionContent(this.description, desc || '');
            setDescriptionGeluAction(this.description, isMlpMatrixSelectionLabel(label));
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
        if (shouldReusePreview && this.currentPreview) {
            this._refreshReusedPreview();
        }

        this._updateAttentionPreview(selection);
        this._applyTokenChipHoverState();
        this.open();
        if (scrollPanelToTop) {
            this._scrollPanelToTop();
        }
        this._scheduleSelectionEquationFit();
        this._scheduleDimensionLabelFit();
        if (!fromHistory) {
            const entry = this._buildHistoryEntry('selection', selection);
            this._pushHistoryEntry(entry);
        } else {
            this._updateHistoryNavigationControls();
        }
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
