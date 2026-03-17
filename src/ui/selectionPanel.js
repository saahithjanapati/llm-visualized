import * as THREE from 'three';
import { FontLoader } from 'three/examples/jsm/loaders/FontLoader.js';
import { WeightMatrixVisualization } from '../components/WeightMatrixVisualization.js';
import { VectorVisualizationInstancedPrism } from '../components/VectorVisualizationInstancedPrism.js';
import { LayerNormalizationVisualization } from '../components/LayerNormalizationVisualization.js';
import { appState } from '../state/appState.js';
import { consoleInfo } from '../utils/runtimeConsole.js';
import { createSciFiMaterial, updateSciFiMaterialColor } from '../utils/sciFiMaterial.js';
import { buildHueRangeOptions, mapAttentionPostScoreToColor, mapValueToColor } from '../utils/colors.js';
import { resolveLogitEntryText } from '../utils/logitTokenText.js';
import {
    formatLayerNormLabel,
    formatLayerNormParamLabel,
    normalizeLayerNormProductStage,
    isPostLayerNormResidualSelection,
    normalizePostLayerNormResidualStage,
    resolvePostLayerNormResidualLabel,
    resolveLayerNormParamSpec
} from '../utils/layerNormLabels.js';
import { initTouchClickFallback } from './touchClickFallback.js';
import { fitSelectionDimensionLabels } from './selectionPanelDimensionFitUtils.js';
import { shouldCaptureMhsaKeyboardInput } from './selectionPanelKeyboardUtils.js';
import {
    clampDesktopSelectionPanelWidth,
    resolveCopyContextButtonLayout,
    resolveDesktopSelectionPanelWidthBounds
} from './selectionPanelLayoutUtils.js';
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
    isQkvMatrixLabel,
    isResidualVectorSelection,
    isSelfAttentionSelection,
    isValueSelection,
    isWeightMatrixLabel,
    isWeightedSumSelection,
    matchesAttentionScoreSelection,
    normalizeSelectionLabel,
    resolveAttentionModeFromSelection,
    resolveSelectionLogitEntry as resolveSelectionLogitEntryMetadata,
    simplifyLayerNormParamDisplayLabel
} from './selectionPanelSelectionUtils.js';
import {
    resolveDescription,
    resolveSelectionEquations,
    resolveSelectionPreviewEquations
} from './selectionPanelNarrativeUtils.js';
import {
    GELU_PANEL_ACTION_OPEN,
    createGeluDetailView,
    isGeluDetailSelection,
    isMlpMatrixSelectionLabel,
    setDescriptionGeluAction
} from './selectionPanelGeluPreview.js';
import {
    SOFTMAX_PANEL_ACTION_OPEN,
    createSoftmaxDetailView,
    getSoftmaxCopyContextContent,
    isPostSoftmaxAttentionSelection,
    resolveSoftmaxPreviewContext,
    setDescriptionSoftmaxAction
} from './selectionPanelSoftmaxPreview.js';
import { TRANSFORMER_VIEW2D_PANEL_ACTION_OPEN } from './selectionPanelTransformerView2dConstants.js';
import {
    describeTransformerView2dTarget,
    resolveTransformerView2dActionContext
} from '../view2d/transformerView2dTargets.js';
import {
    syncTransformerView2dRoute
} from '../view2d/transformerView2dRoute.js';
import {
    normalizeTransformerView2dDetailInteractionTargets,
    resolveTransformerView2dDetailInteractionTargets
} from '../view2d/transformerView2dDetailInteractionTargets.js';
import { renderSelectionPreviewEquations } from './selectionPanelEquationPreviewUtils.js';
import {
    computeAttentionCellSize,
    countVisibleAttentionCellsInRow,
    getAttentionRevealOrder,
    shouldRevealAttentionCell
} from './selectionPanelAttentionRevealUtils.js';
import {
    CAUSAL_MASK_PREVIEW_BLOCKED_VALUE,
    buildAttentionMatrixValues,
    isCausalUpperAttentionCell,
    resolveAttentionMatrixCellValue,
    shouldClearPinnedAttentionOnDocumentPointerDown,
    shouldMuteCausalUpperPreAttentionCell
} from './selectionPanelAttentionMatrixUtils.js';
import {
    readEquationBaseFontPx,
    readEquationContentSize
} from './equationFitUtils.js';
import {
    collectVisibleContextText,
    copyTextToClipboard,
    getDescriptionPlainText,
    setDescriptionContent
} from './selectionPanelCopyUtils.js';
import { createSelectionPanelHistoryNavigation } from './selectionPanelHistoryNavigationUtils.js';
import {
    formatAttentionSubtitleTokenPart,
    formatTokenLabelForPreview,
    normalizeAttentionValuePart
} from './selectionPanelFormatUtils.js';
import {
    isTransformerView2dSelectionSurface,
    normalizeSelectionPanelSurface,
    resolveSelectionPanelSurface,
    SELECTION_PANEL_SURFACES
} from './selectionPanelNavigationSurfaceUtils.js';
import {
    renderSelectionVectorSamplingData,
    resolveSelectionVectorSamplingData
} from './selectionPanelVectorSamplingUtils.js';
import {
    resolveLiveLayerNormNormalizedPreviewSelection,
    resolveLayerNormParameterSummary,
    resolveLayerNormProductStageSummary
} from './selectionPanelLayerNormPreviewUtils.js';
import { resolveSelectionPrimaryActionConfig } from './selectionPanelPrimaryActionUtils.js';
import { buildSelectionPromptContext } from './selectionPanelPromptContextUtils.js';
import { buildSelectionChatPrompt } from './selectionPanelChatPromptUtils.js';
import {
    formatKvCachePhaseLabel,
    isKvCacheInfoSelection,
    normalizeKvCachePhase
} from './kvCacheInfoUtils.js';
import {
    buildMhsaInfoSelection,
    formatMhsaInfoTitle,
    isMhsaInfoSelection
} from './mhsaInfoUtils.js';
import { getSideCopyEntry } from '../animations/mhsa/laneIndex.js';
import {
    buildAttentionHoverInfo,
    resolveAttentionHoverLabel
} from './attentionHoverInfo.js';
import {
    MHSA_INFO_PANEL_ACTION_OPEN,
    setDescriptionMhsaInfoAction
} from './selectionPanelMhsaAction.js';
import {
    buildMhsaTokenMatrixPreviewData
} from './selectionPanelMhsaTokenMatrixUtils.js';
import {
    applyMhsaTokenMatrixLayoutVars,
    clearMhsaTokenMatrixLayoutVars,
    resolveMhsaTokenMatrixLayoutMetrics
} from './selectionPanelMhsaLayoutUtils.js';
import {
    buildMhsaProjectionComponentStateKey,
    resolveMhsaKeyRowSceneFocus,
    resolveMhsaProjectionInputRowSceneFocus,
    resolveMhsaProjectionWeightSceneFocus,
    resolveMhsaValueInputRowSceneFocus,
    resolveMhsaValueRowSceneFocus,
    resolveMhsaTokenMatrixProjectionStageTarget,
    shouldMirrorMhsaHeadOutputRowFocus
} from './selectionPanelMhsaInteractionUtils.js';
import {
    resolveMhsaTokenMatrixFixedLabelScale
} from './selectionPanelMhsaViewportUtils.js';
import {
    createMhsaTokenMatrixCellStore,
    forEachMhsaTokenMatrixCell,
    registerMhsaTokenMatrixCell
} from './selectionPanelMhsaCellStoreUtils.js';
import {
    resolveMhsaTokenMatrixHoverTokenEntries
} from './selectionPanelMhsaTokenHoverUtils.js';
import {
    buildMhsaSceneModel
} from '../view2d/model/buildMhsaSceneModel.js';
import { createHoverLabelOverlay } from './hoverLabelOverlay.js';
import {
    buildSceneLayout
} from '../view2d/layout/buildSceneLayout.js';
import {
    CanvasSceneRenderer
} from '../view2d/render/canvas/CanvasSceneRenderer.js';
import {
    resolveMhsaTokenMatrixCanvasMode,
    shouldRenderMhsaTokenMatrixCanvas,
    VIEW2D_MHSA_CANVAS_MODES
} from '../view2d/runtime/view2dFeatureFlags.js';
import {
    applyMaterialSnapshot,
    copyInstancedVectorSliceToPreview,
    copyInstancedVectorColorsToPreview,
    extractMaterialSnapshot,
    isInstancedVectorSliceInMotion,
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
    PREVIEW_MAX_FIT_VIEWPORT_HEIGHT_PX,
    PREVIEW_MAX_FIT_VIEWPORT_WIDTH_PX,
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
    resolveRenderPixelRatio
} from '../utils/constants.js';
import { getIncompleteUtf8TokenDisplay, getIncompleteUtf8TokenNote } from '../utils/tokenEncodingNotes.js';
import { formatTokenChipDisplayText } from '../utils/tokenChipStyleUtils.js';
import { createTokenChipMesh } from '../utils/tokenChipMeshFactory.js';
import {
    TOKEN_CHIP_HOVER_SYNC_EVENT,
    dispatchTokenChipHoverSync,
    matchesFocusVisibleTarget,
    normalizeTokenChipEntry,
    normalizeTokenChipEntries,
    tokenChipEntryListsMatch,
    tokenChipEntriesMatch
} from './tokenChipHoverSync.js';
import {
    applyTokenChipColors
} from './tokenChipColorUtils.js';
import {
    getPreference,
    setPreference
} from '../utils/preferences.js';
import {
    MHA_FINAL_Q_COLOR,
    MHA_FINAL_K_COLOR,
    MHA_FINAL_V_COLOR,
    MHA_OUTPUT_PROJECTION_MATRIX_COLOR,
    MHA_OUTPUT_PROJECTION_MATRIX_PARAMS,
    MHA_VALUE_CLAMP_MAX,
    MHA_VALUE_HUE_SPREAD,
    MHA_VALUE_KEY_COLOR_COUNT,
    MHA_VALUE_LIGHTNESS_MAX,
    MHA_VALUE_LIGHTNESS_MIN,
    MHA_VALUE_RANGE_MAX,
    MHA_VALUE_RANGE_MIN,
    MHA_VALUE_SPECTRUM_COLOR,
    POSITION_EMBED_COLOR,
    LN_PARAMS
} from '../animations/LayerAnimationConstants.js';
import { getLayerNormParamData } from '../data/layerNormParams.js';
import './selectionPanel.css';

let transformerView2dDetailViewModulePromise = null;

function loadTransformerView2dDetailViewModule() {
    if (!transformerView2dDetailViewModulePromise) {
        transformerView2dDetailViewModulePromise = import('./selectionPanelTransformerView2d.js');
    }
    return transformerView2dDetailViewModulePromise;
}
import {
    getAttentionBiasVectorSample,
    getAttentionBiasHeadSample,
    getMlpBiasVectorSample
} from '../data/biasParams.js';
import {
    MLP_DOWN_BIAS_TOOLTIP_LABEL,
    MLP_UP_BIAS_TOOLTIP_LABEL
} from '../utils/mlpLabels.js';

const ATTENTION_VECTOR_PANEL_ACTION_OPEN = 'open-attention-vector';
const LAYERNORM_PANEL_ACTION_OPEN = 'open-layernorm';
const LAYERNORM_PARAM_PANEL_ACTION_OPEN = 'open-layernorm-param';
const QKV_SOURCE_VECTOR_PANEL_ACTION_OPEN = 'open-qkv-source-vector';
const QKV_WEIGHT_MATRIX_PANEL_ACTION_OPEN = 'open-qkv-weight-matrix';
const ATTENTION_SECTION_COLLAPSED_PREF_KEY = 'selectionPanelAttentionSectionCollapsed';
const TRANSFORMER_VIEW2D_PAUSE_REASON = 'detail-transformer-view2d';
const ATTENTION_HEAD_COUNT = Math.max(1, Math.round(D_MODEL / D_HEAD));
const ATTENTION_SCORE_PREVIEW_DECIMALS = 3;
const SELECTION_PANEL_DEV_MODE_EVENT = 'selectionPanelDevModeChanged';
const TOKEN_CHIP_WARNING_STYLE = Object.freeze({
    ...TOKEN_CHIP_STYLE,
    minWidth: 420,
    height: 132,
    textSize: 34,
    textColor: 0xb45309,
    secondaryTextColor: 0x92400e
});
let tokenChipFont = null;
let tokenChipFontPromise = null;

function getAttentionSectionCollapsedPreference() {
    try {
        const storage = globalThis?.localStorage;
        if (!storage || typeof storage.getItem !== 'function') return false;
        return getPreference(ATTENTION_SECTION_COLLAPSED_PREF_KEY, false) === true;
    } catch (_error) {
        return false;
    }
}

function setAttentionSectionCollapsedPreference(collapsed) {
    try {
        const storage = globalThis?.localStorage;
        if (!storage || typeof storage.setItem !== 'function') return;
        setPreference(ATTENTION_SECTION_COLLAPSED_PREF_KEY, collapsed === true);
    } catch (_error) {
        // Ignore persistence failures and keep the in-memory toggle working.
    }
}

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

const DEFAULT_METADATA_DIMENSION_LABELS = Object.freeze({
    input: 'Input dimension',
    output: 'Output dimension',
    inputSummary: 'input dimension',
    outputSummary: 'output dimension'
});

const VOCAB_EMBEDDING_DIMENSION_LABELS = Object.freeze({
    input: 'Vocabulary size',
    output: 'Embedding dimension',
    inputSummary: 'vocabulary size',
    outputSummary: 'embedding dimension'
});

const POSITION_EMBEDDING_DIMENSION_LABELS = Object.freeze({
    input: 'Max positions',
    output: 'Embedding dimension',
    inputSummary: 'max positions',
    outputSummary: 'embedding dimension'
});

const UNEMBEDDING_DIMENSION_LABELS = Object.freeze({
    input: 'Embedding dimension',
    output: 'Vocabulary size',
    inputSummary: 'embedding dimension',
    outputSummary: 'vocabulary size'
});

function resolveMetadataDimensionLabels(label = '') {
    const lower = String(label || '').toLowerCase();
    if (hasTopVocabEmbeddingLabel(lower) || lower.includes('unembedding')) {
        return UNEMBEDDING_DIMENSION_LABELS;
    }
    if (hasVocabEmbeddingLabel(lower)) {
        return VOCAB_EMBEDDING_DIMENSION_LABELS;
    }
    if (hasPositionEmbeddingLabel(lower)) {
        return POSITION_EMBEDDING_DIMENSION_LABELS;
    }
    return DEFAULT_METADATA_DIMENSION_LABELS;
}

function formatDims(inputDim, outputDim, dimensionLabels = DEFAULT_METADATA_DIMENSION_LABELS) {
    if (!Number.isFinite(inputDim) || !Number.isFinite(outputDim)) return 'TBD';
    return `${dimensionLabels.inputSummary}: ${formatNumber(inputDim)} | ${dimensionLabels.outputSummary}: ${formatNumber(outputDim)}`;
}

function buildMetadata(
    params = 'TBD',
    inputDim = null,
    outputDim = null,
    length = null,
    biasDim = null,
    dimensionLabels = DEFAULT_METADATA_DIMENSION_LABELS
) {
    const hasDims = Number.isFinite(inputDim) && Number.isFinite(outputDim);
    const hasLength = Number.isFinite(length);
    const hasBiasDim = Number.isFinite(biasDim);
    const paramCount = hasDims ? formatNumber(inputDim * outputDim) : params;
    return {
        params: paramCount,
        dims: hasDims ? formatDims(inputDim, outputDim, dimensionLabels) : 'TBD',
        inputDim: hasDims ? formatNumber(inputDim) : 'TBD',
        outputDim: hasDims ? formatNumber(outputDim) : 'TBD',
        inputDimLabel: dimensionLabels.input,
        outputDimLabel: dimensionLabels.output,
        length: hasLength ? formatNumber(length) : 'TBD',
        hasLength,
        biasDim: hasBiasDim ? formatNumber(biasDim) : '',
        hasBiasDim,
        hasDims
    };
}

function createInlineDetailActionButton(label = '', action = '', payload = null) {
    if (typeof document === 'undefined') return null;
    const safeLabel = typeof label === 'string' ? label.trim() : '';
    const safeAction = typeof action === 'string' ? action.trim() : '';
    if (!safeLabel || !safeAction) return null;

    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'detail-description-action-link detail-description-inline-action-link';
    button.dataset.detailAction = safeAction;
    button.setAttribute('aria-label', safeLabel);
    button.textContent = safeLabel;

    if (payload && typeof payload === 'object') {
        const safePayload = Object.fromEntries(
            Object.entries(payload).filter(([, value]) => value !== undefined)
        );
        if (Object.keys(safePayload).length > 0) {
            button.dataset.detailPayload = encodeURIComponent(JSON.stringify(safePayload));
        }
    }

    return button;
}

function setLayerNormParameterSummaryContent(element, parameterSummary = null) {
    if (!element) return;
    if (!parameterSummary) {
        element.textContent = '';
        return;
    }

    const perParameterCount = Number.isFinite(parameterSummary?.perParameterCount)
        ? Math.max(1, Math.floor(parameterSummary.perParameterCount))
        : D_MODEL;
    const totalParameterCount = Number.isFinite(parameterSummary?.totalParameterCount)
        ? Math.max(1, Math.floor(parameterSummary.totalParameterCount))
        : (perParameterCount * 2);
    element.textContent = formatNumber(totalParameterCount);
}

function readSceneSelectionNumber(node, key) {
    const direct = node?.userData?.[key];
    if (Number.isFinite(direct)) return Math.floor(direct);
    const activationValue = node?.userData?.activationData?.[key];
    if (Number.isFinite(activationValue)) return Math.floor(activationValue);
    return null;
}

function readSceneSelectionString(node, key) {
    const direct = node?.userData?.[key];
    if (typeof direct === 'string' && direct.trim().length) return direct;
    const activationValue = node?.userData?.activationData?.[key];
    if (typeof activationValue === 'string' && activationValue.trim().length) return activationValue;
    return '';
}

function normalizeQkvActionKind(kind = 'Q') {
    const upper = String(kind || '').toUpperCase();
    if (upper === 'K') return 'K';
    if (upper === 'V') return 'V';
    return 'Q';
}

function hasExplicitQkvVectorLabelText(value = '') {
    const lower = String(value || '').toLowerCase();
    return lower.includes('query vector')
        || lower.includes('key vector')
        || lower.includes('value vector');
}

function isRelabeledQkvVectorCandidate({
    label = '',
    category = null
} = {}) {
    if (hasExplicitQkvVectorLabelText(label)) return true;
    const upperCategory = String(category || '').toUpperCase();
    return upperCategory === 'Q' || upperCategory === 'K' || upperCategory === 'V';
}

function hasPreQkvResidualLabelText(value = '') {
    const lower = String(value || '').toLowerCase();
    return lower.includes('post-layernorm residual')
        || lower.includes('post layernorm residual');
}

function isPreQkvResidualCopyCandidate({
    label = '',
    category = null
} = {}) {
    const upperCategory = String(category || '').toUpperCase();
    if (upperCategory !== 'Q' && upperCategory !== 'K' && upperCategory !== 'V') {
        return false;
    }
    return hasPreQkvResidualLabelText(label);
}

function resolveLiveLayerNormParamVectorRef(vectorRef = null) {
    const detachedCarrier = vectorRef?.userData?.lnDetachedCarrier || null;
    if (detachedCarrier?.group?.parent) {
        return detachedCarrier;
    }
    return vectorRef;
}

function resolveLayerNormParamPreviewSpec(label = '', selectionInfo = null) {
    const activationStage = String(getActivationDataFromSelection(selectionInfo)?.stage || '').toLowerCase();
    const explicitKind = findUserDataString(selectionInfo, 'layerNormKind') || null;
    const paramSpec = resolveLayerNormParamSpec({
        label,
        stage: activationStage,
        explicitKind
    });
    if (!paramSpec) return null;
    const layerIndex = findUserDataNumber(selectionInfo, 'layerIndex');
    return {
        layerNormKind: paramSpec.layerNormKind,
        param: paramSpec.param,
        layerIndex: Number.isFinite(layerIndex) ? Math.floor(layerIndex) : null
    };
}

export function resolveLayerNormParamPreviewInstanceCount(selectionInfo = null, previewData = []) {
    const vectorRefCount = selectionInfo?.info?.vectorRef?.instanceCount;
    if (Number.isFinite(vectorRefCount) && vectorRefCount > 0) {
        return Math.max(1, Math.floor(vectorRefCount));
    }
    const dataLength = Array.isArray(previewData) || ArrayBuffer.isView(previewData)
        ? previewData.length
        : 0;
    if (dataLength > 0) {
        return Math.max(1, Math.ceil(dataLength / PRISM_DIMENSIONS_PER_UNIT));
    }
    return PREVIEW_VECTOR_BODY_INSTANCES;
}

function getQkvWeightMatrixLabel(kind = 'Q') {
    const safeKind = normalizeQkvActionKind(kind);
    if (safeKind === 'K') return 'Key Weight Matrix';
    if (safeKind === 'V') return 'Value Weight Matrix';
    return 'Query Weight Matrix';
}

function getLayerNormParamLabel(layerNormKind = null, param = 'scale') {
    return formatLayerNormParamLabel(layerNormKind, param);
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
        normalizePreviewKeyValue(findUserDataString(selection, 'layerNormKind')),
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
    if (isPostLayerNormResidualSelection({
        label,
        stage: getActivationDataFromSelection(selectionInfo)?.stage || ''
    })) {
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

function colorToRgbaCss(color, alpha = 1) {
    if (!color) return 'rgba(0, 0, 0, 0)';
    const target = color.isColor ? color : new THREE.Color(color);
    const safeAlpha = Number.isFinite(alpha) ? THREE.MathUtils.clamp(alpha, 0, 1) : 1;
    const r = Math.round(target.r * 255);
    const g = Math.round(target.g * 255);
    const b = Math.round(target.b * 255);
    return `rgba(${r}, ${g}, ${b}, ${safeAlpha.toFixed(3)})`;
}

function darkenColor(color, factor = 1) {
    const target = color?.isColor ? color.clone() : new THREE.Color(color);
    const safeFactor = Number.isFinite(factor) ? THREE.MathUtils.clamp(factor, 0, 1) : 1;
    return target.multiplyScalar(safeFactor);
}

function resolveCausalMaskCellColor(isBlocked = false) {
    if (isBlocked) return new THREE.Color(0x000000);
    return darkenColor(
        mapValueToColor(0, { clampMax: ATTENTION_PRE_COLOR_CLAMP }),
        ATTENTION_PREVIEW_COLOR_DARKEN_FACTOR
    );
}

function buildCausalMaskLegendGradient() {
    const blockedCss = colorToCss(resolveCausalMaskCellColor(true));
    const allowedCss = colorToCss(resolveCausalMaskCellColor(false));
    return `linear-gradient(90deg, ${blockedCss} 0%, ${blockedCss} 50%, ${allowedCss} 50%, ${allowedCss} 100%)`;
}

function resolveAttentionPreviewCellColor(value, mode) {
    const safeMode = mode === 'mask'
        ? 'mask'
        : (mode === 'post' ? 'post' : 'pre');
    if (safeMode === 'post') {
        return mapAttentionPostScoreToColor(value);
    }
    if (safeMode === 'mask') {
        return resolveCausalMaskCellColor(value <= CAUSAL_MASK_PREVIEW_BLOCKED_VALUE);
    }
    const previewValue = value;
    const baseColor = mapValueToColor(previewValue, { clampMax: ATTENTION_PRE_COLOR_CLAMP });
    return darkenColor(baseColor, ATTENTION_PREVIEW_COLOR_DARKEN_FACTOR);
}

function isCausalMaskSelection(selectionInfo = null) {
    const activation = getActivationDataFromSelection(selectionInfo);
    const stageLower = String(activation?.stage || '').toLowerCase();
    if (stageLower === 'attention.mask') return true;
    const lower = String(selectionInfo?.label || '').toLowerCase();
    return lower.includes('causal mask') || lower.includes('attention mask');
}

function resolveAttentionSelectionDisplayMode(selectionInfo = null, fallbackMode = 'pre') {
    if (isCausalMaskSelection(selectionInfo)) return 'mask';
    const stageLower = String(getActivationDataFromSelection(selectionInfo)?.stage || '').toLowerCase();
    if (stageLower === 'attention.post') return 'post';
    if (stageLower === 'attention.pre') return 'pre';
    const preferredMode = resolveAttentionModeFromSelection(selectionInfo);
    if (preferredMode === 'post' || preferredMode === 'pre') return preferredMode;
    return fallbackMode === 'post' ? 'post' : 'pre';
}

function resolveAttentionDisplayMode(mode = 'pre') {
    if (mode === 'mask') return 'mask';
    return mode === 'post' ? 'post' : 'pre';
}

function isCausalMaskBlocked({
    maskValue = null,
    isMasked = false,
    queryTokenIndex = null,
    keyTokenIndex = null
} = {}) {
    if (maskValue === Number.NEGATIVE_INFINITY) return true;
    if (isMasked === true) return true;
    return isCausalUpperAttentionCell(queryTokenIndex, keyTokenIndex);
}

function resolveCausalMaskPreviewValue({
    maskValue = null,
    isMasked = false,
    queryTokenIndex = null,
    keyTokenIndex = null
} = {}) {
    return isCausalMaskBlocked({
        maskValue,
        isMasked,
        queryTokenIndex,
        keyTokenIndex
    })
        ? CAUSAL_MASK_PREVIEW_BLOCKED_VALUE
        : 0;
}

function formatCausalMaskPreviewScore({
    maskValue = null,
    isMasked = false,
    queryTokenIndex = null,
    keyTokenIndex = null,
    fallback = ATTENTION_VALUE_PLACEHOLDER
} = {}) {
    if (
        maskValue === 0
        || isMasked === true
        || maskValue === Number.NEGATIVE_INFINITY
        || (
            Number.isFinite(queryTokenIndex)
            && Number.isFinite(keyTokenIndex)
        )
    ) {
        return isCausalMaskBlocked({
            maskValue,
            isMasked,
            queryTokenIndex,
            keyTokenIndex
        })
            ? '-∞'
            : '0';
    }
    return fallback;
}

function formatAttentionDisplayScore({
    mode = 'pre',
    value = null,
    maskValue = null,
    isMasked = false,
    queryTokenIndex = null,
    keyTokenIndex = null,
    fallback = ATTENTION_VALUE_PLACEHOLDER
} = {}) {
    if (resolveAttentionDisplayMode(mode) === 'mask') {
        return formatCausalMaskPreviewScore({
            maskValue,
            isMasked,
            queryTokenIndex,
            keyTokenIndex,
            fallback
        });
    }
    return formatAttentionPreviewScore(value, fallback);
}

function buildAttentionMatrixCellTitle({
    rowLabel = '',
    colLabel = '',
    mode = 'pre',
    value = null,
    queryTokenIndex = null,
    keyTokenIndex = null,
    maskValue = null,
    isMasked = false
} = {}) {
    const safeMode = resolveAttentionDisplayMode(mode);
    const modeLabel = safeMode === 'mask'
        ? 'causal mask'
        : (safeMode === 'post' ? 'post' : 'pre');
    const displayValue = formatAttentionDisplayScore({
        mode: safeMode,
        value,
        maskValue,
        isMasked,
        queryTokenIndex,
        keyTokenIndex,
        fallback: 'n/a'
    });
    return `${rowLabel} → ${colLabel} (${modeLabel}): ${displayValue}`;
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
    const isScoreStage = stageLower === 'attention.pre'
        || stageLower === 'attention.post'
        || stageLower === 'attention.mask';
    const isAttentionScore = isAttentionScoreSelection(normalizedLabel, selectionInfo)
        || isScoreStage;
    if (!isAttentionScore) return null;

    const mode = resolveAttentionSelectionDisplayMode(
        selectionInfo,
        stageLower.includes('post') ? 'post' : 'pre'
    );
    const score = mode === 'mask'
        ? resolveCausalMaskPreviewValue({
            maskValue: activation.maskValue,
            isMasked: activation.isMasked === true,
            queryTokenIndex: activation.tokenIndex,
            keyTokenIndex: activation.keyTokenIndex
        })
        : resolveAttentionMatrixCellValue({
            mode,
            value: mode === 'post' ? activation.postScore : activation.preScore,
            queryTokenIndex: activation.tokenIndex,
            keyTokenIndex: activation.keyTokenIndex
        });
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
    const scoreText = formatAttentionDisplayScore({
        mode,
        value: score,
        maskValue: activation.maskValue,
        isMasked: activation.isMasked === true,
        queryTokenIndex: activation.tokenIndex,
        keyTokenIndex: activation.keyTokenIndex,
        fallback: 'n/a'
    });
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

function clamp01(value) {
    if (!Number.isFinite(value)) return 0;
    return Math.min(1, Math.max(0, value));
}

function formatAttentionPreviewScore(value, fallback = ATTENTION_VALUE_PLACEHOLDER) {
    if (!Number.isFinite(value)) return fallback;
    return value.toFixed(ATTENTION_SCORE_PREVIEW_DECIMALS);
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

function hasPositionEmbeddingLabel(lower) {
    return lower.includes('position embedding') || lower.includes('positional embedding');
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
    if (lower === MLP_UP_BIAS_TOOLTIP_LABEL.toLowerCase()) return FINAL_MLP_COLOR;
    if (lower === MLP_DOWN_BIAS_TOOLTIP_LABEL.toLowerCase()) return FINAL_MLP_COLOR;
    if (hasVocabEmbeddingLabel(lower) || lower.includes('unembedding')) {
        return hasTopVocabEmbeddingLabel(lower) || lower.includes('unembedding')
            ? FINAL_VOCAB_TOP_COLOR
            : MHA_FINAL_Q_COLOR;
    }
    if (hasPositionEmbeddingLabel(lower)) return POSITION_EMBED_COLOR;
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

function isPlainPreviewUserDataObject(value) {
    if (!value || Object.prototype.toString.call(value) !== '[object Object]') return false;
    const proto = Object.getPrototypeOf(value);
    return proto === Object.prototype || proto === null;
}

function sanitizePreviewUserDataValue(value, seen = new WeakSet()) {
    if (value === null) return null;
    const type = typeof value;
    if (type === 'string' || type === 'number' || type === 'boolean') return value;
    if (Array.isArray(value)) {
        if (seen.has(value)) return undefined;
        seen.add(value);
        const sanitized = value
            .map((entry) => sanitizePreviewUserDataValue(entry, seen))
            .filter((entry) => entry !== undefined);
        seen.delete(value);
        return sanitized;
    }
    if (type !== 'object') return undefined;
    if (
        value.isObject3D
        || value.isMaterial
        || value.isBufferGeometry
        || value.isTexture
        || value.isColor
        || value.isVector2
        || value.isVector3
        || value.isVector4
        || value.isMatrix3
        || value.isMatrix4
        || value.isQuaternion
    ) {
        return undefined;
    }
    if (!isPlainPreviewUserDataObject(value)) return undefined;
    if (seen.has(value)) return undefined;
    seen.add(value);
    const sanitized = {};
    Object.entries(value).forEach(([key, entry]) => {
        const nextValue = sanitizePreviewUserDataValue(entry, seen);
        if (nextValue !== undefined) {
            sanitized[key] = nextValue;
        }
    });
    seen.delete(value);
    return sanitized;
}

function cloneObjectForPreview(root) {
    if (!root || root.isScene) return null;
    const originals = [];
    if (typeof root.traverse === 'function') {
        root.traverse((child) => {
            originals.push({
                child,
                userData: child.userData
            });
            child.userData = sanitizePreviewUserDataValue(child.userData) || {};
        });
    } else {
        originals.push({
            child: root,
            userData: root.userData
        });
        root.userData = sanitizePreviewUserDataValue(root.userData) || {};
    }
    try {
        return root.clone(true);
    } finally {
        originals.forEach(({ child, userData }) => {
            child.userData = userData;
        });
    }
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
    const clone = cloneObjectForPreview(root);
    if (!clone) return null;
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
    const clone = cloneObjectForPreview(root);
    if (!clone) return null;
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
    const clone = cloneObjectForPreview(source);
    if (!clone) return null;
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

function resolveExactVectorSliceSource(selectionInfo, label = '') {
    const vectorRef = selectionInfo?.info?.vectorRef || null;
    if (vectorRef?.isBatchedVectorRef && vectorRef._batch?.mesh?.isInstancedMesh) {
        const prismCount = Number.isFinite(vectorRef._batch.prismCount)
            ? Math.max(1, Math.floor(vectorRef._batch.prismCount))
            : Math.max(1, Math.floor(resolveVectorPreviewInstanceCount(selectionInfo, label) || PREVIEW_VECTOR_BODY_INSTANCES));
        const vectorIndex = Number.isFinite(vectorRef._index) ? Math.max(0, Math.floor(vectorRef._index)) : 0;
        return {
            sourceMesh: vectorRef._batch.mesh,
            sourceOffset: vectorIndex * prismCount,
            instanceCount: prismCount
        };
    }

    if (vectorRef?.mesh?.isInstancedMesh) {
        const instanceCount = Number.isFinite(vectorRef.instanceCount)
            ? Math.max(1, Math.floor(vectorRef.instanceCount))
            : Math.max(1, Math.floor(resolveVectorPreviewInstanceCount(selectionInfo, label) || PREVIEW_VECTOR_BODY_INSTANCES));
        const meshCount = Number.isFinite(vectorRef.mesh.count)
            ? Math.max(1, Math.floor(vectorRef.mesh.count))
            : Math.max(1, Math.floor(vectorRef.mesh.instanceMatrix?.count || instanceCount));
        const hintedVectorIndex = Number(selectionInfo?.info?.vectorIndex);
        const rawInstanceId = Number(selectionInfo?.hit?.instanceId);
        const sourceOffset = Number.isFinite(hintedVectorIndex) && hintedVectorIndex >= 0
            ? Math.max(0, Math.floor(hintedVectorIndex) * instanceCount)
            : (
                Number.isFinite(rawInstanceId) && rawInstanceId >= 0 && meshCount > instanceCount
                    ? Math.max(0, Math.floor(rawInstanceId / instanceCount) * instanceCount)
                    : 0
            );
        return {
            sourceMesh: vectorRef.mesh,
            sourceOffset,
            instanceCount
        };
    }

    const vectorMesh = findVectorSourceMesh(selectionInfo);
    if (!vectorMesh?.isInstancedMesh) return null;

    const instanceCount = Math.max(1, Math.floor(resolveVectorPreviewInstanceCount(selectionInfo, label) || PREVIEW_VECTOR_BODY_INSTANCES));
    const rawInstanceId = Number(selectionInfo?.hit?.instanceId);
    const sourceOffset = Number.isFinite(rawInstanceId) && rawInstanceId >= 0
        ? Math.max(0, Math.floor(rawInstanceId / instanceCount) * instanceCount)
        : 0;
    return {
        sourceMesh: vectorMesh,
        sourceOffset,
        instanceCount
    };
}

function buildExactVectorMeshSlicePreview(selectionInfo, label = '') {
    const sliceSource = resolveExactVectorSliceSource(selectionInfo, label);
    if (!sliceSource?.sourceMesh?.isInstancedMesh || !(sliceSource.instanceCount > 0)) {
        return null;
    }
    if (isInstancedVectorSliceInMotion(
        sliceSource.sourceMesh,
        sliceSource.sourceOffset,
        sliceSource.instanceCount
    )) {
        return null;
    }

    const clone = cloneObjectForPreview(sliceSource.sourceMesh);
    if (!clone?.isInstancedMesh) {
        return null;
    }
    clone.matrixAutoUpdate = true;
    clone.count = Math.max(1, Math.floor(sliceSource.instanceCount));
    clone.frustumCulled = false;
    clone.userData = sanitizePreviewUserDataValue(sliceSource.sourceMesh.userData) || {};

    sliceSource.sourceMesh.updateWorldMatrix(true, true);
    sliceSource.sourceMesh.matrixWorld.decompose(clone.position, clone.quaternion, clone.scale);

    const group = new THREE.Group();
    group.add(clone);

    const previewGeometries = cloneGeometriesForPreview(group);
    const previewMaterials = cloneMaterialsForPreview(group);
    const copied = copyInstancedVectorSliceToPreview(
        {
            mesh: clone,
            instanceCount: clone.count
        },
        sliceSource.sourceMesh,
        sliceSource.sourceOffset,
        sliceSource.instanceCount
    );
    if (!copied) {
        previewGeometries.forEach((geo) => geo && geo.dispose && geo.dispose());
        previewMaterials.forEach((mat) => mat && mat.dispose && mat.dispose());
        return null;
    }

    if (clone.geometry) {
        if (!clone.geometry.boundingBox && typeof clone.geometry.computeBoundingBox === 'function') {
            clone.geometry.computeBoundingBox();
        }
        if (typeof clone.geometry.computeBoundingSphere === 'function') {
            clone.geometry.computeBoundingSphere();
        }
    }
    if (clone.instanceColor) {
        clone.instanceColor.needsUpdate = true;
    }

    return {
        object: group,
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

function resolveSelectionPreviewFitOptions(label, selectionInfo, previewRoot, previewView = null, isSmallScreen = false) {
    const isVectorPreview = isLikelyVectorSelection(label, selectionInfo);
    const isQkvMatrixPreview = isQkvMatrixLabel(label);
    const isOutputProjPreview = label.toLowerCase().includes('output projection matrix');
    const paddingMultiplier = isVectorPreview
        ? PREVIEW_VECTOR_PADDING_MULT
        : (isQkvMatrixPreview ? 0.75 : (isOutputProjPreview ? 0.85 : 1));
    const distanceMultiplier = isVectorPreview
        ? PREVIEW_VECTOR_DISTANCE_MULT
        : (isQkvMatrixPreview ? 0.85 : (isOutputProjPreview ? 0.8 : 1));
    const laneZoom = getLaneZoomMultiplier(previewRoot);
    const isMatrixPreview = isWeightMatrixLabel(label) || isOutputProjPreview;
    const matrixPaddingBoost = (isSmallScreen && isMatrixPreview) ? PREVIEW_MOBILE_MATRIX_PADDING_MULT : 1;
    const matrixDistanceBoost = (isSmallScreen && isMatrixPreview) ? PREVIEW_MOBILE_MATRIX_DISTANCE_MULT : 1;
    const viewPaddingBoost = Number.isFinite(previewView?.fitPaddingMultiplier)
        ? previewView.fitPaddingMultiplier
        : 1;
    const viewDistanceBoost = Number.isFinite(previewView?.fitDistanceMultiplier)
        ? previewView.fitDistanceMultiplier
        : 1;
    return {
        paddingMultiplier: paddingMultiplier * laneZoom * matrixPaddingBoost * viewPaddingBoost,
        distanceMultiplier: distanceMultiplier * laneZoom * matrixDistanceBoost * viewDistanceBoost
    };
}

function resolveMetadata(label, kind = null, selectionInfo = null) {
    const lower = (label || '').toLowerCase();
    const dimensionLabels = resolveMetadataDimensionLabels(label);
    if (lower.startsWith('token:') || lower.startsWith('position:')) {
        const oneHotLength = lower.startsWith('position:') ? CONTEXT_LEN : VOCAB_SIZE;
        return buildMetadata('TBD', null, null, oneHotLength, null, dimensionLabels);
    }
    if (lower.includes('query weight matrix')) {
        return buildMetadata(formatNumber(D_MODEL * D_HEAD), D_MODEL, D_HEAD, null, D_HEAD, dimensionLabels);
    }
    if (lower.includes('key weight matrix')) {
        return buildMetadata(formatNumber(D_MODEL * D_HEAD), D_MODEL, D_HEAD, null, D_HEAD, dimensionLabels);
    }
    if (lower.includes('value weight matrix')) {
        return buildMetadata(formatNumber(D_MODEL * D_HEAD), D_MODEL, D_HEAD, null, D_HEAD, dimensionLabels);
    }
    if (lower.includes('output projection matrix')) {
        return buildMetadata(formatNumber(D_MODEL * D_MODEL), D_MODEL, D_MODEL, null, D_MODEL, dimensionLabels);
    }
    if (lower.includes('mlp up weight matrix')) {
        return buildMetadata(formatNumber(D_MODEL * D_MODEL * 4), D_MODEL, D_MODEL * 4, null, D_MODEL * 4, dimensionLabels);
    }
    if (lower.includes('mlp down weight matrix')) {
        return buildMetadata(formatNumber(D_MODEL * D_MODEL * 4), D_MODEL * 4, D_MODEL, null, D_MODEL, dimensionLabels);
    }
    if (hasTopVocabEmbeddingLabel(lower) || lower.includes('unembedding')) {
        return buildMetadata(formatNumber(VOCAB_SIZE * D_MODEL), D_MODEL, VOCAB_SIZE, null, null, dimensionLabels);
    }
    if (hasVocabEmbeddingLabel(lower)) {
        return buildMetadata(formatNumber(VOCAB_SIZE * D_MODEL), VOCAB_SIZE, D_MODEL, null, null, dimensionLabels);
    }
    if (hasPositionEmbeddingLabel(lower)) {
        return buildMetadata(formatNumber(CONTEXT_LEN * D_MODEL), CONTEXT_LEN, D_MODEL, null, null, dimensionLabels);
    }
    if (isLikelyVectorSelection(label, selectionInfo)) {
        return buildMetadata('TBD', null, null, resolveVectorLength(label, selectionInfo), null, dimensionLabels);
    }
    if (lower.includes('attention') || (kind === 'mergedKV')) {
        return buildMetadata('TBD', null, null, null, null, dimensionLabels);
    }
    return buildMetadata('TBD', null, null, null, null, dimensionLabels);
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

function extractChosenTokenText(label) {
    if (!label) return '';
    const match = String(label).match(/^chosen token\s*:\s*(.*)$/i);
    if (!match) return '';
    const extracted = match[1] || '';
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
    return resolveSelectionLogitEntryMetadata(selectionInfo);
}

function resolveLogitEntryTokenId(entry) {
    const rawTokenId = Number(entry?.token_id ?? entry?.tokenId);
    return Number.isFinite(rawTokenId) ? Math.floor(rawTokenId) : null;
}

function resolveLogitEntryProbability(entry) {
    const rawProbability = Number(entry?.probability ?? entry?.prob);
    return Number.isFinite(rawProbability) ? rawProbability : null;
}

function resolveLogitPreviewTokenText(label, selectionInfo) {
    const entry = resolveLogitSelectionEntry(selectionInfo);
    const previewTokenId = resolveLogitSelectionTokenId(label, entry, selectionInfo);
    const incompleteDisplay = getIncompleteUtf8TokenDisplay(previewTokenId);
    if (incompleteDisplay) return incompleteDisplay;
    const entryTokenText = resolveLogitEntryText(entry);
    if (entryTokenText) {
        const formatted = formatTokenLabelForPreview(sanitizeLogitTokenForPreview(entryTokenText));
        if (formatted) return formatted;
    }
    if (Number.isFinite(previewTokenId)) return `#${previewTokenId}`;
    const labelTokenMatch = String(label || '').match(/token\s+"([^"]+)"/i);
    if (labelTokenMatch && labelTokenMatch[1]) {
        const formatted = formatTokenLabelForPreview(labelTokenMatch[1]);
        if (formatted) return formatted;
    }
    const chosenTokenMatch = String(label || '').match(/^chosen token:\s*(.+)$/i);
    if (chosenTokenMatch && chosenTokenMatch[1]) {
        const formatted = formatTokenLabelForPreview(chosenTokenMatch[1]);
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

function resolveLogitSelectionTokenId(label, entry, selectionInfo = null) {
    const entryTokenId = resolveLogitEntryTokenId(entry);
    if (Number.isFinite(entryTokenId)) return entryTokenId;
    return resolvePreviewTokenId(label, selectionInfo);
}

function resolveLogitSelectionProbability(label, entry, selectionInfo = null) {
    const entryProbability = resolveLogitEntryProbability(entry);
    if (Number.isFinite(entryProbability)) return entryProbability;
    const infoProbability = Number(selectionInfo?.info?.probability ?? selectionInfo?.info?.prob);
    if (Number.isFinite(infoProbability)) return infoProbability;
    const labelProbMatch = String(label || '').match(/\bp\s+(-?\d*\.?\d+(?:e[-+]?\d+)?)\b/i);
    if (!labelProbMatch) return null;
    const parsed = Number(labelProbMatch[1]);
    return Number.isFinite(parsed) ? parsed : null;
}

function formatLogitProbability(value) {
    if (!Number.isFinite(value)) return '';
    const percentage = value * 100;
    const abs = Math.abs(percentage);
    if (abs === 0) return '0%';
    if (abs < 0.0001) return `${percentage.toExponential(2)}%`;
    if (abs < 0.01) return `${percentage.toFixed(4).replace(/\.?0+$/, '')}%`;
    return `${percentage.toFixed(2).replace(/\.?0+$/, '')}%`;
}

function resolveLogitSelectionHeader(label, selectionInfo) {
    const lower = String(label || '').toLowerCase();
    const isChosenTokenLabel = lower.startsWith('chosen token:');
    const isSingleLogitLabel = lower === 'logit' || lower.startsWith('logit ');
    const isLogitBar = isLogitBarSelection(label, selectionInfo);
    if (!isLogitBar && !isChosenTokenLabel && !isSingleLogitLabel) return null;
    const entry = resolveLogitSelectionEntry(selectionInfo);
    const hasEntryData = !!(
        entry
        && typeof entry === 'object'
        && (
            typeof entry.token === 'string'
            || Number.isFinite(resolveLogitEntryTokenId(entry))
            || Number.isFinite(resolveLogitEntryProbability(entry))
            || Number.isFinite(entry.logit)
        )
    );
    const isAggregateLogitBars = lower.includes('top logit bars');
    if (isAggregateLogitBars && !hasEntryData) return null;
    if (!hasEntryData && !isSingleLogitLabel && !isChosenTokenLabel) return null;

    const tokenText = resolveLogitPreviewTokenText(label, selectionInfo);
    const tokenId = resolveLogitSelectionTokenId(label, entry, selectionInfo);
    const probability = resolveLogitSelectionProbability(label, entry, selectionInfo);
    const subtitleParts = [];
    if (isChosenTokenLabel) subtitleParts.push('Chosen token');
    if (Number.isFinite(tokenId)) subtitleParts.push(`ID ${tokenId}`);
    if (Number.isFinite(probability)) subtitleParts.push(`Probability ${formatLogitProbability(probability)}`);

    return {
        title: tokenText || (isChosenTokenLabel ? 'Chosen token' : 'Logit Token'),
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
    const tokenId = resolveLogitSelectionTokenId(label, resolveLogitSelectionEntry(selectionInfo), selectionInfo);
    return buildTokenChipPreview(tokenText, { tokenId });
}

function createTokenChipShared(labelText, tokenId = null) {
    const incompleteDisplay = getIncompleteUtf8TokenDisplay(tokenId);
    const rawText = incompleteDisplay || ((typeof labelText === 'string') ? labelText : '');
    const text = rawText.trim().length ? rawText : SPACE_TOKEN_DISPLAY;
    return createTokenChipMesh({
        labelText: text,
        secondaryText: Number.isFinite(tokenId) ? String(Math.floor(tokenId)) : '',
        style: incompleteDisplay ? TOKEN_CHIP_WARNING_STYLE : TOKEN_CHIP_STYLE,
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

function extractExplicitPreviewVectorData(selectionInfo) {
    const candidates = [
        selectionInfo?.info?.activationData?.values,
        selectionInfo?.info?.values,
        selectionInfo?.info?.vectorData
    ];
    for (const candidate of candidates) {
        const normalized = normalizePreviewVectorDataArray(candidate);
        if (normalized?.length) {
            return normalized;
        }
    }
    return null;
}

function normalizePreviewVectorDataArray(values) {
    if (!(Array.isArray(values) || ArrayBuffer.isView(values)) || values.length <= 0) {
        return null;
    }
    return Array.from(values).map((value) => (Number.isFinite(value) ? value : 0));
}

function samplePreviewVectorData(values = null, targetLength = 0) {
    const normalized = normalizePreviewVectorDataArray(values);
    const safeTargetLength = Math.max(1, Math.floor(targetLength || 1));
    if (!normalized?.length) {
        return new Array(safeTargetLength).fill(0);
    }
    if (normalized.length <= safeTargetLength) {
        return normalized.slice();
    }
    return Array.from({ length: safeTargetLength }, (_, index) => {
        const start = Math.floor((index * normalized.length) / safeTargetLength);
        const end = Math.max(start + 1, Math.floor(((index + 1) * normalized.length) / safeTargetLength));
        let sum = 0;
        for (let cursor = start; cursor < end; cursor += 1) {
            sum += normalized[cursor];
        }
        return sum / Math.max(1, end - start);
    });
}

function inferMlpBiasPreviewKind(label = '', selectionInfo = null) {
    const lower = String(label || '').toLowerCase();
    const stage = String(getActivationDataFromSelection(selectionInfo)?.stage || '').toLowerCase();
    if (stage === 'mlp.up.bias' || lower === MLP_UP_BIAS_TOOLTIP_LABEL.toLowerCase()) {
        return 'up';
    }
    if (stage === 'mlp.down.bias' || lower === MLP_DOWN_BIAS_TOOLTIP_LABEL.toLowerCase()) {
        return 'down';
    }
    return '';
}

function inferAttentionBiasPreviewKind(label = '', selectionInfo = null) {
    const lower = String(label || '').toLowerCase();
    const stage = String(getActivationDataFromSelection(selectionInfo)?.stage || '').toLowerCase();
    if (stage === 'qkv.q.bias' || lower === 'query bias vector') return 'q';
    if (stage === 'qkv.k.bias' || lower === 'key bias vector') return 'k';
    if (stage === 'qkv.v.bias' || lower === 'value bias vector') return 'v';
    return '';
}

function inferAttentionHeadVectorPreviewKind(label = '', selectionInfo = null) {
    const lower = String(label || '').toLowerCase();
    const stage = String(getActivationDataFromSelection(selectionInfo)?.stage || '').toLowerCase();
    if (stage === 'qkv.q' || lower === 'query vector') return 'q';
    if (stage === 'qkv.k' || lower === 'key vector') return 'k';
    if (stage === 'qkv.v' || lower === 'value vector') return 'v';
    return '';
}

function isOutputProjectionBiasPreviewSelection(label = '', selectionInfo = null) {
    const lower = String(label || '').toLowerCase();
    const stage = String(getActivationDataFromSelection(selectionInfo)?.stage || '').toLowerCase();
    return stage === 'attention.output_projection.bias' || lower === 'output projection bias vector';
}

function isAttentionOutputProjectionVectorSelection(label = '', selectionInfo = null) {
    const lower = String(label || '').toLowerCase();
    const stage = String(getActivationDataFromSelection(selectionInfo)?.stage || '').toLowerCase();
    return stage === 'attention.output_projection'
        || lower === 'attention output vector'
        || lower === 'model-width attention output vector';
}

function resolveMlpBiasPreviewSampleCount(kind = 'up') {
    return kind === 'down'
        ? Math.max(1, Math.ceil(D_MODEL / PRISM_DIMENSIONS_PER_UNIT))
        : Math.max(1, Math.ceil((D_MODEL * 4) / PRISM_DIMENSIONS_PER_UNIT));
}

function resolveOutputProjectionBiasPreviewSampleCount() {
    return Math.max(1, Math.ceil(D_MODEL / PRISM_DIMENSIONS_PER_UNIT));
}

function resolveSelectionLayerIndex(selectionInfo = null) {
    const directLayerIndex = selectionInfo?.info?.activationData?.layerIndex;
    if (Number.isFinite(directLayerIndex)) {
        return Math.max(0, Math.floor(directLayerIndex));
    }
    const nestedLayerIndex = selectionInfo?.info?.layerIndex;
    if (Number.isFinite(nestedLayerIndex)) {
        return Math.max(0, Math.floor(nestedLayerIndex));
    }
    const discoveredLayerIndex = findUserDataNumber(selectionInfo, 'layerIndex');
    return Number.isFinite(discoveredLayerIndex) ? Math.max(0, Math.floor(discoveredLayerIndex)) : null;
}

function resolveSelectionHeadIndex(selectionInfo = null) {
    const directHeadIndex = selectionInfo?.info?.activationData?.headIndex;
    if (Number.isFinite(directHeadIndex)) {
        return Math.max(0, Math.floor(directHeadIndex));
    }
    const nestedHeadIndex = selectionInfo?.info?.headIndex;
    if (Number.isFinite(nestedHeadIndex)) {
        return Math.max(0, Math.floor(nestedHeadIndex));
    }
    const discoveredHeadIndex = findUserDataNumber(selectionInfo, 'headIndex');
    return Number.isFinite(discoveredHeadIndex) ? Math.max(0, Math.floor(discoveredHeadIndex)) : null;
}

function resolveMlpBiasPreviewData(selectionInfo = null, kind = 'up') {
    const sampleCount = resolveMlpBiasPreviewSampleCount(kind);
    const layerIndex = resolveSelectionLayerIndex(selectionInfo);
    if (Number.isFinite(layerIndex)) {
        const sampledValues = normalizePreviewVectorDataArray(getMlpBiasVectorSample(layerIndex, kind));
        if (sampledValues?.length) {
            return sampledValues;
        }
    }
    const rawValues = extractPreviewVectorData(selectionInfo);
    if (!rawValues?.length) return null;
    if (rawValues.length === sampleCount) {
        return rawValues;
    }
    return samplePreviewVectorData(rawValues, sampleCount);
}

function resolveAttentionBiasPreviewData(selectionInfo = null, kind = 'q') {
    const explicitValues = normalizePreviewVectorDataArray(
        selectionInfo?.info?.activationData?.values
        || selectionInfo?.info?.values
    );
    if (explicitValues?.length) {
        return explicitValues.slice(0, 1);
    }
    const layerIndex = resolveSelectionLayerIndex(selectionInfo);
    const headIndex = resolveSelectionHeadIndex(selectionInfo);
    if (!Number.isFinite(layerIndex) || !Number.isFinite(headIndex)) return null;
    const biasKind = kind === 'k' ? 'key' : (kind === 'v' ? 'value' : 'query');
    const biasValue = getAttentionBiasHeadSample(layerIndex, biasKind, headIndex);
    return Number.isFinite(biasValue) ? [biasValue] : null;
}

function resolveAttentionHeadVectorPreviewData(selectionInfo = null) {
    const explicitValues = normalizePreviewVectorDataArray(
        selectionInfo?.info?.activationData?.values
        || selectionInfo?.info?.values
    );
    if (explicitValues?.length) {
        return explicitValues;
    }
    return extractPreviewVectorData(selectionInfo);
}

function resolveAttentionWeightedSumPreviewData(selectionInfo = null) {
    const explicitValues = normalizePreviewVectorDataArray(
        selectionInfo?.info?.activationData?.values
        || selectionInfo?.info?.values
    );
    if (explicitValues?.length) {
        return explicitValues;
    }
    return extractPreviewVectorData(selectionInfo);
}

function resolveOutputProjectionBiasPreviewData(selectionInfo = null) {
    const sampleCount = resolveOutputProjectionBiasPreviewSampleCount();
    const explicitValues = normalizePreviewVectorDataArray(
        selectionInfo?.info?.activationData?.values
        || selectionInfo?.info?.values
    );
    if (explicitValues?.length) {
        return explicitValues.length === sampleCount
            ? explicitValues
            : samplePreviewVectorData(explicitValues, sampleCount);
    }
    const layerIndex = resolveSelectionLayerIndex(selectionInfo);
    if (Number.isFinite(layerIndex)) {
        const sampledValues = normalizePreviewVectorDataArray(getAttentionBiasVectorSample(layerIndex, 'output'));
        if (sampledValues?.length) {
            return sampledValues.length === sampleCount
                ? sampledValues
                : samplePreviewVectorData(sampledValues, sampleCount);
        }
    }
    const rawValues = extractPreviewVectorData(selectionInfo);
    if (!rawValues?.length) return null;
    return rawValues.length === sampleCount
        ? rawValues
        : samplePreviewVectorData(rawValues, sampleCount);
}

function extractMlpMiddlePreviewVectorData(selectionInfo) {
    const vectorRef = selectionInfo?.info?.vectorRef || null;
    const batchRefs = vectorRef?.isBatchedVectorRef && Array.isArray(vectorRef?._batch?._vectorRefs)
        ? vectorRef._batch._vectorRefs
        : null;
    if (batchRefs?.length) {
        const combined = [];
        const orderedRefs = batchRefs
            .filter((candidate) => candidate && typeof candidate === 'object')
            .slice()
            .sort((a, b) => {
                const left = Number.isFinite(a?._index) ? Math.floor(a._index) : 0;
                const right = Number.isFinite(b?._index) ? Math.floor(b._index) : 0;
                return left - right;
            });
        orderedRefs.forEach((candidate) => {
            const segmentData = normalizePreviewVectorDataArray(
                candidate?.userData?.activationData?.values
                || candidate?.rawData
            );
            if (segmentData?.length) {
                combined.push(...segmentData);
            }
        });
        if (combined.length) return combined;
    }

    const candidates = [
        selectionInfo?.info?.values,
        selectionInfo?.info?.vectorData,
        selectionInfo?.info?.activationData?.values,
        selectionInfo?.object?.userData?.activationData?.values,
        selectionInfo?.hit?.object?.userData?.activationData?.values,
        vectorRef?.userData?.activationData?.values,
        vectorRef?.rawData
    ];
    for (const candidate of candidates) {
        const normalized = normalizePreviewVectorDataArray(candidate);
        if (normalized?.length) return normalized;
    }
    return null;
}

function resolveMlpMiddlePreviewInstanceCount(selectionInfo, previewData = null) {
    const vectorRef = selectionInfo?.info?.vectorRef || null;
    const batch = vectorRef?.isBatchedVectorRef ? vectorRef?._batch : null;
    const batchPrismCount = Number.isFinite(batch?.prismCount)
        ? Math.max(1, Math.floor(batch.prismCount))
        : null;
    const batchVectorCount = Number.isFinite(batch?.vectorCount)
        ? Math.max(1, Math.floor(batch.vectorCount))
        : (
            Array.isArray(batch?._vectorRefs) && batch._vectorRefs.length
                ? Math.max(1, batch._vectorRefs.length)
                : null
        );
    if (batchPrismCount && batchVectorCount) {
        return batchPrismCount * batchVectorCount;
    }

    const sourceMesh = findVectorSourceMesh(selectionInfo);
    const meshPrismCount = Number.isFinite(sourceMesh?.userData?.prismCount)
        ? Math.max(1, Math.floor(sourceMesh.userData.prismCount))
        : null;
    const meshCount = Number.isFinite(sourceMesh?.count)
        ? Math.max(1, Math.floor(sourceMesh.count))
        : Math.max(0, Math.floor(sourceMesh?.instanceMatrix?.count || 0));
    if (meshPrismCount && meshCount > meshPrismCount) {
        return meshCount;
    }

    const dataLength = Array.isArray(previewData) || ArrayBuffer.isView(previewData)
        ? previewData.length
        : 0;
    if (dataLength > 0) {
        const maxExpandedPrismCount = Math.max(1, Math.ceil((D_MODEL * 4) / PRISM_DIMENSIONS_PER_UNIT));
        if (dataLength <= maxExpandedPrismCount) {
            return Math.max(1, dataLength);
        }
        return Math.max(1, Math.ceil(dataLength / PRISM_DIMENSIONS_PER_UNIT));
    }
    return null;
}

function buildMlpMiddleVectorPreview(selectionInfo, label = '') {
    const previewData = extractMlpMiddlePreviewVectorData(selectionInfo);
    const instanceCount = resolveMlpMiddlePreviewInstanceCount(selectionInfo, previewData);
    if (!(Number.isFinite(instanceCount) && instanceCount > 0)) return null;

    const vec = createPreviewVector({
        colorHex: resolveVectorPreviewColor(label, selectionInfo),
        data: null,
        instanceCount
    });

    if (Array.isArray(previewData) && previewData.length > 0) {
        applyDataToPreviewVector(vec, previewData);
    }

    const batchMesh = selectionInfo?.info?.vectorRef?.isBatchedVectorRef
        ? selectionInfo.info.vectorRef?._batch?.mesh
        : null;
    const colorSourceMesh = batchMesh?.isInstancedMesh
        ? batchMesh
        : findVectorSourceMesh(selectionInfo);
    if (colorSourceMesh?.isInstancedMesh) {
        copyInstancedVectorColorsToPreview(vec, colorSourceMesh, 0, instanceCount);
    }

    return {
        object: vec.group,
        dispose: () => {
            if (vec.mesh?.geometry) vec.mesh.geometry.dispose();
            if (vec.mesh?.material) vec.mesh.material.dispose();
        }
    };
}

function buildMlpBiasPreviewColorOptions(previewData = []) {
    const safeData = Array.isArray(previewData) ? previewData : [];
    const maxAbsValue = safeData.reduce((maxValue, value) => (
        Number.isFinite(value) ? Math.max(maxValue, Math.abs(value)) : maxValue
    ), 0);
    const clampRange = maxAbsValue > 1e-6 ? maxAbsValue : 1;
    const biasBaseColor = new THREE.Color(FINAL_MLP_COLOR);
    const biasBaseHsl = { h: 0, s: 0, l: 0 };
    biasBaseColor.getHSL(biasBaseHsl);
    biasBaseColor.setHSL(
        (biasBaseHsl.h + 0.018) % 1,
        Math.min(1, Math.max(0.82, biasBaseHsl.s)),
        biasBaseHsl.l
    );
    return buildHueRangeOptions(biasBaseColor, {
        hueSpread: 0.1,
        valueMin: -clampRange,
        valueMax: clampRange,
        minLightness: 0.36,
        maxLightness: 0.72,
        saturation: Math.min(1, Math.max(0.84, biasBaseHsl.s))
    });
}

function buildAttentionBiasPreviewColorOptions(kind = 'q') {
    const baseColor = kind === 'k'
        ? MHA_FINAL_K_COLOR
        : (kind === 'v' ? MHA_VALUE_SPECTRUM_COLOR : MHA_FINAL_Q_COLOR);
    return buildHueRangeOptions(baseColor, {
        hueSpread: MHA_VALUE_HUE_SPREAD,
        minLightness: MHA_VALUE_LIGHTNESS_MIN,
        maxLightness: MHA_VALUE_LIGHTNESS_MAX,
        valueMin: MHA_VALUE_RANGE_MIN,
        valueMax: MHA_VALUE_RANGE_MAX,
        valueClampMax: MHA_VALUE_CLAMP_MAX
    });
}

function buildAttentionWeightedSumPreviewColorOptions() {
    return buildHueRangeOptions(MHA_VALUE_SPECTRUM_COLOR, {
        hueSpread: MHA_VALUE_HUE_SPREAD,
        minLightness: MHA_VALUE_LIGHTNESS_MIN,
        maxLightness: MHA_VALUE_LIGHTNESS_MAX,
        valueMin: MHA_VALUE_RANGE_MIN,
        valueMax: MHA_VALUE_RANGE_MAX,
        valueClampMax: MHA_VALUE_CLAMP_MAX
    });
}

function buildAttentionHeadVectorPreview(selectionInfo, label = '') {
    const kind = inferAttentionHeadVectorPreviewKind(label, selectionInfo);
    if (!kind) return null;
    const outputLength = Math.max(1, Math.floor(resolveVectorLength(label, selectionInfo) || D_HEAD));
    const previewData = resolveAttentionHeadVectorPreviewData(selectionInfo);
    if (!previewData?.length) return null;
    const vec = createPreviewVector({
        colorHex: resolveVectorPreviewColor(label, selectionInfo),
        data: null,
        instanceCount: Math.max(1, Math.ceil(outputLength / PRISM_DIMENSIONS_PER_UNIT))
    });
    const processedData = previewData.slice(0, outputLength);
    const numKeyColors = processedData.length <= 1
        ? 1
        : Math.min(MHA_VALUE_KEY_COLOR_COUNT, processedData.length);
    vec.group.userData = vec.group.userData || {};
    vec.group.userData.label = label || (
        kind === 'k'
            ? 'Key Vector'
            : (kind === 'v' ? 'Value Vector' : 'Query Vector')
    );
    vec.applyProcessedVisuals(
        processedData,
        outputLength,
        {
            numKeyColors,
            generationOptions: buildAttentionBiasPreviewColorOptions(kind)
        },
        {
            setHiddenToBlack: false,
            hideByScaleOnly: true
        },
        processedData
    );
    return {
        object: vec.group,
        view: {
            fitDistanceMultiplier: 1.18,
            fitPaddingMultiplier: 1.05
        },
        dispose: () => {
            vec.dispose();
        }
    };
}

function buildOutputProjectionBiasPreviewColorOptions() {
    return buildHueRangeOptions(MHA_OUTPUT_PROJECTION_MATRIX_COLOR, {
        valueMin: -2,
        valueMax: 2,
        minLightness: 0.34,
        maxLightness: 0.72
    });
}

function buildAttentionBiasVectorPreview(selectionInfo, label = '') {
    const kind = inferAttentionBiasPreviewKind(label, selectionInfo);
    if (!kind) return null;
    const previewData = resolveAttentionBiasPreviewData(selectionInfo, kind);
    if (!previewData?.length) return null;

    const vec = createPreviewVector({
        colorHex: resolveVectorPreviewColor(label, selectionInfo),
        data: null,
        instanceCount: 1
    });
    vec.group.userData = vec.group.userData || {};
    vec.group.userData.label = label || (
        kind === 'k'
            ? 'Key Bias Vector'
            : (kind === 'v' ? 'Value Bias Vector' : 'Query Bias Vector')
    );
    applyDataToPreviewVector(vec, previewData, {
        colorOptions: buildAttentionBiasPreviewColorOptions(kind),
        numKeyColors: 1
    });

    return {
        object: vec.group,
        view: {
            fitDistanceMultiplier: 1.18,
            fitPaddingMultiplier: 1.05
        },
        dispose: () => {
            vec.dispose();
        }
    };
}

function buildAttentionWeightedSumVectorPreview(selectionInfo, label = '') {
    if (!isWeightedSumSelection(label, selectionInfo)) return null;
    const outputLength = Math.max(1, Math.floor(resolveVectorLength(label, selectionInfo) || D_HEAD));
    const previewData = resolveAttentionWeightedSumPreviewData(selectionInfo);
    if (!previewData?.length) return null;
    const vec = createPreviewVector({
        colorHex: MHA_FINAL_V_COLOR,
        data: null,
        instanceCount: Math.max(1, Math.ceil(outputLength / PRISM_DIMENSIONS_PER_UNIT))
    });
    vec.group.userData = vec.group.userData || {};
    vec.group.userData.label = label || 'Attention Weighted Sum';
    applyDataToPreviewVector(vec, previewData.slice(0, outputLength), {
        colorOptions: buildAttentionWeightedSumPreviewColorOptions(),
        numKeyColors: 1
    });
    return {
        object: vec.group,
        view: {
            fitDistanceMultiplier: 1.18,
            fitPaddingMultiplier: 1.05
        },
        dispose: () => {
            vec.dispose();
        }
    };
}

function buildOutputProjectionBiasVectorPreview(selectionInfo, label = '') {
    if (!isOutputProjectionBiasPreviewSelection(label, selectionInfo)) return null;
    const previewData = resolveOutputProjectionBiasPreviewData(selectionInfo);
    if (!previewData?.length) return null;
    const sampleCount = Math.max(1, previewData.length || resolveOutputProjectionBiasPreviewSampleCount());
    const previewLabel = label || 'Output Projection Bias Vector';
    const vec = createPreviewVector({
        colorHex: MHA_OUTPUT_PROJECTION_MATRIX_COLOR,
        data: null,
        instanceCount: sampleCount
    });
    vec.group.userData = vec.group.userData || {};
    vec.group.userData.label = previewLabel;
    applyDataToPreviewVector(vec, previewData, {
        colorOptions: buildOutputProjectionBiasPreviewColorOptions(),
        numKeyColors: sampleCount
    });
    return {
        object: vec.group,
        view: {
            fitDistanceMultiplier: 1.14,
            fitPaddingMultiplier: 1.04
        },
        dispose: () => {
            vec.dispose();
        }
    };
}

function buildAttentionOutputProjectionVectorPreview(selectionInfo, label = '') {
    if (!isAttentionOutputProjectionVectorSelection(label, selectionInfo)) return null;
    const previewData = extractPreviewVectorData(selectionInfo);
    if (!previewData?.length) return null;
    const outputLength = Math.max(1, Math.floor(resolveVectorLength(label, selectionInfo) || D_MODEL));
    const vec = createPreviewVector({
        colorHex: resolveVectorPreviewColor(label, selectionInfo),
        data: null,
        instanceCount: Math.max(1, Math.ceil(outputLength / PRISM_DIMENSIONS_PER_UNIT))
    });
    vec.group.userData = vec.group.userData || {};
    vec.group.userData.label = label || 'Attention Output Vector';
    applyDataToPreviewVector(vec, previewData);
    return {
        object: vec.group,
        view: {
            fitDistanceMultiplier: 1.14,
            fitPaddingMultiplier: 1.04
        },
        dispose: () => {
            vec.dispose();
        }
    };
}

function buildMlpBiasVectorPreview(selectionInfo, label = '') {
    const kind = inferMlpBiasPreviewKind(label, selectionInfo);
    if (!kind) return null;
    const previewData = resolveMlpBiasPreviewData(selectionInfo, kind);
    if (!previewData?.length) return null;
    const sampleCount = resolveMlpBiasPreviewSampleCount(kind);
    const previewLabel = label || (kind === 'down' ? MLP_DOWN_BIAS_TOOLTIP_LABEL : MLP_UP_BIAS_TOOLTIP_LABEL);
    const vec = createPreviewVector({
        colorHex: FINAL_MLP_COLOR,
        data: null,
        instanceCount: sampleCount
    });
    vec.group.userData = vec.group.userData || {};
    vec.group.userData.label = previewLabel;
    applyDataToPreviewVector(vec, previewData, {
        colorOptions: buildMlpBiasPreviewColorOptions(previewData),
        numKeyColors: sampleCount
    });
    return {
        object: vec.group,
        view: {
            fitDistanceMultiplier: 1.14,
            fitPaddingMultiplier: 1.04
        },
        dispose: () => {
            vec.dispose();
        }
    };
}

function applyDataToPreviewVector(vec, data, {
    colorOptions = null,
    minimumKeyColors = 2,
    numKeyColors = null
} = {}) {
    const isArrayLike = Array.isArray(data) || ArrayBuffer.isView(data);
    if (!vec || !isArrayLike || data.length === 0) return;
    const raw = Array.from(data);
    if (typeof vec.updateDataInternal === 'function') {
        vec.updateDataInternal(raw);
    } else {
        vec.updateDataAndSnapVisuals(raw);
    }
    const resolvedNumKeyColors = Number.isFinite(numKeyColors)
        ? Math.max(1, Math.floor(numKeyColors))
        : Math.min(30, Math.max(minimumKeyColors, raw.length));
    vec.updateKeyColorsFromData(raw, resolvedNumKeyColors, colorOptions, raw);
    if (typeof vec.updateInstanceGeometryAndColors === 'function') {
        vec.updateInstanceGeometryAndColors();
    }
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
        selectionInfo?.info?.prismCount,
        selectionInfo?.object?.userData?.prismCount,
        selectionInfo?.hit?.object?.userData?.prismCount,
        vectorRef?.instanceCount,
        vectorRef?._batch?.prismCount,
        vectorRef?.mesh?.userData?.prismCount,
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

function isLayerNormSingleVectorFallbackSelection(label = '', selectionInfo = null) {
    const activationData = getActivationDataFromSelection(selectionInfo);
    const stageLower = String(activationData?.stage || '').toLowerCase();
    const sourceStageLower = String(activationData?.sourceStage || '').toLowerCase();
    const stageCandidates = [stageLower, sourceStageLower].filter(Boolean);

    if (stageCandidates.some((stage) => Boolean(normalizeLayerNormProductStage(stage)))) {
        return true;
    }

    return stageCandidates.some((stage) => (
        stage === 'ln1.norm'
        || stage === 'ln2.norm'
        || stage === 'final_ln.input'
        || stage === 'final_ln.norm'
        || stage === 'final_ln.output'
    ));
}

export function buildVectorClonePreview(selectionInfo, label = '') {
    const weightedSumSelection = isWeightedSumSelection(label, selectionInfo);
    const kvCacheVectorSelection = isKvCacheVectorSelection(selectionInfo);
    const mlpMiddleVectorSelection = isMlpMiddleVectorSelection(label, selectionInfo);
    const activationStageLower = String(getActivationDataFromSelection(selectionInfo)?.stage || '').toLowerCase();
    const isMlpDownVectorSelection = activationStageLower === 'mlp.down'
        || String(label || '').toLowerCase().startsWith('mlp down projection');
    if (weightedSumSelection) {
        // Use the exact runtime vector geometry for weighted-sum selections.
        const directClone = buildDirectClonePreview(selectionInfo)
            || buildSelectionClonePreview(selectionInfo, label);
        if (directClone) return directClone;
    }

    const attentionWeightedSumPreview = buildAttentionWeightedSumVectorPreview(selectionInfo, label);
    if (attentionWeightedSumPreview) return attentionWeightedSumPreview;

    const attentionOutputProjectionPreview = buildAttentionOutputProjectionVectorPreview(selectionInfo, label);
    if (attentionOutputProjectionPreview) return attentionOutputProjectionPreview;

    const attentionBiasVectorPreview = buildAttentionBiasVectorPreview(selectionInfo, label);
    if (attentionBiasVectorPreview) return attentionBiasVectorPreview;

    const outputProjectionBiasVectorPreview = buildOutputProjectionBiasVectorPreview(selectionInfo, label);
    if (outputProjectionBiasVectorPreview) return outputProjectionBiasVectorPreview;

    const mlpBiasVectorPreview = buildMlpBiasVectorPreview(selectionInfo, label);
    if (mlpBiasVectorPreview) return mlpBiasVectorPreview;

    if (mlpMiddleVectorSelection) {
        const expandedVectorPreview = buildMlpMiddleVectorPreview(selectionInfo, label);
        if (expandedVectorPreview) return expandedVectorPreview;
    }

    const exactVectorSlicePreview = buildExactVectorMeshSlicePreview(selectionInfo, label);
    if (exactVectorSlicePreview) return exactVectorSlicePreview;

    const vectorRef = selectionInfo?.info?.vectorRef || null;
    const vectorMesh = findVectorSourceMesh(selectionInfo);
    if (!vectorRef && !vectorMesh) {
        const attentionHeadVectorPreview = buildAttentionHeadVectorPreview(selectionInfo, label);
        if (attentionHeadVectorPreview) return attentionHeadVectorPreview;

        const fallbackData = extractPreviewVectorData(selectionInfo);
        if (
            (isResidualVectorSelection(label, selectionInfo)
            || isPostLayerNormResidualSelection({
                label,
                stage: getActivationDataFromSelection(selectionInfo)?.stage || ''
            })
            || isLayerNormSingleVectorFallbackSelection(label, selectionInfo)
            || isMlpDownVectorSelection)
            && Array.isArray(fallbackData)
            && fallbackData.length > 0
        ) {
            const vec = createPreviewVector({
                colorHex: resolveVectorPreviewColor(label, selectionInfo),
                data: null,
                instanceCount: Math.max(1, Math.floor(resolveVectorPreviewInstanceCount(selectionInfo, label) || PREVIEW_VECTOR_BODY_INSTANCES))
            });
            applyDataToPreviewVector(vec, fallbackData);
            return {
                object: vec.group,
                dispose: () => {
                    if (vec.mesh?.geometry) vec.mesh.geometry.dispose();
                    if (vec.mesh?.material) vec.mesh.material.dispose();
                }
            };
        }
        return null;
    }

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
        || stageLower === 'attention.post'
        || stageLower === 'attention.mask';
    if (!isAttentionScore) return null;

    const mode = resolveAttentionSelectionDisplayMode(
        selectionInfo,
        stageLower.includes('post') ? 'post' : 'pre'
    );
    const score = mode === 'mask'
        ? resolveCausalMaskPreviewValue({
            maskValue: activation?.maskValue,
            isMasked: activation?.isMasked === true,
            queryTokenIndex: activation?.tokenIndex,
            keyTokenIndex: activation?.keyTokenIndex
        })
        : resolveAttentionMatrixCellValue({
            mode,
            value: mode === 'post' ? activation?.postScore : activation?.preScore,
            queryTokenIndex: activation?.tokenIndex,
            keyTokenIndex: activation?.keyTokenIndex
        });
    const color = TMP_COLOR;
    let usedLiveColor = false;
    if (mode !== 'mask' && source?.isInstancedMesh) {
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
            color.copy(mapAttentionPostScoreToColor(Number.isFinite(score) ? score : 0.5));
        } else if (mode === 'mask') {
            color.copy(resolveAttentionPreviewCellColor(score, 'mask'));
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
    mesh.scale.setScalar(0.64);
    return {
        object: mesh,
        view: {
            fitDistanceMultiplier: 1.22,
            fitPaddingMultiplier: 1.04
        },
        dispose: () => {
            geometry.dispose();
            material.dispose();
        }
    };
}

export function buildLayerNormPreview(label, selectionInfo, engine = null) {
    const clonePreview = buildSelectionClonePreview(selectionInfo, label)
        || buildDirectClonePreview(selectionInfo);
    if (clonePreview) {
        applyFinalColorToObject(clonePreview.object, 0xffffff);
        return clonePreview;
    }

    const baseHoles = Number.isFinite(LN_PARAMS.numberOfHoles) ? LN_PARAMS.numberOfHoles : PREVIEW_SOLID_LANES;
    const previewDepth = Math.max(
        180,
        Math.min(PREVIEW_MATRIX_DEPTH, LN_PARAMS.depth * (PREVIEW_SOLID_LANES / Math.max(1, baseHoles)))
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

function shouldShowVectorLegendForSelection(selectionInfo = null) {
    const label = selectionInfo?.label || '';
    if (isResidualVectorSelection(label, selectionInfo) && isLikelyVectorSelection(label, selectionInfo)) {
        return true;
    }
    return !!resolveLayerNormParamPreviewSpec(label, selectionInfo);
}

export function buildLayerNormParamVectorPreview(label, selectionInfo) {
    const spec = resolveLayerNormParamPreviewSpec(label, selectionInfo);
    if (!spec) return null;

    const previewLength = Math.max(1, Math.floor(resolveVectorLength(label, selectionInfo) || D_MODEL));
    const previewData = getLayerNormParamData(spec.layerIndex, spec.layerNormKind, spec.param, previewLength)
        || extractPreviewVectorData(selectionInfo);
    if (!Array.isArray(previewData) || previewData.length === 0) {
        return buildVectorClonePreview(selectionInfo, label);
    }
    const previewInstanceCount = resolveLayerNormParamPreviewInstanceCount(selectionInfo, previewData);

    const vec = createPreviewVector({
        colorHex: MHA_FINAL_Q_COLOR,
        data: null,
        instanceCount: previewInstanceCount
    });
    vec.group.userData = vec.group.userData || {};
    vec.group.userData.label = getLayerNormParamLabel(spec.layerNormKind, spec.param);
    applyDataToPreviewVector(vec, previewData);

    return {
        object: vec.group,
        dispose: () => {
            vec.dispose();
        }
    };
}

function resolvePreviewObject(label, selectionInfo, engine = null) {
    const previewSelectionInfo = resolveLiveLayerNormNormalizedPreviewSelection(selectionInfo, engine) || selectionInfo;
    const lower = (label || '').toLowerCase();
    if (isLogitBarSelection(label, previewSelectionInfo) || lower.startsWith('chosen token:')) {
        return buildLogitBarPreview(label, previewSelectionInfo);
    }
    const attentionSpherePreview = buildAttentionSpherePreview(previewSelectionInfo);
    if (attentionSpherePreview) return attentionSpherePreview;
    const layerNormParamPreview = buildLayerNormParamVectorPreview(label, previewSelectionInfo);
    if (layerNormParamPreview) return layerNormParamPreview;
    if (isLayerNormSolidSelection(label)) {
        return buildLayerNormPreview(label, previewSelectionInfo, engine);
    }
    const isVectorSelection = isLikelyVectorSelection(label, previewSelectionInfo);
    if (isVectorSelection) {
        const vectorClone = buildVectorClonePreview(previewSelectionInfo, label);
        if (vectorClone) return vectorClone;
    }
    if (isQkvMatrixLabel(lower)) {
        const type = inferQkvType(label, previewSelectionInfo);
        const matrixColor = type === 'Q' ? MHA_FINAL_Q_COLOR : (type === 'V' ? MHA_FINAL_V_COLOR : MHA_FINAL_K_COLOR);
        const clonePreview = buildSelectionClonePreview(previewSelectionInfo, label);
        if (clonePreview) return clonePreview;
        const preview = buildWeightMatrixPreview(MHA_MATRIX_PARAMS, matrixColor);
        const snapshot = extractMaterialSnapshot(previewSelectionInfo);
        if (snapshot) {
            applyMaterialSnapshot(preview.object, snapshot);
            applyFinalColorToObject(preview.object, matrixColor);
        }
        return preview;
    }
    const clonePreview = buildSelectionClonePreview(previewSelectionInfo, label)
        || buildDirectClonePreview(previewSelectionInfo);
    if (clonePreview) {
        if (isLayerNormLabel(label)) {
            applyFinalColorToObject(clonePreview.object, 0xffffff);
        }
        return clonePreview;
    }
    if (lower.startsWith('token:') || lower.startsWith('position:')) {
        const clonePreview = buildSelectionClonePreview(previewSelectionInfo, label);
        if (clonePreview) return clonePreview;
        return buildTokenChipPreview(extractTokenText(label), {
            tokenId: lower.startsWith('token:') ? resolvePreviewTokenId(label, previewSelectionInfo) : null
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
        const clonePreview = buildSelectionClonePreview(previewSelectionInfo, label)
            || buildDirectClonePreview(previewSelectionInfo);
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
    if (hasPositionEmbeddingLabel(lower)) {
        const clonePreview = buildSelectionClonePreview(previewSelectionInfo, label)
            || buildDirectClonePreview(previewSelectionInfo);
        if (clonePreview) return clonePreview;
        return buildWeightMatrixPreview(EMBEDDING_MATRIX_PARAMS_POSITION, POSITION_EMBED_COLOR);
    }
    if (isWeightMatrixLabel(lower)) {
        const color = resolveFinalPreviewColor(label);
        return buildWeightMatrixPreview(MHA_MATRIX_PARAMS, color);
    }

    if (lower.includes('query vector')) {
        return buildDirectClonePreview(previewSelectionInfo)
            || buildSelectionClonePreview(previewSelectionInfo, label)
            || buildVectorPreview(MHA_FINAL_Q_COLOR, previewSelectionInfo);
    }
    if (lower.includes('key vector')) {
        return buildDirectClonePreview(previewSelectionInfo)
            || buildSelectionClonePreview(previewSelectionInfo, label)
            || buildVectorPreview(MHA_FINAL_K_COLOR, previewSelectionInfo);
    }
    if (lower.includes('value vector')) {
        return buildDirectClonePreview(previewSelectionInfo)
            || buildSelectionClonePreview(previewSelectionInfo, label)
            || buildVectorPreview(MHA_FINAL_V_COLOR, previewSelectionInfo);
    }
    if (previewSelectionInfo?.kind === 'mergedKV') {
        const category = (previewSelectionInfo.info?.category === 'V') ? 'V' : 'K';
        return buildDirectClonePreview(previewSelectionInfo)
            || buildSelectionClonePreview(previewSelectionInfo, label)
            || buildVectorPreview(category === 'V' ? MHA_FINAL_V_COLOR : MHA_FINAL_K_COLOR, previewSelectionInfo);
    }
    if (isLikelyVectorSelection(label, previewSelectionInfo)) {
        return buildDirectClonePreview(previewSelectionInfo)
            || buildSelectionClonePreview(previewSelectionInfo, label)
            || buildVectorPreview(null, previewSelectionInfo);
    }

    if (isAttentionScoreSelection(label, previewSelectionInfo)) {
        return buildDirectClonePreview(previewSelectionInfo)
            || buildSelectionClonePreview(previewSelectionInfo, label)
            || buildStackedBoxPreview(0x1b1b1b);
    }

    if (lower.includes('attention')) {
        return buildStackedBoxPreview(0x1b1b1b);
    }

    return buildStackedBoxPreview(0x202020);
}

function fitObjectToView(object, camera, options = {}) {
    if (!object) return;
    const previousRotation = object.rotation?.isEuler ? object.rotation.clone() : null;
    if (previousRotation) {
        // Fit against a stable neutral pose so repeated refits do not drift as
        // the preview spins.
        object.rotation.set(0, 0, 0);
    }
    try {
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
        const viewportWidth = Number.isFinite(options.viewportWidth) ? options.viewportWidth : null;
        const viewportHeight = Number.isFinite(options.viewportHeight) ? options.viewportHeight : null;
        const maxViewportWidthPx = Number.isFinite(options.maxViewportWidthPx) ? options.maxViewportWidthPx : null;
        const maxViewportHeightPx = Number.isFinite(options.maxViewportHeightPx) ? options.maxViewportHeightPx : null;
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
        // Once the preview canvas exceeds the design viewport, hold the
        // object's screen footprint roughly steady instead of letting the
        // rotating mesh keep growing with every panel resize.
        const viewportDistanceScale = Math.max(
            1,
            (viewportWidth && maxViewportWidthPx && maxViewportWidthPx > 0)
                ? (viewportWidth / maxViewportWidthPx)
                : 1,
            (viewportHeight && maxViewportHeightPx && maxViewportHeightPx > 0)
                ? (viewportHeight / maxViewportHeightPx)
                : 1
        );
        const distance = Math.max(distX, distY)
            * PREVIEW_ROTATION_ENVELOPE_MARGIN
            * PREVIEW_BASE_DISTANCE_MULT
            * distanceMult
            * viewportDistanceScale;
        camera.near = Math.max(0.1, distance / 50);
        camera.far = Math.max(distance * 20, distance + scaledMax * 4);
        camera.position.set(0, 0, distance * 1.1);
        camera.lookAt(0, 0, 0);
        camera.updateProjectionMatrix();
    } finally {
        if (previousRotation) {
            object.rotation.copy(previousRotation);
        }
    }
}

export class SelectionPanel {
    constructor(options = {}) {
        this.panel = document.getElementById('detailPanel');
        this.hudStack = document.getElementById('hudStack');
        this.hudPanel = document.getElementById('hudPanel');
        this.resizeHandle = document.getElementById('detailPanelResizeHandle');
        this._rootStyleTarget = document.documentElement || null;
        this.title = document.getElementById('detailTitle');
        this.subtitle = document.getElementById('detailSubtitle');
        this.subtitleSecondary = document.getElementById('detailSubtitleSecondary');
        this.subtitleTertiary = document.getElementById('detailSubtitleTertiary');
        this.params = document.getElementById('detailParams');
        this.paramsRow = document.getElementById('detailParamsRow');
        this.layerNormStageRow = document.getElementById('detailLayerNormStageRow');
        this.layerNormStageLabel = document.getElementById('detailLayerNormStageLabel');
        this.layerNormStageValue = document.getElementById('detailLayerNormStageValue');
        this.inputDim = document.getElementById('detailInputDim');
        this.inputDimLabel = document.getElementById('detailInputDimLabel');
        this.inputDimHalf = document.getElementById('detailInputDimHalf');
        this.outputDim = document.getElementById('detailOutputDim');
        this.outputDimLabel = document.getElementById('detailOutputDimLabel');
        this.outputDimHalf = document.getElementById('detailOutputDimHalf');
        this.biasDimRow = document.getElementById('detailBiasDimRow');
        this.biasDimLabel = document.getElementById('detailBiasDimLabel');
        this.biasDim = document.getElementById('detailBiasDim');
        this.previewMetaSection = document.getElementById('detailPreviewMeta');
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
        this.transformerView2dActionRow = document.getElementById('detailView2dActionRow');
        this.transformerView2dActionBtn = document.getElementById('detailView2dActionBtn');
        this.transformerView2dActionBtnLabel = document.getElementById('detailView2dActionBtnLabel');
        this.copyContextBtn = document.getElementById('detailCopyContextBtn');
        this.copyContextBtnLabel = document.getElementById('detailCopyContextBtnLabel');
        this.copyContextBtnAssistant = document.getElementById('detailCopyContextBtnAssistant');
        this.copyContextRow = this.copyContextBtn?.closest('.detail-copy-context') || null;
        this.closeBtn = document.getElementById('detailClose');
        this.fullscreenToggleBtn = document.getElementById('detailFullscreenToggle');
        this.previewRoot = this.panel?.querySelector('.detail-preview') || null;
        this.canvas = document.getElementById('detailCanvas');
        this.infoPreview = document.getElementById('detailInfoPreview');
        this.infoPreviewEyebrow = document.getElementById('detailInfoPreviewEyebrow');
        this.infoPreviewTitle = document.getElementById('detailInfoPreviewTitle');
        this.infoPreviewSummary = document.getElementById('detailInfoPreviewSummary');
        this.infoPreviewPhase = document.getElementById('detailInfoPreviewPhase');
        this.infoPreviewStepPrefill = document.getElementById('detailInfoPreviewStepPrefill');
        this.infoPreviewStepCache = document.getElementById('detailInfoPreviewStepCache');
        this.infoPreviewStepDecode = document.getElementById('detailInfoPreviewStepDecode');
        this.infoPreviewCellStoreLabel = document.getElementById('detailInfoPreviewCellStoreLabel');
        this.infoPreviewCellPassLabel = document.getElementById('detailInfoPreviewCellPassLabel');
        this.infoPreviewCellBenefitLabel = document.getElementById('detailInfoPreviewCellBenefitLabel');
        this.infoPreviewCellStore = document.getElementById('detailInfoPreviewCellStore');
        this.infoPreviewCellPass = document.getElementById('detailInfoPreviewCellPass');
        this.infoPreviewCellBenefit = document.getElementById('detailInfoPreviewCellBenefit');
        this.mhsaTokenMatrixPreview = document.getElementById('detailMhsaTokenMatrixPreview');
        this.mhsaTokenMatrixStatus = document.getElementById('detailMhsaTokenMatrixStatus');
        this.mhsaTokenMatrixHover = document.getElementById('detailMhsaTokenMatrixHover');
        this.mhsaTokenMatrixHoverToken = document.getElementById('detailMhsaTokenMatrixHoverToken');
        this.mhsaTokenMatrixHoverDim = document.getElementById('detailMhsaTokenMatrixHoverDim');
        this.mhsaTokenMatrixHoverBand = document.getElementById('detailMhsaTokenMatrixHoverBand');
        this.mhsaTokenMatrixHoverValue = document.getElementById('detailMhsaTokenMatrixHoverValue');
        this.mhsaTokenMatrixCanvas = document.getElementById('detailMhsaTokenMatrixCanvas');
        this.mhsaTokenMatrixBody = document.getElementById('detailMhsaTokenMatrixBody');
        if (!this.mhsaTokenMatrixCanvas && this.mhsaTokenMatrixPreview && typeof document !== 'undefined') {
            const canvasEl = document.createElement('canvas');
            canvasEl.id = 'detailMhsaTokenMatrixCanvas';
            canvasEl.className = 'mhsa-token-matrix-preview__canvas';
            canvasEl.hidden = true;
            canvasEl.setAttribute('aria-hidden', 'true');
            this.mhsaTokenMatrixPreview.insertBefore(canvasEl, this.mhsaTokenMatrixBody || null);
            this.mhsaTokenMatrixCanvas = canvasEl;
        }
        this.description = document.getElementById('detailDescription');
        this.equationsSection = document.getElementById('detailEquations');
        this.equationsBody = document.getElementById('detailEquationsBody');
        this.dataEyebrow = document.getElementById('detailDataEyebrow');
        this.dataTitle = document.getElementById('detailDataTitle');
        this.dataBlurb = document.getElementById('detailDataBlurb');
        this.dataEl = document.getElementById('detailData');
        this.dataSection = document.getElementById('detailDataSection');
        this.attentionRoot = document.getElementById('detailAttention');
        this.attentionTitle = document.getElementById('detailAttentionTitle');
        this.attentionHeaderMeta = document.getElementById('detailAttentionHeaderMeta');
        this.attentionContextLabel = document.getElementById('detailAttentionContext');
        this.attentionToggle = document.getElementById('detailAttentionToggle');
        this.attentionToggleLabel = document.getElementById('detailAttentionToggleLabel');
        this.attentionToggleRow = this.attentionToggle?.closest('.detail-attention-toggle')
            || this.attentionToggle?.closest('label')
            || null;
        this.attentionCollapseBtn = document.getElementById('detailAttentionCollapseBtn');
        this.attentionCollapseLabel = document.getElementById('detailAttentionCollapseLabel');
        this.attentionBody = document.getElementById('detailAttentionBody');
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
        this._desktopPanelWidthPx = null;
        this._panelResizeDrag = {
            active: false,
            pointerId: null,
            startX: 0,
            startWidthPx: 0
        };

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
        this._isMhsaInfoSelectionActive = false;
        this._mhsaFullscreenActive = false;
        this._lastFrameTime = performance.now();
        this._rotationSpeedMult = 1;
        this._currentPreviewSelectionKey = null;
        this._lastFitOptions = null;
        this._mobilePauseActive = false;
        this._mobileFocusActive = false;
        this._pauseMainFlowOnMobileFocus = options.pauseMainFlowOnMobileFocus === true;
        this._pendingResizeRaf = null;
        this._previewRafId = null;
        this._previewPausedForPanelResize = false;
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
        this._panelTokenHoverEntries = [];
        this._mirroredTokenHoverEntries = [];
        this._panelKeyboardEngaged = false;
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
        this._softmaxDetailView = createSoftmaxDetailView(this.panel);
        this._softmaxDetailOpen = false;
        this._softmaxSourceSelection = null;
        this._transformerView2dDetailView = null;
        this._transformerView2dDetailViewPromise = null;
        this._transformerView2dDetailOpen = false;
        this._transformerView2dSourceSelection = null;
        this._currentTransformerView2dContext = null;
        this._transformerView2dSelectionSidebarDockRecords = [];
        this._transformerView2dSelectionSidebarDocked = false;
        this._transformerView2dSelectionSidebarRestoreTimer = null;
        this._suppressNextCloseClick = false;
        this._setTransformerView2dActionButtonState(null);
        this._lastSelection = null;
        this._lastSelectionLabel = '';
        this._historyEntries = [];
        this._historyIndex = -1;
        this._createHistoryNavigationControls();

        this._animate = this._animate.bind(this);
        this._onResize = this._onResize.bind(this);
        this._onKeydown = this._onKeydown.bind(this);
        this._onKeyup = this._onKeyup.bind(this);
        this._onCopyContextClick = this._onCopyContextClick.bind(this);
        this._onFullscreenToggleClick = this._onFullscreenToggleClick.bind(this);
        this._onCloseClick = this._onCloseClick.bind(this);
        this._onClosePointerDown = this._onClosePointerDown.bind(this);
        this._onDocumentPointerDown = this._onDocumentPointerDown.bind(this);
        this._onDocumentFocusIn = this._onDocumentFocusIn.bind(this);
        this._blockPreviewGesture = this._blockPreviewGesture.bind(this);
        this._onResizeHandlePointerDown = this._onResizeHandlePointerDown.bind(this);
        this._onResizeHandlePointerMove = this._onResizeHandlePointerMove.bind(this);
        this._onResizeHandlePointerUp = this._onResizeHandlePointerUp.bind(this);
        this._onResizeHandleKeydown = this._onResizeHandleKeydown.bind(this);
        this._onAttentionPointerMove = this._onAttentionPointerMove.bind(this);
        this._onAttentionPointerDown = this._onAttentionPointerDown.bind(this);
        this._clearAttentionHover = this._clearAttentionHover.bind(this);
        this._onAttentionCollapseClick = this._onAttentionCollapseClick.bind(this);
        this._onMhsaTokenMatrixPointerMove = this._onMhsaTokenMatrixPointerMove.bind(this);
        this._onMhsaTokenMatrixPointerDown = this._onMhsaTokenMatrixPointerDown.bind(this);
        this._onMhsaTokenMatrixPointerUp = this._onMhsaTokenMatrixPointerUp.bind(this);
        this._onMhsaTokenMatrixWheel = this._onMhsaTokenMatrixWheel.bind(this);
        this._clearMhsaTokenMatrixHover = this._clearMhsaTokenMatrixHover.bind(this);
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
        this._onDevModeChanged = this._onDevModeChanged.bind(this);

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
        this._attentionSectionCollapsed = getAttentionSectionCollapsedPreference();
        this.attentionMode = this.attentionToggle?.checked ? 'post' : 'pre';
        this._attentionDisplayModeOverride = null;
        this._updateAttentionToggleLabel(this.attentionMode);
        this._updateAttentionTitle();
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
        this._mhsaTokenMatrixData = null;
        this._mhsaTokenMatrixSceneModel = null;
        this._mhsaTokenMatrixSceneLayout = null;
        this._mhsaTokenMatrixCanvasRenderer = this.mhsaTokenMatrixCanvas
            ? new CanvasSceneRenderer({ canvas: this.mhsaTokenMatrixCanvas })
            : null;
        this._mhsaTokenMatrixHoverOverlay = typeof document !== 'undefined'
            ? createHoverLabelOverlay({
                documentRef: document,
                parent: document.body,
                zIndex: 14
            })
            : null;
        this._mhsaTokenMatrixCanvasRenderFrame = null;
        this._mhsaTokenMatrixCanvasDebugSignature = '';
        this._mhsaTokenMatrixRenderToken = 0;
        this._mhsaTokenMatrixHoverCell = null;
        this._mhsaTokenMatrixHoverRow = null;
        this._mhsaTokenMatrixHoverCol = null;
        this._mhsaTokenMatrixHoverKind = null;
        this._mhsaTokenMatrixHoverSource = null;
        this._mhsaTokenMatrixHoverStage = null;
        this._mhsaTokenMatrixPinned = false;
        this._mhsaTokenMatrixPinnedKind = null;
        this._mhsaTokenMatrixPinnedRow = null;
        this._mhsaTokenMatrixPinnedCol = null;
        this._mhsaTokenMatrixPinnedSource = null;
        this._mhsaTokenMatrixPinnedStage = null;
        this._mhsaTokenMatrixPinnedFocusKey = null;
        this._mhsaTokenMatrixRowEls = [];
        this._mhsaTokenMatrixQueryRowEls = [];
        this._mhsaTokenMatrixCompactRowEls = [];
        this._mhsaTokenMatrixTransposeColEls = [];
        this._mhsaTokenMatrixScoreCellEls = [];
        this._mhsaTokenMatrixStaticScoreCellEls = [];
        this._mhsaTokenMatrixMaskCellEls = [];
        this._mhsaTokenMatrixPostCellEls = [];
        this._mhsaTokenMatrixPostCopyCellEls = [];
        this._mhsaTokenMatrixPostCopyRowEls = [];
        this._mhsaTokenMatrixMirroredPostCellEls = [];
        this._mhsaTokenMatrixXMatrixEl = [];
        this._mhsaTokenMatrixQueryMatrixEl = [];
        this._mhsaTokenMatrixProjectionStageEls = [];
        this._mhsaTokenMatrixProjectionComponentEls = [];
        this._mhsaTokenMatrixAttentionFocusEls = null;
        this._mhsaTokenMatrixConnectorEls = {};
        this._mhsaTokenMatrixConnectorFrame = null;
        this._mhsaTokenMatrixLayoutMetrics = null;
        this._mhsaTokenMatrixSceneFocusState = {
            projectionStages: [],
            projectionComponents: [],
            attentionBlocks: [],
            connectors: []
        };
        this._mhsaTokenMatrixTransposeMatrixEl = null;
        this._mhsaTokenMatrixQueryStageIndex = null;
        this._mhsaTokenMatrixKeyStageIndex = null;
        this._mhsaTokenMatrixValueStageIndex = null;
        this._mhsaTokenMatrixWorkspace = null;
        this._mhsaTokenMatrixOverlay = null;
        this._mhsaTokenMatrixPan = {
            active: false,
            pointerId: null,
            startX: 0,
            startY: 0,
            startPanX: 0,
            startPanY: 0,
            moved: false,
            downTargetInfo: null
        };
        this._mhsaTokenMatrixTouchGesture = {
            pointers: new Map(),
            pinchActive: false,
            startDistance: 0,
            startScale: 1,
            anchorLocalX: 0,
            anchorLocalY: 0
        };
        this._mhsaTokenMatrixViewport = {
            panX: 0,
            panY: 0,
            scale: 1,
            minScale: 0.36,
            maxScale: 3.6,
            initialized: false,
            hasInteracted: false
        };
        this._mhsaTokenMatrixKeyboardMotion = {
            activeKeys: new Set(),
            rafId: null,
            lastTime: 0
        };
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
        this._applyAttentionCollapseState();

        this.closeBtn?.addEventListener('click', this._onCloseClick);
        this.copyContextBtn?.addEventListener('click', this._onCopyContextClick);
        this.fullscreenToggleBtn?.addEventListener('click', this._onFullscreenToggleClick);
        this.closeBtn?.addEventListener('pointerdown', this._onClosePointerDown);
        this.resizeHandle?.addEventListener('pointerdown', this._onResizeHandlePointerDown);
        this.resizeHandle?.addEventListener('keydown', this._onResizeHandleKeydown);
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
        this.attentionCollapseBtn?.addEventListener('click', this._onAttentionCollapseClick);
        if (this.attentionMatrix) {
            this.attentionMatrix.addEventListener('pointermove', this._onAttentionPointerMove);
            this.attentionMatrix.addEventListener('pointerdown', this._onAttentionPointerDown);
            this.attentionMatrix.addEventListener('pointerleave', this._clearAttentionHover);
        }
        if (this.mhsaTokenMatrixBody) {
            this.mhsaTokenMatrixBody.addEventListener('pointerover', this._onMhsaTokenMatrixPointerMove);
            this.mhsaTokenMatrixBody.addEventListener('pointermove', this._onMhsaTokenMatrixPointerMove);
            this.mhsaTokenMatrixBody.addEventListener('pointerdown', this._onMhsaTokenMatrixPointerDown);
            this.mhsaTokenMatrixBody.addEventListener('pointerup', this._onMhsaTokenMatrixPointerUp);
            this.mhsaTokenMatrixBody.addEventListener('pointercancel', this._onMhsaTokenMatrixPointerUp);
            this.mhsaTokenMatrixBody.addEventListener('pointerleave', this._clearMhsaTokenMatrixHover);
            this.mhsaTokenMatrixBody.addEventListener('wheel', this._onMhsaTokenMatrixWheel, { passive: false });
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
        document.addEventListener('keyup', this._onKeyup);
        document.addEventListener('pointerdown', this._onDocumentPointerDown, { capture: true });
        document.addEventListener('focusin', this._onDocumentFocusIn, { capture: true });
        window.addEventListener(SELECTION_PANEL_DEV_MODE_EVENT, this._onDevModeChanged);
        this._touchClickCleanup = initTouchClickFallback(this.panel, {
            selector: '.toggle-row, .detail-attention-collapse, .detail-token-nav-chip[data-token-nav="true"], .detail-history-btn, .detail-description-action-link, .detail-copy-context-btn, .detail-attention-score-link[data-attention-score-link="true"]'
        });
        this._observeResize();
        this._onResize();
        this._updateResizeHandleState();
        this._updateMhsaFullscreenToggle();
        if (typeof document !== 'undefined' && document.fonts?.ready) {
            document.fonts.ready.then(() => {
                this._applyCopyContextButtonLayout();
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
                this._applyCopyContextButtonLayout();
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

    _applyCopyContextButtonLayout() {
        const buttons = [this.transformerView2dActionBtn, this.copyContextBtn].filter(Boolean);
        if (buttons.length === 0) return;
        buttons.forEach((button) => {
            if (button.hidden) return;
            const rect = button.getBoundingClientRect();
            const widthPx = rect.width
                || button.clientWidth
                || button.offsetWidth
                || button.parentElement?.getBoundingClientRect?.().width
                || 0;
            const layout = resolveCopyContextButtonLayout(widthPx);
            button.style.setProperty('--detail-copy-context-font-size', `${layout.fontSizePx}px`);
            button.style.setProperty('--detail-copy-context-icon-size', `${layout.iconSizePx}px`);
            button.style.setProperty('--detail-copy-context-assistant-size', `${layout.assistantSizePx}px`);
            button.style.setProperty('--detail-copy-context-gap', `${layout.gapPx}px`);
            button.style.setProperty('--detail-copy-context-padding-inline', `${layout.paddingInlinePx}px`);
            button.style.setProperty('--detail-copy-context-padding-block', `${layout.paddingBlockPx}px`);
            button.style.setProperty('--detail-copy-context-radius', `${layout.borderRadiusPx}px`);
        });
    }

    _applyDimensionLabelFit() {
        fitSelectionDimensionLabels({
            inputDimLabel: this.inputDimLabel,
            outputDimLabel: this.outputDimLabel,
            biasDimLabel: this.biasDimLabel
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

    _onDevModeChanged() {
        if (!this.isOpen || !this._lastSelection) return;
        this.showSelection(this._lastSelection, { fromHistory: true });
    }

    _setTransformerView2dActionButtonState(actionConfig = null) {
        if (!this.transformerView2dActionBtn) return;
        const enabled = !!actionConfig;
        if (this.transformerView2dActionRow) {
            this.transformerView2dActionRow.hidden = !enabled;
            this.transformerView2dActionRow.setAttribute('aria-hidden', enabled ? 'false' : 'true');
        }
        this.transformerView2dActionBtn.hidden = !enabled;
        this.transformerView2dActionBtn.setAttribute('aria-hidden', enabled ? 'false' : 'true');
        if (!enabled) {
            this.transformerView2dActionBtn.dataset.detailAction = TRANSFORMER_VIEW2D_PANEL_ACTION_OPEN;
            if (this.transformerView2dActionBtnLabel) {
                this.transformerView2dActionBtnLabel.textContent = 'View in 2D / matrix form';
            }
            this.transformerView2dActionBtn.setAttribute('aria-label', 'View in 2D / matrix form');
            this.transformerView2dActionBtn.removeAttribute('title');
            return;
        }
        const label = String(actionConfig?.label || '').trim() || 'View in 2D / matrix form';
        const ariaLabel = String(actionConfig?.ariaLabel || '').trim() || label;
        const title = String(actionConfig?.title || '').trim() || ariaLabel;
        const action = String(actionConfig?.action || '').trim() || TRANSFORMER_VIEW2D_PANEL_ACTION_OPEN;
        this.transformerView2dActionBtn.dataset.detailAction = action;
        if (this.transformerView2dActionBtnLabel) {
            this.transformerView2dActionBtnLabel.textContent = label;
        }
        this.transformerView2dActionBtn.setAttribute('aria-label', ariaLabel);
        this.transformerView2dActionBtn.title = title;
        this._applyCopyContextButtonLayout();
    }

    _resolveSelectionPrimaryActionConfig({
        view2dContext = null
    } = {}) {
        return resolveSelectionPrimaryActionConfig({
            view2dContext,
            isSmallScreen: this._isSmallScreen()
        });
    }

    _setCopyContextButtonLabel(text) {
        const label = String(text || '').trim() || COPY_CONTEXT_BUTTON_DEFAULT_LABEL;
        const defaultLabel = this._copyContextDefaultLabel || COPY_CONTEXT_BUTTON_DEFAULT_LABEL;
        const isDefaultLabel = label === defaultLabel;
        if (this.copyContextBtnLabel) {
            this.copyContextBtnLabel.textContent = label;
        } else if (this.copyContextBtn) {
            this.copyContextBtn.textContent = label;
        }
        if (this.copyContextBtn) {
            this.copyContextBtn.dataset.copyContextLayout = isDefaultLabel ? 'default' : 'feedback';
        }
        this.copyContextBtnAssistant?.setAttribute('aria-hidden', 'true');
        this._applyCopyContextButtonLayout();
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
        const safeMode = resolveAttentionDisplayMode(mode);
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
        if (safeMode === 'mask') {
            return buildCausalMaskLegendGradient();
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
                this._buildAttentionLegendGradient(this._resolveActiveAttentionDisplayMode())
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

        const mode = this._resolveActiveAttentionDisplayMode();
        const value = mode === 'post'
            ? sampleT
            : (mode === 'mask'
                ? (sampleT < 0.5 ? CAUSAL_MASK_PREVIEW_BLOCKED_VALUE : 0)
                : THREE.MathUtils.lerp(-ATTENTION_PRE_COLOR_CLAMP, ATTENTION_PRE_COLOR_CLAMP, sampleT));
        return {
            ratio: safeRatio,
            value,
            valueLabel: mode === 'mask'
                ? formatCausalMaskPreviewScore({
                    maskValue: value <= CAUSAL_MASK_PREVIEW_BLOCKED_VALUE + 1e-6
                        ? Number.NEGATIVE_INFINITY
                        : 0,
                    fallback: ''
                })
                : this._formatLegendHoverValue(value, {
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

    _buildSelectionPromptContextSummary(selection = null, vectorTokenMetadata = null) {
        const promptTokenIndices = Array.isArray(this.attentionTokenIndices) && this.attentionTokenIndices.length
            ? this.attentionTokenIndices
            : this.laneTokenIndices;
        const promptTokenLabels = Array.isArray(this.attentionTokenLabels) && this.attentionTokenLabels.length
            ? this.attentionTokenLabels
            : this.tokenLabels;
        if (!Array.isArray(promptTokenIndices) || !promptTokenIndices.length) return '';
        if (!Array.isArray(promptTokenLabels) || !promptTokenLabels.length) return '';

        const tokenIndex = Number.isFinite(vectorTokenMetadata?.tokenIndex)
            ? Math.floor(vectorTokenMetadata.tokenIndex)
            : findUserDataNumber(selection, 'tokenIndex');
        const tokenId = Number.isFinite(vectorTokenMetadata?.tokenId)
            ? Math.floor(vectorTokenMetadata.tokenId)
            : findUserDataNumber(selection, 'tokenId');
        let tokenText = typeof vectorTokenMetadata?.tokenText === 'string'
            ? vectorTokenMetadata.tokenText
            : findUserDataString(selection, 'tokenLabel');
        if ((!tokenText || !tokenText.trim().length)
            && Number.isFinite(tokenIndex)
            && this.activationSource
            && typeof this.activationSource.getTokenString === 'function') {
            const resolvedTokenText = this.activationSource.getTokenString(tokenIndex);
            if (typeof resolvedTokenText === 'string' && resolvedTokenText.trim().length) {
                tokenText = resolvedTokenText;
            }
        }

        const { entries, activeIndex } = buildSelectionPromptContext({
            activationSource: this.activationSource,
            laneTokenIndices: promptTokenIndices,
            tokenLabels: promptTokenLabels,
            selectedTokenIndex: Number.isFinite(tokenIndex) ? Math.floor(tokenIndex) : null,
            selectedTokenId: Number.isFinite(tokenId) ? Math.floor(tokenId) : null,
            selectedTokenText: typeof tokenText === 'string' ? tokenText : ''
        });
        if (!entries.length || activeIndex < 0) return '';

        return entries
            .map((entry, index) => {
                const label = entry.titleText || entry.tokenLabel || entry.displayText || `Token ${index + 1}`;
                return index === activeIndex ? `[[${label}]]` : label;
            })
            .join(' | ');
    }

    _resolveCopyContextKvState(selection = null) {
        const layerIndex = findUserDataNumber(selection, 'layerIndex');
        const layers = Array.isArray(this.engine?._layers) ? this.engine._layers : [];
        const layer = Number.isFinite(layerIndex) && layerIndex >= 0 && layerIndex < layers.length
            ? layers[Math.floor(layerIndex)]
            : null;
        const mhsa = layer?.mhsaAnimation || null;
        const kvCacheModeEnabled = !!(
            appState.kvCacheModeEnabled
            || layer?._kvCacheModeEnabled
            || (typeof mhsa?._isKvCacheModeEnabled === 'function' && mhsa._isKvCacheModeEnabled())
        );

        return {
            kvCacheModeEnabled,
            kvCachePrefillActive: !!appState.kvCachePrefillActive,
            kvCachePassIndex: Number.isFinite(appState.kvCachePassIndex)
                ? Math.floor(appState.kvCachePassIndex)
                : null,
            kvCacheDecodeActive: !!(layer?._kvCacheDecodeActive || mhsa?._kvCacheDecodeActive),
            selectionIsCachedKv: isKvCacheVectorSelection(selection)
        };
    }

    _buildSelectionContextPayload() {
        const selection = this._lastSelection;
        const normalizedLabel = this._lastSelectionLabel
            || normalizeSelectionLabel(selection?.label || '', selection);
        const title = (this.title?.textContent || '').trim();
        const subtitle = (this.subtitle?.textContent || '').trim();
        const subtitleSecondary = (this.subtitleSecondary?.textContent || '').trim();
        const subtitleTertiary = (this.subtitleTertiary?.textContent || '').trim();
        const descriptionText = String(this._currentSelectionDescription || '').trim();
        const equationText = String(this._currentSelectionEquations || '').trim();
        const vectorTokenMetadata = selection
            ? this._resolveVectorTokenPosition(selection, normalizedLabel)
            : null;
        const attentionContext = selection ? this._resolveAttentionContext(selection) : null;
        const attentionScoreSummary = selection
            ? resolveAttentionScoreSelectionSummary(selection, attentionContext)
            : null;
        const promptContextSummary = this._buildSelectionPromptContextSummary(selection, vectorTokenMetadata);

        if (this._softmaxDetailOpen) {
            const softmaxContent = getSoftmaxCopyContextContent();
            return buildSelectionChatPrompt({
                selection,
                normalizedLabel,
                title,
                subtitle,
                subtitleSecondary,
                subtitleTertiary,
                panelContentsBlurb: softmaxContent.panelContentsBlurb,
                descriptionText: softmaxContent.descriptionText,
                equationText: softmaxContent.equationText,
                promptContextSummary,
                attentionScoreSummary,
                vectorTokenMetadata,
                activationSource: this.activationSource,
                kvState: this._resolveCopyContextKvState(selection)
            });
        }

        const metaLines = [
            ...collectVisibleContextText(this.previewMetaSection, {
                excludeSelectors: '#detailCopyContextBtn, #detailClose, #detailFullscreenToggle'
            }),
            ...collectVisibleContextText(this.metaSection, {
                excludeSelectors: '#detailCopyContextBtn, #detailClose, #detailFullscreenToggle'
            })
        ];

        let legendLines = [];
        if (this.vectorLegend?.classList.contains('is-visible')) {
            legendLines = collectVisibleContextText(this.vectorLegend, {
                excludeSelectors: '.legend-hover-tooltip, .legend-hover-marker'
            });
        }

        let attentionLines = [];
        if (this.attentionRoot?.classList.contains('is-visible')) {
            attentionLines = collectVisibleContextText(this.attentionRoot, {
                excludeSelectors: '.legend-hover-tooltip, .legend-hover-marker'
            });
        }

        let dataLines = [];
        if (this.dataSection && this.dataSection.style.display !== 'none') {
            dataLines = collectVisibleContextText(this.dataSection);
        }

        return buildSelectionChatPrompt({
            selection,
            normalizedLabel,
            title,
            subtitle,
            subtitleSecondary,
            subtitleTertiary,
            descriptionText,
            equationText,
            metaLines,
            legendLines,
            attentionLines,
            dataLines,
            promptContextSummary,
            attentionScoreSummary,
            vectorTokenMetadata,
            activationSource: this.activationSource,
            kvState: this._resolveCopyContextKvState(selection)
        });
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
        return readEquationBaseFontPx(this.equationsBody, 12);
    }

    _readSelectionEquationContentSize() {
        return readEquationContentSize(this.equationsBody);
    }

    _applySelectionEquationFit() {
        if (!this.isReady || !this.isOpen || !this.equationsSection || !this.equationsBody) return;
        if (!this.equationsSection.classList.contains('is-visible')) return;

        const bodyRect = this.equationsBody.getBoundingClientRect();
        const availableWidth = Math.max(0, bodyRect.width);
        if (!(availableWidth > 0)) return;
        const fitWidth = Math.max(0, availableWidth - DETAIL_EQUATION_FIT_BUFFER_PX);
        if (!(fitWidth > 0)) return;

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
                fits: fitted.width <= fitWidth + 0.5,
                size: fitted
            };
        };

        const maxFontPx = Math.max(
            DETAIL_EQUATION_FONT_MIN_PX,
            Math.min(
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
        this._syncDesktopPanelWidthToViewport();
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
            if (this._softmaxDetailOpen) {
                this._softmaxDetailView?.resizeAndRender();
            }
            if (this._transformerView2dDetailOpen) {
                this._transformerView2dDetailView?.resizeAndRender();
            }
            return;
        }
        if (this.currentPreview && this._previewPausedForPanelResize) {
            // Resizing the canvas clears the WebGL buffer, so redraw the frozen
            // preview without refitting until the drag finishes.
            this._renderPreviewSnapshot();
        } else if (this.currentPreview) {
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
                fitObjectToView(
                    this.currentPreview,
                    this.camera,
                    this._resolveCurrentPreviewFitOptions(width, height)
                );
                this._noteFit(width, height);
            }
        }
        this._updateMobileState();
        this._updateMhsaFullscreenToggle();
        this._syncSceneShift();
        this._applyCopyContextButtonLayout();
        this._scheduleSelectionEquationFit();
        this._scheduleDimensionLabelFit();
        this._refreshVisibleLegendGradients();
        this._refreshLegendHover();
        if (this._isMhsaInfoSelectionActive && this._mhsaTokenMatrixWorkspace && !this.mhsaTokenMatrixBody?.hidden) {
            requestAnimationFrame(() => {
                if (!this._isMhsaInfoSelectionActive || !this._mhsaTokenMatrixWorkspace || this.mhsaTokenMatrixBody?.hidden) {
                    return;
                }
                if (!this._mhsaTokenMatrixViewport?.hasInteracted) {
                    const centeredViewport = this._resolveMhsaTokenMatrixViewportCenter();
                    if (centeredViewport) {
                        this._mhsaTokenMatrixViewport.panX = centeredViewport.panX;
                        this._mhsaTokenMatrixViewport.panY = centeredViewport.panY;
                        this._mhsaTokenMatrixViewport.initialized = true;
                    }
                }
                this._applyMhsaTokenMatrixViewport();
            });
        }
        if (this._isMhsaInfoSelectionActive && !this.mhsaTokenMatrixPreview?.hidden) {
            this._scheduleMhsaTokenMatrixCanvasRender();
        }
        if (this._geluDetailOpen) {
            this._geluDetailView?.resizeAndRender();
        }
        if (this._softmaxDetailOpen) {
            this._softmaxDetailView?.resizeAndRender();
        }
        if (this._transformerView2dDetailOpen) {
            this._transformerView2dDetailView?.resizeAndRender();
        }
    }

    _finalizePendingReveal() {
        if (!this.isReady || !this._pendingReveal) return;
        this._pendingReveal = false;
        if (this.currentPreview) {
            fitObjectToView(
                this.currentPreview,
                this.camera,
                this._resolveCurrentPreviewFitOptions(
                    this._pendingRevealSize?.width,
                    this._pendingRevealSize?.height
                )
            );
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

    _resolvePreviewViewportSize(width = null, height = null) {
        let nextWidth = (Number.isFinite(width) && width > 0) ? Math.floor(width) : null;
        let nextHeight = (Number.isFinite(height) && height > 0) ? Math.floor(height) : null;
        if (nextWidth !== null && nextHeight !== null) {
            return { width: nextWidth, height: nextHeight };
        }

        const rect = this.canvas?.getBoundingClientRect?.();
        nextWidth = Math.max(
            1,
            Math.floor(rect?.width || this.canvas?.clientWidth || this.canvas?.width || 0)
        );
        nextHeight = Math.max(
            1,
            Math.floor(rect?.height || this.canvas?.clientHeight || this.canvas?.height || 0)
        );
        return { width: nextWidth, height: nextHeight };
    }

    _resolvePreviewFitViewportCap(width = null, height = null) {
        return {
            width: PREVIEW_MAX_FIT_VIEWPORT_WIDTH_PX,
            height: PREVIEW_MAX_FIT_VIEWPORT_HEIGHT_PX
        };
    }

    _resolveCurrentPreviewFitOptions(width = null, height = null) {
        const viewport = this._resolvePreviewViewportSize(width, height);
        const fitViewportCap = this._resolvePreviewFitViewportCap(viewport.width, viewport.height);
        return {
            ...(this._lastFitOptions || {}),
            viewportWidth: viewport.width,
            viewportHeight: viewport.height,
            maxViewportWidthPx: fitViewportCap.width,
            maxViewportHeightPx: fitViewportCap.height
        };
    }

    _cancelScheduledResize() {
        if (this._pendingResizeRaf) {
            cancelAnimationFrame(this._pendingResizeRaf);
            this._pendingResizeRaf = null;
        }
        if (this._pendingResizeTimeout) {
            clearTimeout(this._pendingResizeTimeout);
            this._pendingResizeTimeout = null;
        }
    }

    _scheduleResize() {
        if (!this.isReady) return;
        this._cancelScheduledResize();
        this._pendingResizeRaf = requestAnimationFrame(() => {
            this._pendingResizeRaf = null;
            this._onResize();
        });
        this._pendingResizeTimeout = setTimeout(() => {
            this._pendingResizeTimeout = null;
            this._onResize();
        }, 280);
    }

    _setPanelKeyboardEngaged(engaged) {
        const next = !!engaged;
        if (this._panelKeyboardEngaged === next) return;
        this._panelKeyboardEngaged = next;
        if (!next) {
            this._clearMhsaTokenMatrixKeyboardMotion();
        }
    }

    _panelOwnsKeyboard() {
        return shouldCaptureMhsaKeyboardInput({
            isOpen: this.isOpen,
            isMhsaInfoSelectionActive: this._isMhsaInfoSelectionActive,
            mhsaTokenMatrixHidden: !!this.mhsaTokenMatrixBody?.hidden,
            panel: this.panel,
            keyboardEngaged: this._panelKeyboardEngaged,
            activeElement: (typeof document !== 'undefined') ? document.activeElement : null
        });
    }

    _onKeydown(event) {
        const keyboardTarget = event?.target instanceof Element ? event.target : null;
        const controlKey = this._normalizeMhsaKeyboardControlKey(event?.key);
        if (
            this._panelOwnsKeyboard()
            && !this._isTextEntryTarget(keyboardTarget)
            && !event.ctrlKey
            && !event.metaKey
            && !event.altKey
        ) {
            if (controlKey) {
                this._mhsaTokenMatrixKeyboardMotion.activeKeys.add(controlKey);
                this._startMhsaTokenMatrixKeyboardMotion();
                event.preventDefault();
                return;
            }
        }

        if (event.key === 'Escape' && this.isOpen) {
            if (this._transformerView2dDetailOpen) {
                const didCloseTransformerView2dSelectionSidebar = this._closeTransformerView2dSelectionSidebar({
                    restoreSections: false,
                    restartLoop: false
                });
                if (didCloseTransformerView2dSelectionSidebar) {
                    event.preventDefault();
                    event.stopPropagation();
                    return;
                }
                const didClearTransformerView2dLock = this._transformerView2dDetailView?.clearSelectionLock?.({
                    scheduleRender: true
                }) === true;
                if (didClearTransformerView2dLock) {
                    event.preventDefault();
                    event.stopPropagation();
                    return;
                }
                this.close({ clearHistory: false });
                return;
            }
            if (this._mhsaFullscreenActive && this.fullscreenToggleBtn) {
                this._setMhsaFullscreen(false);
                return;
            }
            this.close({ clearHistory: false });
        }
    }

    _onKeyup(event) {
        const controlKey = this._normalizeMhsaKeyboardControlKey(event?.key);
        if (!controlKey || !this._mhsaTokenMatrixKeyboardMotion) return;
        this._mhsaTokenMatrixKeyboardMotion.activeKeys.delete(controlKey);
        if (!this._mhsaTokenMatrixKeyboardMotion.activeKeys.size) {
            this._clearMhsaTokenMatrixKeyboardMotion();
        }
    }

    _canToggleMhsaFullscreen() {
        return !!(
            this.isOpen
            && (
                this._transformerView2dDetailOpen
                || (this._isMhsaInfoSelectionActive && !this._isSmallScreen())
            )
        );
    }

    _syncMhsaViewRoute(active) {
        if (typeof window === 'undefined' || typeof window.history?.replaceState !== 'function') return;
        let nextUrl;
        try {
            nextUrl = new URL(window.location.href);
        } catch (_) {
            return;
        }
        const shouldActivate = active === true;
        const currentView = String(nextUrl.searchParams.get('view') || '').trim().toLowerCase();
        const currentHash = String(nextUrl.hash || '').replace(/^#/, '').trim().toLowerCase();
        if (shouldActivate) {
            if (currentView !== 'mhsa') {
                nextUrl.searchParams.set('view', 'mhsa');
            }
            if (currentHash === 'mhsa') {
                nextUrl.hash = '';
            }
        } else {
            if (currentView === 'mhsa') {
                nextUrl.searchParams.delete('view');
            }
            if (currentHash === 'mhsa') {
                nextUrl.hash = '';
            }
        }
        const nextHref = `${nextUrl.pathname}${nextUrl.search}${nextUrl.hash}`;
        const currentHref = `${window.location.pathname}${window.location.search}${window.location.hash}`;
        if (nextHref !== currentHref) {
            window.history.replaceState(window.history.state, '', nextHref);
        }
    }

    _syncMhsaFullscreenDocumentState() {
        if (typeof document === 'undefined' || !document.body) return;
        document.body.classList.toggle('mhsa-fullscreen-active', !!this._mhsaFullscreenActive);
    }

    _updateMhsaFullscreenToggle() {
        if (!this.fullscreenToggleBtn) return;
        const canToggle = this._canToggleMhsaFullscreen();
        const shouldShowToggle = canToggle && !this._transformerView2dDetailOpen;
        let clearedFullscreen = false;
        if (!canToggle && this._mhsaFullscreenActive) {
            this._mhsaFullscreenActive = false;
            this.panel?.classList.remove('is-mhsa-fullscreen');
            clearedFullscreen = true;
        }
        this._syncMhsaFullscreenDocumentState();
        this.fullscreenToggleBtn.hidden = !shouldShowToggle;
        this.fullscreenToggleBtn.setAttribute('aria-hidden', shouldShowToggle ? 'false' : 'true');
        this.fullscreenToggleBtn.dataset.fullscreen = this._mhsaFullscreenActive ? 'true' : 'false';
        this.fullscreenToggleBtn.textContent = this._mhsaFullscreenActive ? 'Exit' : 'Full';
        this.fullscreenToggleBtn.setAttribute(
            'aria-label',
            this._mhsaFullscreenActive ? 'Exit full screen' : 'Enter full screen'
        );
        this.fullscreenToggleBtn.title = this._mhsaFullscreenActive ? 'Exit full screen' : 'Enter full screen';
        if (clearedFullscreen) {
            this._updateResizeHandleState();
            this._syncSceneShift({ immediate: true });
            this._scheduleResize();
        }
    }

    _setMhsaFullscreen(enabled) {
        const next = !!enabled && this._canToggleMhsaFullscreen();
        if (this._mhsaFullscreenActive === next) {
            this._updateMhsaFullscreenToggle();
            return;
        }
        this._mhsaFullscreenActive = next;
        this.panel?.classList.toggle('is-mhsa-fullscreen', next);
        this._syncMhsaFullscreenDocumentState();
        this._updateMhsaFullscreenToggle();
        this._updateResizeHandleState();
        this._syncSceneShift({ immediate: true });
        this._scheduleResize();
        this._applyCopyContextButtonLayout();
        this._scheduleSelectionEquationFit();
        this._scheduleDimensionLabelFit();
    }

    _onFullscreenToggleClick(event) {
        if (event?.cancelable) event.preventDefault();
        event?.stopPropagation?.();
        if (!this._canToggleMhsaFullscreen()) return;
        this._setMhsaFullscreen(!this._mhsaFullscreenActive);
    }

    _handleCloseButtonAction(event = null) {
        if (!this.isOpen) return false;
        if (this._transformerView2dDetailOpen) {
            const didCloseTransformerView2dSelectionSidebar = this._closeTransformerView2dSelectionSidebar({
                restoreSections: false,
                restartLoop: false
            });
            if (didCloseTransformerView2dSelectionSidebar) {
                if (event?.cancelable) event.preventDefault();
                event?.stopPropagation?.();
                return true;
            }
            const didClearTransformerView2dLock = this._transformerView2dDetailView?.clearSelectionLock?.({
                scheduleRender: true
            }) === true;
            if (didClearTransformerView2dLock) {
                if (event?.cancelable) event.preventDefault();
                event?.stopPropagation?.();
                return true;
            }
            if (event?.cancelable) event.preventDefault();
            event?.stopPropagation?.();
            this.close({ clearHistory: false });
            return true;
        }
        if (this._mhsaFullscreenActive && this.fullscreenToggleBtn) {
            if (event?.cancelable) event.preventDefault();
            event?.stopPropagation?.();
            this._setMhsaFullscreen(false);
            return true;
        }
        if (event?.cancelable) event.preventDefault();
        event?.stopPropagation?.();
        this.close({ clearHistory: false });
        return true;
    }

    _onCloseClick(event) {
        if (this._suppressNextCloseClick) {
            this._suppressNextCloseClick = false;
            if (event?.cancelable) event.preventDefault();
            event?.stopPropagation?.();
            return;
        }
        this._handleCloseButtonAction(event);
    }

    _onClosePointerDown(event) {
        if (!this.isOpen) return;
        if (event?.pointerType === 'mouse' && Number.isFinite(event.button) && event.button !== 0) return;
        this._suppressNextCloseClick = true;
        this._handleCloseButtonAction(event);
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

    _isLandscapeViewport() {
        if (typeof window === 'undefined') return true;
        if (typeof window.matchMedia === 'function') {
            return window.matchMedia('(orientation: landscape)').matches;
        }
        return window.innerWidth >= window.innerHeight;
    }

    _canResizeDesktopPanel() {
        return !this._mhsaFullscreenActive && !this._isSmallScreen() && this._isLandscapeViewport();
    }

    _resolveDesktopPanelWidthBounds() {
        if (typeof window === 'undefined') {
            return resolveDesktopSelectionPanelWidthBounds();
        }
        const viewportWidth = window.innerWidth
            || document.documentElement?.clientWidth
            || this.hudStack?.getBoundingClientRect?.().width
            || 0;
        return resolveDesktopSelectionPanelWidthBounds({ viewportWidth });
    }

    _getCurrentDesktopPanelWidthPx() {
        const rectWidth = this.hudStack?.getBoundingClientRect?.().width;
        if (Number.isFinite(rectWidth) && rectWidth > 0) {
            return rectWidth;
        }
        return this._desktopPanelWidthPx;
    }

    _updateResizeHandleState() {
        const canResize = !!(this.resizeHandle && this.isOpen && this._canResizeDesktopPanel());
        this.hudStack?.classList.toggle('is-resizable', canResize);
        if (!canResize && this._panelResizeDrag.active) {
            this._cancelPanelResizeDrag({ finalizePreview: this.isOpen });
        }
        if (!this.resizeHandle) return;

        const currentWidth = clampDesktopSelectionPanelWidth(
            this._getCurrentDesktopPanelWidthPx(),
            this._resolveDesktopPanelWidthBounds()
        );
        const { minWidthPx, maxWidthPx } = this._resolveDesktopPanelWidthBounds();
        this.resizeHandle.setAttribute('aria-hidden', canResize ? 'false' : 'true');
        this.resizeHandle.setAttribute('aria-valuemin', String(Math.round(minWidthPx)));
        this.resizeHandle.setAttribute('aria-valuemax', String(Math.round(maxWidthPx)));
        this.resizeHandle.setAttribute('aria-valuenow', String(Math.round(currentWidth)));
        this.resizeHandle.tabIndex = canResize ? 0 : -1;
    }

    _setDesktopPanelWidth(widthPx, {
        syncSceneShift = false,
        immediateSceneShift = false,
        scheduleResize = true
    } = {}) {
        if (!this._rootStyleTarget || !this._canResizeDesktopPanel()) {
            this._updateResizeHandleState();
            return 0;
        }

        const nextWidth = clampDesktopSelectionPanelWidth(
            widthPx,
            this._resolveDesktopPanelWidthBounds()
        );
        this._desktopPanelWidthPx = nextWidth;
        this._rootStyleTarget.style.setProperty('--hud-stack-desktop-width', `${nextWidth}px`);
        this._updateResizeHandleState();
        if (syncSceneShift) {
            this._syncSceneShift({ immediate: immediateSceneShift });
        }
        if (scheduleResize) {
            this._scheduleResize();
        }
        return nextWidth;
    }

    _syncDesktopPanelWidthToViewport() {
        if (this._desktopPanelWidthPx === null || !this._canResizeDesktopPanel()) {
            this._updateResizeHandleState();
            return;
        }
        this._setDesktopPanelWidth(this._desktopPanelWidthPx, {
            syncSceneShift: false,
            scheduleResize: false
        });
    }

    _cancelPanelResizeDrag({ finalizePreview = false } = {}) {
        const pointerId = this._panelResizeDrag.pointerId;
        if (Number.isFinite(pointerId) && typeof this.resizeHandle?.releasePointerCapture === 'function') {
            try {
                this.resizeHandle.releasePointerCapture(pointerId);
            } catch (_) { /* no-op */ }
        }
        this._panelResizeDrag.active = false;
        this._panelResizeDrag.pointerId = null;
        this.hudStack?.classList.remove('is-resizing');
        if (typeof document !== 'undefined' && document.body) {
            document.body.classList.remove('detail-panel-resizing');
        }
        if (typeof window !== 'undefined') {
            window.removeEventListener('pointermove', this._onResizeHandlePointerMove);
            window.removeEventListener('pointerup', this._onResizeHandlePointerUp);
            window.removeEventListener('pointercancel', this._onResizeHandlePointerUp);
        }
        this._previewPausedForPanelResize = false;
        if (!finalizePreview) return;

        this._cancelScheduledResize();
        if (this._pendingRevealTimer) {
            clearTimeout(this._pendingRevealTimer);
            this._pendingRevealTimer = null;
        }
        this._onResize();
        if (this._pendingReveal) {
            this._finalizePendingReveal();
        }
        const now = performance.now();
        this._lastFrameTime = now;
        this._renderPreviewFrame(now);
        this._startLoop();
    }

    _onResizeHandlePointerDown(event) {
        if (!this.isOpen || !this._canResizeDesktopPanel()) return;
        if (!Number.isFinite(event?.clientX)) return;
        if (event?.pointerType === 'mouse' && Number.isFinite(event.button) && event.button !== 0) return;

        const touchLike = isTouchLikePointerEvent(event);
        if (event.cancelable) event.preventDefault();
        event.stopPropagation();

        const startWidthPx = this._getCurrentDesktopPanelWidthPx();
        if (!(Number.isFinite(startWidthPx) && startWidthPx > 0)) return;

        if (this.engine && typeof this.engine.resetInteractionState === 'function') {
            this.engine.resetInteractionState();
        }
        this._panelResizeDrag.active = true;
        this._panelResizeDrag.pointerId = Number.isFinite(event.pointerId) ? event.pointerId : null;
        this._panelResizeDrag.startX = event.clientX;
        this._panelResizeDrag.startWidthPx = startWidthPx;
        this.hudStack?.classList.add('is-resizing');
        if (typeof document !== 'undefined' && document.body) {
            if (touchLike) {
                document.body.classList.add('touch-ui');
            }
            document.body.classList.add('detail-panel-resizing');
        }
        if (Number.isFinite(event.pointerId) && typeof this.resizeHandle?.setPointerCapture === 'function') {
            try {
                this.resizeHandle.setPointerCapture(event.pointerId);
            } catch (_) { /* no-op */ }
        }
        if (typeof window !== 'undefined') {
            window.addEventListener('pointermove', this._onResizeHandlePointerMove, { passive: false });
            window.addEventListener('pointerup', this._onResizeHandlePointerUp);
            window.addEventListener('pointercancel', this._onResizeHandlePointerUp);
        }
        this._setHoverLabelSuppression(true);
        this._previewPausedForPanelResize = true;
        this._stopLoop();
    }

    _onResizeHandlePointerMove(event) {
        if (!this._panelResizeDrag.active) return;
        if (Number.isFinite(this._panelResizeDrag.pointerId) && event?.pointerId !== this._panelResizeDrag.pointerId) {
            return;
        }
        if (!Number.isFinite(event?.clientX)) return;

        if (event.cancelable) event.preventDefault();
        const deltaX = this._panelResizeDrag.startX - event.clientX;
        const nextWidth = this._panelResizeDrag.startWidthPx + deltaX;
        this._setDesktopPanelWidth(nextWidth, {
            syncSceneShift: true,
            immediateSceneShift: true
        });
    }

    _onResizeHandlePointerUp(event) {
        if (!this._panelResizeDrag.active) return;
        if (Number.isFinite(this._panelResizeDrag.pointerId) && event?.pointerId !== this._panelResizeDrag.pointerId) {
            return;
        }
        if (event?.cancelable) event.preventDefault();
        event?.stopPropagation?.();
        this._cancelPanelResizeDrag({ finalizePreview: true });
        this._syncHoverLabelSuppressionFromHoverState();
        this._syncSceneShift({ immediate: true });
    }

    _onResizeHandleKeydown(event) {
        if (!this.isOpen || !this._canResizeDesktopPanel()) return;
        const { minWidthPx, maxWidthPx } = this._resolveDesktopPanelWidthBounds();
        const currentWidth = clampDesktopSelectionPanelWidth(
            this._getCurrentDesktopPanelWidthPx(),
            this._resolveDesktopPanelWidthBounds()
        );
        const step = event?.shiftKey ? 48 : 24;
        let nextWidth = null;
        if (event.key === 'ArrowLeft') {
            nextWidth = currentWidth + step;
        } else if (event.key === 'ArrowRight') {
            nextWidth = currentWidth - step;
        } else if (event.key === 'Home') {
            nextWidth = minWidthPx;
        } else if (event.key === 'End') {
            nextWidth = maxWidthPx;
        }
        if (!Number.isFinite(nextWidth)) return;

        event.preventDefault();
        event.stopPropagation();
        this._setDesktopPanelWidth(nextWidth, {
            syncSceneShift: true,
            immediateSceneShift: true
        });
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
        this._setTransformerView2dActionButtonState(this._resolveSelectionPrimaryActionConfig({
            view2dContext: this._currentTransformerView2dContext
        }));
    }

    _syncSceneShift({ immediate = false } = {}) {
        if (!this.pipeline || typeof this.pipeline.setScreenShiftPixels !== 'function') return;
        if (typeof window === 'undefined') return;

        let nextShift = 0;
        const shouldShift = this.isOpen && !this._isSmallScreen() && !this._mhsaFullscreenActive;
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

        const historyNavigation = createSelectionPanelHistoryNavigation(document);
        if (!historyNavigation?.nav) return;

        header.insertBefore(historyNavigation.nav, titleGroup);
        this._updateHistoryNavigationControls();
    }

    _buildHistoryEntry(type, selection = null, {
        selectionSurface = null
    } = {}) {
        if (!selection || !selection.label) return null;
        const label = normalizeSelectionLabel(selection.label, selection);
        const keyBase = buildSelectionPreviewKey(label, selection);
        const resolvedSelectionSurface = type === 'transformer-view2d'
            ? SELECTION_PANEL_SURFACES.TRANSFORMER_VIEW2D
            : resolveSelectionPanelSurface({
                selectionSurface,
                transformerView2dDetailOpen: this._transformerView2dDetailOpen
            });
        return {
            type,
            selection,
            label,
            selectionSurface: resolvedSelectionSurface,
            key: `${resolvedSelectionSurface}:${type}:${keyBase}`
        };
    }

    _historyEntriesEqual(a, b) {
        return !!a && !!b && a.type === b.type && a.key === b.key;
    }

    _getHistoryNavigationButtons(action = '') {
        if (!this.panel || typeof this.panel.querySelectorAll !== 'function') return [];
        const normalizedAction = String(action || '').trim();
        if (!normalizedAction) return [];
        return Array.from(
            this.panel.querySelectorAll(`[data-detail-action="${normalizedAction}"]`)
        );
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
        this._getHistoryNavigationButtons(PANEL_ACTION_HISTORY_BACK).forEach((button) => {
            button.disabled = !hasBack;
            button.dataset.disabled = hasBack ? 'false' : 'true';
        });
        this._getHistoryNavigationButtons(PANEL_ACTION_HISTORY_FORWARD).forEach((button) => {
            button.disabled = !hasForward;
            button.dataset.disabled = hasForward ? 'false' : 'true';
        });
    }

    _buildHistorySelectionOptions(entry = null) {
        const selectionSurface = normalizeSelectionPanelSurface(
            entry?.selectionSurface,
            SELECTION_PANEL_SURFACES.PANEL
        );
        return {
            fromHistory: true,
            selectionSurface,
            preserveTransformerView2d: selectionSurface === SELECTION_PANEL_SURFACES.TRANSFORMER_VIEW2D
        };
    }

    _applyHistoryEntry(entry) {
        if (!entry || !entry.selection) return false;
        const historySelectionOptions = this._buildHistorySelectionOptions(entry);
        if (entry.type === 'gelu') {
            this.showSelection(entry.selection, historySelectionOptions);
            this._openGeluDetailPreview({
                fromHistory: true,
                sourceSelection: entry.selection
            });
            return true;
        }
        if (entry.type === 'softmax') {
            this.showSelection(entry.selection, historySelectionOptions);
            this._openSoftmaxDetailPreview({
                fromHistory: true,
                sourceSelection: entry.selection
            });
            return true;
        }
        if (entry.type === 'transformer-view2d') {
            this.showSelection(entry.selection, historySelectionOptions);
            this._openTransformerView2dPreview({
                fromHistory: true,
                sourceSelection: entry.selection
            });
            return true;
        }
        if (entry.type === 'selection') {
            this.showSelection(entry.selection, historySelectionOptions);
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

        this.title.classList.add('detail-title--token-context');
        const fragment = document.createDocumentFragment();

        const prefix = document.createElement('span');
        prefix.className = 'detail-title-text-part';
        prefix.textContent = safePrefixText;

        const chip = document.createElement('span');
        chip.className = 'detail-subtitle-token-chip detail-title-token-chip';
        applyTokenChipColors(chip, {
            tokenLabel: safeTokenText,
            tokenIndex: safeTokenIndex,
            tokenId: safeTokenId
        }, Number.isFinite(safeTokenIndex) ? safeTokenIndex : 0);
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

    _setSubtitleTertiaryText(text = '') {
        if (!this.subtitleTertiary) return;
        this.subtitleTertiary.textContent = String(text || '');
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

        this._setSubtitleSecondaryText('');
        this.subtitleSecondary.classList.add('detail-subtitle--token-context');

        const fragment = document.createDocumentFragment();
        const prefix = document.createElement('span');
        prefix.className = 'detail-subtitle-context-label';
        prefix.textContent = `${prefixText} `;

        const chip = document.createElement('span');
        chip.className = 'detail-subtitle-token-chip detail-subtitle-secondary-token-chip';
        applyTokenChipColors(chip, {
            tokenLabel: safeTokenText,
            tokenIndex: safeTokenIndex,
            tokenId: safeTokenId
        }, Number.isFinite(safeTokenIndex) ? safeTokenIndex : 0);
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

    _setPanelTokenHoverEntries(entries = null, { emit = true } = {}) {
        const normalizedEntries = normalizeTokenChipEntries(entries);
        const unchanged = tokenChipEntryListsMatch(this._panelTokenHoverEntries, normalizedEntries);
        if (!unchanged) {
            this._panelTokenHoverEntries = normalizedEntries;
            this._panelTokenHoverEntry = normalizedEntries[0] || null;
            if (emit) {
                dispatchTokenChipHoverSync(normalizedEntries, {
                    active: normalizedEntries.length > 0,
                    source: this._tokenHoverSyncSource
                });
            }
        }
        this._applyTokenChipHoverState();
    }

    _setPanelTokenHoverEntry(entry = null, { emit = true } = {}) {
        this._setPanelTokenHoverEntries(entry, { emit });
    }

    _applyTokenChipHoverState() {
        if (!this.panel) return;
        const chips = this.panel.querySelectorAll('.detail-token-nav-chip');
        chips.forEach((chip) => {
            const canNavigate = chip.dataset.tokenNav === 'true';
            const chipEntry = canNavigate ? this._extractPanelTokenChipEntry(chip) : null;
            const isActive = !!chipEntry && (
                this._panelTokenHoverEntries.some((candidate) => (
                    tokenChipEntriesMatch(chipEntry, candidate)
                ))
                || this._mirroredTokenHoverEntries.some((candidate) => (
                    tokenChipEntriesMatch(chipEntry, candidate)
                ))
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
        if (!matchesFocusVisibleTarget(chip)) return;
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
        this._mirroredTokenHoverEntries = detail.active ? normalizeTokenChipEntries(detail) : [];
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
            applyTokenChipColors(chip, {
                tokenLabel: safeTokenText,
                tokenIndex,
                tokenId
            }, Number.isFinite(tokenIndex) ? Math.floor(tokenIndex) : 0);
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

    _findPositionChipSceneObject({
        tokenIndex = null,
        tokenId = null,
        tokenText = '',
        positionIndex = null
    } = {}) {
        const scene = this.engine?.scene;
        if (!scene || typeof scene.traverse !== 'function') return null;

        const normalizedTokenText = formatTokenLabelForPreview(tokenText);
        const safePositionIndex = Number.isFinite(positionIndex) ? Math.max(0, Math.floor(positionIndex)) : null;
        let fallbackMatch = null;
        let visibleMatch = null;

        scene.traverse((node) => {
            if (!node || !node.userData || node.isScene) return;
            const label = typeof node.userData.label === 'string'
                ? node.userData.label
                : '';
            if (!label.toLowerCase().startsWith('position:')) return;

            const nodeTokenIndex = Number.isFinite(node.userData.tokenIndex)
                ? Math.floor(node.userData.tokenIndex)
                : null;
            const nodeTokenId = Number.isFinite(node.userData.tokenId)
                ? Math.floor(node.userData.tokenId)
                : null;
            const nodeTokenText = formatTokenLabelForPreview(node.userData.tokenLabel);
            const nodePositionIndex = Number.isFinite(node.userData.positionIndex)
                ? Math.floor(node.userData.positionIndex)
                : (() => {
                    const parsed = Number(extractTokenText(label));
                    return Number.isFinite(parsed) && parsed > 0 ? Math.floor(parsed) : null;
                })();

            if (Number.isFinite(tokenIndex) && Number.isFinite(nodeTokenIndex) && nodeTokenIndex !== tokenIndex) return;
            if (Number.isFinite(tokenId) && Number.isFinite(nodeTokenId) && nodeTokenId !== tokenId) return;
            if (Number.isFinite(safePositionIndex) && Number.isFinite(nodePositionIndex) && nodePositionIndex !== safePositionIndex) {
                return;
            }
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

    _parseDetailActionPayload(actionEl) {
        const rawPayload = actionEl?.dataset?.detailPayload;
        if (typeof rawPayload !== 'string' || !rawPayload.length) return {};
        try {
            const parsed = JSON.parse(decodeURIComponent(rawPayload));
            return parsed && typeof parsed === 'object' ? parsed : {};
        } catch (_) {
            return {};
        }
    }

    _buildRuntimeVectorSelection({
        vectorRef = null,
        label = 'Vector',
        kind = 'vector',
        activationData = null,
        layerIndex = null,
        headIndex = null,
        tokenIndex = null,
        tokenId = null,
        tokenLabel = ''
    } = {}) {
        if (!vectorRef) return null;
        const resolvedActivationData = activationData && typeof activationData === 'object'
            ? activationData
            : (vectorRef.userData?.activationData || null);
        const normalizedLayerIndex = Number.isFinite(layerIndex) ? Math.floor(layerIndex) : null;
        const normalizedHeadIndex = Number.isFinite(headIndex) ? Math.floor(headIndex) : null;
        const normalizedTokenIndex = Number.isFinite(tokenIndex) ? Math.floor(tokenIndex) : null;
        const normalizedTokenId = Number.isFinite(tokenId) ? Math.floor(tokenId) : null;
        const normalizedTokenLabel = typeof tokenLabel === 'string' && tokenLabel.trim().length
            ? formatTokenLabelForPreview(tokenLabel)
            : '';
        const resolvedLabel = String(
            label
            || activationData?.label
            || vectorRef.group?.userData?.label
            || vectorRef.mesh?.userData?.label
            || 'Vector'
        ).trim() || 'Vector';
        const info = {
            vectorRef
        };
        if (resolvedActivationData && typeof resolvedActivationData === 'object') {
            info.activationData = resolvedActivationData;
        }
        if (Number.isFinite(normalizedLayerIndex)) info.layerIndex = normalizedLayerIndex;
        if (Number.isFinite(normalizedHeadIndex)) info.headIndex = normalizedHeadIndex;
        if (Number.isFinite(normalizedTokenIndex)) info.tokenIndex = normalizedTokenIndex;
        if (Number.isFinite(normalizedTokenId)) info.tokenId = normalizedTokenId;
        if (normalizedTokenLabel) info.tokenLabel = normalizedTokenLabel;

        const hasMesh = vectorRef.mesh?.isInstancedMesh === true;
        const object = hasMesh
            ? vectorRef.mesh
            : (vectorRef.group || vectorRef.mesh || null);
        const batchPrismCount = vectorRef.isBatchedVectorRef && Number.isFinite(vectorRef?._batch?.prismCount)
            ? Math.max(1, Math.floor(vectorRef._batch.prismCount))
            : null;
        const sharedPrismCount = Number.isFinite(vectorRef.instanceCount)
            ? Math.max(1, Math.floor(vectorRef.instanceCount))
            : null;
        const instanceId = hasMesh
            ? (
                vectorRef.isBatchedVectorRef && Number.isFinite(vectorRef._index) && batchPrismCount
                    ? Math.max(0, Math.floor(vectorRef._index) * batchPrismCount)
                    : 0
            )
            : null;
        if (Number.isFinite(vectorRef?._index)) info.vectorIndex = Math.max(0, Math.floor(vectorRef._index));
        if (Number.isFinite(sharedPrismCount)) info.prismIndex = 0;
        if (Number.isFinite(sharedPrismCount)) info.prismCount = sharedPrismCount;

        return {
            label: resolvedLabel,
            kind: hasMesh
                ? (object?.userData?.instanceKind || kind || 'instanced')
                : (kind || 'vector'),
            info,
            ...(object ? { object } : {}),
            ...(Number.isFinite(instanceId) && object ? {
                hit: {
                    object,
                    instanceId
                }
            } : {})
        };
    }

    _findLayerLaneByToken({
        layerIndex = null,
        tokenIndex = null,
        tokenId = null,
        tokenLabel = ''
    } = {}) {
        const layers = Array.isArray(this.engine?._layers) ? this.engine._layers : [];
        const resolvedLayerIndex = Number.isFinite(layerIndex) ? Math.floor(layerIndex) : null;
        const layer = Number.isFinite(resolvedLayerIndex)
            && resolvedLayerIndex >= 0
            && resolvedLayerIndex < layers.length
            ? layers[resolvedLayerIndex]
            : null;
        if (!layer || !Array.isArray(layer.lanes) || !layer.lanes.length) {
            return {
                layer,
                lane: null
            };
        }

        const normalizedTokenIndex = Number.isFinite(tokenIndex) ? Math.floor(tokenIndex) : null;
        const normalizedTokenId = Number.isFinite(tokenId) ? Math.floor(tokenId) : null;
        const normalizedTokenLabel = typeof tokenLabel === 'string' && tokenLabel.trim().length
            ? formatTokenLabelForPreview(tokenLabel)
            : '';

        let lane = null;
        if (Number.isFinite(normalizedTokenIndex)) {
            lane = layer.lanes.find((candidate) => (
                candidate
                && Number.isFinite(candidate.tokenIndex)
                && Math.floor(candidate.tokenIndex) === normalizedTokenIndex
            )) || null;
        }
        if (!lane && Number.isFinite(normalizedTokenId) && this.activationSource && typeof this.activationSource.getTokenId === 'function') {
            lane = layer.lanes.find((candidate) => {
                if (!candidate || !Number.isFinite(candidate.tokenIndex)) return false;
                const candidateTokenId = this.activationSource.getTokenId(Math.floor(candidate.tokenIndex));
                return Number.isFinite(candidateTokenId) && Math.floor(candidateTokenId) === normalizedTokenId;
            }) || null;
        }
        if (!lane && normalizedTokenLabel) {
            lane = layer.lanes.find((candidate) => (
                candidate
                && typeof candidate.tokenLabel === 'string'
                && formatTokenLabelForPreview(candidate.tokenLabel) === normalizedTokenLabel
            )) || null;
        }

        return {
            layer,
            lane
        };
    }

    _findRuntimeAttentionVectorSelection({
        vectorKind = 'Q',
        layerIndex = null,
        tokenIndex = null,
        tokenId = null,
        tokenLabel = '',
        headIndex = null
    } = {}) {
        const safeKind = normalizeQkvActionKind(vectorKind);
        const resolvedLayerIndex = Number.isFinite(layerIndex) ? Math.floor(layerIndex) : null;
        const resolvedHeadIndex = Number.isFinite(headIndex) ? Math.floor(headIndex) : null;
        if (!Number.isFinite(resolvedLayerIndex) || !Number.isFinite(resolvedHeadIndex)) return null;

        const { layer, lane } = this._findLayerLaneByToken({
            layerIndex: resolvedLayerIndex,
            tokenIndex,
            tokenId,
            tokenLabel
        });
        if (!layer || !lane) return null;

        let vectorRef = null;
        if (safeKind === 'Q' || safeKind === 'V') {
            const sideCopyEntry = getSideCopyEntry(lane, resolvedHeadIndex, safeKind);
            vectorRef = sideCopyEntry?.vec || null;
        } else if (safeKind === 'K') {
            vectorRef = Array.isArray(lane.upwardCopies)
                ? lane.upwardCopies[resolvedHeadIndex] || null
                : null;
            if (!vectorRef) {
                const mhsa = layer?.mhsaAnimation || null;
                const normalizedTokenIndex = Number.isFinite(tokenIndex) ? Math.floor(tokenIndex) : null;
                vectorRef = Array.isArray(mhsa?._tempKOutputVectors)
                    ? mhsa._tempKOutputVectors.find((candidate) => (
                        candidate
                        && Number.isFinite(candidate.userData?.headIndex)
                        && Math.floor(candidate.userData.headIndex) === resolvedHeadIndex
                        && (
                            !Number.isFinite(normalizedTokenIndex)
                            || (
                                Number.isFinite(candidate.userData?.parentLane?.tokenIndex)
                                && Math.floor(candidate.userData.parentLane.tokenIndex) === normalizedTokenIndex
                            )
                        )
                    )) || null
                    : null;
            }
        }
        if (!vectorRef) return null;

        const label = safeKind === 'Q'
            ? 'Query Vector'
            : (safeKind === 'V' ? 'Value Vector' : 'Key Vector');
        return this._buildRuntimeVectorSelection({
            vectorRef,
            label,
            kind: 'vector',
            layerIndex: resolvedLayerIndex,
            headIndex: resolvedHeadIndex,
            tokenIndex,
            tokenId,
            tokenLabel
        });
    }

    _findRuntimeQkvSourceVectorSelection({
        layerIndex = null,
        headIndex = null,
        tokenIndex = null,
        tokenId = null,
        tokenLabel = ''
    } = {}) {
        const resolvedLayerIndex = Number.isFinite(layerIndex) ? Math.floor(layerIndex) : null;
        const resolvedHeadIndex = Number.isFinite(headIndex) ? Math.floor(headIndex) : null;
        if (!Number.isFinite(resolvedLayerIndex) || !Number.isFinite(resolvedHeadIndex)) return null;

        const { lane } = this._findLayerLaneByToken({
            layerIndex: resolvedLayerIndex,
            tokenIndex,
            tokenId,
            tokenLabel
        });
        if (!lane) return null;

        const qSideCopy = getSideCopyEntry(lane, resolvedHeadIndex, 'Q')?.vec || null;
        const kCopy = Array.isArray(lane.upwardCopies)
            ? lane.upwardCopies[resolvedHeadIndex] || null
            : null;
        const vSideCopy = getSideCopyEntry(lane, resolvedHeadIndex, 'V')?.vec || null;
        const vectorRef = qSideCopy || kCopy || vSideCopy;
        if (!vectorRef) return null;
        const outputStage = normalizePostLayerNormResidualStage('ln1.output') || 'ln1.output';
        const sourceStage = normalizePostLayerNormResidualStage(outputStage, { preferLegacy: true });
        const resolvedLabel = resolvePostLayerNormResidualLabel({ stage: outputStage });

        return this._buildRuntimeVectorSelection({
            vectorRef,
            label: resolvedLabel,
            kind: 'vector',
            activationData: {
                label: resolvedLabel,
                stage: outputStage,
                sourceStage,
                layerIndex: resolvedLayerIndex,
                ...(Number.isFinite(resolvedHeadIndex) ? { headIndex: resolvedHeadIndex } : {}),
                ...(Number.isFinite(tokenIndex) ? { tokenIndex: Math.floor(tokenIndex) } : {}),
                ...(Number.isFinite(tokenId) ? { tokenId: Math.floor(tokenId) } : {}),
                ...(typeof tokenLabel === 'string' && tokenLabel.trim().length
                    ? { tokenLabel: formatTokenLabelForPreview(tokenLabel) }
                    : {})
            },
            layerIndex: resolvedLayerIndex,
            headIndex: resolvedHeadIndex,
            tokenIndex,
            tokenId,
            tokenLabel
        });
    }

    _findRuntimeResidualVectorSelection({
        stage = 'ln1.output',
        layerIndex = null,
        tokenIndex = null,
        tokenId = null,
        tokenLabel = ''
    } = {}) {
        const resolvedLayerIndex = Number.isFinite(layerIndex) ? Math.floor(layerIndex) : null;
        if (!Number.isFinite(resolvedLayerIndex)) return null;

        const { lane } = this._findLayerLaneByToken({
            layerIndex: resolvedLayerIndex,
            tokenIndex,
            tokenId,
            tokenLabel
        });
        if (!lane) return null;

        const normalizedStage = normalizePostLayerNormResidualStage(stage) || 'ln1.output';
        const sourceStage = normalizePostLayerNormResidualStage(normalizedStage, { preferLegacy: true });
        let vectorRef = null;
        if (normalizedStage === 'ln2.output') {
            vectorRef = lane.resultVecLN2 || lane.movingVecLN2 || lane.finalVecAfterMlp || null;
        } else if (normalizedStage === 'ln1.output') {
            vectorRef = lane.resultVec || lane.postAdditionVec || lane.originalVec || null;
        }
        if (!vectorRef) return null;

        const resolvedLabel = resolvePostLayerNormResidualLabel({ stage: normalizedStage || 'ln1.output' });
        const activationData = {
            label: resolvedLabel,
            stage: normalizedStage || 'ln1.output',
            sourceStage,
            layerIndex: resolvedLayerIndex,
            ...(Number.isFinite(tokenIndex) ? { tokenIndex: Math.floor(tokenIndex) } : {}),
            ...(Number.isFinite(tokenId) ? { tokenId: Math.floor(tokenId) } : {}),
            ...(typeof tokenLabel === 'string' && tokenLabel.trim().length
                ? { tokenLabel: formatTokenLabelForPreview(tokenLabel) }
                : {})
        };

        return this._buildRuntimeVectorSelection({
            vectorRef,
            label: resolvedLabel,
            kind: 'vector',
            activationData,
            layerIndex: resolvedLayerIndex,
            tokenIndex,
            tokenId,
            tokenLabel
        });
    }

    _findRuntimeMlpMiddleVectorSelection({
        stage = 'mlp.up',
        label = '',
        layerIndex = null,
        tokenIndex = null,
        tokenId = null,
        tokenLabel = ''
    } = {}) {
        const resolvedLayerIndex = Number.isFinite(layerIndex) ? Math.floor(layerIndex) : null;
        if (!Number.isFinite(resolvedLayerIndex)) return null;

        const { lane } = this._findLayerLaneByToken({
            layerIndex: resolvedLayerIndex,
            tokenIndex,
            tokenId,
            tokenLabel
        });
        if (!lane || !Array.isArray(lane.expandedVecSegments) || !lane.expandedVecSegments.length) return null;

        const normalizedStage = String(stage || '').trim().toLowerCase();
        const vectorRef = lane.expandedVecSegments.find((candidate) => {
            const candidateStage = String(candidate?.userData?.activationData?.stage || '').toLowerCase();
            return normalizedStage && candidateStage === normalizedStage;
        }) || lane.expandedVecSegments[0] || null;
        if (!vectorRef) return null;

        const resolvedLabel = String(
            label
            || vectorRef.userData?.activationData?.label
            || vectorRef.group?.userData?.label
            || vectorRef.mesh?.userData?.label
            || 'MLP Up Projection'
        ).trim() || 'MLP Up Projection';
        const activationData = {
            ...(vectorRef.userData?.activationData && typeof vectorRef.userData.activationData === 'object'
                ? vectorRef.userData.activationData
                : {}),
            label: resolvedLabel,
            ...(normalizedStage ? { stage: normalizedStage } : {}),
            layerIndex: resolvedLayerIndex,
            ...(Number.isFinite(tokenIndex) ? { tokenIndex: Math.floor(tokenIndex) } : {}),
            ...(Number.isFinite(tokenId) ? { tokenId: Math.floor(tokenId) } : {}),
            ...(typeof tokenLabel === 'string' && tokenLabel.trim().length
                ? { tokenLabel: formatTokenLabelForPreview(tokenLabel) }
                : {})
        };

        return this._buildRuntimeVectorSelection({
            vectorRef,
            label: resolvedLabel,
            kind: 'vector',
            activationData,
            layerIndex: resolvedLayerIndex,
            tokenIndex,
            tokenId,
            tokenLabel
        });
    }

    _findRuntimeMlpDownVectorSelection({
        layerIndex = null,
        tokenIndex = null,
        tokenId = null,
        tokenLabel = ''
    } = {}) {
        const resolvedLayerIndex = Number.isFinite(layerIndex) ? Math.floor(layerIndex) : null;
        if (!Number.isFinite(resolvedLayerIndex)) return null;

        const { lane } = this._findLayerLaneByToken({
            layerIndex: resolvedLayerIndex,
            tokenIndex,
            tokenId,
            tokenLabel
        });
        if (!lane) return null;

        const vectorRef = lane.finalVecAfterMlp || null;
        if (!vectorRef) return null;

        const resolvedLabel = 'MLP Down Projection';
        const activationData = {
            ...(vectorRef.userData?.activationData && typeof vectorRef.userData.activationData === 'object'
                ? vectorRef.userData.activationData
                : {}),
            label: resolvedLabel,
            stage: 'mlp.down',
            layerIndex: resolvedLayerIndex,
            ...(Number.isFinite(tokenIndex) ? { tokenIndex: Math.floor(tokenIndex) } : {}),
            ...(Number.isFinite(tokenId) ? { tokenId: Math.floor(tokenId) } : {}),
            ...(typeof tokenLabel === 'string' && tokenLabel.trim().length
                ? { tokenLabel: formatTokenLabelForPreview(tokenLabel) }
                : {})
        };

        return this._buildRuntimeVectorSelection({
            vectorRef,
            label: resolvedLabel,
            kind: 'vector',
            activationData,
            layerIndex: resolvedLayerIndex,
            tokenIndex,
            tokenId,
            tokenLabel
        });
    }

    _readVectorEntryNumber(entry, key) {
        if (!entry || typeof entry !== 'object') return null;
        const direct = entry[key];
        if (Number.isFinite(direct)) return Math.floor(direct);
        const activationValue = entry.activationData?.[key];
        if (Number.isFinite(activationValue)) return Math.floor(activationValue);
        return null;
    }

    _readVectorEntryString(entry, key) {
        if (!entry || typeof entry !== 'object') return '';
        const direct = entry[key];
        if (typeof direct === 'string' && direct.trim().length) return direct;
        const activationValue = entry.activationData?.[key];
        if (typeof activationValue === 'string' && activationValue.trim().length) return activationValue;
        return '';
    }

    _collectSceneVectorEntryCandidates(node) {
        if (!node?.isInstancedMesh || !node.userData) return [];
        const prismCount = Number.isFinite(node.userData.prismCount)
            ? Math.max(1, Math.floor(node.userData.prismCount))
            : 1;
        const candidates = [];
        const pushCandidate = (entry, vectorIndex) => {
            if (!entry || typeof entry !== 'object') return;
            const safeVectorIndex = Number.isFinite(vectorIndex) ? Math.max(0, Math.floor(vectorIndex)) : 0;
            candidates.push({
                entry,
                vectorIndex: safeVectorIndex,
                instanceId: safeVectorIndex * prismCount,
                prismCount
            });
        };

        const vectorEntries = Array.isArray(node.userData.vectorEntries) ? node.userData.vectorEntries : null;
        if (vectorEntries?.length) {
            vectorEntries.forEach((entry, vectorIndex) => pushCandidate(entry, vectorIndex));
            return candidates;
        }

        const instanceEntries = Array.isArray(node.userData.instanceEntries) ? node.userData.instanceEntries : null;
        if (!instanceEntries?.length) return candidates;
        for (let instanceId = 0; instanceId < instanceEntries.length; instanceId += prismCount) {
            pushCandidate(instanceEntries[instanceId], Math.floor(instanceId / prismCount));
        }
        return candidates;
    }

    _findSceneVectorEntrySelection({
        stageMatcher = null,
        layerIndex = null,
        headIndex = null,
        tokenIndex = null,
        tokenId = null,
        tokenLabel = '',
        defaultLabel = 'Vector'
    } = {}) {
        const scene = this.engine?.scene || null;
        if (!scene || typeof scene.traverse !== 'function' || typeof stageMatcher !== 'function') return null;

        const normalizedTokenText = formatTokenLabelForPreview(tokenLabel);
        let bestSelection = null;
        let bestScore = -Infinity;

        scene.traverse((node) => {
            if (!node || node.isScene) return;
            const candidates = this._collectSceneVectorEntryCandidates(node);
            if (!candidates.length) return;

            for (const candidate of candidates) {
                const entry = candidate.entry;
                const activationData = entry.activationData && typeof entry.activationData === 'object'
                    ? entry.activationData
                    : null;
                const stageLower = String(activationData?.stage || '').toLowerCase();
                if (!stageMatcher(stageLower, entry, node)) continue;

                const entryLayerIndex = this._readVectorEntryNumber(entry, 'layerIndex');
                const entryHeadIndex = this._readVectorEntryNumber(entry, 'headIndex');
                const entryTokenIndex = this._readVectorEntryNumber(entry, 'tokenIndex');
                const entryTokenId = this._readVectorEntryNumber(entry, 'tokenId')
                    ?? this._readVectorEntryNumber(entry, 'token_id');
                const entryTokenText = formatTokenLabelForPreview(this._readVectorEntryString(entry, 'tokenLabel'));

                if (Number.isFinite(layerIndex) && entryLayerIndex !== Math.floor(layerIndex)) continue;
                if (Number.isFinite(headIndex) && entryHeadIndex !== Math.floor(headIndex)) continue;
                if (Number.isFinite(tokenIndex) && entryTokenIndex !== Math.floor(tokenIndex)) continue;
                if (Number.isFinite(tokenId) && Number.isFinite(entryTokenId) && entryTokenId !== Math.floor(tokenId)) continue;
                if (
                    normalizedTokenText
                    && normalizedTokenText !== ATTENTION_VALUE_PLACEHOLDER
                    && entryTokenText
                    && entryTokenText !== ATTENTION_VALUE_PLACEHOLDER
                    && entryTokenText !== normalizedTokenText
                ) {
                    continue;
                }

                let score = 120;
                if (node.visible !== false) score += 12;
                if (entry.vectorRef) score += 16;
                if (Number.isFinite(entryLayerIndex) && Number.isFinite(layerIndex) && entryLayerIndex === Math.floor(layerIndex)) score += 14;
                if (Number.isFinite(entryHeadIndex) && Number.isFinite(headIndex) && entryHeadIndex === Math.floor(headIndex)) score += 16;
                if (Number.isFinite(entryTokenIndex) && Number.isFinite(tokenIndex) && entryTokenIndex === Math.floor(tokenIndex)) score += 18;
                if (Number.isFinite(entryTokenId) && Number.isFinite(tokenId) && entryTokenId === Math.floor(tokenId)) score += 10;
                if (normalizedTokenText && entryTokenText && entryTokenText === normalizedTokenText) score += 8;

                if (score <= bestScore) continue;

                const label = (typeof entry.label === 'string' && entry.label.trim().length)
                    ? entry.label
                    : defaultLabel;
                const info = {
                    ...(entry && typeof entry === 'object' ? entry : {})
                };
                if (activationData) info.activationData = activationData;
                if (Number.isFinite(entryLayerIndex)) info.layerIndex = entryLayerIndex;
                if (Number.isFinite(entryHeadIndex)) info.headIndex = entryHeadIndex;
                if (Number.isFinite(entryTokenIndex)) info.tokenIndex = entryTokenIndex;
                if (Number.isFinite(entryTokenId)) info.tokenId = entryTokenId;
                if (entryTokenText) info.tokenLabel = entryTokenText;
                if (Number.isFinite(candidate.vectorIndex)) info.vectorIndex = candidate.vectorIndex;
                if (Number.isFinite(candidate.instanceId)) info.prismIndex = 0;
                if (Number.isFinite(candidate.prismCount)) info.prismCount = candidate.prismCount;

                bestSelection = {
                    label,
                    kind: node.userData?.instanceKind || 'instanced',
                    info,
                    object: node,
                    hit: {
                        object: node,
                        instanceId: candidate.instanceId
                    }
                };
                bestScore = score;
            }
        });

        return bestSelection;
    }

    _resolveTransformerView2dSelectionTokenContext(selection = null) {
        const tokenIndex = Number.isFinite(findUserDataNumber(selection, 'tokenIndex'))
            ? Math.floor(findUserDataNumber(selection, 'tokenIndex'))
            : null;
        const queryTokenIndex = Number.isFinite(findUserDataNumber(selection, 'queryTokenIndex'))
            ? Math.floor(findUserDataNumber(selection, 'queryTokenIndex'))
            : tokenIndex;
        const keyTokenIndex = Number.isFinite(findUserDataNumber(selection, 'keyTokenIndex'))
            ? Math.floor(findUserDataNumber(selection, 'keyTokenIndex'))
            : null;
        const directTokenId = findUserDataNumber(selection, 'tokenId');
        const tokenId = Number.isFinite(directTokenId)
            ? Math.floor(directTokenId)
            : (
                Number.isFinite(tokenIndex)
                && this.activationSource
                && typeof this.activationSource.getTokenId === 'function'
                    ? Math.floor(this.activationSource.getTokenId(tokenIndex))
                    : null
            );

        let tokenLabel = findUserDataString(selection, 'tokenLabel');
        if (!tokenLabel && Number.isFinite(tokenIndex)) {
            if (this.activationSource && typeof this.activationSource.getTokenRawString === 'function') {
                tokenLabel = this.activationSource.getTokenRawString(tokenIndex) || '';
            }
            if (!tokenLabel && this.activationSource && typeof this.activationSource.getTokenString === 'function') {
                tokenLabel = this.activationSource.getTokenString(tokenIndex) || '';
            }
        }

        return {
            tokenIndex,
            queryTokenIndex,
            keyTokenIndex,
            tokenId,
            tokenLabel: formatTokenLabelForPreview(tokenLabel)
        };
    }

    _inferTransformerView2dFallbackSelectionKind({
        label = '',
        stageLower = '',
        sourceSelection = null
    } = {}) {
        const lower = String(label || '').trim().toLowerCase();
        if (sourceSelection?.kind === 'attentionSphere' || isAttentionScoreSelection(label, sourceSelection)) {
            return 'attentionSphere';
        }
        if (
            lower === 'layernorm'
            || lower === 'layernorm (top)'
            || lower === 'multilayer perceptron'
        ) {
            return 'label';
        }
        if (lower.includes('matrix') || isWeightMatrixLabel(label) || isQkvMatrixLabel(label)) {
            return 'matrix';
        }
        if (
            lower.includes('vector')
            || lower.includes('weighted sum')
            || lower.includes('activation')
            || lower.includes('residual stream')
            || stageLower.startsWith('layer.incoming')
            || stageLower.startsWith('residual.')
            || stageLower.startsWith('qkv.')
            || stageLower.startsWith('attention.')
            || stageLower.startsWith('mlp.')
            || stageLower.startsWith('ln1.')
            || stageLower.startsWith('ln2.')
            || stageLower.startsWith('final_ln.')
        ) {
            return 'vector';
        }
        return sourceSelection?.kind || 'label';
    }

    _buildTransformerView2dFallbackSelection(selection = null) {
        if (!selection?.label) return null;
        const label = normalizeSelectionLabel(selection.label, selection);
        const activationData = getActivationDataFromSelection(selection);
        const stageLower = String(activationData?.stage || '').toLowerCase();
        const layerIndex = findUserDataNumber(selection, 'layerIndex');
        const headIndex = findUserDataNumber(selection, 'headIndex');
        const tokenContext = this._resolveTransformerView2dSelectionTokenContext(selection);
        const info = selection?.info && typeof selection.info === 'object'
            ? { ...selection.info }
            : {};

        if (activationData && typeof activationData === 'object') {
            info.activationData = { ...activationData };
        }
        if (Number.isFinite(layerIndex)) info.layerIndex = Math.floor(layerIndex);
        if (Number.isFinite(headIndex)) info.headIndex = Math.floor(headIndex);
        if (Number.isFinite(tokenContext.tokenIndex)) info.tokenIndex = tokenContext.tokenIndex;
        if (Number.isFinite(tokenContext.queryTokenIndex)) info.queryTokenIndex = tokenContext.queryTokenIndex;
        if (Number.isFinite(tokenContext.keyTokenIndex)) info.keyTokenIndex = tokenContext.keyTokenIndex;
        if (Number.isFinite(tokenContext.tokenId)) info.tokenId = tokenContext.tokenId;
        if (tokenContext.tokenLabel) info.tokenLabel = tokenContext.tokenLabel;

        return this._attachActivationSourceValuesToSelection({
            label,
            kind: this._inferTransformerView2dFallbackSelectionKind({
                label,
                stageLower,
                sourceSelection: selection
            }),
            info
        });
    }

    _isBroadTransformerView2dSceneVectorSelection(selection = null) {
        if (!selection?.object && !selection?.hit?.object) return false;
        if (selection?.info?.vectorRef) return false;
        if (Number.isFinite(selection?.hit?.instanceId)) return false;

        const prismCountCandidates = [
            selection?.info?.prismCount,
            selection?.object?.userData?.prismCount,
            selection?.hit?.object?.userData?.prismCount
        ];
        if (prismCountCandidates.some((count) => Number.isFinite(count) && count > 0)) {
            return false;
        }

        const vectorMesh = findVectorSourceMesh(selection);
        if (!vectorMesh?.isInstancedMesh) return false;

        const meshCount = Number.isFinite(vectorMesh.count)
            ? vectorMesh.count
            : vectorMesh.instanceMatrix?.count;
        return Number.isFinite(meshCount) && meshCount > 1;
    }

    _preferTransformerView2dFallbackVectorSelection(selection = null, fallbackSelection = null) {
        const fallbackStage = normalizePostLayerNormResidualStage(
            getActivationDataFromSelection(fallbackSelection)?.stage
            || getActivationDataFromSelection(fallbackSelection)?.sourceStage
            || ''
        );
        const fallbackHasHeadContext = Number.isFinite(findUserDataNumber(fallbackSelection, 'headIndex'));
        const fallbackExplicitValues = extractExplicitPreviewVectorData(fallbackSelection);
        if (
            fallbackStage
            && fallbackHasHeadContext
            && fallbackExplicitValues?.length
        ) {
            return fallbackSelection || selection;
        }

        if (!this._isBroadTransformerView2dSceneVectorSelection(selection)) {
            return selection;
        }
        return fallbackSelection || selection;
    }

    _normalizeTransformerView2dSceneLookupLabel(label = '', selectionInfo = null) {
        const normalizedLabel = normalizeSelectionLabel(label, selectionInfo);
        const simplifiedLabel = simplifyLayerNormParamDisplayLabel(normalizedLabel, selectionInfo);
        return String(simplifiedLabel || normalizedLabel || label).trim().toLowerCase();
    }

    _selectionLabelMatchesSceneCandidate(selection = null, label = '', sceneSelectionInfo = null) {
        const targetLabel = this._normalizeTransformerView2dSceneLookupLabel(selection?.label || '', selection);
        const candidateLabel = this._normalizeTransformerView2dSceneLookupLabel(label, sceneSelectionInfo);
        return !!targetLabel && !!candidateLabel && targetLabel === candidateLabel;
    }

    _findTransformerView2dFallbackSceneSelection(selection = null) {
        if (!selection?.label) return null;

        const label = normalizeSelectionLabel(selection.label, selection);
        const activationData = getActivationDataFromSelection(selection);
        const stageLower = String(activationData?.stage || '').toLowerCase();
        const kind = (typeof selection?.kind === 'string' && selection.kind.trim().length)
            ? selection.kind
            : this._inferTransformerView2dFallbackSelectionKind({
                label,
                stageLower,
                sourceSelection: selection
            });
        const layerIndex = findUserDataNumber(selection, 'layerIndex');
        const headIndex = findUserDataNumber(selection, 'headIndex');
        const tokenContext = this._resolveTransformerView2dSelectionTokenContext(selection);

        const labelMatcher = (sceneLabel = '', node = null) => this._selectionLabelMatchesSceneCandidate(
            selection,
            sceneLabel,
            node ? { object: node } : null
        );

        if (isLikelyVectorSelection(label, selection)) {
            const instancedSelection = this._findSceneVectorEntrySelection({
                stageMatcher: (_stage, entry, node) => {
                    const entrySelectionInfo = entry?.activationData
                        ? { info: { activationData: entry.activationData } }
                        : null;
                    return [
                        entry?.label,
                        entry?.activationData?.label,
                        entry?.vectorRef?.group?.userData?.label,
                        entry?.vectorRef?.mesh?.userData?.label,
                        node?.userData?.label,
                        node?.userData?.activationData?.label
                    ].some((candidateLabel) => this._selectionLabelMatchesSceneCandidate(
                        selection,
                        candidateLabel,
                        entrySelectionInfo || (node ? { object: node } : null)
                    ));
                },
                layerIndex,
                headIndex,
                tokenIndex: tokenContext.tokenIndex,
                tokenId: tokenContext.tokenId,
                tokenLabel: tokenContext.tokenLabel,
                defaultLabel: label
            });
            if (instancedSelection) return instancedSelection;
        }

        return this._findGenericSceneObjectSelection({
            label: '',
            layerIndex,
            headIndex,
            tokenIndex: tokenContext.tokenIndex,
            tokenId: tokenContext.tokenId,
            tokenLabel: tokenContext.tokenLabel,
            kind,
            fallbackLabel: label,
            labelMatcher
        });
    }

    _resolveActivationSourceVectorValuesForSelection(selection = null) {
        const activation = getActivationDataFromSelection(selection);
        const stageLower = String(activation?.stage || '').toLowerCase();
        const sourceStageLower = String(activation?.sourceStage || '').toLowerCase();
        const tokenIndex = findUserDataNumber(selection, 'tokenIndex');
        const layerIndex = findUserDataNumber(selection, 'layerIndex');
        const headIndex = findUserDataNumber(selection, 'headIndex');
        const source = this.activationSource;
        if (!source) return null;

        const baseVectorLength = typeof source.getBaseVectorLength === 'function'
            ? Math.max(1, Math.floor(source.getBaseVectorLength() || D_MODEL))
            : D_MODEL;
        const stageCandidates = [...new Set([sourceStageLower, stageLower].filter(Boolean).flatMap((stage) => {
            if (stage === 'ln1.input') return [stage, 'layer.incoming'];
            if (stage === 'ln1.product') return [stage, 'ln1.scale'];
            if (stage === 'ln1.output') return [stage, 'ln1.shift'];
            if (stage === 'ln2.input') return [stage, 'residual.post_attention'];
            if (stage === 'ln2.product') return [stage, 'ln2.scale'];
            if (stage === 'ln2.output') return [stage, 'ln2.shift'];
            if (stage === 'final_ln.product') return [stage, 'final_ln.scale'];
            if (stage === 'final_ln.output') return [stage, 'final_ln.shift'];
            return [stage];
        }))];
        const tryRead = (reader) => {
            const values = typeof reader === 'function' ? reader() : null;
            return (Array.isArray(values) || ArrayBuffer.isView(values)) && values.length > 0
                ? Array.from(values, (value) => (Number.isFinite(value) ? value : 0))
                : null;
        };

        for (const stage of stageCandidates) {
            if (stage === 'qkv.q' || stage === 'qkv.k' || stage === 'qkv.v') {
                const kind = stage.slice('qkv.'.length, 'qkv.'.length + 1);
                const values = Number.isFinite(layerIndex)
                    && Number.isFinite(headIndex)
                    && Number.isFinite(tokenIndex)
                    && typeof source.getLayerQKVVector === 'function'
                    ? tryRead(() => source.getLayerQKVVector(layerIndex, kind, headIndex, tokenIndex, null))
                    : null;
                if (values) return values;
                continue;
            }

            if (stage === 'attention.weighted_sum') {
                const values = Number.isFinite(layerIndex)
                    && Number.isFinite(headIndex)
                    && Number.isFinite(tokenIndex)
                    && typeof source.getAttentionWeightedSum === 'function'
                    ? tryRead(() => source.getAttentionWeightedSum(layerIndex, headIndex, tokenIndex, null))
                    : null;
                if (values) return values;
                continue;
            }

            if (stage === 'attention.output_projection') {
                const values = Number.isFinite(layerIndex)
                    && Number.isFinite(tokenIndex)
                    && typeof source.getAttentionOutputProjection === 'function'
                    ? tryRead(() => source.getAttentionOutputProjection(layerIndex, tokenIndex, baseVectorLength))
                    : null;
                if (values) return values;
                continue;
            }

            if (stage === 'layer.incoming') {
                const values = Number.isFinite(layerIndex)
                    && Number.isFinite(tokenIndex)
                    && typeof source.getLayerIncoming === 'function'
                    ? tryRead(() => source.getLayerIncoming(layerIndex, tokenIndex, baseVectorLength))
                    : null;
                if (values) return values;
                continue;
            }

            if (stage === 'residual.post_attention') {
                const values = Number.isFinite(layerIndex)
                    && Number.isFinite(tokenIndex)
                    && typeof source.getPostAttentionResidual === 'function'
                    ? tryRead(() => source.getPostAttentionResidual(layerIndex, tokenIndex, baseVectorLength))
                    : null;
                if (values) return values;
                continue;
            }

            if (stage === 'residual.post_mlp') {
                const values = Number.isFinite(layerIndex)
                    && Number.isFinite(tokenIndex)
                    && typeof source.getPostMlpResidual === 'function'
                    ? tryRead(() => source.getPostMlpResidual(layerIndex, tokenIndex, baseVectorLength))
                    : null;
                if (values) return values;
                continue;
            }

            if (stage === 'mlp.up') {
                const values = Number.isFinite(layerIndex)
                    && Number.isFinite(tokenIndex)
                    && typeof source.getMlpUp === 'function'
                    ? tryRead(() => source.getMlpUp(layerIndex, tokenIndex, baseVectorLength * 4))
                    : null;
                if (values) return values;
                continue;
            }

            if (stage === 'mlp.activation') {
                const values = Number.isFinite(layerIndex)
                    && Number.isFinite(tokenIndex)
                    && typeof source.getMlpActivation === 'function'
                    ? tryRead(() => source.getMlpActivation(layerIndex, tokenIndex, baseVectorLength * 4))
                    : null;
                if (values) return values;
                continue;
            }

            if (stage === 'mlp.down') {
                const values = Number.isFinite(layerIndex)
                    && Number.isFinite(tokenIndex)
                    && typeof source.getMlpDown === 'function'
                    ? tryRead(() => source.getMlpDown(layerIndex, tokenIndex, baseVectorLength))
                    : null;
                if (values) return values;
                continue;
            }

            if (stage === 'ln1.norm' || stage === 'ln1.scale' || stage === 'ln1.output' || stage === 'ln1.shift') {
                const lnStage = stage === 'ln1.output'
                    ? 'shift'
                    : stage.slice('ln1.'.length);
                const values = Number.isFinite(layerIndex)
                    && Number.isFinite(tokenIndex)
                    && typeof source.getLayerLn1 === 'function'
                    ? tryRead(() => source.getLayerLn1(layerIndex, lnStage, tokenIndex, baseVectorLength))
                    : null;
                if (values) return values;
                continue;
            }

            if (stage === 'ln2.norm' || stage === 'ln2.scale' || stage === 'ln2.output' || stage === 'ln2.shift') {
                const lnStage = stage === 'ln2.output'
                    ? 'shift'
                    : stage.slice('ln2.'.length);
                const values = Number.isFinite(layerIndex)
                    && Number.isFinite(tokenIndex)
                    && typeof source.getLayerLn2 === 'function'
                    ? tryRead(() => source.getLayerLn2(layerIndex, lnStage, tokenIndex, baseVectorLength))
                    : null;
                if (values) return values;
                continue;
            }

            if (stage === 'final_ln.input' || stage === 'final_ln.norm' || stage === 'final_ln.scale' || stage === 'final_ln.output' || stage === 'final_ln.shift') {
                const finalStage = stage === 'final_ln.output'
                    ? 'shift'
                    : stage.slice('final_ln.'.length);
                const values = Number.isFinite(tokenIndex)
                    && typeof source.getFinalLayerNorm === 'function'
                    ? tryRead(() => source.getFinalLayerNorm(finalStage, tokenIndex, baseVectorLength))
                    : null;
                if (values) return values;
            }
        }

        return null;
    }

    _attachActivationSourceValuesToSelection(selection = null) {
        if (!selection?.label) return selection;
        const info = selection.info && typeof selection.info === 'object' ? selection.info : {};
        const activationData = info.activationData && typeof info.activationData === 'object'
            ? info.activationData
            : null;
        const existingValues = activationData?.values
            || info.vectorData
            || info.values;
        if ((Array.isArray(existingValues) || ArrayBuffer.isView(existingValues)) && existingValues.length > 0) {
            return selection;
        }

        const values = this._resolveActivationSourceVectorValuesForSelection(selection);
        if (!values) return selection;

        return {
            ...selection,
            info: {
                ...info,
                values,
                activationData: {
                    ...(activationData || {}),
                    values
                }
            }
        };
    }

    _findGenericSceneObjectSelection({
        label = '',
        activationStages = null,
        layerIndex = null,
        headIndex = null,
        tokenIndex = null,
        tokenId = null,
        tokenLabel = '',
        kind = 'label',
        fallbackLabel = '',
        labelMatcher = null
    } = {}) {
        const scene = this.engine?.scene || null;
        if (!scene || typeof scene.traverse !== 'function') return null;

        const normalizedLabel = String(label || '').trim().toLowerCase();
        const normalizedStages = Array.isArray(activationStages)
            ? activationStages
                .map((stage) => String(stage || '').trim().toLowerCase())
                .filter(Boolean)
            : [];
        const normalizedTokenText = formatTokenLabelForPreview(tokenLabel);
        let bestSelection = null;
        let bestScore = -Infinity;

        scene.traverse((node) => {
            if (!node || node.isScene || !node.userData) return;

            const nodeStageLower = String(node.userData.activationData?.stage || '').toLowerCase();
            if (normalizedStages.length && !normalizedStages.includes(nodeStageLower)) return;

            const nodeLabel = String(
                readSceneSelectionString(node, 'label')
                || node.userData.label
                || ''
            ).trim();
            const nodeLabelLower = nodeLabel.toLowerCase();
            if (normalizedLabel.length && nodeLabelLower !== normalizedLabel) return;
            if (typeof labelMatcher === 'function' && !labelMatcher(nodeLabel, node)) return;

            const nodeLayerIndex = readSceneSelectionNumber(node, 'layerIndex');
            const nodeHeadIndex = readSceneSelectionNumber(node, 'headIndex');
            const nodeTokenIndex = readSceneSelectionNumber(node, 'tokenIndex');
            const nodeTokenId = readSceneSelectionNumber(node, 'tokenId');
            const nodeTokenText = formatTokenLabelForPreview(readSceneSelectionString(node, 'tokenLabel'));

            if (Number.isFinite(layerIndex) && Number.isFinite(nodeLayerIndex) && nodeLayerIndex !== Math.floor(layerIndex)) return;
            if (Number.isFinite(headIndex) && Number.isFinite(nodeHeadIndex) && nodeHeadIndex !== Math.floor(headIndex)) return;
            if (Number.isFinite(tokenIndex) && Number.isFinite(nodeTokenIndex) && nodeTokenIndex !== Math.floor(tokenIndex)) return;
            if (Number.isFinite(tokenId) && Number.isFinite(nodeTokenId) && nodeTokenId !== Math.floor(tokenId)) return;
            if (
                normalizedTokenText
                && normalizedTokenText !== ATTENTION_VALUE_PLACEHOLDER
                && nodeTokenText
                && nodeTokenText !== ATTENTION_VALUE_PLACEHOLDER
                && nodeTokenText !== normalizedTokenText
            ) {
                return;
            }

            let score = 100;
            if (node.visible !== false) score += 12;
            if (node.type === 'Group') score += 6;
            if (normalizedLabel.length && nodeLabelLower === normalizedLabel) score += 24;
            if (normalizedStages.length && normalizedStages.includes(nodeStageLower)) score += 10;
            if (Number.isFinite(nodeLayerIndex) && Number.isFinite(layerIndex) && nodeLayerIndex === Math.floor(layerIndex)) score += 18;
            if (Number.isFinite(nodeHeadIndex) && Number.isFinite(headIndex) && nodeHeadIndex === Math.floor(headIndex)) score += 18;
            if (Number.isFinite(nodeTokenIndex) && Number.isFinite(tokenIndex) && nodeTokenIndex === Math.floor(tokenIndex)) score += 16;
            if (Number.isFinite(nodeTokenId) && Number.isFinite(tokenId) && nodeTokenId === Math.floor(tokenId)) score += 10;
            if (normalizedTokenText && nodeTokenText && nodeTokenText === normalizedTokenText) score += 8;

            if (score <= bestScore) return;

            const info = {};
            if (node.userData.activationData && typeof node.userData.activationData === 'object') {
                info.activationData = node.userData.activationData;
            }
            if (Number.isFinite(nodeLayerIndex)) info.layerIndex = nodeLayerIndex;
            if (Number.isFinite(nodeHeadIndex)) info.headIndex = nodeHeadIndex;
            if (Number.isFinite(nodeTokenIndex)) info.tokenIndex = nodeTokenIndex;
            if (Number.isFinite(nodeTokenId)) info.tokenId = nodeTokenId;
            if (nodeTokenText) info.tokenLabel = nodeTokenText;

            bestSelection = {
                label: nodeLabel || fallbackLabel || label || 'Selection',
                kind,
                info,
                object: node
            };
            bestScore = score;
        });

        return bestSelection;
    }

    _findGenericSceneVectorSelection({
        activationStages = null,
        layerIndex = null,
        headIndex = null,
        tokenIndex = null,
        tokenId = null,
        tokenLabel = '',
        defaultLabel = 'Vector',
        labelMatcher = null,
        allowGenericObjectFallback = true
    } = {}) {
        const normalizedStages = Array.isArray(activationStages)
            ? activationStages
                .map((stage) => String(stage || '').trim().toLowerCase())
                .filter(Boolean)
            : [];
        if (!normalizedStages.length) return null;

        const instancedSelection = this._findSceneVectorEntrySelection({
            stageMatcher: (stageLower) => normalizedStages.includes(stageLower),
            layerIndex,
            headIndex,
            tokenIndex,
            tokenId,
            tokenLabel,
            defaultLabel
        });
        if (
            instancedSelection
            && (
                typeof labelMatcher !== 'function'
                || labelMatcher(instancedSelection.label, instancedSelection?.object || null)
            )
        ) {
            return instancedSelection;
        }

        if (allowGenericObjectFallback === false) {
            return null;
        }

        return this._findGenericSceneObjectSelection({
            activationStages: normalizedStages,
            layerIndex,
            headIndex,
            tokenIndex,
            tokenId,
            tokenLabel,
            kind: 'vector',
            fallbackLabel: defaultLabel,
            labelMatcher
        });
    }

    _findAttentionVectorSceneSelection({
        vectorKind = 'Q',
        layerIndex = null,
        tokenIndex = null,
        tokenId = null,
        tokenLabel = '',
        headIndex = null
    } = {}) {
        const scene = this.engine?.scene || null;
        if (!scene || typeof scene.traverse !== 'function') return null;

        const safeKind = normalizeQkvActionKind(vectorKind);
        const desiredStage = safeKind === 'Q'
            ? 'qkv.q'
            : (safeKind === 'V' ? 'qkv.v' : 'qkv.k');
        const instancedSelection = this._findSceneVectorEntrySelection({
            stageMatcher: (stageLower, entry) => {
                const cachedKv = entry?.vectorRef?.userData?.cachedKv === true
                    || entry?.vectorRef?.userData?.kvCachePersistent === true;
                const entryCategory = String(entry?.category || entry?.vectorRef?.userData?.vectorCategory || '').toUpperCase();
                const isLiveMatch = stageLower === desiredStage;
                const isCachedVectorMatch = cachedKv && entryCategory === safeKind && (safeKind === 'K' || safeKind === 'V');
                return isLiveMatch || isCachedVectorMatch;
            },
            layerIndex,
            headIndex,
            tokenIndex,
            tokenId,
            tokenLabel,
            defaultLabel: safeKind === 'Q'
                ? 'Query Vector'
                : (safeKind === 'V' ? 'Value Vector' : 'Key Vector')
        });
        if (instancedSelection) return instancedSelection;

        const normalizedTokenText = formatTokenLabelForPreview(tokenLabel);
        let bestSelection = null;
        let bestScore = -Infinity;

        const readNumber = (node, key) => {
            const direct = node?.userData?.[key];
            if (Number.isFinite(direct)) return Math.floor(direct);
            const activationValue = node?.userData?.activationData?.[key];
            if (Number.isFinite(activationValue)) return Math.floor(activationValue);
            return null;
        };
        const readString = (node, key) => {
            const direct = node?.userData?.[key];
            if (typeof direct === 'string' && direct.trim().length) return direct;
            const activationValue = node?.userData?.activationData?.[key];
            if (typeof activationValue === 'string' && activationValue.trim().length) return activationValue;
            return '';
        };

        scene.traverse((node) => {
            if (!node || node.isScene || !node.userData) return;
            const stageLower = String(node.userData.activationData?.stage || '').toLowerCase();
            const cachedKv = node.userData.cachedKv === true || node.userData.kvCachePersistent === true;
            const vectorCategory = String(node.userData.vectorCategory || '').toUpperCase();
            const isLiveMatch = stageLower === desiredStage;
            const isCachedVectorMatch = cachedKv && vectorCategory === safeKind && (safeKind === 'K' || safeKind === 'V');
            if (!isLiveMatch && !isCachedVectorMatch) return;

            const nodeHeadIndex = readNumber(node, 'headIndex');
            const nodeLayerIndex = readNumber(node, 'layerIndex');
            const nodeTokenIndex = readNumber(node, 'tokenIndex');
            const nodeTokenId = readNumber(node, 'tokenId');
            const nodeTokenText = formatTokenLabelForPreview(readString(node, 'tokenLabel'));

            if (Number.isFinite(layerIndex) && Number.isFinite(nodeLayerIndex) && nodeLayerIndex !== Math.floor(layerIndex)) return;
            if (Number.isFinite(headIndex) && Number.isFinite(nodeHeadIndex) && nodeHeadIndex !== Math.floor(headIndex)) return;
            if (Number.isFinite(tokenIndex) && Number.isFinite(nodeTokenIndex) && nodeTokenIndex !== Math.floor(tokenIndex)) return;
            if (Number.isFinite(tokenId) && Number.isFinite(nodeTokenId) && nodeTokenId !== Math.floor(tokenId)) return;
            if (
                normalizedTokenText
                && normalizedTokenText !== ATTENTION_VALUE_PLACEHOLDER
                && nodeTokenText
                && nodeTokenText !== ATTENTION_VALUE_PLACEHOLDER
                && nodeTokenText !== normalizedTokenText
            ) {
                return;
            }

            let score = isLiveMatch ? 100 : 60;
            if (node.visible !== false) score += 12;
            if (node.type === 'Group') score += 4;
            if (Number.isFinite(nodeLayerIndex) && Number.isFinite(layerIndex) && nodeLayerIndex === Math.floor(layerIndex)) score += 18;
            if (Number.isFinite(nodeHeadIndex) && Number.isFinite(headIndex) && nodeHeadIndex === Math.floor(headIndex)) score += 16;
            if (Number.isFinite(nodeTokenIndex) && Number.isFinite(tokenIndex) && nodeTokenIndex === Math.floor(tokenIndex)) score += 20;
            if (Number.isFinite(nodeTokenId) && Number.isFinite(tokenId) && nodeTokenId === Math.floor(tokenId)) score += 10;
            if (normalizedTokenText && nodeTokenText && nodeTokenText === normalizedTokenText) score += 8;

            if (score <= bestScore) return;

            const label = typeof node.userData.label === 'string' && node.userData.label.trim().length
                ? node.userData.label
                : (
                    safeKind === 'Q'
                        ? 'Query Vector'
                        : (safeKind === 'V'
                            ? (isCachedVectorMatch ? 'Cached Value Vector' : 'Value Vector')
                            : (isCachedVectorMatch ? 'Cached Key Vector' : 'Key Vector'))
                );
            const info = {};
            const activationData = node.userData.activationData;
            if (activationData && typeof activationData === 'object') info.activationData = activationData;
            if (Number.isFinite(nodeLayerIndex)) info.layerIndex = nodeLayerIndex;
            if (Number.isFinite(nodeHeadIndex)) info.headIndex = nodeHeadIndex;
            if (Number.isFinite(nodeTokenIndex)) info.tokenIndex = nodeTokenIndex;
            if (Number.isFinite(nodeTokenId)) info.tokenId = nodeTokenId;
            if (nodeTokenText) info.tokenLabel = nodeTokenText;

            bestSelection = {
                label,
                kind: 'vector',
                info,
                object: node
            };
            bestScore = score;
        });

        return bestSelection;
    }

    _findQkvSourceVectorSceneSelection({
        layerIndex = null,
        headIndex = null,
        tokenIndex = null,
        tokenId = null,
        tokenLabel = ''
    } = {}) {
        const scene = this.engine?.scene || null;
        if (!scene || typeof scene.traverse !== 'function') return null;
        const postLayerNormStage = normalizePostLayerNormResidualStage('ln1.output') || 'ln1.output';

        if (Number.isFinite(headIndex)) {
            const copiedSelection = this._findSceneVectorEntrySelection({
                stageMatcher: (_stageLower, entry, node) => {
                    const entryCategory = entry?.category || entry?.vectorRef?.userData?.vectorCategory;
                    const entryActivationLabel = entry?.activationData?.label || '';
                    const vectorRefGroupLabel = entry?.vectorRef?.group?.userData?.label || '';
                    const vectorRefMeshLabel = entry?.vectorRef?.mesh?.userData?.label || '';
                    const nodeLabel = node?.userData?.label || '';
                    const nodeActivationLabel = node?.userData?.activationData?.label || '';
                    return isPreQkvResidualCopyCandidate({ label: entry?.label, category: entryCategory })
                        || isPreQkvResidualCopyCandidate({ label: entryActivationLabel, category: entryCategory })
                        || isPreQkvResidualCopyCandidate({ label: vectorRefGroupLabel, category: entryCategory })
                        || isPreQkvResidualCopyCandidate({ label: vectorRefMeshLabel, category: entryCategory })
                        || isPreQkvResidualCopyCandidate({ label: nodeLabel, category: node?.userData?.vectorCategory })
                        || isPreQkvResidualCopyCandidate({ label: nodeActivationLabel, category: node?.userData?.vectorCategory });
                },
                layerIndex,
                headIndex,
                tokenIndex,
                tokenId,
                tokenLabel,
                defaultLabel: resolvePostLayerNormResidualLabel({ stage: postLayerNormStage })
            });
            if (copiedSelection) return copiedSelection;
        }

        const instancedSelection = this._findSceneVectorEntrySelection({
            stageMatcher: (stageLower, entry, node) => {
                if (normalizePostLayerNormResidualStage(stageLower) !== postLayerNormStage) return false;
                const entryCategory = entry?.category || entry?.vectorRef?.userData?.vectorCategory;
                const entryActivationLabel = entry?.activationData?.label || '';
                const vectorRefGroupLabel = entry?.vectorRef?.group?.userData?.label || '';
                const vectorRefMeshLabel = entry?.vectorRef?.mesh?.userData?.label || '';
                const nodeLabel = node?.userData?.label || '';
                const nodeActivationLabel = node?.userData?.activationData?.label || '';
                if (isRelabeledQkvVectorCandidate({ label: entry?.label, category: entryCategory })) return false;
                if (isRelabeledQkvVectorCandidate({ label: entryActivationLabel, category: entryCategory })) return false;
                if (isRelabeledQkvVectorCandidate({ label: vectorRefGroupLabel, category: entryCategory })) return false;
                if (isRelabeledQkvVectorCandidate({ label: vectorRefMeshLabel, category: entryCategory })) return false;
                if (isRelabeledQkvVectorCandidate({ label: nodeLabel, category: node?.userData?.vectorCategory })) return false;
                if (isRelabeledQkvVectorCandidate({ label: nodeActivationLabel, category: node?.userData?.vectorCategory })) return false;
                return true;
            },
            layerIndex,
            tokenIndex,
            tokenId,
            tokenLabel,
            defaultLabel: resolvePostLayerNormResidualLabel({ stage: postLayerNormStage })
        });
        if (instancedSelection) return instancedSelection;

        const normalizedTokenText = formatTokenLabelForPreview(tokenLabel);
        let bestSelection = null;
        let bestScore = -Infinity;

        scene.traverse((node) => {
            if (!node || node.isScene || !node.userData) return;
            const stageLower = String(node.userData.activationData?.stage || '').toLowerCase();
            const normalizedStage = normalizePostLayerNormResidualStage(stageLower);
            if (normalizedStage !== postLayerNormStage) return;
            if (isRelabeledQkvVectorCandidate({
                label: node.userData.label || node.userData.activationData?.label || '',
                category: node.userData.vectorCategory
            })) {
                return;
            }

            const nodeLayerIndex = readSceneSelectionNumber(node, 'layerIndex');
            const nodeTokenIndex = readSceneSelectionNumber(node, 'tokenIndex');
            const nodeTokenId = readSceneSelectionNumber(node, 'tokenId');
            const nodeTokenText = formatTokenLabelForPreview(readSceneSelectionString(node, 'tokenLabel'));

            if (Number.isFinite(layerIndex) && nodeLayerIndex !== Math.floor(layerIndex)) return;
            if (Number.isFinite(tokenIndex) && nodeTokenIndex !== Math.floor(tokenIndex)) return;
            if (Number.isFinite(tokenId) && Number.isFinite(nodeTokenId) && nodeTokenId !== Math.floor(tokenId)) return;
            if (
                normalizedTokenText
                && normalizedTokenText !== ATTENTION_VALUE_PLACEHOLDER
                && nodeTokenText
                && nodeTokenText !== ATTENTION_VALUE_PLACEHOLDER
                && nodeTokenText !== normalizedTokenText
            ) {
                return;
            }

            let score = 100;
            if (node.visible !== false) score += 12;
            if (node.type === 'Group') score += 6;
            if (Number.isFinite(nodeLayerIndex) && Number.isFinite(layerIndex) && nodeLayerIndex === Math.floor(layerIndex)) score += 20;
            if (Number.isFinite(nodeTokenIndex) && Number.isFinite(tokenIndex) && nodeTokenIndex === Math.floor(tokenIndex)) score += 20;
            if (Number.isFinite(nodeTokenId) && Number.isFinite(tokenId) && nodeTokenId === Math.floor(tokenId)) score += 10;
            if (normalizedTokenText && nodeTokenText && nodeTokenText === normalizedTokenText) score += 8;

            if (score <= bestScore) return;

            const label = typeof node.userData.label === 'string' && node.userData.label.trim().length
                ? resolvePostLayerNormResidualLabel({
                    label: node.userData.label,
                    stage: normalizedStage
                })
                : resolvePostLayerNormResidualLabel({ stage: normalizedStage || postLayerNormStage });
            const info = {};
            if (node.userData.activationData && typeof node.userData.activationData === 'object') {
                const sourceStage = node.userData.activationData.sourceStage
                    || normalizePostLayerNormResidualStage(normalizedStage, { preferLegacy: true });
                info.activationData = {
                    ...node.userData.activationData,
                    stage: normalizedStage || node.userData.activationData.stage,
                    ...(sourceStage ? { sourceStage } : {})
                };
            }
            if (Number.isFinite(nodeLayerIndex)) info.layerIndex = nodeLayerIndex;
            if (Number.isFinite(nodeTokenIndex)) info.tokenIndex = nodeTokenIndex;
            if (Number.isFinite(nodeTokenId)) info.tokenId = nodeTokenId;
            if (nodeTokenText) info.tokenLabel = nodeTokenText;

            bestSelection = {
                label,
                kind: 'vector',
                info,
                object: node
            };
            bestScore = score;
        });

        return bestSelection;
    }

    _findQkvWeightMatrixSceneSelection({
        matrixKind = 'Q',
        headIndex = null,
        layerIndex = null
    } = {}) {
        const scene = this.engine?.scene || null;
        if (!scene || typeof scene.traverse !== 'function') return null;

        const desiredLabel = getQkvWeightMatrixLabel(matrixKind);
        const desiredLabelLower = desiredLabel.toLowerCase();
        let bestSelection = null;
        let bestScore = -Infinity;

        scene.traverse((node) => {
            if (!node || node.isScene || !node.userData) return;
            const labelLower = String(node.userData.label || '').trim().toLowerCase();
            if (labelLower !== desiredLabelLower) return;

            const nodeHeadIndex = readSceneSelectionNumber(node, 'headIndex');
            const nodeLayerIndex = readSceneSelectionNumber(node, 'layerIndex');

            if (Number.isFinite(headIndex) && nodeHeadIndex !== Math.floor(headIndex)) return;
            if (Number.isFinite(layerIndex) && nodeLayerIndex !== Math.floor(layerIndex)) return;

            let score = 100;
            if (node.visible !== false) score += 12;
            if (node.type === 'Group') score += 6;
            if (Number.isFinite(nodeHeadIndex) && Number.isFinite(headIndex) && nodeHeadIndex === Math.floor(headIndex)) score += 20;
            if (Number.isFinite(nodeLayerIndex) && Number.isFinite(layerIndex) && nodeLayerIndex === Math.floor(layerIndex)) score += 20;

            if (score <= bestScore) return;

            const info = {};
            if (Number.isFinite(nodeHeadIndex)) info.headIndex = nodeHeadIndex;
            if (Number.isFinite(nodeLayerIndex)) info.layerIndex = nodeLayerIndex;

            bestSelection = {
                label: desiredLabel,
                kind: 'matrix',
                info,
                object: node
            };
            bestScore = score;
        });

        return bestSelection;
    }

    _findSelectionLayerNormCarrier(selection = null) {
        const candidates = [selection?.object, selection?.hit?.object];
        for (const candidate of candidates) {
            let current = candidate;
            while (current && !current.isScene) {
                const label = String(current.userData?.label || '').toLowerCase();
                if (label === 'layernorm' || label === 'layernorm (top)') {
                    return current;
                }
                current = current.parent;
            }
        }
        return null;
    }

    _inferLayerNormKindFromSelection(selection = null) {
        const explicitKind = findUserDataString(selection, 'layerNormKind');
        const rawLabelLower = String(selection?.label || '').toLowerCase();
        const normalizedLabelLower = normalizeSelectionLabel(selection?.label || '', selection).toLowerCase();
        const combinedLabelLower = `${rawLabelLower} ${normalizedLabelLower}`;
        const stageLower = String(getActivationDataFromSelection(selection)?.stage || '').toLowerCase();

        if (explicitKind === 'ln1' || explicitKind === 'ln2' || explicitKind === 'final') return explicitKind;
        if (stageLower.startsWith('ln1.')) return 'ln1';
        if (stageLower.startsWith('ln2.')) return 'ln2';
        if (stageLower.startsWith('final_ln')) return 'final';
        if (
            combinedLabelLower.includes('layernorm (top)')
            || combinedLabelLower.includes('final ln')
            || combinedLabelLower.includes('top layernorm')
        ) {
            return 'final';
        }
        if (combinedLabelLower.includes('ln1')) return 'ln1';
        if (combinedLabelLower.includes('ln2')) return 'ln2';

        const layerIndex = findUserDataNumber(selection, 'layerIndex');
        const layerNormCarrier = this._findSelectionLayerNormCarrier(selection);
        const layers = Array.isArray(this.engine?._layers) ? this.engine._layers : [];
        if (!layerNormCarrier || !Number.isFinite(layerIndex) || layerIndex < 0 || layerIndex >= layers.length) {
            return null;
        }

        const layer = layers[Math.floor(layerIndex)];
        const searchRoot = layer?.raycastRoot || layer?.root || null;
        if (!searchRoot || typeof searchRoot.traverse !== 'function') return null;

        const layerNormGroups = [];
        searchRoot.traverse((node) => {
            if (!node || node === searchRoot) return;
            const label = String(node.userData?.label || '').toLowerCase();
            if (label !== 'layernorm') return;
            layerNormGroups.push(node);
        });
        if (!layerNormGroups.length) return null;
        layerNormGroups.sort((a, b) => {
            const ay = Number.isFinite(a.position?.y) ? a.position.y : 0;
            const by = Number.isFinite(b.position?.y) ? b.position.y : 0;
            return ay - by;
        });

        const matchedIndex = layerNormGroups.findIndex((candidate) => candidate === layerNormCarrier);
        if (matchedIndex === 0) return 'ln1';
        if (matchedIndex === 1) return 'ln2';
        return null;
    }

    _findLayerNormSceneSelection({
        layerNormKind = null,
        layerIndex = null
    } = {}) {
        const safeKind = (layerNormKind === 'ln1' || layerNormKind === 'ln2' || layerNormKind === 'final')
            ? layerNormKind
            : null;
        const buildSelection = (object, fallbackLabel) => {
            if (!object) return null;
            const objectLabel = typeof object.userData?.label === 'string'
                ? object.userData.label.trim()
                : '';
            const useFallback = !objectLabel.length
                || (safeKind && objectLabel.toLowerCase() === 'layernorm');
            return {
                label: useFallback ? fallbackLabel : objectLabel,
                kind: 'label',
                info: {
                    ...(Number.isFinite(layerIndex) ? { layerIndex: Math.floor(layerIndex) } : {}),
                    ...(safeKind ? { layerNormKind: safeKind } : {})
                },
                object
            };
        };

        const layers = Array.isArray(this.engine?._layers) ? this.engine._layers : [];
        if ((safeKind === 'ln1' || safeKind === 'ln2') && Number.isFinite(layerIndex) && layerIndex >= 0 && layerIndex < layers.length) {
            const layer = layers[Math.floor(layerIndex)];
            const object = safeKind === 'ln1' ? layer?.ln1?.group : layer?.ln2?.group;
            const selection = buildSelection(object, formatLayerNormLabel(safeKind));
            if (selection) return selection;
        }

        const scene = this.engine?.scene || null;
        if (!scene || typeof scene.traverse !== 'function') return null;

        let bestSelection = null;
        let bestScore = -Infinity;
        scene.traverse((node) => {
            if (!node || node.isScene || !node.userData) return;
            const labelLower = String(node.userData.label || '').toLowerCase();
            const isTopLayerNorm = labelLower === 'layernorm (top)';
            const isLayerNorm = labelLower === 'layernorm' || isTopLayerNorm;
            if (!isLayerNorm) return;
            if (safeKind === 'final' && !isTopLayerNorm) return;
            if ((safeKind === 'ln1' || safeKind === 'ln2') && isTopLayerNorm) return;

            let score = 100;
            if (node.visible !== false) score += 12;
            if (safeKind === 'final' && isTopLayerNorm) score += 20;
            if (score <= bestScore) return;

            bestSelection = buildSelection(
                node,
                isTopLayerNorm ? formatLayerNormLabel('final') : formatLayerNormLabel(safeKind)
            );
            bestScore = score;
        });

        return bestSelection;
    }

    _buildFallbackLayerNormSelection({
        layerNormKind = null,
        layerIndex = null
    } = {}) {
        const safeKind = layerNormKind === 'final'
            ? 'final'
            : (layerNormKind === 'ln2' ? 'ln2' : 'ln1');
        const label = formatLayerNormLabel(safeKind);
        return {
            label,
            kind: 'label',
            info: {
                ...(Number.isFinite(layerIndex) ? { layerIndex: Math.floor(layerIndex) } : {}),
                layerNormKind: safeKind
            }
        };
    }

    _findLayerNormParamSceneSelection({
        layerNormKind = null,
        param = 'scale',
        layerIndex = null
    } = {}) {
        const safeParam = param === 'shift' ? 'shift' : 'scale';
        const safeKind = (layerNormKind === 'ln1' || layerNormKind === 'ln2' || layerNormKind === 'final')
            ? layerNormKind
            : null;

        const resolvePreferredLaneIndex = (layer = null) => {
            const explicitLaneIndex = findUserDataNumber(this._lastSelection, 'laneIndex');
            if (Number.isFinite(explicitLaneIndex)) return Math.max(0, Math.floor(explicitLaneIndex));

            const explicitTokenIndex = findUserDataNumber(this._lastSelection, 'tokenIndex');
            if (!layer || !Array.isArray(layer.lanes) || !Number.isFinite(explicitTokenIndex)) {
                return null;
            }
            const matchIndex = layer.lanes.findIndex((lane) => (
                lane
                && Number.isFinite(lane.tokenIndex)
                && Math.floor(lane.tokenIndex) === Math.floor(explicitTokenIndex)
            ));
            return matchIndex >= 0 ? matchIndex : null;
        };

        const buildVectorRefSelection = ({
            label,
            kind = 'vector',
            vectorRef = null,
            object = null,
            hit = null,
            layerIndex: resolvedLayerIndex = null
        } = {}) => {
            const resolvedVectorRef = resolveLiveLayerNormParamVectorRef(vectorRef);
            if (!resolvedVectorRef) return null;
            const activationData = resolvedVectorRef.userData?.activationData || null;
            const usesDetachedCarrier = resolvedVectorRef !== vectorRef;
            const info = {
                vectorRef: resolvedVectorRef
            };
            if (activationData && typeof activationData === 'object') {
                info.activationData = activationData;
            }
            const tokenIndex = findUserDataNumber({ info }, 'tokenIndex');
            const tokenLabel = findUserDataString({ info }, 'tokenLabel');
            const resolvedObject = usesDetachedCarrier
                ? (
                    resolvedVectorRef.group
                    || resolvedVectorRef.mesh
                    || object
                    || null
                )
                : (
                    object
                    || resolvedVectorRef.group
                    || resolvedVectorRef.mesh
                    || null
                );
            const resolvedHit = usesDetachedCarrier ? null : hit;
            if (Number.isFinite(resolvedLayerIndex)) info.layerIndex = Math.floor(resolvedLayerIndex);
            if (Number.isFinite(tokenIndex)) info.tokenIndex = Math.floor(tokenIndex);
            if (typeof tokenLabel === 'string' && tokenLabel.trim().length) info.tokenLabel = tokenLabel;
            return {
                label,
                kind,
                info,
                ...(resolvedObject ? { object: resolvedObject } : {}),
                ...(resolvedHit ? { hit: resolvedHit } : {})
            };
        };

        const layers = Array.isArray(this.engine?._layers) ? this.engine._layers : [];
        const targetLayer = Number.isFinite(layerIndex) && layerIndex >= 0 && layerIndex < layers.length
            ? layers[Math.floor(layerIndex)]
            : null;

        const scene = this.engine?.scene || null;
        if (!safeKind) return null;

        if (scene && typeof scene.traverse === 'function') {
            const desiredStage = `${safeKind}.param.${safeParam}`;
            let bestSelection = null;
            let bestScore = -Infinity;

            scene.traverse((node) => {
                if (!node || node.isScene || !node.userData) return;
                const activationStage = String(node.userData.activationData?.stage || '').toLowerCase();
                if (activationStage !== desiredStage) return;

                const nodeLayerIndex = Number.isFinite(node.userData.activationData?.layerIndex)
                    ? Math.floor(node.userData.activationData.layerIndex)
                    : (Number.isFinite(node.userData.layerIndex) ? Math.floor(node.userData.layerIndex) : null);
                if (Number.isFinite(layerIndex) && Number.isFinite(nodeLayerIndex) && nodeLayerIndex !== Math.floor(layerIndex)) {
                    return;
                }

                let score = 100;
                if (node.visible !== false) score += 12;
                if (node.type === 'Group') score += 6;
                if (Number.isFinite(nodeLayerIndex) && Number.isFinite(layerIndex) && nodeLayerIndex === Math.floor(layerIndex)) {
                    score += 18;
                }
                if (score <= bestScore) return;

                const label = typeof node.userData.label === 'string' && node.userData.label.trim().length
                    ? node.userData.label
                    : `${safeKind.toUpperCase()} ${safeParam === 'scale' ? 'Scale' : 'Shift'}`;
                const info = {};
                if (node.userData.activationData && typeof node.userData.activationData === 'object') {
                    info.activationData = node.userData.activationData;
                }
                if (Number.isFinite(nodeLayerIndex)) info.layerIndex = nodeLayerIndex;

                bestSelection = {
                    label,
                    kind: 'vector',
                    info,
                    object: node
                };
                bestScore = score;
            });

            if (bestSelection) return bestSelection;
        }

        if (safeKind === 'final') {
            const placeholders = this.pipeline?._topLnParamPlaceholders || null;
            const refs = safeParam === 'scale' ? placeholders?.scale : placeholders?.shift;
            if (Array.isArray(refs) && refs.length) {
                const preferredIndex = resolvePreferredLaneIndex(null);
                const candidateRef = Number.isFinite(preferredIndex) && refs[preferredIndex]?.group
                    ? refs[preferredIndex]
                    : refs.find((candidate) => candidate?.group);
                if (candidateRef?.group) {
                    return buildVectorRefSelection({
                        label: getLayerNormParamLabel('final', safeParam),
                        kind: 'vector',
                        vectorRef: candidateRef,
                        object: candidateRef.group,
                        layerIndex: Number.isFinite(placeholders?.layerIndex) ? Math.floor(placeholders.layerIndex) : null
                    });
                }
            }
        }

        const bankKey = safeKind === 'ln2'
            ? (safeParam === 'scale' ? 'ln2Scale' : 'ln2Shift')
            : (safeKind === 'ln1'
                ? (safeParam === 'scale' ? 'ln1Scale' : 'ln1Shift')
                : null);
        const bank = bankKey && targetLayer?._lnParamBanks
            ? targetLayer._lnParamBanks[bankKey]
            : null;
        if (bank && typeof bank.getVectorRef === 'function' && bank.mesh) {
            const preferredIndex = resolvePreferredLaneIndex(targetLayer);
            const fallbackIndex = Number.isFinite(preferredIndex)
                ? preferredIndex
                : 0;
            const clampedIndex = Math.max(0, Math.min(
                Math.max(0, (Number.isFinite(bank.vectorCount) ? bank.vectorCount : 1) - 1),
                Math.floor(fallbackIndex)
            ));
            const vectorRef = bank.getVectorRef(clampedIndex);
            if (vectorRef) {
                if (typeof bank.updateVectorRaycastInfo === 'function') {
                    bank.updateVectorRaycastInfo(clampedIndex, vectorRef);
                }
                const prismCount = Number.isFinite(bank.prismCount) ? Math.max(1, Math.floor(bank.prismCount)) : 1;
                const instanceId = clampedIndex * prismCount;
                const entry = Array.isArray(bank.mesh.userData?.instanceEntries)
                    ? bank.mesh.userData.instanceEntries[instanceId]
                    : null;
                const selection = buildVectorRefSelection({
                    label: getLayerNormParamLabel(safeKind, safeParam),
                    kind: bank.mesh.userData?.instanceKind || 'instanced',
                    vectorRef,
                    object: bank.mesh,
                    hit: {
                        object: bank.mesh,
                        instanceId
                    },
                    layerIndex: Number.isFinite(layerIndex) ? Math.floor(layerIndex) : null
                });
                if (selection) {
                    selection.info = {
                        ...(entry && typeof entry === 'object' ? entry : {}),
                        ...(selection.info || {})
                    };
                    return selection;
                }
            }
        }

        return null;
    }

    _buildFallbackLayerNormParamSelection({
        layerNormKind = null,
        param = 'scale',
        layerIndex = null
    } = {}) {
        const safeParam = param === 'shift' ? 'shift' : 'scale';
        const safeKind = layerNormKind === 'final'
            ? 'final'
            : (layerNormKind === 'ln2' ? 'ln2' : 'ln1');
        const label = getLayerNormParamLabel(safeKind, safeParam);
        const stage = safeKind === 'final'
            ? `final_ln.param.${safeParam}`
            : `${safeKind}.param.${safeParam}`;
        const info = {
            activationData: {
                label,
                stage,
                ...(Number.isFinite(layerIndex) ? { layerIndex: Math.floor(layerIndex) } : {}),
                notes: safeParam === 'scale'
                    ? 'LayerNorm scale parameter'
                    : 'LayerNorm shift parameter'
            }
        };
        if (Number.isFinite(layerIndex)) info.layerIndex = Math.floor(layerIndex);
        return {
            label,
            kind: 'vector',
            info
        };
    }

    _buildFallbackAttentionVectorSelection({
        vectorKind = 'Q',
        layerIndex = null,
        tokenIndex = null,
        tokenId = null,
        tokenLabel = '',
        headIndex = null,
        cachedKv = false
    } = {}) {
        const safeKind = normalizeQkvActionKind(vectorKind);
        const isCachedKv = cachedKv === true && (safeKind === 'K' || safeKind === 'V');
        const label = safeKind === 'Q'
            ? 'Query Vector'
            : (safeKind === 'V'
                ? (isCachedKv ? 'Cached Value Vector' : 'Value Vector')
                : (isCachedKv ? 'Cached Key Vector' : 'Key Vector'));
        const info = {};
        if (Number.isFinite(layerIndex)) info.layerIndex = Math.floor(layerIndex);
        if (Number.isFinite(tokenIndex)) info.tokenIndex = Math.floor(tokenIndex);
        if (Number.isFinite(tokenId)) info.tokenId = Math.floor(tokenId);
        if (typeof tokenLabel === 'string' && tokenLabel.trim().length) {
            info.tokenLabel = formatTokenLabelForPreview(tokenLabel);
        }
        if (Number.isFinite(headIndex)) info.headIndex = Math.floor(headIndex);
        if (isCachedKv) {
            info.cachedKv = true;
            info.category = safeKind;
        }
        const stage = safeKind === 'Q' ? 'qkv.q' : (safeKind === 'V' ? 'qkv.v' : 'qkv.k');
        info.activationData = {
            label,
            stage,
            ...(isCachedKv ? { cachedKv: true, cacheKind: safeKind.toLowerCase() } : {}),
            ...(Number.isFinite(layerIndex) ? { layerIndex: Math.floor(layerIndex) } : {}),
            ...(Number.isFinite(headIndex) ? { headIndex: Math.floor(headIndex) } : {}),
            ...(Number.isFinite(tokenIndex) ? { tokenIndex: Math.floor(tokenIndex) } : {}),
            ...(typeof tokenLabel === 'string' && tokenLabel.trim().length ? { tokenLabel: formatTokenLabelForPreview(tokenLabel) } : {})
        };
        return this._attachActivationSourceValuesToSelection({
            label,
            kind: 'vector',
            info
        });
    }

    _buildFallbackQkvSourceVectorSelection({
        layerIndex = null,
        tokenIndex = null,
        tokenId = null,
        tokenLabel = ''
    } = {}) {
        const outputStage = normalizePostLayerNormResidualStage('ln1.output') || 'ln1.output';
        const sourceStage = normalizePostLayerNormResidualStage(outputStage, { preferLegacy: true });
        const label = resolvePostLayerNormResidualLabel({ stage: outputStage });
        const info = {};
        if (Number.isFinite(layerIndex)) info.layerIndex = Math.floor(layerIndex);
        if (Number.isFinite(tokenIndex)) info.tokenIndex = Math.floor(tokenIndex);
        if (Number.isFinite(tokenId)) info.tokenId = Math.floor(tokenId);
        if (typeof tokenLabel === 'string' && tokenLabel.trim().length) {
            info.tokenLabel = formatTokenLabelForPreview(tokenLabel);
        }
        info.activationData = {
            label,
            stage: outputStage,
            sourceStage,
            ...(Number.isFinite(layerIndex) ? { layerIndex: Math.floor(layerIndex) } : {}),
            ...(Number.isFinite(tokenIndex) ? { tokenIndex: Math.floor(tokenIndex) } : {}),
            ...(typeof tokenLabel === 'string' && tokenLabel.trim().length ? { tokenLabel: formatTokenLabelForPreview(tokenLabel) } : {})
        };
        return this._attachActivationSourceValuesToSelection({
            label,
            kind: 'vector',
            info
        });
    }

    _buildFallbackQkvWeightMatrixSelection({
        matrixKind = 'Q',
        headIndex = null,
        layerIndex = null
    } = {}) {
        const label = getQkvWeightMatrixLabel(matrixKind);
        const info = {};
        if (Number.isFinite(headIndex)) info.headIndex = Math.floor(headIndex);
        if (Number.isFinite(layerIndex)) info.layerIndex = Math.floor(layerIndex);
        return {
            label,
            kind: 'matrix',
            info
        };
    }

    _buildFallbackMlpDownVectorSelection({
        layerIndex = null,
        tokenIndex = null,
        tokenId = null,
        tokenLabel = ''
    } = {}) {
        const label = 'MLP Down Projection';
        const info = {};
        if (Number.isFinite(layerIndex)) info.layerIndex = Math.floor(layerIndex);
        if (Number.isFinite(tokenIndex)) info.tokenIndex = Math.floor(tokenIndex);
        if (Number.isFinite(tokenId)) info.tokenId = Math.floor(tokenId);
        if (typeof tokenLabel === 'string' && tokenLabel.trim().length) {
            info.tokenLabel = formatTokenLabelForPreview(tokenLabel);
        }
        info.activationData = {
            label,
            stage: 'mlp.down',
            ...(Number.isFinite(layerIndex) ? { layerIndex: Math.floor(layerIndex) } : {}),
            ...(Number.isFinite(tokenIndex) ? { tokenIndex: Math.floor(tokenIndex) } : {}),
            ...(typeof tokenLabel === 'string' && tokenLabel.trim().length
                ? { tokenLabel: formatTokenLabelForPreview(tokenLabel) }
                : {})
        };
        return this._attachActivationSourceValuesToSelection({
            label,
            kind: 'vector',
            info
        });
    }

    _resolveTransformerView2dCanvasSelection(selection = null) {
        const fallbackSelection = this._buildTransformerView2dFallbackSelection(selection);
        if (!fallbackSelection?.label) return null;
        if (selection?.object || selection?.hit?.object || selection?.info?.vectorRef) {
            return selection;
        }
        const sceneLabelFallbackSelection = () => this._findTransformerView2dFallbackSceneSelection(fallbackSelection);
        const preferFallbackIfBroadVector = (resolvedSelection = null) => (
            this._preferTransformerView2dFallbackVectorSelection(resolvedSelection, fallbackSelection)
        );

        const label = fallbackSelection.label;
        const lower = label.toLowerCase();
        const activation = getActivationDataFromSelection(fallbackSelection);
        const stageLower = String(activation?.stage || '').toLowerCase();
        const sourceStageLower = String(activation?.sourceStage || '').toLowerCase();
        const normalizedPostLayerNormStage = normalizePostLayerNormResidualStage(stageLower);
        const normalizedPostLayerNormSourceStage = normalizePostLayerNormResidualStage(sourceStageLower);
        const liveVectorStages = [...new Set([
            sourceStageLower,
            stageLower,
            normalizedPostLayerNormSourceStage,
            normalizedPostLayerNormStage
        ].filter(Boolean))];
        const layerIndex = findUserDataNumber(fallbackSelection, 'layerIndex');
        const headIndex = findUserDataNumber(fallbackSelection, 'headIndex');
        const positionIndex = findUserDataNumber(fallbackSelection, 'positionIndex');
        const tokenContext = this._resolveTransformerView2dSelectionTokenContext(fallbackSelection);
        const buildChipSelection = ({
            chipObject = null,
            resolvedLabel = '',
            tokenLabel = '',
            kind = 'label'
        } = {}) => {
            const activationData = fallbackSelection?.info?.activationData;
            const info = fallbackSelection?.info && typeof fallbackSelection.info === 'object'
                ? { ...fallbackSelection.info }
                : {};
            if (Number.isFinite(tokenContext.tokenIndex)) info.tokenIndex = tokenContext.tokenIndex;
            if (Number.isFinite(tokenContext.tokenId)) info.tokenId = tokenContext.tokenId;
            if (Number.isFinite(positionIndex)) info.positionIndex = Math.floor(positionIndex);
            if (tokenLabel) info.tokenLabel = tokenLabel;
            if (activationData && typeof activationData === 'object') {
                info.activationData = {
                    ...activationData,
                    ...(resolvedLabel ? { label: resolvedLabel } : {})
                };
            }
            return {
                label: resolvedLabel || fallbackSelection.label,
                kind,
                info,
                ...(chipObject ? { object: chipObject } : {})
            };
        };
        const findMatrixSceneSelectionByLabelAliases = (labelAliases = []) => {
            const normalizedAliases = Array.isArray(labelAliases)
                ? labelAliases
                    .map((candidate) => String(candidate || '').trim().toLowerCase())
                    .filter(Boolean)
                : [];
            if (!normalizedAliases.length) return null;
            return this._findGenericSceneObjectSelection({
                label: '',
                kind: 'matrix',
                fallbackLabel: label,
                labelMatcher: (sceneLabel = '', node = null) => {
                    const normalizedSceneLabel = this._normalizeTransformerView2dSceneLookupLabel(
                        sceneLabel,
                        node ? { object: node } : null
                    );
                    return normalizedAliases.includes(normalizedSceneLabel);
                }
            });
        };

        if (lower.startsWith('token:')) {
            const tokenText = tokenContext.tokenLabel || formatTokenLabelForPreview(extractTokenText(label));
            const tokenObj = this._findTokenChipSceneObject({
                tokenIndex: tokenContext.tokenIndex,
                tokenId: tokenContext.tokenId,
                tokenText
            });
            const objectLabel = typeof tokenObj?.userData?.label === 'string' ? tokenObj.userData.label : '';
            return buildChipSelection({
                chipObject: tokenObj,
                resolvedLabel: objectLabel.toLowerCase().startsWith('token:')
                    ? objectLabel
                    : (tokenText ? `Token: ${tokenText}` : label),
                tokenLabel: tokenText
            });
        }

        if (lower.startsWith('position:')) {
            const tokenText = tokenContext.tokenLabel || formatTokenLabelForPreview(extractTokenText(label));
            const positionObj = this._findPositionChipSceneObject({
                tokenIndex: tokenContext.tokenIndex,
                tokenId: tokenContext.tokenId,
                tokenText,
                positionIndex
            });
            const objectLabel = typeof positionObj?.userData?.label === 'string' ? positionObj.userData.label : '';
            return buildChipSelection({
                chipObject: positionObj,
                resolvedLabel: objectLabel.toLowerCase().startsWith('position:')
                    ? objectLabel
                    : label,
                tokenLabel: tokenText
            });
        }

        if (
            stageLower.startsWith('generation.chosen')
            || lower.startsWith('chosen token:')
            || lower === 'chosen token'
        ) {
            const chosenTokenText = tokenContext.tokenLabel
                || formatTokenLabelForPreview(extractChosenTokenText(label));
            const tokenObj = this._findTokenChipSceneObject({
                tokenIndex: tokenContext.tokenIndex,
                tokenId: tokenContext.tokenId,
                tokenText: chosenTokenText
            });
            const objectLabel = typeof tokenObj?.userData?.label === 'string' ? tokenObj.userData.label : '';
            return buildChipSelection({
                chipObject: tokenObj,
                resolvedLabel: objectLabel.toLowerCase().startsWith('token:')
                    ? objectLabel
                    : (chosenTokenText ? `Token: ${chosenTokenText}` : label),
                tokenLabel: chosenTokenText
            });
        }

        if (lower === 'vocabulary embedding matrix') {
            return findMatrixSceneSelectionByLabelAliases([
                'vocabulary embedding matrix'
            ]) || fallbackSelection;
        }

        if (lower === 'position embedding matrix') {
            return findMatrixSceneSelectionByLabelAliases([
                'position embedding matrix'
            ]) || fallbackSelection;
        }

        if (lower === 'vocabulary unembedding matrix' || lower === 'vocabulary unembedding') {
            const resolvedSelection = findMatrixSceneSelectionByLabelAliases([
                'vocabulary unembedding matrix',
                'vocabulary unembedding',
                'vocab unembedding',
                'vocabulary embedding (top)',
                'vocab embedding (top)'
            ]);
            return resolvedSelection
                ? {
                    ...resolvedSelection,
                    label
                }
                : fallbackSelection;
        }

        if (
            stageLower === 'attention.mask'
            || lower.includes('causal mask')
            || lower.includes('attention mask')
        ) {
            const link = (
                Number.isFinite(layerIndex)
                && Number.isFinite(headIndex)
                && Number.isFinite(tokenContext.queryTokenIndex)
                && Number.isFinite(tokenContext.keyTokenIndex)
            )
                ? {
                    mode: 'pre',
                    layerIndex: Math.floor(layerIndex),
                    headIndex: Math.floor(headIndex),
                    sourceTokenIndex: tokenContext.queryTokenIndex,
                    targetTokenIndex: tokenContext.keyTokenIndex
                }
                : null;
            return this._buildTransformerView2dCausalMaskSelection(fallbackSelection, link)
                || fallbackSelection;
        }

        if (
            isAttentionScoreSelection(label, fallbackSelection)
            && Number.isFinite(layerIndex)
            && Number.isFinite(headIndex)
            && Number.isFinite(tokenContext.queryTokenIndex)
            && Number.isFinite(tokenContext.keyTokenIndex)
        ) {
            const link = {
                mode: resolveAttentionModeFromSelection(fallbackSelection) === 'post' ? 'post' : 'pre',
                layerIndex: Math.floor(layerIndex),
                headIndex: Math.floor(headIndex),
                sourceTokenIndex: tokenContext.queryTokenIndex,
                targetTokenIndex: tokenContext.keyTokenIndex
            };
            return this._findAttentionScoreSceneSelection(link)
                || this._buildFallbackAttentionScoreSelection(link);
        }

        const layerNormParamSpec = resolveLayerNormParamPreviewSpec(label, fallbackSelection);
        if (layerNormParamSpec) {
            return preferFallbackIfBroadVector(
                this._findLayerNormParamSceneSelection(layerNormParamSpec)
                || sceneLabelFallbackSelection()
            )
                || this._buildFallbackLayerNormParamSelection(layerNormParamSpec);
        }

        const isLayerNormModuleSelection = (
            isLayerNormLabel(label, fallbackSelection)
            && !lower.includes('vector')
            && !lower.includes('scale')
            && !lower.includes('shift')
        );
        if (isLayerNormModuleSelection) {
            const layerNormKind = this._inferLayerNormKindFromSelection(fallbackSelection);
            return this._findLayerNormSceneSelection({
                layerNormKind,
                layerIndex
            }) || sceneLabelFallbackSelection() || this._buildFallbackLayerNormSelection({
                layerNormKind,
                layerIndex
            });
        }

        if (lower.includes('query weight matrix')) {
            return this._findQkvWeightMatrixSceneSelection({
                matrixKind: 'Q',
                headIndex,
                layerIndex
            }) || sceneLabelFallbackSelection() || this._buildFallbackQkvWeightMatrixSelection({
                matrixKind: 'Q',
                headIndex,
                layerIndex
            });
        }
        if (lower.includes('key weight matrix')) {
            return this._findQkvWeightMatrixSceneSelection({
                matrixKind: 'K',
                headIndex,
                layerIndex
            }) || sceneLabelFallbackSelection() || this._buildFallbackQkvWeightMatrixSelection({
                matrixKind: 'K',
                headIndex,
                layerIndex
            });
        }
        if (lower.includes('value weight matrix')) {
            return this._findQkvWeightMatrixSceneSelection({
                matrixKind: 'V',
                headIndex,
                layerIndex
            }) || sceneLabelFallbackSelection() || this._buildFallbackQkvWeightMatrixSelection({
                matrixKind: 'V',
                headIndex,
                layerIndex
            });
        }

        if (stageLower === 'attention.weighted_value' || lower.includes('weighted value vector')) {
            return this._findGenericSceneVectorSelection({
                activationStages: ['attention.weighted_value'],
                layerIndex,
                headIndex,
                tokenIndex: tokenContext.tokenIndex,
                tokenId: tokenContext.tokenId,
                tokenLabel: tokenContext.tokenLabel,
                defaultLabel: 'Weighted Value Vector'
            }) || sceneLabelFallbackSelection() || fallbackSelection;
        }

        if (
            stageLower === 'qkv.q'
            || stageLower === 'qkv.k'
            || stageLower === 'qkv.v'
            || lower.includes('query vector')
            || lower.includes('key vector')
            || lower.includes('value vector')
            || lower.includes('cached key vector')
            || lower.includes('cached value vector')
        ) {
            const cachedKvSelection = isKvCacheVectorSelection(fallbackSelection)
                || lower.includes('cached key vector')
                || lower.includes('cached value vector');
            const vectorKind = (
                stageLower === 'qkv.k'
                || lower.includes('cached key vector')
                || lower.includes('key vector')
            )
                ? 'K'
                : (
                    stageLower === 'qkv.v'
                    || lower.includes('cached value vector')
                    || lower.includes('value vector')
                )
                    ? 'V'
                    : 'Q';
            return this._findRuntimeAttentionVectorSelection({
                vectorKind,
                layerIndex,
                headIndex,
                tokenIndex: tokenContext.tokenIndex,
                tokenId: tokenContext.tokenId,
                tokenLabel: tokenContext.tokenLabel
            }) || this._findAttentionVectorSceneSelection({
                vectorKind,
                layerIndex,
                headIndex,
                tokenIndex: tokenContext.tokenIndex,
                tokenId: tokenContext.tokenId,
                tokenLabel: tokenContext.tokenLabel
            }) || sceneLabelFallbackSelection() || this._buildFallbackAttentionVectorSelection({
                vectorKind,
                layerIndex,
                headIndex,
                tokenIndex: tokenContext.tokenIndex,
                tokenId: tokenContext.tokenId,
                tokenLabel: tokenContext.tokenLabel,
                cachedKv: cachedKvSelection
            });
        }

        if (normalizedPostLayerNormStage === 'ln1.output' || normalizedPostLayerNormSourceStage === 'ln1.output') {
            return preferFallbackIfBroadVector(this._findRuntimeQkvSourceVectorSelection({
                layerIndex,
                headIndex,
                tokenIndex: tokenContext.tokenIndex,
                tokenId: tokenContext.tokenId,
                tokenLabel: tokenContext.tokenLabel
            }) || this._findRuntimeResidualVectorSelection({
                stage: 'ln1.output',
                layerIndex,
                tokenIndex: tokenContext.tokenIndex,
                tokenId: tokenContext.tokenId,
                tokenLabel: tokenContext.tokenLabel
            }) || this._findQkvSourceVectorSceneSelection({
                layerIndex,
                headIndex,
                tokenIndex: tokenContext.tokenIndex,
                tokenId: tokenContext.tokenId,
                tokenLabel: tokenContext.tokenLabel
            }) || sceneLabelFallbackSelection() || this._buildFallbackQkvSourceVectorSelection({
                layerIndex,
                tokenIndex: tokenContext.tokenIndex,
                tokenId: tokenContext.tokenId,
                tokenLabel: tokenContext.tokenLabel
            }));
        }

        if (normalizedPostLayerNormStage === 'ln2.output' || normalizedPostLayerNormSourceStage === 'ln2.output') {
            return preferFallbackIfBroadVector(this._findRuntimeResidualVectorSelection({
                stage: 'ln2.output',
                layerIndex,
                tokenIndex: tokenContext.tokenIndex,
                tokenId: tokenContext.tokenId,
                tokenLabel: tokenContext.tokenLabel
            }) || this._findGenericSceneVectorSelection({
                activationStages: ['ln2.output', 'ln2.shift'],
                layerIndex,
                headIndex,
                tokenIndex: tokenContext.tokenIndex,
                tokenId: tokenContext.tokenId,
                tokenLabel: tokenContext.tokenLabel,
                defaultLabel: label
            }) || sceneLabelFallbackSelection() || fallbackSelection);
        }

        const layerNormProductStage = liveVectorStages.find((stage) => (
            Boolean(normalizeLayerNormProductStage(stage))
        )) || '';
        if (layerNormProductStage) {
            const normalizedProductStage = normalizeLayerNormProductStage(layerNormProductStage);
            const legacyProductStage = normalizeLayerNormProductStage(layerNormProductStage, {
                preferLegacy: true
            });
            return this._findGenericSceneVectorSelection({
                activationStages: [...new Set([
                    legacyProductStage,
                    normalizedProductStage
                ].filter(Boolean))],
                layerIndex,
                headIndex,
                tokenIndex: tokenContext.tokenIndex,
                tokenId: tokenContext.tokenId,
                tokenLabel: tokenContext.tokenLabel,
                defaultLabel: label,
                allowGenericObjectFallback: false
            }) || fallbackSelection;
        }

        const residualStreamStage = liveVectorStages.find((stage) => (
            stage === 'layer.incoming'
            || stage === 'residual.post_attention'
            || stage === 'residual.post_mlp'
        )) || '';
        if (residualStreamStage) {
            return this._findGenericSceneVectorSelection({
                activationStages: [residualStreamStage],
                layerIndex,
                headIndex,
                tokenIndex: tokenContext.tokenIndex,
                tokenId: tokenContext.tokenId,
                tokenLabel: tokenContext.tokenLabel,
                defaultLabel: label,
                allowGenericObjectFallback: false
            }) || fallbackSelection;
        }

        const genericMatrixSelection = (
            lower === 'output projection matrix'
            || lower === 'output projection bias vector'
            || lower === 'query bias vector'
            || lower === 'key bias vector'
            || lower === 'value bias vector'
            || lower === 'mlp up weight matrix'
            || lower === 'bias vector for mlp up matrix'
            || lower === 'mlp down weight matrix'
            || lower === 'bias vector · mlp down'
        );
        if (genericMatrixSelection) {
            return this._findGenericSceneObjectSelection({
                label,
                layerIndex,
                headIndex,
                kind: lower.includes('matrix') ? 'matrix' : 'vector',
                fallbackLabel: label
            }) || sceneLabelFallbackSelection() || fallbackSelection;
        }

        const mlpMiddleVectorStage = liveVectorStages.find((stage) => (
            stage === 'mlp.up'
            || stage === 'mlp.activation'
        )) || '';
        const mlpMiddleVectorSelection = (
            lower.includes('mlp up projection')
            || lower.includes('mlp activation')
            || lower.includes('mlp expanded segments')
            || mlpMiddleVectorStage === 'mlp.up'
            || mlpMiddleVectorStage === 'mlp.activation'
        );
        if (mlpMiddleVectorSelection) {
            return this._findRuntimeMlpMiddleVectorSelection({
                stage: mlpMiddleVectorStage || (lower.includes('mlp activation') ? 'mlp.activation' : 'mlp.up'),
                label,
                layerIndex,
                tokenIndex: tokenContext.tokenIndex,
                tokenId: tokenContext.tokenId,
                tokenLabel: tokenContext.tokenLabel
            }) || this._findGenericSceneVectorSelection({
                activationStages: mlpMiddleVectorStage ? [mlpMiddleVectorStage] : liveVectorStages,
                layerIndex,
                headIndex,
                tokenIndex: tokenContext.tokenIndex,
                tokenId: tokenContext.tokenId,
                tokenLabel: tokenContext.tokenLabel,
                defaultLabel: label
            }) || sceneLabelFallbackSelection() || fallbackSelection;
        }

        if (stageLower === 'mlp.down' || lower.startsWith('mlp down projection')) {
            return this._findRuntimeMlpDownVectorSelection({
                layerIndex,
                tokenIndex: tokenContext.tokenIndex,
                tokenId: tokenContext.tokenId,
                tokenLabel: tokenContext.tokenLabel
            }) || this._findGenericSceneVectorSelection({
                activationStages: ['mlp.down'],
                layerIndex,
                headIndex,
                tokenIndex: tokenContext.tokenIndex,
                tokenId: tokenContext.tokenId,
                tokenLabel: tokenContext.tokenLabel,
                defaultLabel: label
            }) || sceneLabelFallbackSelection() || this._buildFallbackMlpDownVectorSelection({
                layerIndex,
                tokenIndex: tokenContext.tokenIndex,
                tokenId: tokenContext.tokenId,
                tokenLabel: tokenContext.tokenLabel
            });
        }

        const genericVectorSelection = (
            isResidualVectorSelection(label, fallbackSelection)
            || isWeightedSumSelection(label, fallbackSelection)
            || liveVectorStages.includes('attention.weighted_sum')
            || liveVectorStages.includes('attention.output_projection')
            || liveVectorStages.includes('layer.incoming')
            || liveVectorStages.includes('residual.post_attention')
            || liveVectorStages.includes('residual.post_mlp')
            || liveVectorStages.some((stage) => Boolean(normalizePostLayerNormResidualStage(stage)))
            || liveVectorStages.includes('ln1.scale')
            || liveVectorStages.includes('ln2.scale')
            || liveVectorStages.includes('final_ln.scale')
            || liveVectorStages.includes('mlp.up')
            || liveVectorStages.includes('mlp.activation')
            || liveVectorStages.includes('mlp.down')
            || liveVectorStages.includes('final_ln.norm')
            || liveVectorStages.includes('ln1.norm')
            || liveVectorStages.includes('ln2.norm')
            || liveVectorStages.includes('final_ln.input')
            || liveVectorStages.includes('final_ln.product')
            || liveVectorStages.includes('final_ln.output')
            || liveVectorStages.includes('attention.concatenate')
        );
        if (genericVectorSelection) {
            const resolvedSelection = this._findGenericSceneVectorSelection({
                activationStages: liveVectorStages,
                layerIndex,
                headIndex,
                tokenIndex: tokenContext.tokenIndex,
                tokenId: tokenContext.tokenId,
                tokenLabel: tokenContext.tokenLabel,
                defaultLabel: label
            }) || sceneLabelFallbackSelection();
            const hasLayerNormVectorStage = liveVectorStages.some((stage) => (
                stage.startsWith('ln1.')
                || stage.startsWith('ln2.')
                || stage.startsWith('final_ln.')
            ));
            return (hasLayerNormVectorStage
                ? preferFallbackIfBroadVector(resolvedSelection)
                : resolvedSelection)
                || fallbackSelection;
        }

        return sceneLabelFallbackSelection() || fallbackSelection;
    }

    _openTransformerView2dCanvasSelection(selection = null) {
        const resolvedSelection = this._resolveTransformerView2dCanvasSelection(selection);
        if (!resolvedSelection?.label) return false;
        this.showSelection(resolvedSelection, {
            scrollPanelToTop: false,
            preserveTransformerView2d: true
        });
        return true;
    }

    _openLayerNormFromAction(actionEl) {
        const payload = this._parseDetailActionPayload(actionEl);
        const payloadKind = typeof payload?.layerNormKind === 'string' ? payload.layerNormKind : null;
        const layerNormKind = (payloadKind === 'ln1' || payloadKind === 'ln2' || payloadKind === 'final')
            ? payloadKind
            : this._inferLayerNormKindFromSelection(this._lastSelection);
        const payloadLayerIndex = Number(payload?.layerIndex);
        const selectionLayerIndex = findUserDataNumber(this._lastSelection, 'layerIndex');
        const resolvedLayerIndex = Number.isFinite(payloadLayerIndex)
            ? Math.floor(payloadLayerIndex)
            : (Number.isFinite(selectionLayerIndex) ? Math.floor(selectionLayerIndex) : null);

        const selection = this._findLayerNormSceneSelection({
            layerNormKind,
            layerIndex: resolvedLayerIndex
        }) || this._buildFallbackLayerNormSelection({
            layerNormKind,
            layerIndex: resolvedLayerIndex
        });

        if (!selection) return false;
        this.showSelection(selection, { scrollPanelToTop: true });
        return true;
    }

    _openAttentionVectorFromAction(actionEl) {
        const payload = this._parseDetailActionPayload(actionEl);
        const vectorKind = normalizeQkvActionKind(payload?.vectorKind);
        const layerIndex = Number(payload?.layerIndex);
        const headIndex = Number(payload?.headIndex);
        const tokenIndex = Number(payload?.tokenIndex);
        const tokenId = Number(payload?.tokenId);
        const tokenLabel = typeof payload?.tokenLabel === 'string' ? payload.tokenLabel : '';

        const selection = this._findAttentionVectorSceneSelection({
            vectorKind,
            layerIndex: Number.isFinite(layerIndex) ? Math.floor(layerIndex) : null,
            headIndex: Number.isFinite(headIndex) ? Math.floor(headIndex) : null,
            tokenIndex: Number.isFinite(tokenIndex) ? Math.floor(tokenIndex) : null,
            tokenId: Number.isFinite(tokenId) ? Math.floor(tokenId) : null,
            tokenLabel
        }) || this._buildFallbackAttentionVectorSelection({
            vectorKind,
            layerIndex: Number.isFinite(layerIndex) ? Math.floor(layerIndex) : null,
            headIndex: Number.isFinite(headIndex) ? Math.floor(headIndex) : null,
            tokenIndex: Number.isFinite(tokenIndex) ? Math.floor(tokenIndex) : null,
            tokenId: Number.isFinite(tokenId) ? Math.floor(tokenId) : null,
            tokenLabel
        });

        if (!selection) return false;
        this.showSelection(selection, { scrollPanelToTop: true });
        return true;
    }

    _openMhsaInfoFromAction() {
        const sourceSelection = this._lastSelection;
        const layerIndex = findUserDataNumber(sourceSelection, 'layerIndex');
        const headIndex = findUserDataNumber(sourceSelection, 'headIndex');
        const mhsaSelection = buildMhsaInfoSelection({
            layerIndex: Number.isFinite(layerIndex) ? Math.floor(layerIndex) : null,
            headIndex: Number.isFinite(headIndex) ? Math.floor(headIndex) : null
        });
        this.showSelection(mhsaSelection, { scrollPanelToTop: true });
        return true;
    }

    _openQkvSourceVectorFromAction(actionEl) {
        const payload = this._parseDetailActionPayload(actionEl);
        const layerIndex = Number(payload?.layerIndex);
        const tokenIndex = Number(payload?.tokenIndex);
        const tokenId = Number(payload?.tokenId);
        const tokenLabel = typeof payload?.tokenLabel === 'string' ? payload.tokenLabel : '';

        const selection = this._findQkvSourceVectorSceneSelection({
            layerIndex: Number.isFinite(layerIndex) ? Math.floor(layerIndex) : null,
            tokenIndex: Number.isFinite(tokenIndex) ? Math.floor(tokenIndex) : null,
            tokenId: Number.isFinite(tokenId) ? Math.floor(tokenId) : null,
            tokenLabel
        }) || this._buildFallbackQkvSourceVectorSelection({
            layerIndex: Number.isFinite(layerIndex) ? Math.floor(layerIndex) : null,
            tokenIndex: Number.isFinite(tokenIndex) ? Math.floor(tokenIndex) : null,
            tokenId: Number.isFinite(tokenId) ? Math.floor(tokenId) : null,
            tokenLabel
        });

        if (!selection) return false;
        this.showSelection(selection, { scrollPanelToTop: true });
        return true;
    }

    _openQkvWeightMatrixFromAction(actionEl) {
        const payload = this._parseDetailActionPayload(actionEl);
        const matrixKind = normalizeQkvActionKind(payload?.matrixKind);
        const headIndex = Number(payload?.headIndex);
        const layerIndex = Number(payload?.layerIndex);

        const selection = this._findQkvWeightMatrixSceneSelection({
            matrixKind,
            headIndex: Number.isFinite(headIndex) ? Math.floor(headIndex) : null,
            layerIndex: Number.isFinite(layerIndex) ? Math.floor(layerIndex) : null
        }) || this._buildFallbackQkvWeightMatrixSelection({
            matrixKind,
            headIndex: Number.isFinite(headIndex) ? Math.floor(headIndex) : null,
            layerIndex: Number.isFinite(layerIndex) ? Math.floor(layerIndex) : null
        });

        if (!selection) return false;
        this.showSelection(selection, { scrollPanelToTop: true });
        return true;
    }

    _openLayerNormParamFromAction(actionEl) {
        const payload = this._parseDetailActionPayload(actionEl);
        const param = String(payload?.param || '').toLowerCase() === 'shift' ? 'shift' : 'scale';
        const payloadKind = typeof payload?.layerNormKind === 'string' ? payload.layerNormKind : null;
        const layerNormKind = (payloadKind === 'ln1' || payloadKind === 'ln2' || payloadKind === 'final')
            ? payloadKind
            : this._inferLayerNormKindFromSelection(this._lastSelection);
        const payloadLayerIndex = Number(payload?.layerIndex);
        const selectionLayerIndex = findUserDataNumber(this._lastSelection, 'layerIndex');
        const resolvedLayerIndex = Number.isFinite(payloadLayerIndex)
            ? Math.floor(payloadLayerIndex)
            : (Number.isFinite(selectionLayerIndex) ? Math.floor(selectionLayerIndex) : null);
        const finalLayerIndex = layerNormKind === 'final' && !Number.isFinite(resolvedLayerIndex)
            ? (Array.isArray(this.engine?._layers) && this.engine._layers.length
                ? this.engine._layers.length - 1
                : null)
            : resolvedLayerIndex;

        const selection = this._findLayerNormParamSceneSelection({
            layerNormKind,
            param,
            layerIndex: finalLayerIndex
        }) || this._buildFallbackLayerNormParamSelection({
            layerNormKind,
            param,
            layerIndex: finalLayerIndex
        });

        if (!selection) return false;
        this.showSelection(selection, { scrollPanelToTop: true });
        return true;
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
        if (action === SOFTMAX_PANEL_ACTION_OPEN) {
            return this._openSoftmaxDetailPreview();
        }
        if (action === TRANSFORMER_VIEW2D_PANEL_ACTION_OPEN) {
            const openResult = this._openTransformerView2dPreview();
            if (openResult && typeof openResult.then === 'function') {
                void openResult.catch((error) => {
                    console.error('Failed to open transformer 2D view:', error);
                });
            }
            return true;
        }
        if (action === MHSA_INFO_PANEL_ACTION_OPEN) {
            return this._openMhsaInfoFromAction();
        }
        if (action === ATTENTION_VECTOR_PANEL_ACTION_OPEN) {
            return this._openAttentionVectorFromAction(actionEl);
        }
        if (action === LAYERNORM_PANEL_ACTION_OPEN) {
            return this._openLayerNormFromAction(actionEl);
        }
        if (action === LAYERNORM_PARAM_PANEL_ACTION_OPEN) {
            return this._openLayerNormParamFromAction(actionEl);
        }
        if (action === QKV_SOURCE_VECTOR_PANEL_ACTION_OPEN) {
            return this._openQkvSourceVectorFromAction(actionEl);
        }
        if (action === QKV_WEIGHT_MATRIX_PANEL_ACTION_OPEN) {
            return this._openQkvWeightMatrixFromAction(actionEl);
        }
        return false;
    }

    _openGeluDetailPreview({ fromHistory = false, sourceSelection = null } = {}) {
        const resolvedSelection = sourceSelection || this._lastSelection;
        const sourceLabel = this._lastSelectionLabel;
        const resolvedLabel = resolvedSelection?.label
            ? normalizeSelectionLabel(resolvedSelection.label, resolvedSelection)
            : sourceLabel;
        const activationStage = getActivationDataFromSelection(resolvedSelection)?.stage || '';
        if (!resolvedSelection || !isGeluDetailSelection({ label: resolvedLabel, stage: activationStage })) return;
        this._geluSourceSelection = resolvedSelection;
        this._geluDetailOpen = true;
        this.panel.classList.add('is-gelu-view-open');
        this._setTitleText('GELU Activation');
        if (this.subtitle) {
            this.subtitle.classList.remove('detail-subtitle--qkv-token-context');
            this.subtitle.textContent = 'Gaussian Error Linear Unit used in GPT-2 MLP blocks';
        }
        this._setSubtitleSecondaryText('');
        this._setSubtitleTertiaryText('');
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

    _openSoftmaxDetailPreview({ fromHistory = false, sourceSelection = null } = {}) {
        const resolvedSelection = sourceSelection || this._lastSelection;
        const sourceLabel = this._lastSelectionLabel;
        const resolvedLabel = resolvedSelection?.label
            ? normalizeSelectionLabel(resolvedSelection.label, resolvedSelection)
            : sourceLabel;
        if (!resolvedSelection || !isPostSoftmaxAttentionSelection(resolvedSelection, resolvedLabel)) return false;

        const softmaxContext = resolveSoftmaxPreviewContext(resolvedSelection, this.activationSource);
        if (!softmaxContext) return false;

        this._softmaxSourceSelection = resolvedSelection;
        this._softmaxDetailOpen = true;
        this.panel.classList.add('is-softmax-view-open');
        this._setTitleText('Softmax');
        if (this.subtitle) {
            this.subtitle.classList.remove('detail-subtitle--qkv-token-context');
            this.subtitle.textContent = 'Turns raw scores into normalized weights';
        }
        this._setSubtitleSecondaryText('');
        this._setSubtitleTertiaryText('Why the model uses it in attention.');
        this._softmaxDetailView?.setContext(softmaxContext);
        this._softmaxDetailView?.setVisible(true);
        this._softmaxDetailView?.resizeAndRender();
        this._stopLoop();
        this._setAttentionVisibility(false);
        this._setPanelTokenHoverEntry(null, { emit: true });
        if (!fromHistory) {
            const entry = this._buildHistoryEntry('softmax', resolvedSelection);
            this._pushHistoryEntry(entry);
        } else {
            this._updateHistoryNavigationControls();
        }
        return true;
    }

    async _ensureTransformerView2dDetailView() {
        if (this._transformerView2dDetailView) {
            return this._transformerView2dDetailView;
        }
        if (this._transformerView2dDetailViewPromise) {
            return this._transformerView2dDetailViewPromise;
        }

        this._transformerView2dDetailViewPromise = loadTransformerView2dDetailViewModule()
            .then(({ createTransformerView2dDetailView }) => {
                if (this._transformerView2dDetailView) {
                    return this._transformerView2dDetailView;
                }
                const detailView = createTransformerView2dDetailView(this.panel, {
                    onExitTo3d: () => {
                        this.close({ clearHistory: false });
                    },
                    onOpenSelection: (selection) => {
                        return this._openTransformerView2dCanvasSelection(selection);
                    },
                    onCloseSelection: () => {
                        return this._closeTransformerView2dSelectionSidebar();
                    }
                });
                this._transformerView2dDetailView = detailView;
                return detailView;
            })
            .finally(() => {
                this._transformerView2dDetailViewPromise = null;
            });

        return this._transformerView2dDetailViewPromise;
    }

    openTransformerView2d({
        semanticTarget = null,
        focusLabel = '',
        detailSemanticTargets = null,
        detailFocusLabel = '',
        detailInteractionTargets = null,
        transitionMode = '',
        syncRoute = true
    } = {}) {
        return this._openTransformerView2dPreview({
            sourceSelection: null,
            syncRoute,
            view2dContext: {
                semanticTarget,
                focusLabel: String(focusLabel || '').trim() || describeTransformerView2dTarget(semanticTarget),
                detailSemanticTargets,
                detailFocusLabel: String(detailFocusLabel || '').trim(),
                detailInteractionTargets,
                transitionMode: String(transitionMode || '').trim(),
                actionLabel: 'View in 2D / matrix form'
            }
        });
    }

    _getTransformerView2dSelectionSidebarSections() {
        return [
            this.previewRoot,
            this.vectorLegend,
            this.equationsSection,
            this.promptContextRow,
            this.previewMetaSection,
            this.description,
            this.metaSection,
            this.dataSection,
            this.attentionRoot,
            this.copyContextRow
        ].filter(Boolean);
    }

    _ensureTransformerView2dSelectionSidebarDockRecords() {
        if (this._transformerView2dSelectionSidebarDockRecords.length) {
            return this._transformerView2dSelectionSidebarDockRecords;
        }
        const seen = new Set();
        this._transformerView2dSelectionSidebarDockRecords = this._getTransformerView2dSelectionSidebarSections()
            .filter((node) => {
                if (!node || seen.has(node)) return false;
                seen.add(node);
                return true;
            })
            .map((node) => ({
                node,
                parent: node.parentNode || null,
                nextSibling: node.nextSibling || null
            }));
        return this._transformerView2dSelectionSidebarDockRecords;
    }

    _dockTransformerView2dSelectionSidebarSections() {
        const sidebarBody = this._transformerView2dDetailView?.getSelectionSidebarBody?.() || null;
        if (!sidebarBody) return false;
        const records = this._ensureTransformerView2dSelectionSidebarDockRecords();
        records.forEach(({ node }) => {
            if (!node || node.parentNode === sidebarBody) return;
            sidebarBody.appendChild(node);
        });
        this._transformerView2dSelectionSidebarDocked = true;
        return records.length > 0;
    }

    _restoreTransformerView2dSelectionSidebarSections() {
        const records = this._transformerView2dSelectionSidebarDockRecords;
        if (!records.length) {
            this._transformerView2dSelectionSidebarDocked = false;
            return false;
        }
        records.forEach(({ node, parent, nextSibling }) => {
            if (!node || !parent) return;
            if (nextSibling && nextSibling.parentNode === parent) {
                parent.insertBefore(node, nextSibling);
                return;
            }
            parent.appendChild(node);
        });
        this._transformerView2dSelectionSidebarDocked = false;
        return true;
    }

    _buildTransformerView2dSelectionSidebarHeaderContent() {
        const buildClassName = (baseClass, sourceEl, fallbackClass) => {
            const sourceClass = typeof sourceEl?.className === 'string'
                ? sourceEl.className.trim()
                : '';
            return [baseClass, sourceClass || fallbackClass].filter(Boolean).join(' ');
        };
        return {
            titleHtml: this.title?.innerHTML || '',
            titleClassName: buildClassName(
                'detail-transformer-view2d-selection-sidebar-title',
                this.title,
                'detail-title'
            ),
            subtitleHtml: this.subtitle?.innerHTML || '',
            subtitleClassName: buildClassName(
                'detail-transformer-view2d-selection-sidebar-subtitle',
                this.subtitle,
                'detail-subtitle'
            ),
            subtitleSecondaryHtml: this.subtitleSecondary?.innerHTML || '',
            subtitleSecondaryClassName: buildClassName(
                'detail-transformer-view2d-selection-sidebar-subtitle',
                this.subtitleSecondary,
                'detail-subtitle'
            ),
            subtitleTertiaryHtml: this.subtitleTertiary?.innerHTML || '',
            subtitleTertiaryClassName: buildClassName(
                'detail-transformer-view2d-selection-sidebar-subtitle',
                this.subtitleTertiary,
                'detail-subtitle'
            )
        };
    }

    _syncTransformerView2dSelectionSidebarHeader(headerContent = null) {
        const detailView = this._transformerView2dDetailView;
        if (!detailView?.setSelectionSidebarHeaderContent) return;
        detailView.setSelectionSidebarHeaderContent(
            headerContent || this._buildTransformerView2dSelectionSidebarHeaderContent()
        );
    }

    _showTransformerView2dSelectionSidebar({ scrollToTop = true } = {}) {
        if (this._transformerView2dSelectionSidebarRestoreTimer !== null) {
            clearTimeout(this._transformerView2dSelectionSidebarRestoreTimer);
            this._transformerView2dSelectionSidebarRestoreTimer = null;
        }
        if (!this._dockTransformerView2dSelectionSidebarSections()) return false;
        this._syncTransformerView2dSelectionSidebarHeader();
        this._transformerView2dDetailView?.setSelectionSidebarVisible?.(true);
        if (scrollToTop) {
            this._transformerView2dDetailView?.scrollSelectionSidebarToTop?.();
        }
        this._scheduleSelectionEquationFit();
        this._scheduleDimensionLabelFit();
        if (this.isOpen) {
            this._onResize();
            this._renderPreviewSnapshot();
            this._startLoop();
        } else {
            this._scheduleResize();
        }
        this._scheduleResize();
        return true;
    }

    _closeTransformerView2dSelectionSidebar({
        restoreSections = false,
        restartLoop = false
    } = {}) {
        const sidebarVisible = this._transformerView2dDetailView?.isSelectionSidebarVisible?.() === true;
        if (!restoreSections && !sidebarVisible) {
            return false;
        }
        this._transformerView2dDetailView?.clearSelectionLock?.({
            scheduleRender: false
        });
        this._transformerView2dDetailView?.setSelectionSidebarVisible?.(false);
        if (restoreSections) {
            if (this._transformerView2dSelectionSidebarRestoreTimer !== null) {
                clearTimeout(this._transformerView2dSelectionSidebarRestoreTimer);
                this._transformerView2dSelectionSidebarRestoreTimer = null;
            }
            this._restoreTransformerView2dSelectionSidebarSections();
        } else if (this._isSmallScreen()) {
            if (this._transformerView2dSelectionSidebarRestoreTimer !== null) {
                clearTimeout(this._transformerView2dSelectionSidebarRestoreTimer);
            }
            this._transformerView2dSelectionSidebarRestoreTimer = setTimeout(() => {
                this._transformerView2dSelectionSidebarRestoreTimer = null;
                if (this._transformerView2dDetailView?.isSelectionSidebarVisible?.()) return;
                this._restoreTransformerView2dSelectionSidebarSections();
                this._scheduleResize();
            }, 240);
        }
        if (restartLoop && this.isOpen) {
            this._startLoop();
        } else {
            this._stopLoop();
        }
        this._scheduleResize();
        return true;
    }

    _openTransformerView2dPreview(args = {}) {
        if (this._transformerView2dDetailView) {
            return this._openTransformerView2dPreviewWithDetailView(this._transformerView2dDetailView, args);
        }
        return this._ensureTransformerView2dDetailView().then((detailView) => {
            return this._openTransformerView2dPreviewWithDetailView(detailView, args);
        });
    }

    _openTransformerView2dPreviewWithDetailView(detailView, {
        fromHistory = false,
        sourceSelection = null,
        view2dContext = null,
        syncRoute = true
    } = {}) {
        const resolvedSelection = sourceSelection || this._lastSelection;
        const sourceLabel = this._lastSelectionLabel;
        const resolvedLabel = resolvedSelection?.label
            ? normalizeSelectionLabel(resolvedSelection.label, resolvedSelection)
            : sourceLabel;
        const resolvedView2dContext = view2dContext
            || resolveTransformerView2dActionContext(resolvedSelection, resolvedLabel);
        if (!resolvedView2dContext) return false;
        const resolvedDetailInteractionTargets = resolvedView2dContext.detailInteractionTargets
            ? normalizeTransformerView2dDetailInteractionTargets(
                resolvedView2dContext.detailInteractionTargets
            )
            : resolveTransformerView2dDetailInteractionTargets(
                resolvedSelection,
                resolvedView2dContext.detailSemanticTargets
            );
        const hydratedView2dContext = {
            ...resolvedView2dContext,
            ...(resolvedDetailInteractionTargets.length
                ? { detailInteractionTargets: resolvedDetailInteractionTargets }
                : {})
        };
        const shouldAutoOpenSelectionSidebar = !!(
            resolvedSelection
            && !(this._isSmallScreen && this._isSmallScreen())
        );
        const selectionSidebarHeaderContent = resolvedSelection
            ? this._buildTransformerView2dSelectionSidebarHeaderContent()
            : null;

        this._closeTransformerView2dSelectionSidebar({
            restoreSections: true,
            restartLoop: false
        });
        this._isMhsaInfoSelectionActive = false;
        this._syncMhsaViewRoute(false);
        this._setInfoPreview(null);
        this._transformerView2dSourceSelection = resolvedSelection || null;
        this._currentTransformerView2dContext = hydratedView2dContext;
        this._transformerView2dDetailOpen = true;
        this.panel.classList.add('is-transformer-view2d-open');
        this._setTitleText('2D Transformer Canvas');
        if (this.subtitle) {
            this.subtitle.classList.remove('detail-subtitle--qkv-token-context');
            this.subtitle.textContent = 'Scalable semantic canvas for the current transformer state';
        }
        const resolvedCanvasFocusLabel = String(
            hydratedView2dContext.detailFocusLabel
            || hydratedView2dContext.focusLabel
            || ''
        ).trim();
        this._setSubtitleSecondaryText(`Focus: ${resolvedCanvasFocusLabel}`);
        this._setSubtitleTertiaryText('Prototype view. Drag or use one finger to pan, and pinch or scroll to zoom.');
        this.open();
        this._setHoverLabelSuppression(true);
        detailView?.setVisible(true);
        detailView?.open({
            activationSource: this.activationSource,
            tokenIndices: Array.isArray(this.attentionTokenIndices) ? this.attentionTokenIndices : this.laneTokenIndices,
            tokenLabels: Array.isArray(this.attentionTokenLabels) ? this.attentionTokenLabels : this.tokenLabels,
            semanticTarget: hydratedView2dContext.semanticTarget,
            focusLabel: hydratedView2dContext.focusLabel,
            initialOverviewSelectionLockTarget: hydratedView2dContext.initialOverviewSelectionLockTarget,
            detailSemanticTargets: hydratedView2dContext.detailSemanticTargets,
            detailFocusLabel: hydratedView2dContext.detailFocusLabel,
            detailInteractionTargets: hydratedView2dContext.detailInteractionTargets,
            transitionMode: hydratedView2dContext.transitionMode,
            initialSelectionSidebarVisible: shouldAutoOpenSelectionSidebar,
            isSmallScreen: this._isSmallScreen && this._isSmallScreen()
        });
        if (shouldAutoOpenSelectionSidebar) {
            this._showTransformerView2dSelectionSidebar({ scrollToTop: true });
            if (selectionSidebarHeaderContent) {
                detailView?.setSelectionSidebarHeaderContent?.(
                    selectionSidebarHeaderContent
                );
            }
        }
        const shouldKeepSelectionPreviewLoop = !!(
            resolvedSelection
            && this.currentPreview
            && this._transformerView2dDetailView?.isSelectionSidebarVisible?.() === true
        );
        this.engine?.pause?.(TRANSFORMER_VIEW2D_PAUSE_REASON);
        if (shouldKeepSelectionPreviewLoop) {
            this._startLoop();
        } else {
            this._stopLoop();
        }
        this._setAttentionVisibility(false);
        this._setPanelTokenHoverEntry(null, { emit: true });
        if (syncRoute) {
            syncTransformerView2dRoute({
                active: true,
                semanticTarget: hydratedView2dContext.semanticTarget
            });
        }
        if (!fromHistory && resolvedSelection) {
            const entry = this._buildHistoryEntry('transformer-view2d', resolvedSelection);
            this._pushHistoryEntry(entry);
        } else {
            this._updateHistoryNavigationControls();
        }
        if (this._canToggleMhsaFullscreen()) {
            this._setMhsaFullscreen(true);
        }
        return true;
    }

    _closeTransformerView2dPreview({ restoreSelection = false, restartLoop = true } = {}) {
        if (!this._transformerView2dDetailOpen) return false;
        this._closeTransformerView2dSelectionSidebar({
            restoreSections: true,
            restartLoop: false
        });
        this._transformerView2dDetailOpen = false;
        this.panel.classList.remove('is-transformer-view2d-open');
        this._transformerView2dDetailView?.setVisible(false);
        syncTransformerView2dRoute({ active: false });
        this.engine?.resume?.(TRANSFORMER_VIEW2D_PAUSE_REASON);
        this._syncHoverLabelSuppressionFromHoverState();
        if (this._mhsaFullscreenActive) {
            this._setMhsaFullscreen(false);
        }
        const sourceSelection = this._transformerView2dSourceSelection;
        this._transformerView2dSourceSelection = null;
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

    _closeSoftmaxDetailPreview({ restoreSelection = false, restartLoop = true } = {}) {
        if (!this._softmaxDetailOpen) return false;
        this._softmaxDetailOpen = false;
        this.panel.classList.remove('is-softmax-view-open');
        this._softmaxDetailView?.setVisible(false);
        const sourceSelection = this._softmaxSourceSelection;
        this._softmaxSourceSelection = null;
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

    _formatAttentionPanelContext(context = null) {
        const parts = [];
        const layerIndex = Number.isFinite(context?.layerIndex) ? Math.floor(context.layerIndex) : null;
        const headIndex = Number.isFinite(context?.headIndex) ? Math.floor(context.headIndex) : null;
        if (Number.isFinite(layerIndex)) parts.push(`Layer ${layerIndex + 1}`);
        if (Number.isFinite(headIndex)) parts.push(`Head ${headIndex + 1}`);
        return parts.join(', ');
    }

    _resolveActiveAttentionDisplayMode() {
        return resolveAttentionDisplayMode(this._attentionDisplayModeOverride || this.attentionMode);
    }

    _updateAttentionTitle(context = null) {
        if (!this.attentionTitle) return;
        this.attentionTitle.textContent = this._resolveActiveAttentionDisplayMode() === 'mask'
            ? 'Causal Mask'
            : 'Attention score matrix';
        if (!this.attentionContextLabel) return;
        const contextParts = [];
        const layerIndex = Number.isFinite(context?.layerIndex) ? Math.floor(context.layerIndex) : null;
        const headIndex = Number.isFinite(context?.headIndex) ? Math.floor(context.headIndex) : null;
        if (Number.isFinite(layerIndex)) contextParts.push(`Layer ${layerIndex + 1}`);
        if (Number.isFinite(headIndex)) contextParts.push(`Head ${headIndex + 1}`);
        this.attentionContextLabel.textContent = contextParts.join(' • ');
    }

    _applyAttentionCollapseState() {
        const collapsed = this._attentionSectionCollapsed === true;
        const displayMode = this._resolveActiveAttentionDisplayMode();
        const hideToggleRow = collapsed || displayMode === 'mask';
        if (this.attentionRoot) {
            this.attentionRoot.classList.toggle('is-collapsed', collapsed);
            this.attentionRoot.dataset.collapsed = collapsed ? 'true' : 'false';
        }
        if (this.attentionHeaderMeta) {
            this.attentionHeaderMeta.hidden = collapsed;
            this.attentionHeaderMeta.setAttribute('aria-hidden', collapsed ? 'true' : 'false');
        }
        if (this.attentionToggleRow) {
            this.attentionToggleRow.hidden = hideToggleRow;
            this.attentionToggleRow.setAttribute('aria-hidden', hideToggleRow ? 'true' : 'false');
        }
        if (this.attentionBody) {
            this.attentionBody.hidden = collapsed;
        }
        if (this.attentionCollapseBtn) {
            const contextLabel = this._formatAttentionPanelContext(this._attentionContext);
            const actionBase = displayMode === 'mask'
                ? (collapsed ? 'Maximize causal mask' : 'Minimize causal mask')
                : (collapsed ? 'Maximize attention score matrix' : 'Minimize attention score matrix');
            const actionLabel = contextLabel
                ? `${actionBase} for ${contextLabel}`
                : actionBase;
            this.attentionCollapseBtn.setAttribute('aria-expanded', collapsed ? 'false' : 'true');
            this.attentionCollapseBtn.setAttribute('aria-label', actionLabel);
            this.attentionCollapseBtn.title = actionLabel;
        }
        if (this.attentionCollapseLabel) {
            this.attentionCollapseLabel.textContent = collapsed ? 'Maximize' : 'Minimize';
        }
        this._updateAttentionTitle(this._attentionContext);
        if (collapsed) {
            this._hideLegendHover('attention');
        }
    }

    _setAttentionSectionCollapsed(collapsed, { persist = true } = {}) {
        const next = collapsed === true;
        this._attentionSectionCollapsed = next;
        if (persist) {
            setAttentionSectionCollapsedPreference(next);
        }
        this._applyAttentionCollapseState();
        if (
            !next
            && this.attentionRoot?.classList.contains('is-visible')
            && this._attentionContext
            && this.activationSource
        ) {
            this._renderAttentionPreview();
        }
    }

    _onAttentionCollapseClick(event) {
        event?.preventDefault?.();
        this._setAttentionSectionCollapsed(!this._attentionSectionCollapsed);
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

        const chip = document.createElement('span');
        chip.className = 'attention-value-token-chip';
        applyTokenChipColors(chip, {
            tokenLabel: safeText,
            tokenIndex: resolvedTokenIndex,
            tokenId: resolvedTokenId
        }, Number.isFinite(resolvedTokenIndex) ? resolvedTokenIndex : fallbackSeed);
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
            const rawScore = this.activationSource.getAttentionScore(
                link.layerIndex,
                link.mode,
                link.headIndex,
                link.sourceTokenIndex,
                link.targetTokenIndex
            );
            const score = resolveAttentionMatrixCellValue({
                mode: link.mode,
                value: rawScore,
                queryTokenIndex: link.sourceTokenIndex,
                keyTokenIndex: link.targetTokenIndex
            });
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

    _buildTransformerView2dCausalMaskSelection(selection = null, link = null) {
        if (!link) return null;
        const liveSelection = this._findAttentionScoreSceneSelection(link)
            || this._buildFallbackAttentionScoreSelection(link);
        if (!liveSelection?.label) return null;

        const fallbackSelection = this._buildTransformerView2dFallbackSelection(selection);
        const fallbackInfo = fallbackSelection?.info && typeof fallbackSelection.info === 'object'
            ? fallbackSelection.info
            : {};
        const fallbackActivation = fallbackInfo.activationData && typeof fallbackInfo.activationData === 'object'
            ? fallbackInfo.activationData
            : {};
        const liveInfo = liveSelection.info && typeof liveSelection.info === 'object'
            ? { ...liveSelection.info }
            : {};
        const liveActivation = liveInfo.activationData && typeof liveInfo.activationData === 'object'
            ? liveInfo.activationData
            : {};
        const label = fallbackSelection?.label || fallbackActivation.label || 'Causal Mask';

        return {
            ...liveSelection,
            label,
            kind: 'attentionSphere',
            info: {
                ...liveInfo,
                ...fallbackInfo,
                activationData: {
                    ...liveActivation,
                    ...fallbackActivation,
                    label,
                    stage: 'attention.mask'
                }
            }
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

            const partEl = document.createElement('div');
            partEl.className = 'detail-attention-context-part';

            const roleEl = document.createElement('span');
            roleEl.className = 'detail-attention-context-role';
            roleEl.textContent = `${roleText}:`;

            const chipEl = document.createElement('span');
            chipEl.className = 'detail-attention-context-chip';
            applyTokenChipColors(chipEl, {
                tokenLabel: tokenText,
                tokenIndex,
                tokenId
            }, Number.isFinite(tokenIndex) ? tokenIndex : fallbackIndex);
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

            const detailEl = document.createElement('span');
            detailEl.className = 'detail-attention-context-detail';
            detailEl.append(chipEl, positionEl);

            partEl.append(roleEl, detailEl);
            return partEl;
        };

        this.subtitleSecondary.classList.add('detail-subtitle--attention-context');
        const sourcePart = buildContextPart(context.source, 'Source', 0);
        const targetPart = buildContextPart(context.target, 'Target', 1);
        const mainContext = document.createElement('div');
        mainContext.className = 'detail-attention-context-main';
        mainContext.append(sourcePart, targetPart);

        const scoreValue = normalizeAttentionValuePart(context?.score?.value, ATTENTION_VALUE_PLACEHOLDER);
        const scoreWrap = document.createElement('div');
        scoreWrap.className = 'detail-attention-context-score';
        const scoreValueEl = document.createElement('div');
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

    _clearAttentionCellRevealAnimation(cell) {
        if (!cell) return;
        cell.classList.remove(
            'post-softmax-reveal',
            'post-softmax-reveal-focus',
            'pre-softmax-reveal',
            'pre-softmax-reveal-focus'
        );
        cell.style.animationDelay = '';
        cell.style.animationDuration = '';
    }

    _applyAttentionCellRevealAnimation(cell, className, delayMs, durationMs) {
        if (!cell) return;
        const hadClass = cell.classList.contains(className);
        this._clearAttentionCellRevealAnimation(cell);
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
        const mode = this._resolveActiveAttentionDisplayMode();
        const tokenLabels = this._attentionContext.tokenLabels || [];
        const count = this._attentionCells.length;
        const useFocusedPostReveal = mode === 'post' && this._hasAttentionFocusCell();
        const useFocusedPreReveal = mode !== 'post' && this._shouldSuppressAttentionEntryHighlight();
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
            const preAnimStagger = mode !== 'post' && visibleCellsInRow > 1
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
                    this._clearAttentionCellRevealAnimation(cell);
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
                    cell.title = buildAttentionMatrixCellTitle({
                        rowLabel,
                        colLabel,
                        mode,
                        value,
                        queryTokenIndex: Number(cell.dataset.rowTokenIndex),
                        keyTokenIndex: Number(cell.dataset.colTokenIndex)
                    });
                    cell.dataset.value = String(value);
                    cell.classList.remove('is-empty');
                    hasAnyValue = true;
                    const isFocusedCell = cell.classList.contains('is-hovered')
                        || cell.classList.contains('is-pinned');
                    if (isFocusedCell) {
                        this._clearAttentionCellRevealAnimation(cell);
                    } else if (mode === 'post') {
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
                            this._clearAttentionCellRevealAnimation(cell);
                        }
                    } else {
                        if (wasEmpty) {
                            const revealOrder = getAttentionRevealOrder(row, col, count, ATTENTION_PREVIEW_TRIANGLE);
                            this._applyAttentionCellRevealAnimation(
                                cell,
                                useFocusedPreReveal ? 'pre-softmax-reveal-focus' : 'pre-softmax-reveal',
                                revealOrder * preAnimStagger,
                                preAnimDuration
                            );
                        } else {
                            this._clearAttentionCellRevealAnimation(cell);
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
                    this._clearAttentionCellRevealAnimation(cell);
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
        this._setAttentionMatrixInteractivity(hasAnyValue);
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

    _setAttentionMatrixInteractivity(isInteractive) {
        if (!this.attentionMatrix) return;
        this.attentionMatrix.dataset.interactive = isInteractive ? 'true' : 'false';
    }

    _updateAttentionPreview(selection) {
        if (!this.attentionRoot) return;
        const context = this._resolveAttentionContext(selection);
        this._attentionContext = context;
        this._attentionDisplayModeOverride = isCausalMaskSelection(selection) ? 'mask' : null;
        this._updateAttentionTitle(context);
        this._attentionSelectionSummary = resolveAttentionScoreSelectionSummary(selection, context);
        const preferredMode = resolveAttentionModeFromSelection(selection);
        if (this._attentionDisplayModeOverride !== 'mask' && preferredMode && preferredMode !== this.attentionMode) {
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
        const displayMode = this._resolveActiveAttentionDisplayMode();
        this._updateAttentionTitle(context);
        this._applyAttentionCollapseState();
        if (!context || (displayMode !== 'mask' && !this.activationSource)) {
            this._updateAttentionToggleLabel(this.attentionMode);
            this._setAttentionVisibility(false);
            this._setAttentionMatrixInteractivity(false);
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
        const mode = displayMode;
        const hasRenderableValues = hasSource || mode === 'mask';
        this._updateAttentionToggleLabel(this.attentionMode);
        this._updateAttentionLegend(mode);
        const values = hasRenderableValues
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
            if (!hasRenderableValues) {
                this.attentionNote.textContent = 'Attention scores unavailable (no capture loaded).';
            } else {
                this.attentionNote.textContent = trimmed
                    ? `Showing first ${tokenIndices.length} of ${totalCount} tokens`
                    : '';
            }
        }
        this._attentionValueDefault = hasRenderableValues
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
                    attentionScoreLink: mode === 'mask'
                        ? null
                        : this._buildAttentionScoreValueLink({
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
                const rowTokenIndex = tokenIndices[row];
                const colTokenIndex = tokenIndices[col];
                const isVisible = ATTENTION_PREVIEW_TRIANGLE === 'upper'
                    ? col >= row
                    : (ATTENTION_PREVIEW_TRIANGLE === 'lower'
                        ? col <= row
                        : true);
                const value = values ? values[row]?.[col] : null;
                if (!isVisible) {
                    cell.classList.add('is-hidden');
                } else if (!Number.isFinite(value)) {
                    cell.classList.add('is-empty');
                } else {
                    if (shouldMuteCausalUpperPreAttentionCell({
                        mode,
                        queryTokenIndex: rowTokenIndex,
                        keyTokenIndex: colTokenIndex
                    })) {
                        cell.classList.add('is-causal-upper-muted');
                    }
                    const reveal = shouldRevealAttentionCell(progress, row, col, mode, decodeProfile);
                    if (reveal) {
                        const color = resolveAttentionPreviewCellColor(value, mode);
                        cell.style.backgroundColor = colorToCss(color);
                        const rowLabel = tokenLabels[row] || '';
                        const colLabel = tokenLabels[col] || '';
                        cell.title = buildAttentionMatrixCellTitle({
                            rowLabel,
                            colLabel,
                            mode,
                            value,
                            queryTokenIndex: rowTokenIndex,
                            keyTokenIndex: colTokenIndex
                        });
                        cell.dataset.value = String(value);
                        hasAnyValue = true;
                    } else {
                        cell.classList.add('is-empty');
                    }
                    cell.dataset.rowLabel = tokenLabels[row] || '';
                    cell.dataset.colLabel = tokenLabels[col] || '';
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
        this._setAttentionMatrixInteractivity(hasAnyValue);
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
        const safeMode = resolveAttentionDisplayMode(mode);
        const edgeEpsilon = 1e-6;
        for (let i = 0; i < this.attentionLegendTicks.length; i += 1) {
            const tick = this.attentionLegendTicks[i];
            if (!tick) continue;
            const ratio = Number(tick.dataset.ratio);
            if (!Number.isFinite(ratio)) {
                tick.dataset.label = '';
                continue;
            }
            if (safeMode === 'mask') {
                if (ratio <= edgeEpsilon) {
                    tick.dataset.label = '-∞';
                } else if (ratio >= (1 - edgeEpsilon)) {
                    tick.dataset.label = '0';
                } else {
                    tick.dataset.label = '';
                }
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
        const safeMode = resolveAttentionDisplayMode(mode);
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

        const gradient = this._buildAttentionLegendGradient(safeMode);
        this.attentionLegend.style.setProperty('--attention-legend-gradient', gradient);
        this.attentionLegend.dataset.mid = '';
        this.attentionLegendLow.textContent = '';
        this.attentionLegendHigh.textContent = '';
        this._updateAttentionLegendTickLabels(safeMode);
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
        const show = shouldShowVectorLegendForSelection(selection);
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
        const displayMode = this._resolveActiveAttentionDisplayMode();
        const emphasizeTokenLabels = displayMode === 'post';
        const leftToken = this._attentionTokenElsLeft[row];
        const topToken = this._attentionTokenElsTop[col];
        if (isSameCell) {
            cell.classList.toggle('is-pinned', isPinnedSelection);
            if (this.attentionMatrix) this.attentionMatrix.classList.add('has-focus-cell');
            this._clearAttentionCellRevealAnimation(cell);
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
        this._clearAttentionCellRevealAnimation(cell);
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
        const scoreText = formatAttentionDisplayScore({
            mode: displayMode,
            value: valueNum,
            queryTokenIndex: sourceTokenIndex,
            keyTokenIndex: targetTokenIndex
        });
        this._setAttentionValue({
            source: sourceText,
            target: targetText,
            score: scoreText,
            sourceTokenIndex,
            targetTokenIndex,
            attentionScoreLink: displayMode === 'mask'
                ? null
                : this._buildAttentionScoreValueLink({
                    mode: displayMode,
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

    _setMhsaTokenMatrixStatus(message = '') {
        if (!this.mhsaTokenMatrixStatus) return;
        const text = typeof message === 'string' ? message.trim() : '';
        this.mhsaTokenMatrixStatus.textContent = text;
        this.mhsaTokenMatrixStatus.hidden = !text;
        this.mhsaTokenMatrixStatus.setAttribute('aria-hidden', text ? 'false' : 'true');
    }

    _setMhsaTokenMatrixHoverValue(payload = null) {
        if (!this.mhsaTokenMatrixHover) return;
        this.mhsaTokenMatrixHover.hidden = true;
        this.mhsaTokenMatrixHover.dataset.empty = 'true';
        this.mhsaTokenMatrixHover.setAttribute('aria-hidden', 'true');
        if (
            !payload
            || !Number.isFinite(payload.clientX)
            || !Number.isFinite(payload.clientY)
            || typeof payload.label !== 'string'
            || !payload.label.length
        ) {
            this._mhsaTokenMatrixHoverOverlay?.hide();
            return;
        }
        const shown = this._mhsaTokenMatrixHoverOverlay?.show({
            clientX: payload.clientX,
            clientY: payload.clientY,
            label: payload.label,
            info: payload.info || null,
            activationSource: this.activationSource
        });
        if (!shown) {
            this._mhsaTokenMatrixHoverOverlay?.hide();
        }
    }

    _resolveMhsaTokenMatrixScoreHoverPayload({
        rowIndex = null,
        colIndex = null,
        sourceType = 'pre'
    } = {}) {
        const scoreStage = this._mhsaTokenMatrixData?.attentionScoreStage || null;
        if (!scoreStage || !Number.isFinite(rowIndex) || !Number.isFinite(colIndex)) return null;

        const safeRowIndex = Math.max(0, Math.floor(rowIndex));
        const safeColIndex = Math.max(0, Math.floor(colIndex));
        const stageKey = sourceType === 'mask'
            ? 'mask'
            : (sourceType === 'masked-input'
                ? 'masked-input'
                : (sourceType === 'post' ? 'post' : 'pre'));
        const sourceRows = stageKey === 'mask'
            ? scoreStage.maskRows
            : (stageKey === 'post' ? scoreStage.postRows : scoreStage.outputRows);
        const rowData = Array.isArray(sourceRows) ? sourceRows[safeRowIndex] : null;
        const cellData = Array.isArray(rowData?.cells) ? rowData.cells[safeColIndex] : null;
        if (!cellData) return null;

        const layerIndex = findUserDataNumber(this._lastSelection, 'layerIndex');
        const headIndex = findUserDataNumber(this._lastSelection, 'headIndex');
        const queryTokenIndex = Number.isFinite(cellData.queryTokenIndex)
            ? Math.floor(cellData.queryTokenIndex)
            : (Number.isFinite(rowData?.tokenIndex) ? Math.floor(rowData.tokenIndex) : null);
        const keyTokenIndex = Number.isFinite(cellData.keyTokenIndex)
            ? Math.floor(cellData.keyTokenIndex)
            : null;
        const queryTokenLabel = cellData.queryTokenLabel || cellData.rowTokenLabel || rowData?.tokenLabel || '';
        const keyTokenLabel = cellData.keyTokenLabel || cellData.colTokenLabel || '';
        const info = buildAttentionHoverInfo({
            stageKey,
            layerIndex: Number.isFinite(layerIndex) ? Math.floor(layerIndex) : null,
            headIndex: Number.isFinite(headIndex) ? Math.floor(headIndex) : null,
            queryTokenIndex,
            queryTokenLabel,
            keyTokenIndex,
            keyTokenLabel,
            preScore: typeof cellData.preScore === 'number' && !Number.isNaN(cellData.preScore)
                ? cellData.preScore
                : null,
            postScore: typeof cellData.postScore === 'number' && !Number.isNaN(cellData.postScore)
                ? cellData.postScore
                : null,
            maskValue: typeof cellData.maskValue === 'number' && !Number.isNaN(cellData.maskValue)
                ? cellData.maskValue
                : null,
            isMasked: cellData.isMasked === true
        });
        if (!info) return null;

        return {
            label: resolveAttentionHoverLabel(stageKey),
            info
        };
    }

    _resolveMhsaTokenMatrixAttentionBlockHoverPayload(focusKey = '') {
        const safeFocusKey = String(focusKey || '').trim().toLowerCase();
        const stageKey = safeFocusKey === 'score'
            ? 'pre'
            : (safeFocusKey === 'masked-input'
                ? 'masked-input'
                : ((safeFocusKey === 'mask')
                    ? 'mask'
                    : ((safeFocusKey === 'post' || safeFocusKey === 'post-copy')
                        ? 'post'
                        : '')));
        if (!stageKey) return null;

        const layerIndex = findUserDataNumber(this._lastSelection, 'layerIndex');
        const headIndex = findUserDataNumber(this._lastSelection, 'headIndex');
        const info = buildAttentionHoverInfo({
            stageKey,
            layerIndex: Number.isFinite(layerIndex) ? Math.floor(layerIndex) : null,
            headIndex: Number.isFinite(headIndex) ? Math.floor(headIndex) : null
        });
        if (!info) return null;

        return {
            label: resolveAttentionHoverLabel(stageKey),
            info
        };
    }

    _resolveMhsaTokenMatrixHoverTokenEntries(targetInfo = null) {
        return resolveMhsaTokenMatrixHoverTokenEntries(targetInfo, {
            previewData: this._mhsaTokenMatrixData,
            tokenIndices: Array.isArray(this.attentionTokenIndices)
                ? this.attentionTokenIndices
                : this.laneTokenIndices,
            tokenLabels: Array.isArray(this.attentionTokenLabels)
                ? this.attentionTokenLabels
                : this.tokenLabels,
            activationSource: this.activationSource
        });
    }

    _setMhsaTokenMatrixSceneElementState(element, hasSceneFocus, isActive) {
        if (Array.isArray(element)) {
            element.forEach((entry) => {
                this._setMhsaTokenMatrixSceneElementState(entry, hasSceneFocus, isActive);
            });
            return;
        }
        if (!element) return;
        element.classList.toggle('is-scene-focus-active', hasSceneFocus && isActive);
        element.classList.toggle('is-scene-focus-dimmed', hasSceneFocus && !isActive);
    }

    _resolveMhsaTokenMatrixAttentionFocusKeys({
        query = false,
        transpose = false,
        score = false,
        includeWeightedOutput = true
    } = {}) {
        const keys = [];
        if (query) keys.push('query');
        if (transpose) keys.push('transpose');
        if (query || transpose || score) {
            keys.push(
                'paren-pre-open',
                'multiply',
                'paren-pre-close',
                'divide',
                'scale',
                'equals-pre'
            );
        }
        if (score) {
            keys.push(
                'score',
                'softmax',
                'softmax-paren-open',
                'masked-input',
                'add',
                'mask',
                'softmax-paren-close',
                'equals-post',
                'post'
            );
            if (includeWeightedOutput) {
                keys.push(
                    'post-copy',
                    'head-output-multiply',
                    'value-post',
                    'head-output-equals',
                    'head-output'
                );
            }
        }
        return keys;
    }

    _resolveMhsaTokenMatrixMaskFocusKeys({
        query = false,
        transpose = false
    } = {}) {
        return [
            ...this._resolveMhsaTokenMatrixAttentionFocusKeys({
                query,
                transpose,
                score: false
            }),
            'score',
            'softmax',
            'softmax-paren-open',
            'masked-input',
            'add',
            'mask',
            'softmax-paren-close'
        ];
    }

    _resolveMhsaTokenMatrixProjectionKind(stageIndex = null) {
        if (!Number.isFinite(stageIndex)) return '';
        const match = this._mhsaTokenMatrixProjectionStageEls.find((entry) => entry?.stageIndex === stageIndex);
        return typeof match?.kind === 'string' ? match.kind.toLowerCase() : '';
    }

    _applyMhsaTokenMatrixSceneFocus(config = null) {
        const projectionStageIndices = Array.isArray(config?.projectionStages)
            ? config.projectionStages
            : [];
        const projectionComponentKeys = Array.isArray(config?.projectionComponents)
            ? config.projectionComponents
            : [];
        const attentionBlockKeys = Array.isArray(config?.attentionBlocks)
            ? config.attentionBlocks
            : [];
        const connectorKeys = Array.isArray(config?.connectors)
            ? config.connectors
            : [];
        const activeProjectionStages = new Set(
            projectionStageIndices
                .filter(Number.isFinite)
                .map((stageIndex) => Math.floor(stageIndex))
        );
        const activeProjectionComponents = new Set(
            projectionComponentKeys
                .filter((key) => typeof key === 'string' && key.length)
        );
        const activeAttentionBlocks = new Set(
            attentionBlockKeys
                .filter((key) => typeof key === 'string' && key.length)
        );
        const activeConnectors = new Set(
            connectorKeys
                .filter((key) => typeof key === 'string' && key.length)
        );
        const hasSceneFocus = activeProjectionStages.size > 0
            || activeProjectionComponents.size > 0
            || activeAttentionBlocks.size > 0
            || activeConnectors.size > 0;
        const isHeadOutputTerminalFocus = activeProjectionStages.size === 0
            && activeProjectionComponents.size === 0
            && activeConnectors.size === 0
            && activeAttentionBlocks.has('head-output');
        const hasProjectionComponentFocus = activeProjectionComponents.size > 0;

        this._mhsaTokenMatrixSceneFocusState = {
            projectionStages: [...activeProjectionStages],
            projectionComponents: [...activeProjectionComponents],
            attentionBlocks: [...activeAttentionBlocks],
            connectors: [...activeConnectors]
        };
        this.mhsaTokenMatrixBody?.classList.toggle('has-scene-focus', hasSceneFocus);
        this.mhsaTokenMatrixBody?.classList.toggle('has-head-output-terminal-focus', isHeadOutputTerminalFocus);

        this._mhsaTokenMatrixProjectionStageEls.forEach((entry) => {
            const stageIndex = Number.isFinite(entry?.stageIndex) ? entry.stageIndex : null;
            const isActive = Number.isFinite(stageIndex) && activeProjectionStages.has(stageIndex);
            this._setMhsaTokenMatrixSceneElementState(
                entry?.stageEl,
                hasSceneFocus && !hasProjectionComponentFocus,
                isActive
            );
        });

        this._mhsaTokenMatrixProjectionComponentEls.forEach((entry) => {
            const stateKey = buildMhsaProjectionComponentStateKey(entry?.stageIndex, entry?.componentKey);
            this._setMhsaTokenMatrixSceneElementState(
                entry?.element,
                hasSceneFocus && hasProjectionComponentFocus,
                activeProjectionComponents.has(stateKey)
            );
        });

        Object.entries(this._mhsaTokenMatrixAttentionFocusEls || {}).forEach(([key, element]) => {
            this._setMhsaTokenMatrixSceneElementState(
                element,
                hasSceneFocus,
                activeAttentionBlocks.has(key)
            );
        });

        Object.entries(this._mhsaTokenMatrixConnectorEls || {}).forEach(([key, element]) => {
            this._setMhsaTokenMatrixSceneElementState(
                element,
                hasSceneFocus,
                activeConnectors.has(key)
            );
        });
    }

    _applyMhsaTokenMatrixViewport() {
        const workspace = this._mhsaTokenMatrixWorkspace;
        const viewport = this._mhsaTokenMatrixViewport;
        if (!workspace || !viewport) return;
        const numericScale = Number.isFinite(viewport.scale) && viewport.scale > 0 ? viewport.scale : 1;
        const fixedLabelScale = resolveMhsaTokenMatrixFixedLabelScale(numericScale);
        this.mhsaTokenMatrixBody?.style.setProperty(
            '--mhsa-token-matrix-fixed-label-scale',
            fixedLabelScale.toFixed(4)
        );
        this.mhsaTokenMatrixBody?.style.setProperty(
            '--mhsa-token-matrix-viewport-scale',
            numericScale.toFixed(4)
        );
        const translate = `translate(${viewport.panX.toFixed(1)}px, ${viewport.panY.toFixed(1)}px)`;
        const scale = numericScale.toFixed(4);
        workspace.style.removeProperty('zoom');
        workspace.style.transform = `${translate} scale(${scale})`;
        this._scheduleMhsaTokenMatrixConnectorUpdate();
        this._scheduleMhsaTokenMatrixCanvasRender();
    }

    _resolveMhsaTokenMatrixViewportCenter(scaleOverride = null) {
        const viewport = this._mhsaTokenMatrixViewport;
        const workspace = this._mhsaTokenMatrixWorkspace;
        const body = this.mhsaTokenMatrixBody;
        if (!viewport || !workspace || !body || body.hidden) return null;

        const scale = Number.isFinite(scaleOverride) && scaleOverride > 0
            ? scaleOverride
            : (Number.isFinite(viewport.scale) && viewport.scale > 0 ? viewport.scale : 1);
        const bodyRect = body.getBoundingClientRect?.();
        const bodyWidth = Math.max(1, body.clientWidth || bodyRect?.width || 0);
        const bodyHeight = Math.max(1, body.clientHeight || bodyRect?.height || 0);
        const workspaceRect = workspace.getBoundingClientRect?.();
        const contentWidth = Math.max(
            1,
            workspace.scrollWidth
            || workspace.offsetWidth
            || workspace.clientWidth
            || ((workspaceRect?.width || 0) / scale)
            || 1
        );
        const contentHeight = Math.max(
            1,
            workspace.scrollHeight
            || workspace.offsetHeight
            || workspace.clientHeight
            || ((workspaceRect?.height || 0) / scale)
            || 1
        );

        return {
            panX: (bodyWidth - (contentWidth * scale)) * 0.5,
            panY: (bodyHeight - (contentHeight * scale)) * 0.5
        };
    }

    _cancelMhsaTokenMatrixConnectorUpdate() {
        if (!Number.isFinite(this._mhsaTokenMatrixConnectorFrame)) return;
        cancelAnimationFrame(this._mhsaTokenMatrixConnectorFrame);
        this._mhsaTokenMatrixConnectorFrame = null;
    }

    _scheduleMhsaTokenMatrixConnectorUpdate() {
        if (!this._mhsaTokenMatrixWorkspace || !this._mhsaTokenMatrixOverlay || this.mhsaTokenMatrixBody?.hidden) {
            return;
        }
        if (Number.isFinite(this._mhsaTokenMatrixConnectorFrame)) return;
        this._mhsaTokenMatrixConnectorFrame = requestAnimationFrame(() => {
            this._mhsaTokenMatrixConnectorFrame = null;
            if (!this._mhsaTokenMatrixWorkspace || !this._mhsaTokenMatrixOverlay || this.mhsaTokenMatrixBody?.hidden) {
                return;
            }
            this._updateMhsaTokenMatrixConnectors();
        });
    }

    _resetMhsaTokenMatrixViewport() {
        if (!this._mhsaTokenMatrixViewport) return;
        this._mhsaTokenMatrixViewport.scale = 1;
        const centeredViewport = this._resolveMhsaTokenMatrixViewportCenter(1);
        this._mhsaTokenMatrixViewport.panX = centeredViewport?.panX ?? 0;
        this._mhsaTokenMatrixViewport.panY = centeredViewport?.panY ?? 0;
        this._mhsaTokenMatrixViewport.initialized = true;
        this._mhsaTokenMatrixViewport.hasInteracted = false;
        this._applyMhsaTokenMatrixViewport();
    }

    _nudgeMhsaTokenMatrixViewport(deltaX = 0, deltaY = 0) {
        if (!this._mhsaTokenMatrixViewport || !this._mhsaTokenMatrixWorkspace) return false;
        const nextDeltaX = Number.isFinite(deltaX) ? deltaX : 0;
        const nextDeltaY = Number.isFinite(deltaY) ? deltaY : 0;
        if (nextDeltaX === 0 && nextDeltaY === 0) return false;
        this._mhsaTokenMatrixViewport.panX += nextDeltaX;
        this._mhsaTokenMatrixViewport.panY += nextDeltaY;
        this._mhsaTokenMatrixViewport.initialized = true;
        this._mhsaTokenMatrixViewport.hasInteracted = true;
        this._applyMhsaTokenMatrixViewport();
        if (this._mhsaTokenMatrixPinned) {
            this._syncMhsaTokenMatrixPinnedClasses();
        } else {
            this._clearMhsaTokenMatrixHover(true);
        }
        return true;
    }

    _zoomMhsaTokenMatrixViewport(factor = 1, anchorX = null, anchorY = null) {
        if (!this._mhsaTokenMatrixViewport || !this._mhsaTokenMatrixWorkspace || !this.mhsaTokenMatrixBody) {
            return false;
        }
        const viewport = this._mhsaTokenMatrixViewport;
        const currentScale = Number.isFinite(viewport.scale) && viewport.scale > 0 ? viewport.scale : 1;
        const nextFactor = Number.isFinite(factor) ? factor : 1;
        if (!(nextFactor > 0)) return false;
        const nextScale = THREE.MathUtils.clamp(
            currentScale * nextFactor,
            viewport.minScale,
            viewport.maxScale
        );
        if (Math.abs(nextScale - currentScale) <= 0.0001) return false;

        const bodyRect = this.mhsaTokenMatrixBody.getBoundingClientRect();
        const pointerX = Number.isFinite(anchorX) ? anchorX : (bodyRect.width / 2);
        const pointerY = Number.isFinite(anchorY) ? anchorY : (bodyRect.height / 2);
        const localX = (pointerX - viewport.panX) / currentScale;
        const localY = (pointerY - viewport.panY) / currentScale;

        viewport.scale = nextScale;
        viewport.panX = pointerX - (localX * nextScale);
        viewport.panY = pointerY - (localY * nextScale);
        viewport.initialized = true;
        viewport.hasInteracted = true;
        this._applyMhsaTokenMatrixViewport();
        if (this._mhsaTokenMatrixPinned) {
            this._syncMhsaTokenMatrixPinnedClasses();
        } else {
            this._clearMhsaTokenMatrixHover(true);
        }
        return true;
    }

    _normalizeMhsaKeyboardControlKey(key = '') {
        const lower = String(key || '').toLowerCase();
        if (lower === 'arrowleft' || lower === 'a') return 'pan-left';
        if (lower === 'arrowright' || lower === 'd') return 'pan-right';
        if (lower === 'arrowup' || lower === 'w') return 'pan-up';
        if (lower === 'arrowdown' || lower === 's') return 'pan-down';
        if (key === '+' || key === '=' || lower === 'add' || lower === 'numpadadd') return 'zoom-in';
        if (key === '-' || key === '_' || lower === 'subtract' || lower === 'numpadsubtract') return 'zoom-out';
        return '';
    }

    _clearMhsaTokenMatrixKeyboardMotion() {
        if (!this._mhsaTokenMatrixKeyboardMotion) return;
        this._mhsaTokenMatrixKeyboardMotion.activeKeys.clear();
        this._mhsaTokenMatrixKeyboardMotion.lastTime = 0;
        if (this._mhsaTokenMatrixKeyboardMotion.rafId !== null) {
            cancelAnimationFrame(this._mhsaTokenMatrixKeyboardMotion.rafId);
            this._mhsaTokenMatrixKeyboardMotion.rafId = null;
        }
    }

    _startMhsaTokenMatrixKeyboardMotion() {
        if (!this._mhsaTokenMatrixKeyboardMotion || this._mhsaTokenMatrixKeyboardMotion.rafId !== null) return;
        this._mhsaTokenMatrixKeyboardMotion.lastTime = performance.now();
        this._mhsaTokenMatrixKeyboardMotion.rafId = requestAnimationFrame((time) => {
            this._tickMhsaTokenMatrixKeyboardMotion(time);
        });
    }

    _tickMhsaTokenMatrixKeyboardMotion(time) {
        if (!this._mhsaTokenMatrixKeyboardMotion) return;
        this._mhsaTokenMatrixKeyboardMotion.rafId = null;
        if (
            !this.isOpen
            || !this._isMhsaInfoSelectionActive
            || this.mhsaTokenMatrixBody?.hidden
            || !this._mhsaTokenMatrixWorkspace
            || !this._mhsaTokenMatrixKeyboardMotion.activeKeys.size
        ) {
            this._clearMhsaTokenMatrixKeyboardMotion();
            return;
        }

        const lastTime = Number.isFinite(this._mhsaTokenMatrixKeyboardMotion.lastTime)
            ? this._mhsaTokenMatrixKeyboardMotion.lastTime
            : time;
        const dt = Math.min(32, Math.max(8, time - lastTime));
        this._mhsaTokenMatrixKeyboardMotion.lastTime = time;

        const activeKeys = this._mhsaTokenMatrixKeyboardMotion.activeKeys;
        const panXDir = (activeKeys.has('pan-left') ? 1 : 0) + (activeKeys.has('pan-right') ? -1 : 0);
        const panYDir = (activeKeys.has('pan-up') ? 1 : 0) + (activeKeys.has('pan-down') ? -1 : 0);
        const zoomDir = (activeKeys.has('zoom-in') ? 1 : 0) + (activeKeys.has('zoom-out') ? -1 : 0);

        const panSpeedPxPerSec = 620;
        if (panXDir !== 0 || panYDir !== 0) {
            const distance = (panSpeedPxPerSec * dt) / 1000;
            this._nudgeMhsaTokenMatrixViewport(
                panXDir * distance,
                panYDir * distance
            );
        }

        if (zoomDir !== 0) {
            const zoomFactor = Math.exp(zoomDir * dt * 0.00145);
            this._zoomMhsaTokenMatrixViewport(zoomFactor);
        }

        if (this._mhsaTokenMatrixKeyboardMotion.activeKeys.size) {
            this._mhsaTokenMatrixKeyboardMotion.rafId = requestAnimationFrame((nextTime) => {
                this._tickMhsaTokenMatrixKeyboardMotion(nextTime);
            });
        }
    }

    _isTextEntryTarget(target) {
        if (!(target instanceof Element)) return false;
        const tagName = String(target.tagName || '').toLowerCase();
        if (tagName === 'input' || tagName === 'textarea' || tagName === 'select') return true;
        return target.isContentEditable === true;
    }

    _updateMhsaTokenMatrixConnectors() {
        const workspace = this._mhsaTokenMatrixWorkspace;
        const overlay = this._mhsaTokenMatrixOverlay;
        const body = this.mhsaTokenMatrixBody;
        if (!workspace || !overlay || !body) return;
        const CONNECTOR_GAP_PX = this._mhsaTokenMatrixLayoutMetrics?.connectorGaps?.default || 10;
        const viewportScale = Number.isFinite(this._mhsaTokenMatrixViewport?.scale) && this._mhsaTokenMatrixViewport.scale > 0
            ? this._mhsaTokenMatrixViewport.scale
            : 1;

        const connectorKeys = ['q', 'k', 'pre', 'post', 'v'];
        const connectors = connectorKeys
            .map((kind) => ({
                kind,
                source: workspace.querySelector(`[data-mhsa-connector-source="${kind}"]`),
                target: workspace.querySelector(`[data-mhsa-connector-target="${kind}"]`)
            }))
            .filter((entry) => entry.source && entry.target);

        const workspaceRect = workspace.getBoundingClientRect?.();
        if (!workspaceRect || !(workspaceRect.width > 0) || !(workspaceRect.height > 0)) return;
        const width = Math.max(
            1,
            Math.ceil(
                workspace.scrollWidth
                || workspace.clientWidth
                || (workspaceRect.width / viewportScale)
                || 1
            )
        );
        const height = Math.max(
            1,
            Math.ceil(
                workspace.scrollHeight
                || workspace.clientHeight
                || (workspaceRect.height / viewportScale)
                || 1
            )
        );
        overlay.setAttribute('viewBox', `0 0 ${width} ${height}`);
        overlay.setAttribute('width', String(width));
        overlay.setAttribute('height', String(height));

        const toLocalRect = (element) => {
            if (!element) return null;
            const rect = element.getBoundingClientRect?.();
            if (!rect) return null;
            return {
                left: (rect.left - workspaceRect.left) / viewportScale,
                right: (rect.right - workspaceRect.left) / viewportScale,
                top: (rect.top - workspaceRect.top) / viewportScale,
                bottom: (rect.bottom - workspaceRect.top) / viewportScale,
                width: rect.width / viewportScale,
                height: rect.height / viewportScale
            };
        };
        const resolveAnchorPoint = (element) => {
            const rect = toLocalRect(element);
            if (!rect) return null;
            const side = String(element?.dataset?.mhsaConnectorAnchorSide || 'center').toLowerCase();
            const anchorInsetRaw = Number(element?.dataset?.mhsaConnectorAnchorInset);
            const anchorInset = Number.isFinite(anchorInsetRaw)
                ? Math.max(0, anchorInsetRaw)
                : 0;
            const insetX = Math.min(Math.max(0, rect.width / 2), anchorInset);
            const insetY = Math.min(Math.max(0, rect.height / 2), anchorInset);
            switch (side) {
            case 'left':
                return { x: rect.left, y: rect.top + (rect.height / 2) };
            case 'right':
                return { x: rect.right, y: rect.top + (rect.height / 2) };
            case 'top':
                return { x: rect.left + (rect.width / 2), y: rect.top };
            case 'bottom':
                return { x: rect.left + (rect.width / 2), y: rect.bottom };
            case 'top-left':
                return { x: rect.left + insetX, y: rect.top };
            case 'top-right':
                return { x: rect.right - insetX, y: rect.top };
            case 'bottom-left':
                return { x: rect.left + insetX, y: rect.bottom };
            case 'bottom-right':
                return { x: rect.right - insetX, y: rect.bottom };
            default:
                return { x: rect.left + (rect.width / 2), y: rect.top + (rect.height / 2) };
            }
        };
        const resolveConnectorGap = (element, fallbackGap = CONNECTOR_GAP_PX) => {
            const raw = Number(element?.dataset?.mhsaConnectorGap);
            return Number.isFinite(raw) ? Math.max(0, raw) : fallbackGap;
        };
        const offsetAnchorPoint = (point, side, gap = CONNECTOR_GAP_PX) => {
            if (!point) return null;
            const safeGap = Number.isFinite(gap) ? Math.max(0, gap) : CONNECTOR_GAP_PX;
            switch (String(side || '').toLowerCase()) {
            case 'left':
                return { x: point.x - safeGap, y: point.y };
            case 'right':
                return { x: point.x + safeGap, y: point.y };
            case 'top':
            case 'top-left':
            case 'top-right':
                return { x: point.x, y: point.y - safeGap };
            case 'bottom':
            case 'bottom-left':
            case 'bottom-right':
                return { x: point.x, y: point.y + safeGap };
            default:
                return { x: point.x, y: point.y };
            }
        };
        const buildConnectorPathPoints = (start, end, route = 'horizontal', options = {}) => {
            if (!start || !end) return [];
            const targetSide = String(options?.targetSide || '').toLowerCase();
            const targetVerticalSide = targetSide.startsWith('top')
                ? 'top'
                : (targetSide.startsWith('bottom') ? 'bottom' : targetSide);
            const sourceGap = Number.isFinite(options?.sourceGap) ? Math.max(0, options.sourceGap) : CONNECTOR_GAP_PX;
            const targetGap = Number.isFinite(options?.targetGap) ? Math.max(0, options.targetGap) : CONNECTOR_GAP_PX;
            const dx = Math.abs(end.x - start.x);
            const dy = Math.abs(end.y - start.y);
            if (dx < 0.5 || dy < 0.5) {
                return [start, end];
            }
            if (route === 'underpass') {
                const sweepY = Math.max(start.y, end.y) + Math.max(12, sourceGap, targetGap);
                return [
                    start,
                    { x: start.x, y: sweepY },
                    { x: end.x, y: sweepY },
                    end
                ];
            }
            if (targetVerticalSide === 'top' || targetVerticalSide === 'bottom') {
                return [
                    start,
                    { x: end.x, y: start.y },
                    end
                ];
            }
            if (route === 'vertical') {
                const midY = (start.y + end.y) / 2;
                return [
                    start,
                    { x: start.x, y: midY },
                    { x: end.x, y: midY },
                    end
                ];
            }
            const midX = (start.x + end.x) / 2;
            return [
                start,
                { x: midX, y: start.y },
                { x: midX, y: end.y },
                end
            ];
        };
        const buildConnectorPathData = (points = []) => {
            if (!Array.isArray(points) || points.length < 2) return '';
            return points
                .map((point, index) => `${index === 0 ? 'M' : 'L'} ${point.x.toFixed(2)} ${point.y.toFixed(2)}`)
                .join(' ');
        };
        const buildConnectorArrowTipPathData = (points = []) => {
            if (!Array.isArray(points) || points.length < 2) return '';
            const head = points[points.length - 1];
            let tail = null;
            for (let idx = points.length - 2; idx >= 0; idx -= 1) {
                const candidate = points[idx];
                if (!candidate) continue;
                const dx = head.x - candidate.x;
                const dy = head.y - candidate.y;
                if (Math.hypot(dx, dy) > 0.5) {
                    tail = candidate;
                    break;
                }
            }
            if (!tail) return '';
            const dx = head.x - tail.x;
            const dy = head.y - tail.y;
            const length = Math.hypot(dx, dy);
            if (!(length > 0.5)) return '';

            const unitX = dx / length;
            const unitY = dy / length;
            const normalX = -unitY;
            const normalY = unitX;
            const screenScale = Number.isFinite(viewportScale) && viewportScale > 0 ? viewportScale : 1;
            const arrowLength = 6.8 / screenScale;
            const arrowHalfWidth = 2.8 / screenScale;
            const overlap = 0.8 / screenScale;
            const baseX = head.x - (unitX * Math.max(0.1, arrowLength - overlap));
            const baseY = head.y - (unitY * Math.max(0.1, arrowLength - overlap));
            const leftX = baseX + (normalX * arrowHalfWidth);
            const leftY = baseY + (normalY * arrowHalfWidth);
            const rightX = baseX - (normalX * arrowHalfWidth);
            const rightY = baseY - (normalY * arrowHalfWidth);

            return [
                `M ${leftX.toFixed(2)} ${leftY.toFixed(2)}`,
                `L ${head.x.toFixed(2)} ${head.y.toFixed(2)}`,
                `L ${rightX.toFixed(2)} ${rightY.toFixed(2)}`,
                'Z'
            ].join(' ');
        };

        const pathGroup = document.createElementNS('http://www.w3.org/2000/svg', 'g');
        pathGroup.setAttribute('class', 'mhsa-token-matrix-preview__connector-layer');
        const connectorEls = {};

        connectors.forEach(({ source, target, kind }) => {
            const sourceSide = String(source?.dataset?.mhsaConnectorAnchorSide || 'center').toLowerCase();
            const targetSide = String(target?.dataset?.mhsaConnectorAnchorSide || 'center').toLowerCase();
            const sourceGap = resolveConnectorGap(source);
            const targetGap = resolveConnectorGap(target);
            const start = offsetAnchorPoint(
                resolveAnchorPoint(source),
                sourceSide,
                sourceGap
            );
            const end = offsetAnchorPoint(
                resolveAnchorPoint(target),
                targetSide,
                targetGap
            );
            const route = String(target?.dataset?.mhsaConnectorRoute || source?.dataset?.mhsaConnectorRoute || 'horizontal')
                .toLowerCase();
            const pathPoints = buildConnectorPathPoints(start, end, route, {
                sourceSide,
                targetSide,
                kind,
                sourceGap,
                targetGap
            });
            const pathData = buildConnectorPathData(pathPoints);
            if (!pathData) return;

            const linePath = document.createElementNS('http://www.w3.org/2000/svg', 'path');
            linePath.setAttribute('class', 'mhsa-token-matrix-preview__connector-path mhsa-token-matrix-preview__focusable');
            linePath.setAttribute('d', pathData);
            linePath.setAttribute('stroke', '#ffffff');
            const arrowTipData = buildConnectorArrowTipPathData(pathPoints);
            const connectorGroupEls = [linePath];
            pathGroup.append(linePath);
            if (arrowTipData) {
                const tipPath = document.createElementNS('http://www.w3.org/2000/svg', 'path');
                tipPath.setAttribute('class', 'mhsa-token-matrix-preview__connector-tip mhsa-token-matrix-preview__focusable');
                tipPath.setAttribute('d', arrowTipData);
                tipPath.setAttribute('fill', 'rgba(216, 222, 232, 0.92)');
                tipPath.setAttribute('stroke', 'none');
                connectorGroupEls.push(tipPath);
                pathGroup.append(tipPath);
            }

            connectorEls[kind] = connectorGroupEls;
        });

        overlay.replaceChildren(pathGroup);
        this._mhsaTokenMatrixConnectorEls = connectorEls;
        this._applyMhsaTokenMatrixSceneFocus(this._mhsaTokenMatrixSceneFocusState);
    }

    _setMhsaTransposeFocus(colIndex = null) {
        const hasFocus = Number.isFinite(colIndex);
        this._mhsaTokenMatrixTransposeMatrixEl?.classList.toggle('has-focus', hasFocus);
        this._mhsaTokenMatrixTransposeColEls.forEach((colStates, index) => {
            const entries = Array.isArray(colStates) ? colStates : [colStates];
            entries.forEach((entry) => {
                entry?.colEl?.classList.toggle('is-active', hasFocus && index === colIndex);
                entry?.colEl?.classList.toggle('is-dimmed', hasFocus && index !== colIndex);
            });
        });
    }

    _setMhsaScoreCellFocus(rowIndex = null, colIndex = null, options = {}) {
        const hasFocus = Number.isFinite(rowIndex) && Number.isFinite(colIndex);
        this._setMhsaPostRowFocus(null);
        const allCellMatrices = [
            this._mhsaTokenMatrixScoreCellEls,
            this._mhsaTokenMatrixStaticScoreCellEls,
            this._mhsaTokenMatrixMaskCellEls,
            this._mhsaTokenMatrixPostCellEls,
            this._mhsaTokenMatrixPostCopyCellEls,
            this._mhsaTokenMatrixMirroredPostCellEls
        ].filter(Array.isArray);
        allCellMatrices.forEach((cellMatrix) => {
            forEachMhsaTokenMatrixCell(cellMatrix, (cellEl) => {
                if (!cellEl || cellEl.classList.contains('is-empty')) return;
                cellEl.classList.remove('is-active', 'is-dimmed');
            });
        });
        if (!hasFocus) return;

        const cellMatrices = [
            options.includeScoreCells === false ? null : this._mhsaTokenMatrixScoreCellEls,
            options.includeStaticScoreCells === false ? null : this._mhsaTokenMatrixStaticScoreCellEls,
            options.includeMaskCells === false ? null : this._mhsaTokenMatrixMaskCellEls,
            options.includePostCells === false ? null : this._mhsaTokenMatrixPostCellEls,
            options.includePostCopyCells === false ? null : this._mhsaTokenMatrixPostCopyCellEls,
            options.includeMirroredPostCells === false ? null : this._mhsaTokenMatrixMirroredPostCellEls
        ].filter(Array.isArray);
        cellMatrices.forEach((cellMatrix) => {
            forEachMhsaTokenMatrixCell(cellMatrix, (cellEl, focusRowIndex, focusColIndex) => {
                if (!cellEl || cellEl.classList.contains('is-empty')) return;
                const isActive = hasFocus && focusRowIndex === rowIndex && focusColIndex === colIndex;
                cellEl.classList.toggle('is-active', isActive);
                cellEl.classList.toggle('is-dimmed', hasFocus && !isActive);
            });
        });
    }

    _setMhsaPostRowFocus(rowIndex = null, options = {}) {
        const hasFocus = Number.isFinite(rowIndex);
        const rowStores = [
            options.includePostCopyRows === false ? null : this._mhsaTokenMatrixPostCopyRowEls
        ].filter(Array.isArray);
        rowStores.forEach((rowStates) => {
            rowStates.forEach((entries, focusRowIndex) => {
                const rowEntries = Array.isArray(entries) ? entries : [entries];
                rowEntries.forEach((entry) => {
                    const rowEl = entry?.rowEl || entry;
                    if (!rowEl) return;
                    const isActive = hasFocus && focusRowIndex === rowIndex;
                    rowEl.classList.toggle('is-active', isActive);
                    rowEl.classList.toggle('is-dimmed', hasFocus && !isActive);
                });
            });
        });
    }

    _setMhsaAttentionMatrixAxisFocus({ rowIndex = null, colIndex = null } = {}, options = {}) {
        const hasRowFocus = Number.isFinite(rowIndex);
        const hasColFocus = Number.isFinite(colIndex);
        const hasFocus = hasRowFocus || hasColFocus;
        const cellMatrices = [
            this._mhsaTokenMatrixScoreCellEls,
            this._mhsaTokenMatrixStaticScoreCellEls,
            this._mhsaTokenMatrixMaskCellEls,
            this._mhsaTokenMatrixPostCellEls,
            options.includePostCopyCells === true ? this._mhsaTokenMatrixPostCopyCellEls : null
        ].filter(Array.isArray);
        cellMatrices.forEach((cellMatrix) => {
            forEachMhsaTokenMatrixCell(cellMatrix, (cellEl, focusRowIndex, focusColIndex) => {
                if (!cellEl || cellEl.classList.contains('is-empty')) return;
                const matchesRow = !hasRowFocus || focusRowIndex === rowIndex;
                const matchesCol = !hasColFocus || focusColIndex === colIndex;
                cellEl.classList.toggle('is-dimmed', hasFocus && !(matchesRow && matchesCol));
                cellEl.classList.remove('is-active');
            });
        });
    }

    _applyMhsaKeyLinkFocus(rowIndex, { sourceType = 'query' } = {}) {
        const keyStageIndex = Number.isFinite(this._mhsaTokenMatrixKeyStageIndex)
            ? this._mhsaTokenMatrixKeyStageIndex
            : null;
        if (!Number.isFinite(rowIndex) || !Number.isFinite(keyStageIndex)) return;
        const firstProjectionStageIndex = this._mhsaTokenMatrixProjectionStageEls.find((entry) => Number.isFinite(entry?.stageIndex))
            ?.stageIndex ?? null;
        const keyRowSceneFocus = resolveMhsaKeyRowSceneFocus({
            keyStageIndex,
            firstProjectionStageIndex
        });
        const focusedXStages = new Set(
            [firstProjectionStageIndex, keyStageIndex].filter(Number.isFinite)
        );

        this._applyMhsaRowFocus(rowIndex, {
            sourceType: 'query',
            stageIndex: keyStageIndex
        });

        this._mhsaTokenMatrixHoverKind = 'key-link';
        this._mhsaTokenMatrixHoverSource = sourceType === 'transpose' ? 'transpose' : 'query';
        this._mhsaTokenMatrixHoverStage = keyStageIndex;
        this._mhsaTokenMatrixBody?.classList.add('has-focus-column');
        this._setMhsaTransposeFocus(rowIndex);
        this._setMhsaAttentionMatrixAxisFocus({ colIndex: rowIndex }, { includePostCopyCells: false });
        this._mhsaTokenMatrixXMatrixEl.forEach((entry) => {
            const matrixEl = entry?.matrixEl || entry;
            const stageIndex = Number.isFinite(entry?.stageIndex) ? entry.stageIndex : null;
            matrixEl?.classList.toggle('has-focus', focusedXStages.has(stageIndex));
        });
        this._mhsaTokenMatrixRowEls.forEach((rowStates, focusRowIndex) => {
            const entries = Array.isArray(rowStates) ? rowStates : [rowStates];
            entries.forEach((rowState) => {
                const stageIndex = Number.isFinite(rowState?.stageIndex) ? rowState.stageIndex : null;
                const isFocusedStage = focusedXStages.has(stageIndex);
                const isActive = isFocusedStage && focusRowIndex === rowIndex;
                rowState?.rowEl?.classList.toggle('is-active', isActive);
                rowState?.rowEl?.classList.toggle('is-dimmed', !isActive);
                rowState?.labelEl?.classList.toggle('is-highlighted', isActive);
            });
        });
        this._mhsaTokenMatrixQueryRowEls.forEach((rowStates, focusRowIndex) => {
            const entries = Array.isArray(rowStates) ? rowStates : [rowStates];
            entries.forEach((rowState) => {
                const isKeyStage = rowState?.stageIndex === keyStageIndex;
                const isActive = isKeyStage && focusRowIndex === rowIndex;
                rowState?.rowEl?.classList.toggle('is-active', isActive);
                rowState?.rowEl?.classList.toggle('is-dimmed', !isActive);
            });
        });
        this._mhsaTokenMatrixCompactRowEls.forEach((rowStates) => {
            const entries = Array.isArray(rowStates) ? rowStates : [rowStates];
            entries.forEach((rowState) => {
                rowState?.rowEl?.classList.remove('is-active');
                rowState?.rowEl?.classList.add('is-dimmed');
            });
        });
        if (keyRowSceneFocus) {
            this._applyMhsaTokenMatrixSceneFocus({
                projectionStages: [],
                projectionComponents: keyRowSceneFocus.projectionComponents,
                attentionBlocks: keyRowSceneFocus.attentionBlocks,
                connectors: keyRowSceneFocus.connectors
            });
        }
    }

    _applyMhsaScoreCellFocus(rowIndex, colIndex, { sourceType = 'score' } = {}) {
        const queryStageIndex = Number.isFinite(this._mhsaTokenMatrixQueryStageIndex)
            ? this._mhsaTokenMatrixQueryStageIndex
            : null;
        const keyStageIndex = Number.isFinite(this._mhsaTokenMatrixKeyStageIndex)
            ? this._mhsaTokenMatrixKeyStageIndex
            : null;
        const safeSourceType = sourceType === 'mask'
            ? 'mask'
            : (sourceType === 'masked-input'
                ? 'masked-input'
                : (sourceType === 'pre' ? 'pre' : 'post'));
        const isPreScoreFocus = safeSourceType === 'pre';
        const isMaskedInputFocus = safeSourceType === 'masked-input';
        const isMaskFocus = safeSourceType === 'mask';
        const isPostScoreFocus = safeSourceType === 'post';
        const isSoftmaxStageFocus = isMaskedInputFocus || isMaskFocus;
        if (!Number.isFinite(rowIndex) || !Number.isFinite(colIndex)) return;
        if (!Number.isFinite(queryStageIndex) || !Number.isFinite(keyStageIndex)) return;
        if (
            this._mhsaTokenMatrixHoverKind === 'score'
            && this._mhsaTokenMatrixHoverRow === rowIndex
            && this._mhsaTokenMatrixHoverCol === colIndex
            && this._mhsaTokenMatrixHoverSource === safeSourceType
        ) {
            return;
        }

        this._mhsaTokenMatrixHoverKind = 'score';
        this._mhsaTokenMatrixHoverRow = rowIndex;
        this._mhsaTokenMatrixHoverCol = colIndex;
        this._mhsaTokenMatrixHoverCell = `${rowIndex}:${colIndex}`;
        this._mhsaTokenMatrixHoverSource = safeSourceType;
        this._mhsaTokenMatrixHoverStage = queryStageIndex;
        this._mhsaTokenMatrixBody?.classList.add('has-focus-row', 'has-focus-column');

        const focusedStages = new Set([queryStageIndex, keyStageIndex]);
        this._mhsaTokenMatrixXMatrixEl.forEach((entry) => {
            const stageIndex = Number.isFinite(entry?.stageIndex) ? entry.stageIndex : null;
            entry?.matrixEl?.classList.toggle('has-focus', focusedStages.has(stageIndex));
        });
        this._mhsaTokenMatrixQueryMatrixEl.forEach((entry) => {
            const stageIndex = Number.isFinite(entry?.stageIndex) ? entry.stageIndex : null;
            entry?.matrixEl?.classList.toggle('has-focus', focusedStages.has(stageIndex));
        });

        this._mhsaTokenMatrixRowEls.forEach((rowStates, focusRowIndex) => {
            const entries = Array.isArray(rowStates) ? rowStates : [rowStates];
            entries.forEach((rowState) => {
                const stageIndex = Number.isFinite(rowState?.stageIndex) ? rowState.stageIndex : null;
                const targetRow = stageIndex === queryStageIndex
                    ? rowIndex
                    : (stageIndex === keyStageIndex ? colIndex : null);
                const isFocusedStage = focusedStages.has(stageIndex);
                const isActive = isFocusedStage && focusRowIndex === targetRow;
                rowState?.rowEl?.classList.toggle('is-active', isActive);
                rowState?.rowEl?.classList.toggle('is-dimmed', isFocusedStage && !isActive);
                rowState?.labelEl?.classList.toggle('is-highlighted', isActive);
            });
        });

        this._mhsaTokenMatrixQueryRowEls.forEach((rowStates, focusRowIndex) => {
            const entries = Array.isArray(rowStates) ? rowStates : [rowStates];
            entries.forEach((rowState) => {
                const stageIndex = Number.isFinite(rowState?.stageIndex) ? rowState.stageIndex : null;
                const targetRow = stageIndex === queryStageIndex
                    ? rowIndex
                    : (stageIndex === keyStageIndex ? colIndex : null);
                const isFocusedStage = focusedStages.has(stageIndex);
                const isActive = isFocusedStage && focusRowIndex === targetRow;
                rowState?.rowEl?.classList.toggle('is-active', isActive);
                rowState?.rowEl?.classList.toggle('is-dimmed', isFocusedStage && !isActive);
            });
        });
        this._mhsaTokenMatrixCompactRowEls.forEach((rowStates, focusRowIndex) => {
            const entries = Array.isArray(rowStates) ? rowStates : [rowStates];
            entries.forEach((rowState) => {
                const stageIndex = Number.isFinite(rowState?.stageIndex) ? rowState.stageIndex : null;
                const isFocusedStage = isPostScoreFocus && stageIndex === queryStageIndex;
                const isActive = isFocusedStage && focusRowIndex === rowIndex;
                rowState?.rowEl?.classList.toggle('is-active', isActive);
                rowState?.rowEl?.classList.toggle('is-dimmed', isFocusedStage && !isActive);
            });
        });

        this._setMhsaTransposeFocus(colIndex);
        this._setMhsaScoreCellFocus(
            rowIndex,
            colIndex,
            isPreScoreFocus
                ? {
                    includeScoreCells: true,
                    includeStaticScoreCells: true,
                    includeMaskCells: false,
                    includePostCells: false,
                    includePostCopyCells: false
                }
                : isMaskFocus
                    ? {
                        includePostCells: false,
                        includePostCopyCells: false,
                        includeMirroredPostCells: false
                    }
                : isSoftmaxStageFocus
                    ? {
                        includePostCopyCells: false,
                        includeMirroredPostCells: false
                    }
                : {}
        );
        const attentionBlocks = isPreScoreFocus
            ? [
                ...this._resolveMhsaTokenMatrixAttentionFocusKeys({
                    query: true,
                    transpose: true,
                    score: false
                }),
                'score',
                'masked-input'
            ]
            : isMaskFocus
                ? this._resolveMhsaTokenMatrixMaskFocusKeys({
                    query: true,
                    transpose: true
                })
            : this._resolveMhsaTokenMatrixAttentionFocusKeys({
                query: true,
                transpose: true,
                score: true,
                includeWeightedOutput: !isSoftmaxStageFocus
            });
        this._applyMhsaTokenMatrixSceneFocus({
            projectionStages: [queryStageIndex, keyStageIndex],
            attentionBlocks,
            connectors: isPreScoreFocus
                ? ['q', 'k', 'pre']
                : isMaskFocus
                    ? ['q', 'k', 'pre']
                : isSoftmaxStageFocus
                    ? ['q', 'k', 'pre', 'post']
                    : ['q', 'k', 'pre', 'post', 'v']
        });
    }

    _applyMhsaRowFocus(rowIndex, options = {}) {
        if (!this._mhsaTokenMatrixRowEls.length || !Number.isFinite(rowIndex)) return;
        const hoverSource = options?.sourceType === 'query'
            ? 'query'
            : (options?.sourceType === 'value-post' ? 'value-post' : 'x');
        const hoverStage = Number.isFinite(options?.stageIndex) ? Math.floor(options.stageIndex) : null;
        if (
            this._mhsaTokenMatrixHoverKind === 'row'
            && this._mhsaTokenMatrixHoverRow === rowIndex
            && this._mhsaTokenMatrixHoverSource === hoverSource
            && this._mhsaTokenMatrixHoverStage === hoverStage
        ) {
            return;
        }
        const broadcastAcrossStages = !Number.isFinite(hoverStage);
        this._mhsaTokenMatrixHoverKind = 'row';
        this._mhsaTokenMatrixHoverRow = rowIndex;
        this._mhsaTokenMatrixHoverCol = null;
        this._mhsaTokenMatrixHoverCell = null;
        this._mhsaTokenMatrixHoverSource = hoverSource;
        this._mhsaTokenMatrixHoverStage = hoverStage;
        this._setMhsaScoreCellFocus(null, null);
        this._mhsaTokenMatrixBody?.classList.add('has-focus-row');
        const keyStageIndex = Number.isFinite(this._mhsaTokenMatrixKeyStageIndex)
            ? this._mhsaTokenMatrixKeyStageIndex
            : null;
        const projectionKind = this._resolveMhsaTokenMatrixProjectionKind(hoverStage);
        const firstProjectionStageIndex = this._mhsaTokenMatrixProjectionStageEls.find((entry) => Number.isFinite(entry?.stageIndex))
            ?.stageIndex ?? null;
        const shouldHighlightFirstCopyRow = hoverSource === 'query'
            || hoverSource === 'value-post'
            || (hoverSource === 'x' && (projectionKind === 'q' || projectionKind === 'k' || projectionKind === 'v'));
        const shouldMirrorHeadOutputRow = shouldMirrorMhsaHeadOutputRowFocus({
            hoverSource,
            projectionKind
        });
        const shouldLinkTranspose = projectionKind === 'k'
            || (hoverSource === 'query' && Number.isFinite(keyStageIndex) && hoverStage === keyStageIndex);
        this._mhsaTokenMatrixBody?.classList.toggle('has-focus-column', shouldLinkTranspose);
        this._setMhsaTransposeFocus(shouldLinkTranspose ? rowIndex : null);
        this._mhsaTokenMatrixXMatrixEl.forEach((entry) => {
            const matrixEl = entry?.matrixEl || entry;
            const stageIndex = Number.isFinite(entry?.stageIndex) ? entry.stageIndex : null;
            const isFocused = broadcastAcrossStages
                || stageIndex === hoverStage
                || (shouldHighlightFirstCopyRow
                    && Number.isFinite(firstProjectionStageIndex)
                    && stageIndex === firstProjectionStageIndex);
            matrixEl?.classList.toggle('has-focus', isFocused);
        });
        this._mhsaTokenMatrixQueryMatrixEl.forEach((entry) => {
            const matrixEl = entry?.matrixEl || entry;
            const stageIndex = Number.isFinite(entry?.stageIndex) ? entry.stageIndex : null;
            const isFocused = broadcastAcrossStages || stageIndex === hoverStage;
            matrixEl?.classList.toggle('has-focus', isFocused);
        });
        this._mhsaTokenMatrixRowEls.forEach((rowStates, index) => {
            const entries = Array.isArray(rowStates) ? rowStates : [rowStates];
            entries.forEach((rowState) => {
                const stageIndex = Number.isFinite(rowState?.stageIndex) ? rowState.stageIndex : null;
                const isFocusedStage = broadcastAcrossStages
                    || stageIndex === hoverStage
                    || (shouldHighlightFirstCopyRow
                        && Number.isFinite(firstProjectionStageIndex)
                        && stageIndex === firstProjectionStageIndex);
                const isActive = isFocusedStage && index === rowIndex;
                const isDimmed = isFocusedStage && index !== rowIndex;
                rowState?.rowEl?.classList.toggle('is-active', isActive);
                rowState?.rowEl?.classList.toggle('is-dimmed', isDimmed);
                rowState?.labelEl?.classList.toggle('is-highlighted', isActive);
            });
        });
        this._mhsaTokenMatrixQueryRowEls.forEach((rowStates, index) => {
            const entries = Array.isArray(rowStates) ? rowStates : [rowStates];
            entries.forEach((rowState) => {
                const isFocusedStage = broadcastAcrossStages || rowState?.stageIndex === hoverStage;
                const isActive = isFocusedStage && index === rowIndex;
                const isDimmed = isFocusedStage && index !== rowIndex;
                rowState?.rowEl?.classList.toggle('is-active', isActive);
                rowState?.rowEl?.classList.toggle('is-dimmed', isDimmed);
            });
        });
        this._mhsaTokenMatrixCompactRowEls.forEach((rowStates, index) => {
            const entries = Array.isArray(rowStates) ? rowStates : [rowStates];
            entries.forEach((rowState) => {
                const isHeadOutputRow = rowState?.kind === 'head-output';
                const isFocusedStage = broadcastAcrossStages
                    || rowState?.stageIndex === hoverStage
                    || (shouldMirrorHeadOutputRow && isHeadOutputRow);
                const isActive = isFocusedStage && index === rowIndex;
                const isDimmed = isFocusedStage && index !== rowIndex;
                rowState?.rowEl?.classList.toggle('is-active', isActive);
                rowState?.rowEl?.classList.toggle('is-dimmed', isDimmed);
            });
        });
        const queryStageIndex = Number.isFinite(this._mhsaTokenMatrixQueryStageIndex)
            ? this._mhsaTokenMatrixQueryStageIndex
            : null;
        const shouldFocusAttentionRow = projectionKind === 'q'
            && (hoverSource === 'query' || hoverSource === 'x');
        const shouldFocusAttentionColumn = shouldLinkTranspose;
        const shouldFocusPostCopyAxis = shouldFocusAttentionRow || shouldFocusAttentionColumn;
        this._setMhsaPostRowFocus(shouldFocusAttentionRow ? rowIndex : null);
        this._setMhsaAttentionMatrixAxisFocus(
            shouldFocusAttentionRow
                ? { rowIndex }
                : (shouldFocusAttentionColumn ? { colIndex: rowIndex } : {}),
            { includePostCopyCells: shouldFocusPostCopyAxis }
        );
        const activeProjectionStages = broadcastAcrossStages
            ? this._mhsaTokenMatrixProjectionStageEls
                .map((entry) => (Number.isFinite(entry?.stageIndex) ? entry.stageIndex : null))
                .filter(Number.isFinite)
            : (Number.isFinite(hoverStage) ? [hoverStage] : []);
        const hasAttentionQueryFocus = projectionKind === 'q'
            || (broadcastAcrossStages && Number.isFinite(queryStageIndex));
        const hasAttentionTransposeFocus = shouldLinkTranspose;
        const hasAttentionScoreAxisFocus = shouldFocusAttentionRow || shouldFocusAttentionColumn;
        const hasAttentionValuePostFocus = projectionKind === 'v'
            && (hoverSource === 'value-post' || hoverSource === 'query');
        const keyRowSceneFocus = projectionKind === 'k' && hoverSource === 'query'
            ? resolveMhsaKeyRowSceneFocus({
                keyStageIndex: hoverStage,
                firstProjectionStageIndex
            })
            : null;
        const copiedInputRowSceneFocus = hoverSource === 'x'
            && (projectionKind === 'q' || projectionKind === 'k' || projectionKind === 'v')
            ? resolveMhsaProjectionInputRowSceneFocus({
                stageIndex: hoverStage,
                firstProjectionStageIndex
            })
            : null;
        const valueInputRowSceneFocus = projectionKind === 'v' && hoverSource === 'x'
            ? resolveMhsaValueInputRowSceneFocus({
                valueStageIndex: hoverStage,
                firstProjectionStageIndex
            })
            : null;
        const useValueRowSceneFocus = projectionKind === 'v' && hoverSource === 'value-post';
        const valueRowSceneFocus = useValueRowSceneFocus
            ? resolveMhsaValueRowSceneFocus({
                valueStageIndex: hoverStage,
                firstProjectionStageIndex
            })
            : null;
        const projectionSceneFocus = valueRowSceneFocus
            || copiedInputRowSceneFocus
            || valueInputRowSceneFocus
            || keyRowSceneFocus;
        const attentionBlocks = [
            ...this._resolveMhsaTokenMatrixAttentionFocusKeys({
                query: hasAttentionQueryFocus,
                transpose: hasAttentionTransposeFocus,
                score: hasAttentionScoreAxisFocus,
                includeWeightedOutput: false
            }),
            ...(shouldFocusPostCopyAxis ? ['post-copy'] : []),
            ...(shouldMirrorHeadOutputRow ? ['head-output'] : []),
            ...(hasAttentionValuePostFocus ? ['value-post'] : [])
        ];
        this._applyMhsaTokenMatrixSceneFocus({
            projectionStages: projectionSceneFocus ? [] : activeProjectionStages,
            projectionComponents: projectionSceneFocus?.projectionComponents || [],
            attentionBlocks: projectionSceneFocus?.attentionBlocks || attentionBlocks,
            connectors: projectionSceneFocus?.connectors || [
                ...(hasAttentionQueryFocus ? ['q'] : []),
                ...(hasAttentionTransposeFocus ? ['k'] : []),
                ...(shouldFocusAttentionRow ? ['pre', 'post'] : []),
                ...(hasAttentionValuePostFocus ? ['v'] : [])
            ]
        });
    }

    _applyMhsaHeadOutputRowFocus(rowIndex) {
        if (!Number.isFinite(rowIndex)) return;

        this._mhsaTokenMatrixHoverKind = 'row';
        this._mhsaTokenMatrixHoverRow = rowIndex;
        this._mhsaTokenMatrixHoverCol = null;
        this._mhsaTokenMatrixHoverCell = null;
        this._mhsaTokenMatrixHoverSource = 'head-output';
        this._mhsaTokenMatrixHoverStage = null;
        this._mhsaTokenMatrixBody?.classList.add('has-focus-row');
        this._mhsaTokenMatrixBody?.classList.remove('has-focus-column');
        this._setMhsaTransposeFocus(null);
        this._setMhsaScoreCellFocus(null, null);
        this._setMhsaPostRowFocus(rowIndex);

        this._mhsaTokenMatrixXMatrixEl.forEach((entry) => (entry?.matrixEl || entry)?.classList.remove('has-focus'));
        this._mhsaTokenMatrixQueryMatrixEl.forEach((entry) => (entry?.matrixEl || entry)?.classList.remove('has-focus'));

        this._mhsaTokenMatrixRowEls.forEach((rowStates) => {
            const entries = Array.isArray(rowStates) ? rowStates : [rowStates];
            entries.forEach((rowState) => {
                rowState?.rowEl?.classList.remove('is-active', 'is-dimmed');
                rowState?.labelEl?.classList.remove('is-highlighted');
            });
        });
        this._mhsaTokenMatrixQueryRowEls.forEach((rowStates) => {
            const entries = Array.isArray(rowStates) ? rowStates : [rowStates];
            entries.forEach((rowState) => {
                rowState?.rowEl?.classList.remove('is-active', 'is-dimmed');
            });
        });
        this._mhsaTokenMatrixCompactRowEls.forEach((rowStates, focusRowIndex) => {
            const entries = Array.isArray(rowStates) ? rowStates : [rowStates];
            entries.forEach((rowState) => {
                const isHeadOutputRow = rowState?.kind === 'head-output';
                const isActive = isHeadOutputRow && focusRowIndex === rowIndex;
                const isDimmed = isHeadOutputRow && focusRowIndex !== rowIndex;
                rowState?.rowEl?.classList.toggle('is-active', isActive);
                rowState?.rowEl?.classList.toggle('is-dimmed', isDimmed);
            });
        });

        this._applyMhsaTokenMatrixSceneFocus({
            projectionStages: [],
            attentionBlocks: ['post-copy', 'value-post', 'head-output'],
            connectors: []
        });
    }

    _applyMhsaTokenMatrixProjectionStageFocus(stageIndex = null, options = {}) {
        if (!Number.isFinite(stageIndex)) return false;
        const safeStageIndex = Math.floor(stageIndex);
        const projectionKind = this._resolveMhsaTokenMatrixProjectionKind(safeStageIndex);
        const firstProjectionStageIndex = this._mhsaTokenMatrixProjectionStageEls.find((entry) => Number.isFinite(entry?.stageIndex))
            ?.stageIndex ?? null;
        const componentKey = typeof options?.componentKey === 'string'
            ? options.componentKey.trim().toLowerCase()
            : '';
        const weightFocusConfig = componentKey === 'weight'
            ? resolveMhsaProjectionWeightSceneFocus({
                projectionKind,
                stageIndex: safeStageIndex,
                firstProjectionStageIndex
            })
            : null;
        const hasAttentionQueryFocus = projectionKind === 'q';
        const hasAttentionTransposeFocus = projectionKind === 'k';

        this._mhsaTokenMatrixHoverKind = 'stage';
        this._mhsaTokenMatrixHoverRow = null;
        this._mhsaTokenMatrixHoverCol = null;
        this._mhsaTokenMatrixHoverCell = null;
        this._mhsaTokenMatrixHoverSource = componentKey || null;
        this._mhsaTokenMatrixHoverStage = safeStageIndex;
        this._setMhsaScoreCellFocus(null, null);
        this._mhsaTokenMatrixBody?.classList.remove('has-focus-row', 'has-focus-column');
        this._setMhsaTransposeFocus(null);

        this._mhsaTokenMatrixXMatrixEl.forEach((entry) => {
            const matrixEl = entry?.matrixEl || entry;
            matrixEl?.classList.toggle('has-focus', entry?.stageIndex === safeStageIndex);
        });
        this._mhsaTokenMatrixQueryMatrixEl.forEach((entry) => {
            const matrixEl = entry?.matrixEl || entry;
            matrixEl?.classList.toggle('has-focus', entry?.stageIndex === safeStageIndex);
        });
        this._mhsaTokenMatrixRowEls.forEach((rowStates) => {
            const entries = Array.isArray(rowStates) ? rowStates : [rowStates];
            entries.forEach((rowState) => {
                rowState?.rowEl?.classList.remove('is-active', 'is-dimmed');
                rowState?.labelEl?.classList.remove('is-highlighted');
            });
        });
        this._mhsaTokenMatrixQueryRowEls.forEach((rowStates) => {
            const entries = Array.isArray(rowStates) ? rowStates : [rowStates];
            entries.forEach((rowState) => {
                rowState?.rowEl?.classList.remove('is-active', 'is-dimmed');
            });
        });
        this._mhsaTokenMatrixCompactRowEls.forEach((rowStates) => {
            const entries = Array.isArray(rowStates) ? rowStates : [rowStates];
            entries.forEach((rowState) => {
                rowState?.rowEl?.classList.remove('is-active', 'is-dimmed');
            });
        });

        if (weightFocusConfig) {
            this._applyMhsaTokenMatrixSceneFocus({
                projectionStages: [],
                projectionComponents: weightFocusConfig.projectionComponents,
                attentionBlocks: weightFocusConfig.attentionBlocks,
                connectors: weightFocusConfig.connectors
            });
            return true;
        }

        this._mhsaTokenMatrixBody?.classList.toggle('has-focus-column', hasAttentionTransposeFocus);
        this._applyMhsaTokenMatrixSceneFocus({
            projectionStages: [safeStageIndex],
            attentionBlocks: this._resolveMhsaTokenMatrixAttentionFocusKeys({
                query: hasAttentionQueryFocus,
                transpose: hasAttentionTransposeFocus,
                score: false
            }),
            connectors: [
                ...(hasAttentionQueryFocus ? ['q'] : []),
                ...(hasAttentionTransposeFocus ? ['k'] : [])
            ]
        });
        return true;
    }

    _applyMhsaTokenMatrixAttentionBlockFocus(focusKey = '') {
        const safeKey = String(focusKey || '').trim();
        if (!safeKey) return false;
        const queryStageIndex = Number.isFinite(this._mhsaTokenMatrixQueryStageIndex)
            ? this._mhsaTokenMatrixQueryStageIndex
            : null;
        const keyStageIndex = Number.isFinite(this._mhsaTokenMatrixKeyStageIndex)
            ? this._mhsaTokenMatrixKeyStageIndex
            : null;
        const queryOnlyKeys = new Set(['query']);
        const transposeOnlyKeys = new Set(['transpose']);
        const terminalOnlyKeys = new Set(['head-output']);
        const scoreKeys = new Set([
            'paren-pre-open',
            'multiply',
            'paren-pre-close',
            'divide',
            'scale',
            'equals-pre',
            'score',
            'softmax',
            'softmax-paren-open',
            'masked-input',
            'add',
            'mask',
            'softmax-paren-close',
            'equals-post',
            'post',
            'post-copy',
            'head-output-multiply',
            'value-post',
            'head-output-equals',
            'head-output'
        ]);

        let projectionStages = [];
        let attentionBlocks = [];
        let connectors = [];

        if (queryOnlyKeys.has(safeKey)) {
            projectionStages = Number.isFinite(queryStageIndex) ? [queryStageIndex] : [];
            attentionBlocks = this._resolveMhsaTokenMatrixAttentionFocusKeys({
                query: true,
                transpose: false,
                score: false
            });
            connectors = ['q'];
        } else if (transposeOnlyKeys.has(safeKey)) {
            projectionStages = Number.isFinite(keyStageIndex) ? [keyStageIndex] : [];
            attentionBlocks = this._resolveMhsaTokenMatrixAttentionFocusKeys({
                query: false,
                transpose: true,
                score: false
            });
            connectors = ['k'];
        } else if (terminalOnlyKeys.has(safeKey)) {
            projectionStages = [];
            attentionBlocks = ['post-copy', 'value-post', safeKey];
            connectors = [];
        } else if (scoreKeys.has(safeKey)) {
            const isMaskOnlyFocus = safeKey === 'mask';
            projectionStages = [queryStageIndex, keyStageIndex].filter(Number.isFinite);
            attentionBlocks = isMaskOnlyFocus
                ? this._resolveMhsaTokenMatrixMaskFocusKeys({
                    query: true,
                    transpose: true
                })
                : this._resolveMhsaTokenMatrixAttentionFocusKeys({
                    query: true,
                    transpose: true,
                    score: true,
                    includeWeightedOutput: true
                });
            connectors = isMaskOnlyFocus
                ? ['q', 'k', 'pre']
                : ['q', 'k', 'pre', 'post', 'v'];
        } else {
            return false;
        }

        this._mhsaTokenMatrixHoverKind = 'attention-block';
        this._mhsaTokenMatrixHoverRow = null;
        this._mhsaTokenMatrixHoverCol = null;
        this._mhsaTokenMatrixHoverCell = null;
        this._mhsaTokenMatrixHoverSource = safeKey;
        this._mhsaTokenMatrixHoverStage = null;
        this._mhsaTokenMatrixBody?.classList.remove('has-focus-row', 'has-focus-column');
        this._setMhsaTransposeFocus(null);
        this._setMhsaScoreCellFocus(null, null);
        this._mhsaTokenMatrixXMatrixEl.forEach((entry) => {
            const matrixEl = entry?.matrixEl || entry;
            matrixEl?.classList.toggle('has-focus', projectionStages.includes(entry?.stageIndex));
        });
        this._mhsaTokenMatrixQueryMatrixEl.forEach((entry) => {
            const matrixEl = entry?.matrixEl || entry;
            matrixEl?.classList.toggle('has-focus', projectionStages.includes(entry?.stageIndex));
        });
        this._mhsaTokenMatrixRowEls.forEach((rowStates) => {
            const entries = Array.isArray(rowStates) ? rowStates : [rowStates];
            entries.forEach((rowState) => {
                rowState?.rowEl?.classList.remove('is-active', 'is-dimmed');
                rowState?.labelEl?.classList.remove('is-highlighted');
            });
        });
        this._mhsaTokenMatrixQueryRowEls.forEach((rowStates) => {
            const entries = Array.isArray(rowStates) ? rowStates : [rowStates];
            entries.forEach((rowState) => {
                rowState?.rowEl?.classList.remove('is-active', 'is-dimmed');
            });
        });
        this._mhsaTokenMatrixCompactRowEls.forEach((rowStates) => {
            const entries = Array.isArray(rowStates) ? rowStates : [rowStates];
            entries.forEach((rowState) => {
                rowState?.rowEl?.classList.remove('is-active', 'is-dimmed');
            });
        });

        this._applyMhsaTokenMatrixSceneFocus({
            projectionStages,
            attentionBlocks,
            connectors
        });
        return true;
    }

    _resolveMhsaTokenMatrixInteractionTarget(target, {
        requirePinnableStageComponent = false
    } = {}) {
        if (!(target instanceof Element) || !this.mhsaTokenMatrixBody) return null;

        const rowEl = target.closest('.mhsa-token-matrix-preview__row');
        if (rowEl && this.mhsaTokenMatrixBody.contains(rowEl)) {
            const rowIndex = Number(rowEl.dataset.row);
            const stageIndex = Number(rowEl.dataset.stage);
            if (Number.isFinite(rowIndex)) {
                return {
                    kind: 'row',
                    rowIndex,
                    colIndex: null,
                    sourceType: 'x',
                    stageIndex: Number.isFinite(stageIndex) ? stageIndex : null
                };
            }
        }

        const queryRowEl = target.closest('.mhsa-token-matrix-preview__query-row');
        if (queryRowEl && this.mhsaTokenMatrixBody.contains(queryRowEl)) {
            const rowIndex = Number(queryRowEl.dataset.row);
            const stageIndex = Number(queryRowEl.dataset.stage);
            const matrixKind = String(queryRowEl.dataset.kind || '').toLowerCase();
            if (Number.isFinite(rowIndex)) {
                return {
                    kind: matrixKind === 'k' ? 'key-link' : 'row',
                    rowIndex,
                    colIndex: null,
                    sourceType: matrixKind === 'k' ? 'query' : 'query',
                    stageIndex: Number.isFinite(stageIndex) ? stageIndex : null
                };
            }
        }

        const compactRowEl = target.closest('.mhsa-token-matrix-preview__compact-row');
        if (compactRowEl && this.mhsaTokenMatrixBody.contains(compactRowEl)) {
            const rowIndex = Number(compactRowEl.dataset.row);
            const stageIndex = Number(compactRowEl.dataset.stage);
            const matrixKind = String(compactRowEl.dataset.kind || '').toLowerCase();
            if (Number.isFinite(rowIndex) && matrixKind === 'v') {
                return {
                    kind: 'row',
                    rowIndex,
                    colIndex: null,
                    sourceType: 'value-post',
                    stageIndex: Number.isFinite(stageIndex) ? stageIndex : null
                };
            }
            if (Number.isFinite(rowIndex) && matrixKind === 'head-output') {
                return {
                    kind: 'row',
                    rowIndex,
                    colIndex: null,
                    sourceType: 'head-output',
                    stageIndex: null
                };
            }
        }

        const transposeColEl = target.closest('.mhsa-token-matrix-preview__transpose-col');
        if (transposeColEl && this.mhsaTokenMatrixBody.contains(transposeColEl)) {
            const colIndex = Number(transposeColEl.dataset.col);
            if (Number.isFinite(colIndex)) {
                return {
                    kind: 'key-link',
                    rowIndex: colIndex,
                    colIndex: null,
                    sourceType: 'transpose',
                    stageIndex: Number.isFinite(this._mhsaTokenMatrixKeyStageIndex)
                        ? this._mhsaTokenMatrixKeyStageIndex
                        : null
                };
            }
        }

        const scoreCellEl = target.closest('.mhsa-token-matrix-preview__score-cell, .mhsa-token-matrix-preview__score-cell-static, .mhsa-token-matrix-preview__mask-cell, .mhsa-token-matrix-preview__post-cell');
        if (
            scoreCellEl
            && this.mhsaTokenMatrixBody.contains(scoreCellEl)
            && !scoreCellEl.classList.contains('is-empty')
        ) {
            const rowIndex = Number(scoreCellEl.dataset.row);
            const colIndex = Number(scoreCellEl.dataset.col);
            const sourceType = scoreCellEl.classList.contains('mhsa-token-matrix-preview__mask-cell')
                ? 'mask'
                : (
                    scoreCellEl.classList.contains('mhsa-token-matrix-preview__post-cell')
                        ? 'post'
                        : (
                            scoreCellEl.classList.contains('mhsa-token-matrix-preview__score-cell-static')
                                ? 'masked-input'
                                : 'pre'
                        )
                );
            if (Number.isFinite(rowIndex) && Number.isFinite(colIndex)) {
                return {
                    kind: 'score',
                    rowIndex,
                    colIndex,
                    sourceType,
                    stageIndex: Number.isFinite(this._mhsaTokenMatrixQueryStageIndex)
                        ? this._mhsaTokenMatrixQueryStageIndex
                        : null
                };
            }
        }

        const attentionFocusEl = target.closest('[data-mhsa-attention-focus-key]');
        if (attentionFocusEl && this.mhsaTokenMatrixBody.contains(attentionFocusEl)) {
            const focusKey = String(attentionFocusEl.dataset.mhsaAttentionFocusKey || '').trim();
            if (focusKey) {
                return {
                    kind: 'attention-block',
                    rowIndex: null,
                    colIndex: null,
                    sourceType: null,
                    stageIndex: null,
                    focusKey
                };
            }
        }

        const stageTarget = resolveMhsaTokenMatrixProjectionStageTarget(target, {
            root: this.mhsaTokenMatrixBody,
            requirePinnableComponent: requirePinnableStageComponent
        });
        if (stageTarget) {
            return {
                kind: 'stage',
                rowIndex: null,
                colIndex: null,
                sourceType: null,
                stageIndex: stageTarget.stageIndex,
                focusKey: stageTarget.focusKey
            };
        }

        return null;
    }

    _applyMhsaTokenMatrixResolvedTarget(targetInfo = null) {
        if (!targetInfo || typeof targetInfo !== 'object') return false;
        switch (targetInfo.kind) {
        case 'row':
            if (!Number.isFinite(targetInfo.rowIndex)) return false;
            if (targetInfo.sourceType === 'head-output') {
                this._applyMhsaHeadOutputRowFocus(targetInfo.rowIndex);
                return true;
            }
            this._applyMhsaRowFocus(targetInfo.rowIndex, {
                sourceType: targetInfo.sourceType === 'query'
                    ? 'query'
                    : (targetInfo.sourceType === 'value-post' ? 'value-post' : 'x'),
                stageIndex: Number.isFinite(targetInfo.stageIndex) ? targetInfo.stageIndex : null
            });
            return true;
        case 'key-link':
            if (!Number.isFinite(targetInfo.rowIndex)) return false;
            this._applyMhsaKeyLinkFocus(targetInfo.rowIndex, {
                sourceType: targetInfo.sourceType === 'transpose' ? 'transpose' : 'query'
            });
            return true;
        case 'score':
            if (!Number.isFinite(targetInfo.rowIndex) || !Number.isFinite(targetInfo.colIndex)) return false;
            this._applyMhsaScoreCellFocus(targetInfo.rowIndex, targetInfo.colIndex, {
                sourceType: targetInfo.sourceType
            });
            return true;
        case 'attention-block':
            return this._applyMhsaTokenMatrixAttentionBlockFocus(targetInfo.focusKey);
        case 'stage':
            return this._applyMhsaTokenMatrixProjectionStageFocus(targetInfo.stageIndex, {
                componentKey: targetInfo.focusKey
            });
        default:
            return false;
        }
    }

    _syncMhsaTokenMatrixPinnedClasses() {
        const body = this.mhsaTokenMatrixBody;
        if (!body) return;

        body.classList.toggle('has-pinned-focus', this._mhsaTokenMatrixPinned === true);
        body.querySelectorAll([
            '.mhsa-token-matrix-preview__row.is-pinned',
            '.mhsa-token-matrix-preview__row-label.is-pinned',
            '.mhsa-token-matrix-preview__query-row.is-pinned',
            '.mhsa-token-matrix-preview__compact-row.is-pinned',
            '.mhsa-token-matrix-preview__post-row.is-pinned',
            '.mhsa-token-matrix-preview__transpose-col.is-pinned',
            '.mhsa-token-matrix-preview__score-cell.is-pinned',
            '.mhsa-token-matrix-preview__score-cell-static.is-pinned',
            '.mhsa-token-matrix-preview__mask-cell.is-pinned',
            '.mhsa-token-matrix-preview__post-cell.is-pinned',
            '.mhsa-token-matrix-preview__focusable.is-pinned',
            '.mhsa-token-matrix-preview__connector-path.is-pinned',
            '.mhsa-token-matrix-preview__connector-tip.is-pinned',
            '.mhsa-token-matrix-preview__x-matrix.is-pinned',
            '.mhsa-token-matrix-preview__query-matrix.is-pinned',
            '.mhsa-token-matrix-preview__transpose-matrix.is-pinned'
        ].join(', ')).forEach((element) => {
            element.classList.remove('is-pinned');
        });

        if (!this._mhsaTokenMatrixPinned) return;

        body.querySelectorAll([
            '.mhsa-token-matrix-preview__row.is-active',
            '.mhsa-token-matrix-preview__row-label.is-highlighted',
            '.mhsa-token-matrix-preview__query-row.is-active',
            '.mhsa-token-matrix-preview__compact-row.is-active',
            '.mhsa-token-matrix-preview__post-row.is-active',
            '.mhsa-token-matrix-preview__transpose-col.is-active',
            '.mhsa-token-matrix-preview__score-cell.is-active',
            '.mhsa-token-matrix-preview__score-cell-static.is-active',
            '.mhsa-token-matrix-preview__mask-cell.is-active',
            '.mhsa-token-matrix-preview__post-cell.is-active',
            '.mhsa-token-matrix-preview__focusable.is-scene-focus-active',
            '.mhsa-token-matrix-preview__connector-path.is-scene-focus-active',
            '.mhsa-token-matrix-preview__connector-tip.is-scene-focus-active',
            '.mhsa-token-matrix-preview__x-matrix.has-focus',
            '.mhsa-token-matrix-preview__query-matrix.has-focus',
            '.mhsa-token-matrix-preview__transpose-matrix.has-focus'
        ].join(', ')).forEach((element) => {
            element.classList.add('is-pinned');
        });
    }

    _setPinnedMhsaTokenMatrix(targetInfo = null) {
        if (!targetInfo || typeof targetInfo !== 'object') {
            this._clearPinnedMhsaTokenMatrix();
            return false;
        }

        this._mhsaTokenMatrixPinned = true;
        this._mhsaTokenMatrixPinnedKind = typeof targetInfo.kind === 'string' ? targetInfo.kind : null;
        this._mhsaTokenMatrixPinnedRow = Number.isFinite(targetInfo.rowIndex) ? targetInfo.rowIndex : null;
        this._mhsaTokenMatrixPinnedCol = Number.isFinite(targetInfo.colIndex) ? targetInfo.colIndex : null;
        this._mhsaTokenMatrixPinnedSource = typeof targetInfo.sourceType === 'string' ? targetInfo.sourceType : null;
        this._mhsaTokenMatrixPinnedStage = Number.isFinite(targetInfo.stageIndex) ? targetInfo.stageIndex : null;
        this._mhsaTokenMatrixPinnedFocusKey = typeof targetInfo.focusKey === 'string' ? targetInfo.focusKey : null;
        this._setMhsaTokenMatrixHoverValue(null);
        this._setPanelTokenHoverEntries(null, { emit: true });
        this._applyMhsaTokenMatrixResolvedTarget(targetInfo);
        this._syncMhsaTokenMatrixPinnedClasses();
        return true;
    }

    _restorePinnedMhsaTokenMatrix() {
        if (!this._mhsaTokenMatrixPinned) return false;
        const targetInfo = {
            kind: this._mhsaTokenMatrixPinnedKind,
            rowIndex: this._mhsaTokenMatrixPinnedRow,
            colIndex: this._mhsaTokenMatrixPinnedCol,
            sourceType: this._mhsaTokenMatrixPinnedSource,
            stageIndex: this._mhsaTokenMatrixPinnedStage,
            focusKey: this._mhsaTokenMatrixPinnedFocusKey
        };
        if (!this._applyMhsaTokenMatrixResolvedTarget(targetInfo)) {
            this._clearPinnedMhsaTokenMatrix();
            return false;
        }
        this._syncMhsaTokenMatrixPinnedClasses();
        return true;
    }

    _clearMhsaTokenMatrixHover(force = false) {
        const forceFlag = force === true;
        if (this._mhsaTokenMatrixPinned && !forceFlag) {
            this._setMhsaTokenMatrixHoverValue(null);
            this._setPanelTokenHoverEntries(null, { emit: true });
            return;
        }
        this._mhsaTokenMatrixBody?.classList.remove('has-focus-row', 'has-focus-column');
        this._mhsaTokenMatrixXMatrixEl.forEach((entry) => (entry?.matrixEl || entry)?.classList.remove('has-focus'));
        this._mhsaTokenMatrixQueryMatrixEl.forEach((entry) => (entry?.matrixEl || entry)?.classList.remove('has-focus'));
        this._setMhsaTransposeFocus(null);
        this._setMhsaScoreCellFocus(null, null);
        this._mhsaTokenMatrixRowEls.forEach((rowStates) => {
            const entries = Array.isArray(rowStates) ? rowStates : [rowStates];
            entries.forEach((rowState) => {
                rowState?.rowEl?.classList.remove('is-active', 'is-dimmed');
                rowState?.labelEl?.classList.remove('is-highlighted');
            });
        });
        this._mhsaTokenMatrixQueryRowEls.forEach((rowStates) => {
            const entries = Array.isArray(rowStates) ? rowStates : [rowStates];
            entries.forEach((rowState) => {
                rowState?.rowEl?.classList.remove('is-active', 'is-dimmed');
            });
        });
        this._mhsaTokenMatrixCompactRowEls.forEach((rowStates) => {
            const entries = Array.isArray(rowStates) ? rowStates : [rowStates];
            entries.forEach((rowState) => {
                rowState?.rowEl?.classList.remove('is-active', 'is-dimmed');
            });
        });
        this._mhsaTokenMatrixHoverCell = null;
        this._mhsaTokenMatrixHoverKind = null;
        this._mhsaTokenMatrixHoverRow = null;
        this._mhsaTokenMatrixHoverCol = null;
        this._mhsaTokenMatrixHoverSource = null;
        this._mhsaTokenMatrixHoverStage = null;
        this._setMhsaTokenMatrixHoverValue(null);
        this._setPanelTokenHoverEntries(null, { emit: true });
        this._applyMhsaTokenMatrixSceneFocus(null);
        this._syncMhsaTokenMatrixPinnedClasses();
    }

    _clearPinnedMhsaTokenMatrix() {
        this._mhsaTokenMatrixPinned = false;
        this._mhsaTokenMatrixPinnedKind = null;
        this._mhsaTokenMatrixPinnedRow = null;
        this._mhsaTokenMatrixPinnedCol = null;
        this._mhsaTokenMatrixPinnedSource = null;
        this._mhsaTokenMatrixPinnedStage = null;
        this._mhsaTokenMatrixPinnedFocusKey = null;
        this._clearMhsaTokenMatrixHover(true);
    }

    _onMhsaTokenMatrixPointerMove(event) {
        if (event?.pointerType === 'touch' && this._updateMhsaTokenMatrixTouchPinch(event)) {
            event?.preventDefault?.();
            return;
        }
        if (this._mhsaTokenMatrixPan?.active && this.mhsaTokenMatrixBody) {
            if (
                Number.isFinite(this._mhsaTokenMatrixPan.pointerId)
                && Number.isFinite(event?.pointerId)
                && event.pointerId !== this._mhsaTokenMatrixPan.pointerId
            ) {
                return;
            }
            const deltaX = Number.isFinite(event?.clientX) ? event.clientX - this._mhsaTokenMatrixPan.startX : 0;
            const deltaY = Number.isFinite(event?.clientY) ? event.clientY - this._mhsaTokenMatrixPan.startY : 0;
            const crossedDragThreshold = Math.abs(deltaX) > 2 || Math.abs(deltaY) > 2;
            if (!this._mhsaTokenMatrixPan.moved) {
                if (!crossedDragThreshold) {
                    event?.preventDefault?.();
                    return;
                }
                this._mhsaTokenMatrixPan.moved = true;
                this.mhsaTokenMatrixBody.classList.add('is-panning');
                this._mhsaTokenMatrixViewport.hasInteracted = true;
            }
            this._mhsaTokenMatrixViewport.panX = this._mhsaTokenMatrixPan.startPanX + deltaX;
            this._mhsaTokenMatrixViewport.panY = this._mhsaTokenMatrixPan.startPanY + deltaY;
            this._mhsaTokenMatrixViewport.initialized = true;
            this._applyMhsaTokenMatrixViewport();
            if (this._mhsaTokenMatrixPan.moved) {
                if (this._mhsaTokenMatrixPinned) {
                    this._syncMhsaTokenMatrixPinnedClasses();
                } else {
                    this._clearMhsaTokenMatrixHover(true);
                }
            }
            event?.preventDefault?.();
            return;
        }
        const target = event?.target instanceof Element ? event.target : null;
        if (!target || !this.mhsaTokenMatrixBody) {
            this._setMhsaTokenMatrixHoverValue(null);
            this._setPanelTokenHoverEntries(null, { emit: true });
            return;
        }
        if (this._mhsaTokenMatrixPinned) {
            this._setMhsaTokenMatrixHoverValue(null);
            this._setPanelTokenHoverEntries(null, { emit: true });
            return;
        }
        const targetInfo = this._resolveMhsaTokenMatrixInteractionTarget(target);
        const applied = this._applyMhsaTokenMatrixResolvedTarget(targetInfo);
        if (applied) {
            this._setPanelTokenHoverEntries(
                this._resolveMhsaTokenMatrixHoverTokenEntries(targetInfo),
                { emit: true }
            );
            const hoverPayload = targetInfo?.kind === 'score'
                ? this._resolveMhsaTokenMatrixScoreHoverPayload(targetInfo)
                : (targetInfo?.kind === 'attention-block'
                    ? this._resolveMhsaTokenMatrixAttentionBlockHoverPayload(targetInfo.focusKey)
                    : null);
            this._setMhsaTokenMatrixHoverValue(
                hoverPayload && Number.isFinite(event?.clientX) && Number.isFinite(event?.clientY)
                    ? {
                        ...hoverPayload,
                        clientX: event.clientX,
                        clientY: event.clientY
                    }
                    : null
            );
            return;
        }
        this._setMhsaTokenMatrixHoverValue(null);
        this._clearMhsaTokenMatrixHover();
    }

    _onMhsaTokenMatrixPointerDown(event) {
        if (
            !this.mhsaTokenMatrixBody
            || !this._mhsaTokenMatrixWorkspace
            || (Number.isFinite(event?.button) && event.button !== 0)
        ) {
            return;
        }
        if (event?.pointerType === 'touch') {
            this._trackMhsaTokenMatrixTouchPointer(event);
            if (this._beginMhsaTokenMatrixTouchPinch()) {
                event?.preventDefault?.();
                return;
            }
        }
        if (this._mhsaTokenMatrixTouchGesture?.pinchActive) {
            event?.preventDefault?.();
            return;
        }
        const body = this.mhsaTokenMatrixBody;
        this._startMhsaTokenMatrixPan(event, this._resolveMhsaTokenMatrixInteractionTarget(
            event?.target instanceof Element ? event.target : null,
            { requirePinnableStageComponent: true }
        ));
        event?.preventDefault?.();
    }

    _onMhsaTokenMatrixPointerUp(event) {
        if (event?.pointerType === 'touch') {
            this._untrackMhsaTokenMatrixTouchPointer(event?.pointerId);
            if (this._mhsaTokenMatrixTouchGesture?.pinchActive) {
                this._endMhsaTokenMatrixTouchPinch();
                event?.preventDefault?.();
                return;
            }
        }
        if (!this._mhsaTokenMatrixPan?.active) return;
        if (
            Number.isFinite(this._mhsaTokenMatrixPan.pointerId)
            && Number.isFinite(event?.pointerId)
            && event.pointerId !== this._mhsaTokenMatrixPan.pointerId
        ) {
            return;
        }
        const shouldPinSelection = !this._mhsaTokenMatrixPan.moved
            && (!Number.isFinite(event?.button) || event.button === 0);
        const targetInfo = shouldPinSelection
            ? (
                this._resolveMhsaTokenMatrixInteractionTarget(
                    event?.target instanceof Element ? event.target : null,
                    { requirePinnableStageComponent: true }
                )
                || this._mhsaTokenMatrixPan.downTargetInfo
            )
            : null;
        this._cancelMhsaTokenMatrixPan();
        if (!shouldPinSelection) return;
        if (targetInfo) {
            this._setPinnedMhsaTokenMatrix(targetInfo);
            return;
        }
        if (this._mhsaTokenMatrixPinned) {
            this._clearPinnedMhsaTokenMatrix();
        }
    }

    _cancelMhsaTokenMatrixPan() {
        const body = this.mhsaTokenMatrixBody;
        if (Number.isFinite(this._mhsaTokenMatrixPan?.pointerId) && typeof body?.releasePointerCapture === 'function') {
            try {
                body.releasePointerCapture(this._mhsaTokenMatrixPan.pointerId);
            } catch (_) { /* no-op */ }
        }
        if (body) {
            body.classList.remove('is-panning');
        }
        if (this._mhsaTokenMatrixPan) {
            this._mhsaTokenMatrixPan.active = false;
            this._mhsaTokenMatrixPan.pointerId = null;
            this._mhsaTokenMatrixPan.startX = 0;
            this._mhsaTokenMatrixPan.startY = 0;
            this._mhsaTokenMatrixPan.startPanX = 0;
            this._mhsaTokenMatrixPan.startPanY = 0;
            this._mhsaTokenMatrixPan.moved = false;
            this._mhsaTokenMatrixPan.downTargetInfo = null;
        }
    }

    _trackMhsaTokenMatrixTouchPointer(event) {
        if (!this._mhsaTokenMatrixTouchGesture?.pointers || !Number.isFinite(event?.pointerId)) return;
        this._mhsaTokenMatrixTouchGesture.pointers.set(event.pointerId, {
            clientX: Number.isFinite(event?.clientX) ? event.clientX : 0,
            clientY: Number.isFinite(event?.clientY) ? event.clientY : 0
        });
    }

    _untrackMhsaTokenMatrixTouchPointer(pointerId) {
        if (!this._mhsaTokenMatrixTouchGesture?.pointers || !Number.isFinite(pointerId)) return;
        this._mhsaTokenMatrixTouchGesture.pointers.delete(pointerId);
    }

    _getMhsaTokenMatrixTouchMetrics() {
        if (!this.mhsaTokenMatrixBody || !this._mhsaTokenMatrixTouchGesture?.pointers) return null;
        const points = Array.from(this._mhsaTokenMatrixTouchGesture.pointers.values())
            .filter((point) => Number.isFinite(point?.clientX) && Number.isFinite(point?.clientY));
        if (points.length < 2) return null;
        const [first, second] = points;
        const dx = second.clientX - first.clientX;
        const dy = second.clientY - first.clientY;
        const distance = Math.hypot(dx, dy);
        const bodyRect = this.mhsaTokenMatrixBody.getBoundingClientRect();
        return {
            distance,
            bodyX: ((first.clientX + second.clientX) * 0.5) - bodyRect.left,
            bodyY: ((first.clientY + second.clientY) * 0.5) - bodyRect.top
        };
    }

    _startMhsaTokenMatrixPan(event, downTargetInfo = null) {
        if (!this.mhsaTokenMatrixBody || !this._mhsaTokenMatrixViewport || this._mhsaTokenMatrixTouchGesture?.pinchActive) {
            return false;
        }
        const body = this.mhsaTokenMatrixBody;
        this._mhsaTokenMatrixPan.active = true;
        this._mhsaTokenMatrixPan.pointerId = Number.isFinite(event?.pointerId) ? event.pointerId : null;
        this._mhsaTokenMatrixPan.startX = Number.isFinite(event?.clientX) ? event.clientX : 0;
        this._mhsaTokenMatrixPan.startY = Number.isFinite(event?.clientY) ? event.clientY : 0;
        this._mhsaTokenMatrixPan.startPanX = this._mhsaTokenMatrixViewport.panX;
        this._mhsaTokenMatrixPan.startPanY = this._mhsaTokenMatrixViewport.panY;
        this._mhsaTokenMatrixPan.moved = false;
        this._mhsaTokenMatrixPan.downTargetInfo = downTargetInfo;
        if (Number.isFinite(event?.pointerId) && typeof body.setPointerCapture === 'function') {
            try {
                body.setPointerCapture(event.pointerId);
            } catch (_) { /* no-op */ }
        }
        return true;
    }

    _beginMhsaTokenMatrixTouchPinch() {
        if (!this._mhsaTokenMatrixTouchGesture?.pointers || this._mhsaTokenMatrixTouchGesture.pointers.size < 2) {
            return false;
        }
        const metrics = this._getMhsaTokenMatrixTouchMetrics();
        if (!metrics || !(metrics.distance > 0) || !this._mhsaTokenMatrixViewport || !this.mhsaTokenMatrixBody) {
            return false;
        }
        this._cancelMhsaTokenMatrixPan();
        const viewport = this._mhsaTokenMatrixViewport;
        const currentScale = Number.isFinite(viewport.scale) && viewport.scale > 0 ? viewport.scale : 1;
        this._mhsaTokenMatrixTouchGesture.pinchActive = true;
        this._mhsaTokenMatrixTouchGesture.startDistance = metrics.distance;
        this._mhsaTokenMatrixTouchGesture.startScale = currentScale;
        this._mhsaTokenMatrixTouchGesture.anchorLocalX = (metrics.bodyX - viewport.panX) / currentScale;
        this._mhsaTokenMatrixTouchGesture.anchorLocalY = (metrics.bodyY - viewport.panY) / currentScale;
        this.mhsaTokenMatrixBody.classList.add('is-panning');
        this._mhsaTokenMatrixViewport.hasInteracted = true;
        return true;
    }

    _updateMhsaTokenMatrixTouchPinch(event) {
        if (!this._mhsaTokenMatrixTouchGesture?.pointers || !Number.isFinite(event?.pointerId)) return false;
        const point = this._mhsaTokenMatrixTouchGesture.pointers.get(event.pointerId);
        if (!point) return false;
        point.clientX = Number.isFinite(event?.clientX) ? event.clientX : point.clientX;
        point.clientY = Number.isFinite(event?.clientY) ? event.clientY : point.clientY;
        if (!this._mhsaTokenMatrixTouchGesture.pinchActive) return false;
        const metrics = this._getMhsaTokenMatrixTouchMetrics();
        if (!metrics || !(metrics.distance > 0) || !this._mhsaTokenMatrixViewport) return false;
        const viewport = this._mhsaTokenMatrixViewport;
        const baseScale = Number.isFinite(this._mhsaTokenMatrixTouchGesture.startScale)
            ? this._mhsaTokenMatrixTouchGesture.startScale
            : 1;
        const distanceRatio = metrics.distance / Math.max(1, this._mhsaTokenMatrixTouchGesture.startDistance);
        const nextScale = THREE.MathUtils.clamp(
            baseScale * distanceRatio,
            viewport.minScale,
            viewport.maxScale
        );
        viewport.scale = nextScale;
        viewport.panX = metrics.bodyX - (this._mhsaTokenMatrixTouchGesture.anchorLocalX * nextScale);
        viewport.panY = metrics.bodyY - (this._mhsaTokenMatrixTouchGesture.anchorLocalY * nextScale);
        viewport.initialized = true;
        viewport.hasInteracted = true;
        this._applyMhsaTokenMatrixViewport();
        if (this._mhsaTokenMatrixPinned) {
            this._syncMhsaTokenMatrixPinnedClasses();
        } else {
            this._clearMhsaTokenMatrixHover(true);
        }
        return true;
    }

    _endMhsaTokenMatrixTouchPinch() {
        if (this._mhsaTokenMatrixTouchGesture) {
            this._mhsaTokenMatrixTouchGesture.pinchActive = false;
            this._mhsaTokenMatrixTouchGesture.startDistance = 0;
            this._mhsaTokenMatrixTouchGesture.startScale = 1;
            this._mhsaTokenMatrixTouchGesture.anchorLocalX = 0;
            this._mhsaTokenMatrixTouchGesture.anchorLocalY = 0;
        }
        if (this._mhsaTokenMatrixTouchGesture?.pointers?.size === 1) {
            const [pointerId, point] = this._mhsaTokenMatrixTouchGesture.pointers.entries().next().value || [];
            if (Number.isFinite(pointerId) && point) {
                this._startMhsaTokenMatrixPan({
                    pointerId,
                    clientX: point.clientX,
                    clientY: point.clientY
                }, null);
                return;
            }
        }
        this.mhsaTokenMatrixBody?.classList.remove('is-panning');
    }

    _resetMhsaTokenMatrixTouchGesture() {
        if (!this._mhsaTokenMatrixTouchGesture) return;
        this._mhsaTokenMatrixTouchGesture.pointers.clear();
        this._mhsaTokenMatrixTouchGesture.pinchActive = false;
        this._mhsaTokenMatrixTouchGesture.startDistance = 0;
        this._mhsaTokenMatrixTouchGesture.startScale = 1;
        this._mhsaTokenMatrixTouchGesture.anchorLocalX = 0;
        this._mhsaTokenMatrixTouchGesture.anchorLocalY = 0;
    }

    _onMhsaTokenMatrixWheel(event) {
        if (!this.mhsaTokenMatrixBody || !this._mhsaTokenMatrixWorkspace || !this._mhsaTokenMatrixViewport) {
            return;
        }

        const deltaX = Number.isFinite(event?.deltaX) ? event.deltaX : 0;
        const deltaY = Number.isFinite(event?.deltaY) ? event.deltaY : 0;
        const viewport = this._mhsaTokenMatrixViewport;

        if (event?.ctrlKey || event?.metaKey) {
            const currentScale = Number.isFinite(viewport.scale) && viewport.scale > 0 ? viewport.scale : 1;
            const zoomFactor = Math.exp(-deltaY / 520);
            this._zoomMhsaTokenMatrixViewport(zoomFactor);
        } else {
            viewport.panX -= deltaX;
            viewport.panY -= deltaY;
            viewport.initialized = true;
            viewport.hasInteracted = true;
            this._applyMhsaTokenMatrixViewport();
        }

        if (this._mhsaTokenMatrixPinned) {
            this._syncMhsaTokenMatrixPinnedClasses();
        } else {
            this._clearMhsaTokenMatrixHover(true);
        }
        event?.preventDefault?.();
    }

    _hideMhsaTokenMatrixPreview() {
        this._mhsaTokenMatrixRenderToken += 1;
        this._cancelMhsaTokenMatrixConnectorUpdate();
        this._cancelMhsaTokenMatrixCanvasRender();
        this._mhsaTokenMatrixData = null;
        this._mhsaTokenMatrixRowEls = [];
        this._mhsaTokenMatrixQueryRowEls = [];
        this._mhsaTokenMatrixCompactRowEls = [];
        this._mhsaTokenMatrixTransposeColEls = [];
        this._mhsaTokenMatrixScoreCellEls = [];
        this._mhsaTokenMatrixStaticScoreCellEls = [];
        this._mhsaTokenMatrixMaskCellEls = [];
        this._mhsaTokenMatrixPostCellEls = [];
        this._mhsaTokenMatrixPostCopyCellEls = [];
        this._mhsaTokenMatrixPostCopyRowEls = [];
        this._mhsaTokenMatrixMirroredPostCellEls = [];
        this._mhsaTokenMatrixSceneModel = null;
        this._mhsaTokenMatrixSceneLayout = null;
        this._mhsaTokenMatrixPinnedKind = null;
        this._mhsaTokenMatrixPinnedSource = null;
        this._mhsaTokenMatrixPinnedStage = null;
        this._mhsaTokenMatrixPinnedFocusKey = null;
        this._mhsaTokenMatrixXMatrixEl = [];
        this._mhsaTokenMatrixQueryMatrixEl = [];
        this._mhsaTokenMatrixProjectionStageEls = [];
        this._mhsaTokenMatrixProjectionComponentEls = [];
        this._mhsaTokenMatrixAttentionFocusEls = null;
        this._mhsaTokenMatrixConnectorEls = {};
        this._mhsaTokenMatrixLayoutMetrics = null;
        this._mhsaTokenMatrixSceneFocusState = {
            projectionStages: [],
            projectionComponents: [],
            attentionBlocks: [],
            connectors: []
        };
        this._mhsaTokenMatrixTransposeMatrixEl = null;
        this._mhsaTokenMatrixQueryStageIndex = null;
        this._mhsaTokenMatrixKeyStageIndex = null;
        this._mhsaTokenMatrixValueStageIndex = null;
        this._mhsaTokenMatrixWorkspace = null;
        this._mhsaTokenMatrixOverlay = null;
        this._clearMhsaTokenMatrixKeyboardMotion();
        this._cancelMhsaTokenMatrixPan();
        this._resetMhsaTokenMatrixTouchGesture();
        this._clearPinnedMhsaTokenMatrix();
        this._resetMhsaTokenMatrixViewport();
        this._mhsaTokenMatrixViewport.initialized = false;
        if (this.mhsaTokenMatrixBody) {
            this.mhsaTokenMatrixBody.hidden = true;
            this.mhsaTokenMatrixBody.classList.remove('is-measure-only');
            this.mhsaTokenMatrixBody.classList.remove('has-focus-row', 'has-focus-column', 'has-scene-focus', 'has-pinned-focus');
            this.mhsaTokenMatrixBody.style.removeProperty('--mhsa-token-matrix-fixed-label-scale');
            this.mhsaTokenMatrixBody.style.removeProperty('--mhsa-token-matrix-viewport-scale');
            this.mhsaTokenMatrixBody.style.removeProperty('--mhsa-token-matrix-rows');
            this.mhsaTokenMatrixBody.style.removeProperty('--mhsa-token-matrix-band-count');
            clearMhsaTokenMatrixLayoutVars(this.mhsaTokenMatrixBody);
            this.mhsaTokenMatrixBody.replaceChildren();
        }
        if (this.mhsaTokenMatrixCanvas) {
            this.mhsaTokenMatrixCanvas.hidden = true;
            this.mhsaTokenMatrixCanvas.setAttribute('aria-hidden', 'true');
        }
        this._setMhsaTokenMatrixHoverValue(null);
        this._setMhsaTokenMatrixStatus('');
        if (this.mhsaTokenMatrixPreview) {
            this.mhsaTokenMatrixPreview.hidden = true;
            this.mhsaTokenMatrixPreview.setAttribute('aria-hidden', 'true');
            this.mhsaTokenMatrixPreview.classList.remove('is-canvas-replace');
        }
    }

    _resolveMhsaTokenMatrixCanvasMode() {
        return resolveMhsaTokenMatrixCanvasMode(typeof window !== 'undefined' ? window : null);
    }

    _publishMhsaTokenMatrixCanvasDebugState(stage = '', extra = {}) {
        if (typeof window === 'undefined') return;
        const payload = {
            stage: typeof stage === 'string' ? stage : '',
            renderToken: this._mhsaTokenMatrixRenderToken,
            mode: this._resolveMhsaTokenMatrixCanvasMode(),
            previewHidden: !!this.mhsaTokenMatrixPreview?.hidden,
            canvasPresent: !!this.mhsaTokenMatrixCanvas,
            canvasHidden: !!this.mhsaTokenMatrixCanvas?.hidden,
            bodyHidden: !!this.mhsaTokenMatrixBody?.hidden,
            bodyMeasureOnly: !!this.mhsaTokenMatrixBody?.classList?.contains('is-measure-only'),
            hasSceneModel: !!this._mhsaTokenMatrixSceneModel,
            hasSceneLayout: !!this._mhsaTokenMatrixSceneLayout,
            hasCanvasRenderer: !!this._mhsaTokenMatrixCanvasRenderer,
            scheduledFrame: this._mhsaTokenMatrixCanvasRenderFrame !== null,
            ...extra
        };
        window.__MHSA_TOKEN_MATRIX_CANVAS_DEBUG_STATE = payload;
        const signature = JSON.stringify(payload);
        if (signature !== this._mhsaTokenMatrixCanvasDebugSignature) {
            this._mhsaTokenMatrixCanvasDebugSignature = signature;
            consoleInfo('MHSA canvas debug state:', payload);
        }
    }

    _cancelMhsaTokenMatrixCanvasRender() {
        if (this._mhsaTokenMatrixCanvasRenderFrame === null) return;
        if (typeof cancelAnimationFrame === 'function') {
            cancelAnimationFrame(this._mhsaTokenMatrixCanvasRenderFrame);
        } else {
            clearTimeout(this._mhsaTokenMatrixCanvasRenderFrame);
        }
        this._mhsaTokenMatrixCanvasRenderFrame = null;
    }

    _syncMhsaTokenMatrixCanvasPresentation({ domReady = false } = {}) {
        const mode = this._resolveMhsaTokenMatrixCanvasMode();
        const shouldRenderCanvas = shouldRenderMhsaTokenMatrixCanvas(mode)
            && !!this.mhsaTokenMatrixCanvas
            && !!this._mhsaTokenMatrixSceneModel
            && !!this._mhsaTokenMatrixSceneLayout;
        const shouldUseMeasureOnlyBody = domReady
            && shouldRenderCanvas
            && mode === VIEW2D_MHSA_CANVAS_MODES.REPLACE;

        if (this.mhsaTokenMatrixPreview) {
            this.mhsaTokenMatrixPreview.classList.toggle(
                'is-canvas-replace',
                shouldRenderCanvas && mode === VIEW2D_MHSA_CANVAS_MODES.REPLACE
            );
        }
        if (this.mhsaTokenMatrixCanvas) {
            this.mhsaTokenMatrixCanvas.hidden = !shouldRenderCanvas;
            this.mhsaTokenMatrixCanvas.setAttribute('aria-hidden', shouldRenderCanvas ? 'false' : 'true');
        }
        if (this.mhsaTokenMatrixBody) {
            this.mhsaTokenMatrixBody.hidden = !domReady;
            this.mhsaTokenMatrixBody.classList.toggle('is-measure-only', shouldUseMeasureOnlyBody);
        }
        if (shouldRenderCanvas) {
            this._scheduleMhsaTokenMatrixCanvasRender();
        } else {
            this._cancelMhsaTokenMatrixCanvasRender();
        }
        this._publishMhsaTokenMatrixCanvasDebugState('sync-presentation', {
            domReady,
            shouldRenderCanvas,
            shouldUseMeasureOnlyBody
        });
    }

    _resolveMhsaTokenMatrixCanvasViewportTransform() {
        const sceneBounds = this._mhsaTokenMatrixSceneLayout?.sceneBounds || null;
        const workspace = this._mhsaTokenMatrixWorkspace;
        const viewport = this._mhsaTokenMatrixViewport;
        if (!sceneBounds || !workspace || !viewport) {
            return null;
        }

        // In canvas replace mode the DOM workspace is kept only for measurement,
        // so its centered transform should not drive the canvas camera.
        if (this.mhsaTokenMatrixBody?.classList?.contains('is-measure-only')) {
            return null;
        }

        const workspaceRect = typeof workspace.getBoundingClientRect === 'function'
            ? workspace.getBoundingClientRect()
            : null;
        const workspaceWidth = Math.max(
            1,
            workspace.scrollWidth
            || workspace.offsetWidth
            || workspace.clientWidth
            || workspaceRect?.width
            || 0
        );
        const workspaceHeight = Math.max(
            1,
            workspace.scrollHeight
            || workspace.offsetHeight
            || workspace.clientHeight
            || workspaceRect?.height
            || 0
        );
        const baseScaleX = workspaceWidth / Math.max(1, sceneBounds.width);
        const baseScaleY = workspaceHeight / Math.max(1, sceneBounds.height);
        const baseScale = Number.isFinite(baseScaleX) && baseScaleX > 0
            ? baseScaleX
            : baseScaleY;
        if (!Number.isFinite(baseScale) || baseScale <= 0) {
            return null;
        }

        const viewportScale = Number.isFinite(viewport.scale) && viewport.scale > 0
            ? viewport.scale
            : 1;

        return {
            source: 'dom-workspace',
            scale: baseScale * viewportScale,
            offsetX: Number.isFinite(viewport.panX) ? viewport.panX : 0,
            offsetY: Number.isFinite(viewport.panY) ? viewport.panY : 0,
            baseScale,
            baseScaleX,
            baseScaleY,
            workspaceWidth,
            workspaceHeight,
            viewportScale,
            viewportPanX: Number.isFinite(viewport.panX) ? viewport.panX : 0,
            viewportPanY: Number.isFinite(viewport.panY) ? viewport.panY : 0
        };
    }

    _renderMhsaTokenMatrixCanvasPreview() {
        if (!this.mhsaTokenMatrixCanvas) {
            this._publishMhsaTokenMatrixCanvasDebugState('render-skip', {
                reason: 'missing-canvas'
            });
            return false;
        }
        if (this.mhsaTokenMatrixCanvas.hidden) {
            this._publishMhsaTokenMatrixCanvasDebugState('render-skip', {
                reason: 'canvas-hidden'
            });
            return false;
        }
        if (!this._mhsaTokenMatrixSceneModel) {
            this._publishMhsaTokenMatrixCanvasDebugState('render-skip', {
                reason: 'missing-scene-model'
            });
            return false;
        }
        if (!this._mhsaTokenMatrixSceneLayout) {
            this._publishMhsaTokenMatrixCanvasDebugState('render-skip', {
                reason: 'missing-scene-layout'
            });
            return false;
        }
        if (!this._mhsaTokenMatrixCanvasRenderer) {
            this._mhsaTokenMatrixCanvasRenderer = new CanvasSceneRenderer({
                canvas: this.mhsaTokenMatrixCanvas
            });
        } else {
            this._mhsaTokenMatrixCanvasRenderer.setCanvas(this.mhsaTokenMatrixCanvas);
        }
        this._mhsaTokenMatrixCanvasRenderer.setScene(
            this._mhsaTokenMatrixSceneModel,
            this._mhsaTokenMatrixSceneLayout
        );
        const rect = typeof this.mhsaTokenMatrixCanvas.getBoundingClientRect === 'function'
            ? this.mhsaTokenMatrixCanvas.getBoundingClientRect()
            : null;
        const width = Math.max(
            1,
            Math.floor(rect?.width || this.mhsaTokenMatrixCanvas.clientWidth || this.mhsaTokenMatrixCanvas.width || 0)
        );
        const height = Math.max(
            1,
            Math.floor(rect?.height || this.mhsaTokenMatrixCanvas.clientHeight || this.mhsaTokenMatrixCanvas.height || 0)
        );
        const viewportTransform = this._resolveMhsaTokenMatrixCanvasViewportTransform();
        const didRender = this._mhsaTokenMatrixCanvasRenderer.render({
            width,
            height,
            viewportTransform
        });
        const debugState = this._mhsaTokenMatrixCanvasRenderer.getLastRenderState?.() || null;
        if (typeof window !== 'undefined') {
            window.__MHSA_TOKEN_MATRIX_CANVAS_DEBUG_STATE = {
                ...(debugState || {}),
                didRender,
                renderToken: this._mhsaTokenMatrixRenderToken,
                canvasClientWidth: this.mhsaTokenMatrixCanvas.clientWidth || 0,
                canvasClientHeight: this.mhsaTokenMatrixCanvas.clientHeight || 0,
                sceneNodeCount: Array.isArray(this._mhsaTokenMatrixSceneModel?.nodes)
                    ? this._mhsaTokenMatrixSceneModel.nodes.length
                    : 0
            };
        }
        this._publishMhsaTokenMatrixCanvasDebugState('render-complete', {
            ...(debugState || {}),
            didRender,
            canvasClientWidth: this.mhsaTokenMatrixCanvas.clientWidth || 0,
            canvasClientHeight: this.mhsaTokenMatrixCanvas.clientHeight || 0,
            sceneNodeCount: Array.isArray(this._mhsaTokenMatrixSceneModel?.nodes)
                ? this._mhsaTokenMatrixSceneModel.nodes.length
                : 0,
            viewportTransformSource: viewportTransform?.source || 'auto-fit'
        });
        return didRender;
    }

    _scheduleMhsaTokenMatrixCanvasRender() {
        if (
            !this._isMhsaInfoSelectionActive
            || this.mhsaTokenMatrixPreview?.hidden
            || !this.mhsaTokenMatrixCanvas
            || this.mhsaTokenMatrixCanvas.hidden
        ) {
            this._publishMhsaTokenMatrixCanvasDebugState('schedule-skip', {
                reason: !this._isMhsaInfoSelectionActive
                    ? 'mhsa-inactive'
                    : (this.mhsaTokenMatrixPreview?.hidden
                        ? 'preview-hidden'
                        : (!this.mhsaTokenMatrixCanvas
                            ? 'missing-canvas'
                            : 'canvas-hidden'))
            });
            return;
        }
        if (this._mhsaTokenMatrixCanvasRenderFrame !== null) {
            this._publishMhsaTokenMatrixCanvasDebugState('schedule-skip', {
                reason: 'frame-already-scheduled'
            });
            return;
        }
        const schedule = typeof requestAnimationFrame === 'function'
            ? requestAnimationFrame.bind(window)
            : (cb) => setTimeout(cb, 16);
        this._mhsaTokenMatrixCanvasRenderFrame = schedule(() => {
            this._mhsaTokenMatrixCanvasRenderFrame = null;
            this._renderMhsaTokenMatrixCanvasPreview();
        });
        this._publishMhsaTokenMatrixCanvasDebugState('schedule-frame');
    }

    _renderMhsaTokenMatrixPreview() {
        if (!this.mhsaTokenMatrixPreview || !this.mhsaTokenMatrixBody) return;
        const renderToken = this._mhsaTokenMatrixRenderToken + 1;
        this._mhsaTokenMatrixRenderToken = renderToken;
        this._cancelMhsaTokenMatrixConnectorUpdate();
        this._cancelMhsaTokenMatrixCanvasRender();
        this._mhsaTokenMatrixData = null;
        this._mhsaTokenMatrixRowEls = [];
        this._mhsaTokenMatrixQueryRowEls = [];
        this._mhsaTokenMatrixCompactRowEls = [];
        this._mhsaTokenMatrixTransposeColEls = [];
        this._mhsaTokenMatrixScoreCellEls = [];
        this._mhsaTokenMatrixStaticScoreCellEls = [];
        this._mhsaTokenMatrixMaskCellEls = [];
        this._mhsaTokenMatrixPostCellEls = [];
        this._mhsaTokenMatrixPostCopyCellEls = [];
        this._mhsaTokenMatrixPostCopyRowEls = [];
        this._mhsaTokenMatrixMirroredPostCellEls = [];
        this._mhsaTokenMatrixPinnedKind = null;
        this._mhsaTokenMatrixPinnedSource = null;
        this._mhsaTokenMatrixPinnedStage = null;
        this._mhsaTokenMatrixPinnedFocusKey = null;
        this._mhsaTokenMatrixXMatrixEl = [];
        this._mhsaTokenMatrixQueryMatrixEl = [];
        this._mhsaTokenMatrixProjectionStageEls = [];
        this._mhsaTokenMatrixProjectionComponentEls = [];
        this._mhsaTokenMatrixAttentionFocusEls = null;
        this._mhsaTokenMatrixConnectorEls = {};
        this._mhsaTokenMatrixLayoutMetrics = null;
        this._mhsaTokenMatrixSceneFocusState = {
            projectionStages: [],
            projectionComponents: [],
            attentionBlocks: [],
            connectors: []
        };
        this._mhsaTokenMatrixTransposeMatrixEl = null;
        this._mhsaTokenMatrixQueryStageIndex = null;
        this._mhsaTokenMatrixKeyStageIndex = null;
        this._mhsaTokenMatrixValueStageIndex = null;
        this._mhsaTokenMatrixWorkspace = null;
        this._mhsaTokenMatrixOverlay = null;
        this._clearPinnedMhsaTokenMatrix();
        this.mhsaTokenMatrixPreview.hidden = false;
        this.mhsaTokenMatrixPreview.setAttribute('aria-hidden', 'false');
        this.mhsaTokenMatrixBody.hidden = true;
        this.mhsaTokenMatrixBody.classList.remove('is-measure-only');
        this._publishMhsaTokenMatrixCanvasDebugState('preview-start');
        this._syncMhsaTokenMatrixCanvasPresentation({ domReady: false });
        this.mhsaTokenMatrixBody.classList.remove('has-focus-row', 'has-focus-column', 'has-scene-focus', 'has-pinned-focus');
        this.mhsaTokenMatrixBody.style.removeProperty('--mhsa-token-matrix-fixed-label-scale');
        this.mhsaTokenMatrixBody.style.removeProperty('--mhsa-token-matrix-viewport-scale');
        clearMhsaTokenMatrixLayoutVars(this.mhsaTokenMatrixBody);
        this.mhsaTokenMatrixBody.replaceChildren();
        this._resetMhsaTokenMatrixViewport();
        this._setMhsaTokenMatrixHoverValue(null);
        this._setMhsaTokenMatrixStatus('');
        if (this.mhsaTokenMatrixHover) {
            this.mhsaTokenMatrixHover.hidden = true;
            this.mhsaTokenMatrixHover.setAttribute('aria-hidden', 'true');
        }
        if (this.mhsaTokenMatrixStatus) {
            this.mhsaTokenMatrixStatus.hidden = true;
            this.mhsaTokenMatrixStatus.setAttribute('aria-hidden', 'true');
        }

        try {
            const layerIndex = findUserDataNumber(this._lastSelection, 'layerIndex');
            const headIndex = findUserDataNumber(this._lastSelection, 'headIndex');
            const previewOptions = {
                activationSource: this.activationSource,
                sampleStep: 64,
                tokenIndices: Array.isArray(this.attentionTokenIndices) ? this.attentionTokenIndices : null,
                tokenLabels: Array.isArray(this.attentionTokenLabels) ? this.attentionTokenLabels : null
            };
            if (Number.isFinite(layerIndex)) {
                previewOptions.layerIndex = Math.floor(layerIndex);
            }
            if (Number.isFinite(headIndex)) {
                previewOptions.headIndex = Math.floor(headIndex);
            }
            const previewData = buildMhsaTokenMatrixPreviewData(previewOptions);
            if (
                !previewData?.rowCount
                || !previewData?.columnCount
                || !Array.isArray(previewData.rows)
                || !Array.isArray(previewData.projections)
                || !previewData.projections.length
            ) {
                this._publishMhsaTokenMatrixCanvasDebugState('preview-data-invalid');
                this._setMhsaTokenMatrixStatus('Token matrix unavailable.');
                return;
            }
            if (renderToken !== this._mhsaTokenMatrixRenderToken) return;
            this._mhsaTokenMatrixData = previewData;
            const layoutMetrics = resolveMhsaTokenMatrixLayoutMetrics({
                rowCount: previewData.rowCount,
                isSmallScreen: this._isSmallScreen && this._isSmallScreen()
            });
            this._mhsaTokenMatrixLayoutMetrics = layoutMetrics;
            this._mhsaTokenMatrixSceneModel = null;
            this._mhsaTokenMatrixSceneLayout = null;
            try {
                this._mhsaTokenMatrixSceneModel = buildMhsaSceneModel({
                    previewData,
                    layerIndex: previewOptions.layerIndex,
                    headIndex: previewOptions.headIndex,
                    isSmallScreen: this._isSmallScreen && this._isSmallScreen(),
                    layoutMetrics
                });
                if (this._mhsaTokenMatrixSceneModel) {
                    this._mhsaTokenMatrixSceneLayout = buildSceneLayout(this._mhsaTokenMatrixSceneModel, {
                        isSmallScreen: this._isSmallScreen && this._isSmallScreen(),
                        layoutMetrics
                    });
                }
                this._publishMhsaTokenMatrixCanvasDebugState('scene-built', {
                    rowCount: previewData.rowCount,
                    columnCount: previewData.columnCount,
                    sceneRootCount: Array.isArray(this._mhsaTokenMatrixSceneModel?.nodes)
                        ? this._mhsaTokenMatrixSceneModel.nodes.length
                        : 0
                });
            } catch (sceneModelError) {
                this._publishMhsaTokenMatrixCanvasDebugState('scene-build-error', {
                    error: sceneModelError instanceof Error ? sceneModelError.message : String(sceneModelError)
                });
                console.warn('Failed to build MHSA 2D scene model/layout:', sceneModelError);
            }
            this.mhsaTokenMatrixBody.style.setProperty('--mhsa-token-matrix-rows', String(previewData.rowCount));
            this.mhsaTokenMatrixBody.style.setProperty('--mhsa-token-matrix-band-count', String(previewData.bandCount || 1));
            applyMhsaTokenMatrixLayoutVars(this.mhsaTokenMatrixBody, layoutMetrics);
            const fragment = document.createDocumentFragment();
            const workspaceEl = document.createElement('div');
            workspaceEl.className = 'mhsa-token-matrix-preview__workspace';
            const overlayEl = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
            overlayEl.setAttribute('class', 'mhsa-token-matrix-preview__overlay');
            const stageStackEl = document.createElement('div');
            stageStackEl.className = 'mhsa-token-matrix-preview__stack mhsa-token-matrix-preview__content';
            this._mhsaTokenMatrixXMatrixEl = [];
            this._mhsaTokenMatrixQueryMatrixEl = [];
            this._mhsaTokenMatrixRowEls = Array.from({ length: previewData.rowCount }, () => []);
            this._mhsaTokenMatrixQueryRowEls = Array.from({ length: previewData.rowCount }, () => []);
            this._mhsaTokenMatrixCompactRowEls = Array.from({ length: previewData.rowCount }, () => []);
            this._mhsaTokenMatrixTransposeColEls = Array.from({ length: previewData.rowCount }, () => []);
            this._mhsaTokenMatrixScoreCellEls = createMhsaTokenMatrixCellStore(previewData.rowCount);
            this._mhsaTokenMatrixStaticScoreCellEls = createMhsaTokenMatrixCellStore(previewData.rowCount);
            this._mhsaTokenMatrixMaskCellEls = createMhsaTokenMatrixCellStore(previewData.rowCount);
            this._mhsaTokenMatrixPostCellEls = createMhsaTokenMatrixCellStore(previewData.rowCount);
            this._mhsaTokenMatrixPostCopyCellEls = createMhsaTokenMatrixCellStore(previewData.rowCount);
            this._mhsaTokenMatrixPostCopyRowEls = Array.from({ length: previewData.rowCount }, () => []);
            this._mhsaTokenMatrixMirroredPostCellEls = createMhsaTokenMatrixCellStore(previewData.rowCount);
            this._mhsaTokenMatrixTransposeMatrixEl = null;
            this._mhsaTokenMatrixQueryStageIndex = null;
            this._mhsaTokenMatrixKeyStageIndex = null;
            this._mhsaTokenMatrixValueStageIndex = null;
            this._mhsaTokenMatrixProjectionComponentEls = [];

            const renderMathLabel = (targetEl, labelTex, fallbackText = '') => {
                if (!targetEl) return;
                const safeLabelTex = typeof labelTex === 'string' ? labelTex.trim() : '';
                const katex = (typeof window !== 'undefined') ? window.katex : null;
                if (safeLabelTex && katex && typeof katex.renderToString === 'function') {
                    try {
                        targetEl.innerHTML = katex.renderToString(safeLabelTex, { throwOnError: false, displayMode: false });
                    } catch (_) {
                        targetEl.textContent = fallbackText || safeLabelTex;
                    }
                } else {
                    targetEl.textContent = fallbackText || safeLabelTex;
                }
            };

            const createVisualEl = (modifierClass = '') => {
                const visualEl = document.createElement('div');
                visualEl.className = modifierClass
                    ? `mhsa-token-matrix-preview__visual ${modifierClass}`
                    : 'mhsa-token-matrix-preview__visual';
                return visualEl;
            };

            const createCaptionEl = (labelTex, dims = '') => {
                const captionEl = document.createElement('div');
                captionEl.className = 'mhsa-token-matrix-preview__caption';
                const labelEl = document.createElement('div');
                labelEl.className = 'mhsa-token-matrix-preview__caption-label';
                renderMathLabel(labelEl, labelTex, labelTex);
                captionEl.appendChild(labelEl);
                if (typeof dims === 'string' && dims.trim().length) {
                    const dimsEl = document.createElement('div');
                    dimsEl.className = 'mhsa-token-matrix-preview__caption-dims';
                    dimsEl.textContent = dims;
                    captionEl.appendChild(dimsEl);
                }
                return captionEl;
            };

            const registerProjectionComponentEl = (element, stageIndex, componentKey) => {
                if (!element || !Number.isFinite(stageIndex)) return element;
                const safeComponentKey = String(componentKey || '').trim().toLowerCase();
                if (!safeComponentKey.length) return element;
                element.classList.add('mhsa-token-matrix-preview__focusable');
                this._mhsaTokenMatrixProjectionComponentEls.push({
                    element,
                    stageIndex,
                    componentKey: safeComponentKey
                });
                return element;
            };

            const createOperatorEl = ({
                className = '',
                text = '',
                labelTex = '',
                fallbackText = ''
            } = {}) => {
                const operatorEl = document.createElement('div');
                operatorEl.className = className
                    ? `mhsa-token-matrix-preview__operator ${className}`
                    : 'mhsa-token-matrix-preview__operator';
                if (typeof labelTex === 'string' && labelTex.trim().length) {
                    renderMathLabel(operatorEl, labelTex, fallbackText || text);
                } else {
                    operatorEl.textContent = text;
                }
                return operatorEl;
            };

            const createMhsaRowTokenChipEl = (rowData) => {
                const chipEl = document.createElement('span');
                chipEl.className = 'detail-subtitle-token-chip mhsa-token-matrix-preview__row-label-chip';

                const tokenIndex = Number.isFinite(rowData?.tokenIndex) ? Math.floor(rowData.tokenIndex) : null;
                const fallbackRowIndex = Number.isFinite(rowData?.rowIndex) ? Math.floor(rowData.rowIndex) : 0;
                let tokenLabel = Array.isArray(this.attentionTokenLabels)
                    ? this.attentionTokenLabels[fallbackRowIndex]
                    : null;
                if ((typeof tokenLabel !== 'string' || !tokenLabel.length)
                    && this.activationSource
                    && typeof this.activationSource.getTokenString === 'function'
                    && Number.isFinite(tokenIndex)) {
                    tokenLabel = this.activationSource.getTokenString(tokenIndex);
                }
                if (typeof tokenLabel !== 'string' || !tokenLabel.length) {
                    tokenLabel = rowData?.tokenLabel || '';
                }

                const tokenId = (
                    this.activationSource
                    && typeof this.activationSource.getTokenId === 'function'
                    && Number.isFinite(tokenIndex)
                )
                    ? this.activationSource.getTokenId(tokenIndex)
                    : null;
                const safeTokenId = Number.isFinite(tokenId) ? Math.floor(tokenId) : null;

                applyTokenChipColors(chipEl, {
                    tokenLabel,
                    tokenIndex,
                    tokenId: safeTokenId
                }, Number.isFinite(tokenIndex) ? tokenIndex : fallbackRowIndex);

                chipEl.textContent = formatTokenChipDisplayText(tokenLabel, tokenIndex) || rowData?.tokenLabel || '';
                chipEl.title = rowData?.tokenLabel || tokenLabel || '';
                return chipEl;
            };

            const buildStageEl = (projectionData, stageIndex) => {
                const projectionKind = String(projectionData?.kind || '').toLowerCase();
                const stageEl = document.createElement('div');
                stageEl.className = 'mhsa-token-matrix-preview__stage';
                stageEl.classList.add('mhsa-token-matrix-preview__focusable');
                stageEl.dataset.mhsaProjectionStageIndex = String(stageIndex);
                stageEl.dataset.mhsaProjectionKind = projectionKind;
                const projectionStageEl = document.createElement('div');
                projectionStageEl.className = 'mhsa-token-matrix-preview__projection-stage';
                const xBlockEl = document.createElement('div');
                xBlockEl.className = 'mhsa-token-matrix-preview__x-block';
                const xMatrixEl = document.createElement('div');
                xMatrixEl.className = 'mhsa-token-matrix-preview__x-matrix';
                const multiplyEl = document.createElement('div');
                multiplyEl.className = 'mhsa-token-matrix-preview__operator mhsa-token-matrix-preview__operator--matrix mhsa-token-matrix-preview__operator--xw';
                multiplyEl.textContent = '×';
                const weightBlockEl = document.createElement('div');
                weightBlockEl.className = 'mhsa-token-matrix-preview__weight-block';
                const weightCardEl = document.createElement('div');
                weightCardEl.className = 'mhsa-token-matrix-preview__weight-card mhsa-token-matrix-preview__weight-card--projection';
                weightCardEl.style.background = projectionData.weightGradientCss || 'none';
                if (Array.isArray(projectionData.colorRgb) && projectionData.colorRgb.length === 3) {
                    weightCardEl.style.setProperty('--mhsa-weight-rgb', projectionData.colorRgb.join(', '));
                }
                const plusEl = document.createElement('div');
                plusEl.className = 'mhsa-token-matrix-preview__operator mhsa-token-matrix-preview__operator--wb';
                plusEl.textContent = '+';
                const biasBlockEl = document.createElement('div');
                biasBlockEl.className = 'mhsa-token-matrix-preview__bias-block';
                const biasBarEl = document.createElement('div');
                biasBarEl.className = 'mhsa-token-matrix-preview__bias-bar';
                biasBarEl.style.background = projectionData.biasGradientCss || 'none';
                const equalsEl = document.createElement('div');
                equalsEl.className = 'mhsa-token-matrix-preview__operator mhsa-token-matrix-preview__operator--bq';
                equalsEl.textContent = '=';
                const queryBlockEl = document.createElement('div');
                queryBlockEl.className = 'mhsa-token-matrix-preview__query-block';
                const queryMatrixEl = document.createElement('div');
                queryMatrixEl.className = 'mhsa-token-matrix-preview__query-matrix';
                xMatrixEl.dataset.stage = String(stageIndex);
                queryMatrixEl.dataset.stage = String(stageIndex);
                queryMatrixEl.dataset.kind = projectionKind;

                this._mhsaTokenMatrixXMatrixEl.push({ matrixEl: xMatrixEl, stageIndex });
                this._mhsaTokenMatrixQueryMatrixEl.push({ matrixEl: queryMatrixEl, stageIndex });
                if (projectionKind === 'k') {
                    this._mhsaTokenMatrixKeyStageIndex = stageIndex;
                } else if (projectionKind === 'v') {
                    this._mhsaTokenMatrixValueStageIndex = stageIndex;
                }

                previewData.rows.forEach((rowData) => {
                    const rowEl = document.createElement('div');
                    rowEl.className = 'mhsa-token-matrix-preview__row';
                    rowEl.dataset.row = String(rowData.rowIndex);
                    rowEl.dataset.stage = String(stageIndex);

                    const labelEl = document.createElement('div');
                    labelEl.className = 'mhsa-token-matrix-preview__row-label';
                    labelEl.appendChild(createMhsaRowTokenChipEl(rowData));

                    const barEl = document.createElement('div');
                    barEl.className = 'mhsa-token-matrix-preview__row-bar';
                    barEl.dataset.row = String(rowData.rowIndex);
                    barEl.style.background = rowData.gradientCss || 'none';

                    rowEl.append(labelEl, barEl);
                    xMatrixEl.appendChild(rowEl);
                    this._mhsaTokenMatrixRowEls[rowData.rowIndex]?.push({
                        rowEl,
                        labelEl,
                        barEl,
                        stageIndex
                    });
                });

                projectionData.outputRows.forEach((rowData) => {
                    const queryRowEl = document.createElement('div');
                    queryRowEl.className = 'mhsa-token-matrix-preview__query-row';
                    queryRowEl.dataset.row = String(rowData.rowIndex);
                    queryRowEl.dataset.stage = String(stageIndex);
                    queryRowEl.dataset.kind = projectionKind;

                    const queryRowBarEl = document.createElement('div');
                    queryRowBarEl.className = 'mhsa-token-matrix-preview__query-row-bar';
                    queryRowBarEl.style.background = rowData.gradientCss || 'none';

                    queryRowEl.appendChild(queryRowBarEl);
                    queryMatrixEl.appendChild(queryRowEl);
                    this._mhsaTokenMatrixQueryRowEls[rowData.rowIndex]?.push({
                        rowEl: queryRowEl,
                        barEl: queryRowBarEl,
                        stageIndex
                    });
                });

                const xVisualEl = createVisualEl('mhsa-token-matrix-preview__visual--x');
                xVisualEl.appendChild(xMatrixEl);
                const weightVisualEl = createVisualEl('mhsa-token-matrix-preview__visual--weight');
                weightVisualEl.appendChild(weightCardEl);
                const biasVisualEl = createVisualEl('mhsa-token-matrix-preview__visual--bias');
                biasVisualEl.appendChild(biasBarEl);
                const queryVisualEl = createVisualEl('mhsa-token-matrix-preview__visual--query');
                queryVisualEl.appendChild(queryMatrixEl);
                const connectorColor = Array.isArray(projectionData.colorRgb) && projectionData.colorRgb.length === 3
                    ? `rgb(${projectionData.colorRgb.join(', ')})`
                    : '#d9e8ff';
                const queryConnectorGap = projectionKind === 'v'
                    ? (layoutMetrics?.connectorGaps?.value
                        || layoutMetrics?.connectorGaps?.projection
                        || layoutMetrics?.connectorGaps?.default
                        || 10)
                    : (layoutMetrics?.connectorGaps?.projection
                        || layoutMetrics?.connectorGaps?.default
                        || 10);
                queryMatrixEl.dataset.mhsaConnectorSource = String(projectionData.kind || '').toLowerCase();
                queryMatrixEl.dataset.mhsaConnectorColor = connectorColor;
                queryMatrixEl.dataset.mhsaConnectorAnchorSide = 'right';
                queryMatrixEl.dataset.mhsaConnectorRoute = 'horizontal';
                queryMatrixEl.dataset.mhsaConnectorGap = String(queryConnectorGap);

                registerProjectionComponentEl(xBlockEl, stageIndex, 'input');
                registerProjectionComponentEl(multiplyEl, stageIndex, 'multiply');
                registerProjectionComponentEl(weightBlockEl, stageIndex, 'weight');
                registerProjectionComponentEl(plusEl, stageIndex, 'plus');
                registerProjectionComponentEl(biasBlockEl, stageIndex, 'bias');
                registerProjectionComponentEl(equalsEl, stageIndex, 'equals');
                registerProjectionComponentEl(queryBlockEl, stageIndex, 'output');

                xBlockEl.append(
                    xVisualEl,
                    createCaptionEl('X_{\\ln}', `(${previewData.rowCount}, ${previewData.columnCount})`)
                );
                weightBlockEl.append(
                    weightVisualEl,
                    createCaptionEl(
                        projectionData.weightLabelTex,
                        `(${projectionData.weightRowCount}, ${projectionData.weightColumnCount})`
                    )
                );
                biasBlockEl.append(
                    biasVisualEl,
                    createCaptionEl(projectionData.biasLabelTex, `(1, ${projectionData.outputColumnCount})`)
                );
                queryBlockEl.append(
                    queryVisualEl,
                    createCaptionEl(
                        projectionData.outputLabelTex,
                        `(${projectionData.outputRowCount}, ${projectionData.outputColumnCount})`
                    )
                );

                projectionStageEl.append(
                    weightBlockEl,
                    plusEl,
                    biasBlockEl,
                    equalsEl,
                    queryBlockEl
                );

                stageEl.append(xBlockEl, multiplyEl, projectionStageEl);
                this._mhsaTokenMatrixProjectionStageEls.push({
                    stageEl,
                    stageIndex,
                    kind: projectionKind
                });
                return stageEl;
            };

            const buildAttentionScoreStageEl = (scoreStage, queryStageIndex = 0, connectorColors = {}) => {
                const attentionStageEl = document.createElement('div');
                attentionStageEl.className = 'mhsa-token-matrix-preview__attention-stage';
                const flowEl = document.createElement('div');
                flowEl.className = 'mhsa-token-matrix-preview__attention-flow';
                const attentionFocusEls = {};
                const registerAttentionFocusEl = (key, element) => {
                    if (!element) return element;
                    element.classList.add('mhsa-token-matrix-preview__focusable');
                    element.dataset.mhsaAttentionFocusKey = key;
                    attentionFocusEls[key] = element;
                    return element;
                };

                const buildAttentionMatrixBlock = ({
                    blockClass = 'mhsa-token-matrix-preview__score-block',
                    visualModifierClass = 'mhsa-token-matrix-preview__visual--score',
                    matrixClass = 'mhsa-token-matrix-preview__score-matrix',
                    cellClass = 'mhsa-token-matrix-preview__score-cell',
                    rows = [],
                    rowCount = 0,
                    columnCount = 0,
                    labelTex = '',
                    focusKey = '',
                    interactive = false,
                    connectorSource = '',
                    connectorTarget = '',
                    connectorColor = '',
                    connectorAnchorSide = 'left',
                    connectorRoute = 'horizontal',
                    connectorGap = null,
                    connectorPlacement = 'matrix',
                    titleMode = 'attention',
                    cellStore = null,
                    extraCellStores = [],
                    rowStore = null,
                    rowWrapperClass = ''
                } = {}) => {
                    const blockEl = document.createElement('div');
                    blockEl.className = blockClass;
                    const matrixEl = document.createElement('div');
                    matrixEl.className = matrixClass;
                    const useRowWrappers = Array.isArray(rowStore)
                        && typeof rowWrapperClass === 'string'
                        && rowWrapperClass.length > 0;
                    matrixEl.style.gridTemplateColumns = useRowWrappers
                        ? 'minmax(0, 1fr)'
                        : `repeat(${columnCount}, minmax(0, 1fr))`;
                    matrixEl.style.gridTemplateRows = `repeat(${rowCount}, minmax(0, 1fr))`;
                    if (useRowWrappers) {
                        matrixEl.style.setProperty('--mhsa-token-matrix-column-count', String(Math.max(1, columnCount)));
                    }
                    const connectorEl = connectorPlacement === 'block' ? blockEl : matrixEl;

                    if (typeof connectorSource === 'string' && connectorSource.length) {
                        connectorEl.dataset.mhsaConnectorSource = connectorSource;
                    }
                    if (typeof connectorTarget === 'string' && connectorTarget.length) {
                        connectorEl.dataset.mhsaConnectorTarget = connectorTarget;
                    }
                    if ((connectorSource || connectorTarget) && typeof connectorColor === 'string' && connectorColor.length) {
                        connectorEl.dataset.mhsaConnectorColor = connectorColor;
                    }
                    if (connectorSource || connectorTarget) {
                        connectorEl.dataset.mhsaConnectorAnchorSide = connectorAnchorSide;
                        connectorEl.dataset.mhsaConnectorRoute = connectorRoute;
                        if (Number.isFinite(connectorGap)) {
                            connectorEl.dataset.mhsaConnectorGap = String(Math.max(0, Math.floor(connectorGap)));
                        }
                    }

                    rows.forEach((rowData) => {
                        let rowContentEl = matrixEl;
                        if (useRowWrappers) {
                            const rowWrapperEl = document.createElement('div');
                            rowWrapperEl.className = rowWrapperClass;
                            rowWrapperEl.dataset.row = String(rowData.rowIndex);
                            matrixEl.appendChild(rowWrapperEl);
                            rowContentEl = rowWrapperEl;
                            if (Array.isArray(rowStore[rowData.rowIndex])) {
                                rowStore[rowData.rowIndex].push({
                                    rowEl: rowWrapperEl
                                });
                            }
                        }
                        rowData.cells.forEach((cellData) => {
                            const cellEl = document.createElement('div');
                            const allowNativeTitle = titleMode !== 'attention' && titleMode !== 'mask';
                            cellEl.className = cellClass;
                            cellEl.dataset.row = String(cellData.rowIndex);
                            cellEl.dataset.col = String(cellData.colIndex);
                            if (cellData.isMasked && cellData.useMaskedStyle !== false) {
                                cellEl.classList.add('is-masked');
                            }
                            if (cellData.defaultMuted) {
                                cellEl.classList.add('is-default-muted');
                            }
                            if (cellData.isEmpty) {
                                cellEl.classList.add('is-empty');
                            } else if (typeof cellData.fillCss === 'string' && cellData.fillCss.length) {
                                cellEl.style.background = cellData.fillCss;
                            }

                            if (allowNativeTitle && typeof cellData.title === 'string' && cellData.title.length) {
                                cellEl.title = cellData.title;
                            } else if (allowNativeTitle && titleMode === 'attention' && Number.isFinite(cellData.rawValue)) {
                                cellEl.title = `${cellData.rowTokenLabel} → ${cellData.colTokenLabel}: ${formatAttentionPreviewScore(cellData.rawValue)}`;
                            }

                            rowContentEl.appendChild(cellEl);
                            registerMhsaTokenMatrixCell(cellStore, cellData.rowIndex, cellData.colIndex, cellEl);
                            extraCellStores.forEach((store) => {
                                registerMhsaTokenMatrixCell(store, cellData.rowIndex, cellData.colIndex, cellEl);
                            });
                            if (interactive) {
                                registerMhsaTokenMatrixCell(
                                    this._mhsaTokenMatrixScoreCellEls,
                                    cellData.rowIndex,
                                    cellData.colIndex,
                                    cellEl
                                );
                            }
                        });
                    });

                    const visualEl = createVisualEl(visualModifierClass);
                    visualEl.appendChild(matrixEl);
                    blockEl.append(
                        visualEl,
                        createCaptionEl(labelTex, `(${rowCount}, ${columnCount})`)
                    );
                    return focusKey ? registerAttentionFocusEl(focusKey, blockEl) : blockEl;
                };

                const buildCompactRowBlock = ({
                    blockClass = 'mhsa-token-matrix-preview__query-block',
                    visualModifierClass = 'mhsa-token-matrix-preview__visual--query',
                    matrixClass = 'mhsa-token-matrix-preview__compact-row-matrix',
                    rowClass = 'mhsa-token-matrix-preview__compact-row',
                    rowBarClass = 'mhsa-token-matrix-preview__compact-row-bar',
                    rows = [],
                    rowCount = 0,
                    columnCount = 0,
                    labelTex = '',
                    focusKey = '',
                    connectorSource = '',
                    connectorTarget = '',
                    connectorColor = '',
                    connectorAnchorSide = 'left',
                    connectorRoute = 'horizontal',
                    connectorGap = null,
                    connectorAnchorInset = null,
                    connectorPlacement = 'matrix',
                    rowStore = null,
                    rowKind = '',
                    rowStageIndex = null
                } = {}) => {
                    const blockEl = document.createElement('div');
                    blockEl.className = blockClass;
                    const matrixEl = document.createElement('div');
                    matrixEl.className = matrixClass;
                    matrixEl.style.gridTemplateRows = `repeat(${rowCount}, minmax(0, 1fr))`;
                    const connectorEl = connectorPlacement === 'block' ? blockEl : matrixEl;

                    if (typeof connectorSource === 'string' && connectorSource.length) {
                        connectorEl.dataset.mhsaConnectorSource = connectorSource;
                    }
                    if (typeof connectorTarget === 'string' && connectorTarget.length) {
                        connectorEl.dataset.mhsaConnectorTarget = connectorTarget;
                    }
                    if ((connectorSource || connectorTarget) && typeof connectorColor === 'string' && connectorColor.length) {
                        connectorEl.dataset.mhsaConnectorColor = connectorColor;
                    }
                    if (connectorSource || connectorTarget) {
                        connectorEl.dataset.mhsaConnectorAnchorSide = connectorAnchorSide;
                        connectorEl.dataset.mhsaConnectorRoute = connectorRoute;
                        if (Number.isFinite(connectorGap)) {
                            connectorEl.dataset.mhsaConnectorGap = String(Math.max(0, Math.floor(connectorGap)));
                        }
                        if (Number.isFinite(connectorAnchorInset)) {
                            connectorEl.dataset.mhsaConnectorAnchorInset = String(Math.max(0, Math.floor(connectorAnchorInset)));
                        }
                    }

                    rows.forEach((rowData) => {
                        const rowEl = document.createElement('div');
                        rowEl.className = rowClass;
                        if (Array.isArray(rowStore)) {
                            rowEl.dataset.row = String(rowData.rowIndex);
                            if (Number.isFinite(rowStageIndex)) {
                                rowEl.dataset.stage = String(rowStageIndex);
                            }
                            if (typeof rowKind === 'string' && rowKind.length) {
                                rowEl.dataset.kind = rowKind;
                            }
                        }
                        if (typeof rowData?.title === 'string' && rowData.title.length) {
                            rowEl.title = rowData.title;
                        }

                        const rowBarEl = document.createElement('div');
                        rowBarEl.className = rowBarClass;
                        rowBarEl.style.background = rowData?.gradientCss || 'none';

                        rowEl.appendChild(rowBarEl);
                        matrixEl.appendChild(rowEl);
                        if (Array.isArray(rowStore) && Array.isArray(rowStore[rowData.rowIndex])) {
                            rowStore[rowData.rowIndex].push({
                                rowEl,
                                barEl: rowBarEl,
                                stageIndex: Number.isFinite(rowStageIndex) ? rowStageIndex : null,
                                kind: typeof rowKind === 'string' ? rowKind : ''
                            });
                        }
                    });

                    const visualEl = createVisualEl(visualModifierClass);
                    visualEl.appendChild(matrixEl);
                    blockEl.append(
                        visualEl,
                        createCaptionEl(labelTex, `(${rowCount}, ${columnCount})`)
                    );
                    return focusKey ? registerAttentionFocusEl(focusKey, blockEl) : blockEl;
                };

                const buildScaleBlock = () => {
                    const scaleBlockEl = document.createElement('div');
                    scaleBlockEl.className = 'mhsa-token-matrix-preview__scale-block';
                    const scaleVisualEl = createVisualEl('mhsa-token-matrix-preview__visual--scale');
                    const scaleTextEl = document.createElement('div');
                    scaleTextEl.className = 'mhsa-token-matrix-preview__scale-text';
                    renderMathLabel(scaleTextEl, scoreStage.scaleLabelTex, 'sqrt(d_head)');
                    scaleVisualEl.appendChild(scaleTextEl);
                    scaleBlockEl.append(scaleVisualEl);
                    return registerAttentionFocusEl('scale', scaleBlockEl);
                };

                const buildAttentionSourceQueryBlock = () => {
                    const sourceBlockEl = document.createElement('div');
                    sourceBlockEl.className = 'mhsa-token-matrix-preview__query-block';
                    const queryMatrixEl = document.createElement('div');
                    queryMatrixEl.className = 'mhsa-token-matrix-preview__query-matrix mhsa-token-matrix-preview__query-matrix--input';
                    queryMatrixEl.dataset.stage = String(queryStageIndex);
                    scoreStage.queryRows.forEach((rowData) => {
                        const queryRowEl = document.createElement('div');
                        queryRowEl.className = 'mhsa-token-matrix-preview__query-row';
                        queryRowEl.dataset.row = String(rowData.rowIndex);
                        queryRowEl.dataset.stage = String(queryStageIndex);

                        const queryRowBarEl = document.createElement('div');
                        queryRowBarEl.className = 'mhsa-token-matrix-preview__query-row-bar';
                        queryRowBarEl.style.background = rowData.gradientCss || 'none';

                        queryRowEl.appendChild(queryRowBarEl);
                        queryMatrixEl.appendChild(queryRowEl);
                        this._mhsaTokenMatrixQueryRowEls[rowData.rowIndex]?.push({
                            rowEl: queryRowEl,
                            barEl: queryRowBarEl,
                            stageIndex: queryStageIndex
                        });
                    });

                    const queryVisualEl = createVisualEl('mhsa-token-matrix-preview__visual--query');
                    queryVisualEl.appendChild(queryMatrixEl);
                    queryMatrixEl.dataset.mhsaConnectorTarget = 'q';
                    queryMatrixEl.dataset.mhsaConnectorColor = connectorColors.q || '#84b9ff';
                    queryMatrixEl.dataset.mhsaConnectorAnchorSide = 'top';
                    queryMatrixEl.dataset.mhsaConnectorRoute = 'horizontal';
                    this._mhsaTokenMatrixQueryMatrixEl.push({
                        matrixEl: queryMatrixEl,
                        stageIndex: queryStageIndex
                    });
                    sourceBlockEl.append(
                        queryVisualEl,
                        createCaptionEl(
                            scoreStage.queryLabelTex,
                            `(${scoreStage.queryRowCount}, ${scoreStage.queryColumnCount})`
                        )
                    );
                    return registerAttentionFocusEl('query', sourceBlockEl);
                };

                const buildAttentionSourceTransposeBlock = () => {
                    const transposeBlockEl = document.createElement('div');
                    transposeBlockEl.className = 'mhsa-token-matrix-preview__transpose-block';
                    const transposeMatrixEl = document.createElement('div');
                    transposeMatrixEl.className = 'mhsa-token-matrix-preview__transpose-matrix';
                    this._mhsaTokenMatrixTransposeMatrixEl = transposeMatrixEl;
                    scoreStage.transposeColumns.forEach((columnData) => {
                        const columnEl = document.createElement('div');
                        columnEl.className = 'mhsa-token-matrix-preview__transpose-col';
                        columnEl.dataset.col = String(columnData.colIndex);
                        columnEl.style.background = columnData.fillCss || 'none';
                        if (typeof columnData.tokenLabel === 'string' && columnData.tokenLabel.length) {
                            columnEl.title = columnData.tokenLabel;
                        }
                        transposeMatrixEl.appendChild(columnEl);
                        this._mhsaTokenMatrixTransposeColEls[columnData.colIndex]?.push({
                            colEl: columnEl
                        });
                    });
                    const transposeVisualEl = createVisualEl('mhsa-token-matrix-preview__visual--transpose');
                    transposeVisualEl.appendChild(transposeMatrixEl);
                    transposeBlockEl.dataset.mhsaConnectorTarget = 'k';
                    transposeBlockEl.dataset.mhsaConnectorColor = connectorColors.k || '#4cc47b';
                    transposeBlockEl.dataset.mhsaConnectorAnchorSide = 'bottom';
                    transposeBlockEl.dataset.mhsaConnectorRoute = 'horizontal';
                    transposeBlockEl.dataset.mhsaConnectorGap = String(
                        layoutMetrics?.connectorGaps?.transpose || layoutMetrics?.connectorGaps?.default || 16
                    );
                    transposeBlockEl.append(
                        transposeVisualEl,
                        createCaptionEl(
                            scoreStage.transposeLabelTex,
                            `(${scoreStage.transposeRowCount}, ${scoreStage.transposeColumnCount})`
                        )
                    );
                    return registerAttentionFocusEl('transpose', transposeBlockEl);
                };

                const buildPreScoreBlock = ({
                    connectorSource = 'pre',
                    connectorAnchorSide = 'right',
                    connectorRoute = 'horizontal',
                    connectorPlacement = 'matrix',
                    connectorGap = null
                } = {}) => {
                    return buildAttentionMatrixBlock({
                        rows: scoreStage.outputRows,
                        rowCount: scoreStage.outputRowCount,
                        columnCount: scoreStage.outputColumnCount,
                        labelTex: scoreStage.outputLabelTex,
                        focusKey: 'score',
                        interactive: true,
                        connectorSource,
                        connectorColor: '#f3f6fb',
                        connectorAnchorSide,
                        connectorRoute,
                        connectorPlacement,
                        connectorGap
                    });
                };

                const buildHeadOutputStage = () => {
                    const hasValueRows = Array.isArray(scoreStage.valueRows)
                        && scoreStage.valueRows.some((rowData) => typeof rowData?.gradientCss === 'string' && rowData.gradientCss.length);
                    const hasHeadOutputRows = Array.isArray(scoreStage.headOutputRows)
                        && scoreStage.headOutputRows.some((rowData) => Array.isArray(rowData?.rawValues) && rowData.rawValues.length);
                    if (!hasValueRows || !hasHeadOutputRows) return null;
                    const valueStageIndex = Number.isFinite(this._mhsaTokenMatrixValueStageIndex)
                        ? this._mhsaTokenMatrixValueStageIndex
                        : queryStageIndex;

                    const headOutputStageEl = document.createElement('div');
                    headOutputStageEl.className = 'mhsa-token-matrix-preview__head-output-stage';

                    const postCopyBlockEl = buildAttentionMatrixBlock({
                        blockClass: 'mhsa-token-matrix-preview__post-block mhsa-token-matrix-preview__post-block--head-copy',
                        visualModifierClass: 'mhsa-token-matrix-preview__visual--post',
                        matrixClass: 'mhsa-token-matrix-preview__post-matrix mhsa-token-matrix-preview__post-matrix--head-copy',
                        cellClass: 'mhsa-token-matrix-preview__post-cell',
                        rows: scoreStage.postRows,
                        rowCount: scoreStage.postRowCount,
                        columnCount: scoreStage.postColumnCount,
                        labelTex: scoreStage.postLabelTex,
                        focusKey: 'post-copy',
                        interactive: false,
                        connectorTarget: 'post',
                        connectorColor: '#f3f6fb',
                        connectorAnchorSide: 'left',
                        connectorRoute: 'horizontal',
                        connectorGap: layoutMetrics?.connectorGaps?.post || 8,
                        titleMode: 'attention',
                        cellStore: this._mhsaTokenMatrixPostCopyCellEls,
                        extraCellStores: [this._mhsaTokenMatrixMirroredPostCellEls],
                        rowStore: this._mhsaTokenMatrixPostCopyRowEls,
                        rowWrapperClass: 'mhsa-token-matrix-preview__post-row'
                    });

                    const multiplyOutputEl = createOperatorEl({
                        className: 'mhsa-token-matrix-preview__operator--attention-multiply mhsa-token-matrix-preview__operator--attention-multiply-output',
                        text: '×'
                    });
                    registerAttentionFocusEl('head-output-multiply', multiplyOutputEl);

                    const valuePostBlockEl = buildCompactRowBlock({
                        blockClass: 'mhsa-token-matrix-preview__query-block mhsa-token-matrix-preview__value-post-block',
                        visualModifierClass: 'mhsa-token-matrix-preview__visual--query mhsa-token-matrix-preview__visual--value-post',
                        matrixClass: 'mhsa-token-matrix-preview__compact-row-matrix mhsa-token-matrix-preview__compact-row-matrix--value-post',
                        rows: scoreStage.valueRows,
                        rowCount: scoreStage.valueRowCount,
                        columnCount: scoreStage.valueColumnCount,
                        labelTex: scoreStage.valueLabelTex,
                        focusKey: 'value-post',
                        connectorTarget: 'v',
                        connectorColor: connectorColors.v || '#f28b30',
                        connectorAnchorSide: 'bottom-left',
                        connectorRoute: 'horizontal',
                        connectorGap: layoutMetrics?.connectorGaps?.value || 18,
                        connectorAnchorInset: 4,
                        connectorPlacement: 'block',
                        rowStore: this._mhsaTokenMatrixCompactRowEls,
                        rowKind: 'v',
                        rowStageIndex: valueStageIndex
                    });

                    const equalsOutputEl = createOperatorEl({
                        className: 'mhsa-token-matrix-preview__operator--attention-equals',
                        text: '='
                    });
                    registerAttentionFocusEl('head-output-equals', equalsOutputEl);

                    const headOutputBlockEl = buildCompactRowBlock({
                        blockClass: 'mhsa-token-matrix-preview__query-block mhsa-token-matrix-preview__head-output-block',
                        visualModifierClass: 'mhsa-token-matrix-preview__visual--query mhsa-token-matrix-preview__visual--head-output',
                        matrixClass: 'mhsa-token-matrix-preview__compact-row-matrix mhsa-token-matrix-preview__compact-row-matrix--head-output',
                        rows: scoreStage.headOutputRows,
                        rowCount: scoreStage.headOutputRowCount,
                        columnCount: scoreStage.headOutputColumnCount,
                        labelTex: scoreStage.headOutputLabelTex,
                        focusKey: 'head-output',
                        rowStore: this._mhsaTokenMatrixCompactRowEls,
                        rowKind: 'head-output'
                    });

                    headOutputStageEl.append(
                        postCopyBlockEl,
                        multiplyOutputEl,
                        valuePostBlockEl,
                        equalsOutputEl,
                        headOutputBlockEl
                    );
                    return headOutputStageEl;
                };

                const buildMaskedAttentionStage = (headOutputStageEl = null) => {
                    const softmaxStageEl = document.createElement('div');
                    softmaxStageEl.className = 'mhsa-token-matrix-preview__softmax-stage';

                    const preScoreBlockEl = buildPreScoreBlock({
                        connectorAnchorSide: 'bottom',
                        connectorRoute: 'vertical',
                        connectorPlacement: 'block',
                        connectorGap: layoutMetrics?.connectorGaps?.pre || layoutMetrics?.connectorGaps?.default || 10
                    });
                    preScoreBlockEl.classList.add('mhsa-token-matrix-preview__softmax-pre-block');

                    const softmaxLabelEl = document.createElement('div');
                    softmaxLabelEl.className = 'mhsa-token-matrix-preview__softmax-label';
                    renderMathLabel(softmaxLabelEl, scoreStage.softmaxLabelTex, 'softmax');
                    registerAttentionFocusEl('softmax', softmaxLabelEl);

                    const softmaxFlowEl = document.createElement('div');
                    softmaxFlowEl.className = 'mhsa-token-matrix-preview__softmax-flow';

                    const softmaxPrefixEl = document.createElement('div');
                    softmaxPrefixEl.className = 'mhsa-token-matrix-preview__softmax-prefix';

                    const openParenEl = createOperatorEl({
                        className: 'mhsa-token-matrix-preview__operator--attention-paren mhsa-token-matrix-preview__operator--attention-paren-softmax mhsa-token-matrix-preview__operator--attention-paren-softmax-open',
                        text: '('
                    });
                    registerAttentionFocusEl('softmax-paren-open', openParenEl);

                    const plusEl = createOperatorEl({
                        className: 'mhsa-token-matrix-preview__operator--attention-plus',
                        text: '+'
                    });
                    registerAttentionFocusEl('add', plusEl);

                    const closeParenEl = createOperatorEl({
                        className: 'mhsa-token-matrix-preview__operator--attention-paren mhsa-token-matrix-preview__operator--attention-paren-softmax mhsa-token-matrix-preview__operator--attention-paren-softmax-close',
                        text: ')'
                    });
                    registerAttentionFocusEl('softmax-paren-close', closeParenEl);

                    const equalsEl = createOperatorEl({
                        className: 'mhsa-token-matrix-preview__operator--attention-equals-post',
                        text: '='
                    });
                    registerAttentionFocusEl('equals-post', equalsEl);

                    const maskedInputBlockEl = buildAttentionMatrixBlock({
                        blockClass: 'mhsa-token-matrix-preview__score-block mhsa-token-matrix-preview__score-block--masked-input',
                        visualModifierClass: 'mhsa-token-matrix-preview__visual--score',
                        matrixClass: 'mhsa-token-matrix-preview__score-matrix mhsa-token-matrix-preview__score-matrix--static',
                        cellClass: 'mhsa-token-matrix-preview__score-cell-static',
                        rows: scoreStage.outputRows,
                        rowCount: scoreStage.outputRowCount,
                        columnCount: scoreStage.outputColumnCount,
                        labelTex: scoreStage.outputLabelTex,
                        focusKey: 'masked-input',
                        interactive: false,
                        connectorTarget: 'pre',
                        connectorColor: '#f3f6fb',
                        connectorAnchorSide: 'top',
                        connectorRoute: 'vertical',
                        connectorGap: layoutMetrics?.connectorGaps?.pre || layoutMetrics?.connectorGaps?.default || 10,
                        titleMode: 'attention',
                        cellStore: this._mhsaTokenMatrixStaticScoreCellEls
                    });

                    const maskBlockEl = buildAttentionMatrixBlock({
                        blockClass: 'mhsa-token-matrix-preview__mask-block',
                        visualModifierClass: 'mhsa-token-matrix-preview__visual--mask',
                        matrixClass: 'mhsa-token-matrix-preview__mask-matrix',
                        cellClass: 'mhsa-token-matrix-preview__mask-cell',
                        rows: scoreStage.maskRows,
                        rowCount: scoreStage.outputRowCount,
                        columnCount: scoreStage.outputColumnCount,
                        labelTex: scoreStage.maskLabelTex,
                        focusKey: 'mask',
                        interactive: false,
                        titleMode: 'mask',
                        cellStore: this._mhsaTokenMatrixMaskCellEls
                    });

                    const postBlockEl = buildAttentionMatrixBlock({
                        blockClass: 'mhsa-token-matrix-preview__post-block',
                        visualModifierClass: 'mhsa-token-matrix-preview__visual--post',
                        matrixClass: 'mhsa-token-matrix-preview__post-matrix',
                        cellClass: 'mhsa-token-matrix-preview__post-cell',
                        rows: scoreStage.postRows,
                        rowCount: scoreStage.postRowCount,
                        columnCount: scoreStage.postColumnCount,
                        labelTex: scoreStage.postLabelTex,
                        focusKey: 'post',
                        interactive: false,
                        connectorSource: 'post',
                        connectorColor: '#f3f6fb',
                        connectorAnchorSide: 'right',
                        connectorRoute: 'horizontal',
                        connectorGap: layoutMetrics?.connectorGaps?.post || 8,
                        titleMode: 'attention',
                        cellStore: this._mhsaTokenMatrixPostCellEls,
                        extraCellStores: [this._mhsaTokenMatrixMirroredPostCellEls]
                    });

                    softmaxPrefixEl.append(
                        softmaxLabelEl,
                        openParenEl
                    );

                    softmaxFlowEl.append(
                        softmaxPrefixEl,
                        maskedInputBlockEl,
                        plusEl,
                        maskBlockEl,
                        closeParenEl,
                        equalsEl,
                        postBlockEl
                    );
                    if (headOutputStageEl) {
                        softmaxFlowEl.appendChild(headOutputStageEl);
                    }
                    softmaxStageEl.append(
                        preScoreBlockEl,
                        softmaxFlowEl
                    );
                    return softmaxStageEl;
                };

                const attentionEquationEl = document.createElement('div');
                attentionEquationEl.className = 'mhsa-token-matrix-preview__attention-equation';

                const openParenEl = createOperatorEl({
                    className: 'mhsa-token-matrix-preview__operator--attention-paren mhsa-token-matrix-preview__operator--attention-paren-pre-open',
                    text: '('
                });
                registerAttentionFocusEl('paren-pre-open', openParenEl);

                const multiplyEl = createOperatorEl({
                    className: 'mhsa-token-matrix-preview__operator--attention-multiply',
                    text: '×'
                });
                registerAttentionFocusEl('multiply', multiplyEl);

                const closeParenEl = createOperatorEl({
                    className: 'mhsa-token-matrix-preview__operator--attention-paren mhsa-token-matrix-preview__operator--attention-paren-pre-close',
                    text: ')'
                });
                registerAttentionFocusEl('paren-pre-close', closeParenEl);

                const divideEl = createOperatorEl({
                    className: 'mhsa-token-matrix-preview__operator--attention-divide',
                    text: '/'
                });
                registerAttentionFocusEl('divide', divideEl);

                const equalsEl = createOperatorEl({
                    className: 'mhsa-token-matrix-preview__operator--attention-equals mhsa-token-matrix-preview__operator--attention-equals-pre',
                    text: '='
                });
                registerAttentionFocusEl('equals-pre', equalsEl);

                const headOutputStageEl = buildHeadOutputStage();
                const maskedAttentionStageEl = buildMaskedAttentionStage(headOutputStageEl);

                attentionEquationEl.append(
                    openParenEl,
                    buildAttentionSourceQueryBlock(),
                    multiplyEl,
                    buildAttentionSourceTransposeBlock(),
                    closeParenEl,
                    divideEl,
                    buildScaleBlock(),
                    equalsEl,
                    maskedAttentionStageEl
                );

                flowEl.append(attentionEquationEl);
                attentionStageEl.appendChild(flowEl);
                this._mhsaTokenMatrixAttentionFocusEls = attentionFocusEls;
                return attentionStageEl;
            };

            previewData.projections.forEach((projectionData, stageIndex) => {
                const validProjection = projectionData
                    && projectionData.weightRowCount
                    && projectionData.weightColumnCount
                    && projectionData.outputRowCount
                    && projectionData.outputColumnCount
                    && Array.isArray(projectionData.outputRows);
                if (!validProjection) return;
                const stageEl = buildStageEl(projectionData, stageIndex);
                stageEl.style.gridColumn = '1';
                stageEl.style.gridRow = String(stageIndex + 1);
                stageStackEl.appendChild(stageEl);
            });

            if (
                previewData.attentionScoreStage
                && Array.isArray(previewData.attentionScoreStage.queryRows)
                && Array.isArray(previewData.attentionScoreStage.outputRows)
            ) {
                stageStackEl.classList.add('has-attention-sidecar');
                const queryStageIndex = Math.max(0, previewData.projections.findIndex((projectionData) => projectionData?.kind === 'Q'));
                this._mhsaTokenMatrixQueryStageIndex = queryStageIndex;
                const connectorColors = previewData.projections.reduce((acc, projectionData) => {
                    const kind = String(projectionData?.kind || '').toLowerCase();
                    if (
                        kind
                        && Array.isArray(projectionData?.colorRgb)
                        && projectionData.colorRgb.length === 3
                    ) {
                        acc[kind] = `rgb(${projectionData.colorRgb.join(', ')})`;
                    }
                    return acc;
                }, {});
                const attentionStageEl = buildAttentionScoreStageEl(
                    previewData.attentionScoreStage,
                    queryStageIndex,
                    connectorColors
                );
                attentionStageEl.style.gridColumn = '2';
                attentionStageEl.style.gridRow = '1 / span 2';
                stageStackEl.appendChild(attentionStageEl);
            }

            if (!stageStackEl.childNodes.length) {
                this._setMhsaTokenMatrixStatus('Token matrix unavailable.');
                return;
            }
            workspaceEl.append(stageStackEl, overlayEl);
            fragment.append(workspaceEl);

            this.mhsaTokenMatrixBody.replaceChildren(fragment);
            this._mhsaTokenMatrixWorkspace = workspaceEl;
            this._mhsaTokenMatrixOverlay = overlayEl;
            this._syncMhsaTokenMatrixCanvasPresentation({ domReady: true });
            this._resetMhsaTokenMatrixViewport();
            requestAnimationFrame(() => {
                if (
                    renderToken !== this._mhsaTokenMatrixRenderToken
                    || !this._mhsaTokenMatrixWorkspace
                    || this.mhsaTokenMatrixBody?.hidden
                    || this._mhsaTokenMatrixViewport?.hasInteracted
                ) {
                    return;
                }
                const centeredViewport = this._resolveMhsaTokenMatrixViewportCenter();
                if (!centeredViewport) return;
                this._mhsaTokenMatrixViewport.panX = centeredViewport.panX;
                this._mhsaTokenMatrixViewport.panY = centeredViewport.panY;
                this._mhsaTokenMatrixViewport.initialized = true;
                this._applyMhsaTokenMatrixViewport();
            });
            this._scheduleMhsaTokenMatrixConnectorUpdate();
            this._scheduleMhsaTokenMatrixCanvasRender();
            this._setMhsaTokenMatrixStatus('');
        } catch (error) {
            if (renderToken !== this._mhsaTokenMatrixRenderToken) return;
            console.warn('Failed to build MHSA token matrix preview:', error);
            this._setMhsaTokenMatrixStatus('Failed to load token matrix.');
        }
    }

    _onDocumentPointerDown(event) {
        if (!Number.isFinite(event.clientX) || !Number.isFinite(event.clientY)) return;
        const eventTarget = event.target instanceof Element ? event.target : null;
        const hit = typeof document.elementFromPoint === 'function'
            ? document.elementFromPoint(event.clientX, event.clientY)
            : null;
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
        const attentionBodyRoot = resolveClosest('#detailAttentionBody');
        const insideAttentionBody = !!(
            this.attentionBody
            && attentionBodyRoot === this.attentionBody
        );
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
        const panelHit = (hit && this.panel && this.panel.contains(hit))
            ? hit
            : (eventTarget && this.panel && this.panel.contains(eventTarget) ? eventTarget : null);
        this._setPanelKeyboardEngaged(!!panelHit);
        const mhsaTokenMatrixRoot = resolveClosest('#detailMhsaTokenMatrixBody');
        const insideMhsaTokenMatrix = !!(
            this.mhsaTokenMatrixBody
            && mhsaTokenMatrixRoot === this.mhsaTokenMatrixBody
        );
        const mhsaTokenMatrixTarget = insideMhsaTokenMatrix
            ? this._resolveMhsaTokenMatrixInteractionTarget(
                hit && this.mhsaTokenMatrixBody?.contains(hit) ? hit : eventTarget,
                { requirePinnableStageComponent: true }
            )
            : null;
        const shouldClearPinnedAttention = this.isOpen
            && shouldClearPinnedAttentionOnDocumentPointerDown({
                isPinned: this._attentionPinned,
                hitPanelTokenNavChip,
                hitPanelAttentionScoreLink,
                insideAttentionBody,
                insideAttentionMatrix,
                validMatrixCell,
                panelHit: !!panelHit
            });
        if (shouldClearPinnedAttention) {
            this._clearPinnedAttention({ clearSelectionSummary: true });
        }
        const shouldClearPinnedMhsaTokenMatrix = this.isOpen
            && this._mhsaTokenMatrixPinned
            && !hitPanelTokenNavChip
            && !hitPanelAttentionScoreLink
            && (!insideMhsaTokenMatrix || !mhsaTokenMatrixTarget);
        if (shouldClearPinnedMhsaTokenMatrix) {
            this._clearPinnedMhsaTokenMatrix();
        }
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

    _onDocumentFocusIn(event) {
        const focusTarget = event?.target instanceof Element ? event.target : null;
        this._setPanelKeyboardEngaged(!!(
            focusTarget
            && this.panel
            && this.panel.contains(focusTarget)
        ));
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
        return !!(
            this.isReady
            && this.isOpen
            && this.currentPreview
            && !this._geluDetailOpen
            && !this._softmaxDetailOpen
            && !this._previewPausedForPanelResize
        );
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

    _renderPreviewSnapshot({ syncEnvironment = true } = {}) {
        if (!this.isReady || !this.isOpen || !this.currentPreview || this._geluDetailOpen || this._softmaxDetailOpen) {
            return false;
        }
        if (syncEnvironment) {
            this._syncEnvironment();
        }
        this.renderer.render(this.scene, this.camera);
        return true;
    }

    _renderPreviewFrame(time) {
        if (!this.isReady || !this.isOpen || !this.currentPreview || this._geluDetailOpen || this._softmaxDetailOpen || this._previewPausedForPanelResize) {
            return false;
        }
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
        return this._renderPreviewSnapshot({ syncEnvironment: false });
    }

    _animate(time) {
        this._previewRafId = null;
        if (!this._isPreviewLoopActive()) return;
        this._renderPreviewFrame(time);
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

    _setInfoPreview(config = null) {
        const infoType = (config?.type === 'kvCache' || config?.type === 'mhsa')
            ? config.type
            : null;
        const showInfoPreview = !!infoType;
        this.panel?.classList.toggle('is-info-preview', showInfoPreview);
        this.panel?.classList.toggle('is-mhsa-info-preview', infoType === 'mhsa');
        if (!this.infoPreview) return;

        this.infoPreview.hidden = !showInfoPreview;
        this.infoPreview.setAttribute('aria-hidden', showInfoPreview ? 'false' : 'true');
        if (infoType !== 'mhsa') {
            this._hideMhsaTokenMatrixPreview();
        }
        if (!showInfoPreview) return;

        const setText = (element, value) => {
            if (element) element.textContent = value;
        };
        const setStep = (element, label, active = false) => {
            if (!element) return;
            element.textContent = label;
            element.dataset.active = active ? 'true' : 'false';
        };

        if (infoType === 'mhsa') {
            setText(this.infoPreviewEyebrow, '');
            setText(this.infoPreviewTitle, '');
            setText(this.infoPreviewSummary, '');
            setText(this.infoPreviewPhase, '');
            setStep(this.infoPreviewStepPrefill, '', false);
            setStep(this.infoPreviewStepCache, '', false);
            setStep(this.infoPreviewStepDecode, '', false);
            setText(this.infoPreviewCellStoreLabel, '');
            setText(this.infoPreviewCellPassLabel, '');
            setText(this.infoPreviewCellBenefitLabel, '');
            setText(this.infoPreviewCellStore, '');
            setText(this.infoPreviewCellPass, '');
            setText(this.infoPreviewCellBenefit, '');
            this._renderMhsaTokenMatrixPreview();
            return;
        }

        const phase = normalizeKvCachePhase(config?.phase);
        const phaseLabel = formatKvCachePhaseLabel(phase);
        const passCopy = phase === 'decode'
            ? 'Append one new token and attend over the cached prefix'
            : 'Compute the prompt once and write its cache entries'
            ;

        setText(this.infoPreviewEyebrow, 'Transformer inference');
        setText(this.infoPreviewTitle, 'KV Cache');
        setText(
            this.infoPreviewSummary,
            phase === 'decode'
                ? 'Reuse the cached prefix and compute fresh attention work only for the newest token.'
                : 'Build the prompt cache once so later decode steps can reuse it.'
        );
        setText(this.infoPreviewPhase, phaseLabel);
        setStep(this.infoPreviewStepPrefill, 'Pre-fill', phase === 'prefill');
        setStep(this.infoPreviewStepCache, 'Reuse cache', true);
        setStep(this.infoPreviewStepDecode, 'Decode', phase === 'decode');
        setText(this.infoPreviewCellStoreLabel, 'Stores');
        setText(this.infoPreviewCellPassLabel, 'Current pass');
        setText(this.infoPreviewCellBenefitLabel, 'Why faster');
        setText(this.infoPreviewCellStore, 'Keys + values from earlier tokens');
        setText(this.infoPreviewCellPass, passCopy);
        setText(this.infoPreviewCellBenefit, 'Avoid recomputing the full prefix each step');
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
        this._updateMhsaFullscreenToggle();
        this._updateResizeHandleState();
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
        if (this._transformerView2dSelectionSidebarRestoreTimer !== null) {
            clearTimeout(this._transformerView2dSelectionSidebarRestoreTimer);
            this._transformerView2dSelectionSidebarRestoreTimer = null;
        }
        this._isMhsaInfoSelectionActive = false;
        this._mhsaFullscreenActive = false;
        this.isOpen = false;
        this._syncMhsaViewRoute(false);
        this._syncMhsaFullscreenDocumentState();
        this._stopLoop();
        this._closeSoftmaxDetailPreview({ restoreSelection: false, restartLoop: false });
        this._closeGeluDetailPreview({ restoreSelection: false, restartLoop: false });
        this._closeTransformerView2dPreview({ restoreSelection: false, restartLoop: false });
        this._currentSelectionDescription = '';
        this._currentSelectionEquations = '';
        renderSelectionPreviewEquations(this.equationsBody, []);
        this._lastSelection = null;
        this._lastSelectionLabel = '';
        if (clearHistory) {
            this._resetHistoryNavigation();
        } else {
            this._updateHistoryNavigationControls();
        }
        this._setPanelTokenHoverEntry(null, { emit: true });
        this._mirroredTokenHoverEntries = [];
        this._setPanelKeyboardEngaged(false);
        this._applyTokenChipHoverState();
        this._hideLegendHover();
        this._resetCopyContextFeedback();
        this._cancelPanelResizeDrag();
        this.panel.classList.remove('is-open');
        this.panel.classList.remove('is-info-preview');
        this.panel.classList.remove('is-preview-hidden');
        this.panel.classList.remove('is-softmax-view-open');
        this.panel.classList.remove('is-gelu-view-open');
        this.panel.classList.remove('is-transformer-view2d-open');
        this.panel.classList.remove('is-mhsa-fullscreen');
        this.hudStack?.classList.remove('detail-open');
        this.hudPanel?.classList.remove('detail-open');
        this.panel.setAttribute('aria-hidden', 'true');
        this._setHoverLabelSuppression(false);
        this._updateMobileState();
        this._updateMhsaFullscreenToggle();
        this._updateResizeHandleState();
        this._syncSceneShift();
        if (this.description) setDescriptionContent(this.description, '');
        if (this.equationsBody) setDescriptionContent(this.equationsBody, '');
        if (this.equationsBody) this.equationsBody.style.fontSize = '';
        this._setInfoPreview(null);
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
        this._cancelScheduledResize();
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
        if (resolveLayerNormParamPreviewSpec(label, selection)) return null;

        let laneIndex = findUserDataNumber(selection, 'laneIndex');
        if (!Number.isFinite(laneIndex)) {
            const laneLayoutIndex = findUserDataNumber(selection, 'laneLayoutIndex');
            if (Number.isFinite(laneLayoutIndex)) laneIndex = laneLayoutIndex;
        }
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
        const incompleteDisplayText = getIncompleteUtf8TokenDisplay(tokenId);
        if (incompleteDisplayText) {
            tokenText = incompleteDisplayText;
            tokenDisplayText = incompleteDisplayText;
        }
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
        const isMlpDownSelection = lower.startsWith('mlp down projection')
            || activationStage === 'mlp.down';
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
            const tokenId = resolveLogitSelectionTokenId(label, entry, selection);
            const probability = resolveLogitSelectionProbability(label, entry, selection);
            const tokenIdText = Number.isFinite(tokenId) ? String(Math.floor(tokenId)) : ATTENTION_VALUE_PLACEHOLDER;
            const probabilityText = Number.isFinite(probability)
                ? formatLogitProbability(probability)
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

        const isResidualStreamSelection = lower.includes('residual stream vector')
            || isPostLayerNormResidualSelection({
                label,
                stage: activationStage
            })
            || activationStage.startsWith('embedding.sum')
            || activationStage.startsWith('layer.incoming')
            || activationStage === 'residual.post_attention'
            || activationStage === 'residual.post_mlp';
        if (isResidualStreamSelection) {
            hideRows();
            return metadata;
        }

        if (isMlpDownSelection) {
            hideRows();
            return metadata;
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

        entries.forEach((entry, index) => {
            const chip = document.createElement('span');
            const isSelected = index === activeIndex;
            chip.className = 'detail-subtitle-token-chip detail-token-nav-chip detail-prompt-context-token';
            if (isSelected) {
                chip.classList.add('detail-prompt-context-token--selected');
                applyTokenChipColors(chip, entry, index);
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
        const logitTokenId = resolveLogitEntryTokenId(entry);
        if (Number.isFinite(logitTokenId)) {
            return getIncompleteUtf8TokenNote(logitTokenId);
        }

        const lower = (label || '').toLowerCase();
        if (!lower.includes('logit')) return '';
        const labelIdMatch = String(label || '').match(/\bid\s+(-?\d+)/i);
        if (!labelIdMatch) return '';
        const parsedTokenId = Number(labelIdMatch[1]);
        return getIncompleteUtf8TokenNote(parsedTokenId);
    }

    _resolveDescriptionTokenContext(selection, {
        indexKey = 'tokenIndex',
        labelKey = 'tokenLabel',
        idKey = 'tokenId',
        fallbackIndex = null,
        fallbackLabel = '',
        fallbackId = null
    } = {}) {
        let tokenIndex = findUserDataNumber(selection, indexKey);
        if (!Number.isFinite(tokenIndex) && Number.isFinite(fallbackIndex)) {
            tokenIndex = Math.floor(fallbackIndex);
        }

        let tokenLabel = findUserDataString(selection, labelKey);
        if ((typeof tokenLabel !== 'string' || !tokenLabel.trim().length) && typeof fallbackLabel === 'string' && fallbackLabel.trim().length) {
            tokenLabel = fallbackLabel;
        }
        if ((!tokenLabel || !tokenLabel.trim().length)
            && Number.isFinite(tokenIndex)
            && this.activationSource
            && typeof this.activationSource.getTokenString === 'function') {
            const resolvedLabel = this.activationSource.getTokenString(tokenIndex);
            if (typeof resolvedLabel === 'string' && resolvedLabel.trim().length) {
                tokenLabel = resolvedLabel;
            }
        }
        const formattedTokenLabel = (typeof tokenLabel === 'string' && tokenLabel.trim().length)
            ? formatTokenLabelForPreview(tokenLabel)
            : '';

        let tokenId = findUserDataNumber(selection, idKey);
        if (!Number.isFinite(tokenId) && Number.isFinite(fallbackId)) {
            tokenId = Math.floor(fallbackId);
        }
        if (!Number.isFinite(tokenId)
            && Number.isFinite(tokenIndex)
            && this.activationSource
            && typeof this.activationSource.getTokenId === 'function') {
            tokenId = this.activationSource.getTokenId(tokenIndex);
        }

        return {
            tokenIndex: Number.isFinite(tokenIndex) ? Math.floor(tokenIndex) : null,
            tokenLabel: formattedTokenLabel,
            tokenId: Number.isFinite(tokenId) ? Math.floor(tokenId) : null
        };
    }

    _buildDescriptionSelectionContext(selection, {
        vectorTokenMetadata = null,
        attentionScoreSummary = null
    } = {}) {
        if (!selection || typeof selection !== 'object') return selection;

        const baseInfo = (selection.info && typeof selection.info === 'object') ? selection.info : {};
        const info = { ...baseInfo };
        const baseActivation = (baseInfo.activationData && typeof baseInfo.activationData === 'object')
            ? baseInfo.activationData
            : null;
        const activationData = baseActivation ? { ...baseActivation } : null;

        const applyTokenContext = ({
            indexKey,
            labelKey,
            idKey,
            fallbackIndex = null,
            fallbackLabel = '',
            fallbackId = null
        }) => {
            const tokenContext = this._resolveDescriptionTokenContext(selection, {
                indexKey,
                labelKey,
                idKey,
                fallbackIndex,
                fallbackLabel,
                fallbackId
            });
            if (Number.isFinite(tokenContext.tokenIndex)) {
                info[indexKey] = tokenContext.tokenIndex;
                if (activationData) activationData[indexKey] = tokenContext.tokenIndex;
            }
            if (tokenContext.tokenLabel) {
                info[labelKey] = tokenContext.tokenLabel;
                if (activationData) activationData[labelKey] = tokenContext.tokenLabel;
            }
            if (Number.isFinite(tokenContext.tokenId)) {
                info[idKey] = tokenContext.tokenId;
                if (activationData) activationData[idKey] = tokenContext.tokenId;
            }
        };

        applyTokenContext({
            indexKey: 'tokenIndex',
            labelKey: 'tokenLabel',
            idKey: 'tokenId',
            fallbackIndex: vectorTokenMetadata?.tokenIndex,
            fallbackLabel: vectorTokenMetadata?.tokenDisplayText || vectorTokenMetadata?.tokenText || '',
            fallbackId: vectorTokenMetadata?.tokenId
        });

        applyTokenContext({
            indexKey: 'keyTokenIndex',
            labelKey: 'keyTokenLabel',
            idKey: 'keyTokenId',
            fallbackIndex: attentionScoreSummary?.defaultValue?.targetTokenIndex,
            fallbackLabel: attentionScoreSummary?.defaultValue?.target || '',
            fallbackId: null
        });

        applyTokenContext({
            indexKey: 'queryTokenIndex',
            labelKey: 'queryTokenLabel',
            idKey: 'queryTokenId',
            fallbackIndex: null,
            fallbackLabel: '',
            fallbackId: null
        });

        if (activationData) {
            info.activationData = activationData;
        }

        return {
            ...selection,
            info
        };
    }

    showSelection(selection, options = {}) {
        if (!this.isReady || !selection || !selection.label) return;
        const fromHistory = options?.fromHistory === true;
        const scrollPanelToTop = options?.scrollPanelToTop === true;
        const selectionSurface = resolveSelectionPanelSurface({
            selectionSurface: options?.selectionSurface,
            preserveTransformerView2d: options?.preserveTransformerView2d === true,
            transformerView2dDetailOpen: this._transformerView2dDetailOpen
        });
        const preserveTransformerView2d = isTransformerView2dSelectionSurface({
            selectionSurface
        }) && this._transformerView2dDetailOpen;

        this._closeSoftmaxDetailPreview({ restoreSelection: false, restartLoop: false });
        this._closeGeluDetailPreview({ restoreSelection: false, restartLoop: false });
        if (!preserveTransformerView2d) {
            this._closeTransformerView2dPreview({ restoreSelection: false, restartLoop: false });
        }
        this._resetCopyContextFeedback();
        const label = normalizeSelectionLabel(selection.label, selection);
        this._lastSelection = selection;
        this._lastSelectionLabel = label;
        const displayLabel = simplifyLayerNormParamDisplayLabel(label, selection);
        const lower = label.toLowerCase();
        const isKvCacheInfo = isKvCacheInfoSelection(label, selection);
        const isMhsaInfo = isMhsaInfoSelection(label, selection);
        this._isMhsaInfoSelectionActive = isMhsaInfo;
        this._syncMhsaViewRoute(isMhsaInfo);
        if (!isMhsaInfo && this._mhsaFullscreenActive && !preserveTransformerView2d) {
            this._mhsaFullscreenActive = false;
            this.panel?.classList.remove('is-mhsa-fullscreen');
            this._syncMhsaFullscreenDocumentState();
            this._updateResizeHandleState();
            this._syncSceneShift({ immediate: true });
            this._scheduleResize();
        }
        this._updateMhsaFullscreenToggle();
        const transformerView2dContext = resolveTransformerView2dActionContext(selection, label);
        this._currentTransformerView2dContext = transformerView2dContext;
        this._setTransformerView2dActionButtonState(this._resolveSelectionPrimaryActionConfig({
            view2dContext: transformerView2dContext
        }));
        const kvCachePhase = isKvCacheInfo
            ? normalizeKvCachePhase(findUserDataString(selection, 'kvCachePhase'))
            : 'prefill';
        const kvCachePhaseLabel = isKvCacheInfo
            ? (findUserDataString(selection, 'kvCachePhaseLabel') || formatKvCachePhaseLabel(kvCachePhase))
            : '';
        const infoPreviewConfig = isKvCacheInfo
            ? { type: 'kvCache', phase: kvCachePhase }
            : (isMhsaInfo ? { type: 'mhsa' } : null);
        const hidePreviewForSelection = isKvCacheInfo;
        const suppressScenePreview = isKvCacheInfo || isMhsaInfo;
        const previewSelectionKey = suppressScenePreview ? null : buildSelectionPreviewKey(label, selection);
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
        let subtitleTertiaryText = '';
        const isTokenChipSelection = lower.startsWith('token:');
        const isTokenOrPositionChipSelection = isTokenChipSelection || lower.startsWith('position:');
        const isChosenTokenSelection = lower.startsWith('chosen token:');
        const isPositionEmbeddingSelection = lower.startsWith('position:')
            || lower.includes('position embedding')
            || lower.includes('positional embedding')
            || activationStage.startsWith('embedding.position');
        this._setInfoPreview(infoPreviewConfig);
        this.panel.classList.toggle('is-preview-hidden', hidePreviewForSelection);
        if (logitHeader) {
            this._setTitleText(logitHeader.title);
        } else if (isKvCacheInfo) {
            this._setTitleText('KV Cache');
        } else if (isMhsaInfo) {
            const layerIndex = findUserDataNumber(selection, 'layerIndex');
            this._setTitleText(formatMhsaInfoTitle(
                Number.isFinite(layerIndex) ? Math.floor(layerIndex) : null
            ));
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
            } else if (isKvCacheInfo) {
                this.subtitle.classList.remove('detail-subtitle--qkv-token-context');
                this.subtitle.textContent = kvCachePhase === 'decode'
                    ? 'Enabled • Single-token decode pass'
                    : 'Enabled • Prompt pre-fill pass';
            } else if (isMhsaInfo) {
                this.subtitle.classList.remove('detail-subtitle--qkv-token-context');
                this.subtitle.textContent = '';
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
                    subtitleParts.push(`Attention Head ${headIndex + 1}`);
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
            } else if (isKvCacheInfo) {
                this._setSubtitleSecondaryText(`Current phase: ${kvCachePhaseLabel}`);
            } else if (isMhsaInfo) {
                this._setSubtitleSecondaryText('');
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
        const layerNormParameterSummary = hideLayerNormFields
            ? resolveLayerNormParameterSummary(selection, this.engine)
            : null;
        const layerNormProductStageSummary = resolveLayerNormProductStageSummary(selection, this.engine);
        const isLogitTokenSelection = !!logitHeader;
        const hideTensorDimsField = hideLayerNormFields
            || isAttentionScore
            || isLogitTokenSelection
            || isTokenOrPositionChipSelection
            || isChosenTokenSelection
            || isKvCacheInfo
            || isMhsaInfo;
        const isVectorMetadata = isLikelyVectorSelection(label, selection) || isTokenOrPositionChipSelection;
        if (isVectorMetadata && metadata.hasLength && !isTokenOrPositionChipSelection) {
            subtitleTertiaryText = `Length: ${metadata.length}`;
        }
        const dimsRow = this.inputDim?.closest('.detail-row')
            || this.outputDim?.closest('.detail-row')
            || null;
        if (this.inputDimLabel) {
            this.inputDimLabel.textContent = metadata.inputDimLabel || DEFAULT_METADATA_DIMENSION_LABELS.input;
        }
        if (this.outputDimLabel) {
            this.outputDimLabel.textContent = metadata.outputDimLabel || DEFAULT_METADATA_DIMENSION_LABELS.output;
        }
        if (this.inputDim) this.inputDim.textContent = hideTensorDimsField
            ? ''
            : metadata.inputDim;
        if (this.outputDim) this.outputDim.textContent = hideTensorDimsField || isVectorMetadata ? '' : metadata.outputDim;
        if (this.outputDimHalf) this.outputDimHalf.style.display = (!hideTensorDimsField && isVectorMetadata) ? 'none' : '';
        if (this.inputDimHalf) this.inputDimHalf.style.flexBasis = isVectorMetadata ? '100%' : '';
        if (dimsRow) dimsRow.style.display = hideTensorDimsField || isVectorMetadata ? 'none' : '';
        const showParamCount = (!!layerNormParameterSummary)
            || (!hideTensorDimsField && !isVectorMetadata && metadata.hasDims);
        if (this.paramsRow) this.paramsRow.style.display = showParamCount ? '' : 'none';
        if (this.params) {
            if (layerNormParameterSummary) {
                setLayerNormParameterSummaryContent(this.params, layerNormParameterSummary);
            } else {
                this.params.textContent = showParamCount ? metadata.params : '';
            }
        }
        if (this.layerNormStageRow) {
            this.layerNormStageRow.style.display = layerNormProductStageSummary ? '' : 'none';
        }
        if (this.layerNormStageLabel) {
            this.layerNormStageLabel.textContent = layerNormProductStageSummary?.label || 'Current stage';
        }
        if (this.layerNormStageValue) {
            this.layerNormStageValue.textContent = layerNormProductStageSummary?.value || '';
        }
        const showBiasDim = !hideTensorDimsField && !isVectorMetadata && metadata.hasBiasDim;
        if (this.biasDimRow) this.biasDimRow.style.display = showBiasDim ? '' : 'none';
        if (this.biasDim) this.biasDim.textContent = showBiasDim ? metadata.biasDim : '';
        this._setSubtitleTertiaryText(subtitleTertiaryText);
        if (this.previewMetaSection) {
            const topRows = Array.from(this.previewMetaSection.querySelectorAll('.detail-row, .detail-token-info'));
            const hasVisibleTopRow = !isKvCacheInfo && !isMhsaInfo && topRows.some(row => row.style.display !== 'none');
            this.previewMetaSection.style.display = hasVisibleTopRow ? '' : 'none';
        }
        const vectorTokenMetadata = this._updateVectorTokenPositionRows(selection, label);
        const descriptionSelection = this._buildDescriptionSelectionContext(selection, {
            vectorTokenMetadata,
            attentionScoreSummary
        });
        this._setTokenEncodingNote(this._resolveSelectionTokenEncodingNote(selection, label, vectorTokenMetadata));
        this._updatePromptContextRow(vectorTokenMetadata, { visible: isTokenChipSelection });
        if (this.description) {
            const desc = resolveDescription(label, selection.kind, descriptionSelection);
            this._currentSelectionDescription = getDescriptionPlainText(desc || '');
            setDescriptionContent(this.description, desc || '');
            setDescriptionSoftmaxAction(this.description, isPostSoftmaxAttentionSelection(descriptionSelection, label));
            setDescriptionGeluAction(this.description, isGeluDetailSelection({
                label,
                stage: getActivationDataFromSelection(descriptionSelection)?.stage || ''
            }));
            setDescriptionMhsaInfoAction(this.description, false);
        } else {
            this._currentSelectionDescription = '';
        }
        const previewEquations = resolveSelectionPreviewEquations(label, descriptionSelection);
        if (this.equationsSection && this.equationsBody) {
            const equations = resolveSelectionEquations(label, descriptionSelection);
            this._currentSelectionEquations = equations || '';
            renderSelectionPreviewEquations(this.equationsBody, previewEquations);
            const hasEquations = previewEquations.length > 0;
            this.equationsSection.classList.toggle('is-visible', hasEquations);
            this.equationsSection.setAttribute('aria-hidden', hasEquations ? 'false' : 'true');
            this.equationsBody.style.fontSize = '';
        } else {
            this._currentSelectionEquations = resolveSelectionEquations(label, descriptionSelection) || '';
        }
        const vectorSamplingData = (appState.devMode && !hideLayerNormFields && !isKvCacheInfo && !isMhsaInfo)
            ? resolveSelectionVectorSamplingData({
                label,
                selectionInfo: selection,
                activationSource: this.activationSource,
                fallbackValues: extractPreviewVectorData(selection)
            })
            : null;
        const showVectorSampling = !!vectorSamplingData;
        if (this.dataSection) {
            this.dataSection.style.display = showVectorSampling ? '' : 'none';
        }
        if (this.dataTitle) {
            this.dataTitle.textContent = vectorSamplingData?.title || 'Vector sampling';
        }
        if (this.dataEyebrow) {
            this.dataEyebrow.textContent = 'DEV MODE';
        }
        if (this.dataBlurb) {
            const blurb = typeof vectorSamplingData?.description === 'string'
                ? vectorSamplingData.description.trim()
                : '';
            this.dataBlurb.textContent = blurb;
            this.dataBlurb.hidden = !showVectorSampling || !blurb;
        }
        if (this.dataSection) {
            if (showVectorSampling && vectorSamplingData?.text) {
                this.dataSection.dataset.copyText = vectorSamplingData.text;
            } else {
                delete this.dataSection.dataset.copyText;
            }
        }
        if (this.metaSection) {
            const rows = Array.from(this.metaSection.querySelectorAll('.detail-row, .detail-token-info'));
            const hasVisibleRow = !isKvCacheInfo && !isMhsaInfo && rows.some(row => row.style.display !== 'none');
            this.metaSection.style.display = hasVisibleRow ? '' : 'none';
        }
        renderSelectionVectorSamplingData(this.dataEl, showVectorSampling ? vectorSamplingData : null);
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
            : (suppressScenePreview ? null : resolvePreviewObject(label, selection, this.engine));
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
            const isSmallScreen = this._isSmallScreen();
            this._rotationSpeedMult = 1;
            this._lastFitOptions = resolveSelectionPreviewFitOptions(
                label,
                selection,
                this.currentPreview,
                preview.view,
                isSmallScreen
            );
            if (!this.isOpen) {
                this._pendingReveal = true;
                if (this.canvas) this.canvas.style.opacity = '0';
            } else {
                this._pendingReveal = false;
                if (this.canvas) this.canvas.style.opacity = '1';
                fitObjectToView(
                    this.currentPreview,
                    this.camera,
                    this._resolveCurrentPreviewFitOptions()
                );
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
        if (preserveTransformerView2d) {
            this._showTransformerView2dSelectionSidebar({ scrollToTop: true });
        } else if (isMhsaInfo && !this._isSmallScreen()) {
            this._setMhsaFullscreen(true);
        }
        if (scrollPanelToTop) {
            if (preserveTransformerView2d) {
                this._transformerView2dDetailView?.scrollSelectionSidebarToTop?.();
            } else {
                this._scrollPanelToTop();
            }
        }
        this._scheduleSelectionEquationFit();
        this._scheduleDimensionLabelFit();
        if (!fromHistory) {
            const entry = this._buildHistoryEntry('selection', selection, {
                selectionSurface
            });
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
        return {
            handleSelection: () => {},
            close: () => {},
            updateData: () => {},
            openTransformerView2d: () => false
        };
    }
    return {
        handleSelection: (selection) => panel.showSelection(selection),
        close: () => panel.close(),
        updateData: (data) => panel.updateData(data),
        openTransformerView2d: (config) => panel.openTransformerView2d(config)
    };
}
