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
import {
    createResidualVectorMatrixNode,
    RESIDUAL_VECTOR_MEASURE_COLS
} from './createResidualVectorMatrixNode.js';
import { buildHeadDetailSceneModel } from './buildHeadDetailSceneModel.js';
import { buildLayerNormDetailSceneModel } from './buildLayerNormDetailSceneModel.js';
import { buildMlpDetailSceneModel } from './buildMlpDetailSceneModel.js';
import { buildMhsaSceneModel } from './buildMhsaSceneModel.js';
import { buildOutputProjectionDetailSceneModel } from './buildOutputProjectionDetailSceneModel.js';
import {
    buildPositionEmbeddingModule,
    buildUnembeddingModule,
    buildVocabularyEmbeddingModule
} from './createPositionEmbeddingModule.js';
import { buildTokenChipStackModule } from './createTokenChipStackModule.js';
import { resolveEmbeddingStreamLayoutMetrics } from './embeddingStreamLayoutMetrics.js';
import {
    isPlaceholderTokenLabel,
    resolvePreferredTokenLabel
} from '../../utils/tokenLabelResolution.js';
import { getIncompleteUtf8TokenDisplay } from '../../utils/tokenEncodingNotes.js';
import {
    buildPromptTokenChipEntries,
    resolvePromptTokenChipColorState
} from '../../ui/tokenChipColorUtils.js';
import { formatTokenLabel } from '../../app/gpt-tower/tokenLabels.js';

const DEFAULT_VISIBLE_TOKEN_COUNT = 5;
const SUMMARY_MODEL_COLS = 18;
const SUMMARY_RESIDUAL_COLS = RESIDUAL_VECTOR_MEASURE_COLS;
const SUMMARY_HEAD_COLS = 12;
const SUMMARY_MLP_COLS = 24;
const SUMMARY_LOGIT_COLS = 12;
const CARD_LABEL_HORIZONTAL_INSET = 10;
const INPUT_EMBEDDING_ADD_TO_RESIDUAL_OFFSET_X = -96;

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

function normalizeHeadDetailTarget(target = null) {
    if (!target || typeof target !== 'object') return null;
    const layerIndex = normalizeIndex(target.layerIndex);
    const headIndex = normalizeIndex(target.headIndex);
    if (!Number.isFinite(layerIndex) || !Number.isFinite(headIndex)) {
        return null;
    }
    return {
        layerIndex,
        headIndex
    };
}

function normalizeConcatDetailTarget(target = null) {
    if (!target || typeof target !== 'object') return null;
    const layerIndex = normalizeIndex(target.layerIndex);
    if (!Number.isFinite(layerIndex)) {
        return null;
    }
    return {
        layerIndex
    };
}

function normalizeOutputProjectionDetailTarget(target = null) {
    if (!target || typeof target !== 'object') return null;
    const layerIndex = normalizeIndex(target.layerIndex);
    if (!Number.isFinite(layerIndex)) {
        return null;
    }
    return {
        layerIndex
    };
}

function normalizeMlpDetailTarget(target = null) {
    if (!target || typeof target !== 'object') return null;
    const layerIndex = normalizeIndex(target.layerIndex);
    if (!Number.isFinite(layerIndex)) {
        return null;
    }
    return {
        layerIndex
    };
}

function normalizeLayerNormDetailTarget(target = null, layerCount = null) {
    if (!target || typeof target !== 'object') return null;
    const layerNormKind = String(target.layerNormKind || '').trim().toLowerCase();
    if (layerNormKind !== 'ln1' && layerNormKind !== 'ln2' && layerNormKind !== 'final') {
        return null;
    }
    const normalizedLayerCount = Number.isFinite(layerCount) ? Math.max(1, Math.floor(layerCount)) : null;
    const layerIndex = normalizeIndex(target.layerIndex);
    if (layerNormKind === 'final') {
        return {
            layerNormKind,
            ...(Number.isFinite(layerIndex)
                ? { layerIndex }
                : (normalizedLayerCount
                    ? { layerIndex: normalizedLayerCount - 1 }
                    : {}))
        };
    }
    if (!Number.isFinite(layerIndex)) {
        return null;
    }
    return {
        layerNormKind,
        layerIndex
    };
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
    if (Array.isArray(tokenIndices) && tokenIndices.length) {
        return tokenIndices
            .map((value) => Number(value))
            .filter(Number.isFinite)
            .map((value) => Math.max(0, Math.floor(value)));
    }
    const safeMaxTokens = Math.max(1, Math.floor(maxTokens || DEFAULT_VISIBLE_TOKEN_COUNT));
    const tokenCount = typeof activationSource?.getTokenCount === 'function'
        ? activationSource.getTokenCount()
        : (Array.isArray(activationSource?.meta?.prompt_tokens) ? activationSource.meta.prompt_tokens.length : 0);
    const count = Number.isFinite(tokenCount) && tokenCount > 0
        ? Math.min(safeMaxTokens, Math.floor(tokenCount))
        : safeMaxTokens;
    return Array.from({ length: count }, (_, index) => index);
}

function resolveTokenLabel(activationSource = null, tokenIndex = 0, fallbackLabel = null, fallbackIndex = 0) {
    const rawFallbackLabel = typeof fallbackLabel === 'string' ? fallbackLabel : '';
    const trimmedFallbackLabel = rawFallbackLabel.trim();
    if (trimmedFallbackLabel.length && !isPlaceholderTokenLabel(trimmedFallbackLabel)) {
        return formatTokenLabel(rawFallbackLabel);
    }
    if (rawFallbackLabel.length && !trimmedFallbackLabel.length) {
        return formatTokenLabel(rawFallbackLabel);
    }

    const tokenId = resolveTokenId(activationSource, tokenIndex);
    const incompleteTokenDisplay = getIncompleteUtf8TokenDisplay(tokenId);
    if (incompleteTokenDisplay) {
        return incompleteTokenDisplay;
    }
    const resolvedLabel = resolvePreferredTokenLabel({
        tokenLabel: fallbackLabel,
        tokenIndex,
        activationSource
    });
    if (resolvedLabel.length) return resolvedLabel;
    return `Token ${fallbackIndex + 1}`;
}

function resolveTokenId(activationSource = null, tokenIndex = null) {
    if (!Number.isFinite(tokenIndex) || typeof activationSource?.getTokenId !== 'function') {
        return null;
    }
    const resolvedTokenId = activationSource.getTokenId(Math.floor(tokenIndex));
    return Number.isFinite(resolvedTokenId) ? Math.floor(resolvedTokenId) : null;
}

function resolveVisibleTokenRefs(activationSource = null, tokenIndices = null, tokenLabels = null) {
    return resolveVisibleTokenIndices(activationSource, tokenIndices).map((tokenIndex, rowIndex) => ({
        rowIndex,
        tokenIndex,
        tokenId: resolveTokenId(activationSource, tokenIndex),
        tokenLabel: resolveTokenLabel(
            activationSource,
            tokenIndex,
            Array.isArray(tokenLabels) ? tokenLabels[rowIndex] : null,
            rowIndex
        )
    }));
}

function resolveActivationSourceVectorLength(activationSource = null, fallbackLength = D_MODEL) {
    const sourceLength = Number(
        activationSource && typeof activationSource.getBaseVectorLength === 'function'
            ? activationSource.getBaseVectorLength()
            : null
    );
    if (Number.isFinite(sourceLength) && sourceLength > 0) {
        return Math.max(1, Math.floor(sourceLength));
    }
    return Math.max(1, Math.floor(fallbackLength || D_MODEL));
}

function resolveVisibleTokenRefByIndex(tokenRefs = [], tokenIndex = null) {
    const safeTokenIndex = normalizeIndex(tokenIndex);
    if (!Number.isFinite(safeTokenIndex)) return null;
    const safeTokenRefs = Array.isArray(tokenRefs) ? tokenRefs.filter(Boolean) : [];
    return safeTokenRefs.find((tokenRef) => (
        Number.isFinite(tokenRef?.tokenIndex)
        && Math.floor(tokenRef.tokenIndex) === safeTokenIndex
    )) || null;
}

function resolveVisiblePositionRefs(activationSource = null, tokenIndices = null) {
    return resolveVisibleTokenIndices(activationSource, tokenIndices).map((tokenIndex, rowIndex) => ({
        rowIndex,
        tokenIndex,
        positionIndex: tokenIndex + 1,
        tokenLabel: `${tokenIndex + 1}`
    }));
}

function resolvePromptTokenCount(activationSource = null) {
    const promptTokens = Array.isArray(activationSource?.meta?.prompt_tokens)
        ? activationSource.meta.prompt_tokens
        : null;
    if (promptTokens) {
        return promptTokens.length;
    }
    const completionTokens = Array.isArray(activationSource?.meta?.completion_tokens)
        ? activationSource.meta.completion_tokens
        : null;
    if (completionTokens) {
        const tokenCount = typeof activationSource?.getTokenCount === 'function'
            ? activationSource.getTokenCount()
            : null;
        if (Number.isFinite(tokenCount)) {
            return Math.max(0, Math.floor(tokenCount) - completionTokens.length);
        }
    }
    return null;
}

function resolveKnownTokenCount(activationSource = null) {
    const tokenCount = typeof activationSource?.getTokenCount === 'function'
        ? activationSource.getTokenCount()
        : null;
    if (Number.isFinite(tokenCount)) {
        return Math.max(0, Math.floor(tokenCount));
    }
    const promptTokenCount = Array.isArray(activationSource?.meta?.prompt_tokens)
        ? activationSource.meta.prompt_tokens.length
        : 0;
    const completionTokenCount = Array.isArray(activationSource?.meta?.completion_tokens)
        ? activationSource.meta.completion_tokens.length
        : 0;
    return (promptTokenCount + completionTokenCount) > 0
        ? promptTokenCount + completionTokenCount
        : null;
}

function resolveUnembeddingOutputTokenRefs(tokenRefs = [], activationSource = null) {
    const safeTokenRefs = Array.isArray(tokenRefs) ? tokenRefs.filter(Boolean) : [];
    if (!safeTokenRefs.length) return [];

    const promptTokenCount = resolvePromptTokenCount(activationSource);
    const knownTokenCount = resolveKnownTokenCount(activationSource);

    return safeTokenRefs.map((tokenRef, rowIndex) => {
        const currentTokenIndex = normalizeIndex(tokenRef?.tokenIndex);
        const nextTokenIndex = Number.isFinite(currentTokenIndex)
            ? currentTokenIndex + 1
            : null;
        const nextVisibleTokenRef = resolveVisibleTokenRefByIndex(safeTokenRefs, nextTokenIndex);
        const nextTokenKnown = Number.isFinite(nextTokenIndex) && (
            !!nextVisibleTokenRef
            || (
                Number.isFinite(knownTokenCount)
                && nextTokenIndex < knownTokenCount
            )
        );

        if (!nextTokenKnown) {
            return {
                rowIndex,
                displayMode: 'text',
                displayText: 'NA'
            };
        }

        if (Number.isFinite(promptTokenCount) && nextTokenIndex < promptTokenCount) {
            return {
                rowIndex,
                displayMode: 'text',
                displayText: 'NA'
            };
        }

        return {
            rowIndex,
            tokenIndex: nextTokenIndex,
            tokenId: resolveTokenId(activationSource, nextTokenIndex),
            positionIndex: nextTokenIndex + 1,
            tokenLabel: resolveTokenLabel(
                activationSource,
                nextTokenIndex,
                nextVisibleTokenRef?.tokenLabel || null,
                nextTokenIndex
            )
        };
    });
}

function tokenChipEntriesMatchByIdentity(left = null, right = null) {
    const leftTokenIndex = normalizeIndex(left?.tokenIndex);
    const rightTokenIndex = normalizeIndex(right?.tokenIndex);
    if (Number.isFinite(leftTokenIndex) && Number.isFinite(rightTokenIndex)) {
        return leftTokenIndex === rightTokenIndex;
    }

    const leftTokenId = normalizeIndex(left?.tokenId ?? left?.token_id);
    const rightTokenId = normalizeIndex(right?.tokenId ?? right?.token_id);
    if (Number.isFinite(leftTokenId) && Number.isFinite(rightTokenId)) {
        return leftTokenId === rightTokenId;
    }

    const leftTokenLabel = typeof (left?.tokenLabel ?? left?.token) === 'string'
        ? String(left.tokenLabel ?? left.token)
        : '';
    const rightTokenLabel = typeof (right?.tokenLabel ?? right?.token) === 'string'
        ? String(right.tokenLabel ?? right.token)
        : '';
    return !!leftTokenLabel && leftTokenLabel === rightTokenLabel;
}

function resolvePromptContextColorKey(colorState = null, targetEntry = null) {
    const entries = Array.isArray(colorState?.entries) ? colorState.entries : [];
    const colorKeys = Array.isArray(colorState?.colorKeys) ? colorState.colorKeys : [];
    const matchIndex = entries.findIndex((entry) => tokenChipEntriesMatchByIdentity(entry, targetEntry));
    return matchIndex >= 0 && Number.isFinite(colorKeys[matchIndex])
        ? colorKeys[matchIndex]
        : null;
}

function resolveChosenTokenChipRefs(tokenRefs = [], activationSource = null) {
    const chosenTokenRefs = resolveUnembeddingOutputTokenRefs(tokenRefs, activationSource);
    const visibleTokenRefs = Array.isArray(tokenRefs) ? tokenRefs.filter(Boolean) : [];
    if (!chosenTokenRefs.length || !visibleTokenRefs.length) {
        return chosenTokenRefs;
    }

    const promptEntries = buildPromptTokenChipEntries({
        tokenLabels: visibleTokenRefs.map((tokenRef) => tokenRef?.tokenLabel || ''),
        tokenIndices: visibleTokenRefs.map((tokenRef) => tokenRef?.tokenIndex ?? null),
        tokenIds: visibleTokenRefs.map((tokenRef) => tokenRef?.tokenId ?? null)
    });
    const promptColorState = resolvePromptTokenChipColorState(promptEntries);

    const lastVisibleTokenRef = visibleTokenRefs[visibleTokenRefs.length - 1] || null;
    const nextVisibleContinuationRef = (
        Number.isFinite(lastVisibleTokenRef?.tokenIndex)
            ? chosenTokenRefs.find((tokenRef) => (
                tokenRef?.displayMode !== 'text'
                && normalizeIndex(tokenRef?.tokenIndex) === Math.floor(lastVisibleTokenRef.tokenIndex) + 1
            )) || null
            : null
    );
    const promptWithGeneratedEntries = nextVisibleContinuationRef
        ? buildPromptTokenChipEntries({
            tokenLabels: visibleTokenRefs.map((tokenRef) => tokenRef?.tokenLabel || ''),
            tokenIndices: visibleTokenRefs.map((tokenRef) => tokenRef?.tokenIndex ?? null),
            tokenIds: visibleTokenRefs.map((tokenRef) => tokenRef?.tokenId ?? null),
            generatedToken: {
                tokenLabel: nextVisibleContinuationRef.tokenLabel,
                tokenIndex: nextVisibleContinuationRef.tokenIndex,
                tokenId: nextVisibleContinuationRef.tokenId
            }
        })
        : [];
    const promptWithGeneratedColorState = nextVisibleContinuationRef
        ? resolvePromptTokenChipColorState(promptWithGeneratedEntries)
        : null;

    return chosenTokenRefs.map((tokenRef) => {
        if (tokenRef?.displayMode === 'text') {
            return tokenRef;
        }

        const colorContextEntry = {
            tokenIndex: tokenRef?.tokenIndex,
            tokenId: tokenRef?.tokenId,
            tokenLabel: tokenRef?.tokenLabel
        };
        const colorKey = resolvePromptContextColorKey(promptColorState, colorContextEntry)
            ?? resolvePromptContextColorKey(promptWithGeneratedColorState, colorContextEntry);
        return Number.isFinite(colorKey)
            ? {
                ...tokenRef,
                colorKey
            }
            : tokenRef;
    });
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

function buildHeadDetailPreview({
    activationSource = null,
    tokenRefs = [],
    headDetailTarget = null
} = {}) {
    const resolvedTarget = normalizeHeadDetailTarget(headDetailTarget);
    if (!resolvedTarget || typeof activationSource?.getLayerLn1 !== 'function') {
        return null;
    }
    const baseSemantic = {
        componentKind: 'mhsa',
        layerIndex: resolvedTarget.layerIndex,
        headIndex: resolvedTarget.headIndex,
        stage: 'head-detail',
        role: 'x-ln'
    };
    const sourceVectorLength = resolveActivationSourceVectorLength(activationSource, D_MODEL);
    const rowItems = buildVectorRowItems(tokenRefs, {
        baseSemantic,
        role: 'x-ln-row',
        measureCols: SUMMARY_RESIDUAL_COLS,
        getVector: (tokenRef) => activationSource.getLayerLn1(
            resolvedTarget.layerIndex,
            'shift',
            tokenRef.tokenIndex,
            sourceVectorLength
        )
    });
    if (!rowItems.length) {
        return null;
    }
    return {
        xLnCopies: 3,
        rowItems: rowItems.map((rowItem) => ({
            id: rowItem.id,
            index: rowItem.index,
            label: rowItem.label,
            tokenIndex: rowItem.semantic?.tokenIndex,
            gradientCss: rowItem.gradientCss,
            rawValues: Array.isArray(rowItem.rawValues) ? [...rowItem.rawValues] : null
        }))
    };
}

function buildOutputProjectionDetailPreview({
    outputProjectionDetailTarget = null
} = {}) {
    const resolvedTarget = normalizeOutputProjectionDetailTarget(outputProjectionDetailTarget);
    if (!resolvedTarget) {
        return null;
    }
    return {
        arrowCount: NUM_HEAD_SETS_LAYER
    };
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

function createTextFitMetadata(maxWidth = null) {
    const textFit = {};
    if (Number.isFinite(maxWidth) && maxWidth > 0) {
        textFit.maxWidth = Math.floor(maxWidth);
    }
    return Object.keys(textFit).length ? { textFit } : null;
}

function createPersistentTextMetadata({
    maxWidth = null,
    persistentMinScreenFontPx = null
} = {}) {
    const metadata = {
        ...(createTextFitMetadata(maxWidth) || {})
    };
    if (Number.isFinite(persistentMinScreenFontPx) && persistentMinScreenFontPx > 0) {
        metadata.persistentMinScreenFontPx = Number(persistentMinScreenFontPx);
    }
    return Object.keys(metadata).length ? metadata : null;
}

function createCaptionMetadata({
    position = 'bottom',
    styleKey = null,
    dimensionsTex = '',
    dimensionsText = '',
    minScreenHeightPx = null,
    renderMode = ''
} = {}) {
    const caption = {};
    const safePosition = String(position || '').trim().toLowerCase();
    if (safePosition === 'top' || safePosition === 'inside-top' || safePosition === 'float-top') {
        caption.position = safePosition;
    }
    if (typeof styleKey === 'string' && styleKey.length) {
        caption.styleKey = styleKey;
    }
    if (typeof dimensionsTex === 'string' && dimensionsTex.trim().length) {
        caption.dimensionsTex = dimensionsTex.trim();
    }
    if (typeof dimensionsText === 'string' && dimensionsText.trim().length) {
        caption.dimensionsText = dimensionsText.trim();
    }
    if (Number.isFinite(minScreenHeightPx) && minScreenHeightPx > 0) {
        caption.minScreenHeightPx = Math.max(1, Math.floor(minScreenHeightPx));
    }
    const safeRenderMode = String(renderMode || '').trim().toLowerCase();
    if (safeRenderMode.length) {
        caption.renderMode = safeRenderMode;
    }
    return Object.keys(caption).length ? { caption } : null;
}

function createFittedLabelNode({
    role = 'module-title',
    semantic = {},
    text = '',
    styleKey = VIEW2D_STYLE_KEYS.LABEL,
    maxWidth = null,
    persistentMinScreenFontPx = null
} = {}) {
    return createTextNode({
        role,
        semantic,
        text,
        presentation: VIEW2D_TEXT_PRESENTATIONS.LABEL,
        visual: { styleKey },
        metadata: createPersistentTextMetadata({
            maxWidth,
            persistentMinScreenFontPx
        })
    });
}

function createModuleTitleNode(baseSemantic = {}, text = '') {
    return createFittedLabelNode({
        role: 'module-title',
        semantic: buildSemantic(baseSemantic, { role: 'module-title' }),
        text,
        styleKey: VIEW2D_STYLE_KEYS.LABEL
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

function createModuleCardGroup({
    semantic = {},
    title = '',
    styleKey = VIEW2D_STYLE_KEYS.RESIDUAL,
    cardRole = 'module-card',
    cardWidth = 112,
    cardHeight = 96,
    dimensions = { rows: 1, cols: 1 },
    cardCornerRadius = null,
    textStyleKey = VIEW2D_STYLE_KEYS.LABEL,
    textFitWidth = null,
    textPersistentMinScreenFontPx = null
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
                createFittedLabelNode({
                    role: 'module-title',
                    semantic: buildSemantic(semantic, { role: 'module-title' }),
                    text: title,
                    styleKey: textStyleKey,
                    maxWidth: Number.isFinite(textFitWidth) && textFitWidth > 0
                        ? textFitWidth
                        : Math.max(24, cardWidth - (CARD_LABEL_HORIZONTAL_INSET * 2)),
                    persistentMinScreenFontPx: textPersistentMinScreenFontPx
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
    labelTex = 'x',
    labelText = 'x',
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
    const cardNode = createResidualVectorMatrixNode({
        role: 'module-card',
        semantic: buildSemantic(semantic, { role: 'module-card' }),
        labelTex,
        labelText,
        rowItems,
        rowCount,
        captionPosition: 'float-top',
        captionMinScreenHeightPx: 28
    });

    return {
        node: createGroupNode({
            role: semantic.role || 'module',
            semantic,
            direction: VIEW2D_LAYOUT_DIRECTIONS.OVERLAY,
            gapKey: 'default',
            children: [cardNode]
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
            text: `Attention Head ${headIndex + 1}`,
            presentation: VIEW2D_TEXT_PRESENTATIONS.LABEL,
            visual: { styleKey: VIEW2D_STYLE_KEYS.LABEL },
            metadata: createPersistentTextMetadata({
                maxWidth: Math.max(24, HEAD_CARD_WIDTH - (CARD_LABEL_HORIZONTAL_INSET * 2))
            })
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
    const semantic = {
        componentKind: 'output-projection',
        layerIndex,
        stage: 'attn-out',
        role: 'module'
    };
    const cardWidth = 110;
    const cardHeight = 86;
    const titleMaxWidth = Math.max(24, cardWidth - (CARD_LABEL_HORIZONTAL_INSET * 2));
    const cardNode = createMatrixNode({
        role: 'projection-weight',
        semantic: buildSemantic(semantic, { role: 'projection-weight' }),
        dimensions: {
            rows: D_MODEL,
            cols: D_MODEL
        },
        presentation: VIEW2D_MATRIX_PRESENTATIONS.CARD,
        shape: VIEW2D_MATRIX_SHAPES.MATRIX,
        visual: {
            styleKey: VIEW2D_STYLE_KEYS.OUTPUT_PROJECTION
        },
        metadata: mergeMetadata(
            createCardMetadata(cardWidth, cardHeight, {
                cornerRadius: 14
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
                createGroupNode({
                    role: 'module-title-stack',
                    semantic: buildSemantic(semantic, { role: 'module-title-stack' }),
                    direction: VIEW2D_LAYOUT_DIRECTIONS.VERTICAL,
                    gapKey: 'default',
                    children: [
                        createFittedLabelNode({
                            role: 'module-title-top',
                            semantic: buildSemantic(semantic, { role: 'module-title-top' }),
                            text: 'Output',
                            styleKey: VIEW2D_STYLE_KEYS.LABEL,
                            maxWidth: titleMaxWidth
                        }),
                        createFittedLabelNode({
                            role: 'module-title-bottom',
                            semantic: buildSemantic(semantic, { role: 'module-title-bottom' }),
                            text: 'Projection',
                            styleKey: VIEW2D_STYLE_KEYS.LABEL,
                            maxWidth: titleMaxWidth
                        })
                    ],
                    metadata: {
                        gapOverride: 0
                    }
                })
            ]
        }),
        cardNode
    };
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
    const semantic = {
        componentKind: 'mlp',
        layerIndex,
        stage: 'mlp',
        role: 'module'
    };
    const cardWidth = 148;
    const cardHeight = 92;
    const titleMaxWidth = Math.max(24, cardWidth - (CARD_LABEL_HORIZONTAL_INSET * 2));
    const cardNode = createMatrixNode({
        role: 'module-card',
        semantic: buildSemantic(semantic, { role: 'module-card' }),
        dimensions: {
            rows: D_MODEL,
            cols: D_MODEL * 4
        },
        presentation: VIEW2D_MATRIX_PRESENTATIONS.CARD,
        shape: VIEW2D_MATRIX_SHAPES.MATRIX,
        visual: {
            styleKey: VIEW2D_STYLE_KEYS.MLP
        },
        metadata: mergeMetadata(
            createCardMetadata(cardWidth, cardHeight, {
                cornerRadius: 14
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
                createGroupNode({
                    role: 'module-title-stack',
                    semantic: buildSemantic(semantic, { role: 'module-title-stack' }),
                    direction: VIEW2D_LAYOUT_DIRECTIONS.VERTICAL,
                    gapKey: 'default',
                    children: [
                        createFittedLabelNode({
                            role: 'module-title-top',
                            semantic: buildSemantic(semantic, { role: 'module-title-top' }),
                            text: 'Multilayer',
                            styleKey: VIEW2D_STYLE_KEYS.LABEL,
                            maxWidth: titleMaxWidth
                        }),
                        createFittedLabelNode({
                            role: 'module-title-bottom',
                            semantic: buildSemantic(semantic, { role: 'module-title-bottom' }),
                            text: 'Perceptron',
                            styleKey: VIEW2D_STYLE_KEYS.LABEL,
                            maxWidth: titleMaxWidth
                        })
                    ],
                    metadata: {
                        gapOverride: 0
                    }
                })
            ]
        }),
        cardNode
    };
}

function buildLayerGroup({
    layerIndex = 0,
    activationSource = null,
    tokenRefs = [],
    tokenIndices = null,
    tokenLabels = null,
    includeInputPositionEmbedding = false,
    isSmallScreen = false,
    visualTokens = null
} = {}) {
    const layerSemantic = {
        componentKind: 'layer',
        layerIndex
    };
    const sourceVectorLength = resolveActivationSourceVectorLength(activationSource, D_MODEL);
    const embeddingLayoutMetrics = includeInputPositionEmbedding
        ? resolveEmbeddingStreamLayoutMetrics({
            tokenCount: tokenRefs.length,
            isSmallScreen
        })
        : null;

    const COLUMN_WIDTHS = {
        residual: 136,
        heads: 200,
        outProj: 152,
        addIn: 80,
        ln2: 148,
        mlp: 196,
        addOut: 80,
        detail: 0
    };
    const TOP_ROW_HEIGHT = 58;

    const incomingResidual = createResidualStateModule({
        semantic: {
            componentKind: 'residual',
            layerIndex,
            stage: 'incoming',
            role: 'module'
        },
        labelTex: '\\mathrm{X}',
        labelText: 'X',
        tokenRefs,
        getVector: (tokenRef) => (
            typeof activationSource?.getLayerIncoming === 'function'
                ? activationSource.getLayerIncoming(layerIndex, tokenRef.tokenIndex, sourceVectorLength)
                : null
        )
    });
    const inputVocabularyEmbedding = includeInputPositionEmbedding
        ? buildVocabularyEmbeddingModule({
            semantic: {
                componentKind: 'embedding',
                stage: 'embedding.token',
                role: 'module'
            },
            cardWidth: embeddingLayoutMetrics?.vocabulary?.cardWidth,
            cardHeight: embeddingLayoutMetrics?.vocabulary?.cardHeight
        })
        : null;
    const inputTokenChipStack = includeInputPositionEmbedding
        ? buildTokenChipStackModule({
            semantic: {
                componentKind: 'embedding',
                stage: 'embedding.token'
            },
            tokenRefs,
            chipWidth: embeddingLayoutMetrics?.vocabulary?.chipWidth,
            chipHeight: embeddingLayoutMetrics?.vocabulary?.chipHeight,
            minChipHeight: embeddingLayoutMetrics?.vocabulary?.minChipHeight,
            maxStackHeight: embeddingLayoutMetrics?.vocabulary?.maxStackHeight,
            gap: embeddingLayoutMetrics?.vocabulary?.stackGap,
            labelFontScale: embeddingLayoutMetrics?.vocabulary?.labelFontScale
        })
        : null;
    const inputPositionChipStack = includeInputPositionEmbedding
        ? buildTokenChipStackModule({
            semantic: {
                componentKind: 'embedding',
                stage: 'embedding.position'
            },
            tokenRefs: resolveVisiblePositionRefs(activationSource, tokenIndices),
            stackRole: 'input-position-chip-stack',
            chipRole: 'input-position-chip',
            chipLabelRole: 'input-position-chip-label',
            chipGroupRole: 'input-position-chip-group',
            chipWidth: embeddingLayoutMetrics?.position?.chipWidth,
            chipHeight: embeddingLayoutMetrics?.position?.chipHeight,
            minChipHeight: embeddingLayoutMetrics?.position?.minChipHeight,
            maxStackHeight: embeddingLayoutMetrics?.position?.maxStackHeight,
            gap: embeddingLayoutMetrics?.position?.stackGap,
            colorMode: 'neutral',
            labelFontScale: embeddingLayoutMetrics?.position?.labelFontScale
        })
        : null;
    const inputPositionEmbedding = includeInputPositionEmbedding
        ? buildPositionEmbeddingModule({
            semantic: {
                componentKind: 'embedding',
                stage: 'embedding.position',
                role: 'module'
            },
            cardWidth: embeddingLayoutMetrics?.position?.cardWidth,
            cardHeight: embeddingLayoutMetrics?.position?.cardHeight
        })
        : null;
    const ln1Module = buildLayerNormModule({
        layerIndex,
        stage: 'ln1',
        title: 'LayerNorm 1'
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
        labelTex: '\\mathrm{X}',
        labelText: 'X',
        tokenRefs,
        getVector: (tokenRef) => (
            typeof activationSource?.getPostAttentionResidual === 'function'
                ? activationSource.getPostAttentionResidual(layerIndex, tokenRef.tokenIndex, sourceVectorLength)
                : null
        )
    });
    const ln2Module = buildLayerNormModule({
        layerIndex,
        stage: 'ln2',
        title: 'LayerNorm 2'
    });
    const mlpModule = buildMlpModule({
        layerIndex
    });
    const postMlpAdd = buildResidualAddModule({
        layerIndex,
        stage: 'post-mlp-add'
    });
    const embeddingInputAdd = includeInputPositionEmbedding
        ? createTopAddNode({
            semantic: {
                componentKind: 'embedding',
                stage: 'embedding.sum',
                role: 'module'
            }
        })
        : null;
    const embeddingAddOverlay = embeddingInputAdd
        ? createGroupNode({
            role: 'input-embedding-add-overlay',
            semantic: buildSemantic(layerSemantic, {
                stage: 'input-embedding-add-overlay',
                role: 'input-embedding-add-overlay'
            }),
            direction: VIEW2D_LAYOUT_DIRECTIONS.OVERLAY,
            gapKey: 'default',
            layout: {
                anchorAlign: {
                    axis: 'x',
                    selfNodeId: embeddingInputAdd.cardNode.id,
                    targetNodeId: incomingResidual.cardNode.id,
                    selfAnchor: VIEW2D_ANCHOR_SIDES.RIGHT,
                    targetAnchor: VIEW2D_ANCHOR_SIDES.LEFT,
                    offset: INPUT_EMBEDDING_ADD_TO_RESIDUAL_OFFSET_X
                }
            },
            children: [
                createGroupNode({
                    role: 'input-embedding-add-y-anchor',
                    semantic: buildSemantic(layerSemantic, {
                        stage: 'input-embedding-add-y-anchor',
                        role: 'input-embedding-add-y-anchor'
                    }),
                    direction: VIEW2D_LAYOUT_DIRECTIONS.OVERLAY,
                    gapKey: 'default',
                    layout: {
                        anchorAlign: {
                            axis: 'y',
                            selfNodeId: embeddingInputAdd.cardNode.id,
                            targetNodeId: incomingResidual.cardNode.id,
                            selfAnchor: VIEW2D_ANCHOR_SIDES.CENTER,
                            targetAnchor: VIEW2D_ANCHOR_SIDES.CENTER
                        }
                    },
                    children: [embeddingInputAdd.node]
                })
            ]
        })
        : null;
    const vocabularyEmbeddingOverlay = inputVocabularyEmbedding && embeddingInputAdd
        ? createGroupNode({
            role: 'input-vocabulary-embedding-overlay',
            semantic: buildSemantic(layerSemantic, {
                stage: 'input-vocabulary-embedding-overlay',
                role: 'input-vocabulary-embedding-overlay'
            }),
            direction: VIEW2D_LAYOUT_DIRECTIONS.OVERLAY,
            gapKey: 'default',
            layout: {
                anchorAlign: {
                    axis: 'x',
                    selfNodeId: inputVocabularyEmbedding.cardNode.id,
                    targetNodeId: embeddingInputAdd.cardNode.id,
                    selfAnchor: VIEW2D_ANCHOR_SIDES.RIGHT,
                    targetAnchor: VIEW2D_ANCHOR_SIDES.LEFT,
                    offset: -300
                }
            },
            children: [
                createGroupNode({
                    role: 'input-vocabulary-embedding-y-anchor',
                    semantic: buildSemantic(layerSemantic, {
                        stage: 'input-vocabulary-embedding-y-anchor',
                        role: 'input-vocabulary-embedding-y-anchor'
                    }),
                    direction: VIEW2D_LAYOUT_DIRECTIONS.OVERLAY,
                    gapKey: 'default',
                    layout: {
                        anchorAlign: {
                            axis: 'y',
                            selfNodeId: inputVocabularyEmbedding.cardNode.id,
                            targetNodeId: embeddingInputAdd.cardNode.id,
                            selfAnchor: VIEW2D_ANCHOR_SIDES.CENTER,
                            targetAnchor: VIEW2D_ANCHOR_SIDES.CENTER
                        }
                    },
                    children: [inputVocabularyEmbedding.node]
                })
            ]
        })
        : null;
    const inputTokenChipYAnchor = inputTokenChipStack && inputVocabularyEmbedding
        ? createGroupNode({
            role: 'input-token-chip-y-anchor',
            semantic: buildSemantic(layerSemantic, {
                stage: 'input-token-chip-y-anchor',
                role: 'input-token-chip-y-anchor'
            }),
            direction: VIEW2D_LAYOUT_DIRECTIONS.OVERLAY,
            gapKey: 'default',
            layout: {
                anchorAlign: {
                    axis: 'y',
                    selfNodeId: inputTokenChipStack.node.id,
                    targetNodeId: inputVocabularyEmbedding.cardNode.id,
                    selfAnchor: VIEW2D_ANCHOR_SIDES.CENTER,
                    targetAnchor: VIEW2D_ANCHOR_SIDES.CENTER
                }
            },
            children: [inputTokenChipStack.node]
        })
        : null;
    const inputTokenChipOverlay = inputTokenChipYAnchor && inputVocabularyEmbedding
        ? createGroupNode({
            role: 'input-token-chip-overlay',
            semantic: buildSemantic(layerSemantic, {
                stage: 'input-token-chip-overlay',
                role: 'input-token-chip-overlay'
            }),
            direction: VIEW2D_LAYOUT_DIRECTIONS.OVERLAY,
            gapKey: 'default',
            layout: {
                anchorAlign: {
                    axis: 'x',
                    selfNodeId: inputTokenChipYAnchor.id,
                    targetNodeId: inputVocabularyEmbedding.cardNode.id,
                    selfAnchor: VIEW2D_ANCHOR_SIDES.RIGHT,
                    targetAnchor: VIEW2D_ANCHOR_SIDES.LEFT,
                    offset: -(embeddingLayoutMetrics?.vocabulary?.chipToCardGap || 18)
                }
            },
            children: [inputTokenChipYAnchor]
        })
        : null;
    const positionEmbeddingOverlay = inputPositionEmbedding && inputVocabularyEmbedding
        ? createGroupNode({
            role: 'input-position-embedding-overlay',
            semantic: buildSemantic(layerSemantic, {
                stage: 'input-position-embedding-overlay',
                role: 'input-position-embedding-overlay'
            }),
            direction: VIEW2D_LAYOUT_DIRECTIONS.OVERLAY,
            gapKey: 'default',
            layout: {
                anchorAlign: {
                    axis: 'x',
                    selfNodeId: inputPositionEmbedding.cardNode.id,
                    targetNodeId: inputVocabularyEmbedding.cardNode.id,
                    selfAnchor: VIEW2D_ANCHOR_SIDES.CENTER,
                    targetAnchor: VIEW2D_ANCHOR_SIDES.CENTER
                }
            },
            children: [
                createGroupNode({
                    role: 'input-position-embedding-y-anchor',
                    semantic: buildSemantic(layerSemantic, {
                        stage: 'input-position-embedding-y-anchor',
                        role: 'input-position-embedding-y-anchor'
                    }),
                    direction: VIEW2D_LAYOUT_DIRECTIONS.OVERLAY,
                    gapKey: 'default',
                    layout: {
                        anchorAlign: {
                            axis: 'y',
                            selfNodeId: inputPositionEmbedding.cardNode.id,
                            targetNodeId: inputVocabularyEmbedding.cardNode.id,
                            selfAnchor: VIEW2D_ANCHOR_SIDES.CENTER,
                            targetAnchor: VIEW2D_ANCHOR_SIDES.CENTER,
                            offset: Math.round(
                                (
                                    (Number(embeddingLayoutMetrics?.vocabulary?.cardHeight) || 144)
                                    + (Number(embeddingLayoutMetrics?.position?.cardHeight) || 108)
                                ) / 2
                            ) + (Number(embeddingLayoutMetrics?.vocabularyToPositionGap) || 56)
                        }
                    },
                    children: [inputPositionEmbedding.node]
                })
            ]
        })
        : null;
    const inputPositionChipYAnchor = inputPositionChipStack && inputPositionEmbedding
        ? createGroupNode({
            role: 'input-position-chip-y-anchor',
            semantic: buildSemantic(layerSemantic, {
                stage: 'input-position-chip-y-anchor',
                role: 'input-position-chip-y-anchor'
            }),
            direction: VIEW2D_LAYOUT_DIRECTIONS.OVERLAY,
            gapKey: 'default',
            layout: {
                anchorAlign: {
                    axis: 'y',
                    selfNodeId: inputPositionChipStack.node.id,
                    targetNodeId: inputPositionEmbedding.cardNode.id,
                    selfAnchor: VIEW2D_ANCHOR_SIDES.CENTER,
                    targetAnchor: VIEW2D_ANCHOR_SIDES.CENTER
                }
            },
            children: [inputPositionChipStack.node]
        })
        : null;
    const inputPositionChipOverlay = inputPositionChipYAnchor && inputPositionEmbedding
        ? createGroupNode({
            role: 'input-position-chip-overlay',
            semantic: buildSemantic(layerSemantic, {
                stage: 'input-position-chip-overlay',
                role: 'input-position-chip-overlay'
            }),
            direction: VIEW2D_LAYOUT_DIRECTIONS.OVERLAY,
            gapKey: 'default',
            layout: {
                anchorAlign: {
                    axis: 'x',
                    selfNodeId: inputPositionChipYAnchor.id,
                    targetNodeId: inputPositionEmbedding.cardNode.id,
                    selfAnchor: VIEW2D_ANCHOR_SIDES.RIGHT,
                    targetAnchor: VIEW2D_ANCHOR_SIDES.LEFT,
                    offset: -(embeddingLayoutMetrics?.position?.chipToCardGap || 18)
                }
            },
            children: [inputPositionChipYAnchor]
        })
        : null;

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
        ...(inputTokenChipStack && inputVocabularyEmbedding
            ? [
                {
                    from: inputTokenChipStack.node,
                    to: inputVocabularyEmbedding.cardNode,
                    key: `layer-${layerIndex}-input-token-chip-to-vocabulary-embedding`,
                    gap: 8
                }
            ]
            : []),
        ...(inputVocabularyEmbedding && embeddingInputAdd
            ? [
                {
                    from: inputVocabularyEmbedding.cardNode,
                    to: embeddingInputAdd.cardNode,
                    key: `layer-${layerIndex}-vocabulary-embedding-to-add`,
                    gap: 10
                },
                {
                    from: embeddingInputAdd.cardNode,
                    to: incomingResidual.cardNode,
                    key: `layer-${layerIndex}-embedding-add-to-residual`,
                    gap: 8
                }
            ]
            : []),
        ...(inputPositionEmbedding && embeddingInputAdd
            ? [
                {
                    from: inputPositionEmbedding.cardNode,
                    to: embeddingInputAdd.cardNode,
                    key: `layer-${layerIndex}-position-embedding-to-add`,
                    sourceAnchor: VIEW2D_ANCHOR_SIDES.RIGHT,
                    targetAnchor: VIEW2D_ANCHOR_SIDES.BOTTOM,
                    route: VIEW2D_CONNECTOR_ROUTES.ELBOW,
                    gap: 10
                }
            ]
            : []),
        ...(inputPositionChipStack && inputPositionEmbedding
            ? [
                {
                    from: inputPositionChipStack.node,
                    to: inputPositionEmbedding.cardNode,
                    key: `layer-${layerIndex}-input-position-chip-to-position-embedding`,
                    gap: 8
                }
            ]
            : []),
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
        overlayNodes: [
            embeddingAddOverlay,
            vocabularyEmbeddingOverlay,
            inputTokenChipOverlay,
            positionEmbeddingOverlay,
            inputPositionChipOverlay
        ].filter(Boolean),
        flow
    };
}

function buildFinalLayerNormModule() {
    return buildLayerNormModule({
        stage: 'final-ln',
        title: 'Final LayerNorm'
    });
}

function buildFinalOutputGroup({
    layerIndex = null,
    activationSource = null,
    tokenRefs = [],
    anchorTargetNode = null,
    isSmallScreen = false
} = {}) {
    const embeddingLayoutMetrics = resolveEmbeddingStreamLayoutMetrics({
        tokenCount: tokenRefs.length,
        isSmallScreen
    });
    const unembeddingMetrics = embeddingLayoutMetrics.unembedding;
    const UNEMBEDDING_CARD_WIDTH = unembeddingMetrics.cardWidth;
    const UNEMBEDDING_OUTPUT_GAP = unembeddingMetrics.outputGap;
    const COLUMN_WIDTHS = {
        residual: 136,
        layerNorm: 148,
        unembedding: 232
    };
    const TOP_ROW_HEIGHT = 58;
    const baseSemantic = {
        componentKind: 'transformer',
        stage: 'final-output'
    };
    const sourceVectorLength = resolveActivationSourceVectorLength(activationSource, D_MODEL);

    const outgoingResidual = createResidualStateModule({
        semantic: {
            componentKind: 'residual',
            layerIndex,
            stage: 'outgoing',
            role: 'module'
        },
        labelTex: '\\mathrm{X}',
        labelText: 'X',
        tokenRefs,
        getVector: (tokenRef) => (
            typeof activationSource?.getPostMlpResidual === 'function'
                ? activationSource.getPostMlpResidual(layerIndex, tokenRef.tokenIndex, sourceVectorLength)
                : null
        )
    });
    const finalLayerNorm = buildFinalLayerNormModule();
    const unembeddingModule = buildUnembeddingModule({
        semantic: {
            componentKind: 'logits',
            stage: 'unembedding',
            role: 'module'
        },
        cardWidth: unembeddingMetrics.cardWidth,
        cardHeight: unembeddingMetrics.cardHeight
    });
    const chosenTokenChipStack = buildTokenChipStackModule({
        semantic: {
            componentKind: 'logits',
            stage: 'output'
        },
        tokenRefs: resolveChosenTokenChipRefs(tokenRefs, activationSource),
        stackRole: 'chosen-token-chip-stack',
        chipRole: 'chosen-token-chip',
        chipLabelRole: 'chosen-token-chip-label',
        chipGroupRole: 'chosen-token-chip-group',
        chipWidth: unembeddingMetrics.chipWidth,
        chipHeight: unembeddingMetrics.chipHeight,
        minChipHeight: unembeddingMetrics.minChipHeight,
        maxStackHeight: unembeddingMetrics.maxStackHeight,
        gap: unembeddingMetrics.stackGap,
        labelFontScale: unembeddingMetrics.labelFontScale
    });
    const unembeddingBody = chosenTokenChipStack
        ? createGroupNode({
            role: 'chosen-token-output',
            semantic: {
                componentKind: 'logits',
                stage: 'output',
                role: 'chosen-token-output'
            },
            direction: VIEW2D_LAYOUT_DIRECTIONS.HORIZONTAL,
            gapKey: 'inline',
            align: 'center',
            children: [
                unembeddingModule.node,
                chosenTokenChipStack.node
            ],
            metadata: {
                gapOverride: UNEMBEDDING_OUTPUT_GAP
            }
        })
        : unembeddingModule.node;
    COLUMN_WIDTHS.unembedding = chosenTokenChipStack
        ? (unembeddingMetrics.cardWidth + chosenTokenChipStack.width + UNEMBEDDING_OUTPUT_GAP + 10)
        : COLUMN_WIDTHS.unembedding;

    const node = createGroupNode({
        role: 'final-output',
        semantic: buildSemantic(baseSemantic, { role: 'final-output' }),
        direction: VIEW2D_LAYOUT_DIRECTIONS.HORIZONTAL,
        gapKey: 'stage',
        align: 'center',
        layout: anchorTargetNode?.id
            ? {
                anchorAlign: {
                    axis: 'y',
                    selfNodeId: outgoingResidual.cardNode.id,
                    targetNodeId: anchorTargetNode.id,
                    selfAnchor: VIEW2D_ANCHOR_SIDES.CENTER,
                    targetAnchor: VIEW2D_ANCHOR_SIDES.CENTER
                }
            }
            : null,
        children: [
            createFixedWidthColumn({
                semantic: buildSemantic(baseSemantic, { stage: 'outgoing-residual', role: 'outgoing-residual' }),
                width: COLUMN_WIDTHS.residual,
                height: TOP_ROW_HEIGHT,
                child: outgoingResidual.node
            }),
            createFixedWidthColumn({
                semantic: buildSemantic(baseSemantic, { stage: 'final-layer-norm', role: 'final-layer-norm' }),
                width: COLUMN_WIDTHS.layerNorm,
                height: TOP_ROW_HEIGHT,
                child: finalLayerNorm.node
            }),
            createFixedWidthColumn({
                semantic: buildSemantic(baseSemantic, { stage: 'unembedding', role: 'unembedding' }),
                width: COLUMN_WIDTHS.unembedding,
                height: 156,
                child: unembeddingBody
            })
        ]
    });

    const flow = [];
    if (anchorTargetNode?.id) {
        flow.push({
            from: anchorTargetNode,
            to: outgoingResidual.cardNode,
            key: 'final-output-outgoing-residual'
        });
    }
    flow.push({
        from: outgoingResidual.cardNode,
        to: finalLayerNorm.cardNode,
        key: 'final-output-layer-norm'
    });
    flow.push({
        from: finalLayerNorm.cardNode,
        to: unembeddingModule.cardNode,
        key: 'final-output-unembedding'
    });
    if (chosenTokenChipStack) {
        flow.push({
            from: unembeddingModule.cardNode,
            to: chosenTokenChipStack.node,
            key: 'final-output-chosen-token'
        });
    }

    return {
        node,
        entryNode: outgoingResidual.cardNode,
        exitNode: unembeddingModule.cardNode,
        flow
    };
}

function buildLogitsModule({
    tokenRefs = [],
    activationSource = null,
    includeUnembedding = true,
    title = 'Logits'
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

    const unembeddingNode = includeUnembedding
        ? createMatrixNode({
            role: 'unembedding',
            semantic: buildSemantic(baseSemantic, { stage: 'unembedding', role: 'unembedding' }),
            label: buildLabel('Vocabulary Unembedding Matrix', 'Vocabulary Unembedding Matrix'),
            dimensions: {
                rows: D_MODEL,
                cols: VOCAB_SIZE
            },
            presentation: VIEW2D_MATRIX_PRESENTATIONS.CARD,
            shape: VIEW2D_MATRIX_SHAPES.MATRIX,
            visual: {
                styleKey: VIEW2D_STYLE_KEYS.MATRIX_WEIGHT
            }
        })
        : null;

    const node = createModuleGroup(baseSemantic, title, [
        ...(unembeddingNode ? [unembeddingNode] : []),
        logitsNode
    ]);

    return {
        node,
        entryNode: unembeddingNode || logitsNode,
        exitNode: logitsNode,
        logitsNode,
        unembeddingNode
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
    visualTokens = null,
    headDetailTarget = null,
    concatDetailTarget = null,
    outputProjectionDetailTarget = null,
    mlpDetailTarget = null,
    layerNormDetailTarget = null,
    kvCacheState = null
} = {}) {
    const resolvedLayerCount = Number.isFinite(layerCount) ? Math.max(1, Math.floor(layerCount)) : NUM_LAYERS;
    const tokenRefs = resolveVisibleTokenRefs(activationSource, tokenIndices, tokenLabels);
    const resolvedTokens = visualTokens || resolveView2dVisualTokens();
    const redirectedConcatDetailTarget = normalizeConcatDetailTarget(concatDetailTarget);
    const requestedOutputProjectionDetailTarget = normalizeOutputProjectionDetailTarget(outputProjectionDetailTarget)
        || (
            redirectedConcatDetailTarget
                ? { layerIndex: redirectedConcatDetailTarget.layerIndex }
                : null
        );
    const requestedConcatDetailTarget = null;
    const requestedMlpDetailTarget = requestedOutputProjectionDetailTarget
        ? null
        : normalizeMlpDetailTarget(mlpDetailTarget);
    const requestedLayerNormDetailTarget = (
        requestedOutputProjectionDetailTarget
        || requestedMlpDetailTarget
    )
        ? null
        : normalizeLayerNormDetailTarget(layerNormDetailTarget, resolvedLayerCount);
    const requestedHeadDetailTarget = (
        requestedOutputProjectionDetailTarget
        || requestedMlpDetailTarget
        || requestedLayerNormDetailTarget
    )
        ? null
        : normalizeHeadDetailTarget(headDetailTarget);
    const resolvedHeadDetailTarget = requestedHeadDetailTarget;
    const resolvedConcatDetailTarget = requestedConcatDetailTarget;
    const resolvedOutputProjectionDetailTarget = requestedOutputProjectionDetailTarget;
    const resolvedMlpDetailTarget = requestedMlpDetailTarget;
    const resolvedLayerNormDetailTarget = requestedLayerNormDetailTarget;
    const headDetailPreview = buildHeadDetailPreview({
        activationSource,
        tokenRefs,
        headDetailTarget: resolvedHeadDetailTarget
    });
    const mhsaHeadDetailScene = resolvedHeadDetailTarget
        ? buildMhsaSceneModel({
            activationSource,
            layerIndex: resolvedHeadDetailTarget.layerIndex,
            headIndex: resolvedHeadDetailTarget.headIndex,
            tokenIndices: tokenRefs.map((tokenRef) => tokenRef.tokenIndex),
            tokenLabels: tokenRefs.map((tokenRef) => tokenRef.tokenLabel),
            isSmallScreen,
            visualTokens: resolvedTokens,
            kvCacheState
        })
        : null;
    const headDetailScene = buildHeadDetailSceneModel({
        headDetailPreview,
        headDetailTarget: resolvedHeadDetailTarget,
        visualTokens: resolvedTokens,
        isSmallScreen
    });
    const concatDetailPreview = null;
    const outputProjectionDetailPreview = buildOutputProjectionDetailPreview({
        outputProjectionDetailTarget: resolvedOutputProjectionDetailTarget
    });
    const outputProjectionDetailScene = resolvedOutputProjectionDetailTarget
        ? buildOutputProjectionDetailSceneModel({
            activationSource,
            outputProjectionDetailTarget: resolvedOutputProjectionDetailTarget,
            tokenRefs,
            visualTokens: resolvedTokens,
            isSmallScreen
        })
        : null;
    const mlpDetailScene = resolvedMlpDetailTarget
        ? buildMlpDetailSceneModel({
            activationSource,
            mlpDetailTarget: resolvedMlpDetailTarget,
            tokenRefs,
            visualTokens: resolvedTokens,
            isSmallScreen
        })
        : null;
    const layerNormDetailScene = resolvedLayerNormDetailTarget
        ? buildLayerNormDetailSceneModel({
            activationSource,
            layerNormDetailTarget: resolvedLayerNormDetailTarget,
            tokenRefs,
            layerCount: resolvedLayerCount,
            visualTokens: resolvedTokens,
            isSmallScreen
        })
        : null;

    const layerModules = Array.from({ length: resolvedLayerCount }, (_, layerIndex) => buildLayerGroup({
        layerIndex,
        activationSource,
        tokenRefs,
        tokenIndices,
        tokenLabels,
        includeInputPositionEmbedding: layerIndex === 0,
        isSmallScreen,
        visualTokens: resolvedTokens
    }));
    const finalOutputModule = layerModules.length
        ? buildFinalOutputGroup({
            layerIndex: layerModules.length - 1,
            activationSource,
            tokenRefs,
            anchorTargetNode: layerModules[layerModules.length - 1]?.exitNode || null,
            isSmallScreen
        })
        : null;

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
    if (finalOutputModule) {
        finalOutputModule.flow.forEach(({ from, to, key, sourceAnchor, targetAnchor, route, styleKey, gap }) => {
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
    }

    const rootNodes = [
        ...layerModules.map((module) => module.node),
        ...layerModules.flatMap((module) => module.overlayNodes || []),
        ...(finalOutputModule ? [finalOutputModule.node] : []),
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
            kvCacheState: kvCacheState && typeof kvCacheState === 'object'
                ? { ...kvCacheState }
                : null,
            tokens: resolvedTokens,
            focusBuilder: 'buildMhsaSceneModel',
            headDetailTarget: resolvedHeadDetailTarget,
            headDetailPreview,
            headDetailScene,
            mhsaHeadDetailScene,
            concatDetailTarget: resolvedConcatDetailTarget,
            concatDetailPreview,
            outputProjectionDetailTarget: resolvedOutputProjectionDetailTarget,
            outputProjectionDetailPreview,
            outputProjectionDetailScene,
            mlpDetailTarget: resolvedMlpDetailTarget,
            mlpDetailScene,
            layerNormDetailTarget: resolvedLayerNormDetailTarget,
            layerNormDetailScene
        }
    });
}
