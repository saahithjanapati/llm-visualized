import {
    MHA_FINAL_Q_COLOR,
    MHA_FINAL_K_COLOR,
    MHA_FINAL_V_COLOR,
    MHA_OUTPUT_PROJECTION_MATRIX_COLOR,
    POSITION_EMBED_COLOR,
    MLP_UP_MATRIX_COLOR,
    MLP_DOWN_MATRIX_COLOR
} from '../animations/LayerAnimationConstants.js';
import {
    buildAttentionEquationSet,
    buildAttentionProjectionEquation
} from './attentionEquationTextUtils.js';
import { buildSelectionLayerNormEquation } from './selectionPanelLayerNormEquationTextUtils.js';
import {
    formatLayerNormLabel,
    isLayerNormOutputStage,
    isPostLayerNormResidualSelection
} from '../utils/layerNormLabels.js';
import { D_MODEL, D_HEAD, VOCAB_SIZE, CONTEXT_LEN } from './selectionPanelConstants.js';
import { formatTokenLabelForPreview } from './selectionPanelFormatUtils.js';
import {
    isKvCacheInfoSelection,
    normalizeKvCachePhase
} from './kvCacheInfoUtils.js';
import { isMhsaInfoSelection } from './mhsaInfoUtils.js';
import {
    findUserDataNumber,
    findUserDataString,
    getActivationDataFromSelection,
    resolveSelectionLogitEntry
} from './selectionPanelSelectionUtils.js';
import {
    MLP_DOWN_BIAS_TOOLTIP_LABEL,
    MLP_UP_BIAS_TOOLTIP_LABEL
} from '../utils/mlpLabels.js';

const GPT2_VOCAB_SIZE_TEXT = VOCAB_SIZE.toLocaleString('en-US');
const GPT2_D_MODEL_TEXT = D_MODEL.toLocaleString('en-US');
const GPT2_D_HEAD_TEXT = D_HEAD.toLocaleString('en-US');
const GPT2_CONTEXT_LEN_TEXT = CONTEXT_LEN.toLocaleString('en-US');
const GPT2_NUM_HEADS_TEXT = Math.round(D_MODEL / D_HEAD).toLocaleString('en-US');
const GPT2_MLP_HIDDEN_TEXT = (D_MODEL * 4).toLocaleString('en-US');
const ATTENTION_VECTOR_DETAIL_ACTION = 'open-attention-vector';
const LAYERNORM_DETAIL_ACTION = 'open-layernorm';
const LAYERNORM_PARAM_DETAIL_ACTION = 'open-layernorm-param';
const QKV_SOURCE_VECTOR_DETAIL_ACTION = 'open-qkv-source-vector';
const QKV_WEIGHT_MATRIX_DETAIL_ACTION = 'open-qkv-weight-matrix';
function joinParagraphs(...parts) {
    return parts
        .filter((part) => typeof part === 'string' && part.trim().length > 0)
        .join('\n\n');
}

function inlineMath(tex) {
    const safeTex = typeof tex === 'string' ? tex.trim() : '';
    if (!safeTex) return '';
    return `$${safeTex}$`;
}

function buildLinearMapDimensionSentence({
    inputDimText = GPT2_D_MODEL_TEXT,
    outputDimText = GPT2_D_MODEL_TEXT,
    matrixShapeText = `${inputDimText} x ${outputDimText}`,
    inputName = 'input vector',
    outputName = 'output vector'
} = {}) {
    return `In the notation used in this visualization, its shape is ${matrixShapeText}: it takes a ${inputDimText}-dimensional ${inputName} and produces a ${outputDimText}-dimensional ${outputName}. That means the matrix reads ${inputDimText} input features together and recombines them into ${outputDimText} output features.`;
}

function buildLookupTableDimensionSentence({
    rowCountText = GPT2_VOCAB_SIZE_TEXT,
    outputDimText = GPT2_D_MODEL_TEXT,
    indexName = 'token ID'
} = {}) {
    return `This table has ${rowCountText} rows, and each row is a ${outputDimText}-dimensional vector. The input is a discrete ${indexName}, and the output is the row stored at that index.`;
}

function resolveTokenReference(selectionInfo = null, {
    indexKey = 'tokenIndex',
    idKey = 'tokenId',
    labelKey = 'tokenLabel'
} = {}) {
    const tokenIndexRaw = findUserDataNumber(selectionInfo, indexKey);
    const tokenIdRaw = idKey ? findUserDataNumber(selectionInfo, idKey) : null;
    const tokenIndex = Number.isFinite(tokenIndexRaw) ? Math.floor(tokenIndexRaw) : null;
    const tokenId = Number.isFinite(tokenIdRaw) ? Math.floor(tokenIdRaw) : null;
    const tokenLabelRaw = findUserDataString(selectionInfo, labelKey);
    const tokenLabel = typeof tokenLabelRaw === 'string' && tokenLabelRaw.trim().length
        ? tokenLabelRaw.trim()
        : '';
    const tokenText = tokenLabel || (Number.isFinite(tokenIndex) ? `Token ${tokenIndex + 1}` : '');
    if (!tokenText) return null;
    return { tokenText, tokenIndex, tokenId };
}

function buildInlineTokenNavMarkup(tokenText, tokenIndex = null, tokenId = null) {
    const safeText = typeof tokenText === 'string' ? tokenText.trim() : '';
    if (!safeText) return '';
    const safeTokenIndex = Number.isFinite(tokenIndex) ? Math.floor(tokenIndex) : '';
    const safeTokenId = Number.isFinite(tokenId) ? Math.floor(tokenId) : '';
    return `[[token-nav|${encodeURIComponent(safeText)}|${safeTokenIndex}|${safeTokenId}]]`;
}

function buildInlineDetailActionMarkup(linkText, action, payload = null) {
    const safeText = typeof linkText === 'string' ? linkText.trim() : '';
    const safeAction = typeof action === 'string' ? action.trim() : '';
    if (!safeText || !safeAction) return safeText;
    const encodedPayload = encodeURIComponent(JSON.stringify(payload && typeof payload === 'object' ? payload : {}));
    return `[[detail-action|${encodeURIComponent(safeText)}|${safeAction}|${encodedPayload}]]`;
}

function normalizeQkvKind(kind = 'Q') {
    const upper = String(kind || '').toUpperCase();
    if (upper === 'K') return 'K';
    if (upper === 'V') return 'V';
    return 'Q';
}

function resolveSelectionLayerNormKind(selectionInfo = null) {
    const explicitKind = findUserDataString(selectionInfo, 'layerNormKind');
    const stageLower = String(getActivationDataFromSelection(selectionInfo)?.stage || '').toLowerCase();
    const labelLower = String(selectionInfo?.label || '').toLowerCase();
    if (explicitKind === 'ln1' || explicitKind === 'ln2' || explicitKind === 'final') {
        return explicitKind;
    }
    if (stageLower.startsWith('ln1.')) return 'ln1';
    if (stageLower.startsWith('ln2.')) return 'ln2';
    if (stageLower.startsWith('final_ln')) return 'final';
    if (labelLower.includes('layernorm (top)') || labelLower.includes('final ln') || labelLower.includes('top layernorm')) {
        return 'final';
    }
    if (labelLower.includes('ln1')) return 'ln1';
    if (labelLower.includes('ln2')) return 'ln2';
    return null;
}

function resolveSelectionTokenRef(selectionInfo = null) {
    return resolveTokenReference(selectionInfo);
}

function resolveSelectionKeyTokenRef(selectionInfo = null) {
    return resolveTokenReference(selectionInfo, {
        indexKey: 'keyTokenIndex',
        idKey: 'keyTokenId',
        labelKey: 'keyTokenLabel'
    });
}

function resolveSelectionQueryTokenRef(selectionInfo = null) {
    return resolveTokenReference(selectionInfo, {
        indexKey: 'queryTokenIndex',
        idKey: 'queryTokenId',
        labelKey: 'queryTokenLabel'
    });
}

function buildTokenReference(tokenRef = null, fallback = 'this token') {
    if (!tokenRef) return fallback;
    const chip = buildInlineTokenNavMarkup(tokenRef.tokenText, tokenRef.tokenIndex, tokenRef.tokenId);
    if (!chip) return fallback;
    const genericTokenLabel = /^token\s+\d+$/i.test(tokenRef.tokenText);
    if (Number.isFinite(tokenRef.tokenIndex) && !genericTokenLabel) {
        return `${chip} at position ${tokenRef.tokenIndex + 1}`;
    }
    return chip;
}

function buildSelectionTokenReference(selectionInfo = null, fallback = 'this token') {
    return buildTokenReference(resolveSelectionTokenRef(selectionInfo), fallback);
}

function buildSelectionKeyTokenReference(selectionInfo = null, fallback = 'the source token') {
    return buildTokenReference(resolveSelectionKeyTokenRef(selectionInfo), fallback);
}

function buildSelectionQueryTokenReference(selectionInfo = null, fallback = 'the query token') {
    return buildTokenReference(resolveSelectionQueryTokenRef(selectionInfo), fallback);
}

function resolveSelectionHeadIndex(selectionInfo = null) {
    const headIndexRaw = findUserDataNumber(selectionInfo, 'headIndex');
    return Number.isFinite(headIndexRaw) ? Math.floor(headIndexRaw) : null;
}

function buildSelectionHeadReference(selectionInfo = null, { fallback = 'this attention head', capitalize = false } = {}) {
    const headIndex = resolveSelectionHeadIndex(selectionInfo);
    if (!Number.isFinite(headIndex)) return fallback;
    return `${capitalize ? 'Attention head' : 'attention head'} ${headIndex + 1}`;
}

function resolveSelectionLayerIndex(selectionInfo = null) {
    const layerIndexRaw = findUserDataNumber(selectionInfo, 'layerIndex');
    return Number.isFinite(layerIndexRaw) ? Math.floor(layerIndexRaw) : null;
}

function buildSelectionLayerReference(selectionInfo = null, { fallback = 'this layer', capitalize = true } = {}) {
    const layerIndex = resolveSelectionLayerIndex(selectionInfo);
    if (!Number.isFinite(layerIndex)) return fallback;
    return `${capitalize ? 'Layer' : 'layer'} ${layerIndex + 1}`;
}

function buildSelectionLayerJobReference(selectionInfo = null) {
    const layerIndex = resolveSelectionLayerIndex(selectionInfo);
    if (!Number.isFinite(layerIndex)) return 'this layer';
    return `Layer ${layerIndex + 1}`;
}

function sanitizeOutputTokenText(token) {
    if (token === null || token === undefined) return '';
    return String(token).replace(/\n/g, '\\n').replace(/\t/g, '\\t');
}

function formatOutputTokenLabel(token) {
    const formatted = formatTokenLabelForPreview(sanitizeOutputTokenText(token));
    return formatted || '';
}

function resolveOutputTokenDescriptor(selectionInfo = null, fallback = 'this candidate token') {
    const entry = resolveSelectionLogitEntry(selectionInfo);
    const tokenText = formatOutputTokenLabel(entry?.token);
    const tokenId = Number.isFinite(entry?.token_id)
        ? Math.floor(entry.token_id)
        : (Number.isFinite(entry?.tokenId) ? Math.floor(entry.tokenId) : null);
    if (tokenText && Number.isFinite(tokenId)) {
        return `the token ${JSON.stringify(tokenText)} (ID ${tokenId})`;
    }
    if (tokenText) {
        return `the token ${JSON.stringify(tokenText)}`;
    }
    if (Number.isFinite(tokenId)) {
        return `token ID ${tokenId}`;
    }

    const chosenTokenMatch = String(selectionInfo?.label || '').match(/^chosen token:\s*(.+)$/i);
    if (chosenTokenMatch && chosenTokenMatch[1]) {
        const chosenTokenText = formatOutputTokenLabel(chosenTokenMatch[1]);
        if (chosenTokenText) return `the token ${JSON.stringify(chosenTokenText)}`;
    }

    return fallback;
}

function resolveOutputTokenProbability(selectionInfo = null) {
    const entry = resolveSelectionLogitEntry(selectionInfo);
    if (Number.isFinite(entry?.probability)) return Number(entry.probability);
    if (Number.isFinite(entry?.prob)) return Number(entry.prob);
    return null;
}

function formatProbabilityPercentage(probability) {
    if (!Number.isFinite(probability)) return '';
    const percentage = probability * 100;
    const abs = Math.abs(percentage);
    if (abs === 0) return '0%';
    if (abs < 0.0001) return `${percentage.toExponential(2)}%`;
    if (abs < 0.01) return `${percentage.toFixed(4).replace(/\.?0+$/, '')}%`;
    return `${percentage.toFixed(2).replace(/\.?0+$/, '')}%`;
}

function resolveTopKShortlistText(selectionInfo = null) {
    const barCount = findUserDataNumber(selectionInfo, 'barCount');
    if (!Number.isFinite(barCount) || barCount <= 0) return 'the highest-scoring candidates';
    return `the top ${Math.floor(barCount).toLocaleString('en-US')} candidates`;
}

function buildTopLogitBarsDescription(selectionInfo = null) {
    const shortlistText = resolveTopKShortlistText(selectionInfo);
    return joinParagraphs(
        `These bars are ${shortlistText} for the next token after the final hidden state has been projected through the unembedding matrix into vocabulary logits and then normalized with softmax.`,
        `Each visible bar shows one token's softmax probability, ${inlineMath('p_i = \\frac{e^{\\ell_i}}{\\sum_j e^{\\ell_j}}')}. The denominator sums over the entire vocabulary, not just the bars shown here, so the visible shortlist usually adds up to less than 100%.`,
        `If you explain decoding with top-k sampling, this is the shortlist that step operates on: keep only these highest-scoring candidates, renormalize within that reduced set, and sample from it.`
    );
}

function buildChosenTokenDescription(selectionInfo = null) {
    const tokenDescriptor = resolveOutputTokenDescriptor(selectionInfo, 'the token selected for this decoding step');
    const probability = resolveOutputTokenProbability(selectionInfo);
    const probabilityText = formatProbabilityPercentage(probability);
    const shortlistText = resolveTopKShortlistText(selectionInfo);
    const tokenIndex = findUserDataNumber(selectionInfo, 'tokenIndex');
    const appendSentence = Number.isFinite(tokenIndex)
        ? `Once chosen, it is appended at position ${Math.floor(tokenIndex) + 1} in the running sequence. On the next forward pass, the model looks up that token's embedding, adds the position embedding for the new slot, and uses the extended context to predict the token after it.`
        : 'Once chosen, it is appended to the right end of the running sequence. On the next forward pass, the model looks up that token\'s embedding, adds the next position embedding, and uses the extended context to predict the token after it.';
    return joinParagraphs(
        `This marks ${tokenDescriptor}, the token that the decoding step actually selected from the output distribution.`,
        probabilityText
            ? `The displayed probability here is ${probabilityText}, which is the token's full-distribution softmax probability before any truncation. If you describe decoding with top-k sampling, the sampler first keeps ${shortlistText}, renormalizes inside that subset, and then draws one token; this label shows the winner of that selection step.`
            : `This token is chosen from the softmax distribution over vocabulary logits. If you describe decoding with top-k sampling, the sampler first keeps ${shortlistText}, renormalizes inside that subset, and then draws one token; this label shows the winner of that selection step.`,
        appendSentence
    );
}

function resolveKvCacheInfoPhase(selectionInfo = null) {
    return normalizeKvCachePhase(findUserDataString(selectionInfo, 'kvCachePhase'));
}

function buildKvCacheInfoDescription(selectionInfo = null) {
    const phase = resolveKvCacheInfoPhase(selectionInfo);
    const currentPhaseParagraph = phase === 'decode'
        ? 'Right now the visualization is in decode mode. The prompt has already been cached, so this pass computes the newest token\'s query, key, and value, appends the new key/value pair to the cache, and reuses the older cached entries for everything to the left.'
        : 'Right now the visualization is in pre-fill mode. This is the first pass over the prompt, so the model still computes keys and values for every prompt token once and writes them into the cache for later reuse.';
    const companionPhaseParagraph = phase === 'decode'
        ? 'That decode step depends on the earlier pre-fill pass, which built the initial cache for the prompt tokens and made the full prompt history available without recomputing it again.'
        : 'Once pre-fill finishes, later decode passes do not rebuild those prompt keys and values. They only add one new cache row per generated token and attend against the stored history.';

    return joinParagraphs(
        'KV cache stores each token\'s attention keys and values after they are computed, so later autoregressive steps can reuse them instead of rebuilding the whole attention history from scratch.',
        currentPhaseParagraph,
        companionPhaseParagraph,
        'That reuse is why inference gets faster. Without a KV cache, every generated token would recompute keys and values for the entire prefix again. With the cache, the model mostly does fresh work only for the newest token, which reduces repeated computation and helps long prompts or long generations run much faster.'
    );
}

function buildLogitDescription(selectionInfo = null) {
    const tokenDescriptor = resolveOutputTokenDescriptor(selectionInfo, 'this candidate token');
    const probability = resolveOutputTokenProbability(selectionInfo);
    const probabilityText = formatProbabilityPercentage(probability);
    return joinParagraphs(
        `This is the vocabulary score for ${tokenDescriptor}. The raw logit ${inlineMath('\\ell_i')} is an unnormalized compatibility score between the final hidden state and that vocabulary item, so the number itself is not yet a probability.`,
        probabilityText
            ? `Its displayed probability is ${probabilityText}, computed by softmax over all logits in the vocabulary: ${inlineMath('p_i = \\frac{e^{\\ell_i}}{\\sum_j e^{\\ell_j}}')}. What matters is how this score compares with every other token, not the absolute logit on its own.`
            : `Its probability is computed by softmax over all logits in the vocabulary: ${inlineMath('p_i = \\frac{e^{\\ell_i}}{\\sum_j e^{\\ell_j}}')}. What matters is how this score compares with every other token, not the absolute logit on its own.`,
        'If you later apply top-k sampling, this candidate remains eligible only if it survives into the highest-scoring shortlist, after which the sampler renormalizes over the surviving tokens before drawing the next token.'
    );
}

function buildLayerNormDetailActionMarkup(selectionInfo = null, {
    layerNormKind = null,
    linkText = 'LayerNorm'
} = {}) {
    const safeKind = (layerNormKind === 'ln1' || layerNormKind === 'ln2' || layerNormKind === 'final')
        ? layerNormKind
        : resolveSelectionLayerNormKind(selectionInfo);
    const layerIndex = resolveSelectionLayerIndex(selectionInfo);
    return buildInlineDetailActionMarkup(linkText, LAYERNORM_DETAIL_ACTION, {
        layerNormKind: safeKind,
        layerIndex: Number.isFinite(layerIndex) ? layerIndex : null
    });
}

function buildSelectionLayerNormReference(selectionInfo = null, {
    layerNormKind = null,
    linked = false
} = {}) {
    const safeKind = (layerNormKind === 'ln1' || layerNormKind === 'ln2' || layerNormKind === 'final')
        ? layerNormKind
        : resolveSelectionLayerNormKind(selectionInfo);
    const layerRef = buildSelectionLayerReference(selectionInfo);
    let text = 'this LayerNorm';
    if (safeKind === 'ln1') {
        text = layerRef !== 'this layer'
            ? `${formatLayerNormLabel('ln1')} in ${layerRef}`
            : `${formatLayerNormLabel('ln1')} in this layer`;
    } else if (safeKind === 'ln2') {
        text = layerRef !== 'this layer'
            ? `${formatLayerNormLabel('ln2')} in ${layerRef}`
            : `${formatLayerNormLabel('ln2')} in this layer`;
    } else if (safeKind === 'final') {
        text = 'the final LayerNorm at the top of the model';
    }
    return linked
        ? buildLayerNormDetailActionMarkup(selectionInfo, { layerNormKind: safeKind, linkText: text })
        : text;
}

function buildIncomingResidualSourceSentence(selectionInfo = null) {
    const layerIndex = resolveSelectionLayerIndex(selectionInfo);
    if (!Number.isFinite(layerIndex)) {
        return 'It is the complete token state handed over from the embeddings or from the previous layer. Multi-head self-attention reads from this running state, lets this token attend to earlier positions in the sequence, and writes the resulting update back into the residual stream through residual addition.';
    }
    if (layerIndex === 0) {
        return 'It is the complete token state handed over from the embedding sum into Layer 1. Multi-head self-attention reads from this running state, lets this token attend to earlier positions in the sequence, and writes the resulting update back into the residual stream through residual addition.';
    }
    return `It is the complete token state handed over from the output of Layer ${layerIndex} into Layer ${layerIndex + 1}. Multi-head self-attention reads from this running state, lets this token attend to earlier positions in the sequence, and writes the resulting update back into the residual stream through residual addition.`;
}

function buildAttentionVectorActionMarkup(selectionInfo = null, {
    vectorKind = 'Q',
    linkText = 'vector',
    tokenRef = null
} = {}) {
    const headIndex = resolveSelectionHeadIndex(selectionInfo);
    const layerIndex = resolveSelectionLayerIndex(selectionInfo);
    if (!Number.isFinite(headIndex) || !Number.isFinite(layerIndex) || !tokenRef) return linkText;
    const safeVectorKind = normalizeQkvKind(vectorKind);
    return buildInlineDetailActionMarkup(linkText, ATTENTION_VECTOR_DETAIL_ACTION, {
        vectorKind: safeVectorKind,
        layerIndex,
        headIndex,
        tokenIndex: Number.isFinite(tokenRef.tokenIndex) ? tokenRef.tokenIndex : null,
        tokenId: Number.isFinite(tokenRef.tokenId) ? tokenRef.tokenId : null,
        tokenLabel: tokenRef.tokenText || ''
    });
}

function buildQkvSourceVectorActionMarkup(selectionInfo = null, {
    linkText = 'x_t',
    tokenRef = null
} = {}) {
    const layerIndex = resolveSelectionLayerIndex(selectionInfo);
    if (!tokenRef || !Number.isFinite(layerIndex) || !Number.isFinite(tokenRef.tokenIndex)) return linkText;
    return buildInlineDetailActionMarkup(linkText, QKV_SOURCE_VECTOR_DETAIL_ACTION, {
        layerIndex,
        tokenIndex: tokenRef.tokenIndex,
        tokenId: Number.isFinite(tokenRef.tokenId) ? tokenRef.tokenId : null,
        tokenLabel: tokenRef.tokenText || ''
    });
}

function buildQkvWeightMatrixActionMarkup(selectionInfo = null, {
    matrixKind = 'Q',
    linkText = 'W_Q'
} = {}) {
    const headIndex = resolveSelectionHeadIndex(selectionInfo);
    const layerIndex = resolveSelectionLayerIndex(selectionInfo);
    if (!Number.isFinite(headIndex) || !Number.isFinite(layerIndex)) return linkText;
    return buildInlineDetailActionMarkup(linkText, QKV_WEIGHT_MATRIX_DETAIL_ACTION, {
        matrixKind: normalizeQkvKind(matrixKind),
        headIndex,
        layerIndex
    });
}

function buildLayerNormParamActionMarkup(selectionInfo = null, {
    param = 'scale',
    linkText = 'parameter'
} = {}) {
    const safeParam = param === 'shift' ? 'shift' : 'scale';
    const layerNormKind = resolveSelectionLayerNormKind(selectionInfo);
    const layerIndex = findUserDataNumber(selectionInfo, 'layerIndex');
    return buildInlineDetailActionMarkup(linkText, LAYERNORM_PARAM_DETAIL_ACTION, {
        param: safeParam,
        layerNormKind,
        layerIndex: Number.isFinite(layerIndex) ? Math.floor(layerIndex) : null
    });
}

function buildAttentionScoreLinkedVectorReferences(selectionInfo = null) {
    const queryTokenMeta = resolveSelectionTokenRef(selectionInfo);
    const keyTokenMeta = resolveSelectionKeyTokenRef(selectionInfo);
    const queryTokenRef = buildTokenReference(queryTokenMeta, 'the query token');
    const targetTokenRef = buildTokenReference(queryTokenMeta, 'the target token');
    const sourceTokenRef = buildTokenReference(keyTokenMeta, 'the source token');
    const queryVectorLink = buildAttentionVectorActionMarkup(selectionInfo, {
        vectorKind: 'Q',
        linkText: 'query vector',
        tokenRef: queryTokenMeta
    });
    const keyVectorLink = buildAttentionVectorActionMarkup(selectionInfo, {
        vectorKind: 'K',
        linkText: 'key vector',
        tokenRef: keyTokenMeta
    });
    const valueVectorLink = buildAttentionVectorActionMarkup(selectionInfo, {
        vectorKind: 'V',
        linkText: 'value vector',
        tokenRef: keyTokenMeta
    });
    return {
        queryTokenRef,
        targetTokenRef,
        sourceTokenRef,
        queryVectorLink,
        keyVectorLink,
        valueVectorLink
    };
}

function buildResidualStreamDescription(selectionInfo = null) {
    const tokenRef = buildSelectionTokenReference(selectionInfo, 'this token');
    const layerRef = buildSelectionLayerReference(selectionInfo);
    return joinParagraphs(
        `This is the residual-stream vector representation for ${tokenRef}${layerRef !== 'this layer' ? ` in ${layerRef}` : ''}. The full residual stream is the collection of one such vector per token position, all evolving through the model together.`,
        'At this point in the network, this vector is the token\'s running state: each sublayer reads from it, computes an update, and adds that update back instead of replacing it.',
        'As this vector moves upward through the transformer, it keeps accumulating information from the earlier context. Each attention sublayer can pull in relevant information from other token positions, and each MLP can rewrite that information into more useful local features.',
        'Those refinements build on one another across layers, so the top-layer residual vector at this position is the model\'s final internal summary of what it has inferred here. That final state is what the model turns into logits to predict the next token after this one.',
        'That design is what lets the model preserve earlier information while still refining it with attention and MLP computations. If you want to know what the model currently knows about a specific token, this vector is usually the best place to look.'
    );
}

function buildIncomingResidualDescription(selectionInfo = null) {
    const tokenRef = buildSelectionTokenReference(selectionInfo, 'this token');
    const layerRef = buildSelectionLayerReference(selectionInfo, { fallback: 'a transformer block' });
    const ln1Ref = buildSelectionLayerNormReference(selectionInfo, { layerNormKind: 'ln1', linked: true });
    const layerJobRef = buildSelectionLayerJobReference(selectionInfo);
    return joinParagraphs(
        `This is the residual-stream vector representation for ${tokenRef} as it enters ${layerRef}, before ${ln1Ref} runs.`,
        buildIncomingResidualSourceSentence(selectionInfo),
        `This vector already contains everything the model has accumulated so far for this position as it enters ${layerJobRef}, and ${layerJobRef} refines that state further for the eventual prediction of the subsequent token.`
    );
}

function buildPostAttentionResidualDescription(selectionInfo = null) {
    const tokenRef = buildSelectionTokenReference(selectionInfo, 'this token');
    const layerRef = buildSelectionLayerReference(selectionInfo);
    const ln2Ref = buildSelectionLayerNormReference(selectionInfo, { layerNormKind: 'ln2', linked: true });
    const layerJobRef = buildSelectionLayerJobReference(selectionInfo);
    return joinParagraphs(
        `This is the residual-stream vector representation for ${tokenRef} after the attention update from ${layerRef} has been added in.`,
        `This updated residual is what ${ln2Ref} reads next before the MLP branch in ${layerRef}. The MLP takes this attention-shaped running state, recombines features within this token position, and writes its update back into the residual stream through residual addition.`,
        `This vector already contains everything the model has accumulated so far for this position in ${layerJobRef}, including the attention update from earlier tokens, and the MLP in ${layerJobRef} refines those features before the state moves on toward the next-token prediction.`
    );
}

function buildPostMlpResidualDescription(selectionInfo = null) {
    const tokenRef = buildSelectionTokenReference(selectionInfo, 'this token');
    const layerRef = buildSelectionLayerReference(selectionInfo);
    return joinParagraphs(
        `This is the residual-stream vector representation for ${tokenRef} after the MLP update from ${layerRef} has been added in.`,
        `At this point, ${layerRef} is complete for this token. The vector now contains the previous state, the attention update, and the MLP update all combined through residual addition, and this final block output is what gets passed to the next layer.`,
        'The next layer will keep refining this same running state rather than starting over. By the top of the stack, the accumulated version of this vector is what the model ultimately turns into logits for the next token after this one.'
    );
}

function buildLayerNormNormalizedVectorDescription(selectionInfo = null) {
    const tokenRef = buildSelectionTokenReference(selectionInfo, 'this token');
    const layerRef = buildSelectionLayerReference(selectionInfo);
    return joinParagraphs(
        `This is the LayerNorm-normalized token-state vector for ${tokenRef}${layerRef !== 'this layer' ? ` in ${layerRef}` : ''}, before the learned scale and shift are applied.`,
        'At this moment, the vector has been re-centered and rescaled using statistics from this token\'s own features.',
        'Its job is to present the same token state in a numerically steadier form so the next learned operations see a more stable input.'
    );
}

function buildFinalLayerNormNormalizedVectorDescription(selectionInfo = null) {
    const tokenRef = buildSelectionTokenReference(selectionInfo, 'this token');
    return joinParagraphs(
        `This is the top-of-model LayerNorm-normalized token-state vector for ${tokenRef}, before the learned scale and shift are applied.`,
        'It is the last normalized version of this token\'s state before the model converts hidden features into vocabulary logits.'
    );
}

function buildLayerNormScaledVectorDescription(selectionInfo = null) {
    const tokenRef = buildSelectionTokenReference(selectionInfo, 'this token');
    const layerRef = buildSelectionLayerReference(selectionInfo);
    return joinParagraphs(
        `This is the LayerNorm token-state vector for ${tokenRef}${layerRef !== 'this layer' ? ` in ${layerRef}` : ''} after normalization and after elementwise multiplication by the learned scale vector ${inlineMath('\\gamma')}, but before the learned shift ${inlineMath('\\beta')} is added.`,
        'The scale parameters let the model decide which normalized features should be amplified, damped, or even sign-flipped before the token moves on to the next sublayer.'
    );
}

function buildLn1OutputDescription(selectionInfo = null) {
    const tokenRef = buildSelectionTokenReference(selectionInfo, 'this token');
    const layerRef = buildSelectionLayerReference(selectionInfo);
    const layerNormRef = buildSelectionLayerNormReference(selectionInfo, { layerNormKind: 'ln1', linked: true });
    return joinParagraphs(
        `This is the post-LayerNorm residual vector for ${tokenRef} after ${layerNormRef}. It is the version of that token's state that the self-attention sublayer in ${layerRef} will actually read.`,
        'From here, the layer sends branched copies of this vector into the query, key, and value projections, so this is the shared attention input for every head in the block.'
    );
}

function buildLn2OutputDescription(selectionInfo = null) {
    const tokenRef = buildSelectionTokenReference(selectionInfo, 'this token');
    const layerRef = buildSelectionLayerReference(selectionInfo);
    const layerNormRef = buildSelectionLayerNormReference(selectionInfo, { layerNormKind: 'ln2', linked: true });
    return joinParagraphs(
        `This is the post-LayerNorm residual vector for ${tokenRef} after ${layerNormRef}. It is the version of that token's state that the MLP sublayer in ${layerRef} will read next.`,
        'From here, the vector goes into the MLP up-projection, GELU, and down-projection, so this normalized residual state is the starting point for the feed-forward update in this layer.'
    );
}

const TOKEN_CHIP_DESCRIPTION = joinParagraphs(
    'This is one discrete input token ID. The model uses this ID to select the corresponding row from the vocabulary embedding table so the token can enter the network as a learned feature vector.',
    'That lookup gives the model a dense starting representation for the token, carrying usage patterns and associations learned during training.'
);

const POSITION_CHIP_DESCRIPTION = joinParagraphs(
    'This is the position index for a token in the sequence. The model uses it to select a learned position vector that tells the network where this token sits in the prompt.',
    'Adding that position vector to the token embedding gives the token an order-aware starting state before any attention layers run.'
);

const TOKEN_EMBEDDING_VECTOR_DESCRIPTION = joinParagraphs(
    `This is the learned embedding vector for one token: one row selected from a ${GPT2_VOCAB_SIZE_TEXT} x ${GPT2_D_MODEL_TEXT} embedding table. A word or subword token becomes usable to the model only after this lookup converts it into continuous features.`,
    'At this stage, the vector still has no context about the rest of the sequence. It only reflects what the model has learned about this specific token type in general, because attention has not yet compared it with any other token positions.',
    'The embedding is learned jointly with the rest of the model to make next-token prediction easier. From a research perspective, embedding spaces often organize tokens by distributional behavior: tokens that appear in similar contexts, play similar syntactic roles, share morphology, or refer to related concepts often end up in nearby or systematically related parts of the space.',
    'That does not mean each coordinate has a simple human-readable meaning. Instead, this vector gives the later layers a starting bundle of learned features about how this token tends to behave before any context-specific attention or MLP computation refines it.'
);

const POSITION_EMBEDDING_VECTOR_DESCRIPTION = joinParagraphs(
    'This is the learned position vector for one sequence index. It tells the model where this token sits in the prompt so that order and distance can affect later computations.',
    `GPT-2 learns a separate vector for each position up to its context window, so the same token can start from a different state when it appears earlier or later. That gives later attention layers a way to factor sequence order into their decisions.`,
    'Once this position information is added into the token state, it becomes part of the running representation that the transformer keeps refining layer by layer until the final hidden state is used to predict the next token.'
);

const EMBEDDING_SUM_DESCRIPTION = joinParagraphs(
    'This vector is the sum of token meaning and token position. It is the first full model-state vector for this token and the starting point for everything the transformer does afterward.',
    'From this point on, the model repeatedly rewrites this state through residual updates. Attention adds information gathered from other tokens, and the MLP adds token-local nonlinear feature transformations.'
);

const EMBEDDING_CONNECTOR_TRAIL_DESCRIPTION = joinParagraphs(
    'This connector marks the path from the symbolic inputs into the vector computation. It is not itself a learned parameter or activation; it is a guide showing where token and position information enter the network.',
    'Its purpose is to make the handoff legible: discrete token IDs and position indices become continuous vectors, and those vectors then become the initial residual stream.'
);

const VOCAB_EMBEDDING_MATRIX_DESCRIPTION = joinParagraphs(
    `This is the token embedding table. In GPT-2 small it has ${GPT2_VOCAB_SIZE_TEXT} rows and ${GPT2_D_MODEL_TEXT} columns.`,
    buildLookupTableDimensionSentence({
        rowCountText: GPT2_VOCAB_SIZE_TEXT,
        outputDimText: GPT2_D_MODEL_TEXT,
        indexName: 'token ID'
    }),
    'When the model receives a token ID, it uses that ID to select the corresponding row from this table. That row becomes the token\'s learned starting representation before position is added, giving the rest of the network a continuous feature vector it can compare, transform, and refine.'
);

const POSITIONAL_EMBEDDING_MATRIX_DESCRIPTION = joinParagraphs(
    `This is the table of learned position vectors. It has ${GPT2_CONTEXT_LEN_TEXT} rows and ${GPT2_D_MODEL_TEXT} columns, with one learned row for each sequence position in GPT-2\'s context window.`,
    buildLookupTableDimensionSentence({
        rowCountText: GPT2_CONTEXT_LEN_TEXT,
        outputDimText: GPT2_D_MODEL_TEXT,
        indexName: 'position index'
    }),
    'For each token position, the model selects one row from this table and adds it elementwise to the token embedding. That lets later attention layers treat "the same token in a different place" as a meaningfully different starting state.'
);

const UNEMBEDDING_MATRIX_DESCRIPTION = joinParagraphs(
    'This is the output projection, often called the unembedding matrix. It maps the final model-state vector back into vocabulary space, producing one logit for each token in the vocabulary.',
    buildLinearMapDimensionSentence({
        inputDimText: GPT2_D_MODEL_TEXT,
        outputDimText: GPT2_VOCAB_SIZE_TEXT,
        matrixShapeText: `${GPT2_D_MODEL_TEXT} x ${GPT2_VOCAB_SIZE_TEXT}`,
        inputName: 'final hidden-state vector',
        outputName: 'logit vector over the vocabulary'
    }),
    'Those logits say how compatible the current hidden state is with each possible next token. After softmax, they become probabilities that the sampling step can use. In GPT-2, these weights are tied to the input token embeddings.'
);

function buildLayerNormDescription(selectionInfo = null) {
    const scaleLink = buildLayerNormParamActionMarkup(selectionInfo, {
        param: 'scale',
        linkText: 'scale parameter'
    });
    const shiftLink = buildLayerNormParamActionMarkup(selectionInfo, {
        param: 'shift',
        linkText: 'shift parameter'
    });
    const layerNormRef = buildSelectionLayerNormReference(selectionInfo);
    return joinParagraphs(
        `${layerNormRef.charAt(0).toUpperCase()}${layerNormRef.slice(1)} normalizes one token's feature vector by subtracting its mean and dividing by its standard deviation across features, then applying learned ${scaleLink} and ${shiftLink}. This keeps feature magnitudes under control and gives later computations a more stable input.`,
        'In GPT-2, LayerNorm runs before attention and before the MLP. That means the model normalizes the residual stream first and then feeds that conditioned state into each sublayer.',
        buildLayerNormHadamardFootnote(`the learned ${scaleLink}`)
    );
}

function buildTopLayerNormDescription(selectionInfo = null) {
    const scaleLink = buildLayerNormParamActionMarkup(selectionInfo, {
        param: 'scale',
        linkText: 'scale parameter'
    });
    const shiftLink = buildLayerNormParamActionMarkup(selectionInfo, {
        param: 'shift',
        linkText: 'shift parameter'
    });
    return joinParagraphs(
        `This is the final LayerNorm at the top of the model. It performs the same normalize-then-affine operation as other LayerNorm blocks, but here the learned ${scaleLink} and ${shiftLink} prepare the representation directly for the output vocabulary projection.`,
        'By the time the model reaches this point, all 12 transformer layers have already contributed their updates. This LayerNorm is the last conditioning step before logits are computed.',
        buildLayerNormHadamardFootnote(`the learned ${scaleLink}`)
    );
}

function buildLayerNormHadamardFootnote(scaleRef = inlineMath('\\gamma')) {
    return `* In the equation preview, ${inlineMath('\\odot')} means elementwise multiplication: each feature in the normalized vector is multiplied by the matching feature in ${scaleRef}.`;
}

const LN_SCALE_PARAMETER_DESCRIPTION = joinParagraphs(
    `This is a learned LayerNorm scale parameter vector, usually written as ${inlineMath('\\gamma')}. It is not produced by the current prompt; it is a fixed trained parameter shared across all tokens in this layer.`,
    'After normalization, each feature is multiplied by its corresponding scale value. That gives the model a way to keep some features large, suppress others, and generally choose a useful post-normalization coordinate system.',
    buildLayerNormHadamardFootnote()
);

const LN_SHIFT_PARAMETER_DESCRIPTION = joinParagraphs(
    `This is a learned LayerNorm shift parameter vector, usually written as ${inlineMath('\\beta')}. Like ${inlineMath('\\gamma')}, it is a trained parameter shared across tokens rather than a prompt-dependent activation.`,
    `After the normalized vector is scaled, ${inlineMath('\\beta')} is added featurewise. That lets the model choose a learned baseline or offset for each feature instead of forcing every normalized feature to stay centered at zero.`
);

const FINAL_LN_SCALE_DESCRIPTION = joinParagraphs(
    'This is the learned scale vector for the final LayerNorm at the top of the model. It plays the same mathematical role as other LayerNorm scale vectors, but here it prepares the representation immediately before logits are computed.',
    'Its job is to make the final hidden state land in a feature space that the unembedding matrix can read effectively.',
    buildLayerNormHadamardFootnote()
);

const FINAL_LN_SHIFT_DESCRIPTION = joinParagraphs(
    'This is the learned shift vector for the final LayerNorm at the top of the model. It is added after final scaling and before the unembedding step.',
    'Because the logits are read directly from the post-final-LayerNorm state, this shift influences the final baseline from which vocabulary scores are computed.'
);

function buildQueryWeightMatrixDescription(selectionInfo = null) {
    const headRef = buildSelectionHeadReference(selectionInfo);
    const layerRef = buildSelectionLayerReference(selectionInfo);
    return joinParagraphs(
        `This is the learned query projection matrix for ${headRef} in ${layerRef}. For each token state entering attention, the model multiplies that token's LayerNorm output by this matrix to compute a ${GPT2_D_HEAD_TEXT}-dimensional query vector.`,
        buildLinearMapDimensionSentence({
            inputDimText: GPT2_D_MODEL_TEXT,
            outputDimText: GPT2_D_HEAD_TEXT,
            matrixShapeText: `${GPT2_D_MODEL_TEXT} x ${GPT2_D_HEAD_TEXT}`,
            inputName: 'LayerNorm output vector',
            outputName: 'query vector'
        }),
        `In symbols, this is the ${inlineMath(ATTENTION_QUERY_PROJECTION_TEX)} step for this attention head. The matrix multiplication recombines the full token state into the query space that determines what this attention head will search for.`,
        `The same matrix is applied at every token position, so ${headRef} produces one query vector per token. Those query vectors are then dotted with the key vectors from the same attention head to form raw attention scores.`
    );
}

function buildKeyWeightMatrixDescription(selectionInfo = null) {
    const headRef = buildSelectionHeadReference(selectionInfo);
    const layerRef = buildSelectionLayerReference(selectionInfo);
    return joinParagraphs(
        `This is the learned key projection matrix for ${headRef} in ${layerRef}. For each token state entering attention, the model multiplies that token's LayerNorm output by this matrix to compute a ${GPT2_D_HEAD_TEXT}-dimensional key vector.`,
        buildLinearMapDimensionSentence({
            inputDimText: GPT2_D_MODEL_TEXT,
            outputDimText: GPT2_D_HEAD_TEXT,
            matrixShapeText: `${GPT2_D_MODEL_TEXT} x ${GPT2_D_HEAD_TEXT}`,
            inputName: 'LayerNorm output vector',
            outputName: 'key vector'
        }),
        `In symbols, this is the ${inlineMath(ATTENTION_KEY_PROJECTION_TEX)} step for this attention head. The matrix multiplication places each token into the key space that queries will be compared against.`,
        `The same matrix is applied independently at every token position, so ${headRef} produces a separate key vector for every token in the sequence. Those key vectors are what the head's queries use in their scaled dot products.`
    );
}

function buildValueWeightMatrixDescription(selectionInfo = null) {
    const headRef = buildSelectionHeadReference(selectionInfo);
    const layerRef = buildSelectionLayerReference(selectionInfo);
    return joinParagraphs(
        `This is the learned value projection matrix for ${headRef} in ${layerRef}. For each token state entering attention, the model multiplies that token's LayerNorm output by this matrix to compute a ${GPT2_D_HEAD_TEXT}-dimensional value vector.`,
        buildLinearMapDimensionSentence({
            inputDimText: GPT2_D_MODEL_TEXT,
            outputDimText: GPT2_D_HEAD_TEXT,
            matrixShapeText: `${GPT2_D_MODEL_TEXT} x ${GPT2_D_HEAD_TEXT}`,
            inputName: 'LayerNorm output vector',
            outputName: 'value vector'
        }),
        `In symbols, this is the ${inlineMath(ATTENTION_VALUE_PROJECTION_TEX)} step for this attention head. The matrix multiplication places each token into the value space that the attention head will actually mix after softmax has chosen the weights.`,
        `The same matrix is applied independently at every token position, so ${headRef} produces one value vector per token. Later, each of those value vectors is multiplied by a scalar attention weight and added into the head's weighted sum.`
    );
}

const OUTPUT_PROJECTION_MATRIX_DESCRIPTION = joinParagraphs(
    'This is the learned output projection for the attention block. After all heads produce their own outputs, those outputs are concatenated and this matrix maps the combined vector back to model width.',
    buildLinearMapDimensionSentence({
        inputDimText: GPT2_D_MODEL_TEXT,
        outputDimText: GPT2_D_MODEL_TEXT,
        matrixShapeText: `${GPT2_D_MODEL_TEXT} x ${GPT2_D_MODEL_TEXT}`,
        inputName: `concatenated head-output vector (${GPT2_NUM_HEADS_TEXT} heads x ${GPT2_D_HEAD_TEXT} dimensions each)`,
        outputName: 'model-width attention output vector'
    }),
    'Again, this is applied token by token: each token position has its own concatenated attention-head output, and this same matrix maps each of those concatenated vectors back to residual width.',
    'Its job is to let the model mix information across heads and express the final attention update in the same coordinate system as the residual stream.'
);

const OUTPUT_PROJECTION_BIAS_DESCRIPTION = joinParagraphs(
    `This is the learned output-projection bias ${inlineMath('b_O')}. It is a ${GPT2_D_MODEL_TEXT}-dimensional parameter vector added after the concatenated head output is multiplied by ${inlineMath('W_O')}.`,
    `In symbols, the attention block computes ${inlineMath('O_t = H_t W_O + b_O')}. The same bias is shared across every token position, so it shifts the baseline of the attention update before that update is added back into the residual stream.`
);

const MLP_UP_WEIGHT_MATRIX_DESCRIPTION = joinParagraphs(
    'This is the first learned matrix in the feed-forward network. It expands the token state from model width into a larger hidden space so the model has more room to build nonlinear features.',
    buildLinearMapDimensionSentence({
        inputDimText: GPT2_D_MODEL_TEXT,
        outputDimText: GPT2_MLP_HIDDEN_TEXT,
        matrixShapeText: `${GPT2_D_MODEL_TEXT} x ${GPT2_MLP_HIDDEN_TEXT}`,
        inputName: 'token-state vector',
        outputName: 'expanded MLP hidden vector'
    }),
    'This computation is token-local: every token goes through the same MLP weights independently at its own position. That means the layer produces one expanded hidden vector per token position, all using the same learned matrix.'
);

const MLP_DOWN_WEIGHT_MATRIX_DESCRIPTION = joinParagraphs(
    'This is the second learned matrix in the feed-forward network. It takes the expanded, nonlinearly transformed MLP state and compresses it back down to model width.',
    buildLinearMapDimensionSentence({
        inputDimText: GPT2_MLP_HIDDEN_TEXT,
        outputDimText: GPT2_D_MODEL_TEXT,
        matrixShapeText: `${GPT2_MLP_HIDDEN_TEXT} x ${GPT2_D_MODEL_TEXT}`,
        inputName: 'expanded MLP hidden vector',
        outputName: 'residual-width MLP output vector'
    }),
    'This is again applied independently at every token position, so it produces one residual-width MLP output vector per token.',
    'That down-projection turns the MLP\'s private hidden features into a residual-sized update that can be added back into the main stream.'
);

const MLP_UP_BIAS_DESCRIPTION = joinParagraphs(
    `This is the learned up-projection bias ${inlineMath('b_{\\text{up}}')}. It is a ${GPT2_MLP_HIDDEN_TEXT}-dimensional parameter vector that is added to ${inlineMath('x_{\\text{ln}} W_{\\text{up}}')} before GELU is applied.`,
    'Unlike a token activation, this bias is shared across every token position. Its job is to shift the baseline of each hidden MLP feature so different units become easier or harder to activate.'
);

const MLP_DOWN_BIAS_DESCRIPTION = joinParagraphs(
    `This is the learned down-projection bias ${inlineMath('b_{\\text{down}}')}. It is a ${GPT2_D_MODEL_TEXT}-dimensional parameter vector that is added after the expanded MLP state is multiplied by ${inlineMath('W_{\\text{down}}')}.`,
    'This bias is also shared across all token positions. It lets the MLP set a default residual-width offset before the branch output is added back into the main residual stream.'
);

const MLP_GENERIC_DESCRIPTION = joinParagraphs(
    'The MLP is the token-wise feed-forward network inside each transformer block. It processes each token independently using the same learned weights at every position.',
    'Its role is to create nonlinear feature combinations within each token state. The usual pattern is expand, apply GELU, compress, then add the result back into the residual stream.'
);

const MLP_UP_VECTOR_DESCRIPTION = joinParagraphs(
    'This is the token state after the MLP up-projection and before the GELU nonlinearity. The model has moved into a larger hidden space where it can build richer intermediate features.',
    'At this point the representation is still purely linear in the input token state. The nonlinearity that follows is what makes the MLP more expressive than just another matrix multiply.'
);

const MLP_ACTIVATION_DESCRIPTION = joinParagraphs(
    'This is the MLP state after the GELU nonlinearity. GELU selectively bends and gates features instead of passing everything through linearly.',
    'That nonlinearity is what makes the MLP expressive. It lets the model create feature detectors and interactions that depend on the current activation values before the state is projected back down.'
);

const MLP_DOWN_VECTOR_DESCRIPTION = joinParagraphs(
    'This is the token state after the MLP down-projection, back at model width. It is the MLP branch\'s proposed update to the residual stream.',
    'The next step is residual addition, where this branch output is added to the running token state instead of replacing it.'
);

const MLP_EXPANDED_SEGMENTS_DESCRIPTION = joinParagraphs(
    'This is one expanded MLP activation shown as multiple visible segments. Mathematically it is still one vector; the segmentation is only a display choice to make the 4x larger hidden width easier to see.',
    'The important model fact is that the MLP temporarily moves into a larger feature space, applies a nonlinearity there, and then compresses back down.'
);

const POST_ATTENTION_OUTPUT_DESCRIPTION = joinParagraphs(
    'This is the attention branch output after head concatenation and the output projection. It is the full attention update that this layer wants to write back into the token\'s residual state.',
    'It is no longer a per-head object. By this point, all head-specific weighted sums have already been combined into one model-width vector.'
);

const Q_COPIES_DESCRIPTION = joinParagraphs(
    'These are branched copies of the LayerNorm output that are about to be turned into query vectors. Each attention head gets its own learned projection, so the model duplicates the token state before sending it through head-specific Q matrices.',
    'The important idea is that all heads start from the same token state but then ask different questions about the context.'
);

const K_COPIES_DESCRIPTION = joinParagraphs(
    'These are branched copies of the LayerNorm output that are about to be turned into key vectors. Each attention head uses its own key projection so that different attention heads can organize context in different ways.',
    'The keys derived from these copies are what queries will compare against to decide how much each source token matters.'
);

const V_COPIES_DESCRIPTION = joinParagraphs(
    'These are branched copies of the LayerNorm output that are about to be turned into value vectors. Each attention head learns its own value space because different attention heads may want to pass different kinds of information forward.',
    'Once attention weights are computed, those weights are applied to the resulting value vectors and summed into the attention-head output.'
);

function buildQueryVectorDescription(selectionInfo = null) {
    const tokenMeta = resolveSelectionTokenRef(selectionInfo);
    const tokenRef = buildTokenReference(tokenMeta, 'this token');
    const headRef = buildSelectionHeadReference(selectionInfo);
    const sourceVectorLink = buildQkvSourceVectorActionMarkup(selectionInfo, {
        linkText: inlineMath('x_t'),
        tokenRef: tokenMeta
    });
    const weightMatrixLink = buildQkvWeightMatrixActionMarkup(selectionInfo, {
        matrixKind: 'Q',
        linkText: inlineMath('W_Q')
    });
    return joinParagraphs(
        `This is the ${GPT2_D_HEAD_TEXT}-dimensional query vector for ${tokenRef} in ${headRef}. It is produced by multiplying the post-LayerNorm residual vector ${sourceVectorLink} for this token by the head-specific query matrix ${weightMatrixLink}, which is the ${inlineMath(ATTENTION_QUERY_PROJECTION_TEX)} computation for this attention head.`,
        `The model takes dot products between this query vector and every key vector in ${headRef}, producing one raw attention score per source token before scaling by ${inlineMath('\\frac{1}{\\sqrt{d_h}}')}, causal masking, and softmax.`,
        `Those scores determine how strongly ${tokenRef} will read from each source token when ${headRef} forms its weighted sum of value vectors.`
    );
}

function buildKeyVectorDescription(selectionInfo = null) {
    const tokenMeta = resolveSelectionTokenRef(selectionInfo);
    const tokenRef = buildTokenReference(tokenMeta, 'this token');
    const headRef = buildSelectionHeadReference(selectionInfo);
    const sourceVectorLink = buildQkvSourceVectorActionMarkup(selectionInfo, {
        linkText: inlineMath('x_j'),
        tokenRef: tokenMeta
    });
    const weightMatrixLink = buildQkvWeightMatrixActionMarkup(selectionInfo, {
        matrixKind: 'K',
        linkText: inlineMath('W_K')
    });
    return joinParagraphs(
        `This is the ${GPT2_D_HEAD_TEXT}-dimensional key vector for ${tokenRef} in ${headRef}. It is produced by multiplying the post-LayerNorm residual vector ${sourceVectorLink} for this token by the head-specific key matrix ${weightMatrixLink}, which is the ${inlineMath(ATTENTION_KEY_VECTOR_TEX)} computation for this attention head.`,
        `When a query token computes ${inlineMath(ATTENTION_QK_DOT_PRODUCT_TEX)} in ${headRef}, this vector supplies the ${inlineMath('k_{j,i}')} term for ${tokenRef}. After scaling by ${inlineMath('\\frac{1}{\\sqrt{d_h}}')}, and after masking removes disallowed future positions, that interaction contributes one raw attention score in the query token's row.`,
        `Its job is to determine how strongly other tokens can attend to ${tokenRef} in ${headRef}.`
    );
}

function buildValueVectorDescription(selectionInfo = null) {
    const tokenMeta = resolveSelectionTokenRef(selectionInfo);
    const tokenRef = buildTokenReference(tokenMeta, 'this token');
    const headRef = buildSelectionHeadReference(selectionInfo);
    const sourceVectorLink = buildQkvSourceVectorActionMarkup(selectionInfo, {
        linkText: inlineMath('x_j'),
        tokenRef: tokenMeta
    });
    const weightMatrixLink = buildQkvWeightMatrixActionMarkup(selectionInfo, {
        matrixKind: 'V',
        linkText: inlineMath('W_V')
    });
    return joinParagraphs(
        `This is the ${GPT2_D_HEAD_TEXT}-dimensional value vector for ${tokenRef} in ${headRef}. It is produced by multiplying the post-LayerNorm residual vector ${sourceVectorLink} for this token by the head-specific value matrix ${weightMatrixLink}, which is the ${inlineMath(ATTENTION_VALUE_VECTOR_TEX)} computation for this attention head.`,
        `Once a query token's raw scores have been turned into weights ${inlineMath('\\alpha_{t,j}')} by softmax, this vector is multiplied by its scalar weight and added into the weighted sum.`,
        `That is how information from ${tokenRef} can flow into other tokens through ${headRef}.`
    );
}

function buildQueryBiasVectorDescription(selectionInfo = null) {
    const headRef = buildSelectionHeadReference(selectionInfo);
    const layerRef = buildSelectionLayerReference(selectionInfo);
    const weightMatrixLink = buildQkvWeightMatrixActionMarkup(selectionInfo, {
        matrixKind: 'Q',
        linkText: inlineMath('W_Q')
    });
    return joinParagraphs(
        `This is the learned query bias vector for ${headRef} in ${layerRef}. It is a fixed ${GPT2_D_HEAD_TEXT}-dimensional parameter added after the token state is multiplied by the head-specific query matrix ${weightMatrixLink}.`,
        `In symbols, the head computes ${inlineMath('q_{t,i} = x_t W_Q + b_Q')}. The same bias is reused at every token position in this head, so it acts as a learned offset in query space before any query-key comparisons happen.`,
        'Its job is to shift the baseline of the head\'s query features, making some query directions easier or harder to activate.'
    );
}

function buildKeyBiasVectorDescription(selectionInfo = null) {
    const headRef = buildSelectionHeadReference(selectionInfo);
    const layerRef = buildSelectionLayerReference(selectionInfo);
    const weightMatrixLink = buildQkvWeightMatrixActionMarkup(selectionInfo, {
        matrixKind: 'K',
        linkText: inlineMath('W_K')
    });
    return joinParagraphs(
        `This is the learned key bias vector for ${headRef} in ${layerRef}. It is a fixed ${GPT2_D_HEAD_TEXT}-dimensional parameter added after the token state is multiplied by the head-specific key matrix ${weightMatrixLink}.`,
        `In symbols, the head computes ${inlineMath('k_{j,i} = x_j W_K + b_K')}. The same bias is shared across all token positions in this head, so it sets a default offset in key space for every token.`,
        'That offset changes how easily other tokens\' queries can match against these keys when raw attention scores are formed.'
    );
}

function buildValueBiasVectorDescription(selectionInfo = null) {
    const headRef = buildSelectionHeadReference(selectionInfo);
    const layerRef = buildSelectionLayerReference(selectionInfo);
    const weightMatrixLink = buildQkvWeightMatrixActionMarkup(selectionInfo, {
        matrixKind: 'V',
        linkText: inlineMath('W_V')
    });
    return joinParagraphs(
        `This is the learned value bias vector for ${headRef} in ${layerRef}. It is a fixed ${GPT2_D_HEAD_TEXT}-dimensional parameter added after the token state is multiplied by the head-specific value matrix ${weightMatrixLink}.`,
        `In symbols, the head computes ${inlineMath('v_{j,i} = x_j W_V + b_V')}. The same bias is shared across tokens, so every value vector in this head starts from the same learned baseline offset.`,
        'Its role is to shift the default contents of the value space that this attention head writes into its weighted sum.'
    );
}

function buildCachedKeyVectorDescription(selectionInfo = null) {
    const tokenRef = buildSelectionTokenReference(selectionInfo, 'this cached token');
    const headRef = buildSelectionHeadReference(selectionInfo);
    return joinParagraphs(
        `This is the cached key vector for ${tokenRef} in ${headRef}. It was computed on an earlier decoding step and stored so later queries can still take dot products against it.`,
        'Reusing cached keys is what makes autoregressive decoding efficient: when a new token arrives, the model computes the new query and compares it against the stored past keys instead of rebuilding those old keys every time.'
    );
}

function buildCachedValueVectorDescription(selectionInfo = null) {
    const tokenRef = buildSelectionTokenReference(selectionInfo, 'this cached token');
    const headRef = buildSelectionHeadReference(selectionInfo);
    return joinParagraphs(
        `This is the cached value vector for ${tokenRef} in ${headRef}. It stores the value content from an earlier decoding step so future queries can still read from that token.`,
        `After the new token's attention row is computed in ${headRef}, the model multiplies the relevant post-softmax weights by these cached values and adds them into the weighted sum, without recomputing old value vectors from scratch.`
    );
}

const MERGED_KEY_VECTORS_DESCRIPTION = joinParagraphs(
    'These are the key vectors from multiple attention heads shown together in one combined view. Mathematically they still belong to separate attention heads; they are only grouped here so you can inspect more of the attention state at once.',
    'Each attention head builds its own attention matrix from its own queries and keys, so these vectors can be viewed as parallel key sets rather than one shared pool.'
);

const MERGED_VALUE_VECTORS_DESCRIPTION = joinParagraphs(
    'These are the value vectors from multiple attention heads shown together in one combined view. Each attention head still uses only its own values when forming its weighted sum.',
    'Seeing them together is useful because it makes the multi-head structure visible: different attention heads can carry different kinds of payload even for the same token position.'
);

function buildWeightedValueVectorDescription(selectionInfo = null) {
    const sourceTokenRef = buildSelectionTokenReference(selectionInfo, 'the source token');
    const queryTokenRef = buildSelectionQueryTokenReference(selectionInfo, 'the query token');
    const headRef = buildSelectionHeadReference(selectionInfo);
    return joinParagraphs(
        `This is the weighted value contribution from ${sourceTokenRef} to ${queryTokenRef} in ${headRef}. In equations, this contribution can be written as ${inlineMath('\\tilde{V}_{t,j} = \\alpha_{t,j} V_j')}.`,
        `The scalar ${inlineMath('\\alpha_{t,j}')} comes from the scaled dot product between the query for ${queryTokenRef} and the key for ${sourceTokenRef}, after masking and softmax. Multiplying that scalar by the value vector from ${sourceTokenRef} produces one contribution to the attention-head output.`,
        `The full output of ${headRef} for ${queryTokenRef} is the sum of these weighted contributions over all source tokens, ${inlineMath('H_t = \\sum_j \\tilde{V}_{t,j}')}.`
    );
}

function buildAttentionWeightedSumDescription(selectionInfo = null) {
    const queryTokenRef = buildSelectionTokenReference(selectionInfo, 'the query token');
    const headRef = buildSelectionHeadReference(selectionInfo);
    return joinParagraphs(
        `This is the ${GPT2_D_HEAD_TEXT}-dimensional head output ${inlineMath('H_i')} for ${queryTokenRef} in ${headRef}. It is formed by mixing that head's value vectors with the post-softmax attention weights from ${queryTokenRef}'s attention row.`,
        `In symbols, ${headRef} computes ${inlineMath('H_{t,i} = \\sum_j \\alpha_{t,j}^{(i)} V_j^{(i)}')}. Each source token contributes its value vector, scaled by how much attention ${queryTokenRef} assigns to that token in this head after softmax and masking.`,
        `This is the final vector that ${headRef} contributes for that token before the model concatenates all ${GPT2_NUM_HEADS_TEXT} head outputs and applies the output projection ${inlineMath('W_O')}.`
    );
}

const ATTENTION_GENERIC_DESCRIPTION = joinParagraphs(
    'Self-attention is the mechanism that lets each token build a query vector, take dot products with key vectors in each attention head, scale and mask those scores, apply softmax across each row, and then use the resulting weights to form weighted sums of value vectors.',
    'The Q, K, and V matrix multiplications determine what each attention head measures and what content it can pass forward, so different attention heads can learn different lookup patterns and different kinds of token-to-token interactions.'
);

const MHSA_INFO_DESCRIPTION = joinParagraphs(
    `Multi-head self-attention lets each token project its current state into queries, keys, and values, compare ${inlineMath('QK^\\top')} inside each head, and turn those scores into read weights with ${inlineMath('\\mathrm{softmax}(\\cdot)')}.`,
    `Those weights decide how much of every source token's value vector should be mixed into the output for the current token. GPT-2 runs ${GPT2_NUM_HEADS_TEXT} heads in parallel, so the same token can look for different patterns and relationships at once.`,
    `After that, the head outputs are concatenated and projected back to model width. In short: queries decide what to look for, keys decide where it matches, and values carry the content that gets passed forward.`
);

function buildAttentionScoreGenericDescription(selectionInfo = null) {
    const headRef = buildSelectionHeadReference(selectionInfo);
    const linkedVectors = buildAttentionScoreLinkedVectorReferences(selectionInfo);
    return joinParagraphs(
        `This is one entry in the attention matrix from source token ${linkedVectors.sourceTokenRef} to target token ${linkedVectors.targetTokenRef} in ${headRef}.`,
        `Depending on the stage, it is either the raw scaled dot-product score between the target token's ${linkedVectors.queryVectorLink} and the source token's ${linkedVectors.keyVectorLink}, or the post-softmax weight derived from that score.`,
        `That weight determines how much of the source token's ${linkedVectors.valueVectorLink} is mixed into the output of ${headRef}.`
    );
}

function buildAttentionPreScoreDescription(selectionInfo = null) {
    const headRef = buildSelectionHeadReference(selectionInfo);
    const linkedVectors = buildAttentionScoreLinkedVectorReferences(selectionInfo);
    return joinParagraphs(
        `This is the pre-softmax attention score from source token ${linkedVectors.sourceTokenRef} to target token ${linkedVectors.targetTokenRef} in ${headRef}.`,
        `It is the scaled dot product ${inlineMath(ATTENTION_PRE_SCORE_TEX)} between the target token's ${linkedVectors.queryVectorLink} and the source token's ${linkedVectors.keyVectorLink}, with causal masking applied before the row is normalized.`,
        `At this stage the number is still a raw score. Softmax will turn the full row into normalized attention weights, and this entry will then scale the source token's ${linkedVectors.valueVectorLink}.`
    );
}

function buildAttentionPostScoreDescription(selectionInfo = null) {
    const headRef = buildSelectionHeadReference(selectionInfo);
    const linkedVectors = buildAttentionScoreLinkedVectorReferences(selectionInfo);
    return joinParagraphs(
        `This is the post-softmax attention weight from source token ${linkedVectors.sourceTokenRef} to target token ${linkedVectors.targetTokenRef} in ${headRef}.`,
        `It comes from applying softmax to the row of scaled dot products built from the target token's ${linkedVectors.queryVectorLink} against the head's key vectors, including the source token's ${linkedVectors.keyVectorLink}. For one entry in that row, softmax is ${inlineMath(ATTENTION_POST_WEIGHT_TEX)}.`,
        `This scalar is then multiplied into the source token's ${linkedVectors.valueVectorLink}, so it directly sets how much information from that token enters the output of ${headRef}.`
    );
}

function buildCausalMaskDescription(selectionInfo = null) {
    const activation = getActivationDataFromSelection(selectionInfo);
    const sourceTokenRef = buildSelectionKeyTokenReference(selectionInfo, 'the source token');
    const targetTokenRef = buildSelectionQueryTokenReference(selectionInfo, 'the target token');
    const headRef = buildSelectionHeadReference(selectionInfo);
    const maskValue = activation?.maskValue;
    const isBlocked = activation?.isMasked === true || maskValue === Number.NEGATIVE_INFINITY;
    const valueText = isBlocked ? inlineMath('-\\infty') : inlineMath('0');

    return joinParagraphs(
        `This is one entry of the causal mask ${inlineMath('M_{\\mathrm{causal}}')} from source token ${sourceTokenRef} to target token ${targetTokenRef} in ${headRef}. This mask entry is added to the raw query-key score before softmax so the head knows which source positions the target token is allowed to read from.`,
        isBlocked
            ? `Here the mask value is ${valueText}. That means ${sourceTokenRef} is a future position relative to ${targetTokenRef}, so GPT-style autoregressive self-attention must block this connection. Adding ${inlineMath('-\\infty')} before softmax forces this cell's probability to zero, so the target token cannot attend to information that comes later in the sequence.`
            : `Here the mask value is ${valueText}. That means ${sourceTokenRef} is at or before ${targetTokenRef}, so this connection is allowed. Adding ${inlineMath('0')} leaves the raw query-key score unchanged and lets this source token compete normally inside the softmax row.`,
        `Causal masking is what keeps self-attention compatible with next-token prediction: when the model is generating token ${targetTokenRef}, it may use the current and earlier tokens, but it must not peek at future tokens that have not been generated yet.`
    );
}

const GENERIC_VECTOR_DESCRIPTION = joinParagraphs(
    'This vector is a learned feature representation used somewhere along the model\'s forward pass. A vector in a transformer should be thought of as a bundle of coordinates that the network has learned to make useful.',
    'Different parts of the model reinterpret and recombine these coordinates for different purposes: residual state, queries, keys, values, MLP activations, and logits all come from vectors passed through different learned transformations.'
);

const GENERIC_WEIGHT_MATRIX_DESCRIPTION = joinParagraphs(
    'This is a learned linear transformation matrix. From first principles, multiplying a vector by a matrix means recombining its coordinates into a new set of coordinates.',
    'In the notation used in this visualization, a matrix with shape input_dim x output_dim takes an input vector of length input_dim and produces an output vector of length output_dim. So the input dimension tells you how many features the matrix reads, and the output dimension tells you how many new features it writes.',
    'Different matrices in the model learn different jobs: some build queries, keys, and values; some mix attention-head outputs; some expand or compress MLP features.',
    'In a transformer, the important pattern is usually that the same learned matrix is reused across token positions. So these objects typically produce a separate output vector for each token position in the sequence.'
);

function normalizeSelectionKind(kind) {
    const raw = String(kind || '').trim().toLowerCase();
    if (!raw) return '';
    if (raw === 'mergedkv') return 'merged key/value vector';
    if (raw === 'attentionsphere') return 'attention-matrix entry';
    if (raw === 'batchedvector') return 'batched vector';
    if (raw === 'instanced') return 'instanced scene element';
    if (raw === 'label') return 'scene component';
    return `${raw} component`;
}

function buildContextualFallbackDescription(label, kind, stage = '') {
    const cleanLabel = String(label || '').trim();
    const cleanStage = String(stage || '').trim();
    const kindLabel = normalizeSelectionKind(kind);

        if (cleanStage) {
            if (cleanLabel && kindLabel) {
                return joinParagraphs(
                    `"${cleanLabel}" is an interactive ${kindLabel} mapped to activation stage "${cleanStage}". That stage tells you where this object sits in the forward-pass pipeline.`,
                    'The important question is still the same: what role does this component play in updating the token state, and what other components read from it next?',
                    'Use the stage name, the equation panel, and the surrounding scene context together to place it within the residual, attention, or MLP flow.'
                );
            }
        if (cleanLabel) {
            return joinParagraphs(
                `"${cleanLabel}" is mapped to activation stage "${cleanStage}". That stage tells you where this object participates in the current forward-pass pipeline.`,
                'The best way to interpret it is to ask what vector or parameter it represents and which later operation consumes it.'
            );
        }
        if (kindLabel) {
            return joinParagraphs(
                `This ${kindLabel} is mapped to activation stage "${cleanStage}". That stage marks where it participates in the current forward-pass pipeline.`,
                'From there, you can read it as one piece of the transformer\'s sequence of embedding, normalization, attention, MLP, or output computations.'
            );
        }
        return joinParagraphs(
            `This selection is mapped to activation stage "${cleanStage}". That stage marks where it participates in the current forward-pass pipeline.`,
            'It should be interpreted through the same first-principles question: what information is it carrying or what transformation is it applying at this moment?'
        );
    }

    if (cleanLabel && kindLabel) {
        return joinParagraphs(
            `"${cleanLabel}" is an interactive ${kindLabel} in the scene.`,
            'It contributes somewhere in the residual, attention, MLP, or output flow, and its surrounding stage tells you which part of the transformer is using it.',
            'The surrounding stage, equation, and neighboring components tell you whether it is acting as a token state, a learned parameter, or a transformation.'
        );
    }
    if (cleanLabel) {
        return joinParagraphs(
            `"${cleanLabel}" is an interactive scene component in the current forward pass.`,
            'It contributes somewhere in the residual, attention, MLP, or output pipeline, and should be interpreted by looking at what stage of the layer flow is active around it.'
        );
    }
    if (kindLabel) {
        return joinParagraphs(
            `This ${kindLabel} contributes to the current transformer computation.`,
            'The key question is whether it is storing a token state, storing a learned parameter, or applying a transformation to another object nearby.'
        );
    }
    return joinParagraphs(
        'This interactive scene component participates in the model\'s forward pass at this point in the layer flow.',
        'To interpret it, locate whether it belongs to embeddings, LayerNorm, attention, the MLP, or the output head, then read it as a state, parameter, or transformation within that stage.'
    );
}

const SELECTION_EQUATION_BASE_COLOR = '#8f98a6';
const SELECTION_EQUATION_MUTED_SURFACE_COLOR = '#141417';
const SELECTION_EQUATION_MUTED_ALPHA = 0.42;
const toKatexColorHex = (hex) => `#${Number(hex).toString(16).padStart(6, '0')}`;
const colorizeEquationToken = (hex, token) => `\\textcolor{${hex}}{${token}}`;
function blendEquationColor(backgroundHex, foregroundHex, foregroundAlpha = 1) {
    const safeAlpha = Number.isFinite(foregroundAlpha)
        ? Math.max(0, Math.min(1, foregroundAlpha))
        : 1;
    const normalizeHex = (value, fallback) => {
        const source = typeof value === 'string' ? value.trim() : '';
        const matched = /^#?([0-9a-f]{6})$/i.exec(source);
        return matched ? matched[1] : fallback;
    };
    const bgHex = normalizeHex(backgroundHex, '141417');
    const fgHex = normalizeHex(foregroundHex, 'ffffff');
    const toRgb = (hex) => ({
        r: Number.parseInt(hex.slice(0, 2), 16),
        g: Number.parseInt(hex.slice(2, 4), 16),
        b: Number.parseInt(hex.slice(4, 6), 16)
    });
    const bg = toRgb(bgHex);
    const fg = toRgb(fgHex);
    const mix = (bgChannel, fgChannel) => Math.round((bgChannel * (1 - safeAlpha)) + (fgChannel * safeAlpha));
    const mixedHex = [mix(bg.r, fg.r), mix(bg.g, fg.g), mix(bg.b, fg.b)]
        .map((channel) => channel.toString(16).padStart(2, '0'))
        .join('');
    return `#${mixedHex}`;
}
const buildMutedEquationColor = (hex) => blendEquationColor(
    SELECTION_EQUATION_MUTED_SURFACE_COLOR,
    hex,
    SELECTION_EQUATION_MUTED_ALPHA
);
const colorizeHeadScopedEquationToken = (hex, token, headSubscript = 'i') => {
    const safeToken = typeof token === 'string' ? token.trim() : '';
    const safeSubscript = typeof headSubscript === 'string' && headSubscript.trim().length
        ? headSubscript.trim()
        : 'i';
    if (!safeToken) return '';
    return colorizeEquationToken(hex, `${safeToken}_{${safeSubscript}}`);
};
const SELECTION_EQUATION_COLORS = {
    base: SELECTION_EQUATION_BASE_COLOR,
    mutedWhite: buildMutedEquationColor('#ffffff'),
    q: toKatexColorHex(MHA_FINAL_Q_COLOR),
    qMuted: buildMutedEquationColor(toKatexColorHex(MHA_FINAL_Q_COLOR)),
    k: toKatexColorHex(MHA_FINAL_K_COLOR),
    kMuted: buildMutedEquationColor(toKatexColorHex(MHA_FINAL_K_COLOR)),
    v: toKatexColorHex(MHA_FINAL_V_COLOR),
    vMuted: buildMutedEquationColor(toKatexColorHex(MHA_FINAL_V_COLOR)),
    output: toKatexColorHex(MHA_OUTPUT_PROJECTION_MATRIX_COLOR),
    embeddingVocab: toKatexColorHex(MHA_FINAL_Q_COLOR),
    embeddingPos: toKatexColorHex(POSITION_EMBED_COLOR),
    mlpUp: toKatexColorHex(MLP_UP_MATRIX_COLOR),
    mlpDown: toKatexColorHex(MLP_DOWN_MATRIX_COLOR)
};
const ATTENTION_QUERY_PROJECTION_TEX = 'q_{t,i} = x_t W_Q';
const ATTENTION_KEY_PROJECTION_TEX = 'k_{t,i} = x_t W_K';
const ATTENTION_VALUE_PROJECTION_TEX = 'v_{t,i} = x_t W_V';
const ATTENTION_KEY_VECTOR_TEX = 'k_{j,i} = x_j W_K';
const ATTENTION_VALUE_VECTOR_TEX = 'v_{j,i} = x_j W_V';
const ATTENTION_QK_DOT_PRODUCT_TEX = 'q_{t,i} \\cdot k_{j,i}';
const ATTENTION_PRE_SCORE_TEX = '\\frac{q_{t,i} \\cdot k_{j,i}}{\\sqrt{d_h}}';
const ATTENTION_POST_WEIGHT_TEX = '\\alpha_{t,j} = \\frac{\\exp\\left(s_{t,j}\\right)}{\\sum_k \\exp\\left(s_{t,k}\\right)}';
const NORMALIZED_STREAM_SYMBOL = 'x_{\\text{ln}}';
const SELECTION_EQUATION_SYMBOLS = {
    Q: colorizeEquationToken(SELECTION_EQUATION_COLORS.q, 'Q'),
    K: colorizeEquationToken(SELECTION_EQUATION_COLORS.k, 'K'),
    V: colorizeEquationToken(SELECTION_EQUATION_COLORS.v, 'V'),
    HBright: colorizeEquationToken('#ffffff', 'H'),
    OBright: colorizeEquationToken('#ffffff', 'O'),
    HMuted: colorizeEquationToken(SELECTION_EQUATION_COLORS.mutedWhite, 'H'),
    OMuted: colorizeEquationToken(SELECTION_EQUATION_COLORS.mutedWhite, 'O'),
    QMuted: colorizeEquationToken(SELECTION_EQUATION_COLORS.qMuted, 'Q'),
    KMuted: colorizeEquationToken(SELECTION_EQUATION_COLORS.kMuted, 'K'),
    VMuted: colorizeEquationToken(SELECTION_EQUATION_COLORS.vMuted, 'V'),
    QHeadMuted: colorizeHeadScopedEquationToken(SELECTION_EQUATION_COLORS.qMuted, 'Q'),
    KHeadMuted: colorizeHeadScopedEquationToken(SELECTION_EQUATION_COLORS.kMuted, 'K'),
    VHeadMuted: colorizeHeadScopedEquationToken(SELECTION_EQUATION_COLORS.vMuted, 'V'),
    XLnBright: colorizeEquationToken('#ffffff', NORMALIZED_STREAM_SYMBOL),
    WQ: colorizeEquationToken(SELECTION_EQUATION_COLORS.q, 'W_Q'),
    WQHead: colorizeEquationToken(SELECTION_EQUATION_COLORS.q, 'W_{Q_i}'),
    WK: colorizeEquationToken(SELECTION_EQUATION_COLORS.k, 'W_K'),
    WKHead: colorizeEquationToken(SELECTION_EQUATION_COLORS.k, 'W_{K_i}'),
    WV: colorizeEquationToken(SELECTION_EQUATION_COLORS.v, 'W_V'),
    WVHead: colorizeEquationToken(SELECTION_EQUATION_COLORS.v, 'W_{V_i}'),
    BQ: colorizeEquationToken(SELECTION_EQUATION_COLORS.q, 'b_Q'),
    BQHead: colorizeEquationToken(SELECTION_EQUATION_COLORS.q, 'b_{Q_i}'),
    BK: colorizeEquationToken(SELECTION_EQUATION_COLORS.k, 'b_K'),
    BKHead: colorizeEquationToken(SELECTION_EQUATION_COLORS.k, 'b_{K_i}'),
    BV: colorizeEquationToken(SELECTION_EQUATION_COLORS.v, 'b_V'),
    BVHead: colorizeEquationToken(SELECTION_EQUATION_COLORS.v, 'b_{V_i}'),
    WO: colorizeEquationToken(SELECTION_EQUATION_COLORS.output, 'W_O'),
    BO: colorizeEquationToken(SELECTION_EQUATION_COLORS.output, 'b_O'),
    XTok: 'x_t^{\\text{tok}}',
    XPos: 'x_t^{\\text{pos}}',
    E: colorizeEquationToken(SELECTION_EQUATION_COLORS.embeddingVocab, 'E'),
    P: colorizeEquationToken(SELECTION_EQUATION_COLORS.embeddingPos, 'P'),
    WU: colorizeEquationToken(SELECTION_EQUATION_COLORS.embeddingVocab, 'W_U'),
    WUp: colorizeEquationToken(SELECTION_EQUATION_COLORS.mlpUp, 'W_{\\text{up}}'),
    BUp: colorizeEquationToken(SELECTION_EQUATION_COLORS.mlpUp, 'b_{\\text{up}}'),
    WDown: colorizeEquationToken(SELECTION_EQUATION_COLORS.mlpDown, 'W_{\\text{down}}'),
    BDown: colorizeEquationToken(SELECTION_EQUATION_COLORS.mlpDown, 'b_{\\text{down}}'),
    MLPDown: `${colorizeEquationToken(SELECTION_EQUATION_COLORS.mlpDown, '\\mathrm{MLP}')}\\left(${NORMALIZED_STREAM_SYMBOL}\\right)`
};

function buildEquationEntries(lines, activeIndexes = []) {
    const nonEmpty = Array.isArray(lines)
        ? lines.filter((line) => typeof line === 'string' && line.trim().length > 0)
        : [];
    if (!nonEmpty.length) return [];
    const rawIndexes = Array.isArray(activeIndexes) ? activeIndexes : [activeIndexes];
    const activeSet = new Set();
    rawIndexes.forEach((value) => {
        if (!Number.isFinite(value)) return;
        activeSet.add(Math.max(0, Math.floor(value)));
    });
    const highlightAll = activeSet.size === 0;
    return nonEmpty.map((tex, index) => ({
        tex,
        active: highlightAll ? true : activeSet.has(index)
    }));
}

function formatEquationBlock(lines) {
    if (!Array.isArray(lines) || lines.length === 0) return '';
    const nonEmpty = lines.filter((line) => typeof line === 'string' && line.trim().length > 0);
    if (!nonEmpty.length) return '';
    return nonEmpty.map((line) => `$$${line}$$`).join('\n');
}

function buildNlpEmbeddingEquationEntries(activeIndexes = [0, 1, 2]) {
    return buildEquationEntries([
        `${SELECTION_EQUATION_SYMBOLS.XTok} = ${SELECTION_EQUATION_SYMBOLS.E}[\\mathrm{token}_t]`,
        `${SELECTION_EQUATION_SYMBOLS.XPos} = ${SELECTION_EQUATION_SYMBOLS.P}[t]`,
        `x_t = ${SELECTION_EQUATION_SYMBOLS.XTok} + ${SELECTION_EQUATION_SYMBOLS.XPos}`
    ], activeIndexes);
}

function hasVocabEmbeddingLabel(lower) {
    return lower.includes('vocab embedding') || lower.includes('vocabulary embedding');
}

function hasTopVocabEmbeddingLabel(lower) {
    return lower.includes('vocab embedding (top)')
        || lower.includes('vocabulary embedding (top)')
        || lower.includes('vocab unembedding')
        || lower.includes('vocabulary unembedding');
}

function resolveLayerNormEquationSymbols(lower) {
    if (lower.includes('top')) {
        return {
            input: 'x',
            norm: '\\hat{x}_{\\text{out}}',
            output: 'x_{\\text{final}}'
        };
    }
    if (lower.includes('ln2')) {
        return {
            input: 'x',
            norm: '\\hat{u}',
            output: NORMALIZED_STREAM_SYMBOL
        };
    }
    return {
        input: 'x',
        norm: '\\hat{x}',
        output: NORMALIZED_STREAM_SYMBOL
    };
}

function resolveLayerNormEquationHighlight(lower, stageLower = '') {
    if (isPostLayerNormResidualSelection({ label: lower, stage: stageLower })) {
        return 'output';
    }
    if (
        lower.includes('scale')
        || lower.includes('gamma')
        || stageLower.endsWith('.scale')
        || stageLower.endsWith('.param.scale')
    ) {
        return 'scale';
    }
    if (
        lower.includes('shift')
        || lower.includes('beta')
        || stageLower.endsWith('.param.shift')
    ) {
        return 'shift';
    }
    if (isLayerNormOutputStage(stageLower)) {
        return 'output';
    }
    return 'norm';
}

function resolveAttentionHeadSubscript(selectionInfo) {
    const headIndex = resolveSelectionHeadIndex(selectionInfo);
    return Number.isFinite(headIndex) ? String(headIndex + 1) : null;
}

function buildSelectionEquationEntries(label, selectionInfo = null) {
    const lower = String(label || '').toLowerCase();
    const stageLower = String(getActivationDataFromSelection(selectionInfo)?.stage || '').toLowerCase();
    const attentionEquations = buildAttentionEquationSet({
        Q: SELECTION_EQUATION_SYMBOLS.Q,
        K: SELECTION_EQUATION_SYMBOLS.K,
        V: SELECTION_EQUATION_SYMBOLS.V,
        WQ: SELECTION_EQUATION_SYMBOLS.WQHead,
        WK: SELECTION_EQUATION_SYMBOLS.WKHead,
        WV: SELECTION_EQUATION_SYMBOLS.WVHead,
        BQ: SELECTION_EQUATION_SYMBOLS.BQHead,
        BK: SELECTION_EQUATION_SYMBOLS.BKHead,
        BV: SELECTION_EQUATION_SYMBOLS.BVHead,
        WO: SELECTION_EQUATION_SYMBOLS.WO,
        BO: SELECTION_EQUATION_SYMBOLS.BO
    }, {
        headSubscript: resolveAttentionHeadSubscript(selectionInfo)
    });
    const tokenEmbedEq = `${SELECTION_EQUATION_SYMBOLS.XTok} = ${SELECTION_EQUATION_SYMBOLS.E}[\\mathrm{token}_t]`;
    const posEmbedEq = `${SELECTION_EQUATION_SYMBOLS.XPos} = ${SELECTION_EQUATION_SYMBOLS.P}[t]`;
    const embedSumEq = `x_t = ${SELECTION_EQUATION_SYMBOLS.XTok} + ${SELECTION_EQUATION_SYMBOLS.XPos}`;
    const queryProjectionEq = attentionEquations.queryProjection;
    const keyProjectionEq = attentionEquations.keyProjection;
    const valueProjectionEq = attentionEquations.valueProjection;
    const queryWeightMatrixEq = buildAttentionProjectionEquation({
        outputSymbol: SELECTION_EQUATION_SYMBOLS.QHeadMuted,
        inputSymbol: SELECTION_EQUATION_SYMBOLS.XLnBright,
        weightSymbol: SELECTION_EQUATION_SYMBOLS.WQHead,
        biasSymbol: SELECTION_EQUATION_SYMBOLS.BQHead
    });
    const keyWeightMatrixEq = buildAttentionProjectionEquation({
        outputSymbol: SELECTION_EQUATION_SYMBOLS.KHeadMuted,
        inputSymbol: SELECTION_EQUATION_SYMBOLS.XLnBright,
        weightSymbol: SELECTION_EQUATION_SYMBOLS.WKHead,
        biasSymbol: SELECTION_EQUATION_SYMBOLS.BKHead
    });
    const valueWeightMatrixEq = buildAttentionProjectionEquation({
        outputSymbol: SELECTION_EQUATION_SYMBOLS.VHeadMuted,
        inputSymbol: SELECTION_EQUATION_SYMBOLS.XLnBright,
        weightSymbol: SELECTION_EQUATION_SYMBOLS.WVHead,
        biasSymbol: SELECTION_EQUATION_SYMBOLS.BVHead
    });
    const outputProjectionWeightMatrixEq = buildAttentionProjectionEquation({
        outputSymbol: SELECTION_EQUATION_SYMBOLS.OBright,
        inputSymbol: SELECTION_EQUATION_SYMBOLS.HBright,
        weightSymbol: SELECTION_EQUATION_SYMBOLS.WO,
        biasSymbol: SELECTION_EQUATION_SYMBOLS.BO
    });
    const attentionEquation = attentionEquations.attention;
    const concatEq = attentionEquations.concat;
    const outputProjectionEq = attentionEquations.outputProjection;
    const postAttentionResidualEq = attentionEquations.postAttentionResidual;
    const mlpUpEq = `a = ${NORMALIZED_STREAM_SYMBOL} ${SELECTION_EQUATION_SYMBOLS.WUp} + ${SELECTION_EQUATION_SYMBOLS.BUp}`;
    const mlpGeluEq = 'z = \\mathrm{GELU}(a)';
    const mlpDownEq = `\\mathrm{MLP}(${NORMALIZED_STREAM_SYMBOL}) = z ${SELECTION_EQUATION_SYMBOLS.WDown} + ${SELECTION_EQUATION_SYMBOLS.BDown}`;
    const postMlpResidualEq = `x_{\\text{out}} = u + ${SELECTION_EQUATION_SYMBOLS.MLPDown}`;
    const logitsEq = `\\ell = x_{\\text{final}} ${SELECTION_EQUATION_SYMBOLS.WU}`;
    const probsEq = 'p = \\mathrm{softmax}(\\ell)';
    const kvWriteEq = 'k_t = x_t W_K,\\quad v_t = x_t W_V';
    const kvAppendEq = 'K_{1:t} = [K_{1:t-1}; k_t],\\quad V_{1:t} = [V_{1:t-1}; v_t]';
    const kvReuseEq = 'o_t = \\mathrm{softmax}\\!\\left(\\frac{q_t K_{1:t}^{\\top}}{\\sqrt{d_h}}\\right) V_{1:t}';
    const buildLayerNormEntries = (kind = 'ln1', highlight = 'norm') => {
        const layerNormKey = kind === 'ln2'
            ? 'ln2'
            : (kind === 'top' ? 'top' : 'ln1');
        const symbols = resolveLayerNormEquationSymbols(layerNormKey);
        if (highlight === 'scale' || highlight === 'shift' || highlight === 'output') {
            return [{
                tex: buildSelectionLayerNormEquation({
                    inputSymbol: symbols.input,
                    outputSymbol: symbols.output,
                    highlight
                }),
                active: true
            }];
        }
        const combinedEq = `${symbols.output} = \\gamma \\odot \\frac{${symbols.input} - \\mu}{\\sqrt{\\sigma^2 + \\epsilon}} + \\beta`;
        return buildEquationEntries([combinedEq], [0]);
    };

    if (isMhsaInfoSelection(label, selectionInfo)) {
        return buildEquationEntries([attentionEquation, concatEq, outputProjectionEq], [0, 1, 2]);
    }

    if (isKvCacheInfoSelection(label, selectionInfo)) {
        const phase = resolveKvCacheInfoPhase(selectionInfo);
        return phase === 'decode'
            ? buildEquationEntries([kvAppendEq, kvReuseEq], [1])
            : buildEquationEntries([kvWriteEq, kvAppendEq, kvReuseEq], [0, 1]);
    }

    if (stageLower.startsWith('embedding.token')) {
        return buildEquationEntries([tokenEmbedEq, embedSumEq], [0]);
    }
    if (stageLower.startsWith('embedding.position')) {
        return buildEquationEntries([posEmbedEq, embedSumEq], [0]);
    }
    if (stageLower.startsWith('embedding.sum')) {
        return buildNlpEmbeddingEquationEntries([2]);
    }
    if (stageLower.startsWith('layer.incoming')) {
        return buildLayerNormEntries('ln1', 'norm');
    }
    if (stageLower === 'ln1.norm') {
        return buildLayerNormEntries('ln1', 'norm');
    }
    if (stageLower === 'ln1.scale' || stageLower === 'ln1.output' || stageLower === 'ln1.shift' || stageLower === 'ln1.param.scale' || stageLower === 'ln1.param.shift') {
        return buildLayerNormEntries('ln1', resolveLayerNormEquationHighlight(lower, stageLower));
    }
    if (stageLower === 'ln2.norm') {
        return buildLayerNormEntries('ln2', 'norm');
    }
    if (stageLower === 'ln2.scale' || stageLower === 'ln2.output' || stageLower === 'ln2.shift' || stageLower === 'ln2.param.scale' || stageLower === 'ln2.param.shift') {
        return buildLayerNormEntries('ln2', resolveLayerNormEquationHighlight(lower, stageLower));
    }
    if (stageLower === 'final_ln.norm') {
        return buildLayerNormEntries('top', 'norm');
    }
    if (
        stageLower === 'final_ln.scale'
        || stageLower === 'final_ln.output'
        || stageLower === 'final_ln.shift'
        || stageLower === 'final_ln.param.scale'
        || stageLower === 'final_ln.param.shift'
    ) {
        return buildLayerNormEntries('top', resolveLayerNormEquationHighlight(lower, stageLower));
    }
    if (stageLower === 'qkv.q') {
        return buildEquationEntries([queryProjectionEq, attentionEquation], [0]);
    }
    if (stageLower === 'qkv.q.bias') {
        return buildEquationEntries([queryProjectionEq, attentionEquation], [0]);
    }
    if (stageLower === 'qkv.k') {
        return buildEquationEntries([keyProjectionEq, attentionEquation], [0]);
    }
    if (stageLower === 'qkv.k.bias') {
        return buildEquationEntries([keyProjectionEq, attentionEquation], [0]);
    }
    if (stageLower === 'qkv.v') {
        return buildEquationEntries([valueProjectionEq, attentionEquation], [0]);
    }
    if (stageLower === 'qkv.v.bias') {
        return buildEquationEntries([valueProjectionEq, attentionEquation], [0]);
    }
    if (stageLower === 'attention.pre') {
        return buildEquationEntries([attentionEquation, concatEq], [0]);
    }
    if (stageLower === 'attention.mask') {
        return buildEquationEntries([attentionEquation, concatEq], [0]);
    }
    if (stageLower === 'attention.post') {
        return buildEquationEntries([attentionEquation, concatEq], [0]);
    }
    if (stageLower === 'attention.weighted_value') {
        return buildEquationEntries([attentionEquation, concatEq], [0]);
    }
    if (stageLower === 'attention.output_projection.bias') {
        return buildEquationEntries([concatEq, outputProjectionEq, postAttentionResidualEq], [1]);
    }
    if (stageLower === 'attention.output_projection') {
        return buildEquationEntries([concatEq, outputProjectionEq, postAttentionResidualEq], [1]);
    }
    if (stageLower === 'residual.post_attention') {
        return buildEquationEntries([outputProjectionEq, postAttentionResidualEq], [1]);
    }
    if (stageLower === 'mlp.up') {
        return buildEquationEntries([mlpUpEq, mlpGeluEq, mlpDownEq], [0]);
    }
    if (stageLower === 'mlp.up.bias') {
        return buildEquationEntries([mlpUpEq, mlpGeluEq, mlpDownEq], [0]);
    }
    if (stageLower === 'mlp.activation') {
        return buildEquationEntries([mlpUpEq, mlpGeluEq, mlpDownEq], [1]);
    }
    if (stageLower === 'mlp.down') {
        return buildEquationEntries([mlpGeluEq, mlpDownEq, postMlpResidualEq], [1]);
    }
    if (stageLower === 'mlp.down.bias') {
        return buildEquationEntries([mlpGeluEq, mlpDownEq, postMlpResidualEq], [1]);
    }
    if (stageLower === 'residual.post_mlp') {
        return buildEquationEntries([mlpDownEq, postMlpResidualEq], [1]);
    }

    if (lower.startsWith('token:')) {
        return buildEquationEntries([tokenEmbedEq, embedSumEq], [0]);
    }
    if (lower.startsWith('position:')) {
        return buildEquationEntries([posEmbedEq, embedSumEq], [0]);
    }
    if (lower.includes('embedding sum')) {
        return buildNlpEmbeddingEquationEntries([2]);
    }
    if (lower.includes('token embedding')) {
        return buildEquationEntries([tokenEmbedEq, embedSumEq], [0]);
    }
    if (hasTopVocabEmbeddingLabel(lower) || lower.includes('unembedding')) {
        return buildEquationEntries([logitsEq, probsEq], [0]);
    }
    if (hasVocabEmbeddingLabel(lower)) {
        return buildEquationEntries([tokenEmbedEq], [0]);
    }
    if (lower.includes('position embedding') || lower.includes('positional embedding')) {
        return buildEquationEntries([posEmbedEq, embedSumEq], [0]);
    }
    if (lower.includes('query weight matrix')) {
        return buildEquationEntries([queryWeightMatrixEq, attentionEquation], [0]);
    }
    if (lower.includes('query vector')) {
        return buildEquationEntries([queryProjectionEq, attentionEquation], [0]);
    }
    if (lower.includes('query bias vector')) {
        return buildEquationEntries([queryProjectionEq, attentionEquation], [0]);
    }
    if (lower.includes('key weight matrix')) {
        return buildEquationEntries([keyWeightMatrixEq, attentionEquation], [0]);
    }
    if (lower.includes('key bias vector')) {
        return buildEquationEntries([keyProjectionEq, attentionEquation], [0]);
    }
    if (lower.includes('cached key vector') || lower.includes('key vector')) {
        return buildEquationEntries([keyProjectionEq, attentionEquation], [0]);
    }
    if (lower.includes('weighted value vector')) {
        return buildEquationEntries([attentionEquation, concatEq], [0]);
    }
    if (lower.includes('value weight matrix')) {
        return buildEquationEntries([valueWeightMatrixEq, attentionEquation], [0]);
    }
    if (lower.includes('value bias vector')) {
        return buildEquationEntries([valueProjectionEq, attentionEquation], [0]);
    }
    if (lower.includes('cached value vector') || lower.includes('value vector')) {
        return buildEquationEntries([valueProjectionEq, attentionEquation], [0]);
    }
    if (lower.includes('attention score')) {
        return buildEquationEntries([attentionEquation, concatEq], [0]);
    }
    if (lower.includes('attention weighted sum')) {
        return buildEquationEntries([attentionEquation, concatEq, outputProjectionEq], [0]);
    }
    if (lower.includes('output projection bias vector')) {
        return buildEquationEntries([concatEq, outputProjectionEq, postAttentionResidualEq], [1]);
    }
    if (lower.includes('output projection matrix')) {
        return buildEquationEntries([concatEq, outputProjectionWeightMatrixEq, postAttentionResidualEq], [1]);
    }
    if (lower.includes('post-attention residual')) {
        return buildEquationEntries([outputProjectionEq, postAttentionResidualEq], [1]);
    }
    if (lower.includes('mlp up weight matrix') || lower.includes('mlp up projection')) {
        return buildEquationEntries([mlpUpEq, mlpGeluEq, mlpDownEq], [0]);
    }
    if (lower === MLP_UP_BIAS_TOOLTIP_LABEL.toLowerCase()) {
        return buildEquationEntries([mlpUpEq, mlpGeluEq, mlpDownEq], [0]);
    }
    if (lower.includes('mlp expanded segments')) {
        return buildEquationEntries([mlpUpEq, mlpGeluEq, mlpDownEq], [1]);
    }
    if (lower.includes('mlp down weight matrix') || lower.includes('mlp down projection')) {
        return buildEquationEntries([mlpGeluEq, mlpDownEq, postMlpResidualEq], [1]);
    }
    if (lower === MLP_DOWN_BIAS_TOOLTIP_LABEL.toLowerCase()) {
        return buildEquationEntries([mlpGeluEq, mlpDownEq, postMlpResidualEq], [1]);
    }
    if (lower.includes('post-mlp residual')) {
        return buildEquationEntries([mlpDownEq, postMlpResidualEq], [1]);
    }
    if (lower.includes('top logit bars') || lower === 'logit' || lower.startsWith('logit')) {
        return buildEquationEntries([logitsEq, probsEq], [0]);
    }
    if (lower.startsWith('chosen token:')) {
        return buildEquationEntries([logitsEq, probsEq], [1]);
    }

    const isLayerNormSelection = lower.includes('layernorm')
        || lower.includes('layer norm')
        || lower.includes('ln1')
        || lower.includes('ln2')
        || lower.includes('final ln')
        || isPostLayerNormResidualSelection({ label, stage: stageLower });
    if (isLayerNormSelection) {
        const layerNormKind = lower.includes('top') || lower.includes('final ln')
            ? 'top'
            : (resolveSelectionLayerNormKind(selectionInfo) === 'ln2' || lower.includes('ln2') ? 'ln2' : 'ln1');
        return buildLayerNormEntries(layerNormKind, resolveLayerNormEquationHighlight(lower, stageLower));
    }

    if (lower.includes('causal mask') || lower.includes('attention mask')) {
        return buildEquationEntries([attentionEquation, concatEq], [0]);
    }
    if (lower.includes('attention')) {
        return buildEquationEntries([attentionEquation, concatEq, outputProjectionEq], [0]);
    }
    if (lower.includes('mlp')) {
        return buildEquationEntries([mlpUpEq, mlpGeluEq, mlpDownEq], [0, 1, 2]);
    }
    if (lower.includes('weight matrix')) {
        return buildEquationEntries(['y = xW'], [0]);
    }

    return [];
}

export function resolveSelectionPreviewEquations(label, selectionInfo = null) {
    return buildSelectionEquationEntries(label, selectionInfo);
}

export function resolveSelectionEquations(label, selectionInfo = null) {
    return formatEquationBlock(buildSelectionEquationEntries(label, selectionInfo).map((entry) => entry.tex));
}

export function resolveDescription(label, kind = null, selectionInfo = null) {
    const lower = (label || '').toLowerCase();
    const activation = getActivationDataFromSelection(selectionInfo);
    const stage = activation?.stage || '';
    const stageLower = stage.toLowerCase();

    if (isMhsaInfoSelection(label, selectionInfo)) {
        return MHSA_INFO_DESCRIPTION;
    }

    if (isKvCacheInfoSelection(label, selectionInfo)) {
        return buildKvCacheInfoDescription(selectionInfo);
    }

    if (stageLower) {
        if (stageLower.startsWith('embedding.token')) {
            return TOKEN_EMBEDDING_VECTOR_DESCRIPTION;
        }
        if (stageLower.startsWith('embedding.position')) {
            return POSITION_EMBEDDING_VECTOR_DESCRIPTION;
        }
        if (stageLower.startsWith('embedding.sum')) {
            return EMBEDDING_SUM_DESCRIPTION;
        }
        if (stageLower.startsWith('layer.incoming')) {
            return buildIncomingResidualDescription(selectionInfo);
        }
        if (stageLower === 'ln1.norm' || stageLower === 'ln2.norm') {
            return buildLayerNormNormalizedVectorDescription(selectionInfo);
        }
        if (stageLower === 'ln1.scale' || stageLower === 'ln2.scale') {
            return buildLayerNormScaledVectorDescription(selectionInfo);
        }
        if (stageLower === 'ln1.output' || stageLower === 'ln1.shift') {
            return buildLn1OutputDescription(selectionInfo);
        }
        if (stageLower === 'ln2.output' || stageLower === 'ln2.shift') {
            return buildLn2OutputDescription(selectionInfo);
        }
        if (stageLower === 'qkv.q') {
            return buildQueryVectorDescription(selectionInfo);
        }
        if (stageLower === 'qkv.q.bias') {
            return buildQueryBiasVectorDescription(selectionInfo);
        }
        if (stageLower === 'qkv.k') {
            return buildKeyVectorDescription(selectionInfo);
        }
        if (stageLower === 'qkv.k.bias') {
            return buildKeyBiasVectorDescription(selectionInfo);
        }
        if (stageLower === 'qkv.v') {
            return buildValueVectorDescription(selectionInfo);
        }
        if (stageLower === 'qkv.v.bias') {
            return buildValueBiasVectorDescription(selectionInfo);
        }
        if (stageLower === 'attention.mask') {
            return buildCausalMaskDescription(selectionInfo);
        }
        if (stageLower === 'attention.weighted_value') {
            return buildWeightedValueVectorDescription(selectionInfo);
        }
        if (stageLower === 'attention.output_projection.bias') {
            return OUTPUT_PROJECTION_BIAS_DESCRIPTION;
        }
        if (stageLower === 'attention.output_projection') {
            return POST_ATTENTION_OUTPUT_DESCRIPTION;
        }
        if (stageLower === 'residual.post_attention') {
            return buildPostAttentionResidualDescription(selectionInfo);
        }
        if (stageLower === 'mlp.up') {
            return MLP_UP_VECTOR_DESCRIPTION;
        }
        if (stageLower === 'mlp.up.bias') {
            return MLP_UP_BIAS_DESCRIPTION;
        }
        if (stageLower === 'mlp.activation') {
            return MLP_ACTIVATION_DESCRIPTION;
        }
        if (stageLower === 'mlp.down') {
            return MLP_DOWN_VECTOR_DESCRIPTION;
        }
        if (stageLower === 'mlp.down.bias') {
            return MLP_DOWN_BIAS_DESCRIPTION;
        }
        if (stageLower === 'residual.post_mlp') {
            return buildPostMlpResidualDescription(selectionInfo);
        }
        if (stageLower === 'final_ln.norm') {
            return buildFinalLayerNormNormalizedVectorDescription(selectionInfo);
        }
        if (stageLower === 'final_ln.scale') {
            return FINAL_LN_SCALE_DESCRIPTION;
        }
        if (stageLower === 'final_ln.output' || stageLower === 'final_ln.shift') {
            return FINAL_LN_SHIFT_DESCRIPTION;
        }
    }

    if (
        lower.includes('residual stream vector')
        || stageLower.includes('residual')
    ) {
        return buildResidualStreamDescription(selectionInfo);
    }

    if (lower.startsWith('token:')) {
        return TOKEN_CHIP_DESCRIPTION;
    }
    if (lower.startsWith('position:')) {
        return POSITION_CHIP_DESCRIPTION;
    }
    if (lower.includes('token embedding')) {
        return TOKEN_EMBEDDING_VECTOR_DESCRIPTION;
    }
    if (lower.includes('position embedding')) {
        return POSITION_EMBEDDING_VECTOR_DESCRIPTION;
    }
    if (lower.includes('embedding sum')) {
        return EMBEDDING_SUM_DESCRIPTION;
    }
    if (lower.includes('embedding connector trail')) {
        return EMBEDDING_CONNECTOR_TRAIL_DESCRIPTION;
    }
    if (hasTopVocabEmbeddingLabel(lower)) {
        return UNEMBEDDING_MATRIX_DESCRIPTION;
    }
    if (hasVocabEmbeddingLabel(lower)) {
        return VOCAB_EMBEDDING_MATRIX_DESCRIPTION;
    }
    if (lower.includes('positional embedding')) {
        return POSITIONAL_EMBEDDING_MATRIX_DESCRIPTION;
    }
    if ((lower.includes('ln1') || lower.includes('ln2')) && (lower.includes('scale') || lower.includes('gamma'))) {
        return LN_SCALE_PARAMETER_DESCRIPTION;
    }
    if ((lower.includes('ln1') || lower.includes('ln2')) && (lower.includes('normed') || lower.includes('normalized'))) {
        return buildLayerNormNormalizedVectorDescription(selectionInfo);
    }
    if ((lower.includes('ln1') || lower.includes('ln2')) && (lower.includes('shift') || lower.includes('beta'))) {
        return LN_SHIFT_PARAMETER_DESCRIPTION;
    }
    if (lower.includes('final ln scale')) {
        return FINAL_LN_SCALE_DESCRIPTION;
    }
    if (lower.includes('final ln shift')) {
        return FINAL_LN_SHIFT_DESCRIPTION;
    }
    if (lower.includes('query weight matrix')) {
        return buildQueryWeightMatrixDescription(selectionInfo);
    }
    if (lower.includes('query bias vector')) {
        return buildQueryBiasVectorDescription(selectionInfo);
    }
    if (lower.includes('key weight matrix')) {
        return buildKeyWeightMatrixDescription(selectionInfo);
    }
    if (lower.includes('key bias vector')) {
        return buildKeyBiasVectorDescription(selectionInfo);
    }
    if (lower.includes('value weight matrix')) {
        return buildValueWeightMatrixDescription(selectionInfo);
    }
    if (lower.includes('value bias vector')) {
        return buildValueBiasVectorDescription(selectionInfo);
    }
    if (lower.includes('output projection bias vector')) {
        return OUTPUT_PROJECTION_BIAS_DESCRIPTION;
    }
    if (lower.includes('output projection matrix')) {
        return OUTPUT_PROJECTION_MATRIX_DESCRIPTION;
    }
    if (lower.includes('mlp up weight matrix')) {
        return MLP_UP_WEIGHT_MATRIX_DESCRIPTION;
    }
    if (lower === MLP_UP_BIAS_TOOLTIP_LABEL.toLowerCase()) {
        return MLP_UP_BIAS_DESCRIPTION;
    }
    if (lower.includes('mlp down weight matrix')) {
        return MLP_DOWN_WEIGHT_MATRIX_DESCRIPTION;
    }
    if (lower === MLP_DOWN_BIAS_TOOLTIP_LABEL.toLowerCase()) {
        return MLP_DOWN_BIAS_DESCRIPTION;
    }
    if (lower.includes('mlp up projection')) {
        return MLP_UP_VECTOR_DESCRIPTION;
    }
    if (lower.includes('mlp down projection')) {
        return MLP_DOWN_VECTOR_DESCRIPTION;
    }
    if (lower.includes('mlp expanded segments')) {
        return MLP_EXPANDED_SEGMENTS_DESCRIPTION;
    }
    const activationStage = String(getActivationDataFromSelection(selectionInfo)?.stage || '').toLowerCase();
    const isPostLayerNormResidual = isPostLayerNormResidualSelection({
        label,
        stage: activationStage
    });
    if ((lower.includes('layernorm') || lower.includes('layer norm')) && !isPostLayerNormResidual) {
        if (lower.includes('top')) {
            return buildTopLayerNormDescription(selectionInfo);
        }
        if (lower.includes('scale') || lower.includes('gamma')) {
            return LN_SCALE_PARAMETER_DESCRIPTION;
        }
        if (lower.includes('shift') || lower.includes('beta')) {
            return LN_SHIFT_PARAMETER_DESCRIPTION;
        }
        if (lower.includes('normed') || lower.includes('normalized')) {
            return buildLayerNormNormalizedVectorDescription(selectionInfo);
        }
        return buildLayerNormDescription(selectionInfo);
    }
    if (lower.includes('merged key vectors')) {
        return MERGED_KEY_VECTORS_DESCRIPTION;
    }
    if (lower.includes('merged value vectors')) {
        return MERGED_VALUE_VECTORS_DESCRIPTION;
    }
    if (isPostLayerNormResidual) {
        if (resolveSelectionLayerNormKind(selectionInfo) === 'ln2') {
            return buildLn2OutputDescription(selectionInfo);
        }
        return buildLn1OutputDescription(selectionInfo);
    }
    if (lower.includes('incoming residual')) {
        return buildIncomingResidualDescription(selectionInfo);
    }
    if (lower.includes('post-attention residual')) {
        return buildPostAttentionResidualDescription(selectionInfo);
    }
    if (lower.includes('post-mlp residual')) {
        return buildPostMlpResidualDescription(selectionInfo);
    }
    if (lower.includes('mhsa q copies')) {
        return Q_COPIES_DESCRIPTION;
    }
    if (lower.includes('mhsa k copies')) {
        return K_COPIES_DESCRIPTION;
    }
    if (lower.includes('mhsa v copies')) {
        return V_COPIES_DESCRIPTION;
    }
    if (lower.includes('query vector')) {
        return buildQueryVectorDescription(selectionInfo);
    }
    if (lower.includes('cached key vector')) {
        return buildCachedKeyVectorDescription(selectionInfo);
    }
    if (lower.includes('key vector')) {
        return buildKeyVectorDescription(selectionInfo);
    }
    if (lower.includes('cached value vector')) {
        return buildCachedValueVectorDescription(selectionInfo);
    }
    if (lower.includes('weighted value vector')) {
        return buildWeightedValueVectorDescription(selectionInfo);
    }
    if (lower.includes('value vector')) {
        return buildValueVectorDescription(selectionInfo);
    }
    if (lower.includes('attention weighted sum') || stageLower === 'attention.weighted_sum') {
        return buildAttentionWeightedSumDescription(selectionInfo);
    }
    if (lower.includes('causal mask') || lower.includes('attention mask')) {
        return buildCausalMaskDescription(selectionInfo);
    }
    if (lower.includes('attention score') || stageLower.startsWith('attention.')) {
        if (stageLower === 'attention.pre') {
            return buildAttentionPreScoreDescription(selectionInfo);
        }
        if (stageLower === 'attention.mask') {
            return buildCausalMaskDescription(selectionInfo);
        }
        if (stageLower === 'attention.post') {
            return buildAttentionPostScoreDescription(selectionInfo);
        }
        return buildAttentionScoreGenericDescription(selectionInfo);
    }
    if (lower.includes('attention')) {
        return ATTENTION_GENERIC_DESCRIPTION;
    }
    if (lower.includes('top logit bars')) {
        return buildTopLogitBarsDescription(selectionInfo);
    }
    if (lower.startsWith('chosen token:')) {
        return buildChosenTokenDescription(selectionInfo);
    }
    if (lower === 'logit' || lower.startsWith('logit')) {
        return buildLogitDescription(selectionInfo);
    }
    if (lower.includes('residual')) {
        return buildResidualStreamDescription(selectionInfo);
    }
    if (lower.includes('mlp')) {
        return MLP_GENERIC_DESCRIPTION;
    }
    if (lower.includes('vector')) {
        return GENERIC_VECTOR_DESCRIPTION;
    }
    if (lower.includes('weight matrix')) {
        return GENERIC_WEIGHT_MATRIX_DESCRIPTION;
    }
    return buildContextualFallbackDescription(label, kind, stage);
}
