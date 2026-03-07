import visualizationDescriptionMarkdown from '../../vizualization_description.md?raw';
import {
    ATTENTION_SCORE_DECIMALS,
    ATTENTION_VALUE_PLACEHOLDER
} from './selectionPanelConstants.js';
import { formatTokenLabelForPreview } from './selectionPanelFormatUtils.js';
import {
    findUserDataNumber,
    findUserDataString,
    getActivationDataFromSelection,
    isAttentionScoreSelection,
    isKvCacheVectorSelection,
    isLogitBarSelection,
    isResidualVectorSelection,
    isSelfAttentionSelection,
    isWeightedSumSelection,
    normalizeSelectionLabel,
    resolveAttentionModeFromSelection
} from './selectionPanelSelectionUtils.js';

function joinSections(...sections) {
    return sections
        .filter((section) => typeof section === 'string' && section.trim().length > 0)
        .join('\n\n')
        .replace(/\n{3,}/g, '\n\n')
        .trim();
}

function buildMarkdownSection(title, body) {
    const text = typeof body === 'string' ? body.trim() : '';
    if (!text) return '';
    return `## ${title}\n${text}`;
}

function buildMarkdownBulletSection(title, lines = []) {
    const items = Array.isArray(lines)
        ? lines.filter((line) => typeof line === 'string' && line.trim().length > 0)
        : [];
    if (!items.length) return '';
    return `## ${title}\n${items.map((line) => `- ${line.trim()}`).join('\n')}`;
}

function quoteTokenLabel(label = '') {
    const safeLabel = typeof label === 'string' ? label.trim() : '';
    return safeLabel ? JSON.stringify(safeLabel) : '';
}

function resolveTokenContext(selection, activationSource, {
    indexKey = 'tokenIndex',
    labelKey = 'tokenLabel',
    idKey = 'tokenId',
    fallback = null
} = {}) {
    let tokenIndex = findUserDataNumber(selection, indexKey);
    if (!Number.isFinite(tokenIndex) && Number.isFinite(fallback?.tokenIndex)) {
        tokenIndex = Math.floor(fallback.tokenIndex);
    }

    let tokenLabel = findUserDataString(selection, labelKey);
    if ((typeof tokenLabel !== 'string' || !tokenLabel.trim().length) && typeof fallback?.tokenLabel === 'string') {
        tokenLabel = fallback.tokenLabel;
    }
    if ((typeof tokenLabel !== 'string' || !tokenLabel.trim().length)
        && Number.isFinite(tokenIndex)
        && activationSource
        && typeof activationSource.getTokenString === 'function') {
        const resolvedLabel = activationSource.getTokenString(tokenIndex);
        if (typeof resolvedLabel === 'string' && resolvedLabel.trim().length) {
            tokenLabel = resolvedLabel;
        }
    }

    let tokenId = findUserDataNumber(selection, idKey);
    if (!Number.isFinite(tokenId) && Number.isFinite(fallback?.tokenId)) {
        tokenId = Math.floor(fallback.tokenId);
    }
    if (!Number.isFinite(tokenId)
        && Number.isFinite(tokenIndex)
        && activationSource
        && typeof activationSource.getTokenId === 'function') {
        const resolvedTokenId = activationSource.getTokenId(tokenIndex);
        if (Number.isFinite(resolvedTokenId)) tokenId = Math.floor(resolvedTokenId);
    }

    const formattedLabel = formatTokenLabelForPreview(tokenLabel);
    return {
        tokenIndex: Number.isFinite(tokenIndex) ? Math.floor(tokenIndex) : null,
        tokenLabel: formattedLabel || '',
        tokenId: Number.isFinite(tokenId) ? Math.floor(tokenId) : null
    };
}

function buildTokenSummary(token = null) {
    if (!token) return '';
    const parts = [];
    if (Number.isFinite(token.tokenIndex)) {
        parts.push(`position ${token.tokenIndex + 1}`);
    }
    if (token.tokenLabel) {
        parts.push(`token ${quoteTokenLabel(token.tokenLabel)}`);
    }
    if (Number.isFinite(token.tokenId)) {
        parts.push(`token ID ${token.tokenId}`);
    }
    return parts.join(', ');
}

function describeSceneStage(stage = '', normalizedLabel = '') {
    const lower = String(stage || '').toLowerCase();
    if (!lower) {
        return normalizedLabel
            ? `The selected object is currently shown as ${normalizedLabel}.`
            : '';
    }
    if (lower === 'embedding.token') {
        return 'The scene is showing the token lookup result from the vocabulary embedding table.';
    }
    if (lower === 'embedding.position') {
        return 'The scene is showing the learned position vector associated with this token position.';
    }
    if (lower === 'embedding.sum') {
        return 'The scene is showing the initial residual state formed by adding token and position information together.';
    }
    if (lower === 'layer.incoming') {
        return 'The selected residual vector is entering a transformer block before the first LayerNorm runs.';
    }
    if (lower === 'residual.post_attention') {
        return 'The attention branch has already written its update back into the residual stream, and the token is on its way to LayerNorm 2 and the MLP.';
    }
    if (lower === 'residual.post_mlp') {
        return 'The MLP branch has already written its update back into the residual stream, so this token state is the block output passed upward to the next layer.';
    }
    if (lower === 'attention.pre') {
        return 'The scene is showing a raw scaled dot-product attention score before softmax normalization.';
    }
    if (lower === 'attention.post') {
        return 'The scene is showing a post-softmax attention weight that will be used to scale a value vector.';
    }
    if (lower.startsWith('qkv.q')) {
        return 'The selected vector has already been projected into query space for one attention head.';
    }
    if (lower.startsWith('qkv.k')) {
        return 'The selected vector has already been projected into key space for one attention head.';
    }
    if (lower.startsWith('qkv.v')) {
        return 'The selected vector has already been projected into value space for one attention head.';
    }
    if (lower.startsWith('ln1.')) {
        return 'The selected object is in the first LayerNorm path, immediately before self-attention reads the token state.';
    }
    if (lower.startsWith('ln2.')) {
        return 'The selected object is in the second LayerNorm path, immediately before the MLP reads the token state.';
    }
    if (lower.startsWith('mlp.up')) {
        return 'The scene is showing the MLP expansion step where model-width features are projected into the wider hidden space.';
    }
    if (lower.startsWith('mlp.activation')) {
        return 'The scene is showing the nonlinear MLP activation after the up-projection.';
    }
    if (lower.startsWith('mlp.down')) {
        return 'The scene is showing the MLP down-projection back to residual width.';
    }
    return `The active animation stage is ${JSON.stringify(stage)}.`;
}

function buildSpecificSelectionLines({
    selection = null,
    normalizedLabel = '',
    vectorTokenMetadata = null,
    attentionScoreSummary = null,
    activationSource = null,
    kvState = null
} = {}) {
    const lines = [];
    const labelText = normalizedLabel || normalizeSelectionLabel(selection?.label || '', selection);
    const activation = getActivationDataFromSelection(selection);
    const stage = typeof activation?.stage === 'string' ? activation.stage : '';
    const stageLower = stage.toLowerCase();
    const layerIndex = findUserDataNumber(selection, 'layerIndex');
    const headIndex = findUserDataNumber(selection, 'headIndex');
    const mainToken = resolveTokenContext(selection, activationSource, {
        fallback: {
            tokenIndex: vectorTokenMetadata?.tokenIndex,
            tokenLabel: vectorTokenMetadata?.tokenDisplayText || vectorTokenMetadata?.tokenText || '',
            tokenId: vectorTokenMetadata?.tokenId
        }
    });
    const keyToken = resolveTokenContext(selection, activationSource, {
        indexKey: 'keyTokenIndex',
        labelKey: 'keyTokenLabel',
        idKey: 'keyTokenId'
    });

    lines.push(`Selected item: ${labelText || 'Unknown selection'}`);
    if (selection?.kind) lines.push(`Selection kind: ${selection.kind}`);
    if (Number.isFinite(layerIndex)) lines.push(`Layer: ${layerIndex + 1}`);
    if (Number.isFinite(headIndex)) {
        const isAttentionHeadContext = stageLower.startsWith('attention.') || stageLower.startsWith('qkv.');
        lines.push(`${isAttentionHeadContext ? 'Attention head' : 'Head'}: ${headIndex + 1}`);
    }
    if (stage) lines.push(`Activation stage: ${stage}`);

    const stageSummary = describeSceneStage(stage, labelText);
    if (stageSummary) lines.push(`Current scene state: ${stageSummary}`);

    if (isAttentionScoreSelection(labelText, selection)) {
        const mode = resolveAttentionModeFromSelection(selection)
            || (stageLower.includes('post') ? 'post' : 'pre');
        const querySummary = buildTokenSummary(mainToken);
        const keySummary = buildTokenSummary(keyToken);
        const selectedScore = mode === 'post' ? activation?.postScore : activation?.preScore;
        const scoreText = Number.isFinite(selectedScore)
            ? selectedScore.toFixed(ATTENTION_SCORE_DECIMALS)
            : String(attentionScoreSummary?.defaultValue?.score || '').trim();

        lines.push(`Attention mode: ${mode === 'post' ? 'post-softmax' : 'pre-softmax'}`);
        if (querySummary) lines.push(`Query token: ${querySummary}`);
        if (keySummary) lines.push(`Key token: ${keySummary}`);
        if (scoreText && scoreText !== ATTENTION_VALUE_PLACEHOLDER && scoreText !== 'n/a') {
            lines.push(`Displayed attention score: ${scoreText}`);
        }
        lines.push(mode === 'post'
            ? 'This selected cell is the normalized attention weight for that query/key pair in this head.'
            : 'This selected cell is the raw scaled dot product for that query/key pair in this head, before softmax.'
        );
        return lines;
    }

    if (isResidualVectorSelection(labelText, selection)) {
        const tokenSummary = buildTokenSummary(mainToken);
        if (tokenSummary) lines.push(`Token represented by this residual vector: ${tokenSummary}`);
        lines.push('This selection is one residual-stream vector for a single token position, not the whole residual stream.');
        if (stageLower === 'layer.incoming') {
            lines.push('At this moment, the block is about to read this token state and refine it with attention and then the MLP.');
        } else if (stageLower === 'residual.post_attention') {
            lines.push('At this moment, attention has already added context from other positions into this token state.');
        } else if (stageLower === 'residual.post_mlp') {
            lines.push('At this moment, the full transformer block update has been applied and this token state is ready to move to the next layer.');
        }
        return lines;
    }

    if (isWeightedSumSelection(labelText, selection)) {
        const tokenSummary = buildTokenSummary(mainToken);
        if (tokenSummary) lines.push(`Query token for this weighted sum: ${tokenSummary}`);
        lines.push('This selected vector is the head-wise weighted sum of value vectors after attention weights have been applied.');
        return lines;
    }

    const lowerLabel = String(labelText || '').toLowerCase();
    if (lowerLabel.includes('query vector')) {
        const tokenSummary = buildTokenSummary(mainToken);
        if (tokenSummary) lines.push(`Token represented by this query vector: ${tokenSummary}`);
        lines.push('This vector is the token state projected into query space for one head.');
        return lines;
    }
    if (lowerLabel.includes('key vector')) {
        const tokenSummary = buildTokenSummary(mainToken);
        if (tokenSummary) lines.push(`Token represented by this key vector: ${tokenSummary}`);
        lines.push('This vector is the token state projected into key space for one head.');
        return lines;
    }
    if (lowerLabel.includes('value vector')) {
        const tokenSummary = buildTokenSummary(mainToken);
        if (tokenSummary) lines.push(`Token represented by this value vector: ${tokenSummary}`);
        lines.push(isKvCacheVectorSelection(selection)
            ? 'This value vector is currently being shown as part of the persistent KV cache.'
            : 'This vector is the token state projected into value space for one head.'
        );
        return lines;
    }
    if (isLogitBarSelection(labelText, selection)) {
        const logitEntry = selection?.info?.logitEntry || null;
        const tokenLabel = formatTokenLabelForPreview(logitEntry?.token || logitEntry?.label || '');
        const tokenId = Number.isFinite(logitEntry?.token_id) ? Math.floor(logitEntry.token_id) : null;
        const probability = Number.isFinite(logitEntry?.probability)
            ? logitEntry.probability
            : (Number.isFinite(logitEntry?.prob) ? logitEntry.prob : null);
        if (tokenLabel) lines.push(`Candidate token: token ${quoteTokenLabel(tokenLabel)}`);
        if (Number.isFinite(tokenId)) lines.push(`Candidate token ID: ${tokenId}`);
        if (Number.isFinite(probability)) lines.push(`Displayed probability: ${probability}`);
        lines.push('This selection is one vocabulary candidate in the final output distribution.');
        return lines;
    }
    if (isSelfAttentionSelection(labelText, selection)) {
        const tokenSummary = buildTokenSummary(mainToken);
        const keySummary = buildTokenSummary(keyToken);
        if (tokenSummary) lines.push(`Primary token context: ${tokenSummary}`);
        if (keySummary) lines.push(`Secondary token context: ${keySummary}`);
        lines.push('This selection belongs to the self-attention computation for the current layer and head.');
        return lines;
    }

    const mainTokenSummary = buildTokenSummary(mainToken);
    if (mainTokenSummary) lines.push(`Primary token context: ${mainTokenSummary}`);
    if (kvState?.selectionIsCachedKv) {
        lines.push('This selected object is currently being shown as a cached KV-state object.');
    }
    return lines;
}

export function buildSelectionChatPrompt({
    selection = null,
    normalizedLabel = '',
    title = '',
    subtitle = '',
    subtitleSecondary = '',
    descriptionText = '',
    equationText = '',
    metaLines = [],
    legendLines = [],
    attentionLines = [],
    dataLines = [],
    promptContextSummary = '',
    attentionScoreSummary = null,
    vectorTokenMetadata = null,
    activationSource = null,
    kvState = null
} = {}) {
    const instructionText = [
        'You are helping a user understand an interactive GPT-2 visualization.',
        'Use the reference markdown below as the broad description of the full scene, then use the live selection context afterward as the most specific description of what is currently selected.',
        'Answer the user\'s next question about the model or the visualization using both sources.'
    ].join('\n');

    const panelSummaryLines = [];
    if (title) panelSummaryLines.push(`Panel title: ${title}`);
    if (subtitle) panelSummaryLines.push(`Primary subtitle: ${subtitle}`);
    if (subtitleSecondary) panelSummaryLines.push(`Secondary subtitle: ${subtitleSecondary}`);

    const kvLines = [];
    if (kvState) {
        kvLines.push(`KV cache mode: ${kvState.kvCacheModeEnabled ? 'enabled' : 'disabled'}`);
        if (kvState.kvCacheModeEnabled) {
            if (kvState.kvCacheDecodeActive) {
                kvLines.push('Current KV cache phase: decode / single-query pass');
            } else if (kvState.kvCachePrefillActive) {
                kvLines.push('Current KV cache phase: prefill');
            } else {
                kvLines.push('Current KV cache phase: enabled, but no active decode-only pass was detected');
            }
            if (Number.isFinite(kvState.kvCachePassIndex)) {
                kvLines.push(`KV cache pass index: ${kvState.kvCachePassIndex}`);
            }
        }
        kvLines.push(`Current selection is a cached KV object: ${kvState.selectionIsCachedKv ? 'yes' : 'no'}`);
    }

    return joinSections(
        instructionText,
        buildMarkdownSection('Visualization Reference Markdown', String(visualizationDescriptionMarkdown || '').trim()),
        buildMarkdownBulletSection('Current Selection', buildSpecificSelectionLines({
            selection,
            normalizedLabel,
            vectorTokenMetadata,
            attentionScoreSummary,
            activationSource,
            kvState
        })),
        buildMarkdownBulletSection('Selection Preview Panel', panelSummaryLines),
        buildMarkdownSection('Panel Description', descriptionText),
        buildMarkdownSection('Equation Overlay', equationText),
        buildMarkdownSection('Prompt Context', promptContextSummary),
        buildMarkdownBulletSection('Visible Details', metaLines),
        buildMarkdownBulletSection('Visible Attention Panel', attentionLines),
        buildMarkdownBulletSection('Visible Legend', legendLines),
        buildMarkdownSection('Visible Activation Data', Array.isArray(dataLines) ? dataLines.join('\n') : ''),
        buildMarkdownBulletSection('KV Cache State', kvLines),
        'The user will now ask a question.'
    );
}
