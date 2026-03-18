import visualizationDescriptionMarkdown from '../../vizualization_description.md?raw';
import {
    ATTENTION_SCORE_DECIMALS,
    ATTENTION_VALUE_PLACEHOLDER
} from './selectionPanelConstants.js';
import { formatTokenLabelForPreview } from './selectionPanelFormatUtils.js';
import {
    buildChatPromptInstructionText,
    buildSelfAttentionColorCueLines,
    buildVisualizationStateLines,
    describeSceneStage
} from './selectionPanelChatPromptStateUtils.js';
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

function formatVisualizationModeLabel(mode = '') {
    const normalized = String(mode || '').trim().toLowerCase();
    if (normalized === 'transformer-view2d') return '2D semantic canvas';
    if (normalized === '3d') return '3D scene';
    return 'Unknown / mixed view';
}

function buildSemanticTargetSummary(target = null) {
    if (!target || typeof target !== 'object') return '';
    const parts = [];
    if (typeof target.componentKind === 'string' && target.componentKind.trim().length) {
        parts.push(`component ${target.componentKind.trim()}`);
    }
    if (Number.isFinite(target.layerIndex)) {
        parts.push(`layer ${Math.floor(target.layerIndex) + 1}`);
    }
    if (Number.isFinite(target.headIndex)) {
        parts.push(`head ${Math.floor(target.headIndex) + 1}`);
    }
    if (typeof target.stage === 'string' && target.stage.trim().length) {
        parts.push(`stage ${target.stage.trim()}`);
    }
    if (typeof target.role === 'string' && target.role.trim().length) {
        parts.push(`role ${target.role.trim()}`);
    }
    if (Number.isFinite(target.tokenIndex)) {
        parts.push(`token position ${Math.floor(target.tokenIndex) + 1}`);
    }
    if (Number.isFinite(target.positionIndex)) {
        parts.push(`position index ${Math.floor(target.positionIndex) + 1}`);
    }
    return parts.join(', ');
}

function buildVisualizationSurfaceLines(surfaceState = null) {
    if (!surfaceState || typeof surfaceState !== 'object') return [];
    const lines = [];
    lines.push('I might ask about the whole visualization or any part of it, not only the current live selection.');

    const supports3d = surfaceState.supports3d !== false;
    const supports2d = surfaceState.supports2d !== false;
    if (supports3d && supports2d) {
        lines.push('This visualization has both a 3D scene view and a 2D semantic canvas / matrix-style view of the same model state.');
    } else if (supports2d) {
        lines.push('I have a 2D semantic canvas / matrix-style view available.');
    } else if (supports3d) {
        lines.push('I have a 3D scene view available.');
    }

    const activeModeLabel = formatVisualizationModeLabel(surfaceState.activeMode);
    lines.push(`I am currently in the ${activeModeLabel}.`);

    if (surfaceState.activeMode === 'transformer-view2d') {
        lines.push('Because I am in the 2D semantic canvas, my question may refer to module cards, matrix-style layouts, semantic focus targets, panning, zooming, or detail views rather than 3D camera placement.');
    } else if (surfaceState.activeMode === '3d') {
        lines.push('Because I am in the 3D scene, my question may refer to spatial arrangement, moving vectors, rising matrices, animation stages, or geometric relationships in the scene.');
    }

    const current2dFocusLabel = String(surfaceState.current2dFocusLabel || '').trim();
    if (current2dFocusLabel) {
        lines.push(`My current 2D focus label is: ${current2dFocusLabel}`);
    }

    const current2dDetailFocusLabel = String(surfaceState.current2dDetailFocusLabel || '').trim();
    if (current2dDetailFocusLabel) {
        lines.push(`My current 2D detail focus is: ${current2dDetailFocusLabel}`);
    }

    const current2dSemanticTargetSummary = buildSemanticTargetSummary(surfaceState.current2dSemanticTarget);
    if (current2dSemanticTargetSummary) {
        lines.push(`My current 2D semantic target is: ${current2dSemanticTargetSummary}`);
    }

    const current2dTransitionMode = String(surfaceState.current2dTransitionMode || '').trim();
    if (current2dTransitionMode) {
        lines.push(`The current 2D transition mode is: ${current2dTransitionMode}`);
    }

    if (surfaceState.activeMode === 'transformer-view2d') {
        lines.push(`The 2D selection sidebar is currently visible: ${surfaceState.current2dSelectionSidebarVisible ? 'yes' : 'no'}`);
    }

    const available2dFocusLabel = String(surfaceState.available2dFocusLabel || '').trim();
    if (surfaceState.activeMode !== 'transformer-view2d' && available2dFocusLabel) {
        lines.push(`If I open the matching 2D target for my current selection, it would focus on: ${available2dFocusLabel}`);
    }

    const available2dDetailFocusLabel = String(surfaceState.available2dDetailFocusLabel || '').trim();
    if (surfaceState.activeMode !== 'transformer-view2d' && available2dDetailFocusLabel) {
        lines.push(`If I open this in 2D, the detailed focus would be: ${available2dDetailFocusLabel}`);
    }

    lines.push('The current selection and current visualization-state sections below describe what I am focused on right now and may still be useful even if my question is broader.');
    return lines;
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

    lines.push(`I currently have this selected: ${labelText || 'Unknown selection'}`);
    if (selection?.kind) lines.push(`The selected thing is of kind: ${selection.kind}`);
    if (Number.isFinite(layerIndex)) lines.push(`I am looking at Layer ${layerIndex + 1}.`);
    if (Number.isFinite(headIndex)) {
        const isAttentionHeadContext = stageLower.startsWith('attention.') || stageLower.startsWith('qkv.');
        lines.push(`I am looking at ${isAttentionHeadContext ? 'Attention Head' : 'Head'} ${headIndex + 1}.`);
    }
    if (stage) lines.push(`The current activation stage is: ${stage}`);

    const stageSummary = describeSceneStage(stage, labelText);
    if (stageSummary) lines.push(`What I am seeing right now: ${stageSummary}`);

    if (isAttentionScoreSelection(labelText, selection)) {
        const mode = resolveAttentionModeFromSelection(selection)
            || (stageLower.includes('post') ? 'post' : 'pre');
        const querySummary = buildTokenSummary(mainToken);
        const keySummary = buildTokenSummary(keyToken);
        const selectedScore = mode === 'post' ? activation?.postScore : activation?.preScore;
        const scoreText = Number.isFinite(selectedScore)
            ? selectedScore.toFixed(ATTENTION_SCORE_DECIMALS)
            : String(attentionScoreSummary?.defaultValue?.score || '').trim();

        lines.push(`I am looking at the ${mode === 'post' ? 'post-softmax' : 'pre-softmax'} version of this attention cell.`);
        if (querySummary) lines.push(`The query token here is: ${querySummary}`);
        if (keySummary) lines.push(`The key / source token here is: ${keySummary}`);
        if (scoreText && scoreText !== ATTENTION_VALUE_PLACEHOLDER && scoreText !== 'n/a') {
            lines.push(`The displayed attention score here is: ${scoreText}`);
        }
        lines.push(mode === 'post'
            ? 'This selected cell is the normalized attention weight for that query/key pair in this head.'
            : 'This selected cell is the raw scaled dot product for that query/key pair in this head, before softmax.'
        );
        return lines;
    }

    if (isResidualVectorSelection(labelText, selection)) {
        const tokenSummary = buildTokenSummary(mainToken);
        if (tokenSummary) lines.push(`This residual vector corresponds to: ${tokenSummary}`);
        lines.push('I am looking at one residual-stream vector for a single token position, not the whole residual stream.');
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
        if (tokenSummary) lines.push(`The query token for this weighted sum is: ${tokenSummary}`);
        lines.push('I am looking at the head-wise weighted sum of value vectors after attention weights have been applied.');
        return lines;
    }

    const lowerLabel = String(labelText || '').toLowerCase();
    if (lowerLabel.includes('query vector')) {
        const tokenSummary = buildTokenSummary(mainToken);
        if (tokenSummary) lines.push(`This query vector corresponds to: ${tokenSummary}`);
        lines.push('I am looking at the token state projected into query space for one head.');
        return lines;
    }
    if (lowerLabel.includes('key vector')) {
        const tokenSummary = buildTokenSummary(mainToken);
        if (tokenSummary) lines.push(`This key vector corresponds to: ${tokenSummary}`);
        lines.push('I am looking at the token state projected into key space for one head.');
        return lines;
    }
    if (lowerLabel.includes('value vector')) {
        const tokenSummary = buildTokenSummary(mainToken);
        if (tokenSummary) lines.push(`This value vector corresponds to: ${tokenSummary}`);
        lines.push(isKvCacheVectorSelection(selection)
            ? 'I am seeing this value vector as part of the persistent KV cache.'
            : 'I am looking at the token state projected into value space for one head.'
        );
        return lines;
    }
    if (isLogitBarSelection(labelText, selection)) {
        const logitEntry = selection?.info?.logitEntry || null;
        const tokenLabel = formatTokenLabelForPreview(logitEntry?.token || logitEntry?.label || '');
        const rawTokenId = Number(logitEntry?.token_id ?? logitEntry?.tokenId);
        const tokenId = Number.isFinite(rawTokenId) ? Math.floor(rawTokenId) : null;
        const probability = Number.isFinite(logitEntry?.probability)
            ? logitEntry.probability
            : (Number.isFinite(logitEntry?.prob) ? logitEntry.prob : null);
        if (tokenLabel) lines.push(`The candidate token here is: token ${quoteTokenLabel(tokenLabel)}`);
        if (Number.isFinite(tokenId)) lines.push(`The candidate token ID here is: ${tokenId}`);
        if (Number.isFinite(probability)) lines.push(`The displayed probability here is: ${probability}`);
        lines.push('I am looking at one vocabulary candidate in the final output distribution.');
        return lines;
    }
    if (isSelfAttentionSelection(labelText, selection)) {
        const tokenSummary = buildTokenSummary(mainToken);
        const keySummary = buildTokenSummary(keyToken);
        if (tokenSummary) lines.push(`The primary token context here is: ${tokenSummary}`);
        if (keySummary) lines.push(`The secondary token context here is: ${keySummary}`);
        lines.push('I am looking at part of the self-attention computation for the current layer and head.');
        return lines;
    }

    const mainTokenSummary = buildTokenSummary(mainToken);
    if (mainTokenSummary) lines.push(`The primary token context here is: ${mainTokenSummary}`);
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
    subtitleTertiary = '',
    panelContentsBlurb = '',
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
    kvState = null,
    surfaceState = null
} = {}) {
    const instructionText = buildChatPromptInstructionText();

    const panelSummaryLines = [];
    if (title) panelSummaryLines.push(`Panel title: ${title}`);
    if (subtitle) panelSummaryLines.push(`Primary subtitle: ${subtitle}`);
    if (subtitleSecondary) panelSummaryLines.push(`Secondary subtitle: ${subtitleSecondary}`);
    if (subtitleTertiary) panelSummaryLines.push(`Tertiary subtitle: ${subtitleTertiary}`);

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
        buildMarkdownBulletSection('Question Context', buildVisualizationSurfaceLines(surfaceState)),
        buildMarkdownSection('Visualization Reference Markdown', String(visualizationDescriptionMarkdown || '').trim()),
        buildMarkdownBulletSection('Self-Attention Color Cues', buildSelfAttentionColorCueLines()),
        buildMarkdownBulletSection('Current Selection', buildSpecificSelectionLines({
            selection,
            normalizedLabel,
            vectorTokenMetadata,
            attentionScoreSummary,
            activationSource,
            kvState
        })),
        buildMarkdownBulletSection('Current Visualization State', buildVisualizationStateLines({
            selection,
            normalizedLabel,
            kvState
        })),
        buildMarkdownBulletSection('Selection Preview Panel', panelSummaryLines),
        buildMarkdownSection('Selection Preview Includes', panelContentsBlurb),
        buildMarkdownSection('Panel Description', descriptionText),
        buildMarkdownSection('Equation Overlay', equationText),
        buildMarkdownSection('Prompt Context', promptContextSummary),
        buildMarkdownBulletSection('Visible Details', metaLines),
        buildMarkdownBulletSection('Visible Attention Panel', attentionLines),
        buildMarkdownBulletSection('Visible Legend', legendLines),
        buildMarkdownSection('Visible Activation Data', Array.isArray(dataLines) ? dataLines.join('\n') : ''),
        buildMarkdownBulletSection('KV Cache State', kvLines),
        buildMarkdownSection('My Question', 'My actual question is below.\n\nINSERT YOUR QUESTION HERE')
    );
}
