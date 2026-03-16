import {
    formatLayerNormLabel,
    isLayerNormOutputStage
} from '../utils/layerNormLabels.js';
import {
    MHA_FINAL_K_COLOR,
    MHA_FINAL_Q_COLOR,
    MHA_FINAL_V_COLOR
} from '../animations/LayerAnimationConstants.js';
import {
    getActivationDataFromSelection,
    isAttentionScoreSelection,
    isResidualVectorSelection,
    isWeightedSumSelection,
    normalizeSelectionLabel,
    resolveAttentionModeFromSelection
} from './selectionPanelSelectionUtils.js';

function toColorHex(value = 0) {
    return `#${(value >>> 0).toString(16).padStart(6, '0')}`;
}

function isLayerNormScaleStage(stageLower = '', kind = '') {
    return stageLower === `${kind}.scale`
        || stageLower === `${kind}.product`
        || stageLower === `${kind}.param.scale`;
}

function isLayerNormOutputVectorStage(stageLower = '', kind = '') {
    return isLayerNormOutputStage(stageLower)
        && String(stageLower || '').toLowerCase().startsWith(`${kind}.`);
}

export function buildChatPromptInstructionText() {
    return [
        'You are helping a user understand an interactive GPT-2 visualization.',
        'Answer as if you are watching the same animation alongside the user. Prefer wording like "here we are seeing" or "in this view" instead of describing the animation as belonging to the user or to you.',
        'Use the reference markdown below as the broad description of the full scene, then use the live selection and visualization-state context afterward as the most specific description of what is currently on screen.',
        'If the user\'s next question is about self-attention, it may be useful to refer to query, key, and value vectors by the visualization colors listed below.',
        'Answer the user\'s next question about the model or the visualization using both sources.'
    ].join('\n');
}

export function buildSelfAttentionColorCueLines() {
    return [
        `Query vectors are blue (${toColorHex(MHA_FINAL_Q_COLOR)}).`,
        `Key vectors are green (${toColorHex(MHA_FINAL_K_COLOR)}).`,
        `Value vectors are red (${toColorHex(MHA_FINAL_V_COLOR)}).`,
        'When a self-attention answer would be clearer with color language, use these names to match what we are seeing.'
    ];
}

export function describeSceneStage(stage = '', normalizedLabel = '') {
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
        return `The selected residual vector is entering a transformer block before ${formatLayerNormLabel('ln1')} runs.`;
    }
    if (lower === 'ln1.norm') {
        return `The selected object is in the normalization step of ${formatLayerNormLabel('ln1')}, before the learned scale and shift have fully prepared the token for attention.`;
    }
    if (isLayerNormScaleStage(lower, 'ln1')) {
        return `The selected object is in the scale step of ${formatLayerNormLabel('ln1')}, where learned per-dimension gains are being applied before the final shift.`;
    }
    if (isLayerNormOutputVectorStage(lower, 'ln1')) {
        return `The selected object is at the output of ${formatLayerNormLabel('ln1')}, immediately before self-attention reads the token state.`;
    }
    if (lower === 'attention.pre') {
        return 'The scene is showing a raw scaled dot-product attention score before softmax normalization.';
    }
    if (lower === 'attention.post') {
        return 'The scene is showing a post-softmax attention weight that will be used to scale a value vector.';
    }
    if (lower === 'attention.weighted_value') {
        return 'The scene is showing a value vector after an attention weight has been applied, on its way into the head\'s weighted sum.';
    }
    if (lower === 'attention.output_projection') {
        return 'The scene is showing the recombined head output being projected back to model width before it is added into the residual stream.';
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
    if (lower === 'residual.post_attention') {
        return 'The attention branch has already written its update back into the residual stream, and the token is on its way to LayerNorm 2 and the MLP.';
    }
    if (lower === 'ln2.norm') {
        return `The selected object is in the normalization step of ${formatLayerNormLabel('ln2')}, before the MLP reads the token state.`;
    }
    if (isLayerNormScaleStage(lower, 'ln2')) {
        return `The selected object is in the scale step of ${formatLayerNormLabel('ln2')}, where learned gains are being applied before the final shift into the MLP path.`;
    }
    if (isLayerNormOutputVectorStage(lower, 'ln2')) {
        return `The selected object is at the output of ${formatLayerNormLabel('ln2')}, immediately before the MLP reads the token state.`;
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
    if (lower === 'residual.post_mlp') {
        return 'The MLP branch has already written its update back into the residual stream, so this token state is the block output passed upward to the next layer.';
    }
    if (lower === 'final_ln.norm') {
        return `The scene is showing the final normalization step at ${formatLayerNormLabel('final')} before the last token state is converted into logits.`;
    }
    if (isLayerNormScaleStage(lower, 'final_ln')) {
        return `The scene is showing the scale step of ${formatLayerNormLabel('final')}, where learned gains are being applied before the final shift into the unembedding path.`;
    }
    if (isLayerNormOutputVectorStage(lower, 'final_ln')) {
        return `The scene is showing the output of ${formatLayerNormLabel('final')}, immediately before the model maps the token state into vocabulary logits.`;
    }
    return `The active animation stage is ${JSON.stringify(stage)}.`;
}

export function buildVisualizationStateLines({
    selection = null,
    normalizedLabel = '',
    kvState = null
} = {}) {
    const lines = [];
    const labelText = normalizedLabel || normalizeSelectionLabel(selection?.label || '', selection);
    const activation = getActivationDataFromSelection(selection);
    const stage = typeof activation?.stage === 'string' ? activation.stage : '';
    const stageLower = stage.toLowerCase();

    if (stage) lines.push(`Active stage key: ${stage}`);

    const stageSummary = describeSceneStage(stage, labelText);
    if (stageSummary) lines.push(`Current stage summary: ${stageSummary}`);

    if (stageLower === 'embedding.token') {
        lines.push('We are at the very start of the token path, before position information has been added into the residual stream.');
    } else if (stageLower === 'embedding.position') {
        lines.push('We are looking at positional information before it is added to the token embedding.');
    } else if (stageLower === 'embedding.sum') {
        lines.push('We are looking at the token state after token and position embeddings have been combined, but before the first block processes it.');
    } else if (stageLower === 'layer.incoming') {
        lines.push('We are at the entrance to a transformer block, right before the attention sublayer begins.');
    } else if (stageLower === 'ln1.norm') {
        lines.push(`We are between the incoming residual stream and the learned affine output of ${formatLayerNormLabel('ln1')}.`);
    } else if (isLayerNormScaleStage(stageLower, 'ln1')) {
        lines.push(`We are in the middle of ${formatLayerNormLabel('ln1')}, after normalization and during the learned scaling step.`);
    } else if (isLayerNormOutputVectorStage(stageLower, 'ln1')) {
        lines.push('We are at the handoff from LayerNorm 1 into self-attention, right before Q/K/V projections are read from this token state.');
    } else if (stageLower === 'qkv.q') {
        lines.push('We are in the query-projection view for one attention head, where this token prepares the vector that will look outward across the context.');
    } else if (stageLower === 'qkv.k') {
        lines.push('We are in the key-projection view for one attention head, where this token prepares the vector that other queries will compare against.');
    } else if (stageLower === 'qkv.v') {
        lines.push('We are in the value-projection view for one attention head, where this token prepares the content that can be routed forward by attention weights.');
    } else if (stageLower === 'attention.pre') {
        lines.push('We are in the raw score step of attention, before softmax converts compatibility scores into normalized weights.');
    } else if (stageLower === 'attention.post') {
        lines.push('We are in the normalized attention step, after softmax has distributed the query token\'s focus across the visible context.');
    } else if (stageLower === 'attention.weighted_value') {
        lines.push('We are after attention weighting has started, while value-vector contributions are being scaled and accumulated for the head output.');
    } else if (stageLower === 'attention.output_projection') {
        lines.push('We are after per-head mixing, while the concatenated attention result is being projected back into residual width.');
    } else if (stageLower === 'residual.post_attention') {
        lines.push('We are between the attention sublayer and the MLP: the attention update has already been written back into the residual stream.');
    } else if (stageLower === 'ln2.norm') {
        lines.push(`We are between the post-attention residual stream and the learned affine output of ${formatLayerNormLabel('ln2')}.`);
    } else if (isLayerNormScaleStage(stageLower, 'ln2')) {
        lines.push(`We are in the middle of ${formatLayerNormLabel('ln2')}, after normalization and during the learned scaling step.`);
    } else if (isLayerNormOutputVectorStage(stageLower, 'ln2')) {
        lines.push('We are at the handoff from LayerNorm 2 into the MLP, right before the feed-forward sublayer reads this token state.');
    } else if (stageLower === 'mlp.up') {
        lines.push('We are in the MLP expansion step, where residual-width features fan out into a wider hidden representation.');
    } else if (stageLower === 'mlp.activation') {
        lines.push('We are in the nonlinear part of the MLP, after the up-projection and before the hidden state is compressed back down.');
    } else if (stageLower === 'mlp.down') {
        lines.push('We are in the MLP down-projection step, where the widened hidden representation is being mapped back to residual width.');
    } else if (stageLower === 'residual.post_mlp') {
        lines.push('We are at the end of the block, after both attention and the MLP have updated the residual stream.');
    } else if (stageLower === 'final_ln.norm') {
        lines.push('We are in the final normalization path, just before the last token state is sent into the vocabulary projection.');
    } else if (isLayerNormScaleStage(stageLower, 'final_ln')) {
        lines.push(`We are in the middle of ${formatLayerNormLabel('final')}, after normalization and during the learned scaling step before logits are computed.`);
    } else if (isLayerNormOutputVectorStage(stageLower, 'final_ln')) {
        lines.push('We are at the output of the final normalization path, immediately before the unembedding produces vocabulary logits.');
    }

    if (isAttentionScoreSelection(labelText, selection)) {
        const mode = resolveAttentionModeFromSelection(selection)
            || (stageLower === 'attention.post' ? 'post' : 'pre');
        lines.push(mode === 'post'
            ? 'The selected attention cell is currently being interpreted as a post-softmax weight.'
            : 'The selected attention cell is currently being interpreted as a pre-softmax score.');
    } else if (isWeightedSumSelection(labelText, selection)) {
        lines.push('This view is after attention weights have been applied to value vectors, but before the final output projection writes the result back into the residual stream.');
    } else if (isResidualVectorSelection(labelText, selection) && stageLower === 'residual.post_attention') {
        lines.push('In other words, we are looking at the token state after context from other positions has already been mixed in.');
    }

    if (kvState?.kvCacheModeEnabled) {
        if (kvState.kvCacheDecodeActive) {
            lines.push('KV cache decode mode is active, so the live view may emphasize reusing cached keys and values while computing a single new query.');
        } else if (kvState.kvCachePrefillActive) {
            lines.push('KV cache prefill mode is active, so the live view may emphasize building the cache for the prompt tokens.');
        } else {
            lines.push('KV cache mode is enabled in this scene, even if the current selection is not part of an active decode-only step.');
        }
    }

    return lines;
}
