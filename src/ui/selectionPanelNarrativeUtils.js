import {
    MHA_FINAL_Q_COLOR,
    MHA_FINAL_K_COLOR,
    MHA_FINAL_V_COLOR,
    MHA_OUTPUT_PROJECTION_MATRIX_COLOR,
    POSITION_EMBED_COLOR,
    MLP_UP_MATRIX_COLOR,
    MLP_DOWN_MATRIX_COLOR
} from '../animations/LayerAnimationConstants.js';
import { getActivationDataFromSelection } from './selectionPanelSelectionUtils.js';

const RESIDUAL_STREAM_DESCRIPTION = 'This is the residual stream vector for a token at this point in the model. It is the main path that carries information through the network. Attention and MLP updates are added back into this stream at each layer, so prior context is preserved while new information is incorporated.';

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
            return `"${cleanLabel}" is an interactive ${kindLabel} mapped to activation stage "${cleanStage}". This stage indicates where it participates in the current forward-pass pipeline.`;
        }
        if (cleanLabel) {
            return `"${cleanLabel}" is mapped to activation stage "${cleanStage}". This stage indicates where it participates in the current forward-pass pipeline.`;
        }
        if (kindLabel) {
            return `This ${kindLabel} is mapped to activation stage "${cleanStage}". This stage indicates where it participates in the current forward-pass pipeline.`;
        }
        return `This selection is mapped to activation stage "${cleanStage}". This stage indicates where it participates in the current forward-pass pipeline.`;
    }

    if (cleanLabel && kindLabel) {
        return `"${cleanLabel}" is an interactive ${kindLabel} in the scene. It contributes to the residual, attention, or MLP computation at this point in the layer flow.`;
    }
    if (cleanLabel) {
        return `"${cleanLabel}" is an interactive scene component. It contributes to the residual, attention, or MLP computation at this point in the layer flow.`;
    }
    if (kindLabel) {
        return `This ${kindLabel} contributes to the residual, attention, or MLP computation at this point in the layer flow.`;
    }
    return 'This interactive scene component contributes to the model forward pass at this point in the layer flow.';
}

const toKatexColorHex = (hex) => `#${Number(hex).toString(16).padStart(6, '0')}`;
const colorizeEquationToken = (hex, token) => `\\textcolor{${hex}}{${token}}`;
const SELECTION_EQUATION_COLORS = {
    q: toKatexColorHex(MHA_FINAL_Q_COLOR),
    k: toKatexColorHex(MHA_FINAL_K_COLOR),
    v: toKatexColorHex(MHA_FINAL_V_COLOR),
    output: toKatexColorHex(MHA_OUTPUT_PROJECTION_MATRIX_COLOR),
    embeddingVocab: toKatexColorHex(MHA_FINAL_Q_COLOR),
    embeddingPos: toKatexColorHex(POSITION_EMBED_COLOR),
    mlpUp: toKatexColorHex(MLP_UP_MATRIX_COLOR),
    mlpDown: toKatexColorHex(MLP_DOWN_MATRIX_COLOR)
};
const SELECTION_EQUATION_SYMBOLS = {
    Q: colorizeEquationToken(SELECTION_EQUATION_COLORS.q, 'Q'),
    K: colorizeEquationToken(SELECTION_EQUATION_COLORS.k, 'K'),
    V: colorizeEquationToken(SELECTION_EQUATION_COLORS.v, 'V'),
    WQ: colorizeEquationToken(SELECTION_EQUATION_COLORS.q, 'W_Q'),
    WK: colorizeEquationToken(SELECTION_EQUATION_COLORS.k, 'W_K'),
    WV: colorizeEquationToken(SELECTION_EQUATION_COLORS.v, 'W_V'),
    BQ: colorizeEquationToken(SELECTION_EQUATION_COLORS.q, 'b_Q'),
    BK: colorizeEquationToken(SELECTION_EQUATION_COLORS.k, 'b_K'),
    BV: colorizeEquationToken(SELECTION_EQUATION_COLORS.v, 'b_V'),
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
    MLPDown: colorizeEquationToken(SELECTION_EQUATION_COLORS.mlpDown, '\\mathrm{MLP}(u_{\\text{ln}})')
};

function formatEquationBlock(lines) {
    if (!Array.isArray(lines) || lines.length === 0) return '';
    const nonEmpty = lines.filter((line) => typeof line === 'string' && line.trim().length > 0);
    if (!nonEmpty.length) return '';
    return nonEmpty.map((line) => `$$${line}$$`).join('\n');
}

function buildNlpEmbeddingEquationBlock() {
    return formatEquationBlock([
        `${SELECTION_EQUATION_SYMBOLS.XTok} = ${SELECTION_EQUATION_SYMBOLS.E}[\\mathrm{token}_t]`,
        `${SELECTION_EQUATION_SYMBOLS.XPos} = ${SELECTION_EQUATION_SYMBOLS.P}[t]`,
        `x_t = ${SELECTION_EQUATION_SYMBOLS.XTok} + ${SELECTION_EQUATION_SYMBOLS.XPos}`
    ]);
}

function resolveLayerNormEquationSymbols(lower) {
    if (lower.includes('top')) {
        return {
            input: 'x_{\\text{out}}',
            norm: '\\hat{x}_{\\text{out}}',
            output: 'x_{\\text{final}}'
        };
    }
    if (lower.includes('ln2')) {
        return {
            input: 'u',
            norm: '\\hat{u}',
            output: 'u_{\\text{ln}}'
        };
    }
    return {
        input: 'x',
        norm: '\\hat{x}',
        output: 'x_{\\text{ln}}'
    };
}

function resolveAttentionHeadSubscript(selectionInfo) {
    void selectionInfo;
    return 'i';
}

function buildHeadSpecificAttentionEquation(selectionInfo) {
    const headSubscript = resolveAttentionHeadSubscript(selectionInfo);
    const Qh = `${SELECTION_EQUATION_SYMBOLS.Q}_{${headSubscript}}`;
    const Kh = `${SELECTION_EQUATION_SYMBOLS.K}_{${headSubscript}}`;
    const Vh = `${SELECTION_EQUATION_SYMBOLS.V}_{${headSubscript}}`;
    const Hh = `H_{${headSubscript}}`;
    return `${Hh} = \\mathrm{softmax}\\left(\\frac{${Qh} ${Kh}^\\top}{\\sqrt{d_h}} + M\\right)${Vh}`;
}

export function resolveSelectionEquations(label, selectionInfo = null) {
    const lower = String(label || '').toLowerCase();
    const attentionEquation = buildHeadSpecificAttentionEquation(selectionInfo);
    const isNlpEmbeddingSelection = lower.startsWith('token:')
        || lower.startsWith('position:')
        || lower.includes('token embedding')
        || lower.includes('vocab embedding')
        || lower.includes('positional embedding')
        || lower.includes('embedding sum');

    if (lower.includes('query weight matrix')) {
        return formatEquationBlock([
            `${SELECTION_EQUATION_SYMBOLS.Q} = x_{\\text{ln}} ${SELECTION_EQUATION_SYMBOLS.WQ} + ${SELECTION_EQUATION_SYMBOLS.BQ}`,
            attentionEquation
        ]);
    }
    if (lower.includes('key weight matrix')) {
        return formatEquationBlock([
            `${SELECTION_EQUATION_SYMBOLS.K} = x_{\\text{ln}} ${SELECTION_EQUATION_SYMBOLS.WK} + ${SELECTION_EQUATION_SYMBOLS.BK}`,
            attentionEquation
        ]);
    }
    if (lower.includes('value weight matrix')) {
        return formatEquationBlock([
            `${SELECTION_EQUATION_SYMBOLS.V} = x_{\\text{ln}} ${SELECTION_EQUATION_SYMBOLS.WV} + ${SELECTION_EQUATION_SYMBOLS.BV}`,
            attentionEquation
        ]);
    }
    if (lower.includes('weighted value vector')) {
        return formatEquationBlock([
            '\\tilde{V}_{t,j} = \\alpha_{t,j} V_j',
            'H_t = \\sum_j \\tilde{V}_{t,j}'
        ]);
    }
    if (lower.includes('output projection matrix')) {
        return formatEquationBlock([
            'H = \\mathrm{Concat}(H_i)_{i=1}^{12}',
            `O = H ${SELECTION_EQUATION_SYMBOLS.WO} + ${SELECTION_EQUATION_SYMBOLS.BO}`,
            'u = x + O'
        ]);
    }
    if (lower.includes('mlp up weight matrix')) {
        return formatEquationBlock([
            `a = u_{\\text{ln}} ${SELECTION_EQUATION_SYMBOLS.WUp} + ${SELECTION_EQUATION_SYMBOLS.BUp}`,
            'z = \\mathrm{GELU}(a)'
        ]);
    }
    if (lower.includes('mlp down weight matrix')) {
        return formatEquationBlock([
            `\\mathrm{MLP}(u_{\\text{ln}}) = z ${SELECTION_EQUATION_SYMBOLS.WDown} + ${SELECTION_EQUATION_SYMBOLS.BDown}`,
            `x_{\\text{out}} = u + ${SELECTION_EQUATION_SYMBOLS.MLPDown}`
        ]);
    }
    if (lower.includes('vocab embedding (top)') || lower.includes('unembedding')) {
        return formatEquationBlock([
            `\\ell = x_{\\text{final}} ${SELECTION_EQUATION_SYMBOLS.WU}`,
            'p = \\mathrm{softmax}(\\ell)'
        ]);
    }
    if (isNlpEmbeddingSelection) {
        return buildNlpEmbeddingEquationBlock();
    }

    const isLayerNormSelection = lower.includes('layernorm')
        || lower.includes('layer norm')
        || lower.includes('ln1')
        || lower.includes('ln2');
    if (isLayerNormSelection) {
        const symbols = resolveLayerNormEquationSymbols(lower);
        const normalizeEq = `${symbols.norm} = \\frac{${symbols.input} - \\mu}{\\sqrt{\\sigma^2 + \\epsilon}}`;
        const affineEq = `${symbols.output} = \\gamma \\odot ${symbols.norm} + \\beta`;

        if (lower.includes('normed') || lower.includes('normalized')) {
            return formatEquationBlock([normalizeEq]);
        }
        return formatEquationBlock([normalizeEq, affineEq]);
    }

    if (lower.includes('weight matrix')) {
        return formatEquationBlock(['y = xW']);
    }

    return '';
}

export function resolveDescription(label, kind = null, selectionInfo = null) {
    const lower = (label || '').toLowerCase();
    const activation = getActivationDataFromSelection(selectionInfo);
    const stage = activation?.stage || '';
    const stageLower = stage.toLowerCase();

    if (stageLower) {
        if (stageLower.startsWith('embedding.token')) {
            return 'This is the embedding vector for a specific token: a row of the embedding matrix. It contributes to the initial residual stream after being added to the position embedding. This vector is the starting point for the residual stream.';
        }
        if (stageLower.startsWith('embedding.position')) {
            return 'This is the position embedding vector for this token\'s index. It is added to the token embedding to form the initial residual stream, so order is encoded before any attention or MLP layers run.';
        }
        if (stageLower.startsWith('embedding.sum')) {
            return 'This is the summed input vector after combining token and position information. It is the initial residual stream that enters Layer 1.';
        }
        if (stageLower.startsWith('layer.incoming')) {
            return 'This is the residual stream as it enters a layer, before the first LayerNorm. It carries the accumulated information from earlier layers and will feed into attention and the MLP.';
        }
        if (stageLower === 'ln1.norm' || stageLower === 'ln2.norm') {
            return 'This is the normalized vector for one token inside LayerNorm, before learned scale and shift are applied. It has zero mean and unit variance across features at this step.';
        }
        if (stageLower === 'ln1.scale' || stageLower === 'ln2.scale') {
            return 'This is the LayerNorm-scaled vector after multiplying normalized features by learned per-feature scale parameters.';
        }
        if (stageLower === 'ln1.shift') {
            return 'This is the layer-normed residual stream right before query, key, and value projection. It is copied for each head to make queries, keys, and values, which then determine attention weights.';
        }
        if (stageLower === 'ln2.shift') {
            return 'This is the LayerNorm-2 output vector after scale and shift, right before the MLP block for this token.';
        }
        if (stageLower === 'qkv.q') {
            return 'This is the query vector for one token in one head. It encodes what the token wants to read and is compared with keys to form attention scores.';
        }
        if (stageLower === 'qkv.k') {
            return 'This is the key vector for one token in one head. It encodes what this token offers to other queries in the same head.';
        }
        if (stageLower === 'qkv.v') {
            return 'This is the value vector for one token in one head. Attention weights are used to mix these values into the head output.';
        }
        if (stageLower === 'attention.weighted_value') {
            return 'This is a value vector after being scaled by one post-softmax attention weight. It represents a single weighted contribution before all weighted values are added into the head output.';
        }
        if (stageLower === 'attention.output_projection') {
            return 'This is the attention output vector after concatenating head outputs and applying the output projection. It is the branch result that gets added back into the residual stream.';
        }
        if (stageLower === 'residual.post_attention') {
            return 'This is the residual stream after attention has been added. It represents the model state before the MLP block in this layer.';
        }
        if (stageLower === 'mlp.up') {
            return 'This is the vector after the MLP up-projection, in the larger hidden dimension. It is the input to the nonlinearity before being projected back down.';
        }
        if (stageLower === 'mlp.activation') {
            return 'This is the post-activation MLP vector (after GELU), still in the expanded hidden dimension before down-projection.';
        }
        if (stageLower === 'mlp.down') {
            return 'This is the vector after the MLP down-projection, back at model width. It is the MLP branch output that will be added to the residual stream.';
        }
        if (stageLower === 'residual.post_mlp') {
            return 'This is the residual stream after the MLP output has been added. It is the final output of this layer and the input to the next layer.';
        }
        if (stageLower === 'final_ln.norm') {
            return 'This is the normalized vector in the final LayerNorm at the top of the model, before final scale and shift are applied.';
        }
        if (stageLower === 'final_ln.scale') {
            return 'This is the final LayerNorm scale vector applied to normalized top-of-model features before unembedding.';
        }
        if (stageLower === 'final_ln.shift') {
            return 'This is the final LayerNorm shift vector applied after final scaling, immediately before the unembedding matrix maps to logits.';
        }
    }

    if (
        lower.includes('residual stream vector')
        || stageLower.includes('residual')
    ) {
        return RESIDUAL_STREAM_DESCRIPTION;
    }

    if (lower.startsWith('token:')) {
        return 'This is one input token (a subword piece). The model looks up its embedding vector and adds a position vector. That sum enters the residual stream and is transformed by each layer. Through attention, this token can influence later positions.';
    }
    if (lower.startsWith('position:')) {
        return 'This is the position embedding for a specific index in the sequence. It is added elementwise to the token embedding so the model knows order and distance. Without it, the model would treat the sequence as a bag of tokens. The same word at different positions gets different combined vectors.';
    }
    if (lower.includes('token embedding')) {
        return 'This is the embedding vector for a specific token: a row of the embedding matrix. It contributes to the initial residual stream after being added to the position embedding. This vector is the starting point for the residual stream.';
    }
    if (lower.includes('position embedding')) {
        return 'This is the position embedding vector for this token\'s index. It is added to the token embedding to form the initial residual stream, so order is encoded before any attention or MLP layers run.';
    }
    if (lower.includes('embedding sum')) {
        return 'This is the summed input vector after combining token and position information. It is the initial residual stream that enters Layer 1.';
    }
    if (lower.includes('embedding connector trail')) {
        return 'This connector marks the visual handoff path from the token and position chips into the embedding/residual input flow for the current pass.';
    }
    if (lower.includes('vocab embedding (top)')) {
        return 'This is the output (unembedding) matrix at the top of the model. It maps the final residual stream vector to vocabulary logits. Softmax converts logits to probabilities and the model samples or selects the next token. In GPT-2, these weights are tied to the input embedding.';
    }
    if (lower.includes('vocab embedding')) {
        return 'This matrix converts token IDs into vectors. It is the learned lookup table that produces the initial residual stream input after adding position information. These vectors carry semantic and syntactic information into the model. Everything else in the network builds on these representations.';
    }
    if (lower.includes('positional embedding')) {
        return 'This is the learned matrix of position vectors. Each position has its own vector that is summed with the token embedding at the bottom of the model. This gives the model a sense of order and distance. It lets attention distinguish first versus last occurrences.';
    }
    if ((lower.includes('ln1') || lower.includes('ln2')) && (lower.includes('scale') || lower.includes('gamma'))) {
        const isLn1 = lower.includes('ln1');
        const lnName = isLn1 ? 'LN1' : 'LN2';
        return `This is the ${lnName} scale vector. After normalization, each feature is multiplied by a learned scale before the shifted output is produced. These parameters are shared across tokens but applied per feature.`;
    }
    if ((lower.includes('ln1') || lower.includes('ln2')) && (lower.includes('normed') || lower.includes('normalized'))) {
        return 'This is the normalized vector for one token inside LayerNorm, before learned scale and shift are applied. It has zero mean and unit variance across features at this step.';
    }
    if ((lower.includes('ln1') || lower.includes('ln2')) && (lower.includes('shift') || lower.includes('beta'))) {
        const isLn1 = lower.includes('ln1');
        const lnName = isLn1 ? 'LN1' : 'LN2';
        return `This is the ${lnName} shift vector, the additive term applied after scaling normalized features. It lets the model re-center features after normalization.`;
    }
    if (lower.includes('final ln scale')) {
        return 'This is the final LayerNorm scale vector at the top of the model. It rescales normalized features immediately before unembedding.';
    }
    if (lower.includes('final ln shift')) {
        return 'This is the final LayerNorm shift vector at the top of the model. It is added after final scaling right before logits are computed.';
    }
    if (lower.includes('query weight matrix')) {
        return 'This matrix projects the layer-normed residual stream into query vectors for each head. Queries represent what a token is seeking, and together with keys they determine attention weights.';
    }
    if (lower.includes('key weight matrix')) {
        return 'This matrix projects the layer-normed residual stream into key vectors for each head. A key represents what a token offers to be attended to. Queries score against keys to form attention weights.';
    }
    if (lower.includes('value weight matrix')) {
        return 'This matrix projects the layer-normed residual stream into value vectors for each head. Values hold the content that gets mixed. Attention weights combine these values into head outputs.';
    }
    if (lower.includes('output projection matrix')) {
        return 'After each head produces an output, the head outputs are concatenated. This matrix projects that concatenation back to model width. It combines attention results before they are added back into the residual stream.';
    }
    if (lower.includes('mlp up weight matrix')) {
        return 'This matrix expands the model width to a larger MLP size (often 4x). It is the first linear step in the feed-forward block before the nonlinearity. This is applied independently to each token and increases nonlinear capacity.';
    }
    if (lower.includes('mlp down weight matrix')) {
        return 'This matrix compresses the expanded MLP activations back to model width. It brings nonlinear features back into the residual stream.';
    }
    if (lower.includes('mlp up projection')) {
        return 'This is the vector after the MLP up-projection, in the larger hidden dimension. It is the input to the nonlinearity before being projected back down.';
    }
    if (lower.includes('mlp down projection')) {
        return 'This is the vector after the MLP down-projection, back at model width. It will be added to the residual stream.';
    }
    if (lower.includes('mlp expanded segments')) {
        return 'This is the expanded MLP vector split into multiple segments to show the 4× width. It is one high-dimensional vector, just partitioned for visualization.';
    }
    if (lower.includes('layernorm') || lower.includes('layer norm')) {
        if (lower.includes('top')) {
            return 'This is the final LayerNorm at the top of the model. It normalizes the residual stream before the unembedding step, then applies learned scale and shift.';
        }
        if (lower.includes('scale') || lower.includes('gamma')) {
            return 'This is the LayerNorm scale vector. After normalization, each feature is multiplied by a learned scale. Values are learned per feature and shared across tokens.';
        }
        if (lower.includes('shift') || lower.includes('beta')) {
            return 'This is the LayerNorm shift vector, the additive term applied after scaling normalized features. It lets the model re-center activations and choose a useful baseline.';
        }
        if (lower.includes('normed') || lower.includes('normalized')) {
            return 'This is the normalized vector for one token. Before scale and shift, it has zero mean and unit variance across features. This stabilized vector becomes the input to the next sublayer.';
        }
        return 'LayerNorm normalizes each token\'s features and then applies learned scale and shift. It keeps activations stable and improves training. GPT-2 uses pre-LayerNorm, so this happens before attention and before the MLP in each layer.';
    }
    if (lower.includes('merged key vectors')) {
        return 'These are the key vectors from all heads stacked together. Each head uses its own keys to build its own attention matrix. This view lets you see all heads\' keys at once. Queries from the matching head compare against these keys to form the attention matrix shown below.';
    }
    if (lower.includes('merged value vectors')) {
        return 'These are the value vectors from all heads stacked together. For each head, the attention matrix shown below weights that head\'s values. This view shows all head values at once. The weighted sums become the per-head outputs.';
    }
    if (lower.includes('post-layernorm residual') || lower.includes('post layernorm residual')) {
        return 'This is the layer-normed residual stream right before query, key, and value projection. It is copied for each head to make queries, keys, and values, which then determine attention weights.';
    }
    if (lower.includes('incoming residual')) {
        return 'This is the residual stream as it enters a layer, before the first LayerNorm. It carries the accumulated information from earlier layers and will feed into attention and the MLP.';
    }
    if (lower.includes('post-attention residual')) {
        return 'This is the residual stream after attention has been added. It represents the model state before the MLP block in this layer.';
    }
    if (lower.includes('post-mlp residual')) {
        return 'This is the residual stream after the MLP output has been added. It is the final output of this layer and the input to the next layer.';
    }
    if (lower.includes('mhsa q copies')) {
        return 'These are per-head copies of the layer-normed residual stream that will be projected into queries. Each copy is paired with keys to determine attention weights.';
    }
    if (lower.includes('mhsa k copies')) {
        return 'These are per-head copies of the layer-normed residual stream that will be projected into keys. Each copy is compared with queries to determine attention weights.';
    }
    if (lower.includes('mhsa v copies')) {
        return 'These are per-head copies of the layer-normed residual stream that will be projected into values. Attention weights then mix these values into each head output.';
    }
    if (lower.includes('query vector')) {
        return 'This is the query vector for one token in one head. It encodes what the token wants to read. It is compared with all key vectors to form one row of the attention matrix shown below. Each entry in that row is a score for another token.';
    }
    if (lower.includes('cached key vector')) {
        return 'This is a key vector persisted in the KV cache from an earlier decoding step. New queries attend to it so the model can reuse prior context without recomputing past keys.';
    }
    if (lower.includes('key vector')) {
        return 'This is the key vector for one token in one head. It encodes what this token offers. Queries compare against it to compute attention scores, contributing to one column of the attention matrix shown below. Higher dot products mean this token is more attendable.';
    }
    if (lower.includes('cached value vector')) {
        return 'This is a value vector persisted in the KV cache from an earlier decoding step. Attention weights select from these cached values so prior-token information can flow into the current token update.';
    }
    if (lower.includes('weighted value vector')) {
        return 'This is one value vector multiplied by a specific post-softmax attention weight. It is a per-edge contribution that later gets accumulated into the attention weighted sum for the query token.';
    }
    if (lower.includes('value vector')) {
        return 'This is the value vector for one token in one head. It holds the information other tokens can read. The attention matrix shown below (after softmax) provides weights that mix these values. The head output is a weighted sum of values.';
    }
    if (lower.includes('attention score') || stageLower.startsWith('attention.')) {
        if (stageLower === 'attention.pre') {
            return 'This is a raw attention score from a source token to a target token in one head. It is the scaled similarity between a query and key, before normalization. A causal mask is applied so tokens cannot look ahead. Softmax later converts these scores into attention weights.';
        }
        if (stageLower === 'attention.post') {
            return 'This is a normalized attention weight after softmax. It is one entry in the attention matrix shown below for a head. Each row sums to 1 and shows how the source token distributes its attention. These weights are used to mix values.';
        }
        return 'This is an attention score between two tokens in one head. All scores together form the attention matrix shown below (queries by keys). That matrix controls how much information flows between tokens. After softmax, it becomes weights applied to values.';
    }
    if (lower.includes('attention')) {
        return 'Self-attention lets each token read information from other tokens in the sequence. Each head computes attention weights over tokens and uses them to mix value vectors. Head outputs are then concatenated, projected, and added back to the residual stream.';
    }
    if (lower.includes('attention weighted sum')) {
        return 'This is the weighted-sum output of one attention head for a token. It is produced by multiplying post-softmax attention weights with value vectors and summing across source tokens.';
    }
    if (lower.includes('top logit bars')) {
        return 'These bars come from vocabulary logits before softmax. Each bar is one candidate token, and taller means higher post-softmax probability for this next-token step. The model computes logits from the final residual stream, then softmax converts logits to probabilities for sampling or argmax.';
    }
    if (lower.startsWith('chosen token:')) {
        return 'This marks the token selected from the current logit distribution for this pass. That selected token is what gets appended for the next decode step.';
    }
    if (lower === 'logit' || lower.startsWith('logit ')) {
        return 'This is one vocabulary logit entry before softmax for a candidate next token. In this view, the shown probability is that token\'s post-softmax value.';
    }
    if (lower.includes('residual')) {
        return RESIDUAL_STREAM_DESCRIPTION;
    }
    if (lower.includes('mlp')) {
        return 'The MLP is a token-wise feed-forward network. It expands the vector, applies a nonlinearity (like GELU), then compresses back to model width. This adds nonlinear feature mixing that attention alone cannot provide. It operates independently on each token.';
    }
    if (lower.includes('vector')) {
        return 'This vector represents a token\'s state at a particular stage. It may be part of the residual stream or a derived Q/K/V vector. Its coordinates are learned features. Later layers will transform or mix it.';
    }
    if (lower.includes('weight matrix')) {
        return 'This is a learned linear transformation matrix. Multiplying by this matrix changes the feature mix or coordinate system. Different matrices serve different roles, such as creating Q/K/V vectors or MLP features. These are core trainable parameters.';
    }
    return buildContextualFallbackDescription(label, kind, stage);
}
