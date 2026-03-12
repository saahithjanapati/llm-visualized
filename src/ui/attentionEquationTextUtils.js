function formatHeadScopedSymbol(symbol, headSubscript = null) {
    const safeSymbol = typeof symbol === 'string' ? symbol : '';
    if (!safeSymbol) return '';
    const safeSubscript = typeof headSubscript === 'string' && headSubscript.trim().length
        ? headSubscript.trim()
        : 'i';
    const colorizedTokenMatch = safeSymbol.match(/^\\textcolor\{([^}]+)\}\{(.+)\}$/);
    if (colorizedTokenMatch) {
        const [, color, body] = colorizedTokenMatch;
        return `\\textcolor{${color}}{${body}_{${safeSubscript}}}`;
    }
    return `${safeSymbol}_${safeSubscript}`;
}

function resolveAttentionEquationSymbol(symbol, fallback) {
    return typeof symbol === 'string' ? symbol : fallback;
}

export function buildAttentionProjectionEquation({
    outputSymbol = 'Q',
    inputSymbol = 'x_{\\text{ln}}',
    weightSymbol = 'W_Q',
    biasSymbol = 'b_Q',
    alignEquals = false
} = {}) {
    const output = resolveAttentionEquationSymbol(outputSymbol, 'Q');
    const input = resolveAttentionEquationSymbol(inputSymbol, 'x_{\\text{ln}}');
    const weight = resolveAttentionEquationSymbol(weightSymbol, 'W_Q');
    const bias = resolveAttentionEquationSymbol(biasSymbol, 'b_Q');
    const equals = alignEquals ? '&=' : '=';
    return `${output} ${equals} ${input} ${weight} + ${bias}`;
}

export function buildAttentionEquationSet(symbols = {}, options = {}) {
    const Q = resolveAttentionEquationSymbol(symbols.Q, 'Q');
    const K = resolveAttentionEquationSymbol(symbols.K, 'K');
    const V = resolveAttentionEquationSymbol(symbols.V, 'V');
    const WQ = resolveAttentionEquationSymbol(symbols.WQ, 'W_Q');
    const WK = resolveAttentionEquationSymbol(symbols.WK, 'W_K');
    const WV = resolveAttentionEquationSymbol(symbols.WV, 'W_V');
    const BQ = resolveAttentionEquationSymbol(symbols.BQ, 'b_Q');
    const BK = resolveAttentionEquationSymbol(symbols.BK, 'b_K');
    const BV = resolveAttentionEquationSymbol(symbols.BV, 'b_V');
    const WO = resolveAttentionEquationSymbol(symbols.WO, 'W_O');
    const BO = resolveAttentionEquationSymbol(symbols.BO, 'b_O');
    const safeHeadSubscript = typeof options.headSubscript === 'string' && options.headSubscript.trim().length
        ? options.headSubscript.trim()
        : null;
    const QHead = formatHeadScopedSymbol(Q, safeHeadSubscript);
    const KHead = formatHeadScopedSymbol(K, safeHeadSubscript);
    const VHead = formatHeadScopedSymbol(V, safeHeadSubscript);

    const queryProjection = buildAttentionProjectionEquation({
        outputSymbol: QHead,
        weightSymbol: WQ,
        biasSymbol: BQ
    });
    const keyProjection = buildAttentionProjectionEquation({
        outputSymbol: KHead,
        weightSymbol: WK,
        biasSymbol: BK
    });
    const valueProjection = buildAttentionProjectionEquation({
        outputSymbol: VHead,
        weightSymbol: WV,
        biasSymbol: BV
    });
    const qkvProjection = String.raw`\begin{aligned} ${buildAttentionProjectionEquation({
        outputSymbol: QHead,
        weightSymbol: WQ,
        biasSymbol: BQ,
        alignEquals: true
    })} \\ ${buildAttentionProjectionEquation({
        outputSymbol: KHead,
        weightSymbol: WK,
        biasSymbol: BK,
        alignEquals: true
    })} \\ ${buildAttentionProjectionEquation({
        outputSymbol: VHead,
        weightSymbol: WV,
        biasSymbol: BV,
        alignEquals: true
    })} \end{aligned}`;
    const Qh = QHead;
    const Kh = KHead;
    const Vh = VHead;
    const Hh = formatHeadScopedSymbol('H', safeHeadSubscript);
    const headQualifier = safeHeadSubscript
        ? `,\\; i=${safeHeadSubscript}`
        : ',\\; i=1\\dots 12';
    const attention = `${Hh} = \\mathrm{softmax}\\left(\\frac{${Qh} ${Kh}^\\top}{\\sqrt{d_h}} + M\\right)${Vh}${headQualifier}`;
    const concat = 'H = \\mathrm{Concat}(H_i)_{i=1}^{12}';
    const outputProjection = `O = H ${WO} + ${BO}`;
    const concatProjection = String.raw`\begin{aligned} ${concat} \\ ${outputProjection} \end{aligned}`;
    const postAttentionResidual = 'u = x + O';

    return {
        queryProjection,
        keyProjection,
        valueProjection,
        qkvProjection,
        attention,
        concat,
        outputProjection,
        concatProjection,
        postAttentionResidual
    };
}
