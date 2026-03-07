function formatHeadScopedSymbol(symbol, headSubscript = null) {
    const safeSymbol = typeof symbol === 'string' ? symbol : '';
    if (!safeSymbol) return '';
    return `${safeSymbol}_i`;
}

export function buildAttentionEquationSet(symbols = {}, options = {}) {
    const Q = typeof symbols.Q === 'string' ? symbols.Q : 'Q';
    const K = typeof symbols.K === 'string' ? symbols.K : 'K';
    const V = typeof symbols.V === 'string' ? symbols.V : 'V';
    const WQ = typeof symbols.WQ === 'string' ? symbols.WQ : 'W_Q';
    const WK = typeof symbols.WK === 'string' ? symbols.WK : 'W_K';
    const WV = typeof symbols.WV === 'string' ? symbols.WV : 'W_V';
    const BQ = typeof symbols.BQ === 'string' ? symbols.BQ : 'b_Q';
    const BK = typeof symbols.BK === 'string' ? symbols.BK : 'b_K';
    const BV = typeof symbols.BV === 'string' ? symbols.BV : 'b_V';
    const WO = typeof symbols.WO === 'string' ? symbols.WO : 'W_O';
    const BO = typeof symbols.BO === 'string' ? symbols.BO : 'b_O';
    const safeHeadSubscript = typeof options.headSubscript === 'string' && options.headSubscript.trim().length
        ? options.headSubscript.trim()
        : null;

    const qkvProjection = String.raw`\begin{aligned} ${Q} &= x_{\text{ln}} ${WQ} + ${BQ} \\ ${K} &= x_{\text{ln}} ${WK} + ${BK} \\ ${V} &= x_{\text{ln}} ${WV} + ${BV} \end{aligned}`;
    const Qh = formatHeadScopedSymbol(Q, safeHeadSubscript);
    const Kh = formatHeadScopedSymbol(K, safeHeadSubscript);
    const Vh = formatHeadScopedSymbol(V, safeHeadSubscript);
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
        qkvProjection,
        attention,
        concat,
        outputProjection,
        concatProjection,
        postAttentionResidual
    };
}
