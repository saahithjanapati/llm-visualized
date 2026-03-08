const LAYER_NORM_EQUATION_BASE_COLOR = '#8f98a6';
const LAYER_NORM_EQUATION_HIGHLIGHT_COLOR = '#ffffff';

function colorizeLayerNormEquationToken(color, token) {
    const safeColor = typeof color === 'string' && color.trim().length
        ? color.trim()
        : LAYER_NORM_EQUATION_BASE_COLOR;
    const safeToken = typeof token === 'string' ? token : '';
    return `\\textcolor{${safeColor}}{${safeToken}}`;
}

export function buildSelectionLayerNormEquation({
    inputSymbol = 'x',
    outputSymbol = 'x_{\\text{ln}}',
    highlight = 'norm',
    baseColor = LAYER_NORM_EQUATION_BASE_COLOR,
    highlightColor = LAYER_NORM_EQUATION_HIGHLIGHT_COLOR
} = {}) {
    const safeHighlight = highlight === 'scale' || highlight === 'shift' || highlight === 'output'
        ? highlight
        : 'norm';
    const colorize = (key, token) => colorizeLayerNormEquationToken(
        key === safeHighlight ? highlightColor : baseColor,
        token
    );

    return [
        colorize('output', outputSymbol),
        colorizeLayerNormEquationToken(baseColor, '='),
        colorize('scale', '\\gamma'),
        colorizeLayerNormEquationToken(baseColor, '\\odot'),
        colorize('norm', `\\frac{${inputSymbol} - \\mu}{\\sqrt{\\sigma^2 + \\epsilon}}`),
        colorizeLayerNormEquationToken(baseColor, '+'),
        colorize('shift', '\\beta')
    ].join(' ');
}
