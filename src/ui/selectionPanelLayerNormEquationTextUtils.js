const LAYER_NORM_EQUATION_MUTED_SURFACE_COLOR = '#141417';
const LAYER_NORM_EQUATION_MUTED_ALPHA = 0.42;
const LAYER_NORM_EQUATION_HIGHLIGHT_COLOR = '#ffffff';

function blendLayerNormEquationColor(backgroundHex, foregroundHex, foregroundAlpha = 1) {
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

const LAYER_NORM_EQUATION_BASE_COLOR = blendLayerNormEquationColor(
    LAYER_NORM_EQUATION_MUTED_SURFACE_COLOR,
    '#ffffff',
    LAYER_NORM_EQUATION_MUTED_ALPHA
);

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
