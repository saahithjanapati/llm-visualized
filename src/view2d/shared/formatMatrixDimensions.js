function normalizeDimension(value, fallback = 1) {
    return Number.isFinite(value) && value > 0
        ? Math.max(1, Math.floor(value))
        : fallback;
}

export function formatView2dMatrixDimensions(rows = 1, cols = 1) {
    const safeRows = normalizeDimension(rows);
    const safeCols = normalizeDimension(cols);
    const text = `(${safeRows}, ${safeCols})`;
    return {
        tex: text,
        text
    };
}
