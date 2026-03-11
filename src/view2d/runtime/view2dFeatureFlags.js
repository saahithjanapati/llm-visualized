export const VIEW2D_MHSA_CANVAS_MODES = Object.freeze({
    OFF: 'off',
    REPLACE: 'replace'
});

export function resolveMhsaTokenMatrixCanvasMode(runtimeWindow = null) {
    const source = runtimeWindow && typeof runtimeWindow === 'object'
        ? runtimeWindow
        : (typeof window !== 'undefined' ? window : null);
    const rawValue = source?.__MHSA_TOKEN_MATRIX_CANVAS_MODE;

    if (rawValue === undefined) {
        return VIEW2D_MHSA_CANVAS_MODES.OFF;
    }
    if (rawValue === true) return VIEW2D_MHSA_CANVAS_MODES.REPLACE;
    if (rawValue === false || rawValue === null) {
        return VIEW2D_MHSA_CANVAS_MODES.OFF;
    }

    const normalized = String(rawValue).trim().toLowerCase();
    if (!normalized.length) return VIEW2D_MHSA_CANVAS_MODES.OFF;
    if ([
        VIEW2D_MHSA_CANVAS_MODES.REPLACE,
        'on',
        '1',
        'true',
        'canvas'
    ].includes(normalized)) {
        return VIEW2D_MHSA_CANVAS_MODES.REPLACE;
    }
    if ([
        VIEW2D_MHSA_CANVAS_MODES.OFF,
        '0',
        'false'
    ].includes(normalized)) {
        return VIEW2D_MHSA_CANVAS_MODES.OFF;
    }
    return VIEW2D_MHSA_CANVAS_MODES.OFF;
}

export function shouldRenderMhsaTokenMatrixCanvas(mode = VIEW2D_MHSA_CANVAS_MODES.OFF) {
    return mode === VIEW2D_MHSA_CANVAS_MODES.REPLACE;
}
