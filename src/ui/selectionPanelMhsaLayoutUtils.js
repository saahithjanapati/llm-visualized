const BASE_ROW_COUNT = 5;

const DESKTOP_BOOST_RULES = {
    '--mhsa-token-matrix-content-min-width-boost': { perRow: 72, max: 360 },
    '--mhsa-token-matrix-canvas-pad-x-boost': { perRow: 14, max: 72 },
    '--mhsa-token-matrix-canvas-pad-y-boost': { perRow: 8, max: 40 },
    '--mhsa-token-matrix-stack-row-gap-boost': { perRow: 4, max: 24 },
    '--mhsa-token-matrix-stack-column-gap-boost': { perRow: 16, max: 84 },
    '--mhsa-token-matrix-stage-gap-boost': { perRow: 4, max: 20 },
    '--mhsa-token-matrix-projection-gap-boost': { perRow: 2, max: 12 },
    '--mhsa-token-matrix-attention-flow-gap-boost': { perRow: 10, max: 48 },
    '--mhsa-token-matrix-inline-gap-boost': { perRow: 4, max: 20 },
    '--mhsa-token-matrix-head-output-gap-boost': { perRow: 4, max: 20 },
    '--mhsa-token-matrix-softmax-stage-gap-boost': { perRow: 4, max: 20 },
    '--mhsa-token-matrix-head-copy-offset-boost': { perRow: 16, max: 84 }
};

const MOBILE_BOOST_RULES = {
    '--mhsa-token-matrix-content-min-width-boost': { perRow: 48, max: 240 },
    '--mhsa-token-matrix-canvas-pad-x-boost': { perRow: 10, max: 48 },
    '--mhsa-token-matrix-canvas-pad-y-boost': { perRow: 6, max: 28 },
    '--mhsa-token-matrix-stack-row-gap-boost': { perRow: 3, max: 18 },
    '--mhsa-token-matrix-stack-column-gap-boost': { perRow: 12, max: 56 },
    '--mhsa-token-matrix-stage-gap-boost': { perRow: 3, max: 14 },
    '--mhsa-token-matrix-projection-gap-boost': { perRow: 1, max: 6 },
    '--mhsa-token-matrix-attention-flow-gap-boost': { perRow: 6, max: 28 },
    '--mhsa-token-matrix-inline-gap-boost': { perRow: 2, max: 10 },
    '--mhsa-token-matrix-head-output-gap-boost': { perRow: 2, max: 10 },
    '--mhsa-token-matrix-softmax-stage-gap-boost': { perRow: 2, max: 10 },
    '--mhsa-token-matrix-head-copy-offset-boost': { perRow: 10, max: 52 }
};

const DESKTOP_CONNECTOR_RULES = {
    default: { base: 10, perRow: 2, maxBoost: 10 },
    projection: { base: 10, perRow: 2, maxBoost: 10 },
    transpose: { base: 16, perRow: 2, maxBoost: 12 },
    pre: { base: 10, perRow: 2, maxBoost: 10 },
    post: { base: 8, perRow: 2, maxBoost: 10 },
    value: { base: 18, perRow: 3, maxBoost: 14 }
};

const MOBILE_CONNECTOR_RULES = {
    default: { base: 10, perRow: 1, maxBoost: 6 },
    projection: { base: 10, perRow: 1, maxBoost: 6 },
    transpose: { base: 16, perRow: 1, maxBoost: 6 },
    pre: { base: 10, perRow: 1, maxBoost: 6 },
    post: { base: 8, perRow: 1, maxBoost: 6 },
    value: { base: 18, perRow: 2, maxBoost: 10 }
};

const LAYOUT_VAR_NAMES = Object.freeze([
    ...Object.keys(DESKTOP_BOOST_RULES),
    ...Object.keys(MOBILE_BOOST_RULES).filter((name) => !Object.prototype.hasOwnProperty.call(DESKTOP_BOOST_RULES, name))
]);

function resolveExtraRows(rowCount) {
    const safeRowCount = Number.isFinite(rowCount) ? Math.max(1, Math.floor(rowCount)) : BASE_ROW_COUNT;
    return Math.max(0, safeRowCount - BASE_ROW_COUNT);
}

function resolveCssVars(extraRows, rules) {
    return Object.entries(rules).reduce((acc, [cssVarName, rule]) => {
        const boost = Math.min(rule.max, extraRows * rule.perRow);
        acc[cssVarName] = `${Math.max(0, Math.round(boost))}px`;
        return acc;
    }, {});
}

function resolveConnectorGaps(extraRows, rules) {
    return Object.entries(rules).reduce((acc, [key, rule]) => {
        const boost = Math.min(rule.maxBoost, extraRows * rule.perRow);
        acc[key] = Math.max(0, Math.round(rule.base + boost));
        return acc;
    }, {});
}

export function resolveMhsaTokenMatrixLayoutMetrics({
    rowCount = BASE_ROW_COUNT,
    isSmallScreen = false
} = {}) {
    const safeRowCount = Number.isFinite(rowCount) ? Math.max(1, Math.floor(rowCount)) : BASE_ROW_COUNT;
    const extraRows = resolveExtraRows(safeRowCount);
    const cssRules = isSmallScreen ? MOBILE_BOOST_RULES : DESKTOP_BOOST_RULES;
    const connectorRules = isSmallScreen ? MOBILE_CONNECTOR_RULES : DESKTOP_CONNECTOR_RULES;

    return {
        rowCount: safeRowCount,
        extraRows,
        cssVars: resolveCssVars(extraRows, cssRules),
        connectorGaps: resolveConnectorGaps(extraRows, connectorRules)
    };
}

export function applyMhsaTokenMatrixLayoutVars(targetEl, metrics = null) {
    if (!targetEl?.style) return;
    clearMhsaTokenMatrixLayoutVars(targetEl);
    const cssVars = metrics?.cssVars;
    if (!cssVars || typeof cssVars !== 'object') return;

    Object.entries(cssVars).forEach(([name, value]) => {
        if (typeof value === 'string' && value.length) {
            targetEl.style.setProperty(name, value);
        }
    });
}

export function clearMhsaTokenMatrixLayoutVars(targetEl) {
    if (!targetEl?.style) return;
    LAYOUT_VAR_NAMES.forEach((name) => {
        targetEl.style.removeProperty(name);
    });
}
