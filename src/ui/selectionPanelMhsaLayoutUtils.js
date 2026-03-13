const BASE_ROW_COUNT = 5;

const DESKTOP_BOOST_RULES = {};

const MOBILE_BOOST_RULES = {};

const DENSITY_CSS_VAR_NAMES = Object.freeze({
    rowHeight: '--mhsa-token-row-height',
    rowGap: '--mhsa-token-row-gap'
});

const DESKTOP_DENSITY_RULES = Object.freeze({
    previewRowHeightBase: 13,
    previewRowHeightMin: 10,
    previewRowGap: 0,
    previewTargetExtent: 104,
    gridCellSizeBase: 9,
    gridCellSizeMin: 6,
    gridCellGapBase: 2,
    gridCellGapDense: 1,
    gridTargetExtent: 96
});

const MOBILE_DENSITY_RULES = Object.freeze({
    previewRowHeightBase: 11,
    previewRowHeightMin: 8,
    previewRowGap: 0,
    previewTargetExtent: 86,
    gridCellSizeBase: 8,
    gridCellSizeMin: 5,
    gridCellGapBase: 1,
    gridCellGapDense: 1,
    gridTargetExtent: 82
});

const DESKTOP_CONNECTOR_RULES = {
    default: { base: 10 },
    projection: { base: 10 },
    transpose: { base: 16 },
    pre: { base: 10 },
    post: { base: 8 },
    value: { base: 18 }
};

const MOBILE_CONNECTOR_RULES = {
    default: { base: 10 },
    projection: { base: 10 },
    transpose: { base: 16 },
    pre: { base: 10 },
    post: { base: 8 },
    value: { base: 18 }
};

const LAYOUT_VAR_NAMES = Object.freeze([
    ...Object.values(DENSITY_CSS_VAR_NAMES),
    ...Object.keys(DESKTOP_BOOST_RULES),
    ...Object.keys(MOBILE_BOOST_RULES).filter((name) => !Object.prototype.hasOwnProperty.call(DESKTOP_BOOST_RULES, name))
]);

function clampNumber(value, min, max) {
    return Math.min(max, Math.max(min, value));
}

function resolveAdaptiveUnitSize(rowCount, {
    baseSize = 1,
    minSize = 1,
    gap = 0,
    targetExtent = null
} = {}) {
    const safeRowCount = Number.isFinite(rowCount) ? Math.max(1, Math.floor(rowCount)) : BASE_ROW_COUNT;
    const safeBaseSize = Math.max(1, Math.floor(baseSize));
    const safeMinSize = Math.max(1, Math.min(safeBaseSize, Math.floor(minSize)));
    const safeGap = Math.max(0, Math.floor(gap));
    const safeTargetExtent = Number.isFinite(targetExtent)
        ? Math.max(1, Math.floor(targetExtent))
        : ((safeRowCount * safeBaseSize) + (Math.max(0, safeRowCount - 1) * safeGap));
    const fittedSize = Math.floor(
        (safeTargetExtent - (Math.max(0, safeRowCount - 1) * safeGap)) / safeRowCount
    );
    return clampNumber(fittedSize, safeMinSize, safeBaseSize);
}

function resolveDenseGridGap(rowCount, rules) {
    const safeRowCount = Number.isFinite(rowCount) ? Math.max(1, Math.floor(rowCount)) : BASE_ROW_COUNT;
    return safeRowCount >= BASE_ROW_COUNT + 3
        ? Math.max(0, Math.floor(rules.gridCellGapDense))
        : Math.max(0, Math.floor(rules.gridCellGapBase));
}

function resolveDensityMetrics(rowCount, isSmallScreen = false) {
    const safeRowCount = Number.isFinite(rowCount) ? Math.max(1, Math.floor(rowCount)) : BASE_ROW_COUNT;
    const rules = isSmallScreen ? MOBILE_DENSITY_RULES : DESKTOP_DENSITY_RULES;
    const gridCellGap = resolveDenseGridGap(safeRowCount, rules);
    const previewRowHeight = resolveAdaptiveUnitSize(safeRowCount, {
        baseSize: rules.previewRowHeightBase,
        minSize: rules.previewRowHeightMin,
        gap: rules.previewRowGap,
        targetExtent: rules.previewTargetExtent
    });
    const gridCellSize = resolveAdaptiveUnitSize(safeRowCount, {
        baseSize: rules.gridCellSizeBase,
        minSize: rules.gridCellSizeMin,
        gap: gridCellGap,
        targetExtent: rules.gridTargetExtent
    });

    return {
        previewRowHeight,
        previewRowGap: Math.max(0, Math.floor(rules.previewRowGap)),
        gridCellSize,
        gridCellGap
    };
}

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
        acc[key] = Math.max(0, Math.round(rule.base));
        return acc;
    }, {});
}

function resolveDensityCssVars(densityMetrics = null) {
    return {
        [DENSITY_CSS_VAR_NAMES.rowHeight]: `${Math.max(1, Math.floor(densityMetrics?.previewRowHeight || 0))}px`,
        [DENSITY_CSS_VAR_NAMES.rowGap]: `${Math.max(0, Math.floor(densityMetrics?.previewRowGap || 0))}px`
    };
}

export function resolveMhsaTokenMatrixLayoutMetrics({
    rowCount = BASE_ROW_COUNT,
    isSmallScreen = false
} = {}) {
    const safeRowCount = Number.isFinite(rowCount) ? Math.max(1, Math.floor(rowCount)) : BASE_ROW_COUNT;
    const extraRows = resolveExtraRows(safeRowCount);
    const cssRules = isSmallScreen ? MOBILE_BOOST_RULES : DESKTOP_BOOST_RULES;
    const connectorRules = isSmallScreen ? MOBILE_CONNECTOR_RULES : DESKTOP_CONNECTOR_RULES;
    const densityMetrics = resolveDensityMetrics(safeRowCount, isSmallScreen);

    return {
        rowCount: safeRowCount,
        extraRows,
        cssVars: {
            ...resolveDensityCssVars(densityMetrics),
            ...resolveCssVars(extraRows, cssRules)
        },
        connectorGaps: resolveConnectorGaps(extraRows, connectorRules),
        componentOverrides: {
            gridCellSize: densityMetrics.gridCellSize,
            gridCellGap: densityMetrics.gridCellGap
        }
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
