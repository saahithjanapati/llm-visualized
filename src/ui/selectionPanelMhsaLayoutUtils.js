import {
    resolveMhsaDimensionVisualExtent,
    resolveMhsaTokenVisualExtent
} from '../view2d/shared/mhsaDimensionSizing.js';

const BASE_ROW_COUNT = 5;

const DESKTOP_BOOST_RULES = Object.freeze({
    '--mhsa-token-matrix-stack-column-gap-boost': { perRow: 16, max: 96 },
    '--mhsa-token-matrix-stage-gap-boost': { perRow: 6, max: 36 },
    '--mhsa-token-matrix-projection-gap-boost': { perRow: 2, max: 12 },
    '--mhsa-token-matrix-attention-flow-gap-boost': { perRow: 10, max: 60 },
    '--mhsa-token-matrix-inline-gap-boost': { perRow: 2, max: 12 },
    '--mhsa-token-matrix-head-output-gap-boost': { perRow: 2, max: 12 },
    '--mhsa-token-matrix-softmax-stage-gap-boost': { perRow: 8, max: 48 },
    '--mhsa-token-matrix-head-copy-offset-boost': { perRow: 16, max: 96 }
});

const MOBILE_BOOST_RULES = Object.freeze({
    '--mhsa-token-matrix-stack-column-gap-boost': { perRow: 12, max: 72 },
    '--mhsa-token-matrix-stage-gap-boost': { perRow: 4, max: 24 },
    '--mhsa-token-matrix-projection-gap-boost': { perRow: 2, max: 10 },
    '--mhsa-token-matrix-attention-flow-gap-boost': { perRow: 7, max: 42 },
    '--mhsa-token-matrix-inline-gap-boost': { perRow: 1, max: 8 },
    '--mhsa-token-matrix-head-output-gap-boost': { perRow: 2, max: 10 },
    '--mhsa-token-matrix-softmax-stage-gap-boost': { perRow: 6, max: 36 },
    '--mhsa-token-matrix-head-copy-offset-boost': { perRow: 12, max: 72 }
});

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
    gridCellSizeMin: 1,
    gridCellGapBase: 2,
    gridCellGapDense: 1,
    gridCellGapDenseThreshold: BASE_ROW_COUNT + 3,
    gridCellGapZeroThreshold: BASE_ROW_COUNT + 5,
    gridTargetExtent: 96,
    gridTargetExtentBoostPerExtraRow: 1.5,
    gridTargetExtentBoostMax: 20,
    gridPaddingX: 2,
    gridPaddingY: 2
});

const MOBILE_DENSITY_RULES = Object.freeze({
    previewRowHeightBase: 11,
    previewRowHeightMin: 8,
    previewRowGap: 0,
    previewTargetExtent: 86,
    gridCellSizeBase: 8,
    gridCellSizeMin: 1,
    gridCellGapBase: 1,
    gridCellGapDense: 1,
    gridCellGapDenseThreshold: BASE_ROW_COUNT + 3,
    gridCellGapZeroThreshold: BASE_ROW_COUNT + 5,
    gridTargetExtent: 82,
    gridTargetExtentBoostPerExtraRow: 1.25,
    gridTargetExtentBoostMax: 16,
    gridPaddingX: 2,
    gridPaddingY: 2
});

const DESKTOP_CONNECTOR_RULES = {
    default: { base: 10, perRow: 2, maxExtra: 12 },
    projection: { base: 10, perRow: 2, maxExtra: 12 },
    transpose: { base: 16, perRow: 2, maxExtra: 12 },
    pre: { base: 10, perRow: 2, maxExtra: 12 },
    post: { base: 8, perRow: 2, maxExtra: 12 },
    value: { base: 18, perRow: 3, maxExtra: 18 }
};

const MOBILE_CONNECTOR_RULES = {
    default: { base: 10, perRow: 1.5, maxExtra: 9 },
    projection: { base: 10, perRow: 1.5, maxExtra: 9 },
    transpose: { base: 16, perRow: 1.5, maxExtra: 9 },
    pre: { base: 10, perRow: 1.5, maxExtra: 9 },
    post: { base: 8, perRow: 1.5, maxExtra: 9 },
    value: { base: 18, perRow: 2.5, maxExtra: 15 }
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
    const zeroGapThreshold = Number.isFinite(rules?.gridCellGapZeroThreshold)
        ? Math.max(BASE_ROW_COUNT, Math.floor(rules.gridCellGapZeroThreshold))
        : (BASE_ROW_COUNT + 7);
    const denseGapThreshold = Number.isFinite(rules?.gridCellGapDenseThreshold)
        ? Math.max(BASE_ROW_COUNT, Math.floor(rules.gridCellGapDenseThreshold))
        : (BASE_ROW_COUNT + 3);
    if (safeRowCount >= zeroGapThreshold) {
        return 0;
    }
    return safeRowCount >= denseGapThreshold
        ? Math.max(0, Math.floor(rules.gridCellGapDense))
        : Math.max(0, Math.floor(rules.gridCellGapBase));
}

function resolveAttentionGridTargetExtent(rowCount, {
    isSmallScreen = false,
    rules = null,
    gridPaddingX = 0
} = {}) {
    const safeRowCount = Number.isFinite(rowCount) ? Math.max(1, Math.floor(rowCount)) : BASE_ROW_COUNT;
    const safePaddingX = Math.max(0, Math.floor(gridPaddingX));
    const baseVisualExtent = resolveMhsaTokenVisualExtent(safeRowCount, {
        isSmallScreen
    });
    const extraRows = resolveExtraRows(safeRowCount);
    const boostPerExtraRow = Number.isFinite(rules?.gridTargetExtentBoostPerExtraRow)
        ? Math.max(0, Number(rules.gridTargetExtentBoostPerExtraRow))
        : 0;
    const boostMax = Number.isFinite(rules?.gridTargetExtentBoostMax)
        ? Math.max(0, Number(rules.gridTargetExtentBoostMax))
        : 0;
    const extentBoost = Math.min(boostMax, extraRows * boostPerExtraRow);
    const maxTargetExtent = Number.isFinite(rules?.gridTargetExtent)
        ? Math.max(1, Math.floor(rules.gridTargetExtent))
        : Number.POSITIVE_INFINITY;
    const boostedVisualExtent = Math.min(
        maxTargetExtent,
        baseVisualExtent + Math.round(extentBoost)
    );

    return Math.max(1, boostedVisualExtent - (safePaddingX * 2));
}

function resolveDensityMetrics(rowCount, isSmallScreen = false) {
    const safeRowCount = Number.isFinite(rowCount) ? Math.max(1, Math.floor(rowCount)) : BASE_ROW_COUNT;
    const rules = isSmallScreen ? MOBILE_DENSITY_RULES : DESKTOP_DENSITY_RULES;
    const gridCellGap = resolveDenseGridGap(safeRowCount, rules);
    const gridPaddingX = Math.max(0, Math.floor(rules.gridPaddingX || 0));
    const gridPaddingY = Math.max(0, Math.floor(rules.gridPaddingY || 0));
    const gridTargetExtent = resolveAttentionGridTargetExtent(safeRowCount, {
        isSmallScreen,
        rules,
        gridPaddingX
    });
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
        targetExtent: gridTargetExtent
    });

    return {
        previewRowHeight,
        previewRowGap: Math.max(0, Math.floor(rules.previewRowGap)),
        gridCellSize,
        gridCellGap,
        gridPaddingX,
        gridPaddingY
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
        const extra = Math.min(
            Number.isFinite(rule.maxExtra) ? Math.max(0, Number(rule.maxExtra)) : Number.POSITIVE_INFINITY,
            Math.max(0, extraRows) * Math.max(0, Number(rule.perRow || 0))
        );
        acc[key] = Math.max(0, Math.round(rule.base + extra));
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
            gridCellGap: densityMetrics.gridCellGap,
            gridPaddingX: densityMetrics.gridPaddingX,
            gridPaddingY: densityMetrics.gridPaddingY
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
