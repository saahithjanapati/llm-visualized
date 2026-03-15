import { resolveMhsaTokenVisualExtent } from '../shared/mhsaDimensionSizing.js';

const EMBEDDING_STREAM_CARD_WIDTHS = Object.freeze({
    token: 196,
    position: 148,
    unembedding: 196
});

function resolveScaledExtent(tokenCount = 1, {
    isSmallScreen = false,
    baseExtentPx = 0,
    minExtentPx = 0,
    maxExtentPx = 0,
    baseTokenCount = 5,
    exponent = 0.24
} = {}) {
    return Math.round(resolveMhsaTokenVisualExtent(tokenCount, {
        isSmallScreen,
        baseTokenCount,
        exponent,
        baseExtentPx,
        minExtentPx,
        maxExtentPx
    }));
}

function resolveChipLabelFontScale(tokenCount = 1) {
    const safeTokenCount = Number.isFinite(tokenCount) ? Math.max(1, Math.floor(tokenCount)) : 1;
    if (safeTokenCount >= 18) return 0.78;
    if (safeTokenCount >= 12) return 0.86;
    if (safeTokenCount >= 8) return 0.94;
    return 1;
}

function resolveChipStackGap(tokenCount = 1, {
    isSmallScreen = false
} = {}) {
    const safeTokenCount = Number.isFinite(tokenCount) ? Math.max(1, Math.floor(tokenCount)) : 1;
    if (safeTokenCount >= 16) return 2;
    if (safeTokenCount >= 10) return 3;
    if (safeTokenCount >= 6) return 4;
    return isSmallScreen ? 4 : 6;
}

export function resolveEmbeddingStreamLayoutMetrics({
    tokenCount = 1,
    isSmallScreen = false
} = {}) {
    const safeTokenCount = Number.isFinite(tokenCount) ? Math.max(1, Math.floor(tokenCount)) : 1;
    const defaultChipHeight = isSmallScreen ? 20 : 22;
    const minChipHeight = isSmallScreen ? 8 : 9;
    const stackGap = resolveChipStackGap(safeTokenCount, {
        isSmallScreen
    });
    const labelFontScale = resolveChipLabelFontScale(safeTokenCount);

    const tokenCardHeight = resolveScaledExtent(safeTokenCount, {
        isSmallScreen,
        baseExtentPx: isSmallScreen ? 130 : 144,
        minExtentPx: isSmallScreen ? 118 : 128,
        maxExtentPx: isSmallScreen ? 206 : 232,
        exponent: 0.28
    });
    const positionCardHeight = resolveScaledExtent(safeTokenCount, {
        isSmallScreen,
        baseExtentPx: isSmallScreen ? 112 : 120,
        minExtentPx: isSmallScreen ? 100 : 108,
        maxExtentPx: isSmallScreen ? 170 : 188,
        exponent: 0.24
    });
    const unembeddingCardHeight = resolveScaledExtent(safeTokenCount, {
        isSmallScreen,
        baseExtentPx: isSmallScreen ? 132 : 146,
        minExtentPx: isSmallScreen ? 120 : 130,
        maxExtentPx: isSmallScreen ? 210 : 236,
        exponent: 0.28
    });

    const chipToCardGap = resolveScaledExtent(safeTokenCount, {
        isSmallScreen,
        baseExtentPx: isSmallScreen ? 16 : 18,
        minExtentPx: isSmallScreen ? 12 : 14,
        maxExtentPx: isSmallScreen ? 22 : 24,
        exponent: 0.14
    });
    const vocabularyToPositionGap = resolveScaledExtent(safeTokenCount, {
        isSmallScreen,
        baseExtentPx: isSmallScreen ? 44 : 56,
        minExtentPx: isSmallScreen ? 34 : 44,
        maxExtentPx: isSmallScreen ? 68 : 88,
        exponent: 0.2
    });
    const unembeddingOutputGap = resolveScaledExtent(safeTokenCount, {
        isSmallScreen,
        baseExtentPx: isSmallScreen ? 12 : 14,
        minExtentPx: isSmallScreen ? 10 : 12,
        maxExtentPx: isSmallScreen ? 18 : 20,
        exponent: 0.14
    });

    return {
        vocabulary: {
            cardWidth: EMBEDDING_STREAM_CARD_WIDTHS.token,
            cardHeight: tokenCardHeight,
            chipWidth: 118,
            chipHeight: defaultChipHeight,
            minChipHeight,
            maxStackHeight: tokenCardHeight + (isSmallScreen ? 4 : 8),
            stackGap,
            chipToCardGap,
            labelFontScale
        },
        position: {
            cardWidth: EMBEDDING_STREAM_CARD_WIDTHS.position,
            cardHeight: positionCardHeight,
            chipWidth: 118,
            chipHeight: defaultChipHeight,
            minChipHeight,
            maxStackHeight: positionCardHeight + (isSmallScreen ? 4 : 8),
            stackGap,
            chipToCardGap,
            labelFontScale
        },
        unembedding: {
            cardWidth: EMBEDDING_STREAM_CARD_WIDTHS.unembedding,
            cardHeight: unembeddingCardHeight,
            chipWidth: 118,
            chipHeight: defaultChipHeight,
            minChipHeight,
            maxStackHeight: unembeddingCardHeight + (isSmallScreen ? 4 : 8),
            stackGap,
            chipToCardGap,
            labelFontScale,
            outputGap: unembeddingOutputGap
        },
        vocabularyToPositionGap
    };
}
