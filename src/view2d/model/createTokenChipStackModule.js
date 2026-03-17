import {
    createGroupNode,
    createMatrixNode,
    createTextNode,
    VIEW2D_LAYOUT_DIRECTIONS,
    VIEW2D_MATRIX_PRESENTATIONS,
    VIEW2D_MATRIX_SHAPES,
    VIEW2D_TEXT_PRESENTATIONS
} from '../schema/sceneTypes.js';
import { VIEW2D_STYLE_KEYS } from '../theme/visualTokens.js';
import { resolveTokenChipColors } from '../../ui/tokenChipColorUtils.js';
import {
    formatTokenChipDisplayText,
    resolveTokenChipLabel
} from '../../utils/tokenChipStyleUtils.js';

const DEFAULT_CHIP_WIDTH = 118;
const DEFAULT_CHIP_HEIGHT = 22;
const DEFAULT_STACK_MAX_HEIGHT = 132;
const DEFAULT_STACK_GAP = 6;
const MIN_CHIP_HEIGHT = 18;
const CHIP_CORNER_RADIUS = 999;
const CHIP_LABEL_HORIZONTAL_INSET = 10;
const NEUTRAL_CHIP_COLORS = Object.freeze({
    border: 'rgba(184, 191, 200, 0.92)',
    fill: 'rgba(126, 133, 144, 0.22)',
    fillHover: 'rgba(148, 156, 166, 0.30)',
    fillActive: 'rgba(168, 176, 186, 0.38)'
});

function mergeSemantic(baseSemantic = {}, extra = {}) {
    return {
        ...(baseSemantic && typeof baseSemantic === 'object' ? baseSemantic : {}),
        ...(extra && typeof extra === 'object' ? extra : {})
    };
}

function mergeMetadata(...parts) {
    const merged = parts.reduce((acc, part) => {
        if (!part || typeof part !== 'object') return acc;
        return {
            ...acc,
            ...part
        };
    }, {});
    return Object.keys(merged).length ? merged : null;
}

function createCardMetadata(width = null, height = null, {
    cornerRadius = null,
    hidden = false
} = {}) {
    const card = {};
    if (Number.isFinite(width) && width > 0) card.width = Math.floor(width);
    if (Number.isFinite(height) && height > 0) card.height = Math.floor(height);
    if (Number.isFinite(cornerRadius) && cornerRadius >= 0) card.cornerRadius = Math.floor(cornerRadius);
    const metadata = Object.keys(card).length ? { card } : {};
    if (hidden) metadata.hidden = true;
    return Object.keys(metadata).length ? metadata : null;
}

function createTextFitMetadata(maxWidth = null) {
    if (!Number.isFinite(maxWidth) || maxWidth <= 0) return null;
    return {
        textFit: {
            maxWidth: Math.floor(maxWidth)
        }
    };
}

function resolveChipStackSizing(tokenCount = 0, {
    chipHeight = DEFAULT_CHIP_HEIGHT,
    maxStackHeight = DEFAULT_STACK_MAX_HEIGHT,
    gap = DEFAULT_STACK_GAP,
    minChipHeight = MIN_CHIP_HEIGHT
} = {}) {
    const safeCount = Math.max(1, Math.floor(tokenCount || 1));
    const safeMinChipHeight = Math.max(1, Math.floor(minChipHeight || MIN_CHIP_HEIGHT));
    const safeChipHeight = Math.max(safeMinChipHeight, Math.floor(chipHeight || DEFAULT_CHIP_HEIGHT));
    const safeMaxStackHeight = Math.max(safeChipHeight, Math.floor(maxStackHeight || DEFAULT_STACK_MAX_HEIGHT));
    const preferredGap = Math.max(0, Math.floor(gap || DEFAULT_STACK_GAP));
    const maxChipHeightForStack = Math.floor(
        (safeMaxStackHeight - (preferredGap * Math.max(0, safeCount - 1))) / safeCount
    );
    const resolvedChipHeight = Math.max(
        safeMinChipHeight,
        Math.min(safeChipHeight, maxChipHeightForStack)
    );
    const remainingHeight = Math.max(0, safeMaxStackHeight - (resolvedChipHeight * safeCount));
    const resolvedGap = safeCount > 1
        ? Math.min(preferredGap, Math.floor(remainingHeight / (safeCount - 1)))
        : 0;
    return {
        chipHeight: resolvedChipHeight,
        gap: resolvedGap
    };
}

function buildTokenChipBackground(colors = null) {
    const fillHover = colors?.fillHover || 'rgba(255, 255, 255, 0.12)';
    const fillActive = colors?.fillActive || 'rgba(255, 255, 255, 0.20)';
    return `linear-gradient(135deg, rgba(12, 14, 18, 0.96) 0%, ${fillHover} 54%, ${fillActive} 100%)`;
}

export function buildTokenChipStackModule({
    semantic = {},
    tokenRefs = [],
    stackRole = 'input-token-chip-stack',
    chipRole = 'input-token-chip',
    chipLabelRole = 'input-token-chip-label',
    chipGroupRole = 'input-token-chip-group',
    textOnlySpacerRole = `${chipRole}-placeholder`,
    textOnlyLabelRole = `${chipLabelRole}-text`,
    textOnlyGroupRole = `${chipGroupRole}-text`,
    chipWidth = DEFAULT_CHIP_WIDTH,
    chipHeight = DEFAULT_CHIP_HEIGHT,
    minChipHeight = MIN_CHIP_HEIGHT,
    maxStackHeight = DEFAULT_STACK_MAX_HEIGHT,
    gap = DEFAULT_STACK_GAP,
    colorMode = 'token',
    labelFontScale = 1
} = {}) {
    const safeTokenRefs = Array.isArray(tokenRefs) ? tokenRefs.filter(Boolean) : [];
    if (!safeTokenRefs.length) {
        return null;
    }

    const resolvedChipWidth = Math.max(72, Math.floor(chipWidth || DEFAULT_CHIP_WIDTH));
    const labelMaxWidth = Math.max(24, resolvedChipWidth - (CHIP_LABEL_HORIZONTAL_INSET * 2));
    const sizing = resolveChipStackSizing(safeTokenRefs.length, {
        chipHeight,
        maxStackHeight,
        gap,
        minChipHeight
    });

    const chipEntries = safeTokenRefs.map((tokenRef, index) => {
        const tokenIndex = Number.isFinite(tokenRef?.tokenIndex)
            ? Math.floor(tokenRef.tokenIndex)
            : null;
        const tokenId = Number.isFinite(tokenRef?.tokenId)
            ? Math.floor(tokenRef.tokenId)
            : null;
        const positionIndex = Number.isFinite(tokenRef?.positionIndex)
            ? Math.max(1, Math.floor(tokenRef.positionIndex))
            : (Number.isFinite(tokenIndex) ? tokenIndex + 1 : null);
        const displayMode = String(tokenRef?.displayMode || '').trim().toLowerCase() === 'text'
            ? 'text'
            : 'chip';
        const tokenLabel = resolveTokenChipLabel(tokenRef?.tokenLabel, tokenIndex);
        const displayText = (
            typeof tokenRef?.displayText === 'string' && tokenRef.displayText.length
                ? tokenRef.displayText
                : (
                    formatTokenChipDisplayText(tokenRef?.tokenLabel, tokenIndex)
                    || tokenLabel
                    || `Token ${index + 1}`
                )
        );
        const tokenMetadata = displayMode === 'chip'
            ? mergeMetadata(
                tokenLabel.length ? { tokenLabel } : null,
                Number.isFinite(tokenId) ? { tokenId } : null,
                Number.isFinite(positionIndex) ? { positionIndex } : null
            )
            : null;

        if (displayMode === 'text') {
            const placeholderSemantic = mergeSemantic(semantic, {
                rowIndex: index
            });
            const placeholderSpacerNode = createMatrixNode({
                role: textOnlySpacerRole,
                semantic: mergeSemantic(placeholderSemantic, {
                    role: textOnlySpacerRole
                }),
                dimensions: {
                    rows: 1,
                    cols: 1
                },
                presentation: VIEW2D_MATRIX_PRESENTATIONS.CARD,
                shape: VIEW2D_MATRIX_SHAPES.MATRIX,
                visual: {
                    styleKey: VIEW2D_STYLE_KEYS.RESIDUAL
                },
                metadata: createCardMetadata(resolvedChipWidth, sizing.chipHeight, {
                    cornerRadius: CHIP_CORNER_RADIUS,
                    hidden: true
                })
            });
            const placeholderLabelNode = createTextNode({
                role: textOnlyLabelRole,
                semantic: mergeSemantic(placeholderSemantic, {
                    role: textOnlyLabelRole
                }),
                text: displayText,
                presentation: VIEW2D_TEXT_PRESENTATIONS.LABEL,
                visual: {
                    styleKey: VIEW2D_STYLE_KEYS.LABEL
                },
                metadata: mergeMetadata(
                    createTextFitMetadata(labelMaxWidth),
                    Number.isFinite(labelFontScale) && labelFontScale > 0 && labelFontScale !== 1
                        ? { fontScale: labelFontScale }
                        : null
                )
            });

            return {
                cardNode: null,
                labelNode: placeholderLabelNode,
                node: createGroupNode({
                    role: textOnlyGroupRole,
                    semantic: mergeSemantic(placeholderSemantic, {
                        role: textOnlyGroupRole
                    }),
                    direction: VIEW2D_LAYOUT_DIRECTIONS.OVERLAY,
                    gapKey: 'default',
                    children: [
                        placeholderSpacerNode,
                        placeholderLabelNode
                    ]
                })
            };
        }

        const colors = String(colorMode || '').trim().toLowerCase() === 'neutral'
            ? NEUTRAL_CHIP_COLORS
            : resolveTokenChipColors({
                tokenIndex,
                tokenId,
                tokenLabel,
                ...(Number.isFinite(tokenRef?.colorKey) ? { colorKey: Math.floor(tokenRef.colorKey) } : {}),
                ...(Number.isFinite(tokenRef?.seed) ? { seed: Math.floor(tokenRef.seed) } : {})
            }, Number.isFinite(tokenIndex) ? tokenIndex : index);
        const chipSemantic = mergeSemantic(semantic, {
            role: chipRole,
            rowIndex: index,
            ...(Number.isFinite(tokenIndex) ? { tokenIndex } : {}),
            ...(Number.isFinite(tokenId) ? { tokenId } : {}),
            ...(Number.isFinite(positionIndex) ? { positionIndex } : {})
        });

        const chipCardNode = createMatrixNode({
            role: chipRole,
            semantic: chipSemantic,
            dimensions: {
                rows: 1,
                cols: 1
            },
            presentation: VIEW2D_MATRIX_PRESENTATIONS.CARD,
            shape: VIEW2D_MATRIX_SHAPES.MATRIX,
            visual: {
                accent: colors.border,
                background: buildTokenChipBackground(colors),
                stroke: colors.border,
                disableCardSurfaceEffects: true
            },
            metadata: mergeMetadata(
                createCardMetadata(resolvedChipWidth, sizing.chipHeight, {
                    cornerRadius: CHIP_CORNER_RADIUS
                }),
                tokenMetadata
            )
        });

        const chipLabelNode = createTextNode({
            role: chipLabelRole,
            semantic: mergeSemantic(semantic, {
                role: chipLabelRole,
                rowIndex: index,
                ...(Number.isFinite(tokenIndex) ? { tokenIndex } : {}),
                ...(Number.isFinite(tokenId) ? { tokenId } : {}),
                ...(Number.isFinite(positionIndex) ? { positionIndex } : {})
            }),
            text: displayText,
            presentation: VIEW2D_TEXT_PRESENTATIONS.LABEL,
            visual: {
                styleKey: VIEW2D_STYLE_KEYS.LABEL
            },
            metadata: mergeMetadata(
                createTextFitMetadata(labelMaxWidth),
                tokenMetadata,
                Number.isFinite(labelFontScale) && labelFontScale > 0 && labelFontScale !== 1
                    ? { fontScale: labelFontScale }
                    : null
            )
        });

        return {
            cardNode: chipCardNode,
            labelNode: chipLabelNode,
            node: createGroupNode({
                role: chipGroupRole,
                semantic: mergeSemantic(semantic, {
                    role: chipGroupRole,
                    rowIndex: index,
                    ...(Number.isFinite(tokenIndex) ? { tokenIndex } : {}),
                    ...(Number.isFinite(tokenId) ? { tokenId } : {}),
                    ...(Number.isFinite(positionIndex) ? { positionIndex } : {})
                }),
                direction: VIEW2D_LAYOUT_DIRECTIONS.OVERLAY,
                gapKey: 'default',
                children: [
                    chipCardNode,
                    chipLabelNode
                ],
                metadata: tokenMetadata
            })
        };
    });

    return {
        node: createGroupNode({
            role: stackRole,
            semantic: mergeSemantic(semantic, { role: stackRole }),
            direction: VIEW2D_LAYOUT_DIRECTIONS.VERTICAL,
            gapKey: 'inline',
            children: chipEntries.map((entry) => entry.node),
            metadata: {
                gapOverride: sizing.gap
            }
        }),
        cardNodes: chipEntries.map((entry) => entry.cardNode).filter(Boolean),
        labelNodes: chipEntries.map((entry) => entry.labelNode),
        width: resolvedChipWidth
    };
}
