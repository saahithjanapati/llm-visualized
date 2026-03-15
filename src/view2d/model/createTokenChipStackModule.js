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
    cornerRadius = null
} = {}) {
    const card = {};
    if (Number.isFinite(width) && width > 0) card.width = Math.floor(width);
    if (Number.isFinite(height) && height > 0) card.height = Math.floor(height);
    if (Number.isFinite(cornerRadius) && cornerRadius >= 0) card.cornerRadius = Math.floor(cornerRadius);
    return Object.keys(card).length ? { card } : null;
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
    gap = DEFAULT_STACK_GAP
} = {}) {
    const safeCount = Math.max(1, Math.floor(tokenCount || 1));
    const safeChipHeight = Math.max(MIN_CHIP_HEIGHT, Math.floor(chipHeight || DEFAULT_CHIP_HEIGHT));
    const safeMaxStackHeight = Math.max(safeChipHeight, Math.floor(maxStackHeight || DEFAULT_STACK_MAX_HEIGHT));
    const preferredGap = Math.max(0, Math.floor(gap || DEFAULT_STACK_GAP));
    const maxChipHeightForStack = Math.floor(
        (safeMaxStackHeight - (preferredGap * Math.max(0, safeCount - 1))) / safeCount
    );
    const resolvedChipHeight = Math.max(
        MIN_CHIP_HEIGHT,
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
    chipWidth = DEFAULT_CHIP_WIDTH,
    chipHeight = DEFAULT_CHIP_HEIGHT,
    maxStackHeight = DEFAULT_STACK_MAX_HEIGHT,
    gap = DEFAULT_STACK_GAP
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
        gap
    });

    const chipEntries = safeTokenRefs.map((tokenRef, index) => {
        const tokenIndex = Number.isFinite(tokenRef?.tokenIndex)
            ? Math.floor(tokenRef.tokenIndex)
            : null;
        const positionIndex = Number.isFinite(tokenRef?.positionIndex)
            ? Math.max(1, Math.floor(tokenRef.positionIndex))
            : (Number.isFinite(tokenIndex) ? tokenIndex + 1 : null);
        const tokenLabel = resolveTokenChipLabel(tokenRef?.tokenLabel, tokenIndex);
        const displayText = formatTokenChipDisplayText(tokenRef?.tokenLabel, tokenIndex)
            || tokenLabel
            || `Token ${index + 1}`;
        const colors = resolveTokenChipColors({
            tokenIndex,
            tokenLabel
        }, Number.isFinite(tokenIndex) ? tokenIndex : index);
        const tokenMetadata = mergeMetadata(
            tokenLabel.length ? { tokenLabel } : null,
            Number.isFinite(positionIndex) ? { positionIndex } : null
        );
        const chipSemantic = mergeSemantic(semantic, {
            role: chipRole,
            rowIndex: index,
            ...(Number.isFinite(tokenIndex) ? { tokenIndex } : {}),
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
                ...(Number.isFinite(positionIndex) ? { positionIndex } : {})
            }),
            text: displayText,
            presentation: VIEW2D_TEXT_PRESENTATIONS.LABEL,
            visual: {
                styleKey: VIEW2D_STYLE_KEYS.LABEL
            },
            metadata: mergeMetadata(
                createTextFitMetadata(labelMaxWidth),
                tokenMetadata
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
        cardNodes: chipEntries.map((entry) => entry.cardNode),
        labelNodes: chipEntries.map((entry) => entry.labelNode),
        width: resolvedChipWidth
    };
}
