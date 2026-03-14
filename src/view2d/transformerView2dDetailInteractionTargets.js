import {
    findUserDataNumber
} from '../ui/selectionPanelSelectionUtils.js';
import { resolveMhsaDetailHoverState } from './mhsaDetailInteraction.js';

const DETAIL_INTERACTION_KINDS = Object.freeze({
    NODE: 'node',
    ROW: 'row',
    COLUMN: 'column',
    CELL: 'cell'
});

const DETAIL_INTERACTION_ROW_ROLES = new Set([
    'projection-output',
    'attention-query-source',
    'attention-value-post',
    'attention-head-output',
    'projection-source-xln',
    'head-output-matrix'
]);

const DETAIL_INTERACTION_COLUMN_ROLES = new Set([
    'attention-key-transpose'
]);

const DETAIL_INTERACTION_CELL_ROLES = new Set([
    'attention-pre-score',
    'attention-masked-input',
    'attention-mask',
    'attention-post',
    'attention-post-copy'
]);

function normalizeIndex(value = null) {
    return Number.isFinite(value) ? Math.max(0, Math.floor(value)) : null;
}

function normalizeSemanticTarget(target = null) {
    if (!target || typeof target !== 'object') return null;
    const normalized = Object.entries(target).reduce((acc, [key, value]) => {
        if (!key) return acc;
        if (typeof value === 'number') {
            if (Number.isFinite(value)) {
                acc[key] = Math.floor(value);
            }
            return acc;
        }
        if (typeof value === 'string') {
            const safeValue = value.trim();
            if (safeValue.length) {
                acc[key] = safeValue;
            }
            return acc;
        }
        if (typeof value === 'boolean') {
            acc[key] = value;
        }
        return acc;
    }, {});
    return Object.keys(normalized).length ? normalized : null;
}

function buildDetailInteractionTarget({
    kind = DETAIL_INTERACTION_KINDS.NODE,
    semanticTarget = null,
    tokenIndex = null,
    queryTokenIndex = null,
    keyTokenIndex = null,
    rowIndex = null,
    colIndex = null
} = {}) {
    const normalizedSemanticTarget = normalizeSemanticTarget(semanticTarget);
    if (!normalizedSemanticTarget) return null;
    const normalizedKind = Object.values(DETAIL_INTERACTION_KINDS).includes(kind)
        ? kind
        : DETAIL_INTERACTION_KINDS.NODE;
    return {
        kind: normalizedKind,
        semanticTarget: normalizedSemanticTarget,
        ...(Number.isFinite(tokenIndex) ? { tokenIndex: normalizeIndex(tokenIndex) } : {}),
        ...(Number.isFinite(queryTokenIndex) ? { queryTokenIndex: normalizeIndex(queryTokenIndex) } : {}),
        ...(Number.isFinite(keyTokenIndex) ? { keyTokenIndex: normalizeIndex(keyTokenIndex) } : {}),
        ...(Number.isFinite(rowIndex) ? { rowIndex: normalizeIndex(rowIndex) } : {}),
        ...(Number.isFinite(colIndex) ? { colIndex: normalizeIndex(colIndex) } : {})
    };
}

export function normalizeTransformerView2dDetailInteractionTargets(targets = null) {
    const sourceTargets = Array.isArray(targets)
        ? targets
        : (targets ? [targets] : []);
    const seen = new Set();
    return sourceTargets.reduce((acc, target) => {
        const normalizedTarget = buildDetailInteractionTarget(target);
        if (!normalizedTarget) return acc;
        const key = JSON.stringify(normalizedTarget);
        if (seen.has(key)) return acc;
        seen.add(key);
        acc.push(normalizedTarget);
        return acc;
    }, []);
}

function resolveInteractionKind(role = '', {
    tokenIndex = null,
    queryTokenIndex = null,
    keyTokenIndex = null
} = {}) {
    const safeRole = String(role || '').trim().toLowerCase();
    if (
        DETAIL_INTERACTION_CELL_ROLES.has(safeRole)
        && Number.isFinite(queryTokenIndex)
        && Number.isFinite(keyTokenIndex)
    ) {
        return DETAIL_INTERACTION_KINDS.CELL;
    }
    if (DETAIL_INTERACTION_ROW_ROLES.has(safeRole) && Number.isFinite(tokenIndex)) {
        return DETAIL_INTERACTION_KINDS.ROW;
    }
    if (DETAIL_INTERACTION_COLUMN_ROLES.has(safeRole) && Number.isFinite(tokenIndex)) {
        return DETAIL_INTERACTION_KINDS.COLUMN;
    }
    return DETAIL_INTERACTION_KINDS.NODE;
}

export function resolveTransformerView2dDetailInteractionTargets(selectionInfo = null, detailSemanticTargets = null) {
    const semanticTargets = Array.isArray(detailSemanticTargets)
        ? detailSemanticTargets
        : (detailSemanticTargets ? [detailSemanticTargets] : []);
    if (!semanticTargets.length) return [];

    const tokenIndex = normalizeIndex(findUserDataNumber(selectionInfo, 'tokenIndex'));
    const queryTokenIndexRaw = normalizeIndex(findUserDataNumber(selectionInfo, 'queryTokenIndex'));
    const keyTokenIndex = normalizeIndex(findUserDataNumber(selectionInfo, 'keyTokenIndex'));
    const queryTokenIndex = Number.isFinite(queryTokenIndexRaw) ? queryTokenIndexRaw : tokenIndex;

    return normalizeTransformerView2dDetailInteractionTargets(
        semanticTargets.map((semanticTarget) => {
            const normalizedSemanticTarget = normalizeSemanticTarget(semanticTarget);
            const role = String(normalizedSemanticTarget?.role || '').trim();
            return buildDetailInteractionTarget({
                kind: resolveInteractionKind(role, {
                    tokenIndex,
                    queryTokenIndex,
                    keyTokenIndex
                }),
                semanticTarget: normalizedSemanticTarget,
                tokenIndex,
                queryTokenIndex,
                keyTokenIndex
            });
        })
    );
}

function semanticValuesMatch(left = null, right = null) {
    const leftIsNumber = typeof left === 'number' && Number.isFinite(left);
    const rightIsNumber = typeof right === 'number' && Number.isFinite(right);
    if (leftIsNumber || rightIsNumber) {
        return normalizeIndex(left) === normalizeIndex(right);
    }
    return String(left ?? '').trim() === String(right ?? '').trim();
}

function nodeMatchesSemanticTarget(node = null, semanticTarget = null) {
    if (!node || !semanticTarget || typeof semanticTarget !== 'object') return false;
    return Object.entries(semanticTarget).every(([key, value]) => {
        if (value === null || value === undefined || value === '') return true;
        return semanticValuesMatch(node?.semantic?.[key], value);
    });
}

function findMatchingNode(index = null, semanticTarget = null) {
    const normalizedSemanticTarget = normalizeSemanticTarget(semanticTarget);
    if (!index?.nodesById || !normalizedSemanticTarget) return null;
    for (const node of index.nodesById.values()) {
        if (nodeMatchesSemanticTarget(node, normalizedSemanticTarget)) {
            return node;
        }
    }
    return null;
}

function findRowIndexByTokenIndex(rowItems = [], tokenIndex = null) {
    if (!Array.isArray(rowItems) || !Number.isFinite(tokenIndex)) return null;
    const matchIndex = rowItems.findIndex((rowItem, index) => {
        const semanticTokenIndex = normalizeIndex(rowItem?.semantic?.tokenIndex);
        const itemTokenIndex = normalizeIndex(rowItem?.tokenIndex);
        const itemIndex = normalizeIndex(rowItem?.index);
        return semanticTokenIndex === tokenIndex
            || itemTokenIndex === tokenIndex
            || itemIndex === tokenIndex
            || normalizeIndex(index) === tokenIndex;
    });
    return matchIndex >= 0 ? matchIndex : null;
}

function findColumnIndexByTokenIndex(columnItems = [], tokenIndex = null) {
    if (!Array.isArray(columnItems) || !Number.isFinite(tokenIndex)) return null;
    const matchIndex = columnItems.findIndex((columnItem, index) => {
        const semanticTokenIndex = normalizeIndex(columnItem?.semantic?.tokenIndex);
        const itemTokenIndex = normalizeIndex(columnItem?.tokenIndex);
        const itemIndex = normalizeIndex(columnItem?.index);
        return semanticTokenIndex === tokenIndex
            || itemTokenIndex === tokenIndex
            || itemIndex === tokenIndex
            || normalizeIndex(index) === tokenIndex;
    });
    return matchIndex >= 0 ? matchIndex : null;
}

function buildRowHit(node = null, target = null) {
    if (!node || !Array.isArray(node.rowItems) || !node.rowItems.length) return null;
    const rowIndex = Number.isFinite(target?.rowIndex)
        ? normalizeIndex(target.rowIndex)
        : findRowIndexByTokenIndex(node.rowItems, normalizeIndex(target?.tokenIndex));
    if (!Number.isFinite(rowIndex) || rowIndex >= node.rowItems.length) return null;
    return {
        node,
        rowHit: {
            rowIndex,
            rowItem: node.rowItems[rowIndex] || null
        }
    };
}

function buildColumnHit(node = null, target = null) {
    if (!node || !Array.isArray(node.columnItems) || !node.columnItems.length) return null;
    const colIndex = Number.isFinite(target?.colIndex)
        ? normalizeIndex(target.colIndex)
        : findColumnIndexByTokenIndex(node.columnItems, normalizeIndex(target?.tokenIndex));
    if (!Number.isFinite(colIndex) || colIndex >= node.columnItems.length) return null;
    return {
        node,
        columnHit: {
            colIndex,
            columnItem: node.columnItems[colIndex] || null
        }
    };
}

function findCellIndices(node = null, target = null) {
    if (!node || !Array.isArray(node.rowItems) || !node.rowItems.length) return null;
    const explicitRowIndex = normalizeIndex(target?.rowIndex);
    const explicitColIndex = normalizeIndex(target?.colIndex);
    if (Number.isFinite(explicitRowIndex) && Number.isFinite(explicitColIndex)) {
        return {
            rowIndex: explicitRowIndex,
            colIndex: explicitColIndex
        };
    }

    const queryTokenIndex = Number.isFinite(target?.queryTokenIndex)
        ? normalizeIndex(target.queryTokenIndex)
        : normalizeIndex(target?.tokenIndex);
    const keyTokenIndex = normalizeIndex(target?.keyTokenIndex);
    if (!Number.isFinite(queryTokenIndex) || !Number.isFinite(keyTokenIndex)) {
        return null;
    }

    for (let rowIndex = 0; rowIndex < node.rowItems.length; rowIndex += 1) {
        const rowItem = node.rowItems[rowIndex];
        const rowTokenIndex = normalizeIndex(rowItem?.semantic?.tokenIndex);
        const cells = Array.isArray(rowItem?.cells) ? rowItem.cells : [];
        for (let colIndex = 0; colIndex < cells.length; colIndex += 1) {
            const cellItem = cells[colIndex];
            const cellQueryTokenIndex = normalizeIndex(cellItem?.queryTokenIndex);
            const cellKeyTokenIndex = normalizeIndex(cellItem?.keyTokenIndex);
            const queryMatches = rowTokenIndex === queryTokenIndex || cellQueryTokenIndex === queryTokenIndex;
            if (!queryMatches || cellKeyTokenIndex !== keyTokenIndex) continue;
            return {
                rowIndex,
                colIndex
            };
        }
    }

    return null;
}

function buildCellHit(node = null, target = null) {
    const cellIndices = findCellIndices(node, target);
    if (!cellIndices) return null;
    const rowItem = node?.rowItems?.[cellIndices.rowIndex] || null;
    const cellItem = Array.isArray(rowItem?.cells)
        ? rowItem.cells[cellIndices.colIndex] || null
        : null;
    if (!cellItem) return null;
    return {
        node,
        cellHit: {
            rowIndex: cellIndices.rowIndex,
            colIndex: cellIndices.colIndex,
            cellItem
        }
    };
}

function buildInteractionHit(index = null, target = null) {
    const normalizedTarget = buildDetailInteractionTarget(target);
    if (!normalizedTarget) return null;
    const node = findMatchingNode(index, normalizedTarget.semanticTarget);
    if (!node) return null;

    if (normalizedTarget.kind === DETAIL_INTERACTION_KINDS.ROW) {
        return buildRowHit(node, normalizedTarget);
    }
    if (normalizedTarget.kind === DETAIL_INTERACTION_KINDS.COLUMN) {
        return buildColumnHit(node, normalizedTarget);
    }
    if (normalizedTarget.kind === DETAIL_INTERACTION_KINDS.CELL) {
        return buildCellHit(node, normalizedTarget);
    }
    return { node };
}

export function resolveTransformerView2dDetailInteractionHoverState(index = null, targets = null, options = null) {
    const normalizedTargets = normalizeTransformerView2dDetailInteractionTargets(targets);
    for (const target of normalizedTargets) {
        const hit = buildInteractionHit(index, target);
        if (!hit) continue;
        const hoverState = resolveMhsaDetailHoverState(index, hit, options);
        if (hoverState?.focusState) {
            return hoverState;
        }
    }
    return null;
}
