import { SCENE_FOCUS_BASE_CACHE_KEY } from './sceneFocusState.js';

export function appendUnique(target, value) {
    if (typeof value !== 'string' || !value.length) return;
    if (!target.includes(value)) {
        target.push(value);
    }
}

export function appendAllUnique(target, values = []) {
    values.forEach((value) => appendUnique(target, value));
}

function collectUniqueIds(values = []) {
    const seen = new Set();
    const normalized = [];
    values.forEach((value) => {
        if (typeof value !== 'string' || !value.length || seen.has(value)) return;
        seen.add(value);
        normalized.push(value);
    });
    return {
        seen,
        normalized
    };
}

function collectSelections(items = [], key = 'rowIndex') {
    const seen = new Set();
    const buckets = new Map();
    const normalized = [];
    items.forEach((selection) => {
        if (!selection || typeof selection !== 'object') return;
        const nodeId = typeof selection.nodeId === 'string' ? selection.nodeId : '';
        const indexValue = Number.isFinite(selection[key]) ? Math.max(0, Math.floor(selection[key])) : null;
        if (!nodeId.length || !Number.isFinite(indexValue)) return;
        const signature = `${nodeId}:${indexValue}`;
        if (seen.has(signature)) return;
        seen.add(signature);
        normalized.push({
            nodeId,
            [key]: indexValue
        });
        let bucket = buckets.get(nodeId);
        if (!bucket) {
            bucket = new Set();
            buckets.set(nodeId, bucket);
        }
        bucket.add(indexValue);
    });
    return {
        normalized,
        buckets
    };
}

function collectCellSelections(items = []) {
    const seen = new Set();
    const buckets = new Map();
    const normalized = [];
    items.forEach((selection) => {
        if (!selection || typeof selection !== 'object') return;
        const nodeId = typeof selection.nodeId === 'string' ? selection.nodeId : '';
        const rowIndex = Number.isFinite(selection.rowIndex) ? Math.max(0, Math.floor(selection.rowIndex)) : null;
        const colIndex = Number.isFinite(selection.colIndex) ? Math.max(0, Math.floor(selection.colIndex)) : null;
        if (!nodeId.length || !Number.isFinite(rowIndex) || !Number.isFinite(colIndex)) return;
        const signature = `${nodeId}:${rowIndex}:${colIndex}`;
        if (seen.has(signature)) return;
        seen.add(signature);
        normalized.push({
            nodeId,
            rowIndex,
            colIndex
        });
        let bucket = buckets.get(nodeId);
        if (!bucket) {
            bucket = new Set();
            buckets.set(nodeId, bucket);
        }
        bucket.add(`${rowIndex}:${colIndex}`);
    });
    return {
        normalized,
        buckets
    };
}

function appendUniqueId(target = [], seen = new Set(), value = '') {
    if (typeof value !== 'string' || !value.length || seen.has(value)) return;
    seen.add(value);
    target.push(value);
}

function buildFocusSignature({
    activeNodeIds = [],
    activeConnectorIds = [],
    dimNodeIds = [],
    rowSelections = [],
    columnSelections = [],
    cellSelections = []
} = {}) {
    return [
        `n:${activeNodeIds.join(',')}`,
        `c:${activeConnectorIds.join(',')}`,
        `d:${dimNodeIds.join(',')}`,
        `r:${rowSelections.map((entry) => `${entry.nodeId}:${entry.rowIndex}`).join(',')}`,
        `k:${columnSelections.map((entry) => `${entry.nodeId}:${entry.colIndex}`).join(',')}`,
        `g:${cellSelections.map((entry) => `${entry.nodeId}:${entry.rowIndex}:${entry.colIndex}`).join(',')}`
    ].join('|');
}

export function buildFocusResult({
    label = '',
    info = null,
    activeNodeIds = [],
    activeConnectorIds = [],
    dimNodeIds = [],
    rowSelections = [],
    columnSelections = [],
    cellSelections = [],
    includeFocusState = true
} = {}) {
    if (includeFocusState !== true) {
        return {
            label: String(label || '').trim(),
            info,
            focusState: null,
            signature: ''
        };
    }

    const {
        normalized: normalizedNodeIds,
        seen: normalizedNodeIdSet
    } = collectUniqueIds(activeNodeIds);
    const {
        normalized: normalizedConnectorIds
    } = collectUniqueIds(activeConnectorIds);
    const {
        normalized: normalizedDimNodeIds
    } = collectUniqueIds(dimNodeIds);
    const {
        normalized: normalizedRowSelections,
        buckets: rowSelectionBuckets
    } = collectSelections(rowSelections, 'rowIndex');
    const {
        normalized: normalizedColumnSelections,
        buckets: columnSelectionBuckets
    } = collectSelections(columnSelections, 'colIndex');
    const {
        normalized: normalizedCellSelections,
        buckets: cellSelectionBuckets
    } = collectCellSelections(cellSelections);

    normalizedRowSelections.forEach((selection) => appendUniqueId(normalizedNodeIds, normalizedNodeIdSet, selection.nodeId));
    normalizedColumnSelections.forEach((selection) => appendUniqueId(normalizedNodeIds, normalizedNodeIdSet, selection.nodeId));
    normalizedCellSelections.forEach((selection) => appendUniqueId(normalizedNodeIds, normalizedNodeIdSet, selection.nodeId));
    const filteredDimNodeIds = normalizedDimNodeIds.filter((nodeId) => !normalizedNodeIdSet.has(nodeId));

    const focusState = {
        activeNodeIds: normalizedNodeIds,
        activeConnectorIds: normalizedConnectorIds,
        dimNodeIds: filteredDimNodeIds,
        rowSelections: normalizedRowSelections,
        columnSelections: normalizedColumnSelections,
        cellSelections: normalizedCellSelections
    };
    const focusBase = {
        activeNodeIds: normalizedNodeIdSet,
        activeConnectorIds: new Set(normalizedConnectorIds),
        dimNodeIds: new Set(filteredDimNodeIds),
        rowSelections: rowSelectionBuckets,
        columnSelections: columnSelectionBuckets,
        cellSelections: cellSelectionBuckets,
        normalizedByOpacity: new Map()
    };
    if (Object.isExtensible(focusState)) {
        Object.defineProperty(focusState, SCENE_FOCUS_BASE_CACHE_KEY, {
            value: focusBase,
            configurable: true
        });
    }

    return {
        label: String(label || '').trim(),
        info,
        focusState,
        signature: buildFocusSignature(focusState)
    };
}
