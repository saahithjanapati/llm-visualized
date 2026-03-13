const SCENE_FOCUS_DEFAULT_INACTIVE_OPACITY = 0.18;
const SCENE_FOCUS_MIN_INACTIVE_OPACITY = 0.08;
const SCENE_FOCUS_MAX_INACTIVE_OPACITY = 0.4;
const SCENE_FOCUS_DEFAULT_DIM_NODE_OPACITY = 0.08;
const SCENE_FOCUS_MIN_DIM_NODE_OPACITY = 0.04;
export const SCENE_FOCUS_BASE_CACHE_KEY = Symbol.for('view2d.sceneFocusBase');

function normalizeSceneFocusSelections(items = [], key = 'rowIndex') {
    const selectionMap = new Map();
    if (!Array.isArray(items) || !items.length) return selectionMap;
    items.forEach((item) => {
        const nodeId = typeof item?.nodeId === 'string' ? item.nodeId : '';
        const indexValue = Number.isFinite(item?.[key]) ? Math.max(0, Math.floor(item[key])) : null;
        if (!nodeId.length || !Number.isFinite(indexValue)) return;
        let bucket = selectionMap.get(nodeId);
        if (!bucket) {
            bucket = new Set();
            selectionMap.set(nodeId, bucket);
        }
        bucket.add(indexValue);
    });
    return selectionMap;
}

function normalizeSceneFocusCellSelections(items = []) {
    const selectionMap = new Map();
    if (!Array.isArray(items) || !items.length) return selectionMap;
    items.forEach((item) => {
        const nodeId = typeof item?.nodeId === 'string' ? item.nodeId : '';
        const rowIndex = Number.isFinite(item?.rowIndex) ? Math.max(0, Math.floor(item.rowIndex)) : null;
        const colIndex = Number.isFinite(item?.colIndex) ? Math.max(0, Math.floor(item.colIndex)) : null;
        if (!nodeId.length || !Number.isFinite(rowIndex) || !Number.isFinite(colIndex)) return;
        let bucket = selectionMap.get(nodeId);
        if (!bucket) {
            bucket = new Set();
            selectionMap.set(nodeId, bucket);
        }
        bucket.add(`${rowIndex}:${colIndex}`);
    });
    return selectionMap;
}

function resolveInactiveOpacity(value = SCENE_FOCUS_DEFAULT_INACTIVE_OPACITY) {
    const safeValue = Number.isFinite(value) ? Number(value) : SCENE_FOCUS_DEFAULT_INACTIVE_OPACITY;
    return Math.max(
        SCENE_FOCUS_MIN_INACTIVE_OPACITY,
        Math.min(SCENE_FOCUS_MAX_INACTIVE_OPACITY, safeValue)
    );
}

function resolveDimNodeOpacity(value = SCENE_FOCUS_DEFAULT_DIM_NODE_OPACITY, inactiveOpacity = SCENE_FOCUS_DEFAULT_INACTIVE_OPACITY) {
    const maxOpacity = Math.max(
        SCENE_FOCUS_MIN_DIM_NODE_OPACITY,
        Math.min(resolveInactiveOpacity(inactiveOpacity), SCENE_FOCUS_MAX_INACTIVE_OPACITY)
    );
    const safeValue = Number.isFinite(value) ? Number(value) : SCENE_FOCUS_DEFAULT_DIM_NODE_OPACITY;
    return Math.max(
        SCENE_FOCUS_MIN_DIM_NODE_OPACITY,
        Math.min(maxOpacity, safeValue)
    );
}

function createSceneFocusBase(focusState = null) {
    if (!focusState || typeof focusState !== 'object') return null;
    const cachedBase = focusState[SCENE_FOCUS_BASE_CACHE_KEY] || null;
    if (cachedBase) {
        return cachedBase;
    }

    const activeNodeIds = focusState.activeNodeIds instanceof Set
        ? focusState.activeNodeIds
        : new Set(
            Array.isArray(focusState.activeNodeIds)
                ? focusState.activeNodeIds.filter((value) => typeof value === 'string' && value.length)
                : []
        );
    const activeConnectorIds = focusState.activeConnectorIds instanceof Set
        ? focusState.activeConnectorIds
        : new Set(
            Array.isArray(focusState.activeConnectorIds)
                ? focusState.activeConnectorIds.filter((value) => typeof value === 'string' && value.length)
                : []
        );
    const dimNodeIds = focusState.dimNodeIds instanceof Set
        ? focusState.dimNodeIds
        : new Set(
            Array.isArray(focusState.dimNodeIds)
                ? focusState.dimNodeIds.filter((value) => typeof value === 'string' && value.length)
                : []
        );
    const rowSelections = focusState.rowSelections instanceof Map
        ? focusState.rowSelections
        : normalizeSceneFocusSelections(focusState.rowSelections, 'rowIndex');
    const columnSelections = focusState.columnSelections instanceof Map
        ? focusState.columnSelections
        : normalizeSceneFocusSelections(focusState.columnSelections, 'colIndex');
    const cellSelections = focusState.cellSelections instanceof Map
        ? focusState.cellSelections
        : normalizeSceneFocusCellSelections(focusState.cellSelections);
    const hasSceneFocus = activeNodeIds.size > 0
        || activeConnectorIds.size > 0
        || dimNodeIds.size > 0
        || rowSelections.size > 0
        || columnSelections.size > 0
        || cellSelections.size > 0;
    if (!hasSceneFocus) return null;

    const base = {
        activeNodeIds,
        activeConnectorIds,
        dimNodeIds,
        dimNodeOpacity: resolveDimNodeOpacity(
            focusState.dimNodeOpacity,
            focusState.inactiveOpacity
        ),
        rowSelections,
        columnSelections,
        cellSelections,
        normalizedByOpacity: new Map()
    };

    if (Object.isExtensible(focusState)) {
        Object.defineProperty(focusState, SCENE_FOCUS_BASE_CACHE_KEY, {
            value: base,
            configurable: true
        });
    }

    return base;
}

export function normalizeSceneFocusState(focusState = null, {
    inactiveOpacity = SCENE_FOCUS_DEFAULT_INACTIVE_OPACITY
} = {}) {
    const base = createSceneFocusBase(focusState);
    if (!base) return null;
    const resolvedOpacity = resolveInactiveOpacity(inactiveOpacity);
    const cachedState = base.normalizedByOpacity.get(resolvedOpacity) || null;
    if (cachedState) {
        return cachedState;
    }
    const normalizedState = {
        activeNodeIds: base.activeNodeIds,
        activeConnectorIds: base.activeConnectorIds,
        dimNodeIds: base.dimNodeIds,
        dimNodeOpacity: resolveDimNodeOpacity(base.dimNodeOpacity, resolvedOpacity),
        rowSelections: base.rowSelections,
        columnSelections: base.columnSelections,
        cellSelections: base.cellSelections,
        inactiveOpacity: resolvedOpacity
    };
    if (Object.isExtensible(normalizedState)) {
        Object.defineProperty(normalizedState, SCENE_FOCUS_BASE_CACHE_KEY, {
            value: base,
            configurable: true
        });
    }
    base.normalizedByOpacity.set(resolvedOpacity, normalizedState);
    return normalizedState;
}

export function resolveSceneElementFocusAlpha(isActive = true, sceneFocusState = null) {
    if (!sceneFocusState?.inactiveOpacity) return 1;
    return isActive ? 1 : sceneFocusState.inactiveOpacity;
}

export function resolveSceneNodeFocusAlpha(nodeId = '', sceneFocusState = null) {
    if (!sceneFocusState) return 1;
    if (typeof nodeId !== 'string' || !nodeId.length) {
        return sceneFocusState.inactiveOpacity;
    }
    if (sceneFocusState.activeNodeIds.has(nodeId)) {
        return resolveSceneElementFocusAlpha(true, sceneFocusState);
    }
    if (sceneFocusState.dimNodeIds?.has(nodeId)) {
        return Number.isFinite(sceneFocusState.dimNodeOpacity)
            ? sceneFocusState.dimNodeOpacity
            : sceneFocusState.inactiveOpacity;
    }
    return resolveSceneElementFocusAlpha(false, sceneFocusState);
}

export function resolveSceneRowSelectionAlpha(nodeId = '', rowIndex = null, sceneFocusState = null) {
    if (!sceneFocusState || !Number.isFinite(rowIndex)) return null;
    const bucket = sceneFocusState.rowSelections.get(nodeId);
    if (!bucket || !bucket.size) return null;
    return resolveSceneElementFocusAlpha(bucket.has(Math.max(0, Math.floor(rowIndex))), sceneFocusState);
}

export function resolveSceneColumnSelectionAlpha(nodeId = '', colIndex = null, sceneFocusState = null) {
    if (!sceneFocusState || !Number.isFinite(colIndex)) return null;
    const bucket = sceneFocusState.columnSelections.get(nodeId);
    if (!bucket || !bucket.size) return null;
    return resolveSceneElementFocusAlpha(bucket.has(Math.max(0, Math.floor(colIndex))), sceneFocusState);
}

export function resolveSceneGridCellAlpha(nodeId = '', rowIndex = null, colIndex = null, sceneFocusState = null) {
    if (!sceneFocusState || !Number.isFinite(rowIndex) || !Number.isFinite(colIndex)) return null;
    const rowBucket = sceneFocusState.rowSelections.get(nodeId) || null;
    const colBucket = sceneFocusState.columnSelections.get(nodeId) || null;
    const cellBucket = sceneFocusState.cellSelections.get(nodeId) || null;
    const hasLocalFocus = !!(
        (rowBucket && rowBucket.size)
        || (colBucket && colBucket.size)
        || (cellBucket && cellBucket.size)
    );
    if (!hasLocalFocus) return null;
    const rowMatch = !rowBucket || !rowBucket.size || rowBucket.has(Math.max(0, Math.floor(rowIndex)));
    const colMatch = !colBucket || !colBucket.size || colBucket.has(Math.max(0, Math.floor(colIndex)));
    const cellMatch = !cellBucket || !cellBucket.size || cellBucket.has(
        `${Math.max(0, Math.floor(rowIndex))}:${Math.max(0, Math.floor(colIndex))}`
    );
    return resolveSceneElementFocusAlpha(rowMatch && colMatch && cellMatch, sceneFocusState);
}
