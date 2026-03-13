function cloneBounds(bounds = null) {
    if (!bounds || typeof bounds !== 'object') return null;
    return {
        x: Number.isFinite(bounds.x) ? bounds.x : 0,
        y: Number.isFinite(bounds.y) ? bounds.y : 0,
        width: Number.isFinite(bounds.width) ? bounds.width : 0,
        height: Number.isFinite(bounds.height) ? bounds.height : 0
    };
}

function clonePoint(point = null) {
    if (!point || typeof point !== 'object') return null;
    return {
        x: Number.isFinite(point.x) ? point.x : 0,
        y: Number.isFinite(point.y) ? point.y : 0
    };
}

function cloneAnchors(anchors = {}) {
    return Object.entries(anchors || {}).reduce((acc, [key, point]) => {
        const clonedPoint = clonePoint(point);
        if (clonedPoint) {
            acc[key] = clonedPoint;
        }
        return acc;
    }, {});
}

function translatePointInPlace(point = null, offsetX = 0, offsetY = 0) {
    if (!point || typeof point !== 'object') return;
    point.x += Number.isFinite(offsetX) ? offsetX : 0;
    point.y += Number.isFinite(offsetY) ? offsetY : 0;
}

function translateBoundsInPlace(bounds = null, offsetX = 0, offsetY = 0) {
    if (!bounds || typeof bounds !== 'object') return;
    bounds.x += Number.isFinite(offsetX) ? offsetX : 0;
    bounds.y += Number.isFinite(offsetY) ? offsetY : 0;
}

function cloneArray(items = []) {
    return Array.isArray(items) ? items.map((item) => {
        if (item && typeof item === 'object') {
            return Array.isArray(item) ? [...item] : { ...item };
        }
        return item;
    }) : [];
}

function normalizeSemanticValue(value) {
    if (value === null || value === undefined) return null;
    if (typeof value === 'number') {
        return Number.isFinite(value) ? Math.floor(value) : null;
    }
    if (typeof value === 'string') {
        const safe = value.trim();
        return safe.length ? safe : null;
    }
    if (typeof value === 'boolean') {
        return value;
    }
    return null;
}

function cloneSemantic(semantic = null) {
    if (!semantic || typeof semantic !== 'object') return null;
    const normalized = Object.entries(semantic).reduce((acc, [key, value]) => {
        if (!key) return acc;
        const normalizedValue = normalizeSemanticValue(value);
        if (normalizedValue !== null) {
            acc[key] = normalizedValue;
        }
        return acc;
    }, {});
    return Object.keys(normalized).length ? normalized : null;
}

function getBoundsArea(bounds = null) {
    if (!bounds) return 0;
    const width = Number.isFinite(bounds.width) ? Math.max(0, bounds.width) : 0;
    const height = Number.isFinite(bounds.height) ? Math.max(0, bounds.height) : 0;
    return width * height;
}

function containsPoint(bounds = null, x = 0, y = 0) {
    if (!bounds) return false;
    const minX = Number.isFinite(bounds.x) ? bounds.x : 0;
    const minY = Number.isFinite(bounds.y) ? bounds.y : 0;
    const maxX = minX + (Number.isFinite(bounds.width) ? Math.max(0, bounds.width) : 0);
    const maxY = minY + (Number.isFinite(bounds.height) ? Math.max(0, bounds.height) : 0);
    return x >= minX && x <= maxX && y >= minY && y <= maxY;
}

function scoreSemanticMatch(entrySemantic = null, targetSemantic = null) {
    if (!entrySemantic || !targetSemantic) return null;
    const targetEntries = Object.entries(targetSemantic);
    if (!targetEntries.length) return null;
    for (const [key, value] of targetEntries) {
        if (!Object.prototype.hasOwnProperty.call(entrySemantic, key)) {
            return null;
        }
        if (entrySemantic[key] !== value) {
            return null;
        }
    }
    const extraKeyCount = Object.keys(entrySemantic)
        .filter((key) => !Object.prototype.hasOwnProperty.call(targetSemantic, key))
        .length;
    return {
        matchCount: targetEntries.length,
        extraKeyCount
    };
}

function compareSemanticMatches(a, b) {
    if ((a?.score?.matchCount || 0) !== (b?.score?.matchCount || 0)) {
        return (b?.score?.matchCount || 0) - (a?.score?.matchCount || 0);
    }
    if ((a?.score?.extraKeyCount || 0) !== (b?.score?.extraKeyCount || 0)) {
        return (a?.score?.extraKeyCount || 0) - (b?.score?.extraKeyCount || 0);
    }

    const kindRank = (entry) => {
        const kind = String(entry?.entry?.kind || '');
        if (kind === 'group') return 0;
        if (kind === 'matrix') return 1;
        if (kind === 'text' || kind === 'operator') return 2;
        if (kind === 'connector') return 3;
        return 4;
    };
    const rankDelta = kindRank(a) - kindRank(b);
    if (rankDelta !== 0) return rankDelta;

    if ((a?.entry?.depth || 0) !== (b?.entry?.depth || 0)) {
        return (a?.entry?.depth || 0) - (b?.entry?.depth || 0);
    }

    return getBoundsArea(b?.entry?.bounds) - getBoundsArea(a?.entry?.bounds);
}

function comparePointHitEntries(a, b) {
    if ((a?.depth || 0) !== (b?.depth || 0)) {
        return (b?.depth || 0) - (a?.depth || 0);
    }

    const kindRank = (entry) => {
        const kind = String(entry?.kind || '');
        if (kind === 'matrix') return 0;
        if (kind === 'text' || kind === 'operator') return 1;
        if (kind === 'group') return 2;
        return 3;
    };
    const rankDelta = kindRank(a) - kindRank(b);
    if (rankDelta !== 0) return rankDelta;

    return getBoundsArea(a?.bounds) - getBoundsArea(b?.bounds);
}

export class LayoutRegistry {
    constructor() {
        this._sceneBounds = null;
        this._nodeEntries = new Map();
        this._connectorEntries = new Map();
        this._pointHitEntries = [];
        this._pointHitEntriesDirty = true;
    }

    setSceneBounds(bounds = null) {
        this._sceneBounds = cloneBounds(bounds);
        return this._sceneBounds;
    }

    getSceneBounds() {
        return cloneBounds(this._sceneBounds);
    }

    setNodeEntry(nodeId, entry = {}) {
        if (typeof nodeId !== 'string' || !nodeId.length) return null;
        const normalizedEntry = {
            nodeId,
            kind: typeof entry.kind === 'string' ? entry.kind : '',
            role: typeof entry.role === 'string' ? entry.role : '',
            parentId: typeof entry.parentId === 'string' ? entry.parentId : null,
            depth: Number.isFinite(entry.depth) ? entry.depth : 0,
            bounds: cloneBounds(entry.bounds),
            contentBounds: cloneBounds(entry.contentBounds),
            labelBounds: cloneBounds(entry.labelBounds),
            dimensionBounds: cloneBounds(entry.dimensionBounds),
            anchors: cloneAnchors(entry.anchors),
            semantic: cloneSemantic(entry.semantic),
            layoutData: entry.layoutData && typeof entry.layoutData === 'object'
                ? { ...entry.layoutData }
                : null,
            metadata: entry.metadata && typeof entry.metadata === 'object'
                ? { ...entry.metadata }
                : null
        };
        this._nodeEntries.set(nodeId, normalizedEntry);
        this._pointHitEntriesDirty = true;
        return normalizedEntry;
    }

    getNodeEntry(nodeId) {
        const entry = this._nodeEntries.get(nodeId);
        if (!entry) return null;
        return {
            ...entry,
            bounds: cloneBounds(entry.bounds),
            contentBounds: cloneBounds(entry.contentBounds),
            labelBounds: cloneBounds(entry.labelBounds),
            dimensionBounds: cloneBounds(entry.dimensionBounds),
            anchors: cloneAnchors(entry.anchors),
            semantic: cloneSemantic(entry.semantic),
            layoutData: entry.layoutData ? { ...entry.layoutData } : null,
            metadata: entry.metadata ? { ...entry.metadata } : null
        };
    }

    getBounds(nodeId) {
        return cloneBounds(this._nodeEntries.get(nodeId)?.bounds || this._connectorEntries.get(nodeId)?.bounds);
    }

    getContentBounds(nodeId) {
        return cloneBounds(this._nodeEntries.get(nodeId)?.contentBounds);
    }

    getAnchors(nodeId) {
        return cloneAnchors(this._nodeEntries.get(nodeId)?.anchors);
    }

    resolveAnchor(anchorRef = null) {
        if (!anchorRef || typeof anchorRef !== 'object') return null;
        const entry = this._nodeEntries.get(anchorRef.nodeId);
        if (!entry) return null;
        const anchorKey = typeof anchorRef.anchor === 'string' ? anchorRef.anchor : 'center';
        return clonePoint(entry.anchors?.[anchorKey] || entry.anchors?.center || null);
    }

    setConnectorEntry(nodeId, entry = {}) {
        if (typeof nodeId !== 'string' || !nodeId.length) return null;
        const normalizedEntry = {
            nodeId,
            kind: typeof entry.kind === 'string' ? entry.kind : 'connector',
            role: typeof entry.role === 'string' ? entry.role : '',
            bounds: cloneBounds(entry.bounds),
            source: entry.source && typeof entry.source === 'object' ? { ...entry.source } : null,
            target: entry.target && typeof entry.target === 'object' ? { ...entry.target } : null,
            pathPoints: Array.isArray(entry.pathPoints) ? entry.pathPoints.map((point) => clonePoint(point)) : [],
            semantic: cloneSemantic(entry.semantic),
            metadata: entry.metadata && typeof entry.metadata === 'object'
                ? { ...entry.metadata }
                : null
        };
        this._connectorEntries.set(nodeId, normalizedEntry);
        return normalizedEntry;
    }

    getConnectorEntry(nodeId) {
        const entry = this._connectorEntries.get(nodeId);
        if (!entry) return null;
        return {
            ...entry,
            bounds: cloneBounds(entry.bounds),
            source: entry.source ? { ...entry.source } : null,
            target: entry.target ? { ...entry.target } : null,
            pathPoints: entry.pathPoints.map((point) => clonePoint(point)),
            semantic: cloneSemantic(entry.semantic),
            metadata: entry.metadata ? { ...entry.metadata } : null
        };
    }

    getNodeEntries() {
        return Array.from(this._nodeEntries.values()).map((entry) => this.getNodeEntry(entry.nodeId));
    }

    translateNodeSubtree(nodeId, offsetX = 0, offsetY = 0) {
        if (typeof nodeId !== 'string' || !nodeId.length) return false;
        if (!Number.isFinite(offsetX) && !Number.isFinite(offsetY)) return false;
        const safeOffsetX = Number.isFinite(offsetX) ? offsetX : 0;
        const safeOffsetY = Number.isFinite(offsetY) ? offsetY : 0;
        if (safeOffsetX === 0 && safeOffsetY === 0) return false;
        if (!this._nodeEntries.has(nodeId)) return false;

        const pending = [nodeId];
        while (pending.length) {
            const currentNodeId = pending.shift();
            const currentEntry = this._nodeEntries.get(currentNodeId);
            if (!currentEntry) continue;

            translateBoundsInPlace(currentEntry.bounds, safeOffsetX, safeOffsetY);
            translateBoundsInPlace(currentEntry.contentBounds, safeOffsetX, safeOffsetY);
            translateBoundsInPlace(currentEntry.labelBounds, safeOffsetX, safeOffsetY);
            translateBoundsInPlace(currentEntry.dimensionBounds, safeOffsetX, safeOffsetY);
            Object.values(currentEntry.anchors || {}).forEach((anchorPoint) => {
                translatePointInPlace(anchorPoint, safeOffsetX, safeOffsetY);
            });

            this._nodeEntries.forEach((candidateEntry, candidateNodeId) => {
                if (candidateEntry?.parentId === currentNodeId) {
                    pending.push(candidateNodeId);
                }
            });
        }
        return true;
    }

    getNodeEntriesAtPoint(x = 0, y = 0, { includeGroups = false } = {}) {
        if (!Number.isFinite(x) || !Number.isFinite(y)) return [];
        return this._getPointHitEntries()
            .filter((entry) => includeGroups || entry.kind !== 'group')
            .filter((entry) => containsPoint(entry.bounds, x, y))
            .map((entry) => this.getNodeEntry(entry.nodeId))
            .filter(Boolean);
    }

    _getPointHitEntries() {
        if (this._pointHitEntriesDirty) {
            this._pointHitEntries = Array.from(this._nodeEntries.values())
                .sort(comparePointHitEntries);
            this._pointHitEntriesDirty = false;
        }
        return this._pointHitEntries;
    }

    resolveRawNodeEntryAtPoint(x = 0, y = 0, options = {}) {
        if (!Number.isFinite(x) || !Number.isFinite(y)) return null;
        const includeGroups = options?.includeGroups === true;
        const pointHitEntries = this._getPointHitEntries();
        for (let index = 0; index < pointHitEntries.length; index += 1) {
            const entry = pointHitEntries[index];
            if (!entry) continue;
            if (!includeGroups && entry.kind === 'group') continue;
            if (containsPoint(entry.bounds, x, y)) {
                return entry;
            }
        }
        return null;
    }

    resolveNodeEntryAtPoint(x = 0, y = 0, options = {}) {
        const entry = this.resolveRawNodeEntryAtPoint(x, y, options);
        return entry ? this.getNodeEntry(entry.nodeId) : null;
    }

    getConnectorEntries() {
        return Array.from(this._connectorEntries.values()).map((entry) => this.getConnectorEntry(entry.nodeId));
    }

    getEntriesForSemanticTarget(target = null, { includeConnectors = false } = {}) {
        const targetSemantic = cloneSemantic(target);
        if (!targetSemantic) return [];

        const nodeMatches = Array.from(this._nodeEntries.values()).map((entry) => ({
            entryType: 'node',
            entry,
            score: scoreSemanticMatch(entry.semantic, targetSemantic)
        })).filter((candidate) => candidate.score);
        const connectorMatches = includeConnectors
            ? Array.from(this._connectorEntries.values()).map((entry) => ({
                entryType: 'connector',
                entry,
                score: scoreSemanticMatch(entry.semantic, targetSemantic)
            })).filter((candidate) => candidate.score)
            : [];

        return [...nodeMatches, ...connectorMatches]
            .sort(compareSemanticMatches)
            .map((candidate) => (
                candidate.entryType === 'connector'
                    ? this.getConnectorEntry(candidate.entry.nodeId)
                    : this.getNodeEntry(candidate.entry.nodeId)
            ))
            .filter(Boolean);
    }

    resolveEntryForSemanticTarget(target = null, options = {}) {
        return this.getEntriesForSemanticTarget(target, options)[0] || null;
    }

    resolveNodeIdForSemanticTarget(target = null, options = {}) {
        const entry = this.resolveEntryForSemanticTarget(target, options);
        return typeof entry?.nodeId === 'string' ? entry.nodeId : null;
    }

    resolveBoundsForSemanticTarget(target = null, options = {}) {
        const entry = this.resolveEntryForSemanticTarget(target, options);
        return cloneBounds(entry?.bounds || null);
    }

    resolveAnchorsForSemanticTarget(target = null, options = {}) {
        const entry = this.resolveEntryForSemanticTarget(target, options);
        return cloneAnchors(entry?.anchors || {});
    }

    resolveFocusPathForSemanticTarget(target = null, options = {}) {
        const entry = this.resolveEntryForSemanticTarget(target, options);
        if (!entry || typeof entry.nodeId !== 'string' || !entry.nodeId.length) return [];

        const path = [];
        let current = this._nodeEntries.get(entry.nodeId) || null;
        while (current) {
            path.unshift({
                nodeId: current.nodeId,
                kind: current.kind,
                role: current.role,
                semantic: cloneSemantic(current.semantic),
                bounds: cloneBounds(current.bounds)
            });
            current = typeof current.parentId === 'string' && current.parentId.length
                ? this._nodeEntries.get(current.parentId) || null
                : null;
        }
        return path;
    }

    toJSON() {
        return {
            sceneBounds: this.getSceneBounds(),
            nodes: this.getNodeEntries(),
            connectors: this.getConnectorEntries()
        };
    }
}

export function unionBounds(boundsList = []) {
    const validBounds = boundsList.filter((bounds) => (
        bounds
        && typeof bounds === 'object'
        && Number.isFinite(bounds.x)
        && Number.isFinite(bounds.y)
        && Number.isFinite(bounds.width)
        && Number.isFinite(bounds.height)
    ));
    if (!validBounds.length) {
        return { x: 0, y: 0, width: 0, height: 0 };
    }
    const minX = Math.min(...validBounds.map((bounds) => bounds.x));
    const minY = Math.min(...validBounds.map((bounds) => bounds.y));
    const maxX = Math.max(...validBounds.map((bounds) => bounds.x + bounds.width));
    const maxY = Math.max(...validBounds.map((bounds) => bounds.y + bounds.height));
    return {
        x: minX,
        y: minY,
        width: Math.max(0, maxX - minX),
        height: Math.max(0, maxY - minY)
    };
}

export function translateBounds(bounds = null, offsetX = 0, offsetY = 0) {
    const cloned = cloneBounds(bounds);
    if (!cloned) return null;
    cloned.x += Number.isFinite(offsetX) ? offsetX : 0;
    cloned.y += Number.isFinite(offsetY) ? offsetY : 0;
    return cloned;
}

export function inflateBounds(bounds = null, inset = 0) {
    const cloned = cloneBounds(bounds);
    if (!cloned) return null;
    const safeInset = Number.isFinite(inset) ? inset : 0;
    return {
        x: cloned.x - safeInset,
        y: cloned.y - safeInset,
        width: cloned.width + (safeInset * 2),
        height: cloned.height + (safeInset * 2)
    };
}
