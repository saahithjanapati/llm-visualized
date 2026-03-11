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

function cloneArray(items = []) {
    return Array.isArray(items) ? items.map((item) => {
        if (item && typeof item === 'object') {
            return Array.isArray(item) ? [...item] : { ...item };
        }
        return item;
    }) : [];
}

export class LayoutRegistry {
    constructor() {
        this._sceneBounds = null;
        this._nodeEntries = new Map();
        this._connectorEntries = new Map();
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
            layoutData: entry.layoutData && typeof entry.layoutData === 'object'
                ? { ...entry.layoutData }
                : null,
            metadata: entry.metadata && typeof entry.metadata === 'object'
                ? { ...entry.metadata }
                : null
        };
        this._nodeEntries.set(nodeId, normalizedEntry);
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
            metadata: entry.metadata ? { ...entry.metadata } : null
        };
    }

    getNodeEntries() {
        return Array.from(this._nodeEntries.values()).map((entry) => this.getNodeEntry(entry.nodeId));
    }

    getConnectorEntries() {
        return Array.from(this._connectorEntries.values()).map((entry) => this.getConnectorEntry(entry.nodeId));
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
