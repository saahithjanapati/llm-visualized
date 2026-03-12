export const VIEW2D_SCENE_VERSION = 1;

export const VIEW2D_NODE_KINDS = Object.freeze({
    SCENE: 'scene',
    GROUP: 'group',
    MATRIX: 'matrix',
    TEXT: 'text',
    OPERATOR: 'operator',
    CONNECTOR: 'connector'
});

export const VIEW2D_LAYOUT_DIRECTIONS = Object.freeze({
    HORIZONTAL: 'horizontal',
    VERTICAL: 'vertical',
    OVERLAY: 'overlay'
});

export const VIEW2D_MATRIX_SHAPES = Object.freeze({
    MATRIX: 'matrix',
    VECTOR: 'vector'
});

export const VIEW2D_MATRIX_PRESENTATIONS = Object.freeze({
    GRID: 'grid',
    BANDED_ROWS: 'banded-rows',
    COMPACT_ROWS: 'compact-rows',
    COLUMN_STRIP: 'column-strip',
    CARD: 'card',
    ACCENT_BAR: 'accent-bar'
});

export const VIEW2D_TEXT_PRESENTATIONS = Object.freeze({
    LABEL: 'label',
    CAPTION: 'caption',
    OPERATOR: 'operator'
});

export const VIEW2D_CONNECTOR_ROUTES = Object.freeze({
    HORIZONTAL: 'horizontal',
    VERTICAL: 'vertical',
    ELBOW: 'elbow'
});

export const VIEW2D_ANCHOR_SIDES = Object.freeze({
    LEFT: 'left',
    RIGHT: 'right',
    TOP: 'top',
    BOTTOM: 'bottom',
    CENTER: 'center'
});

const DEFAULT_ANCHORS = Object.freeze([
    VIEW2D_ANCHOR_SIDES.LEFT,
    VIEW2D_ANCHOR_SIDES.RIGHT,
    VIEW2D_ANCHOR_SIDES.TOP,
    VIEW2D_ANCHOR_SIDES.BOTTOM,
    VIEW2D_ANCHOR_SIDES.CENTER
]);

const SEMANTIC_KEY_ORDER = Object.freeze([
    'componentKind',
    'layerIndex',
    'headIndex',
    'stage',
    'role',
    'rowIndex',
    'colIndex',
    'tokenIndex',
    'focusKey',
    'operatorKey'
]);

function normalizeInteger(value) {
    return Number.isFinite(value) ? Math.floor(value) : null;
}

function normalizeSemanticValue(value) {
    if (value === null || value === undefined) return null;
    if (typeof value === 'number') {
        return Number.isFinite(value) ? String(Math.floor(value)) : null;
    }
    const safe = String(value).trim();
    if (!safe.length) return null;
    return safe
        .replace(/\s+/g, '-')
        .replace(/[^a-zA-Z0-9:_-]/g, '_');
}

function normalizeSemantic(semantic = {}) {
    if (!semantic || typeof semantic !== 'object') return {};
    const normalized = {};
    Object.entries(semantic).forEach(([key, value]) => {
        if (!key) return;
        const normalizedValue = ['layerIndex', 'headIndex', 'rowIndex', 'colIndex', 'tokenIndex'].includes(key)
            ? normalizeInteger(value)
            : value;
        if (normalizedValue === null || normalizedValue === undefined || normalizedValue === '') return;
        normalized[key] = normalizedValue;
    });
    return normalized;
}

export function buildSceneNodeId(semantic = {}, suffix = '') {
    const normalized = normalizeSemantic(semantic);
    const orderedEntries = [];

    SEMANTIC_KEY_ORDER.forEach((key) => {
        if (!Object.prototype.hasOwnProperty.call(normalized, key)) return;
        const safeValue = normalizeSemanticValue(normalized[key]);
        if (safeValue !== null) {
            orderedEntries.push(`${key}:${safeValue}`);
        }
    });

    Object.keys(normalized)
        .filter((key) => !SEMANTIC_KEY_ORDER.includes(key))
        .sort()
        .forEach((key) => {
            const safeValue = normalizeSemanticValue(normalized[key]);
            if (safeValue !== null) {
                orderedEntries.push(`${key}:${safeValue}`);
            }
        });

    if (!orderedEntries.length) {
        orderedEntries.push('node:anonymous');
    }
    const safeSuffix = normalizeSemanticValue(suffix);
    return safeSuffix ? `${orderedEntries.join('/') }#${safeSuffix}` : orderedEntries.join('/');
}

export function createAnchorRef(nodeId, anchor = VIEW2D_ANCHOR_SIDES.CENTER) {
    return {
        nodeId: typeof nodeId === 'string' ? nodeId : '',
        anchor: Object.values(VIEW2D_ANCHOR_SIDES).includes(anchor)
            ? anchor
            : VIEW2D_ANCHOR_SIDES.CENTER
    };
}

function createNodeBase({
    id = '',
    kind = VIEW2D_NODE_KINDS.GROUP,
    role = '',
    semantic = {},
    label = null,
    anchors = DEFAULT_ANCHORS,
    layout = null,
    visual = null,
    metadata = null
} = {}) {
    const normalizedSemantic = normalizeSemantic(semantic);
    return {
        id: typeof id === 'string' && id.length
            ? id
            : buildSceneNodeId({ ...normalizedSemantic, role: role || kind }),
        kind,
        role: typeof role === 'string' ? role : '',
        semantic: normalizedSemantic,
        label: label && typeof label === 'object' ? { ...label } : null,
        anchors: Array.isArray(anchors) && anchors.length ? [...anchors] : [...DEFAULT_ANCHORS],
        layout: layout && typeof layout === 'object' ? { ...layout } : null,
        visual: visual && typeof visual === 'object' ? { ...visual } : null,
        metadata: metadata && typeof metadata === 'object' ? { ...metadata } : null
    };
}

export function createSceneModel({
    id = '',
    semantic = {},
    nodes = [],
    metadata = {},
    viewport = null
} = {}) {
    const normalizedSemantic = normalizeSemantic(semantic);
    return {
        id: typeof id === 'string' && id.length
            ? id
            : buildSceneNodeId({ ...normalizedSemantic, role: VIEW2D_NODE_KINDS.SCENE }),
        kind: VIEW2D_NODE_KINDS.SCENE,
        version: VIEW2D_SCENE_VERSION,
        semantic: normalizedSemantic,
        nodes: Array.isArray(nodes) ? [...nodes] : [],
        metadata: metadata && typeof metadata === 'object' ? { ...metadata } : {},
        viewport: viewport && typeof viewport === 'object' ? { ...viewport } : null
    };
}

export function createGroupNode({
    children = [],
    direction = VIEW2D_LAYOUT_DIRECTIONS.HORIZONTAL,
    gapKey = 'default',
    align = 'center',
    ...rest
} = {}) {
    const base = createNodeBase({
        ...rest,
        kind: VIEW2D_NODE_KINDS.GROUP
    });
    return {
        ...base,
        children: Array.isArray(children) ? [...children] : [],
        layout: {
            ...(base.layout || {}),
            direction,
            gapKey,
            align
        }
    };
}

export function createMatrixNode({
    dimensions = {},
    rowItems = [],
    columnItems = [],
    shape = VIEW2D_MATRIX_SHAPES.MATRIX,
    presentation = VIEW2D_MATRIX_PRESENTATIONS.GRID,
    ...rest
} = {}) {
    const base = createNodeBase({
        ...rest,
        kind: VIEW2D_NODE_KINDS.MATRIX
    });
    const rows = normalizeInteger(dimensions.rows) ?? 0;
    const cols = normalizeInteger(dimensions.cols) ?? 0;
    return {
        ...base,
        shape,
        presentation,
        dimensions: {
            rows: Math.max(0, rows),
            cols: Math.max(0, cols)
        },
        rowItems: Array.isArray(rowItems) ? [...rowItems] : [],
        columnItems: Array.isArray(columnItems) ? [...columnItems] : []
    };
}

export function createTextNode({
    text = '',
    tex = '',
    presentation = VIEW2D_TEXT_PRESENTATIONS.LABEL,
    ...rest
} = {}) {
    const base = createNodeBase({
        ...rest,
        kind: VIEW2D_NODE_KINDS.TEXT
    });
    return {
        ...base,
        text: typeof text === 'string' ? text : '',
        tex: typeof tex === 'string' ? tex : '',
        presentation
    };
}

export function createOperatorNode({
    text = '',
    ...rest
} = {}) {
    const base = createNodeBase({
        ...rest,
        kind: VIEW2D_NODE_KINDS.OPERATOR
    });
    return {
        ...base,
        text: typeof text === 'string' ? text : '',
        presentation: VIEW2D_TEXT_PRESENTATIONS.OPERATOR
    };
}

export function createConnectorNode({
    source = null,
    target = null,
    route = VIEW2D_CONNECTOR_ROUTES.HORIZONTAL,
    gap = null,
    sourceGap = null,
    targetGap = null,
    gapKey = 'default',
    ...rest
} = {}) {
    const base = createNodeBase({
        ...rest,
        kind: VIEW2D_NODE_KINDS.CONNECTOR,
        anchors: []
    });
    const resolvedGap = Number.isFinite(gap) ? Math.max(0, Math.floor(gap)) : null;
    return {
        ...base,
        source: source && typeof source === 'object' ? { ...source } : null,
        target: target && typeof target === 'object' ? { ...target } : null,
        route,
        gap: resolvedGap,
        sourceGap: Number.isFinite(sourceGap) ? Math.max(0, Math.floor(sourceGap)) : resolvedGap,
        targetGap: Number.isFinite(targetGap) ? Math.max(0, Math.floor(targetGap)) : resolvedGap,
        gapKey
    };
}

export function walkSceneNodes(sceneOrNodes, visit) {
    if (typeof visit !== 'function') return;
    const queue = Array.isArray(sceneOrNodes)
        ? [...sceneOrNodes]
        : (Array.isArray(sceneOrNodes?.nodes) ? [...sceneOrNodes.nodes] : []);
    while (queue.length) {
        const node = queue.shift();
        if (!node || typeof node !== 'object') continue;
        visit(node);
        if (Array.isArray(node.children) && node.children.length) {
            queue.unshift(...node.children);
        }
    }
}

export function flattenSceneNodes(sceneOrNodes) {
    const nodes = [];
    walkSceneNodes(sceneOrNodes, (node) => {
        nodes.push(node);
    });
    return nodes;
}

export function indexSceneNodes(sceneOrNodes) {
    const index = new Map();
    walkSceneNodes(sceneOrNodes, (node) => {
        if (typeof node?.id === 'string' && node.id.length) {
            index.set(node.id, node);
        }
    });
    return index;
}
