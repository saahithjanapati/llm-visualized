import {
    normalizeProjectionKind,
    resolveMatrixStageKey,
    resolveProjectionKindForNode
} from './mhsaDetailHoverInfo.js';

const CELL_REQUIRED_ATTENTION_ROLES = new Set([
    'attention-pre-score',
    'attention-masked-input',
    'attention-mask',
    'attention-post',
    'attention-post-copy'
]);

function normalizeAxisIndex(hit = null) {
    if (Number.isFinite(hit?.rowIndex)) {
        return Math.max(0, Math.floor(hit.rowIndex));
    }
    if (Number.isFinite(hit?.colIndex)) {
        return Math.max(0, Math.floor(hit.colIndex));
    }
    return null;
}

export function resolveMhsaDetailHoverEntity(hit = null) {
    const node = hit?.node || null;
    if (!node) return null;

    if (hit?.cellHit) {
        const stageKey = resolveMatrixStageKey(node, hit.cellHit);
        if (!stageKey) return null;
        return {
            type: 'attention-cell',
            node,
            stageKey,
            rowIndex: Number.isFinite(hit.cellHit.rowIndex) ? Math.max(0, Math.floor(hit.cellHit.rowIndex)) : null,
            colIndex: Number.isFinite(hit.cellHit.colIndex) ? Math.max(0, Math.floor(hit.cellHit.colIndex)) : null,
            cellItem: hit.cellHit.cellItem || null
        };
    }

    if (CELL_REQUIRED_ATTENTION_ROLES.has(String(node.role || '').trim())) {
        return null;
    }

    if ((hit?.rowHit || hit?.columnHit) && node.role === 'attention-key-transpose') {
        const axisHit = hit.rowHit || hit.columnHit;
        return {
            type: 'transpose-axis',
            node,
            axisIndex: normalizeAxisIndex(axisHit),
            axisHit
        };
    }

    const projectionKind = normalizeProjectionKind(resolveProjectionKindForNode(node));

    if (hit?.rowHit) {
        if (projectionKind) {
            return {
                type: node.role === 'projection-bias' ? 'projection-bias' : 'projection-row',
                node,
                role: String(node.role || '').trim(),
                projectionKind,
                rowHit: hit.rowHit
            };
        }
        if (node.role === 'attention-query-source') {
            return {
                type: 'projection-row',
                node,
                role: String(node.role || '').trim(),
                projectionKind: 'q',
                rowHit: hit.rowHit
            };
        }
        if (node.role === 'projection-source-xln') {
            return {
                type: 'projection-source-row',
                node,
                rowHit: hit.rowHit
            };
        }
        if (node.role === 'attention-value-post') {
            return {
                type: 'weighted-output-row',
                node,
                variant: 'value',
                rowHit: hit.rowHit
            };
        }
        if (node.role === 'attention-head-output') {
            return {
                type: 'weighted-output-row',
                node,
                variant: 'head-output',
                rowHit: hit.rowHit
            };
        }
        if (node.role === 'head-output-matrix') {
            return {
                type: 'output-projection-head-output-row',
                node,
                rowHit: hit.rowHit
            };
        }
    }

    if (projectionKind) {
        return {
            type: node.role === 'projection-weight'
                ? 'projection-weight'
                : (node.role === 'projection-bias' ? 'projection-bias' : 'projection-stage'),
            node,
            projectionKind
        };
    }

    return {
        type: 'attention-role',
        node,
        role: String(node.role || '').trim()
    };
}
