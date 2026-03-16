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
    const componentKind = String(node?.semantic?.componentKind || '').trim().toLowerCase();
    const projectionKind = normalizeProjectionKind(resolveProjectionKindForNode(node));

    if (hit?.cellHit) {
        if (node.role === 'concat-output-matrix' || node.role === 'concat-output-copy-matrix') {
            return {
                type: 'output-projection-concat-output-band',
                node,
                cellHit: hit.cellHit
            };
        }
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

    if (hit?.rowHit) {
        if (componentKind === 'layer-norm') {
            if (node.role === 'layer-norm-input') {
                return {
                    type: 'layer-norm-input-row',
                    node,
                    rowHit: hit.rowHit
                };
            }
            if (node.role === 'layer-norm-normalized' || node.role === 'layer-norm-normalized-copy') {
                return {
                    type: 'layer-norm-normalized-row',
                    node,
                    rowHit: hit.rowHit
                };
            }
            if (node.role === 'layer-norm-scaled' || node.role === 'layer-norm-scaled-copy') {
                return {
                    type: 'layer-norm-scaled-row',
                    node,
                    rowHit: hit.rowHit
                };
            }
            if (node.role === 'layer-norm-output') {
                return {
                    type: 'layer-norm-output-row',
                    node,
                    rowHit: hit.rowHit
                };
            }
            if (node.role === 'layer-norm-scale') {
                return {
                    type: 'layer-norm-scale',
                    node,
                    rowHit: hit.rowHit
                };
            }
            if (node.role === 'layer-norm-shift') {
                return {
                    type: 'layer-norm-shift',
                    node,
                    rowHit: hit.rowHit
                };
            }
        }
        if (componentKind === 'output-projection') {
            if (node.role === 'projection-output') {
                return {
                    type: 'output-projection-output-row',
                    node,
                    rowHit: hit.rowHit
                };
            }
            if (node.role === 'concat-output-copy-matrix') {
                return {
                    type: 'output-projection-concat-output-row',
                    node,
                    rowHit: hit.rowHit
                };
            }
        }
        if (node.role === 'projection-cache-concat-result' && projectionKind) {
            return {
                type: 'projection-cache-concat-result-row',
                node,
                projectionKind,
                rowHit: hit.rowHit
            };
        }
        if (node.role === 'projection-cache-next' && projectionKind) {
            return {
                type: 'projection-cache-concat-result-row',
                node,
                projectionKind,
                rowHit: hit.rowHit
            };
        }
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
        if (node.role === 'mlp-up-output' || node.role === 'mlp-up-output-copy') {
            return {
                type: 'mlp-up-output-row',
                node,
                rowHit: hit.rowHit
            };
        }
        if (node.role === 'mlp-activation-output' || node.role === 'mlp-activation-output-copy') {
            return {
                type: 'mlp-activation-row',
                node,
                rowHit: hit.rowHit
            };
        }
        if (node.role === 'mlp-down-output') {
            return {
                type: 'mlp-down-output-row',
                node,
                rowHit: hit.rowHit
            };
        }
        if (node.role === 'mlp-up-bias') {
            return {
                type: 'mlp-up-bias',
                node,
                rowHit: hit.rowHit
            };
        }
        if (node.role === 'mlp-down-bias') {
            return {
                type: 'mlp-down-bias',
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
        if (node.role === 'head-output-matrix' || node.role === 'concat-head-copy-matrix') {
            return {
                type: 'output-projection-head-output-row',
                node,
                rowHit: hit.rowHit
            };
        }
        if (node.role === 'concat-output-matrix') {
            return {
                type: 'output-projection-concat-output-row',
                node,
                rowHit: hit.rowHit
            };
        }
        if (componentKind === 'output-projection' && node.role === 'projection-bias') {
            return {
                type: 'output-projection-bias',
                node,
                rowHit: hit.rowHit
            };
        }
    }

    if (componentKind === 'output-projection') {
        if (node.role === 'projection-weight') {
            return {
                type: 'output-projection-weight',
                node
            };
        }
        if (node.role === 'projection-bias') {
            return {
                type: 'output-projection-bias',
                node
            };
        }
    }

    if (node.role === 'projection-cache-concat-result' && projectionKind) {
        return {
            type: 'projection-cache-concat-result',
            node,
            projectionKind
        };
    }

    if (node.role === 'projection-cache-next' && projectionKind) {
        return {
            type: 'projection-cache-concat-result',
            node,
            projectionKind
        };
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

    if (node.role === 'mlp-up-weight') {
        return {
            type: 'mlp-up-weight',
            node
        };
    }

    if (node.role === 'mlp-up-bias') {
        return {
            type: 'mlp-up-bias',
            node
        };
    }

    if (node.role === 'mlp-down-weight') {
        return {
            type: 'mlp-down-weight',
            node
        };
    }

    if (node.role === 'mlp-down-bias') {
        return {
            type: 'mlp-down-bias',
            node
        };
    }

    if (componentKind === 'layer-norm') {
        if (node.role === 'layer-norm-scale') {
            return {
                type: 'layer-norm-scale',
                node
            };
        }
        if (node.role === 'layer-norm-shift') {
            return {
                type: 'layer-norm-shift',
                node
            };
        }
    }

    return {
        type: 'attention-role',
        node,
        role: String(node.role || '').trim()
    };
}
