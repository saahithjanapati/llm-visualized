export const MHSA_INFO_REQUEST_EVENT = 'llm-visualized:open-mhsa-info';
export const MHSA_INFO_LABEL = 'Multi-Head Self-Attention Inspector';

function normalizeOptionalIndex(value) {
    const next = Number(value);
    return Number.isFinite(next) ? Math.max(0, Math.floor(next)) : null;
}

export function formatMhsaInfoTitle(layerIndex = null) {
    return Number.isFinite(layerIndex)
        ? `${MHSA_INFO_LABEL} for Layer ${Math.floor(layerIndex) + 1}`
        : MHSA_INFO_LABEL;
}

export function buildMhsaInfoSelection({ layerIndex = null, headIndex = null } = {}) {
    const safeLayerIndex = normalizeOptionalIndex(layerIndex);
    const safeHeadIndex = normalizeOptionalIndex(headIndex);
    const info = {
        activationData: {
            stage: 'attention.overview'
        }
    };
    if (Number.isFinite(safeLayerIndex)) {
        info.layerIndex = safeLayerIndex;
        info.activationData.layerIndex = safeLayerIndex;
    }
    if (Number.isFinite(safeHeadIndex)) {
        info.headIndex = safeHeadIndex;
        info.activationData.headIndex = safeHeadIndex;
    }
    return {
        label: MHSA_INFO_LABEL,
        kind: 'mhsaInfo',
        info
    };
}

export function isMhsaInfoSelection(label = '', selectionInfo = null) {
    if (selectionInfo?.kind === 'mhsaInfo') return true;
    const lower = String(label || '').trim().toLowerCase();
    return lower === 'multi-head self-attention'
        || lower === MHSA_INFO_LABEL.toLowerCase();
}
