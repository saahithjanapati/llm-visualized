export const MHSA_INFO_REQUEST_EVENT = 'llm-visualized:open-mhsa-info';

export function buildMhsaInfoSelection() {
    return {
        label: 'Multi-Head Self-Attention',
        kind: 'mhsaInfo',
        info: {
            activationData: {
                stage: 'attention.overview'
            }
        }
    };
}

export function isMhsaInfoSelection(label = '', selectionInfo = null) {
    if (selectionInfo?.kind === 'mhsaInfo') return true;
    return String(label || '').trim().toLowerCase() === 'multi-head self-attention';
}
