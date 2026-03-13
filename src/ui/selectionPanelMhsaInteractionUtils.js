const MHSA_TOKEN_MATRIX_PROJECTION_STAGE_SELECTOR = '[data-mhsa-projection-stage-index]';

const MHSA_TOKEN_MATRIX_PINNABLE_PROJECTION_STAGE_COMPONENT_SELECTOR = [
    '.mhsa-token-matrix-preview__x-block',
    '.mhsa-token-matrix-preview__weight-block',
    '.mhsa-token-matrix-preview__bias-block',
    '.mhsa-token-matrix-preview__query-block',
    '.mhsa-token-matrix-preview__operator--matrix',
    '.mhsa-token-matrix-preview__operator--xw',
    '.mhsa-token-matrix-preview__operator--wb',
    '.mhsa-token-matrix-preview__operator--bq'
].join(', ');

export function resolveMhsaTokenMatrixProjectionStageTarget(target, {
    root = null,
    requirePinnableComponent = false
} = {}) {
    if (!(target instanceof Element)) return null;

    const stageEl = target.closest(MHSA_TOKEN_MATRIX_PROJECTION_STAGE_SELECTOR);
    if (!(stageEl instanceof Element)) return null;
    if (root instanceof Element && !root.contains(stageEl)) return null;

    if (requirePinnableComponent) {
        const componentEl = target.closest(MHSA_TOKEN_MATRIX_PINNABLE_PROJECTION_STAGE_COMPONENT_SELECTOR);
        if (!(componentEl instanceof Element) || !stageEl.contains(componentEl)) {
            return null;
        }
    }

    const stageIndex = Number(stageEl.dataset.mhsaProjectionStageIndex);
    if (!Number.isFinite(stageIndex)) return null;

    return {
        stageIndex,
        focusKey: String(stageEl.dataset.mhsaProjectionKind || '').toLowerCase()
    };
}
