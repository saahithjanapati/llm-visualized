export const MLP_OVERLAY_STAGE_UP = Object.freeze({
    eqKey: 'mlp_up',
    eqTitle: 'MLP Up Projection'
});

export const MLP_OVERLAY_STAGE_GELU = Object.freeze({
    eqKey: 'mlp_gelu',
    eqTitle: 'GELU Activation'
});

export const MLP_OVERLAY_STAGE_DOWN = Object.freeze({
    eqKey: 'mlp_down',
    eqTitle: 'MLP Down Projection'
});

export function resolveMlpOverlayStage(lanes = []) {
    const safeLanes = Array.isArray(lanes) ? lanes : [];
    const downActive = safeLanes.some((lane) => lane?.mlpDownStarted || lane?.mlpDownComplete);
    if (downActive) return MLP_OVERLAY_STAGE_DOWN;

    const geluActive = safeLanes.some((lane) => lane?.mlpGeluActive || lane?.mlpGeluComplete);
    if (geluActive) return MLP_OVERLAY_STAGE_GELU;

    const upActive = safeLanes.some((lane) => (
        lane?.mlpUpStarted
        || lane?.ln2Phase === 'mlpReady'
        || lane?.ln2Phase === 'done'
    ));
    if (upActive) return MLP_OVERLAY_STAGE_UP;

    return null;
}
