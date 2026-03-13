const TOP_EMBED_ACTIVATION_INSET_FRACTION = 0.08;
const TOP_EMBED_ACTIVATION_INSET_MIN = 1.0;
const TOP_LOGIT_REVEAL_ACTIVATION_PROGRESS = 0.28;

function clamp01(value) {
    if (!Number.isFinite(value)) return 0;
    return Math.max(0, Math.min(1, value));
}

export function resolveTopEmbeddingActivationWindow(entryY, exitY) {
    if (!Number.isFinite(entryY)) return null;
    const safeExitY = Number.isFinite(exitY) ? Math.max(exitY, entryY) : entryY;
    const span = Math.max(0, safeExitY - entryY);
    const activationInset = Math.min(
        Math.max(TOP_EMBED_ACTIVATION_INSET_MIN, span * TOP_EMBED_ACTIVATION_INSET_FRACTION),
        Math.max(0, span - 1e-6)
    );

    return {
        entryY,
        exitY: safeExitY,
        span,
        activationInset,
        activationStartY: entryY + activationInset
    };
}

export function getTopEmbeddingActivationProgress(entryY, exitY, y) {
    const window = resolveTopEmbeddingActivationWindow(entryY, exitY);
    if (!window || !Number.isFinite(y)) return 0;
    if (y < window.activationStartY) return 0;
    if (window.exitY <= window.activationStartY + 1e-6) return 1;
    const denom = Math.max(1e-6, window.exitY - window.activationStartY);
    return denom > 0 ? clamp01((y - window.activationStartY) / denom) : 1;
}

export function easeTopEmbeddingActivationProgress(progress) {
    const t = clamp01(progress);
    return t * t * (3 - 2 * t);
}

export function getTopEmbeddingActivationEasedProgress(entryY, exitY, y) {
    return easeTopEmbeddingActivationProgress(
        getTopEmbeddingActivationProgress(entryY, exitY, y)
    );
}

export function resolveTopLogitRevealY(entryY, exitY, revealProgress = TOP_LOGIT_REVEAL_ACTIVATION_PROGRESS) {
    const window = resolveTopEmbeddingActivationWindow(entryY, exitY);
    if (!window) return null;
    const progress = clamp01(revealProgress);
    return window.activationStartY + (window.exitY - window.activationStartY) * progress;
}
