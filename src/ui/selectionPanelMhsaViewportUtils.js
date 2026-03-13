export function resolveMhsaTokenMatrixFixedLabelScale(viewportScale = 1) {
    const safeViewportScale = Number.isFinite(viewportScale) && viewportScale > 0
        ? viewportScale
        : 1;
    return 1 / safeViewportScale;
}
