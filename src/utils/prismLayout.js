import { VECTOR_LENGTH_PRISM, PRISM_BASE_WIDTH } from './constants.js';

export const PRISM_INSTANCE_WIDTH_SCALE = 1.5;

export function getPrismCenterOffset(length = VECTOR_LENGTH_PRISM) {
    return (length - 1) / 2;
}

export function getPrismSpacing(widthScale = PRISM_INSTANCE_WIDTH_SCALE) {
    return PRISM_BASE_WIDTH * widthScale;
}

export function computeCenteredPrismX(index, length = VECTOR_LENGTH_PRISM, widthScale = PRISM_INSTANCE_WIDTH_SCALE) {
    const centreOffset = getPrismCenterOffset(length);
    const spacing = getPrismSpacing(widthScale);
    return (index - centreOffset) * spacing;
}

export function getCentralPrismIndices(length = VECTOR_LENGTH_PRISM) {
    if (!Number.isFinite(length) || length <= 0) {
        return [0];
    }

    const centreOffset = getPrismCenterOffset(length);
    const leftIndex = Math.max(0, Math.floor(centreOffset));
    const rightIndex = Math.min(length - 1, Math.ceil(centreOffset));

    return leftIndex === rightIndex ? [leftIndex] : [leftIndex, rightIndex];
}
