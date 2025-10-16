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
