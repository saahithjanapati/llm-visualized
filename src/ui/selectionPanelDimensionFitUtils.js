const DETAIL_DIM_LABEL_MAX_FONT_PX = 11;
const DETAIL_DIM_LABEL_MIN_FONT_PX = 7.5;
const DETAIL_DIM_LABEL_MAX_LETTER_SPACING_PX = 0.5;
const DETAIL_DIM_LABEL_MIN_LETTER_SPACING_PX = 0.12;
const DETAIL_DIM_LABEL_FIT_EPSILON_PX = 0.75;
const DETAIL_DIM_LABEL_FIT_ITERATIONS = 8;

const clamp = (value, min, max) => Math.max(min, Math.min(max, value));

function isVisible(el) {
    if (!el || typeof window === 'undefined') return false;
    if (!el.isConnected) return false;
    const style = window.getComputedStyle(el);
    if (!style || style.display === 'none' || style.visibility === 'hidden') return false;
    if (el.getClientRects().length === 0) return false;
    return true;
}

function applyLabelTypography(labelEl, fontPx, letterSpacingPx) {
    labelEl.style.fontSize = `${fontPx.toFixed(3)}px`;
    labelEl.style.letterSpacing = `${letterSpacingPx.toFixed(3)}px`;
}

function fitSingleDimensionLabel(labelEl) {
    if (!isVisible(labelEl)) return;

    applyLabelTypography(
        labelEl,
        DETAIL_DIM_LABEL_MAX_FONT_PX,
        DETAIL_DIM_LABEL_MAX_LETTER_SPACING_PX
    );

    const availableWidth = labelEl.clientWidth;
    if (!Number.isFinite(availableWidth) || availableWidth <= 0) return;

    const baseWidth = labelEl.scrollWidth;
    if (!Number.isFinite(baseWidth) || baseWidth <= availableWidth + DETAIL_DIM_LABEL_FIT_EPSILON_PX) {
        return;
    }

    const quickScale = clamp(
        availableWidth / baseWidth,
        DETAIL_DIM_LABEL_MIN_FONT_PX / DETAIL_DIM_LABEL_MAX_FONT_PX,
        1
    );
    const quickFontPx = DETAIL_DIM_LABEL_MAX_FONT_PX * quickScale;
    const quickLetterSpacingPx = clamp(
        DETAIL_DIM_LABEL_MAX_LETTER_SPACING_PX * quickScale,
        DETAIL_DIM_LABEL_MIN_LETTER_SPACING_PX,
        DETAIL_DIM_LABEL_MAX_LETTER_SPACING_PX
    );
    applyLabelTypography(labelEl, quickFontPx, quickLetterSpacingPx);

    if (labelEl.scrollWidth <= availableWidth + DETAIL_DIM_LABEL_FIT_EPSILON_PX) {
        return;
    }

    let low = DETAIL_DIM_LABEL_MIN_FONT_PX;
    let high = quickFontPx;
    let bestFontPx = low;
    for (let idx = 0; idx < DETAIL_DIM_LABEL_FIT_ITERATIONS; idx += 1) {
        const mid = (low + high) * 0.5;
        applyLabelTypography(labelEl, mid, DETAIL_DIM_LABEL_MIN_LETTER_SPACING_PX);
        const fits = labelEl.scrollWidth <= availableWidth + DETAIL_DIM_LABEL_FIT_EPSILON_PX;
        if (fits) {
            bestFontPx = mid;
            low = mid;
        } else {
            high = mid;
        }
    }
    applyLabelTypography(labelEl, bestFontPx, DETAIL_DIM_LABEL_MIN_LETTER_SPACING_PX);
}

export function fitSelectionDimensionLabels({ inputDimLabel, outputDimLabel }) {
    if (typeof window === 'undefined') return;
    fitSingleDimensionLabel(inputDimLabel);
    fitSingleDimensionLabel(outputDimLabel);
}
