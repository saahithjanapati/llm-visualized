export function hasSelectionPanelKeyboardOwnership({
    panel = null,
    keyboardEngaged = false,
    activeElement = null
} = {}) {
    if (keyboardEngaged) return true;
    if (!panel || typeof panel.contains !== 'function') return false;
    return !!(activeElement && panel.contains(activeElement));
}

export function shouldCaptureMhsaKeyboardInput({
    isOpen = false,
    isMhsaInfoSelectionActive = false,
    mhsaTokenMatrixHidden = true,
    panel = null,
    keyboardEngaged = false,
    activeElement = null
} = {}) {
    if (!isOpen || !isMhsaInfoSelectionActive || mhsaTokenMatrixHidden) return false;
    return hasSelectionPanelKeyboardOwnership({
        panel,
        keyboardEngaged,
        activeElement
    });
}
